// Expectation maximization for the Camel Morph context disambiguator
// (developer tooling; tools/arabic_morph/disambig_train.py drives it).
//
//   ar-morph-disambig-em corpus.bin counts.bin iterations alpha kappa beta threads uniform|unambiguous
//       [init-weights.f64]
//
// corpus.bin (written by disambig_train.py, little-endian):
//   "AMEM", u32 n_tags (tag 0 is the sentence boundary), u32 n_lexpos, u32 n_pos,
//   u32 pos_of_lexpos[n_lexpos],
//   u32 n_types, per type { u32 n, n x (u32 tag, u32 lexpos) },
//   u32 n_sentences, per sentence { u32 n, u32 type[n] }.
// The model is a bigram HMM over candidates (tag, lexpos): a sentence scores
// prod T(tag_i | tag_i-1) * L(lexpos_i | pos), with the boundary tag at
// both ends. Iteration 0 gives every candidate of a token the same
// posterior; each later iteration runs forward-backward with the previous
// parameters (T backed off to the add-alpha tag unigram with weight kappa,
// L add-beta smoothed). counts.bin receives the
// expected counts of the last E step: u32 n_tags, u32 n_lexpos,
// f64 transitions[n_tags][n_tags] (previous tag major), f64 lexpos[n_lexpos].
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Cand {
    uint32_t tag, lp;
};

struct Corpus {
    uint32_t n_tags = 0, n_lp = 0, n_pos = 0;
    std::vector<uint32_t> lp_pos;
    std::vector<uint32_t> type_begin; // n_types + 1
    std::vector<Cand> cands;
    std::vector<uint32_t> sent_begin; // n_sentences + 1
    std::vector<uint32_t> tokens;     // type ids
};

[[noreturn]] void die(const char * msg) {
    std::fprintf(stderr, "ar-morph-disambig-em: %s\n", msg);
    std::exit(1);
}

struct Reader {
    FILE * f;
    uint32_t u32() {
        uint32_t v;
        if (std::fread(&v, 4, 1, f) != 1) die("truncated corpus");
        return v;
    }
};

Corpus load(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) die("cannot open corpus");
    Reader r{ f };
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "AMEM", 4) != 0) die("bad magic");
    Corpus c;
    c.n_tags = r.u32();
    c.n_lp = r.u32();
    c.n_pos = r.u32();
    c.lp_pos.resize(c.n_lp);
    for (auto & p : c.lp_pos) p = r.u32();
    uint32_t n_types = r.u32();
    c.type_begin.push_back(0);
    for (uint32_t t = 0; t < n_types; ++t) {
        uint32_t n = r.u32();
        for (uint32_t i = 0; i < n; ++i) {
            Cand x;
            x.tag = r.u32();
            x.lp = r.u32();
            if (x.tag >= c.n_tags || x.lp >= c.n_lp || x.tag == 0) die("bad candidate");
            c.cands.push_back(x);
        }
        c.type_begin.push_back(uint32_t(c.cands.size()));
    }
    uint32_t n_sent = r.u32();
    c.sent_begin.push_back(0);
    for (uint32_t s = 0; s < n_sent; ++s) {
        uint32_t n = r.u32();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t t = r.u32();
            if (t >= n_types) die("bad type");
            c.tokens.push_back(t);
        }
        c.sent_begin.push_back(uint32_t(c.tokens.size()));
    }
    std::fclose(f);
    return c;
}

struct Counts {
    std::vector<double> trans, lp;
    double loglik = 0;
    explicit Counts(const Corpus & c) : trans(size_t(c.n_tags) * c.n_tags, 0.0), lp(c.n_lp, 0.0) {}
};

struct Params {
    std::vector<double> trans; // T[prev * n + cur]
    std::vector<double> emit;  // L per lexpos
};

Params m_step(const Corpus & c, const Counts & k, double alpha, double kappa, double beta) {
    Params p;
    const size_t n = c.n_tags;
    // T(cur | prev) = (C(prev, cur) + kappa U(cur)) / (C(prev) + kappa), with
    // the add-alpha unigram U(cur) of the tags that follow anything.
    std::vector<double> uni(n, 0.0);
    double total = 0;
    for (size_t a = 0; a < n; ++a)
        for (size_t b = 0; b < n; ++b) uni[b] += k.trans[a * n + b];
    for (size_t b = 0; b < n; ++b) total += uni[b];
    for (size_t b = 0; b < n; ++b) uni[b] = (uni[b] + alpha) / (total + alpha * double(n));
    p.trans.assign(n * n, 0.0);
    for (size_t a = 0; a < n; ++a) {
        double row = 0;
        for (size_t b = 0; b < n; ++b) row += k.trans[a * n + b];
        for (size_t b = 0; b < n; ++b) p.trans[a * n + b] = (k.trans[a * n + b] + kappa * uni[b]) / (row + kappa);
    }
    std::vector<double> pos_total(c.n_pos, 0.0), pos_types(c.n_pos, 0.0);
    for (uint32_t i = 0; i < c.n_lp; ++i) {
        pos_total[c.lp_pos[i]] += k.lp[i];
        pos_types[c.lp_pos[i]] += 1;
    }
    p.emit.resize(c.n_lp);
    for (uint32_t i = 0; i < c.n_lp; ++i) {
        uint32_t q = c.lp_pos[i];
        p.emit[i] = (k.lp[i] + beta) / (pos_total[q] + beta * (pos_types[q] + 1));
    }
    return p;
}

// Initial posteriors without a model. With `prior` empty every candidate of a
// token gets 1/n; otherwise candidate weights are prior[tag], normalized per
// token. Adjacent pairs get the product of their posteriors.
void e_step_uniform(const Corpus & c, Counts & k, const std::vector<double> & prior) {
    const size_t n = c.n_tags;
    auto weight = [&](uint32_t b, uint32_t e, uint32_t x) {
        if (prior.empty()) return 1.0 / double(e - b);
        double z = 0;
        for (uint32_t y = b; y < e; ++y) z += prior[c.cands[y].tag];
        return prior[c.cands[x].tag] / z;
    };
    for (size_t s = 0; s + 1 < c.sent_begin.size(); ++s) {
        uint32_t prev_b = 0, prev_e = 0; // empty range = boundary
        for (uint32_t i = c.sent_begin[s]; i <= c.sent_begin[s + 1]; ++i) {
            bool end = i == c.sent_begin[s + 1];
            uint32_t b = end ? 0 : c.type_begin[c.tokens[i]];
            uint32_t e = end ? 0 : c.type_begin[c.tokens[i] + 1];
            for (uint32_t x = prev_b; x < (prev_e > prev_b ? prev_e : prev_b + 1); ++x) {
                uint32_t pt = prev_e > prev_b ? c.cands[x].tag : 0;
                double wx = prev_e > prev_b ? weight(prev_b, prev_e, x) : 1.0;
                for (uint32_t y = b; y < (e > b ? e : b + 1); ++y) {
                    uint32_t ct = e > b ? c.cands[y].tag : 0;
                    k.trans[pt * n + ct] += wx * (e > b ? weight(b, e, y) : 1.0);
                }
            }
            if (!end)
                for (uint32_t y = b; y < e; ++y) k.lp[c.cands[y].lp] += weight(b, e, y);
            prev_b = b;
            prev_e = e;
        }
    }
}

void e_step_range(const Corpus & c, const Params & p, size_t s0, size_t s1, Counts & k) {
    const size_t n = c.n_tags;
    std::vector<double> alpha, beta, emit, scale;
    std::vector<uint32_t> off;
    for (size_t s = s0; s < s1; ++s) {
        uint32_t t0 = c.sent_begin[s], t1 = c.sent_begin[s + 1];
        uint32_t len = t1 - t0;
        if (!len) continue;
        off.assign(len + 1, 0);
        for (uint32_t i = 0; i < len; ++i) {
            uint32_t ty = c.tokens[t0 + i];
            off[i + 1] = off[i] + (c.type_begin[ty + 1] - c.type_begin[ty]);
        }
        alpha.assign(off[len], 0.0);
        beta.assign(off[len], 0.0);
        emit.assign(off[len], 0.0);
        scale.assign(len + 1, 0.0);
        auto cand = [&](uint32_t i, uint32_t j) -> const Cand & { return c.cands[c.type_begin[c.tokens[t0 + i]] + j]; };
        // Forward.
        for (uint32_t i = 0; i < len; ++i) {
            double sum = 0;
            uint32_t m = off[i + 1] - off[i];
            for (uint32_t j = 0; j < m; ++j) {
                const Cand & x = cand(i, j);
                double e = p.emit[x.lp];
                emit[off[i] + j] = e;
                double in = 0;
                if (i == 0) {
                    in = p.trans[x.tag];
                } else {
                    uint32_t pm = off[i] - off[i - 1];
                    for (uint32_t q = 0; q < pm; ++q)
                        in += alpha[off[i - 1] + q] * p.trans[cand(i - 1, q).tag * n + x.tag];
                }
                double a = in * e;
                alpha[off[i] + j] = a;
                sum += a;
            }
            if (!(sum > 0)) sum = 1e-300;
            for (uint32_t j = 0; j < m; ++j) alpha[off[i] + j] /= sum;
            scale[i] = sum;
        }
        // End transition.
        double end_sum = 0;
        uint32_t lm = off[len] - off[len - 1];
        for (uint32_t j = 0; j < lm; ++j)
            end_sum += alpha[off[len - 1] + j] * p.trans[size_t(cand(len - 1, j).tag) * n];
        if (!(end_sum > 0)) end_sum = 1e-300;
        scale[len] = end_sum;
        double ll = 0;
        for (uint32_t i = 0; i <= len; ++i) ll += std::log(scale[i]);
        k.loglik += ll;
        for (uint32_t j = 0; j < lm; ++j) beta[off[len - 1] + j] = p.trans[size_t(cand(len - 1, j).tag) * n] / end_sum;
        // Backward.
        for (uint32_t i = len - 1; i > 0; --i) {
            uint32_t m = off[i + 1] - off[i], pm = off[i] - off[i - 1];
            for (uint32_t q = 0; q < pm; ++q) {
                uint32_t pt = cand(i - 1, q).tag;
                double b = 0;
                for (uint32_t j = 0; j < m; ++j)
                    b += p.trans[pt * n + cand(i, j).tag] * emit[off[i] + j] * beta[off[i] + j];
                beta[off[i - 1] + q] = b / scale[i];
            }
        }
        // Expected counts.
        for (uint32_t i = 0; i < len; ++i) {
            uint32_t m = off[i + 1] - off[i];
            double z = 0;
            for (uint32_t j = 0; j < m; ++j) z += alpha[off[i] + j] * beta[off[i] + j];
            if (!(z > 0)) z = 1;
            for (uint32_t j = 0; j < m; ++j) k.lp[cand(i, j).lp] += alpha[off[i] + j] * beta[off[i] + j] / z;
            if (i == 0) {
                for (uint32_t j = 0; j < m; ++j) {
                    const Cand & x = cand(0, j);
                    k.trans[x.tag] += p.trans[x.tag] * emit[j] * beta[j] / scale[0];
                }
            } else {
                uint32_t pm = off[i] - off[i - 1];
                for (uint32_t q = 0; q < pm; ++q) {
                    uint32_t pt = cand(i - 1, q).tag;
                    double a = alpha[off[i - 1] + q];
                    for (uint32_t j = 0; j < m; ++j) {
                        uint32_t ct = cand(i, j).tag;
                        k.trans[pt * n + ct] +=
                            a * p.trans[pt * n + ct] * emit[off[i] + j] * beta[off[i] + j] / scale[i];
                    }
                }
            }
        }
        for (uint32_t j = 0; j < lm; ++j) {
            uint32_t pt = cand(len - 1, j).tag;
            k.trans[size_t(pt) * n] += alpha[off[len - 1] + j] * p.trans[size_t(pt) * n] / end_sum;
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    if ((argc != 9 && argc != 10) ||
        (std::strcmp(argv[8], "uniform") != 0 && std::strcmp(argv[8], "unambiguous") != 0)) {
        std::fprintf(stderr, "usage: ar-morph-disambig-em corpus.bin counts.bin iterations alpha kappa beta threads "
                             "uniform|unambiguous [init-weights.f64]\n");
        return 64;
    }
    Corpus c = load(argv[1]);
    int iterations = std::atoi(argv[3]);
    double alpha = std::atof(argv[4]), kappa = std::atof(argv[5]), beta = std::atof(argv[6]);
    int threads = std::atoi(argv[7]);
    if (threads < 1) threads = 1;
    // init "unambiguous" starts from the tag frequencies of the tokens whose
    // candidates all share one tag instead of uniform posteriors.
    std::vector<double> prior;
    if (std::strcmp(argv[8], "unambiguous") == 0) {
        prior.assign(c.n_tags, 1.0);
        for (uint32_t t : c.tokens) {
            uint32_t b = c.type_begin[t], e = c.type_begin[t + 1];
            bool one = true;
            for (uint32_t y = b + 1; y < e; ++y) one = one && c.cands[y].tag == c.cands[b].tag;
            if (one && e > b) prior[c.cands[b].tag] += 1;
        }
    }
    // Optional per-tag multipliers of the initial weights (f64 per tag).
    if (argc == 10) {
        FILE * wf = std::fopen(argv[9], "rb");
        if (!wf) die("cannot open init weights");
        std::vector<double> w(c.n_tags);
        if (std::fread(w.data(), 8, w.size(), wf) != w.size()) die("truncated init weights");
        std::fclose(wf);
        if (prior.empty()) prior.assign(c.n_tags, 1.0);
        for (size_t t = 0; t < w.size(); ++t) prior[t] *= w[t];
    }
    Counts k(c);
    e_step_uniform(c, k, prior);
    for (int it = 1; it <= iterations; ++it) {
        Params p = m_step(c, k, alpha, kappa, beta);
        size_t ns = c.sent_begin.size() - 1;
        std::vector<Counts> part(size_t(threads), Counts{ c });
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&, t] { e_step_range(c, p, ns * t / threads, ns * (t + 1) / threads, part[t]); });
        for (auto & th : pool) th.join();
        Counts sum{ c };
        for (auto & q : part) {
            for (size_t i = 0; i < sum.trans.size(); ++i) sum.trans[i] += q.trans[i];
            for (size_t i = 0; i < sum.lp.size(); ++i) sum.lp[i] += q.lp[i];
            sum.loglik += q.loglik;
        }
        k = std::move(sum);
        std::fprintf(stderr, "{\"iteration\": %d, \"loglik_per_token\": %.6f}\n", it,
                     k.loglik / double(c.tokens.size() ? c.tokens.size() : 1));
    }
    FILE * f = std::fopen(argv[2], "wb");
    if (!f) die("cannot write counts");
    uint32_t hdr[2] = { c.n_tags, c.n_lp };
    std::fwrite(hdr, 4, 2, f);
    std::fwrite(k.trans.data(), 8, k.trans.size(), f);
    std::fwrite(k.lp.data(), 8, k.lp.size(), f);
    std::fclose(f);
    return 0;
}
