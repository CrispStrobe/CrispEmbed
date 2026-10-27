// unlimited_ocr2_dash.cpp — see unlimited_ocr2_dash.h. Independent implementation of
// the documented reference semantics (baidu/Unlimited-OCR-2@04647c7a, MIT).
#include "unlimited_ocr2_dash.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace uocr2 {

coords prefill_coords(const std::vector<int32_t> & ids) {
    coords c;
    const int S = int(ids.size());
    int last_img = -1;
    for (int t = 0; t < S; ++t)
        if (ids[t] == IMG) last_img = t;
    std::vector<int> cum(S);
    for (int t = 0, m = 0; t < S; ++t) cum[t] = (m += ids[t] == PAGE);
    c.page_mode = last_img >= 0 && cum[last_img] > 0;
    c.tok.assign(S, 0);
    c.page.assign(S, 0);
    if (!c.page_mode) {
        for (int t = 0; t < S; ++t) c.tok[t] = t;
        c.cur_tok = S - 1;
        return c;
    }
    const int n_pages = cum[last_img];
    int max_page = 0;
    for (int t = 0; t < S; ++t) {
        c.page[t] = std::max(0, t <= last_img ? cum[t] : cum[t] - n_pages);
        max_page = std::max(max_page, c.page[t]);
    }
    c.n_pg = max_page + 1;
    std::vector<int> running(c.n_pg, 0);
    c.pf_cnt.assign(c.n_pg, 0);
    for (int t = 0; t < S; ++t) {
        c.tok[t] = running[c.page[t]]++;
        if (t <= last_img) ++c.pf_cnt[c.page[t]];
    }
    c.cur_page = c.page[S - 1];
    c.cur_tok = c.tok[S - 1];
    return c;
}

void coords::advance(int32_t fed) {
    if (page_mode && fed == PAGE) {
        cur_page = std::min(cur_page + 1, n_pg - 1);
        cur_tok = pf_cnt[cur_page];
    } else {
        ++cur_tok;
    }
}

void rope_table(int tok, int page, float * co, float * si) {
    // 63 'resample' token pairs (theta 1e4) + 1 page pair (theta 256 -> inv 1.0, page mod 2).
    for (int i = 0; i < 64; ++i) {
        const double inv = i < 63 ? std::pow(10000.0, -double(i) / 63.0) : 1.0;
        const double ang = i < 63 ? double(tok) * inv : double(page % 2) * inv;
        co[i] = co[i + 64] = float(std::cos(ang));
        si[i] = si[i + 64] = float(std::sin(ang));
    }
}

void apply_rope(float * x, const float * co, const float * si) {
    for (int i = 0; i < 64; ++i) {
        const float a = x[i], b = x[i + 64];
        x[i] = a * co[i] - b * si[i];
        x[i + 64] = b * co[i + 64] + a * si[i + 64];
    }
}

void gemma_rms(float * x, int n, const float * w, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float r = 1.0f / std::sqrt(float(ss / n) + eps);
    for (int i = 0; i < n; ++i) x[i] = std::clamp(x[i] * r * (1.0f + w[i]), -1e4f, 1e4f);
}

pages prefill_pages(const std::vector<int32_t> & ids) {
    pages p;
    const int S = int(ids.size());
    int last_img = -1;
    for (int t = 0; t < S; ++t)
        if (ids[t] == IMG) last_img = t;
    std::vector<int> cum(S);
    for (int t = 0, m = 0; t < S; ++t) cum[t] = (m += ids[t] == PAGE);
    p.prompt_pages = last_img >= 0 ? cum[last_img] : 0;
    p.mcum = S ? cum[S - 1] : 0;
    const bool marker_mode = p.prompt_pages > 0;
    p.img_page.assign(S, -1);
    p.q_page.assign(S, -1);
    for (int t = 0, run = 0; t < S; ++t) {
        if (ids[t] == IMG) p.img_page[t] = marker_mode ? cum[t] - 1 : run++ / 273;
        const int raw = cum[t] - 1, q = t <= last_img ? raw : raw - p.prompt_pages;
        p.q_page[t] = q < 0 ? -1 : q;
    }
    p.n_prefill_blocks = (S + BLOCK - 1) / BLOCK;
    p.bmin.assign(p.n_prefill_blocks, -1);
    p.bmax.assign(p.n_prefill_blocks, -1);
    for (int t = 0; t < S; ++t)
        if (p.img_page[t] >= 0) {
            int & lo = p.bmin[t / BLOCK], & hi = p.bmax[t / BLOCK];
            lo = lo < 0 ? p.img_page[t] : std::min(lo, p.img_page[t]);
            hi = std::max(hi, p.img_page[t]);
        }
    return p;
}

void router_keys::summary(int c, const float * head_cls, float * out) const {
    const int r0 = c * BLOCK, r1 = std::min(rows(), r0 + BLOCK);
    const float scale = 1.0f / std::sqrt(float(HD));
    std::vector<float> a(r1 - r0);
    for (int j = r0; j < r1; ++j) {
        float d = 0;
        for (int i = 0; i < HD; ++i) d += ik[size_t(j) * HD + i] * head_cls[i];
        a[j - r0] = std::clamp(d * scale, -1e4f, 1e4f);
    }
    const float m = *std::max_element(a.begin(), a.end());
    float s = 0;
    for (float & x : a) s += (x = std::exp(x - m));
    s = std::max(s, 1e-6f);
    std::fill(out, out + HD, 0.0f);
    for (int j = r0; j < r1; ++j)
        for (int i = 0; i < HD; ++i) out[i] += a[j - r0] / s * ik[size_t(j) * HD + i];
}

void router_keys::append(const float * row, const float * head_cls) {
    ik.insert(ik.end(), row, row + HD);
    if (rows() % BLOCK == 0) { // a block just completed: freeze its summary
        const int c = rows() / BLOCK - 1;
        kbar.resize(size_t(c + 1) * HD);
        summary(c, head_cls, kbar.data() + size_t(c) * HD);
    }
}

void entmax15(const float * xin, int n, float * out) {
    std::vector<float> X(n), Xs(n);
    const float mx = *std::max_element(xin, xin + n);
    for (int i = 0; i < n; ++i) X[i] = (xin[i] - mx) / 2.0f;
    Xs = X;
    std::sort(Xs.begin(), Xs.end(), std::greater<float>());
    double csum = 0, csq = 0;
    std::vector<double> tau(n);
    int support = 0;
    for (int r = 1; r <= n; ++r) {
        csum += Xs[r - 1];
        csq += double(Xs[r - 1]) * Xs[r - 1];
        const double mean = csum / r, mean_sq = csq / r, ss = r * (mean_sq - mean * mean);
        const double delta = (1.0 - ss) / r;
        tau[r - 1] = mean - std::sqrt(std::max(delta, 0.0));
        if (tau[r - 1] <= Xs[r - 1]) support = r;
    }
    const double t = tau[std::max(support, 1) - 1];
    for (int i = 0; i < n; ++i) {
        const double z = std::max(double(X[i]) - t, 0.0);
        out[i] = float(z * z);
    }
}

selection route(const router_keys & rk, const float * iq, const float * head_cls, int pos,
                const std::vector<uint8_t> & candidate) {
    selection sel;
    const int nblk = (rk.rows() + BLOCK - 1) / BLOCK, qblk = pos / BLOCK;
    const float scale = 1.0f / std::sqrt(float(HD)), idx_scale = 128.0f / std::sqrt(float(HD));
    std::vector<float> score(nblk, SCORE_NEG), kb(HD);
    for (int c = 0; c < nblk; ++c) {
        if (c > qblk - LOCAL_BLOCKS) continue; // not remote
        const float * k = c < int(rk.kbar.size() / HD) ? rk.kbar.data() + size_t(c) * HD : (rk.summary(c, head_cls, kb.data()), kb.data());
        float d = 0;
        for (int i = 0; i < HD; ++i) d += iq[i] * k[i];
        score[c] = std::clamp(d * scale, -1e4f, 1e4f);
    }
    std::vector<uint8_t> cand(nblk);
    bool any = false;
    for (int c = 0; c < nblk; ++c) any |= (cand[c] = score[c] > -5e8f && c < int(candidate.size()) && candidate[c]);
    if (!any)
        for (int c = 0; c < nblk; ++c) cand[c] = score[c] > -5e8f;
    std::vector<float> x(nblk);
    for (int c = 0; c < nblk; ++c) x[c] = cand[c] && std::isfinite(score[c] * idx_scale) ? score[c] * idx_scale : ENTMAX_NEG;
    const int M = std::min(TOPM, nblk);
    if (M == 0) return sel;
    std::vector<int> order(nblk);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] > x[b]; });
    std::vector<float> xv(M), prior(M);
    for (int i = 0; i < M; ++i) xv[i] = x[order[i]];
    entmax15(xv.data(), M, prior.data());
    const int K = std::min(TOPK, M);
    double mu = 0;
    int nvalid = 0;
    for (int i = 0; i < K; ++i)
        if (cand[order[i]]) { mu += std::log(std::max(prior[i], FLOOR)); ++nvalid; }
    mu /= std::max(nvalid, 1);
    for (int i = 0; i < K; ++i)
        if (cand[order[i]]) {
            sel.blocks.push_back(order[i]);
            sel.bias.push_back(float((std::log(std::max(prior[i], FLOOR)) - mu) / SIGMA));
        }
    return sel;
}

void attend(const float * q, const float * k, const float * v, int stride, int pos, const selection & sel,
            float * out) {
    const float scale = 1.0f / std::sqrt(float(HD));
    std::vector<int> keys;
    std::vector<float> logit;
    std::vector<float> blk_bias((pos / BLOCK) + 1, NAN);
    for (size_t i = 0; i < sel.blocks.size(); ++i) blk_bias[sel.blocks[i]] = sel.bias[i];
    for (int j = 0; j <= pos; ++j) {
        const bool local = j > pos - LOCAL_WINDOW;
        const float b = blk_bias[j / BLOCK];
        if (!local && std::isnan(b)) continue;
        float d = 0;
        for (int i = 0; i < HD; ++i) d += q[i] * k[size_t(j) * stride + i];
        keys.push_back(j);
        logit.push_back(d * scale + (local ? 0.0f : b));
    }
    const float m = *std::max_element(logit.begin(), logit.end());
    double s = 0;
    for (float & l : logit) s += (l = std::exp(l - m));
    std::fill(out, out + HD, 0.0f);
    for (size_t n = 0; n < keys.size(); ++n)
        for (int i = 0; i < HD; ++i) out[i] += float(logit[n] / s) * v[size_t(keys[n]) * stride + i];
}
} // namespace uocr2
