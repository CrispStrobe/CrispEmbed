// crispembed-ar-morph: C-ABI-only runner for the native Camel Morph analyzer.
//
//   crispembed-ar-morph --manifest full/manifest.json --sha256 <manifest sha> [--backoff NONE] WORD...
//   crispembed-ar-morph --db camel_morph_msa_v1.0.db --sha256 <db sha> --jsonl words.jsonl --out native.jsonl
//
// --jsonl reads one JSON string per line and writes {"i":n,"ok":true,"analyses":[...]}
// lines (analyses as [[key, value], ...] pairs, the camel_reference.py format) or
// {"i":n,"ok":false,"error":"..."}. Timing and peak RSS go to stderr as JSON.
// --compact [--dedup] uses crispembed_ar_morph_analyze_compact in batches of
// --batch words (default 512) and decodes its buffer into the same lines.
// --repeat N analyzes the list N times (timing only; output from the first run).
#include "arabic_morph.h"
#include "core/json.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace {

double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

long peak_rss_kib() {
#if defined(__APPLE__)
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return long(ru.ru_maxrss / 1024); // bytes on macOS
#elif defined(__unix__)
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return long(ru.ru_maxrss); // KiB on Linux
#else
    return -1;
#endif
}

// Current resident set size in KiB (-1 when unknown).
long current_rss_kib() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return -1;
    return long(info.resident_size / 1024);
#elif defined(__linux__)
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) return -1;
    long pages = 0, resident = 0;
    const int n = std::fscanf(f, "%ld %ld", &pages, &resident);
    std::fclose(f);
    return n == 2 ? resident * 4 : -1;
#else
    return -1;
#endif
}

uint32_t rd32(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint32_t rd16(const uint8_t * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
}

// The "keys" array of the info JSON (our own output: plain JSON strings).
std::vector<std::string> info_keys(const std::string & info) {
    std::vector<std::string> keys;
    size_t i = info.find("\"keys\":[");
    if (i == std::string::npos) return keys;
    i += 8;
    while (i < info.size() && info[i] != ']') {
        if (info[i] == '"') {
            std::string k;
            size_t end = 0;
            core_json::json_decode_string(info, i, k, end);
            keys.push_back(k);
            i = end;
        } else {
            ++i;
        }
    }
    return keys;
}

// Decodes one compact record into [[key, value], ...] JSON lists.
std::string compact_to_pairs(const uint8_t * p, const std::vector<std::string> & keys, size_t & n_analyses) {
    const uint32_t n = rd32(p), n_strings = rd32(p + 4);
    const bool wide = p[8] != 0;
    p += 12;
    std::vector<std::string> strs(n_strings);
    for (uint32_t i = 0; i < n_strings; ++i) {
        const uint8_t kind = p[0];
        const uint32_t len = rd32(p + 1);
        const std::string text(reinterpret_cast<const char *>(p + 5), len);
        strs[i] = kind == 1 ? text : "\"" + core_json::json_escape_strict(text) + "\"";
        p += 5 + len;
    }
    std::string out = "[";
    for (uint32_t a = 0; a < n; ++a) {
        if (a) out += ',';
        out += '[';
        const uint32_t nf = rd16(p);
        p += 2;
        for (uint32_t f = 0; f < nf; ++f) {
            uint32_t key, ref;
            if (wide) {
                key = rd16(p);
                ref = rd32(p + 2);
                p += 6;
            } else {
                key = p[0];
                ref = rd16(p + 1);
                if (ref == 0xFFFF) ref = 0xFFFFFFFFu;
                p += 3;
            }
            if (f) out += ',';
            out += "[\"" + core_json::json_escape_strict(keys.at(key)) + "\",";
            out += ref == 0xFFFFFFFFu ? std::string("null") : strs.at(ref);
            out += ']';
        }
        out += ']';
    }
    out += ']';
    n_analyses += n;
    return out;
}

int backoff_of(const std::string & s) {
    const char * names[] = { "NONE", "NOAN_ALL", "NOAN_PROP", "ADD_ALL", "ADD_PROP" };
    for (int i = 0; i < 5; ++i)
        if (s == names[i]) return i;
    return -1;
}

// Converts the analysis objects to [[key, value], ...] pair lists so key order
// survives any JSON reader. The input is our own JSON (objects of scalars).
std::string objects_to_pairs(const std::string & j) {
    std::string out;
    out.reserve(j.size() + j.size() / 4);
    size_t i = 0;
    bool in_obj = false, expect_key = false;
    while (i < j.size()) {
        const char c = j[i];
        if (c == '"') {
            size_t end = i;
            std::string tmp;
            core_json::json_decode_string(j, i, tmp, end);
            if (in_obj && expect_key) out += '[';
            out.append(j, i, end - i);
            i = end;
            continue;
        }
        if (c == '{') {
            out += '[';
            in_obj = true;
            expect_key = true;
        } else if (c == '}') {
            if (j[i - 1] != '{') out += ']';
            out += ']';
            in_obj = false;
        } else if (c == ':' && in_obj) {
            out += ',';
            expect_key = false;
        } else if (c == ',' && in_obj) {
            out += "],";
            expect_key = true;
        } else {
            out += c;
        }
        ++i;
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    std::string manifest, db, sha, jsonl, out_path, backoff = "NONE";
    std::vector<std::string> words;
    bool info = false, compact = false, dedup = false;
    int repeat = 1;
    size_t batch = 512;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--manifest")
            manifest = next();
        else if (a == "--db")
            db = next();
        else if (a == "--sha256")
            sha = next();
        else if (a == "--backoff")
            backoff = next();
        else if (a == "--jsonl")
            jsonl = next();
        else if (a == "--out")
            out_path = next();
        else if (a == "--info")
            info = true;
        else if (a == "--compact")
            compact = true;
        else if (a == "--dedup")
            dedup = true;
        else if (a == "--repeat")
            repeat = std::max(1, std::atoi(next().c_str()));
        else if (a == "--batch")
            batch = size_t(std::max(1, std::atoi(next().c_str())));
        else
            words.push_back(a);
    }
    const int bo = backoff_of(backoff);
    if ((manifest.empty() == db.empty()) || sha.empty() || bo < 0) {
        std::fprintf(stderr,
                     "usage: crispembed-ar-morph (--manifest M | --db D) --sha256 HEX [--backoff MODE] [--info]\n"
                     "       [--jsonl words.jsonl --out out.jsonl [--compact [--dedup] [--batch N]] [--repeat N]]\n"
                     "       [WORD...]\n");
        return 2;
    }
    char err[512];
    const double t0 = now_s();
    crispembed_ar_morph * m = manifest.empty()
                                  ? crispembed_ar_morph_open_db(db.c_str(), sha.c_str(), bo, err, sizeof err)
                                  : crispembed_ar_morph_open(manifest.c_str(), sha.c_str(), bo, err, sizeof err);
    const double load_s = now_s() - t0;
    if (!m) {
        std::fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    if (info) std::printf("%s\n", crispembed_ar_morph_info(m));
    const long rss_after_load = current_rss_kib();

    for (const std::string & w : words) {
        char * json = nullptr;
        size_t len = 0;
        const int rc = crispembed_ar_morph_analyze(m, w.data(), w.size(), &json, &len, err, sizeof err);
        if (rc != CRISPEMBED_AR_MORPH_OK) {
            std::printf("error %d: %s\n", rc, err);
            continue;
        }
        std::printf("%s\n", json);
        crispembed_ar_morph_string_free(json);
    }

    if (!jsonl.empty()) {
        std::ifstream in(jsonl, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "cannot read %s\n", jsonl.c_str());
            return 1;
        }
        std::vector<std::string> list;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::string w;
            size_t end = 0;
            if (!core_json::json_decode_string(line, 0, w, end)) {
                std::fprintf(stderr, "bad JSON string line\n");
                return 1;
            }
            list.push_back(w);
        }
        FILE * out = out_path.empty() ? stdout : std::fopen(out_path.c_str(), "wb");
        if (!out) return 1;
        size_t n_analyses = 0;
        double analyze_s = 0;
        std::vector<double> run_s;
        const std::vector<std::string> keys = info_keys(crispembed_ar_morph_info(m));
        for (int run = 0; run < repeat; ++run) {
            const bool write = run == 0;
            double secs = 0;
            if (compact) {
                for (size_t start = 0; start < list.size(); start += batch) {
                    const size_t end = std::min(list.size(), start + batch);
                    std::string concat;
                    std::vector<uint32_t> lens;
                    for (size_t i = start; i < end; ++i) {
                        concat += list[i];
                        lens.push_back(uint32_t(list[i].size()));
                    }
                    uint8_t * buf = nullptr;
                    size_t len = 0;
                    const double a0 = now_s();
                    const int rc = crispembed_ar_morph_analyze_compact(
                        m, concat.data(), lens.data(), uint32_t(lens.size()), dedup ? CRISPEMBED_AR_MORPH_COMPACT_DEDUP : 0,
                        &buf, &len, err, sizeof err);
                    secs += now_s() - a0;
                    if (rc != CRISPEMBED_AR_MORPH_OK) {
                        std::fprintf(stderr, "compact analysis failed %d: %s\n", rc, err);
                        return 1;
                    }
                    if (write) {
                        const uint8_t * p = buf + 8;
                        for (size_t i = start; i < end; ++i) {
                            const uint32_t status = rd32(p), bytes = rd32(p + 4);
                            p += 8;
                            if (status == CRISPEMBED_AR_MORPH_OK)
                                std::fprintf(out, "{\"i\":%zu,\"ok\":true,\"analyses\":%s}\n", i,
                                             compact_to_pairs(p, keys, n_analyses).c_str());
                            else
                                std::fprintf(out, "{\"i\":%zu,\"ok\":false,\"status\":%u,\"error\":\"rejected\"}\n", i,
                                             status);
                            p += bytes;
                        }
                    }
                    crispembed_ar_morph_buffer_free(buf);
                }
            } else {
                for (size_t i = 0; i < list.size(); ++i) {
                    char * json = nullptr;
                    size_t len = 0;
                    const double a0 = now_s();
                    const int rc =
                        crispembed_ar_morph_analyze(m, list[i].data(), list[i].size(), &json, &len, err, sizeof err);
                    secs += now_s() - a0;
                    if (!write) {
                        if (rc == CRISPEMBED_AR_MORPH_OK) crispembed_ar_morph_string_free(json);
                        continue;
                    }
                    if (rc == CRISPEMBED_AR_MORPH_OK) {
                        const std::string body(json, len);
                        for (size_t k = 0; k + 1 < body.size(); ++k)
                            if (body[k] == '}' && body[k + 1] == ',') ++n_analyses;
                        if (body.size() > 2) ++n_analyses;
                        std::fprintf(out, "{\"i\":%zu,\"ok\":true,\"analyses\":%s}\n", i, objects_to_pairs(body).c_str());
                        crispembed_ar_morph_string_free(json);
                    } else {
                        std::fprintf(out, "{\"i\":%zu,\"ok\":false,\"status\":%d,\"error\":\"%s\"}\n", i, rc,
                                     core_json::json_escape_strict(err).c_str());
                    }
                }
            }
            run_s.push_back(secs);
        }
        std::sort(run_s.begin(), run_s.end());
        analyze_s = run_s[run_s.size() / 2];
        if (out != stdout) std::fclose(out);
        std::fprintf(stderr,
                     "{\"load_s\":%.3f,\"words\":%zu,\"analyses\":%zu,\"analyze_s\":%.3f,\"words_per_s\":%.1f,"
                     "\"rss_after_load_kib\":%ld,\"rss_after_analysis_kib\":%ld,\"peak_rss_kib\":%ld,"
                     "\"runs\":%d,\"mode\":\"%s\"}\n",
                     load_s, list.size(), n_analyses, analyze_s, analyze_s > 0 ? double(list.size()) / analyze_s : 0.0,
                     rss_after_load, current_rss_kib(), peak_rss_kib(), repeat, compact ? "compact" : "json");
    }
    crispembed_ar_morph_free(m);
    return 0;
}
