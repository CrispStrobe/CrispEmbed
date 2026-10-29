// Test-only F32 FAST control on IDs/masks emitted by the actual native text frontend.
// Parity probe: frozen IDs/masks (TSV: id role ids,comma mask,comma) -> <out>/<id>.f32
// Optional 4th argument: threads (default 2). CRISPEMBED_MOEDERN_FAST=1 selects the
// fast path. Prints "<id> <rows> <seconds>" per case and a "total_encode_seconds" line.
#include "moedern_family_engine.h"
#include "core/clean_exit.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
static int body(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "model.gguf cases.tsv outdir [threads]\n");
        return 2;
    }
    const int threads = argc == 5 ? std::atoi(argv[4]) : 2;
    std::string failure;
    auto * ctx = moedern_family::init(argv[1], threads, failure);
    if (!ctx) {
        std::fprintf(stderr, "init: %s\n", failure.c_str());
        return 1;
    }
    std::printf("profile %s\n", moedern_family::get_profile(ctx)->name);
    std::ifstream in(argv[2]);
    std::string line;
    int rc = 0;
    double total = 0;
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string id, role, ids_s, mask_s;
        std::getline(row, id, '\t');
        std::getline(row, role, '\t');
        std::getline(row, ids_s, '\t');
        std::getline(row, mask_s, '\t');
        std::vector<int32_t> ids;
        std::vector<uint8_t> mask;
        std::string tok;
        for (std::istringstream s(ids_s); std::getline(s, tok, ',');) ids.push_back(std::stoi(tok));
        for (std::istringstream s(mask_s); std::getline(s, tok, ',');) mask.push_back(uint8_t(std::stoi(tok)));
        std::vector<float> v;
        const auto start = std::chrono::steady_clock::now();
        const bool ok = moedern_family::encode_tokens(ctx, ids.data(), mask.data(), ids.size(), role == "query", v);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total += seconds;
        if (!ok) {
            std::fprintf(stderr, "%s: %s\n", id.c_str(), moedern_family::error(ctx));
            rc = 1;
            continue;
        }
        std::ofstream(std::string(argv[3]) + "/" + id + ".f32", std::ios::binary)
            .write((const char *)v.data(), v.size() * 4);
        std::printf("%s %zu %.6f\n", id.c_str(), v.size() / 128, seconds);
    }
    std::printf("total_encode_seconds %.6f\n", total);
    moedern_family::free(ctx);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
