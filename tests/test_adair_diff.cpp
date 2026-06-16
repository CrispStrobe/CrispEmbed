// tests/test_adair_diff.cpp — AdaIR parity via crispembed-diff.
#include "adair.h"
#include "crispembed_diff.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    if (c) { printf("  %s[PASS]%s %s\n", GREEN, RESET, n); n_pass++; }
    else   { printf("  %s[FAIL]%s %s\n", RED, RESET, n); n_fail++; }
}
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf ref.gguf\n", argv[0]); return 1; }
    printf("AdaIR — parity test\n  Model: %s\n  Ref: %s\n\n", argv[1], argv[2]);
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;
    adair_context * ctx = adair_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;
    auto [ri, rn] = ref.get_f32("input");
    if (!ri) { adair_free(ctx); return 1; }
    auto sh = ref.shape("input");
    int W = (int)sh[0], H = (int)sh[1];
    printf("  Input: %dx%d\n", W, H);
    std::vector<float> out(3 * H * W);
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = adair_process_float(ctx, ri, W, H, out.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    check("process returns 0", ret == 0);
    printf("  Inference: %.1f ms\n\n", std::chrono::duration<double,std::milli>(t1-t0).count());
    auto r = ref.compare("output", out.data(), 3*H*W);
    printf("  output: cos=%.6f max_abs=%.6f  %s\n",
           r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
    check("output cos >= 0.999", r.is_pass(0.999f));
    adair_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
