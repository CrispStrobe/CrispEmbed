// tests/test_esrgan_diff.cpp — Real-ESRGAN parity via crispembed-diff.
// Usage: ./test-esrgan-diff esrgan-x4-f32.gguf esrgan-ref.gguf

#include "esrgan_sr.h"
#include "crispembed_diff.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]); return 1; }

    printf("Real-ESRGAN (SRVGGNetCompact) — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { fprintf(stderr, "Failed to load reference\n"); return 1; }

    esrgan_context * ctx = esrgan_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;
    printf("  Scale: %dx\n\n", esrgan_get_scale(ctx));

    const int W = 64, H = 64, scale = esrgan_get_scale(ctx);
    auto [ref_in, ref_n] = ref.get_f32("input");
    if (!ref_in || ref_n != 3 * H * W) {
        fprintf(stderr, "Reference missing input\n"); esrgan_free(ctx); return 1;
    }

    int oh = H * scale, ow = W * scale;
    std::vector<float> output(3 * oh * ow);
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = esrgan_process_float(ctx, ref_in, W, H, output.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    check("process returns 0", ret == 0);
    printf("  Inference: %.1f ms (%dx%d → %dx%d)\n\n", ms, W, H, ow, oh);

    printf("=== Output comparison ===\n");
    auto r = ref.compare("output", output.data(), 3 * oh * ow);
    printf("  output: cos=%.6f max_abs=%.6f  %s\n",
           r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
    check("output cos >= 0.999", r.is_pass(0.999f));

    char msg[128];
    snprintf(msg, sizeof(msg), "output max_abs < 0.01 (got %.6f)", r.max_abs);
    check(msg, r.max_abs < 0.01f);

    float mn = 1e9f, mx = -1e9f;
    for (auto v : output) { mn = std::min(mn, v); mx = std::max(mx, v); }
    printf("  Output range: [%.4f, %.4f]\n", mn, mx);
    check("output not all zeros", mx > 0.01f);

    esrgan_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
