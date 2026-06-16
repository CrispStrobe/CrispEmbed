// tests/test_scunet_diff.cpp — SCUNet per-stage parity via crispembed-diff.
// Usage: ./test-scunet-diff model.gguf ref.gguf [detail-ref.gguf]

#include "scunet_denoise.h"
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
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf> [detail-ref.gguf]\n", argv[0]); return 1; }

    printf("SCUNet Swin-Conv-UNet — per-stage parity test\n");
    printf("  Model: %s\n  Ref:   %s\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    // Optional detailed reference for swin block internals
    crispembed_diff::Ref detail;
    bool has_detail = (argc >= 4 && detail.load(argv[3]));
    if (has_detail) printf("  Detail: %s\n", argv[3]);
    printf("\n");

    scunet_context * ctx = scunet_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    auto [ref_in, ref_n] = ref.get_f32("input");
    if (!ref_in || ref_n < 3) { scunet_free(ctx); return 1; }
    auto in_shape = ref.shape("input");
    const int W = (int)in_shape[0], H = (int)in_shape[1];
    printf("  Input: %dx%d\n\n", W, H);

    std::vector<float> output(3 * H * W);
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = scunet_process_float_debug(ctx, ref_in, W, H, output.data(),
        [&](const char * name, const float * data, int n) {
            // Compare with main reference
            if (ref.has(name)) {
                auto r = ref.compare(name, data, n);
                printf("  %-12s cos=%.6f max_abs=%.6f  %s\n",
                       name, r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
            }
            // Compare with detail reference (swin internals)
            if (has_detail && detail.has(name)) {
                auto r = detail.compare(name, data, n);
                printf("  %-12s cos=%.6f max_abs=%.6f  %s  (detail)\n",
                       name, r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
            }
        });
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check("process returns 0", ret == 0);
    printf("  Inference: %.1f ms\n\n", ms);

    printf("=== Final output ===\n");
    auto r = ref.compare("output", output.data(), 3 * H * W);
    printf("  output: cos=%.6f max_abs=%.6f  %s\n",
           r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
    check("output cos >= 0.999", r.is_pass(0.999f));

    scunet_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
