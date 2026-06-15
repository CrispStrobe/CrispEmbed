// tests/test_safmn_diff.cpp — SAFMN parity test via crispembed-diff harness.
//
// Usage:
//   ./test-safmn-diff safmn-x4-f32.gguf safmn-ref.gguf

#include "safmn_sr.h"
#include "crispembed_diff.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    printf("SAFMN Super-Resolution — parity test\n");
    printf("  Model: %s\n", argv[1]);
    printf("  Ref:   %s\n\n", argv[2]);

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        fprintf(stderr, "Failed to load reference\n");
        return 1;
    }

    // Load model
    safmn_context * ctx = safmn_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int scale = safmn_get_scale(ctx);
    printf("  Scale: %dx\n\n", scale);

    // Create same input as reference (deterministic random 64x64 RGB)
    const int W = 64, H = 64;
    srand(42);
    std::vector<float> input_chw(3 * H * W);
    // Match numpy random seed 42
    // The reference uses np.random.seed(42); np.random.rand(64, 64, 3)
    // We need the same data — read it from the reference GGUF instead
    auto [ref_input, ref_input_n] = ref.get_f32("input");
    if (!ref_input || ref_input_n != 3 * H * W) {
        fprintf(stderr, "Reference missing 'input' tensor (need %d, got %zu)\n",
                3 * H * W, ref_input_n);
        safmn_free(ctx);
        return 1;
    }
    memcpy(input_chw.data(), ref_input, 3 * H * W * sizeof(float));

    // Run C++ forward pass
    int out_h = H * scale, out_w = W * scale;
    std::vector<float> output_chw(3 * out_h * out_w);

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = safmn_process_float(ctx, input_chw.data(), W, H, output_chw.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    check("safmn_process_float returns 0", ret == 0);
    printf("  Inference: %.1f ms (64x64 → %dx%d)\n\n", ms, out_w, out_h);

    // Compare output
    printf("=== Output comparison ===\n");
    auto r_out = ref.compare("output", output_chw.data(), 3 * out_h * out_w);
    printf("  output: cos=%.6f max_abs=%.6f  %s\n",
           r_out.cos_min, r_out.max_abs,
           r_out.is_pass(0.999f) ? "PASS" : "FAIL");
    check("output cos >= 0.999", r_out.is_pass(0.999f));

    char msg[128];
    snprintf(msg, sizeof(msg), "output max_abs < 0.01 (got %.6f)", r_out.max_abs);
    check(msg, r_out.max_abs < 0.01f);

    // Check output range is reasonable (SR output should be in similar range as input)
    float out_min = 1e9f, out_max = -1e9f;
    for (int i = 0; i < 3 * out_h * out_w; i++) {
        if (output_chw[i] < out_min) out_min = output_chw[i];
        if (output_chw[i] > out_max) out_max = output_chw[i];
    }
    printf("  Output range: [%.4f, %.4f]\n", out_min, out_max);
    check("output not all zeros", out_max > 0.01f);

    safmn_free(ctx);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
