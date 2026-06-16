// tests/test_instructir_diff.cpp — InstructIR parity via crispembed-diff.
// Usage: ./test-instructir-diff instructir-f32.gguf instructir-ref.gguf

#include "instructir.h"
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
    printf("InstructIR — parity test (task=denoise)\n");
    printf("  Model: %s\n  Ref:   %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    instructir_context * ctx = instructir_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;
    printf("  Tasks: %d\n\n", instructir_get_n_tasks(ctx));

    auto [ref_in, ref_n] = ref.get_f32("input");
    if (!ref_in) { instructir_free(ctx); return 1; }
    auto sh = ref.shape("input");
    const int W = (int)sh[0], H = (int)sh[1];
    printf("  Input: %dx%d\n", W, H);

    std::vector<float> output(3 * H * W);
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = instructir_process_float(ctx, INSTRUCTIR_DENOISE, ref_in, W, H, output.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check("process returns 0", ret == 0);
    printf("  Inference: %.1f ms\n\n", ms);

    // Compare per-stage
    const char * stage_names[] = {"intro", "enc_0", "enc_1", "enc_2", "enc_3",
                                   "middle", "dec_0", "dec_1", "dec_2", "dec_3", "output"};
    for (auto name : stage_names) {
        if (!ref.has(name)) continue;
        // We only have output from the full forward pass, not per-stage
        // So just compare output
    }

    printf("=== Output comparison ===\n");
    auto r = ref.compare("output", output.data(), 3 * H * W);
    printf("  output: cos=%.6f max_abs=%.6f  %s\n",
           r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
    check("output cos >= 0.999", r.is_pass(0.999f));

    char msg[128];
    snprintf(msg, sizeof(msg), "output max_abs < 0.01 (got %.6f)", r.max_abs);
    check(msg, r.max_abs < 0.01f);

    instructir_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
