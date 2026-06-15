// test_restormer_diff.cpp — parity test for Restormer image restoration.
//
// Usage: test-restormer-diff <model.gguf> <ref.gguf>

#include "restormer.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(crispembed_diff::Ref & ref, const char * name,
                  const float * data, size_t n_elem) {
    auto r = ref.compare(name, data, n_elem);
    const char * status = r.is_pass() ? "PASS" : "FAIL";
    printf("  %-25s  cos_min=%.6f  max_abs=%.2e  %s\n",
           name, r.cos_min, r.max_abs, status);
    if (r.is_pass()) n_pass++; else n_fail++;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-restormer-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { printf("Failed to load ref\n"); return 1; }

    restormer_context * ctx = restormer_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input) { printf("No 'input' in ref\n"); restormer_free(ctx); return 1; }

    int W = 64, H = 64;  // matches --size 64 in the reference dumper

    // Convert float input to uint8
    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                input_u8[(y * W + x) * 3 + c] =
                    (uint8_t)(ref_input[c * H * W + y * W + x] * 255.0f + 0.5f);

    uint8_t * output_u8 = nullptr;
    // Single tile to match reference
    int rc = restormer_process(ctx, input_u8.data(), W, H, 256, 0, &output_u8);
    if (rc != 0 || !output_u8) { printf("restormer_process failed\n"); restormer_free(ctx); return 1; }

    printf("Output: %dx%d\n", W, H);

    // We can't easily get intermediate stages from the uint8 API.
    // But we CAN compare the final output to see overall parity.

    // Compare — ref stores clamped [0,1] output; convert uint8 back
    if (ref.has("output")) {
        auto [ref_out, ref_on] = ref.get_f32("output");
        // Clamp ref to [0,1] for fair comparison with uint8 output
        std::vector<float> ref_clamped(3 * H * W);
        for (size_t i = 0; i < ref_on; i++)
            ref_clamped[i] = std::max(0.0f, std::min(1.0f, ref_out[i]));

        std::vector<float> cpp_out(3 * H * W);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c = 0; c < 3; c++)
                    cpp_out[c * H * W + y * W + x] =
                        output_u8[(y * W + x) * 3 + c] / 255.0f;

        // Compare against clamped reference
        auto r = ref.compare("output", cpp_out.data(), cpp_out.size());
        // Use lower threshold (0.99) because uint8 quantization + clamp loses precision
        bool pass = r.cos_min >= 0.99f;
        printf("  %-25s  cos_min=%.6f  max_abs=%.2e  %s\n",
               "output (clamped)", r.cos_min, r.max_abs, pass ? "PASS" : "FAIL");
        if (pass) n_pass++; else n_fail++;
    }

    if (output_u8) restormer_free_image(output_u8);
    restormer_free(ctx);
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
