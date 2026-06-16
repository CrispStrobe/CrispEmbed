// test_hat_diff.cpp — parity test for HAT super-resolution.
//
// Usage: test-hat-diff <model.gguf> <ref.gguf>

#include "hat_sr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-hat-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { printf("Failed to load ref\n"); return 1; }

    hat_sr_context * ctx = hat_sr_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input) { printf("No 'input' in ref\n"); hat_sr_free(ctx); return 1; }

    auto input_shape = ref.shape("input");
    int H = (int)input_shape[1], W = (int)input_shape[2];
    int scale = hat_sr_scale(ctx);
    int ow = W * scale, oh = H * scale;

    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                input_u8[(y * W + x) * 3 + c] =
                    (uint8_t)(ref_input[c * H * W + y * W + x] * 255.0f + 0.5f);

    uint8_t * output_u8 = nullptr;
    int rw = 0, rh = 0;
    int rc = hat_sr_process(ctx, input_u8.data(), W, H, 256, 0, &output_u8, &rw, &rh);
    if (rc != 0 || !output_u8) { printf("hat_sr_process failed\n"); hat_sr_free(ctx); return 1; }
    printf("Output: %dx%d (scale=%d)\n", rw, rh, scale);

    int n_pass = 0, n_fail = 0;

    // Compare intermediate stages to isolate divergence
    for (auto & name : {"conv_first", "rhag_0", "rhag_1", "deep_features"}) {
        if (!ref.has(name)) continue;
        // These are [C, H, W] format — can't easily extract from uint8 API
        // Just note which stages we have for reference
        auto shape = ref.shape(name);
        printf("  ref stage %s: [%lld,%lld,%lld]\n", name,
               (long long)shape[0], (long long)shape[1], (long long)shape[2]);
    }

    if (ref.has("output")) {
        std::vector<float> cpp_out(3 * oh * ow);
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                for (int c = 0; c < 3; c++)
                    cpp_out[c * oh * ow + y * ow + x] =
                        output_u8[(y * ow + x) * 3 + c] / 255.0f;
        auto r = ref.compare("output", cpp_out.data(), cpp_out.size());
        bool pass = r.cos_min >= 0.99f;
        printf("  %-25s  cos_min=%.6f  max_abs=%.2e  %s\n",
               "output", r.cos_min, r.max_abs, pass ? "PASS" : "FAIL");
        if (pass) n_pass++; else n_fail++;
    }

    if (output_u8) hat_sr_free_image(output_u8);
    hat_sr_free(ctx);
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
