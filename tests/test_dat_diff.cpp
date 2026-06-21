// test_dat_diff.cpp — parity test for DAT-light super-resolution.
//
// Compares the C++ DAT forward against a GENUINE reference: the real PyTorch
// DAT-light model run on gguf-reconstructed weights (see
// tools/dump_dat_reference_from_gguf.py). Only the final output stage is
// compared via the public API (uint8). Like test_swinir_diff, the output is
// uint8-clamped vs a raw-float ref, so we gate on the image-level (global +
// per-channel) cosine, not crispembed_diff's fragile worst-row metric.
//
// Usage: test-dat-diff <model.gguf> <ref.gguf>

#include "dat_sr.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-dat-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { printf("Failed to load reference %s\n", argv[2]); return 1; }

    auto in_shape = ref.shape("input");          // [W,H,3] (gguf-reversed of [3,H,W])
    if (in_shape.size() != 3) { printf("Reference missing 'input'\n"); return 1; }
    int W = (int)in_shape[0], H = (int)in_shape[1];
    printf("Reference input: %dx%d\n", W, H);

    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input || ref_n == 0) { printf("Reference missing 'input' data\n"); return 1; }

    // [3,H,W] float [0,1] → uint8 RGB interleaved
    std::vector<uint8_t> input_u8((size_t)W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                input_u8[(y * W + x) * 3 + c] =
                    (uint8_t)(ref_input[c * H * W + y * W + x] * 255.0f + 0.5f);

    printf("Loading DAT model: %s\n", argv[1]);
    dat_sr_context * ctx = dat_sr_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    uint8_t * output = nullptr;
    int ow = 0, oh = 0;
    int rc = dat_sr_process(ctx, input_u8.data(), W, H, 256, 256, &output, &ow, &oh);
    if (rc != 0 || !output) { printf("dat_sr_process failed\n"); dat_sr_free(ctx); return 1; }
    printf("Output: %dx%d\n", ow, oh);

    int n_fail = 0;
    if (ref.has("output")) {
        auto [ref_out, ref_on] = ref.get_f32("output");
        if (ref_out && ref_on == (size_t)(3 * oh * ow)) {
            std::vector<float> cpp_out((size_t)3 * oh * ow);
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int c = 0; c < 3; c++)
                        cpp_out[c * oh * ow + y * ow + x] =
                            output[(y * ow + x) * 3 + c] / 255.0f;

            // The public API clamps to uint8 [0,1]; the DAT ref is raw float and
            // ~19% of pixels fall outside [0,1] on the seeded-random input (clamp
            // ceiling cos≈0.9965). Clamp the ref the same way so we measure
            // engine-vs-achievable, not engine-vs-unrepresentable-float.
            std::vector<float> ref_clamped((size_t)3 * oh * ow);
            for (size_t i = 0; i < (size_t)3 * oh * ow; i++)
                ref_clamped[i] = std::max(0.0f, std::min(1.0f, ref_out[i]));
            ref_out = ref_clamped.data();

            auto cosine = [](const float * a, const float * b, size_t n) {
                double dot = 0, na = 0, nb = 0;
                for (size_t i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
                return (na > 1e-18 && nb > 1e-18) ? dot / (std::sqrt(na)*std::sqrt(nb)) : 0.0;
            };
            size_t chan = (size_t)oh * ow;
            double cos_global = cosine(cpp_out.data(), ref_out, 3 * chan);
            double cos_min_ch = 1.0, max_abs = 0;
            for (int c = 0; c < 3; c++)
                cos_min_ch = std::min(cos_min_ch, cosine(cpp_out.data() + c*chan, ref_out + c*chan, chan));
            for (size_t i = 0; i < 3 * chan; i++)
                max_abs = std::max(max_abs, (double)std::fabs(cpp_out[i] - ref_out[i]));

            bool pass = cos_global >= 0.99 && cos_min_ch >= 0.99;
            printf("  %-22s  cos=%.6f  cos_ch_min=%.6f  max_abs=%.2e  %s\n",
                   "output", cos_global, cos_min_ch, max_abs, pass ? "PASS" : "FAIL");
            printf("    (note: engine uint8 vs DAT ref clamped to [0,1]; residual is uint8 quant)\n");
            if (!pass) n_fail++;
        } else {
            printf("  output: shape mismatch (ref=%zu, cpp=%d)\n", ref_on, 3 * oh * ow);
            n_fail++;
        }
    }

    dat_sr_free_image(output);
    dat_sr_free(ctx);
    printf("\n%d passed, %d failed\n", n_fail ? 0 : 1, n_fail);
    return n_fail > 0 ? 1 : 0;
}
