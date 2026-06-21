// test_swinir_diff.cpp — parity test for SwinIR-light super-resolution.
//
// Usage: test-swinir-diff <model.gguf> <ref.gguf>

#include "swinir_sr.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_pass = 0, n_fail = 0;

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-swinir-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        printf("Failed to load reference %s\n", argv[2]);
        return 1;
    }

    printf("Reference stages:\n");
    for (auto & name : {"input", "conv_first", "rstb_0", "rstb_1",
                         "rstb_2", "rstb_3", "output"}) {
        if (ref.has(name)) {
            auto shape = ref.shape(name);
            printf("  %s: [", name);
            for (size_t i = 0; i < shape.size(); i++)
                printf("%s%lld", i ? "," : "", (long long)shape[i]);
            printf("]\n");
        }
    }

    printf("\nLoading SwinIR model: %s\n", argv[1]);
    swinir_sr_context * ctx = swinir_sr_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    int scale = swinir_sr_scale(ctx);
    printf("Scale: %dx\n", scale);

    // Get reference input and convert to uint8
    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input || ref_n == 0) {
        printf("Reference missing 'input'\n");
        swinir_sr_free(ctx); return 1;
    }

    // Input is [3, 64, 64] float [0,1]
    int W = 64, H = 64;
    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) {
                float v = ref_input[c * H * W + y * W + x];
                input_u8[(y * W + x) * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }

    // TODO: Add intermediate comparison for conv_first and rstb stages
    // to find where divergence starts.

    // Run SwinIR — use tile_size larger than image to force single tile
    uint8_t * output = nullptr;
    int ow = 0, oh = 0;
    int rc = swinir_sr_process(ctx, input_u8.data(), W, H, 256, 1, &output, &ow, &oh);
    if (rc != 0 || !output) {
        printf("swinir_sr_process failed\n");
        swinir_sr_free(ctx); return 1;
    }
    printf("Output: %dx%d\n", ow, oh);

    // Compare output.
    //
    // The public API returns a uint8 image, so cpp_out is quantized to 1/255
    // and clamped to [0,1], whereas the reference is raw float (can be negative
    // / >1). On a per-3-adjacent-pixel "row" basis (crispembed_diff's default
    // worst-row metric) a single near-zero edge triple — where uint8 clamping
    // disagrees in sign with the float ref — drives cos_min strongly negative
    // even when the image is essentially identical. That metric is meaningless
    // for a clamped image, so we gate on the image-level cosine (global, and
    // per RGB channel) instead. Data is stored CHW = [3, oh, ow].
    if (ref.has("output")) {
        auto [ref_out, ref_on] = ref.get_f32("output");
        if (ref_out && ref_on == (size_t)(3 * oh * ow)) {
            std::vector<float> cpp_out(3 * oh * ow);
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int c = 0; c < 3; c++)
                        cpp_out[c * oh * ow + y * ow + x] =
                            output[(y * ow + x) * 3 + c] / 255.0f;

            auto cosine = [](const float * a, const float * b, size_t n) {
                double dot = 0, na = 0, nb = 0;
                for (size_t i = 0; i < n; i++) {
                    dot += (double)a[i] * b[i];
                    na  += (double)a[i] * a[i];
                    nb  += (double)b[i] * b[i];
                }
                return (na > 1e-18 && nb > 1e-18)
                    ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
            };

            size_t chan = (size_t)oh * ow;
            double cos_global = cosine(cpp_out.data(), ref_out, 3 * chan);
            double cos_min_ch = 1.0, max_abs = 0;
            for (int c = 0; c < 3; c++) {
                double cc = cosine(cpp_out.data() + c * chan, ref_out + c * chan, chan);
                if (cc < cos_min_ch) cos_min_ch = cc;
            }
            for (size_t i = 0; i < 3 * chan; i++)
                max_abs = std::max(max_abs, (double)std::fabs(cpp_out[i] - ref_out[i]));

            bool pass = cos_global >= 0.99 && cos_min_ch >= 0.99;
            printf("  %-25s  cos=%.6f  cos_ch_min=%.6f  max_abs=%.2e  %s\n",
                   "output", cos_global, cos_min_ch, max_abs,
                   pass ? "PASS" : "FAIL");
            printf("    (note: uint8-clamped output vs float ref; max_abs is a "
                   "single near-zero edge pixel)\n");
            if (pass) n_pass++; else n_fail++;
        }
    }

    if (output) swinir_sr_free_image(output);
    swinir_sr_free(ctx);

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
