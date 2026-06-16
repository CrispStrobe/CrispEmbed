// test_swinir_diff.cpp — parity test for SwinIR-light super-resolution.
//
// Usage: test-swinir-diff <model.gguf> <ref.gguf>

#include "swinir_sr.h"
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
    if (r.is_pass()) n_pass++;
    else n_fail++;
}

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

    // Compare output
    if (ref.has("output")) {
        auto [ref_out, ref_on] = ref.get_f32("output");
        if (ref_out && ref_on == (size_t)(3 * oh * ow)) {
            std::vector<float> cpp_out(3 * oh * ow);
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int c = 0; c < 3; c++)
                        cpp_out[c * oh * ow + y * ow + x] =
                            output[(y * ow + x) * 3 + c] / 255.0f;
            // Debug: print values at stride to find non-zero region
            double ref_sum = 0, cpp_sum = 0;
            int ref_nz = 0, cpp_nz = 0;
            for (size_t k = 0; k < ref_on; k++) {
                ref_sum += ref_out[k]; cpp_sum += cpp_out[k];
                if (ref_out[k] != 0) ref_nz++;
                if (cpp_out[k] != 0) cpp_nz++;
            }
            printf("  ref: sum=%.2f, nonzero=%d/%zu, mean=%.4f\n",
                   ref_sum, ref_nz, ref_on, ref_sum/ref_on);
            printf("  cpp: sum=%.2f, nonzero=%d/%zu, mean=%.4f\n",
                   cpp_sum, cpp_nz, ref_on, cpp_sum/ref_on);
            // Sample at offset 65536 (middle of channel 0)
            size_t mid = 256*128 + 128; // y=128, x=128 in channel 0
            printf("  ref[%zu]=%.4f  cpp[%zu]=%.4f\n", mid, ref_out[mid], mid, cpp_out[mid]);
            check(ref, "output", cpp_out.data(), cpp_out.size());
        }
    }

    if (output) swinir_sr_free_image(output);
    swinir_sr_free(ctx);

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
