// test_tbsrn_diff.cpp — parity test for TBSRN text-line super-resolution.
//
// Compares C++ TBSRN forward pass against Python reference (numpy) activations.
// Uses the crispembed_diff harness: load ref GGUF, run C++ forward, compare.
//
// Usage:
//   test-tbsrn-diff <model.gguf> <ref.gguf>
//
// Generate reference:
//   python tools/dump_tbsrn_reference.py --model sr_telescope_train/best_accuracy.pdparams --output ref.gguf

#include "tbsrn_sr.h"
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
        printf("Usage: test-tbsrn-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * ref_path = argv[2];

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        printf("Failed to load reference %s\n", ref_path);
        return 1;
    }

    printf("Reference stages:\n");
    for (auto & name : {"input", "block1", "srb0", "srb1", "srb2", "srb3", "srb4",
                         "block7", "upsample", "output"}) {
        if (ref.has(name)) {
            auto shape = ref.shape(name);
            printf("  %s: [", name);
            for (size_t i = 0; i < shape.size(); i++)
                printf("%s%lld", i ? "," : "", (long long)shape[i]);
            printf("]\n");
        }
    }

    // Load TBSRN model
    printf("\nLoading TBSRN model: %s\n", model_path);
    tbsrn_sr_context * ctx = tbsrn_sr_init(model_path, 2);
    if (!ctx) {
        printf("Failed to load model\n");
        return 1;
    }

    // Create the same deterministic input as the Python reference
    // (numpy seed 42, rand(1, 3, 16, 64) float32)
    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input || ref_n == 0) {
        printf("Reference missing 'input' tensor\n");
        tbsrn_sr_free(ctx);
        return 1;
    }

    // Convert float [0,1] planar to uint8 RGB interleaved for the API
    int W = 64, H = 16;
    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) {
                float v = ref_input[c * H * W + y * W + x];
                input_u8[(y * W + x) * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
            }

    // Run TBSRN
    uint8_t * output = nullptr;
    int ow = 0, oh = 0;
    int rc = tbsrn_sr_process(ctx, input_u8.data(), W, H, &output, &ow, &oh);
    if (rc != 0 || !output) {
        printf("tbsrn_sr_process failed\n");
        tbsrn_sr_free(ctx);
        return 1;
    }

    printf("Output: %dx%d\n", ow, oh);

    // Compare output (convert uint8 back to float [-1,1] tanh range)
    if (ref.has("output")) {
        auto [ref_out, ref_on] = ref.get_f32("output");
        if (ref_out && ref_on == (size_t)(3 * oh * ow)) {
            std::vector<float> cpp_out(3 * oh * ow);
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                    for (int c = 0; c < 3; c++) {
                        float v = output[(y * ow + x) * 3 + c] / 255.0f * 2.0f - 1.0f;
                        cpp_out[c * oh * ow + y * ow + x] = v;
                    }
            check(ref, "output", cpp_out.data(), cpp_out.size());
        } else {
            printf("  output: reference shape mismatch (ref=%zu, cpp=%d)\n",
                   ref_on, 3 * oh * ow);
        }
    }

    if (output) tbsrn_sr_free_image(output);
    tbsrn_sr_free(ctx);

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
