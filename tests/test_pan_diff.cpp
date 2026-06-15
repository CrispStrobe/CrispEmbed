// test_pan_diff.cpp — parity test for PAN whole-image super-resolution.
//
// Usage: test-pan-diff <model.gguf> <ref.gguf>

#include "pan_sr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
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
        printf("Usage: test-pan-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { printf("Failed to load ref\n"); return 1; }

    pan_sr_context * ctx = pan_sr_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    auto [ref_input, ref_n] = ref.get_f32("input");
    if (!ref_input) { printf("No 'input' in ref\n"); pan_sr_free(ctx); return 1; }

    int W = 32, H = 32, scale = pan_sr_scale(ctx);
    int ow = W * scale, oh = H * scale;

    // Convert float input to uint8 for the API
    std::vector<uint8_t> input_u8(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                input_u8[(y * W + x) * 3 + c] =
                    (uint8_t)(ref_input[c * H * W + y * W + x] * 255.0f + 0.5f);

    uint8_t * output_u8 = nullptr;
    int rw = 0, rh = 0;
    // Single tile (no tiling) to match reference
    int rc = pan_sr_process(ctx, input_u8.data(), W, H, 256, 0, &output_u8, &rw, &rh);
    if (rc != 0 || !output_u8) { printf("pan_sr_process failed\n"); pan_sr_free(ctx); return 1; }
    printf("Output: %dx%d (scale=%d)\n", rw, rh, scale);

    // Compare clamped output: ref stores output clamped to [0,1], we convert
    // uint8 back to [0,1] range.
    if (ref.has("output")) {
        std::vector<float> cpp_out(3 * oh * ow);
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                for (int c = 0; c < 3; c++)
                    cpp_out[c * oh * ow + y * ow + x] =
                        output_u8[(y * ow + x) * 3 + c] / 255.0f;
        check(ref, "output", cpp_out.data(), cpp_out.size());
    }

    if (output_u8) pan_sr_free_image(output_u8);
    pan_sr_free(ctx);
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
