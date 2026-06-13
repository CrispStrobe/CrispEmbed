// test_mixtex_diff.cpp — MixTex encoder parity test.
// Usage: test-mixtex-diff <model.gguf> <ref.gguf>
//
// Sets g_diff_ref so the engine's inline diff_compare calls fire.
// The engine compares enc_embed, enc_stage_0..3, enc_output at each
// stage during the forward pass.

#include "mixtex_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>

// Defined in mixtex_ocr.cpp
extern crispembed_diff::Ref* g_diff_ref;

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    // Load reference
    printf("Loading reference: %s\n", argv[2]);
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        fprintf(stderr, "Failed to load reference\n");
        return 1;
    }
    printf("Reference tensors:\n");
    for (auto &name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf("  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++)
            printf("%s%lld", i ? "," : "", (long long)s[i]);
        printf("]\n");
    }

    // Set global diff ref so engine compares inline
    g_diff_ref = &ref;

    // Load model
    printf("\nLoading model: %s\n", argv[1]);
    mixtex_ocr_context *ctx = mixtex_ocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Run on synthetic image (same as ref dumper's default)
    const mixtex_ocr_hparams *hp = mixtex_ocr_get_hparams(ctx);
    int w = hp->image_w, h = hp->image_h;
    printf("\nSynthetic image: %dx%d\n", w, h);

    // Generate same synthetic image as Python dumper:
    // White background with black lines (simple "x^2" shape)
    std::vector<uint8_t> img(w * h * 3, 255);  // white
    // Horizontal line
    for (int y = h/2-2; y < h/2+2; y++)
        for (int x = w/4; x < 3*w/4; x++)
            img[(y*w+x)*3+0] = img[(y*w+x)*3+1] = img[(y*w+x)*3+2] = 0;
    // Vertical bar
    for (int y = h/3; y < 2*h/3; y++)
        for (int x = w/2-2; x < w/2+2; x++)
            img[(y*w+x)*3+0] = img[(y*w+x)*3+1] = img[(y*w+x)*3+2] = 0;
    // Small superscript
    for (int y = h/3-5; y < h/3+5; y++)
        for (int x = w/2+20; x < w/2+30; x++)
            img[(y*w+x)*3+0] = img[(y*w+x)*3+1] = img[(y*w+x)*3+2] = 0;

    int out_len;
    printf("\nRunning OCR...\n");
    const char *result = mixtex_ocr_recognize(ctx, img.data(), w, h, 3, &out_len);
    printf("Result (%d chars): %s\n", out_len, result ? result : "(null)");

    g_diff_ref = nullptr;
    mixtex_ocr_free(ctx);
    printf("\nDone.\n");
    return 0;
}
