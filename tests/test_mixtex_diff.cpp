// test_mixtex_diff.cpp — MixTex encoder/decoder parity test.
// Usage: test-mixtex-diff <model.gguf> <ref.gguf>
//
// Sets MIXTEX_DIFF_REF (and MIXTEX_ENC_STAGE_REF / MIXTEX_DEC_REF) so the
// engine's inline diff_compare calls fire during the forward pass.

#include "mixtex_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    const char *ref_path = argv[2];

    // Verify the reference file loads
    printf("Loading reference: %s\n", ref_path);
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
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

    // Set env vars so the engine compares inline during forward pass
    // (portable: MSVC has no setenv)
#ifdef _WIN32
    _putenv_s("MIXTEX_DIFF_REF", ref_path);
    _putenv_s("MIXTEX_ENC_STAGE_REF", ref_path);
    _putenv_s("MIXTEX_DEC_REF", ref_path);
#else
    setenv("MIXTEX_DIFF_REF", ref_path, 1);
    setenv("MIXTEX_ENC_STAGE_REF", ref_path, 1);
    setenv("MIXTEX_DEC_REF", ref_path, 1);
#endif

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

    mixtex_ocr_free(ctx);

    // Clean up env vars
#ifdef _WIN32
    _putenv_s("MIXTEX_DIFF_REF", "");
    _putenv_s("MIXTEX_ENC_STAGE_REF", "");
    _putenv_s("MIXTEX_DEC_REF", "");
#else
    unsetenv("MIXTEX_DIFF_REF");
    unsetenv("MIXTEX_ENC_STAGE_REF");
    unsetenv("MIXTEX_DEC_REF");
#endif

    printf("\nDone.\n");
    return 0;
}
