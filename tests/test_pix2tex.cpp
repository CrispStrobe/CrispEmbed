// test_pix2tex.cpp — Test pix2tex math OCR on raw f32 images.
// Usage: ./test-pix2tex model.gguf image.f32 WxH
#include "math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.gguf image.f32 WxH\n", argv[0]);
        return 1;
    }
    int W, H;
    sscanf(argv[3], "%dx%d", &W, &H);

    auto * ctx = math_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    std::vector<float> pixels(W * H);
    FILE * f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }
    fread(pixels.data(), sizeof(float), W * H, f);
    fclose(f);

    int len = 0;
    const char * result = math_ocr_recognize(ctx, pixels.data(), W, H, &len);
    if (result) fprintf(stderr, "LaTeX (%d): %s\n", len, result);
    else fprintf(stderr, "(no result)\n");

    math_ocr_free(ctx);
    return result ? 0 : 1;
}
