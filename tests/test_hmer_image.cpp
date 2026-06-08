// tests/test_hmer_image.cpp — run HMER OCR on an external image file.
// Usage: test-hmer-image <model.gguf> <image.f32> <WxH>
//   image.f32: raw grayscale float32, row-major, [0..1]
#include "hmer_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.f32> <WxH>\n", argv[0]);
        return 1;
    }

    hmer_ocr_context* ctx = hmer_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Parse WxH
    int w = 0, h = 0;
    if (sscanf(argv[3], "%dx%d", &w, &h) != 2) {
        fprintf(stderr, "Bad dimensions: %s (expected WxH)\n", argv[3]);
        return 1;
    }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }

    std::vector<float> pixels(w * h);
    size_t read = fread(pixels.data(), sizeof(float), w * h, f);
    fclose(f);
    if ((int)read != w * h) {
        fprintf(stderr, "Read %zu floats, expected %d\n", read, w * h);
        return 1;
    }

    int len = 0;
    const char* result = hmer_ocr_recognize(ctx, pixels.data(), w, h, &len);
    if (result) {
        printf("LaTeX: %s\n", result);
    } else {
        printf("LaTeX: [NULL]\n");
    }

    hmer_ocr_free(ctx);
    return 0;
}
