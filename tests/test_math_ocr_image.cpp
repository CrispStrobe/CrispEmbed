// tests/test_math_ocr_image.cpp — run math OCR on an external image file.
// Usage: test_math_ocr_image <model.gguf> <image.bin> <width> <height>
//   image.bin: raw grayscale float32, row-major, [0..1]
#include "math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.bin> <width> <height>\n", argv[0]);
        return 1;
    }

    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int w = atoi(argv[3]), h = atoi(argv[4]);
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }

    std::vector<float> pixels(w * h);
    fread(pixels.data(), sizeof(float), w * h, f);
    fclose(f);

    fprintf(stderr, "Image: %dx%d\n", w, h);
    int len = 0;
    const char* result = math_ocr_recognize(ctx, pixels.data(), w, h, &len);
    if (result) {
        // Clean BPE space tokens for display
        std::string clean;
        for (int i = 0; i < len; i++) {
            if ((unsigned char)result[i] == 0xc4 && i + 1 < len && (unsigned char)result[i+1] == 0xa0) {
                clean += ' '; i++; // Ġ = U+0120 = 0xC4 0xA0 in UTF-8
            } else {
                clean += result[i];
            }
        }
        printf("LaTeX: %s\n", clean.c_str());
    } else {
        printf("Result: NULL\n");
    }

    math_ocr_free(ctx);
    return 0;
}
