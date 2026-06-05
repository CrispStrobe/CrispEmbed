// Full-length decode on a real handwritten image
#include "src/math_ocr.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "ggml/examples/stb_image.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf image.png\n", argv[0]); return 1; }

    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int w, h, ch;
    unsigned char* data = stbi_load(argv[2], &w, &h, &ch, 1);
    if (!data) { fprintf(stderr, "Failed to load image\n"); math_ocr_free(ctx); return 1; }

    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) gray[i] = data[i] / 255.0f;
    stbi_image_free(data);

    fprintf(stderr, "Image: %dx%d → running full OCR...\n", w, h);

    int out_len = 0;
    const char* result = math_ocr_recognize(ctx, gray.data(), w, h, &out_len);

    if (result && out_len > 0) {
        // Clean BPE space tokens
        std::string clean;
        for (int i = 0; i < out_len; i++) {
            unsigned char c = result[i];
            if (c == 0xC4 && i+1 < out_len && (unsigned char)result[i+1] == 0xA0) {
                clean += ' '; i++;
            } else if (c == '<') {
                // Skip <s>, </s>, <pad>
                int j = i;
                while (j < out_len && result[j] != '>') j++;
                i = j;
            } else {
                clean += result[i];
            }
        }
        printf("Result: %s\n", clean.c_str());
    } else {
        printf("No result\n");
    }

    math_ocr_free(ctx);
    return 0;
}
