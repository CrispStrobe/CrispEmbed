// tests/test_mixtex.cpp — basic MixTex load + recognize test
#include "mixtex_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mixtex.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    mixtex_ocr_context* ctx = mixtex_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    const mixtex_ocr_hparams* hp = mixtex_ocr_get_hparams(ctx);
    printf("Loaded: %dx%d, vocab=%d, dec_layers=%d\n",
           hp->image_w, hp->image_h, hp->vocab_size, hp->dec_layers);

    // Create synthetic formula image (white bg, black cross shape)
    int w = 200, h = 100;
    std::vector<uint8_t> img(w * h * 3, 255);
    // Horizontal line
    for (int x = 30; x < 170; x++)
        for (int dy = -1; dy <= 1; dy++)
            for (int c = 0; c < 3; c++)
                img[((50 + dy) * w + x) * 3 + c] = 0;
    // Vertical line
    for (int y = 20; y < 80; y++)
        for (int dx = -1; dx <= 1; dx++)
            for (int c = 0; c < 3; c++)
                img[(y * w + (100 + dx)) * 3 + c] = 0;

    printf("Running OCR on synthetic %dx%d...\n", w, h);
    int out_len = 0;
    const char* result = mixtex_ocr_recognize(ctx, img.data(), w, h, 3, &out_len);
    printf("Result: \"%s\" (len=%d)\n", result ? result : "(null)", out_len);

    mixtex_ocr_free(ctx);
    printf("PASS\n");
    return 0;
}
