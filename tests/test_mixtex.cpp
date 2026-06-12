// tests/test_mixtex.cpp — basic MixTex load + recognize test
// Usage: ./test-mixtex <mixtex.gguf> [image.png|jpg|bmp]
#include "mixtex_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// stb_image (implementation in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *ch, int desired_ch);
    void stbi_image_free(void *p);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mixtex.gguf> [image.png|jpg|bmp]\n", argv[0]);
        return 1;
    }

    mixtex_ocr_context* ctx = mixtex_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    const mixtex_ocr_hparams* hp = mixtex_ocr_get_hparams(ctx);
    printf("Loaded: %dx%d, vocab=%d, dec_layers=%d\n",
           hp->image_w, hp->image_h, hp->vocab_size, hp->dec_layers);

    int w, h;
    std::vector<uint8_t> img;

    if (argc >= 3) {
        int ch;
        stbi_uc* px = stbi_load(argv[2], &w, &h, &ch, 3);
        if (!px) {
            fprintf(stderr, "Failed to load image: %s\n", argv[2]);
            mixtex_ocr_free(ctx); return 1;
        }
        img.assign(px, px + w * h * 3);
        stbi_image_free(px);
        printf("Loaded image: %s (%dx%d)\n", argv[2], w, h);
    } else {
        // Synthetic formula (white bg, black cross)
        w = 200; h = 100;
        img.resize(w * h * 3, 255);
        for (int x = 30; x < 170; x++)
            for (int dy = -1; dy <= 1; dy++)
                for (int c = 0; c < 3; c++)
                    img[((50 + dy) * w + x) * 3 + c] = 0;
        for (int y = 20; y < 80; y++)
            for (int dx = -1; dx <= 1; dx++)
                for (int c = 0; c < 3; c++)
                    img[(y * w + (100 + dx)) * 3 + c] = 0;
        printf("Using synthetic %dx%d image\n", w, h);
    }

    printf("Running OCR on %dx%d...\n", w, h);
    int out_len = 0;
    const char* result = mixtex_ocr_recognize(ctx, img.data(), w, h, 3, &out_len);
    printf("Result: \"%s\" (len=%d)\n", result ? result : "(null)", out_len);

    mixtex_ocr_free(ctx);
    printf("PASS\n");
    return 0;
}
