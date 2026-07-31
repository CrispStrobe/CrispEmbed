#include "easyocr_ocr.h"
#include "stb_image.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <easyocr.gguf> <image>\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, ch = 0;
    unsigned char * px = stbi_load(argv[2], &w, &h, &ch, 0);
    if (!px) return 3;
    easyocr_ocr_context * ctx = easyocr_ocr_init(argv[1], 4);
    if (!ctx) {
        stbi_image_free(px);
        return 4;
    }
    int n = 0;
    const char * text = easyocr_ocr_recognize(ctx, px, w, h, ch, &n);
    if (!text) {
        easyocr_ocr_free(ctx);
        stbi_image_free(px);
        return 5;
    }
    printf("%s\n", text);
    easyocr_ocr_free(ctx);
    stbi_image_free(px);
    return n >= 0 ? 0 : 6;
}
