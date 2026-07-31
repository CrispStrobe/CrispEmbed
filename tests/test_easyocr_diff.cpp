#include "easyocr_ocr.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "stb_image.h"

#include <cstdio>
#include <string>

static int easyocr_diff_main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <easyocr.gguf> <reference.gguf> <image>\n", argv[0]);
        return 2;
    }
    easyocr_ocr_context * ctx = easyocr_ocr_init(argv[1], 1);
    if (!ctx) return 3;
    int width = 0, height = 0, channels = 0;
    unsigned char * pixels = stbi_load(argv[3], &width, &height, &channels, 0);
    if (!pixels) {
        easyocr_ocr_free(ctx);
        return 4;
    }
    int text_len = 0;
    const char * text = easyocr_ocr_recognize(ctx, pixels, width, height, channels, &text_len);
    if (!text) {
        stbi_image_free(pixels);
        easyocr_ocr_free(ctx);
        return 5;
    }
    printf("decoded=%s\n", text);
    crispembed_diff::Ref reference;
    if (!reference.load(argv[2])) {
        stbi_image_free(pixels);
        easyocr_ocr_free(ctx);
        return 6;
    }
    const std::string expected = reference.meta("easyocr.decoded");
    const bool decoded_match = expected.empty() || expected == text;
    const int failures = easyocr_ocr_diff(ctx, argv[2]);
    if (!decoded_match) fprintf(stderr, "decoded output mismatch: mine=%s ref=%s\n", text, expected.c_str());
    stbi_image_free(pixels);
    easyocr_ocr_free(ctx);
    return failures == 0 && decoded_match ? 0 : 6;
}

int main(int argc, char ** argv) {
    const int rc = easyocr_diff_main(argc, argv);
    core_util::clean_exit(rc);
    return rc;
}
