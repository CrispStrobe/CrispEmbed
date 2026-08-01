#include "core/clean_exit.h"
#include "ppocrv6_ocr.h"

#include <cstdio>

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rec.gguf> <image>\n", argv[0]);
        return 1;
    }
    int w = 0, h = 0, ch = 0;
    auto * pixels = stbi_load(argv[2], &w, &h, &ch, 3);
    if (!pixels) return 1;
    auto * ctx = ppocrv6_ocr_init(argv[1], 1);
    if (!ctx) {
        stbi_image_free(pixels);
        return 1;
    }
    int n = 0;
    const char * text = ppocrv6_ocr_recognize_raw(ctx, pixels, w, h, 3, &n);
    printf("text=%s\n", text ? text : "<null>");
    ppocrv6_ocr_free(ctx);
    stbi_image_free(pixels);
    core_util::clean_exit(text ? 0 : 1);
}
