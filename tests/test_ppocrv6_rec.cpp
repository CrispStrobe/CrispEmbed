#include "core/clean_exit.h"
#include "ppocrv6_ocr.h"

#include <cstdio>

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rec.gguf> <image> [image ...]\n", argv[0]);
        return 1;
    }
    auto * ctx = ppocrv6_ocr_init(argv[1], 1);
    if (!ctx) return 1;
    bool ok = true;
    const char * text = nullptr;
    for (int i = 2; i < argc; ++i) {
        int w = 0, h = 0, ch = 0;
        auto * pixels = stbi_load(argv[i], &w, &h, &ch, 3);
        if (!pixels) {
            fprintf(stderr, "failed to load image: %s\n", argv[i]);
            ok = false;
            continue;
        }
        int n = 0;
        text = ppocrv6_ocr_recognize_raw(ctx, pixels, w, h, 3, &n);
        printf("text=%s\n", text ? text : "<null>");
        ok = ok && text != nullptr;
        stbi_image_free(pixels);
    }
    ppocrv6_ocr_free(ctx);
    core_util::clean_exit(ok && text ? 0 : 1);
}
