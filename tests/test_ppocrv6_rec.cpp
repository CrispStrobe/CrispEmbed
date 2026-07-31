#include "ppocrv6_ocr.h"

#include "../ggml/examples/stb_image.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL IMAGE\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, ch = 0;
    auto * pixels = stbi_load(argv[2], &w, &h, &ch, 3);
    if (!pixels) return 3;
    auto * ctx = ppocrv6_ocr_init(argv[1], 1);
    if (!ctx) { stbi_image_free(pixels); return 4; }
    int len = 0;
    const char * text = ppocrv6_ocr_recognize_raw(ctx, pixels, w, h, 3, &len);
    std::printf("text=%.*s\n", len, text ? text : "");
    ppocrv6_ocr_free(ctx);
    stbi_image_free(pixels);
    return text ? 0 : 5;
}
