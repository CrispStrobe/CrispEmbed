// test_bttr.cpp — Smoke test for BTTR handwritten math OCR.
// Usage: ./test-bttr model.gguf [image.f32 WxH]

#include "bttr_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> create_test_image(int w, int h) {
    // Simple "26" (white on black) matching CROHME style
    std::vector<float> img(w * h, 0.0f);
    int cx = w / 2, cy = h / 2;
    auto set = [&](int y, int x) {
        if (y >= 0 && y < h && x >= 0 && x < w) img[y*w + x] = 1.0f;
    };
    // "2"
    for (int x = cx-15; x < cx-5; x++) { set(cy-10, x); set(cy, x); set(cy+10, x); }
    for (int y = cy-10; y < cy; y++) { set(y, cx-5); }
    for (int y = cy; y < cy+10; y++) { set(y, cx-15); }
    // "6"
    for (int x = cx+5; x < cx+15; x++) { set(cy-10, x); set(cy, x); set(cy+10, x); }
    for (int y = cy-10; y < cy+10; y++) { set(y, cx+5); }
    for (int y = cy; y < cy+10; y++) { set(y, cx+15); }
    return img;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.f32 WxH]\n", argv[0]);
        return 1;
    }

    bttr_ocr_context * ctx = bttr_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    int W, H;
    std::vector<float> img;

    if (argc >= 4) {
        sscanf(argv[3], "%dx%d", &W, &H);
        img.resize(W * H);
        FILE * f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }
        fread(img.data(), sizeof(float), W*H, f);
        fclose(f);
    } else {
        W = 76; H = 56;
        img = create_test_image(W, H);
    }

    int len = 0;
    const char * result = bttr_ocr_recognize(ctx, img.data(), W, H, &len);
    if (result) fprintf(stderr, "LaTeX (%d): %s\n", len, result);
    else fprintf(stderr, "(no result)\n");

    bttr_ocr_free(ctx);
    return result ? 0 : 1;
}
