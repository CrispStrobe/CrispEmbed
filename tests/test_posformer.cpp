// test_posformer.cpp — Smoke test for PosFormer handwritten math OCR.
// Usage: ./test-posformer model.gguf [image.bmp]

#include "posformer_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Minimal BMP loader (uncompressed, 24-bit or 8-bit)
static bool load_bmp_gray(const char * path, std::vector<float> & gray,
                          int & w, int & h) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;

    unsigned char header[54];
    if (fread(header, 1, 54, f) != 54) { fclose(f); return false; }

    w = *(int*)&header[18];
    h = *(int*)&header[22];
    int bpp = *(short*)&header[28];
    int offset = *(int*)&header[10];
    int abs_h = h < 0 ? -h : h;

    fseek(f, offset, SEEK_SET);
    gray.resize(w * abs_h);

    int row_bytes = ((w * (bpp / 8) + 3) / 4) * 4;
    std::vector<unsigned char> row(row_bytes);

    for (int y = 0; y < abs_h; y++) {
        if (fread(row.data(), 1, row_bytes, f) != (size_t)row_bytes) {
            fclose(f); return false;
        }
        int dy = (h > 0) ? (abs_h - 1 - y) : y;  // BMP is bottom-up if h > 0
        for (int x = 0; x < w; x++) {
            if (bpp == 24 || bpp == 32) {
                int b = row[x * (bpp/8)], g = row[x*(bpp/8)+1], r = row[x*(bpp/8)+2];
                gray[dy * w + x] = (0.299f*r + 0.587f*g + 0.114f*b) / 255.0f;
            } else if (bpp == 8) {
                gray[dy * w + x] = row[x] / 255.0f;
            }
        }
    }
    h = abs_h;
    fclose(f);
    return true;
}

static std::vector<float> create_test_image(int w, int h) {
    std::vector<float> img(w * h, 0.0f);
    int cx = w / 2, cy = h / 2;
    auto set = [&](int y, int x) {
        if (y >= 0 && y < h && x >= 0 && x < w) img[y*w + x] = 1.0f;
    };
    for (int x = cx-15; x < cx-5; x++) { set(cy-10, x); set(cy, x); set(cy+10, x); }
    for (int y = cy-10; y < cy; y++) { set(y, cx-5); }
    for (int y = cy; y < cy+10; y++) { set(y, cx-15); }
    for (int x = cx+5; x < cx+15; x++) { set(cy-10, x); set(cy, x); set(cy+10, x); }
    for (int y = cy-10; y < cy+10; y++) { set(y, cx+5); }
    for (int y = cy; y < cy+10; y++) { set(y, cx+15); }
    return img;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.bmp]\n", argv[0]);
        return 1;
    }

    posformer_ocr_context * ctx = posformer_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int W, H;
    std::vector<float> img;

    if (argc >= 3) {
        if (!load_bmp_gray(argv[2], img, W, H)) {
            fprintf(stderr, "Can't load BMP: %s\n", argv[2]);
            posformer_ocr_free(ctx);
            return 1;
        }
        fprintf(stderr, "Loaded BMP: %dx%d\n", W, H);
    } else {
        W = 76; H = 56;
        img = create_test_image(W, H);
    }

    int len = 0;
    const char * result = posformer_ocr_recognize(ctx, img.data(), W, H, &len);
    if (result) {
        fprintf(stderr, "LaTeX (%d): %s\n", len, result);
        printf("%s\n", result);
    } else {
        fprintf(stderr, "(no result)\n");
    }

    posformer_ocr_free(ctx);
    return result ? 0 : 1;
}
