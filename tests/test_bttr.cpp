// test_bttr.cpp — Smoke test for BTTR handwritten math OCR.
// Usage: ./test-bttr model.gguf [image.bmp | image.f32 WxH]

#include "bttr_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
        int dy = (h > 0) ? (abs_h - 1 - y) : y;
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

// stb_image (implementation in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *ch, int desired_ch);
    void stbi_image_free(void *p);
}

static bool ends_with(const char* s, const char* suffix) {
    int sl = strlen(s), sufl = strlen(suffix);
    if (sl < sufl) return false;
    return strcmp(s + sl - sufl, suffix) == 0;
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
        fprintf(stderr, "Usage: %s <model.gguf> [image.bmp | image.f32 WxH]\n", argv[0]);
        return 1;
    }

    bttr_ocr_context * ctx = bttr_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    int W, H;
    std::vector<float> img;

    if (argc >= 3 && (ends_with(argv[2], ".bmp") || ends_with(argv[2], ".BMP"))) {
        if (!load_bmp_gray(argv[2], img, W, H)) {
            fprintf(stderr, "Failed to load BMP: %s\n", argv[2]);
            bttr_ocr_free(ctx); return 1;
        }
        fprintf(stderr, "Loaded BMP: %dx%d\n", W, H);
    } else if (argc >= 3 && (ends_with(argv[2], ".png") || ends_with(argv[2], ".jpg") ||
               ends_with(argv[2], ".PNG") || ends_with(argv[2], ".JPG") ||
               ends_with(argv[2], ".jpeg") || ends_with(argv[2], ".JPEG"))) {
        int ch;
        stbi_uc* px = stbi_load(argv[2], &W, &H, &ch, 1); // load as grayscale
        if (!px) { fprintf(stderr, "Failed to load: %s\n", argv[2]); bttr_ocr_free(ctx); return 1; }
        img.resize(W * H);
        for (int i = 0; i < W * H; i++) img[i] = px[i] / 255.0f;
        stbi_image_free(px);
        fprintf(stderr, "Loaded image: %dx%d\n", W, H);
    } else if (argc >= 4) {
        // Raw float32 file
        sscanf(argv[3], "%dx%d", &W, &H);
        img.resize(W * H);
        FILE * f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }
        fread(img.data(), sizeof(float), W*H, f);
        fclose(f);
    } else {
        W = 76; H = 56;
        img = create_test_image(W, H);
        fprintf(stderr, "Using synthetic %dx%d image\n", W, H);
    }

    int len = 0;
    const char * result = bttr_ocr_recognize(ctx, img.data(), W, H, &len);
    if (result) fprintf(stderr, "LaTeX (%d): %s\n", len, result);
    else fprintf(stderr, "(no result)\n");

    bttr_ocr_free(ctx);
    return result ? 0 : 1;
}
