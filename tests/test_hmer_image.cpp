// tests/test_hmer_image.cpp — run HMER OCR on an external image file.
// Usage: test-hmer-image <model.gguf> <image.bmp | image.f32 WxH>
#include "hmer_ocr.h"
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
        if (fread(row.data(), 1, row_bytes, f) != (size_t)row_bytes) { fclose(f); return false; }
        int dy = (h > 0) ? (abs_h - 1 - y) : y;
        for (int x = 0; x < w; x++) {
            if (bpp == 24 || bpp == 32) {
                int b = row[x*(bpp/8)], g = row[x*(bpp/8)+1], r = row[x*(bpp/8)+2];
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
    return sl >= sufl && strcmp(s + sl - sufl, suffix) == 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.bmp | image.f32 WxH>\n", argv[0]);
        return 1;
    }

    hmer_ocr_context* ctx = hmer_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int w = 0, h = 0;
    std::vector<float> pixels;

    if (ends_with(argv[2], ".bmp") || ends_with(argv[2], ".BMP")) {
        if (!load_bmp_gray(argv[2], pixels, w, h)) {
            fprintf(stderr, "Failed to load BMP: %s\n", argv[2]);
            hmer_ocr_free(ctx); return 1;
        }
        fprintf(stderr, "Loaded BMP: %dx%d\n", w, h);
    } else if (ends_with(argv[2], ".png") || ends_with(argv[2], ".jpg") ||
               ends_with(argv[2], ".PNG") || ends_with(argv[2], ".JPG") ||
               ends_with(argv[2], ".jpeg")) {
        int ch;
        stbi_uc* px = stbi_load(argv[2], &w, &h, &ch, 1);
        if (!px) { fprintf(stderr, "Failed to load: %s\n", argv[2]); hmer_ocr_free(ctx); return 1; }
        pixels.resize(w * h);
        for (int i = 0; i < w * h; i++) pixels[i] = px[i] / 255.0f;
        stbi_image_free(px);
        fprintf(stderr, "Loaded image: %dx%d\n", w, h);
    } else if (argc >= 4) {
        if (sscanf(argv[3], "%dx%d", &w, &h) != 2) {
            fprintf(stderr, "Bad dimensions: %s\n", argv[3]);
            hmer_ocr_free(ctx); return 1;
        }
        FILE* f = fopen(argv[2], "rb");
        if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }
        pixels.resize(w * h);
        fread(pixels.data(), sizeof(float), w * h, f);
        fclose(f);
    } else {
        fprintf(stderr, "Usage: %s <model.gguf> <image.bmp | image.f32 WxH>\n", argv[0]);
        hmer_ocr_free(ctx); return 1;
    }

    int len = 0;
    const char* result = hmer_ocr_recognize(ctx, pixels.data(), w, h, &len);
    if (result) printf("LaTeX: %s\n", result);
    else printf("LaTeX: [NULL]\n");

    hmer_ocr_free(ctx);
    return 0;
}
