// tests/test_cc_detect.cpp — test CC-based text line detector
#include "cc_detect.h"
#include <cstdio>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : nullptr;

    int w, h;
    uint8_t * gray;

    if (path) {
        int ch;
        gray = stbi_load(path, &w, &h, &ch, 1);
        if (!gray) { fprintf(stderr, "Cannot load: %s\n", path); return 1; }
        printf("Image: %dx%d (from %s)\n", w, h, path);
    } else {
        // Synthetic: 800x600, white background, 5 text lines
        w = 800; h = 600;
        gray = (uint8_t *)malloc(w * h);
        memset(gray, 255, w * h);
        // 5 horizontal text lines (dark bands)
        for (int line = 0; line < 5; line++) {
            int y0 = 80 + line * 100;
            for (int y = y0; y < y0 + 20; y++)
                for (int x = 50; x < 750; x++)
                    gray[y * w + x] = 10 + (x % 30 < 5 ? 200 : 0); // simulated chars
        }
        printf("Synthetic: %dx%d (5 text lines)\n", w, h);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    int n = 0;
    cc_text_region * regions = cc_detect_lines(gray, w, h, &n);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Detected %d regions in %.1f ms\n\n", n, ms);
    for (int i = 0; i < n; i++) {
        printf("  [%2d] (%d, %d) %dx%d\n", i, regions[i].x, regions[i].y,
               regions[i].w, regions[i].h);
    }

    cc_detect_free(regions);
    if (path) stbi_image_free(gray); else free(gray);
    return 0;
}
