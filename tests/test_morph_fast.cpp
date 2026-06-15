// tests/test_morph_fast.cpp — benchmark 1-bit morph vs float morph
#include "morph_fast.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Float-based morph from scan_cleanup (copy for comparison)
static void min_pool_2d(const float *src, int w, int h, int k, float *dst) {
    int half = k/2;
    std::vector<float> tmp(w*h);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float mn=1.0f; int x0=std::max(0,x-half),x1=std::min(w-1,x+half);
        for (int xx=x0;xx<=x1;xx++) mn=std::min(mn,src[y*w+xx]); tmp[y*w+x]=mn;
    }
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float mn=1.0f; int y0=std::max(0,y-half),y1=std::min(h-1,y+half);
        for (int yy=y0;yy<=y1;yy++) mn=std::min(mn,tmp[yy*w+x]); dst[y*w+x]=mn;
    }
}
static void max_pool_2d(const float *src, int w, int h, int k, float *dst) {
    int half = k/2;
    std::vector<float> tmp(w*h);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float mx=0.0f; int x0=std::max(0,x-half),x1=std::min(w-1,x+half);
        for (int xx=x0;xx<=x1;xx++) mx=std::max(mx,src[y*w+xx]); tmp[y*w+x]=mx;
    }
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
        float mx=0.0f; int y0=std::max(0,y-half),y1=std::min(h-1,y+half);
        for (int yy=y0;yy<=y1;yy++) mx=std::max(mx,tmp[yy*w+x]); dst[y*w+x]=mx;
    }
}

int main() {
    // Simulate a 2000x3000 document image (typical A4 at 300dpi)
    const int W = 2000, H = 3000;
    const int K = 51; // typical background whitening kernel

    printf("Image: %dx%d, kernel: %d\n\n", W, H, K);

    // Create synthetic image: white background with some dark text
    std::vector<float> gray(W * H, 1.0f);
    for (int y = 100; y < 130; y++)
        for (int x = 200; x < 1800; x++)
            gray[y * W + x] = 0.1f; // dark text line

    // ── Float morph (current scan_cleanup approach) ──
    std::vector<float> eroded(W * H), background(W * H);
    auto t0 = std::chrono::high_resolution_clock::now();
    min_pool_2d(gray.data(), W, H, K, eroded.data());
    max_pool_2d(eroded.data(), W, H, K, background.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double float_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Float morph open: %.1f ms\n", float_ms);

    // ── 1-bit morph (new fast approach) ──
    int wpl = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    uint32_t * bits = morph_float_to_1bit(gray.data(), W, H, 0.5f, &wpl);
    uint32_t * opened = morph_open_brick(bits, W, H, wpl, K, K);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bit_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("1-bit morph open: %.1f ms\n", bit_ms);

    printf("\nSpeedup: %.1fx\n", float_ms / bit_ms);

    // Memory comparison
    size_t float_mem = 2 * W * H * sizeof(float); // eroded + background
    size_t bit_mem = 2 * wpl * H * sizeof(uint32_t); // bits + opened
    printf("Float memory: %.1f MB\n", float_mem / 1e6);
    printf("1-bit memory: %.1f MB (%.1fx less)\n", bit_mem / 1e6,
           (double)float_mem / bit_mem);

    // Verify correctness: check that opened mask matches expected
    // (text region should be eroded away by large kernel)
    int fg_count = 0;
    for (int y = 0; y < H; y++) {
        const uint32_t * line = opened + y * wpl;
        for (int x = 0; x < W; x++) {
            if ((line[x >> 5] >> (31 - (x & 31))) & 1) fg_count++;
        }
    }
    printf("\nForeground pixels after open: %d (expect 0 for K=%d > text height 30)\n",
           fg_count, K);

    morph_free(bits);
    morph_free(opened);
    return 0;
}
