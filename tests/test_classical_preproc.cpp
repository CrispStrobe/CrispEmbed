// tests/test_classical_preproc.cpp — unit + live tests for classical preprocessing.
//
// Unit tests: synthetic images with known properties (skew, speckles, gradient).
// Live tests: real document images (if available).
//
// Usage:
//   ./test-classical-preproc [image.png]   (live test on a real image)
//   ./test-classical-preproc               (unit tests only)

#include "classical_preproc.h"
#include "morph_fast.h"
#include "cc_detect.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ANSI colors
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

// ── Synthetic image helpers ────────────────────────────────────────

static std::vector<uint8_t> make_white(int w, int h) {
    return std::vector<uint8_t>(w * h, 255);
}

static void draw_hline(std::vector<uint8_t> & img, int w, int y, int x0, int x1, uint8_t val) {
    for (int x = x0; x < x1 && x < w; x++)
        if (y >= 0 && y < (int)img.size() / w) img[y * w + x] = val;
}

static void add_speckles(std::vector<uint8_t> & img, int w, int h, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        int x = rand() % w, y = rand() % h;
        img[y * w + x] = 0; // single dark pixel
    }
}

static void add_gradient(std::vector<uint8_t> & img, int w, int h, int dark_left, int bright_right) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float f = (float)x / w;
            int bg = (int)(dark_left + f * (bright_right - dark_left));
            int val = std::min(255, (int)img[y * w + x] + (bg - 255));
            img[y * w + x] = (uint8_t)std::max(0, val);
        }
}

// ── Unit tests ─────────────────────────────────────────────────────

static void test_adaptive_otsu() {
    printf("\n=== Adaptive Otsu binarization ===\n");
    int w = 400, h = 300;

    // Test 1: uniform image → should binarize cleanly
    auto img = make_white(w, h);
    for (int y = 100; y < 120; y++) draw_hline(img, w, y, 50, 350, 20);
    std::vector<uint8_t> out(w * h);
    adaptive_otsu(img.data(), w, h, 0, 0, 0, out.data());
    int dark = 0;
    for (int i = 0; i < w*h; i++) if (out[i] == 0) dark++;
    check("uniform: text detected", dark > 3000 && dark < 10000);

    // Test 2: gradient background → adaptive should still find text
    auto grad_img = make_white(w, h);
    for (int y = 100; y < 120; y++) draw_hline(grad_img, w, y, 50, 350, 20);
    add_gradient(grad_img, w, h, 100, 250); // dark left, bright right
    adaptive_otsu(grad_img.data(), w, h, 32, 32, 3, out.data());
    // Count dark pixels in the text region
    int text_dark = 0;
    for (int y = 100; y < 120; y++)
        for (int x = 50; x < 350; x++)
            if (out[y * w + x] == 0) text_dark++;
    check("gradient: text found despite uneven lighting", text_dark > 2000);
}

static void test_find_skew() {
    printf("\n=== Differential square-sum deskew ===\n");
    int w = 600, h = 400;

    // Test 1: no skew → angle ≈ 0
    auto img = make_white(w, h);
    for (int line = 0; line < 5; line++) {
        int y = 60 + line * 60;
        for (int dy = 0; dy < 6; dy++)  // 6px thick to survive 4x reduction
            draw_hline(img, w, y + dy, 50, 550, 10);
    }
    float angle = 0, conf = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    find_skew_angle(img.data(), w, h, &angle, &conf);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  No skew: angle=%.3f° conf=%.1f (%.1f ms)\n", angle, conf, ms);
    check("no skew: angle near 0", fabsf(angle) < 1.0f);

    // Test 2: +3° skew → should detect ~3°
    auto skewed = make_white(w, h);
    float skew_deg = 3.0f;
    float skew_rad = skew_deg * 3.14159f / 180.0f;
    for (int line = 0; line < 5; line++) {
        int base_y = 60 + line * 60;
        for (int x = 50; x < 550; x++) {
            int y = base_y + (int)(tanf(skew_rad) * (x - w/2));
            for (int dy = 0; dy < 6; dy++)
                if (y+dy >= 0 && y+dy < h) skewed[(y+dy) * w + x] = 10;
        }
    }
    find_skew_angle(skewed.data(), w, h, &angle, &conf);
    printf("  +3° skew: detected=%.3f° (expect ~-3°) conf=%.1f\n", angle, conf);
    // Returns the deskew angle (negative of skew), so expect ~-3°
    check("+3° skew: deskew angle within 1.5°", fabsf(angle + skew_deg) < 1.5f);
}

static void test_despeckle() {
    printf("\n=== CC despeckle ===\n");
    int w = 200, h = 200;
    auto img = make_white(w, h);
    // Add text line
    for (int y = 90; y < 110; y++) draw_hline(img, w, y, 30, 170, 10);
    // Add 50 isolated speckle pixels
    add_speckles(img, w, h, 50, 42);

    std::vector<uint8_t> out(w * h);
    despeckle_gray(img.data(), w, h, 5, 5, out.data());

    // Count isolated dark pixels (should be mostly removed)
    // The text line should remain
    int text_dark = 0, speckle_dark = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            if (out[y * w + x] == 0) {
                if (y >= 88 && y <= 112 && x >= 28 && x <= 172)
                    text_dark++;
                else
                    speckle_dark++;
            }
        }
    printf("  Text pixels: %d, speckle pixels remaining: %d\n", text_dark, speckle_dark);
    check("text preserved", text_dark > 1000);
    check("speckles removed", speckle_dark < 10);
}

static void test_background_norm() {
    printf("\n=== Background normalization ===\n");
    int w = 400, h = 300;
    auto img = make_white(w, h);
    // Dark text
    for (int y = 130; y < 150; y++) draw_hline(img, w, y, 50, 350, 20);
    // Add strong gradient: left side dark (100), right side bright (250)
    add_gradient(img, w, h, 100, 250);

    std::vector<uint8_t> out(w * h);
    background_norm(img.data(), w, h, 0, 0, out.data());

    // After normalization, background should be more uniform
    // Compare std dev of background pixels (y < 120)
    double sum = 0, sum2 = 0;
    int cnt = 0;
    for (int y = 10; y < 120; y++) {
        for (int x = 10; x < w - 10; x++) {
            double v = out[y * w + x];
            sum += v; sum2 += v * v; cnt++;
        }
    }
    double mean = sum / cnt;
    double std_dev = sqrt(sum2 / cnt - mean * mean);

    double orig_sum = 0, orig_sum2 = 0;
    for (int y = 10; y < 120; y++) {
        for (int x = 10; x < w - 10; x++) {
            double v = img[y * w + x];
            orig_sum += v; orig_sum2 += v * v;
        }
    }
    double orig_std = sqrt(orig_sum2 / cnt - (orig_sum/cnt) * (orig_sum/cnt));

    printf("  Background std dev: before=%.1f, after=%.1f\n", orig_std, std_dev);
    check("background uniformity improved", std_dev < orig_std * 0.5);
}

static void test_cc_detect() {
    printf("\n=== CC text line detection ===\n");
    int w = 600, h = 400;
    auto img = make_white(w, h);
    // 5 text lines: dark characters on white background
    for (int line = 0; line < 5; line++) {
        int y = 50 + line * 60;
        // Characters: 8px wide blocks separated by 7px gaps
        for (int cx = 40; cx < 560; cx += 15) {
            for (int dy = 0; dy < 14; dy++)
                for (int dx = 0; dx < 8; dx++)
                    if (y+dy < h && cx+dx < w)
                        img[(y+dy) * w + (cx+dx)] = 30; // dark enough for Otsu
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int n = 0;
    cc_text_region * regions = cc_detect_lines(img.data(), w, h, &n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  Detected %d regions in %.1f ms (expect 5)\n", n, ms);
    check("correct number of lines", n == 5);
    if (regions) {
        // Verify regions are sorted top-to-bottom
        bool sorted = true;
        for (int i = 1; i < n; i++)
            if (regions[i].y < regions[i-1].y) sorted = false;
        check("regions sorted top-to-bottom", sorted);
    }
    cc_detect_free(regions);
}

static void test_morph_fast() {
    printf("\n=== 1-bit DWA morphology ===\n");
    int w = 200, h = 200;
    auto img = make_white(w, h);
    // Dark rectangle
    for (int y = 80; y < 120; y++)
        for (int x = 60; x < 140; x++)
            img[y * w + x] = 0;

    int wpl = 0;
    uint32_t * bits = morph_u8_to_1bit(img.data(), w, h, 128, &wpl);

    // Count foreground before
    int fg_before = 0;
    for (int i = 0; i < wpl * h; i++) {
#ifdef __GNUC__
        fg_before += __builtin_popcount(bits[i]);
#else
        uint32_t v = bits[i];
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        fg_before += (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
    }

    // Erode with 5x5 → should shrink rectangle
    uint32_t * eroded = morph_erode_brick(bits, w, h, wpl, 5, 5);
    int fg_eroded = 0;
    for (int i = 0; i < wpl * h; i++) {
#ifdef __GNUC__
        fg_eroded += __builtin_popcount(eroded[i]);
#else
        uint32_t v = eroded[i];
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        fg_eroded += (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
    }
    check("erode shrinks foreground", fg_eroded < fg_before);
    check("erode preserves some foreground", fg_eroded > 0);

    // Dilate the eroded → should grow back (open)
    uint32_t * opened = morph_dilate_brick(eroded, w, h, wpl, 5, 5);
    int fg_opened = 0;
    for (int i = 0; i < wpl * h; i++) {
#ifdef __GNUC__
        fg_opened += __builtin_popcount(opened[i]);
#else
        uint32_t v = opened[i];
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        fg_opened += (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
    }
    check("open: fg <= original", fg_opened <= fg_before);
    printf("  FG: before=%d, eroded=%d, opened=%d\n", fg_before, fg_eroded, fg_opened);

    morph_free(bits);
    morph_free(eroded);
    morph_free(opened);
}

// ── Live test on real image ────────────────────────────────────────

static void live_test(const char * path) {
    printf("\n=== Live test: %s ===\n", path);
    int w, h, ch;
    uint8_t * img = stbi_load(path, &w, &h, &ch, 1);
    if (!img) { printf("  Cannot load image\n"); return; }
    printf("  Image: %dx%d\n", w, h);

    // Deskew
    float angle = 0, conf = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    find_skew_angle(img, w, h, &angle, &conf);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Skew: %.3f° (conf=%.1f, %.1f ms)\n", angle, conf,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // CC detect
    int n = 0;
    t0 = std::chrono::high_resolution_clock::now();
    cc_text_region * regions = cc_detect_lines(img, w, h, &n);
    t1 = std::chrono::high_resolution_clock::now();
    printf("  CC detect: %d regions (%.1f ms)\n", n,
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    for (int i = 0; i < n && i < 10; i++)
        printf("    [%d] (%d,%d) %dx%d\n", i, regions[i].x, regions[i].y,
               regions[i].w, regions[i].h);
    cc_detect_free(regions);

    // Background norm
    std::vector<uint8_t> normed(w * h);
    t0 = std::chrono::high_resolution_clock::now();
    background_norm(img, w, h, 0, 0, normed.data());
    t1 = std::chrono::high_resolution_clock::now();
    printf("  Background norm: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Adaptive Otsu
    std::vector<uint8_t> binarized(w * h);
    t0 = std::chrono::high_resolution_clock::now();
    adaptive_otsu(img, w, h, 0, 0, 0, binarized.data());
    t1 = std::chrono::high_resolution_clock::now();
    printf("  Adaptive Otsu: %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    stbi_image_free(img);
}

// ── Main ───────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    printf("Classical preprocessing — unit tests\n");

    test_morph_fast();
    test_adaptive_otsu();
    test_find_skew();
    test_despeckle();
    test_background_norm();
    test_cc_detect();

    if (argc > 1) {
        for (int i = 1; i < argc; i++)
            live_test(argv[i]);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
