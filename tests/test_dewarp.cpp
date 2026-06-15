// tests/test_dewarp.cpp — unit + live tests for page dewarping
#include "dewarp.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

// Create a synthetic curved document: 5 textlines with sinusoidal warp
static std::vector<uint8_t> make_curved_doc(int w, int h, float amplitude) {
    std::vector<uint8_t> img(w * h, 255);
    for (int line = 0; line < 5; line++) {
        int base_y = 60 + line * 60;
        for (int x = 30; x < w - 30; x++) {
            // Sinusoidal vertical displacement
            float curve = amplitude * sinf(3.14159f * x / w);
            int y = base_y + (int)curve;
            // Draw 8px-thick text line
            for (int dy = 0; dy < 8; dy++) {
                if (y + dy >= 0 && y + dy < h)
                    img[(y + dy) * w + x] = 20;
            }
        }
    }
    return img;
}

// Measure how straight the textlines are after dewarping:
// For each row of dark pixels, compute the standard deviation of their y positions
static float measure_straightness(const uint8_t * img, int w, int h, int threshold) {
    // Find horizontal runs of dark pixels (text) and measure y variation
    std::vector<float> y_positions;
    for (int y = 0; y < h; y++) {
        int dark_count = 0;
        for (int x = 0; x < w; x++)
            if (img[y * w + x] < threshold) dark_count++;
        if (dark_count > w / 4)  // mostly dark row = text
            y_positions.push_back((float)y);
    }

    if (y_positions.size() < 3) return 999.0f;

    // Group into textlines (gaps > 3px separate lines)
    std::vector<std::vector<float>> lines;
    lines.push_back({y_positions[0]});
    for (size_t i = 1; i < y_positions.size(); i++) {
        if (y_positions[i] - y_positions[i-1] > 3)
            lines.push_back({});
        lines.back().push_back(y_positions[i]);
    }

    // For each textline, measure the y spread
    float total_spread = 0;
    int count = 0;
    for (auto & l : lines) {
        if (l.size() < 2) continue;
        float mn = l.front(), mx = l.back();
        total_spread += mx - mn;
        count++;
    }
    return count > 0 ? total_spread / count : 0;
}

static void test_straight_document() {
    printf("\n=== Straight document (no curvature) ===\n");
    int w = 600, h = 400;
    auto img = make_curved_doc(w, h, 0.0f); // no curve
    std::vector<uint8_t> out(w * h);
    int ow = 0, oh = 0;

    int ret = dewarp_page(img.data(), w, h, out.data(), &ow, &oh);
    check("returns success (0) for straight doc", ret == 0);
    check("output dimensions unchanged", ow == w && oh == h);
}

static void test_curved_document() {
    printf("\n=== Curved document (sinusoidal warp) ===\n");
    int w = 600, h = 400;
    float amplitude = 20.0f;
    auto img = make_curved_doc(w, h, amplitude);
    std::vector<uint8_t> out(w * h);
    int ow = 0, oh = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = dewarp_page(img.data(), w, h, out.data(), &ow, &oh);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  Dewarp returned %d in %.1f ms\n", ret, ms);
    check("returns success (0)", ret == 0);

    // Measure straightness before and after
    float before = measure_straightness(img.data(), w, h, 128);
    float after = measure_straightness(out.data(), w, h, 128);
    printf("  Line spread: before=%.1f px, after=%.1f px\n", before, after);
    check("text lines straighter after dewarp", after < before);
    check("significant improvement (>50%)", after < before * 0.5f);
}

static void test_small_image() {
    printf("\n=== Small image (should fail gracefully) ===\n");
    int w = 50, h = 50;
    std::vector<uint8_t> img(w * h, 128);
    std::vector<uint8_t> out(w * h);
    int ow = 0, oh = 0;

    int ret = dewarp_page(img.data(), w, h, out.data(), &ow, &oh);
    check("returns failure (1) for tiny image", ret == 1);
    check("output is copy of input", memcmp(out.data(), img.data(), w * h) == 0);
}

static void test_live_image(const char * path) {
    printf("\n=== Live test: %s ===\n", path);
    int w, h, ch;
    uint8_t * img = stbi_load(path, &w, &h, &ch, 1);
    if (!img) { printf("  Cannot load image\n"); return; }
    printf("  Image: %dx%d\n", w, h);

    std::vector<uint8_t> out(w * h);
    int ow = 0, oh = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = dewarp_page(img, w, h, out.data(), &ow, &oh);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  Dewarp: ret=%d, %.1f ms\n", ret, ms);
    if (ret == 0) {
        float before = measure_straightness(img, w, h, 128);
        float after = measure_straightness(out.data(), w, h, 128);
        printf("  Line spread: before=%.1f, after=%.1f\n", before, after);
    }

    stbi_image_free(img);
}

int main(int argc, char ** argv) {
    printf("Page dewarping — unit tests\n");

    test_small_image();
    test_straight_document();
    test_curved_document();

    if (argc > 1) {
        for (int i = 1; i < argc; i++)
            test_live_image(argv[i]);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
