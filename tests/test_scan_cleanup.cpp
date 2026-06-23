// test_scan_cleanup.cpp — test scan cleanup operations on synthetic images
//
// Usage: test-scan-cleanup [image.png]
//   No args: runs synthetic tests (no model needed)
//   With arg: processes a real image and writes cleaned output

#include "scan_cleanup.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// Generate a synthetic grayscale image with horizontal text-like lines
static std::vector<uint8_t> make_text_image(int w, int h) {
    std::vector<uint8_t> img(w * h, 255);  // white background
    // Draw dark horizontal lines (simulating text)
    for (int y = 20; y < h - 20; y += 30) {
        for (int x = 30; x < w - 30; x++) {
            if (y + 5 < h) {
                for (int dy = 0; dy < 3; dy++) {
                    img[(y + dy) * w + x] = 30;  // dark text
                }
            }
        }
    }
    return img;
}

// Generate an image with dark borders (simulating scanner border)
static std::vector<uint8_t> make_bordered_image(int w, int h, int border) {
    std::vector<uint8_t> img(w * h, 200);  // light gray content
    // Add text lines
    for (int y = border + 20; y < h - border - 20; y += 25) {
        for (int x = border + 20; x < w - border - 20; x++) {
            img[y * w + x] = 40;
        }
    }
    // Dark borders
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (y < border || y >= h - border || x < border || x >= w - border) {
                img[y * w + x] = 10;  // very dark
            }
        }
    }
    return img;
}

// Generate a skewed image by rotating content
static std::vector<uint8_t> make_skewed_image(int w, int h, float angle_deg) {
    std::vector<uint8_t> img(w * h, 255);
    float rad = angle_deg * (float)M_PI / 180.0f;
    float cx = w / 2.0f, cy = h / 2.0f;
    float cos_a = cosf(rad), sin_a = sinf(rad);

    // Draw rotated horizontal lines
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Inverse rotate to find source
            float dx = x - cx, dy = y - cy;
            float sx = cos_a * dx + sin_a * dy + cx;
            float sy = -sin_a * dx + cos_a * dy + cy;

            // Draw text lines every 30 pixels in source space
            int isy = (int)sy;
            if (sx > 40 && sx < w - 40 && isy > 20 && isy < h - 20) {
                int line_pos = (isy - 20) % 30;
                if (line_pos < 3) {
                    img[y * w + x] = 30;
                }
            }
        }
    }
    return img;
}

// Generate an image with uneven background
static std::vector<uint8_t> make_uneven_bg_image(int w, int h) {
    std::vector<uint8_t> img(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Gradient background (simulating uneven lighting)
            float bg = 180.0f + 60.0f * sinf((float)x / w * 3.14f)
                              + 40.0f * cosf((float)y / h * 3.14f);
            bg = std::min(255.0f, std::max(0.0f, bg));

            // Add text lines
            int line_pos = (y - 20) % 30;
            if (line_pos < 3 && x > 20 && x < w - 20 && y > 20 && y < h - 20) {
                img[y * w + x] = (uint8_t)(bg * 0.15f);  // text is dark relative to bg
            } else {
                img[y * w + x] = (uint8_t)bg;
            }
        }
    }
    return img;
}

static void test_otsu() {
    printf("\n=== Otsu binarization ===\n");

    // Bimodal image: half black, half white
    const int w = 100, h = 100;
    std::vector<float> gray(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            gray[y * w + x] = (x < w / 2) ? 0.2f : 0.8f;
        }
    }

    float t = scan_cleanup_otsu(gray.data(), w, h);
    printf("  Otsu threshold for bimodal image: %.3f\n", t);
    // Otsu finds the optimal threshold — any value in (0.2, 0.8) separates the modes
    CHECK(t > 0.19f && t < 0.81f, "threshold between modes");
}

static void test_sauvola() {
    printf("\n=== Sauvola binarization ===\n");

    const int w = 200, h = 200;
    auto img = make_uneven_bg_image(w, h);

    // Convert to float
    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) gray[i] = img[i] / 255.0f;

    std::vector<float> bin(w * h);
    scan_cleanup_sauvola(gray.data(), w, h, 25, 0.2f, bin.data());

    // Count black and white pixels
    int n_black = 0, n_white = 0;
    for (int i = 0; i < w * h; i++) {
        if (bin[i] < 0.5f) n_black++; else n_white++;
    }
    printf("  Sauvola: %d black, %d white pixels\n", n_black, n_white);
    CHECK(n_black > 0 && n_white > 0, "both black and white pixels present");
    CHECK(n_white > n_black, "more white (background) than black (text)");
}

static void test_border_crop() {
    printf("\n=== Border crop ===\n");

    const int w = 300, h = 400, border = 30;
    auto img = make_bordered_image(w, h, border);

    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) gray[i] = img[i] / 255.0f;

    int x0, y0, x1, y1;
    scan_cleanup_find_content_rect(gray.data(), w, h, 0.15f, &x0, &y0, &x1, &y1);
    printf("  Detected content rect: (%d,%d) - (%d,%d)\n", x0, y0, x1, y1);
    printf("  Expected ~(%d,%d) - (%d,%d)\n", border, border, w - border - 1, h - border - 1);

    CHECK(x0 >= border - 5 && x0 <= border + 5, "left border detected");
    CHECK(y0 >= border - 5 && y0 <= border + 5, "top border detected");
    CHECK(x1 >= w - border - 5 && x1 <= w - border + 5, "right border detected");
    CHECK(y1 >= h - border - 5 && y1 <= h - border + 5, "bottom border detected");
}

static void test_deskew() {
    printf("\n=== Deskew ===\n");

    const int w = 400, h = 300;
    float test_angle = 3.0f;

    auto img = make_skewed_image(w, h, test_angle);

    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) gray[i] = img[i] / 255.0f;

    float detected = scan_cleanup_detect_angle(gray.data(), w, h, 15.0f);
    printf("  Applied skew: %.1f deg, detected: %.1f deg\n", test_angle, detected);
    CHECK(fabsf(detected - test_angle) < 2.0f, "detected angle within 2 degrees");

    // Regression: axis-aligned (un-skewed) text must NOT trigger a spurious
    // rotation. The old single-max-bin Hough reported ~4 deg on clean
    // horizontal text, which deskew then "corrected" — distorting the page and
    // shifting content into the wrong VLM vision-grid cells.
    auto flat = make_text_image(w, h);
    std::vector<float> fgray(w * h);
    for (int i = 0; i < w * h; i++) fgray[i] = flat[i] / 255.0f;
    float flat_angle = scan_cleanup_detect_angle(fgray.data(), w, h, 15.0f);
    printf("  Axis-aligned text detected: %.2f deg (expect ~0)\n", flat_angle);
    CHECK(fabsf(flat_angle) < 1.0f, "no spurious skew on axis-aligned text");

    // And a larger genuine skew is still detected/corrected.
    auto img8 = make_skewed_image(w, h, 8.0f);
    std::vector<float> g8(w * h);
    for (int i = 0; i < w * h; i++) g8[i] = img8[i] / 255.0f;
    float d8 = scan_cleanup_detect_angle(g8.data(), w, h, 15.0f);
    printf("  Applied skew: 8.0 deg, detected: %.1f deg\n", d8);
    CHECK(fabsf(d8 - 8.0f) < 2.0f, "8 deg skew still detected");
}

static void test_background_whiten() {
    printf("\n=== Background whitening ===\n");

    const int w = 200, h = 200;
    auto img = make_uneven_bg_image(w, h);

    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) gray[i] = img[i] / 255.0f;

    // Measure background variance before
    float mean_before = 0;
    int n_bg = 0;
    for (int y = 20; y < h - 20; y++) {
        int line_pos = (y - 20) % 30;
        if (line_pos >= 3) {
            for (int x = 20; x < w - 20; x++) {
                mean_before += gray[y * w + x];
                n_bg++;
            }
        }
    }
    mean_before /= n_bg;

    float var_before = 0;
    for (int y = 20; y < h - 20; y++) {
        int line_pos = (y - 20) % 30;
        if (line_pos >= 3) {
            for (int x = 20; x < w - 20; x++) {
                float d = gray[y * w + x] - mean_before;
                var_before += d * d;
            }
        }
    }
    var_before /= n_bg;

    // Whiten
    std::vector<float> whitened(w * h);
    scan_cleanup_whiten(gray.data(), w, h, 51, whitened.data());

    // Measure background variance after
    float mean_after = 0;
    n_bg = 0;
    for (int y = 20; y < h - 20; y++) {
        int line_pos = (y - 20) % 30;
        if (line_pos >= 3) {
            for (int x = 20; x < w - 20; x++) {
                mean_after += whitened[y * w + x];
                n_bg++;
            }
        }
    }
    mean_after /= n_bg;

    float var_after = 0;
    for (int y = 20; y < h - 20; y++) {
        int line_pos = (y - 20) % 30;
        if (line_pos >= 3) {
            for (int x = 20; x < w - 20; x++) {
                float d = whitened[y * w + x] - mean_after;
                var_after += d * d;
            }
        }
    }
    var_after /= n_bg;

    printf("  Background variance: before=%.6f, after=%.6f\n", var_before, var_after);
    CHECK(var_after < var_before, "background variance reduced after whitening");
}

static void test_full_pipeline() {
    printf("\n=== Full pipeline ===\n");

    const int w = 300, h = 400;
    auto img = make_bordered_image(w, h, 25);

    auto * ctx = scan_cleanup_init(nullptr, 1);
    CHECK(ctx != nullptr, "context created");

    auto params = scan_cleanup_defaults();
    params.binarize = 0;

    uint8_t * out = nullptr;
    int ow = 0, oh = 0;
    int rc = scan_cleanup_process(ctx, img.data(), w, h, 1, params, &out, &ow, &oh);
    CHECK(rc == 0, "process returned success");
    CHECK(out != nullptr, "output buffer allocated");
    CHECK(ow > 0 && oh > 0, "output dimensions valid");
    // Deskew may slightly expand, but crop should remove borders.
    // Net result should be roughly similar or smaller.
    printf("  Input: %dx%d, Output: %dx%d\n", w, h, ow, oh);
    CHECK(ow <= w + 10 && oh <= h + 10, "output dimensions reasonable");

    if (out) scan_cleanup_free_image(out);
    scan_cleanup_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        // Process a real image file
        // TODO: load via stb_image and process
        printf("Real image processing not yet implemented in test.\n");
        printf("Usage: %s   (runs synthetic tests)\n", argv[0]);
        return 1;
    }

    printf("scan_cleanup test suite\n");
    printf("=======================\n");

    test_otsu();
    test_sauvola();
    test_border_crop();
    test_deskew();
    test_background_whiten();
    test_full_pipeline();

    printf("\n=======================\n");
    printf("Results: %d PASS, %d FAIL\n", n_pass, n_fail);

    return n_fail > 0 ? 1 : 0;
}
