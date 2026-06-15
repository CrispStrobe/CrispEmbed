// test_text_sr.cpp — test text super-resolution
//
// Usage: test-text-sr <model.gguf> [image.png]
//   Loads the SR GGUF model, runs on a synthetic or provided image,
//   verifies output dimensions and pixel range.

#include "text_sr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// stbi_load / stbi_write_png forward declarations
extern "C" {
unsigned char* stbi_load(const char* filename, int* x, int* y,
                         int* channels_in_file, int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
}

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// Generate a synthetic RGB image with text-like horizontal lines
static std::vector<uint8_t> make_text_image(int w, int h) {
    std::vector<uint8_t> img(w * h * 3, 245);  // near-white background
    for (int y = 10; y < h - 10; y += 20) {
        for (int x = 15; x < w - 15; x++) {
            for (int dy = 0; dy < 3 && y + dy < h; dy++) {
                int idx = ((y + dy) * w + x) * 3;
                img[idx + 0] = 20;  // dark text
                img[idx + 1] = 20;
                img[idx + 2] = 20;
            }
        }
    }
    return img;
}

static void test_with_model(const char* model_path, const uint8_t* pixels,
                            int w, int h) {
    printf("Loading SR model: %s\n", model_path);
    text_sr_context* ctx = text_sr_init(model_path, 2);
    CHECK(ctx != nullptr, "text_sr_init succeeds");
    if (!ctx) return;

    int r = text_sr_upscale_factor(ctx);
    printf("  upscale factor: %d\n", r);
    CHECK(r == 2 || r == 4, "upscale factor is 2 or 4");

    int expected_w = w * r;
    int expected_h = h * r;

    uint8_t* output = nullptr;
    int ow = 0, oh = 0;
    int rc = text_sr_process(ctx, pixels, w, h, 0, 0, &output, &ow, &oh);
    CHECK(rc == 0, "text_sr_process returns 0");
    CHECK(output != nullptr, "output is non-null");
    CHECK(ow == expected_w, "output width matches expected");
    CHECK(oh == expected_h, "output height matches expected");

    if (output && ow > 0 && oh > 0) {
        // Verify pixel range
        int min_val = 255, max_val = 0;
        for (int i = 0; i < ow * oh * 3; i++) {
            if (output[i] < min_val) min_val = output[i];
            if (output[i] > max_val) max_val = output[i];
        }
        printf("  output pixel range: [%d, %d]\n", min_val, max_val);
        CHECK(max_val > 0, "output has non-zero pixels");
        CHECK(max_val - min_val > 10, "output has reasonable dynamic range");

        // Verify that the output is not all the same color
        double mean_r = 0, mean_g = 0, mean_b = 0;
        int n = ow * oh;
        for (int i = 0; i < n; i++) {
            mean_r += output[i * 3 + 0];
            mean_g += output[i * 3 + 1];
            mean_b += output[i * 3 + 2];
        }
        mean_r /= n; mean_g /= n; mean_b /= n;
        printf("  output mean RGB: (%.1f, %.1f, %.1f)\n", mean_r, mean_g, mean_b);

        // Compute variance to ensure it's not a flat image
        double var = 0;
        for (int i = 0; i < n; i++) {
            double d = output[i * 3] - mean_r;
            var += d * d;
        }
        var /= n;
        printf("  output variance (R channel): %.1f\n", var);
        CHECK(var > 1.0, "output has spatial variance (not flat)");
    }

    if (output) text_sr_free_image(output);
    text_sr_free(ctx);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: test-text-sr <model.gguf> [image.png]\n");
        printf("  Runs text super-resolution on a synthetic or provided image.\n");
        return 1;
    }

    const char* model_path = argv[1];

    if (argc >= 3) {
        // Use provided image
        const char* image_path = argv[2];
        int w = 0, h = 0, c = 0;
        uint8_t* pixels = stbi_load(image_path, &w, &h, &c, 3);
        if (!pixels) {
            printf("Failed to load %s\n", image_path);
            return 1;
        }
        printf("Loaded %s: %dx%d\n", image_path, w, h);
        test_with_model(model_path, pixels, w, h);
        stbi_image_free(pixels);
    } else {
        // Synthetic test
        int w = 128, h = 64;
        printf("Using synthetic %dx%d text image\n", w, h);
        auto img = make_text_image(w, h);
        test_with_model(model_path, img.data(), w, h);
    }

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
