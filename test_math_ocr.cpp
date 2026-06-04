// Quick test for the ggml graph decoder
#include "src/math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Minimal: create a simple gray test image (white background, dark text simulation)
static std::vector<uint8_t> make_test_image(int w, int h) {
    std::vector<uint8_t> img(w * h, 255); // white
    // Draw a simple dark cross pattern in the center
    for (int y = h/3; y < 2*h/3; y++) img[y * w + w/2] = 30;
    for (int x = w/3; x < 2*w/3; x++) img[h/2 * w + x] = 30;
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", argv[1]);
    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded OK\n");

    // Use a synthetic test image (grayscale)
    const int W = 224, H = 224;
    auto img = make_test_image(W, H);
    fprintf(stderr, "Test image: %dx%d grayscale\n", W, H);

    // Convert to float [0..1]
    std::vector<float> gray(W * H);
    for (int i = 0; i < W * H; i++) gray[i] = img[i] / 255.0f;

    int out_len = 0;
    const char* result = math_ocr_recognize(ctx, gray.data(), W, H, &out_len);

    if (result && out_len > 0) {
        printf("OCR result (%d chars): %s\n", out_len, result);
    } else {
        fprintf(stderr, "OCR returned no result (len=%d)\n", out_len);
    }

    math_ocr_free(ctx);
    fprintf(stderr, "Done.\n");
    return 0;
}
