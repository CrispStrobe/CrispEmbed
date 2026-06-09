// test_ocr_detect.cpp — smoke test for DBNet text detection.
//
// Usage: test-ocr-detect model.gguf [image.png]

#include "ocr_detect.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dbnet.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argc > 2 ? argv[2] : nullptr;

    // Load model
    ocr_detect::context* ctx = nullptr;
    if (!ocr_detect::load(&ctx, model_path, 4)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    if (image_path) {
        // Detect from file
        auto boxes = ocr_detect::detect_file(ctx, image_path);
        printf("Detected %zu text regions:\n", boxes.size());
        for (size_t i = 0; i < boxes.size(); i++) {
            auto& b = boxes[i];
            printf("  [%zu] (%.0f, %.0f)-(%.0f, %.0f) score=%.3f\n",
                   i, b.x, b.y, b.x + b.w, b.y + b.h, b.score);
        }

        // Print prob map stats
        int ph, pw;
        const float* pm = ocr_detect::get_prob_map(ctx, &ph, &pw);
        if (pm) {
            float min_v = 1, max_v = 0;
            int above_thresh = 0;
            for (int i = 0; i < ph * pw; i++) {
                if (pm[i] < min_v) min_v = pm[i];
                if (pm[i] > max_v) max_v = pm[i];
                if (pm[i] > 0.3f) above_thresh++;
            }
            printf("Prob map: %dx%d, range [%.4f, %.4f], %d pixels > 0.3\n",
                   pw, ph, min_v, max_v, above_thresh);
        }
    } else {
        // Just test loading
        printf("Model loaded successfully. Pass an image to test detection.\n");
    }

    ocr_detect::free(ctx);
    return 0;
}
