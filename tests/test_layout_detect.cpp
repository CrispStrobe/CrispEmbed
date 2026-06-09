// test_layout_detect.cpp — smoke test for document layout detection.
// Usage: test-layout-detect model.gguf [image.png]

#include "layout_detect.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <layout.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    layout_detect::context* ctx = nullptr;
    if (!layout_detect::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        return 1;
    }

    printf("Model loaded successfully.\n");

    if (argc > 2) {
        auto regions = layout_detect::detect_file(ctx, argv[2]);
        printf("Detected %zu regions:\n", regions.size());
        for (size_t i = 0; i < regions.size(); i++) {
            auto& r = regions[i];
            printf("  [%zu] %s (%.0f,%.0f)-(%.0f,%.0f) score=%.3f\n",
                   i, r.label_name, r.x1, r.y1, r.x2, r.y2, r.score);
        }
    }

    layout_detect::free(ctx);
    return 0;
}
