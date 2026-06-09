// test_ocr_pipeline.cpp — end-to-end OCR: detect + recognize.
//
// Usage: test-ocr-pipeline <dbnet.gguf> <trocr.gguf> <image.png>

#include "ocr_pipeline.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <dbnet.gguf> <trocr.gguf> <image.png>\n", argv[0]);
        return 1;
    }

    ocr_pipeline::context* ctx = nullptr;
    if (!ocr_pipeline::load(&ctx, argv[1], argv[2], 4)) {
        fprintf(stderr, "Failed to load models\n");
        return 1;
    }

    auto results = ocr_pipeline::run_file(ctx, argv[3]);

    printf("\n=== OCR Results (%zu regions) ===\n", results.size());
    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        printf("[%2zu] (%.0f,%.0f)-(%.0f,%.0f) conf=%.2f  \"%s\"\n",
               i, r.box.x, r.box.y,
               r.box.x + r.box.w, r.box.y + r.box.h,
               r.confidence, r.text.c_str());
    }

    if (results.empty()) {
        printf("(no text detected)\n");
    }

    ocr_pipeline::free(ctx);
    return 0;
}
