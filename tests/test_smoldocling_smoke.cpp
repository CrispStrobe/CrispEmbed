// tests/test_smoldocling_smoke.cpp — Quick smoke test for SmolDocling OCR.
// Usage: ./test-smoldocling-smoke model.gguf test.png

#include "../src/smoldocling_ocr.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf image.png\n", argv[0]);
        return 1;
    }

    printf("SmolDocling smoke test\n");
    printf("  Model: %s\n  Image: %s\n\n", argv[1], argv[2]);

    smoldocling_context * ctx = smoldocling_init(argv[1], 1);
    if (!ctx) {
        fprintf(stderr, "Failed to init model\n");
        return 1;
    }

    int out_len = 0;
    const char * text = smoldocling_recognize(ctx, argv[2], &out_len);
    if (text) {
        printf("Output (%d chars):\n%s\n", out_len, text);
    } else {
        printf("Recognition returned null\n");
    }

    smoldocling_free(ctx);
    return text ? 0 : 1;
}
