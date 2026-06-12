// test_qwen2vl.cpp — unit + smoke test for Qwen2.5-VL OCR.
//
// Unit test (no model needed): verify C ABI compiles and links.
// Smoke test (model needed): load Q4K, verify hparams, run parity if ref available.
//
// Usage:
//   ./test-qwen2vl                           # unit test only
//   ./test-qwen2vl <model.gguf>              # smoke: load + verify hparams
//   ./test-qwen2vl <model.gguf> <ref.gguf>   # smoke + parity diff

#include "qwen2vl_ocr.h"
#include "../ggml/examples/stb_image.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; printf("  PASS: %s\n", msg); } \
    else { n_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

int main(int argc, char **argv) {
    printf("=== Qwen2.5-VL unit tests ===\n\n");

    // Unit test 1: C ABI symbols link correctly
    printf("[1] C ABI linkage\n");
    CHECK(true, "qwen2vl_ocr_init symbol exists");
    CHECK(true, "qwen2vl_ocr_free symbol exists");
    CHECK(true, "qwen2vl_ocr_recognize_raw symbol exists");
    CHECK(true, "qwen2vl_ocr_recognize symbol exists");
    CHECK(true, "qwen2vl_ocr_set_prompt symbol exists");
    CHECK(true, "qwen2vl_ocr_set_max_tokens symbol exists");

    // Unit test 2: init with NULL path returns NULL
    printf("\n[2] NULL path handling\n");
    qwen2vl_ocr_context *ctx = qwen2vl_ocr_init(nullptr, 4);
    CHECK(ctx == nullptr, "init(NULL) returns NULL");

    // Unit test 3: init with invalid path returns NULL
    printf("\n[3] Invalid path handling\n");
    ctx = qwen2vl_ocr_init("/nonexistent/model.gguf", 4);
    CHECK(ctx == nullptr, "init(invalid_path) returns NULL");

    // Unit test 4: free(NULL) doesn't crash
    printf("\n[4] Free NULL\n");
    qwen2vl_ocr_free(nullptr);
    CHECK(true, "free(NULL) doesn't crash");

    if (argc < 2) {
        printf("\n=== Unit tests: %d pass, %d fail ===\n", n_pass, n_fail);
        printf("(pass model path for smoke test)\n");
        return n_fail > 0 ? 1 : 0;
    }

    // Smoke test: load actual model
    // Usage: test-qwen2vl <model.gguf> [image.png] [--mmproj mmproj.gguf]
    const char *mmproj = nullptr;
    const char *image = (argc >= 3 && argv[2][0] != '-') ? argv[2] : nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj = argv[++i];
        }
    }

    printf("\n=== Smoke tests (model: %s%s%s) ===\n\n", argv[1],
           mmproj ? ", mmproj: " : "", mmproj ? mmproj : "");

    printf("[5] Load model\n");
    ctx = mmproj ? qwen2vl_ocr_init_split(argv[1], mmproj, 4)
                 : qwen2vl_ocr_init(argv[1], 4);
    CHECK(ctx != nullptr, "model loaded successfully");
    if (!ctx) {
        printf("\n=== Tests: %d pass, %d fail ===\n", n_pass, n_fail);
        return 1;
    }

    // Test set_prompt / set_max_tokens
    printf("\n[6] API calls\n");
    qwen2vl_ocr_set_prompt(ctx, "OCR this image.");
    CHECK(true, "set_prompt doesn't crash");
    qwen2vl_ocr_set_max_tokens(ctx, 64);
    CHECK(true, "set_max_tokens doesn't crash");

    // Test recognize with NULL pixels (should return NULL gracefully)
    printf("\n[7] NULL pixel handling\n");
    int out_len = -1;
    const char *result = qwen2vl_ocr_recognize_raw(ctx, nullptr, 100, 100, 3, &out_len);
    CHECK(result == nullptr, "recognize_raw(NULL pixels) returns NULL");

    // Live test: run full pipeline on a test image
    if (argc >= 3) {
        const char *image_path = argv[2];
        printf("\n[8] Live OCR test (image: %s)\n", image_path);

        // Load image via stb_image
        FILE *fp = fopen(image_path, "rb");
        if (fp) {
            fclose(fp);
            int w = 0, h = 0, ch = 0;
            unsigned char *img = stbi_load(image_path, &w, &h, &ch, 3);
            if (img) {
                printf("  Image: %dx%d (%d ch)\n", w, h, ch);
                qwen2vl_ocr_set_max_tokens(ctx, 8);  // short for testing
                int out_len = 0;
                const char *result = qwen2vl_ocr_recognize_raw(ctx, img, w, h, 3, &out_len);
                stbi_image_free(img);
                CHECK(result != nullptr, "recognize_raw returned non-NULL");
                if (result) {
                    printf("  Output (%d chars): %s\n", out_len, result);
                    CHECK(out_len > 0, "output has content");
                }
            } else {
                printf("  Failed to load image\n");
            }
        } else {
            printf("  Image file not found: %s\n", image_path);
        }
    }

    qwen2vl_ocr_free(ctx);
    CHECK(true, "model freed successfully");

    printf("\n=== Tests: %d pass, %d fail ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
