// test_hmer.cpp — Quick smoke test for HMER handwritten math OCR.
//
// Loads the GGUF model, creates the same synthetic test image used by
// test_hmer_parity.py, and runs inference. Compare output against the
// PyTorch reference.
//
// Build: (from build dir)
//   cmake .. && make test-hmer
// Run:
//   ./test-hmer /mnt/storage/models/hmer-hw-f32.gguf

#include "hmer_ocr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Create the same synthetic "2" image as test_hmer_parity.py
static std::vector<float> create_test_image(int width, int height) {
    std::vector<float> img(width * height, 1.0f);  // white background

    int cx = width / 2, cy = height / 2;

    auto set = [&](int y, int x) {
        if (y >= 0 && y < height && x >= 0 && x < width)
            img[y * width + x] = 0.0f;
    };

    // Top horizontal
    for (int x = cx - 10; x < cx + 10; x++) { set(cy - 15, x); set(cy - 14, x); }
    // Right vertical (top half)
    for (int y = cy - 15; y < cy; y++) { set(y, cx + 9); set(y, cx + 8); }
    // Middle horizontal
    for (int x = cx - 10; x < cx + 10; x++) { set(cy, x); set(cy + 1, x); }
    // Left vertical (bottom half)
    for (int y = cy; y < cy + 15; y++) { set(y, cx - 10); set(y, cx - 9); }
    // Bottom horizontal
    for (int x = cx - 10; x < cx + 10; x++) { set(cy + 14, x); set(cy + 15, x); }

    return img;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [threads]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int n_threads = argc > 2 ? atoi(argv[2]) : 4;

    fprintf(stderr, "Loading model: %s (threads=%d)\n", model_path, n_threads);
    hmer_ocr_context * ctx = hmer_ocr_init(model_path, n_threads);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const hmer_ocr_hparams * hp = hmer_ocr_get_hparams(ctx);
    fprintf(stderr, "Model loaded: blocks=[%d,%d,%d] vocab=%d\n",
            hp->block_config[0], hp->block_config[1], hp->block_config[2],
            hp->output_size);

    // Create test image (same as Python)
    int W = 128, H = 64;
    auto img = create_test_image(W, H);
    fprintf(stderr, "Test image: %dx%d\n", W, H);

    // Run inference
    int out_len = 0;
    const char * result = hmer_ocr_recognize(ctx, img.data(), W, H, &out_len);

    if (result) {
        fprintf(stderr, "\n=== RESULT ===\n");
        fprintf(stderr, "LaTeX (%d chars): %s\n", out_len, result);

        // Expected from PyTorch reference (corrected attention feedback):
        const char * expected = "\\log _ { 2 } ( \\frac { 5 } { 2 - 1 } ) ( \\frac { 9 x \\rightarrow - \\cos \\theta } { \\cos z } )";
        if (strcmp(result, expected) == 0) {
            fprintf(stderr, "PASS: matches PyTorch reference!\n");
        } else {
            fprintf(stderr, "MISMATCH!\n");
            fprintf(stderr, "  Expected: %s\n", expected);
            fprintf(stderr, "  Got:      %s\n", result);
        }
    } else {
        fprintf(stderr, "Recognition failed!\n");
    }

    hmer_ocr_free(ctx);
    return result ? 0 : 1;
}
