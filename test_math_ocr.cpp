// Quick test for the ggml graph decoder
#include "src/math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", argv[1]);
    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    fprintf(stderr, "Model loaded OK\n");

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx);
    if (hp) {
        fprintf(stderr, "Hparams: enc=%dL dec=%dL D_enc=%d D_dec=%d vocab=%d\n",
                hp->enc_layers, hp->dec_layers, hp->enc_hidden, hp->dec_d_model, hp->vocab_size);
    }

    // Use a synthetic 224x224 grayscale image
    const int W = 224, H = 224;
    std::vector<float> gray(W * H, 0.9f); // mostly white
    // Draw a simple + sign
    for (int y = 80; y < 144; y++) gray[y * W + 112] = 0.1f;
    for (int x = 80; x < 144; x++) gray[112 * W + x] = 0.1f;

    fprintf(stderr, "Running OCR on %dx%d synthetic image...\n", W, H);
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
