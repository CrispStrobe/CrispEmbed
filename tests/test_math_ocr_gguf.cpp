// tests/test_math_ocr_gguf.cpp — smoke test for math_ocr GGUF loading + inference.
#include "math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }

    printf("Loading model: %s\n", argv[1]);
    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model.\n"); return 1; }

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx);
    printf("Model loaded:\n");
    printf("  Encoder: %dL, %dH, hidden=%d\n", hp->enc_layers, hp->enc_heads, hp->enc_hidden);
    printf("  Decoder: %dL, %dH, d_model=%d, vocab=%d\n",
           hp->dec_layers, hp->dec_heads, hp->dec_d_model, hp->vocab_size);

    // Gray test image: white with a dark bar
    int S = hp->image_size;
    std::vector<float> img(S * S, 0.8f);
    for (int y = S/2 - 2; y < S/2 + 2; y++)
        for (int x = S/4; x < 3*S/4; x++)
            img[y * S + x] = 0.1f;

    printf("\nRunning OCR...\n");
    int out_len = 0;
    const char* result = math_ocr_recognize(ctx, img.data(), S, S, &out_len);
    if (result) {
        int show = out_len < 80 ? out_len : 80;
        printf("Result (%d chars): \"%.80s%s\"\n", out_len, result, out_len > 80 ? "..." : "");
    } else {
        printf("Result: NULL\n");
    }

    math_ocr_free(ctx);
    printf("Done.\n");
    return 0;
}
