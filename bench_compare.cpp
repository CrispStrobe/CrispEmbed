// bench_compare.cpp — run BOTH graph and scalar decoders on each image,
// compare logits per step with cosine similarity.
// Compile with -DDECODER_VALIDATE to enable the comparison.

#define DECODER_VALIDATE
#include "src/math_ocr.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "ggml/examples/stb_image.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.png> [image2 ...]\n", argv[0]);
        return 1;
    }

    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx);
    printf("Model: enc=%dL dec=%dL D=%d vocab=%d\n\n",
           hp->enc_layers, hp->dec_layers, hp->dec_d_model, hp->vocab_size);

    for (int i = 2; i < argc; i++) {
        int w, h, ch;
        unsigned char* data = stbi_load(argv[i], &w, &h, &ch, 1);
        if (!data) { fprintf(stderr, "skip: %s\n", argv[i]); continue; }

        std::vector<float> gray(w * h);
        for (int j = 0; j < w * h; j++) gray[j] = data[j] / 255.0f;
        stbi_image_free(data);

        printf("=== %s (%dx%d) ===\n", argv[i], w, h);

        auto t0 = std::chrono::high_resolution_clock::now();
        int out_len = 0;
        const char* result = math_ocr_recognize(ctx, gray.data(), w, h, &out_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Clean BPE spaces for display
        std::string clean;
        if (result) {
            for (int j = 0; j < out_len; j++) {
                if ((unsigned char)result[j] == 0xC4 && j+1 < out_len && (unsigned char)result[j+1] == 0xA0) {
                    clean += ' ';
                    j++;
                } else {
                    clean += result[j];
                }
            }
        }
        printf("  Result: %s\n", clean.c_str());
        printf("  Time:   %.1f ms\n\n", ms);
    }

    math_ocr_free(ctx);
    return 0;
}
