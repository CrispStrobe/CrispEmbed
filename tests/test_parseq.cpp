// test_parseq.cpp — basic smoke test for PARSeq scene text OCR.
// Usage: test-parseq <model.gguf> [image.png]

#include "../src/parseq_ocr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// stb_image for loading test images
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * image_path = argc > 2 ? argv[2] : nullptr;

    printf("Loading PARSeq model: %s\n", model_path);
    auto * ctx = parseq_ocr_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "Failed to init PARSeq model\n");
        return 1;
    }

    const auto * hp = parseq_ocr_get_hparams(ctx);
    printf("  embed_dim=%d enc_layers=%d heads=%d patches=%d\n",
           hp->embed_dim, hp->enc_layers, hp->enc_heads, hp->n_patches);
    printf("  vocab_size=%d max_label_len=%d\n", hp->vocab_size, hp->max_label_len);

    if (image_path) {
        printf("\nLoading image: %s\n", image_path);
        int w, h, ch;
        unsigned char * img = stbi_load(image_path, &w, &h, &ch, 3);
        if (!img) {
            fprintf(stderr, "Failed to load image: %s\n", image_path);
            parseq_ocr_free(ctx);
            return 1;
        }
        printf("  Image: %dx%d channels=%d\n", w, h, ch);

        int out_len = 0;
        const char * text = parseq_ocr_recognize_raw(ctx, img, w, h, 3, &out_len);
        stbi_image_free(img);

        if (text) {
            printf("  Recognized: '%s' (len=%d)\n", text, out_len);
        } else {
            printf("  Recognition failed!\n");
        }
    } else {
        // Synthetic test: white image with no text
        printf("\nRunning with synthetic white image (32x128)...\n");
        int w = 128, h = 32;
        std::vector<unsigned char> white(w * h * 3, 255);
        int out_len = 0;
        const char * text = parseq_ocr_recognize_raw(ctx, white.data(), w, h, 3, &out_len);
        if (text) {
            printf("  Result: '%s' (len=%d)\n", text, out_len);
        } else {
            printf("  Recognition returned null\n");
        }
    }

    parseq_ocr_free(ctx);
    printf("\nPARSeq test PASSED (model loads and runs)\n");
    return 0;
}
