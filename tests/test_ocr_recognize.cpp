// test_ocr_recognize.cpp — test TrOCR text recognition on an image file.
//
// Uses the existing math_ocr API (which supports any TrOCR model).
// Usage: test-ocr-recognize <trocr.gguf> <image.png>

#include "math_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

// stb_image for loading test images
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <trocr.gguf> <image.png>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    math_ocr_context* ctx = math_ocr_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx);
    printf("Model: enc=%dL/%d dec=%dL/%d vocab=%d\n",
           hp->enc_layers, hp->enc_hidden,
           hp->dec_layers, hp->dec_d_model,
           hp->vocab_size);
    printf("Special tokens: bos=%d eos=%d pad=%d start=%d\n",
           hp->bos_token, hp->eos_token, hp->pad_token, hp->decoder_start_token);

    // Load image
    int img_w, img_h, img_c;
    unsigned char* img = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        math_ocr_free(ctx);
        return 1;
    }
    printf("Image: %dx%d (%d channels)\n", img_w, img_h, img_c);

    int out_len = 0;
    const char* text = math_ocr_recognize_raw(ctx, img, img_w, img_h, 3, &out_len);
    stbi_image_free(img);

    if (text) {
        printf("\nRecognized text (%d chars):\n  \"%s\"\n", out_len, text);
    } else {
        printf("\nRecognition failed\n");
    }

    math_ocr_free(ctx);
    return text ? 0 : 1;
}
