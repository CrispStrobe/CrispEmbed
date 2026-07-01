// test_glm_ocr_image.cpp — GLM-OCR E2E with real image.
// Usage: test-glm-ocr-image <model.gguf> [max_tokens]

#include "glm_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [max_tokens]\n", argv[0]);
        return 1;
    }
    int max_tokens = (argc > 2) ? atoi(argv[2]) : 50;

    glm_ocr::context ctx;
    if (!glm_ocr::load(ctx, argv[1], 4, 1)) return 1;

    // Synthetic gradient image (336x336)
    const int img_size = (int)ctx.m.vhp.image_size;
    std::vector<float> pixels(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float val = (float)(y * img_size + x) / (float)(img_size * img_size);
                pixels[c * img_size * img_size + y * img_size + x] =
                    (val - ctx.m.vhp.image_mean[c]) / ctx.m.vhp.image_std[c];
            }

    // Vision encode
    printf("Encoding vision...\n");
    glm_ocr::vision_result vr;
    if (!glm_ocr::encode_vision(ctx, pixels.data(), img_size, img_size, vr)) {
        fprintf(stderr, "Vision failed\n");
        glm_ocr::free_(ctx);
        return 1;
    }
    printf("  Vision: %d tokens, %d dim\n", vr.n_tokens, vr.hidden_dim);

    // Build chat prompt:
    // [gMASK]<sop>\n<|user|>\n<|begin_of_image|><|image|>*N<|end_of_image|>\nOCR this image\n<|assistant|>\n
    int32_t img_token_id = (int32_t)ctx.m.lhp.image_token_id;  // 59280
    std::vector<int32_t> prompt;

    // [gMASK]=59248, <sop>=59250
    prompt.push_back(59248);  // [gMASK]
    prompt.push_back(59250);  // <sop>
    // \n<|user|>\n
    prompt.push_back(198);    // \n (approximate — GLM tokenizer)
    prompt.push_back(59253);  // <|user|>
    prompt.push_back(198);    // \n
    // <|begin_of_image|>
    prompt.push_back(59256);  // <|begin_of_image|>
    // <|image|> * n_vision_tokens
    for (int i = 0; i < vr.n_tokens; i++)
        prompt.push_back(img_token_id);
    // <|end_of_image|>
    prompt.push_back(59257);  // <|end_of_image|>
    // \nOCR this image\n<|assistant|>\n
    prompt.push_back(198);    // \n
    // "OCR this image" — approximate token IDs
    prompt.push_back(42555);  // OCR
    prompt.push_back(1917);   // this
    prompt.push_back(4656);   // image
    prompt.push_back(198);    // \n
    prompt.push_back(59254);  // <|assistant|>
    prompt.push_back(198);    // \n

    printf("Prompt: %zu tokens (%d image)\n", prompt.size(), vr.n_tokens);

    // Generate
    printf("\nGenerating (max %d tokens)...\n", max_tokens);
    glm_ocr::generate_result gen;
    if (!glm_ocr::generate(ctx,
            vr.hidden, vr.n_tokens, vr.hidden_dim,
            prompt.data(), (int)prompt.size(),
            max_tokens, gen)) {
        fprintf(stderr, "Generation failed\n");
        free(vr.hidden);
        glm_ocr::free_(ctx);
        return 1;
    }

    printf("\n=== Output (%zu tokens) ===\n%s\n=== End ===\n",
           gen.token_ids.size(), gen.text.c_str());

    free(vr.hidden);
    glm_ocr::free_(ctx);
    return 0;
}
