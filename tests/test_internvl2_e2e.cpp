// test_internvl2_e2e.cpp — end-to-end generation test for InternVL2.
//
// Usage: test-internvl2-e2e <model.gguf> [max_tokens]
//
// Runs: synthetic image → vision encode → greedy generation → print tokens.

#include "internvl2_ocr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [max_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int max_tokens = (argc > 2) ? atoi(argv[2]) : 20;

    // Load model
    printf("Loading model: %s\n", model_path);
    internvl2_ocr::context ctx;
    if (!internvl2_ocr::load(ctx, model_path, 4, 1)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Prepare synthetic image
    const int img_size = (int)ctx.m.vhp.image_size;
    std::vector<float> pixels(3 * img_size * img_size);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < img_size; y++) {
            for (int x = 0; x < img_size; x++) {
                float val = (float)(y * img_size + x) / (float)(img_size * img_size);
                pixels[c * img_size * img_size + y * img_size + x] =
                    (val - ctx.m.vhp.image_mean[c]) / ctx.m.vhp.image_std[c];
            }
        }
    }

    // Vision encode
    printf("\nEncoding vision...\n");
    internvl2_ocr::vision_pipeline_result vpr;
    if (!internvl2_ocr::encode_vision(ctx, pixels.data(), 1, vpr)) {
        fprintf(stderr, "Vision encode failed\n");
        internvl2_ocr::free_(ctx);
        return 1;
    }
    printf("Vision: %d tokens, %d dim\n", vpr.n_image_tokens, vpr.embed_dim);

    // Build a simple prompt: BOS token + generate
    printf("\nGenerating (max %d tokens)...\n", max_tokens);
    int32_t prompt[] = {1};  // BOS

    internvl2_ocr::generate_result gen;
    if (!internvl2_ocr::generate(ctx,
            vpr.image_embeds, vpr.n_image_tokens, vpr.embed_dim,
            prompt, 1, max_tokens, gen)) {
        fprintf(stderr, "Generation failed\n");
        free(vpr.image_embeds);
        internvl2_ocr::free_(ctx);
        return 1;
    }

    printf("\nGenerated %zu tokens:", gen.token_ids.size());
    for (int32_t id : gen.token_ids) {
        printf(" %d", id);
    }
    printf("\n");

    free(vpr.image_embeds);
    internvl2_ocr::free_(ctx);
    printf("Done.\n");
    return 0;
}
