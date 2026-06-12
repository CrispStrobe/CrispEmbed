// test_internvl2_image.cpp — InternVL2 OCR on a real image file.
//
// Usage: test-internvl2-image <model.gguf> <image.png> [max_tokens] [prompt]

#include "internvl2_ocr.h"
#include "image_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.png> [max_tokens] [prompt]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 200;
    const char *user_prompt = (argc > 4) ? argv[4] : "Describe this image in detail.";

    // Load model
    printf("Loading model: %s\n", model_path);
    internvl2_ocr::context ctx;
    if (!internvl2_ocr::load(ctx, model_path, 4, 1)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("  Vocab: %d tokens\n", ctx.tok.vocab_size);

    // Preprocess image with dynamic tiling
    printf("Processing image: %s\n", image_path);
    image_preproc::internvl_config icfg;
    icfg.image_size = (int)ctx.m.vhp.image_size;
    icfg.max_dynamic_patch = (int)ctx.m.lhp.max_dynamic_patch;
    icfg.min_dynamic_patch = (int)ctx.m.lhp.min_dynamic_patch;
    icfg.use_thumbnail = ctx.m.lhp.use_thumbnail;
    for (int c = 0; c < 3; c++) {
        icfg.mean[c] = ctx.m.vhp.image_mean[c];
        icfg.std[c] = ctx.m.vhp.image_std[c];
    }

    image_preproc::internvl_result ires;
    if (!image_preproc::preprocess_internvl_file(image_path, icfg, ires)) {
        fprintf(stderr, "Failed to preprocess image\n");
        internvl2_ocr::free_(ctx);
        return 1;
    }
    printf("  Tiles: %d (%dx%d grid", ires.n_tiles, ires.grid_rows, ires.grid_cols);
    if (ctx.m.lhp.use_thumbnail) printf(" + thumbnail");
    printf(")\n");

    // Vision encode all tiles
    printf("Encoding vision (%d tiles)...\n", ires.n_tiles);
    internvl2_ocr::vision_pipeline_result vpr;
    if (!internvl2_ocr::encode_vision(ctx, ires.tiles.data(), ires.n_tiles, vpr)) {
        fprintf(stderr, "Vision encode failed\n");
        internvl2_ocr::free_(ctx);
        return 1;
    }
    printf("  Vision: %d tokens, %d dim\n", vpr.n_image_tokens, vpr.embed_dim);

    // Build chat prompt
    int32_t img_token_id = (int32_t)ctx.m.lhp.image_token_id;
    std::vector<int32_t> prompt;

    // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    int32_t sys_tokens[] = {92543, 9081, 364, 2770, 657, 395, 11100, 17993, 281, 92542, 364};
    prompt.insert(prompt.end(), sys_tokens, sys_tokens + 11);

    // <|im_start|>user\n
    int32_t user_start[] = {92543, 1404, 364};
    prompt.insert(prompt.end(), user_start, user_start + 3);

    // <IMG_CONTEXT> * n_image_tokens
    for (int i = 0; i < vpr.n_image_tokens; i++) {
        prompt.push_back(img_token_id);
    }

    // \n + user prompt tokens (hardcoded for "Describe this image in detail.")
    // TODO: use proper tokenizer encode
    int32_t user_text[] = {364, 3471, 2321, 435, 7856, 281, 92542, 364};
    prompt.insert(prompt.end(), user_text, user_text + 8);

    // <|im_start|>assistant\n
    int32_t asst_start[] = {92543, 525, 11353, 364};
    prompt.insert(prompt.end(), asst_start, asst_start + 4);

    printf("Prompt: %zu tokens (%d image)\n", prompt.size(), vpr.n_image_tokens);

    // Generate
    printf("\nGenerating (max %d tokens)...\n", max_tokens);
    internvl2_ocr::generate_result gen;
    if (!internvl2_ocr::generate(ctx,
            vpr.image_embeds, vpr.n_image_tokens, vpr.embed_dim,
            prompt.data(), (int)prompt.size(), max_tokens, gen)) {
        fprintf(stderr, "Generation failed\n");
        free(vpr.image_embeds);
        internvl2_ocr::free_(ctx);
        return 1;
    }

    printf("\n=== Output (%zu tokens) ===\n%s\n=== End ===\n",
           gen.token_ids.size(), gen.text.c_str());

    free(vpr.image_embeds);
    internvl2_ocr::free_(ctx);
    return 0;
}
