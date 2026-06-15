// test_qwen2vl_e2e.cpp — end-to-end OCR test for Qwen2.5-VL.
//
// Usage: test-qwen2vl-e2e <model.gguf> <tokens.bin> <image.png>
//
// tokens.bin: output of tools/qwen2vl_tokenize.py --format bin
//   Format: 4-byte uint32 count, then count × 4-byte int32 token IDs
//
// Runs: image preprocessing → vision encoder → token splicing →
//       LLM decoder → greedy generation → print output text

#include "qwen2vl_ocr.h"
#include "crispembed_diff.h"
#include "image_preprocess.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <tokens.bin> <image.png>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[0 + 1];
    const char *tokens_path = argv[1 + 1];
    const char *image_path = argv[2 + 1];
    int max_tokens = (argc > 4) ? atoi(argv[4]) : 128;

    // ── Load token IDs from binary file ─────────────────────────
    printf("Loading tokens: %s\n", tokens_path);
    FILE *tf = fopen(tokens_path, "rb");
    if (!tf) { fprintf(stderr, "Cannot open %s\n", tokens_path); return 1; }

    uint32_t n_tokens = 0;
    fread(&n_tokens, sizeof(uint32_t), 1, tf);
    std::vector<int32_t> token_ids(n_tokens);
    fread(token_ids.data(), sizeof(int32_t), n_tokens, tf);
    fclose(tf);

    printf("  %u tokens loaded\n", n_tokens);

    // Count image_pad tokens (151655)
    int n_image_pad = 0;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (token_ids[i] == 151655) n_image_pad++;
    }
    printf("  %d image_pad tokens\n", n_image_pad);

    // ── Load model ──────────────────────────────────────────────
    printf("\nLoading model: %s\n", model_path);
    qwen2vl_ocr::context ctx;
    if (!qwen2vl_ocr::load(ctx, model_path, 4, 1)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Preprocess image ────────────────────────────────────────
    printf("\nPreprocessing image: %s\n", image_path);

    // Use CrispEmbed image_preprocess for Qwen2VL-compatible preprocessing
    // For now, use the Python-generated patches from the reference test
    // TODO: integrate image_preprocess.cpp

    // Quick hack: load patches from the reference GGUF if available
    // For the real test, we'd preprocess the image in C++
    printf("  (using Python-preprocessed patches from reference)\n");

    // For end-to-end test without C++ preprocessing, we need patches.
    // Let's check if we have a reference GGUF with input_patches:
    const char *ref_path = "/tmp/qwen2vl-ref-full.gguf";
    FILE *check = fopen(ref_path, "rb");
    if (!check) {
        fprintf(stderr, "Need %s with input_patches for image data\n", ref_path);
        fprintf(stderr, "Run: python tools/dump_qwen2vl_reference.py first\n");
        qwen2vl_ocr::free_(ctx);
        return 1;
    }
    fclose(check);

    // Load patches from reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "Failed to load reference GGUF\n");
        qwen2vl_ocr::free_(ctx);
        return 1;
    }

    auto [patches_data, patches_n] = ref.get_f32("input_patches");
    if (!patches_data) {
        fprintf(stderr, "No input_patches in reference\n");
        qwen2vl_ocr::free_(ctx);
        return 1;
    }

    // Derive grid
    auto vis_shape = ref.shape("vis_patch_embed");
    int n_patches = (vis_shape.size() >= 2) ? (int)vis_shape[1] : 0;
    auto merger_shape = ref.shape("vis_merger_output");
    int n_merged = (merger_shape.size() >= 2) ? (int)merger_shape[1] : 0;

    int32_t grid_thw[3] = {1, 0, 0};
    for (int h = 2; h * h <= n_patches * 2; h += 2) {
        if (n_patches % h == 0) {
            int w = n_patches / h;
            if (w % 2 == 0 && (h / 2) * (w / 2) == n_merged) {
                if (abs(h - w) < abs(grid_thw[1] - grid_thw[2]) || grid_thw[1] == 0) {
                    grid_thw[1] = h;
                    grid_thw[2] = w;
                }
            }
        }
    }
    printf("  Grid: %dx%d, %d patches, %d merged\n",
           grid_thw[1], grid_thw[2], n_patches, n_merged);

    if (n_image_pad != n_merged) {
        fprintf(stderr, "WARNING: token image_pad count (%d) != merged tokens (%d)\n",
                n_image_pad, n_merged);
    }

    // ── Run vision encoder ──────────────────────────────────────
    printf("\nRunning vision encoder...\n");
    qwen2vl_ocr::vision_result vis_result;
    if (!qwen2vl_ocr::encode_vision(ctx, patches_data, n_patches,
                                     grid_thw, vis_result)) {
        fprintf(stderr, "Vision encoder failed\n");
        qwen2vl_ocr::free_(ctx);
        return 1;
    }
    printf("  %d merged tokens, %d dim\n", vis_result.n_merged, vis_result.embed_dim);

    // ── Generate text ───────────────────────────────────────────
    printf("\nGenerating (max %d tokens)...\n", max_tokens);
    qwen2vl_ocr::generate_result gen_result;
    if (!qwen2vl_ocr::generate(ctx, vis_result.image_embeds,
                                vis_result.n_merged, vis_result.embed_dim,
                                grid_thw,  // actual image grid for mRoPE
                                token_ids.data(), (int)n_tokens,
                                max_tokens, gen_result)) {
        fprintf(stderr, "Generation failed\n");
        qwen2vl_ocr::vision_result_free(vis_result);
        qwen2vl_ocr::free_(ctx);
        return 1;
    }

    printf("\n=== Generated %zu tokens ===\n", gen_result.token_ids.size());
    printf("Token IDs: [");
    for (size_t i = 0; i < gen_result.token_ids.size(); i++) {
        printf("%s%d", i ? "," : "", gen_result.token_ids[i]);
    }
    printf("]\n");

    // Clean up
    qwen2vl_ocr::vision_result_free(vis_result);
    qwen2vl_ocr::free_(ctx);

    printf("\nDone.\n");
    return 0;
}
