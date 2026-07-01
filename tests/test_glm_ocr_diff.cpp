// test_glm_ocr_diff.cpp — per-layer parity test for GLM-OCR vision encoder.
//
// Usage: test-glm-ocr-diff <model.gguf> <ref.gguf>

#include "glm_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    // Load reference
    printf("Loading reference: %s\n", argv[2]);
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        fprintf(stderr, "Failed to load reference\n");
        return 1;
    }
    printf("Reference tensors:\n");
    for (auto &name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf("  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++)
            printf("%s%lld", i ? "," : "", (long long)s[i]);
        printf("]\n");
    }

    // Load model
    printf("\nLoading model: %s\n", argv[1]);
    glm_ocr::context ctx;
    ctx.diff_ref_path = argv[2];
    if (!glm_ocr::load(ctx, argv[1], 4, 2)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Synthetic gradient image
    const int img_size = (int)ctx.m.vhp.image_size;
    std::vector<float> pixels(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float val = (float)(y * img_size + x) / (float)(img_size * img_size);
                pixels[c * img_size * img_size + y * img_size + x] =
                    (val - ctx.m.vhp.image_mean[c]) / ctx.m.vhp.image_std[c];
            }

    // Run vision
    printf("\nRunning vision encoder...\n");
    glm_ocr::vision_result vr;
    if (!glm_ocr::encode_vision(ctx, pixels.data(), img_size, img_size, vr)) {
        fprintf(stderr, "Vision encode failed\n");
        glm_ocr::free_(ctx);
        return 1;
    }
    printf("Vision output: %d tokens, %d dim\n", vr.n_tokens, vr.hidden_dim);

    free(vr.hidden);

    // Run LLM forward (if reference has LLM layers)
    {
        crispembed_diff::Ref ref2;
        ref2.load(argv[2]);
        if (ref2.has("llm_embed")) {
            printf("\nRunning LLM decoder...\n");
            int32_t test_tokens[] = {1, 100, 200, 300, 400};
            glm_ocr::llm_result lr;
            if (glm_ocr::run_llm_forward(ctx, test_tokens, 5, lr)) {
                printf("LLM output: %d tokens, %d dim\n", lr.n_tokens, lr.hidden_dim);
                free(lr.hidden);
                if (lr.logits) free(lr.logits);
            } else {
                fprintf(stderr, "LLM forward failed\n");
            }
        }
    }

    glm_ocr::free_(ctx);
    printf("\nDone.\n");
    return 0;
}
