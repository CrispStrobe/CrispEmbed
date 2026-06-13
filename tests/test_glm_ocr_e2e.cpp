// test_glm_ocr_e2e.cpp — GLM-OCR generation test.
// Usage: test-glm-ocr-e2e <model.gguf> [max_tokens]

#include "glm_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [max_tokens]\n", argv[0]);
        return 1;
    }
    int max_tokens = (argc > 2) ? atoi(argv[2]) : 20;

    glm_ocr::context ctx;
    if (!glm_ocr::load(ctx, argv[1], 4, 1)) return 1;
    printf("  Vocab: %d tokens\n", ctx.tok.vocab_size);

    // Simple text-only generation test with BOS token
    printf("\nGenerating (max %d tokens)...\n", max_tokens);
    int32_t prompt[] = {1, 100, 200};
    glm_ocr::generate_result gen;
    if (!glm_ocr::generate(ctx, nullptr, 0, 0, prompt, 3, max_tokens, gen)) {
        fprintf(stderr, "Generation failed\n");
        glm_ocr::free_(ctx);
        return 1;
    }

    printf("\nGenerated %zu tokens:", gen.token_ids.size());
    for (int32_t id : gen.token_ids) printf(" %d", id);
    printf("\n");
    if (!gen.text.empty())
        printf("\nDecoded: %s\n", gen.text.c_str());

    glm_ocr::free_(ctx);
    printf("Done.\n");
    return 0;
}
