// tests/test_bert_ner_diff.cpp — BERT NER per-layer parity test.
// Usage: ./test-bert-ner-diff bert-base-ner-f32.gguf bert-ner-ref.gguf

#include "crispembed.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    auto* ctx = crispembed_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Get raw hidden states for comparison
    int n_tok = 0, dim = 0;
    const float* raw = crispembed_encode_tokens_raw(ctx, "Barack Obama was born in Hawaii", &n_tok, &dim);
    if (!raw) { fprintf(stderr, "encode failed\n"); return 1; }

    printf("C++ tokens: %d, dim: %d\n", n_tok, dim);
    printf("C++ first token first5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
           raw[0], raw[1], raw[2], raw[3], raw[4]);

    // Compare final hidden state
    auto r = ref.compare("final_hidden", raw, (size_t)n_tok * dim, dim);
    printf("\n%-20s %10.6f %10.2e %s\n", "final_hidden", r.cos_min, r.max_abs,
           r.is_pass() ? "PASS" : "FAIL");

    // Also check embedding stage if Python tokens match
    auto [ref_ids, ref_n] = ref.get_f32("input_ids");
    if (ref_ids && ref_n == (size_t)n_tok) {
        printf("\nToken count matches (both %d)\n", n_tok);

        const int32_t* cpp_ids = crispembed_last_token_ids(ctx);
        bool ids_match = true;
        for (int i = 0; i < n_tok; i++) {
            if (cpp_ids[i] != (int32_t)ref_ids[i]) {
                printf("  Token %d: C++=%d, Python=%d MISMATCH\n",
                       i, cpp_ids[i], (int32_t)ref_ids[i]);
                ids_match = false;
            }
        }
        if (ids_match) printf("  All token IDs match\n");
    } else {
        printf("\nToken count MISMATCH: C++=%d, Python=%zu\n", n_tok, ref_n);
    }

    crispembed_free(ctx);
    return r.is_pass() ? 0 : 1;
}
