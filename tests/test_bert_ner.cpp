// tests/test_bert_ner.cpp — BERT NER parity test
#include "crispembed.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <bert-ner.gguf> [text]\n", argv[0]);
        return 1;
    }
    const char* text = argc > 2 ? argv[2] : "Barack Obama was born in Hawaii";

    // Test 1: raw hidden states
    auto* ctx = crispembed_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int n_tok = 0, dim = 0;
    const float* raw = crispembed_encode_tokens_raw(ctx, text, &n_tok, &dim);
    if (!raw) { fprintf(stderr, "encode_tokens_raw failed\n"); return 1; }

    const int32_t* ids = crispembed_last_token_ids(ctx);
    printf("T=%d dim=%d\n", n_tok, dim);
    for (int t = 0; t < n_tok; t++) {
        const char* ts = crispembed_token_str(ctx, ids[t]);
        printf("  [%d] %-15s  first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               t, ts, raw[t*dim], raw[t*dim+1], raw[t*dim+2], raw[t*dim+3], raw[t*dim+4]);
    }

    // Test 2: NER extraction
    printf("\nNER extraction:\n");
    void* nctx = crispembed_ner_init(argv[1], 4);
    if (!nctx) { fprintf(stderr, "Failed to load NER\n"); crispembed_free(ctx); return 1; }

    crispembed_ner_entity* ents = nullptr;
    int n = crispembed_ner_extract(nctx, text, nullptr, 0, 0.0f, &ents);
    printf("  %d entities:\n", n);
    for (int i = 0; i < n; i++) {
        printf("    \"%s\" → %s [%d,%d) score=%.3f\n",
               ents[i].text, ents[i].label,
               ents[i].start_char, ents[i].end_char, ents[i].score);
    }

    crispembed_ner_free(nctx);
    crispembed_free(ctx);
    return 0;
}
