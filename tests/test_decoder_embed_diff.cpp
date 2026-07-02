// tests/test_decoder_embed_diff.cpp — decoder-embedding (Qwen3/Gemma3) parity via crispembed-diff.
// Usage: ./test-decoder-embed-diff model.gguf ref.gguf
//
// Single-stage guardrail: crispembed_encode returns the final (last-token-pooled, L2-norm)
// text embedding; we compare "final_embedding" vs an independent HF-AutoModel reference
// (tools/dump_decoder_embed_reference.py). cosine is scale-invariant; a graph-scramble
// regression craters cos to ~0.

#include "crispembed.h"
#include "crispembed_diff.h"
#include <cstdio>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    if (c) {
        printf("  %s[PASS]%s %s\n", GREEN, RESET, n);
        n_pass++;
    } else {
        printf("  %s[FAIL]%s %s\n", RED, RESET, n);
        n_fail++;
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf ref.gguf\n", argv[0]);
        return 1;
    }
    // MUST match tools/dump_decoder_embed_reference.py --text default.
    const char * text = "The quick brown fox jumps over the lazy dog";

    printf("decoder_embed (Qwen3/Gemma3) — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    crispembed_context * ctx = crispembed_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int dim = 0;
    const float * emb = crispembed_encode(ctx, text, &dim);
    check("encode returns non-null", emb != nullptr);
    printf("  embedding dim: %d\n", dim);

    if (emb && dim > 0) {
        auto r = ref.compare("final_embedding", emb, dim);
        // 0.98 floor: q8_0 GGUF vs f32 HF ref → ~0.99; a scramble craters to ~0.
        printf("  final_embedding: cos_min=%.6f max_abs=%.6f  %s\n", r.cos_min, r.max_abs,
               r.is_pass(0.98f) ? "PASS" : "FAIL");
        check("final_embedding cos >= 0.98", r.is_pass(0.98f));
    }

    crispembed_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
