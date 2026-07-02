// tests/test_lfm2_colbert_diff.cpp — LFM2.5-ColBERT parity via crispembed-diff.
// Usage: ./test-lfm2-colbert-diff lfm2-colbert-f32.gguf lfm2-colbert-ref.gguf

#include "crispembed.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>

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

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf ref.gguf\n", argv[0]);
        return 1;
    }
    printf("LFM2.5-ColBERT — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    // Load model via crispembed C API
    crispembed_context * ctx = crispembed_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    // Check ColBERT capability
    int has_cb = crispembed_has_colbert(ctx);
    check("has_colbert = true", has_cb != 0);

    // Encode the same text as the reference
    const char * text = "query: The quick brown fox";
    int n_tokens = 0, colbert_dim = 0;
    const float * multivec = crispembed_encode_multivec(ctx, text, &n_tokens, &colbert_dim);

    check("encode_multivec returns non-null", multivec != nullptr);
    printf("  Tokens: %d, ColBERT dim: %d\n", n_tokens, colbert_dim);

    if (multivec && n_tokens > 0) {
        // Compare with reference
        auto r = ref.compare("colbert_output", multivec, n_tokens * colbert_dim);
        printf("  colbert_output: cos=%.6f max_abs=%.6f  %s\n", r.cos_min, r.max_abs,
               r.is_pass(0.99f) ? "PASS" : "FAIL");
        // 0.99 floor: the ref is f32 (AutoModel) but the shipped GGUF is q8_0 -> ~0.998
        // colbert_output; a real scramble craters to ~0. (Set the ref+model both f32 for 0.999.)
        check("colbert cos >= 0.99", r.is_pass(0.99f));
    }

    crispembed_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
