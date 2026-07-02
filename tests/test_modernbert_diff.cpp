// tests/test_modernbert_diff.cpp — ModernBERT encoder parity test.
// Usage: ./test-modernbert-diff gte-modernbert-base-q8_0.gguf modernbert-ref.gguf
//
// The text is >128 tokens so ModernBERT's local sliding-window layers restrict
// attention — this guards the SWA mask (src/crispembed.cpp swa_mask), not just the
// backbone. Must match tools/dump_modernbert_reference.py TEXT exactly.

#include "crispembed.h"
#include "crispembed_diff.h"
#include <cmath>
#include <cstdio>

static const char * TEXT = "Machine learning is a subset of artificial intelligence that enables systems "
                           "to learn from data. Transformers use self attention to model long range "
                           "dependencies across an entire sequence. ModernBERT alternates global and local "
                           "attention layers to process long documents efficiently while keeping quality "
                           "high. The sliding window restricts most layers to a local neighborhood, and "
                           "every third layer attends globally to mix information across the whole passage. "
                           "Berlin is the capital of Germany and the Eiffel Tower stands in Paris while "
                           "water boils at one hundred degrees Celsius at sea level near the open ocean.";

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    auto * ctx = crispembed_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    int n_tok = 0, dim = 0;
    const float * raw = crispembed_encode_tokens_raw(ctx, TEXT, &n_tok, &dim);
    if (!raw) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    printf("C++ tokens: %d, dim: %d\n", n_tok, dim);

    auto r = ref.compare("final_hidden", raw, (size_t)n_tok * dim, dim);
    // Canonical parseable format (tests/regression/run_one.py _DIFF_LINE):
    //   <stage>: cos_min=<f> max_abs=<f> PASS|FAIL
    printf("final_hidden: cos_min=%.6f max_abs=%.2e %s\n", r.cos_min, r.max_abs,
           r.is_pass(0.99f) ? "PASS" : "FAIL"); // 0.99: q8_0 model vs f32 ref; SWA scramble craters cos

    auto [ref_ids, ref_n] = ref.get_f32("input_ids");
    if (ref_ids && ref_n == (size_t)n_tok) {
        const int32_t * cpp_ids = crispembed_last_token_ids(ctx);
        bool ids_match = true;
        for (int i = 0; i < n_tok; i++) {
            if (cpp_ids[i] != (int32_t)ref_ids[i]) {
                printf("  Token %d: C++=%d, Python=%d MISMATCH\n", i, cpp_ids[i], (int32_t)ref_ids[i]);
                ids_match = false;
            }
        }
        printf(ids_match ? "  All %d token IDs match\n" : "  token id mismatch\n", n_tok);
    } else {
        printf("\nToken count MISMATCH: C++=%d, Python=%zu\n", n_tok, ref_n);
    }

    crispembed_free(ctx);
    return r.is_pass(0.99f) ? 0 : 1;
}
