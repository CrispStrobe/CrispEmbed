// tests/test_clip_text_diff.cpp — CLIP/SigLIP text encoder parity via crispembed-diff.
// Usage: ./test-clip-text-diff clip-text.gguf clip-text-ref.gguf
//
// Single-stage guardrail: the C API exposes only the final pooled text embedding, so
// we compare "final_embedding" vs an independent HF-AutoModel reference
// (tools/dump_clip_text_reference.py, text "a photo of a fox"). cosine is scale-invariant,
// so an L2-norm mismatch is harmless; a graph-scramble regression craters cos to ~0.

#include "crispembed.h"
#include "crispembed_diff.h"
#include "core/clean_exit.h"
#include <cstdio>
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
        fprintf(stderr, "Usage: %s clip-text.gguf ref.gguf [text]\n", argv[0]);
        return 1;
    }
    // MUST match the --text the reference was dumped with. The default is the
    // regression fixture; pass argv[3] (with a matching
    // `dump_clip_text_reference.py --text`) to probe a different one.
    //
    // The default is all-lowercase and punctuation-free, which is precisely
    // why it could not see the missing SigLIP normalizer (Lowercase + ASCII
    // punctuation strip). Use a text with capitals and punctuation to exercise
    // that stage.
    const char * text = (argc > 3) ? argv[3] : "a photo of a fox";

    printf("CLIP/SigLIP text encoder — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n  Text:  \"%s\"\n\n", argv[1], argv[2], text);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    crispembed_clip_text_context * ctx = crispembed_clip_text_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int dim = 0;
    const float * emb = crispembed_clip_text_encode(ctx, text, &dim);
    check("encode returns non-null", emb != nullptr);
    printf("  embedding dim: %d\n", dim);

    if (emb && dim > 0) {
        auto r = ref.compare("final_embedding", emb, dim);
        // Canonical "stage: cos_min=X max_abs=Y" so tests/regression/run_one.py's
        // _DIFF_PATTERNS[0] parses this as a gated stage. The stage token must be
        // the first non-space run (no leading "vs ").
        printf("  final_embedding: cos_min=%.6f max_abs=%.6f (post text_proj)\n", r.cos_min, r.max_abs);
        if (ref.has("pre_proj")) {
            auto rp = ref.compare("pre_proj", emb, dim);
            // Diagnostic only (expected ~0, confirming text_proj IS applied). The
            // leading "vs " keeps this OFF run_one.py's _DIFF_PATTERNS[0] anchor so
            // it is never treated as a gated stage — only "final_embedding" is.
            printf("  vs pre_proj (EOS hidden, no proj): cos_min=%.6f max_abs=%.6f\n", rp.cos_min, rp.max_abs);
        }
        check("final_embedding cos >= 0.99", r.is_pass(0.99f));
    }

    crispembed_clip_text_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
