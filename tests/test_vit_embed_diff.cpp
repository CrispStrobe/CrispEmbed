// tests/test_vit_embed_diff.cpp — SigLIP/CLIP ViT vision encoder parity via crispembed-diff.
// Usage: ./test-vit-embed-diff vit.gguf vit-ref.gguf [image.png]
//
// Single-stage guardrail (nafnet-style): the C API only exposes the final image
// embedding, so we compare "final_embedding" vs an independent HF-AutoModel reference
// (tools/dump_vit_reference.py). cosine is scale-invariant, so an L2-norm mismatch
// between engine and ref is harmless; a graph-scramble regression craters cos to ~0.

#include "crispembed.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    if (c) { printf("  %s[PASS]%s %s\n", GREEN, RESET, n); n_pass++; }
    else   { printf("  %s[FAIL]%s %s\n", RED, RESET, n); n_fail++; }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s vit.gguf ref.gguf [image.png]\n", argv[0]); return 1; }
    // Image: argv[3], else the in-repo fox.png (run_one passes it via diff.args; the
    // fallbacks cover a plain manual invocation from the repo root or build/).
    const char * candidates[] = {
        argc > 3 ? argv[3] : nullptr,
        "tests/regression/images/fox.png",
        "../tests/regression/images/fox.png",
        "../../tests/regression/images/fox.png",
    };
    const char * image = nullptr;
    for (const char * c : candidates) {
        if (!c) continue;
        if (FILE * f = fopen(c, "rb")) { fclose(f); image = c; break; }
    }
    if (!image) { fprintf(stderr, "test image not found\n"); return 1; }

    printf("ViT (SigLIP/CLIP) — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n  Image: %s\n\n", argv[1], argv[2], image);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    crispembed_vit_context * ctx = crispembed_vit_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int dim = 0;
    const float * emb = crispembed_vit_encode_file(ctx, image, &dim);
    check("encode returns non-null", emb != nullptr);
    printf("  embedding dim: %d\n", dim);

    if (emb && dim > 0) {
        auto r = ref.compare("final_embedding", emb, dim);
        // 0.98 floor: f16 GGUF vs f32 HF ref + slow-vs-engine image preprocessing
        // (resize/normalize) gives ~0.9915 on CPU; backend FP variance can shave a bit
        // more. A graph-scramble regression craters to ~0, so 0.98 still catches it.
        printf("  final_embedding: cos_min=%.6f max_abs=%.6f  %s\n",
               r.cos_min, r.max_abs, r.is_pass(0.98f) ? "PASS" : "FAIL");
        check("final_embedding cos >= 0.98", r.is_pass(0.98f));
    }

    crispembed_vit_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
