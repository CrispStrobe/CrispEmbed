// tests/test_smoldocling_diff.cpp — SmolDocling parity via crispembed-diff.
// Compares C++ vision encoder + connector output against HF reference.
//
// Usage: ./test-smoldocling-diff model.gguf ref.gguf image.png

#include "../src/smoldocling_ocr.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

extern "C" {
    unsigned char * stbi_load(const char *, int *, int *, int *, int);
    void stbi_image_free(void *);
}

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.gguf ref.gguf image.png\n", argv[0]);
        return 1;
    }
    printf("SmolDocling — per-stage parity test\n");
    printf("  Model: %s\n  Ref:   %s\n  Image: %s\n\n", argv[1], argv[2], argv[3]);

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    // Load model
    smoldocling_context * ctx = smoldocling_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    // Load test image
    int w, h, c;
    unsigned char * img = stbi_load(argv[3], &w, &h, &c, 3);
    check("image loads", img != nullptr);
    if (!img) { smoldocling_free(ctx); return 1; }

    // Stage 1: Vision encoder output (after post_layernorm)
    printf("\n--- Vision encoder ---\n");
    int vis_tokens = 0, vis_dim = 0;
    float * vis_out = smoldocling_debug_vision(ctx, img, w, h, 3, &vis_tokens, &vis_dim);
    check("vision returns data", vis_out != nullptr);
    printf("  Tokens: %d, Dim: %d\n", vis_tokens, vis_dim);
    if (vis_out) {
        printf("  C++ first4: [%.4f, %.4f, %.4f, %.4f]\n",
               vis_out[0], vis_out[1], vis_out[2], vis_out[3]);

        // Compare against vis_post_ln reference
        auto r = ref.compare("vis_post_ln", vis_out, vis_tokens * vis_dim);
        printf("  vis_post_ln: cos=%.6f max_abs=%.6f  %s\n",
               r.cos_min, r.max_abs, r.is_pass(0.99f) ? "PASS" : "FAIL");
        check("vis_post_ln cos >= 0.99", r.is_pass(0.99f));

        free(vis_out);
    }

    // Stage 2: Connector output
    printf("\n--- Connector ---\n");
    int conn_tokens = 0, conn_dim = 0;
    float * conn_out = smoldocling_debug_connector(ctx, img, w, h, 3, &conn_tokens, &conn_dim);
    check("connector returns data", conn_out != nullptr);
    printf("  Tokens: %d, Dim: %d\n", conn_tokens, conn_dim);
    if (conn_out) {
        printf("  C++ first4: [%.4f, %.4f, %.4f, %.4f]\n",
               conn_out[0], conn_out[1], conn_out[2], conn_out[3]);

        auto r = ref.compare("connector_output", conn_out, conn_tokens * conn_dim);
        printf("  connector_output: cos=%.6f max_abs=%.6f  %s\n",
               r.cos_min, r.max_abs, r.is_pass(0.99f) ? "PASS" : "FAIL");
        check("connector cos >= 0.99", r.is_pass(0.99f));

        free(conn_out);
    }

    stbi_image_free(img);
    smoldocling_free(ctx);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
