// tests/test_pix2struct_diff.cpp -- Pix2Struct encoder parity via crispembed-diff.
// Usage: ./test-pix2struct-diff pix2struct-base-f32.gguf pix2struct-ref.gguf

#include "pix2struct.h"
#include "crispembed_diff.h"
#include <chrono>
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
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf ref.gguf\n", argv[0]); return 1; }
    printf("Pix2Struct -- encoder parity test\n");
    printf("  Model: %s\n  Ref: %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    pix2struct_context * ctx = pix2struct_init(argv[1], 1);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    // Get preprocessed patches from reference
    auto [patches, patches_n] = ref.get_f32("flattened_patches");
    auto patches_shape = ref.shape("flattened_patches");
    if (!patches || patches_n == 0) {
        fprintf(stderr, "Reference missing flattened_patches\n");
        pix2struct_free(ctx); return 1;
    }
    int n_patches = (int)patches_shape[1]; // ggml: ne[0]=770, ne[1]=n_patches
    int patch_dim = (int)patches_shape[0]; // 770
    printf("  Patches: %d x %d\n", n_patches, patch_dim);

    // Run encoder
    int out_dim = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    const float * enc_out = pix2struct_encode_patches(ctx, patches, n_patches, &out_dim);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    check("encoder returns non-null", enc_out != nullptr);
    printf("  Encoder: %d x %d in %.1f ms\n\n", n_patches, out_dim, ms);

    // Compare encoder output
    printf("=== Encoder output ===\n");
    auto r = ref.compare("encoder_output", enc_out, n_patches * out_dim);
    printf("  encoder_output: cos=%.6f max_abs=%.6f  %s\n",
           r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
    check("encoder cos >= 0.999", r.is_pass(0.999f));

    char msg[128];
    snprintf(msg, sizeof(msg), "encoder max_abs < 0.01 (got %.6f)", r.max_abs);
    check(msg, r.max_abs < 0.01f);

    // Decoder step 0 parity (if dec-ref available as argv[3])
    if (argc >= 4) {
        crispembed_diff::Ref dec_ref;
        if (dec_ref.load(argv[3])) {
            printf("\n=== Decoder step 0 ===\n");
            std::vector<float> logits(50244); // vocab_size
            int ret2 = pix2struct_decode_step0(ctx, logits.data());
            check("decode_step0 returns 0", ret2 == 0);

            auto r2 = dec_ref.compare("logits_step0", logits.data(), 50244);
            printf("  logits: cos=%.6f max_abs=%.6f  %s\n",
                   r2.cos_min, r2.max_abs, r2.is_pass(0.999f) ? "PASS" : "FAIL");
            check("logits cos >= 0.999", r2.is_pass(0.999f));

            // Check argmax
            int argmax = 0;
            for (int i = 1; i < 50244; i++)
                if (logits[i] > logits[argmax]) argmax = i;
            printf("  argmax: %d (expected 411)\n", argmax);
            check("argmax == 411 (token '<')", argmax == 411);
        }
    }

    pix2struct_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
