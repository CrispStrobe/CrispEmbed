// tests/test_ppformulanet_l.cpp — PP-FormulaNet-L encoder diff test.
//
// Compares C++ encoder + decoder against Python reference (GGUF archive).
// Usage:
//   ./test-ppformulanet-l <model.gguf> <ref.gguf>
//
// Or without reference (smoke test):
//   ./test-ppformulanet-l <model.gguf>

#include "ppformulanet_l_ocr.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static std::vector<float> create_test_image(int S) {
    // Same synthetic image as dump_ppformulanet_l_reference.py:
    // gray 0.8 with dark bar at center
    std::vector<float> img(S * S, 0.8f);
    for (int y = S/2 - 3; y < S/2 + 3; y++)
        for (int x = S/4; x < 3*S/4; x++)
            img[y * S + x] = 0.1f;
    return img;
}

// Load a tensor from reference GGUF
static std::vector<float> load_ref_tensor(gguf_context* gctx, ggml_context* mctx,
                                           const char* name, int* out_n) {
    ggml_tensor* t = ggml_get_tensor(mctx, name);
    if (!t) {
        if (out_n) *out_n = 0;
        return {};
    }
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    }
    if (out_n) *out_n = n;
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [ref.gguf]\n", argv[0]);
        return 1;
    }

    ppformulanet_l_ocr_context* ctx = ppformulanet_l_ocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto* hp = ppformulanet_l_ocr_get_hparams(ctx);
    int S = hp->image_size;
    printf("Model: image=%d enc=%dL/%dH/%d (ws=%d) dec=%dL/%dH/%d vocab=%d\n",
           S, hp->enc_layers, hp->enc_heads, hp->enc_hidden, hp->window_size,
           hp->dec_layers, hp->dec_heads, hp->dec_d_model, hp->vocab_size);

    // Run inference with synthetic image
    auto img = create_test_image(S);
    int len = 0;
    const char* result = ppformulanet_l_ocr_recognize(ctx, img.data(), S, S, &len);
    printf("Result: \"%s\" (len=%d)\n", result ? result : "(null)", len);

    // Get encoder output for comparison
    int n_tok = 0, hidden = 0;
    const float* enc_out = ppformulanet_l_ocr_get_encoder_output(ctx, &n_tok, &hidden);
    printf("Encoder output: %d tokens x %d hidden\n", n_tok, hidden);

    // If reference GGUF provided, compare
    if (argc >= 3) {
        printf("\n--- Comparing with reference: %s ---\n", argv[2]);

        ggml_context* mctx = nullptr;
        gguf_init_params params = {};
        params.ctx = &mctx;
        params.no_alloc = false;
        gguf_context* ref = gguf_init_from_file(argv[2], params);
        if (!ref) {
            fprintf(stderr, "Failed to load reference GGUF\n");
            ppformulanet_l_ocr_free(ctx);
            return 1;
        }

        // Compare projected encoder output
        int ref_n = 0;
        auto ref_proj = load_ref_tensor(ref, mctx, "proj_output", &ref_n);
        if (!ref_proj.empty() && enc_out) {
            int n = std::min(ref_n, n_tok * hidden);
            float cos = cosine_sim(enc_out, ref_proj.data(), n);
            float mad = max_abs_diff(enc_out, ref_proj.data(), n);
            printf("proj_output: cos=%.6f max_abs_diff=%.6f (%d elements)\n", cos, mad, n);
        }

        // Compare logits
        auto ref_logits = load_ref_tensor(ref, mctx, "logits_step0", &ref_n);
        if (!ref_logits.empty()) {
            printf("ref logits[0:5]: %.4f %.4f %.4f %.4f %.4f\n",
                   ref_logits[0], ref_logits[1], ref_logits[2], ref_logits[3], ref_logits[4]);
            // Find top token
            int ref_best = 0;
            for (int i = 1; i < ref_n; i++)
                if (ref_logits[i] > ref_logits[ref_best]) ref_best = i;
            printf("ref top token: %d (score %.3f)\n", ref_best, ref_logits[ref_best]);
        }

        ggml_free(mctx);
        gguf_free(ref);
    }

    ppformulanet_l_ocr_free(ctx);
    return 0;
}
