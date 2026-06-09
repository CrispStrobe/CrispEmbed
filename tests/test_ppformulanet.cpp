// tests/test_ppformulanet.cpp — PPFormulaNet-S encoder diff test.
//
// Compares C++ encoder output against Python reference (GGUF archive).
// Usage:
//   ./test-ppformulanet <model.gguf> <ref.gguf>
//
// Or without reference (smoke test):
//   ./test-ppformulanet <model.gguf>

#include "ppformulanet_ocr.h"
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

static std::vector<float> create_test_image(int S) {
    // Same synthetic image as dump_ppformulanet_reference.py:
    // gray 0.8 with dark bar at center
    std::vector<float> img(S * S, 0.8f);
    for (int y = S/2 - 2; y < S/2 + 2; y++)
        for (int x = S/4; x < 3*S/4; x++)
            img[y * S + x] = 0.1f;
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [ref.gguf]\n", argv[0]);
        return 1;
    }

    ppformulanet_ocr_context* ctx = ppformulanet_ocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto* hp = ppformulanet_ocr_get_hparams(ctx);
    int S = hp->image_size;
    printf("Model: image=%d enc=%d dec=%dL/%dH/%d vocab=%d\n",
           S, hp->enc_hidden, hp->dec_layers, hp->dec_heads,
           hp->dec_d_model, hp->vocab_size);

    // Run inference
    auto img = create_test_image(S);
    int len = 0;
    const char* result = ppformulanet_ocr_recognize(ctx, img.data(), S, S, &len);
    if (result) {
        printf("\nLaTeX output (%d chars): %s\n", len, result);
    } else {
        printf("\n(no result)\n");
    }

    // Get encoder output for comparison
    int n_tok = 0, hidden = 0;
    const float* enc = ppformulanet_ocr_get_encoder_output(ctx, &n_tok, &hidden);
    if (enc) {
        printf("Encoder output: %d tokens × %d hidden\n", n_tok, hidden);
    }

    // Compare against reference if provided
    if (argc >= 3 && enc) {
        printf("\n--- Diff against reference ---\n");

        // Load reference GGUF using ggml's GGUF reader
        ggml_context* ref_ctx = nullptr;
        struct gguf_init_params params = { /*.no_alloc=*/ false, /*.ctx=*/ &ref_ctx };
        gguf_context* gctx = gguf_init_from_file(argv[2], params);
        if (!gctx) {
            fprintf(stderr, "Failed to load reference: %s\n", argv[2]);
        } else {

            // Find enc_output tensor
            int enc_idx = gguf_find_tensor(gctx, "enc_output");
            if (enc_idx >= 0 && ref_ctx) {
                ggml_tensor* ref_t = ggml_get_tensor(ref_ctx, "enc_output");
                if (ref_t && ref_t->type == GGML_TYPE_F32) {
                    const float* ref_data = (const float*)ref_t->data;
                    int ref_n = (int)ggml_nelements(ref_t);
                    int total = std::min(n_tok * hidden, ref_n);
                    float cos = cosine_sim(enc, ref_data, total);
                    printf("enc_output: cos=%.6f (%d elements) %s\n",
                           cos, total, cos > 0.99f ? "PASS" : "FAIL");

                    // Per-token comparison
                    for (int t : {0, 1, 10, 50, 100, 143}) {
                        if (t >= n_tok || (t + 1) * hidden > ref_n) continue;
                        float tc = cosine_sim(enc + t*hidden, ref_data + t*hidden, hidden);
                        float max_abs = 0;
                        for (int i = 0; i < hidden; i++) {
                            float d = fabsf(enc[t*hidden+i] - ref_data[t*hidden+i]);
                            if (d > max_abs) max_abs = d;
                        }
                        printf("  token %3d: cos=%.6f max_abs=%.4e  cpp=[%.4f %.4f %.4f] ref=[%.4f %.4f %.4f] %s\n",
                               t, tc, max_abs,
                               enc[t*hidden], enc[t*hidden+1], enc[t*hidden+2],
                               ref_data[t*hidden], ref_data[t*hidden+1], ref_data[t*hidden+2],
                               tc > 0.99f ? "PASS" : "FAIL");
                    }
                } else {
                    printf("enc_output tensor not found or wrong type\n");
                }
            } else {
                printf("enc_output not found in reference (idx=%d)\n", enc_idx);
            }

            // Compare logits_step0
            ggml_tensor* logits_ref = ggml_get_tensor(ref_ctx, "logits_step0");
            if (logits_ref && logits_ref->type == GGML_TYPE_F32) {
                // Run one decoder step and compare logits
                // (The recognize call already ran the decoder, but we need step 0 logits)
                // For now, compare the token sequence
                ggml_tensor* gen_ids = ggml_get_tensor(ref_ctx, "generated_ids");
                if (gen_ids) {
                    const int32_t* ids = (const int32_t*)gen_ids->data;
                    int n_ids = (int)ggml_nelements(gen_ids);
                    printf("\nRef generated IDs (%d): ", n_ids);
                    for (int i = 0; i < std::min(n_ids, 20); i++)
                        printf("%d ", ids[i]);
                    printf("\n");
                }

                const float* lref = (const float*)logits_ref->data;
                int n_logits = (int)ggml_nelements(logits_ref);
                printf("Ref logits_step0: %d values\n", n_logits);

                // Find top-5 in reference
                printf("Ref top-5: ");
                std::vector<std::pair<float,int>> scored(n_logits);
                for (int i = 0; i < n_logits; i++) scored[i] = {lref[i], i};
                std::partial_sort(scored.begin(), scored.begin()+5, scored.end(),
                    [](auto& a, auto& b){ return a.first > b.first; });
                for (int i = 0; i < 5; i++)
                    printf("%d(%.3f) ", scored[i].second, scored[i].first);
                printf("\n");
            }

            gguf_free(gctx);
            if (ref_ctx) ggml_free(ref_ctx);
        }
    }

    // If 3rd arg is a .bin file, load and run as preprocessed CHW
    if (argc >= 3) {
        const char* ext = strrchr(argv[2], '.');
        if (ext && strcmp(ext, ".bin") == 0) {
            FILE* bf = fopen(argv[2], "rb");
            if (bf) {
                fseek(bf, 0, SEEK_END);
                long bsz = ftell(bf);
                fseek(bf, 0, SEEK_SET);
                int nf = bsz / sizeof(float);
                std::vector<float> chw(nf);
                fread(chw.data(), sizeof(float), nf, bf);
                fclose(bf);
                printf("\n--- Running from preprocessed CHW (%d floats) ---\n", nf);

                // Declare extern since it's not in the header
                extern const char* ppformulanet_ocr_recognize_chw(
                    ppformulanet_ocr_context*, const float*, int*);
                int olen = 0;
                const char* res = ppformulanet_ocr_recognize_chw(ctx, chw.data(), &olen);
                if (res) printf("LaTeX: %s\n", res);
                else printf("(no result)\n");
            }
        }
    }

    ppformulanet_ocr_free(ctx);
    return 0;
}
