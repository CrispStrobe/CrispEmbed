// tests/test_math_ocr_diff.cpp — compare C++ encoder output vs ONNX reference
#include "math_ocr.h"
#include <cmath>
#include <cstdio>
#include <vector>

float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref_enc_output.bin>\n", argv[0]);
        return 1;
    }

    math_ocr_context* ctx = math_ocr_init(argv[1], 4);
    if (!ctx) return 1;

    // Load reference
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Can't open ref: %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    int n_ref = sz / sizeof(float);
    std::vector<float> ref(n_ref);
    fread(ref.data(), sizeof(float), n_ref, f);
    fclose(f);
    printf("Reference: %d floats (578×384=%d)\n", n_ref, 578*384);

    const auto* hp = math_ocr_get_hparams(ctx);
    int S = hp->image_size;

    // Same test image
    std::vector<float> img(S*S, 0.8f);
    for (int y = S/2-2; y < S/2+2; y++)
        for (int x = S/4; x < 3*S/4; x++) img[y*S+x] = 0.1f;

    math_ocr_recognize(ctx, img.data(), S, S, nullptr);

    int n_tok = 0, hidden = 0;
    const float* enc = math_ocr_get_encoder_output(ctx, &n_tok, &hidden);
    if (!enc) { fprintf(stderr, "No encoder output\n"); return 1; }
    printf("C++ output: %d tokens × %d hidden\n\n", n_tok, hidden);

    // Overall cosine similarity
    int total = std::min(n_tok * hidden, n_ref);
    float cos = cosine_sim(enc, ref.data(), total);
    printf("Overall cosine similarity: %.6f %s\n\n", cos, cos > 0.99 ? "PASS" : "FAIL");

    // Per-token comparison
    for (int t : {0, 1, 2, 50, 100, 200, 500, 577}) {
        if (t >= n_tok || t * hidden + hidden > n_ref) continue;
        float tc = cosine_sim(enc + t*hidden, ref.data() + t*hidden, hidden);
        float max_abs = 0;
        for (int i = 0; i < hidden; i++) {
            float d = fabsf(enc[t*hidden+i] - ref[t*hidden+i]);
            if (d > max_abs) max_abs = d;
        }
        printf("  token %3d: cos=%.6f max_abs=%.4f  cpp=[%.4f %.4f %.4f] ref=[%.4f %.4f %.4f] %s\n",
               t, tc, max_abs,
               enc[t*hidden], enc[t*hidden+1], enc[t*hidden+2],
               ref[t*hidden], ref[t*hidden+1], ref[t*hidden+2],
               tc > 0.99 ? "PASS" : "FAIL");
    }

    math_ocr_free(ctx);
    return 0;
}
