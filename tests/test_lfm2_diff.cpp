// tests/test_lfm2_diff.cpp — LFM2 per-layer parity via crispembed_diff.
// Usage: ./build/test-lfm2-diff lfm2-embed-q8_0.gguf /tmp/lfm2-ref.gguf ["text"]

#include "lfm2_embed.h"
#include "crispembed_diff.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

#include <cstdio>
#include <string>
#include <vector>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check_cos(const char * label, float cos_min, float thresh = 0.999f) {
    bool ok = cos_min >= thresh;
    printf("  %s[%s]%s %-44s cos=%.6f  (thr=%.3f)\n",
           ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET,
           label, cos_min, thresh);
    if (ok) n_pass++; else n_fail++;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf ref.gguf [\"text\"]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path   = argv[2];
    const char * text       = argc >= 4 ? argv[3] : "hello world";

    printf("LFM2 per-layer parity test\n");
    printf("  Model: %s\n  Ref:   %s\n  Text:  %s\n\n", model_path, ref_path, text);

    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) { fprintf(stderr, "Failed to load ref\n"); return 1; }

    // Backend
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_METAL
    backend = ggml_backend_metal_init();
#endif
    if (!backend) backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "No backend\n"); return 1; }

    // Load model
    lfm2_embed_ctx * ctx = lfm2_embed_load(model_path, backend);
    if (!ctx) {
        fprintf(stderr, "lfm2_embed_load failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    printf("Model loaded OK\n\n");

    // Run dump-mode encode
    auto entries = lfm2_embed_encode_dump(ctx, text);
    if (entries.empty()) {
        fprintf(stderr, "lfm2_embed_encode_dump returned empty\n");
        lfm2_embed_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Compare each stage against the Python reference
    printf("=== Per-layer cosine (ggml vs HF float32) ===\n");
    // Reference shapes: (T, H) for 2D, (H,) for 1D.
    // lfm2_embed.cpp stores 2D tensors as (H, T) in ggml —
    // memory layout is identical to Python's (T, H) row-major, so we compare flat.
    for (auto & e : entries) {
        // For 2D tensors (T tokens × H hidden), use row_dim=0 so each "row" is
        // one token's full 1024-dim hidden state — avoids artificially low cosines
        // from comparing tiny 3-element (per-T) slices.
        int rd = (e.T > 1) ? 0 : -1;
        auto r = ref.compare(e.name.c_str(), e.data.data(), e.data.size(), rd);
        if (!r.found) {
            printf("  %-44s [not in ref — skip]\n", e.name.c_str());
            continue;
        }
        const float thresh = (e.name == "cls_norm") ? 0.99f : 0.999f;
        check_cos(e.name.c_str(), r.cos_min, thresh);
        if (r.cos_min < thresh) {
            printf("        max_abs=%.2e  mean_abs=%.2e\n", r.max_abs, r.mean_abs);
            // Print first few C++ values for manual inspection
            printf("        cpp[:4]:");
            for (int k = 0; k < 4 && k < (int)e.data.size(); k++)
                printf(" %.5f", e.data[k]);
            printf("\n");
            // And reference values
            auto [ref_data, ref_n] = ref.get_f32(e.name.c_str());
            if (ref_data) {
                printf("        ref[:4]:");
                for (int k = 0; k < 4 && k < (int)ref_n; k++)
                    printf(" %.5f", ref_data[k]);
                printf("\n");
            }
        }
    }

    printf("\n--- Summary ---\n");
    printf("  PASS: %d   FAIL: %d\n", n_pass, n_fail);

    lfm2_embed_free(ctx);
    ggml_backend_free(backend);
    return n_fail > 0 ? 1 : 0;
}
