// test_paddleocr_vl_diff.cpp — parity test for PaddleOCR-VL vision+projector.
//
// Usage: test-paddleocr-vl-diff <model.gguf> <ref.gguf>
//
// Loads the model via qwen2vl_ocr engine with diff_ref_path set.
// Uses synthetic patches matching the reference grid_thw.

#include "qwen2vl_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *ref_path = argv[2];

    int n_pass = 0, n_fail = 0;

    // ── Load reference ──
    printf("Loading reference: %s\n", ref_path);
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "Failed to load reference GGUF\n");
        return 1;
    }

    printf("Reference tensors:\n");
    for (auto &name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf("  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++)
            printf("%s%lld", i ? "," : "", (long long)s[i]);
        printf("]\n");
    }

    // Grid from reference dumper (200x60 image with min_pixels=784 → 56x196 → 4x14 grid)
    int32_t grid_thw[3] = {1, 4, 14};
    int n_patches = grid_thw[0] * grid_thw[1] * grid_thw[2];
    printf("Grid: t=%d h=%d w=%d → %d patches\n",
           grid_thw[0], grid_thw[1], grid_thw[2], n_patches);

    // ── Load model ──
    printf("\nLoading model: %s\n", model_path);
    qwen2vl_ocr::context ctx;
    ctx.diff_ref_path = ref_path;
    if (!qwen2vl_ocr::load(ctx, model_path, 4, 1)) {
        fprintf(stderr, "[FAIL] model load\n");
        return 1;
    }
    n_pass++;
    printf("[PASS] model loads\n");
    printf("  has_position_embed: %s\n",
           ctx.m.vhp.has_position_embed ? "true" : "false");

    // ── Create synthetic patches ──
    // Use random-like deterministic data (won't match Python's preprocessing
    // but we can still verify the graph runs and check shapes)
    int patch_flat = 3 * 14 * 14;  // channels * P * P
    // For proper parity, we'd use the same patches as Python.
    // For now, just check the model loads and runs.
    std::vector<float> patches(n_patches * patch_flat, 0.0f);
    for (int i = 0; i < (int)patches.size(); i++) {
        patches[i] = sinf((float)i * 0.01f) * 0.5f;
    }

    // ── Run vision encoder ──
    printf("\nRunning vision encoder...\n");
    qwen2vl_ocr::vision_result vis_out;
    if (!qwen2vl_ocr::encode_vision(ctx, patches.data(), n_patches,
                                     grid_thw, vis_out)) {
        fprintf(stderr, "[FAIL] vision encoder\n");
        n_fail++;
    } else {
        n_pass++;
        printf("[PASS] vision encoder runs: %d merged, %d dim\n",
               vis_out.n_merged, vis_out.embed_dim);

        // Check shape matches reference
        if (ref.has("projector_output")) {
            auto s = ref.shape("projector_output");
            // GGUF stores in column-major: [dim, n_tokens] for a [n_tokens, dim] tensor
            int ref_dim = (s.size() >= 2) ? (int)s[0] : vis_out.embed_dim;
            int ref_tokens = (s.size() >= 2) ? (int)s[1] : (int)s[0] / ref_dim;
            if (ref_tokens == vis_out.n_merged && ref_dim == vis_out.embed_dim) {
                n_pass++;
                printf("[PASS] projector shape: %d x %d\n",
                       vis_out.n_merged, vis_out.embed_dim);
            } else {
                n_fail++;
                printf("[FAIL] projector shape: got %d x %d, ref %d x %d\n",
                       vis_out.n_merged, vis_out.embed_dim,
                       ref_tokens, ref_dim);
            }
        }
    }

    qwen2vl_ocr::vision_result_free(vis_out);
    qwen2vl_ocr::free_(ctx);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
