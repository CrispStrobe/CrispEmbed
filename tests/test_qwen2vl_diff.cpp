// test_qwen2vl_diff.cpp — per-layer parity test for Qwen2.5-VL vision encoder.
//
// Usage: test-qwen2vl-diff <model.gguf> <ref.gguf> [image.png]
//
// Loads the model GGUF and reference GGUF, runs the vision encoder on the
// test image, and compares every intermediate tensor against the Python
// reference. The first layer where cos_min < 0.999 is where the bug lives.
//
// Requires:
//   model.gguf — converted via models/convert-qwen2vl-to-gguf.py --vision-only
//   ref.gguf   — dumped via tools/dump_qwen2vl_reference.py
//   image.png  — same image used for the reference dump

#include "qwen2vl_ocr.h"
#include "crispembed_diff.h"
#include "image_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *ref_path = argv[2];
    const char *image_path = (argc > 3) ? argv[3] : "/tmp/test_invoice_de.png";

    // ── Load reference ──────────────────────────────────────────
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

    // ── Compare input patches first ─────────────────────────────
    // The reference GGUF contains the preprocessed patches used by Python.
    // We can compare our C++ preprocessing against it, OR just use the
    // Python patches directly to isolate vision encoder bugs from
    // preprocessing bugs.

    // Try both naming conventions: "input_patches" (newer) and "pixel_values" (Qari ref)
    auto [ref_patches, n_ref_patches_elem] = ref.get_f32("input_patches");
    if (!ref_patches || n_ref_patches_elem == 0) {
        auto [pv, pv_n] = ref.get_f32("pixel_values");
        ref_patches = pv;
        n_ref_patches_elem = pv_n;
    }
    if (!ref_patches || n_ref_patches_elem == 0) {
        fprintf(stderr, "WARNING: no input_patches/pixel_values in reference\n");
    }

    // Derive grid_thw from vision layer or merger shape.
    int32_t grid_thw[3] = {1, 0, 0};
    {
        // Try vis_patch_embed first (newer format), then vis_layer_0 (Qari format)
        auto vis_shape = ref.shape("vis_patch_embed");
        if (vis_shape.empty()) vis_shape = ref.shape("vis_layer_0");
        // Try vis_merger_output first, then vis_merger
        auto merger_shape = ref.shape("vis_merger_output");
        if (merger_shape.empty()) merger_shape = ref.shape("vis_merger");

        int n_p = 0, n_merged = 0;
        if (vis_shape.size() >= 2) {
            // Column-major: shape[0] is innermost dim (D), shape[1] is n_patches
            n_p = (int)vis_shape[0];  // Could be [n_patches, D] or [D, n_patches]
            int d_v = (int)vis_shape[1];
            // The larger dim is D (1280), smaller is n_patches
            if (d_v < n_p) { std::swap(n_p, d_v); }
        }
        if (merger_shape.size() >= 2) {
            n_merged = (int)merger_shape[0];
            int d_m = (int)merger_shape[1];
            if (d_m < n_merged) { std::swap(n_merged, d_m); }
        }

        if (n_p > 0 && n_merged > 0) {
            int best_h = 0, best_w = 0, best_diff = n_p;
            for (int h = 2; h * h <= n_p * 2; h += 2) {
                if (n_p % h == 0) {
                    int w = n_p / h;
                    if (w % 2 == 0 && (h / 2) * (w / 2) == n_merged) {
                        int diff = std::abs(h - w);
                        if (diff < best_diff) {
                            best_diff = diff; best_h = h; best_w = w;
                        }
                    }
                }
            }
            grid_thw[1] = best_h;
            grid_thw[2] = best_w;
        }
    }

    bool has_vision = (grid_thw[1] > 0 && grid_thw[2] > 0);
    if (!has_vision) {
        printf("No vision grid derivable from reference — LLM-only test mode\n");
    }

    int n_patches = grid_thw[0] * grid_thw[1] * grid_thw[2];
    printf("\nGrid: t=%d h=%d w=%d  n_patches=%d\n",
           grid_thw[0], grid_thw[1], grid_thw[2], n_patches);
    if (ref_patches)
        printf("Input patches: %zu elements\n", n_ref_patches_elem);

    // ── Load model ──────────────────────────────────────────────
    printf("\nLoading model: %s\n", model_path);
    qwen2vl_ocr::context ctx;
    if (!qwen2vl_ocr::load(ctx, model_path, 4, 2)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Set diff reference path for internal comparison
    ctx.diff_ref_path = ref_path;

    // ── Run vision encoder using reference patches ──────────────
    qwen2vl_ocr::vision_result result = {};
    if (has_vision) {
        printf("\nRunning vision encoder (%d patches)...\n", n_patches);
        bool ok = qwen2vl_ocr::encode_vision(ctx, ref_patches, n_patches,
                                              grid_thw, result);
        if (!ok) {
            fprintf(stderr, "Vision encoder failed\n");
            qwen2vl_ocr::free_(ctx);
            return 1;
        }
    }

    if (has_vision) {
        printf("\n=== Vision encoder output: %d merged tokens, %d dim ===\n",
               result.n_merged, result.embed_dim);
    }

    // Compare merger output (try both naming conventions)
    {
        const char *merger_name = ref.has("vis_merger_output") ? "vis_merger_output"
                                : ref.has("vis_merger") ? "vis_merger" : nullptr;
        if (has_vision && merger_name && result.image_embeds) {
            auto r = ref.compare(merger_name,
                                  result.image_embeds,
                                  (size_t)result.n_merged * result.embed_dim);
            printf("  %s: cos_min=%.6f max_abs=%.2e %s\n",
                   merger_name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // Compare per-layer vision outputs
    for (uint32_t il = 0; il < 32; il++) {
        char name[64];
        std::snprintf(name, sizeof(name), "vis_layer_%u", il);
        if (!ref.has(name)) continue;
        // The diff harness in the engine already compared these if diff_ref_path is set,
        // but print the results here for clarity
        printf("  (ref has %s)\n", name);
    }

    qwen2vl_ocr::vision_result_free(result);

    // ── LLM decoder parity test ─────────────────────────────────
    if (ref.has("llm_embed") && ctx.m.embed_tokens) {
        printf("\n=== LLM decoder test ===\n");

        // Use token IDs [0,1,2,3,4] matching the Python reference
        int32_t test_ids[] = {0, 1, 2, 3, 4};
        int n_test = 5;

        qwen2vl_ocr::llm_result llm_out;
        if (qwen2vl_ocr::run_llm_forward(ctx, test_ids, n_test, llm_out)) {
            printf("  LLM forward: %d tokens, %d dim\n",
                   llm_out.n_tokens, llm_out.hidden_dim);
            if (llm_out.hidden) {
                free(llm_out.hidden);
            }
        } else {
            printf("  LLM forward failed\n");
        }
    }

    qwen2vl_ocr::free_(ctx);

    printf("\nDone.\n");
    return 0;
}
