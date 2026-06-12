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

    auto [ref_patches, n_ref_patches_elem] = ref.get_f32("input_patches");
    if (!ref_patches || n_ref_patches_elem == 0) {
        fprintf(stderr, "WARNING: no input_patches in reference — "
                "cannot compare preprocessing\n");
    }

    // Derive grid_thw from vis_patch_embed shape.
    // vis_patch_embed is (n_patches, D_v=1280). We know the image dimensions
    // from the reference dump. For the test image (640x480), after Qwen2VL
    // preprocessing: 644x476 → patches = 34x46 = 1564.
    // The patch embed output shape tells us n_patches.
    int32_t grid_thw[3] = {1, 0, 0};
    {
        auto vis_shape = ref.shape("vis_patch_embed");
        if (vis_shape.size() >= 2) {
            // GGUF stores (D, N) in column-major — shape[1] is n_patches
            int n_p = (int)vis_shape[1];
            auto merger_shape = ref.shape("vis_merger_output");
            if (merger_shape.size() >= 2) {
                int n_merged = (int)merger_shape[1];
                // n_patches = h * w, n_merged = (h/2)*(w/2)
                // Find h,w closest to square (prefer portrait-ish)
                int best_h = 0, best_w = 0;
                int best_diff = n_p;
                for (int h = 2; h * h <= n_p * 2; h += 2) {
                    if (n_p % h == 0) {
                        int w = n_p / h;
                        if (w % 2 == 0 && (h / 2) * (w / 2) == n_merged) {
                            int diff = std::abs(h - w);
                            if (diff < best_diff) {
                                best_diff = diff;
                                best_h = h;
                                best_w = w;
                            }
                        }
                    }
                }
                grid_thw[1] = best_h;
                grid_thw[2] = best_w;
            }
        }
    }

    bool has_vision = (grid_thw[1] > 0 && grid_thw[2] > 0);
    if (!has_vision) {
        printf("No vision tensors in reference — LLM-only test mode\n");
    }

    int n_patches = grid_thw[0] * grid_thw[1] * grid_thw[2];
    printf("\nGrid: t=%d h=%d w=%d  n_patches=%d\n",
           grid_thw[0], grid_thw[1], grid_thw[2], n_patches);

    // Print reference tensor info
    if (ref.has("vis_patch_embed")) {
        auto [pe_data, pe_n] = ref.get_f32("vis_patch_embed");
        printf("vis_patch_embed ref: n_elem=%zu, first5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               pe_n, pe_data[0], pe_data[1], pe_data[2], pe_data[3], pe_data[4]);
    }
    if (ref.has("input_patches")) {
        auto [p_data, p_n] = ref.get_f32("input_patches");
        printf("input_patches ref: n_elem=%zu, first5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               p_n, p_data[0], p_data[1], p_data[2], p_data[3], p_data[4]);
        auto s = ref.shape("input_patches");
        printf("  shape: [%lld", (long long)s[0]);
        for (size_t i = 1; i < s.size(); i++) printf(", %lld", (long long)s[i]);
        printf("]  (GGUF column-major)\n");
    }

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

    // Compare merger output
    if (has_vision && ref.has("vis_merger_output")) {
        auto [ref_merger, n_merger] = ref.get_f32("vis_merger_output");
        if (result.image_embeds && ref_merger) {
            auto r = ref.compare("vis_merger_output",
                                  result.image_embeds,
                                  (size_t)result.n_merged * result.embed_dim);
            printf("  vis_merger_output: cos_min=%.6f max_abs=%.2e %s\n",
                   r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
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
