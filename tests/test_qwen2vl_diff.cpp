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

    const char *patches_name = ref.has("input_patches") ? "input_patches" : "pixel_values";
    const char *merger_name = ref.has("vis_merger_output") ? "vis_merger_output" : "vis_merger";

    auto [ref_patches, n_ref_patches_elem] = ref.get_f32(patches_name);
    if (!ref_patches || n_ref_patches_elem == 0) {
        fprintf(stderr, "WARNING: no input patches in reference — "
                "cannot compare preprocessing\n");
    }

    // Derive grid_thw from vis_patch_embed shape.
    // vis_patch_embed is (n_patches, D_v=1280). We know the image dimensions
    // from the reference dump. For the test image (640x480), after Qwen2VL
    // preprocessing: 644x476 → patches = 34x46 = 1564.
    // The patch embed output shape tells us n_patches.
    int32_t grid_thw[3] = {1, 0, 0};
    {
        auto vis_shape = ref.shape(ref.has("vis_patch_embed") ? "vis_patch_embed" : patches_name);
        if (vis_shape.size() >= 2) {
            int n_p = 0;
            if (ref.has("vis_patch_embed")) {
                // Newer refs store GGUF shape (D, N).
                n_p = (int)vis_shape[1];
            } else {
                // Qari Kaggle refs store pixel_values as numpy shape (N, D).
                n_p = (int)vis_shape[0];
            }
            auto merger_shape = ref.shape(merger_name);
            if (merger_shape.size() >= 2) {
                int n_merged = ref.has("vis_merger_output")
                    ? (int)merger_shape[1]
                    : (int)merger_shape[0];
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
    if (ref.has(patches_name)) {
        auto [p_data, p_n] = ref.get_f32(patches_name);
        printf("%s ref: n_elem=%zu, first5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               patches_name,
               p_n, p_data[0], p_data[1], p_data[2], p_data[3], p_data[4]);
        auto s = ref.shape(patches_name);
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
    if (has_vision && ref.has(merger_name)) {
        auto [ref_merger, n_merger] = ref.get_f32(merger_name);
        if (result.image_embeds && ref_merger) {
            auto r = ref.compare(merger_name,
                                  result.image_embeds,
                                  (size_t)result.n_merged * result.embed_dim);
            printf("  %s: cos_min=%.6f max_abs=%.2e %s\n",
                   merger_name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            // Per-row cos distribution
            int D = result.embed_dim;
            int nbad = 0;
            for (int t = 0; t < result.n_merged; t++) {
                const float *a = result.image_embeds + (size_t)t * D;
                const float *b = ref_merger + (size_t)t * D;
                double dot=0, na=0, nb=0;
                for (int k=0;k<D;k++){dot+=a[k]*b[k];na+=a[k]*a[k];nb+=b[k]*b[k];}
                double c = dot/(sqrt(na)*sqrt(nb)+1e-9);
                if (c < 0.999) { nbad++;
                    if (nbad <= 6)
                        printf("    row %d cos=%.5f |a|=%.1f |b|=%.1f\n",
                               t, c, sqrt(na), sqrt(nb));
                }
            }
            printf("    rows<0.999: %d/%d\n", nbad, result.n_merged);
        }
    }

    // ── LLM decoder parity test ─────────────────────────────────
    if ((ref.has("llm_embed") || ref.has("llm_layer_0") || ref.has("last_logits")) &&
        ref.has("token_ids") && ctx.m.embed_tokens) {
        printf("\n=== LLM decoder test ===\n");

        auto [ids_f, n_ids] = ref.get_f32("token_ids");
        std::vector<int32_t> test_ids(n_ids);
        for (size_t i = 0; i < n_ids; i++) test_ids[i] = (int32_t)ids_f[i];

        // Optionally splice the REFERENCE merger embeds to isolate LLM from vision.
        const float *use_embeds = result.image_embeds;
        int use_n_merged = result.n_merged;
        std::vector<float> ref_emb_copy;
        if (getenv("LLM_FROM_REF") && ref.has(merger_name)) {
            auto [rm, nrm] = ref.get_f32(merger_name);
            ref_emb_copy.assign(rm, rm + nrm);
            use_embeds = ref_emb_copy.data();
            use_n_merged = (int)(nrm / (size_t)result.embed_dim);
            printf("  [using REFERENCE merger embeds for LLM]\n");
        }

        qwen2vl_ocr::image_input img = {};
        if (use_embeds) {
            img.image_embeds = use_embeds;
            img.n_image_tokens = use_n_merged;
            img.grid_thw = grid_thw;
            img.n_images = 1;
        }

        qwen2vl_ocr::llm_result llm_out;
        if (qwen2vl_ocr::run_llm_forward(ctx, test_ids.data(), (int)test_ids.size(),
                                          llm_out, use_embeds ? &img : nullptr)) {
            printf("  LLM forward: %d tokens, %d dim\n",
                   llm_out.n_tokens, llm_out.hidden_dim);
            if (llm_out.logits && ref.has("last_logits")) {
                size_t off = (size_t)(llm_out.n_tokens - 1) * llm_out.vocab_size;
                auto r = ref.compare("last_logits", llm_out.logits + off,
                                     (size_t)llm_out.vocab_size);
                printf("  last_logits: cos_min=%.6f max_abs=%.2e %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            if (llm_out.logits) {
                free(llm_out.logits);
            }
            if (llm_out.hidden) {
                free(llm_out.hidden);
            }
            if (llm_out.kv_graph_ctx) {
                ggml_free(llm_out.kv_graph_ctx);
            }
        } else {
            printf("  LLM forward failed\n");
        }
    }

    // ── Decisive: generate using REFERENCE (ground-truth) merger embeds ──
    // If this OCRs correctly, our vision fidelity is the limiter, not the LLM.
    if (getenv("GEN_FROM_REF") && ref.has(merger_name) && ref.has("token_ids")) {
        printf("\n=== GENERATE from REFERENCE merger embeds ===\n");
        auto [ref_merger, n_merger] = ref.get_f32(merger_name);
        auto [ids_f, n_ids] = ref.get_f32("token_ids");
        int n_merged_ref = (int)(n_merger / (size_t)result.embed_dim);
        std::vector<int32_t> pids;
        if (getenv("QARI_PROMPT")) {
            // Build Qari long-OCR prompt, no system message.
            std::vector<int32_t> longp = {
                38214,374,279,2168,315,825,2150,315,264,2197,11,438,1632,438,
                1045,7112,62533,2213,429,572,8597,27432,369,432,13,4599,470,279,
                14396,1467,13042,315,419,2197,438,421,498,1033,5290,432,17712,13,
                3155,537,58023,3277,13};
            pids = {151644,872,198,151652};
            for (int i=0;i<n_merged_ref;i++) pids.push_back(151655);
            pids.push_back(151653);
            pids.insert(pids.end(), longp.begin(), longp.end());
            pids.insert(pids.end(), {151645,198,151644,77091,198});
            printf("  [Qari long prompt, %zu tokens]\n", pids.size());
        } else {
            pids.resize(n_ids);
            for (size_t i = 0; i < n_ids; i++) pids[i] = (int32_t)ids_f[i];
        }
        qwen2vl_ocr::generate_result g = {};
        bool ok = qwen2vl_ocr::generate(
            ctx, ref_merger, n_merged_ref, result.embed_dim, grid_thw,
            pids.data(), (int)pids.size(), 48, g);
        if (ok) {
            printf("  gen tokens: ");
            for (int id : g.token_ids) printf("%d ", id);
            printf("\n");
        }
    }

    qwen2vl_ocr::vision_result_free(result);

    qwen2vl_ocr::free_(ctx);

    printf("\nDone.\n");
    return 0;
}
