// test_internvl2_diff.cpp — per-layer parity test for InternVL2/2.5.
//
// Usage: test-internvl2-diff <model.gguf> <ref.gguf>
//
// Loads the model GGUF and reference GGUF, runs the vision encoder +
// projector on a synthetic test image, and compares every intermediate
// tensor against the Python reference.

#include "internvl2_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf( "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *ref_path = argv[2];

    // ── Load reference ──────────────────────────────────────────
    printf("Loading reference: %s\n", ref_path);
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        printf( "Failed to load reference GGUF\n");
        return 1;
    }

    printf( "Reference tensors:\n");
    for (auto &name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf( "  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++)
            printf( "%s%lld", i ? "," : "", (long long)s[i]);
        printf( "]\n");
    }

    // ── Load model ──────────────────────────────────────────────
    printf( "\nLoading model: %s\n", model_path);
    internvl2_ocr::context ctx;
    ctx.diff_ref_path = ref_path;

    if (!internvl2_ocr::load(ctx, model_path, 4, 2)) {
        printf( "Failed to load model\n");
        return 1;
    }

    // ── Prepare synthetic test image ────────────────────────────
    const int img_size = (int)ctx.m.vhp.image_size;
    std::vector<float> pixels(3 * img_size * img_size);

    printf("\nUsing synthetic gradient image (%dx%d)\n", img_size, img_size);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < img_size; y++) {
            for (int x = 0; x < img_size; x++) {
                float val = (float)(y * img_size + x) / (float)(img_size * img_size);
                pixels[c * img_size * img_size + y * img_size + x] =
                    (val - ctx.m.vhp.image_mean[c]) / ctx.m.vhp.image_std[c];
            }
        }
    }

    // ── Run vision encoder ──────────────────────────────────────
    printf( "\nRunning vision encoder...\n");
    internvl2_ocr::vision_result vr;
    printf( "  calling encode_vision_tile...\n");
    if (!internvl2_ocr::encode_vision_tile(ctx, pixels.data(), vr)) {
        printf( "Vision encode failed\n");
        internvl2_ocr::free_(ctx);
        return 1;
    }
    printf("Vision output: %d tokens, %d dim\n", vr.n_tokens, vr.hidden_dim);

    // ── Run pixel unshuffle + projector ─────────────────────────
    printf("\nRunning projector...\n");
    const float *no_cls = vr.hidden + vr.hidden_dim;
    int n_patches = vr.n_tokens - 1;

    internvl2_ocr::project_result pr;
    if (!internvl2_ocr::project_vision(ctx, no_cls, n_patches, pr)) {
        printf( "Projection failed\n");
        free(vr.hidden);
        internvl2_ocr::free_(ctx);
        return 1;
    }
    printf("Projector output: %d tokens, %d dim\n", pr.n_tokens, pr.embed_dim);

    // ── Compare projector output ────────────────────────────────
    {
        auto [ref_data, ref_n] = ref.get_f32("vis_proj_output");
        if (ref_data && ref_n > 0) {
            auto r = ref.compare("vis_proj_output", pr.embeds,
                                 pr.n_tokens * pr.embed_dim);
            printf("  vis_proj_output: cos=%.6f max_abs=%.6f %s\n",
                   r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // ── Run LLM forward (if reference has LLM layers) ───────────
    {
        auto [ref_data, ref_n] = ref.get_f32("llm_embed");
        if (ref_data && ref_n > 0) {
            printf("\nRunning LLM decoder...\n");
            int32_t test_tokens[] = {1, 100, 200, 300, 400};
            int n_tokens = 5;

            internvl2_ocr::llm_result lr;
            if (internvl2_ocr::run_llm_forward(ctx, test_tokens, n_tokens, lr)) {
                printf("LLM output: %d tokens, %d dim\n",
                       lr.n_tokens, lr.hidden_dim);
                free(lr.hidden);
                if (lr.logits) free(lr.logits);
            } else {
                printf( "LLM forward failed\n");
            }
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────
    free(vr.hidden);
    free(pr.embeds);
    internvl2_ocr::free_(ctx);

    printf("\nDone.\n");
    return 0;
}
