// tests/test_bidirlm_vision_diff.cpp — BidirLM-Omni vision-tower parity via crispembed-diff.
// Usage: ./test-bidirlm-vision-diff bidirlm-omni.gguf bidirlm-vision-ref.gguf
//
// Feeds the reference's stored pixel_values/image_grid_thw (the exact HF
// preprocessor output) to crispembed_encode_image_raw and compares the tower's
// image_embeds + each deepstack slab against the HF BidirLMOmniVisionModel
// reference (tools/dump_bidirlm_vision_reference.py). cosine is scale-invariant,
// so an L2-norm mismatch is harmless; a graph-scramble regression craters cos.
// Validated on q8_0 (image_embeds cos 0.997, deepstack 0.9998/0.9937); q4_k sits
// at its quant floor (~0.97) and is not the CI target.

#include "crispembed.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    printf("  %s%s%s %s\n", c ? GREEN "[PASS]" : RED "[FAIL]", "", RESET, n);
    if (c)
        n_pass++;
    else
        n_fail++;
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s bidirlm-omni.gguf bidirlm-vision-ref.gguf [cos_threshold]\n", argv[0]);
        return 1;
    }
    const char * gguf = argv[1];
    const char * ref_path = argv[2];
    const float thresh = argc > 3 ? (float)atof(argv[3]) : 0.99f;

    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "failed to load ref: %s\n", ref_path);
        return 1;
    }

    auto pv = ref.get_f32("pixel_values");
    auto gt = ref.get_f32("image_grid_thw");
    if (!pv.first || !gt.first) {
        fprintf(stderr, "ref missing pixel_values / image_grid_thw\n");
        return 1;
    }
    const int n_images = (int)(gt.second / 3);
    std::vector<int32_t> grid(gt.second);
    for (size_t i = 0; i < gt.second; i++) grid[i] = (int32_t)llroundf(gt.first[i]);
    // n_patches = pre-merge patch count = Σ t*h*w per image (robust to GGUF's
    // column-major tensor-dim order, unlike reading pixel_values' shape).
    int n_patches = 0;
    for (int im = 0; im < n_images; im++) n_patches += grid[im * 3 + 0] * grid[im * 3 + 1] * grid[im * 3 + 2];

    crispembed_context * ctx = crispembed_init(gguf, 4);
    if (!ctx) {
        fprintf(stderr, "crispembed_init failed: %s\n", gguf);
        return 1;
    }

    int n_merged = 0, dim = 0, n_ds = 0;
    const float * out =
        crispembed_encode_image_raw(ctx, pv.first, n_patches, grid.data(), n_images, &n_merged, &dim, &n_ds);
    check("encode_image_raw returns vision features", out != nullptr && n_merged > 0 && dim > 0);
    if (!out || n_merged <= 0) {
        crispembed_free(ctx);
        return 1;
    }
    printf("  n_merged=%d dim=%d n_deepstack=%d\n", n_merged, dim, n_ds);

    // Emit `<stage>: cos=<mean> max_abs=<x>` — the per-token MEAN cosine, matching
    // tests/test_bidirlm_vision.py's validated metric and run_one.py's parser (the
    // esrgan/safmn `cos=` convention). cos_min is the worst single row — a few
    // massive-activation deepstack rows (max_abs ~5) quantize poorly and tank it
    // without indicating a bug; a real graph scramble craters the mean too.
    const size_t per = (size_t)n_merged * dim;
    auto r_img = ref.compare("image_embeds", out, per);
    printf("  image_embeds: cos=%.6f max_abs=%.3e %s  (min-row=%.4f)\n", r_img.cos_mean, r_img.max_abs,
           r_img.cos_mean >= thresh ? "PASS" : "FAIL", r_img.cos_min);
    check("image_embeds cos >= threshold", r_img.found && r_img.cos_mean >= thresh);

    for (int k = 0; k < n_ds; k++) {
        std::string nm = "deepstack." + std::to_string(k);
        auto r = ref.compare(nm, out + (size_t)(1 + k) * per, per);
        printf("  %s: cos=%.6f max_abs=%.3e %s  (min-row=%.4f)\n", nm.c_str(), r.cos_mean, r.max_abs,
               r.cos_mean >= thresh ? "PASS" : "FAIL", r.cos_min);
        check((nm + " cos >= threshold").c_str(), r.found && r.cos_mean >= thresh);
    }

    crispembed_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
