// test_sam2_diff.cpp — layer-by-layer parity of SAM 2.1 (src/sam2.cpp) against PyTorch on the CPU.
//
// Usage: test-sam2-diff <model.gguf> <ref.gguf>
//   ref.gguf from tools/dump_sam2_reference.py (input_image, hiera.patch, hiera.block.NN, neck.N,
//   image_embed, high_res_0/1, point_coords, point_labels, mask_logits, iou).
// Passes when every stage has cosine >= 0.999 (CRISPEMBED_DIFF_COS overrides) and the four final
// masks (logits > 0) agree with the reference on at least 99.5 % of their union.

#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "sam2.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(const crispembed_diff::Ref & ref, const std::string & name, const float * data, size_t n_elem) {
    auto r = ref.compare(name, data, n_elem);
    const bool ok = r.is_pass_global();
    printf("  %-16s cos=%.6f  max_abs=%.3e  mean_abs=%.3e  %s\n", name.c_str(), r.cos_global, r.max_abs, r.mean_abs,
           ok ? "PASS" : "FAIL");
    (ok ? n_pass : n_fail)++;
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-sam2-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        printf("cannot load %s\n", argv[2]);
        return 1;
    }
    sam2_context * ctx = sam2_init(argv[1], 4);
    if (!ctx) return 1;
    auto [input, n_input] = ref.get_f32("input_image");
    const int S = sam2_image_size(ctx), M = sam2_mask_size(ctx);
    if (!input || n_input != (size_t)3 * S * S) {
        printf("no input_image of 3 x %d x %d in the reference\n", S, S);
        sam2_free(ctx);
        return 1;
    }
    sam2_set_probing(ctx, true);
    if (sam2_set_image_f32(ctx, input) != 0) {
        sam2_free(ctx);
        return 1;
    }
    printf("encoder (%s):\n", sam2_backend_name(ctx));
    for (auto & [name, data] : sam2_probes(ctx))
        if (ref.has(name)) check(ref, name, data.data(), data.size());

    auto [coords, n_coords] = ref.get_f32("point_coords");
    auto [labels_f, n_labels] = ref.get_f32("point_labels");
    std::vector<int> labels(n_labels);
    for (size_t i = 0; i < n_labels; i++) labels[i] = (int)std::lround(labels_f[i]);
    std::vector<float> logits((size_t)4 * M * M);
    float scores[4];
    if (sam2_predict(ctx, coords, labels.data(), (int)n_labels, logits.data(), scores) != 0) {
        sam2_free(ctx);
        return 1;
    }
    printf("decoder (%d points):\n", (int)n_labels);
    check(ref, "mask_logits", logits.data(), logits.size());
    check(ref, "iou", scores, 4);
    auto [ref_logits, n_ref] = ref.get_f32("mask_logits");
    for (int t = 0; t < 4 && n_ref == logits.size(); t++) {
        size_t both = 0, either = 0;
        for (int i = 0; i < M * M; i++) {
            const bool a = logits[(size_t)t * M * M + i] > 0.0f, b = ref_logits[(size_t)t * M * M + i] > 0.0f;
            both += a && b;
            either += a || b;
        }
        const double iou = either ? (double)both / (double)either : 1.0;
        printf("  mask %d           IoU=%.6f  score=%.4f  %s\n", t, iou, scores[t], iou >= 0.995 ? "PASS" : "FAIL");
        (iou >= 0.995 ? n_pass : n_fail)++;
    }
    sam2_free(ctx);
    printf("%d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
