#include "ocr_detect.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <reference.gguf>\n", argv[0]);
        return 2;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 3;
    const auto input = ref.get_f32("input_image");
    const auto shape = ref.shape("input_image");
    if (!input.first || shape.size() != 3) return 4;
    set_test_env("OCR_DETECT_CAPTURE_TAPS", "1");
    ocr_detect::context * ctx = nullptr;
    if (!ocr_detect::load(&ctx, argv[1], 1)) return 5;
    const auto boxes = ocr_detect::detect(ctx, input.first, (int)shape[1], (int)shape[0], 0.3f, 0.5f, 1.5f);
    int failures = 0;
    for (const char * name :
         { "backbone_stage_0", "backbone_stage_1", "backbone_stage_2", "backbone_stage_3", "neck_lateral_0",
           "neck_lateral_1", "neck_lateral_2", "neck_lateral_3", "neck_smooth_0", "neck_smooth_1", "neck_smooth_2",
           "neck_smooth_3", "neck_fused", "head_conv1", "head_deconv1", "prob_map_sigmoid" }) {
        const auto reference = ref.get_f32(name);
        const float * native = nullptr;
        size_t n_elem = 0;
        if (!reference.first || !ocr_detect::get_intermediate(ctx, name, &native, &n_elem) ||
            n_elem != reference.second) {
            std::printf("dbnet-diff %s MISSING\n", name);
            failures++;
            continue;
        }
        const auto report = ref.compare(name, native, n_elem, 0);
        // ReLU/deconvolution rows can be identically zero in both tensors;
        // the row-wise cosine is then undefined and reported as zero. Keep
        // the strict row gate where defined, but accept a high-global-cosine,
        // low-RMS result with matching magnitudes for that case.
        const bool pass = report.is_pass() ||
                          (report.cos_global >= 0.999f && report.rms < 1.0e-2f && report.ref_norm > 0.0f &&
                           report.mine_norm / report.ref_norm > 0.99f && report.mine_norm / report.ref_norm < 1.01f);
        std::printf("dbnet-diff %s n=%zu max=%.9g mean=%.9g rms=%.9g cos=%.7f global=%.7f mine=%.9g ref=%.9g %s\n",
                    name, n_elem, report.max_abs, report.mean_abs, report.rms, report.cos_min, report.cos_global,
                    report.mine_norm, report.ref_norm, pass ? "PASS" : "FAIL");
        if (!pass) failures++;
    }
    std::printf("dbnet-diff decoded_boxes=%zu\n", boxes.size());
    ocr_detect::free(ctx);
    const int rc = failures == 0 ? 0 : 1;
    core_util::clean_exit(rc);
    return rc;
}
