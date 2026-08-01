#include "ocr_detect.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

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
    ocr_detect::context * ctx = nullptr;
    if (!ocr_detect::load(&ctx, argv[1], 1)) return 5;
    const auto boxes = ocr_detect::detect(ctx, input.first, (int)shape[1], (int)shape[0], 0.3f, 0.5f, 1.5f);
    int map_h = 0, map_w = 0;
    const float * prob = ocr_detect::get_prob_map(ctx, &map_h, &map_w);
    const auto reference = ref.get_f32("prob_map_sigmoid");
    if (!prob || !reference.first || (size_t)map_h * map_w != reference.second) {
        ocr_detect::free(ctx);
        return 6;
    }
    const auto report = ref.compare("prob_map_sigmoid", prob, reference.second, 0);
    std::printf(
        "dbnet-diff prob_map_sigmoid n=%zu max=%.9g mean=%.9g rms=%.9g cos=%.7f global=%.7f mine=%.9g ref=%.9g %s\n",
        reference.second, report.max_abs, report.mean_abs, report.rms, report.cos_min, report.cos_global,
        report.mine_norm, report.ref_norm, report.is_pass() ? "PASS" : "FAIL");
    std::printf("dbnet-diff decoded_boxes=%zu\n", boxes.size());
    ocr_detect::free(ctx);
    const int rc = report.is_pass() ? 0 : 1;
    core_util::clean_exit(rc);
    return rc;
}
