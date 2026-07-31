#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "ppocrv6_det.h"

#include <cstdio>
#include <string>

static int run(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <det.gguf> <ref.gguf> <image>\n", argv[0]);
        return 1;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;
    auto * ctx = ppocrv6_det::init(argv[1], 1);
    if (!ctx) return 1;
    auto boxes = ppocrv6_det::detect_file(ctx, argv[3], 0.2f);
    bool pass = true;
    for (const char * name : { "stem1", "stem2b", "stem_pooled", "stem3", "stem4", "block0_dw", "block0_pool",
                               "block0_gate", "block0_se", "block0_cm1", "block0_out" }) {
        size_t n = 0;
        const float * data = ppocrv6_det::last_stage(ctx, name, &n);
        auto r = data ? ref.compare("ppocrv6." + std::string(name), data, n) : crispembed_diff::Report{};
        printf("%s cos_min=%.9f cos_mean=%.9f |mine|=%.9g |ref|=%.9g max_abs=%.9g rms=%.9g %s\n", name, r.cos_min,
               r.cos_mean, r.mine_norm, r.ref_norm, r.max_abs, r.rms, r.found && r.is_pass(0.999f) ? "PASS" : "FAIL");
        pass = pass && r.found && r.is_pass(0.999f);
    }
    for (int i = 0; i < 4; ++i) {
        size_t n = 0;
        const float * data = ppocrv6_det::last_stage(ctx, ("backbone_stage" + std::to_string(i)).c_str(), &n);
        auto r = data ? ref.compare("ppocrv6.backbone_stage" + std::to_string(i), data, n) : crispembed_diff::Report{};
        printf("backbone_stage%d cos_min=%.9f cos_mean=%.9f |mine|=%.9g |ref|=%.9g max_abs=%.9g rms=%.9g %s\n", i,
               r.cos_min, r.cos_mean, r.mine_norm, r.ref_norm, r.max_abs, r.rms,
               r.found && r.is_pass(0.999f) ? "PASS" : "FAIL");
        pass = pass && r.found && r.is_pass(0.999f);
    }
    {
        size_t n = 0;
        const float * data = ppocrv6_det::last_stage(ctx, "neck_output", &n);
        auto r = data ? ref.compare("ppocrv6.neck_output", data, n) : crispembed_diff::Report{};
        printf("neck_output cos_min=%.9f cos_mean=%.9f |mine|=%.9g |ref|=%.9g max_abs=%.9g rms=%.9g %s\n", r.cos_min,
               r.cos_mean, r.mine_norm, r.ref_norm, r.max_abs, r.rms, r.found && r.is_pass(0.999f) ? "PASS" : "FAIL");
        pass = pass && r.found && r.is_pass(0.999f);
    }
    int h = 0, w = 0;
    const float * prob = ppocrv6_det::last_probability(ctx, &h, &w);
    printf("boxes=%zu probability=%dx%d\n", boxes.size(), w, h);
    if (!prob) {
        ppocrv6_det::free(ctx);
        return 1;
    }
    auto r = ref.compare("ppocrv6.prob_map_sigmoid", prob, (size_t)h * w);
    printf("prob_map_sigmoid cos_min=%.9f cos_mean=%.9f |mine|=%.9g |ref|=%.9g max_abs=%.9g rms=%.9g %s\n", r.cos_min,
           r.cos_mean, r.mine_norm, r.ref_norm, r.max_abs, r.rms, r.found && r.is_pass(0.999f) ? "PASS" : "FAIL");
    ppocrv6_det::free(ctx);
    return pass && r.found && r.is_pass(0.999f) ? 0 : 1;
}

int main(int argc, char ** argv) {
    int rc = run(argc, argv);
    core_util::clean_exit(rc);
}
