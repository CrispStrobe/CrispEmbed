#include "qwen35_vision.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool compare(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto report = ref.compare(name, values.data(), values.size());
    const bool ok = report.found && ref.get_f32(name).second == values.size() && report.n_elem == values.size() &&
                    report.cos_min >= 0.99999f && report.max_abs <= 0.002f && std::isfinite(report.max_abs) &&
                    std::isfinite(report.cos_min);
    std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs, report.mean_abs,
                report.mine_norm, report.ref_norm);
    return ok;
}


static bool include(const char * name, void *) {
    return std::string(name).find("v.") == 0;
}
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    if (argc == 4 && std::string(argv[3]) != "--synthetic") return 2;
    gate("1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-vision", loaded))
        return 2;
    qwen35_ocr::vision_weights weights;
    if (!qwen35_ocr::map_small_vision(loaded.tensors, weights)) return 2;
    qwen35_ocr::vision_config config;
    if (argc == 4) {
        config.width = 24;
        config.heads = 3;
        config.intermediate = 48;
        config.output = 16;
        config.patch_width = 12;
        config.position_side = 7;
    }
    const auto patches = ref.get_f32("input.pixel_values");
    const auto shape = ref.shape("input.pixel_values");
    const auto grid = ref.get_f32("input.image_grid_thw");
    if (!patches.first || shape.size() != 2 || shape[0] != config.patch_width ||
        patches.second != size_t(shape[0] * shape[1]) || !grid.first || grid.second % 3)
        return 2;
    std::vector<int> dimensions;
    for (size_t i = 0; i < grid.second; ++i) {
        if (!std::isfinite(grid.first[i]) || grid.first[i] < 1 || grid.first[i] > 1024 ||
            std::floor(grid.first[i]) != grid.first[i])
            return 2;
        dimensions.push_back(int(grid.first[i]));
    }
    qwen35_ocr::vision_outputs output;
    bool ok = loaded.tensors.size() == 153 &&
              qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, output);
    if (!ok) {
        std::printf("FAIL native vision initialization/execution\n");
        return 1;
    }
    ok &= compare(ref, "prefill.vision.patch_embed", output.patch);
    ok &= compare(ref, "prefill.vision.position", output.position);
    ok &= compare(ref, "prefill.vision.embedded", output.embedded);
    ok &= compare(ref, "prefill.vision.rotary_cos", output.rotary_cos);
    ok &= compare(ref, "prefill.vision.rotary_sin", output.rotary_sin);
    for (size_t i = 0; i < output.hidden.size(); ++i) {
        if (!compare(ref, "prefill.vision.layer." + std::to_string(i), output.hidden[i])) {
            ok = false;
            break;
        }
    }
    ok &= compare(ref, "prefill.vision.merger.norm", output.merger_norm);
    ok &= compare(ref, "prefill.vision.merger.up", output.merger_up);
    ok &= compare(ref, "prefill.vision.merger.activation", output.merger_activation);
    ok &= compare(ref, "prefill.vision.merger", output.merged);
    // Repeated calls reset all temporary geometry/state. Fail-closed controls.
    qwen35_ocr::vision_outputs repeated;
    ok &= qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    ok &= repeated.merged == output.merged;
    gate("0");
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    gate("1");
    auto invalid = dimensions;
    invalid[1] += 1;
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), invalid, 4, repeated);
    invalid = dimensions;
    invalid[0] += 1;
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), invalid, 4, repeated);
    auto missing = weights;
    missing.patch = nullptr;
    ok &= !qwen35_ocr::vision_cpu(missing, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    auto perturbed = output.position;
    for (float & x : perturbed) x += 0.1f;
    ok &= ref.compare("prefill.vision.position", perturbed.data(), perturbed.size()).max_abs > 0.002f;
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("%s native composed Qwen3.5 vision from packed pixels, positional geometry, 12 blocks and merger; "
                "image frontend pending\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
