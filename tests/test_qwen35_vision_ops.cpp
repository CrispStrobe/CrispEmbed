#include "core/clean_exit.h"
#include "qwen35_vision_operators_impl.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static bool include(const char * name, void *) {
    return std::string(name).find("v.") == 0;
}
static std::vector<float> array(const crispembed_diff::Ref & ref, const std::string & name) {
    const auto value = ref.get_f32(name);
    return value.first ? std::vector<float>(value.first, value.first + value.second) : std::vector<float>();
}
using projection_inputs = std::map<std::string, std::vector<float>>;
static bool observe(const char * name, const float * input, int width, int tokens, void * user) {
    if (!name || !input || width < 1 || tokens < 1) return false;
    auto & inputs = *static_cast<projection_inputs *>(user);
    return inputs.emplace(name, std::vector<float>(input, input + size_t(width) * tokens)).second;
}
static int failures = 0, measurements = 0;
static bool measure(const crispembed_diff::Ref & ref, const char * mode, const std::string & name,
                    const std::vector<float> & values) {
    const auto r = ref.compare(name, values.data(), values.size());
    if (!r.found || values.empty() || r.n_elem != values.size() || ref.get_f32(name).second != values.size() ||
        !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }) ||
        !std::isfinite(r.cos_min) || !std::isfinite(r.max_abs))
        return false;
    ++measurements;
    std::printf(
        "VISION_METRIC %s %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
        mode, name.c_str(), r.n_elem, r.cos_min, r.cos_global, r.max_abs, r.mean_abs, r.rms, r.mine_norm, r.ref_norm);
    std::printf("MODE %s\n", mode);
    if (!qwen35_compare(ref, name, values)) ++failures;
    return true;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3 && (argc != 4 || std::string(argv[3]) != "--gate-controls")) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-vision-ops", loaded))
        return 2;
    qwen35_ocr::vision_weights weights;
    qwen35_ocr::vision_config config;
    if (loaded.tensors.size() != 153 || !qwen35_ocr::map_small_vision(loaded.tensors, weights)) return 2;
    auto pixels = array(ref, "input.pixel_values");
    auto original_grid = array(ref, "input.image_grid_thw");
    if (original_grid.size() != 3 || original_grid[0] != 1) return 2;
    std::vector<int> grid;
    for (float x : original_grid) {
        if (!std::isfinite(x) || x < 1 || x > 1024 || std::floor(x) != x) return 2;
        grid.push_back(int(x));
    }
    const int tokens = grid[1] * grid[2];
    if (pixels.size() != size_t(tokens) * config.patch_width) return 2;
    qwen35_ocr::vision_outputs native;
    projection_inputs observed;
    gate("CRISPEMBED_QWEN35_IMATRIX", "1");
    if (!qwen35_ocr::set_projection_observer(observe, &observed)) return 2;
    const bool ran = qwen35_ocr::vision_cpu(weights, config, pixels.data(), tokens, grid, 4, native);
    qwen35_ocr::set_projection_observer(nullptr, nullptr);
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    if (!ran || observed.size() != 51) return 2;
    bool ok = true;
    auto composed = [&](const std::string & name, const std::vector<float> & values) {
        ok &= measure(ref, "composed", "prefill.vision." + name, values);
    };
    composed("patch_embed", native.patch);
    composed("position", native.position);
    composed("embedded", native.embedded);
    composed("rotary_cos", native.rotary_cos);
    composed("rotary_sin", native.rotary_sin);
    for (size_t i = 0; i < native.hidden.size(); ++i) {
        const auto prefix = "probe." + std::to_string(i) + ".";
        const auto & p = native.probes[i];
        const auto weight_prefix = "v.blocks." + std::to_string(i) + ".";
        const auto & norm1 = observed.at(weight_prefix + "attn.qkv.weight");
        const auto & pooled = observed.at(weight_prefix + "attn.proj.weight");
        const auto & norm2 = observed.at(weight_prefix + "mlp.linear_fc1.weight");
        const auto & activation = observed.at(weight_prefix + "mlp.linear_fc2.weight");
        composed(prefix + "norm1", norm1);
        composed(prefix + "pooled", pooled);
        composed(prefix + "norm2", norm2);
        composed(prefix + "activation", activation);
        if (!p.norm1.empty()) {
            ok &= norm1 == p.norm1 && pooled == p.pooled && norm2 == p.norm2 && activation == p.activation;
            composed(prefix + "qkv", p.qkv);
            composed(prefix + "attention", p.attention);
            composed(prefix + "up", p.up);
            composed(prefix + "down", p.down);
        }
        composed("layer." + std::to_string(i), native.hidden[i]);
    }
    composed("merger.norm", native.merger_norm);
    composed("merger.up", native.merger_up);
    composed("merger.activation", native.merger_activation);
    composed("merger", native.merged);
    const auto cosine = array(ref, "prefill.vision.rotary_cos"), sine = array(ref, "prefill.vision.rotary_sin");
    auto isolated = [&](int layer, const std::string & op, const std::string & input_name, const std::string & name) {
        const auto input = array(ref, input_name);
        std::vector<float> values;
        if (!qwen35_ocr::vision_test_operator_cpu(weights, config, layer, op.c_str(), input, grid, cosine, sine, 4,
                                                  values)) {
            ok = false;
            std::printf("EXECUTION_FAILURE %s\n", name.c_str());
            return;
        }
        ok &= measure(ref, "same-input", name, values);
    };
    isolated(-1, "patch_embed", "input.pixel_values", "prefill.vision.patch_embed");
    for (int i = 0; i < 12; ++i) {
        const auto prefix = "prefill.vision.probe." + std::to_string(i) + ".";
        for (const auto & pair : { std::pair<const char *, const char *>{ "norm1", "input" },
                                   { "qkv", "norm1" },
                                   { "pooled", "qkv" },
                                   { "attention", "pooled" },
                                   { "norm2", "norm2_input" },
                                   { "up", "norm2" },
                                   { "activation", "up" },
                                   { "down", "activation" } })
            isolated(i, pair.first, prefix + pair.second, prefix + pair.first);
    }
    isolated(-1, "merger.norm", "prefill.vision.merger.input", "prefill.vision.merger.norm");
    isolated(-1, "merger.up", "prefill.vision.merger.norm", "prefill.vision.merger.up");
    isolated(-1, "merger.activation", "prefill.vision.merger.up", "prefill.vision.merger.activation");
    isolated(-1, "merger", "prefill.vision.merger.activation", "prefill.vision.merger");
    std::vector<float> rejected;
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "0");
    ok &= !qwen35_ocr::vision_test_operator_cpu(weights, config, -1, "patch_embed", pixels, grid, cosine, sine, 4,
                                                rejected);
    if (argc == 4) {
        qwen35_ocr::vision_outputs ordinary;
        ok &= qwen35_ocr::vision_cpu(weights, config, pixels.data(), tokens, grid, 4, ordinary);
        ok &= ordinary.merged == native.merged && ordinary.hidden == native.hidden;
        ok &= ordinary.probes.size() == 12 && ordinary.probes[0].norm1.empty() && !ordinary.probes[5].norm1.empty();
    }
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    gate("CRISPEMBED_QWEN35_OCR", "0");
    ok &= !qwen35_ocr::vision_test_operator_cpu(weights, config, -1, "patch_embed", pixels, grid, cosine, sine, 4,
                                                rejected);
    gate("CRISPEMBED_QWEN35_OCR", "1");
    pixels.pop_back();
    ok &= !qwen35_ocr::vision_test_operator_cpu(weights, config, -1, "patch_embed", pixels, grid, cosine, sine, 4,
                                                rejected);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("VISION_OPS finite=%d metrics=%d qualified_failures=%d controls=%d observer=%zu\n", ok, measurements,
                failures, ok, observed.size());
    // Failures are retained diagnostic outcomes, never tolerance changes.
    return ok && measurements == 178 ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
