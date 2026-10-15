#include "core/clean_exit.h"
#include "qwen35_attention_audit.h"
#include "qwen35_math.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <string>

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
    const auto x = ref.get_f32(name);
    return x.first ? std::vector<float>(x.first, x.first + x.second) : std::vector<float>();
}
static bool finite(const std::vector<float> & x, size_t n) {
    return n > 0 && x.size() == n && std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
static int failures = 0, metrics = 0, controls = 0;
static bool equal(const std::string & name, const std::vector<float> & x, const std::vector<float> & y) {
    const bool ok = !x.empty() && x == y;
    double error = x.size() == y.size() ? 0 : INFINITY;
    if (x.size() == y.size())
        for (size_t i = 0; i < x.size(); ++i) error = std::max(error, std::abs(double(x[i]) - y[i]));
    ++controls;
    std::printf("ATTN_CONTROL %s pass=%d max_abs=%.9g n=%zu/%zu\n", name.c_str(), ok, error, x.size(), y.size());
    return ok;
}
static bool measure(const crispembed_diff::Ref & ref, const std::string & mode, const std::string & stage,
                    const std::vector<float> & values) {
    const auto m = ref.compare(stage, values.data(), values.size());
    if (!finite(values, ref.get_f32(stage).second) || !m.found || m.n_elem != values.size() ||
        !std::isfinite(m.cos_min) || !std::isfinite(m.max_abs))
        return false;
    ++metrics;
    std::printf(
        "ATTN_METRIC %s %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
        mode.c_str(), stage.c_str(), m.n_elem, m.cos_min, m.cos_global, m.max_abs, m.mean_abs, m.rms, m.mine_norm,
        m.ref_norm);
    std::printf("MODE %s\n", mode.c_str());
    if (!qwen35_compare(ref, stage, values)) ++failures;
    return true;
}
static bool project(const qwen35_ocr::vision_layer_weights & w, const std::vector<float> & norm, int tokens,
                    std::vector<float> & qkv) {
    std::vector<std::vector<float>> ys;
    if (!qwen35_ocr::project_cpu({ w.qkv }, norm.data(), 768, tokens, 4, ys)) return false;
    std::vector<float> bias(2304);
    if (!qwen35_ocr::tensor_floats(w.qkv_bias, bias.data())) return false;
    qkv = std::move(ys[0]);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] += bias[i % bias.size()];
    return finite(qkv, size_t(tokens) * 2304);
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    const auto original_grid = array(ref, "input.image_grid_thw");
    if (original_grid.size() != 3 || original_grid[0] != 1) return 2;
    std::vector<int> grid;
    for (float x : original_grid) {
        if (!std::isfinite(x) || x < 1 || x > 1024 || std::floor(x) != x) return 2;
        grid.push_back(int(x));
    }
    const int tokens = grid[1] * grid[2];
    const auto pixels = array(ref, "input.pixel_values");
    qwen35_ocr::vision_config config;
    if (tokens > 1024 || !finite(pixels, size_t(tokens) * config.patch_width)) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    qwen35_ocr::vision_weights weights;
    if (!backend ||
        !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-attention-trace", loaded) ||
        loaded.tensors.size() != 153 || !qwen35_ocr::map_small_vision(loaded.tensors, weights))
        return 2;
    const auto original_cosine = array(ref, "prefill.vision.rotary_cos"),
               original_sine = array(ref, "prefill.vision.rotary_sin");
    if (!finite(original_cosine, size_t(tokens) * 64) || !finite(original_sine, original_cosine.size())) return 2;
    const auto indices = array(ref, "input.vision_attention_queries");
    if (indices.size() != size_t(std::min(16, tokens))) return 2;
    std::vector<int> queries;
    for (size_t i = 0; i < indices.size(); ++i) {
        const int expected = indices.size() > 1 ? int(i * (tokens - 1) / (indices.size() - 1)) : 0;
        if (indices[i] != expected) return 2;
        queries.push_back(expected);
    }
    qwen35_ocr::vision_outputs native;
    if (!qwen35_ocr::vision_cpu(weights, config, pixels.data(), tokens, grid, 4, native) || native.probes.size() != 12)
        return 2;
    bool ok = true, native_controls = true;
    using namespace qwen35_attention_audit;
    auto record_trace = [&](const std::string & mode, const std::string & p, const trace_result & t, bool full,
                            const std::vector<float> * production_pooled = nullptr) {
        bool valid = true;
        if (full) {
            valid &= measure(ref, mode, p + "trace.query", t.query);
            valid &= measure(ref, mode, p + "trace.key", t.key);
            valid &= measure(ref, mode, p + "trace.value", t.value);
        }
        valid &= measure(ref, mode, p + "trace.scores", t.scores);
        valid &= measure(ref, mode, p + "trace.probabilities", t.probabilities);
        valid &= measure(ref, mode, p + "trace.pooled_sample", t.pooled_sample);
        valid &= measure(ref, mode, p + "pooled", production_pooled ? *production_pooled : t.pooled);
        return valid;
    };
    for (int index : { 5, 11 }) {
        const auto p = "prefill.vision.probe." + std::to_string(index) + ".";
        const auto & probe = native.probes[index];
        if (!finite(probe.qkv, size_t(tokens) * 2304) || !finite(probe.norm1, size_t(tokens) * 768)) return 2;
        std::vector<float> reproduced;
        if (!project(weights.layers[index], probe.norm1, tokens, reproduced)) return 2;
        const auto actual = trace(probe.qkv, config, tokens, native.rotary_cos, native.rotary_sin, queries);
        native_controls &= equal("native-qkv." + std::to_string(index), reproduced, probe.qkv);
        native_controls &= equal("native-pooled." + std::to_string(index), actual.pooled, probe.pooled);
        native_controls &= equal("native-probabilities-pool." + std::to_string(index),
                                 original_probabilities_pool(actual, tokens, config, queries), actual.pooled_sample);
        const auto legacy =
            trace(probe.qkv, config, tokens, native.rotary_cos, native.rotary_sin, queries, nullptr, false);
        double legacy_error = 0, pinned_error = 0;
        for (size_t i = 0; i < probe.pooled.size(); ++i) {
            legacy_error = std::max(legacy_error, std::abs(double(legacy.pooled[i]) - probe.pooled[i]));
            pinned_error = std::max(pinned_error, std::abs(double(actual.pooled[i]) - probe.pooled[i]));
        }
        std::printf("ATTN_ROUNDING block=%d legacy_max_abs=%.9g pinned_max_abs=%.9g n=%zu\n", index, legacy_error,
                    pinned_error, probe.pooled.size());
        // Retain measurements even on failed controls, but never validate their interpretation.
        ok &= measure(ref, "composed", p + "norm1", probe.norm1);
        ok &= measure(ref, "composed", p + "qkv", probe.qkv);
        ok &= record_trace("composed", p, actual, true, &probe.pooled);
        ok &= measure(ref, "composed", "prefill.vision.layer." + std::to_string(index), native.hidden[index]);
        const auto original_qkv = array(ref, p + "qkv"), norm = array(ref, p + "norm1");
        trace_result original;
        original.query = array(ref, p + "trace.query");
        original.key = array(ref, p + "trace.key");
        original.value = array(ref, p + "trace.value");
        original.probabilities = array(ref, p + "trace.probabilities");
        if (!finite(original_qkv, size_t(tokens) * 2304) || !finite(norm, size_t(tokens) * 768) ||
            !finite(original.query, norm.size()) || !finite(original.key, norm.size()) ||
            !finite(original.value, norm.size()) ||
            !finite(original.probabilities, size_t(config.heads) * queries.size() * tokens))
            return 2;
        std::vector<float> projected;
        if (!project(weights.layers[index], norm, tokens, projected)) return 2;
        for (bool native_geometry : { false, true }) {
            const auto & co = native_geometry ? native.rotary_cos : original_cosine;
            const auto & si = native_geometry ? native.rotary_sin : original_sine;
            const std::string suffix = native_geometry ? "native-geometry" : "original-geometry";
            ok &= record_trace("original-qkv-" + suffix, p, trace(original_qkv, config, tokens, co, si, queries), true);
            ok &= measure(ref, "original-norm-" + suffix, p + "qkv", projected);
            ok &= record_trace("original-norm-" + suffix, p, trace(projected, config, tokens, co, si, queries), true);
        }
        ok &= record_trace("original-qk-v", p, trace({}, config, tokens, {}, {}, queries, &original), false);
        ok &= measure(ref, "original-probabilities-v", p + "trace.pooled_sample",
                      original_probabilities_pool(original, tokens, config, queries));
    }
    ok &= measure(ref, "composed", "prefill.vision.merger", native.merged);
    ok &= measure(ref, "composed", "prefill.vision.rotary_cos", native.rotary_cos);
    ok &= measure(ref, "composed", "prefill.vision.rotary_sin", native.rotary_sin);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("ATTN_AUDIT finite=%d metrics=%d qualified_failures=%d controls=%d exact_controls=%d\n", ok, metrics,
                failures, native_controls, controls);
    return ok && native_controls && metrics == 93 && controls == 6 && failures == 0 ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
