#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "qwen35_math.h"
#include "qwen35_vision.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

using inputs = std::map<std::string, std::vector<float>>;
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
static bool observe(const char * name, const float * x, int width, int tokens, void * user) {
    if (!name || !x || width < 1 || tokens < 1) return false;
    return static_cast<inputs *>(user)->emplace(name, std::vector<float>(x, x + size_t(width) * tokens)).second;
}
static std::vector<float> array(const crispembed_diff::Ref & ref, const std::string & name) {
    const auto x = ref.get_f32(name);
    return x.first ? std::vector<float>(x.first, x.first + x.second) : std::vector<float>();
}
static bool run(const qwen35_ocr::vision_weights & w, const qwen35_ocr::vision_config & c,
                const std::vector<float> & pixels, const std::vector<int> & grid, qwen35_ocr::vision_outputs & out,
                inputs & captured) {
    captured.clear();
    gate("CRISPEMBED_QWEN35_IMATRIX", "1");
    if (!qwen35_ocr::set_projection_observer(observe, &captured)) return false;
    const bool ok = qwen35_ocr::vision_cpu(w, c, pixels.data(), grid[1] * grid[2], grid, 4, out);
    qwen35_ocr::set_projection_observer(nullptr, nullptr);
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    return ok && captured.size() == 3 + 4 * w.layers.size();
}
static bool equal(const std::string & name, const std::vector<float> & x, const std::vector<float> & y) {
    const bool ok = !x.empty() && x == y;
    double error = 0;
    if (x.size() == y.size())
        for (size_t i = 0; i < x.size(); ++i) error = std::max(error, std::abs(double(x[i]) - y[i]));
    else
        error = INFINITY;
    std::printf("BLOCK_CONTROL %s pass=%d max_abs=%.9g n=%zu/%zu\n", name.c_str(), ok, error, x.size(), y.size());
    return ok;
}
// This is native projection replay, not expected-value reconstruction. The
// reconstructed residual/down sum must equal the ordinary native block output.
static bool project(ggml_tensor * w, ggml_tensor * b, const std::vector<float> & x, int tokens,
                    std::vector<float> & y) {
    std::vector<std::vector<float>> ys;
    if (!w || !b || x.size() != size_t(tokens) * w->ne[0] ||
        !qwen35_ocr::project_cpu({ w }, x.data(), int(w->ne[0]), tokens, 4, ys))
        return false;
    std::vector<float> bias(size_t(w->ne[1]));
    if (!qwen35_ocr::tensor_floats(b, bias.data())) return false;
    y = std::move(ys[0]);
    for (size_t j = 0; j < y.size(); ++j) y[j] += bias[j % bias.size()];
    return true;
}
static bool residual(const qwen35_ocr::vision_layer_weights & w, int i, const std::vector<float> & input,
                     const inputs & captured, int tokens, const std::vector<float> & expected,
                     std::vector<float> & middle, std::vector<float> & attention, std::vector<float> & down) {
    const auto p = "v.blocks." + std::to_string(i) + ".";
    if (!project(w.out, w.out_bias, captured.at(p + "attn.proj.weight"), tokens, attention) ||
        !project(w.down, w.down_bias, captured.at(p + "mlp.linear_fc2.weight"), tokens, down) ||
        attention.size() != input.size() || down.size() != input.size())
        return false;
    middle = input;
    auto output = input;
    for (size_t j = 0; j < input.size(); ++j) {
        middle[j] += attention[j];
        output[j] = middle[j] + down[j];
    }
    return equal("residual-sum." + std::to_string(i), output, expected);
}
static int failures = 0, metrics = 0;
static bool measure(const crispembed_diff::Ref & ref, const std::string & mode, const std::string & stage,
                    const std::vector<float> & values) {
    const auto m = ref.compare(stage, values.data(), values.size());
    if (!m.found || values.empty() || m.n_elem != values.size() || ref.get_f32(stage).second != values.size() ||
        !std::isfinite(m.cos_min) || !std::isfinite(m.max_abs) ||
        !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
        return false;
    ++metrics;
    std::printf(
        "VISION_METRIC %s %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
        mode.c_str(), stage.c_str(), m.n_elem, m.cos_min, m.cos_global, m.max_abs, m.mean_abs, m.rms, m.mine_norm,
        m.ref_norm);
    std::printf("MODE %s\n", mode.c_str());
    if (!qwen35_compare(ref, stage, values)) ++failures;
    return true;
}
// Identity patch weights and zero positional additions are validation fixtures.
// They deliver a captured input to the unchanged, linked production block loop.
static bool replay(const qwen35_ocr::vision_weights & w, const qwen35_ocr::vision_config & c, ggml_tensor * identity,
                   ggml_tensor * zero_bias, ggml_tensor * zero_position, int first, int count,
                   const std::vector<float> & input, const std::vector<int> & grid, qwen35_ocr::vision_outputs & out,
                   inputs & captured) {
    const int tokens = grid[1] * grid[2];
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_QWEN35_VISION_PROBES") || first < 0 ||
        count < 1 || first + count > int(w.layers.size()) || input.size() != size_t(tokens) * c.width ||
        !std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
        return false;
    auto local = w;
    local.patch = identity;
    local.patch_bias = zero_bias;
    local.position = zero_position;
    local.layers.assign(w.layers.begin() + first, w.layers.begin() + first + count);
    std::vector<float> packed(size_t(tokens) * c.patch_width, 0.0f);
    for (int t = 0; t < tokens; ++t)
        std::copy_n(input.data() + size_t(t) * c.width, c.width, packed.data() + size_t(t) * c.patch_width);
    return run(local, c, packed, grid, out, captured) && equal("identity-input", out.embedded, input);
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    qwen35_ocr::vision_weights weights;
    qwen35_ocr::vision_config config;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-blocks", loaded) ||
        loaded.tensors.size() != 153 || !qwen35_ocr::map_small_vision(loaded.tensors, weights) ||
        weights.layers.size() != 12)
        return 2;
    const auto source_grid = array(ref, "input.image_grid_thw");
    if (source_grid.size() != 3 || source_grid[0] != 1) return 2;
    std::vector<int> grid;
    for (float v : source_grid) {
        if (!std::isfinite(v) || v < 1 || v > 1024 || std::floor(v) != v) return 2;
        grid.push_back(int(v));
    }
    const int tokens = grid[1] * grid[2];
    const auto pixels = array(ref, "input.pixel_values");
    if (tokens > 1024 || pixels.size() != size_t(tokens) * config.patch_width) return 2;
    auto * fixtures = ggml_init({ 1024 * 1024, nullptr, true });
    if (!fixtures) return 2;
    auto * identity = ggml_new_tensor_2d(fixtures, GGML_TYPE_F32, config.patch_width, config.width);
    auto * zero_bias = ggml_new_tensor_1d(fixtures, GGML_TYPE_F32, config.width);
    auto * zero_position =
        ggml_new_tensor_2d(fixtures, GGML_TYPE_F32, config.width, config.position_side * config.position_side);
    auto * fixture_buffer = ggml_backend_alloc_ctx_tensors(fixtures, backend);
    if (!fixture_buffer || !ggml_backend_buffer_is_host(fixture_buffer)) return 2;
    for (auto * t : { identity, zero_bias, zero_position })
        std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), 0.0f);
    ggml_set_name(identity, "audit.identity.patch");
    for (int i = 0; i < config.width; ++i)
        static_cast<float *>(identity->data)[size_t(i) * config.patch_width + i] = 1.0f;
    qwen35_ocr::vision_outputs native, ordinary;
    inputs captured, ignored;
    if (!run(weights, config, pixels, grid, native, captured) || native.hidden.size() != 12 || captured.size() != 51)
        return 2;
    bool ok = qwen35_ocr::vision_cpu(weights, config, pixels.data(), tokens, grid, 4, ordinary);
    ok &= equal("observer-off-merger", native.merged, ordinary.merged) && native.hidden == ordinary.hidden;
    auto boundaries = [&](const std::string & mode, int i, const std::vector<float> & input, const inputs & observed,
                          const std::vector<float> & output) {
        const auto p = "prefill.vision.probe." + std::to_string(i) + ".", w = "v.blocks." + std::to_string(i) + ".";
        std::vector<float> middle, attention, down;
        if (!residual(weights.layers[i], i, input, observed, tokens, output, middle, attention, down)) return false;
        bool valid = measure(ref, mode, p + "input", input) && measure(ref, mode, p + "norm2_input", middle) &&
                     measure(ref, mode, p + "attention", attention) && measure(ref, mode, p + "down", down) &&
                     measure(ref, mode, "prefill.vision.layer." + std::to_string(i), output);
        for (const auto & pair : { std::pair<const char *, const char *>{ "norm1", "attn.qkv.weight" },
                                   { "pooled", "attn.proj.weight" },
                                   { "norm2", "mlp.linear_fc1.weight" },
                                   { "activation", "mlp.linear_fc2.weight" } })
            valid &= measure(ref, mode, p + pair.first, observed.at(w + pair.second));
        return valid;
    };
    for (int i = 0; i < 12; ++i)
        ok &= boundaries("composed", i, i ? native.hidden[i - 1] : native.embedded, captured, native.hidden[i]);
    ok &= measure(ref, "composed", "prefill.vision.merger", native.merged);
    ok &= measure(ref, "composed", "prefill.vision.rotary_cos", native.rotary_cos);
    ok &= measure(ref, "composed", "prefill.vision.rotary_sin", native.rotary_sin);
    for (int i = 0; i < 12; ++i) {
        const auto & input = i ? native.hidden[i - 1] : native.embedded;
        qwen35_ocr::vision_outputs control;
        if (!replay(weights, config, identity, zero_bias, zero_position, i, 1, input, grid, control, ignored)) return 1;
        ok &= equal("native-block." + std::to_string(i), control.hidden[0], native.hidden[i]);
        for (const auto & pair : captured)
            if (pair.first.find("v.blocks." + std::to_string(i) + ".") == 0)
                ok &= equal(pair.first, ignored.at(pair.first), pair.second);
        ok &= control.rotary_cos == native.rotary_cos && control.rotary_sin == native.rotary_sin;
    }
    std::printf("BLOCK_CONTROL_COMPLETE pass=%d blocks=12\n", ok);
    if (!ok) return 1;
    qwen35_ocr::vision_outputs last;
    for (int i = 0; i < 12; ++i) {
        const auto input = array(ref, "prefill.vision.probe." + std::to_string(i) + ".input");
        qwen35_ocr::vision_outputs block;
        if (!replay(weights, config, identity, zero_bias, zero_position, i, 1, input, grid, block, ignored)) return 1;
        ok &= boundaries("block-input", i, input, ignored, block.hidden[0]);
        if (i == 11) last = std::move(block);
    }
    for (int first : { 0, 4, 8, 10, 11 }) {
        const auto input = array(ref, "prefill.vision.probe." + std::to_string(first) + ".input");
        qwen35_ocr::vision_outputs suffix;
        if (first == 11)
            suffix = std::move(last);
        else if (!replay(weights, config, identity, zero_bias, zero_position, first, 12 - first, input, grid, suffix,
                         ignored))
            return 1;
        const auto mode = "suffix-input-" + std::to_string(first);
        for (int j = first; j < 12; ++j)
            ok &= measure(ref, mode, "prefill.vision.layer." + std::to_string(j), suffix.hidden[j - first]);
        ok &= measure(ref, mode, "prefill.vision.merger.norm", suffix.merger_norm);
        ok &= measure(ref, mode, "prefill.vision.merger.up", suffix.merger_up);
        ok &= measure(ref, mode, "prefill.vision.merger.activation", suffix.merger_activation);
        ok &= measure(ref, mode, "prefill.vision.merger", suffix.merged);
    }
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "0");
    ok &= !replay(weights, config, identity, zero_bias, zero_position, 0, 1, native.embedded, grid, ordinary, ignored);
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    gate("CRISPEMBED_QWEN35_OCR", "0");
    ok &= !replay(weights, config, identity, zero_bias, zero_position, 0, 1, native.embedded, grid, ordinary, ignored);
    gate("CRISPEMBED_QWEN35_OCR", "1");
    ok &= !replay(weights, config, identity, zero_bias, zero_position, 0, 1, {}, grid, ordinary, ignored);
    ggml_backend_buffer_free(fixture_buffer);
    ggml_free(fixtures);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("VISION_BLOCKS finite=%d metrics=%d qualified_failures=%d controls=%d observer=51\n", ok, metrics,
                failures, ok);
    return ok && metrics == 266 && failures == 0 ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
