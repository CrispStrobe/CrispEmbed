#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "qwen35_math.h"
#include "qwen35_vision.h"
#include "qwen35_moments_audit.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
struct stop_after_norm {
    std::string target;
    std::vector<float> output;
    int calls = 0;
};
static bool observe(const char * name, const float * x, int width, int tokens, void * user) {
    auto & p = *static_cast<stop_after_norm *>(user);
    ++p.calls;
    if (p.target != name) return p.calls == 1;
    p.output.assign(x, x + size_t(width) * tokens);
    return false; // Deliberate abort before QKV projection; LayerNorm is complete.
}
static bool stopped(const qwen35_ocr::vision_weights & w, const qwen35_ocr::vision_config & c,
                    const std::vector<float> & packed, const std::vector<int> & grid, qwen35_ocr::vision_outputs & out,
                    std::vector<float> & normalized) {
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_QWEN35_VISION_PROBES") ||
        w.layers.empty() || grid.size() != 3 || packed.size() != size_t(grid[1]) * grid[2] * c.patch_width)
        return false;
    stop_after_norm p{ ggml_get_name(w.layers[0].qkv), {}, 0 };
    gate("CRISPEMBED_QWEN35_IMATRIX", "1");
    if (!qwen35_ocr::set_projection_observer(observe, &p)) return false;
    const bool completed = qwen35_ocr::vision_cpu(w, c, packed.data(), grid[1] * grid[2], grid, 4, out);
    qwen35_ocr::set_projection_observer(nullptr, nullptr);
    gate("CRISPEMBED_QWEN35_IMATRIX", "0");
    normalized = std::move(p.output);
    return !completed && p.calls == 2 && normalized.size() == size_t(grid[1]) * grid[2] * c.width &&
           std::all_of(normalized.begin(), normalized.end(), [](float x) { return std::isfinite(x); });
}
static int failures = 0, metrics = 0;
static bool measure(const crispembed_diff::Ref & ref, const std::string & mode, const std::string & stage,
                    const std::vector<float> & values) {
    const auto m = ref.compare(stage, values.data(), values.size());
    if (!m.found || values.empty() || ref.get_f32(stage).second != values.size() || m.n_elem != values.size() ||
        !std::isfinite(m.cos_min) || !std::isfinite(m.max_abs) ||
        !std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
        return false;
    ++metrics;
    std::printf(
        "NORM_METRIC %s %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
        mode.c_str(), stage.c_str(), m.n_elem, m.cos_min, m.cos_global, m.max_abs, m.mean_abs, m.rms, m.mine_norm,
        m.ref_norm);
    std::printf("MODE %s\n", mode.c_str());
    if (!qwen35_compare(ref, stage, values)) ++failures;
    return true;
}
static bool write(const std::string & path, const std::vector<float> & values) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return bool(f);
}
static int crispembed_test_main(int argc, char ** argv) {
    const bool geometry = argc == 6 && std::string(argv[1]) == "--geometry";
    const bool cascade = argc == 4 && std::string(argv[3]) == "--cascade";
    if (!geometry && argc != 3 && !cascade) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    gate("CRISPEMBED_QWEN35_VISION_CASCADE_NORM", "0");
    gate("CRISPEMBED_QWEN35_VISION_PRECISE_NORM", "0");
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    qwen35_ocr::vision_config config;
    qwen35_ocr::vision_weights weights;
    if (!backend ||
        !core_gguf::load_weights_filtered(argv[geometry ? 2 : 1], backend, include, nullptr, "qwen35-norm", loaded) ||
        loaded.tensors.size() != 153 || !qwen35_ocr::map_small_vision(loaded.tensors, weights))
        return 2;
    crispembed_diff::Ref ref;
    std::vector<int> grid{ 1, 0, 0 };
    if (geometry) {
        grid[1] = std::stoi(argv[3]);
        grid[2] = std::stoi(argv[4]);
    } else {
        if (!ref.load(argv[2])) return 2;
        const auto g = array(ref, "input.image_grid_thw");
        if (g.size() != 3 || g[0] != 1) return 2;
        for (int i = 1; i < 3; ++i) {
            if (!std::isfinite(g[i]) || g[i] < 1 || g[i] > 1024 || std::floor(g[i]) != g[i]) return 2;
            grid[i] = int(g[i]);
        }
    }
    if (grid[1] < 1 || grid[1] > 1024 || grid[2] < 1 || grid[2] > 1024 || grid[1] * grid[2] > 1024 || grid[1] % 2 ||
        grid[2] % 2)
        return 2;
    const int tokens = grid[1] * grid[2];
    if (geometry) {
        std::vector<float> packed(size_t(tokens) * config.patch_width, 0.0f), normalized;
        qwen35_ocr::vision_outputs a, b;
        bool ok = stopped(weights, config, packed, grid, a, normalized);
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = float(int(i % 7) - 3) * 0.01f;
        ok &= stopped(weights, config, packed, grid, b, normalized);
        ok &= a.rotary_cos == b.rotary_cos && a.rotary_sin == b.rotary_sin &&
              a.rotary_cos.size() == size_t(tokens) * (config.width / config.heads) &&
              a.rotary_sin.size() == a.rotary_cos.size();
        ok &= write(std::string(argv[5]) + "/native-rotary-cos.bin", a.rotary_cos);
        ok &= write(std::string(argv[5]) + "/native-rotary-sin.bin", a.rotary_sin);
        std::printf("GEOMETRY finite=%d controls=%d elements=%zu\n", ok, ok, a.rotary_cos.size());
        core_gguf::free_weights(loaded);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    auto * fixtures = ggml_init({ 1024 * 1024, nullptr, true });
    if (!fixtures) return 2;
    auto * identity = ggml_new_tensor_2d(fixtures, GGML_TYPE_F32, config.patch_width, config.width);
    auto * zero_bias = ggml_new_tensor_1d(fixtures, GGML_TYPE_F32, config.width);
    auto * zero_position =
        ggml_new_tensor_2d(fixtures, GGML_TYPE_F32, config.width, config.position_side * config.position_side);
    auto * buffer = ggml_backend_alloc_ctx_tensors(fixtures, backend);
    if (!buffer || !ggml_backend_buffer_is_host(buffer)) return 2;
    for (auto * t : { identity, zero_bias, zero_position })
        std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), 0.0f);
    ggml_set_name(identity, "audit.identity.patch");
    for (int i = 0; i < config.width; ++i)
        static_cast<float *>(identity->data)[size_t(i) * config.patch_width + i] = 1.0f;
    bool finite = true, controls = true;
    int exact_controls = 0;
    const char * modes[] = { "two-pass-f32",
                             "two-pass-f64",
                             "welford-4",
                             "welford-8",
                             "welford-16",
                             "original-moments-fused",
                             "original-moments-split",
                             "welford-4-split",
                             "welford-8-split",
                             "welford-16-split",
                             "two-pass-f32-fused-variance" };
    for (int node = 0; node < 25; ++node) {
        const int layer = node == 24 ? 0 : node / 2;
        const bool second = node < 24 && node % 2;
        const auto prefix = node == 24
                                ? std::string("prefill.vision.merger.norm")
                                : "prefill.vision.probe." + std::to_string(layer) + (second ? ".norm2" : ".norm1");
        const auto input_name =
            node == 24 ? std::string("prefill.vision.merger.input")
                       : "prefill.vision.probe." + std::to_string(layer) + (second ? ".norm2_input" : ".input");
        auto local = weights;
        local.layers = { weights.layers[layer] };
        local.layers[0].norm1 = node == 24 ? weights.merger_norm
                                : second   ? weights.layers[layer].norm2
                                           : weights.layers[layer].norm1;
        local.layers[0].norm1_bias = node == 24 ? weights.merger_norm_bias
                                     : second   ? weights.layers[layer].norm2_bias
                                                : weights.layers[layer].norm1_bias;
        local.patch = identity;
        local.patch_bias = zero_bias;
        local.position = zero_position;
        const auto input = array(ref, input_name);
        if (input.size() != size_t(tokens) * config.width) return 2;
        std::vector<float> gamma(config.width), beta(config.width), packed(size_t(tokens) * config.patch_width, 0.0f);
        if (!qwen35_ocr::tensor_floats(local.layers[0].norm1, gamma.data()) ||
            !qwen35_ocr::tensor_floats(local.layers[0].norm1_bias, beta.data()))
            return 2;
        for (int t = 0; t < tokens; ++t)
            std::copy_n(input.data() + size_t(t) * config.width, config.width,
                        packed.data() + size_t(t) * config.patch_width);
        std::vector<float> native_f32;
        double split_error = INFINITY;
        for (int mode = 0; mode < 11; ++mode) {
            std::vector<float> output, means, rstds;
            const int statistic_mode = mode == 10 ? 0 : mode < 5 ? mode : mode < 7 ? 0 : mode - 5;
            norm_audit::normalize(input, gamma, beta, statistic_mode, output, means, rstds, mode == 10);
            if (mode == 5 || mode == 6) {
                means = array(ref, prefix + ".mean");
                rstds = array(ref, prefix + ".rstd");
                if (means.size() != size_t(tokens) || rstds.size() != size_t(tokens)) return 2;
            }
            if (mode >= 5 && mode < 10) norm_audit::affine(input, gamma, beta, means, rstds, mode == 5, output);
            if (mode < 2 || (cascade && mode == 2)) {
                gate("CRISPEMBED_QWEN35_VISION_PRECISE_NORM", mode == 1 ? "1" : "0");
                gate("CRISPEMBED_QWEN35_VISION_CASCADE_NORM", mode == 2 ? "1" : "0");
                qwen35_ocr::vision_outputs probe;
                std::vector<float> actual;
                const bool delivered = stopped(local, config, packed, grid, probe, actual) && probe.embedded == input;
                const bool exact = delivered && actual == output;
                double error = 0;
                if (actual.size() != output.size())
                    error = INFINITY;
                else
                    for (size_t j = 0; j < actual.size(); ++j)
                        error = std::max(error, std::abs(double(actual[j]) - output[j]));
                std::printf("NORM_INPUT %s %s pass=%d\n", modes[mode], prefix.c_str(), delivered);
                std::printf("NORM_CONTROL %s %s pass=%d max_abs=%.9g n=%zu/%zu\n", modes[mode], prefix.c_str(), exact,
                            error, actual.size(), output.size());
                if (mode == 0) {
                    native_f32 = actual;
                    split_error = error;
                }
                gate("CRISPEMBED_QWEN35_VISION_CASCADE_NORM", "0");
                controls &= exact;
                exact_controls += int(exact);
            }
            if (mode == 10) {
                double fused_error = 0;
                if (native_f32.size() != output.size()) return 2;
                for (size_t j = 0; j < output.size(); ++j)
                    fused_error = std::max(fused_error, std::abs(double(native_f32[j]) - output[j]));
                std::printf("VARIANCE_CONTRAST %s split_max_abs=%.9g fused_max_abs=%.9g n=%zu\n", prefix.c_str(),
                            split_error, fused_error, output.size());
            }
            finite &= measure(ref, modes[mode], prefix, output);
            finite &= measure(ref, modes[mode], prefix + ".mean", means);
            finite &= measure(ref, modes[mode], prefix + ".rstd", rstds);
        }
    }
    for (int i = 0; i < 12; ++i)
        finite &= measure(ref, "python-native-geometry", "prefill.vision.layer." + std::to_string(i),
                          array(ref, "native_geometry.vision.layer." + std::to_string(i)));
    qwen35_ocr::vision_outputs rejected;
    std::vector<float> normalized, packed(size_t(tokens) * config.patch_width, 0.0f);
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "0");
    controls &= !stopped(weights, config, packed, grid, rejected, normalized);
    gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    gate("CRISPEMBED_QWEN35_OCR", "0");
    controls &= !stopped(weights, config, packed, grid, rejected, normalized);
    gate("CRISPEMBED_QWEN35_OCR", "1");
    controls &= !stopped(weights, config, {}, grid, rejected, normalized);
    ggml_backend_buffer_free(buffer);
    ggml_free(fixtures);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("NORM_AUDIT finite=%d metrics=%d qualified_failures=%d controls=%d exact_controls=%d\n", finite,
                metrics, failures, controls, exact_controls);
    return finite && metrics == 837 && controls && failures == 0 ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
