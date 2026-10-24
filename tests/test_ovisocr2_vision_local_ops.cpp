// Test-only replay of two actual block-11 operators on frozen original inputs.
// The original reference and A/B raw vectors are immutable external inputs.
#include "core/clean_exit.h"
#include "qwen35_vision_operators_impl.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
constexpr int kLayer = 11;
constexpr int kWidth = 768;
const char * kPrior = "prefill.vision.layer.10";
const char * kAttention = "prefill.vision.probe.11.attention";
const char * kNorm = "prefill.vision.probe.11.norm2";
const char * kUp = "prefill.vision.probe.11.up";

bool include_vision(const char * name, void *) { return name && std::string(name).rfind("v.", 0) == 0; }

void set_gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

const char * env_value(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "unset";
}

std::vector<float> reference(const crispembed_diff::Ref & ref, const char * name, size_t expected) {
    const auto value = ref.get_f32(name);
    if (!value.first || value.second != expected) return {};
    const std::vector<float> out(value.first, value.first + value.second);
    return std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }) ? out : std::vector<float>();
}

std::vector<float> raw(const std::string & directory, const char * name, size_t expected) {
    const auto path = directory + "/" + name + ".f32";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streampos(std::streamoff(expected * sizeof(float)))) return {};
    std::vector<float> out(expected);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(out.data()), std::streamsize(expected * sizeof(float)));
    if (!file || !std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); })) return {};
    return out;
}

bool exact(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    const bool same = !a.empty() && a.size() == b.size() &&
                      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    std::printf("LOCAL_OP_CONTROL %s exact=%d n=%zu/%zu\n", name, int(same), a.size(), b.size());
    return same;
}

std::vector<float> residual_input(const std::vector<float> & prior, const std::vector<float> & attention) {
    if (prior.empty() || prior.size() != attention.size()) return {};
    std::vector<float> sum = prior;
    // Original Qwen3_5VisionBlock.forward does hidden_states + self.attn(...)
    // immediately before self.norm2(hidden_states).
    for (size_t i = 0; i < sum.size(); ++i) sum[i] += attention[i];
    if (!std::all_of(sum.begin(), sum.end(), [](float x) { return std::isfinite(x); })) return {};
    return sum;
}

void input_metric(const std::vector<float> & original, const std::vector<float> & native) {
    size_t worst = 0;
    double max_abs = 0, total_abs = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        const double error = std::abs(double(native[i]) - original[i]);
        total_abs += error;
        if (error > max_abs) { max_abs = error; worst = i; }
    }
    std::printf("LOCAL_OP_INPUT norm2 n=%zu max_abs=%.9g mean_abs=%.9g worst_index=%zu row=%zu "
                "native=%.9g original=%.9g\n", original.size(), max_abs, total_abs / original.size(),
                worst, worst / kWidth, native[worst], original[worst]);
}

bool measure(const crispembed_diff::Ref & ref, const char * mode, const char * stage,
             const std::vector<float> & values) {
    const auto oracle = ref.get_f32(stage);
    if (!oracle.first || values.empty() || oracle.second != values.size()) return false;
    size_t worst = 0;
    double max_abs = 0, total_abs = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(oracle.first[i])) return false;
        const double error = std::abs(double(values[i]) - oracle.first[i]);
        total_abs += error;
        if (error > max_abs) { max_abs = error; worst = i; }
    }
    const auto report = ref.compare(stage, values.data(), values.size());
    if (!report.found || !std::isfinite(report.cos_min) || !std::isfinite(report.max_abs)) return false;
    const bool qualified = qwen35_compare(ref, stage, values);
    std::printf("LOCAL_OP_METRIC %s %s n=%zu max_abs=%.9g mean_abs=%.9g cos_min=%.9g "
                "worst_index=%zu row=%zu candidate=%.9g original=%.9g existing_budget_pass=%d\n",
                mode, stage, values.size(), max_abs, total_abs / values.size(), report.cos_min,
                worst, worst / (std::string(stage) == kUp ? 3072 : kWidth), values[worst], oracle.first[worst],
                int(qualified));
    return true; // A numerical failure is the diagnostic result, not infrastructure failure.
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    static_assert(sizeof(float) == 4, "raw vision vectors are F32");
    if (argc != 4) {
        std::fprintf(stderr, "usage: test-ovisocr2-vision-local-ops MODEL_GGUF ORIGINAL_REF_GGUF A_B_RAW_DIR\n");
        return 2;
    }
    set_gate("CRISPEMBED_QWEN35_OCR", "1");
    set_gate("CRISPEMBED_QWEN35_VISION_PROBES", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    const auto grid_value = ref.get_f32("input.image_grid_thw");
    if (!grid_value.first || grid_value.second != 3 || grid_value.first[0] != 1 ||
        !std::isfinite(grid_value.first[1]) || !std::isfinite(grid_value.first[2]) ||
        grid_value.first[1] < 1 || grid_value.first[2] < 1 ||
        grid_value.first[1] > 1024 || grid_value.first[2] > 1024 ||
        std::floor(grid_value.first[1]) != grid_value.first[1] ||
        std::floor(grid_value.first[2]) != grid_value.first[2]) return 2;
    const std::vector<int> grid{1, int(grid_value.first[1]), int(grid_value.first[2])};
    const size_t tokens = size_t(grid[1]) * grid[2];
    if (tokens > 1024) return 2;
    const auto original_prior = reference(ref, kPrior, tokens * kWidth);
    const auto original_attention = reference(ref, kAttention, tokens * kWidth);
    const auto original_norm = reference(ref, kNorm, tokens * kWidth);
    const auto original_up = reference(ref, kUp, tokens * 3072);
    const auto native_prior = raw(argv[3], kPrior, tokens * kWidth);
    const auto native_attention = raw(argv[3], kAttention, tokens * kWidth);
    const auto native_norm = raw(argv[3], kNorm, tokens * kWidth);
    const auto native_up = raw(argv[3], kUp, tokens * 3072);
    if (original_prior.empty() || original_attention.empty() || original_norm.empty() || original_up.empty() ||
        native_prior.empty() || native_attention.empty() || native_norm.empty() || native_up.empty()) return 2;
    const auto original_input = residual_input(original_prior, original_attention);
    const auto native_input = residual_input(native_prior, native_attention);
    if (original_input.empty() || native_input.empty()) return 2;
    input_metric(original_input, native_input);
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include_vision, nullptr,
                                                      "ovisocr2-vision-local-ops", loaded)) return 2;
    qwen35_ocr::vision_weights weights;
    qwen35_ocr::vision_config config;
    if (loaded.tensors.size() != 153 || !qwen35_ocr::map_small_vision(loaded.tensors, weights)) return 2;
    const std::vector<float> rotary; // Norm and up use geometry but not rotary values.
    std::vector<float> replay_native_norm, replay_native_up, replay_original_norm, replay_original_up;
    const bool ran = qwen35_ocr::vision_test_operator_cpu(weights, config, kLayer, "norm2", native_input,
            grid, rotary, rotary, 4, replay_native_norm) &&
        qwen35_ocr::vision_test_operator_cpu(weights, config, kLayer, "up", native_norm,
            grid, rotary, rotary, 4, replay_native_up) &&
        qwen35_ocr::vision_test_operator_cpu(weights, config, kLayer, "norm2", original_input,
            grid, rotary, rotary, 4, replay_original_norm) &&
        qwen35_ocr::vision_test_operator_cpu(weights, config, kLayer, "up", original_norm,
            grid, rotary, rotary, 4, replay_original_up);
    const bool controls = ran && exact("native.norm2", replay_native_norm, native_norm) &&
                          exact("native.up", replay_native_up, native_up);
    bool complete = controls;
    if (complete) {
        complete &= measure(ref, "native_input", kNorm, replay_native_norm);
        complete &= measure(ref, "same_original_input", kNorm, replay_original_norm);
        complete &= measure(ref, "native_input", kUp, replay_native_up);
        complete &= measure(ref, "same_original_input", kUp, replay_original_up);
    }
    std::printf("LOCAL_OP_DIAGNOSTIC_COMPLETE complete=%d qualification_claim=0 "
                "precise_norm=%s cascade_norm=%s precise_attn=%s\n", int(complete),
                env_value("CRISPEMBED_QWEN35_VISION_PRECISE_NORM"),
                env_value("CRISPEMBED_QWEN35_VISION_CASCADE_NORM"),
                env_value("CRISPEMBED_QWEN35_VISION_PRECISE_ATTN"));
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    return complete ? 0 : 2;
}

int main(int argc, char ** argv) { core_util::clean_exit(crispembed_test_main(argc, argv)); }
