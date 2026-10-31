// Test-only composed native CPU vision versus the frozen qianziwen prefill.
//
// usage: test-xiaomi-vision-prefill <model-f32.gguf> <qianziwen-prefill.gguf>
// exit:  0 every stage within the strict gate, 1 a numerical mismatch,
//        2 setup, shape or control failure
//
// Scope: qwen35_ocr::vision_cpu end to end -- patch embed, interpolated
// position, embedded sum, rotary cos/sin, all twelve blocks and the merger
// (norm, fc1 "up", activation, fc2). The image frontend, decoder, decoded text
// and any quantized artifact are NOT exercised. Verdicts use a literal
// max_abs 0.002 / cos_min 0.99999 and never Report::is_pass(), which would
// honour CRISPEMBED_DIFF_COS_THRESHOLD from the environment.
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_image_bounds.h"
#include "qwen35_vision.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static constexpr float kMaxAbs = 0.002f, kCosMin = 0.99999f;
static constexpr int kTokens = 1984, kWidth = 768, kPatchWidth = 1536, kRotary = 64;
static constexpr const char * kPrefix = "prefill.vision.";

static bool include_vision(const char * name, void *) {
    return std::strncmp(name, "v.", 2) == 0;
}

static bool finite(const float * values, size_t count) {
    return values && count && std::all_of(values, values + count, [](float x) { return std::isfinite(x); });
}

// Literal gate. Never Report::is_pass(): that path consults an env override.
static bool strict_pass(const crispembed_diff::Report & report, size_t count) {
    return report.found && report.n_elem == count && std::isfinite(report.max_abs) && std::isfinite(report.mean_abs) &&
           std::isfinite(report.rms) && std::isfinite(report.cos_min) && std::isfinite(report.cos_mean) &&
           std::isfinite(report.cos_global) && std::isfinite(report.mine_norm) && std::isfinite(report.ref_norm) &&
           report.max_abs <= kMaxAbs && report.cos_min >= kCosMin;
}

static uint64_t fnv1a(const float * values, size_t count) {
    uint64_t hash = 14695981039346656037ull;
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(values);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

struct loaded {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    ~loaded() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

struct stage {
    std::string name;
    const std::vector<float> * values;
    int64_t width, rows;
};

static int run(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model-f32.gguf> <qianziwen-prefill.gguf>\n", argv[0]);
        return 2;
    }
#ifdef _WIN32
    if (_putenv_s("CRISPEMBED_QWEN35_OCR", "1") != 0) return 2;
#else
    if (setenv("CRISPEMBED_QWEN35_OCR", "1", 1) != 0) return 2;
#endif
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    const auto pixels = ref.get_f32("input.pixel_values");
    const auto grid_values = ref.get_f32("input.image_grid_thw");
    if (ref.shape("input.pixel_values") != std::vector<int64_t>{ kPatchWidth, kTokens } ||
        ref.shape("input.image_grid_thw") != std::vector<int64_t>{ 3, 1 } ||
        pixels.second != size_t(kPatchWidth) * kTokens || grid_values.second != 3 ||
        !finite(pixels.first, pixels.second) || !finite(grid_values.first, grid_values.second)) {
        std::fprintf(stderr, "frozen prefill lacks the pinned 1984x1536 pixel / grid pair\n");
        return 2;
    }
    std::vector<int> grid;
    for (size_t i = 0; i < grid_values.second; ++i) {
        if (grid_values.first[i] < 1 || grid_values.first[i] > 1024 ||
            std::floor(grid_values.first[i]) != grid_values.first[i])
            return 2;
        grid.push_back(int(grid_values.first[i]));
    }
    if (int64_t(grid[0]) * grid[1] * grid[2] != kTokens) {
        std::fprintf(stderr, "grid does not match the frozen token count\n");
        return 2;
    }

    loaded model;
    model.backend = ggml_backend_cpu_init();
    if (!model.backend || !core_gguf::load_weights_filtered(argv[1], model.backend, include_vision, nullptr,
                                                            "xiaomi-vision-prefill", model.weights))
        return 2;
    for (const auto & entry : model.weights.tensors)
        if (entry.second->type != GGML_TYPE_F32) {
            std::fprintf(stderr, "non-F32 weight %s; this stage requires the F32 export\n", entry.first.c_str());
            return 2;
        }
    qwen35_ocr::vision_weights weights;
    if (model.weights.tensors.size() != 153 || !qwen35_ocr::map_small_vision(model.weights.tensors, weights) ||
        weights.layers.size() != 12) {
        std::fprintf(stderr, "expected the 153-tensor twelve-block vision tower\n");
        return 2;
    }
    qwen35_ocr::vision_config config;
    config.max_patches = qwen35_ocr::kXiaomiPatchLimit;
    qwen35_ocr::vision_outputs output;
    std::printf("AUDIT_BACKEND CPU (qwen35_ocr::vision_cpu is CPU-only) threads=4 grid=%d,%d,%d tokens=%d "
                "input_fnv1a=%016llx\n",
                grid[0], grid[1], grid[2], kTokens,
                static_cast<unsigned long long>(fnv1a(pixels.first, pixels.second)));
    std::fflush(stdout);
    if (!qwen35_ocr::vision_cpu(weights, config, pixels.first, kTokens, grid, 4, output) ||
        output.hidden.size() != 12) {
        std::fprintf(stderr, "native vision_cpu failed\n");
        return 2;
    }

    std::vector<stage> stages = {
        { kPrefix + std::string("patch_embed"), &output.patch, kWidth, kTokens },
        { kPrefix + std::string("position"), &output.position, kWidth, kTokens },
        { kPrefix + std::string("embedded"), &output.embedded, kWidth, kTokens },
        { kPrefix + std::string("rotary_cos"), &output.rotary_cos, kRotary, kTokens },
        { kPrefix + std::string("rotary_sin"), &output.rotary_sin, kRotary, kTokens },
    };
    for (size_t i = 0; i < output.hidden.size(); ++i)
        stages.push_back({ kPrefix + std::string("layer.") + std::to_string(i), &output.hidden[i], kWidth, kTokens });
    stages.push_back({ kPrefix + std::string("merger.norm"), &output.merger_norm, kWidth, kTokens });
    stages.push_back({ kPrefix + std::string("merger.up"), &output.merger_up, 3072, 496 });
    stages.push_back({ kPrefix + std::string("merger.activation"), &output.merger_activation, 3072, 496 });
    stages.push_back({ kPrefix + std::string("merger"), &output.merged, 1024, 496 });

    // Negative control perturbs the ORACLE, so it stays meaningful even when the
    // native stage already fails. It runs first so a CRISPEMBED_DIFF_DUMP_DIR
    // ends up holding the real native trace, not the corruption.
    const auto oracle = ref.get_f32(kPrefix + std::string("patch_embed"));
    if (!finite(oracle.first, oracle.second)) return 2;
    std::vector<float> corrupted(oracle.first, oracle.first + oracle.second);
    corrupted[0] += 1.0f;
    std::string dump;
    if (const char * current = std::getenv("CRISPEMBED_DIFF_DUMP_DIR"); current && *current) dump = current;
    if (!dump.empty()) setenv("CRISPEMBED_DIFF_DUMP_DIR", "", 1);
    const auto negative = ref.compare(kPrefix + std::string("patch_embed"), corrupted.data(), corrupted.size());
    if (!dump.empty()) setenv("CRISPEMBED_DIFF_DUMP_DIR", dump.c_str(), 1);
    if (!negative.found || negative.n_elem != corrupted.size() || !std::isfinite(negative.max_abs) ||
        negative.max_abs <= kMaxAbs || strict_pass(negative, corrupted.size())) {
        std::fprintf(stderr, "perturbed negative control was not rejected; harness cannot discriminate\n");
        return 2;
    }
    std::printf("NEGATIVE_CONTROL rejected=1 name=%spatch_embed max_abs=%.9g\n", kPrefix, negative.max_abs);
    std::fflush(stdout);

    bool all = true;
    for (const auto & current : stages) {
        const auto & values = *current.values;
        const auto expected = ref.get_f32(current.name);
        if (ref.shape(current.name) != std::vector<int64_t>{ current.width, current.rows } ||
            expected.second != size_t(current.width) * current.rows || values.size() != expected.second ||
            !finite(values.data(), values.size()) || !finite(expected.first, expected.second)) {
            std::fprintf(stderr, "invalid shape, size or non-finite data at %s\n", current.name.c_str());
            return 2;
        }
        const auto report = ref.compare(current.name, values.data(), values.size());
        const bool pass = strict_pass(report, values.size());
        if (!std::isfinite(report.max_abs) || !std::isfinite(report.mean_abs) || !std::isfinite(report.rms) ||
            !std::isfinite(report.cos_min) || !std::isfinite(report.cos_mean) || !std::isfinite(report.cos_global) ||
            !std::isfinite(report.mine_norm) || !std::isfinite(report.ref_norm))
            return 2;
        all &= pass;
        std::printf("STAGE {\"name\":\"%s\",\"n\":%zu,\"shape\":[%lld,%lld],\"pass\":%s,\"max_abs\":%.9g,"
                    "\"mean_abs\":%.9g,\"rms\":%.9g,\"cos_min\":%.9g,\"cos_mean\":%.9g,\"cos_global\":%.9g,"
                    "\"native_norm\":%.9g,\"reference_norm\":%.9g,\"input_fnv1a\":\"%016llx\"}\n",
                    current.name.c_str(), report.n_elem, static_cast<long long>(current.width),
                    static_cast<long long>(current.rows), pass ? "true" : "false", report.max_abs, report.mean_abs,
                    report.rms, report.cos_min, report.cos_mean, report.cos_global, report.mine_norm, report.ref_norm,
                    static_cast<unsigned long long>(fnv1a(values.data(), values.size())));
        std::fflush(stdout);
    }
    return all ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
