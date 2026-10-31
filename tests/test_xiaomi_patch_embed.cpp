// Test-only earliest-stage audit: native F32 patch projection from the pinned
// F32 export against the original BF16 activation widened to F32 in the frozen
// qianziwen prefill archive.
//
// usage: test-xiaomi-patch-embed <model-f32.gguf> <qianziwen-prefill.gguf>
// exit:  0 strict parity, 1 numerical mismatch, 2 setup/control failure
//
// Scope: this isolates only prefill.vision.patch_embed (the projection plus
// bias of qwen35_ocr::vision_cpu's linear()). It says nothing about later
// vision stages, the decoder, or decoded output. The archive input is the
// frozen processor output, so the image frontend is not exercised here either.
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_math.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

static constexpr const char * kWeight = "v.patch_embed.proj.weight";
static constexpr const char * kBias = "v.patch_embed.proj.bias";
static constexpr const char * kPixels = "input.pixel_values";
static constexpr const char * kOutput = "prefill.vision.patch_embed";
// Xiaomi-OCR-0 vision: hidden 768, patch width 3*2*16*16, patch_size 16.
static constexpr int64_t kPatchWidth = 1536, kWidth = 768;
// Observation thresholds, not tuned: the reference is BF16-rounded while the
// exported weights are F32, so this strict gate may legitimately fail. Report
// measurements either way; the return code carries the verdict.
static constexpr float kMaxAbs = 0.002f, kCosMin = 0.99999f;

static bool include_patch_tensors(const char * name, void *) {
    return std::strcmp(name, kWeight) == 0 || std::strcmp(name, kBias) == 0;
}

static bool enable_ocr_gate() {
#ifdef _WIN32
    return _putenv_s("CRISPEMBED_QWEN35_OCR", "1") == 0;
#else
    return setenv("CRISPEMBED_QWEN35_OCR", "1", 1) == 0;
#endif
}

static bool finite(const float * values, size_t count) {
    return values && count && std::all_of(values, values + count, [](float x) { return std::isfinite(x); });
}

// FNV-1a over the frozen input bytes, so a recorded measurement can be tied to
// the exact archive pixels it consumed.
static uint64_t digest(const float * values, size_t count) {
    uint64_t h = 14695981039346656037ull;
    const unsigned char * p = reinterpret_cast<const unsigned char *>(values);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Deliberately not Report::is_pass(): that honours CRISPEMBED_DIFF_COS_THRESHOLD
// from the environment, and this gate must not be relaxed by a stray variable.
static bool strict_pass(const crispembed_diff::Report & report, size_t count) {
    return report.found && report.n_elem == count && std::isfinite(report.max_abs) && std::isfinite(report.mean_abs) &&
           std::isfinite(report.rms) && std::isfinite(report.cos_min) && std::isfinite(report.cos_mean) &&
           std::isfinite(report.cos_global) && std::isfinite(report.mine_norm) && std::isfinite(report.ref_norm) &&
           report.max_abs <= kMaxAbs && report.cos_min >= kCosMin;
}

static void print_report(const char * label, const crispembed_diff::Report & report, bool pass) {
    std::printf("%s pass=%d shape_ne=[", label, int(pass));
    for (size_t i = 0; i < report.shape.size(); ++i)
        std::printf("%s%lld", i ? "," : "", static_cast<long long>(report.shape[i]));
    std::printf("] n=%zu cos_min=%.9f cos_mean=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g "
                "native_norm=%.9g reference_norm=%.9g\n",
                report.n_elem, report.cos_min, report.cos_mean, report.cos_global, report.max_abs, report.mean_abs,
                report.rms, report.mine_norm, report.ref_norm);
}

struct loaded {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    ~loaded() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model-f32.gguf> <qianziwen-prefill.gguf>\n", argv[0]);
        return 2;
    }
    if (!enable_ocr_gate()) return 2; // project_cpu is default-off
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;

    const auto pixels = ref.get_f32(kPixels);
    const auto oracle = ref.get_f32(kOutput);
    const auto pixel_shape = ref.shape(kPixels), output_shape = ref.shape(kOutput);
    if (pixel_shape.size() != 2 || pixel_shape[0] != kPatchWidth || pixel_shape[1] != 1984 ||
        output_shape.size() != 2 || output_shape[0] != kWidth || output_shape[1] != pixel_shape[1] ||
        pixels.second != size_t(pixel_shape[0] * pixel_shape[1]) ||
        oracle.second != size_t(output_shape[0] * output_shape[1]) || !finite(pixels.first, pixels.second) ||
        !finite(oracle.first, oracle.second)) {
        std::fprintf(stderr, "frozen prefill reference lacks a finite pixel/patch_embed pair\n");
        return 2;
    }
    const int tokens = int(pixel_shape[1]);
    const uint64_t input_digest = digest(pixels.first, pixels.second);

    loaded r;
    r.backend = ggml_backend_cpu_init();
    if (!r.backend || !core_gguf::load_weights_filtered(argv[1], r.backend, include_patch_tensors, nullptr,
                                                        "xiaomi-patch-embed", r.weights))
        return 2;
    auto * weight = core_gguf::try_get(r.weights.tensors, kWeight);
    auto * bias = core_gguf::try_get(r.weights.tensors, kBias);
    if (r.weights.tensors.size() != 2 || !weight || !bias || weight->type != GGML_TYPE_F32 ||
        bias->type != GGML_TYPE_F32 || weight->ne[0] != kPatchWidth || weight->ne[1] != kWidth || weight->ne[2] != 1 ||
        weight->ne[3] != 1 || bias->ne[0] != kWidth || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        std::fprintf(stderr, "expected exactly the F32 patch weight [%d,%d] and bias [%d] with the pinned shapes\n",
                     int(kPatchWidth), int(kWidth), int(kWidth));
        return 2;
    }

    // Same operation as vision_cpu's linear(): F32 CPU projection, then bias.
    std::vector<std::vector<float>> projections;
    if (!qwen35_ocr::project_cpu({ weight }, pixels.first, int(kPatchWidth), tokens, 4, projections) ||
        projections.size() != 1 || projections[0].size() != oracle.second) {
        std::fprintf(stderr, "native patch projection failed\n");
        return 2;
    }
    std::vector<float> biases(kWidth);
    if (!qwen35_ocr::tensor_floats(bias, biases.data()) || !finite(biases.data(), biases.size())) return 2;
    auto actual = std::move(projections[0]);
    for (size_t i = 0; i < actual.size(); ++i) actual[i] += biases[i % kWidth];
    if (!finite(actual.data(), actual.size())) return 2;

    // Negative control starts from the ORACLE, not native output, so it stays
    // meaningful when native parity already fails. It is compared first so an
    // optional CRISPEMBED_DIFF_DUMP_DIR ends holding the real output.
    std::vector<float> perturbed(oracle.first, oracle.first + oracle.second);
    perturbed[0] += 1.0f;
    if (!finite(perturbed.data(), perturbed.size())) return 2;
    const auto negative = ref.compare(kOutput, perturbed.data(), perturbed.size());
    const bool negative_rejected = negative.found && negative.n_elem == perturbed.size() &&
                                   std::isfinite(negative.max_abs) && negative.max_abs > kMaxAbs &&
                                   !strict_pass(negative, perturbed.size());
    print_report("PATCH_NEGATIVE_CONTROL", negative, negative_rejected);

    const auto report = ref.compare(kOutput, actual.data(), actual.size());
    const bool pass = strict_pass(report, actual.size());
    std::printf("PATCH_INPUT shape_ne=[%d,%d] weight=F32_export reference=BF16_widened_to_F32 "
                "stage=prefill.vision.patch_embed only input_fnv1a=%016llx\n",
                int(kPatchWidth), tokens, static_cast<unsigned long long>(input_digest));
    print_report("PATCH_METRIC", report, pass);
    if (!negative_rejected) {
        std::fprintf(stderr, "perturbed negative control was not rejected; harness is not discriminating\n");
        return 2;
    }
    return pass ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
