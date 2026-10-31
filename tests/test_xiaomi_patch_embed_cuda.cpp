// TEST ONLY: replay the frozen qianziwen patch embedding on ggml CUDA.
// usage: test-xiaomi-patch-embed-cuda <model.gguf> <reference.gguf> <output-f32-path>
// exit: 0 strict BF16-reference parity, 1 numerical mismatch, 2 setup/control failure
// The raw output is token-major F32 [1984,768], without a header, and is
// written before the parity verdict, including on a numerical mismatch.
// This is not a production CUDA engine or an end-to-end OCR test.
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

static constexpr const char * kWeight = "v.patch_embed.proj.weight";
static constexpr const char * kBias = "v.patch_embed.proj.bias";
static constexpr const char * kPixels = "input.pixel_values";
static constexpr const char * kOutput = "prefill.vision.patch_embed";
static constexpr int64_t kPatchWidth = 1536, kWidth = 768, kTokens = 1984;
// Identical to the CPU probe; these are deliberately strict against BF16.
static constexpr float kMaxAbs = 0.002f, kCosMin = 0.99999f;

static bool include_patch_tensors(const char * name, void *) {
    return std::strcmp(name, kWeight) == 0 || std::strcmp(name, kBias) == 0;
}

static bool finite_values(const float * values, size_t count) {
    return values && count && std::all_of(values, values + count, [](float x) { return std::isfinite(x); });
}

static uint64_t digest(const float * values, size_t count) {
    uint64_t h = 14695981039346656037ull;
    const auto * p = reinterpret_cast<const unsigned char *>(values);
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool strict_pass(const crispembed_diff::Report & r, size_t count) {
    return r.found && r.n_elem == count && std::isfinite(r.max_abs) && std::isfinite(r.mean_abs) &&
           std::isfinite(r.rms) && std::isfinite(r.cos_min) && std::isfinite(r.cos_mean) &&
           std::isfinite(r.cos_global) && std::isfinite(r.mine_norm) && std::isfinite(r.ref_norm) &&
           r.max_abs <= kMaxAbs && r.cos_min >= kCosMin;
}

static void print_report(const char * label, const crispembed_diff::Report & r, bool pass) {
    std::printf("%s pass=%d shape_ne=[", label, int(pass));
    for (size_t i = 0; i < r.shape.size(); ++i) std::printf("%s%lld", i ? "," : "", static_cast<long long>(r.shape[i]));
    std::printf("] n=%zu cos_min=%.9f cos_mean=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g "
                "native_norm=%.9g reference_norm=%.9g\n",
                r.n_elem, r.cos_min, r.cos_mean, r.cos_global, r.max_abs, r.mean_abs, r.rms, r.mine_norm, r.ref_norm);
}

// ggml's CUDA cuBLAS code permits this variable to override GGML_PREC_F32.
// Do not silently change the user's environment or accept reduced precision.
static bool f32_compute_allowed() {
    const char * value = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (!value || !*value) return true;
    std::string mode(value);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return mode == "auto" || mode == "f32" || mode == "fp32";
}

static bool same_existing_file(const char * a, const char * b) {
    std::error_code ec;
    const bool same = std::filesystem::equivalent(a, b, ec);
    return !ec && same;
}

struct resources {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    std::vector<uint8_t> arena;
    ggml_context * graph_ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ~resources() {
        if (alloc) ggml_gallocr_free(alloc);
        if (graph_ctx) ggml_free(graph_ctx);
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference.gguf> <output-f32-path>\n", argv[0]);
        return 2;
    }
    if (!*argv[3] || same_existing_file(argv[3], argv[1]) || same_existing_file(argv[3], argv[2])) {
        std::fprintf(stderr, "output must not overwrite a model or reference file\n");
        return 2;
    }
    if (!f32_compute_allowed()) {
        std::fprintf(stderr, "GGML_CUDA_CUBLAS_COMPUTE_TYPE overrides F32; refusing probe\n");
        return 2;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    const auto pixels = ref.get_f32(kPixels);
    const auto oracle = ref.get_f32(kOutput);
    const auto pixel_shape = ref.shape(kPixels), output_shape = ref.shape(kOutput);
    if (pixel_shape != std::vector<int64_t>{ kPatchWidth, kTokens } ||
        output_shape != std::vector<int64_t>{ kWidth, kTokens } || pixels.second != size_t(kPatchWidth * kTokens) ||
        oracle.second != size_t(kWidth * kTokens) || !finite_values(pixels.first, pixels.second) ||
        !finite_values(oracle.first, oracle.second)) {
        std::fprintf(stderr, "frozen prefill reference lacks a finite pixel/patch_embed pair of pinned shape\n");
        return 2;
    }

    resources r;
    // No registry auto-selection, scheduler, or CPU fallback: device 0 is CUDA.
    if (ggml_backend_cuda_get_device_count() < 1 || !(r.backend = ggml_backend_cuda_init(0)) ||
        !ggml_backend_is_cuda(r.backend)) {
        std::fprintf(stderr, "CUDA device 0 unavailable; refusing CPU fallback\n");
        return 2;
    }
    if (!core_gguf::load_weights_filtered(argv[1], r.backend, include_patch_tensors, nullptr, "xiaomi-patch-embed-cuda",
                                          r.weights))
        return 2;
    auto * weight = core_gguf::try_get(r.weights.tensors, kWeight);
    auto * bias = core_gguf::try_get(r.weights.tensors, kBias);
    if (r.weights.tensors.size() != 2 || !weight || !bias || weight->type != GGML_TYPE_F32 ||
        bias->type != GGML_TYPE_F32 || weight->ne[0] != kPatchWidth || weight->ne[1] != kWidth || weight->ne[2] != 1 ||
        weight->ne[3] != 1 || bias->ne[0] != kWidth || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        std::fprintf(stderr, "expected exactly the pinned F32 patch weight and bias shapes\n");
        return 2;
    }

    // Metadata-only context; the graph allocator owns CUDA buffers. The two
    // preallocated weights are already CUDA-resident and are not reallocated.
    r.arena.resize(ggml_tensor_overhead() * 8 + ggml_graph_overhead());
    r.graph_ctx = ggml_init({ r.arena.size(), r.arena.data(), true });
    if (!r.graph_ctx) return 2;
    auto * input = ggml_new_tensor_2d(r.graph_ctx, GGML_TYPE_F32, kPatchWidth, kTokens);
    ggml_set_input(input);
    auto * projection = ggml_mul_mat(r.graph_ctx, weight, input);
    ggml_mul_mat_set_prec(projection, GGML_PREC_F32);
    auto * output = ggml_add(r.graph_ctx, projection, bias);
    ggml_set_output(output);
    if (!ggml_backend_supports_op(r.backend, projection) || !ggml_backend_supports_op(r.backend, output)) {
        std::fprintf(stderr, "CUDA backend does not support the patch graph operations\n");
        return 2;
    }
    auto * graph = ggml_new_graph(r.graph_ctx);
    ggml_build_forward_expand(graph, output);
    r.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r.backend));
    if (!r.alloc || !ggml_gallocr_alloc_graph(r.alloc, graph)) {
        std::fprintf(stderr, "CUDA patch graph allocation failed\n");
        return 2;
    }
    ggml_backend_tensor_set(input, pixels.first, 0, pixels.second * sizeof(float));
    if (ggml_backend_graph_compute(r.backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CUDA patch graph computation failed\n");
        return 2;
    }
    std::vector<float> actual(oracle.second);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    std::printf("AUDIT_BACKEND CUDA device=0 name=%s graph=mul_mat_f32_plus_bias\n", ggml_backend_name(r.backend));

    FILE * file = std::fopen(argv[3], "wb");
    if (!file) {
        std::fprintf(stderr, "cannot open CUDA output file '%s'\n", argv[3]);
        return 2;
    }
    const bool wrote = std::fwrite(actual.data(), sizeof(float), actual.size(), file) == actual.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        std::fprintf(stderr, "failed writing CUDA output file '%s'\n", argv[3]);
        return 2;
    }

    // Match the CPU probe's oracle-derived negative control and strict gate.
    // Compare the real output last so optional diff dumps end with actual data.
    std::vector<float> perturbed(oracle.first, oracle.first + oracle.second);
    perturbed[0] += 1.0f;
    const auto negative = ref.compare(kOutput, perturbed.data(), perturbed.size());
    const bool rejected = negative.found && negative.n_elem == perturbed.size() && std::isfinite(negative.max_abs) &&
                          negative.max_abs > kMaxAbs && !strict_pass(negative, perturbed.size());
    print_report("PATCH_NEGATIVE_CONTROL", negative, rejected);
    const auto report = ref.compare(kOutput, actual.data(), actual.size());
    const bool pass = finite_values(actual.data(), actual.size()) && strict_pass(report, actual.size());
    std::printf("PATCH_INPUT shape_ne=[%lld,%lld] weight=F32_export reference=BF16_widened_to_F32 "
                "stage=prefill.vision.patch_embed only input_fnv1a=%016llx\n",
                static_cast<long long>(kPatchWidth), static_cast<long long>(kTokens),
                static_cast<unsigned long long>(digest(pixels.first, pixels.second)));
    print_report("PATCH_CUDA_METRIC", report, pass);
    if (!rejected) {
        std::fprintf(stderr, "negative control was not rejected\n");
        return 2;
    }
    return pass ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
