#pragma once
#include "lighton4b_upstream_physical_semantics.h"
#include "lighton4b_physical_f16_attention.h"
#include "qwen35_weight_cache.h"
#include "lighton4b_physical_q6_tiled.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <memory>
#include <cstring>
namespace qwen35_ocr {
inline bool lighton4b_physical_native_ops_master(bool physical) {
#if defined(__EMSCRIPTEN__)
    return false;
#else
    return physical && lighton4b_upstream_physical_mode_valid(true) && core_env::on("CRISPEMBED_QWEN35_OCR") &&
           core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") && core_env::on("CRISPEMBED_LIGHTON4B_QUANT") &&
           core_env::on("CRISPEMBED_LIGHTON4B_Q6_K") && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS");
#endif
}
// Diagnostic selector is inert unless the explicit physical master is enabled.
// Bit1=layer projections, bit2=RMS, bit4=tied head. Unknown values fail closed.
inline uint8_t lighton4b_physical_native_ops_mode(bool physical) {
    if (!lighton4b_physical_native_ops_master(physical)) return 0;
    const char * value = std::getenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION");
    if (!value || !*value || std::strcmp(value, "all") == 0) return 7;
    if (core_env::spells_off(value)) return 0;
    if (std::strcmp(value, "quant") == 0) return 5;
    if (std::strcmp(value, "rms") == 0) return 2;
    if (std::strcmp(value, "layers") == 0) return 1;
    if (std::strcmp(value, "head") == 0) return 4;
    return 255;
}
inline bool lighton4b_physical_native_ops_valid(bool physical) {
    return lighton4b_physical_native_ops_mode(physical) != 255;
}
inline bool lighton4b_physical_native_ops(bool physical) {
    const auto mode = lighton4b_physical_native_ops_mode(physical);
    return mode != 0 && mode != 255;
}
inline bool lighton4b_physical_native_component(bool physical, uint8_t bit) {
    const auto mode = lighton4b_physical_native_ops_mode(physical);
    return mode != 255 && (mode & bit) != 0;
}
inline bool lighton4b_physical_q4_repack_mode(bool physical) {
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK");
}
inline bool lighton4b_physical_q4_repack_valid(bool physical) {
    return !lighton4b_physical_q4_repack_mode(physical) ||
           (lighton4b_physical_native_component(physical, 1) && lighton4b_physical_f16_attention_available());
}
inline bool lighton4b_physical_q6_tiled_valid(bool physical) {
    return !lighton4b_physical_q6_tiled_mode(physical) ||
           (lighton4b_physical_native_component(physical, 1) && lighton4b_physical_f16_attention_available() &&
            lighton4b_physical_q6_tiled_available());
}
// Borrow original quant storage, or retain explicitly owned Q4 repacked bytes.
// No F32 weight materialization/cache and no shared trait mutation.
inline bool lighton4b_physical_matmul(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width,
                                      int tokens, int threads, std::vector<std::vector<float>> & output,
                                      bool head = false, converted_weight_cache * weights_cache = nullptr) {
    output.clear();
    if (!lighton4b_physical_native_component(true, head ? 4 : 1) || !hidden || matrices.empty() ||
        matrices.size() > 4 || width < 1 || width > 32768 || tokens < 1 || tokens > 16 || threads < 1 || threads > 256)
        return false;
    const bool repack = lighton4b_physical_q4_repack_mode(true);
    if (!lighton4b_physical_q4_repack_valid(true) || !lighton4b_physical_q6_tiled_valid(true) ||
        (!head && bool(weights_cache && weights_cache->physical_repack_enabled()) != repack))
        return false;
    size_t elements = size_t(width) * tokens;
    for (auto * m : matrices) {
        if (!m || m->ne[0] != width || m->ne[1] < 1 || m->ne[1] > 300000 || m->ne[2] != 1 || m->ne[3] != 1 ||
            !ggml_is_contiguous(m) || !m->data || !m->buffer || !ggml_backend_buffer_is_host(m->buffer) ||
            (m->type != GGML_TYPE_Q4_K && m->type != GGML_TYPE_Q6_K && m->type != GGML_TYPE_F32))
            return false;
        elements += size_t(m->ne[1]) * tokens;
    }
    if (elements > 16ull * 1024 * 1024 ||
        !std::all_of(hidden, hidden + size_t(width) * tokens, [](float x) { return std::isfinite(x); }))
        return false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ 1024 * 1024 + elements * sizeof(float), nullptr, false }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    std::memcpy(input->data, hidden, size_t(width) * tokens * sizeof(float));
    auto * graph = ggml_new_graph(ctx.get());
    std::vector<ggml_tensor *> results;
    std::vector<std::vector<float>> tiled_results(
        lighton4b_physical_q6_tiled_mode(true) && tokens >= 8 ? matrices.size() : 0);
    std::vector<std::shared_ptr<ggml_tensor>> leases;
    for (auto * source : matrices) {
        if (source->type == GGML_TYPE_Q6_K && tokens >= 8 && lighton4b_physical_q6_tiled_mode(true)) {
            auto & values = tiled_results[results.size()];
            if (!lighton4b_physical_q6_tiled_cpu(source->data, ggml_nbytes(source), hidden, width, source->ne[1],
                                                 tokens, threads, values))
                return false;
            results.push_back(nullptr);
            continue;
        }
        ggml_tensor * weight = nullptr;
        if (repack && source->type == GGML_TYPE_Q4_K) {
            if (head || !weights_cache) return false; // authenticated physical head is Q6
            auto lease = weights_cache->get_physical_repacked(source, ggml_backend_get_device(backend.get()));
            if (!lease) return false; // explicit mode never falls back to a different arithmetic route
            weight = lease.get();
            leases.push_back(std::move(lease));
        } else {
            ggml_set_no_alloc(ctx.get(), true);
            weight = ggml_new_tensor_2d(ctx.get(), source->type, width, source->ne[1]);
            ggml_set_no_alloc(ctx.get(), false);
            weight->data = source->data;
        }
        auto * result = ggml_mul_mat(ctx.get(), weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        ggml_build_forward_expand(graph, result);
        results.push_back(result);
    }
    if (std::any_of(results.begin(), results.end(), [](auto * value) { return value != nullptr; }) &&
        ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
        return false;
    std::vector<std::vector<float>> complete;
    for (size_t i = 0; i < results.size(); ++i) {
        auto * result = results[i];
        if (!result) {
            complete.push_back(std::move(tiled_results[i]));
            continue;
        }
        auto * values = static_cast<const float *>(result->data);
        const auto count = size_t(ggml_nelements(result));
        if (!std::all_of(values, values + count, [](float x) { return std::isfinite(x); })) return false;
        complete.emplace_back(values, values + count);
    }
    output = std::move(complete);
    return true;
}
inline bool lighton4b_physical_rms(const float * hidden, const float * gamma, int width, int tokens, int threads,
                                   std::vector<float> & output, float eps = 1e-6f) {
    output.clear();
    if (!lighton4b_physical_native_component(true, 2) || !hidden || !gamma || width < 1 || width > 4096 || tokens < 1 ||
        tokens > 4096 || size_t(width) * tokens > 1048576 || threads < 1 || threads > 256 || !std::isfinite(eps) ||
        eps <= 0 || !std::all_of(hidden, hidden + size_t(width) * tokens, [](float x) { return std::isfinite(x); }) ||
        !std::all_of(gamma, gamma + width, [](float x) { return std::isfinite(x); }))
        return false;
    const size_t elements = size_t(width) * tokens;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ 1024 * 1024 + (3 * elements + width) * sizeof(float), nullptr, false }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    auto * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    std::memcpy(input->data, hidden, elements * sizeof(float));
    std::memcpy(weight->data, gamma, size_t(width) * sizeof(float));
    auto * result = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, eps), weight);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    auto * values = static_cast<const float *>(result->data);
    if (!std::all_of(values, values + elements, [](float x) { return std::isfinite(x); })) return false;
    output.assign(values, values + elements);
    return true;
}
} // namespace qwen35_ocr
