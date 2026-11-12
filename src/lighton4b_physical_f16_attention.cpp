#include "lighton4b_physical_f16_attention.h"
#include "lighton4b_physical_rope.h"
#include <cstring>
#include "qwen35_attention.h"
#include <algorithm>
#include <cmath>
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "llamafile/sgemm.h"
#include <atomic>
#include <memory>
extern "C" void ggml_vec_dot_f16(int, float *, size_t, ggml_fp16_t *, size_t, ggml_fp16_t *, size_t, int);
#endif
namespace qwen35_ocr {
bool lighton4b_physical_f16_attention_available() {
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__) && defined(__linux__) &&                   \
    defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
namespace {
struct f16_operator {
    ggml_threadpool * pool;
    bool pv;
    int padded;
    std::atomic_bool accepted{ true };
};
void physical_f16_product(ggml_tensor * dst, const ggml_tensor *, const ggml_tensor * cache, const ggml_tensor * rhs,
                          int ith, int nth, void * user) {
    auto & state = *static_cast<f16_operator *>(user);
    if (nth != 4) {
        state.accepted = false;
        return;
    }
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth) static_cast<float *>(dst->data)[i] = 0;
    if (rhs->type != GGML_TYPE_F16 || cache->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        state.accepted = false;
        return;
    }
    const int n = int(rhs->ne[1]), m = state.pv ? 256 : state.padded, k = state.pv ? state.padded : 256;
    const int lda = state.pv ? 2048 : 1024;
    auto * output = static_cast<float *>(dst->data);
    auto * left = static_cast<ggml_fp16_t *>(cache->data);
    auto * right = static_cast<ggml_fp16_t *>(rhs->data);
    if (n == 1) {
        // Source SGEMM explicitly declines one query. Its F16 CPU trait then
        // uses this original paired-F16 dot, not the fork's global F16/F32 trait.
        for (int index = ith; index < 16 * m; index += nth) {
            const int h = index / m, row = index % m;
            auto * a = left + (state.pv ? (h / 4) * 256 * 2048 : (h / 4) * 256) + row * lda;
            ggml_vec_dot_f16(k, output + index, 0, a, 0, right + h * k, 0, 1);
        }
        return;
    }
    ggml_compute_params params{};
    params.ith = ith;
    params.nth = nth;
    params.threadpool = state.pool;
    for (int h = 0; h < 16; ++h) {
        auto * a = left + (state.pv ? (h / 4) * 256 * 2048 : (h / 4) * 256);
        if (!llamafile_sgemm(&params, m, n, k, a, lda, right + h * n * k, k, output + h * n * m, m, GGML_TYPE_F16,
                             GGML_TYPE_F16, GGML_TYPE_F32)) {
            state.accepted = false;
            return; // Unsupported is never a kernel substitution.
        }
    }
}
} // namespace
#endif
bool lighton4b_physical_f16_attention_cpu(const std::vector<float> & queries, const std::vector<float> & fused,
                                          const attention_cache & cache, int past, int tokens, int threads,
                                          std::vector<float> & pooled, attention_cache & canonical_cache) {
    pooled.clear();
    canonical_cache = {};
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
    if (!lighton4b_physical_f16_attention_mode(cache.upstream_physical) ||
        !lighton4b_physical_f16_attention_valid(cache.upstream_physical) || !cache.physical_f16_attention ||
        threads != 4 || tokens < 1 || tokens > 16 || past < 0 || past > 2048 - tokens ||
        cache.tokens != past + tokens || cache.batch != 1 || cache.heads != 4 || cache.width != 256 ||
        cache.keys.size() != size_t(cache.tokens) * 1024 || cache.values.size() != cache.keys.size() ||
        queries.size() != size_t(tokens) * 4096 || fused.size() != size_t(tokens) * 8192)
        return false;
    auto finite = [](const std::vector<float> & x) {
        return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
    };
    if (!finite(queries) || !finite(fused) || !finite(cache.keys) || !finite(cache.values)) return false;
    const int total = cache.tokens, padded = ((total + 255) / 256) * 256;
    // Bounded transient context only, never model weights or a full F32 model.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 64ull << 20, nullptr, false }), ggml_free);
    if (!ctx) return false;
    auto pp = ggml_threadpool_params_default(4);
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pp),
                                                                           ggml_threadpool_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!pool || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), 4);
    ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
    auto * c = ctx.get();
    auto * key = ggml_new_tensor_2d(c, GGML_TYPE_F16, 1024, 2048);
    auto * value = ggml_new_tensor_2d(c, GGML_TYPE_F16, 2048, 1024);
    std::fill_n(static_cast<ggml_fp16_t *>(key->data), 1024 * 2048, ggml_fp32_to_fp16(0));
    std::fill_n(static_cast<ggml_fp16_t *>(value->data), 1024 * 2048, ggml_fp32_to_fp16(0));
    auto canonical = cache;
    for (int h = 0; h < 4; ++h)
        for (int t = 0; t < total; ++t)
            for (int i = 0; i < 256; ++i) {
                const size_t ix = (size_t(h) * total + t) * 256 + i;
                const auto kh = ggml_fp32_to_fp16(cache.keys[ix]), vh = ggml_fp32_to_fp16(cache.values[ix]);
                const float kf = ggml_fp16_to_fp32(kh), vf = ggml_fp16_to_fp32(vh);
                if (!std::isfinite(kf) || !std::isfinite(vf) ||
                    (t < past && (kf != cache.keys[ix] || vf != cache.values[ix])))
                    return false;
                canonical.keys[ix] = kf;
                canonical.values[ix] = vf;
                static_cast<ggml_fp16_t *>(key->data)[t * 1024 + h * 256 + i] = kh;
                static_cast<ggml_fp16_t *>(value->data)[(h * 256 + i) * 2048 + t] = vh;
            }
    auto * query = ggml_new_tensor_3d(c, GGML_TYPE_F16, 256, tokens, 16);
    for (int h = 0; h < 16; ++h)
        for (int t = 0; t < tokens; ++t)
            for (int i = 0; i < 256; ++i) {
                const auto half = ggml_fp32_to_fp16(queries[(size_t(t) * 16 + h) * 256 + i]);
                if (!std::isfinite(ggml_fp16_to_fp32(half))) return false;
                static_cast<ggml_fp16_t *>(query->data)[(h * tokens + t) * 256 + i] = half;
            }
    f16_operator qk{ pool.get(), false, padded }, pv{ pool.get(), true, padded };
    auto * qshape = ggml_new_tensor_3d(c, GGML_TYPE_F32, padded, tokens, 16);
    auto * logits = ggml_map_custom3(c, qshape, key, query, physical_f16_product, 4, &qk);
    auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, padded, 32);
    for (int t = 0; t < 32; ++t)
        for (int s = 0; s < padded; ++s)
            static_cast<float *>(mask->data)[t * padded + s] = (t < tokens && s <= past + t) ? 0.f : -INFINITY;
    auto * probs = ggml_soft_max_ext(c, logits, mask, .0625f, 0.f);
    auto * ph = ggml_cast(c, probs, GGML_TYPE_F16);
    auto * vshape = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, tokens, 16);
    auto * raw = ggml_map_custom3(c, vshape, value, ph, physical_f16_product, 4, &pv);
    auto * context = ggml_cont_2d(c, ggml_permute(c, raw, 0, 2, 1, 3), 4096, tokens);
    auto * gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4096, tokens);
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 16; ++h)
            std::copy_n(fused.data() + (size_t(t) * 16 + h) * 512 + 256, 256,
                        static_cast<float *>(gate->data) + (size_t(t) * 16 + h) * 256);
    auto * gated = ggml_mul(c, context, ggml_sigmoid(c, gate));
    auto * graph = ggml_new_graph_custom(c, 128, false);
    ggml_build_forward_expand(graph, gated);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS || !qk.accepted || !pv.accepted)
        return false;
    auto * data = static_cast<float *>(gated->data);
    std::vector<float> result(data, data + size_t(tokens) * 4096);
    if (!finite(result)) return false;
    pooled = std::move(result);
    canonical_cache = std::move(canonical);
    return true;
#else
    (void)queries;
    (void)fused;
    (void)cache;
    (void)past;
    (void)tokens;
    (void)threads;
    return false;
#endif
}
// Pinned source6c01da627 IMROPE parameters. YARN is inactive (ext0/scale1).
bool lighton4b_physical_rope_cpu(const std::vector<float> & normalized, const attention_config & config,
                                  const float * positions, int tokens, int heads, int threads,
                                  std::vector<float> & rotated) {
    // A rejected aliased call must preserve its input; ordinary failures clear output.
    if (&normalized == &rotated) return false;
    rotated.clear();
    try {
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
    if (!lighton4b_physical_rope_mode(true) || !lighton4b_physical_rope_valid(true) || !positions ||
        tokens < 1 || tokens > 16 || (heads != 4 && heads != 16) || threads < 1 || threads > 256 ||
        config.head_dim != 256 || config.query_heads != 16 || config.kv_heads != 4 || config.rotary_dim != 64 ||
        config.theta != 10000000.f || config.mrope_section[0] != 11 || config.mrope_section[1] != 11 ||
        config.mrope_section[2] != 10 || normalized.size() != size_t(tokens) * heads * 256 ||
        !std::all_of(normalized.begin(), normalized.end(), [](float v) { return std::isfinite(v); })) return false;
    std::vector<int32_t> axes(size_t(tokens) * 4, 0);
    for (int i = 0; i < 3 * tokens; ++i) {
        const float v = positions[i];
        if (!std::isfinite(v) || v < 0 || v >= 262144 || std::trunc(v) != v) return false;
        axes[size_t(i)] = int32_t(v);
    }
    // Two activation tensors only, bounded by the physical caller's16-token chunk; no persistent cache.
    const size_t bytes = normalized.size() * sizeof(float);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ 2 * bytes + axes.size() * sizeof(int32_t) + (1ull << 20), nullptr, false }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, heads, tokens);
    auto * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4 * tokens);
    std::memcpy(x->data, normalized.data(), bytes);
    std::memcpy(pos->data, axes.data(), axes.size() * sizeof(int32_t));
    int sections[4] = { 11, 11, 10, 0 };
    auto * y = ggml_rope_multi(ctx.get(), x, pos, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE,
                              262144, 10000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    const auto * data = static_cast<const float *>(y->data);
    std::vector<float> result(data, data + normalized.size());
    if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); })) return false;
    rotated = std::move(result);
    return true;
#else
    (void)normalized; (void)config; (void)positions; (void)tokens; (void)heads; (void)threads;
    return false;
#endif
    } catch (...) {
        rotated.clear();
        return false;
    }
}
} // namespace qwen35_ocr
