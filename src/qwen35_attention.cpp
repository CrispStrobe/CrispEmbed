#include "lighton4b_physical_native_ops.h"
#include "lighton4b_physical_f16_attention.h"
#include "lighton4b_upstream_physical_semantics.h"
#include "qwen35_attention.h"
#include "qwen35_math.h"
#include "core/env_gate.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace qwen35_ocr {
static bool matrix(ggml_tensor * t, int64_t columns, int64_t rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool finite(const std::vector<float> & x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
static bool norm_head(const float * x, const float * weights, int width, float eps, float * y, bool physical) {
    if (lighton4b_physical_native_component(physical, 2)) {
        std::vector<float> values;
        if (!lighton4b_physical_rms(x, weights, width, 1, 1, values, eps)) return false;
        std::copy(values.begin(), values.end(), y);
        return true;
    }
    const float scale = rms_scale_cpu(x, width, eps);
    for (int i = 0; i < width; ++i) y[i] = (x[i] * scale) * lighton4b_upstream_gamma(weights[i], physical);
    return true;
}
static void rotate(float * x, const attention_config & c, const float * positions, int bt, int count) {
    const int half = c.rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        int axis = 0;
        if (i % 3 == 1 && i < 3 * c.mrope_section[1]) axis = 1;
        if (i % 3 == 2 && i < 3 * c.mrope_section[2]) axis = 2;
        const float frequency = 1.0f / std::pow(c.theta, float(2 * i) / c.rotary_dim);
        const float angle = frequency * positions[axis * count + bt];
        const float cosine = std::cos(angle), sine = std::sin(angle);
        const float left = x[i], right = x[i + half];
        x[i] = left * cosine - right * sine;
        x[i + half] = right * cosine + left * sine;
    }
}
bool attention_cpu(const attention_weights & w, const attention_config & c, const float * hidden,
                   const float * positions, int batch, int tokens, int threads, const attention_cache * previous,
                   attention_outputs & output, converted_weight_cache * weights_cache) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !hidden || !positions || !w.q || batch < 1 || batch > 4 ||
        tokens < 1 || tokens > 4096 || threads < 1 || threads > 256 || c.query_heads < 1 || c.query_heads > 128 ||
        c.kv_heads < 1 || c.kv_heads > c.query_heads || c.query_heads % c.kv_heads || c.head_dim < 1 ||
        c.head_dim > 256 || c.rotary_dim < 2 || c.rotary_dim > c.head_dim || c.rotary_dim % 2 ||
        !std::isfinite(c.theta) || c.theta <= 1 || !std::isfinite(c.eps) || c.eps <= 0)
        return false;
    for (int section : c.mrope_section)
        if (section < 0 || section > c.rotary_dim / 2) return false;
    if (c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] != c.rotary_dim / 2) return false;
    const int width = int(w.q->ne[0]), d = c.head_dim, qh = c.query_heads, kh = c.kv_heads;
    if (w.upstream_physical && (!lighton4b_upstream_physical_mode_valid(true) || batch != 1 || tokens > 16 ||
                                width != 2560 || d != 256 || qh != 16 || kh != 4 || c.rotary_dim != 64))
        return false;
    if (!lighton4b_physical_native_ops_valid(w.upstream_physical) ||
        !lighton4b_physical_f16_attention_valid(w.upstream_physical))
        return false;
    const bool physical_f16 = lighton4b_physical_f16_attention_mode(w.upstream_physical);
    const bool source_chunks = lighton4b_source_prefill_chunks_mode(w.upstream_physical);
    if (!lighton4b_source_prefill_chunks_valid(w.upstream_physical) ||
        (previous && previous->physical_source_prefill_chunks != source_chunks))
        return false;
    const bool q6_tiled = lighton4b_physical_q6_tiled_mode(w.upstream_physical);
    if (!lighton4b_physical_q6_tiled_valid(w.upstream_physical) ||
        (previous && previous->physical_q6_tiled != q6_tiled))
        return false;
    const bool repack = lighton4b_physical_q4_repack_mode(w.upstream_physical);
    if (!lighton4b_physical_q4_repack_valid(w.upstream_physical) || w.physical_q4_repack != repack ||
        (w.upstream_physical && bool(weights_cache && weights_cache->physical_repack_enabled()) != repack) ||
        (previous && previous->physical_q4_repack != repack))
        return false;
    if (previous && (previous->upstream_physical != w.upstream_physical ||
                     previous->physical_native_ops != lighton4b_physical_native_ops_mode(w.upstream_physical) ||
                     previous->physical_f16_attention != physical_f16))
        return false;
    if (width < 1 || width > 8192 || !matrix(w.q, width, 2 * qh * d) || !matrix(w.k, width, kh * d) ||
        !matrix(w.v, width, kh * d) || !matrix(w.out, qh * d, width) || !matrix(w.q_norm, d, 1) ||
        !matrix(w.k_norm, d, 1))
        return false;
    const int past = previous ? previous->tokens : 0, total = past + tokens, count = batch * tokens;
    if (physical_f16 && (threads != 4 || total > 2048)) return false;
    if (past < 0 || total > 4096 || size_t(batch) * total * kh * d > 16 * 1024 * 1024 ||
        size_t(count) * qh * d > 16 * 1024 * 1024)
        return false;
    if (previous &&
        (previous->batch != batch || previous->heads != kh || previous->width != d ||
         previous->keys.size() != size_t(batch) * kh * past * d || previous->values.size() != previous->keys.size() ||
         !finite(previous->keys) || !finite(previous->values)))
        return false;
    for (int i = 0; i < 3 * count; ++i)
        if (!std::isfinite(positions[i])) return false;
    std::vector<std::vector<float>> projected;
    if (!project_cpu({ w.q, w.k, w.v }, hidden, width, count, threads, projected, weights_cache, w.upstream_physical))
        return false;
    output.q = std::move(projected[0]);
    output.k = std::move(projected[1]);
    output.v = std::move(projected[2]);
    std::vector<float> qweight(d), kweight(d);
    if (!tensor_floats(w.q_norm, qweight.data()) || !tensor_floats(w.k_norm, kweight.data())) return false;
    output.q_norm.resize(size_t(count) * qh * d);
    output.k_norm.resize(size_t(count) * kh * d);
    for (int bt = 0; bt < count; ++bt) {
        for (int h = 0; h < qh; ++h)
            if (!norm_head(output.q.data() + (size_t(bt) * qh + h) * 2 * d, qweight.data(), d, c.eps,
                           output.q_norm.data() + (size_t(bt) * qh + h) * d, w.upstream_physical))
                return false;
        for (int h = 0; h < kh; ++h)
            if (!norm_head(output.k.data() + (size_t(bt) * kh + h) * d, kweight.data(), d, c.eps,
                           output.k_norm.data() + (size_t(bt) * kh + h) * d, w.upstream_physical))
                return false;
    }
    auto queries = output.q_norm, keys = output.k_norm;
    for (int bt = 0; bt < count; ++bt) {
        for (int h = 0; h < qh; ++h) rotate(queries.data() + (size_t(bt) * qh + h) * d, c, positions, bt, count);
        for (int h = 0; h < kh; ++h) rotate(keys.data() + (size_t(bt) * kh + h) * d, c, positions, bt, count);
    }
    auto & cache = output.cache;
    cache.upstream_physical = w.upstream_physical;
    cache.physical_native_ops = lighton4b_physical_native_ops_mode(w.upstream_physical);
    cache.physical_f16_attention = physical_f16;
    cache.physical_q4_repack = repack;
    cache.physical_q6_tiled = q6_tiled;
    cache.physical_source_prefill_chunks = source_chunks;
    cache.batch = batch;
    cache.heads = kh;
    cache.tokens = total;
    cache.width = d;
    cache.keys.resize(size_t(batch) * kh * total * d);
    cache.values.resize(cache.keys.size());
    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < kh; ++h) {
            const size_t base = (size_t(b) * kh + h) * total * d;
            if (past) {
                const size_t old = (size_t(b) * kh + h) * past * d;
                std::copy_n(previous->keys.data() + old, size_t(past) * d, cache.keys.data() + base);
                std::copy_n(previous->values.data() + old, size_t(past) * d, cache.values.data() + base);
            }
            for (int t = 0; t < tokens; ++t) {
                const size_t src = (size_t(b * tokens + t) * kh + h) * d, dst = base + size_t(past + t) * d;
                std::copy_n(keys.data() + src, d, cache.keys.data() + dst);
                std::copy_n(output.v.data() + src, d, cache.values.data() + dst);
            }
        }
    if (physical_f16) {
        std::vector<float> pooled;
        attention_cache canonical;
        if (!lighton4b_physical_f16_attention_cpu(queries, output.q, cache, past, tokens, threads, pooled, canonical)) {
            output = {};
            return false;
        }
        output.pooled = std::move(pooled);
        cache = std::move(canonical);
    } else {
        output.pooled.resize(size_t(count) * qh * d);
        std::vector<float> scores(total);
        const float scale = 1.0f / std::sqrt(float(d));
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < qh; ++h) {
                    const int limit = past + t + 1;
                    const size_t base = (size_t(b) * kh + h / (qh / kh)) * total * d;
                    const size_t qbase = (size_t(b * tokens + t) * qh + h) * d;
                    float maximum = -INFINITY;
                    for (int s = 0; s < limit; ++s) {
                        float dot = 0;
                        for (int i = 0; i < d; ++i) dot += queries[qbase + i] * cache.keys[base + size_t(s) * d + i];
                        scores[s] = dot * scale;
                        maximum = std::max(maximum, scores[s]);
                    }
                    float sum = 0;
                    for (int s = 0; s < limit; ++s) {
                        scores[s] = std::exp(scores[s] - maximum);
                        sum += scores[s];
                    }
                    for (int i = 0; i < d; ++i) {
                        float value = 0;
                        for (int s = 0; s < limit; ++s)
                            value += (scores[s] / sum) * cache.values[base + size_t(s) * d + i];
                        const size_t gate = (size_t(b * tokens + t) * qh + h) * 2 * d + d + i;
                        output.pooled[qbase + i] = value / (1.0f + std::exp(-output.q[gate]));
                    }
                }
    }
    projected.clear();
    if (!project_cpu({ w.out }, output.pooled.data(), qh * d, count, threads, projected, weights_cache,
                     w.upstream_physical))
        return false;
    output.attention = std::move(projected[0]);
    return finite(output.attention) && finite(cache.keys) && finite(cache.values);
}
} // namespace qwen35_ocr
