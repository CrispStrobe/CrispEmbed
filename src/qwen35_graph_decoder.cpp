// Stored-type MiniCPM decoder candidate, composed from this repository's
// qwen35_decoder/attention/linear arithmetic. No upstream engine is copied.
// See docs/minicpm-ocr-graph-v1.md.
#include "qwen35_graph_decoder.h"
#include "qwen35_math.h"
#include "core/env_gate.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace qwen35_ocr {
namespace {
bool enabled() {
    return core_env::on("CRISPEMBED_MINICPM_OCR_GRAPH") && core_env::on("CRISPEMBED_QWEN35_OCR") &&
           !core_env::on("CRISPEMBED_QWEN35_IMATRIX");
}
struct context_free {
    void operator()(ggml_context * p) const { ggml_free(p); }
};
struct allocation_free {
    void operator()(ggml_gallocr * p) const { ggml_gallocr_free(p); }
};
bool stored_project(ggml_backend_t backend, const std::vector<ggml_tensor *> & matrices, const float * hidden,
                    int width, int tokens, int threads, std::vector<std::vector<float>> & output) {
    output.clear();
    if (!enabled() || !backend || !hidden || matrices.empty() || matrices.size() > 4 || width < 1 || width > 32768 ||
        tokens < 1 || tokens > 4096 || threads < 1 || threads > 256)
        return false;
    size_t elements = size_t(width) * tokens;
    if (!std::all_of(hidden, hidden + elements, [](float v) { return std::isfinite(v); })) return false;
    for (auto * weight : matrices) {
        if (!weight || weight->ne[0] != width || weight->ne[1] < 1 || weight->ne[1] > 300000 || weight->ne[2] != 1 ||
            weight->ne[3] != 1 || !matrix_type_supported(weight) || !weight->buffer ||
            !ggml_backend_buffer_is_host(weight->buffer))
            return false;
        elements += size_t(weight->ne[1]) * tokens;
        if (elements > 128ull * 1024 * 1024) return false;
    }
    // Weight tensors remain owned by the loaded model through synchronous
    // compute; the graph allocator owns only input/output/scratch buffers.
    const size_t meta = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
    std::unique_ptr<ggml_context, context_free> ctx(ggml_init({ meta, nullptr, true }));
    if (!ctx) return false;
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    ggml_set_input(input);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    std::vector<ggml_tensor *> results;
    for (auto * weight : matrices) {
        auto * result = ggml_mul_mat(ctx.get(), weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        results.push_back(result);
    }
    std::unique_ptr<ggml_gallocr, allocation_free> alloc(ggml_gallocr_new(ggml_backend_cpu_buffer_type()));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc.get(), graph)) return false;
    ggml_backend_tensor_set(input, hidden, size_t(0), size_t(width) * tokens * sizeof(float));
    ggml_backend_cpu_set_n_threads(backend, threads);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    std::vector<std::vector<float>> pending;
    for (auto * result : results) {
        pending.emplace_back(size_t(ggml_nelements(result)));
        auto & values = pending.back();
        ggml_backend_tensor_get(result, values.data(), 0, values.size() * sizeof(float));
        if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); })) return false;
    }
    output = std::move(pending);
    return true;
}
static bool shape(ggml_tensor * t, int columns, int rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool rms(const std::vector<float> & x, ggml_tensor * norm, int width, float eps, std::vector<float> & y) {
    if (!shape(norm, width, 1)) return false;
    std::vector<float> weights(width);
    if (!tensor_floats(norm, weights.data())) return false;
    y.resize(x.size());
    for (size_t base = 0; base < x.size(); base += width) {
        const float scale = rms_scale_cpu(x.data() + base, width, eps);
        for (int i = 0; i < width; ++i) y[base + i] = (x[base + i] * scale) * (1.0f + weights[i]);
    }
    return true;
}
static bool matrix(ggml_tensor * t, int64_t columns, int64_t rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool finite(const std::vector<float> & x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
static void norm_head(const float * x, const float * weights, int width, float eps, float * y) {
    const float scale = rms_scale_cpu(x, width, eps);
    for (int i = 0; i < width; ++i) y[i] = (x[i] * scale) * (1.0f + weights[i]);
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
bool graph_attention(ggml_backend_t backend, const attention_weights & w, const attention_config & c,
                     const float * hidden, const float * positions, int batch, int tokens, int threads,
                     const attention_cache * previous, attention_outputs & output) {
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
    if (width < 1 || width > 8192 || !matrix(w.q, width, 2 * qh * d) || !matrix(w.k, width, kh * d) ||
        !matrix(w.v, width, kh * d) || !matrix(w.out, qh * d, width) || !matrix(w.q_norm, d, 1) ||
        !matrix(w.k_norm, d, 1))
        return false;
    const int past = previous ? previous->tokens : 0, total = past + tokens, count = batch * tokens;
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
    if (!stored_project(backend, { w.q, w.k, w.v }, hidden, width, count, threads, projected)) return false;
    output.q = std::move(projected[0]);
    output.k = std::move(projected[1]);
    output.v = std::move(projected[2]);
    std::vector<float> qweight(d), kweight(d);
    if (!tensor_floats(w.q_norm, qweight.data()) || !tensor_floats(w.k_norm, kweight.data())) return false;
    output.q_norm.resize(size_t(count) * qh * d);
    output.k_norm.resize(size_t(count) * kh * d);
    for (int bt = 0; bt < count; ++bt) {
        for (int h = 0; h < qh; ++h)
            norm_head(output.q.data() + (size_t(bt) * qh + h) * 2 * d, qweight.data(), d, c.eps,
                      output.q_norm.data() + (size_t(bt) * qh + h) * d);
        for (int h = 0; h < kh; ++h)
            norm_head(output.k.data() + (size_t(bt) * kh + h) * d, kweight.data(), d, c.eps,
                      output.k_norm.data() + (size_t(bt) * kh + h) * d);
    }
    auto queries = output.q_norm, keys = output.k_norm;
    for (int bt = 0; bt < count; ++bt) {
        for (int h = 0; h < qh; ++h) rotate(queries.data() + (size_t(bt) * qh + h) * d, c, positions, bt, count);
        for (int h = 0; h < kh; ++h) rotate(keys.data() + (size_t(bt) * kh + h) * d, c, positions, bt, count);
    }
    auto & cache = output.cache;
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
                    for (int s = 0; s < limit; ++s) value += (scores[s] / sum) * cache.values[base + size_t(s) * d + i];
                    const size_t gate = (size_t(b * tokens + t) * qh + h) * 2 * d + d + i;
                    output.pooled[qbase + i] = value / (1.0f + std::exp(-output.q[gate]));
                }
            }
    projected.clear();
    if (!stored_project(backend, { w.out }, output.pooled.data(), qh * d, count, threads, projected)) return false;
    output.attention = std::move(projected[0]);
    return finite(output.attention) && finite(cache.keys) && finite(cache.values);
}
bool graph_linear(ggml_backend_t backend, const linear_weights & weights, const float * hidden, int batch, int tokens,
                  int threads, const float * conv_history, const float * state, projected_linear_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !hidden || batch < 1 || batch > 4 || tokens < 1 || tokens > 4096 ||
        threads < 1 || threads > 256 || !weights.a_log || !weights.norm || !weights.qkv || !weights.conv)
        return false;
    const int64_t heads = weights.a_log->ne[0], width = weights.norm->ne[0], hidden_width = weights.qkv->ne[0];
    const int64_t kernel = weights.conv->ne[0];
    if (heads < 1 || heads > 128 || width < 1 || width > 256 || hidden_width < 1 || hidden_width > 8192 || kernel < 1 ||
        kernel > 16)
        return false;
    const int64_t group = heads * width;
    if (!shape(weights.a_log, heads, 1) || !shape(weights.dt_bias, heads, 1) || !shape(weights.norm, width, 1) ||
        !shape(weights.qkv, hidden_width, 3 * group) || !shape(weights.z, hidden_width, group) ||
        !shape(weights.a, hidden_width, heads) || !shape(weights.b, hidden_width, heads) ||
        !shape(weights.out, group, hidden_width) || weights.conv->ne[1] != 1 || weights.conv->ne[2] != 3 * group ||
        weights.conv->ne[3] != 1)
        return false;
    std::vector<std::vector<float>> projections;
    if (!stored_project(backend, { weights.qkv, weights.z, weights.a, weights.b }, hidden, int(hidden_width),
                        batch * tokens, threads, projections))
        return false;
    output.qkv = std::move(projections[0]);
    output.z = std::move(projections[1]);
    output.a = std::move(projections[2]);
    output.b = std::move(projections[3]);
    std::vector<float> conv(size_t(3 * group * kernel)), a_log(size_t(heads), 0.0f), dt_bias(size_t(heads), 0.0f),
        norm(size_t(width), 0.0f);
    if (!tensor_floats(weights.conv, conv.data()) || !tensor_floats(weights.a_log, a_log.data()) ||
        !tensor_floats(weights.dt_bias, dt_bias.data()) || !tensor_floats(weights.norm, norm.data()))
        return false;
    const linear_inputs input{
        batch,           tokens,      int(heads),        int(width),      int(kernel),
        threads,         1e-6f,       output.qkv.data(), output.z.data(), output.a.data(),
        output.b.data(), conv.data(), a_log.data(),      dt_bias.data(),  norm.data(),
        conv_history,    state,
    };
    if (!linear_core_cpu(input, output.core)) return false;
    projections.clear();
    if (!stored_project(backend, { weights.out }, output.core.values.data(), int(group), batch * tokens, threads,
                        projections))
        return false;
    output.attention = std::move(projections[0]);
    return true;
}
bool graph_run(ggml_backend_t backend, const decoder_weights & w, const attention_config & c, const float * input,
               const float * positions, int batch, int tokens, int threads, const decoder_cache * previous,
               decoder_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !input || !positions || !w.embedding || w.layers.empty() ||
        w.layers.size() > 64 || batch < 1 || batch > 4 || tokens < 1 || tokens > 4096 || threads < 1 || threads > 256 ||
        !std::isfinite(c.eps) || c.eps <= 0)
        return false;
    const int width = int(w.embedding->ne[0]), past = previous ? previous->tokens : 0;
    if (width < 1 || width > 8192 || size_t(batch) * tokens * width > 16 * 1024 * 1024 || past < 0 ||
        past + tokens > 4096 ||
        ((!w.head || w.head->ne[1] < 1 || w.head->ne[1] > 300000 || !shape(w.head, width, int(w.head->ne[1])))))
        return false;
    if (previous && (previous->batch != batch || previous->layers.size() != w.layers.size())) return false;
    const size_t count = size_t(batch) * tokens * width;
    std::vector<float> hidden(input, input + count), normalized;
    if (!std::all_of(hidden.begin(), hidden.end(), [](float x) { return std::isfinite(x); })) return false;
    output.cache.batch = batch;
    output.cache.tokens = past + tokens;
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const auto & layer = w.layers[i];
        const auto * cache = previous ? &previous->layers[i] : nullptr;
        if (!rms(hidden, layer.input_norm, width, c.eps, normalized)) return false;
        decoder_layer_cache next;
        std::vector<float> mixed;
        if (layer.full_attention) {
            attention_outputs attention;
            output.attention_inputs[int(i)] = normalized;
            if (cache && cache->attention.tokens != past) return false;
            if (!graph_attention(backend, layer.attention, c, normalized.data(), positions, batch, tokens, threads,
                                 cache ? &cache->attention : nullptr, attention))
                return false;
            output.attention_keys[int(i)] = attention.k_norm;
            mixed = std::move(attention.attention);
            next.attention = std::move(attention.cache);
        } else {
            projected_linear_outputs linear;
            if (cache &&
                (!layer.linear.a_log || !layer.linear.norm || !layer.linear.conv ||
                 cache->conv.size() != size_t(batch) * size_t(3 * layer.linear.a_log->ne[0] * layer.linear.norm->ne[0] *
                                                              layer.linear.conv->ne[0]) ||
                 cache->recurrent.size() !=
                     size_t(batch) * layer.linear.a_log->ne[0] * layer.linear.norm->ne[0] * layer.linear.norm->ne[0]))
                return false;
            if (!graph_linear(backend, layer.linear, normalized.data(), batch, tokens, threads,
                              cache ? cache->conv.data() : nullptr, cache ? cache->recurrent.data() : nullptr, linear))
                return false;
            mixed = std::move(linear.attention);
            next.conv = std::move(linear.core.conv_history);
            next.recurrent = std::move(linear.core.state);
        }
        if (mixed.size() != count) return false;
        for (size_t j = 0; j < count; ++j) hidden[j] += mixed[j];
        if (!rms(hidden, layer.post_norm, width, c.eps, normalized) || !layer.gate || layer.gate->ne[1] < 1 ||
            layer.gate->ne[1] > 32768 || !shape(layer.gate, width, int(layer.gate->ne[1])) ||
            !shape(layer.up, width, int(layer.gate->ne[1])) || !shape(layer.down, int(layer.gate->ne[1]), width))
            return false;
        std::vector<std::vector<float>> projections;
        if (!stored_project(backend, { layer.gate, layer.up }, normalized.data(), width, batch * tokens, threads,
                            projections))
            return false;
        for (size_t j = 0; j < projections[0].size(); ++j) {
            const float gate = projections[0][j];
            projections[0][j] = (gate / (1.0f + std::exp(-gate))) * projections[1][j];
        }
        const auto activated = std::move(projections[0]);
        projections.clear();
        if (!stored_project(backend, { layer.down }, activated.data(), int(layer.gate->ne[1]), batch * tokens, threads,
                            projections))
            return false;
        for (size_t j = 0; j < count; ++j) hidden[j] += projections[0][j];
        output.hidden.push_back(hidden);
        output.cache.layers.push_back(std::move(next));
    }
    if (!rms(hidden, w.norm, width, c.eps, output.norm)) return false;
    std::vector<float> last(size_t(batch) * width);
    for (int b = 0; b < batch; ++b)
        std::copy_n(output.norm.data() + size_t((b * tokens) + tokens - 1) * width, width,
                    last.data() + size_t(b) * width);
    std::vector<std::vector<float>> logits;
    if (!stored_project(backend, { w.head }, last.data(), width, batch, threads, logits)) return false;
    output.logits = std::move(logits[0]);
    return true;
}
} // namespace
struct graph_decoder::implementation {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ~implementation() {
        if (backend) ggml_backend_free(backend);
    }
};
graph_decoder::graph_decoder() = default;
graph_decoder::~graph_decoder() = default;
void graph_decoder::clear() {
    impl_.reset();
}
bool graph_decoder::project(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width, int tokens,
                            int threads, std::vector<std::vector<float>> & output) {
    output.clear();
    if (!enabled()) return false;
    try {
        if (!impl_) impl_ = std::make_unique<implementation>();
        return stored_project(impl_->backend, matrices, hidden, width, tokens, threads, output);
    } catch (...) {
        output.clear();
        return false;
    }
}
bool graph_decoder::run(const decoder_weights & w, const attention_config & c, const float * input,
                        const float * positions, int tokens, int threads, const decoder_cache * previous,
                        decoder_outputs & output) {
    output = {};
    // Deliberately narrow first profile: MiniCPM-OCR-1.0, batch1, equal-axis
    // 1-D RoPE. Ovis/Josev/other engines retain their original decoder.
    if (!enabled() || !shape(w.embedding, 1024, 248100) || !shape(w.norm, 1024, 1) || w.head != w.embedding ||
        w.layers.size() != 24 || c.query_heads != 8 || c.kv_heads != 2 || c.head_dim != 256 || c.rotary_dim != 64 ||
        c.theta != 10000000.0f || c.eps != 1e-6f || c.mrope_section[0] != 11 || c.mrope_section[1] != 11 ||
        c.mrope_section[2] != 10 || !positions || tokens < 1 || tokens > 4096)
        return false;
    const int past = previous ? previous->tokens : 0;
    if (past < 0 || past > 4096 - tokens) return false;
    for (int t = 0; t < tokens; ++t)
        for (int axis = 0; axis < 3; ++axis)
            if (positions[axis * tokens + t] != float(past + t)) return false;
    for (size_t i = 0; i < w.layers.size(); ++i)
        if (w.layers[i].full_attention != (i % 4 == 3)) return false;
    try {
        if (!impl_) impl_ = std::make_unique<implementation>();
        if (!impl_->backend) return false;
        decoder_outputs pending;
        if (!graph_run(impl_->backend, w, c, input, positions, 1, tokens, threads, previous, pending)) return false;
        output = std::move(pending);
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}
} // namespace qwen35_ocr
