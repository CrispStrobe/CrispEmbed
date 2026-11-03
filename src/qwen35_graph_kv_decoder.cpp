// qwen35_graph_kv_decoder.cpp - see qwen35_graph_kv_decoder.h. Default-off
// (CRISPEMBED_QWEN35_GRAPH_DECODER); the reference path is decoder_cpu.
//
// Arithmetic map against decoder_cpu / attention_cpu / linear_cpu / linear_core_cpu:
//   RMSNorm (zero-centered, (x*scale)*(1+w))  custom op calling rms_scale_cpu   (same scalar code)
//   q/k head norms                            same custom op                    (same scalar code)
//   RoPE (partial 64/256, rotate_half)        custom op, reference expression   (same scalar code)
//   attention softmax(QK^T*scale)V            ggml mul_mat + soft_max_ext       (reordered sums)
//   attention output gate value/(1+exp(-g))   custom op                         (same scalar code)
//   depthwise causal conv (kernel 4)          ggml_ssm_conv                     (same tap order, F32 sum)
//   conv SiLU x*sigmoid(x), softplus, beta    custom ops                        (same scalar code)
//   q/k l2norm, gated delta rule              same ggml ops as delta_cpu        (same ops)
//   gated RMSNorm (double square sum)         custom op                         (same scalar code)
//   SwiGLU (g/(1+exp(-g)))*u                  custom op                         (same scalar code)
//   matrices                                  see CRISPEMBED_QWEN35_GRAPH_MATMUL in the header
#include "qwen35_graph_kv_decoder.h"

#include "qwen35_math.h"
#include "core/env_gate.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace qwen35_ocr {
namespace {

enum class matmul_mode { exact, native, f32 };

// Matrices the "f32" mode copies once at init (non-F32 only).
std::vector<ggml_tensor *> f32_copy_sources(const decoder_weights & weights) {
    std::vector<ggml_tensor *> sources;
    auto add = [&](ggml_tensor * t) {
        if (t && t->type != GGML_TYPE_F32) sources.push_back(t);
    };
    for (const auto & l : weights.layers) {
        add(l.gate);
        add(l.up);
        add(l.down);
        if (l.full_attention) {
            add(l.attention.q);
            add(l.attention.k);
            add(l.attention.v);
            add(l.attention.out);
        } else {
            add(l.linear.qkv);
            add(l.linear.z);
            add(l.linear.a);
            add(l.linear.b);
            add(l.linear.out);
        }
    }
    add(weights.head);
    return sources;
}

matmul_mode mode_from_env() {
    const char * v = std::getenv("CRISPEMBED_QWEN35_GRAPH_MATMUL");
    // Default "f32" since the 2026-11-02 MiniCPM default flip (the only caller); "exact"
    // and "native" stay selectable.
    if (v && std::strcmp(v, "native") == 0) return matmul_mode::native;
    if (v && std::strcmp(v, "exact") == 0) return matmul_mode::exact;
    return matmul_mode::f32;
}

bool is_matrix(const ggml_tensor * t, int64_t columns, int64_t rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}

// Row partition of a contiguous [ne0, rows] view for custom ops.
void row_range(int64_t rows, int ith, int nth, int64_t & r0, int64_t & r1) {
    r0 = rows * ith / nth;
    r1 = rows * (ith + 1) / nth;
}
int64_t row_count(const ggml_tensor * t) {
    return ggml_nelements(t) / t->ne[0];
}

float sigmoid_ref(float x) { // = qwen35_delta.cpp sigmoid()
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

struct norm_param {
    std::vector<float> weights; // raw checkpoint values; (1 + w) is formed per element as in rms()
    float eps = 1e-6f;
};
struct linear_param {
    std::vector<float> a_log, dt_bias, norm;
    float norm_eps = 1e-6f;
};
struct rope_param {
    attention_config config;
    int past = 0;
};

// y = (x * rms_scale_cpu(x)) * (1 + w), per row of ne0 (= qwen35_decoder.cpp rms / attention norm_head).
void op_rms(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * user) {
    const auto * p = static_cast<const norm_param *>(user);
    const int width = int(a->ne[0]);
    int64_t r0, r1;
    row_range(row_count(a), ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        const float * x = static_cast<const float *>(a->data) + r * width;
        float * y = static_cast<float *>(dst->data) + r * width;
        const float scale = rms_scale_cpu(x, width, p->eps);
        for (int i = 0; i < width; ++i) y[i] = (x[i] * scale) * (1.0f + p->weights[size_t(i)]);
    }
}

// Partial NeoX RoPE on [head_dim, heads, tokens] (= qwen35_attention.cpp rotate with equal axes).
void op_rope(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * user) {
    const auto * p = static_cast<const rope_param *>(user);
    const auto & c = p->config;
    const int d = int(a->ne[0]), heads = int(a->ne[1]), half = c.rotary_dim / 2;
    int64_t r0, r1;
    row_range(row_count(a), ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        const float * x = static_cast<const float *>(a->data) + r * d;
        float * y = static_cast<float *>(dst->data) + r * d;
        std::memcpy(y, x, size_t(d) * sizeof(float));
        const float position = float(p->past + int(r / heads));
        for (int i = 0; i < half; ++i) {
            const float frequency = 1.0f / std::pow(c.theta, float(2 * i) / c.rotary_dim);
            const float angle = frequency * position;
            const float cosine = std::cos(angle), sine = std::sin(angle);
            const float left = y[i], right = y[i + half];
            y[i] = left * cosine - right * sine;
            y[i + half] = right * cosine + left * sine;
        }
    }
}

// value / (1 + exp(-gate)) (= attention_cpu pooled output gate).
void op_gate(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void *) {
    int64_t r0, r1;
    const int64_t n = ggml_nelements(a);
    row_range(n, ith, nth, r0, r1);
    const float * v = static_cast<const float *>(a->data);
    const float * g = static_cast<const float *>(b->data);
    float * y = static_cast<float *>(dst->data);
    for (int64_t i = r0; i < r1; ++i) y[i] = v[i] / (1.0f + std::exp(-g[i]));
}

// (g / (1 + exp(-g))) * u (= decoder_impl SwiGLU).
void op_swiglu(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void *) {
    int64_t r0, r1;
    row_range(ggml_nelements(a), ith, nth, r0, r1);
    const float * g = static_cast<const float *>(a->data);
    const float * u = static_cast<const float *>(b->data);
    float * y = static_cast<float *>(dst->data);
    for (int64_t i = r0; i < r1; ++i) y[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
}

// sum * sigmoid(sum) after the depthwise conv (= linear_core_cpu).
void op_conv_silu(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    int64_t r0, r1;
    row_range(ggml_nelements(a), ith, nth, r0, r1);
    const float * x = static_cast<const float *>(a->data);
    float * y = static_cast<float *>(dst->data);
    for (int64_t i = r0; i < r1; ++i) y[i] = x[i] * sigmoid_ref(x[i]);
}

// g = -exp(A_log) * softplus(a + dt_bias) on [heads, tokens] (= linear_core_cpu).
void op_decay(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * user) {
    const auto * p = static_cast<const linear_param *>(user);
    const int heads = int(a->ne[0]);
    int64_t r0, r1;
    row_range(row_count(a), ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r)
        for (int h = 0; h < heads; ++h) {
            const float time = static_cast<const float *>(a->data)[r * heads + h] + p->dt_bias[size_t(h)];
            const float softplus = time > 20.0f ? time : std::log1p(std::exp(time));
            static_cast<float *>(dst->data)[r * heads + h] = -std::exp(p->a_log[size_t(h)]) * softplus;
        }
}

void op_beta(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    int64_t r0, r1;
    row_range(ggml_nelements(a), ith, nth, r0, r1);
    for (int64_t i = r0; i < r1; ++i)
        static_cast<float *>(dst->data)[i] = sigmoid_ref(static_cast<const float *>(a->data)[i]);
}

// Gated RMSNorm per [width] row: delta * scale * w * z * sigmoid(z), double square sum (= linear_core_cpu).
void op_gated_norm(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * user) {
    const auto * p = static_cast<const linear_param *>(user);
    const size_t width = size_t(a->ne[0]);
    int64_t r0, r1;
    row_range(row_count(a), ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        const float * delta = static_cast<const float *>(a->data) + size_t(r) * width;
        const float * z = static_cast<const float *>(b->data) + size_t(r) * width;
        float * y = static_cast<float *>(dst->data) + size_t(r) * width;
        double square = 0;
        for (size_t column = 0; column < width; ++column) square += double(delta[column]) * delta[column];
        const float scale = 1.0f / std::sqrt(float(square / width) + p->norm_eps);
        for (size_t column = 0; column < width; ++column)
            y[column] = delta[column] * scale * p->norm[column] * z[column] * sigmoid_ref(z[column]);
    }
}

// Decode-size product with F32 activations: dequantize one stored weight row at a
// time (F16/quantized -> F32) and use the F32 dot. No whole-matrix conversion.
void op_rowdot(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t columns = w->ne[0], rows = w->ne[1], tokens = x->ne[1];
    const auto to_float = ggml_get_type_traits(w->type)->to_float;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
    std::vector<float> row(static_cast<size_t>(columns));
    int64_t r0, r1;
    row_range(rows, ith, nth, r0, r1);
    for (int64_t r = r0; r < r1; ++r) {
        const auto * source = static_cast<const uint8_t *>(w->data) + r * w->nb[1];
        const float * values = reinterpret_cast<const float *>(source);
        if (w->type != GGML_TYPE_F32) {
            to_float(source, row.data(), columns);
            values = row.data();
        }
        for (int64_t t = 0; t < tokens; ++t) {
            float result = 0;
            dot(int(columns), &result, 0, values, 0, static_cast<const uint8_t *>(x->data) + t * x->nb[1], 0, 1);
            *reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + r * dst->nb[0] + t * dst->nb[1]) = result;
        }
    }
}

struct layer_state {
    // full attention
    ggml_tensor *keys = nullptr, *values = nullptr;
    // delta-net
    ggml_tensor *conv_state = nullptr, *recurrent = nullptr, *conv_weight = nullptr;
};

} // namespace

struct graph_kv_decoder::impl {
    decoder_weights w;
    attention_config c;
    ggml_backend_t backend = nullptr;
    matmul_mode mode = matmul_mode::exact;
    int width = 0, ffn = 0, capacity = 0, past = 0;
    int lin_heads = 0, lin_width = 0, lin_kernel = 0;
    ggml_context * state_ctx = nullptr;
    ggml_backend_buffer_t state_buf = nullptr;
    ggml_context * const_ctx = nullptr;
    ggml_backend_buffer_t const_buf = nullptr;
    ggml_context * f32_ctx = nullptr;
    ggml_backend_buffer_t f32_buf = nullptr;
    std::map<const ggml_tensor *, ggml_tensor *> f32_copy;
    ggml_tensor * l2_eps = nullptr;
    std::vector<layer_state> states;
    // Stable addresses: sized once in init(), never resized afterwards.
    std::vector<norm_param> input_norms, post_norms, q_norms, k_norms;
    std::vector<linear_param> linears;
    norm_param final_norm;
    rope_param rope;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> meta;

    ~impl() {
        if (allocr) ggml_gallocr_free(allocr);
        if (state_buf) ggml_backend_buffer_free(state_buf);
        if (state_ctx) ggml_free(state_ctx);
        if (const_buf) ggml_backend_buffer_free(const_buf);
        if (const_ctx) ggml_free(const_ctx);
        if (f32_buf) ggml_backend_buffer_free(f32_buf);
        if (f32_ctx) ggml_free(f32_ctx);
    }

    ggml_tensor * matrix(ggml_tensor * t) const {
        const auto found = f32_copy.find(t);
        return found == f32_copy.end() ? t : found->second;
    }
    ggml_tensor * mm(ggml_context * g, ggml_tensor * weight, ggml_tensor * x) const {
        weight = matrix(weight);
        ggml_tensor * y;
        if (mode == matmul_mode::native || weight->type == GGML_TYPE_F32) {
            y = ggml_mul_mat(g, weight, x);
        } else if (x->ne[1] <= 8) {
            ggml_tensor * args[] = { weight, x };
            y = ggml_custom_4d(g, GGML_TYPE_F32, weight->ne[1], x->ne[1], 1, 1, args, 2, op_rowdot, GGML_N_TASKS_MAX,
                               nullptr);
        } else {
            y = ggml_mul_mat(g, ggml_cast(g, weight, GGML_TYPE_F32), x);
        }
        if (y->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
};

graph_kv_decoder::graph_kv_decoder() : p_(new impl) {}
graph_kv_decoder::~graph_kv_decoder() = default;

bool graph_kv_decoder::enabled() {
    return !core_env::explicitly_off("CRISPEMBED_QWEN35_GRAPH_DECODER");
}

uint64_t graph_kv_decoder::f32_copy_bytes(const decoder_weights & weights) {
    if (mode_from_env() != matmul_mode::f32) return 0;
    uint64_t bytes = 0;
    for (const auto * t : f32_copy_sources(weights)) bytes += uint64_t(ggml_nelements(t)) * sizeof(float);
    return bytes;
}

const char * graph_kv_decoder::mode() const {
    switch (p_->mode) {
    case matmul_mode::native:
        return "native";
    case matmul_mode::f32:
        return "f32";
    default:
        return "exact";
    }
}

int graph_kv_decoder::tokens() const {
    return p_->past;
}

void graph_kv_decoder::reset() {
    p_->past = 0;
    if (p_->state_buf) ggml_backend_buffer_clear(p_->state_buf, 0);
}

bool graph_kv_decoder::init(const decoder_weights & weights, const attention_config & config, ggml_backend_t backend,
                            int capacity, bool allow_f32_copies) {
    if (!enabled() || !backend || !ggml_backend_is_cpu(backend) || !weights.embedding || !weights.head ||
        !weights.norm || weights.layers.empty() || weights.layers.size() > 64 || capacity < 1 || capacity > 32768)
        return false;
    p_.reset(new impl);
    auto & p = *p_;
    p.w = weights;
    p.c = config;
    p.backend = backend;
    p.capacity = capacity;
    p.mode = mode_from_env();
    if (p.mode == matmul_mode::f32 && !allow_f32_copies) p.mode = matmul_mode::exact;
    p.width = int(weights.embedding->ne[0]);
    const auto & c = config;
    const int d = c.head_dim, qh = c.query_heads, kh = c.kv_heads, width = p.width;
    if (width < 1 || width > 8192 || qh < 1 || kh < 1 || qh % kh || d < 2 || c.rotary_dim < 2 || c.rotary_dim > d ||
        c.rotary_dim % 2 || !is_matrix(weights.head, width, weights.head->ne[1]) || !is_matrix(weights.norm, width, 1))
        return false;
    for (const auto & l : weights.layers) {
        if (!l.gate || !is_matrix(l.input_norm, width, 1) || !is_matrix(l.post_norm, width, 1)) return false;
        const int64_t f = l.gate->ne[1];
        if (!is_matrix(l.gate, width, f) || !is_matrix(l.up, width, f) || !is_matrix(l.down, f, width)) return false;
        p.ffn = int(f);
        if (l.full_attention) {
            const auto & a = l.attention;
            if (!is_matrix(a.q, width, 2 * qh * d) || !is_matrix(a.k, width, kh * d) ||
                !is_matrix(a.v, width, kh * d) || !is_matrix(a.out, qh * d, width) || !is_matrix(a.q_norm, d, 1) ||
                !is_matrix(a.k_norm, d, 1))
                return false;
        } else {
            const auto & x = l.linear;
            if (!x.a_log || !x.norm || !x.conv) return false;
            const int64_t heads = x.a_log->ne[0], hw = x.norm->ne[0], kernel = x.conv->ne[0], group = heads * hw;
            if (heads < 1 || heads > 128 || hw < 1 || hw > 256 || kernel < 2 || kernel > 16 ||
                !is_matrix(x.a_log, heads, 1) || !is_matrix(x.dt_bias, heads, 1) || !is_matrix(x.norm, hw, 1) ||
                !is_matrix(x.qkv, width, 3 * group) || !is_matrix(x.z, width, group) || !is_matrix(x.a, width, heads) ||
                !is_matrix(x.b, width, heads) || !is_matrix(x.out, group, width) || x.conv->ne[1] != 1 ||
                x.conv->ne[2] != 3 * group || x.conv->ne[3] != 1)
                return false;
            p.lin_heads = int(heads);
            p.lin_width = int(hw);
            p.lin_kernel = int(kernel);
        }
    }
    const size_t n_layers = weights.layers.size();
    auto floats = [](ggml_tensor * t, std::vector<float> & out) {
        out.assign(size_t(ggml_nelements(t)), 0.0f);
        return tensor_floats(t, out.data());
    };
    p.input_norms.resize(n_layers);
    p.post_norms.resize(n_layers);
    p.q_norms.resize(n_layers);
    p.k_norms.resize(n_layers);
    p.linears.resize(n_layers);
    p.states.resize(n_layers);
    p.final_norm.eps = c.eps;
    if (!floats(weights.norm, p.final_norm.weights)) return false;
    const size_t tensors = 4 * n_layers + 8;
    p.state_ctx = ggml_init({ tensors * ggml_tensor_overhead(), nullptr, true });
    p.const_ctx = ggml_init({ tensors * ggml_tensor_overhead(), nullptr, true });
    if (!p.state_ctx || !p.const_ctx) return false;
    p.l2_eps = ggml_new_tensor_1d(p.const_ctx, GGML_TYPE_F32, 1);
    for (size_t i = 0; i < n_layers; ++i) {
        const auto & l = weights.layers[i];
        p.input_norms[i].eps = p.post_norms[i].eps = c.eps;
        if (!floats(l.input_norm, p.input_norms[i].weights) || !floats(l.post_norm, p.post_norms[i].weights))
            return false;
        auto & s = p.states[i];
        if (l.full_attention) {
            p.q_norms[i].eps = p.k_norms[i].eps = c.eps;
            if (!floats(l.attention.q_norm, p.q_norms[i].weights) || !floats(l.attention.k_norm, p.k_norms[i].weights))
                return false;
            s.keys = ggml_new_tensor_3d(p.state_ctx, GGML_TYPE_F32, d, capacity, kh);
            s.values = ggml_new_tensor_3d(p.state_ctx, GGML_TYPE_F32, d, capacity, kh);
        } else {
            auto & x = p.linears[i];
            if (!floats(l.linear.a_log, x.a_log) || !floats(l.linear.dt_bias, x.dt_bias) ||
                !floats(l.linear.norm, x.norm))
                return false;
            const int64_t channels = 3 * int64_t(p.lin_heads) * p.lin_width;
            s.conv_state = ggml_new_tensor_2d(p.state_ctx, GGML_TYPE_F32, p.lin_kernel - 1, channels);
            s.recurrent = ggml_new_tensor_4d(p.state_ctx, GGML_TYPE_F32, p.lin_width, p.lin_width, p.lin_heads, 1);
            s.conv_weight = ggml_new_tensor_2d(p.const_ctx, GGML_TYPE_F32, p.lin_kernel, channels);
        }
    }
    p.state_buf = ggml_backend_alloc_ctx_tensors(p.state_ctx, backend);
    p.const_buf = ggml_backend_alloc_ctx_tensors(p.const_ctx, backend);
    if (!p.state_buf || !p.const_buf) return false;
    reset();
    const float l2_eps = 1e-6f; // delta_cpu: official l2norm rsqrt(sum(x*x) + 1e-6)
    ggml_backend_tensor_set(p.l2_eps, &l2_eps, 0, sizeof(l2_eps));
    for (size_t i = 0; i < n_layers; ++i) {
        if (weights.layers[i].full_attention) continue;
        std::vector<float> conv; // [kernel, 1, channels] -> [kernel, channels], F32 as linear_cpu
        if (!floats(weights.layers[i].linear.conv, conv)) return false;
        ggml_backend_tensor_set(p.states[i].conv_weight, conv.data(), 0, conv.size() * sizeof(float));
    }
    if (p.mode == matmul_mode::f32) {
        const std::vector<ggml_tensor *> sources = f32_copy_sources(weights);
        p.f32_ctx = ggml_init({ (sources.size() + 1) * ggml_tensor_overhead(), nullptr, true });
        if (!p.f32_ctx) return false;
        for (auto * t : sources) p.f32_copy[t] = ggml_new_tensor_2d(p.f32_ctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
        p.f32_buf = ggml_backend_alloc_ctx_tensors(p.f32_ctx, backend);
        if (!p.f32_buf) return false;
        for (auto & entry : p.f32_copy) {
            const auto * source = entry.first;
            auto * target = entry.second;
            const auto to_float = ggml_get_type_traits(source->type)->to_float;
            if (!to_float || !source->data || !target->data) return false;
            for (int64_t r = 0; r < source->ne[1]; ++r)
                to_float(static_cast<const uint8_t *>(source->data) + r * source->nb[1],
                         static_cast<float *>(target->data) + r * source->ne[0], source->ne[0]);
        }
    }
    p.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return p.allocr != nullptr;
}

bool graph_kv_decoder::forward(const float * hidden, int tokens, int threads, std::vector<float> & logits,
                               std::vector<std::vector<float>> * layer_capture, std::vector<float> * norm_capture) {
    logits.clear();
    auto & p = *p_;
    if (!enabled() || !p.allocr || !hidden || tokens < 1 || threads < 1 || threads > 256 ||
        p.past + tokens > p.capacity)
        return false;
    const auto & c = p.c;
    const int T = tokens, D = p.width, d = c.head_dim, qh = c.query_heads, kh = c.kv_heads;
    const int Lk = p.past + T, H = p.lin_heads, W = p.lin_width, K = p.lin_kernel, group = H * W;
    const int64_t vocab = p.w.head->ne[1];
    if (!std::all_of(hidden, hidden + size_t(T) * D, [](float x) { return std::isfinite(x); })) return false;
    static constexpr int kNodes = 16384;
    p.meta.assign(ggml_tensor_overhead() * kNodes + ggml_graph_overhead_custom(kNodes, false), 0);
    ggml_context * g = ggml_init({ p.meta.size(), p.meta.data(), true });
    if (!g) return false;
    struct ctx_guard {
        ggml_context * g;
        ~ctx_guard() { ggml_free(g); }
    } guard{ g };
    ggml_cgraph * gf = ggml_new_graph_custom(g, kNodes, false);
    p.rope.config = c;
    p.rope.past = p.past;

    ggml_tensor * input = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_input(input);
    ggml_tensor * x = input;
    ggml_tensor * mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, Lk, T);
        ggml_set_input(mask);
    }
    auto rms = [&](ggml_tensor * t, norm_param & np) {
        return ggml_map_custom1(g, ggml_is_contiguous(t) ? t : ggml_cont(g, t), op_rms, GGML_N_TASKS_MAX, &np);
    };
    const float scale = 1.0f / std::sqrt(float(d));
    std::vector<ggml_tensor *> captured;
    for (size_t li = 0; li < p.w.layers.size(); ++li) {
        const auto & l = p.w.layers[li];
        auto & s = p.states[li];
        ggml_tensor * xn = rms(x, p.input_norms[li]);
        ggml_tensor * mixed;
        if (l.full_attention) {
            const auto & a = l.attention;
            ggml_tensor * qfull = p.mm(g, a.q, xn); // [2d*qh, T]: per head [query d | gate d]
            const size_t fs = ggml_element_size(qfull);
            ggml_tensor * query = ggml_view_3d(g, qfull, d, qh, T, 2 * d * fs, qfull->nb[1], 0);
            ggml_tensor * gate = ggml_cont(g, ggml_view_3d(g, qfull, d, qh, T, 2 * d * fs, qfull->nb[1], d * fs));
            ggml_tensor * qn = rms(query, p.q_norms[li]);
            ggml_tensor * kn = ggml_map_custom1(g, ggml_reshape_3d(g, p.mm(g, a.k, xn), d, kh, T), op_rms,
                                                GGML_N_TASKS_MAX, &p.k_norms[li]);
            ggml_tensor * qr = ggml_map_custom1(g, qn, op_rope, GGML_N_TASKS_MAX, &p.rope); // [d, qh, T]
            ggml_tensor * kr = ggml_map_custom1(g, kn, op_rope, GGML_N_TASKS_MAX, &p.rope); // [d, kh, T]
            ggml_tensor * v = ggml_reshape_3d(g, p.mm(g, a.v, xn), d, kh, T);
            const size_t row = s.keys->nb[1], plane = s.keys->nb[2];
            ggml_tensor * kdst = ggml_view_3d(g, s.keys, d, T, kh, row, plane, size_t(p.past) * row);
            ggml_tensor * vdst = ggml_view_3d(g, s.values, d, T, kh, row, plane, size_t(p.past) * row);
            ggml_build_forward_expand(gf, ggml_cpy(g, ggml_permute(g, kr, 0, 2, 1, 3), kdst));
            ggml_build_forward_expand(gf, ggml_cpy(g, ggml_permute(g, v, 0, 2, 1, 3), vdst));
            ggml_tensor * kall = ggml_view_3d(g, s.keys, d, Lk, kh, row, plane, 0);
            ggml_tensor * vall = ggml_view_3d(g, s.values, d, Lk, kh, row, plane, 0);
            ggml_tensor * q = ggml_cont(g, ggml_permute(g, qr, 0, 2, 1, 3)); // [d, T, qh]
            ggml_tensor * kq = ggml_mul_mat(g, kall, q);                     // [Lk, T, qh]; head h -> kv h/(qh/kh)
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(g, kq, mask, scale, 0.0f);
            ggml_tensor * vt = ggml_cont(g, ggml_permute(g, vall, 1, 0, 2, 3)); // [Lk, d, kh]
            ggml_tensor * o = ggml_mul_mat(g, vt, kq);                          // [d, T, qh]
            ggml_mul_mat_set_prec(o, GGML_PREC_F32);
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3)); // [d, qh, T]
            ggml_tensor * pooled = ggml_map_custom2(g, o, gate, op_gate, GGML_N_TASKS_MAX, nullptr);
            mixed = p.mm(g, a.out, ggml_reshape_2d(g, pooled, qh * d, T));
        } else {
            const auto & lw = l.linear;
            auto & lp = p.linears[li];
            ggml_tensor * qkv = p.mm(g, lw.qkv, xn); // [3*group, T]
            ggml_tensor * z = p.mm(g, lw.z, xn);     // [group, T]
            ggml_tensor * av = p.mm(g, lw.a, xn);    // [H, T]
            ggml_tensor * bv = p.mm(g, lw.b, xn);    // [H, T]
            const int64_t channels = 3 * int64_t(group);
            ggml_tensor * sx = ggml_concat(g, s.conv_state, ggml_cont(g, ggml_transpose(g, qkv)), 0); // [K-1+T, ch]
            ggml_build_forward_expand(
                gf, ggml_cpy(g, ggml_view_2d(g, sx, K - 1, channels, sx->nb[1], size_t(T) * ggml_element_size(sx)),
                             s.conv_state));
            ggml_tensor * conv = ggml_ssm_conv(g, ggml_reshape_3d(g, sx, K - 1 + T, channels, 1), s.conv_weight);
            conv = ggml_map_custom1(g, ggml_reshape_2d(g, conv, channels, T), op_conv_silu, GGML_N_TASKS_MAX, nullptr);
            const size_t cs = ggml_element_size(conv);
            auto part = [&](int index) {
                return ggml_cont(g, ggml_view_4d(g, conv, W, H, T, 1, size_t(W) * cs, conv->nb[1], conv->nb[1] * T,
                                                 size_t(index) * group * cs));
            };
            auto l2 = [&](ggml_tensor * t) { // = delta_cpu normalize(); scalar eps broadcast (add1 is deprecated)
                return ggml_div(g, t, ggml_sqrt(g, ggml_add(g, ggml_sum_rows(g, ggml_sqr(g, t)), p.l2_eps)));
            };
            ggml_tensor * query = l2(part(0));
            ggml_tensor * key = l2(part(1));
            ggml_tensor * value = part(2);
            ggml_tensor * decay =
                ggml_reshape_4d(g, ggml_map_custom1(g, av, op_decay, GGML_N_TASKS_MAX, &lp), 1, H, T, 1);
            ggml_tensor * beta =
                ggml_reshape_4d(g, ggml_map_custom1(g, bv, op_beta, GGML_N_TASKS_MAX, nullptr), 1, H, T, 1);
            ggml_tensor * delta = ggml_gated_delta_net(g, query, key, value, decay, beta, s.recurrent, 1);
            const size_t ds = ggml_element_size(delta);
            ggml_tensor * core = ggml_view_2d(g, delta, W, int64_t(H) * T, size_t(W) * ds, 0);
            ggml_tensor * state = ggml_view_1d(g, delta, int64_t(W) * W * H, size_t(T) * group * ds);
            ggml_build_forward_expand(gf, ggml_cpy(g, state, ggml_reshape_1d(g, s.recurrent, int64_t(W) * W * H)));
            ggml_tensor * normed = ggml_map_custom2(g, ggml_cont(g, core), ggml_reshape_2d(g, z, W, int64_t(H) * T),
                                                    op_gated_norm, GGML_N_TASKS_MAX, &lp);
            mixed = p.mm(g, lw.out, ggml_reshape_2d(g, normed, group, T));
        }
        x = ggml_add(g, x, mixed);
        ggml_tensor * hn = rms(x, p.post_norms[li]);
        ggml_tensor * act =
            ggml_map_custom2(g, p.mm(g, l.gate, hn), p.mm(g, l.up, hn), op_swiglu, GGML_N_TASKS_MAX, nullptr);
        x = ggml_add(g, x, p.mm(g, l.down, act));
        if (layer_capture) {
            ggml_set_output(x);
            captured.push_back(x);
        }
    }
    ggml_tensor * normed_all = nullptr;
    ggml_tensor * last;
    if (norm_capture) {
        normed_all = rms(x, p.final_norm);
        ggml_set_output(normed_all);
        last = ggml_view_2d(g, normed_all, D, 1, normed_all->nb[1], size_t(T - 1) * normed_all->nb[1]);
        last = ggml_cont(g, last);
    } else {
        last = rms(ggml_view_2d(g, x, D, 1, x->nb[1], size_t(T - 1) * x->nb[1]), p.final_norm);
    }
    ggml_tensor * out = p.mm(g, p.w.head, last);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    if (normed_all) ggml_build_forward_expand(gf, normed_all);
    for (auto * t : captured) ggml_build_forward_expand(gf, t);
    if (!ggml_gallocr_alloc_graph(p.allocr, gf)) return false;
    ggml_backend_tensor_set(input, hidden, 0, size_t(T) * D * sizeof(float));
    if (mask) {
        std::vector<float> m(size_t(Lk) * T, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int k = p.past + t + 1; k < Lk; ++k) m[size_t(t) * Lk + k] = -INFINITY;
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    ggml_backend_cpu_set_n_threads(p.backend, threads);
    if (ggml_backend_graph_compute(p.backend, gf) != GGML_STATUS_SUCCESS) return false;
    logits.resize(size_t(vocab));
    ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));
    if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) {
        logits.clear();
        return false;
    }
    if (layer_capture) {
        layer_capture->assign(captured.size(), {});
        for (size_t i = 0; i < captured.size(); ++i) {
            (*layer_capture)[i].resize(size_t(ggml_nelements(captured[i])));
            ggml_backend_tensor_get(captured[i], (*layer_capture)[i].data(), 0,
                                    (*layer_capture)[i].size() * sizeof(float));
        }
    }
    if (norm_capture) {
        norm_capture->resize(size_t(ggml_nelements(normed_all)));
        ggml_backend_tensor_get(normed_all, norm_capture->data(), 0, norm_capture->size() * sizeof(float));
    }
    p.past += T;
    return true;
}
} // namespace qwen35_ocr
