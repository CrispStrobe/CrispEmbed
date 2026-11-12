// qwen35_ocr_gpu.cpp - see qwen35_ocr_gpu.h. Default-off gate CRISPEMBED_QWEN35_OCR_GPU.
//
// Standard ggml ops only; every operation is cited against the pinned upstream Python,
// transformers 5.5.4 models/qwen3_5/modeling_qwen3_5.py ("MQ:<line>"), and against the
// qualified CPU reference it replaces (the nine CPU-only custom ops of
// qwen35_graph_kv_decoder.cpp and the scalar loops of qwen35_vision.cpp):
//
// decoder (one graph per call, persistent KV + delta-net state on the backend)
//   Qwen3_5RMSNorm x*rsqrt(mean(x^2)+eps)*(1+w)  MQ:711-729   rms_norm + mul (1+w)        (was op_rms)
//   q/k head norms (same class on head_dim)      MQ:644-645   rms_norm + mul (1+w)        (was op_rms)
//   interleaved multimodal RoPE, partial 64/256  MQ:85-173   rope_multi IMROPE, n_dims 64 (was op_rope)
//   attention output gate o*sigmoid(gate)        MQ:689       sigmoid + mul               (was op_gate)
//   causal conv1d + SiLU                         MQ:474       ssm_conv + silu             (was op_conv_silu)
//   q/k l2norm x*rsqrt(sum(x^2)+1e-6)            MQ:228-231   sqr/sum_rows/add/sqrt/div   (unchanged)
//   g = -exp(A_log)*softplus(a+dt_bias)          MQ:493       add + softplus + mul        (was op_decay)
//   beta = sigmoid(b)                            MQ:491       sigmoid                     (was op_beta)
//   gated delta rule                             MQ:234-354   gated_delta_net             (unchanged)
//   Qwen3_5RMSNormGated norm(x)*w*silu(z)        MQ:175-191   rms_norm + mul + silu + mul (was op_gated_norm)
//   MLP down(silu(gate(x))*up(x))                MQ:695-709   swiglu_split                (was op_swiglu)
//   matrices                                     -            mul_mat on the stored type, F32 precision
//                                                             (was op_rowdot in "exact" mode)
// vision (one graph per image)
//   patch embed (Conv3d == linear on flattened patches)  MQ:828-846
//   pre-LN block x+attn(norm1(x)); x+mlp(norm2(x))      MQ:961-993   norm + mul + add
//   qkv -> (q,k,v), vision RoPE q*cos+rotate_half(q)*sin MQ:864-875, 536-540, 878-958
//     cos/sin and the interpolated position embedding are host inputs computed by
//     vision_host_inputs (the same code vision_cpu runs)
//   non-causal attention per frame (cu_seqlens)          MQ:930-952   block-diagonal mask
//   MLP gelu_pytorch_tanh 0.5x(1+tanh(c(x+0.044715x^3))) MQ:815-826   explicit tanh formula
//   patch merger LN -> view(-1, 4*768) -> fc1 -> GELU(erf) -> fc2  MQ:848-862   gelu_erf
#include "qwen35_ocr_gpu.h"

#include "core/env_gate.h"
#include "core/gpu_backend_pref.h"
#include "qwen35_math.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>

namespace qwen35_ocr {
namespace {
constexpr int kNodes = 16384;

bool is_matrix(const ggml_tensor * t, int64_t columns, int64_t rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}

struct dec_layer {
    ggml_tensor *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *neg_exp_alog = nullptr, *dt_bias = nullptr, *gated_norm = nullptr, *conv = nullptr;
    ggml_tensor *keys = nullptr, *values = nullptr, *conv_state = nullptr, *recurrent = nullptr;
};
struct vis_layer {
    ggml_tensor *n1 = nullptr, *n1b = nullptr, *n2 = nullptr, *n2b = nullptr;
    ggml_tensor *qkv_b = nullptr, *out_b = nullptr, *up_b = nullptr, *down_b = nullptr;
};
} // namespace

struct ggml_pipeline::impl {
    decoder_weights dw;
    vision_weights vw;
    attention_config ac;
    vision_config vc;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool gpu = false;
    int threads = 1, capacity = 0, past = 0, width = 0, H = 0, W = 0, K = 0;
    ggml_context *weight_ctx = nullptr, *const_ctx = nullptr, *state_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr, const_buf = nullptr, state_buf = nullptr;
    std::map<const ggml_tensor *, ggml_tensor *> device;
    std::vector<dec_layer> layers;
    std::vector<vis_layer> vlayers;
    ggml_tensor *final_norm = nullptr, *l2_eps = nullptr;
    ggml_tensor *patch_b = nullptr, *merger_n = nullptr, *merger_nb = nullptr, *merger_up_b = nullptr,
                *merger_down_b = nullptr;
    std::vector<uint8_t> meta;
    std::set<std::string> logged;

    ~impl() {
        if (sched) ggml_backend_sched_free(sched);
        if (state_buf) ggml_backend_buffer_free(state_buf);
        if (const_buf) ggml_backend_buffer_free(const_buf);
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (state_ctx) ggml_free(state_ctx);
        if (const_ctx) ggml_free(const_ctx);
        if (weight_ctx) ggml_free(weight_ctx);
        if (cpu && cpu != backend) ggml_backend_free(cpu);
        if (backend) ggml_backend_free(backend);
    }
    ggml_tensor * dev(const ggml_tensor * t) const {
        const auto found = device.find(t);
        return found == device.end() ? const_cast<ggml_tensor *>(t) : found->second;
    }
    ggml_tensor * mm(ggml_context * g, const ggml_tensor * weight, ggml_tensor * x) const {
        ggml_tensor * y = ggml_mul_mat(g, dev(weight), x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    }
    void set_threads() const {
        if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, threads);
        if (cpu && cpu != backend) ggml_backend_cpu_set_n_threads(cpu, threads);
    }
    // Places the graph, reports placement once per graph kind (CRISPEMBED_QWEN35_OCR_GPU_LOG),
    // and fails if any node had to leave the primary backend while a GPU is in use and
    // CRISPEMBED_QWEN35_OCR_GPU_STRICT is set.
    bool place(ggml_cgraph * gf, const char * kind) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) return false;
        if (!logged.count(kind) &&
            (core_env::on("CRISPEMBED_QWEN35_OCR_GPU_LOG") || core_env::on("CRISPEMBED_QWEN35_OCR_GPU_STRICT"))) {
            logged.insert(kind);
            const int n = ggml_graph_n_nodes(gf);
            int primary = 0, other = 0;
            std::map<std::string, int> fallback;
            for (int i = 0; i < n; ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, node);
                const bool view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                                  node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
                if (!b && view) continue; // no compute; lives in its source's buffer
                if (b == backend) {
                    ++primary;
                } else {
                    ++other;
                    ++fallback[ggml_op_desc(node)];
                }
            }
            std::string ops;
            for (const auto & e : fallback) ops += " " + e.first + "x" + std::to_string(e.second);
            std::fprintf(stderr,
                         "qwen35_ocr ggml-ops %s graph: backend=%s nodes=%d on_backend=%d other=%d splits=%d%s%s\n",
                         kind, ggml_backend_name(backend), n, primary, other, ggml_backend_sched_get_n_splits(sched),
                         ops.empty() ? "" : " fallback:", ops.c_str());
            if (other && gpu && core_env::on("CRISPEMBED_QWEN35_OCR_GPU_STRICT")) return false;
        }
        return true;
    }
};

ggml_pipeline::ggml_pipeline() : p_(new impl) {}
ggml_pipeline::~ggml_pipeline() = default;

bool ggml_pipeline::enabled() {
    return core_env::on("CRISPEMBED_QWEN35_OCR_GPU");
}
const char * ggml_pipeline::backend_name() const {
    return p_->backend ? ggml_backend_name(p_->backend) : "none";
}
bool ggml_pipeline::on_gpu() const {
    return p_->gpu;
}
int ggml_pipeline::tokens() const {
    return p_->past;
}
void ggml_pipeline::reset() {
    p_->past = 0;
    if (p_->state_buf) ggml_backend_buffer_clear(p_->state_buf, 0);
}

bool ggml_pipeline::init(const decoder_weights & dw, const vision_weights & vw, const attention_config & ac,
                         const vision_config & vc, int capacity, int threads) {
    if (!enabled() || !dw.embedding || !dw.head || !dw.norm || dw.layers.empty() || dw.layers.size() > 64 ||
        vw.layers.empty() || vw.layers.size() > 32 || capacity < 1 || capacity > 32768 || threads < 1 || threads > 256)
        return false;
    p_.reset(new impl);
    auto & p = *p_;
    p.dw = dw;
    p.vw = vw;
    p.ac = ac;
    p.vc = vc;
    p.capacity = capacity;
    p.threads = threads;
    p.width = int(dw.embedding->ne[0]);
    // CRISPEMBED_GPU_BACKEND / --gpu-backend preference; CPU when no GPU is built or found.
    p.backend = crispasr_init_gpu_backend();
    if (!p.backend) p.backend = ggml_backend_cpu_init();
    if (!p.backend) return false;
    p.gpu = !ggml_backend_is_cpu(p.backend);
    p.cpu = p.gpu ? ggml_backend_cpu_init() : p.backend;
    if (!p.cpu) return false;

    // Validate the small qwen35 schedule (same checks as graph_kv_decoder::init).
    const int d = ac.head_dim, qh = ac.query_heads, kh = ac.kv_heads, width = p.width;
    if (width < 1 || width > 8192 || qh < 1 || kh < 1 || qh % kh || d < 2 || ac.rotary_dim < 2 || ac.rotary_dim > d ||
        ac.rotary_dim % 2 || ac.mrope_section[0] + ac.mrope_section[1] + ac.mrope_section[2] != ac.rotary_dim / 2 ||
        !is_matrix(dw.head, width, dw.head->ne[1]) || !is_matrix(dw.norm, width, 1))
        return false;
    for (const auto & l : dw.layers) {
        if (!l.gate || !is_matrix(l.input_norm, width, 1) || !is_matrix(l.post_norm, width, 1)) return false;
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
                !is_matrix(x.qkv, width, 3 * group) || !is_matrix(x.out, group, width) || x.conv->ne[2] != 3 * group)
                return false;
            p.H = int(heads);
            p.W = int(hw);
            p.K = int(kernel);
        }
    }
    const int vwid = vc.width;
    if (vwid < 1 || vc.heads < 1 || vwid % vc.heads || !is_matrix(vw.patch, vc.patch_width, vwid) ||
        !is_matrix(vw.merger_up, vwid * vc.merge * vc.merge, vwid * vc.merge * vc.merge) ||
        !is_matrix(vw.merger_down, vwid * vc.merge * vc.merge, vc.output))
        return false;

    // Matrices: device copies in their stored type (GPU), or the mapped CPU tensors (CPU).
    std::vector<ggml_tensor *> matrices = { dw.head, vw.patch, vw.merger_up, vw.merger_down };
    for (const auto & l : dw.layers) {
        for (auto * t : { l.gate, l.up, l.down }) matrices.push_back(t);
        if (l.full_attention)
            for (auto * t : { l.attention.q, l.attention.k, l.attention.v, l.attention.out }) matrices.push_back(t);
        else
            for (auto * t : { l.linear.qkv, l.linear.z, l.linear.a, l.linear.b, l.linear.out }) matrices.push_back(t);
    }
    for (const auto & l : vw.layers)
        for (auto * t : { l.qkv, l.out, l.up, l.down }) matrices.push_back(t);
    for (auto * t : matrices)
        if (!t || !t->data) return false;
    if (p.gpu) {
        std::sort(matrices.begin(), matrices.end());
        matrices.erase(std::unique(matrices.begin(), matrices.end()), matrices.end());
        p.weight_ctx = ggml_init({ (matrices.size() + 1) * ggml_tensor_overhead(), nullptr, true });
        if (!p.weight_ctx) return false;
        // Precision gates (default off): F32 device copies of the vision and/or decoder matrices, so the
        // backend multiplies in F32 instead of rounding the activations to the weight type (cuBLAS F16
        // GEMM). Same values as the stored weights, widened exactly; the CPU backend path is unchanged.
        std::set<const ggml_tensor *> widen;
        if (core_env::on("CRISPEMBED_QWEN35_OCR_GPU_F32_VISION")) {
            for (auto * t : { vw.patch, vw.merger_up, vw.merger_down }) widen.insert(t);
            for (const auto & l : vw.layers)
                for (auto * t : { l.qkv, l.out, l.up, l.down }) widen.insert(t);
        }
        if (core_env::on("CRISPEMBED_QWEN35_OCR_GPU_F32_DECODER"))
            for (auto * t : matrices)
                if (!widen.count(t)) {
                    bool vision = t == vw.patch || t == vw.merger_up || t == vw.merger_down;
                    for (const auto & l : vw.layers) vision |= t == l.qkv || t == l.out || t == l.up || t == l.down;
                    if (!vision) widen.insert(t);
                }
        for (auto * t : matrices) {
            const bool wide = widen.count(t) && t->type != GGML_TYPE_F32 && ggml_get_type_traits(t->type)->to_float;
            p.device[t] = wide ? ggml_new_tensor(p.weight_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, t->ne)
                               : ggml_dup_tensor(p.weight_ctx, t);
        }
        p.weight_buf = ggml_backend_alloc_ctx_tensors(p.weight_ctx, p.backend);
        if (!p.weight_buf) return false;
        ggml_backend_buffer_set_usage(p.weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & e : p.device) {
            if (e.second->type == e.first->type) {
                ggml_backend_tensor_set(e.second, e.first->data, 0, ggml_nbytes(e.first));
                continue;
            }
            std::vector<float> wide(size_t(ggml_nelements(e.first)));
            const int64_t row = e.first->ne[0], rows = ggml_nelements(e.first) / row;
            for (int64_t r = 0; r < rows; ++r)
                ggml_get_type_traits(e.first->type)
                    ->to_float(static_cast<const char *>(e.first->data) + r * e.first->nb[1], wide.data() + r * row, row);
            ggml_backend_tensor_set(e.second, wide.data(), 0, wide.size() * sizeof(float));
        }
    }

    // Small F32 constants (norm weights, biases, decay factors, conv taps) and the states.
    const size_t n_layers = dw.layers.size();
    const size_t n_const = 16 * n_layers + 16 * vw.layers.size() + 16;
    p.const_ctx = ggml_init({ n_const * ggml_tensor_overhead(), nullptr, true });
    p.state_ctx = ggml_init({ (4 * n_layers + 4) * ggml_tensor_overhead(), nullptr, true });
    if (!p.const_ctx || !p.state_ctx) return false;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> uploads;
    bool ok = true;
    // F32 vector of `src` (any stored type); transform applied per element.
    auto vec = [&](ggml_tensor * src, float (*f)(float) = nullptr) -> ggml_tensor * {
        if (!src) {
            ok = false;
            return nullptr;
        }
        std::vector<float> values(size_t(ggml_nelements(src)));
        if (!tensor_floats(src, values.data())) ok = false;
        if (f)
            for (float & v : values) v = f(v);
        ggml_tensor * t = ggml_new_tensor_1d(p.const_ctx, GGML_TYPE_F32, int64_t(values.size()));
        uploads.emplace_back(t, std::move(values));
        return t;
    };
    auto one_plus = [](float w) { return 1.0f + w; };    // MQ:724 (1.0 + self.weight)
    auto neg_exp = [](float a) { return -std::exp(a); }; // MQ:493 -A_log.exp()
    p.final_norm = vec(dw.norm, one_plus);
    p.l2_eps = ggml_new_tensor_1d(p.const_ctx, GGML_TYPE_F32, 1);
    uploads.emplace_back(p.l2_eps, std::vector<float>{ 1e-6f }); // MQ:228 l2norm eps
    p.layers.resize(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        const auto & l = dw.layers[i];
        auto & s = p.layers[i];
        s.in_norm = vec(l.input_norm, one_plus);
        s.post_norm = vec(l.post_norm, one_plus);
        if (l.full_attention) {
            s.q_norm = vec(l.attention.q_norm, one_plus);
            s.k_norm = vec(l.attention.k_norm, one_plus);
            s.keys = ggml_new_tensor_3d(p.state_ctx, GGML_TYPE_F32, d, capacity, kh);
            s.values = ggml_new_tensor_3d(p.state_ctx, GGML_TYPE_F32, d, capacity, kh);
        } else {
            s.neg_exp_alog = vec(l.linear.a_log, neg_exp);
            s.dt_bias = vec(l.linear.dt_bias);
            s.gated_norm = vec(l.linear.norm);
            const int64_t channels = 3 * int64_t(p.H) * p.W;
            std::vector<float> conv(size_t(ggml_nelements(l.linear.conv)));
            if (!tensor_floats(l.linear.conv, conv.data())) return false;
            s.conv = ggml_new_tensor_2d(p.const_ctx, GGML_TYPE_F32, p.K, channels);
            uploads.emplace_back(s.conv, std::move(conv));
            s.conv_state = ggml_new_tensor_2d(p.state_ctx, GGML_TYPE_F32, p.K - 1, channels);
            s.recurrent = ggml_new_tensor_4d(p.state_ctx, GGML_TYPE_F32, p.W, p.W, p.H, 1);
        }
    }
    p.patch_b = vec(vw.patch_bias);
    p.merger_n = vec(vw.merger_norm);
    p.merger_nb = vec(vw.merger_norm_bias);
    p.merger_up_b = vec(vw.merger_up_bias);
    p.merger_down_b = vec(vw.merger_down_bias);
    p.vlayers.resize(vw.layers.size());
    for (size_t i = 0; i < vw.layers.size(); ++i) {
        const auto & l = vw.layers[i];
        auto & s = p.vlayers[i];
        s.n1 = vec(l.norm1);
        s.n1b = vec(l.norm1_bias);
        s.n2 = vec(l.norm2);
        s.n2b = vec(l.norm2_bias);
        s.qkv_b = vec(l.qkv_bias);
        s.out_b = vec(l.out_bias);
        s.up_b = vec(l.up_bias);
        s.down_b = vec(l.down_bias);
    }
    if (!ok) return false;
    p.const_buf = ggml_backend_alloc_ctx_tensors(p.const_ctx, p.backend);
    p.state_buf = ggml_backend_alloc_ctx_tensors(p.state_ctx, p.backend);
    if (!p.const_buf || !p.state_buf) return false;
    for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second.data(), 0, u.second.size() * sizeof(float));
    reset();

    std::vector<ggml_backend_t> backends = { p.backend };
    if (p.cpu != p.backend) backends.push_back(p.cpu);
    p.sched = ggml_backend_sched_new(backends.data(), nullptr, int(backends.size()), kNodes, false, false);
    p.meta.resize(ggml_tensor_overhead() * kNodes + ggml_graph_overhead_custom(kNodes, false));
    if (core_env::on("CRISPEMBED_QWEN35_OCR_GPU_LOG"))
        std::fprintf(stderr, "qwen35_ocr ggml-ops: backend=%s gpu=%d device_matrices=%zu\n",
                     ggml_backend_name(p.backend), int(p.gpu), p.device.size());
    return p.sched != nullptr;
}

bool ggml_pipeline::vision(const float * patches, int tokens, const std::vector<int> & grid,
                           std::vector<float> & merged) {
    merged.clear();
    auto & p = *p_;
    const auto & c = p.vc;
    if (!p.sched || !patches || tokens < 1 || tokens > 4096) return false;
    const int group = c.merge * c.merge;
    if (tokens % group) return false;
    std::vector<int> offsets;
    std::vector<float> position, cosine, sine;
    if (!vision_host_inputs(p.vw, c, tokens, grid, offsets, position, cosine, sine)) return false;
    const int N = tokens, V = c.width, heads = c.heads, d = V / heads;
    ggml_context * g = ggml_init({ p.meta.size(), p.meta.data(), true });
    if (!g) return false;
    struct guard_t {
        ggml_context * g;
        ~guard_t() { ggml_free(g); }
    } guard{ g };
    ggml_cgraph * gf = ggml_new_graph_custom(g, kNodes, false);
    ggml_tensor * in = ggml_new_tensor_2d(g, GGML_TYPE_F32, c.patch_width, N);
    ggml_tensor * pos = ggml_new_tensor_2d(g, GGML_TYPE_F32, V, N);
    ggml_tensor * cos_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, 1, N);
    ggml_tensor * sin_t = ggml_new_tensor_3d(g, GGML_TYPE_F32, d, 1, N);
    for (auto * t : { in, pos, cos_t, sin_t }) ggml_set_input(t);
    ggml_tensor * mask = nullptr;
    if (offsets.size() > 2) { // several frames/images: attention stays inside each (MQ:930-952)
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, N, N);
        ggml_set_input(mask);
    }
    auto linear = [&](const ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
        return ggml_add(g, p.mm(g, w, x), b);
    };
    auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        return ggml_add(g, ggml_mul(g, ggml_norm(g, x, c.eps), w), b);
    };
    // q*cos + rotate_half(q)*sin; rotate_half(x) = cat(-x2, x1) (MQ:536-540, 864-875).
    auto rotate = [&](ggml_tensor * x) { // x [d, heads, N] contiguous
        ggml_tensor * x1 = ggml_cont(g, ggml_view_3d(g, x, d / 2, heads, N, x->nb[1], x->nb[2], 0));
        ggml_tensor * x2 =
            ggml_cont(g, ggml_view_3d(g, x, d / 2, heads, N, x->nb[1], x->nb[2], (d / 2) * sizeof(float)));
        ggml_tensor * half = ggml_concat(g, ggml_neg(g, x2), x1, 0);
        return ggml_add(g, ggml_mul(g, x, cos_t), ggml_mul(g, half, sin_t));
    };
    ggml_tensor * x = ggml_add(g, linear(p.vw.patch, p.patch_b, in), pos); // MQ:828-846 + pos embed
    const float scale = 1.0f / std::sqrt(float(d));
    for (size_t li = 0; li < p.vw.layers.size(); ++li) {
        const auto & l = p.vw.layers[li];
        const auto & s = p.vlayers[li];
        ggml_tensor * qkv = linear(l.qkv, s.qkv_b, layer_norm(x, s.n1, s.n1b)); // [3V, N]
        const size_t es = sizeof(float);
        auto part = [&](int k) {
            return ggml_cont(g, ggml_view_3d(g, qkv, d, heads, N, size_t(d) * es, qkv->nb[1], size_t(k) * V * es));
        };
        ggml_tensor * q = ggml_cont(g, ggml_permute(g, rotate(part(0)), 0, 2, 1, 3)); // [d, N, heads]
        ggml_tensor * k = ggml_cont(g, ggml_permute(g, rotate(part(1)), 0, 2, 1, 3)); // [d, N, heads]
        ggml_tensor * vt = ggml_cont(g, ggml_permute(g, part(2), 1, 2, 0, 3));        // [N, d, heads]
        ggml_tensor * kq = ggml_mul_mat(g, k, q);                                     // [N, N, heads]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(g, kq, mask, scale, 0.0f);
        ggml_tensor * o = ggml_mul_mat(g, vt, kq); // [d, N, heads]
        ggml_mul_mat_set_prec(o, GGML_PREC_F32);
        o = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3)), V, N);
        x = ggml_add(g, x, linear(l.out, s.out_b, o));
        ggml_tensor * up = linear(l.up, s.up_b, layer_norm(x, s.n2, s.n2b));
        // gelu_pytorch_tanh, written out (ggml_gelu's CPU kernel uses an F16 table).
        ggml_tensor * inner = ggml_add(g, up, ggml_scale(g, ggml_mul(g, ggml_sqr(g, up), up), 0.044715f));
        ggml_tensor * act =
            ggml_mul(g, up, ggml_scale_bias(g, ggml_tanh(g, ggml_scale(g, inner, 0.7978845608028654f)), 0.5f, 0.5f));
        x = ggml_add(g, x, linear(l.down, s.down_b, act));
    }
    ggml_tensor * m = ggml_reshape_2d(g, layer_norm(x, p.merger_n, p.merger_nb), V * group, N / group);
    m = ggml_gelu_erf(g, linear(p.vw.merger_up, p.merger_up_b, m));
    ggml_tensor * out = linear(p.vw.merger_down, p.merger_down_b, m);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    if (!p.place(gf, "vision")) return false;
    ggml_backend_tensor_set(in, patches, 0, size_t(N) * c.patch_width * sizeof(float));
    ggml_backend_tensor_set(pos, position.data(), 0, position.size() * sizeof(float));
    ggml_backend_tensor_set(cos_t, cosine.data(), 0, cosine.size() * sizeof(float));
    ggml_backend_tensor_set(sin_t, sine.data(), 0, sine.size() * sizeof(float));
    if (mask) {
        std::vector<float> values(size_t(N) * N, -INFINITY);
        for (size_t f = 1; f < offsets.size(); ++f)
            for (int r = offsets[f - 1]; r < offsets[f]; ++r)
                for (int col = offsets[f - 1]; col < offsets[f]; ++col) values[size_t(r) * N + col] = 0.0f;
        ggml_backend_tensor_set(mask, values.data(), 0, values.size() * sizeof(float));
    }
    p.set_threads();
    if (ggml_backend_sched_graph_compute(p.sched, gf) != GGML_STATUS_SUCCESS) return false;
    merged.resize(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out, merged.data(), 0, merged.size() * sizeof(float));
    return std::all_of(merged.begin(), merged.end(), [](float v) { return std::isfinite(v); });
}

bool ggml_pipeline::forward(const float * hidden, int tokens, const float * positions, std::vector<float> & logits) {
    logits.clear();
    auto & p = *p_;
    if (!p.sched || !hidden || !positions || tokens < 1 || p.past + tokens > p.capacity) return false;
    const auto & c = p.ac;
    const int T = tokens, D = p.width, d = c.head_dim, qh = c.query_heads, kh = c.kv_heads;
    const int Lk = p.past + T, H = p.H, W = p.W, K = p.K, group = H * W;
    const int64_t vocab = p.dw.head->ne[1];
    ggml_context * g = ggml_init({ p.meta.size(), p.meta.data(), true });
    if (!g) return false;
    struct guard_t {
        ggml_context * g;
        ~guard_t() { ggml_free(g); }
    } guard{ g };
    ggml_cgraph * gf = ggml_new_graph_custom(g, kNodes, false);
    ggml_tensor * input = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_tensor * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 4 * int64_t(T));
    ggml_set_input(input);
    ggml_set_input(pos);
    ggml_tensor * mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, Lk, T);
        ggml_set_input(mask);
    }
    // Qwen3_5RMSNorm (MQ:711-729): x * rsqrt(mean(x^2) + eps) * (1 + w).
    auto rms = [&](ggml_tensor * t, ggml_tensor * w1) { return ggml_mul(g, ggml_rms_norm(g, t, c.eps), w1); };
    int sections[GGML_MROPE_SECTIONS] = { c.mrope_section[0], c.mrope_section[1], c.mrope_section[2], 0 };
    // Interleaved multimodal RoPE (MQ:157-173, partial_rotary_factor 0.25): NEOX pairs (i, i+32)
    // over the first 64 dims; sector i%3 selects t/h/w as qwen35_attention.cpp rotate().
    auto rope = [&](ggml_tensor * t) {
        return ggml_rope_multi(g, t, pos, nullptr, c.rotary_dim, sections, GGML_ROPE_TYPE_IMROPE, 0, c.theta, 1.0f,
                               0.0f, 1.0f, 0.0f, 0.0f);
    };
    const float scale = 1.0f / std::sqrt(float(d));
    ggml_tensor * x = input;
    for (size_t li = 0; li < p.dw.layers.size(); ++li) {
        const auto & l = p.dw.layers[li];
        auto & s = p.layers[li];
        ggml_tensor * xn = rms(x, s.in_norm);
        ggml_tensor * mixed;
        if (l.full_attention) { // Qwen3_5Attention (MQ:620-693)
            const auto & a = l.attention;
            ggml_tensor * qfull = p.mm(g, a.q, xn); // [2d*qh, T]: per head [query d | gate d] (MQ:658-661)
            const size_t fs = ggml_element_size(qfull);
            ggml_tensor * query = ggml_cont(g, ggml_view_3d(g, qfull, d, qh, T, 2 * d * fs, qfull->nb[1], 0));
            ggml_tensor * gate = ggml_cont(g, ggml_view_3d(g, qfull, d, qh, T, 2 * d * fs, qfull->nb[1], d * fs));
            ggml_tensor * qr = rope(rms(query, s.q_norm));                                          // [d, qh, T]
            ggml_tensor * kr = rope(rms(ggml_reshape_3d(g, p.mm(g, a.k, xn), d, kh, T), s.k_norm)); // [d, kh, T]
            ggml_tensor * v = ggml_reshape_3d(g, p.mm(g, a.v, xn), d, kh, T);
            const size_t row = s.keys->nb[1], plane = s.keys->nb[2];
            ggml_tensor * kdst = ggml_view_3d(g, s.keys, d, T, kh, row, plane, size_t(p.past) * row);
            ggml_tensor * vdst = ggml_view_3d(g, s.values, d, T, kh, row, plane, size_t(p.past) * row);
            ggml_build_forward_expand(gf, ggml_cpy(g, ggml_permute(g, kr, 0, 2, 1, 3), kdst));
            ggml_build_forward_expand(gf, ggml_cpy(g, ggml_permute(g, v, 0, 2, 1, 3), vdst));
            ggml_tensor * kall = ggml_view_3d(g, s.keys, d, Lk, kh, row, plane, 0);
            ggml_tensor * vall = ggml_view_3d(g, s.values, d, Lk, kh, row, plane, 0);
            ggml_tensor * q = ggml_cont(g, ggml_permute(g, qr, 0, 2, 1, 3)); // [d, T, qh]
            ggml_tensor * kq = ggml_mul_mat(g, kall, q);                     // [Lk, T, qh]; GQA broadcast
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(g, kq, mask, scale, 0.0f);
            ggml_tensor * vt = ggml_cont(g, ggml_permute(g, vall, 1, 0, 2, 3)); // [Lk, d, kh]
            ggml_tensor * o = ggml_mul_mat(g, vt, kq);                          // [d, T, qh]
            ggml_mul_mat_set_prec(o, GGML_PREC_F32);
            o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3));             // [d, qh, T]
            ggml_tensor * pooled = ggml_mul(g, o, ggml_sigmoid(g, gate)); // MQ:689
            mixed = p.mm(g, a.out, ggml_reshape_2d(g, pooled, qh * d, T));
        } else { // Qwen3_5GatedDeltaNet (MQ:356-534)
            const auto & lw = l.linear;
            ggml_tensor * qkv = p.mm(g, lw.qkv, xn); // [3*group, T]
            ggml_tensor * z = p.mm(g, lw.z, xn);     // [group, T]
            ggml_tensor * av = p.mm(g, lw.a, xn);    // [H, T]
            ggml_tensor * bv = p.mm(g, lw.b, xn);    // [H, T]
            const int64_t channels = 3 * int64_t(group);
            ggml_tensor * sx = ggml_concat(g, s.conv_state, ggml_cont(g, ggml_transpose(g, qkv)), 0); // [K-1+T, ch]
            ggml_build_forward_expand(
                gf, ggml_cpy(g, ggml_view_2d(g, sx, K - 1, channels, sx->nb[1], size_t(T) * ggml_element_size(sx)),
                             s.conv_state));
            ggml_tensor * conv = ggml_ssm_conv(g, ggml_reshape_3d(g, sx, K - 1 + T, channels, 1), s.conv);
            conv = ggml_silu(g, ggml_reshape_2d(g, conv, channels, T)); // MQ:474
            const size_t cs = ggml_element_size(conv);
            auto part = [&](int index) {
                return ggml_cont(g, ggml_view_4d(g, conv, W, H, T, 1, size_t(W) * cs, conv->nb[1], conv->nb[1] * T,
                                                 size_t(index) * group * cs));
            };
            auto l2 = [&](ggml_tensor * t) { // MQ:228-231
                return ggml_div(g, t, ggml_sqrt(g, ggml_add(g, ggml_sum_rows(g, ggml_sqr(g, t)), p.l2_eps)));
            };
            ggml_tensor * query = l2(part(0));
            ggml_tensor * key = l2(part(1));
            ggml_tensor * value = part(2);
            // MQ:493 g = -exp(A_log) * softplus(a + dt_bias); MQ:491 beta = sigmoid(b).
            ggml_tensor * decay = ggml_reshape_4d(
                g, ggml_mul(g, ggml_softplus(g, ggml_add(g, av, s.dt_bias)), s.neg_exp_alog), 1, H, T, 1);
            ggml_tensor * beta = ggml_reshape_4d(g, ggml_sigmoid(g, bv), 1, H, T, 1);
            ggml_tensor * delta = ggml_gated_delta_net(g, query, key, value, decay, beta, s.recurrent, 1);
            const size_t ds = ggml_element_size(delta);
            ggml_tensor * core = ggml_view_2d(g, delta, W, int64_t(H) * T, size_t(W) * ds, 0);
            ggml_tensor * state = ggml_view_1d(g, delta, int64_t(W) * W * H, size_t(T) * group * ds);
            ggml_build_forward_expand(gf, ggml_cpy(g, state, ggml_reshape_1d(g, s.recurrent, int64_t(W) * W * H)));
            // Qwen3_5RMSNormGated (MQ:175-191): rms_norm(core) * w * silu(z).
            ggml_tensor * zr = ggml_reshape_2d(g, z, W, int64_t(H) * T);
            ggml_tensor * normed =
                ggml_mul(g, ggml_mul(g, ggml_rms_norm(g, ggml_cont(g, core), c.eps), s.gated_norm), ggml_silu(g, zr));
            mixed = p.mm(g, lw.out, ggml_reshape_2d(g, normed, group, T));
        }
        x = ggml_add(g, x, mixed);
        ggml_tensor * hn = rms(x, s.post_norm);
        ggml_tensor * act = ggml_swiglu_split(g, p.mm(g, l.gate, hn), p.mm(g, l.up, hn)); // MQ:707
        x = ggml_add(g, x, p.mm(g, l.down, act));
    }
    ggml_tensor * last = rms(ggml_view_2d(g, x, D, 1, x->nb[1], size_t(T - 1) * x->nb[1]), p.final_norm);
    ggml_tensor * out = p.mm(g, p.dw.head, last);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    if (!p.place(gf, T > 1 ? "prefill" : "decode")) return false;
    ggml_backend_tensor_set(input, hidden, 0, size_t(T) * D * sizeof(float));
    std::vector<int32_t> ids(size_t(4) * T);
    for (int t = 0; t < T; ++t) {
        for (int axis = 0; axis < 3; ++axis)
            ids[size_t(axis) * T + t] = int32_t(std::lround(positions[size_t(axis) * T + t]));
        ids[size_t(3) * T + t] = ids[t]; // unused 4th section; t keeps rotate()'s default axis
    }
    ggml_backend_tensor_set(pos, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (mask) {
        std::vector<float> m(size_t(Lk) * T, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int k = p.past + t + 1; k < Lk; ++k) m[size_t(t) * Lk + k] = -INFINITY;
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    p.set_threads();
    if (ggml_backend_sched_graph_compute(p.sched, gf) != GGML_STATUS_SUCCESS) return false;
    logits.resize(size_t(vocab));
    ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));
    if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); })) {
        logits.clear();
        return false;
    }
    p.past += T;
    return true;
}
} // namespace qwen35_ocr
