// ppformulanet_l_ocr.cpp — PP-FormulaNet-L (SAM-ViT + MBart) via ggml.
//
// SAM-ViT encoder: windowed + global attention with decomposed relative
// position bias. Neck + multi-modal projector converts to decoder input.
// MBart PRE-LN decoder with KV caching for autoregressive generation.

#include "ppformulanet_l_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FP16/quantized → F32 dequantization
// ---------------------------------------------------------------------------

static std::vector<float> to_f32(const ggml_tensor* t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, out.data(), n);
        } else {
            memset(out.data(), 0, n * sizeof(float));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// CPU-side math helpers
// ---------------------------------------------------------------------------

static void layernorm_cpu(const float* in, float* out, int D,
                          const float* w, const float* b, float eps = 1e-5f) {
    double mean = 0;
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= D;
    double var = 0;
    for (int i = 0; i < D; i++) { double d = in[i] - mean; var += d * d; }
    var /= D;
    float s = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < D; i++)
        out[i] = ((in[i] - (float)mean) * s) * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

static void layernorm_cpu(const float* in, float* out, int D,
                          const ggml_tensor* w, const ggml_tensor* b, float eps = 1e-5f) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    layernorm_cpu(in, out, D,
                  wv.empty() ? nullptr : wv.data(),
                  bv.empty() ? nullptr : bv.data(), eps);
}

// LayerNorm2d: normalize over channel dim for NCHW tensor
// Input/output shape: (C, H, W), normalize over C for each spatial position
static void layernorm2d_cpu(const float* in, float* out,
                            int C, int H, int W,
                            const float* w, const float* b, float eps = 1e-6f) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double mean = 0;
            for (int c = 0; c < C; c++)
                mean += in[c * H * W + y * W + x];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; c++) {
                double d = in[c * H * W + y * W + x] - mean;
                var += d * d;
            }
            var /= C;
            float s = 1.0f / sqrtf((float)var + eps);
            for (int c = 0; c < C; c++) {
                float v = (in[c * H * W + y * W + x] - (float)mean) * s;
                out[c * H * W + y * W + x] = v * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
            }
        }
    }
}

static void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                        const float* w, const float* b) {
    for (int o = 0; o < out_dim; o++) {
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++)
            s += in[i] * w[o * in_dim + i];
        out[o] = s;
    }
}

static void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                        const ggml_tensor* w, const ggml_tensor* b) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    linear_cpu(in, out, in_dim, out_dim, wv.data(), bv.empty() ? nullptr : bv.data());
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void mha_1q_cpu(const float* q, const float* k, const float* v,
                        float* out, int n_kv, int D, int n_heads) {
    int hd = D / n_heads;
    std::vector<float> result(D, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_kv);
        for (int ki = 0; ki < n_kv; ki++) {
            float s = 0;
            for (int d = 0; d < hd; d++)
                s += q[off + d] * k[ki * D + off + d];
            scores[ki] = s / sqrtf((float)hd);
        }
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        for (int ki = 0; ki < n_kv; ki++) scores[ki] /= sum;
        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_kv; ki++)
                s += scores[ki] * v[ki * D + off + d];
            result[off + d] = s;
        }
    }
    memcpy(out, result.data(), D * sizeof(float));
}

// 2D convolution (NCHW layout), handles groups and arbitrary padding
static void conv2d_cpu(const float* in, float* out,
                        const float* weight, const float* bias,
                        int in_ch, int out_ch, int H, int W,
                        int kh, int kw, int stride, int pad,
                        int groups) {
    int out_H = (H + 2 * pad - kh) / stride + 1;
    int out_W = (W + 2 * pad - kw) / stride + 1;
    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;

    for (int oc = 0; oc < out_ch; oc++) {
        int g = oc / ch_per_group_out;
        float b = bias ? bias[oc] : 0.0f;
        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                float sum = b;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    int actual_ic = g * ch_per_group_in + ic;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * stride - pad + ky;
                            int ix = ox * stride - pad + kx;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                float pixel = in[actual_ic * H * W + iy * W + ix];
                                int w_idx = oc * (ch_per_group_in * kh * kw)
                                            + ic * kh * kw + ky * kw + kx;
                                sum += pixel * weight[w_idx];
                            }
                        }
                    }
                }
                out[oc * out_H * out_W + oy * out_W + ox] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structs: ViT Encoder layer
// ---------------------------------------------------------------------------

struct vit_layer {
    ggml_tensor *ln1_w, *ln1_b;   // pre-attn LayerNorm
    ggml_tensor *qkv_w, *qkv_b;  // fused QKV (3*hidden, hidden)
    ggml_tensor *proj_w, *proj_b; // output projection
    ggml_tensor *rel_pos_h;       // (2*ws-1, head_dim) or (2*48-1, head_dim) for global
    ggml_tensor *rel_pos_w;
    ggml_tensor *ln2_w, *ln2_b;   // pre-FFN LayerNorm
    ggml_tensor *mlp_lin1_w, *mlp_lin1_b;
    ggml_tensor *mlp_lin2_w, *mlp_lin2_b;
    bool is_global;               // global attention (no windowing)
    int window_size;              // 0 for global, 14 for windowed
};

// Decoder layer (MBart PRE-LN)
struct ppfnl_dec_layer {
    ggml_tensor *self_ln_w, *self_ln_b;
    ggml_tensor *self_q_w, *self_q_b, *self_k_w, *self_k_b, *self_v_w, *self_v_b;
    ggml_tensor *self_out_w, *self_out_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b, *cross_k_w, *cross_k_b, *cross_v_w, *cross_v_b;
    ggml_tensor *cross_out_w, *cross_out_b;
    ggml_tensor *ff_ln_w, *ff_ln_b, *ff_up_w, *ff_up_b, *ff_down_w, *ff_down_b;
};

struct ppformulanet_l_ocr_context {
    ppformulanet_l_ocr_hparams hparams;
    std::vector<int> global_attn_indexes;

    // Encoder: Patch Embed
    ggml_tensor *patch_embed_w, *patch_embed_b;
    ggml_tensor *pos_embed;   // (1, 48, 48, 768)

    // Encoder: ViT Layers
    std::vector<vit_layer> enc_layers;

    // Encoder: Neck
    ggml_tensor *neck_conv1_w;   // (256, 768, 1, 1)
    ggml_tensor *neck_conv2_w;   // (256, 256, 3, 3)
    ggml_tensor *neck_ln1_w, *neck_ln1_b;
    ggml_tensor *neck_ln2_w, *neck_ln2_b;

    // Encoder: Multi-modal Projector
    ggml_tensor *proj_conv1_w;   // (512, 256, 3, 3)
    ggml_tensor *proj_conv2_w;   // (1024, 512, 3, 3)
    ggml_tensor *proj_linear1_w, *proj_linear1_b;
    ggml_tensor *proj_linear2_w, *proj_linear2_b;

    // Decoder
    ggml_tensor *tok_embed, *pos_embed_dec;
    ggml_tensor *embed_ln_w, *embed_ln_b;
    ggml_tensor *final_ln_w, *final_ln_b;
    ggml_tensor *lm_head_w;
    std::vector<ppfnl_dec_layer> dec_layers_vec;

    // Infrastructure
    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads;
    std::string result_buf;

    // Cached encoder output (after projector: 144 tokens × 512d)
    std::vector<float> proj_out;
    int n_enc_tokens = 0;
    int proj_dim = 0;  // 512

    // Cross-attention K/V cache
    std::vector<std::vector<float>> cross_k_cache;
    std::vector<std::vector<float>> cross_v_cache;

    // Precomputed rel_pos lookup tables (built once at init).
    std::vector<std::vector<float>> rp_h_per_layer;
    std::vector<std::vector<float>> rp_w_per_layer;
};

// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

static ggml_tensor* F(const std::map<std::string, ggml_tensor*>& m, const char* n) {
    auto it = m.find(n);
    if (it != m.end()) return it->second;
    std::string alt(n);
    for (auto& c : alt) if (c == '.') c = '_';
    it = m.find(alt);
    return it != m.end() ? it->second : nullptr;
}

static void map_tensors(ppformulanet_l_ocr_context* ctx) {
    const auto& m = ctx->wl.tensors;
    const auto& hp = ctx->hparams;
    char buf[256];

    // Patch embed
    ctx->patch_embed_w = F(m, "enc.patch_embed.weight");
    ctx->patch_embed_b = F(m, "enc.patch_embed.bias");
    ctx->pos_embed     = F(m, "enc.pos_embed");

    // ViT layers
    ctx->enc_layers.resize(hp.enc_layers);
    for (int i = 0; i < hp.enc_layers; i++) {
        auto& l = ctx->enc_layers[i];
        bool is_global = false;
        for (int g : ctx->global_attn_indexes)
            if (g == i) { is_global = true; break; }
        l.is_global = is_global;
        l.window_size = is_global ? 0 : hp.window_size;

        auto EL = [&](const char* s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.layers.%d.%s", i, s);
            return F(m, buf);
        };
        l.ln1_w = EL("ln1.weight");       l.ln1_b = EL("ln1.bias");
        l.qkv_w = EL("attn.qkv.weight");  l.qkv_b = EL("attn.qkv.bias");
        l.proj_w = EL("attn.proj.weight"); l.proj_b = EL("attn.proj.bias");
        l.rel_pos_h = EL("attn.rel_pos_h");
        l.rel_pos_w = EL("attn.rel_pos_w");
        l.ln2_w = EL("ln2.weight");       l.ln2_b = EL("ln2.bias");
        l.mlp_lin1_w = EL("mlp.lin1.weight"); l.mlp_lin1_b = EL("mlp.lin1.bias");
        l.mlp_lin2_w = EL("mlp.lin2.weight"); l.mlp_lin2_b = EL("mlp.lin2.bias");
    }

    // Neck
    ctx->neck_conv1_w = F(m, "enc.neck.conv1.weight");
    ctx->neck_conv2_w = F(m, "enc.neck.conv2.weight");
    ctx->neck_ln1_w = F(m, "enc.neck.ln1.weight"); ctx->neck_ln1_b = F(m, "enc.neck.ln1.bias");
    ctx->neck_ln2_w = F(m, "enc.neck.ln2.weight"); ctx->neck_ln2_b = F(m, "enc.neck.ln2.bias");

    // Projector
    ctx->proj_conv1_w = F(m, "enc.proj.conv1.weight");
    ctx->proj_conv2_w = F(m, "enc.proj.conv2.weight");
    ctx->proj_linear1_w = F(m, "enc.proj.linear1.weight");
    ctx->proj_linear1_b = F(m, "enc.proj.linear1.bias");
    ctx->proj_linear2_w = F(m, "enc.proj.linear2.weight");
    ctx->proj_linear2_b = F(m, "enc.proj.linear2.bias");

    // Decoder
    ctx->tok_embed     = F(m, "dec.embed_tokens.weight");
    ctx->pos_embed_dec = F(m, "dec.embed_positions.weight");
    ctx->embed_ln_w    = F(m, "dec.embed_ln.weight");
    ctx->embed_ln_b    = F(m, "dec.embed_ln.bias");
    ctx->final_ln_w    = F(m, "dec.final_ln.weight");
    ctx->final_ln_b    = F(m, "dec.final_ln.bias");
    ctx->lm_head_w     = F(m, "dec.lm_head.weight");

    ctx->dec_layers_vec.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto& l = ctx->dec_layers_vec[i];
        auto DL = [&](const char* s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "dec.layers.%d.%s", i, s);
            return F(m, buf);
        };
        l.self_ln_w = DL("self_attn_ln.weight"); l.self_ln_b = DL("self_attn_ln.bias");
        l.self_q_w = DL("self_attn.q.weight"); l.self_q_b = DL("self_attn.q.bias");
        l.self_k_w = DL("self_attn.k.weight"); l.self_k_b = DL("self_attn.k.bias");
        l.self_v_w = DL("self_attn.v.weight"); l.self_v_b = DL("self_attn.v.bias");
        l.self_out_w = DL("self_attn.out.weight"); l.self_out_b = DL("self_attn.out.bias");
        l.cross_ln_w = DL("cross_attn_ln.weight"); l.cross_ln_b = DL("cross_attn_ln.bias");
        l.cross_q_w = DL("cross_attn.q.weight"); l.cross_q_b = DL("cross_attn.q.bias");
        l.cross_k_w = DL("cross_attn.k.weight"); l.cross_k_b = DL("cross_attn.k.bias");
        l.cross_v_w = DL("cross_attn.v.weight"); l.cross_v_b = DL("cross_attn.v.bias");
        l.cross_out_w = DL("cross_attn.out.weight"); l.cross_out_b = DL("cross_attn.out.bias");
        l.ff_ln_w = DL("ffn_ln.weight"); l.ff_ln_b = DL("ffn_ln.bias");
        l.ff_up_w = DL("ffn.up.weight"); l.ff_up_b = DL("ffn.up.bias");
        l.ff_down_w = DL("ffn.down.weight"); l.ff_down_b = DL("ffn.down.bias");
    }
}

// ---------------------------------------------------------------------------
// Graph helpers (shared with ggml graph encoder)
// ---------------------------------------------------------------------------

static ggml_tensor* ensure_f32(ggml_context* g, ggml_tensor* t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

static ggml_tensor* g_ln(ggml_context* g, ggml_tensor* x,
                          ggml_tensor* w, ggml_tensor* b, float eps = 1e-6f) {
    if (!w) return x;
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, ensure_f32(g, w));
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

static ggml_tensor* g_linear(ggml_context* g, ggml_tensor* x,
                              ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

// Get relative positional embedding via interpolation (matches HF exactly).
// rel_pos: shape (L, head_dim), L = 2*input_size - 1
// Returns: (q_size, k_size, head_dim)
static std::vector<float> get_rel_pos(int q_size, int k_size,
                                       const float* rel_pos, int L, int head_dim) {
    int max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    std::vector<float> resized(head_dim * max_rel_dist);
    for (int c = 0; c < head_dim; c++) {
        for (int i = 0; i < max_rel_dist; i++) {
            float src = (float)i * (L - 1) / std::max(max_rel_dist - 1, 1);
            int lo = (int)src;
            int hi = std::min(lo + 1, L - 1);
            float frac = src - lo;
            resized[i * head_dim + c] =
                rel_pos[lo * head_dim + c] * (1.0f - frac) +
                rel_pos[hi * head_dim + c] * frac;
        }
    }
    float q_scale = std::max((float)k_size / q_size, 1.0f);
    float k_scale = std::max((float)q_size / k_size, 1.0f);
    float offset = (float)(k_size - 1) * q_scale;
    std::vector<float> result(q_size * k_size * head_dim);
    for (int qi = 0; qi < q_size; qi++) {
        for (int ki = 0; ki < k_size; ki++) {
            int idx = (int)(qi * q_scale - ki * k_scale + offset);
            idx = std::max(0, std::min(idx, max_rel_dist - 1));
            for (int c = 0; c < head_dim; c++)
                result[(qi * k_size + ki) * head_dim + c] = resized[idx * head_dim + c];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// SAM-ViT Encoder -- full ggml graph (attention + decomposed RPE in-graph)
// ---------------------------------------------------------------------------
//
// One ggml graph per layer. For windowed layers, tokens are rearranged into
// window order on CPU before the graph and unpartitioned after. The graph does:
//   LN1 -> QKV -> split -> attention with decomposed RPE -> proj -> residual
//   -> LN2 -> MLP -> residual
//
// The decomposed RPE computation in-graph uses two matmuls + broadcast adds:
//   rp_h: [hd, aH, aH] static table (precomputed at init)
//   rp_w: [hd, aW, aW] static table
//   rel_h = mul_mat(rp_h, Q_reshaped) -> broadcast-add to scores along kx
//   rel_w = mul_mat(rp_w, Q_permuted) -> broadcast-add to scores along ky
//
// All 16 windows are processed in parallel as a batch dimension.

// Full-layer graph: LN1 → QKV → attention → proj → residual → LN2 → MLP → residual
// Used for global layers (no padding issue) and can be used for windowed layers
// when the caller has already applied LN1 externally (skip_ln1=true).
static ggml_cgraph* build_full_layer_graph(ggml_context* g,
                                            ppformulanet_l_ocr_context* ctx,
                                            int li, int C, int T,
                                            int aH, int aW, int nW, int n_heads,
                                            bool skip_ln1 = false) {
    auto& layer = ctx->enc_layers[li];
    int hd = C / n_heads;
    int wN = aH * aW;       // tokens per window (or total for global)
    int batch = n_heads * nW; // merged head+window batch dim
    float attn_scale = 1.0f / sqrtf((float)hd);

    ggml_cgraph* gf = ggml_new_graph_custom(g, 512, false);

    // --- Inputs ---
    // For windowed layers with skip_ln1: this receives already-LN'd + partitioned data
    ggml_tensor* inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
    ggml_set_name(inp, "layer_input");
    ggml_set_input(inp);

    // For windowed layers with skip_ln1: separate residual input (original partitioned hidden)
    ggml_tensor* res_inp = nullptr;
    if (skip_ln1) {
        res_inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
        ggml_set_name(res_inp, "residual_input");
        ggml_set_input(res_inp);
    }

    ggml_tensor* rp_h_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aH, aH);
    ggml_set_name(rp_h_in, "rp_h");
    ggml_set_input(rp_h_in);

    ggml_tensor* rp_w_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aW, aW);
    ggml_set_name(rp_w_in, "rp_w");
    ggml_set_input(rp_w_in);

    // --- Pre-LN (skipped for windowed layers where caller applied it) ---
    ggml_tensor* cur = skip_ln1 ? inp : g_ln(g, inp, layer.ln1_w, layer.ln1_b, 1e-6f);

    // --- Fused QKV: [C, T] -> [3C, T] ---
    ggml_tensor* qkv = g_linear(g, cur, layer.qkv_w, layer.qkv_b);

    // --- Split Q, K, V: each [C, T] ---
    ggml_tensor* Q = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], 0));
    ggml_tensor* K = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1],
                                                (size_t)C * sizeof(float)));
    ggml_tensor* V = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1],
                                                (size_t)2 * C * sizeof(float)));

    // Reshape Q/K/V: [C, T] -> [hd, n_heads, wN, nW]
    // Then permute to [hd, wN, n_heads, nW] and merge -> [hd, wN, batch]
    Q = ggml_reshape_4d(g, Q, hd, n_heads, wN, nW);
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(g, Q, hd, wN, batch);

    K = ggml_reshape_4d(g, K, hd, n_heads, wN, nW);
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(g, K, hd, wN, batch);

    V = ggml_reshape_4d(g, V, hd, n_heads, wN, nW);
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(g, V, hd, wN, batch);

    // --- Attention scores: [wN, wN, batch] ---
    ggml_tensor* scores = ggml_mul_mat(g, K, Q);
    scores = ggml_scale(g, scores, attn_scale);

    // --- Decomposed relative position bias ---
    // Q is [hd, wN, batch]. Reshape to [hd, aW, aH, batch].
    // Tokens are row-major: tok = qy*aW + qx, so in ggml col-major dim1=aW (qx fastest).
    ggml_tensor* Q_4d = ggml_reshape_4d(g, Q, hd, aW, aH, batch);

    // H-component: rp_h[d, ky, qy], Q[d, qx, qy, b]
    // mul_mat contracts d, batches qy, broadcasts batch -> [ky, qx, qy, b]
    ggml_tensor* rp_h_4d = ggml_reshape_4d(g, rp_h_in, hd, aH, aH, 1);
    ggml_tensor* rel_h = ggml_mul_mat(g, rp_h_4d, Q_4d);
    // Merge (qx,qy) -> wN, then reshape for broadcast along kx
    rel_h = ggml_reshape_3d(g, rel_h, aH, wN, batch);
    rel_h = ggml_reshape_4d(g, rel_h, 1, aH, wN, batch);

    // W-component: need Q as [d, qy, qx, b] -- permute dims 1,2
    ggml_tensor* Q_w = ggml_cont(g, ggml_permute(g, Q_4d, 0, 2, 1, 3));
    // rp_w[d, kx, qx], Q_w[d, qy, qx, b] -> [kx, qy, qx, b]
    ggml_tensor* rp_w_4d = ggml_reshape_4d(g, rp_w_in, hd, aW, aW, 1);
    ggml_tensor* rel_w = ggml_mul_mat(g, rp_w_4d, Q_w);
    // Permute to [kx, qx, qy, b] so q merge gives correct order
    rel_w = ggml_cont(g, ggml_permute(g, rel_w, 0, 2, 1, 3));
    // Merge (qx,qy) -> wN, reshape for broadcast along ky
    rel_w = ggml_reshape_3d(g, rel_w, aW, wN, batch);
    rel_w = ggml_reshape_4d(g, rel_w, aW, 1, wN, batch);

    // Add both bias components to scores
    // Reshape scores to [aW, aH, wN, batch] = [kx, ky, q_idx, b]
    scores = ggml_reshape_4d(g, scores, aW, aH, wN, batch);
    scores = ggml_add(g, scores, rel_h);  // broadcast [1,aH,...] -> [aW,aH,...]
    scores = ggml_add(g, scores, rel_w);  // broadcast [aW,1,...] -> [aW,aH,...]
    scores = ggml_reshape_3d(g, scores, wN, wN, batch);

    // --- Softmax ---
    scores = ggml_soft_max_ext(g, scores, nullptr, 1.0f, 0.0f);

    // --- Attention output: scores @ V ---
    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);

    // Reshape back to [C, T]
    attn = ggml_reshape_4d(g, attn, hd, wN, n_heads, nW);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, C, T);

    // --- Output projection ---
    attn = g_linear(g, attn, layer.proj_w, layer.proj_b);

    // --- Residual 1: use res_inp if provided, else inp ---
    ggml_tensor* res_base = skip_ln1 ? res_inp : inp;
    cur = ggml_add(g, res_base, attn);

    // --- Pre-LN + MLP ---
    ggml_tensor* residual = cur;
    cur = g_ln(g, cur, layer.ln2_w, layer.ln2_b, 1e-6f);
    ggml_tensor* up = g_linear(g, cur, layer.mlp_lin1_w, layer.mlp_lin1_b);
    up = ggml_gelu(g, up);
    cur = g_linear(g, up, layer.mlp_lin2_w, layer.mlp_lin2_b);
    cur = ggml_add(g, residual, cur);

    ggml_set_name(cur, "layer_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Rearrange hidden state from raster order to window order.
// hidden: [N, C] row-major (token tok at hidden[tok*C]).
// win_out: [wN*nW, C] with tokens reordered into contiguous windows.
static void window_partition(const float* hidden, float* win_out,
                              int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws;
    int pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws;
    int wN = ws * ws;

    memset(win_out, 0, (size_t)nWh * nWw * wN * C * sizeof(float));

    for (int wh = 0; wh < nWh; wh++) {
        for (int ww = 0; ww < nWw; ww++) {
            int win_idx = wh * nWw + ww;
            for (int y = 0; y < ws; y++) {
                int sy = wh * ws + y;
                if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) {
                    int sx = ww * ws + x;
                    if (sx >= nP) continue;
                    int src_tok = sy * nP + sx;
                    int dst_tok = win_idx * wN + y * ws + x;
                    memcpy(win_out + dst_tok * C, hidden + src_tok * C, C * sizeof(float));
                }
            }
        }
    }
}

// Reverse: window order back to raster order.
static void window_unpartition(const float* win_in, float* hidden,
                                int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws;
    int pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws;
    int wN = ws * ws;

    for (int wh = 0; wh < nWh; wh++) {
        for (int ww = 0; ww < nWw; ww++) {
            int win_idx = wh * nWw + ww;
            for (int y = 0; y < ws; y++) {
                int sy = wh * ws + y;
                if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) {
                    int sx = ww * ws + x;
                    if (sx >= nP) continue;
                    int dst_tok = sy * nP + sx;
                    int src_tok = win_idx * wN + y * ws + x;
                    memcpy(hidden + dst_tok * C, win_in + src_tok * C, C * sizeof(float));
                }
            }
        }
    }
}

// Reformat RPE table from [(q*aH+k)*hd+d] row-major to ggml [hd, aH, aH] col-major.
static void reformat_rp_table(const float* rp_in, float* rp_out, int aH, int hd) {
    for (int q = 0; q < aH; q++)
        for (int k = 0; k < aH; k++)
            for (int d = 0; d < hd; d++)
                rp_out[d + k * hd + q * aH * hd] = rp_in[(q * aH + k) * hd + d];
}

static void run_encoder_graph(ppformulanet_l_ocr_context* ctx,
                               const float* rgb_chw, int imgH, int imgW) {
    const auto& hp = ctx->hparams;
    int C = hp.enc_hidden;       // 768
    int PS = hp.patch_size;      // 16
    int nP = hp.n_patches;       // 48
    int N = nP * nP;             // 2304
    int n_heads = hp.enc_heads;  // 12
    int hd = C / n_heads;        // 64
    int ws = hp.window_size;     // 14

    // ---------------------------------------------------------------
    // Step 1: Patch embedding on CPU
    // ---------------------------------------------------------------
    auto pe_w = to_f32(ctx->patch_embed_w);
    auto pe_b = to_f32(ctx->patch_embed_b);
    auto pos = to_f32(ctx->pos_embed);

    std::vector<float> hidden(N * C);

    int patch_dim = 3 * PS * PS;
    for (int py = 0; py < nP; py++) {
        for (int px = 0; px < nP; px++) {
            int tok = py * nP + px;
            std::vector<float> patch(patch_dim);
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < PS; ky++)
                    for (int kx = 0; kx < PS; kx++)
                        patch[c * PS * PS + ky * PS + kx] =
                            rgb_chw[c * imgH * imgW + (py * PS + ky) * imgW + (px * PS + kx)];
            for (int o = 0; o < C; o++) {
                float s = pe_b[o];
                for (int i = 0; i < patch_dim; i++)
                    s += pe_w[o * patch_dim + i] * patch[i];
                hidden[tok * C + o] = s + pos[tok * C + o];
            }
        }
    }

    fprintf(stderr, "ppfnl: patch_embed done, shape=(%d, %d)\n", nP, C);

    // ---------------------------------------------------------------
    // Step 2: Per-layer full-graph execution
    // ---------------------------------------------------------------
    auto t_start = std::chrono::steady_clock::now();

    // Dequant LN weights once (used for CPU LN on windowed layers)
    auto ln1_ws = std::vector<std::vector<float>>(hp.enc_layers);
    auto ln1_bs = std::vector<std::vector<float>>(hp.enc_layers);
    for (int li = 0; li < hp.enc_layers; li++) {
        if (!ctx->enc_layers[li].is_global) {
            ln1_ws[li] = to_f32(ctx->enc_layers[li].ln1_w);
            ln1_bs[li] = to_f32(ctx->enc_layers[li].ln1_b);
        }
    }

    for (int li = 0; li < hp.enc_layers; li++) {
        auto& layer = ctx->enc_layers[li];
        bool is_global = layer.is_global;
        int aH, aW;
        if (is_global) { aH = nP; aW = nP; }
        else { aH = ws; aW = ws; }
        int wN = aH * aW;

        int nW, T;
        if (is_global) {
            nW = 1;
            T = N;
        } else {
            int pad_h = (ws - nP % ws) % ws;
            int pad_w = (ws - nP % ws) % ws;
            int pH = nP + pad_h, pW = nP + pad_w;
            nW = (pH / ws) * (pW / ws);
            T = wN * nW;
        }

        // For windowed layers: apply LN1 on CPU BEFORE partition (matches HF ordering).
        // HF: LN(unpadded) → pad with zeros → QKV.
        // For global layers: LN1 is done inside the graph (no padding issue).
        bool skip_ln1 = !is_global;
        std::vector<float> ln1_hidden;
        if (skip_ln1) {
            ln1_hidden.resize(N * C);
            for (int n = 0; n < N; n++)
                layernorm_cpu(hidden.data() + n * C, ln1_hidden.data() + n * C, C,
                              ln1_ws[li].data(), ln1_bs[li].data(), 1e-6f);
        }

        // Rearrange to window order for windowed layers
        std::vector<float> graph_input;
        std::vector<float> residual_input;
        if (is_global) {
            graph_input.assign(hidden.begin(), hidden.end());
        } else {
            // graph_input = LN1'd hidden, partitioned (padded with zeros)
            graph_input.resize(T * C, 0.0f);
            window_partition(ln1_hidden.data(), graph_input.data(), nP, ws, C);
            // residual_input = original hidden, partitioned (for residual connection)
            residual_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), residual_input.data(), nP, ws, C);
        }

        // RPE tables in ggml layout
        std::vector<float> rp_h_ggml(aH * aH * hd);
        std::vector<float> rp_w_ggml(aW * aW * hd);
        reformat_rp_table(ctx->rp_h_per_layer[li].data(), rp_h_ggml.data(), aH, hd);
        reformat_rp_table(ctx->rp_w_per_layer[li].data(), rp_w_ggml.data(), aW, hd);

        // Build and execute graph
        size_t meta_size = 8 * 1024 * 1024;
        std::vector<uint8_t> meta_buf(meta_size);
        ggml_init_params ip = { meta_size, meta_buf.data(), true };
        ggml_context* gc = ggml_init(ip);

        ggml_cgraph* gf = build_full_layer_graph(gc, ctx, li, C, T, aH, aW, nW, n_heads,
                                                  skip_ln1);

        ggml_backend_sched_reset(ctx->sched);
        ggml_backend_sched_alloc_graph(ctx->sched, gf);

        ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "layer_input");
        ggml_backend_tensor_set(inp_t, graph_input.data(), 0, (size_t)T * C * sizeof(float));

        if (skip_ln1) {
            ggml_tensor* res_t = ggml_graph_get_tensor(gf, "residual_input");
            ggml_backend_tensor_set(res_t, residual_input.data(), 0, (size_t)T * C * sizeof(float));
        }

        ggml_tensor* rph_t = ggml_graph_get_tensor(gf, "rp_h");
        ggml_backend_tensor_set(rph_t, rp_h_ggml.data(), 0, (size_t)aH * aH * hd * sizeof(float));

        ggml_tensor* rpw_t = ggml_graph_get_tensor(gf, "rp_w");
        ggml_backend_tensor_set(rpw_t, rp_w_ggml.data(), 0, (size_t)aW * aW * hd * sizeof(float));

        ggml_backend_sched_graph_compute(ctx->sched, gf);

        ggml_tensor* out_t = ggml_graph_get_tensor(gf, "layer_output");
        std::vector<float> graph_output(T * C);
        ggml_backend_tensor_get(out_t, graph_output.data(), 0, (size_t)T * C * sizeof(float));
        ggml_free(gc);

        // Unpartition
        if (is_global) {
            memcpy(hidden.data(), graph_output.data(), N * C * sizeof(float));
        } else {
            window_unpartition(graph_output.data(), hidden.data(), nP, ws, C);
        }

        if (li < 3 || li == hp.enc_layers - 1) {
            fprintf(stderr, "ppfnl: enc_layer_%d done (%s, T=%d, nW=%d)\n",
                    li, is_global ? "global" : "windowed", T, nW);
        }

    }

    auto t_end = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    fprintf(stderr, "ppfnl: ViT graph compute: %.0f ms (%d layers)\n", ms, hp.enc_layers);

    // ---------------------------------------------------------------
    // Step 3: Neck + Projector (on CPU -- small ops)
    // ---------------------------------------------------------------
    int nC = hp.output_channels;
    std::vector<float> nchw(C * nP * nP);
    for (int y = 0; y < nP; y++)
        for (int x = 0; x < nP; x++)
            for (int c = 0; c < C; c++)
                nchw[c * nP * nP + y * nP + x] = hidden[(y * nP + x) * C + c];

    auto nc1_w = to_f32(ctx->neck_conv1_w);
    std::vector<float> neck1(nC * nP * nP);
    conv2d_cpu(nchw.data(), neck1.data(), nc1_w.data(), nullptr,
               C, nC, nP, nP, 1, 1, 1, 0, 1);
    auto nln1_w = to_f32(ctx->neck_ln1_w);
    auto nln1_b = to_f32(ctx->neck_ln1_b);
    layernorm2d_cpu(neck1.data(), neck1.data(), nC, nP, nP, nln1_w.data(), nln1_b.data());

    auto nc2_w = to_f32(ctx->neck_conv2_w);
    std::vector<float> neck2(nC * nP * nP);
    conv2d_cpu(neck1.data(), neck2.data(), nc2_w.data(), nullptr,
               nC, nC, nP, nP, 3, 3, 1, 1, 1);
    auto nln2_w = to_f32(ctx->neck_ln2_w);
    auto nln2_b = to_f32(ctx->neck_ln2_b);
    layernorm2d_cpu(neck2.data(), neck2.data(), nC, nP, nP, nln2_w.data(), nln2_b.data());

    fprintf(stderr, "ppfnl: neck done, shape=(%d, %d, %d)\n", nC, nP, nP);

    int mid_ch = 512, out_ch = 1024;
    int h1 = nP / 2, w1 = nP / 2;
    auto pc1_w = to_f32(ctx->proj_conv1_w);
    std::vector<float> pc1(mid_ch * h1 * w1);
    conv2d_cpu(neck2.data(), pc1.data(), pc1_w.data(), nullptr,
               nC, mid_ch, nP, nP, 3, 3, 2, 1, 1);

    int h2 = h1 / 2, w2 = w1 / 2;
    auto pc2_w = to_f32(ctx->proj_conv2_w);
    std::vector<float> pc2(out_ch * h2 * w2);
    conv2d_cpu(pc1.data(), pc2.data(), pc2_w.data(), nullptr,
               mid_ch, out_ch, h1, w1, 3, 3, 2, 1, 1);

    int n_tokens = h2 * w2;
    std::vector<float> flat(n_tokens * out_ch);
    for (int y = 0; y < h2; y++)
        for (int x = 0; x < w2; x++)
            for (int c = 0; c < out_ch; c++)
                flat[(y * w2 + x) * out_ch + c] = pc2[c * h2 * w2 + y * w2 + x];

    auto pl1_w = to_f32(ctx->proj_linear1_w);
    auto pl1_b = to_f32(ctx->proj_linear1_b);
    auto pl2_w = to_f32(ctx->proj_linear2_w);
    auto pl2_b = to_f32(ctx->proj_linear2_b);

    int dec_dim = hp.dec_d_model;
    ctx->proj_out.resize(n_tokens * dec_dim);
    ctx->n_enc_tokens = n_tokens;
    ctx->proj_dim = dec_dim;

    for (int n = 0; n < n_tokens; n++) {
        std::vector<float> l1(out_ch);
        linear_cpu(flat.data() + n * out_ch, l1.data(), out_ch, out_ch,
                   pl1_w.data(), pl1_b.data());
        linear_cpu(l1.data(), ctx->proj_out.data() + n * dec_dim, out_ch, dec_dim,
                   pl2_w.data(), pl2_b.data());
    }

    fprintf(stderr, "ppfnl: projector done, shape=(%d, %d)\n", n_tokens, dec_dim);
}


// ---------------------------------------------------------------------------
// SAM-ViT Encoder — CPU scalar (fallback, kept for reference/debugging)
// ---------------------------------------------------------------------------

// Single ViT layer forward pass.
// hidden: BHWC format (B*nW or B, H, W, C), modifies in-place.
// For windowed layers: caller handles window partition/unpartition.
static void vit_layer_attn(const vit_layer& layer,
                            float* hidden, int B, int H, int W, int C,
                            int n_heads) {
    int hd = C / n_heads;
    int N = H * W;  // sequence length

    auto qkv_w = to_f32(layer.qkv_w);
    auto qkv_b = to_f32(layer.qkv_b);
    auto proj_w = to_f32(layer.proj_w);
    auto proj_b = to_f32(layer.proj_b);
    auto rel_h = to_f32(layer.rel_pos_h);
    auto rel_w = to_f32(layer.rel_pos_w);
    int rel_h_L = layer.rel_pos_h ? (int)layer.rel_pos_h->ne[1] : 0;

    // Get decomposed relative position tables
    auto rp_h = get_rel_pos(H, H, rel_h.data(), rel_h_L, hd);  // (H, H, hd)
    auto rp_w = get_rel_pos(W, W, rel_w.data(), rel_h_L, hd);  // (W, W, hd)

    // Process each batch element
    for (int bi = 0; bi < B; bi++) {
        float* h = hidden + bi * N * C;  // (N, C) = (H*W, C)

        // Compute QKV: (N, C) × (C, 3C)^T → (N, 3C)
        std::vector<float> qkv(N * 3 * C);
        for (int n = 0; n < N; n++) {
            linear_cpu(h + n * C, qkv.data() + n * 3 * C, C, 3 * C,
                       qkv_w.data(), qkv_b.data());
        }

        // Split into Q, K, V: each (N, C)
        // Reshape to (n_heads, N, hd) for attention
        // Attention: (B*nH, N, hd) × (B*nH, hd, N) → (B*nH, N, N)
        std::vector<float> attn(n_heads * N * N, 0.0f);
        float scale = 1.0f / sqrtf((float)hd);

        for (int head = 0; head < n_heads; head++) {
            for (int qi = 0; qi < N; qi++) {
                const float* q_ptr = qkv.data() + qi * 3 * C + head * hd;  // Q
                for (int ki = 0; ki < N; ki++) {
                    const float* k_ptr = qkv.data() + ki * 3 * C + C + head * hd;  // K
                    float s = 0;
                    for (int d = 0; d < hd; d++)
                        s += q_ptr[d] * k_ptr[d];
                    attn[head * N * N + qi * N + ki] = s * scale;
                }
            }
        }

        // Add decomposed relative position bias
        // For each head h, position (qy, qx) attending to (ky, kx):
        //   bias = sum_d( q[h, qy*W+qx, d] * rp_h[qy, ky, d] )
        //        + sum_d( q[h, qy*W+qx, d] * rp_w[qx, kx, d] )
        for (int head = 0; head < n_heads; head++) {
            for (int qy = 0; qy < H; qy++) {
                for (int qx = 0; qx < W; qx++) {
                    int qi = qy * W + qx;
                    const float* q_ptr = qkv.data() + qi * 3 * C + head * hd;

                    // Compute rel_h contribution: q @ rp_h[qy, :, :] → (H,)
                    std::vector<float> rh_scores(H, 0.0f);
                    for (int ky = 0; ky < H; ky++) {
                        float s = 0;
                        for (int d = 0; d < hd; d++)
                            s += q_ptr[d] * rp_h[(qy * H + ky) * hd + d];
                        rh_scores[ky] = s;
                    }
                    // Compute rel_w contribution: q @ rp_w[qx, :, :] → (W,)
                    std::vector<float> rw_scores(W, 0.0f);
                    for (int kx = 0; kx < W; kx++) {
                        float s = 0;
                        for (int d = 0; d < hd; d++)
                            s += q_ptr[d] * rp_w[(qx * W + kx) * hd + d];
                        rw_scores[kx] = s;
                    }
                    // Add to attention: bias[qi, ky*W+kx] = rh[ky] + rw[kx]
                    for (int ky = 0; ky < H; ky++) {
                        for (int kx = 0; kx < W; kx++) {
                            int ki = ky * W + kx;
                            attn[head * N * N + qi * N + ki] += rh_scores[ky] + rw_scores[kx];
                        }
                    }
                }
            }
        }

        // Softmax per head per query
        for (int head = 0; head < n_heads; head++) {
            for (int qi = 0; qi < N; qi++) {
                float* row = attn.data() + head * N * N + qi * N;
                float maxv = *std::max_element(row, row + N);
                float sum = 0;
                for (int ki = 0; ki < N; ki++) {
                    row[ki] = expf(row[ki] - maxv);
                    sum += row[ki];
                }
                for (int ki = 0; ki < N; ki++) row[ki] /= sum;
            }
        }

        // Attention output: attn @ V → (n_heads, N, hd)
        // Then reshape to (N, C) and project
        std::vector<float> attn_out(N * C, 0.0f);
        for (int head = 0; head < n_heads; head++) {
            for (int qi = 0; qi < N; qi++) {
                for (int ki = 0; ki < N; ki++) {
                    float a = attn[head * N * N + qi * N + ki];
                    const float* v_ptr = qkv.data() + ki * 3 * C + 2 * C + head * hd;
                    for (int d = 0; d < hd; d++)
                        attn_out[qi * C + head * hd + d] += a * v_ptr[d];
                }
            }
        }

        // Output projection: (N, C) → (N, C)
        std::vector<float> proj_out(N * C);
        for (int n = 0; n < N; n++) {
            linear_cpu(attn_out.data() + n * C, proj_out.data() + n * C, C, C,
                       proj_w.data(), proj_b.data());
        }

        // Write back
        memcpy(h, proj_out.data(), N * C * sizeof(float));
    }
}

// Full ViT layer: pre-LN → attn (with windowing) → residual → pre-LN → MLP → residual
// hidden: shape (B, H, W, C) stored as flat array, row-major
static void run_vit_layer(const vit_layer& layer,
                           float* hidden, int B, int H, int W, int C,
                           int n_heads, int ws) {
    int N = H * W;

    auto ln1_w = to_f32(layer.ln1_w);
    auto ln1_b = to_f32(layer.ln1_b);
    auto ln2_w = to_f32(layer.ln2_w);
    auto ln2_b = to_f32(layer.ln2_b);
    auto mlp1_w = to_f32(layer.mlp_lin1_w);
    auto mlp1_b = to_f32(layer.mlp_lin1_b);
    auto mlp2_w = to_f32(layer.mlp_lin2_w);
    auto mlp2_b = to_f32(layer.mlp_lin2_b);
    int mlp_dim = (int)layer.mlp_lin1_w->ne[1];

    for (int bi = 0; bi < B; bi++) {
        float* h = hidden + bi * N * C;

        // Save residual
        std::vector<float> residual(h, h + N * C);

        // Pre-LN (over last dim C, for each spatial position)
        for (int n = 0; n < N; n++)
            layernorm_cpu(h + n * C, h + n * C, C,
                          ln1_w.data(), ln1_b.data(), 1e-6f);

        // Window partition (if windowed)
        int aH = H, aW = W, nB = 1;
        std::vector<float> windowed;
        int pad_h = 0, pad_w = 0;
        if (layer.window_size > 0) {
            // Pad to multiples of ws
            pad_h = (ws - H % ws) % ws;
            pad_w = (ws - W % ws) % ws;
            int pH = H + pad_h, pW = W + pad_w;

            // Pad: (H, W, C) → (pH, pW, C)
            std::vector<float> padded(pH * pW * C, 0.0f);
            for (int y = 0; y < H; y++)
                memcpy(padded.data() + y * pW * C, h + y * W * C, W * C * sizeof(float));

            // Reshape to windows: (nH, ws, nW, ws, C) → (nH*nW, ws, ws, C)
            int nH = pH / ws, nW = pW / ws;
            int nWindows = nH * nW;
            windowed.resize(nWindows * ws * ws * C);
            for (int wh = 0; wh < nH; wh++) {
                for (int ww = 0; ww < nW; ww++) {
                    for (int y = 0; y < ws; y++) {
                        for (int x = 0; x < ws; x++) {
                            int src_y = wh * ws + y;
                            int src_x = ww * ws + x;
                            int src_idx = src_y * pW * C + src_x * C;
                            int dst_idx = (wh * nW + ww) * ws * ws * C + y * ws * C + x * C;
                            memcpy(windowed.data() + dst_idx, padded.data() + src_idx, C * sizeof(float));
                        }
                    }
                }
            }
            aH = ws; aW = ws; nB = nWindows;
        } else {
            windowed.assign(h, h + N * C);
            nB = 1;
        }

        // Attention
        vit_layer_attn(layer, windowed.data(), nB, aH, aW, C, n_heads);

        // Window unpartition
        if (layer.window_size > 0) {
            int pH = H + pad_h, pW = W + pad_w;
            int nH = pH / ws, nW = pW / ws;
            std::vector<float> unpadded(pH * pW * C);
            for (int wh = 0; wh < nH; wh++) {
                for (int ww = 0; ww < nW; ww++) {
                    for (int y = 0; y < ws; y++) {
                        for (int x = 0; x < ws; x++) {
                            int dst_y = wh * ws + y;
                            int dst_x = ww * ws + x;
                            int dst_idx = dst_y * pW * C + dst_x * C;
                            int src_idx = (wh * nW + ww) * ws * ws * C + y * ws * C + x * C;
                            memcpy(unpadded.data() + dst_idx, windowed.data() + src_idx, C * sizeof(float));
                        }
                    }
                }
            }
            // Crop to original size
            for (int y = 0; y < H; y++)
                memcpy(h + y * W * C, unpadded.data() + y * pW * C, W * C * sizeof(float));
        } else {
            memcpy(h, windowed.data(), N * C * sizeof(float));
        }

        // Residual add
        for (int i = 0; i < N * C; i++) h[i] += residual[i];

        // Pre-LN for MLP
        std::vector<float> ln2_out(N * C);
        for (int n = 0; n < N; n++)
            layernorm_cpu(h + n * C, ln2_out.data() + n * C, C,
                          ln2_w.data(), ln2_b.data(), 1e-6f);

        // MLP: Linear → GELU → Linear
        std::vector<float> mlp_out(N * C);
        for (int n = 0; n < N; n++) {
            std::vector<float> up(mlp_dim);
            linear_cpu(ln2_out.data() + n * C, up.data(), C, mlp_dim,
                       mlp1_w.data(), mlp1_b.data());
            for (int i = 0; i < mlp_dim; i++) up[i] = gelu(up[i]);
            linear_cpu(up.data(), mlp_out.data() + n * C, mlp_dim, C,
                       mlp2_w.data(), mlp2_b.data());
        }

        // Residual add
        for (int i = 0; i < N * C; i++) h[i] += mlp_out[i];
    }
}

// Run full encoder: patch embed → 12 ViT layers → neck → projector
static void run_encoder(ppformulanet_l_ocr_context* ctx,
                         const float* rgb_chw, int imgH, int imgW) {
    const auto& hp = ctx->hparams;
    int C = hp.enc_hidden;       // 768
    int PS = hp.patch_size;      // 16
    int nP = hp.n_patches;       // 48

    // 1. Patch embedding: Conv2d(3→768, k=16, s=16) on (3, 768, 768)
    //    Output: (768, 48, 48) → transpose to (48, 48, 768)
    auto pe_w = to_f32(ctx->patch_embed_w);
    auto pe_b = to_f32(ctx->patch_embed_b);
    std::vector<float> patches_nchw(C * nP * nP);
    conv2d_cpu(rgb_chw, patches_nchw.data(), pe_w.data(), pe_b.data(),
               3, C, imgH, imgW, PS, PS, PS, 0, 1);

    // Transpose (C, H, W) → (H, W, C) for ViT layers
    std::vector<float> hidden(nP * nP * C);
    for (int y = 0; y < nP; y++)
        for (int x = 0; x < nP; x++)
            for (int c = 0; c < C; c++)
                hidden[(y * nP + x) * C + c] = patches_nchw[c * nP * nP + y * nP + x];

    // 2. Add positional embedding (1, 48, 48, 768) → squeeze batch
    auto pos = to_f32(ctx->pos_embed);
    for (int i = 0; i < nP * nP * C; i++) hidden[i] += pos[i];

    fprintf(stderr, "ppfnl: patch_embed done, shape=(%d, %d, %d)\n", nP, nP, C);

    // 3. ViT layers
    for (int li = 0; li < hp.enc_layers; li++) {
        auto& layer = ctx->enc_layers[li];
        run_vit_layer(layer, hidden.data(), 1, nP, nP, C,
                      hp.enc_heads, hp.window_size);
        if (li < 3 || li == hp.enc_layers - 1) {
            fprintf(stderr, "ppfnl: enc_layer_%d done (%s)\n",
                    li, layer.is_global ? "global" : "windowed");
        }
    }

    // 4. Neck: permute (H, W, C) → (C, H, W), then Conv1×1 + LN2d + Conv3×3 + LN2d
    int nC = hp.output_channels;  // 256
    std::vector<float> nchw(C * nP * nP);
    for (int y = 0; y < nP; y++)
        for (int x = 0; x < nP; x++)
            for (int c = 0; c < C; c++)
                nchw[c * nP * nP + y * nP + x] = hidden[(y * nP + x) * C + c];

    // Conv1×1 (768 → 256)
    auto nc1_w = to_f32(ctx->neck_conv1_w);
    std::vector<float> neck1(nC * nP * nP);
    conv2d_cpu(nchw.data(), neck1.data(), nc1_w.data(), nullptr,
               C, nC, nP, nP, 1, 1, 1, 0, 1);

    // LayerNorm2d
    auto nln1_w = to_f32(ctx->neck_ln1_w);
    auto nln1_b = to_f32(ctx->neck_ln1_b);
    layernorm2d_cpu(neck1.data(), neck1.data(), nC, nP, nP,
                    nln1_w.data(), nln1_b.data());

    // Conv3×3 (256 → 256, padding=1)
    auto nc2_w = to_f32(ctx->neck_conv2_w);
    std::vector<float> neck2(nC * nP * nP);
    conv2d_cpu(neck1.data(), neck2.data(), nc2_w.data(), nullptr,
               nC, nC, nP, nP, 3, 3, 1, 1, 1);

    // LayerNorm2d
    auto nln2_w = to_f32(ctx->neck_ln2_w);
    auto nln2_b = to_f32(ctx->neck_ln2_b);
    layernorm2d_cpu(neck2.data(), neck2.data(), nC, nP, nP,
                    nln2_w.data(), nln2_b.data());

    fprintf(stderr, "ppfnl: neck done, shape=(%d, %d, %d)\n", nC, nP, nP);

    // 5. Multi-modal projector
    // Conv3×3(256→512, s=2, p=1) → Conv3×3(512→1024, s=2, p=1)
    // → flatten(2).transpose(1,2) → Linear(1024→1024) → Linear(1024→512)
    int mid_ch = 512, out_ch = 1024;  // from config
    int h1 = nP / 2, w1 = nP / 2;    // 24×24 after stride 2

    auto pc1_w = to_f32(ctx->proj_conv1_w);
    std::vector<float> pc1(mid_ch * h1 * w1);
    conv2d_cpu(neck2.data(), pc1.data(), pc1_w.data(), nullptr,
               nC, mid_ch, nP, nP, 3, 3, 2, 1, 1);

    int h2 = h1 / 2, w2 = w1 / 2;   // 12×12 after stride 2
    auto pc2_w = to_f32(ctx->proj_conv2_w);
    std::vector<float> pc2(out_ch * h2 * w2);
    conv2d_cpu(pc1.data(), pc2.data(), pc2_w.data(), nullptr,
               mid_ch, out_ch, h1, w1, 3, 3, 2, 1, 1);

    // Flatten (1024, 12, 12) → (144, 1024) then linear projections
    int n_tokens = h2 * w2;  // 144
    std::vector<float> flat(n_tokens * out_ch);
    for (int y = 0; y < h2; y++)
        for (int x = 0; x < w2; x++)
            for (int c = 0; c < out_ch; c++)
                flat[(y * w2 + x) * out_ch + c] = pc2[c * h2 * w2 + y * w2 + x];

    auto pl1_w = to_f32(ctx->proj_linear1_w);
    auto pl1_b = to_f32(ctx->proj_linear1_b);
    auto pl2_w = to_f32(ctx->proj_linear2_w);
    auto pl2_b = to_f32(ctx->proj_linear2_b);

    int dec_dim = hp.dec_d_model;  // 512
    ctx->proj_out.resize(n_tokens * dec_dim);
    ctx->n_enc_tokens = n_tokens;
    ctx->proj_dim = dec_dim;

    for (int n = 0; n < n_tokens; n++) {
        std::vector<float> l1(out_ch);
        linear_cpu(flat.data() + n * out_ch, l1.data(), out_ch, out_ch,
                   pl1_w.data(), pl1_b.data());
        linear_cpu(l1.data(), ctx->proj_out.data() + n * dec_dim, out_ch, dec_dim,
                   pl2_w.data(), pl2_b.data());
    }

    fprintf(stderr, "ppfnl: projector done, shape=(%d, %d)\n", n_tokens, dec_dim);
}

// ---------------------------------------------------------------------------
// Decoder: precompute cross-attention K/V from projected encoder output
// ---------------------------------------------------------------------------

static void precompute_cross_kv(ppformulanet_l_ocr_context* ctx) {
    const auto& hp = ctx->hparams;
    int D = hp.dec_d_model;
    int n_enc = ctx->n_enc_tokens;

    ctx->cross_k_cache.resize(hp.dec_layers);
    ctx->cross_v_cache.resize(hp.dec_layers);

    for (int li = 0; li < hp.dec_layers; li++) {
        auto& l = ctx->dec_layers_vec[li];
        auto kw = to_f32(l.cross_k_w);
        auto kb = to_f32(l.cross_k_b);
        auto vw = to_f32(l.cross_v_w);
        auto vb = to_f32(l.cross_v_b);

        ctx->cross_k_cache[li].resize(n_enc * D);
        ctx->cross_v_cache[li].resize(n_enc * D);

        for (int t = 0; t < n_enc; t++) {
            linear_cpu(ctx->proj_out.data() + t * D,
                       ctx->cross_k_cache[li].data() + t * D,
                       D, D, kw.data(), kb.data());
            linear_cpu(ctx->proj_out.data() + t * D,
                       ctx->cross_v_cache[li].data() + t * D,
                       D, D, vw.data(), vb.data());
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder: single autoregressive step
// ---------------------------------------------------------------------------

static std::vector<float> decoder_step(
    ppformulanet_l_ocr_context* ctx, int token, int step,
    std::vector<std::vector<float>>& kv_k,
    std::vector<std::vector<float>>& kv_v) {

    const auto& hp = ctx->hparams;
    int D = hp.dec_d_model;
    int V = hp.vocab_size;
    int n_enc = ctx->n_enc_tokens;
    float scale = sqrtf((float)D);

    // Token embedding (scaled) + position embedding
    auto tok_data = to_f32(ctx->tok_embed);
    auto pos_data = to_f32(ctx->pos_embed_dec);

    std::vector<float> x(D);
    for (int i = 0; i < D; i++)
        x[i] = tok_data[token * D + i] * scale + pos_data[(step + 2) * D + i];

    // Embedding LayerNorm
    layernorm_cpu(x.data(), x.data(), D, ctx->embed_ln_w, ctx->embed_ln_b);

    // Decoder layers
    for (int li = 0; li < hp.dec_layers; li++) {
        auto& l = ctx->dec_layers_vec[li];

        // --- Self-attention (PRE-LN) ---
        std::vector<float> residual(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.self_ln_w, l.self_ln_b);

        std::vector<float> q(D), k(D), v(D);
        linear_cpu(x.data(), q.data(), D, D, l.self_q_w, l.self_q_b);
        linear_cpu(x.data(), k.data(), D, D, l.self_k_w, l.self_k_b);
        linear_cpu(x.data(), v.data(), D, D, l.self_v_w, l.self_v_b);

        kv_k[li].insert(kv_k[li].end(), k.begin(), k.end());
        kv_v[li].insert(kv_v[li].end(), v.begin(), v.end());
        int n_kv = (int)(kv_k[li].size() / D);

        std::vector<float> sa(D);
        mha_1q_cpu(q.data(), kv_k[li].data(), kv_v[li].data(),
                   sa.data(), n_kv, D, hp.dec_heads);

        std::vector<float> sa_proj(D);
        linear_cpu(sa.data(), sa_proj.data(), D, D, l.self_out_w, l.self_out_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + sa_proj[i];

        // --- Cross-attention (PRE-LN) ---
        residual.assign(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.cross_ln_w, l.cross_ln_b);

        std::vector<float> cq(D);
        linear_cpu(x.data(), cq.data(), D, D, l.cross_q_w, l.cross_q_b);

        std::vector<float> ca(D);
        mha_1q_cpu(cq.data(), ctx->cross_k_cache[li].data(),
                   ctx->cross_v_cache[li].data(),
                   ca.data(), n_enc, D, hp.dec_heads);

        std::vector<float> ca_proj(D);
        linear_cpu(ca.data(), ca_proj.data(), D, D, l.cross_out_w, l.cross_out_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + ca_proj[i];

        // --- FFN (PRE-LN) ---
        residual.assign(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.ff_ln_w, l.ff_ln_b);

        std::vector<float> ff_up(hp.dec_ffn_dim);
        linear_cpu(x.data(), ff_up.data(), D, hp.dec_ffn_dim, l.ff_up_w, l.ff_up_b);
        for (int i = 0; i < hp.dec_ffn_dim; i++) ff_up[i] = gelu(ff_up[i]);
        std::vector<float> ff_down(D);
        linear_cpu(ff_up.data(), ff_down.data(), hp.dec_ffn_dim, D, l.ff_down_w, l.ff_down_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + ff_down[i];
    }

    // Final LayerNorm
    layernorm_cpu(x.data(), x.data(), D, ctx->final_ln_w, ctx->final_ln_b);

    // LM head (no bias)
    std::vector<float> logits(V);
    linear_cpu(x.data(), logits.data(), D, V, ctx->lm_head_w, nullptr);

    return logits;
}

static std::vector<int> greedy_decode(ppformulanet_l_ocr_context* ctx) {
    const auto& hp = ctx->hparams;
    const int max_steps = std::min(hp.max_seq_len, 512);

    std::vector<int> tokens;
    int tok = hp.decoder_start_token;

    std::vector<std::vector<float>> kv_k(hp.dec_layers);
    std::vector<std::vector<float>> kv_v(hp.dec_layers);

    for (int step = 0; step < max_steps; step++) {
        auto logits = decoder_step(ctx, tok, step, kv_k, kv_v);

        int best = 0;
        float best_s = logits[0];
        for (int v = 1; v < hp.vocab_size; v++)
            if (logits[v] > best_s) { best_s = logits[v]; best = v; }

        if (step < 5) {
            fprintf(stderr, "ppfnl: dec step %d: tok=%d best=%d (%.3f)\n",
                    step, tok, best, best_s);
        }

        if (best == hp.eos_token || best == hp.pad_token) break;
        tokens.push_back(best);
        tok = best;
    }

    return tokens;
}

// Detokenize: concatenate vocab strings, replacing GPT-2 BPE Ġ (U+0120) with space
static void detokenize(ppformulanet_l_ocr_context* ctx, const std::vector<int>& tokens) {
    ctx->result_buf.clear();
    for (int tok : tokens) {
        if (tok < 0 || tok >= (int)ctx->vocab.size()) continue;
        const auto& s = ctx->vocab[tok];
        for (size_t i = 0; i < s.size(); ) {
            if (i + 1 < s.size() &&
                (unsigned char)s[i] == 0xC4 && (unsigned char)s[i+1] == 0xA0) {
                ctx->result_buf += ' ';
                i += 2;
            } else {
                ctx->result_buf += s[i];
                i++;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Init / Free / API
// ---------------------------------------------------------------------------

ppformulanet_l_ocr_context* ppformulanet_l_ocr_init(const char* model_path, int n_threads) {
    auto ctx = std::make_unique<ppformulanet_l_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "ppfnl: can't open %s\n", model_path);
        return nullptr;
    }

    auto& hp = ctx->hparams;
    hp.image_size       = core_gguf::kv_u32(gctx, "ppfnl.encoder.image_size", 768);
    hp.patch_size       = core_gguf::kv_u32(gctx, "ppfnl.encoder.patch_size", 16);
    hp.enc_hidden       = core_gguf::kv_u32(gctx, "ppfnl.encoder.hidden_size", 768);
    hp.enc_layers       = core_gguf::kv_u32(gctx, "ppfnl.encoder.n_layers", 12);
    hp.enc_heads        = core_gguf::kv_u32(gctx, "ppfnl.encoder.n_heads", 12);
    hp.enc_mlp_dim      = core_gguf::kv_u32(gctx, "ppfnl.encoder.mlp_dim", 3072);
    hp.window_size      = core_gguf::kv_u32(gctx, "ppfnl.encoder.window_size", 14);
    hp.n_patches        = hp.image_size / hp.patch_size;
    hp.output_channels  = core_gguf::kv_u32(gctx, "ppfnl.encoder.output_channels", 256);

    hp.dec_layers       = core_gguf::kv_u32(gctx, "ppfnl.decoder.decoder_layers", 8);
    hp.dec_heads        = core_gguf::kv_u32(gctx, "ppfnl.decoder.decoder_attention_heads", 16);
    hp.dec_d_model      = core_gguf::kv_u32(gctx, "ppfnl.decoder.d_model", 512);
    hp.dec_ffn_dim      = core_gguf::kv_u32(gctx, "ppfnl.decoder.decoder_ffn_dim", 2048);
    hp.vocab_size       = core_gguf::kv_u32(gctx, "ppfnl.decoder.vocab_size", 50000);
    hp.max_seq_len      = core_gguf::kv_u32(gctx, "ppfnl.decoder.max_position_embeddings", 1024);
    hp.bos_token        = core_gguf::kv_u32(gctx, "ppfnl.decoder.bos_token_id", 0);
    hp.eos_token        = core_gguf::kv_u32(gctx, "ppfnl.decoder.eos_token_id", 2);
    hp.pad_token        = core_gguf::kv_u32(gctx, "ppfnl.decoder.pad_token_id", 1);
    hp.decoder_start_token = core_gguf::kv_u32(gctx, "ppfnl.decoder.decoder_start_token_id", 0);

    // Read global attention indexes
    ctx->global_attn_indexes = core_gguf::kv_i32_array(gctx, "ppfnl.encoder.global_attn_indexes");
    if (ctx->global_attn_indexes.empty()) {
        ctx->global_attn_indexes = {2, 5, 8, 11};  // default
    }

    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "ppfnl: encoder=%dL/%dH/%d (ws=%d) decoder=%dL/%dH/%d vocab=%d(%zu)\n",
            hp.enc_layers, hp.enc_heads, hp.enc_hidden, hp.window_size,
            hp.dec_layers, hp.dec_heads, hp.dec_d_model,
            hp.vocab_size, ctx->vocab.size());

    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "ppfnl", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        return nullptr;
    }
    fprintf(stderr, "ppfnl: %zu tensors loaded\n", ctx->wl.tensors.size());

    // Create scheduler for graph compute
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 8192, false, false);

    map_tensors(ctx.get());

    // Precompute rel_pos lookup tables (once at init, not per-inference)
    ctx->rp_h_per_layer.resize(hp.enc_layers);
    ctx->rp_w_per_layer.resize(hp.enc_layers);
    for (int li = 0; li < hp.enc_layers; li++) {
        auto& layer = ctx->enc_layers[li];
        int aH = layer.is_global ? hp.n_patches : hp.window_size;
        int head_dim = hp.enc_hidden / hp.enc_heads;
        auto rh = to_f32(layer.rel_pos_h);
        auto rw = to_f32(layer.rel_pos_w);
        int rel_L = layer.rel_pos_h ? (int)layer.rel_pos_h->ne[1] : 0;
        ctx->rp_h_per_layer[li] = get_rel_pos(aH, aH, rh.data(), rel_L, head_dim);
        ctx->rp_w_per_layer[li] = get_rel_pos(aH, aH, rw.data(), rel_L, head_dim);
    }

    return ctx.release();
}

void ppformulanet_l_ocr_free(ppformulanet_l_ocr_context* ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const ppformulanet_l_ocr_hparams* ppformulanet_l_ocr_get_hparams(
    const ppformulanet_l_ocr_context* ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

const char* ppformulanet_l_ocr_recognize(ppformulanet_l_ocr_context* ctx,
                                          const float* pixels,
                                          int width, int height, int* out_len) {
    if (!ctx || !pixels) return nullptr;
    const int S = ctx->hparams.image_size;

    const float MEAN = 0.7931f;
    const float STD  = 0.1738f;

    float scale = std::min((float)S / width, (float)S / height);
    int new_w = std::min((int)(width * scale), S);
    int new_h = std::min((int)(height * scale), S);
    int pad_left = (S - new_w) / 2;
    int pad_top  = (S - new_h) / 2;

    float black_norm = (0.0f - MEAN) / STD;
    std::vector<float> rgb(3 * S * S, black_norm);

    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            float src_x = (float)x / scale;
            float src_y = (float)y / scale;
            int ox = std::min((int)src_x, width - 1);
            int oy = std::min((int)src_y, height - 1);
            float v = pixels[oy * width + ox];
            float normed = (v - MEAN) / STD;
            int dy = pad_top + y;
            int dx = pad_left + x;
            for (int c = 0; c < 3; c++)
                rgb[c * S * S + dy * S + dx] = normed;
        }
    }

    run_encoder_graph(ctx, rgb.data(), S, S);
    precompute_cross_kv(ctx);
    auto tokens = greedy_decode(ctx);
    detokenize(ctx, tokens);

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char* ppformulanet_l_ocr_recognize_raw(ppformulanet_l_ocr_context* ctx,
                                              const uint8_t* pixel_bytes,
                                              int width, int height, int channels,
                                              int* out_len) {
    if (!ctx || !pixel_bytes) return nullptr;

    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        if (channels == 1) {
            gray[i] = pixel_bytes[i] / 255.0f;
        } else if (channels == 3) {
            gray[i] = (0.299f * pixel_bytes[i*3] + 0.587f * pixel_bytes[i*3+1]
                       + 0.114f * pixel_bytes[i*3+2]) / 255.0f;
        } else if (channels == 4) {
            gray[i] = (0.299f * pixel_bytes[i*4] + 0.587f * pixel_bytes[i*4+1]
                       + 0.114f * pixel_bytes[i*4+2]) / 255.0f;
        }
    }

    return ppformulanet_l_ocr_recognize(ctx, gray.data(), width, height, out_len);
}

const char* ppformulanet_l_ocr_recognize_chw(ppformulanet_l_ocr_context* ctx,
                                              const float* chw_data,
                                              int* out_len) {
    if (!ctx || !chw_data) return nullptr;
    const int S = ctx->hparams.image_size;

    run_encoder_graph(ctx, chw_data, S, S);
    precompute_cross_kv(ctx);
    auto tokens = greedy_decode(ctx);
    detokenize(ctx, tokens);

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float* ppformulanet_l_ocr_get_encoder_output(
    const ppformulanet_l_ocr_context* ctx,
    int* out_n_tokens, int* out_hidden) {
    if (!ctx || ctx->proj_out.empty()) return nullptr;
    if (out_n_tokens) *out_n_tokens = ctx->n_enc_tokens;
    if (out_hidden) *out_hidden = ctx->proj_dim;
    return ctx->proj_out.data();
}
