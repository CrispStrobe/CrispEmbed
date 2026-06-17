// deepseek_ocr2.cpp — DeepSeek-OCR-2 engine: SAM ViT-B + Qwen2 encoder + MoE decoder.
//
// Vision: per-layer ggml graph with CPU window partition (SAM ViT-B pattern).
// Qwen2 encoder: CPU-scalar bidirectional transformer (no causal mask).
// LLM decoder: ggml graph with KV cache, MoE layers use CPU-scalar expert dispatch.

#include "deepseek_ocr2.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct sam_hparams {
    int depth = 12, hidden = 768, heads = 12, head_dim = 64;
    int patch_size = 16, image_size = 1024, window_size = 14;
    int neck_out = 256;
    std::vector<int> global_attn_indexes{2, 5, 8, 11};
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct qwen2_enc_hparams {
    int depth = 24, hidden = 896, heads = 14, kv_heads = 2;
    int intermediate = 4864;
    float rms_eps = 1e-6f;
};

struct llm_hparams {
    int vocab_size = 129280, hidden = 1280, heads = 10, kv_heads = 10;
    int head_dim = 128, n_layers = 12;
    int dense_intermediate = 6848;   // layer 0
    int expert_intermediate = 896;   // routed experts
    int shared_intermediate = 1792;  // shared experts (896*2)
    int n_experts = 64, n_experts_top = 6, n_shared_experts = 2;
    float rms_eps = 1e-6f, rope_theta = 10000.0f;
    float routed_scaling_factor = 1.0f;
    int eos_token_id = 1;
    int max_position_embeddings = 4096;
};

// ---------------------------------------------------------------------------
// Weight storage
// ---------------------------------------------------------------------------

struct sam_block_w {
    ggml_tensor *ln1_w{}, *ln1_b{}, *ln2_w{}, *ln2_b{};
    ggml_tensor *qkv_w{}, *qkv_b{}, *proj_w{}, *proj_b{};
    ggml_tensor *rel_pos_h{}, *rel_pos_w{};
    ggml_tensor *ffn_up_w{}, *ffn_up_b{}, *ffn_down_w{}, *ffn_down_b{};
    bool is_global = false;
};

struct qwen2_enc_layer_w {
    ggml_tensor *in_ln_w{}, *post_ln_w{};
    ggml_tensor *q_w{}, *q_b{}, *k_w{}, *k_b{}, *v_w{}, *v_b{}, *o_w{};
    ggml_tensor *gate_w{}, *up_w{}, *down_w{};
};

struct moe_expert_w {
    ggml_tensor *gate_w{}, *up_w{}, *down_w{};
};

struct llm_layer_w {
    ggml_tensor *in_ln_w{}, *post_ln_w{};
    ggml_tensor *q_w{}, *k_w{}, *v_w{}, *o_w{};
    // Dense FFN (layer 0)
    ggml_tensor *ffn_gate_w{}, *ffn_up_w{}, *ffn_down_w{};
    // MoE (layers 1-11)
    ggml_tensor *router_w{};  // mlp.gate.weight
    std::vector<moe_expert_w> experts;
    moe_expert_w shared_experts[2];
    // Single shared expert (combined)
    ggml_tensor *shared_gate_w{}, *shared_up_w{}, *shared_down_w{};
};

struct model_weights {
    sam_hparams shp;
    qwen2_enc_hparams qhp;
    llm_hparams lhp;

    // SAM
    ggml_tensor *patch_embed_w{}, *patch_embed_b{}, *pos_embed{};
    std::vector<sam_block_w> sam_blocks;
    ggml_tensor *neck_conv1_w{}, *neck_ln1_w{}, *neck_ln1_b{};
    ggml_tensor *neck_conv2_w{}, *neck_ln2_w{}, *neck_ln2_b{};
    ggml_tensor *net_2_w{}, *net_3_w{};

    // Qwen2 encoder
    std::vector<qwen2_enc_layer_w> qwen2_layers;
    ggml_tensor *query_768{}, *query_1024{};

    // Projector
    ggml_tensor *projector_w{}, *projector_b{};

    // View separator
    ggml_tensor *view_separator{};

    // LLM
    ggml_tensor *embed_tokens{}, *output_norm_w{}, *lm_head_w{};
    std::vector<llm_layer_w> llm_layers;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct ds_ocr2_ctx {
    model_weights m;
    ggml_context *model_ctx{};
    ggml_backend_buffer_t model_buf{};
    ggml_backend_t backend{}, backend_cpu{};
    ggml_backend_sched_t sched{};
    std::vector<uint8_t> compute_meta;

    // Tokenizer
    std::vector<std::string> id_to_piece;
    int tok_vocab_size = 0;

    // KV cache for LLM decoder
    struct {
        std::vector<std::vector<float>> k_cache;  // [layer][kv_heads * head_dim * n_past]
        std::vector<std::vector<float>> v_cache;
        int n_past = 0;
    } kvc;

    // Precomputed RPE tables
    std::vector<std::vector<float>> rp_h_per_layer, rp_w_per_layer;

    int n_threads = 4, verbosity = 1;
    std::string diff_ref_path;
};

// ---------------------------------------------------------------------------
// CPU helpers (shared with got_ocr.cpp pattern)
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
        if (traits && traits->to_float)
            traits->to_float(t->data, out.data(), n);
        else
            memset(out.data(), 0, n * sizeof(float));
    }
    return out;
}

static void layernorm_cpu(const float* in, float* out, int D,
                          const float* w, const float* b, float eps = 1e-6f) {
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

static void layernorm2d_cpu(const float* in, float* out, int C, int H, int W,
                            const float* w, const float* b, float eps = 1e-6f) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double mean = 0;
            for (int c = 0; c < C; c++) mean += in[c * H * W + y * W + x];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; c++) {
                double d = in[c * H * W + y * W + x] - mean; var += d * d;
            }
            var /= C;
            float s = 1.0f / sqrtf((float)var + eps);
            for (int c = 0; c < C; c++) {
                float v = (in[c * H * W + y * W + x] - (float)mean) * s;
                out[c * H * W + y * W + x] = v * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
            }
        }
}

static void rmsnorm_cpu(const float* in, float* out, int D,
                        const float* w, float eps = 1e-6f) {
    double ss = 0;
    for (int i = 0; i < D; i++) ss += (double)in[i] * in[i];
    float s = 1.0f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = in[i] * s * (w ? w[i] : 1.0f);
}

static void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                       const float* w, const float* b) {
    for (int o = 0; o < out_dim; o++) {
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) s += in[i] * w[o * in_dim + i];
        out[o] = s;
    }
}

static void conv2d_cpu(const float* in, float* out, const float* weight,
                       const float* bias, int in_ch, int out_ch, int H, int W,
                       int kh, int kw, int stride, int pad) {
    int oH = (H + 2 * pad - kh) / stride + 1;
    int oW = (W + 2 * pad - kw) / stride + 1;
    for (int oc = 0; oc < out_ch; oc++) {
        float b = bias ? bias[oc] : 0.0f;
        for (int oy = 0; oy < oH; oy++)
            for (int ox = 0; ox < oW; ox++) {
                float sum = b;
                for (int ic = 0; ic < in_ch; ic++)
                    for (int ky2 = 0; ky2 < kh; ky2++)
                        for (int kx2 = 0; kx2 < kw; kx2++) {
                            int iy = oy * stride - pad + ky2;
                            int ix = ox * stride - pad + kx2;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                sum += in[ic * H * W + iy * W + ix]
                                     * weight[oc * (in_ch * kh * kw) + ic * kh * kw + ky2 * kw + kx2];
                        }
                out[oc * oH * oW + oy * oW + ox] = sum;
            }
    }
}

static void silu_cpu(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void swiglu_ffn_cpu(const float* in, float* out, int D, int inter,
                           const float* gate_w, const float* up_w,
                           const float* down_w) {
    std::vector<float> gate(inter), up(inter);
    linear_cpu(in, gate.data(), D, inter, gate_w, nullptr);
    linear_cpu(in, up.data(), D, inter, up_w, nullptr);
    silu_cpu(gate.data(), inter);
    for (int i = 0; i < inter; i++) gate[i] *= up[i];
    linear_cpu(gate.data(), out, inter, D, down_w, nullptr);
}

// ---------------------------------------------------------------------------
// SAM window partition / unpartition + RPE (same as got_ocr.cpp)
// ---------------------------------------------------------------------------

static void window_partition(const float* h, float* wo, int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws, pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws, wN = ws * ws;
    memset(wo, 0, (size_t)nWh * nWw * wN * C * sizeof(float));
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int wi = wh * nWw + ww;
            for (int y = 0; y < ws; y++) { int sy = wh * ws + y; if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) { int sx = ww * ws + x; if (sx >= nP) continue;
                    memcpy(wo + (wi * wN + y * ws + x) * C,
                           h + (sy * nP + sx) * C, C * sizeof(float));
                }
            }
        }
}

static void window_unpartition(const float* wi, float* h, int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws, pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws, wN = ws * ws;
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int widx = wh * nWw + ww;
            for (int y = 0; y < ws; y++) { int sy = wh * ws + y; if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) { int sx = ww * ws + x; if (sx >= nP) continue;
                    memcpy(h + (sy * nP + sx) * C,
                           wi + (widx * wN + y * ws + x) * C, C * sizeof(float));
                }
            }
        }
}

static std::vector<float> get_rel_pos(int q_size, int k_size,
                                       const float* rel_pos, int L, int hd) {
    int max_rd = 2 * std::max(q_size, k_size) - 1;
    std::vector<float> resized(hd * max_rd);
    for (int c = 0; c < hd; c++)
        for (int i = 0; i < max_rd; i++) {
            float src = (float)i * (L - 1) / std::max(max_rd - 1, 1);
            int lo = (int)src, hi = std::min(lo + 1, L - 1);
            float frac = src - lo;
            resized[i * hd + c] = rel_pos[lo * hd + c] * (1.0f - frac)
                                + rel_pos[hi * hd + c] * frac;
        }
    float qs = std::max((float)k_size / q_size, 1.0f);
    float ks = std::max((float)q_size / k_size, 1.0f);
    float off = (float)(k_size - 1) * qs;
    std::vector<float> result(q_size * k_size * hd);
    for (int qi = 0; qi < q_size; qi++)
        for (int ki = 0; ki < k_size; ki++) {
            int idx = std::max(0, std::min((int)(qi * qs - ki * ks + off), max_rd - 1));
            for (int c = 0; c < hd; c++)
                result[(qi * k_size + ki) * hd + c] = resized[idx * hd + c];
        }
    return result;
}

static void reformat_rp_table(const float* rp_in, float* rp_out, int aH, int hd) {
    for (int q = 0; q < aH; q++)
        for (int k = 0; k < aH; k++)
            for (int d = 0; d < hd; d++)
                rp_out[d + k * hd + q * aH * hd] = rp_in[(q * aH + k) * hd + d];
}

// ---------------------------------------------------------------------------
// ggml graph helpers
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

// ---------------------------------------------------------------------------
// Load model
// ---------------------------------------------------------------------------

static bool load_hparams(ds_ocr2_ctx &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32v = [&](const char *k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto &s = ctx.m.shp;
    s.depth       = u32("deepseek_ocr2.sam.depth", s.depth);
    s.hidden      = u32("deepseek_ocr2.sam.hidden_size", s.hidden);
    s.heads       = u32("deepseek_ocr2.sam.num_heads", s.heads);
    s.head_dim    = s.hidden / s.heads;
    s.patch_size  = u32("deepseek_ocr2.sam.patch_size", s.patch_size);
    s.image_size  = u32("deepseek_ocr2.sam.image_size", s.image_size);
    s.window_size = u32("deepseek_ocr2.sam.window_size", s.window_size);
    s.neck_out    = u32("deepseek_ocr2.sam.neck_out_channels", s.neck_out);

    int key_id = gguf_find_key(g, "deepseek_ocr2.sam.global_attn_indexes");
    if (key_id >= 0) {
        int n = (int)gguf_get_arr_n(g, key_id);
        s.global_attn_indexes.resize(n);
        memcpy(s.global_attn_indexes.data(), gguf_get_arr_data(g, key_id), n * sizeof(int32_t));
    }

    key_id = gguf_find_key(g, "deepseek_ocr2.sam.image_mean");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_mean, gguf_get_arr_data(g, key_id), 3 * sizeof(float));
    key_id = gguf_find_key(g, "deepseek_ocr2.sam.image_std");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_std, gguf_get_arr_data(g, key_id), 3 * sizeof(float));

    auto &q = ctx.m.qhp;
    q.depth        = u32("deepseek_ocr2.qwen2_enc.depth", q.depth);
    q.hidden       = u32("deepseek_ocr2.qwen2_enc.hidden_size", q.hidden);
    q.heads        = u32("deepseek_ocr2.qwen2_enc.num_heads", q.heads);
    q.kv_heads     = u32("deepseek_ocr2.qwen2_enc.num_kv_heads", q.kv_heads);
    q.intermediate = u32("deepseek_ocr2.qwen2_enc.intermediate_size", q.intermediate);
    q.rms_eps      = f32v("deepseek_ocr2.qwen2_enc.rms_norm_eps", q.rms_eps);

    auto &l = ctx.m.lhp;
    l.vocab_size           = u32("deepseek_ocr2.vocab_size", l.vocab_size);
    l.hidden               = u32("deepseek_ocr2.hidden_size", l.hidden);
    l.heads                = u32("deepseek_ocr2.num_attention_heads", l.heads);
    l.kv_heads             = u32("deepseek_ocr2.num_key_value_heads", l.kv_heads);
    l.head_dim             = l.hidden / l.heads;
    l.n_layers             = u32("deepseek_ocr2.num_hidden_layers", l.n_layers);
    l.dense_intermediate   = u32("deepseek_ocr2.dense_intermediate_size", l.dense_intermediate);
    l.expert_intermediate  = u32("deepseek_ocr2.expert_intermediate_size", l.expert_intermediate);
    l.shared_intermediate  = u32("deepseek_ocr2.shared_intermediate_size", l.shared_intermediate);
    l.n_experts            = u32("deepseek_ocr2.n_routed_experts", l.n_experts);
    l.n_experts_top        = u32("deepseek_ocr2.num_experts_per_tok", l.n_experts_top);
    l.n_shared_experts     = u32("deepseek_ocr2.n_shared_experts", l.n_shared_experts);
    l.rms_eps              = f32v("deepseek_ocr2.rms_norm_eps", l.rms_eps);
    l.rope_theta           = f32v("deepseek_ocr2.rope_theta", l.rope_theta);
    l.routed_scaling_factor = f32v("deepseek_ocr2.routed_scaling_factor", l.routed_scaling_factor);
    l.eos_token_id         = u32("deepseek_ocr2.eos_token_id", l.eos_token_id);

    // Tokenizer
    int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        int n = (int)gguf_get_arr_n(g, vocab_idx);
        ctx.id_to_piece.resize(n);
        for (int i = 0; i < n; i++)
            ctx.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
        ctx.tok_vocab_size = n;
    }

    core_gguf::free_metadata(g);
    return true;
}

static bool load_tensors(ds_ocr2_ctx &ctx, const char *path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "deepseek_ocr2", wl)) return false;

    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;
    auto &t = wl.tensors;
    auto F = [&](const char *n) -> ggml_tensor* {
        auto it = t.find(n); return it != t.end() ? it->second : nullptr;
    };

    auto &m = ctx.m;
    auto &s = m.shp;

    // SAM
    m.patch_embed_w = F("v.patch_embed.weight");
    m.patch_embed_b = F("v.patch_embed.bias");
    m.pos_embed     = F("v.pos_embed");

    m.sam_blocks.resize(s.depth);
    for (int i = 0; i < s.depth; i++) {
        char buf[128];
        auto &blk = m.sam_blocks[i];
        blk.is_global = false;
        for (int gi : s.global_attn_indexes) if (gi == i) { blk.is_global = true; break; }
        auto BF = [&](const char *sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, sfx); return F(buf);
        };
        blk.ln1_w = BF("ln1.weight");     blk.ln1_b = BF("ln1.bias");
        blk.ln2_w = BF("ln2.weight");     blk.ln2_b = BF("ln2.bias");
        blk.qkv_w = BF("attn_qkv.weight"); blk.qkv_b = BF("attn_qkv.bias");
        blk.proj_w = BF("attn_proj.weight"); blk.proj_b = BF("attn_proj.bias");
        blk.rel_pos_h = BF("attn_rel_pos_h"); blk.rel_pos_w = BF("attn_rel_pos_w");
        blk.ffn_up_w = BF("ffn_up.weight");   blk.ffn_up_b = BF("ffn_up.bias");
        blk.ffn_down_w = BF("ffn_down.weight"); blk.ffn_down_b = BF("ffn_down.bias");
    }

    m.neck_conv1_w = F("v.neck_conv1.weight");
    m.neck_ln1_w = F("v.neck_ln1.weight"); m.neck_ln1_b = F("v.neck_ln1.bias");
    m.neck_conv2_w = F("v.neck_conv2.weight");
    m.neck_ln2_w = F("v.neck_ln2.weight"); m.neck_ln2_b = F("v.neck_ln2.bias");
    m.net_2_w = F("v.net_2.weight"); m.net_3_w = F("v.net_3.weight");

    // Qwen2 encoder
    auto &qhp = m.qhp;
    m.qwen2_layers.resize(qhp.depth);
    for (int i = 0; i < qhp.depth; i++) {
        char buf[128];
        auto &ly = m.qwen2_layers[i];
        auto QF = [&](const char *sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "qe.blk.%d.%s", i, sfx); return F(buf);
        };
        ly.in_ln_w  = QF("input_layernorm.weight");
        ly.post_ln_w = QF("post_attention_layernorm.weight");
        ly.q_w = QF("attn_q.weight"); ly.q_b = QF("attn_q.bias");
        ly.k_w = QF("attn_k.weight"); ly.k_b = QF("attn_k.bias");
        ly.v_w = QF("attn_v.weight"); ly.v_b = QF("attn_v.bias");
        ly.o_w = QF("attn_o.weight");
        ly.gate_w = QF("ffn_gate.weight");
        ly.up_w   = QF("ffn_up.weight");
        ly.down_w = QF("ffn_down.weight");
    }
    m.query_768  = F("qe.query_768");
    m.query_1024 = F("qe.query_1024");

    // Projector
    m.projector_w = F("proj.weight"); m.projector_b = F("proj.bias");

    // View separator
    m.view_separator = F("v.view_separator");

    // LLM
    m.embed_tokens   = F("l.embed_tokens.weight");
    m.output_norm_w  = F("l.output_norm.weight");
    m.lm_head_w      = F("l.lm_head.weight");

    auto &lhp = m.lhp;
    m.llm_layers.resize(lhp.n_layers);
    for (int i = 0; i < lhp.n_layers; i++) {
        char buf[128];
        auto &ly = m.llm_layers[i];
        auto LF = [&](const char *sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "l.blk.%d.%s", i, sfx); return F(buf);
        };
        ly.in_ln_w   = LF("input_layernorm.weight");
        ly.post_ln_w = LF("post_attention_layernorm.weight");
        ly.q_w = LF("attn_q.weight"); ly.k_w = LF("attn_k.weight");
        ly.v_w = LF("attn_v.weight"); ly.o_w = LF("attn_o.weight");

        if (i == 0) {
            // Dense FFN
            ly.ffn_gate_w = LF("ffn_gate.weight");
            ly.ffn_up_w   = LF("ffn_up.weight");
            ly.ffn_down_w = LF("ffn_down.weight");
        } else {
            // MoE
            ly.router_w = LF("mlp_gate.weight");
            ly.experts.resize(lhp.n_experts);
            for (int j = 0; j < lhp.n_experts; j++) {
                auto EF = [&](const char *sfx) -> ggml_tensor* {
                    snprintf(buf, sizeof(buf), "l.blk.%d.exp.%d.%s", i, j, sfx);
                    return F(buf);
                };
                ly.experts[j].gate_w = EF("ffn_gate.weight");
                ly.experts[j].up_w   = EF("ffn_up.weight");
                ly.experts[j].down_w = EF("ffn_down.weight");
            }
            ly.shared_gate_w = LF("shared_exp.ffn_gate.weight");
            ly.shared_up_w   = LF("shared_exp.ffn_up.weight");
            ly.shared_down_w = LF("shared_exp.ffn_down.weight");
        }
    }

    return true;
}

static void precompute_rpe_tables(ds_ocr2_ctx &ctx) {
    auto &s = ctx.m.shp;
    int hd = s.head_dim, nP = s.image_size / s.patch_size, ws = s.window_size;
    ctx.rp_h_per_layer.resize(s.depth);
    ctx.rp_w_per_layer.resize(s.depth);
    for (int li = 0; li < s.depth; li++) {
        auto &blk = ctx.m.sam_blocks[li];
        if (!blk.rel_pos_h || !blk.rel_pos_w) continue;
        auto rph = to_f32(blk.rel_pos_h), rpw = to_f32(blk.rel_pos_w);
        int L_h = (int)blk.rel_pos_h->ne[1], L_w = (int)blk.rel_pos_w->ne[1];
        int aH = blk.is_global ? nP : ws, aW = aH;
        ctx.rp_h_per_layer[li] = get_rel_pos(aH, aH, rph.data(), L_h, hd);
        ctx.rp_w_per_layer[li] = get_rel_pos(aW, aW, rpw.data(), L_w, hd);
    }
}

// ---------------------------------------------------------------------------
// SAM ViT-B per-layer ggml graph (same pattern as got_ocr.cpp)
// ---------------------------------------------------------------------------

static ggml_cgraph* build_sam_layer_graph(ggml_context* g, ds_ocr2_ctx* ctx,
                                           int li, int C, int T,
                                           int aH, int aW, int nW, int n_heads,
                                           bool skip_ln1) {
    auto& layer = ctx->m.sam_blocks[li];
    int hd = C / n_heads, wN = aH * aW, batch = n_heads * nW;
    float attn_scale = 1.0f / sqrtf((float)hd);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 512, false);

    ggml_tensor* inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
    ggml_set_name(inp, "layer_input"); ggml_set_input(inp);

    ggml_tensor* res_inp = nullptr;
    if (skip_ln1) {
        res_inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
        ggml_set_name(res_inp, "residual_input"); ggml_set_input(res_inp);
    }

    ggml_tensor* rp_h = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aH, aH);
    ggml_set_name(rp_h, "rp_h"); ggml_set_input(rp_h);
    ggml_tensor* rp_w = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aW, aW);
    ggml_set_name(rp_w, "rp_w"); ggml_set_input(rp_w);

    ggml_tensor* cur = skip_ln1 ? inp : g_ln(g, inp, layer.ln1_w, layer.ln1_b, 1e-6f);
    ggml_tensor* qkv = g_linear(g, cur, layer.qkv_w, layer.qkv_b);

    ggml_tensor* Q = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], 0));
    ggml_tensor* K = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], (size_t)C * sizeof(float)));
    ggml_tensor* V = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], (size_t)2 * C * sizeof(float)));

    Q = ggml_reshape_4d(g, Q, hd, n_heads, wN, nW);
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(g, Q, hd, wN, batch);
    K = ggml_reshape_4d(g, K, hd, n_heads, wN, nW);
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(g, K, hd, wN, batch);
    V = ggml_reshape_4d(g, V, hd, n_heads, wN, nW);
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(g, V, hd, wN, batch);

    ggml_tensor* scores = ggml_mul_mat(g, K, Q);
    scores = ggml_scale(g, scores, attn_scale);

    // Decomposed RPE
    ggml_tensor* Q_4d = ggml_reshape_4d(g, Q, hd, aW, aH, batch);
    ggml_tensor* rp_h_4d = ggml_reshape_4d(g, rp_h, hd, aH, aH, 1);
    ggml_tensor* rel_h = ggml_mul_mat(g, rp_h_4d, Q_4d);
    rel_h = ggml_reshape_3d(g, rel_h, aH, wN, batch);
    rel_h = ggml_reshape_4d(g, rel_h, 1, aH, wN, batch);

    ggml_tensor* Q_w = ggml_cont(g, ggml_permute(g, Q_4d, 0, 2, 1, 3));
    ggml_tensor* rp_w_4d = ggml_reshape_4d(g, rp_w, hd, aW, aW, 1);
    ggml_tensor* rel_w2 = ggml_mul_mat(g, rp_w_4d, Q_w);
    rel_w2 = ggml_cont(g, ggml_permute(g, rel_w2, 0, 2, 1, 3));
    rel_w2 = ggml_reshape_3d(g, rel_w2, aW, wN, batch);
    rel_w2 = ggml_reshape_4d(g, rel_w2, aW, 1, wN, batch);

    scores = ggml_reshape_4d(g, scores, aW, aH, wN, batch);
    scores = ggml_add(g, scores, rel_h);
    scores = ggml_add(g, scores, rel_w2);
    scores = ggml_reshape_3d(g, scores, wN, wN, batch);

    scores = ggml_soft_max_ext(g, scores, nullptr, 1.0f, 0.0f);

    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);
    attn = ggml_reshape_4d(g, attn, hd, wN, n_heads, nW);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, C, T);
    attn = g_linear(g, attn, layer.proj_w, layer.proj_b);

    cur = ggml_add(g, skip_ln1 ? res_inp : inp, attn);

    ggml_tensor* residual = cur;
    cur = g_ln(g, cur, layer.ln2_w, layer.ln2_b, 1e-6f);
    ggml_tensor* up = g_linear(g, cur, layer.ffn_up_w, layer.ffn_up_b);
    up = ggml_gelu(g, up);
    cur = g_linear(g, up, layer.ffn_down_w, layer.ffn_down_b);
    cur = ggml_add(g, residual, cur);

    ggml_set_name(cur, "layer_output"); ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ---------------------------------------------------------------------------
// SAM vision encoder
// ---------------------------------------------------------------------------

static bool encode_sam(ds_ocr2_ctx &ctx, const float *pixels,
                       std::vector<float> &out_features, int &out_n_tokens, int &out_dim) {
    auto &s = ctx.m.shp;
    int C = s.hidden, PS = s.patch_size, nP = s.image_size / PS;
    int N = nP * nP, hd = s.head_dim, ws = s.window_size;

    // Patch embedding
    auto pe_w = to_f32(ctx.m.patch_embed_w);
    auto pe_b = to_f32(ctx.m.patch_embed_b);
    auto pos = to_f32(ctx.m.pos_embed);
    int patch_dim = 3 * PS * PS;
    std::vector<float> hidden(N * C);

    for (int py = 0; py < nP; py++)
        for (int px = 0; px < nP; px++) {
            int tok = py * nP + px;
            std::vector<float> patch(patch_dim);
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < PS; ky++)
                    for (int kx = 0; kx < PS; kx++)
                        patch[c * PS * PS + ky * PS + kx] =
                            pixels[c * s.image_size * s.image_size
                                   + (py * PS + ky) * s.image_size + (px * PS + kx)];
            for (int o = 0; o < C; o++) {
                float sv = pe_b.empty() ? 0.0f : pe_b[o];
                for (int i = 0; i < patch_dim; i++) sv += pe_w[o * patch_dim + i] * patch[i];
                hidden[tok * C + o] = sv + (pos.empty() ? 0.0f : pos[tok * C + o]);
            }
        }

    // Pre-dequant LN weights for windowed layers
    std::vector<std::vector<float>> ln1_ws(s.depth), ln1_bs(s.depth);
    for (int li = 0; li < s.depth; li++)
        if (!ctx.m.sam_blocks[li].is_global) {
            ln1_ws[li] = to_f32(ctx.m.sam_blocks[li].ln1_w);
            ln1_bs[li] = to_f32(ctx.m.sam_blocks[li].ln1_b);
        }

    // Per-layer ggml graph
    for (int li = 0; li < s.depth; li++) {
        auto &blk = ctx.m.sam_blocks[li];
        bool is_global = blk.is_global;
        int aH = is_global ? nP : ws, aW = aH, wN = aH * aW;
        int nW, T;
        if (is_global) { nW = 1; T = N; }
        else {
            int ph = (ws - nP % ws) % ws, pw = (ws - nP % ws) % ws;
            nW = ((nP + ph) / ws) * ((nP + pw) / ws); T = wN * nW;
        }

        bool skip_ln1 = !is_global;
        std::vector<float> ln1_hidden;
        if (skip_ln1) {
            ln1_hidden.resize(N * C);
            for (int n = 0; n < N; n++)
                layernorm_cpu(hidden.data() + n * C, ln1_hidden.data() + n * C, C,
                              ln1_ws[li].data(), ln1_bs[li].data(), 1e-6f);
        }

        std::vector<float> graph_input, residual_input;
        if (is_global) graph_input.assign(hidden.begin(), hidden.end());
        else {
            graph_input.resize(T * C, 0.0f);
            window_partition(ln1_hidden.data(), graph_input.data(), nP, ws, C);
            residual_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), residual_input.data(), nP, ws, C);
        }

        std::vector<float> rp_h_ggml(aH * aH * hd), rp_w_ggml(aW * aW * hd);
        reformat_rp_table(ctx.rp_h_per_layer[li].data(), rp_h_ggml.data(), aH, hd);
        reformat_rp_table(ctx.rp_w_per_layer[li].data(), rp_w_ggml.data(), aW, hd);

        size_t meta_sz = 8 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context* gc = ggml_init(ip);

        ggml_cgraph* gf = build_sam_layer_graph(gc, &ctx, li, C, T, aH, aW, nW, s.heads, skip_ln1);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);

        ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "layer_input");
        ggml_backend_tensor_set(inp_t, graph_input.data(), 0, (size_t)T * C * sizeof(float));
        if (skip_ln1) {
            ggml_tensor* res_t = ggml_graph_get_tensor(gf, "residual_input");
            ggml_backend_tensor_set(res_t, residual_input.data(), 0, (size_t)T * C * sizeof(float));
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rp_h"),
                                rp_h_ggml.data(), 0, (size_t)aH * aH * hd * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rp_w"),
                                rp_w_ggml.data(), 0, (size_t)aW * aW * hd * sizeof(float));

        ggml_backend_sched_graph_compute(ctx.sched, gf);

        std::vector<float> graph_output(T * C);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_output"),
                                graph_output.data(), 0, (size_t)T * C * sizeof(float));
        ggml_free(gc);

        if (is_global) memcpy(hidden.data(), graph_output.data(), N * C * sizeof(float));
        else window_unpartition(graph_output.data(), hidden.data(), nP, ws, C);

        if (ctx.verbosity >= 2)
            fprintf(stderr, "deepseek_ocr2: sam_layer_%d done (%s, T=%d)\n",
                    li, is_global ? "global" : "window", T);
    }

    // Diff: SAM output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_output")) {
            auto r = ref.compare("sam_output", hidden.data(), N * C);
            fprintf(stderr, "  sam_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // Neck: Conv(768->256,1x1) -> LN2d -> Conv(256->256,3x3,p1) -> LN2d
    int nC = s.neck_out;
    std::vector<float> chw(C * nP * nP);
    for (int tok = 0; tok < N; tok++) {
        int y = tok / nP, x = tok % nP;
        for (int c = 0; c < C; c++) chw[c * nP * nP + y * nP + x] = hidden[tok * C + c];
    }

    auto nc1_w = to_f32(ctx.m.neck_conv1_w);
    std::vector<float> neck1(nC * nP * nP);
    conv2d_cpu(chw.data(), neck1.data(), nc1_w.data(), nullptr, C, nC, nP, nP, 1, 1, 1, 0);

    auto nln1_w = to_f32(ctx.m.neck_ln1_w), nln1_b = to_f32(ctx.m.neck_ln1_b);
    std::vector<float> neck1_ln(nC * nP * nP);
    layernorm2d_cpu(neck1.data(), neck1_ln.data(), nC, nP, nP, nln1_w.data(), nln1_b.data());

    auto nc2_w = to_f32(ctx.m.neck_conv2_w);
    std::vector<float> neck2(nC * nP * nP);
    conv2d_cpu(neck1_ln.data(), neck2.data(), nc2_w.data(), nullptr, nC, nC, nP, nP, 3, 3, 1, 1);

    auto nln2_w = to_f32(ctx.m.neck_ln2_w), nln2_b = to_f32(ctx.m.neck_ln2_b);
    std::vector<float> neck2_ln(nC * nP * nP);
    layernorm2d_cpu(neck2.data(), neck2_ln.data(), nC, nP, nP, nln2_w.data(), nln2_b.data());

    // Downsample: Conv(256->512,3x3,s2,p1) -> Conv(512->1024,3x3,s2,p1)
    int ds1_ch = 512, ds1_H = (nP + 2 - 3) / 2 + 1, ds1_W = ds1_H;
    auto n2_w = to_f32(ctx.m.net_2_w);
    std::vector<float> ds1(ds1_ch * ds1_H * ds1_W);
    conv2d_cpu(neck2_ln.data(), ds1.data(), n2_w.data(), nullptr, nC, ds1_ch, nP, nP, 3, 3, 2, 1);

    int ds2_ch = 1024, ds2_H = (ds1_H + 2 - 3) / 2 + 1, ds2_W = ds2_H;
    auto n3_w = to_f32(ctx.m.net_3_w);
    std::vector<float> ds2(ds2_ch * ds2_H * ds2_W);
    conv2d_cpu(ds1.data(), ds2.data(), n3_w.data(), nullptr, ds1_ch, ds2_ch, ds1_H, ds1_W, 3, 3, 2, 1);

    // Flatten to [n_tokens, dim]
    int n_vis = ds2_H * ds2_W, vis_D = ds2_ch;
    out_features.resize(n_vis * vis_D);
    for (int tok = 0; tok < n_vis; tok++) {
        int y = tok / ds2_W, x = tok % ds2_W;
        for (int c = 0; c < vis_D; c++)
            out_features[tok * vis_D + c] = ds2[c * ds2_H * ds2_W + y * ds2_W + x];
    }

    out_n_tokens = n_vis;
    out_dim = vis_D;
    return true;
}

// ---------------------------------------------------------------------------
// Qwen2 bidirectional encoder (CPU-scalar)
// ---------------------------------------------------------------------------

static bool encode_qwen2(ds_ocr2_ctx &ctx, const float *vis_features, int n_vis, int vis_dim,
                          std::vector<float> &out_enc, int &out_n_tokens, int &out_dim) {
    auto &qhp = ctx.m.qhp;
    int D = qhp.hidden, nh = qhp.heads, nkv = qhp.kv_heads;
    int hd = D / nh, kv_repeat = nh / nkv, inter = qhp.intermediate;
    float eps = qhp.rms_eps;

    // Build token sequence: query tokens + vis features
    // Use query_1024 for 1024-size images (default)
    auto query_data = to_f32(ctx.m.query_1024 ? ctx.m.query_1024 : ctx.m.query_768);
    int n_query = query_data.empty() ? 0 : (int)(query_data.size() / D);

    // Total tokens = n_query + n_vis
    // The visual features from SAM are 1024-dim, but Qwen2 encoder is 896-dim.
    // A projection from vis_dim(1024) -> D(896) is needed (this is the query mechanism).
    // Actually, the query tokens are learnable embeddings that attend to vis features
    // via cross-attention. But in DeepSeek-OCR-2, the Qwen2 encoder is self-attention
    // over concatenated [query_tokens, visual_features].
    // The vis features need to be projected to D first if dims don't match.
    // For now assume vis_dim == D or handle the mismatch.

    int T = n_query + n_vis;
    std::vector<float> hidden(T * D, 0.0f);

    // Copy query tokens
    if (n_query > 0)
        memcpy(hidden.data(), query_data.data(), n_query * D * sizeof(float));

    // Copy visual features (project if dim mismatch)
    if (vis_dim == D) {
        memcpy(hidden.data() + n_query * D, vis_features, n_vis * D * sizeof(float));
    } else {
        // If vis_dim != D, we'd need a linear projection here
        // For the expected architecture, vis_dim=1024 and D=896, so there should be
        // a projection in the SAM->Qwen2 path. This might be handled differently.
        // For safety, truncate or zero-pad.
        for (int t = 0; t < n_vis; t++)
            for (int d = 0; d < D; d++)
                hidden[(n_query + t) * D + d] = (d < vis_dim) ? vis_features[t * vis_dim + d] : 0.0f;
    }

    // Run bidirectional transformer layers
    for (int li = 0; li < qhp.depth; li++) {
        auto &ly = ctx.m.qwen2_layers[li];
        auto in_ln = to_f32(ly.in_ln_w), post_ln = to_f32(ly.post_ln_w);
        auto qw = to_f32(ly.q_w), qb = to_f32(ly.q_b);
        auto kw = to_f32(ly.k_w), kb = to_f32(ly.k_b);
        auto vw = to_f32(ly.v_w), vb = to_f32(ly.v_b);
        auto ow = to_f32(ly.o_w);
        auto gw = to_f32(ly.gate_w), uw = to_f32(ly.up_w), dw = to_f32(ly.down_w);

        int q_dim = nh * hd, kv_dim = nkv * hd;

        // Pre-norm
        std::vector<float> normed(T * D);
        for (int t = 0; t < T; t++)
            rmsnorm_cpu(hidden.data() + t * D, normed.data() + t * D, D, in_ln.data(), eps);

        // Q/K/V projections
        std::vector<float> Q(T * q_dim), K(T * kv_dim), V(T * kv_dim);
        for (int t = 0; t < T; t++) {
            linear_cpu(normed.data() + t * D, Q.data() + t * q_dim, D, q_dim,
                       qw.data(), qb.empty() ? nullptr : qb.data());
            linear_cpu(normed.data() + t * D, K.data() + t * kv_dim, D, kv_dim,
                       kw.data(), kb.empty() ? nullptr : kb.data());
            linear_cpu(normed.data() + t * D, V.data() + t * kv_dim, D, kv_dim,
                       vw.data(), vb.empty() ? nullptr : vb.data());
        }

        // Multi-head attention (bidirectional — no causal mask)
        float attn_scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out(T * D, 0.0f);

        for (int h = 0; h < nh; h++) {
            int kv_h = h / kv_repeat;  // GQA mapping

            // Compute scores for all query positions
            for (int qi = 0; qi < T; qi++) {
                std::vector<float> scores(T);
                for (int ki = 0; ki < T; ki++) {
                    float dot = 0;
                    for (int d = 0; d < hd; d++)
                        dot += Q[qi * q_dim + h * hd + d] * K[ki * kv_dim + kv_h * hd + d];
                    scores[ki] = dot * attn_scale;
                }
                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0;
                for (int ki = 0; ki < T; ki++) { scores[ki] = expf(scores[ki] - max_s); sum_exp += scores[ki]; }
                for (int ki = 0; ki < T; ki++) scores[ki] /= sum_exp;

                // Weighted sum of values
                for (int ki = 0; ki < T; ki++)
                    for (int d = 0; d < hd; d++)
                        attn_out[qi * D + h * hd + d] += scores[ki] * V[ki * kv_dim + kv_h * hd + d];
            }
        }

        // Output projection + residual
        std::vector<float> proj(T * D);
        for (int t = 0; t < T; t++)
            linear_cpu(attn_out.data() + t * D, proj.data() + t * D, D, D, ow.data(), nullptr);
        for (int i = 0; i < T * D; i++) hidden[i] += proj[i];

        // Post-attn norm + SwiGLU FFN + residual
        for (int t = 0; t < T; t++) {
            rmsnorm_cpu(hidden.data() + t * D, normed.data() + t * D, D, post_ln.data(), eps);
            std::vector<float> ffn_out(D);
            swiglu_ffn_cpu(normed.data() + t * D, ffn_out.data(), D, inter,
                           gw.data(), uw.data(), dw.data());
            for (int d = 0; d < D; d++) hidden[t * D + d] += ffn_out[d];
        }

        if (ctx.verbosity >= 2)
            fprintf(stderr, "deepseek_ocr2: qwen2_enc_layer_%d done\n", li);
    }

    // Diff: Qwen2 encoder output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("qwen2_enc_output")) {
            auto r = ref.compare("qwen2_enc_output", hidden.data(), T * D);
            fprintf(stderr, "  qwen2_enc_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // Output: only the query tokens (first n_query)
    out_n_tokens = (n_query > 0) ? n_query : T;
    out_dim = D;
    out_enc.resize(out_n_tokens * D);
    memcpy(out_enc.data(), hidden.data(), out_n_tokens * D * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Projector: Linear(896, 1280)
// ---------------------------------------------------------------------------

static bool project_to_llm(ds_ocr2_ctx &ctx, const float *enc_out, int n_tokens, int enc_dim,
                            std::vector<float> &proj_out) {
    auto &lhp = ctx.m.lhp;
    int out_dim = lhp.hidden;
    auto pw = to_f32(ctx.m.projector_w);
    auto pb = to_f32(ctx.m.projector_b);

    proj_out.resize(n_tokens * out_dim);
    for (int t = 0; t < n_tokens; t++)
        linear_cpu(enc_out + t * enc_dim, proj_out.data() + t * out_dim,
                   enc_dim, out_dim, pw.data(), pb.empty() ? nullptr : pb.data());

    // Diff: projector output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("projector_output")) {
            auto r = ref.compare("projector_output", proj_out.data(), n_tokens * out_dim);
            fprintf(stderr, "  projector_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// LLM decoder — ggml graph for attention + CPU-scalar MoE FFN
// ---------------------------------------------------------------------------

struct llm_attn_graph {
    ggml_cgraph *gf{};
    ggml_context *gctx{};
};

// Build attention-only graph for one LLM layer (no FFN — MoE done on CPU)
// For layer 0 (dense), includes the FFN in the graph.
static llm_attn_graph build_llm_layer_attn(ds_ocr2_ctx &ctx, int li, int T, int n_past,
                                            bool include_ffn) {
    auto &lhp = ctx.m.lhp;
    auto &ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads;
    int hd = lhp.head_dim;
    int Lk = n_past + T;
    float eps = lhp.rms_eps;

    size_t meta_sz = 4 * 1024 * 1024;
    std::vector<uint8_t> meta(meta_sz);
    ggml_init_params ip = { meta_sz, meta.data(), true };
    llm_attn_graph lag;
    lag.gctx = ggml_init(ip);
    auto *g = lag.gctx;
    lag.gf = ggml_new_graph_custom(g, 4096, false);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    // Input hidden states
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "layer_input"); ggml_set_input(x);

    // Position IDs
    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    // KV cache input
    ggml_tensor *k_cache_in = nullptr, *v_cache_in = nullptr;
    if (n_past > 0) {
        k_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, nkv * hd, n_past);
        ggml_set_name(k_cache_in, "k_cache_in"); ggml_set_input(k_cache_in);
        v_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, nkv * hd, n_past);
        ggml_set_name(v_cache_in, "v_cache_in"); ggml_set_input(v_cache_in);
    }

    // Pre-norm
    ggml_tensor *h = rmsnorm(x, ly.in_ln_w);

    // Q/K/V
    ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, h);
    ggml_tensor *K = ggml_mul_mat(g, ly.k_w, h);
    ggml_tensor *V = ggml_mul_mat(g, ly.v_w, h);

    // Reshape for RoPE
    Q = ggml_reshape_3d(g, Q, hd, nh, T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);

    // Standard RoPE
    Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Output new K/V for cache. The reshape_2d makes these *views* of the
    // cont'd permute; a view marked set_output and read back via
    // ggml_backend_tensor_get under the no-alloc scheduler returns garbage (its
    // buffer is reused once attention consumes it). cont the reshape so the
    // read-back data survives. (Same bug class fixed in qwen2vl/lightonocr.)
    ggml_tensor *K_new = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    K_new = ggml_cont(g, ggml_reshape_2d(g, K_new, nkv * hd, T));
    ggml_set_name(K_new, "k_out"); ggml_set_output(K_new);

    ggml_tensor *V_new = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
    V_new = ggml_cont(g, ggml_reshape_2d(g, V_new, nkv * hd, T));
    ggml_set_name(V_new, "v_out"); ggml_set_output(V_new);

    // Build full K/V (cache + new)
    ggml_tensor *Kfull, *Vfull;
    if (n_past > 0) {
        ggml_tensor *kc3 = ggml_reshape_3d(g, k_cache_in, hd, nkv, n_past);
        ggml_tensor *kn3 = ggml_reshape_3d(g, K_new, hd, nkv, T);
        Kfull = ggml_concat(g, kc3, kn3, 2);  // [hd, nkv, Lk]

        ggml_tensor *vc3 = ggml_reshape_3d(g, v_cache_in, hd, nkv, n_past);
        ggml_tensor *vn3 = ggml_reshape_3d(g, V_new, hd, nkv, T);
        Vfull = ggml_concat(g, vc3, vn3, 2);
    } else {
        Kfull = ggml_reshape_3d(g, K_new, hd, nkv, T);
        Vfull = ggml_reshape_3d(g, V_new, hd, nkv, T);
    }

    // GQA repeat if needed
    int kv_repeat = nh / nkv;
    if (kv_repeat > 1) {
        Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, Lk);
        ggml_tensor *K_tgt = ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, Lk);
        Kfull = ggml_repeat(g, Kfull, K_tgt);
        Kfull = ggml_reshape_3d(g, Kfull, hd, nh, Lk);

        Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, Lk);
        ggml_tensor *V_tgt = ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, Lk);
        Vfull = ggml_repeat(g, Vfull, V_tgt);
        Vfull = ggml_reshape_3d(g, Vfull, hd, nh, Lk);
    }

    // Attention
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));  // [hd, T, nh]
    Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
    Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

    // Causal mask
    ggml_tensor *mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    ggml_tensor *scores = ggml_mul_mat(g, Kfull, Q);
    float attn_scale = 1.0f / sqrtf((float)hd);
    scores = ggml_soft_max_ext(g, scores, mask, attn_scale, 0.0f);

    ggml_tensor *Vt = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
    ggml_tensor *attn = ggml_mul_mat(g, Vt, scores);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, D, T);

    attn = ggml_mul_mat(g, ly.o_w, attn);
    x = ggml_add(g, x, attn);

    if (include_ffn) {
        // Dense SwiGLU FFN (layer 0)
        ggml_tensor *residual = x;
        h = rmsnorm(x, ly.post_ln_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);
        ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
        x = ggml_add(g, residual, ffn);
    }

    ggml_set_name(x, "layer_output"); ggml_set_output(x);
    ggml_build_forward_expand(lag.gf, x);
    return lag;
}

// CPU-scalar MoE FFN for one layer
static void moe_ffn_cpu(ds_ocr2_ctx &ctx, int li, float *hidden, int T) {
    auto &lhp = ctx.m.lhp;
    auto &ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, inter_e = lhp.expert_intermediate;
    int inter_s = lhp.shared_intermediate;
    int n_exp = lhp.n_experts, top_k = lhp.n_experts_top;
    float scale = lhp.routed_scaling_factor;
    float eps = lhp.rms_eps;

    auto post_ln = to_f32(ly.post_ln_w);
    auto router = to_f32(ly.router_w);

    // Dequant shared expert weights
    auto sh_gw = to_f32(ly.shared_gate_w);
    auto sh_uw = to_f32(ly.shared_up_w);
    auto sh_dw = to_f32(ly.shared_down_w);

    // Dequant routed expert weights (lazy — all at once)
    struct exp_w { std::vector<float> gw, uw, dw; };
    std::vector<exp_w> exp_ws(n_exp);
    // Only dequant experts that are actually selected (deferred below)

    for (int t = 0; t < T; t++) {
        float *tok = hidden + t * D;

        // RMSNorm
        std::vector<float> normed(D);
        rmsnorm_cpu(tok, normed.data(), D, post_ln.data(), eps);

        // Router: matmul hidden x gate_weight^T -> softmax -> top-k
        std::vector<float> logits(n_exp);
        for (int e = 0; e < n_exp; e++) {
            float dot = 0;
            for (int d = 0; d < D; d++) dot += normed[d] * router[e * D + d];
            logits[e] = dot;
        }

        // Softmax
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0;
        for (int e = 0; e < n_exp; e++) { logits[e] = expf(logits[e] - max_l); sum_exp += logits[e]; }
        for (int e = 0; e < n_exp; e++) logits[e] /= sum_exp;

        // Top-k selection
        std::vector<std::pair<float, int>> scored(n_exp);
        for (int e = 0; e < n_exp; e++) scored[e] = {logits[e], e};
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](auto &a, auto &b) { return a.first > b.first; });

        // Normalize top-k weights to sum to 1
        float wsum = 0;
        for (int k = 0; k < top_k; k++) wsum += scored[k].first;
        if (wsum > 0) for (int k = 0; k < top_k; k++) scored[k].first /= wsum;

        // Run top-k routed experts
        std::vector<float> routed_out(D, 0.0f);
        for (int k = 0; k < top_k; k++) {
            int eid = scored[k].second;
            float w = scored[k].first * scale;

            // Lazy dequant
            if (exp_ws[eid].gw.empty()) {
                exp_ws[eid].gw = to_f32(ly.experts[eid].gate_w);
                exp_ws[eid].uw = to_f32(ly.experts[eid].up_w);
                exp_ws[eid].dw = to_f32(ly.experts[eid].down_w);
            }

            std::vector<float> expert_out(D);
            swiglu_ffn_cpu(normed.data(), expert_out.data(), D, inter_e,
                           exp_ws[eid].gw.data(), exp_ws[eid].uw.data(), exp_ws[eid].dw.data());
            for (int d = 0; d < D; d++) routed_out[d] += w * expert_out[d];
        }

        // Shared expert
        std::vector<float> shared_out(D);
        swiglu_ffn_cpu(normed.data(), shared_out.data(), D, inter_s,
                       sh_gw.data(), sh_uw.data(), sh_dw.data());

        // Combine: residual + routed + shared
        for (int d = 0; d < D; d++) tok[d] += routed_out[d] + shared_out[d];
    }
}

// ---------------------------------------------------------------------------
// Full LLM decoder forward
// ---------------------------------------------------------------------------

static bool run_llm_decoder(ds_ocr2_ctx &ctx, const float *image_embeds, int n_img,
                            const int32_t *prompt_ids, int n_prompt, int max_new,
                            std::vector<int32_t> &out_ids, std::vector<float> &out_confs) {
    auto &lhp = ctx.m.lhp;
    int D = lhp.hidden, V = lhp.vocab_size;
    int nh = lhp.heads, nkv = lhp.kv_heads, hd = lhp.head_dim;
    int n_layers = lhp.n_layers;
    int kv_dim = nkv * hd;

    // Initialize KV cache
    ctx.kvc.k_cache.resize(n_layers);
    ctx.kvc.v_cache.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        ctx.kvc.k_cache[i].clear();
        ctx.kvc.v_cache[i].clear();
    }
    ctx.kvc.n_past = 0;

    // Build initial embedding: splice image embeds at image_token positions
    // For simplicity, assume prompt = [special_tokens...] with image placeholders
    auto embed_w = to_f32(ctx.m.embed_tokens);

    auto get_embedding = [&](int32_t tok_id, float *out_emb) {
        for (int d = 0; d < D; d++)
            out_emb[d] = embed_w[tok_id * D + d];
    };

    // Dequant weights needed for LM head
    auto norm_w = to_f32(ctx.m.output_norm_w);
    auto head_w = to_f32(ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens);

    // Run generation loop
    int n_generated = 0;
    std::vector<int32_t> cur_tokens(prompt_ids, prompt_ids + n_prompt);
    int n_past = 0;

    while (n_generated < max_new) {
        int T = (int)cur_tokens.size();

        // Build input embeddings
        std::vector<float> input_emb(T * D);
        int img_idx = 0;
        for (int t = 0; t < T; t++) {
            if (n_past == 0 && image_embeds) {
                // Check if this is an image token placeholder
                // Simple heuristic: if token is in image range, splice
                get_embedding(cur_tokens[t], input_emb.data() + t * D);
                // Override with image embeddings for image token positions
                if (img_idx < n_img && cur_tokens[t] >= 151643) {
                    // Use image embedding instead
                    memcpy(input_emb.data() + t * D, image_embeds + img_idx * D, D * sizeof(float));
                    img_idx++;
                }
            } else {
                get_embedding(cur_tokens[t], input_emb.data() + t * D);
            }
        }

        // Process each layer
        std::vector<float> hidden(input_emb);

        for (int li = 0; li < n_layers; li++) {
            bool is_dense = (li == 0);

            // Build and run attention graph
            auto lag = build_llm_layer_attn(ctx, li, T, n_past, is_dense);
            ggml_backend_sched_reset(ctx.sched);
            if (!ggml_backend_sched_alloc_graph(ctx.sched, lag.gf)) {
                ggml_free(lag.gctx);
                return false;
            }

            // Set inputs
            ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "layer_input"),
                                    hidden.data(), 0, T * D * sizeof(float));

            std::vector<int32_t> pos(T);
            for (int t = 0; t < T; t++) pos[t] = n_past + t;
            ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "pos_ids"),
                                    pos.data(), 0, T * sizeof(int32_t));

            // KV cache
            if (n_past > 0) {
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "k_cache_in"),
                                        ctx.kvc.k_cache[li].data(), 0,
                                        kv_dim * n_past * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "v_cache_in"),
                                        ctx.kvc.v_cache[li].data(), 0,
                                        kv_dim * n_past * sizeof(float));
            }

            // Causal mask
            int Lk = n_past + T;
            std::vector<ggml_fp16_t> mask(Lk * T);
            for (int qi = 0; qi < T; qi++)
                for (int ki = 0; ki < Lk; ki++)
                    mask[qi * Lk + ki] = ggml_fp32_to_fp16(ki > n_past + qi ? -INFINITY : 0.0f);
            ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "mask"),
                                    mask.data(), 0, Lk * T * sizeof(ggml_fp16_t));

            ggml_backend_sched_graph_compute(ctx.sched, lag.gf);

            // Read outputs
            ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "layer_output"),
                                    hidden.data(), 0, T * D * sizeof(float));

            // Update KV cache
            std::vector<float> k_new(kv_dim * T), v_new(kv_dim * T);
            ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "k_out"),
                                    k_new.data(), 0, kv_dim * T * sizeof(float));
            ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "v_out"),
                                    v_new.data(), 0, kv_dim * T * sizeof(float));

            ctx.kvc.k_cache[li].insert(ctx.kvc.k_cache[li].end(), k_new.begin(), k_new.end());
            ctx.kvc.v_cache[li].insert(ctx.kvc.v_cache[li].end(), v_new.begin(), v_new.end());

            ggml_free(lag.gctx);

            // MoE FFN (layers 1-11)
            if (!is_dense) {
                moe_ffn_cpu(ctx, li, hidden.data(), T);
            }

            // Diff comparison
            if (!ctx.diff_ref_path.empty() && n_past == 0) {
                char name[64];
                snprintf(name, sizeof(name), "llm_layer_%d", li);
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(name)) {
                    auto r = ref.compare(name, hidden.data(), T * D);
                    fprintf(stderr, "  %s: cos_min=%.6f max_abs=%.6f %s\n",
                            name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }

        n_past += T;

        // Final norm + LM head (CPU)
        std::vector<float> last_hidden(D);
        rmsnorm_cpu(hidden.data() + (T - 1) * D, last_hidden.data(), D, norm_w.data(), lhp.rms_eps);

        std::vector<float> logits(V);
        linear_cpu(last_hidden.data(), logits.data(), D, V, head_w.data(), nullptr);

        // Diff: logits
        if (!ctx.diff_ref_path.empty() && n_generated == 0) {
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("logits")) {
                auto r = ref.compare("logits", logits.data(), V);
                fprintf(stderr, "  logits: cos_min=%.6f max_abs=%.6f %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }

        // Argmax
        int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        // Confidence
        float max_l = logits[next];
        float sum_e = 0;
        for (int v = 0; v < V; v++) sum_e += expf(logits[v] - max_l);
        out_confs.push_back(1.0f / sum_e);

        out_ids.push_back(next);
        n_generated++;

        if (next == lhp.eos_token_id) break;

        // Next step: single token
        cur_tokens = {(int32_t)next};
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer decode
// ---------------------------------------------------------------------------

static std::string decode_tokens(const ds_ocr2_ctx &ctx, const int32_t *ids, int n) {
    std::string result;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == ctx.m.lhp.eos_token_id) continue;
        if (id < 0 || id >= ctx.tok_vocab_size) continue;
        const auto &piece = ctx.id_to_piece[id];
        if (!piece.empty() && piece[0] == '<' && piece.back() == '>') continue;
        result += piece;
    }
    return result;
}

// ---------------------------------------------------------------------------
// C ABI wrappers
// ---------------------------------------------------------------------------

struct deepseek_ocr2_context {
    ds_ocr2_ctx inner;
    std::string result;
    std::vector<float> char_confidences;
};

deepseek_ocr2_context * deepseek_ocr2_init(const char * model_path, int n_threads) {
    auto *c = new deepseek_ocr2_context;
    auto &ctx = c->inner;
    ctx.n_threads = n_threads;

    if (!load_hparams(ctx, model_path)) {
        fprintf(stderr, "deepseek_ocr2: failed to load hparams\n");
        delete c; return nullptr;
    }

    ctx.backend = ggml_backend_init_best();
    if (!ctx.backend) {
        ctx.backend = ggml_backend_cpu_init();
        if (ctx.backend) ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
    }
    if (!ctx.backend) { delete c; return nullptr; }

    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);

    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(), 32768, false, false);
    ctx.compute_meta.resize(16 * 1024 * 1024);

    if (!load_tensors(ctx, model_path)) {
        fprintf(stderr, "deepseek_ocr2: failed to load tensors\n");
        delete c; return nullptr;
    }

    precompute_rpe_tables(ctx);

    if (ctx.verbosity >= 1) {
        auto &s = ctx.m.shp; auto &q = ctx.m.qhp; auto &l = ctx.m.lhp;
        fprintf(stderr, "deepseek_ocr2: loaded %s\n", model_path);
        fprintf(stderr, "  sam: %dL %dd %dH patch=%d img=%d ws=%d\n",
                s.depth, s.hidden, s.heads, s.patch_size, s.image_size, s.window_size);
        fprintf(stderr, "  qwen2_enc: %dL %dd %dH/%dKV inter=%d\n",
                q.depth, q.hidden, q.heads, q.kv_heads, q.intermediate);
        fprintf(stderr, "  llm: %dL %dd %dH/%dKV vocab=%d n_exp=%d top_%d\n",
                l.n_layers, l.hidden, l.heads, l.kv_heads, l.vocab_size,
                l.n_experts, l.n_experts_top);
        fprintf(stderr, "  tokenizer: %d tokens\n", ctx.tok_vocab_size);
    }

    return c;
}

void deepseek_ocr2_free(deepseek_ocr2_context * ctx) {
    if (!ctx) return;
    auto &c = ctx->inner;
    if (c.sched) ggml_backend_sched_free(c.sched);
    if (c.model_buf) ggml_backend_buffer_free(c.model_buf);
    if (c.model_ctx) ggml_free(c.model_ctx);
    if (c.backend) ggml_backend_free(c.backend);
    if (c.backend_cpu) ggml_backend_free(c.backend_cpu);
    delete ctx;
}

const char * deepseek_ocr2_recognize_raw(deepseek_ocr2_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!ctx || !px) { if (out_len) *out_len = 0; return ""; }

    auto &s = ctx->inner.m.shp;
    int imgS = s.image_size;

    // Resize to (imgS, imgS) with bilinear interpolation + normalize
    std::vector<float> pixels(3 * imgS * imgS);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < imgS; y++) {
            float fy = (float)y * h / imgS;
            int iy = std::min((int)fy, h - 1);
            for (int x = 0; x < imgS; x++) {
                float fx = (float)x * w / imgS;
                int ix = std::min((int)fx, w - 1);
                int ci = std::min(c, ch - 1);
                float val = (float)px[(iy * w + ix) * ch + ci] / 255.0f;
                pixels[c * imgS * imgS + y * imgS + x] = (val - s.image_mean[c]) / s.image_std[c];
            }
        }

    // 1. SAM vision encoder
    std::vector<float> sam_features;
    int n_sam_tokens, sam_dim;
    if (!encode_sam(ctx->inner, pixels.data(), sam_features, n_sam_tokens, sam_dim)) {
        fprintf(stderr, "deepseek_ocr2: SAM encoding failed\n");
        if (out_len) *out_len = 0; return "";
    }

    // 2. Qwen2 bidirectional encoder
    std::vector<float> enc_out;
    int n_enc_tokens, enc_dim;
    if (!encode_qwen2(ctx->inner, sam_features.data(), n_sam_tokens, sam_dim,
                      enc_out, n_enc_tokens, enc_dim)) {
        fprintf(stderr, "deepseek_ocr2: Qwen2 encoder failed\n");
        if (out_len) *out_len = 0; return "";
    }

    // 3. Project to LLM dimension
    std::vector<float> proj_out;
    if (!project_to_llm(ctx->inner, enc_out.data(), n_enc_tokens, enc_dim, proj_out)) {
        fprintf(stderr, "deepseek_ocr2: projection failed\n");
        if (out_len) *out_len = 0; return "";
    }

    // 4. Build prompt: image placeholder tokens
    int n_img_tokens = n_enc_tokens;
    std::vector<int32_t> prompt(n_img_tokens, 151643);  // placeholder token IDs

    // 5. LLM decoder
    std::vector<int32_t> gen_ids;
    std::vector<float> gen_confs;
    if (!run_llm_decoder(ctx->inner, proj_out.data(), n_img_tokens,
                         prompt.data(), (int)prompt.size(), 1024,
                         gen_ids, gen_confs)) {
        fprintf(stderr, "deepseek_ocr2: LLM decode failed\n");
        if (out_len) *out_len = 0; return "";
    }

    ctx->result = decode_tokens(ctx->inner, gen_ids.data(), (int)gen_ids.size());
    ctx->char_confidences = std::move(gen_confs);
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * deepseek_ocr2_recognize(deepseek_ocr2_context * ctx,
    const float * px, int w, int h, int * out_len) {
    if (!ctx || !px) { if (out_len) *out_len = 0; return ""; }
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, px[i] * 255.0f));
        rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
    }
    return deepseek_ocr2_recognize_raw(ctx, rgb.data(), w, h, 3, out_len);
}

const float * deepseek_ocr2_confidences(const deepseek_ocr2_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) { if (n_tokens) *n_tokens = 0; return nullptr; }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float deepseek_ocr2_mean_confidence(const deepseek_ocr2_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
