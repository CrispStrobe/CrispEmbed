// unlimited_ocr.cpp — Unlimited-OCR engine: SAM ViT-B + CLIP-L/14 + MoE decoder.
//
// Vision: per-layer ggml graph with CPU window partition (SAM ViT-B pattern).
// CLIP encoder: ggml graph bidirectional transformer (pre-LN ViT, quick_gelu).
// Fusion: concat(CLIP[:,1:], SAM.flatten) → Linear(2048, 1280).
// LLM decoder: ggml graph with KV cache, MoE layers use CPU-scalar expert dispatch.

#include "unlimited_ocr.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"
#include "core/bpe.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct sam_hparams {
    int depth = 12, hidden = 768, heads = 12, head_dim = 64;
    int patch_size = 16, image_size = 1024, window_size = 14;
    int neck_out = 256;
    std::vector<int> global_attn_indexes{2, 5, 8, 11};
    // BasicImageTransform: simple [-1,1] normalization (mean=std=0.5)
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
};

struct clip_hparams {
    int depth = 24, hidden = 1024, heads = 16;
    int ffn_hidden = 4096;
    int image_size = 224, patch_size = 14;
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

struct clip_block_w {
    ggml_tensor *ln1_w{}, *ln1_b{}, *ln2_w{}, *ln2_b{};
    ggml_tensor *qkv_w{}, *qkv_b{};   // fused Q/K/V
    ggml_tensor *proj_w{}, *proj_b{};
    ggml_tensor *ffn_up_w{}, *ffn_up_b{}, *ffn_down_w{}, *ffn_down_b{};
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
    // Experts stacked as [in, out, n_exp] for ggml_mul_mat_id (Metal MoE path).
    ggml_tensor *gate_exps{}, *up_exps{}, *down_exps{};
};

struct model_weights {
    sam_hparams shp;
    clip_hparams chp;
    llm_hparams lhp;

    // SAM
    ggml_tensor *patch_embed_w{}, *patch_embed_b{}, *pos_embed{};
    std::vector<sam_block_w> sam_blocks;
    ggml_tensor *neck_conv1_w{}, *neck_ln1_w{}, *neck_ln1_b{};
    ggml_tensor *neck_conv2_w{}, *neck_ln2_w{}, *neck_ln2_b{};
    ggml_tensor *net_2_w{}, *net_3_w{};

    // CLIP
    ggml_tensor *cls_token{};      // [1024]
    ggml_tensor *clip_patch_embed{};  // conv2d weight (unused — SAM provides patch embeds)
    ggml_tensor *clip_pos_embed{}; // [257, 1024]
    ggml_tensor *clip_pre_ln_w{}, *clip_pre_ln_b{};
    std::vector<clip_block_w> clip_blocks;

    // Image newline
    ggml_tensor *image_newline{};  // [1280]

    // Projector: Linear(2048, 1280)
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

struct uocr_ctx {
    model_weights m;
    ggml_context *model_ctx{};
    ggml_backend_buffer_t model_buf{};
    core_gguf::WeightLoad model_wl;
    ggml_backend_t backend{}, backend_cpu{};
    ggml_backend_sched_t sched{};
    std::vector<uint8_t> compute_meta;

    // Stacked MoE expert weights for the ggml_mul_mat_id path.
    ggml_context *moe_ctx{};
    ggml_backend_buffer_t moe_buf{};
    bool moe_metal = false;

    // Tokenizer
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    int tok_vocab_size = 0;

    // KV cache for LLM decoder — stored as F16
    struct {
        std::vector<std::vector<ggml_fp16_t>> k_cache;
        std::vector<std::vector<ggml_fp16_t>> v_cache;
        int n_past = 0;
    } kvc;

    // Precomputed RPE tables (raw = get_rel_pos output)
    std::vector<std::vector<float>> rp_h_per_layer, rp_w_per_layer;
    // Precomputed RPE tables (ggml-formatted = after reformat_rp_table)
    std::vector<std::vector<float>> rp_h_ggml_per_layer, rp_w_ggml_per_layer;

    int n_threads = 1, verbosity = 1;
    bool bench = false;
    std::string diff_ref_path;
};

// ---------------------------------------------------------------------------
// CPU helpers
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
                       int kh, int kw, int stride, int pad, int n_threads = 1) {
    int oH = (H + 2 * pad - kh) / stride + 1;
    int oW = (W + 2 * pad - kw) / stride + 1;
    auto plane = [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; oc++) {
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
    };
    int nt = std::max(1, std::min(n_threads, out_ch));
    if (nt <= 1) { plane(0, out_ch); return; }
    std::vector<std::thread> pool;
    int chunk = (out_ch + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int o0 = t * chunk, o1 = std::min(out_ch, o0 + chunk);
        if (o0 < o1) pool.emplace_back(plane, o0, o1);
    }
    for (auto& th : pool) th.join();
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
// SAM window partition / unpartition + RPE
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

static bool load_hparams(uocr_ctx &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32v = [&](const char *k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto &s = ctx.m.shp;
    s.depth       = u32("unlimited_ocr.sam.depth", s.depth);
    s.hidden      = u32("unlimited_ocr.sam.hidden_size", s.hidden);
    s.heads       = u32("unlimited_ocr.sam.num_heads", s.heads);
    s.head_dim    = s.hidden / s.heads;
    s.patch_size  = u32("unlimited_ocr.sam.patch_size", s.patch_size);
    s.image_size  = u32("unlimited_ocr.sam.image_size", s.image_size);
    s.window_size = u32("unlimited_ocr.sam.window_size", s.window_size);
    s.neck_out    = u32("unlimited_ocr.sam.neck_out_channels", s.neck_out);

    int key_id = gguf_find_key(g, "unlimited_ocr.sam.global_attn_indexes");
    if (key_id >= 0) {
        int n = (int)gguf_get_arr_n(g, key_id);
        s.global_attn_indexes.resize(n);
        memcpy(s.global_attn_indexes.data(), gguf_get_arr_data(g, key_id), n * sizeof(int32_t));
    }

    key_id = gguf_find_key(g, "unlimited_ocr.sam.image_mean");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_mean, gguf_get_arr_data(g, key_id), 3 * sizeof(float));
    key_id = gguf_find_key(g, "unlimited_ocr.sam.image_std");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_std, gguf_get_arr_data(g, key_id), 3 * sizeof(float));

    // CLIP hparams
    auto &c = ctx.m.chp;
    c.depth      = u32("unlimited_ocr.clip.depth", c.depth);
    c.hidden     = u32("unlimited_ocr.clip.hidden_size", c.hidden);
    c.heads      = u32("unlimited_ocr.clip.num_heads", c.heads);
    c.ffn_hidden = u32("unlimited_ocr.clip.intermediate_size", c.ffn_hidden);

    auto &l = ctx.m.lhp;
    l.vocab_size           = u32("unlimited_ocr.vocab_size", l.vocab_size);
    l.hidden               = u32("unlimited_ocr.hidden_size", l.hidden);
    l.heads                = u32("unlimited_ocr.num_attention_heads", l.heads);
    l.kv_heads             = u32("unlimited_ocr.num_key_value_heads", l.kv_heads);
    l.head_dim             = l.hidden / l.heads;
    l.n_layers             = u32("unlimited_ocr.num_hidden_layers", l.n_layers);
    l.dense_intermediate   = u32("unlimited_ocr.dense_intermediate_size", l.dense_intermediate);
    l.expert_intermediate  = u32("unlimited_ocr.expert_intermediate_size", l.expert_intermediate);
    l.shared_intermediate  = u32("unlimited_ocr.shared_intermediate_size", l.shared_intermediate);
    l.n_experts            = u32("unlimited_ocr.n_routed_experts", l.n_experts);
    l.n_experts_top        = u32("unlimited_ocr.num_experts_per_tok", l.n_experts_top);
    l.n_shared_experts     = u32("unlimited_ocr.n_shared_experts", l.n_shared_experts);
    l.rms_eps              = f32v("unlimited_ocr.rms_norm_eps", l.rms_eps);
    l.rope_theta           = f32v("unlimited_ocr.rope_theta", l.rope_theta);
    l.routed_scaling_factor = f32v("unlimited_ocr.routed_scaling_factor", l.routed_scaling_factor);
    l.eos_token_id         = u32("unlimited_ocr.eos_token_id", l.eos_token_id);

    // Tokenizer
    int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        int n = (int)gguf_get_arr_n(g, vocab_idx);
        ctx.id_to_piece.resize(n);
        ctx.token_to_id.reserve(n * 2);
        for (int i = 0; i < n; i++) {
            ctx.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
            ctx.token_to_id[ctx.id_to_piece[i]] = i;
        }
        ctx.tok_vocab_size = n;
    }
    int merges_idx = gguf_find_key(g, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        int n = (int)gguf_get_arr_n(g, merges_idx);
        ctx.merge_rank.reserve(n * 2);
        for (int i = 0; i < n; i++)
            ctx.merge_rank[gguf_get_arr_str(g, merges_idx, i)] = i;
    }

    core_gguf::free_metadata(g);
    return true;
}

static bool load_tensors(uocr_ctx &ctx, const char *path) {
    bool try_mmap = getenv("UOCR_MMAP") != nullptr;
    if (!core_gguf::load_weights(path, ctx.backend, "unlimited_ocr", ctx.model_wl, try_mmap))
        return false;

    ctx.model_ctx = ctx.model_wl.ctx;
    ctx.model_buf = ctx.model_wl.buf;
    auto &t = ctx.model_wl.tensors;
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

    // CLIP
    m.cls_token      = F("c.cls_token");
    m.clip_patch_embed = F("c.patch_embed.weight");
    m.clip_pos_embed = F("c.pos_embed");
    m.clip_pre_ln_w  = F("c.pre_ln.weight");
    m.clip_pre_ln_b  = F("c.pre_ln.bias");

    auto &chp = m.chp;
    m.clip_blocks.resize(chp.depth);
    for (int i = 0; i < chp.depth; i++) {
        char buf[128];
        auto &blk = m.clip_blocks[i];
        auto CF = [&](const char *sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "c.blk.%d.%s", i, sfx); return F(buf);
        };
        blk.ln1_w = CF("ln1.weight");     blk.ln1_b = CF("ln1.bias");
        blk.ln2_w = CF("ln2.weight");     blk.ln2_b = CF("ln2.bias");
        blk.qkv_w = CF("attn_qkv.weight"); blk.qkv_b = CF("attn_qkv.bias");
        blk.proj_w = CF("attn_proj.weight"); blk.proj_b = CF("attn_proj.bias");
        blk.ffn_up_w = CF("ffn_up.weight");   blk.ffn_up_b = CF("ffn_up.bias");
        blk.ffn_down_w = CF("ffn_down.weight"); blk.ffn_down_b = CF("ffn_down.bias");
    }

    // Image newline
    m.image_newline = F("v.image_newline");

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

static void precompute_rpe_tables(uocr_ctx &ctx) {
    auto &s = ctx.m.shp;
    int hd = s.head_dim, nP = s.image_size / s.patch_size, ws = s.window_size;
    ctx.rp_h_per_layer.resize(s.depth);
    ctx.rp_w_per_layer.resize(s.depth);
    ctx.rp_h_ggml_per_layer.resize(s.depth);
    ctx.rp_w_ggml_per_layer.resize(s.depth);
    for (int li = 0; li < s.depth; li++) {
        auto &blk = ctx.m.sam_blocks[li];
        if (!blk.rel_pos_h || !blk.rel_pos_w) continue;
        auto rph = to_f32(blk.rel_pos_h), rpw = to_f32(blk.rel_pos_w);
        int L_h = (int)blk.rel_pos_h->ne[1], L_w = (int)blk.rel_pos_w->ne[1];
        int aH = blk.is_global ? nP : ws, aW = aH;
        ctx.rp_h_per_layer[li] = get_rel_pos(aH, aH, rph.data(), L_h, hd);
        ctx.rp_w_per_layer[li] = get_rel_pos(aW, aW, rpw.data(), L_w, hd);
        // Pre-reformat for ggml layout (UOCR_OPT_RPE_CACHE path)
        ctx.rp_h_ggml_per_layer[li].resize((size_t)aH * aH * hd);
        ctx.rp_w_ggml_per_layer[li].resize((size_t)aW * aW * hd);
        reformat_rp_table(ctx.rp_h_per_layer[li].data(), ctx.rp_h_ggml_per_layer[li].data(), aH, hd);
        reformat_rp_table(ctx.rp_w_per_layer[li].data(), ctx.rp_w_ggml_per_layer[li].data(), aW, hd);
    }
}

// ---------------------------------------------------------------------------
// SAM ViT-B per-layer ggml graph
// ---------------------------------------------------------------------------

static ggml_cgraph* build_sam_layer_graph(ggml_context* g, uocr_ctx* ctx,
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
    up = ggml_gelu_erf(g, up);  // SAM MLPBlock uses nn.GELU (exact erf), not tanh approx
    cur = g_linear(g, up, layer.ffn_down_w, layer.ffn_down_b);
    cur = ggml_add(g, residual, cur);

    ggml_set_name(cur, "layer_output"); ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ---------------------------------------------------------------------------
// SAM vision encoder
// ---------------------------------------------------------------------------

static ggml_cgraph* build_sam_patch_graph(ggml_context* g, int imgS, int PS, int C, int nP) {
    ggml_cgraph* gf = ggml_new_graph(g);
    ggml_tensor* px = ggml_new_tensor_4d(g, GGML_TYPE_F32, imgS, imgS, 3, 1);
    ggml_set_name(px, "px"); ggml_set_input(px);
    ggml_tensor* w = ggml_new_tensor_4d(g, GGML_TYPE_F32, PS, PS, 3, C);
    ggml_set_name(w, "w_patch"); ggml_set_input(w);
    ggml_tensor* bias = ggml_new_tensor_1d(g, GGML_TYPE_F32, C);
    ggml_set_name(bias, "pe_b"); ggml_set_input(bias);
    ggml_tensor* pos = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nP * nP);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);

    ggml_tensor* x = ggml_conv_2d(g, w, px, PS, PS, 0, 0, 1, 1);
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));
    x = ggml_reshape_2d(g, x, C, nP * nP);
    x = ggml_add(g, x, bias);
    x = ggml_add(g, x, pos);
    ggml_set_name(x, "patch_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static ggml_cgraph* build_sam_neck_graph(ggml_context* g, int nP, int C, int nC,
                                         int ds1_ch, int ds2_ch) {
    ggml_cgraph* gf = ggml_new_graph(g);
    ggml_tensor* chw = ggml_new_tensor_4d(g, GGML_TYPE_F32, nP, nP, C, 1);
    ggml_set_name(chw, "chw"); ggml_set_input(chw);
    auto in4 = [&](const char* nm, int kw, int kh, int ic, int oc) {
        ggml_tensor* t = ggml_new_tensor_4d(g, GGML_TYPE_F32, kw, kh, ic, oc);
        ggml_set_name(t, nm); ggml_set_input(t); return t;
    };
    auto in1 = [&](const char* nm, int n) {
        ggml_tensor* t = ggml_new_tensor_1d(g, GGML_TYPE_F32, n);
        ggml_set_name(t, nm); ggml_set_input(t); return t;
    };
    ggml_tensor* w_nc1 = in4("w_nc1", 1, 1, C, nC);
    ggml_tensor* w_nc2 = in4("w_nc2", 3, 3, nC, nC);
    ggml_tensor* w_n2  = in4("w_n2", 3, 3, nC, ds1_ch);
    ggml_tensor* w_n3  = in4("w_n3", 3, 3, ds1_ch, ds2_ch);
    ggml_tensor *ln1w = in1("ln1w", nC), *ln1b = in1("ln1b", nC);
    ggml_tensor *ln2w = in1("ln2w", nC), *ln2b = in1("ln2b", nC);

    auto ln2d = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* xp = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));
        xp = ggml_norm(g, xp, 1e-6f);
        xp = ggml_add(g, ggml_mul(g, xp, w), b);
        return ggml_cont(g, ggml_permute(g, xp, 2, 0, 1, 3));
    };

    ggml_tensor* x = ggml_conv_2d(g, w_nc1, chw, 1, 1, 0, 0, 1, 1);
    x = ln2d(x, ln1w, ln1b);
    x = ggml_conv_2d(g, w_nc2, x, 1, 1, 1, 1, 1, 1);
    x = ln2d(x, ln2w, ln2b);
    x = ggml_conv_2d(g, w_n2, x, 2, 2, 1, 1, 1, 1);
    x = ggml_conv_2d(g, w_n3, x, 2, 2, 1, 1, 1, 1);
    int ds2 = (nP + 2 - 3) / 2 + 1; ds2 = (ds2 + 2 - 3) / 2 + 1;
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));
    x = ggml_reshape_2d(g, x, ds2_ch, ds2 * ds2);
    ggml_set_name(x, "neck_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// Bilinear interpolation of 2D position embedding grid.
// pos_in: [nP_in*nP_in, C], pos_out: [nP_out*nP_out, C]
static void interpolate_pos_embed_2d(const float* pos_in, float* pos_out,
                                      int nP_in, int nP_out, int C) {
    float scale = (float)nP_in / nP_out;
    for (int oy = 0; oy < nP_out; oy++) {
        float sy = (oy + 0.5f) * scale - 0.5f;
        int iy = (int)floorf(sy);
        float fy = sy - iy;
        int iy0 = std::max(0, std::min(iy, nP_in - 1));
        int iy1 = std::max(0, std::min(iy + 1, nP_in - 1));
        for (int ox = 0; ox < nP_out; ox++) {
            float sx = (ox + 0.5f) * scale - 0.5f;
            int ix = (int)floorf(sx);
            float fx = sx - ix;
            int ix0 = std::max(0, std::min(ix, nP_in - 1));
            int ix1 = std::max(0, std::min(ix + 1, nP_in - 1));
            int out_idx = (oy * nP_out + ox) * C;
            int i00 = (iy0 * nP_in + ix0) * C;
            int i01 = (iy0 * nP_in + ix1) * C;
            int i10 = (iy1 * nP_in + ix0) * C;
            int i11 = (iy1 * nP_in + ix1) * C;
            float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx;
            float w10 = fy * (1 - fx), w11 = fy * fx;
            for (int c = 0; c < C; c++)
                pos_out[out_idx + c] = w00 * pos_in[i00 + c] + w01 * pos_in[i01 + c]
                                     + w10 * pos_in[i10 + c] + w11 * pos_in[i11 + c];
        }
    }
}

static bool encode_sam(uocr_ctx &ctx, const float *pixels, int sam_img_size,
                       std::vector<float> &out_features, int &out_n_tokens, int &out_dim) {
    auto &s = ctx.m.shp;
    int C = s.hidden, PS = s.patch_size;
    int nP_orig = s.image_size / PS;
    int nP = sam_img_size / PS;
    int N = nP * nP, hd = s.head_dim, ws = s.window_size;
    auto _sam_t = std::chrono::steady_clock::now();
    auto sam_mark = [&](const char *w) {
        if (!getenv("UOCR_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] sam.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _sam_t).count());
        _sam_t = now;
    };

    // Patch embedding
    auto pe_w = to_f32(ctx.m.patch_embed_w);
    auto pe_b = to_f32(ctx.m.patch_embed_b);
    auto pos_orig = to_f32(ctx.m.pos_embed);
    // Interpolate position embedding if running at reduced resolution
    std::vector<float> pos;
    if (nP != nP_orig) {
        pos.resize((size_t)N * C);
        interpolate_pos_embed_2d(pos_orig.data(), pos.data(), nP_orig, nP, C);
    } else {
        pos = std::move(pos_orig);
    }
    int patch_dim = 3 * PS * PS;
    std::vector<float> hidden(N * C);

    if (!getenv("UOCR_SAM_CONV_CPU")) {
        size_t meta_sz = 8 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context* gc = ggml_init(ip);
        ggml_cgraph* gf = build_sam_patch_graph(gc, sam_img_size, PS, C, nP);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "px"), pixels, 0,
                                (size_t)3 * sam_img_size * sam_img_size * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_patch"), pe_w.data(), 0, pe_w.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pe_b"), pe_b.data(), 0, pe_b.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos.data(), 0, pos.size() * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, gf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "patch_out"), hidden.data(), 0,
                                (size_t)N * C * sizeof(float));
        ggml_free(gc);
    } else {
        auto patch_rows = [&](int py0, int py1) {
            std::vector<float> patch(patch_dim);
            for (int py = py0; py < py1; py++)
                for (int px = 0; px < nP; px++) {
                    int tok = py * nP + px;
                    for (int c = 0; c < 3; c++)
                        for (int ky = 0; ky < PS; ky++)
                            for (int kx = 0; kx < PS; kx++)
                                patch[c * PS * PS + ky * PS + kx] =
                                    pixels[c * s.image_size * s.image_size
                                           + (py * PS + ky) * s.image_size + (px * PS + kx)];
                    if (tok == 1000 && getenv("UOCR_DBG")) {
                        fprintf(stderr, "  [dbg] tok1000 patch[0..3]: %.6f %.6f %.6f %.6f\n",
                                patch[0], patch[1], patch[2], patch[3]);
                        fprintf(stderr, "  [dbg] tok1000 pixel[240,640]: %.6f\n",
                                pixels[0 * s.image_size * s.image_size + 240 * s.image_size + 640]);
                    }
                    if (tok == 0 && getenv("UOCR_DBG")) {
                        fprintf(stderr, "  [dbg] tok0 patch[0..3]: %.6f %.6f %.6f %.6f\n",
                                patch[0], patch[1], patch[2], patch[3]);
                    }
                    for (int o = 0; o < C; o++) {
                        float sv = pe_b.empty() ? 0.0f : pe_b[o];
                        for (int i = 0; i < patch_dim; i++) sv += pe_w[o * patch_dim + i] * patch[i];
                        hidden[tok * C + o] = sv + (pos.empty() ? 0.0f : pos[tok * C + o]);
                    }
                }
        };
        int nt = std::max(1, std::min(ctx.n_threads, nP));
        if (nt <= 1) patch_rows(0, nP);
        else {
            std::vector<std::thread> pool;
            int chunk = (nP + nt - 1) / nt;
            for (int t = 0; t < nt; t++) {
                int y0 = t * chunk, y1 = std::min(nP, y0 + chunk);
                if (y0 < y1) pool.emplace_back(patch_rows, y0, y1);
            }
            for (auto& th : pool) th.join();
        }
    }

    sam_mark("patch_embed");

    // Diff: patch embedding output (before any transformer layers)
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_patch_embed")) {
            auto r = ref.compare("sam_patch_embed", hidden.data(), N * C);
            fprintf(stderr, "  sam_patch_embed: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }
    if (getenv("UOCR_DBG")) {
        fprintf(stderr, "  [dbg] sam_patch_embed[0..7]:");
        for (int i = 0; i < std::min(8, N * C); i++) fprintf(stderr, " %.4f", hidden[i]);
        fprintf(stderr, "\n");
        int toks[] = {0, 1, 100, 1000, 2048, 4000};
        for (int ti = 0; ti < 6; ti++) {
            int tok = toks[ti];
            if (tok < N)
                fprintf(stderr, "  [dbg] tok%4d[0:4]: %.7f %.7f %.7f %.7f\n", tok,
                        hidden[tok*C], hidden[tok*C+1], hidden[tok*C+2], hidden[tok*C+3]);
        }
    }

    // Pre-dequant LN weights for windowed layers
    std::vector<std::vector<float>> ln1_ws(s.depth), ln1_bs(s.depth);
    for (int li = 0; li < s.depth; li++)
        if (!ctx.m.sam_blocks[li].is_global) {
            ln1_ws[li] = to_f32(ctx.m.sam_blocks[li].ln1_w);
            ln1_bs[li] = to_f32(ctx.m.sam_blocks[li].ln1_b);
        }

    // Hoist ggml metadata buffer outside the per-layer loop (UOCR_OPT_REUSE_CTX)
    size_t meta_sz = 8 * 1024 * 1024;
    std::vector<uint8_t> mb(meta_sz);

    // Per-layer ggml graph
    for (int li = 0; li < s.depth; li++) {
        auto _slt = std::chrono::steady_clock::now();
        auto &blk = ctx.m.sam_blocks[li];
        bool is_global = blk.is_global;
        int aH = is_global ? nP : ws, aW = aH, wN = aH * aW;
        int nW, T;
        if (is_global) { nW = 1; T = N; }
        else {
            int ph = (ws - nP % ws) % ws, pw = (ws - nP % ws) % ws;
            nW = ((nP + ph) / ws) * ((nP + pw) / ws); T = wN * nW;
        }

        // UOCR_OPT_GRAPH_LN: do LN inside the ggml graph (no CPU LN + separate residual)
        bool graph_ln = getenv("UOCR_OPT_GRAPH_LN") != nullptr;
        bool skip_ln1 = !is_global && !graph_ln;
        std::vector<float> ln1_hidden;
        if (skip_ln1) {
            ln1_hidden.resize(N * C);
            for (int n = 0; n < N; n++)
                layernorm_cpu(hidden.data() + n * C, ln1_hidden.data() + n * C, C,
                              ln1_ws[li].data(), ln1_bs[li].data(), 1e-6f);
        }

        std::vector<float> graph_input, residual_input;
        if (is_global) graph_input.assign(hidden.begin(), hidden.end());
        else if (skip_ln1) {
            graph_input.resize(T * C, 0.0f);
            window_partition(ln1_hidden.data(), graph_input.data(), nP, ws, C);
            residual_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), residual_input.data(), nP, ws, C);
        } else {
            // graph_ln path: partition raw data, graph does LN internally
            graph_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), graph_input.data(), nP, ws, C);
        }

        if (getenv("UOCR_DBG")) fprintf(stderr, "  [dbg] sam li=%d is_global=%d aH=%d nW=%d T=%d rp_h.sz=%zu\n",
                li, is_global, aH, nW, T, ctx.rp_h_per_layer[li].size());
        // RPE tables: use precomputed if sizes match, else recompute for reduced resolution
        const float *rp_h_ptr, *rp_w_ptr;
        std::vector<float> rp_h_recomp, rp_w_recomp, rp_h_ggml_rc, rp_w_ggml_rc;
        bool rpe_size_match = !ctx.rp_h_ggml_per_layer[li].empty() &&
            (int)ctx.rp_h_ggml_per_layer[li].size() == aH * aH * hd;
        if (rpe_size_match) {
            rp_h_ptr = ctx.rp_h_ggml_per_layer[li].data();
            rp_w_ptr = ctx.rp_w_ggml_per_layer[li].data();
        } else {
            // Recompute RPE for reduced resolution (global layers change from nP_orig to nP)
            auto &blk2 = ctx.m.sam_blocks[li];
            auto rph = to_f32(blk2.rel_pos_h), rpw = to_f32(blk2.rel_pos_w);
            int L_h = (int)blk2.rel_pos_h->ne[1], L_w = (int)blk2.rel_pos_w->ne[1];
            rp_h_recomp = get_rel_pos(aH, aH, rph.data(), L_h, hd);
            rp_w_recomp = get_rel_pos(aW, aW, rpw.data(), L_w, hd);
            rp_h_ggml_rc.resize((size_t)aH * aH * hd);
            rp_w_ggml_rc.resize((size_t)aW * aW * hd);
            reformat_rp_table(rp_h_recomp.data(), rp_h_ggml_rc.data(), aH, hd);
            reformat_rp_table(rp_w_recomp.data(), rp_w_ggml_rc.data(), aW, hd);
            rp_h_ptr = rp_h_ggml_rc.data();
            rp_w_ptr = rp_w_ggml_rc.data();
        }

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
                                rp_h_ptr, 0, (size_t)aH * aH * hd * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rp_w"),
                                rp_w_ptr, 0, (size_t)aW * aW * hd * sizeof(float));

        ggml_backend_sched_graph_compute(ctx.sched, gf);

        std::vector<float> graph_output(T * C);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_output"),
                                graph_output.data(), 0, (size_t)T * C * sizeof(float));
        ggml_free(gc);

        if (is_global) memcpy(hidden.data(), graph_output.data(), N * C * sizeof(float));
        else window_unpartition(graph_output.data(), hidden.data(), nP, ws, C);

        // Per-layer SAM diff check
        if (!ctx.diff_ref_path.empty()) {
            char nm[32]; snprintf(nm, sizeof(nm), "sam_layer_%d", li);
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                auto r = ref.compare(nm, hidden.data(), N * C);
                fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                        nm, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }
        if (ctx.verbosity >= 2)
            fprintf(stderr, "unlimited_ocr: sam_layer_%d done (%s, T=%d)\n",
                    li, is_global ? "global" : "window", T);
    }

    // Diff: pre-neck ViT output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_vit_output")) {
            auto r = ref.compare("sam_vit_output", hidden.data(), N * C);
            fprintf(stderr, "  sam_vit_output: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    sam_mark("layers");
    // Neck: Conv(768->256,1x1) -> LN2d -> Conv(256->256,3x3,p1) -> LN2d
    int nC = s.neck_out;
    std::vector<float> chw(C * nP * nP);
    for (int tok = 0; tok < N; tok++) {
        int y = tok / nP, x = tok % nP;
        for (int c = 0; c < C; c++) chw[c * nP * nP + y * nP + x] = hidden[tok * C + c];
    }

    // Derive downsample channels from weights
    int ds1_ch = (int)ctx.m.net_2_w->ne[1];
    int ds2_ch = (int)ctx.m.net_3_w->ne[1];
    int ds1_H = (nP + 2 - 3) / 2 + 1;
    int ds2_H = (ds1_H + 2 - 3) / 2 + 1, ds2_W = ds2_H;
    int n_vis = ds2_H * ds2_W, vis_D = ds2_ch;
    out_features.resize((size_t)n_vis * vis_D);

    if (!getenv("UOCR_SAM_CONV_CPU")) {
        auto nc1 = to_f32(ctx.m.neck_conv1_w), nc2 = to_f32(ctx.m.neck_conv2_w);
        auto n2 = to_f32(ctx.m.net_2_w), n3 = to_f32(ctx.m.net_3_w);
        auto l1w = to_f32(ctx.m.neck_ln1_w), l1b = to_f32(ctx.m.neck_ln1_b);
        auto l2w = to_f32(ctx.m.neck_ln2_w), l2b = to_f32(ctx.m.neck_ln2_b);
        size_t meta_sz = 16 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context* gc = ggml_init(ip);
        ggml_cgraph* gf = build_sam_neck_graph(gc, nP, C, nC, ds1_ch, ds2_ch);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);
        auto setn = [&](const char* nm, const std::vector<float>& v) {
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), v.data(), 0, v.size() * sizeof(float));
        };
        setn("chw", chw); setn("w_nc1", nc1); setn("w_nc2", nc2); setn("w_n2", n2); setn("w_n3", n3);
        setn("ln1w", l1w); setn("ln1b", l1b); setn("ln2w", l2w); setn("ln2b", l2b);
        ggml_backend_sched_graph_compute(ctx.sched, gf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "neck_out"), out_features.data(), 0,
                                (size_t)n_vis * vis_D * sizeof(float));
        ggml_free(gc);
    } else {
        auto nc1_w = to_f32(ctx.m.neck_conv1_w);
        std::vector<float> neck1(nC * nP * nP);
        conv2d_cpu(chw.data(), neck1.data(), nc1_w.data(), nullptr, C, nC, nP, nP, 1, 1, 1, 0, ctx.n_threads);
        auto nln1_w = to_f32(ctx.m.neck_ln1_w), nln1_b = to_f32(ctx.m.neck_ln1_b);
        std::vector<float> neck1_ln(nC * nP * nP);
        layernorm2d_cpu(neck1.data(), neck1_ln.data(), nC, nP, nP, nln1_w.data(), nln1_b.data());
        auto nc2_w = to_f32(ctx.m.neck_conv2_w);
        std::vector<float> neck2(nC * nP * nP);
        conv2d_cpu(neck1_ln.data(), neck2.data(), nc2_w.data(), nullptr, nC, nC, nP, nP, 3, 3, 1, 1, ctx.n_threads);
        auto nln2_w = to_f32(ctx.m.neck_ln2_w), nln2_b = to_f32(ctx.m.neck_ln2_b);
        std::vector<float> neck2_ln(nC * nP * nP);
        layernorm2d_cpu(neck2.data(), neck2_ln.data(), nC, nP, nP, nln2_w.data(), nln2_b.data());
        auto n2_w = to_f32(ctx.m.net_2_w);
        std::vector<float> ds1((size_t)ds1_ch * ds1_H * ds1_H);
        conv2d_cpu(neck2_ln.data(), ds1.data(), n2_w.data(), nullptr, nC, ds1_ch, nP, nP, 3, 3, 2, 1, ctx.n_threads);
        auto n3_w = to_f32(ctx.m.net_3_w);
        std::vector<float> ds2((size_t)ds2_ch * ds2_H * ds2_W);
        conv2d_cpu(ds1.data(), ds2.data(), n3_w.data(), nullptr, ds1_ch, ds2_ch, ds1_H, ds1_H, 3, 3, 2, 1, ctx.n_threads);
        for (int tok = 0; tok < n_vis; tok++) {
            int y = tok / ds2_W, x = tok % ds2_W;
            for (int c = 0; c < vis_D; c++)
                out_features[tok * vis_D + c] = ds2[c * ds2_H * ds2_W + y * ds2_W + x];
        }
    }

    sam_mark("neck_downsample");

    // SAM reduced res: upsample from reduced spatial grid to match 1024 output
    // At 1024: ds2_H=16, n_vis=256. At 512: ds2_H=8, n_vis=64.
    // CLIP expects 256 tokens. Bilinear upsample 8×8 → 16×16.
    int target_nvis = (s.image_size / s.patch_size / 2 / 2);  // = 16 for 1024
    target_nvis = target_nvis * target_nvis;  // = 256
    if (n_vis < target_nvis && nP != nP_orig) {
        int src_h = ds2_H, src_w = ds2_W;
        int dst_h = s.image_size / s.patch_size / 2 / 2;
        int dst_w = dst_h;
        int dst_n = dst_h * dst_w;
        std::vector<float> upsampled((size_t)dst_n * vis_D);
        float scale_y = (float)src_h / dst_h;
        float scale_x = (float)src_w / dst_w;
        for (int dy = 0; dy < dst_h; dy++) {
            float sy = (dy + 0.5f) * scale_y - 0.5f;
            int iy = (int)floorf(sy); float fy = sy - iy;
            int iy0 = std::max(0, std::min(iy, src_h - 1));
            int iy1 = std::max(0, std::min(iy + 1, src_h - 1));
            for (int dx = 0; dx < dst_w; dx++) {
                float sx = (dx + 0.5f) * scale_x - 0.5f;
                int ix = (int)floorf(sx); float fx = sx - ix;
                int ix0 = std::max(0, std::min(ix, src_w - 1));
                int ix1 = std::max(0, std::min(ix + 1, src_w - 1));
                float w00 = (1-fy)*(1-fx), w01 = (1-fy)*fx, w10 = fy*(1-fx), w11 = fy*fx;
                int di = (dy * dst_w + dx) * vis_D;
                int s00 = (iy0*src_w+ix0)*vis_D, s01 = (iy0*src_w+ix1)*vis_D;
                int s10 = (iy1*src_w+ix0)*vis_D, s11 = (iy1*src_w+ix1)*vis_D;
                for (int c = 0; c < vis_D; c++)
                    upsampled[di+c] = w00*out_features[s00+c] + w01*out_features[s01+c]
                                    + w10*out_features[s10+c] + w11*out_features[s11+c];
            }
        }
        out_features = std::move(upsampled);
        n_vis = dst_n;
        if (getenv("UOCR_DBG"))
            fprintf(stderr, "  [opt] SAM reduced res: upsampled %dx%d → %dx%d (%d tokens)\n",
                    src_h, src_w, dst_h, dst_w, dst_n);
    }

    out_n_tokens = n_vis;
    out_dim = vis_D;

    // Diff: final SAM output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_output")) {
            auto r = ref.compare("sam_output", out_features.data(),
                                 (size_t)out_n_tokens * out_dim);
            fprintf(stderr, "  sam_output: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CLIP-L/14 encoder (ggml graph, all 24 layers in one graph)
// ---------------------------------------------------------------------------

// Builds a single ggml graph covering all 24 CLIP transformer layers.
// Input: SAM features as patch embeddings (256, 1024) — replaces the CLIP
// conv patch embedding. CLS token + position embeddings + pre-LayerNorm
// are applied first, then 24 pre-LN ViT layers with quick_gelu FFN.
// Output: (257, 1024) including CLS; caller skips CLS to get (256, 1024).
static ggml_cgraph* build_clip_enc_full_graph(ggml_context* g, uocr_ctx* ctx, int T) {
    // T = 257 (CLS + 256 patches)
    auto &chp = ctx->m.chp;
    int D = chp.hidden, nh = chp.heads;
    int hd = D / nh;
    int n_layers = chp.depth;

    bool clip_debug = (getenv("UOCR_CLIP_DBG") != nullptr);
    ggml_cgraph* gf = ggml_new_graph_custom(g, clip_debug ? 8192 : 4096, false);

    // Input: [D, T] — CLS prepended + pos embed + pre_layernorm already done by caller?
    // No — we do it in-graph for performance.
    ggml_tensor* x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "clip_input"); ggml_set_input(x);

    // Pre-LayerNorm (applies to the full sequence after CLS + pos_embed)
    x = g_ln(g, x, ctx->m.clip_pre_ln_w, ctx->m.clip_pre_ln_b, 1e-5f);

    if (clip_debug) {
        ggml_set_name(x, "clip_pre_ln_out"); ggml_set_output(x);
    }

    for (int li = 0; li < n_layers; li++) {
        auto &blk = ctx->m.clip_blocks[li];

        // Pre-LN ViT: LayerNorm → MHSA → residual → LayerNorm → FFN → residual
        ggml_tensor* res = x;
        ggml_tensor* h = g_ln(g, x, blk.ln1_w, blk.ln1_b, 1e-5f);

        // Fused QKV: [3*D, T]
        ggml_tensor* qkv = g_linear(g, h, blk.qkv_w, blk.qkv_b);

        // Split into Q, K, V each [D, T]
        ggml_tensor* Q = ggml_cont(g, ggml_view_2d(g, qkv, D, T, qkv->nb[1], 0));
        ggml_tensor* K = ggml_cont(g, ggml_view_2d(g, qkv, D, T, qkv->nb[1], (size_t)D * sizeof(float)));
        ggml_tensor* V = ggml_cont(g, ggml_view_2d(g, qkv, D, T, qkv->nb[1], (size_t)2 * D * sizeof(float)));

        // Reshape to multi-head [hd, nh, T] → permute to [hd, T, nh]
        // Do NOT ggml_cont() — flash_attn_ext uses strides natively.
        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        Q = ggml_permute(g, Q, 0, 2, 1, 3);  // [hd, T, nh] (non-contiguous view)
        K = ggml_reshape_3d(g, K, hd, nh, T);
        K = ggml_permute(g, K, 0, 2, 1, 3);
        V = ggml_reshape_3d(g, V, hd, nh, T);
        V = ggml_permute(g, V, 0, 2, 1, 3);

        // SDPA — bidirectional, no mask (same pattern as vit_embed.cpp)
        ggml_tensor* attn = ggml_flash_attn_ext(g, Q, K, V, nullptr,
                                                 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        // flash_attn_ext output: [hd, nh, T] — reshape directly to [D, T]
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection
        attn = g_linear(g, attn, blk.proj_w, blk.proj_b);
        x = ggml_add(g, res, attn);

        // FFN with quick_gelu
        res = x;
        h = g_ln(g, x, blk.ln2_w, blk.ln2_b, 1e-5f);
        ggml_tensor* up = g_linear(g, h, blk.ffn_up_w, blk.ffn_up_b);
        up = ggml_gelu_quick(g, up);
        ggml_tensor* down = g_linear(g, up, blk.ffn_down_w, blk.ffn_down_b);
        x = ggml_add(g, res, down);

        // Per-layer output for diff bisection (only first 2 layers to avoid OOM)
        if (clip_debug && li < 2) {
            char nm[32]; snprintf(nm, sizeof(nm), "clip_layer_%d", li);
            ggml_set_name(x, nm); ggml_set_output(x);
        }
    }

    ggml_set_name(x, "clip_output"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// Encode SAM features through the CLIP-L/14 vision encoder.
// Input: sam_features_chw in CHW format (ds2_ch, ds2_H, ds2_W) — typically (1024, 16, 16).
// The SAM output is already in row-major [n_vis, vis_D] = [256, 1024] layout from encode_sam.
// Output: clip_out [256, 1024] (CLS skipped).
static bool encode_clip(uocr_ctx &ctx, const float *sam_features, int n_vis, int sam_dim,
                        std::vector<float> &clip_out) {
    auto &chp = ctx.m.chp;
    int D = chp.hidden;  // 1024

    // SAM output is [n_vis, sam_dim] = [256, 1024] — these ARE the patch embeddings.
    // sam_dim should equal D (both 1024).
    int n_patches = n_vis;  // 256
    int T = 1 + n_patches;  // 257 (CLS + patches)

    // Build input: CLS token + SAM features as patch embeddings + position embeddings
    auto cls = to_f32(ctx.m.cls_token);    // [1024]
    auto pos = to_f32(ctx.m.clip_pos_embed); // [257, 1024] — stored as [257 * 1024]

    std::vector<float> input(T * D);
    // Position 0: CLS token
    for (int d = 0; d < D; d++)
        input[0 * D + d] = cls[d];
    // Positions 1..256: SAM features as patch embeddings
    if (sam_dim == D) {
        memcpy(input.data() + D, sam_features, (size_t)n_patches * D * sizeof(float));
    } else {
        for (int t = 0; t < n_patches; t++)
            for (int d = 0; d < D; d++)
                input[(1 + t) * D + d] = (d < sam_dim) ? sam_features[t * sam_dim + d] : 0.0f;
    }

    if (getenv("UOCR_CLIP_DBG")) {
        fprintf(stderr, "  [dbg] clip input (before pos) tok0[0:4]: %.6f %.6f %.6f %.6f\n",
                input[0], input[1], input[2], input[3]);
        fprintf(stderr, "  [dbg] clip input (before pos) tok1[0:4]: %.6f %.6f %.6f %.6f\n",
                input[D], input[D+1], input[D+2], input[D+3]);
        fprintf(stderr, "  [dbg] sam_features[0:4]: %.6f %.6f %.6f %.6f\n",
                sam_features[0], sam_features[1], sam_features[2], sam_features[3]);
        fprintf(stderr, "  [dbg] pos_embed tok1[0:4]: %.6f %.6f %.6f %.6f\n",
                pos[D], pos[D+1], pos[D+2], pos[D+3]);
    }

    // Add position embeddings
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            input[t * D + d] += pos[t * D + d];

    // Build and run the CLIP encoder graph (all 24 layers in one graph)
    size_t meta_sz = 32 * 1024 * 1024;
    std::vector<uint8_t> mb(meta_sz);
    ggml_init_params ip = { meta_sz, mb.data(), true };
    ggml_context* gc = ggml_init(ip);
    ggml_cgraph* gf = build_clip_enc_full_graph(gc, &ctx, T);
    ggml_backend_sched_reset(ctx.sched);
    ggml_backend_sched_alloc_graph(ctx.sched, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "clip_input"),
                            input.data(), 0, (size_t)T * D * sizeof(float));
    ggml_backend_sched_graph_compute(ctx.sched, gf);

    // Read full output [257, 1024] for diff comparison
    std::vector<float> full_output(T * D);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "clip_output"),
                            full_output.data(), 0, (size_t)T * D * sizeof(float));

    // Pre-LN output check
    if (getenv("UOCR_CLIP_DBG")) {
        ggml_tensor* pln = ggml_graph_get_tensor(gf, "clip_pre_ln_out");
        if (pln) {
            std::vector<float> pln_data(T * D);
            ggml_backend_tensor_get(pln, pln_data.data(), 0, (size_t)T * D * sizeof(float));
            fprintf(stderr, "  [dbg] clip_pre_ln tok0[0:4]: %.6f %.6f %.6f %.6f\n",
                    pln_data[0], pln_data[1], pln_data[2], pln_data[3]);
            fprintf(stderr, "  [dbg] clip_pre_ln tok1[0:4]: %.6f %.6f %.6f %.6f\n",
                    pln_data[D], pln_data[D+1], pln_data[D+2], pln_data[D+3]);
        }
    }

    // Per-layer CLIP diff (when UOCR_CLIP_DBG=1)
    if (getenv("UOCR_CLIP_DBG") && !ctx.diff_ref_path.empty()) {
        // Print C++ layer 0 first values for manual comparison
        {
            ggml_tensor* l0 = ggml_graph_get_tensor(gf, "clip_layer_0");
            if (l0) {
                std::vector<float> l0d(T * D);
                ggml_backend_tensor_get(l0, l0d.data(), 0, T * D * sizeof(float));
                fprintf(stderr, "  [dbg] C++ clip_layer_0 tok0[0:8]:");
                for (int i = 0; i < 8; i++) fprintf(stderr, " %.7f", l0d[i]);
                fprintf(stderr, "\n  [dbg] C++ clip_layer_0 tok1[0:8]:");
                for (int i = 0; i < 8; i++) fprintf(stderr, " %.7f", l0d[D+i]);
                fprintf(stderr, "\n");
            }
        }
        for (int li = 0; li < chp.depth; li++) {
            char nm[32]; snprintf(nm, sizeof(nm), "clip_layer_%d", li);
            ggml_tensor* lt = ggml_graph_get_tensor(gf, nm);
            if (!lt) continue;
            std::vector<float> layer_out(T * D);
            ggml_backend_tensor_get(lt, layer_out.data(), 0, (size_t)T * D * sizeof(float));
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                auto r = ref.compare(nm, layer_out.data(), (size_t)T * D);
                fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                        nm, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }
    }
    ggml_free(gc);

    // Diff: full CLIP output (257x1024, including CLS)
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("clip_output")) {
            auto r = ref.compare("clip_output", full_output.data(), (size_t)T * D);
            fprintf(stderr, "  clip_output: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // Skip CLS → output (256, 1024)
    clip_out.resize((size_t)n_patches * D);
    memcpy(clip_out.data(), full_output.data() + D, (size_t)n_patches * D * sizeof(float));

    return true;
}

// ---------------------------------------------------------------------------
// Fusion & Projection: concat(CLIP, SAM) → Linear(2048, 1280)
// ---------------------------------------------------------------------------

static bool fuse_and_project(uocr_ctx &ctx, const float *clip_features,
                              const float *sam_features, int n_tokens,
                              int clip_dim, int sam_dim,
                              std::vector<float> &proj_out) {
    // Fuse: concat along feature dimension
    int fused_dim = clip_dim + sam_dim;  // 1024 + 1024 = 2048
    std::vector<float> fused((size_t)n_tokens * fused_dim);
    for (int t = 0; t < n_tokens; t++) {
        memcpy(fused.data() + t * fused_dim,
               clip_features + t * clip_dim, clip_dim * sizeof(float));
        memcpy(fused.data() + t * fused_dim + clip_dim,
               sam_features + t * sam_dim, sam_dim * sizeof(float));
    }

    // Diff: fused features
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("fused_features")) {
            auto r = ref.compare("fused_features", fused.data(), (size_t)n_tokens * fused_dim);
            fprintf(stderr, "  fused_features: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    // Project: Linear(2048, 1280)
    auto &lhp = ctx.m.lhp;
    int out_dim = lhp.hidden;  // 1280
    auto pw = to_f32(ctx.m.projector_w);
    auto pb = to_f32(ctx.m.projector_b);

    proj_out.resize((size_t)n_tokens * out_dim);
    for (int t = 0; t < n_tokens; t++)
        linear_cpu(fused.data() + t * fused_dim, proj_out.data() + t * out_dim,
                   fused_dim, out_dim, pw.data(), pb.empty() ? nullptr : pb.data());

    // Diff: projector output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("projector_output")) {
            auto r = ref.compare("projector_output", proj_out.data(), (size_t)n_tokens * out_dim);
            fprintf(stderr, "  projector_output: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Vision features assembly (with image_newline)
// ---------------------------------------------------------------------------

// Reshape projected features into spatial grid, append image_newline per row,
// flatten, append view_separator. Output: (h*w + h + 1, D) = (272 + 1, 1280).
static bool assemble_vision_features(uocr_ctx &ctx, const float *proj_features,
                                      int n_tokens, int D,
                                      std::vector<float> &vis_features,
                                      int &out_n_vis) {
    int hw = n_tokens;  // 256
    int side = (int)sqrtf((float)hw);  // 16
    assert(side * side == hw);

    auto img_newline = to_f32(ctx.m.image_newline);  // [D]

    // For each row: 16 features + 1 newline → 17 per row, 16 rows → 272
    int features_per_row = side + 1;  // 17
    int n_grid = side * features_per_row;  // 272
    int n_total = n_grid + 1;  // 273 (+ view_separator)

    vis_features.resize((size_t)n_total * D);

    for (int row = 0; row < side; row++) {
        // Copy the row's features (row-major grid; verified correct — a transposed
        // grid wrongly maps the bottom image line to the top position).
        memcpy(vis_features.data() + (size_t)(row * features_per_row) * D,
               proj_features + (size_t)(row * side) * D,
               (size_t)side * D * sizeof(float));
        // Append image_newline
        memcpy(vis_features.data() + (size_t)(row * features_per_row + side) * D,
               img_newline.data(), D * sizeof(float));
    }

    // Append view_separator at the end
    auto vsep = to_f32(ctx.m.view_separator);
    memcpy(vis_features.data() + (size_t)n_grid * D, vsep.data(), D * sizeof(float));

    out_n_vis = n_total;

    // Diff: vision features
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("vision_features")) {
            auto r = ref.compare("vision_features", vis_features.data(), (size_t)n_total * D);
            fprintf(stderr, "  vision_features: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// LLM decoder — ggml graph for attention + CPU-scalar MoE FFN
// ---------------------------------------------------------------------------

static bool stack_moe_experts(uocr_ctx &ctx) {
    int n_exp = ctx.m.lhp.n_experts;
    int n_moe = 0;
    for (auto &ly : ctx.m.llm_layers)
        if ((int)ly.experts.size() == n_exp && ly.experts[0].gate_w) n_moe++;
    if (n_moe == 0) return false;

    ggml_init_params ip = { (size_t)n_moe * 3 * ggml_tensor_overhead() + 4096, nullptr, true };
    ctx.moe_ctx = ggml_init(ip);
    if (!ctx.moe_ctx) return false;

    for (auto &ly : ctx.m.llm_layers) {
        if ((int)ly.experts.size() != n_exp || !ly.experts[0].gate_w) continue;
        auto &e0 = ly.experts[0];
        ly.gate_exps = ggml_new_tensor_3d(ctx.moe_ctx, e0.gate_w->type, e0.gate_w->ne[0], e0.gate_w->ne[1], n_exp);
        ly.up_exps   = ggml_new_tensor_3d(ctx.moe_ctx, e0.up_w->type,   e0.up_w->ne[0],   e0.up_w->ne[1],   n_exp);
        ly.down_exps = ggml_new_tensor_3d(ctx.moe_ctx, e0.down_w->type, e0.down_w->ne[0], e0.down_w->ne[1], n_exp);
    }

    ctx.moe_buf = ggml_backend_alloc_ctx_tensors(ctx.moe_ctx, ctx.backend);
    if (!ctx.moe_buf) { ggml_free(ctx.moe_ctx); ctx.moe_ctx = nullptr; return false; }

    std::vector<uint8_t> tmp;
    auto fill = [&](ggml_tensor *stacked, const std::vector<moe_expert_w> &exps,
                    ggml_tensor *moe_expert_w::*member) {
        for (int e = 0; e < n_exp; e++) {
            ggml_tensor *src = exps[e].*member;
            size_t nb = ggml_nbytes(src);
            if (nb != stacked->nb[2]) return false;
            tmp.resize(nb);
            ggml_backend_tensor_get(src, tmp.data(), 0, nb);
            ggml_backend_tensor_set(stacked, tmp.data(), (size_t)e * stacked->nb[2], nb);
        }
        return true;
    };
    for (auto &ly : ctx.m.llm_layers) {
        if (!ly.gate_exps) continue;
        if (!fill(ly.gate_exps, ly.experts, &moe_expert_w::gate_w) ||
            !fill(ly.up_exps,   ly.experts, &moe_expert_w::up_w) ||
            !fill(ly.down_exps, ly.experts, &moe_expert_w::down_w))
            return false;
    }
    return true;
}

// Persistent T=1 decode graph
struct PdGraph {
    std::vector<uint8_t> meta;
    ggml_context* gctx = nullptr;
    ggml_cgraph*  gf   = nullptr;
    ggml_tensor* t_cur_tok_id = nullptr;
    ggml_tensor* t_pos_ids    = nullptr;
    ggml_tensor* t_mask       = nullptr;
    std::vector<ggml_tensor*> t_k_cache;
    std::vector<ggml_tensor*> t_v_cache;
    ggml_tensor* t_logits = nullptr;
    std::vector<ggml_tensor*> t_k_new;
    std::vector<ggml_tensor*> t_layer_out;  // per-layer hidden state for debugging
    std::vector<ggml_tensor*> t_v_new;
    int max_kv = 0;
};

struct llm_attn_graph {
    ggml_cgraph *gf{};
    ggml_context *gctx{};
    std::vector<uint8_t> meta;
};

static PdGraph build_persistent_decode_graph(uocr_ctx &ctx, int max_kv) {
    auto &lhp = ctx.m.lhp;
    int D = lhp.hidden, V = lhp.vocab_size;
    int nh = lhp.heads, nkv = lhp.kv_heads, hd = lhp.head_dim;
    int n_layers = lhp.n_layers, kv_dim = nkv * hd;
    float eps = lhp.rms_eps;
    int max_kv_in = max_kv - 1;

    PdGraph pd;
    pd.max_kv = max_kv;
    pd.t_k_cache.resize(n_layers); pd.t_v_cache.resize(n_layers);
    pd.t_k_new.resize(n_layers);   pd.t_v_new.resize(n_layers);

    pd.meta.resize(8 * 1024 * 1024);
    ggml_init_params ip = {pd.meta.size(), pd.meta.data(), true};
    pd.gctx = ggml_init(ip);
    auto* g = pd.gctx;
    pd.gf = ggml_new_graph_custom(g, 8192, false);

    auto rmsnorm = [&](ggml_tensor* t, ggml_tensor* w) -> ggml_tensor* {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    // Use pre-dequanted F32 embedding as input (matches rebuild path's get_embedding)
    pd.t_cur_tok_id = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, 1);
    ggml_set_name(pd.t_cur_tok_id, "pd_emb"); ggml_set_input(pd.t_cur_tok_id);
    ggml_tensor* x = pd.t_cur_tok_id;

    pd.t_pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pd.t_pos_ids, "pd_pos"); ggml_set_input(pd.t_pos_ids);

    pd.t_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, max_kv, 1);
    ggml_set_name(pd.t_mask, "pd_mask"); ggml_set_input(pd.t_mask);

    for (int li = 0; li < n_layers; li++) {
        bool is_dense      = (li == 0);
        bool moe_in_graph  = ctx.moe_metal && !is_dense;
        auto& ly = ctx.m.llm_layers[li];

        char kn[16], vn[16], kon[16], von[16];
        snprintf(kn,  sizeof(kn),  "pd_kc%d", li);
        snprintf(vn,  sizeof(vn),  "pd_vc%d", li);
        snprintf(kon, sizeof(kon), "pd_kn%d", li);
        snprintf(von, sizeof(von), "pd_vn%d", li);

        // Use F32 for PD KV cache when UOCR_OPT_PD_F32=1 to avoid F16 precision issues
        ggml_type kv_type = getenv("UOCR_OPT_PD_F32") ? GGML_TYPE_F32 : GGML_TYPE_F16;
        pd.t_k_cache[li] = ggml_new_tensor_2d(g, kv_type, kv_dim, max_kv_in);
        ggml_set_name(pd.t_k_cache[li], kn); ggml_set_input(pd.t_k_cache[li]);
        pd.t_v_cache[li] = ggml_new_tensor_2d(g, kv_type, kv_dim, max_kv_in);
        ggml_set_name(pd.t_v_cache[li], vn); ggml_set_input(pd.t_v_cache[li]);

        ggml_tensor* h = rmsnorm(x, ly.in_ln_w);
        ggml_tensor* Q = ggml_mul_mat(g, ly.q_w, h);
        ggml_tensor* K = ggml_mul_mat(g, ly.k_w, h);
        ggml_tensor* V = ggml_mul_mat(g, ly.v_w, h);

        Q = ggml_reshape_3d(g, Q, hd, nh, 1);
        K = ggml_reshape_3d(g, K, hd, nkv, 1);
        V = ggml_reshape_3d(g, V, hd, nkv, 1);

        Q = ggml_rope_ext(g, Q, pd.t_pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, pd.t_pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor* Kc = ggml_cont(g, K);
        ggml_tensor* Vc = ggml_cont(g, V);

        pd.t_k_new[li] = ggml_cont(g, ggml_reshape_2d(g, Kc, kv_dim, 1));
        ggml_set_name(pd.t_k_new[li], kon); ggml_set_output(pd.t_k_new[li]);
        pd.t_v_new[li] = ggml_cont(g, ggml_reshape_2d(g, Vc, kv_dim, 1));
        ggml_set_name(pd.t_v_new[li], von); ggml_set_output(pd.t_v_new[li]);

        ggml_tensor* kp_3d = ggml_reshape_3d(g, pd.t_k_cache[li], hd, nkv, max_kv_in);
        ggml_tensor* kp    = kp_3d->type == GGML_TYPE_F32 ? kp_3d : ggml_cast(g, kp_3d, GGML_TYPE_F32);
        ggml_tensor* Kfull = ggml_concat(g, kp, Kc, 2);
        ggml_tensor* vp_3d = ggml_reshape_3d(g, pd.t_v_cache[li], hd, nkv, max_kv_in);
        ggml_tensor* vp    = vp_3d->type == GGML_TYPE_F32 ? vp_3d : ggml_cast(g, vp_3d, GGML_TYPE_F32);
        ggml_tensor* Vfull = ggml_concat(g, vp, Vc, 2);

        Q     = ggml_cont(g, ggml_permute(g, Q,     0, 2, 1, 3));
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

        float attn_scale = 1.0f / sqrtf((float)hd);
        ggml_tensor* attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull,
                                                 pd.t_mask, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(g, attn, D, 1);
        attn = ggml_mul_mat(g, ly.o_w, attn);
        x = ggml_add(g, x, attn);

        if (is_dense) {
            ggml_tensor* res  = x;
            h = rmsnorm(x, ly.post_ln_w);
            ggml_tensor* gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
            ggml_tensor* up   = ggml_mul_mat(g, ly.ffn_up_w, h);
            x = ggml_add(g, res, ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up)));
        } else if (moe_in_graph) {
            int n_exp = lhp.n_experts, Kk = lhp.n_experts_top;
            ggml_tensor* res = x;
            ggml_tensor* hn  = rmsnorm(x, ly.post_ln_w);
            ggml_tensor* lgt = ggml_mul_mat(g, ly.router_w, hn);
            ggml_tensor* prb = ggml_soft_max(g, lgt);
            ggml_tensor* ids = ggml_top_k(g, prb, Kk);
            ggml_tensor* p3  = ggml_reshape_3d(g, prb, 1, n_exp, 1);
            ggml_tensor* tw  = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), Kk, 1);
            tw  = ggml_scale(g, tw, lhp.routed_scaling_factor);
            ggml_tensor* hn3 = ggml_reshape_3d(g, hn, D, 1, 1);
            ggml_tensor* hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, Kk, 1));
            ggml_tensor* gt  = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));
            ggml_tensor* up  = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
            ggml_tensor* dn  = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gt, up), ids);
            ggml_tensor* dnp = ggml_cont(g, ggml_permute(g, dn, 1, 0, 2, 3));
            ggml_tensor* wc  = ggml_reshape_3d(g, tw, Kk, 1, 1);
            ggml_tensor* rt  = ggml_reshape_2d(g, ggml_mul_mat(g, wc, dnp), D, 1);
            ggml_tensor* sg  = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
            ggml_tensor* su  = ggml_mul_mat(g, ly.shared_up_w, hn);
            ggml_tensor* sh  = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));
            x = ggml_add(g, res, ggml_add(g, rt, sh));
        }

        // Debug: mark per-layer output for comparison
        if (getenv("UOCR_PD_DBG")) {
            char ln[16]; snprintf(ln, sizeof(ln), "pd_l%d", li);
            ggml_tensor* xd = ggml_cont(g, x);
            ggml_set_name(xd, ln); ggml_set_output(xd);
            pd.t_layer_out.push_back(xd);
        }
    }

    ggml_tensor* normed = rmsnorm(x, ctx.m.output_norm_w);
    ggml_tensor* lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    pd.t_logits = ggml_mul_mat(g, lm_w, normed);
    ggml_set_name(pd.t_logits, "pd_logits"); ggml_set_output(pd.t_logits);

    ggml_build_forward_expand(pd.gf, pd.t_logits);
    for (int li = 0; li < n_layers; li++) {
        ggml_build_forward_expand(pd.gf, pd.t_k_new[li]);
        ggml_build_forward_expand(pd.gf, pd.t_v_new[li]);
    }
    for (auto* lo : pd.t_layer_out)
        ggml_build_forward_expand(pd.gf, lo);
    return pd;
}

static llm_attn_graph build_llm_layer_attn(uocr_ctx &ctx, int li, int T, int n_past,
                                            bool include_ffn, bool include_moe = false) {
    auto &lhp = ctx.m.lhp;
    auto &ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads;
    int hd = lhp.head_dim;
    int Lk = n_past + T;
    float eps = lhp.rms_eps;

    size_t meta_sz = 4 * 1024 * 1024;
    llm_attn_graph lag;
    lag.meta.resize(meta_sz);
    ggml_init_params ip = { meta_sz, lag.meta.data(), true };
    lag.gctx = ggml_init(ip);
    auto *g = lag.gctx;
    lag.gf = ggml_new_graph_custom(g, 4096, false);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "layer_input"); ggml_set_input(x);

    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor *k_cache_in = nullptr, *v_cache_in = nullptr;
    if (n_past > 0) {
        k_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, nkv * hd, n_past);
        ggml_set_name(k_cache_in, "k_cache_in"); ggml_set_input(k_cache_in);
        v_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, nkv * hd, n_past);
        ggml_set_name(v_cache_in, "v_cache_in"); ggml_set_input(v_cache_in);
    }

    ggml_tensor *h = rmsnorm(x, ly.in_ln_w);
    ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, h);
    ggml_tensor *K = ggml_mul_mat(g, ly.k_w, h);
    ggml_tensor *V = ggml_mul_mat(g, ly.v_w, h);

    Q = ggml_reshape_3d(g, Q, hd, nh, T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);

    Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor *Kc = ggml_cont(g, K);
    ggml_tensor *Vc = ggml_cont(g, V);

    ggml_tensor *K_new = ggml_cont(g, ggml_reshape_2d(g, Kc, nkv * hd, T));
    ggml_set_name(K_new, "k_out"); ggml_set_output(K_new);
    ggml_tensor *V_new = ggml_cont(g, ggml_reshape_2d(g, Vc, nkv * hd, T));
    ggml_set_name(V_new, "v_out"); ggml_set_output(V_new);

    ggml_tensor *Kfull, *Vfull;
    if (n_past > 0) {
        ggml_tensor *kc3 = ggml_reshape_3d(g, k_cache_in, hd, nkv, n_past);
        Kfull = ggml_concat(g, kc3, Kc, 2);
        ggml_tensor *vc3 = ggml_reshape_3d(g, v_cache_in, hd, nkv, n_past);
        Vfull = ggml_concat(g, vc3, Vc, 2);
    } else {
        Kfull = Kc;
        Vfull = Vc;
    }

    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
    Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

    ggml_tensor *mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    float attn_scale = 1.0f / sqrtf((float)hd);
    ggml_tensor *attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, mask, attn_scale, 0.0f, 0.0f);
    if (getenv("UOCR_FA_F32")) ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);  // debug
    // flash_attn_ext output is [hd, nh, T]; reshape directly to [D, T] (llama.cpp
    // pattern). An intervening permute(0,2,1,3) scrambles head/token data whenever
    // T>1 (it is only a no-op for the T=1 decode step), corrupting the prefill.
    attn = ggml_reshape_2d(g, attn, D, T);
    attn = ggml_mul_mat(g, ly.o_w, attn);
    x = ggml_add(g, x, attn);

    if (include_ffn) {
        ggml_tensor *residual = x;
        h = rmsnorm(x, ly.post_ln_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);
        ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
        x = ggml_add(g, residual, ffn);
    } else if (include_moe) {
        int n_exp = lhp.n_experts, K = lhp.n_experts_top;
        ggml_tensor *residual = x;
        ggml_tensor *hn = rmsnorm(x, ly.post_ln_w);

        ggml_tensor *logits = ggml_mul_mat(g, ly.router_w, hn);
        ggml_tensor *probs  = ggml_soft_max(g, logits);
        ggml_tensor *ids    = ggml_top_k(g, probs, K);
        ggml_tensor *p3     = ggml_reshape_3d(g, probs, 1, n_exp, T);
        ggml_tensor *top_w  = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), K, T);
        top_w = ggml_scale(g, top_w, lhp.routed_scaling_factor);

        ggml_tensor *hn3 = ggml_reshape_3d(g, hn, D, 1, T);
        ggml_tensor *hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, K, T));
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));
        ggml_tensor *up   = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
        ggml_tensor *down = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gate, up), ids);

        ggml_tensor *down_p = ggml_cont(g, ggml_permute(g, down, 1, 0, 2, 3));
        ggml_tensor *w_col  = ggml_reshape_3d(g, top_w, K, 1, T);
        ggml_tensor *routed = ggml_reshape_2d(g, ggml_mul_mat(g, w_col, down_p), D, T);

        ggml_tensor *sg = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
        ggml_tensor *su = ggml_mul_mat(g, ly.shared_up_w, hn);
        ggml_tensor *shared = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));

        x = ggml_add(g, residual, ggml_add(g, routed, shared));
    }

    ggml_set_name(x, "layer_output"); ggml_set_output(x);
    ggml_build_forward_expand(lag.gf, x);
    ggml_build_forward_expand(lag.gf, K_new);
    ggml_build_forward_expand(lag.gf, V_new);
    return lag;
}

// CPU-scalar MoE FFN for one layer
static void moe_ffn_cpu(uocr_ctx &ctx, int li, float *hidden, int T) {
    auto &lhp = ctx.m.lhp;
    auto &ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, inter_e = lhp.expert_intermediate;
    int inter_s = lhp.shared_intermediate;
    int n_exp = lhp.n_experts, top_k = lhp.n_experts_top;
    float scale = lhp.routed_scaling_factor;
    float eps = lhp.rms_eps;

    auto post_ln = to_f32(ly.post_ln_w);
    auto router = to_f32(ly.router_w);

    auto sh_gw = to_f32(ly.shared_gate_w);
    auto sh_uw = to_f32(ly.shared_up_w);
    auto sh_dw = to_f32(ly.shared_down_w);

    struct exp_w { std::vector<float> gw, uw, dw; };
    std::vector<exp_w> exp_ws(n_exp);

    std::vector<float> normed_all((size_t)T * D);
    std::vector<std::array<int, 16>> tk_idx(T);
    std::vector<std::array<float, 16>> tk_w(T);
    std::vector<char> used(n_exp, 0);
    for (int t = 0; t < T; t++) {
        float *normed = normed_all.data() + (size_t)t * D;
        rmsnorm_cpu(hidden + t * D, normed, D, post_ln.data(), eps);
        std::vector<float> logits(n_exp);
        for (int e = 0; e < n_exp; e++) {
            float dot = 0;
            for (int d = 0; d < D; d++) dot += normed[d] * router[e * D + d];
            logits[e] = dot;
        }
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0;
        for (int e = 0; e < n_exp; e++) { logits[e] = expf(logits[e] - max_l); sum_exp += logits[e]; }
        for (int e = 0; e < n_exp; e++) logits[e] /= sum_exp;
        std::vector<std::pair<float, int>> scored(n_exp);
        for (int e = 0; e < n_exp; e++) scored[e] = {logits[e], e};
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](auto &a, auto &b) { return a.first > b.first; });
        for (int k = 0; k < top_k; k++) {
            tk_idx[t][k] = scored[k].second;
            tk_w[t][k]   = scored[k].first * scale;
            used[scored[k].second] = 1;
        }
    }

    for (int e = 0; e < n_exp; e++)
        if (used[e]) {
            exp_ws[e].gw = to_f32(ly.experts[e].gate_w);
            exp_ws[e].uw = to_f32(ly.experts[e].up_w);
            exp_ws[e].dw = to_f32(ly.experts[e].down_w);
        }

    int nthreads = std::max(1, ctx.n_threads);
    if (nthreads > T) nthreads = std::max(1, T);
    auto worker = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            const float *normed = normed_all.data() + (size_t)t * D;
            float *tok = hidden + t * D;
            std::vector<float> routed_out(D, 0.0f), expert_out(D);
            for (int k = 0; k < top_k; k++) {
                int eid = tk_idx[t][k]; float w = tk_w[t][k];
                swiglu_ffn_cpu(normed, expert_out.data(), D, inter_e,
                               exp_ws[eid].gw.data(), exp_ws[eid].uw.data(), exp_ws[eid].dw.data());
                for (int d = 0; d < D; d++) routed_out[d] += w * expert_out[d];
            }
            std::vector<float> shared_out(D);
            swiglu_ffn_cpu(normed, shared_out.data(), D, inter_s,
                           sh_gw.data(), sh_uw.data(), sh_dw.data());
            for (int d = 0; d < D; d++) tok[d] += routed_out[d] + shared_out[d];
        }
    };
    if (nthreads <= 1) worker(0, T);
    else {
        std::vector<std::thread> pool;
        int chunk = (T + nthreads - 1) / nthreads;
        for (int ti = 0; ti < nthreads; ti++) {
            int t0 = ti * chunk, t1 = std::min(T, t0 + chunk);
            if (t0 < t1) pool.emplace_back(worker, t0, t1);
        }
        for (auto &th : pool) th.join();
    }
}

// ---------------------------------------------------------------------------
// Fused multi-layer decode graph (UOCR_OPT_FUSED_DECODE)
// Builds all 12 LLM layers in one ggml graph, eliminating 11 graph builds per step.
// ---------------------------------------------------------------------------

struct fused_decode_graph {
    ggml_cgraph *gf{};
    ggml_context *gctx{};
    std::vector<uint8_t> meta;
    ggml_tensor *t_input{};    // [D, T]
    ggml_tensor *t_pos{};      // [T]
    ggml_tensor *t_mask{};     // [Lk, T] F16
    std::vector<ggml_tensor*> t_k_in;   // per-layer [kv_dim, n_past]
    std::vector<ggml_tensor*> t_v_in;   // per-layer [kv_dim, n_past]
    std::vector<ggml_tensor*> t_k_out;  // per-layer [kv_dim, T]
    std::vector<ggml_tensor*> t_v_out;  // per-layer [kv_dim, T]
    ggml_tensor *t_output{};   // [D, T]
};

static fused_decode_graph build_fused_decode(uocr_ctx &ctx, int T, int n_past) {
    auto &lhp = ctx.m.lhp;
    int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads;
    int hd = lhp.head_dim, n_layers = lhp.n_layers;
    int Lk = n_past + T, kv_dim = nkv * hd;
    float eps = lhp.rms_eps;

    fused_decode_graph fd;
    fd.t_k_in.resize(n_layers); fd.t_v_in.resize(n_layers);
    fd.t_k_out.resize(n_layers); fd.t_v_out.resize(n_layers);

    // Large metadata buffer for all 12 layers
    size_t meta_sz = 32 * 1024 * 1024;
    fd.meta.resize(meta_sz);
    ggml_init_params ip = {meta_sz, fd.meta.data(), true};
    fd.gctx = ggml_init(ip);
    auto *g = fd.gctx;
    fd.gf = ggml_new_graph_custom(g, 16384, false);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    fd.t_input = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(fd.t_input, "fd_input"); ggml_set_input(fd.t_input);

    fd.t_pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(fd.t_pos, "fd_pos"); ggml_set_input(fd.t_pos);

    fd.t_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(fd.t_mask, "fd_mask"); ggml_set_input(fd.t_mask);

    ggml_tensor *x = fd.t_input;

    for (int li = 0; li < n_layers; li++) {
        auto &ly = ctx.m.llm_layers[li];
        bool is_dense = (li == 0);
        bool moe_in_graph = ctx.moe_metal && !is_dense;

        // Per-layer KV cache inputs
        char kn[16], vn[16], kon[16], von[16];
        snprintf(kn, sizeof(kn), "fd_ki%d", li);
        snprintf(vn, sizeof(vn), "fd_vi%d", li);
        snprintf(kon, sizeof(kon), "fd_ko%d", li);
        snprintf(von, sizeof(von), "fd_vo%d", li);

        ggml_tensor *k_cache_in = nullptr, *v_cache_in = nullptr;
        if (n_past > 0) {
            k_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, n_past);
            ggml_set_name(k_cache_in, kn); ggml_set_input(k_cache_in);
            fd.t_k_in[li] = k_cache_in;
            v_cache_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, n_past);
            ggml_set_name(v_cache_in, vn); ggml_set_input(v_cache_in);
            fd.t_v_in[li] = v_cache_in;
        }

        // Attention
        ggml_tensor *h = rmsnorm(x, ly.in_ln_w);
        ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, h);
        ggml_tensor *K = ggml_mul_mat(g, ly.k_w, h);
        ggml_tensor *V = ggml_mul_mat(g, ly.v_w, h);

        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K = ggml_reshape_3d(g, K, hd, nkv, T);
        V = ggml_reshape_3d(g, V, hd, nkv, T);

        Q = ggml_rope_ext(g, Q, fd.t_pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, fd.t_pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor *Kc = ggml_cont(g, K);
        ggml_tensor *Vc = ggml_cont(g, V);

        // Output new K/V for cache update
        fd.t_k_out[li] = ggml_cont(g, ggml_reshape_2d(g, Kc, kv_dim, T));
        ggml_set_name(fd.t_k_out[li], kon); ggml_set_output(fd.t_k_out[li]);
        fd.t_v_out[li] = ggml_cont(g, ggml_reshape_2d(g, Vc, kv_dim, T));
        ggml_set_name(fd.t_v_out[li], von); ggml_set_output(fd.t_v_out[li]);

        // Concatenate with KV cache
        ggml_tensor *Kfull, *Vfull;
        if (n_past > 0) {
            Kfull = ggml_concat(g, ggml_reshape_3d(g, k_cache_in, hd, nkv, n_past), Kc, 2);
            Vfull = ggml_concat(g, ggml_reshape_3d(g, v_cache_in, hd, nkv, n_past), Vc, 2);
        } else {
            Kfull = Kc; Vfull = Vc;
        }

        Q     = ggml_cont(g, ggml_permute(g, Q,     0, 2, 1, 3));
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

        float attn_scale = 1.0f / sqrtf((float)hd);
        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, fd.t_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);
        attn = ggml_mul_mat(g, ly.o_w, attn);
        x = ggml_add(g, x, attn);

        // FFN
        if (is_dense) {
            ggml_tensor *residual = x;
            h = rmsnorm(x, ly.post_ln_w);
            ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
            ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);
            ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
            x = ggml_add(g, residual, ffn);
        } else if (moe_in_graph) {
            int n_exp = lhp.n_experts, Kk = lhp.n_experts_top;
            ggml_tensor *residual = x;
            ggml_tensor *hn = rmsnorm(x, ly.post_ln_w);
            ggml_tensor *lgt = ggml_mul_mat(g, ly.router_w, hn);
            ggml_tensor *prb = ggml_soft_max(g, lgt);
            ggml_tensor *ids = ggml_top_k(g, prb, Kk);
            ggml_tensor *p3  = ggml_reshape_3d(g, prb, 1, n_exp, T);
            ggml_tensor *tw  = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), Kk, T);
            tw = ggml_scale(g, tw, lhp.routed_scaling_factor);
            ggml_tensor *hn3 = ggml_reshape_3d(g, hn, D, 1, T);
            ggml_tensor *hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, Kk, T));
            ggml_tensor *gt  = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));
            ggml_tensor *up  = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
            ggml_tensor *dn  = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gt, up), ids);
            ggml_tensor *dnp = ggml_cont(g, ggml_permute(g, dn, 1, 0, 2, 3));
            ggml_tensor *wc  = ggml_reshape_3d(g, tw, Kk, 1, T);
            ggml_tensor *rt  = ggml_reshape_2d(g, ggml_mul_mat(g, wc, dnp), D, T);
            ggml_tensor *sg  = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
            ggml_tensor *su  = ggml_mul_mat(g, ly.shared_up_w, hn);
            ggml_tensor *sh  = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));
            x = ggml_add(g, residual, ggml_add(g, rt, sh));
        }
        // Note: if !is_dense && !moe_in_graph, MoE is done on CPU after graph eval
    }

    fd.t_output = x;
    ggml_set_name(fd.t_output, "fd_output"); ggml_set_output(fd.t_output);

    ggml_build_forward_expand(fd.gf, fd.t_output);
    for (int li = 0; li < n_layers; li++) {
        ggml_build_forward_expand(fd.gf, fd.t_k_out[li]);
        ggml_build_forward_expand(fd.gf, fd.t_v_out[li]);
    }
    return fd;
}

// ---------------------------------------------------------------------------
// Full LLM decoder forward
// ---------------------------------------------------------------------------

static bool run_llm_decoder(uocr_ctx &ctx, const float *prompt_embeds, int n_prompt, int max_new,
                            std::vector<int32_t> &out_ids, std::vector<float> &out_confs,
                            const std::vector<int32_t> &prompt_ids = {}) {
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

    ggml_tensor* emb_t = ctx.m.embed_tokens;
    const auto* emb_tt = ggml_get_type_traits(emb_t->type);
    const size_t emb_row_bytes = ggml_row_size(emb_t->type, D);
    std::vector<uint8_t> emb_row;
    auto get_embedding = [&](int32_t tok_id, float *out_emb) {
        const size_t off = (size_t)tok_id * emb_row_bytes;
        if (emb_t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(emb_t, out_emb, off, (size_t)D * sizeof(float));
        } else {
            emb_row.resize(emb_row_bytes);
            ggml_backend_tensor_get(emb_t, emb_row.data(), off, emb_row_bytes);
            emb_tt->to_float(emb_row.data(), out_emb, D);
        }
    };

    auto norm_w = to_f32(ctx.m.output_norm_w);
    ggml_tensor *lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    bool lmhead_cpu = getenv("UOCR_LMHEAD_CPU") != nullptr;
    std::vector<float> head_w;
    if (lmhead_cpu) head_w = to_f32(lm_w);

    bool no_kv = getenv("UOCR_NO_KV") != nullptr;
    bool fused_decode = getenv("UOCR_OPT_FUSED_DECODE") != nullptr;
    std::vector<float> full_emb(prompt_embeds, prompt_embeds + (size_t)n_prompt * D);

    int n_generated = 0;
    std::vector<int32_t> cur_tokens;
    int n_past = 0;

    static constexpr int PD_GEN_CAP = 96;
    // The persistent-decode (PD) graph is a speed optimization but still diverges
    // from the per-step rebuild path on vision-heavy prefills (its first decode
    // token matches after the KV zero-init fix, but later steps drift — a residual
    // Metal flash_attn numerics issue with the zero-padded KV layout). The rebuild
    // path is verified correct (byte-identical to the CPU-MoE reference), so it is
    // the default. Opt into PD with UOCR_PD=1 once the divergence is fixed.
    bool use_pd  = getenv("UOCR_PD") && !getenv("UOCR_DECODE_REBUILD") &&
                   ctx.moe_metal && !no_kv && !lmhead_cpu;
    int  pd_max_kv = use_pd ? std::min(n_prompt + std::min(max_new, PD_GEN_CAP),
                                        lhp.max_position_embeddings) : 0;
    PdGraph pd;
    std::vector<ggml_fp16_t> pd_mask;
    std::vector<float> pd_k_tmp(kv_dim), pd_v_tmp(kv_dim);
    bool pd_ready = false;
    if (use_pd && pd_max_kv > n_prompt) {
        pd = build_persistent_decode_graph(ctx, pd_max_kv);
        pd_mask.assign(pd_max_kv, ggml_fp32_to_fp16(-INFINITY));
        pd_mask[pd_max_kv - 1] = ggml_fp32_to_fp16(0.0f);
    } else {
        use_pd = false;
    }

    auto _decode_t0 = std::chrono::steady_clock::now();
    int _decode_gen_steps = 0;
    while (n_generated < max_new) {
        int T = no_kv ? (int)(full_emb.size() / D) : ((n_past == 0) ? n_prompt : (int)cur_tokens.size());
        if (getenv("UOCR_DBG"))
            fprintf(stderr, "  [dbg] decode step gen=%d n_past=%d T=%d pd=%d\n",
                    n_generated, n_past, T, (int)use_pd);

        bool did_pd = false;
        std::vector<float> logits(V, 0.0f);

        // === Persistent decode path for T=1 generation steps ===
        if (use_pd && n_past > 0) {
            if (n_past >= pd_max_kv) {
                if (getenv("UOCR_DBG"))
                    fprintf(stderr, "  [pd] KV full (n_past=%d >= max_kv=%d), fall back\n",
                            n_past, pd_max_kv);
                use_pd = false;
            }
        }
        if (use_pd && n_past > 0) {
            if (!pd_ready) {
                ggml_backend_sched_reset(ctx.sched);
                if (!ggml_backend_sched_alloc_graph(ctx.sched, pd.gf)) {
                    fprintf(stderr, "[unlimited_ocr] pd alloc failed — falling back to per-step rebuild\n");
                    use_pd = false;
                } else {
                    for (int ki = 0; ki < n_past; ki++)
                        pd_mask[ki] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, pd_mask.data(), 0,
                                            (size_t)pd_max_kv * sizeof(ggml_fp16_t));
                    // Zero the entire KV cache first. The graph allocates max_kv_in
                    // slots but only n_past are valid; the remaining slots are read
                    // by flash_attn before the (-inf) mask is applied, and on the
                    // shared scheduler they hold leftover garbage from the vision
                    // graphs. NaN/Inf garbage survives the mask (NaN + -inf = NaN)
                    // and corrupts every logit. Zeroing makes masked slots inert.
                    size_t kv_elem_sz = ggml_type_size(pd.t_k_cache[0]->type);
                    bool pd_f32 = (pd.t_k_cache[0]->type == GGML_TYPE_F32);
                    size_t kv_full = (size_t)(pd_max_kv - 1) * kv_dim;
                    std::vector<uint8_t> kv_zero(kv_full * kv_elem_sz, 0);
                    size_t kv_b_src = (size_t)n_past * kv_dim * sizeof(ggml_fp16_t);
                    size_t kv_b_dst = (size_t)n_past * kv_dim * kv_elem_sz;
                    // If PD is F32, convert the F16 KV cache to F32 for upload
                    std::vector<float> kv_f32_tmp;
                    if (pd_f32) kv_f32_tmp.resize((size_t)n_past * kv_dim);
                    for (int li = 0; li < n_layers; li++) {
                        ggml_backend_tensor_set(pd.t_k_cache[li], kv_zero.data(), 0,
                                                kv_full * kv_elem_sz);
                        ggml_backend_tensor_set(pd.t_v_cache[li], kv_zero.data(), 0,
                                                kv_full * kv_elem_sz);
                        if (pd_f32) {
                            for (int i = 0; i < n_past * kv_dim; i++)
                                kv_f32_tmp[i] = ggml_fp16_to_fp32(ctx.kvc.k_cache[li][i]);
                            ggml_backend_tensor_set(pd.t_k_cache[li], kv_f32_tmp.data(), 0, kv_b_dst);
                            for (int i = 0; i < n_past * kv_dim; i++)
                                kv_f32_tmp[i] = ggml_fp16_to_fp32(ctx.kvc.v_cache[li][i]);
                            ggml_backend_tensor_set(pd.t_v_cache[li], kv_f32_tmp.data(), 0, kv_b_dst);
                        } else {
                            ggml_backend_tensor_set(pd.t_k_cache[li],
                                                    ctx.kvc.k_cache[li].data(), 0, kv_b_src);
                            ggml_backend_tensor_set(pd.t_v_cache[li],
                                                    ctx.kvc.v_cache[li].data(), 0, kv_b_src);
                        }
                    }
                    pd_ready = true;
                }
            } else {
                int prev = n_past - 1;
                if (prev < pd_max_kv - 1) {
                    pd_mask[prev] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, &pd_mask[prev],
                                            (size_t)prev * sizeof(ggml_fp16_t),
                                            sizeof(ggml_fp16_t));
                }
                size_t kv_el = ggml_type_size(pd.t_k_cache[0]->type);
                bool pd_f32_inc = (pd.t_k_cache[0]->type == GGML_TYPE_F32);
                size_t off = (size_t)(n_past - 1) * kv_dim * kv_el;
                size_t sz  = (size_t)kv_dim * kv_el;
                std::vector<float> kv_inc_f32;
                if (pd_f32_inc) kv_inc_f32.resize(kv_dim);
                for (int li = 0; li < n_layers; li++) {
                    if (pd_f32_inc) {
                        for (int i = 0; i < kv_dim; i++)
                            kv_inc_f32[i] = ggml_fp16_to_fp32(ctx.kvc.k_cache[li][(n_past-1)*kv_dim + i]);
                        ggml_backend_tensor_set(pd.t_k_cache[li], kv_inc_f32.data(), off, sz);
                        for (int i = 0; i < kv_dim; i++)
                            kv_inc_f32[i] = ggml_fp16_to_fp32(ctx.kvc.v_cache[li][(n_past-1)*kv_dim + i]);
                        ggml_backend_tensor_set(pd.t_v_cache[li], kv_inc_f32.data(), off, sz);
                    } else {
                        ggml_backend_tensor_set(pd.t_k_cache[li],
                            ctx.kvc.k_cache[li].data() + (n_past - 1) * kv_dim, off, sz);
                        ggml_backend_tensor_set(pd.t_v_cache[li],
                            ctx.kvc.v_cache[li].data() + (n_past - 1) * kv_dim, off, sz);
                    }
                }
            }

            if (use_pd) {
                // Populate embedding via CPU dequant (matches rebuild path)
                std::vector<float> pd_emb(D);
                get_embedding(cur_tokens[0], pd_emb.data());
                ggml_backend_tensor_set(pd.t_cur_tok_id, pd_emb.data(), 0, D * sizeof(float));
                int32_t pos_id = n_past;
                ggml_backend_tensor_set(pd.t_pos_ids, &pos_id, 0, sizeof(int32_t));

                ggml_backend_sched_graph_compute(ctx.sched, pd.gf);

                // Dump per-layer hidden state for divergence debugging
                if (getenv("UOCR_PD_DBG") && n_generated >= 2 && n_generated <= 3) {
                    for (size_t i = 0; i < pd.t_layer_out.size(); i++) {
                        float buf[4];
                        ggml_backend_tensor_get(pd.t_layer_out[i], buf, 0, 4*sizeof(float));
                        fprintf(stderr, "  [pd_dbg] gen=%d layer=%zu: %.7f %.7f %.7f %.7f\n",
                                n_generated, i, buf[0], buf[1], buf[2], buf[3]);
                    }
                }

                ggml_backend_tensor_get(pd.t_logits, logits.data(), 0, (size_t)V * sizeof(float));

                for (int li = 0; li < n_layers; li++) {
                    ggml_backend_tensor_get(pd.t_k_new[li], pd_k_tmp.data(), 0, kv_dim * sizeof(float));
                    ggml_backend_tensor_get(pd.t_v_new[li], pd_v_tmp.data(), 0, kv_dim * sizeof(float));
                    size_t base = ctx.kvc.k_cache[li].size();
                    ctx.kvc.k_cache[li].resize(base + kv_dim);
                    ctx.kvc.v_cache[li].resize(base + kv_dim);
                    for (int i = 0; i < kv_dim; i++) {
                        ctx.kvc.k_cache[li][base + i] = ggml_fp32_to_fp16(pd_k_tmp[i]);
                        ctx.kvc.v_cache[li][base + i] = ggml_fp32_to_fp16(pd_v_tmp[i]);
                    }
                }
                n_past++;
                did_pd = true;
            }
        }

        if (!did_pd && fused_decode && ctx.moe_metal) {
            // === Fused multi-layer decode path (UOCR_OPT_FUSED_DECODE) ===
            std::vector<float> input_emb(T * D);
            if (no_kv) {
                memcpy(input_emb.data(), full_emb.data(), (size_t)T * D * sizeof(float));
            } else if (n_past == 0) {
                memcpy(input_emb.data(), prompt_embeds, (size_t)T * D * sizeof(float));
            } else {
                for (int t = 0; t < T; t++)
                    get_embedding(cur_tokens[t], input_emb.data() + t * D);
            }

            auto fd = build_fused_decode(ctx, T, n_past);
            ggml_backend_sched_reset(ctx.sched);
            if (ggml_backend_sched_alloc_graph(ctx.sched, fd.gf)) {
                ggml_backend_tensor_set(fd.t_input, input_emb.data(), 0, T * D * sizeof(float));

                std::vector<int32_t> pos(T);
                for (int t = 0; t < T; t++) pos[t] = n_past + t;
                ggml_backend_tensor_set(fd.t_pos, pos.data(), 0, T * sizeof(int32_t));

                int Lk = n_past + T;
                std::vector<ggml_fp16_t> mask(Lk * T);
                for (int qi = 0; qi < T; qi++)
                    for (int ki = 0; ki < Lk; ki++)
                        mask[qi * Lk + ki] = ggml_fp32_to_fp16(ki > n_past + qi ? -INFINITY : 0.0f);
                ggml_backend_tensor_set(fd.t_mask, mask.data(), 0, Lk * T * sizeof(ggml_fp16_t));

                // Set per-layer KV caches
                if (n_past > 0) {
                    int kv_n = kv_dim * n_past;
                    std::vector<float> k_f32(kv_n), v_f32(kv_n);
                    for (int li = 0; li < n_layers; li++) {
                        for (int i = 0; i < kv_n; i++) {
                            k_f32[i] = ggml_fp16_to_fp32(ctx.kvc.k_cache[li][i]);
                            v_f32[i] = ggml_fp16_to_fp32(ctx.kvc.v_cache[li][i]);
                        }
                        ggml_backend_tensor_set(fd.t_k_in[li], k_f32.data(), 0, kv_n * sizeof(float));
                        ggml_backend_tensor_set(fd.t_v_in[li], v_f32.data(), 0, kv_n * sizeof(float));
                    }
                }

                ggml_backend_sched_graph_compute(ctx.sched, fd.gf);

                // Read output hidden state
                std::vector<float> hidden(T * D);
                ggml_backend_tensor_get(fd.t_output, hidden.data(), 0, T * D * sizeof(float));

                // Store new K/V in cache
                int kv_nt = kv_dim * T;
                std::vector<float> k_new(kv_nt), v_new(kv_nt);
                for (int li = 0; li < n_layers; li++) {
                    ggml_backend_tensor_get(fd.t_k_out[li], k_new.data(), 0, kv_nt * sizeof(float));
                    ggml_backend_tensor_get(fd.t_v_out[li], v_new.data(), 0, kv_nt * sizeof(float));
                    if (!no_kv) {
                        size_t base = ctx.kvc.k_cache[li].size();
                        ctx.kvc.k_cache[li].resize(base + kv_nt);
                        ctx.kvc.v_cache[li].resize(base + kv_nt);
                        for (int i = 0; i < kv_nt; i++) {
                            ctx.kvc.k_cache[li][base + i] = ggml_fp32_to_fp16(k_new[i]);
                            ctx.kvc.v_cache[li][base + i] = ggml_fp32_to_fp16(v_new[i]);
                        }
                    }
                }

                if (!no_kv) n_past += T;

                // LM head
                std::vector<float> last_hidden(D);
                rmsnorm_cpu(hidden.data() + (T - 1) * D, last_hidden.data(), D, norm_w.data(), lhp.rms_eps);
                if (lmhead_cpu) {
                    linear_cpu(last_hidden.data(), logits.data(), D, V, head_w.data(), nullptr);
                } else {
                    size_t meta_sz2 = 1024 * 1024;
                    std::vector<uint8_t> mb2(meta_sz2);
                    ggml_init_params ip2 = {meta_sz2, mb2.data(), true};
                    ggml_context* gc2 = ggml_init(ip2);
                    ggml_tensor* lh_in = ggml_new_tensor_2d(gc2, GGML_TYPE_F32, D, 1);
                    ggml_set_name(lh_in, "lh_in"); ggml_set_input(lh_in);
                    ggml_tensor* lh_out = ggml_mul_mat(gc2, lm_w, lh_in);
                    ggml_set_name(lh_out, "lh_out"); ggml_set_output(lh_out);
                    ggml_cgraph* gf2 = ggml_new_graph(gc2);
                    ggml_build_forward_expand(gf2, lh_out);
                    ggml_backend_sched_reset(ctx.sched);
                    ggml_backend_sched_alloc_graph(ctx.sched, gf2);
                    ggml_backend_tensor_set(lh_in, last_hidden.data(), 0, D * sizeof(float));
                    ggml_backend_sched_graph_compute(ctx.sched, gf2);
                    ggml_backend_tensor_get(lh_out, logits.data(), 0, V * sizeof(float));
                    ggml_free(gc2);
                }

                did_pd = true;  // skip the per-layer rebuild path below
            }
            ggml_free(fd.gctx);
        }

        if (!did_pd) {
            // === Original per-step rebuild path ===
            std::vector<float> input_emb(T * D);
            if (no_kv) {
                memcpy(input_emb.data(), full_emb.data(), (size_t)T * D * sizeof(float));
            } else if (n_past == 0) {
                memcpy(input_emb.data(), prompt_embeds, (size_t)T * D * sizeof(float));
            } else {
                for (int t = 0; t < T; t++)
                    get_embedding(cur_tokens[t], input_emb.data() + t * D);
            }

            std::vector<float> hidden(input_emb);

            auto _rb_t0 = std::chrono::steady_clock::now();
            long long _rb_build_ms = 0, _rb_set_ms = 0, _rb_compute_ms = 0;
            for (int li = 0; li < n_layers; li++) {
                bool is_dense = (li == 0);
                bool moe_in_graph = ctx.moe_metal && !is_dense;

                auto _t1 = std::chrono::steady_clock::now();
                auto lag = build_llm_layer_attn(ctx, li, T, n_past, is_dense, moe_in_graph);
                ggml_backend_sched_reset(ctx.sched);
                if (!ggml_backend_sched_alloc_graph(ctx.sched, lag.gf)) {
                    ggml_free(lag.gctx);
                    return false;
                }

                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "layer_input"),
                                        hidden.data(), 0, T * D * sizeof(float));

                std::vector<int32_t> pos(T);
                for (int t = 0; t < T; t++) pos[t] = n_past + t;
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "pos_ids"),
                                        pos.data(), 0, T * sizeof(int32_t));

                if (n_past > 0) {
                    int kv_n = kv_dim * n_past;
                    std::vector<float> k_f32(kv_n), v_f32(kv_n);
                    for (int i = 0; i < kv_n; i++) {
                        k_f32[i] = ggml_fp16_to_fp32(ctx.kvc.k_cache[li][i]);
                        v_f32[i] = ggml_fp16_to_fp32(ctx.kvc.v_cache[li][i]);
                    }
                    ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "k_cache_in"),
                                            k_f32.data(), 0, kv_n * sizeof(float));
                    ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "v_cache_in"),
                                            v_f32.data(), 0, kv_n * sizeof(float));
                }

                int Lk = n_past + T;
                std::vector<ggml_fp16_t> mask(Lk * T);
                for (int qi = 0; qi < T; qi++)
                    for (int ki = 0; ki < Lk; ki++)
                        mask[qi * Lk + ki] = ggml_fp32_to_fp16(ki > n_past + qi ? -INFINITY : 0.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "mask"),
                                        mask.data(), 0, Lk * T * sizeof(ggml_fp16_t));

                auto _t2 = std::chrono::steady_clock::now();
                _rb_build_ms += std::chrono::duration_cast<std::chrono::milliseconds>(_t2 - _t1).count();

                ggml_backend_sched_graph_compute(ctx.sched, lag.gf);

                auto _t3 = std::chrono::steady_clock::now();
                _rb_compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(_t3 - _t2).count();

                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "layer_output"),
                                        hidden.data(), 0, T * D * sizeof(float));

                // Dump per-layer hidden state for divergence debugging
                if (getenv("UOCR_PD_DBG") && n_generated >= 2 && n_generated <= 3) {
                    fprintf(stderr, "  [rb_dbg] gen=%d layer=%d: %.7f %.7f %.7f %.7f\n",
                            n_generated, li, hidden[0], hidden[1], hidden[2], hidden[3]);
                }

                int kv_nt = kv_dim * T;
                std::vector<float> k_new(kv_nt), v_new(kv_nt);
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "k_out"),
                                        k_new.data(), 0, kv_nt * sizeof(float));
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "v_out"),
                                        v_new.data(), 0, kv_nt * sizeof(float));

                if (!no_kv) {
                    size_t base = ctx.kvc.k_cache[li].size();
                    ctx.kvc.k_cache[li].resize(base + kv_nt);
                    ctx.kvc.v_cache[li].resize(base + kv_nt);
                    for (int i = 0; i < kv_nt; i++) {
                        ctx.kvc.k_cache[li][base + i] = ggml_fp32_to_fp16(k_new[i]);
                        ctx.kvc.v_cache[li][base + i] = ggml_fp32_to_fp16(v_new[i]);
                    }
                }

                ggml_free(lag.gctx);

                if (!is_dense && !moe_in_graph) {
                    moe_ffn_cpu(ctx, li, hidden.data(), T);
                }

                // Diff comparison
                if (!ctx.diff_ref_path.empty() && n_past == 0) {
                    char name[64];
                    snprintf(name, sizeof(name), "llm_layer_%d", li);
                    crispembed_diff::Ref ref;
                    if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(name)) {
                        auto r = ref.compare(name, hidden.data(), T * D);
                        fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                                name, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    }
                }
            }

            if (getenv("UOCR_DECODE_TIMING") && n_generated <= 2) {
                fprintf(stderr, "  [rb_timing] gen=%d T=%d build=%lldms compute=%lldms\n",
                        n_generated, T, _rb_build_ms, _rb_compute_ms);
            }

            if (!no_kv) n_past += T;

            std::vector<float> last_hidden(D);
            rmsnorm_cpu(hidden.data() + (T - 1) * D, last_hidden.data(), D, norm_w.data(), lhp.rms_eps);

            if (lmhead_cpu) {
                linear_cpu(last_hidden.data(), logits.data(), D, V, head_w.data(), nullptr);
            } else {
                size_t meta_sz = 1 * 1024 * 1024;
                std::vector<uint8_t> mb(meta_sz);
                ggml_init_params ip = { meta_sz, mb.data(), true };
                ggml_context* gc = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph(gc);
                ggml_tensor* in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, 1);
                ggml_set_name(in, "lmh_in"); ggml_set_input(in);
                ggml_tensor* out = ggml_mul_mat(gc, lm_w, in);
                ggml_set_name(out, "lmh_out"); ggml_set_output(out);
                ggml_build_forward_expand(gf, out);
                ggml_backend_sched_reset(ctx.sched);
                ggml_backend_sched_alloc_graph(ctx.sched, gf);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "lmh_in"),
                                        last_hidden.data(), 0, (size_t)D * sizeof(float));
                ggml_backend_sched_graph_compute(ctx.sched, gf);
                ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "lmh_out"),
                                        logits.data(), 0, (size_t)V * sizeof(float));
                ggml_free(gc);
            }

            // Diff: logits
            if (!ctx.diff_ref_path.empty() && n_generated == 0) {
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("logits")) {
                    auto r = ref.compare("logits", logits.data(), V);
                    fprintf(stderr, "  logits: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                            r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        } // end !did_pd

        // Sliding-window no_repeat_ngram logits processor — REQUIRED by this model
        // (the HF model card calls infer() with no_repeat_ngram_size=35,
        // ngram_window=128). Mirrors SlidingWindowNoRepeatNgramProcessor over the
        // full input_ids (prompt placeholders + generated). Without it the
        // detection-box decode gets stuck repeating a partial box.
        int nrng = 35, nwin = 128;
        if (const char *e = getenv("UOCR_NO_REPEAT_NGRAM")) nrng = atoi(e);
        if (const char *e = getenv("UOCR_NGRAM_WINDOW")) nwin = atoi(e);
        if (nrng > 1) {
            // full sequence so far = prompt_ids + generated out_ids
            int n_pre = (int)prompt_ids.size();
            int seq_len = n_pre + (int)out_ids.size();
            auto at = [&](int i) -> int32_t {
                return i < n_pre ? prompt_ids[i] : out_ids[i - n_pre];
            };
            if (seq_len >= nrng) {
                int k = nrng - 1;
                int search_start = std::max(0, seq_len - nwin);
                int search_end = seq_len - nrng + 1;  // exclusive
                for (int idx = search_start; idx < search_end; idx++) {
                    bool match = true;
                    for (int j = 0; j < k; j++)
                        if (at(idx + j) != at(seq_len - k + j)) { match = false; break; }
                    if (match) {
                        int banned = at(idx + k);
                        if (banned >= 0 && banned < V) logits[banned] = -INFINITY;
                    }
                }
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

        if (getenv("UOCR_DBG")) {
            const char *pc = (next >= 0 && next < ctx.tok_vocab_size) ? ctx.id_to_piece[next].c_str() : "?";
            fprintf(stderr, "  [gen %d] id=%d piece=%s\n", n_generated - 1, next, pc);
        }

        _decode_gen_steps++;
        if (next == lhp.eos_token_id) break;

        cur_tokens = {(int32_t)next};
        if (no_kv) {
            size_t off = full_emb.size();
            full_emb.resize(off + D);
            get_embedding(next, full_emb.data() + off);
        }
    }

    if (getenv("UOCR_DBG")) {
        auto _decode_t1 = std::chrono::steady_clock::now();
        long long _decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   _decode_t1 - _decode_t0).count();
        fprintf(stderr, "  [decode] total=%lldms steps=%d prefill+gen use_pd=%d\n",
                _decode_ms, _decode_gen_steps, (int)(n_past > 0 && !getenv("UOCR_DECODE_REBUILD")));
    }
    if (pd.gctx) ggml_free(pd.gctx);
    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer decode
// ---------------------------------------------------------------------------

static std::string decode_tokens(const uocr_ctx &ctx, const int32_t *ids, int n) {
    static std::unordered_map<uint32_t, uint8_t> byte_decoder = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        const auto &enc = core_bpe::byte_encoder();
        for (int b = 0; b < 256; b++) m[(uint32_t)enc[b]] = (uint8_t)b;
        return m;
    }();

    std::string merged;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == ctx.m.lhp.eos_token_id) continue;
        if (id < 0 || id >= ctx.tok_vocab_size) continue;
        const auto &piece = ctx.id_to_piece[id];
        if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>') continue;
        merged += piece;
    }

    std::string result;
    size_t i = 0;
    while (i < merged.size()) {
        unsigned char c = (unsigned char)merged[i];
        size_t len = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0) ? 2 : ((c & 0xF0) == 0xE0) ? 3 : 4;
        if (i + len > merged.size()) len = 1;
        uint32_t cp = 0;
        if (len == 1) cp = c;
        else if (len == 2) cp = ((c & 0x1F) << 6) | (merged[i+1] & 0x3F);
        else if (len == 3) cp = ((c & 0x0F) << 12) | ((merged[i+1] & 0x3F) << 6) | (merged[i+2] & 0x3F);
        else cp = ((c & 0x07) << 18) | ((merged[i+1] & 0x3F) << 12) | ((merged[i+2] & 0x3F) << 6) | (merged[i+3] & 0x3F);
        auto it = byte_decoder.find(cp);
        if (it != byte_decoder.end()) result.push_back((char)it->second);
        else result.append(merged, i, len);
        i += len;
    }
    return result;
}

// ---------------------------------------------------------------------------
// C ABI wrappers
// ---------------------------------------------------------------------------

struct unlimited_ocr_context {
    uocr_ctx inner;
    std::string result;
    std::vector<float> char_confidences;
};

unlimited_ocr_context * unlimited_ocr_init(const char * model_path, int n_threads) {
    auto *c = new unlimited_ocr_context;
    auto &ctx = c->inner;
    ctx.n_threads = n_threads;

    if (const char *ref = getenv("UOCR_REF")) ctx.diff_ref_path = ref;

    if (!load_hparams(ctx, model_path)) {
        fprintf(stderr, "unlimited_ocr: failed to load hparams\n");
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

    auto _it = std::chrono::steady_clock::now();
    auto init_ms = [&](const char *w) {
        if (!getenv("UOCR_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] init.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _it).count());
        _it = now;
    };
    if (!load_tensors(ctx, model_path)) {
        fprintf(stderr, "unlimited_ocr: failed to load tensors\n");
        delete c; return nullptr;
    }
    init_ms("load_tensors");

    precompute_rpe_tables(ctx);

    if (!getenv("UOCR_MOE_CPU")) {
        ctx.moe_metal = stack_moe_experts(ctx);
        if (!ctx.moe_metal)
            fprintf(stderr, "unlimited_ocr: MoE expert stacking failed — using CPU MoE\n");
    }
    init_ms("stack_moe_experts");

    ctx.bench = (std::getenv("CRISPEMBED_UNLIMITED_OCR_BENCH") != nullptr);

    if (ctx.verbosity >= 1) {
        auto &s = ctx.m.shp; auto &ch = ctx.m.chp; auto &l = ctx.m.lhp;
        fprintf(stderr, "unlimited_ocr: loaded %s\n", model_path);
        fprintf(stderr, "  sam: %dL %dd %dH patch=%d img=%d ws=%d\n",
                s.depth, s.hidden, s.heads, s.patch_size, s.image_size, s.window_size);
        fprintf(stderr, "  clip: %dL %dd %dH ffn=%d\n",
                ch.depth, ch.hidden, ch.heads, ch.ffn_hidden);
        fprintf(stderr, "  llm: %dL %dd %dH/%dKV vocab=%d n_exp=%d top_%d\n",
                l.n_layers, l.hidden, l.heads, l.kv_heads, l.vocab_size,
                l.n_experts, l.n_experts_top);
        fprintf(stderr, "  tokenizer: %d tokens\n", ctx.tok_vocab_size);
    }

    return c;
}

void unlimited_ocr_free(unlimited_ocr_context * ctx) {
    if (!ctx) return;
    auto &c = ctx->inner;
    if (c.sched) ggml_backend_sched_free(c.sched);
    if (c.moe_buf) ggml_backend_buffer_free(c.moe_buf);
    if (c.moe_ctx) ggml_free(c.moe_ctx);
    c.model_buf = nullptr;
    c.model_ctx = nullptr;
    core_gguf::free_weights(c.model_wl);
    if (c.backend) ggml_backend_free(c.backend);
    if (c.backend_cpu) ggml_backend_free(c.backend_cpu);
    delete ctx;
}

const char * unlimited_ocr_recognize_raw(unlimited_ocr_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!ctx || !px) { if (out_len) *out_len = 0; return ""; }

    if (getenv("UOCR_DBG"))
        fprintf(stderr, "unlimited_ocr: recognize_raw input: %dx%d ch=%d\n", w, h, ch);

    // Isolation test: UOCR_TEXT_TEST runs the LLM decoder as a pure language model
    if (const char *tt = getenv("UOCR_TEXT_TEST")) {
        auto &mdl = ctx->inner.m;
        int D = mdl.lhp.hidden;
        auto embed_w = to_f32(mdl.embed_tokens);
        std::vector<int32_t> ids = {0};  // bos
        auto more = core_bpe::tokenize_simple(ctx->inner.token_to_id, ctx->inner.merge_rank, tt);
        ids.insert(ids.end(), more.begin(), more.end());
        std::vector<float> pe((size_t)ids.size() * D);
        for (size_t i = 0; i < ids.size(); i++)
            memcpy(pe.data() + i * D, embed_w.data() + (size_t)ids[i] * D, D * sizeof(float));
        fprintf(stderr, "  [TEXT_TEST] prompt=\"%s\" ids:", tt);
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
        std::vector<int32_t> g; std::vector<float> gc;
        run_llm_decoder(ctx->inner, pe.data(), (int)ids.size(), 40, g, gc);
        ctx->result = decode_tokens(ctx->inner, g.data(), (int)g.size());
        if (out_len) *out_len = (int)ctx->result.size();
        return ctx->result.c_str();
    }

    auto &s = ctx->inner.m.shp;
    int imgS = s.image_size;
    // UOCR_OPT_SAM_RES=N: process SAM at reduced resolution for speedup
    // Must be a multiple of patch_size (16). Examples: 512 (~5x), 768 (~2.5x)
    if (const char* sr = getenv("UOCR_OPT_SAM_RES")) {
        int res = atoi(sr);
        if (res > 0 && res % s.patch_size == 0 && res < s.image_size) {
            imgS = res;
            if (getenv("UOCR_DBG"))
                fprintf(stderr, "  [opt] SAM reduced res: %d → %d\n", s.image_size, imgS);
        }
    }

    // Preprocess: ImageOps.pad → resize preserving aspect ratio, center, pad with gray
    float scale = std::min((float)imgS / w, (float)imgS / h);
    int rw = std::max(1, (int)lroundf(w * scale));
    int rh = std::max(1, (int)lroundf(h * scale));
    int ox = (imgS - rw) / 2;
    int oy = (imgS - rh) / 2;

    std::vector<float> pixels(3 * imgS * imgS);
    for (int c = 0; c < 3; c++) {
        int ci = std::min(c, ch - 1);
        for (int y = 0; y < imgS; y++) {
            for (int x = 0; x < imgS; x++) {
                float val;
                if (x < ox || x >= ox + rw || y < oy || y >= oy + rh) {
                    // Match Python ImageOps.pad: color=tuple(int(x*255) for x in mean)
                    // int(0.5*255)=127, 127/255.0=0.498039 (NOT exactly 0.5)
                    val = (float)((int)(s.image_mean[c] * 255.0f)) / 255.0f;
                } else {
                    // Bicubic interpolation (matches PIL Image.BICUBIC / ImageOps.pad)
                    float sx = (x - ox + 0.5f) / scale - 0.5f;
                    float sy = (y - oy + 0.5f) / scale - 0.5f;
                    int ix = (int)floorf(sx), iy = (int)floorf(sy);
                    float fx = sx - ix, fy = sy - iy;

                    auto P = [&](int xx, int yy) -> float {
                        xx = std::min(std::max(xx, 0), w - 1);
                        yy = std::min(std::max(yy, 0), h - 1);
                        return (float)px[(yy * w + xx) * ch + ci] / 255.0f;
                    };
                    // PIL Keys bicubic kernel (a=-1): matches PIL Image.BICUBIC
                    // k(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1         for |t|<=1
                    //         a|t|^3 - 5a|t|^2 + 8a|t| - 4a      for 1<|t|<2
                    auto cubic = [](float t) -> float {
                        constexpr float a = -0.5f;  // PIL BICUBIC = Catmull-Rom
                        t = fabsf(t);
                        if (t <= 1.0f) return ((a + 2.0f)*t - (a + 3.0f))*t*t + 1.0f;
                        if (t < 2.0f) return ((a*t - 5.0f*a)*t + 8.0f*a)*t - 4.0f*a;
                        return 0.0f;
                    };
                    float sum = 0, wsum = 0;
                    for (int ky = -1; ky <= 2; ky++) {
                        float wy = cubic(fy - ky);
                        for (int kx = -1; kx <= 2; kx++) {
                            float ww = wy * cubic(fx - kx);
                            sum += ww * P(ix + kx, iy + ky);
                            wsum += ww;
                        }
                    }
                    val = (wsum > 0) ? sum / wsum : 0.0f;
                    val = std::min(std::max(val, 0.0f), 1.0f);  // clamp like PIL
                }
                pixels[c * imgS * imgS + y * imgS + x] = (val - s.image_mean[c]) / s.image_std[c];
            }
        }
    }

    const bool bench = ctx->inner.bench;
    auto t_total = std::chrono::steady_clock::now();
    bool dbg_t = getenv("UOCR_DBG") != nullptr;
    auto _ts = std::chrono::steady_clock::now();
    auto stage_ms = [&](const char *name) {
        if (!dbg_t) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] %s %lldms\n", name,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _ts).count());
        _ts = now;
    };

    auto &lhp = ctx->inner.m.lhp;
    int D = lhp.hidden;
    std::vector<float> vis_features;
    int n_vis_total = 0;

    // UOCR_INJECT_VIS: skip the SAM/CLIP towers entirely and feed the decoder
    // the reference's assembled vision_features directly. Isolates the decoder
    // (perfect, HF-equal vision input) AND avoids the vision Metal buffers so
    // f16/q8_0 fit in memory. Used to prove decode bugs are not "quantization".
    if (getenv("UOCR_INJECT_VIS") && !ctx->inner.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx->inner.diff_ref_path.c_str())) {
            auto [vd, vn] = ref.get_f32("vision_features");
            if (vd && vn % D == 0) {
                vis_features.assign(vd, vd + vn);
                n_vis_total = (int)(vn / D);
                fprintf(stderr, "  [INJECT_VIS] using reference vision_features (%d tokens)\n",
                        n_vis_total);
            }
        }
        if (n_vis_total == 0) { fprintf(stderr, "unlimited_ocr: INJECT_VIS failed\n");
                                if (out_len) *out_len = 0; return ""; }
    } else {

    // 1. SAM vision encoder
    auto t_sam = std::chrono::steady_clock::now();
    std::vector<float> sam_features;
    int n_sam_tokens, sam_dim;
    if (!encode_sam(ctx->inner, pixels.data(), imgS, sam_features, n_sam_tokens, sam_dim)) {
        fprintf(stderr, "unlimited_ocr: SAM encoding failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("sam");
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_sam).count();
        fprintf(stderr, "[unlimited_ocr-bench] sam_encoder: %lldms\n", (long long)ms);
    }

    // UOCR_INJECT_REF: replace C++ SAM output with Python reference SAM output
    // to isolate CLIP encoder bugs from SAM quantization noise.
    if (getenv("UOCR_INJECT_REF") && !ctx->inner.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx->inner.diff_ref_path.c_str())) {
            auto [ref_data, ref_n] = ref.get_f32("sam_output");
            if (ref_data && ref_n == (size_t)n_sam_tokens * sam_dim) {
                memcpy(sam_features.data(), ref_data, ref_n * sizeof(float));
                fprintf(stderr, "  [INJECT] replaced SAM output with reference (%zu floats)\n", ref_n);
            }
        }
    }

    // 2. CLIP-L/14 encoder — receives SAM features as patch embeddings
    auto t_clip = std::chrono::steady_clock::now();
    std::vector<float> clip_out;
    if (!encode_clip(ctx->inner, sam_features.data(), n_sam_tokens, sam_dim, clip_out)) {
        fprintf(stderr, "unlimited_ocr: CLIP encoding failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("clip");
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_clip).count();
        fprintf(stderr, "[unlimited_ocr-bench] clip_encoder: %lldms\n", (long long)ms);
    }

    // 3. Fusion + Projection: concat(CLIP, SAM) → Linear(2048, 1280)
    auto t_proj = std::chrono::steady_clock::now();
    std::vector<float> proj_out;
    int clip_dim = ctx->inner.m.chp.hidden;  // 1024
    if (!fuse_and_project(ctx->inner, clip_out.data(), sam_features.data(),
                          n_sam_tokens, clip_dim, sam_dim, proj_out)) {
        fprintf(stderr, "unlimited_ocr: fusion/projection failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("fuse_project");

    // 4. Vision features assembly (with image_newline)
    if (!assemble_vision_features(ctx->inner, proj_out.data(), n_sam_tokens, D,
                                   vis_features, n_vis_total)) {
        fprintf(stderr, "unlimited_ocr: vision features assembly failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("vision_assembly");
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_proj).count();
        fprintf(stderr, "[unlimited_ocr-bench] projection+assembly: %lldms\n", (long long)ms);
    }

    fprintf(stderr, "unlimited_ocr: stages done — sam=%d/%d clip=%d/%d proj=%d vis=%d\n",
            n_sam_tokens, sam_dim, (int)clip_out.size() / clip_dim, clip_dim,
            n_sam_tokens, n_vis_total);
    } // end else (real vision path)

    // 5. Assemble the LLM prompt embeddings
    //    [bos] + [n_vis_total vision features] + tokenize("\nFree OCR.")
    auto &mdl = ctx->inner.m;

    // Per-row embed dequant
    ggml_tensor* emb_t = mdl.embed_tokens;
    const auto* emb_tt = ggml_get_type_traits(emb_t->type);
    const size_t emb_row_bytes = ggml_row_size(emb_t->type, D);
    std::vector<uint8_t> emb_row_buf(emb_row_bytes);

    // Instruction: the model's prompt is "<image>document parsing." (per the HF
    // model card — NOT "Free OCR.", which is a different DeepSeek-OCR checkpoint's
    // prompt and makes this model emit its training-instruction boilerplate). No
    // leading newline: "document parsing." directly follows the <image> block.
    // "document parsing." → [document=34030, Ġparsing=76466, .=16] (verified
    // against the model's tokenizer.json). Override with UOCR_INSTR.
    std::vector<int32_t> instr_ids = {34030, 76466, 16};
    if (const char *ov = getenv("UOCR_INSTR")) {
        instr_ids.clear();
        const char *p = ov;
        while (*p) {
            char *end; long v = strtol(p, &end, 10);
            if (end == p) { p++; continue; }
            instr_ids.push_back((int32_t)v);
            p = (*end == ',') ? end + 1 : end;
        }
    }

    int n_prompt = 1 /*bos*/ + n_vis_total + (int)instr_ids.size();
    std::vector<float> prompt_embeds((size_t)n_prompt * D);

    int row = 0;
    auto put_tok = [&](int32_t id) {
        float *dst = prompt_embeds.data() + (size_t)row * D;
        const size_t off = (size_t)id * emb_row_bytes;
        if (emb_t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(emb_t, dst, off, (size_t)D * sizeof(float));
        } else {
            ggml_backend_tensor_get(emb_t, emb_row_buf.data(), off, emb_row_bytes);
            emb_tt->to_float(emb_row_buf.data(), dst, D);
        }
        row++;
    };
    // Token-id view of the prompt, for the no_repeat_ngram processor (HF runs it
    // over the full input_ids incl. the image placeholders). Vision rows map to
    // the <image> placeholder id (128815).
    std::vector<int32_t> prompt_ids;
    prompt_ids.reserve(n_prompt);
    prompt_ids.push_back(0);  // bos
    put_tok(0);  // bos = <|begin_of_sentence|>

    // Vision features (already includes image_newline per row + view_separator)
    memcpy(prompt_embeds.data() + (size_t)row * D, vis_features.data(),
           (size_t)n_vis_total * D * sizeof(float));
    row += n_vis_total;
    for (int i = 0; i < n_vis_total; i++) prompt_ids.push_back(128815);  // <image>

    for (int32_t id : instr_ids) { put_tok(id); prompt_ids.push_back(id); }

    if (getenv("UOCR_DBG")) {
        fprintf(stderr, "  [dbg] prompt: bos + %d vis + %zu instr = %d tokens; instr_ids:",
                n_vis_total, instr_ids.size(), n_prompt);
        for (int32_t id : instr_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    // 6. LLM decoder
    auto t_llm = std::chrono::steady_clock::now();
    std::vector<int32_t> gen_ids;
    std::vector<float> gen_confs;
    int max_new = 1024;
    if (const char* mn = getenv("UOCR_MAX_NEW")) max_new = atoi(mn);
    if (!run_llm_decoder(ctx->inner, prompt_embeds.data(), n_prompt, max_new,
                         gen_ids, gen_confs, prompt_ids)) {
        fprintf(stderr, "unlimited_ocr: LLM decode failed\n");
        if (out_len) *out_len = 0; return "";
    }
    if (bench) {
        auto llm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_llm).count();
        fprintf(stderr, "[unlimited_ocr-bench] llm_decoder: %lldms\n", (long long)llm_ms);
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[unlimited_ocr-bench] total: %lldms\n", (long long)total_ms);
    }

    if (getenv("UOCR_DBG")) {
        fprintf(stderr, "  [dbg] gen_ids (%zu):", gen_ids.size());
        for (int id : gen_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    ctx->result = decode_tokens(ctx->inner, gen_ids.data(), (int)gen_ids.size());
    ctx->char_confidences = std::move(gen_confs);
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * unlimited_ocr_recognize(unlimited_ocr_context * ctx,
    const float * px, int w, int h, int * out_len) {
    if (!ctx || !px) { if (out_len) *out_len = 0; return ""; }
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, px[i] * 255.0f));
        rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
    }
    return unlimited_ocr_recognize_raw(ctx, rgb.data(), w, h, 3, out_len);
}

const float * unlimited_ocr_confidences(const unlimited_ocr_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) { if (n_tokens) *n_tokens = 0; return nullptr; }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float unlimited_ocr_mean_confidence(const unlimited_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
