// deepseek_ocr2.cpp — DeepSeek-OCR-2 engine: SAM ViT-B + Qwen2 encoder + MoE decoder.
//
// Vision: per-layer ggml graph with CPU window partition (SAM ViT-B pattern).
// Qwen2 encoder: CPU-scalar bidirectional transformer (no causal mask).
// LLM decoder: ggml graph with KV cache, MoE layers use CPU-scalar expert dispatch.

#include "deepseek_ocr2.h"
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
    // DeepSeek-OCR2 BasicImageTransform: simple [-1,1] normalization (mean=std=0.5),
    // NOT CLIP normalization. (processor_config.json image_mean/std = 0.5.)
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
};

struct qwen2_enc_hparams {
    int depth = 24, hidden = 896, heads = 14, kv_heads = 2;
    int intermediate = 4864;
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
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
    // Experts stacked as [in, out, n_exp] for ggml_mul_mat_id (Metal MoE path).
    // Built at load by stack_moe_experts() when the graph MoE path is active.
    ggml_tensor *gate_exps{}, *up_exps{}, *down_exps{};
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
    ggml_tensor *query_768{}, *query_1024{}, *qe_output_norm{};

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
    ggml_context *model_ctx{};            // alias into model_wl (do not free separately)
    ggml_backend_buffer_t model_buf{};    // alias into model_wl
    core_gguf::WeightLoad model_wl;       // owns ctx/buf (+ the mmap on the no-copy path)
    ggml_backend_t backend{}, backend_cpu{};
    ggml_backend_sched_t sched{};
    std::vector<uint8_t> compute_meta;

    // Stacked MoE expert weights ([in,out,n_exp]) for the ggml_mul_mat_id path.
    ggml_context *moe_ctx{};
    ggml_backend_buffer_t moe_buf{};
    bool moe_metal = false;  // true once experts are stacked + the graph path is on

    // Tokenizer
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
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
    bool bench = false;
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
                       int kh, int kw, int stride, int pad, int n_threads = 1) {
    int oH = (H + 2 * pad - kh) / stride + 1;
    int oW = (W + 2 * pad - kw) / stride + 1;
    // Each output channel writes its own plane → parallelize over oc. This is
    // the SAM neck/downsample hot path (~10 s scalar); threading it is exact.
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

static bool load_tensors(ds_ocr2_ctx &ctx, const char *path) {
    // DS_MMAP=1 opts into the no-copy mmap load (Metal/CPU unified memory):
    // point the weight buffer at the mmap'd file instead of copying ~2 GB into
    // a fresh buffer — halves resident memory. On Metal the pages still get
    // wired on first use, so it's a memory win more than a first-load-time win;
    // default stays the proven copy path. Falls back automatically if
    // unsupported. Validated equal by tests/test_gguf_loader_mmap.
    bool try_mmap = getenv("DS_MMAP") != nullptr;
    if (!core_gguf::load_weights(path, ctx.backend, "deepseek_ocr2", ctx.model_wl, try_mmap))
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
    m.qe_output_norm = F("qe.output_norm.weight");

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

// Metal graph for the SAM neck + downsample (conv 768->256 1x1, LN2d, conv
// 256->256 3x3, LN2d, conv 256->512 3x3 s2, conv 512->896 3x3 s2). Replaces the
// CPU conv2d_cpu chain. Conv kernels are fed as F32 inputs (the GGUF stores them
// Q8_0 — can't reshape a quantized tensor to [1,1,IC,OC]). Default path;
// DS_SAM_CONV_CPU=1 restores the CPU chain. Validated equal via DS_REF sam_output.
// Metal graph for the SAM patch embedding (conv 3->C, PS×PS, stride PS) + the
// absolute position embedding. Replaces the scalar/threaded per-patch matmul.
// Produces hidden as [C, N] (== hidden[tok*C+c]). Kernel/pos fed as F32.
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

    ggml_tensor* x = ggml_conv_2d(g, w, px, PS, PS, 0, 0, 1, 1);   // [nP,nP,C]
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));             // [C,nP,nP]
    x = ggml_reshape_2d(g, x, C, nP * nP);                        // [C, N]
    x = ggml_add(g, x, bias);                                     // + per-channel bias
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

    // LayerNorm over the channel axis (ne[2]) at each spatial position.
    auto ln2d = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* xp = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));  // [C,W,H]
        xp = ggml_norm(g, xp, 1e-6f);
        xp = ggml_add(g, ggml_mul(g, xp, w), b);
        return ggml_cont(g, ggml_permute(g, xp, 2, 0, 1, 3));            // [W,H,C]
    };

    ggml_tensor* x = ggml_conv_2d(g, w_nc1, chw, 1, 1, 0, 0, 1, 1);  // [nP,nP,nC]
    x = ln2d(x, ln1w, ln1b);
    x = ggml_conv_2d(g, w_nc2, x, 1, 1, 1, 1, 1, 1);                 // [nP,nP,nC]
    x = ln2d(x, ln2w, ln2b);
    x = ggml_conv_2d(g, w_n2, x, 2, 2, 1, 1, 1, 1);                  // [ds1,ds1,ds1_ch]
    x = ggml_conv_2d(g, w_n3, x, 2, 2, 1, 1, 1, 1);                  // [ds2,ds2,ds2_ch]
    int ds2 = (nP + 2 - 3) / 2 + 1; ds2 = (ds2 + 2 - 3) / 2 + 1;
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));               // [C, W, H]
    x = ggml_reshape_2d(g, x, ds2_ch, ds2 * ds2);                   // [C, n_vis] = out_features
    ggml_set_name(x, "neck_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static bool encode_sam(ds_ocr2_ctx &ctx, const float *pixels,
                       std::vector<float> &out_features, int &out_n_tokens, int &out_dim) {
    auto &s = ctx.m.shp;
    int C = s.hidden, PS = s.patch_size, nP = s.image_size / PS;
    int N = nP * nP, hd = s.head_dim, ws = s.window_size;
    auto _sam_t = std::chrono::steady_clock::now();
    auto sam_mark = [&](const char *w) {
        if (!getenv("DS_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] sam.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _sam_t).count());
        _sam_t = now;
    };

    // Patch embedding
    auto pe_w = to_f32(ctx.m.patch_embed_w);
    auto pe_b = to_f32(ctx.m.patch_embed_b);
    auto pos = to_f32(ctx.m.pos_embed);
    int patch_dim = 3 * PS * PS;
    std::vector<float> hidden(N * C);

    if (!getenv("DS_SAM_CONV_CPU")) {
        // Patch embed on Metal (conv 3->C, PS×PS stride PS) + position embed.
        size_t meta_sz = 8 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context* gc = ggml_init(ip);
        ggml_cgraph* gf = build_sam_patch_graph(gc, s.image_size, PS, C, nP);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "px"), pixels, 0,
                                (size_t)3 * s.image_size * s.image_size * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_patch"), pe_w.data(), 0, pe_w.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pe_b"), pe_b.data(), 0, pe_b.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos.data(), 0, pos.size() * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, gf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "patch_out"), hidden.data(), 0,
                                (size_t)N * C * sizeof(float));
        ggml_free(gc);
    } else {
        // Threaded CPU per-patch matmul (exact fallback).
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
    // Pre-dequant LN weights for windowed layers
    std::vector<std::vector<float>> ln1_ws(s.depth), ln1_bs(s.depth);
    for (int li = 0; li < s.depth; li++)
        if (!ctx.m.sam_blocks[li].is_global) {
            ln1_ws[li] = to_f32(ctx.m.sam_blocks[li].ln1_w);
            ln1_bs[li] = to_f32(ctx.m.sam_blocks[li].ln1_b);
        }

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

        if (getenv("DS_DBG")) fprintf(stderr, "  [dbg] sam li=%d is_global=%d aH=%d nW=%d T=%d rp_h.sz=%zu\n",
                li, is_global, aH, nW, T, ctx.rp_h_per_layer[li].size());
        std::vector<float> rp_h_ggml(aH * aH * hd), rp_w_ggml(aW * aW * hd);
        reformat_rp_table(ctx.rp_h_per_layer[li].data(), rp_h_ggml.data(), aH, hd);
        reformat_rp_table(ctx.rp_w_per_layer[li].data(), rp_w_ggml.data(), aW, hd);
        if (getenv("DS_DBG")) fprintf(stderr, "  [dbg] sam li=%d reformat ok, building graph\n", li);

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
        if (getenv("DS_DBG"))
            fprintf(stderr, "  [time] sam_li=%d (%s T=%d) %lldms\n", li,
                    is_global ? "global" : "window", T,
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - _slt).count());
    }

    // Diff: pre-neck ViT output (4096x768). The reference dump does not capture
    // this intermediate; named distinctly so it never collides with the final
    // SAM output below.
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_vit_output")) {
            auto r = ref.compare("sam_vit_output", hidden.data(), N * C);
            fprintf(stderr, "  sam_vit_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
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

    // net_2 = 256->512, net_3 = 512->896 (= Qwen2 dim); derive channels from the
    // weights (ne[1]), not the config's nominal [512,1024] (which overruns).
    int ds1_ch = (int)ctx.m.net_2_w->ne[1];
    int ds2_ch = (int)ctx.m.net_3_w->ne[1];
    int ds1_H = (nP + 2 - 3) / 2 + 1;
    int ds2_H = (ds1_H + 2 - 3) / 2 + 1, ds2_W = ds2_H;
    int n_vis = ds2_H * ds2_W, vis_D = ds2_ch;
    out_features.resize((size_t)n_vis * vis_D);

    if (!getenv("DS_SAM_CONV_CPU")) {
        // Neck + downsample on Metal (ggml_conv_2d), ~20-40x vs the CPU convs and
        // no thread-scheduling variance. Conv kernels fed as F32 (GGUF stores them
        // Q8_0; can't reshape a quantized tensor to [1,1,IC,OC]). DS_SAM_CONV_CPU=1
        // restores the threaded CPU chain. Validated equal via DS_REF sam_output.
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
    out_n_tokens = n_vis;
    out_dim = vis_D;

    // Diff: final SAM output (post neck + downsample), [256, 896] — this is the
    // tensor the reference dump's "sam_output" corresponds to (sam_model output).
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_output")) {
            auto r = ref.compare("sam_output", out_features.data(),
                                 (size_t)out_n_tokens * out_dim);
            fprintf(stderr, "  sam_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Qwen2 bidirectional encoder (CPU-scalar)
// ---------------------------------------------------------------------------

// NEOX (rotate_half) RoPE applied in-place to one head_dim vector at `pos`.
static void apply_rope_neox(float *v, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int j = 0; j < half; j++) {
        float freq = 1.0f / powf(theta, (float)(2 * j) / hd);
        float a = pos * freq, c = cosf(a), s = sinf(a);
        float x1 = v[j], x2 = v[j + half];
        v[j]        = x1 * c - x2 * s;
        v[j + half] = x2 * c + x1 * s;
    }
}

// Graph-based qwen2 encoder layer (runs on ctx.sched / Metal). One layer over
// all T = n_vis + n_query tokens at once — bidirectional, no KV cache. Mirrors
// build_llm_layer_attn (NEOX RoPE, GQA interleave, soft_max_ext+mask) but adds
// q/k/v biases and an always-on SwiGLU FFN. The graph is built + executed +
// freed within one scope by the caller (SAM pattern), so the meta buffer never
// outlives the context. Gated by DS_QWEN2_SCALAR=1 (falls back to the CPU path).
static ggml_cgraph* build_qwen2_enc_layer_graph(ggml_context* g, ds_ocr2_ctx* ctx, int li, int T) {
    auto &qhp = ctx->m.qhp;
    auto &ly  = ctx->m.qwen2_layers[li];
    int D = qhp.hidden, nh = qhp.heads, nkv = qhp.kv_heads;
    int hd = D / nh, kv_repeat = nh / nkv;
    float eps = qhp.rms_eps;

    ggml_cgraph* gf = ggml_new_graph_custom(g, 2048, false);
    auto rmsnorm = [&](ggml_tensor* t, ggml_tensor* w) {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    ggml_tensor* x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "layer_input"); ggml_set_input(x);
    ggml_tensor* pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);
    ggml_tensor* mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);  // [keys, queries]
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    // Pre-norm + Q/K/V (+ bias)
    ggml_tensor* h = rmsnorm(x, ly.in_ln_w);
    ggml_tensor* Q = ggml_mul_mat(g, ly.q_w, h);
    ggml_tensor* K = ggml_mul_mat(g, ly.k_w, h);
    ggml_tensor* V = ggml_mul_mat(g, ly.v_w, h);
    if (ly.q_b) Q = ggml_add(g, Q, ensure_f32(g, ly.q_b));
    if (ly.k_b) K = ggml_add(g, K, ensure_f32(g, ly.k_b));
    if (ly.v_b) V = ggml_add(g, V, ensure_f32(g, ly.v_b));

    Q = ggml_reshape_3d(g, Q, hd, nh, T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);
    Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      qhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                      qhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor* Kfull = ggml_cont(g, K);
    ggml_tensor* Vfull = ggml_cont(g, V);
    if (kv_repeat > 1) {  // GQA interleave (not tile)
        Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, T);
        Kfull = ggml_repeat(g, Kfull, ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, T));
        Kfull = ggml_reshape_3d(g, Kfull, hd, nh, T);
        Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, T);
        Vfull = ggml_repeat(g, Vfull, ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, T));
        Vfull = ggml_reshape_3d(g, Vfull, hd, nh, T);
    }

    Q     = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));      // [hd, T, nh]
    Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
    Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

    ggml_tensor* scores = ggml_mul_mat(g, Kfull, Q);          // [T(keys), T(queries), nh]
    scores = ggml_soft_max_ext(g, scores, mask, 1.0f / sqrtf((float)hd), 0.0f);
    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, D, T);
    attn = ggml_mul_mat(g, ly.o_w, attn);
    x = ggml_add(g, x, attn);

    // SwiGLU FFN + residual
    ggml_tensor* res = x;
    h = rmsnorm(x, ly.post_ln_w);
    ggml_tensor* gate = ggml_silu(g, ggml_mul_mat(g, ly.gate_w, h));
    ggml_tensor* up   = ggml_mul_mat(g, ly.up_w, h);
    x = ggml_add(g, res, ggml_mul_mat(g, ly.down_w, ggml_mul(g, gate, up)));

    ggml_set_name(x, "layer_output"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

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

    // Blueprint (CustomQwen2): x_combined = cat([visual, queries]) — VISUAL
    // FIRST, then the learned query tokens. token_type 0=visual (non-causal),
    // 1=query (causal). RoPE applied at positions 0..T-1. Returns the query
    // half (y[:, n_vis:]).
    int T = n_vis + n_query;
    std::vector<float> hidden(T * D, 0.0f);
    if (vis_dim == D)
        memcpy(hidden.data(), vis_features, (size_t)n_vis * D * sizeof(float));
    else
        for (int t = 0; t < n_vis; t++)
            for (int d = 0; d < D; d++)
                hidden[t * D + d] = (d < vis_dim) ? vis_features[t * vis_dim + d] : 0.0f;
    if (n_query > 0)
        memcpy(hidden.data() + (size_t)n_vis * D, query_data.data(),
               (size_t)n_query * D * sizeof(float));

    // Run the 24 bidirectional transformer layers. Default: ggml graph on
    // ctx.sched (Metal). DS_QWEN2_SCALAR=1 forces the CPU-scalar reference path.
    if (!getenv("DS_QWEN2_SCALAR")) {
        std::vector<int32_t> pos(T);
        for (int t = 0; t < T; t++) pos[t] = t;
        // Bidirectional-vis + causal-query mask, shared across layers. Layout
        // matches soft_max_ext: [keys, queries], mask[qi*T + ki].
        std::vector<ggml_fp16_t> emask((size_t)T * T);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int qi = 0; qi < T; qi++) {
            bool qv = qi < n_vis;
            for (int ki = 0; ki < T; ki++) {
                bool ok = qv ? (ki < n_vis) : (ki < n_vis || ki <= qi);
                emask[(size_t)qi * T + ki] = ok ? z : ninf;
            }
        }
        for (int li = 0; li < qhp.depth; li++) {
            size_t meta_sz = 16 * 1024 * 1024;
            std::vector<uint8_t> mb(meta_sz);
            ggml_init_params ip = { meta_sz, mb.data(), true };
            ggml_context* gc = ggml_init(ip);
            ggml_cgraph* gf = build_qwen2_enc_layer_graph(gc, &ctx, li, T);
            ggml_backend_sched_reset(ctx.sched);
            ggml_backend_sched_alloc_graph(ctx.sched, gf);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "layer_input"),
                                    hidden.data(), 0, (size_t)T * D * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                                    pos.data(), 0, (size_t)T * sizeof(int32_t));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"),
                                    emask.data(), 0, (size_t)T * T * sizeof(ggml_fp16_t));
            ggml_backend_sched_graph_compute(ctx.sched, gf);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_output"),
                                    hidden.data(), 0, (size_t)T * D * sizeof(float));
            ggml_free(gc);

            if (!ctx.diff_ref_path.empty()) {
                char nm[32]; snprintf(nm, sizeof(nm), "qwen2_layer_%d", li);
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                    auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                    fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                            nm, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }
    } else
    // Run bidirectional transformer layers (CPU-scalar reference path)
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

        // NEOX RoPE at positions 0..T-1 (Qwen2, rope_theta from config).
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < nh; h++)
                apply_rope_neox(Q.data() + t * q_dim + h * hd, hd, t, qhp.rope_theta);
            for (int h = 0; h < nkv; h++)
                apply_rope_neox(K.data() + t * kv_dim + h * hd, hd, t, qhp.rope_theta);
        }

        // Multi-head attention with the token-type mask: visual tokens
        // (< n_vis) attend to visual only (bidirectional); query tokens (>=
        // n_vis) attend to all visual + causally to earlier/equal queries.
        float attn_scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out(T * D, 0.0f);

        for (int h = 0; h < nh; h++) {
            int kv_h = h / kv_repeat;  // GQA mapping

            // Compute scores for all query positions
            for (int qi = 0; qi < T; qi++) {
                bool qi_vis = qi < n_vis;
                std::vector<float> scores(T);
                for (int ki = 0; ki < T; ki++) {
                    bool allowed = qi_vis ? (ki < n_vis)
                                          : (ki < n_vis || ki <= qi);
                    if (!allowed) { scores[ki] = -INFINITY; continue; }
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

        // Diff: post-attention hidden (residual + attn, pre-FFN). Splits each
        // layer into attention-half vs FFN-half to localize the divergence.
        if (!ctx.diff_ref_path.empty()) {
            char nm[40]; snprintf(nm, sizeof(nm), "qwen2_layer_%d_postattn", li);
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                        nm, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }

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

        // Diff: per-layer qwen2 hidden state (full [vis+query] seq, pre-final-norm)
        // for bisecting the encoder divergence.
        if (!ctx.diff_ref_path.empty()) {
            char nm[32]; snprintf(nm, sizeof(nm), "qwen2_layer_%d", li);
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n",
                        nm, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }
    }

    // Final RMSNorm (Qwen2Model.norm) over all tokens.
    if (ctx.m.qe_output_norm) {
        auto fn = to_f32(ctx.m.qe_output_norm);
        std::vector<float> tmp(D);
        for (int t = 0; t < T; t++) {
            rmsnorm_cpu(hidden.data() + t * D, tmp.data(), D, fn.data(), eps);
            memcpy(hidden.data() + t * D, tmp.data(), D * sizeof(float));
        }
    }

    // Output = the query half (blueprint y[:, n_vis:]): positions n_vis..T-1.
    out_n_tokens = (n_query > 0) ? n_query : T;
    out_dim = D;
    out_enc.resize((size_t)out_n_tokens * D);
    memcpy(out_enc.data(), hidden.data() + (size_t)n_vis * D,
           (size_t)out_n_tokens * D * sizeof(float));

    // Diff: Qwen2 encoder output (post final-norm, query half) — this is the
    // tensor the reference dump's "qwen2_enc_output" corresponds to.
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("qwen2_enc_output")) {
            auto r = ref.compare("qwen2_enc_output", out_enc.data(),
                                 (size_t)out_n_tokens * D);
            fprintf(stderr, "  qwen2_enc_output: cos_min=%.6f max_abs=%.6f %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }
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

// Stack the 64 per-expert weights of each MoE layer into [in, out, n_exp]
// tensors so the decode graph can dispatch them with ggml_mul_mat_id on Metal
// (instead of the per-token CPU-scalar moe_ffn_cpu). Keeps the quantized blocks
// as-is — each expert is the same shape/type, so its bytes are a contiguous
// slice. Allocates a dedicated backend buffer (the per-expert tensors stay in
// model_buf for the DS_MOE_CPU fallback). Returns false on any failure → caller
// falls back to the CPU MoE.
static bool stack_moe_experts(ds_ocr2_ctx &ctx) {
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
            if (nb != stacked->nb[2]) return false;  // slice size must match
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

// Persistent T=1 decode graph: built once after prefill, reused for every
// generation step. All tensor shapes are fixed (max_kv = n_prompt+max_new).
// The scheduler's is_alloc flag stays true after the one-time sched_alloc_graph
// call, so subsequent sched_graph_compute calls skip the alloc step entirely.
// Gate with DS_DECODE_REBUILD=1 to revert to the old per-step rebuild path.
struct PdGraph {
    std::vector<uint8_t> meta;
    ggml_context* gctx = nullptr;
    ggml_cgraph*  gf   = nullptr;
    // Direct tensor pointers for fast per-step set/get (no name lookup needed).
    ggml_tensor* t_cur_tok_id = nullptr;   // [1] I32
    ggml_tensor* t_pos_ids    = nullptr;   // [1] I32
    ggml_tensor* t_mask       = nullptr;   // [max_kv, 1] F16
    std::vector<ggml_tensor*> t_k_cache;   // [n_layers], each [kv_dim, max_kv-1] F32
    std::vector<ggml_tensor*> t_v_cache;
    ggml_tensor* t_logits = nullptr;       // [V, 1] F32
    std::vector<ggml_tensor*> t_k_new;    // [n_layers], each [kv_dim, 1] F32
    std::vector<ggml_tensor*> t_v_new;
    int max_kv = 0;
};

struct llm_attn_graph {
    ggml_cgraph *gf{};
    ggml_context *gctx{};
    // The no-alloc ggml context places its tensor/graph metadata in `meta`, so
    // the buffer must outlive the returned graph. Holding it here (moved out with
    // the struct) fixes a use-after-free: a local meta buffer was freed on return,
    // leaving gf/gctx dangling — a latent crash that surfaced once the fast
    // (graph) qwen2 encoder let the decoder prefill actually run.
    std::vector<uint8_t> meta;
};

// Build a single persistent T=1 decode graph covering all 12 LLM layers + LM head.
// All tensor shapes are fixed so the graph can be allocated once and reused.
// KV cache slots: k_cache_in [kv_dim, max_kv-1] holds past tokens 0..n_past-1;
// current-step K/V are computed inside the graph and concatenated onto the cache.
// The mask [max_kv, 1] controls which positions participate (0 = attend, -INF = mask).
static PdGraph build_persistent_decode_graph(ds_ocr2_ctx &ctx, int max_kv) {
    auto &lhp = ctx.m.lhp;
    int D = lhp.hidden, V = lhp.vocab_size;
    int nh = lhp.heads, nkv = lhp.kv_heads, hd = lhp.head_dim;
    int n_layers = lhp.n_layers, kv_dim = nkv * hd;
    float eps = lhp.rms_eps;
    int max_kv_in = max_kv - 1;  // past-token slots (current step is the concat tail)

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

    // Token embedding: look up on the backend (dequants quantized embed_tokens in-graph).
    pd.t_cur_tok_id = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pd.t_cur_tok_id, "pd_tok"); ggml_set_input(pd.t_cur_tok_id);
    ggml_tensor* x = ggml_get_rows(g, ctx.m.embed_tokens, pd.t_cur_tok_id);  // [D, 1]

    pd.t_pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pd.t_pos_ids, "pd_pos"); ggml_set_input(pd.t_pos_ids);

    // Shared causal mask for all layers: [max_kv, 1] F16.
    // Layout: Kfull = concat(k_cache_in[0..max_kv-2], K_new[max_kv-1]).
    // Positions 0..n_past-1 attend (0.0), n_past..max_kv-2 are garbage (-INF),
    // max_kv-1 is the current step's K (0.0). Caller updates incrementally.
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

        pd.t_k_cache[li] = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, max_kv_in);
        ggml_set_name(pd.t_k_cache[li], kn); ggml_set_input(pd.t_k_cache[li]);
        pd.t_v_cache[li] = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, max_kv_in);
        ggml_set_name(pd.t_v_cache[li], vn); ggml_set_input(pd.t_v_cache[li]);

        // Pre-attention norm + Q/K/V
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

        ggml_tensor* Kc = ggml_cont(g, K);  // [hd, nkv, 1]
        ggml_tensor* Vc = ggml_cont(g, V);

        // Cache outputs — independent cont so the attention path can't trample them.
        pd.t_k_new[li] = ggml_cont(g, ggml_reshape_2d(g, Kc, kv_dim, 1));
        ggml_set_name(pd.t_k_new[li], kon); ggml_set_output(pd.t_k_new[li]);
        pd.t_v_new[li] = ggml_cont(g, ggml_reshape_2d(g, Vc, kv_dim, 1));
        ggml_set_name(pd.t_v_new[li], von); ggml_set_output(pd.t_v_new[li]);

        // Full K/V: concat(past_cache[max_kv-1], current[1]) → [hd, nkv, max_kv]
        ggml_tensor* kp    = ggml_reshape_3d(g, pd.t_k_cache[li], hd, nkv, max_kv_in);
        ggml_tensor* Kfull = ggml_concat(g, kp, Kc, 2);
        ggml_tensor* vp    = ggml_reshape_3d(g, pd.t_v_cache[li], hd, nkv, max_kv_in);
        ggml_tensor* Vfull = ggml_concat(g, vp, Vc, 2);

        // GQA expand (for this model nh==nkv so the block is skipped)
        int kv_repeat = nh / nkv;
        if (kv_repeat > 1) {
            Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, max_kv);
            Kfull = ggml_reshape_3d(g, ggml_repeat(g, Kfull,
                        ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, max_kv)),
                        hd, nh, max_kv);
            Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, max_kv);
            Vfull = ggml_reshape_3d(g, ggml_repeat(g, Vfull,
                        ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, max_kv)),
                        hd, nh, max_kv);
        }

        // Attention
        Q     = ggml_cont(g, ggml_permute(g, Q,     0, 2, 1, 3));  // [hd, 1,      nh]
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));  // [hd, max_kv, nh]
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(g, Kfull, Q);  // [max_kv, nh, 1]
        scores = ggml_soft_max_ext(g, scores, pd.t_mask, 1.0f / sqrtf((float)hd), 0.0f);

        ggml_tensor* Vt   = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
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
    }

    // Output norm + LM head (in-graph so we avoid a separate per-token graph)
    ggml_tensor* normed = rmsnorm(x, ctx.m.output_norm_w);
    ggml_tensor* lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    pd.t_logits = ggml_mul_mat(g, lm_w, normed);  // [V, 1]
    ggml_set_name(pd.t_logits, "pd_logits"); ggml_set_output(pd.t_logits);

    ggml_build_forward_expand(pd.gf, pd.t_logits);
    for (int li = 0; li < n_layers; li++) {
        ggml_build_forward_expand(pd.gf, pd.t_k_new[li]);
        ggml_build_forward_expand(pd.gf, pd.t_v_new[li]);
    }
    return pd;
}

// Build attention-only graph for one LLM layer (no FFN — MoE done on CPU)
// For layer 0 (dense), includes the FFN in the graph.
static llm_attn_graph build_llm_layer_attn(ds_ocr2_ctx &ctx, int li, int T, int n_past,
                                            bool include_ffn, bool include_moe = false) {
    auto &lhp = ctx.m.lhp;
    auto &ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads;
    int hd = lhp.head_dim;
    int Lk = n_past + T;
    float eps = lhp.rms_eps;

    size_t meta_sz = 4 * 1024 * 1024;
    llm_attn_graph lag;
    lag.meta.resize(meta_sz);  // owned by lag; survives the return (move preserves data ptr)
    ggml_init_params ip = { meta_sz, lag.meta.data(), true };
    lag.gctx = ggml_init(ip);
    auto *g = lag.gctx;
    lag.gf = ggml_new_graph_custom(g, 4096, false);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        // ensure_f32: the norm weight is F16 in an all-F16 GGUF, and ggml's
        // elementwise mul does not support an f32×f16 operand pair.
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
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

    // Materialize K/V once for the attention path. K/V come from rope as
    // [hd, nkv, T] (memory: hd fastest, then nkv, then T) — exactly the layout
    // the cache reload expects (reshape_3d(hd, nkv, n_past)). Do NOT permute the
    // token/head axes (an earlier permute(0,2,1,3) transposed T<->nkv and
    // scrambled the cache).
    ggml_tensor *Kc = ggml_cont(g, K);  // [hd, nkv, T]
    ggml_tensor *Vc = ggml_cont(g, V);

    // Cache outputs: an INDEPENDENT cont (not aliasing Kc/Vc). The attention
    // path below consumes Kc/Vc; under the no-alloc scheduler their buffers get
    // recycled once attention reads them, so a cache view sharing that buffer
    // would read garbage on the read-back (prefill attention still works, but
    // the cache is poisoned — exactly the "first token right, rest garbage"
    // symptom). Giving k_out/v_out their own cont copies the data into a
    // dedicated buffer that survives. Matches the verified qwen2vl_ocr path
    // (which conts K separately for cache vs attention).
    ggml_tensor *K_new = ggml_cont(g, ggml_reshape_2d(g, Kc, nkv * hd, T));
    ggml_set_name(K_new, "k_out"); ggml_set_output(K_new);

    ggml_tensor *V_new = ggml_cont(g, ggml_reshape_2d(g, Vc, nkv * hd, T));
    ggml_set_name(V_new, "v_out"); ggml_set_output(V_new);

    // Build full K/V (cache + new) for attention — use Kc/Vc, not the cache outs.
    ggml_tensor *Kfull, *Vfull;
    if (n_past > 0) {
        ggml_tensor *kc3 = ggml_reshape_3d(g, k_cache_in, hd, nkv, n_past);
        Kfull = ggml_concat(g, kc3, Kc, 2);  // [hd, nkv, Lk]

        ggml_tensor *vc3 = ggml_reshape_3d(g, v_cache_in, hd, nkv, n_past);
        Vfull = ggml_concat(g, vc3, Vc, 2);
    } else {
        Kfull = Kc;
        Vfull = Vc;
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
    } else if (include_moe) {
        // DeepSeek-V2 MoE FFN on Metal via ggml_mul_mat_id. Router → softmax →
        // top-k (raw probs; norm_topk_prob=False, routed_scaling_factor=1.0) →
        // per-expert SwiGLU dispatch → weighted sum + a (combined) shared expert.
        int n_exp = lhp.n_experts, K = lhp.n_experts_top;
        ggml_tensor *residual = x;
        ggml_tensor *hn = rmsnorm(x, ly.post_ln_w);  // [D, T]

        ggml_tensor *logits = ggml_mul_mat(g, ly.router_w, hn);    // [n_exp, T]
        ggml_tensor *probs  = ggml_soft_max(g, logits);
        ggml_tensor *ids    = ggml_top_k(g, probs, K);             // [K, T] I32
        ggml_tensor *p3     = ggml_reshape_3d(g, probs, 1, n_exp, T);
        ggml_tensor *top_w  = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), K, T);  // [K, T]
        top_w = ggml_scale(g, top_w, lhp.routed_scaling_factor);

        ggml_tensor *hn3 = ggml_reshape_3d(g, hn, D, 1, T);
        ggml_tensor *hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, K, T));
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));  // [inter,K,T]
        ggml_tensor *up   = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
        ggml_tensor *down = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gate, up), ids);  // [D,K,T]

        // Weighted sum over the K experts: down [D,K,T] → [K,D,T], w [K,1,T] → [1,D,T].
        ggml_tensor *down_p = ggml_cont(g, ggml_permute(g, down, 1, 0, 2, 3));
        ggml_tensor *w_col  = ggml_reshape_3d(g, top_w, K, 1, T);
        ggml_tensor *routed = ggml_reshape_2d(g, ggml_mul_mat(g, w_col, down_p), D, T);

        // Combined shared expert (always active), SwiGLU on the same normed input.
        ggml_tensor *sg = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
        ggml_tensor *su = ggml_mul_mat(g, ly.shared_up_w, hn);
        ggml_tensor *shared = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));

        x = ggml_add(g, residual, ggml_add(g, routed, shared));
    }

    ggml_set_name(x, "layer_output"); ggml_set_output(x);
    ggml_build_forward_expand(lag.gf, x);
    // k_out/v_out (the cache copies) are NOT ancestors of layer_output (the
    // attention path consumes Kc/Vc, not these), so expand them explicitly via
    // their pointers — a graph lookup by name would miss them (not yet added).
    ggml_build_forward_expand(lag.gf, K_new);
    ggml_build_forward_expand(lag.gf, V_new);
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

    struct exp_w { std::vector<float> gw, uw, dw; };
    std::vector<exp_w> exp_ws(n_exp);

    // Pass 1 (serial, cheap): RMSNorm + route every token, recording its
    // normed vector and its top-k experts/weights. Also note which experts get
    // used so we dequant only those.
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
        // norm_topk_prob=False, routed_scaling_factor=1.0 → raw top-k softmax probs.
        for (int k = 0; k < top_k; k++) {
            tk_idx[t][k] = scored[k].second;
            tk_w[t][k]   = scored[k].first * scale;
            used[scored[k].second] = 1;
        }
    }

    // Dequant the used routed experts once (parallel-unsafe lazy dequant is gone).
    for (int e = 0; e < n_exp; e++)
        if (used[e]) {
            exp_ws[e].gw = to_f32(ly.experts[e].gate_w);
            exp_ws[e].uw = to_f32(ly.experts[e].up_w);
            exp_ws[e].dw = to_f32(ly.experts[e].down_w);
        }

    // Pass 2 (parallel over tokens): each token's expert FFNs are independent
    // and write only to its own row, so split the token range across threads.
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
// Full LLM decoder forward
// ---------------------------------------------------------------------------

// Runs the MoE decoder. `prompt_embeds` is the fully-assembled prompt embedding
// matrix [n_prompt x D] (bos + image features + view-separator + instruction
// token embeddings), built by the caller. Generation continues until EOS.
static bool run_llm_decoder(ds_ocr2_ctx &ctx, const float *prompt_embeds, int n_prompt, int max_new,
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

    // Per-row embedding: dequant only the requested row on demand instead of
    // the whole 128k×1280 table (~655 MB held for the entire decode). The decode
    // touches ~one row per generated token, so the table f32 copy was pure waste.
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

    // Dequant weights needed for LM head
    auto norm_w = to_f32(ctx.m.output_norm_w);
    // LM head: default runs as a Metal quantized mul_mat (DS_LMHEAD_CPU=1 forces
    // the scalar path). Only dequant the 662 MB head weight for the CPU path.
    ggml_tensor *lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    bool lmhead_cpu = getenv("DS_LMHEAD_CPU") != nullptr;
    std::vector<float> head_w;
    if (lmhead_cpu) head_w = to_f32(lm_w);

    // Diagnostic: DS_NO_KV disables the KV cache and re-runs the entire growing
    // sequence each step (n_past always 0). Slow but a ground-truth reference to
    // isolate cache bugs from prefill bugs.
    bool no_kv = getenv("DS_NO_KV") != nullptr;
    std::vector<float> full_emb(prompt_embeds, prompt_embeds + (size_t)n_prompt * D);

    // Run generation loop
    int n_generated = 0;
    std::vector<int32_t> cur_tokens;  // tokens generated so far (single-token steps)
    int n_past = 0;

    // Persistent T=1 decode graph: allocate once after prefill, reuse every gen step.
    // DS_DECODE_REBUILD=1 forces the old per-step build path (useful for bisection).
    // Cap the graph window: attending over n_prompt+max_new positions when only n_prompt
    // are valid at the first step wastes 5× compute. We limit the extension so the
    // fixed-size attention overhead stays ≤1.4× of the actual n_past. Steps beyond
    // this cap fall back to the per-step rebuild path.
    static constexpr int PD_GEN_CAP = 96;
    bool use_pd  = !getenv("DS_DECODE_REBUILD") && ctx.moe_metal && !no_kv && !lmhead_cpu;
    int  pd_max_kv = use_pd ? std::min(n_prompt + std::min(max_new, PD_GEN_CAP),
                                        lhp.max_position_embeddings) : 0;
    PdGraph pd;
    std::vector<ggml_fp16_t> pd_mask;
    std::vector<float> pd_k_tmp(kv_dim), pd_v_tmp(kv_dim);
    bool pd_ready = false;
    if (use_pd && pd_max_kv > n_prompt) {
        pd = build_persistent_decode_graph(ctx, pd_max_kv);
        // Mask init: all -INF; position max_kv-1 (current token's K slot) is always 0.
        pd_mask.assign(pd_max_kv, ggml_fp32_to_fp16(-INFINITY));
        pd_mask[pd_max_kv - 1] = ggml_fp32_to_fp16(0.0f);
    } else {
        use_pd = false;
    }

    auto _decode_t0 = std::chrono::steady_clock::now();
    int _decode_gen_steps = 0;
    while (n_generated < max_new) {
        // Prefill (n_past==0) processes the whole assembled prompt; subsequent
        // steps process one freshly-generated token at a time. In no_kv mode the
        // whole sequence is reprocessed every step.
        int T = no_kv ? (int)(full_emb.size() / D) : ((n_past == 0) ? n_prompt : (int)cur_tokens.size());
        if (getenv("DS_DBG"))
            fprintf(stderr, "  [dbg] decode step gen=%d n_past=%d T=%d pd=%d\n",
                    n_generated, n_past, T, (int)use_pd);

        bool did_pd = false;
        std::vector<float> logits(V, 0.0f);

        // === Persistent decode path for T=1 generation steps ===
        if (use_pd && n_past > 0) {
            // Fall back once the KV cache window is full (n_past == pd_max_kv would
            // need pd_max_kv past entries but k_cache_in only holds pd_max_kv-1).
            if (n_past >= pd_max_kv) {
                if (getenv("DS_DBG"))
                    fprintf(stderr, "  [pd] KV full (n_past=%d >= max_kv=%d), fall back\n",
                            n_past, pd_max_kv);
                use_pd = false;
            }
        }
        if (use_pd && n_past > 0) {
            if (!pd_ready) {
                // First gen step: allocate the persistent graph then upload prefill KV.
                // Prefill's last sched_compute left is_alloc=true; reset to permit alloc.
                ggml_backend_sched_reset(ctx.sched);
                auto _pa_t0 = std::chrono::steady_clock::now();
                if (!ggml_backend_sched_alloc_graph(ctx.sched, pd.gf)) {
                    fprintf(stderr, "[ds_ocr2] pd alloc failed — falling back to per-step rebuild\n");
                    use_pd = false;
                } else {
                    if (getenv("DS_DBG_VERBOSE")) {
                        auto _pa_t1 = std::chrono::steady_clock::now();
                        fprintf(stderr, "  [pd] initial alloc=%lldms max_kv=%d\n",
                                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_pa_t1-_pa_t0).count(),
                                pd_max_kv);
                    }
                    // Unlock all prefill positions in the mask.
                    for (int ki = 0; ki < n_past; ki++)
                        pd_mask[ki] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, pd_mask.data(), 0,
                                            (size_t)pd_max_kv * sizeof(ggml_fp16_t));
                    // Upload prefill KV cache (valid positions 0..n_past-1).
                    size_t kv_b = (size_t)n_past * kv_dim * sizeof(float);
                    for (int li = 0; li < n_layers; li++) {
                        ggml_backend_tensor_set(pd.t_k_cache[li],
                                                ctx.kvc.k_cache[li].data(), 0, kv_b);
                        ggml_backend_tensor_set(pd.t_v_cache[li],
                                                ctx.kvc.v_cache[li].data(), 0, kv_b);
                    }
                    pd_ready = true;
                }
            } else {
                // Subsequent gen steps: incrementally unlock the slot for the token
                // appended to the KV cache last step (position n_past-1 in the cache,
                // which maps to Kfull position n_past-1 via k_cache_in[0..max_kv-2]).
                int prev = n_past - 1;
                if (prev < pd_max_kv - 1) {
                    pd_mask[prev] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, &pd_mask[prev],
                                            (size_t)prev * sizeof(ggml_fp16_t),
                                            sizeof(ggml_fp16_t));
                }
                // Upload the one new KV entry appended last step at cache index n_past-1.
                size_t off = (size_t)(n_past - 1) * kv_dim * sizeof(float);
                size_t sz  = (size_t)kv_dim * sizeof(float);
                for (int li = 0; li < n_layers; li++) {
                    ggml_backend_tensor_set(pd.t_k_cache[li],
                        ctx.kvc.k_cache[li].data() + (n_past - 1) * kv_dim, off, sz);
                    ggml_backend_tensor_set(pd.t_v_cache[li],
                        ctx.kvc.v_cache[li].data() + (n_past - 1) * kv_dim, off, sz);
                }
            }

            if (use_pd) {
                int32_t tok_id = cur_tokens[0];
                ggml_backend_tensor_set(pd.t_cur_tok_id, &tok_id, 0, sizeof(int32_t));
                int32_t pos_id = n_past;
                ggml_backend_tensor_set(pd.t_pos_ids, &pos_id, 0, sizeof(int32_t));

                // is_alloc=true from the one-time sched_alloc above → compute skips alloc.
                auto _pd_t0 = std::chrono::steady_clock::now();
                ggml_backend_sched_graph_compute(ctx.sched, pd.gf);
                if (getenv("DS_DBG_VERBOSE")) {
                    auto _pd_t1 = std::chrono::steady_clock::now();
                    fprintf(stderr, "  [pd] gen=%d n_past=%d compute=%lldms\n", n_generated, n_past,
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_pd_t1 - _pd_t0).count());
                }

                ggml_backend_tensor_get(pd.t_logits, logits.data(), 0, (size_t)V * sizeof(float));

                for (int li = 0; li < n_layers; li++) {
                    ggml_backend_tensor_get(pd.t_k_new[li], pd_k_tmp.data(), 0, kv_dim * sizeof(float));
                    ggml_backend_tensor_get(pd.t_v_new[li], pd_v_tmp.data(), 0, kv_dim * sizeof(float));
                    ctx.kvc.k_cache[li].insert(ctx.kvc.k_cache[li].end(),
                                               pd_k_tmp.begin(), pd_k_tmp.end());
                    ctx.kvc.v_cache[li].insert(ctx.kvc.v_cache[li].end(),
                                               pd_v_tmp.begin(), pd_v_tmp.end());
                }
                n_past++;
                did_pd = true;
            }
        }

        if (!did_pd) {
            // === Original per-step rebuild path (prefill + fallback gen) ===

            // Build input embeddings
            std::vector<float> input_emb(T * D);
            if (no_kv) {
                memcpy(input_emb.data(), full_emb.data(), (size_t)T * D * sizeof(float));
            } else if (n_past == 0) {
                memcpy(input_emb.data(), prompt_embeds, (size_t)T * D * sizeof(float));
            } else {
                for (int t = 0; t < T; t++)
                    get_embedding(cur_tokens[t], input_emb.data() + t * D);
            }

            // Process each layer
            std::vector<float> hidden(input_emb);

            for (int li = 0; li < n_layers; li++) {
                bool is_dense = (li == 0);
                // MoE in-graph (Metal) when experts were stacked; else CPU fallback.
                bool moe_in_graph = ctx.moe_metal && !is_dense;

                // Build and run attention graph
                auto _rb_t0 = std::chrono::steady_clock::now();
                auto lag = build_llm_layer_attn(ctx, li, T, n_past, is_dense, moe_in_graph);
                auto _rb_t1 = std::chrono::steady_clock::now();
                ggml_backend_sched_reset(ctx.sched);
                auto _rb_t2 = std::chrono::steady_clock::now();
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

                auto _t0 = std::chrono::steady_clock::now();
                ggml_backend_sched_graph_compute(ctx.sched, lag.gf);
                auto _t1 = std::chrono::steady_clock::now();

                // Read outputs
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "layer_output"),
                                        hidden.data(), 0, T * D * sizeof(float));

                // Update KV cache
                std::vector<float> k_new(kv_dim * T), v_new(kv_dim * T);
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "k_out"),
                                        k_new.data(), 0, kv_dim * T * sizeof(float));
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "v_out"),
                                        v_new.data(), 0, kv_dim * T * sizeof(float));

                if (!no_kv) {
                    ctx.kvc.k_cache[li].insert(ctx.kvc.k_cache[li].end(), k_new.begin(), k_new.end());
                    ctx.kvc.v_cache[li].insert(ctx.kvc.v_cache[li].end(), v_new.begin(), v_new.end());
                }

                ggml_free(lag.gctx);

                // MoE FFN (layers 1-11): in-graph on Metal (done above) or CPU here.
                auto _t2 = std::chrono::steady_clock::now();
                if (!is_dense && !moe_in_graph) {
                    moe_ffn_cpu(ctx, li, hidden.data(), T);
                }
                auto _t3 = std::chrono::steady_clock::now();
                if (getenv("DS_DBG_VERBOSE"))
                    fprintf(stderr, "  [dbg] llm li=%d T=%d build=%lldms alloc=%lldms attn=%lldms moe=%lldms\n",
                            li, T,
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_rb_t1-_rb_t0).count(),
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_t0-_rb_t2).count(),
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_t1-_t0).count(),
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_t3-_t2).count());

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

            if (!no_kv) n_past += T;

            // Final norm + LM head
            std::vector<float> last_hidden(D);
            rmsnorm_cpu(hidden.data() + (T - 1) * D, last_hidden.data(), D, norm_w.data(), lhp.rms_eps);

            if (lmhead_cpu) {
                linear_cpu(last_hidden.data(), logits.data(), D, V, head_w.data(), nullptr);
            } else {
                // logits = lm_w @ last_hidden on Metal (quantized; ~165M mults/token
                // off the scalar path). Build+run+free in scope (no dangling graph).
                size_t meta_sz = 1 * 1024 * 1024;
                std::vector<uint8_t> mb(meta_sz);
                ggml_init_params ip = { meta_sz, mb.data(), true };
                ggml_context* gc = ggml_init(ip);
                ggml_cgraph* gf = ggml_new_graph(gc);
                ggml_tensor* in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, 1);
                ggml_set_name(in, "lmh_in"); ggml_set_input(in);
                ggml_tensor* out = ggml_mul_mat(gc, lm_w, in);  // [V, 1]
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
                    fprintf(stderr, "  logits: cos_min=%.6f max_abs=%.6f %s\n",
                            r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        } // end !did_pd

        // Argmax
        int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        // Confidence
        float max_l = logits[next];
        float sum_e = 0;
        for (int v = 0; v < V; v++) sum_e += expf(logits[v] - max_l);
        out_confs.push_back(1.0f / sum_e);

        out_ids.push_back(next);
        n_generated++;

        if (getenv("DS_DBG")) {
            const char *pc = (next >= 0 && next < ctx.tok_vocab_size) ? ctx.id_to_piece[next].c_str() : "?";
            fprintf(stderr, "  [gen %d] id=%d piece=%s\n", n_generated - 1, next, pc);
        }

        _decode_gen_steps++;
        if (next == lhp.eos_token_id) break;

        // Next step: single token (or, in no_kv mode, append to the full sequence)
        cur_tokens = {(int32_t)next};
        if (no_kv) {
            size_t off = full_emb.size();
            full_emb.resize(off + D);
            get_embedding(next, full_emb.data() + off);
        }
    }

    if (getenv("DS_DBG")) {
        auto _decode_t1 = std::chrono::steady_clock::now();
        long long _decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   _decode_t1 - _decode_t0).count();
        fprintf(stderr, "  [decode] total=%lldms steps=%d prefill+gen use_pd=%d\n",
                _decode_ms, _decode_gen_steps, (int)(n_past > 0 && !getenv("DS_DECODE_REBUILD")));
    }
    if (pd.gctx) ggml_free(pd.gctx);
    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer decode
// ---------------------------------------------------------------------------

static std::string decode_tokens(const ds_ocr2_ctx &ctx, const int32_t *ids, int n) {
    // Inverse of the GPT-2 byte-level BPE byte_encoder(): codepoint -> raw byte.
    // The vocab pieces live in "byte-encoded unicode" space (e.g. 'Ġ' = space),
    // so we concatenate the pieces then map each UTF-8 codepoint back to its
    // original byte to recover real text.
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
        // Skip special marker tokens like <｜...｜>.
        if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>') continue;
        merged += piece;
    }

    // Decode UTF-8 codepoints of `merged` back to raw bytes.
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
        else result.append(merged, i, len);  // not in map: keep as-is
        i += len;
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

    // Parity harness: when DS_REF points at a crispembed_diff GGUF dump, each
    // stage (sam_output, qwen2_enc_output, projector_output, llm logits) is
    // compared against the reference and a cos_min/max_abs line is printed.
    // See tools/dump_deepseek_ocr2_reference.py.
    if (const char *ref = getenv("DS_REF")) ctx.diff_ref_path = ref;

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

    auto _it = std::chrono::steady_clock::now();
    auto init_ms = [&](const char *w) {
        if (!getenv("DS_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] init.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _it).count());
        _it = now;
    };
    if (!load_tensors(ctx, model_path)) {
        fprintf(stderr, "deepseek_ocr2: failed to load tensors\n");
        delete c; return nullptr;
    }
    init_ms("load_tensors");

    precompute_rpe_tables(ctx);

    // Stack MoE experts for the Metal ggml_mul_mat_id decode path (default).
    // DS_MOE_CPU=1 keeps the per-token CPU-scalar moe_ffn_cpu (slower, but the
    // reference path / fallback for platforms where mul_mat_id misbehaves).
    if (!getenv("DS_MOE_CPU")) {
        ctx.moe_metal = stack_moe_experts(ctx);
        if (!ctx.moe_metal)
            fprintf(stderr, "deepseek_ocr2: MoE expert stacking failed — using CPU MoE\n");
    }
    init_ms("stack_moe_experts");

    ctx.bench = (std::getenv("CRISPEMBED_DEEPSEEK_OCR2_BENCH") != nullptr);

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
    if (c.moe_buf) ggml_backend_buffer_free(c.moe_buf);
    if (c.moe_ctx) ggml_free(c.moe_ctx);
    // model_buf/model_ctx alias model_wl — free via free_weights (also unmaps).
    c.model_buf = nullptr;
    c.model_ctx = nullptr;
    core_gguf::free_weights(c.model_wl);
    if (c.backend) ggml_backend_free(c.backend);
    if (c.backend_cpu) ggml_backend_free(c.backend_cpu);
    delete ctx;
}

const char * deepseek_ocr2_recognize_raw(deepseek_ocr2_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!ctx || !px) { if (out_len) *out_len = 0; return ""; }

    // Isolation test: DS_TEXT_TEST runs the LLM decoder as a pure language
    // model (no vision) to verify the decoder/MoE/rope in isolation.
    if (const char *tt = getenv("DS_TEXT_TEST")) {
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

    // Preprocess like the HF reference: ImageOps.pad(image, (imgS, imgS)) —
    // resize preserving aspect ratio to fit inside imgS×imgS, center, and pad
    // the borders with the mean colour (gray 127). Then normalize to [-1,1]
    // (mean=std=0.5). The padded border normalizes to exactly 0.
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
                    val = s.image_mean[c];  // gray padding -> normalizes to 0
                } else {
                    // Bilinear sample from source at the un-scaled position.
                    float sx = (x - ox + 0.5f) / scale - 0.5f;
                    float sy = (y - oy + 0.5f) / scale - 0.5f;
                    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
                    float dx = sx - x0, dy = sy - y0;
                    int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                    x0 = std::min(std::max(x0, 0), w - 1);
                    y0 = std::min(std::max(y0, 0), h - 1);
                    auto P = [&](int xx, int yy) { return (float)px[(yy * w + xx) * ch + ci] / 255.0f; };
                    val = P(x0,y0)*(1-dx)*(1-dy) + P(x1,y0)*dx*(1-dy)
                        + P(x0,y1)*(1-dx)*dy     + P(x1,y1)*dx*dy;
                }
                pixels[c * imgS * imgS + y * imgS + x] = (val - s.image_mean[c]) / s.image_std[c];
            }
        }
    }

    const bool bench = ctx->inner.bench;
    auto t_total = std::chrono::steady_clock::now();
    bool dbg_t = getenv("DS_DBG") != nullptr;
    auto _ts = std::chrono::steady_clock::now();
    auto stage_ms = [&](const char *name) {
        if (!dbg_t) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] %s %lldms\n", name,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _ts).count());
        _ts = now;
    };

    // 1. SAM vision encoder
    auto t_sam = std::chrono::steady_clock::now();
    std::vector<float> sam_features;
    int n_sam_tokens, sam_dim;
    if (!encode_sam(ctx->inner, pixels.data(), sam_features, n_sam_tokens, sam_dim)) {
        fprintf(stderr, "deepseek_ocr2: SAM encoding failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("sam");
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_sam).count();
        fprintf(stderr, "[deepseek_ocr2-bench] sam_encoder: %lldms\n", (long long)ms);
    }

    // 2. Qwen2 bidirectional encoder
    auto t_proj = std::chrono::steady_clock::now();
    std::vector<float> enc_out;
    int n_enc_tokens, enc_dim;
    if (!encode_qwen2(ctx->inner, sam_features.data(), n_sam_tokens, sam_dim,
                      enc_out, n_enc_tokens, enc_dim)) {
        fprintf(stderr, "deepseek_ocr2: Qwen2 encoder failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("qwen2_enc");

    // 3. Project to LLM dimension
    std::vector<float> proj_out;
    if (!project_to_llm(ctx->inner, enc_out.data(), n_enc_tokens, enc_dim, proj_out)) {
        fprintf(stderr, "deepseek_ocr2: projection failed\n");
        if (out_len) *out_len = 0; return "";
    }
    stage_ms("projector");
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_proj).count();
        fprintf(stderr, "[deepseek_ocr2-bench] projection: %lldms\n", (long long)ms);
    }

    fprintf(stderr, "deepseek_ocr2: stages done — sam=%d/%d qwen2=%d/%d proj=%d image tokens\n",
            n_sam_tokens, sam_dim, n_enc_tokens, enc_dim, n_enc_tokens);

    // 4. Assemble the LLM prompt embeddings. The HF reference (infer + plain
    //    template, prompt "<image>\nFree OCR.") builds the token sequence:
    //        [bos] + <image>*N + <view_sep> + tokenize("\nFree OCR.")
    //    where the N image placeholders + the view-separator placeholder are
    //    masked-scatter-replaced by [global_features (N), view_seperator (1)].
    //    We build the embedding matrix directly: text positions use
    //    embed_tokens, image positions use the projected vision features, and
    //    the separator position uses the learned v.view_separator embedding.
    auto &mdl = ctx->inner.m;
    auto &lhp = mdl.lhp;
    int D = lhp.hidden;
    auto vsep = to_f32(mdl.view_separator);  // [D] — tiny (1 row), to_f32 is fine

    // Per-row embed dequant: avoid materialising the full 128k×1280 (~655 MB) table.
    ggml_tensor* emb_t = mdl.embed_tokens;
    const auto* emb_tt = ggml_get_type_traits(emb_t->type);
    const size_t emb_row_bytes = ggml_row_size(emb_t->type, D);
    std::vector<uint8_t> emb_row_buf(emb_row_bytes);

    // Instruction text after <image>. DeepSeek-OCR2 plain prompt: "\nFree OCR."
    std::vector<int32_t> instr_ids =
        core_bpe::tokenize_simple(ctx->inner.token_to_id, ctx->inner.merge_rank, "\nFree OCR.");

    int n_img_tokens = n_enc_tokens;            // 256 global features
    int n_prompt = 1 /*bos*/ + n_img_tokens + 1 /*view_sep*/ + (int)instr_ids.size();
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
    put_tok(0);  // bos = <｜begin▁of▁sentence｜>
    for (int i = 0; i < n_img_tokens; i++) {
        memcpy(prompt_embeds.data() + (size_t)row * D, proj_out.data() + (size_t)i * D,
               D * sizeof(float));
        row++;
    }
    memcpy(prompt_embeds.data() + (size_t)row * D, vsep.data(), D * sizeof(float));  // view separator
    row++;
    for (int32_t id : instr_ids) put_tok(id);

    if (getenv("DS_DBG")) {
        fprintf(stderr, "  [dbg] prompt: bos + %d img + sep + %zu instr = %d tokens; instr_ids:",
                n_img_tokens, instr_ids.size(), n_prompt);
        for (int32_t id : instr_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    // 5. LLM decoder
    auto t_llm = std::chrono::steady_clock::now();
    std::vector<int32_t> gen_ids;
    std::vector<float> gen_confs;
    if (!run_llm_decoder(ctx->inner, prompt_embeds.data(), n_prompt, 1024,
                         gen_ids, gen_confs)) {
        fprintf(stderr, "deepseek_ocr2: LLM decode failed\n");
        if (out_len) *out_len = 0; return "";
    }
    if (bench) {
        auto llm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_llm).count();
        fprintf(stderr, "[deepseek_ocr2-bench] llm_decoder: %lldms\n", (long long)llm_ms);
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[deepseek_ocr2-bench] total: %lldms\n", (long long)total_ms);
    }

    if (getenv("DS_DBG")) {
        fprintf(stderr, "  [dbg] gen_ids (%zu):", gen_ids.size());
        for (int id : gen_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
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
