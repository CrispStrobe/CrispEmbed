// got_ocr.cpp — GOT-OCR2 engine: SAM ViT-B + Qwen2-0.5B via ggml.
//
// Vision: per-layer ggml graph with CPU window partition (SAM ViT-B pattern).
// LLM: ggml graph with KV cache (standard Qwen2 pattern).

#include "got_ocr.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
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
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using core_cpu::to_f32;
using core_cpu::layernorm_cpu;
using core_cpu::layernorm2d_cpu;
using core_cpu::linear_cpu;
using core_cpu::conv2d_cpu;

// ---------------------------------------------------------------------------
// Window partition / unpartition (SAM ViT pattern)
// ---------------------------------------------------------------------------

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

// Decomposed RPE: get rel_pos table for given q_size, k_size.
// rel_pos: (L, head_dim), L = 2*input_size - 1
// Returns: (q_size * k_size * head_dim) in row-major [qi][ki][d]
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

// Reformat RPE table from row-major [(q*aH+k)*hd+d] to ggml col-major [hd, aH, aH].
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

static bool load_hparams(got_ocr::context &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *key, uint32_t def) -> uint32_t {
        return core_gguf::kv_u32(g, key, def);
    };
    auto f32v = [&](const char *key, float def) -> float {
        return core_gguf::kv_f32(g, key, def);
    };

    auto &v = ctx.m.vhp;
    v.depth = u32("got_ocr.vision.depth", v.depth);
    v.hidden_size = u32("got_ocr.vision.hidden_size", v.hidden_size);
    v.intermediate_size = u32("got_ocr.vision.intermediate_size", v.intermediate_size);
    v.num_heads = u32("got_ocr.vision.num_heads", v.num_heads);
    v.head_dim = v.hidden_size / v.num_heads;
    v.patch_size = u32("got_ocr.vision.patch_size", v.patch_size);
    v.image_size = u32("got_ocr.vision.image_size", v.image_size);
    v.window_size = u32("got_ocr.vision.window_size", v.window_size);
    v.neck_out_channels = u32("got_ocr.vision.neck_out_channels", v.neck_out_channels);

    // Global attention indexes
    {
        int key_id = gguf_find_key(g, "got_ocr.vision.global_attn_indexes");
        if (key_id >= 0) {
            int n = (int)gguf_get_arr_n(g, key_id);
            v.global_attn_indexes.resize(n);
            const void *data = gguf_get_arr_data(g, key_id);
            memcpy(v.global_attn_indexes.data(), data, n * sizeof(int32_t));
        } else {
            v.global_attn_indexes = {2, 5, 8, 11};
        }
    }

    // Image mean/std
    {
        int key_id = gguf_find_key(g, "got_ocr.vision.image_mean");
        if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3) {
            const void *data = gguf_get_arr_data(g, key_id);
            memcpy(v.image_mean, data, 3 * sizeof(float));
        }
    }
    {
        int key_id = gguf_find_key(g, "got_ocr.vision.image_std");
        if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3) {
            const void *data = gguf_get_arr_data(g, key_id);
            memcpy(v.image_std, data, 3 * sizeof(float));
        }
    }

    auto &l = ctx.m.lhp;
    l.vocab_size = u32("got_ocr.vocab_size", l.vocab_size);
    l.hidden_size = u32("got_ocr.hidden_size", l.hidden_size);
    l.intermediate_size = u32("got_ocr.intermediate_size", l.intermediate_size);
    l.num_hidden_layers = u32("got_ocr.num_hidden_layers", l.num_hidden_layers);
    l.num_attention_heads = u32("got_ocr.num_attention_heads", l.num_attention_heads);
    l.num_key_value_heads = u32("got_ocr.num_key_value_heads", l.num_key_value_heads);
    l.head_dim = u32("got_ocr.head_dim", l.head_dim);
    l.max_position_embeddings = u32("got_ocr.max_position_embeddings", l.max_position_embeddings);
    l.rms_norm_eps = f32v("got_ocr.rms_norm_eps", l.rms_norm_eps);
    l.rope_theta = f32v("got_ocr.rope_theta", l.rope_theta);
    l.image_token_id = u32("got_ocr.image_token_id", l.image_token_id);
    l.image_start_token_id = u32("got_ocr.image_start_token_id", l.image_start_token_id);
    l.image_end_token_id = u32("got_ocr.image_end_token_id", l.image_end_token_id);
    l.image_token_len = u32("got_ocr.image_token_len", l.image_token_len);
    l.eos_token_id = u32("got_ocr.tokenizer.eos_id", l.eos_token_id);

    // Tokenizer
    {
        int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (vocab_idx >= 0) {
            int n = (int)gguf_get_arr_n(g, vocab_idx);
            ctx.tok.id_to_piece.resize(n);
            for (int i = 0; i < n; i++)
                ctx.tok.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
            ctx.tok.vocab_size = n;
            ctx.tok.eos_id = (int)l.eos_token_id;
        }
    }

    core_gguf::free_metadata(g);
    return true;
}

static bool load_tensors(got_ocr::context &ctx, const char *path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "got_ocr", wl)) return false;

    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;
    auto &t = wl.tensors;
    auto F = [&](const char *n) -> ggml_tensor* {
        auto it = t.find(n);
        return it != t.end() ? it->second : nullptr;
    };

    auto &m = ctx.m;
    auto &v = m.vhp;
    auto &l = m.lhp;

    // Vision
    m.patch_embed_w = F("v.patch_embed.weight");
    m.patch_embed_b = F("v.patch_embed.bias");
    m.pos_embed = F("v.pos_embed");

    m.vis_blocks.resize(v.depth);
    for (uint32_t i = 0; i < v.depth; i++) {
        char buf[128];
        auto &blk = m.vis_blocks[i];
        blk.is_global = false;
        for (int g : v.global_attn_indexes)
            if (g == (int)i) { blk.is_global = true; break; }

        auto BF = [&](const char *s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, s);
            return F(buf);
        };
        blk.ln1_w = BF("ln1.weight");     blk.ln1_b = BF("ln1.bias");
        blk.ln2_w = BF("ln2.weight");     blk.ln2_b = BF("ln2.bias");
        blk.qkv_w = BF("attn_qkv.weight"); blk.qkv_b = BF("attn_qkv.bias");
        blk.proj_w = BF("attn_proj.weight"); blk.proj_b = BF("attn_proj.bias");
        blk.rel_pos_h = BF("attn_rel_pos_h");
        blk.rel_pos_w = BF("attn_rel_pos_w");
        blk.ffn_up_w = BF("ffn_up.weight");   blk.ffn_up_b = BF("ffn_up.bias");
        blk.ffn_down_w = BF("ffn_down.weight"); blk.ffn_down_b = BF("ffn_down.bias");
    }

    // Neck
    m.neck_conv1_w = F("v.neck_conv1.weight");
    m.neck_ln1_w = F("v.neck_ln1.weight"); m.neck_ln1_b = F("v.neck_ln1.bias");
    m.neck_conv2_w = F("v.neck_conv2.weight");
    m.neck_ln2_w = F("v.neck_ln2.weight"); m.neck_ln2_b = F("v.neck_ln2.bias");
    m.net_2_w = F("v.net_2.weight");
    m.net_3_w = F("v.net_3.weight");
    m.projector_w = F("v.projector.weight");
    m.projector_b = F("v.projector.bias");

    // LLM
    m.embed_tokens = F("l.embed_tokens.weight");
    m.llm_layers.resize(l.num_hidden_layers);
    for (uint32_t i = 0; i < l.num_hidden_layers; i++) {
        char buf[128];
        auto &ly = m.llm_layers[i];
        auto LF = [&](const char *s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "l.blk.%d.%s", i, s);
            return F(buf);
        };
        ly.input_layernorm_w = LF("input_layernorm.weight");
        ly.post_attention_layernorm_w = LF("post_attention_layernorm.weight");
        ly.q_w = LF("attn_q.weight"); ly.q_b = LF("attn_q.bias");
        ly.k_w = LF("attn_k.weight"); ly.k_b = LF("attn_k.bias");
        ly.v_w = LF("attn_v.weight"); ly.v_b = LF("attn_v.bias");
        ly.o_w = LF("attn_o.weight");
        ly.ffn_gate_w = LF("ffn_gate.weight");
        ly.ffn_up_w = LF("ffn_up.weight");
        ly.ffn_down_w = LF("ffn_down.weight");
    }
    m.output_norm_w = F("l.output_norm.weight");
    m.lm_head_w = F("l.lm_head.weight");  // nullptr if tied

    return true;
}

// Precompute RPE tables for each layer (called once at load)
static void precompute_rpe_tables(got_ocr::context &ctx) {
    auto &v = ctx.m.vhp;
    int n_layers = (int)v.depth;
    int hd = (int)v.head_dim;
    int nP = (int)(v.image_size / v.patch_size);
    int ws = (int)v.window_size;

    ctx.rp_h_per_layer.resize(n_layers);
    ctx.rp_w_per_layer.resize(n_layers);

    for (int li = 0; li < n_layers; li++) {
        auto &blk = ctx.m.vis_blocks[li];
        if (!blk.rel_pos_h || !blk.rel_pos_w) continue;

        auto rph_raw = to_f32(blk.rel_pos_h);
        auto rpw_raw = to_f32(blk.rel_pos_w);
        int L_h = (int)blk.rel_pos_h->ne[1]; // (L, hd) in ggml = ne[0]=hd, ne[1]=L
        int L_w = (int)blk.rel_pos_w->ne[1];

        int aH, aW;
        if (blk.is_global) { aH = nP; aW = nP; }
        else { aH = ws; aW = ws; }

        auto rph = get_rel_pos(aH, aH, rph_raw.data(), L_h, hd);
        auto rpw = get_rel_pos(aW, aW, rpw_raw.data(), L_w, hd);
        ctx.rp_h_per_layer[li] = std::move(rph);
        ctx.rp_w_per_layer[li] = std::move(rpw);
    }
}

bool got_ocr::load(context &ctx, const char *gguf_path, int n_threads, int verbosity) {
    ctx.n_threads = n_threads;
    ctx.verbosity = verbosity;

    if (!load_hparams(ctx, gguf_path)) {
        fprintf(stderr, "got_ocr: failed to load hparams\n");
        return false;
    }

    // GOT_OCR_FORCE_CPU=1 forces the CPU backend (parity with the other engines).
    // Useful to sidestep Metal op gaps and to A/B Metal-vs-CPU outputs while
    // debugging vision-encoder correctness (issue #25).
    bool force_cpu = (std::getenv("GOT_OCR_FORCE_CPU") &&
                      atoi(std::getenv("GOT_OCR_FORCE_CPU")));
    ctx.backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx.backend) ctx.backend = ggml_backend_cpu_init();  // fallback if init_best failed
    if (!ctx.backend) return false;
    if (ggml_backend_is_cpu(ctx.backend)) ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);

    // Scheduler. ggml_backend_sched_new asserts the LAST backend is CPU, so
    // when the best backend is a GPU (Metal/CUDA/Vulkan) append a CPU backend
    // as the fallback — otherwise this aborts on every GPU machine.
    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);
    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(), 32768, false, false);
    ctx.compute_meta.resize(16 * 1024 * 1024);

    if (!load_tensors(ctx, gguf_path)) {
        fprintf(stderr, "got_ocr: failed to load tensors\n");
        return false;
    }

    precompute_rpe_tables(ctx);

    ctx.bench = (std::getenv("CRISPEMBED_GOT_OCR_BENCH") != nullptr);

    if (verbosity >= 1) {
        auto &v = ctx.m.vhp;
        auto &l = ctx.m.lhp;
        fprintf(stderr, "got_ocr: loaded %s\n", gguf_path);
        fprintf(stderr, "  vision: %dL %dd %dH patch=%d img=%d ws=%d\n",
                v.depth, v.hidden_size, v.num_heads, v.patch_size, v.image_size, v.window_size);
        fprintf(stderr, "  llm: %dL %dd %dH/%dKV inter=%d vocab=%d\n",
                l.num_hidden_layers, l.hidden_size, l.num_attention_heads,
                l.num_key_value_heads, l.intermediate_size, l.vocab_size);
        fprintf(stderr, "  rope_theta=%.0f rms_eps=%g\n", l.rope_theta, l.rms_norm_eps);
        fprintf(stderr, "  tokenizer: %d tokens\n", ctx.tok.vocab_size);
    }
    return true;
}

void got_ocr::free_(context &ctx) {
    if (ctx.kvc.buf) ggml_backend_buffer_free(ctx.kvc.buf);
    if (ctx.kvc.ctx) ggml_free(ctx.kvc.ctx);
    if (ctx.sched) ggml_backend_sched_free(ctx.sched);
    if (ctx.model_buf) ggml_backend_buffer_free(ctx.model_buf);
    if (ctx.model_ctx) ggml_free(ctx.model_ctx);
    if (ctx.backend) ggml_backend_free(ctx.backend);
    if (ctx.backend_cpu) ggml_backend_free(ctx.backend_cpu);
    ctx = {};
}

// ---------------------------------------------------------------------------
// Vision encoder: per-layer ggml graph (SAM ViT-B)
// ---------------------------------------------------------------------------

static ggml_cgraph* build_vis_layer_graph(ggml_context* g,
                                           got_ocr::context* ctx,
                                           int li, int C, int T,
                                           int aH, int aW, int nW, int n_heads,
                                           bool skip_ln1 = false) {
    auto& layer = ctx->m.vis_blocks[li];
    int hd = C / n_heads;
    int wN = aH * aW;
    int batch = n_heads * nW;
    float attn_scale = 1.0f / sqrtf((float)hd);

    ggml_cgraph* gf = ggml_new_graph_custom(g, 512, false);

    // Input
    ggml_tensor* inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
    ggml_set_name(inp, "layer_input");
    ggml_set_input(inp);

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

    // Pre-LN (LayerNorm, not RMSNorm)
    ggml_tensor* cur = skip_ln1 ? inp : g_ln(g, inp, layer.ln1_w, layer.ln1_b, 1e-6f);

    // Fused QKV
    ggml_tensor* qkv = g_linear(g, cur, layer.qkv_w, layer.qkv_b);

    // Split Q, K, V
    ggml_tensor* Q = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], 0));
    ggml_tensor* K = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1],
                                                (size_t)C * sizeof(float)));
    ggml_tensor* V = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1],
                                                (size_t)2 * C * sizeof(float)));

    // Reshape: [C, T] → [hd, n_heads, wN, nW] → permute to [hd, wN, n_heads, nW]
    Q = ggml_reshape_4d(g, Q, hd, n_heads, wN, nW);
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(g, Q, hd, wN, batch);

    K = ggml_reshape_4d(g, K, hd, n_heads, wN, nW);
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(g, K, hd, wN, batch);

    V = ggml_reshape_4d(g, V, hd, n_heads, wN, nW);
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(g, V, hd, wN, batch);

    // Attention scores
    ggml_tensor* scores = ggml_mul_mat(g, K, Q);
    scores = ggml_scale(g, scores, attn_scale);

    // Decomposed RPE
    ggml_tensor* Q_4d = ggml_reshape_4d(g, Q, hd, aW, aH, batch);
    ggml_tensor* rp_h_4d = ggml_reshape_4d(g, rp_h_in, hd, aH, aH, 1);
    ggml_tensor* rel_h = ggml_mul_mat(g, rp_h_4d, Q_4d);
    rel_h = ggml_reshape_3d(g, rel_h, aH, wN, batch);
    rel_h = ggml_reshape_4d(g, rel_h, 1, aH, wN, batch);

    ggml_tensor* Q_w = ggml_cont(g, ggml_permute(g, Q_4d, 0, 2, 1, 3));
    ggml_tensor* rp_w_4d = ggml_reshape_4d(g, rp_w_in, hd, aW, aW, 1);
    ggml_tensor* rel_w = ggml_mul_mat(g, rp_w_4d, Q_w);
    rel_w = ggml_cont(g, ggml_permute(g, rel_w, 0, 2, 1, 3));
    rel_w = ggml_reshape_3d(g, rel_w, aW, wN, batch);
    rel_w = ggml_reshape_4d(g, rel_w, aW, 1, wN, batch);

    scores = ggml_reshape_4d(g, scores, aW, aH, wN, batch);
    scores = ggml_add(g, scores, rel_h);
    scores = ggml_add(g, scores, rel_w);
    scores = ggml_reshape_3d(g, scores, wN, wN, batch);

    // Softmax
    scores = ggml_soft_max_ext(g, scores, nullptr, 1.0f, 0.0f);

    // Attention output
    ggml_tensor* Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);

    attn = ggml_reshape_4d(g, attn, hd, wN, n_heads, nW);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, C, T);

    // Output projection
    attn = g_linear(g, attn, layer.proj_w, layer.proj_b);

    // Residual
    ggml_tensor* res_base = skip_ln1 ? res_inp : inp;
    cur = ggml_add(g, res_base, attn);

    // Pre-LN + GELU MLP
    ggml_tensor* residual = cur;
    cur = g_ln(g, cur, layer.ln2_w, layer.ln2_b, 1e-6f);
    ggml_tensor* up = g_linear(g, cur, layer.ffn_up_w, layer.ffn_up_b);
    up = ggml_gelu(g, up);
    cur = g_linear(g, up, layer.ffn_down_w, layer.ffn_down_b);
    cur = ggml_add(g, residual, cur);

    ggml_set_name(cur, "layer_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

bool got_ocr::encode_vision(context &ctx, const float *pixels, vision_result &out) {
    const bool bench = ctx.bench;
    auto t_vis_total = std::chrono::steady_clock::now();
    auto &v = ctx.m.vhp;
    int C = (int)v.hidden_size;
    int PS = (int)v.patch_size;
    int nP = (int)(v.image_size / v.patch_size);
    int N = nP * nP;
    int n_heads = (int)v.num_heads;
    int hd = C / n_heads;
    int ws = (int)v.window_size;
    int imgS = (int)v.image_size;

    // Patch embedding: im2col + ggml matmul (gated: CRISPEMBED_GOT_OCR_SCALAR_PATCH=1 for fallback)
    int patch_dim = 3 * PS * PS;
    std::vector<float> hidden(N * C);

    // im2col: extract non-overlapping patches → [N, patch_dim]
    std::vector<float> im2col(N * patch_dim);
    for (int py = 0; py < nP; py++)
        for (int px = 0; px < nP; px++) {
            int tok = py * nP + px;
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < PS; ky++)
                    for (int kx = 0; kx < PS; kx++)
                        im2col[tok * patch_dim + c * PS * PS + ky * PS + kx] =
                            pixels[c * imgS * imgS + (py * PS + ky) * imgS + (px * PS + kx)];
        }

    static const bool scalar_patch = (std::getenv("CRISPEMBED_GOT_OCR_SCALAR_PATCH") != nullptr);
    if (scalar_patch) {
        auto pe_w = to_f32(ctx.m.patch_embed_w);
        auto pe_b = to_f32(ctx.m.patch_embed_b);
        for (int t = 0; t < N; t++)
            for (int o = 0; o < C; o++) {
                float s = pe_b.empty() ? 0.0f : pe_b[o];
                for (int i = 0; i < patch_dim; i++)
                    s += pe_w[o * patch_dim + i] * im2col[t * patch_dim + i];
                hidden[t * C + o] = s;
            }
    } else {
        // ggml graph: matmul weight × im2col → [C, N]
        // patch_embed_w: [PS, PS, 3, C] in ggml → reshape to [patch_dim, C]
        size_t buf_sz = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
        ggml_init_params eip{buf_sz, nullptr, true};
        ggml_context *eg = ggml_init(eip);

        ggml_tensor *w = ggml_reshape_2d(eg, ctx.m.patch_embed_w, patch_dim, C);
        ggml_tensor *inp = ggml_new_tensor_2d(eg, GGML_TYPE_F32, patch_dim, N);
        ggml_set_name(inp, "im2col"); ggml_set_input(inp);
        ggml_tensor *out = ggml_mul_mat(eg, w, inp);  // [C, N]
        // Add bias if present
        if (ctx.m.patch_embed_b) {
            out = ggml_add(eg, out, ctx.m.patch_embed_b);
        }
        ggml_set_name(out, "pe_out"); ggml_set_output(out);

        ggml_cgraph *egf = ggml_new_graph(eg);
        ggml_build_forward_expand(egf, out);

        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, egf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(egf, "im2col"),
                                im2col.data(), 0, N * patch_dim * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, egf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(egf, "pe_out"),
                                hidden.data(), 0, N * C * sizeof(float));
        ggml_free(eg);
    }

    // Add position embedding
    auto pos = to_f32(ctx.m.pos_embed);
    if (!pos.empty())
        for (int i = 0; i < N * C; i++) hidden[i] += pos[i];

    if (ctx.verbosity >= 2)
        fprintf(stderr, "got_ocr: patch_embed done (%d, %d)\n", N, C);

    // Dequant LN weights for windowed layers
    auto ln1_ws = std::vector<std::vector<float>>(v.depth);
    auto ln1_bs = std::vector<std::vector<float>>(v.depth);
    for (uint32_t li = 0; li < v.depth; li++) {
        if (!ctx.m.vis_blocks[li].is_global) {
            ln1_ws[li] = to_f32(ctx.m.vis_blocks[li].ln1_w);
            ln1_bs[li] = to_f32(ctx.m.vis_blocks[li].ln1_b);
        }
    }

    auto t_start = std::chrono::steady_clock::now();

    // Per-layer ggml graph
    for (uint32_t li = 0; li < v.depth; li++) {
        auto &layer = ctx.m.vis_blocks[li];
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

        // For windowed: LN1 on CPU before partition
        bool skip_ln1 = !is_global;
        std::vector<float> ln1_hidden;
        if (skip_ln1) {
            ln1_hidden.resize(N * C);
            for (int n = 0; n < N; n++)
                layernorm_cpu(hidden.data() + n * C, ln1_hidden.data() + n * C, C,
                              ln1_ws[li].data(), ln1_bs[li].data(), 1e-6f);
        }

        std::vector<float> graph_input;
        std::vector<float> residual_input;
        if (is_global) {
            graph_input.assign(hidden.begin(), hidden.end());
        } else {
            graph_input.resize(T * C, 0.0f);
            window_partition(ln1_hidden.data(), graph_input.data(), nP, ws, C);
            residual_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), residual_input.data(), nP, ws, C);
        }

        // RPE tables
        std::vector<float> rp_h_ggml(aH * aH * hd);
        std::vector<float> rp_w_ggml(aW * aW * hd);
        reformat_rp_table(ctx.rp_h_per_layer[li].data(), rp_h_ggml.data(), aH, hd);
        reformat_rp_table(ctx.rp_w_per_layer[li].data(), rp_w_ggml.data(), aW, hd);

        // Build graph
        size_t meta_size = 8 * 1024 * 1024;
        std::vector<uint8_t> meta_buf(meta_size);
        ggml_init_params ip = { meta_size, meta_buf.data(), true };
        ggml_context* gc = ggml_init(ip);

        ggml_cgraph* gf = build_vis_layer_graph(gc, &ctx, li, C, T, aH, aW, nW, n_heads,
                                                 skip_ln1);

        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);

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

        ggml_backend_sched_graph_compute(ctx.sched, gf);

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

        // Per-layer diff comparison (must happen here, before hidden is overwritten)
        if (!ctx.diff_ref_path.empty()) {
            char name[64];
            snprintf(name, sizeof(name), "vis_layer_%d", li);
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(name)) {
                auto r = ref.compare(name, hidden.data(), N * C);
                fprintf(stderr, "  %s: cos_min=%.6f max_abs=%.6f %s\n",
                        name, r.cos_min, r.max_abs,
                        r.cos_min >= 0.999 ? "PASS" : "FAIL");
            }
        }

        if (ctx.verbosity >= 2)
            fprintf(stderr, "got_ocr: vis_layer_%d done (%s, T=%d)\n",
                    li, is_global ? "global" : "window", T);
    }

    auto t_end = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    if (ctx.verbosity >= 1)
        fprintf(stderr, "got_ocr: ViT done %.0f ms (%d layers)\n", ms, v.depth);
    if (bench) {
        auto vit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t_end - t_vis_total).count();
        fprintf(stderr, "[got_ocr-bench] vision_encoder: %lldms\n", (long long)vit_ms);
    }

    // ── Neck + Downsample + Projector ────────────────────────────
    // Gated: CRISPEMBED_GOT_OCR_SCALAR_NECK=1 for CPU-scalar fallback
    int nC = (int)v.neck_out_channels;
    int ds1_out_ch = 512;
    int ds1_H = (nP + 2 * 1 - 3) / 2 + 1;  // 32
    int ds1_W = ds1_H;
    int ds2_out_ch = 1024;
    int ds2_H = (ds1_H + 2 * 1 - 3) / 2 + 1;  // 16
    int ds2_W = ds2_H;
    int n_vis_tokens = ds2_H * ds2_W;  // 256
    int vis_D = ds2_out_ch;             // 1024
    std::vector<float> proj_out(n_vis_tokens * vis_D);

    static const bool scalar_neck = (std::getenv("CRISPEMBED_GOT_OCR_SCALAR_NECK") != nullptr);

    // Permute to CHW: [N, C] = [nP*nP, 768] → [nP, nP, 768] in ggml = (W, H, C)
    std::vector<float> chw(C * nP * nP);
    for (int tok = 0; tok < N; tok++) {
        int y = tok / nP, x = tok % nP;
        for (int c = 0; c < C; c++)
            chw[c * nP * nP + y * nP + x] = hidden[tok * C + c];
    }

    if (scalar_neck) {
        // ── Scalar CPU fallback ──
        auto nc1_w = to_f32(ctx.m.neck_conv1_w);
        std::vector<float> neck1(nC * nP * nP);
        conv2d_cpu(chw.data(), neck1.data(), nc1_w.data(), nullptr,
                   C, nC, nP, nP, 1, 1, 1, 0);
        auto nln1_w = to_f32(ctx.m.neck_ln1_w);
        auto nln1_b = to_f32(ctx.m.neck_ln1_b);
        std::vector<float> neck1_ln(nC * nP * nP);
        layernorm2d_cpu(neck1.data(), neck1_ln.data(), nC, nP, nP,
                        nln1_w.data(), nln1_b.data(), 1e-6f);
        auto nc2_w = to_f32(ctx.m.neck_conv2_w);
        std::vector<float> neck2(nC * nP * nP);
        conv2d_cpu(neck1_ln.data(), neck2.data(), nc2_w.data(), nullptr,
                   nC, nC, nP, nP, 3, 3, 1, 1);
        auto nln2_w = to_f32(ctx.m.neck_ln2_w);
        auto nln2_b = to_f32(ctx.m.neck_ln2_b);
        std::vector<float> neck2_ln(nC * nP * nP);
        layernorm2d_cpu(neck2.data(), neck2_ln.data(), nC, nP, nP,
                        nln2_w.data(), nln2_b.data(), 1e-6f);
        auto n2_w = to_f32(ctx.m.net_2_w);
        std::vector<float> ds1(ds1_out_ch * ds1_H * ds1_W);
        conv2d_cpu(neck2_ln.data(), ds1.data(), n2_w.data(), nullptr,
                   nC, ds1_out_ch, nP, nP, 3, 3, 2, 1);
        auto n3_w = to_f32(ctx.m.net_3_w);
        std::vector<float> ds2(ds2_out_ch * ds2_H * ds2_W);
        conv2d_cpu(ds1.data(), ds2.data(), n3_w.data(), nullptr,
                   ds1_out_ch, ds2_out_ch, ds1_H, ds1_W, 3, 3, 2, 1);
        std::vector<float> vis_flat(n_vis_tokens * vis_D);
        for (int tok = 0; tok < n_vis_tokens; tok++) {
            int y = tok / ds2_W, x = tok % ds2_W;
            for (int c = 0; c < vis_D; c++)
                vis_flat[tok * vis_D + c] = ds2[c * ds2_H * ds2_W + y * ds2_W + x];
        }
        auto pw = to_f32(ctx.m.projector_w);
        auto pb = to_f32(ctx.m.projector_b);
        for (int tok = 0; tok < n_vis_tokens; tok++)
            linear_cpu(vis_flat.data() + tok * vis_D,
                       proj_out.data() + tok * vis_D,
                       vis_D, vis_D, pw.data(),
                       pb.empty() ? nullptr : pb.data());
    } else {
        // ── ggml graph: conv + LN2d + downsample + projector ──
        // Build a single graph for the entire neck+projector pipeline.
        // LN2d = permute to (C,W,H) → ggml_norm along ne[0]=C → mul+add → permute back.
        auto prep_conv_w = [](ggml_context *g, ggml_tensor *w, int IC, int KH, int KW) -> ggml_tensor* {
            // Dequantize BEFORE reshaping. A quantized (e.g. q8_0) weight reshaped
            // to ne0=KW (1 for a 1x1 conv) breaks the block alignment and aborts
            // the Metal CPY used by ggml_cast (ne00 % blck_size != 0). Casting the
            // un-reshaped weight (ne0 = a multiple of the block) avoids that.
            if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
            if (ggml_n_dims(w) <= 2) {
                int64_t OC = w->ne[1];
                w = ggml_reshape_4d(g, w, KW, KH, IC, OC);
            }
            return w;
        };
        auto g_ln2d = [](ggml_context *g, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b, float eps) -> ggml_tensor* {
            // x: (W, H, C). LN2d normalizes along C for each (w,h).
            // Permute to (C, W, H) → norm along ne[0]=C → mul weight → add bias → permute back
            int W_ = (int)x->ne[0], H_ = (int)x->ne[1], C_ = (int)x->ne[2];
            // Bring channels to ne[0] so ggml_norm normalizes along C and the
            // (C,) weight/bias broadcast correctly. permute(1,2,0,3): src axis2
            // (C) -> dest0, src axis0 (W) -> dest1, src axis1 (H) -> dest2.
            ggml_tensor *xp = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));  // (W,H,C) -> (C, W, H)
            xp = ggml_norm(g, xp, eps);
            // Cast LN weight/bias to F32: they are stored F16 in the GGUF, and a
            // binary op with an F32 tensor + F16 operand has no CPU kernel
            // ((f32,f16) is the one unsupported combo) — aborts under
            // GOT_OCR_FORCE_CPU and is implicit-converted on Metal. Mirrors g_ln().
            if (w) xp = ggml_mul(g, xp, ensure_f32(g, w));  // w is (C,) broadcast along ne[1,2]
            if (b) xp = ggml_add(g, xp, ensure_f32(g, b));
            return ggml_cont(g, ggml_permute(g, xp, 2, 0, 1, 3));  // (C,W,H) -> back to (W, H, C)
        };

        size_t buf_sz = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(512, false);
        ggml_init_params nip{buf_sz, nullptr, true};
        ggml_context *ng = ggml_init(nip);

        // Input: CHW as (W, H, C) in ggml
        ggml_tensor *x = ggml_new_tensor_3d(ng, GGML_TYPE_F32, nP, nP, C);
        ggml_set_name(x, "neck_in"); ggml_set_input(x);

        // Conv1: 1×1, 768→256
        x = ggml_conv_2d_direct(ng, prep_conv_w(ng, ctx.m.neck_conv1_w, C, 1, 1),
                                 x, 1, 1, 0, 0, 1, 1);
        // LN2d 1
        x = g_ln2d(ng, x, ctx.m.neck_ln1_w, ctx.m.neck_ln1_b, 1e-6f);
        // Conv2: 3×3, 256→256, pad=1
        x = ggml_conv_2d_direct(ng, prep_conv_w(ng, ctx.m.neck_conv2_w, nC, 3, 3),
                                 x, 1, 1, 1, 1, 1, 1);
        // LN2d 2
        x = g_ln2d(ng, x, ctx.m.neck_ln2_w, ctx.m.neck_ln2_b, 1e-6f);
        // Downsample conv1: 3×3, 256→512, stride=2, pad=1
        x = ggml_conv_2d_direct(ng, prep_conv_w(ng, ctx.m.net_2_w, nC, 3, 3),
                                 x, 2, 2, 1, 1, 1, 1);
        // Downsample conv2: 3×3, 512→1024, stride=2, pad=1
        x = ggml_conv_2d_direct(ng, prep_conv_w(ng, ctx.m.net_3_w, ds1_out_ch, 3, 3),
                                 x, 2, 2, 1, 1, 1, 1);
        // x is now (ds2_W, ds2_H, 1024) in ggml

        // Flatten + permute to (vis_D, n_vis_tokens) = (1024, 256)
        // CHW (W,H,C) → (C, W*H) → this is what ggml_reshape_2d gives us directly
        // Bring channels to ne[0] then flatten spatial into tokens. ggml input is
        // (W,H,C); permute(1,2,0,3) -> (C,W,H), and reshape_2d(C, W*H) then yields
        // token = h*W + w, matching the numpy reference (x_chw.reshape(C,-1).T).
        // NOTE: the previous (2,0,1,3) produced (H,C,W), scrambling the projector
        // input — that was the got-ocr2 garbage-output bug.
        x = ggml_cont(ng, ggml_permute(ng, x, 1, 2, 0, 3));  // (W,H,C) → (C, W, H)
        x = ggml_reshape_2d(ng, x, vis_D, n_vis_tokens);      // (1024, 256) = (channel, token)

        // Projector: Linear(1024, 1024) with bias
        // projector_w is [1024, 1024] in ggml. mul_mat(w, x) → [1024, 256]
        x = ggml_mul_mat(ng, ctx.m.projector_w, x);
        if (ctx.m.projector_b)
            x = ggml_add(ng, x, ctx.m.projector_b);

        ggml_set_name(x, "proj_out"); ggml_set_output(x);

        ggml_cgraph *ngf = ggml_new_graph_custom(ng, 512, false);
        ggml_build_forward_expand(ngf, x);

        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, ngf)) {
            fprintf(stderr, "got_ocr: neck graph alloc failed, falling back to scalar\n");
            ggml_free(ng);
            // Fall through to error — shouldn't happen
            return false;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(ngf, "neck_in"),
                                chw.data(), 0, C * nP * nP * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, ngf);

        // Read output: (vis_D, n_vis_tokens) = (1024, 256)
        ggml_backend_tensor_get(ggml_graph_get_tensor(ngf, "proj_out"),
                                proj_out.data(), 0, n_vis_tokens * vis_D * sizeof(float));
        ggml_free(ng);
    }

    // Diff: projector output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            if (ref.has("vis_proj_output")) {
                auto r = ref.compare("vis_proj_output", proj_out.data(), n_vis_tokens * vis_D);
                fprintf(stderr, "  vis_proj_output: cos_min=%.6f max_abs=%.6f %s\n",
                        r.cos_min, r.max_abs,
                        r.cos_min >= 0.999 ? "PASS" : "FAIL");
            }
            // Intermediate diff comparisons only available in scalar neck path
            // (ggml path fuses all ops into one graph)
        }
    }

    // Output
    out.hidden = (float *)malloc(n_vis_tokens * vis_D * sizeof(float));
    memcpy(out.hidden, proj_out.data(), n_vis_tokens * vis_D * sizeof(float));
    out.n_tokens = n_vis_tokens;
    out.hidden_dim = vis_D;

    if (bench) {
        auto np_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_end).count();
        fprintf(stderr, "[got_ocr-bench] neck_projector: %lldms\n", (long long)np_ms);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

std::string got_ocr::tokenizer::decode(const int32_t *ids, int n) const {
    std::string result;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == eos_id) continue;
        if (id < 0 || id >= vocab_size) continue;
        const auto &piece = id_to_piece[id];
        // Skip special tokens
        if (!piece.empty() && piece[0] == '<' && piece.back() == '>') continue;
        result += piece;
    }
    return result;
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

static bool alloc_kv_cache(got_ocr::context &ctx, int max_seq) {
    auto &l = ctx.m.lhp;
    if (ctx.kvc.allocated && ctx.kvc.max_seq >= max_seq) {
        ctx.kvc.n_past = 0;
        // Clear
        ggml_backend_buffer_clear(ctx.kvc.buf, 0);
        return true;
    }
    if (ctx.kvc.buf) ggml_backend_buffer_free(ctx.kvc.buf);
    if (ctx.kvc.ctx) ggml_free(ctx.kvc.ctx);

    size_t ctx_size = 2 * ggml_tensor_overhead() + 1024;
    ggml_init_params ip = { ctx_size, nullptr, true };
    ctx.kvc.ctx = ggml_init(ip);

    int hd = (int)l.head_dim;
    int nkv = (int)l.num_key_value_heads;
    int nl = (int)l.num_hidden_layers;
    ctx.kvc.k = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16, hd, max_seq, nkv, nl);
    ctx.kvc.v = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16, hd, max_seq, nkv, nl);

    ctx.kvc.buf = ggml_backend_alloc_ctx_tensors(ctx.kvc.ctx, ctx.backend);
    if (!ctx.kvc.buf) return false;
    ggml_backend_buffer_clear(ctx.kvc.buf, 0);

    ctx.kvc.max_seq = max_seq;
    ctx.kvc.n_past = 0;
    ctx.kvc.allocated = true;
    return true;
}

// ---------------------------------------------------------------------------
// LLM graph
// ---------------------------------------------------------------------------

struct llm_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *token_in = nullptr;
    ggml_tensor *pos_in = nullptr;
    ggml_tensor *mask_in = nullptr;
    ggml_tensor *img_embeds = nullptr;
    ggml_tensor *splice_mask = nullptr;
    ggml_tensor *output = nullptr;
    ggml_tensor *logits_out = nullptr;
    std::vector<ggml_tensor *> layer_outputs;
};

static llm_graph build_llm_graph(got_ocr::context &ctx, int n_tokens, int n_past,
                                  bool use_kv_cache) {
    auto &m = ctx.m;
    auto &l = m.lhp;
    int D = (int)l.hidden_size;
    int T = n_tokens;
    int Lk = use_kv_cache ? (n_past + T) : T;
    int n_layers = (int)l.num_hidden_layers;
    int nh = (int)l.num_attention_heads;
    int nkv = (int)l.num_key_value_heads;
    int hd = (int)l.head_dim;
    float eps = l.rms_norm_eps;

    int tpl = use_kv_cache ? 80 : 60;
    size_t meta_size = ((size_t)n_layers * tpl + 300) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(32768, false);

    llm_graph lg;
    ggml_init_params ip = { meta_size, ctx.compute_meta.data(), true };
    lg.gctx = ggml_init(ip);
    auto *g = lg.gctx;

    lg.gf = ggml_new_graph_custom(g, 32768, false);

    // Token input
    lg.token_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(lg.token_in, "token_ids");
    ggml_set_input(lg.token_in);

    // Embedding
    ggml_tensor *x = ggml_get_rows(g, m.embed_tokens, lg.token_in);

    // Splice (prefill only)
    if (n_past == 0) {
        lg.img_embeds = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(lg.img_embeds, "img_embeds");
        ggml_set_input(lg.img_embeds);

        lg.splice_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(lg.splice_mask, "splice_mask");
        ggml_set_input(lg.splice_mask);


        x = ggml_add(g, ggml_mul(g, x, lg.splice_mask), lg.img_embeds);
    }

    // Position IDs (standard 1D RoPE)
    lg.pos_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(lg.pos_in, "pos_ids");
    ggml_set_input(lg.pos_in);

    // Causal mask
    lg.mask_in = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(lg.mask_in, "mask");
    ggml_set_input(lg.mask_in);

    // Transformer layers
    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {

        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    for (int i = 0; i < n_layers; i++) {
        auto &ly = m.llm_layers[i];

        // Pre-norm
        ggml_tensor *h = rmsnorm(x, ly.input_layernorm_w);

        // Q/K/V projections
        ggml_tensor *Q = g_linear(g, h, ly.q_w, ly.q_b);
        ggml_tensor *K = g_linear(g, h, ly.k_w, ly.k_b);
        ggml_tensor *V = g_linear(g, h, ly.v_w, ly.v_b);

        // Reshape for RoPE: [D, T] → [hd, nh, T]
        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K = ggml_reshape_3d(g, K, hd, nkv, T);
        V = ggml_reshape_3d(g, V, hd, nkv, T);

        // Standard RoPE
        Q = ggml_rope_ext(g, Q, lg.pos_in, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, 0, l.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, lg.pos_in, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, 0, l.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor *Kfull, *Vfull;

        if (use_kv_cache) {
            // Permute for cache: [hd, nh, T] → [hd, T, nh, 1]
            ggml_tensor *K_perm = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
            K_perm = ggml_reshape_4d(g, K_perm, hd, T, nkv, 1);
            ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
            V_perm = ggml_reshape_4d(g, V_perm, hd, T, nkv, 1);

            // Write to cache
            size_t k_nb1 = ctx.kvc.k->nb[1];
            size_t k_nb3 = ctx.kvc.k->nb[3];
            ggml_tensor *k_view = ggml_view_4d(g, ctx.kvc.k,
                hd, T, nkv, 1,
                ctx.kvc.k->nb[1], ctx.kvc.k->nb[2], ctx.kvc.k->nb[3],
                (size_t)i * k_nb3 + (size_t)n_past * k_nb1);
            ggml_build_forward_expand(lg.gf, ggml_cpy(g, K_perm, k_view));

            size_t v_nb1 = ctx.kvc.v->nb[1];
            size_t v_nb3 = ctx.kvc.v->nb[3];
            ggml_tensor *v_view = ggml_view_4d(g, ctx.kvc.v,
                hd, T, nkv, 1,
                ctx.kvc.v->nb[1], ctx.kvc.v->nb[2], ctx.kvc.v->nb[3],
                (size_t)i * v_nb3 + (size_t)n_past * v_nb1);
            ggml_build_forward_expand(lg.gf, ggml_cpy(g, V_perm, v_view));

            // Read full cache
            Kfull = ggml_view_3d(g, ctx.kvc.k,
                hd, Lk, nkv,
                ctx.kvc.k->nb[1], ctx.kvc.k->nb[2],
                (size_t)i * k_nb3);
            Vfull = ggml_view_3d(g, ctx.kvc.v,
                hd, Lk, nkv,
                ctx.kvc.v->nb[1], ctx.kvc.v->nb[2],
                (size_t)i * v_nb3);
        } else {
            // No cache: just permute
            Kfull = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
            Vfull = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
        }

        // flash_attn_ext handles GQA natively; GOT-OCR2 is MHA (nh==nkv)

        // Flash attention
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, lg.mask_in,
                                                 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection
        attn = ggml_mul_mat(g, ly.o_w, attn);

        // Residual
        x = ggml_add(g, x, attn);

        // FFN
        h = rmsnorm(x, ly.post_attention_layernorm_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);

        ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
        x = ggml_add(g, x, ffn);

        ggml_set_name(x, "layer_out");
        ggml_set_output(x);
        lg.layer_outputs.push_back(x);
    }

    // Final norm
    x = rmsnorm(x, m.output_norm_w);
    ggml_set_name(x, "final_norm");
    ggml_set_output(x);
    lg.output = x;

    // LM head: tied to embed_tokens if lm_head_w is null
    ggml_tensor *head_w = m.lm_head_w ? m.lm_head_w : m.embed_tokens;
    ggml_tensor *logits = ggml_mul_mat(g, head_w, x);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    lg.logits_out = logits;

    ggml_build_forward_expand(lg.gf, logits);
    return lg;
}

// ---------------------------------------------------------------------------
// LLM forward (parity test — uncached, no splice)
// ---------------------------------------------------------------------------

bool got_ocr::run_llm_forward(context &ctx, const int32_t *token_ids, int n_tokens,
                               llm_result &out) {
    int T = n_tokens;
    auto &l = ctx.m.lhp;
    int D = (int)l.hidden_size;

    llm_graph lg = build_llm_graph(ctx, T, 0, false);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        fprintf(stderr, "got_ocr: llm graph alloc failed\n");
        ggml_free(lg.gctx);
        return false;
    }

    // Token IDs
    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // Position IDs (sequential)
    std::vector<int32_t> pos_data(T);
    for (int j = 0; j < T; j++) pos_data[j] = j;
    ggml_backend_tensor_set(lg.pos_in, pos_data.data(), 0, T * sizeof(int32_t));

    // Causal mask
    std::vector<ggml_fp16_t> mask_data(T * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < T; ki++)
            mask_data[qi * T + ki] = ggml_fp32_to_fp16(ki > qi ? -INFINITY : 0.0f);
    ggml_backend_tensor_set(lg.mask_in, mask_data.data(), 0, T * T * sizeof(ggml_fp16_t));

    // Splice disabled: mask=1.0 (keep text), embeds=0.0
    {
        std::vector<float> ones(D * T, 1.0f);
        ggml_backend_tensor_set(lg.splice_mask, ones.data(), 0, D * T * sizeof(float));
        std::vector<float> zeros(D * T, 0.0f);
        ggml_backend_tensor_set(lg.img_embeds, zeros.data(), 0, D * T * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    // Read output
    out.hidden = (float *)malloc(D * T * sizeof(float));
    ggml_backend_tensor_get(lg.output, out.hidden, 0, D * T * sizeof(float));
    out.n_tokens = T;
    out.hidden_dim = D;
    out.vocab_size = (int)l.vocab_size;

    int V = (int)l.vocab_size;
    out.logits = (float *)malloc(V * T * sizeof(float));
    ggml_backend_tensor_get(lg.logits_out, out.logits, 0, V * T * sizeof(float));

    // Diff
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            if (ref.has("llm_embed")) {
                // Can't easily compare embed since we read final norm output
            }
            for (int i = 0; i < (int)lg.layer_outputs.size(); i++) {
                char name[64];
                snprintf(name, sizeof(name), "llm_layer_%d", i);
                if (ref.has(name)) {
                    std::vector<float> layer_out(D * T);
                    ggml_backend_tensor_get(lg.layer_outputs[i], layer_out.data(),
                                            0, D * T * sizeof(float));
                    auto r = ref.compare(name, layer_out.data(), D * T);
                    fprintf(stderr, "  %s: cos_min=%.6f max_abs=%.6f %s\n",
                            name, r.cos_min, r.max_abs,
                            r.cos_min >= 0.999 ? "PASS" : "FAIL");
                }
            }
        }
    }

    ggml_free(lg.gctx);
    return true;
}

// ---------------------------------------------------------------------------
// Cached step helper
// ---------------------------------------------------------------------------

struct splice_data {
    std::map<int, int> token_to_image;
};

static bool run_cached_step(got_ocr::context &ctx,
                             const int32_t *token_ids, int n_tokens, int n_past,
                             std::vector<float> &last_logits_out,
                             const splice_data *sd = nullptr) {
    auto &l = ctx.m.lhp;
    int T = n_tokens;
    int D = (int)l.hidden_size;
    int V = (int)l.vocab_size;

    llm_graph lg = build_llm_graph(ctx, T, n_past, true);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) return false;

    // Token IDs
    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // Position IDs
    std::vector<int32_t> pos_data(T);
    for (int j = 0; j < T; j++) pos_data[j] = n_past + j;
    ggml_backend_tensor_set(lg.pos_in, pos_data.data(), 0, T * sizeof(int32_t));

    // Causal mask
    int Lk = n_past + T;
    std::vector<ggml_fp16_t> mask_data(Lk * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < Lk; ki++)
            mask_data[qi * Lk + ki] =
                ggml_fp32_to_fp16(ki > n_past + qi ? -INFINITY : 0.0f);
    ggml_backend_tensor_set(lg.mask_in, mask_data.data(), 0, Lk * T * sizeof(ggml_fp16_t));

    // Splice
    if (n_past == 0 && lg.img_embeds && lg.splice_mask) {
        std::vector<float> img_data(D * T, 0.0f);
        std::vector<float> mask_f(D * T, 1.0f);

        if (sd) {
            for (auto &[tok_pos, img_idx] : sd->token_to_image) {
                if (tok_pos < T) {
                    for (int d = 0; d < D; d++) {
                        mask_f[tok_pos * D + d] = 0.0f;
                        // img_data filled separately
                    }
                }
            }
        }

        ggml_backend_tensor_set(lg.splice_mask, mask_f.data(), 0, D * T * sizeof(float));
        ggml_backend_tensor_set(lg.img_embeds, img_data.data(), 0, D * T * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    // Read last token's logits
    last_logits_out.resize(V);
    size_t offset = (size_t)(T - 1) * V * sizeof(float);
    ggml_backend_tensor_get(lg.logits_out, last_logits_out.data(), offset, V * sizeof(float));

    ggml_free(lg.gctx);
    return true;
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

static int argmax_no_repeat_ngram(const float *logits, int V,
                                  const std::vector<int32_t> &hist, int ngram) {
    std::unordered_set<int> banned;
    const int k = ngram - 1;
    const int n = (int)hist.size();
    if (ngram > 1 && n >= k && k > 0) {
        for (int i = 0; i + k < n; i++) {
            bool match = true;
            for (int j = 0; j < k; j++) {
                if (hist[i + j] != hist[n - k + j]) { match = false; break; }
            }
            if (match) banned.insert(hist[i + k]);
        }
    }
    int best_id = -1;
    float best = -INFINITY;
    for (int v = 0; v < V; v++) {
        if (!banned.empty() && banned.count(v)) continue;
        if (logits[v] > best) { best = logits[v]; best_id = v; }
    }
    if (best_id < 0) {
        for (int v = 0; v < V; v++) if (logits[v] > best) { best = logits[v]; best_id = v; }
    }
    return best_id;
}

bool got_ocr::generate(context &ctx,
                       const float *image_embeds, int n_image_tokens, int embed_dim,
                       const int32_t *prompt_ids, int n_prompt,
                       int max_new_tokens,
                       generate_result &out) {
    const bool bench = ctx.bench;
    auto t_gen_total = std::chrono::steady_clock::now();
    auto &l = ctx.m.lhp;
    int D = (int)l.hidden_size;

    // KV cache
    int total_seq = n_prompt + max_new_tokens + 16;
    if (!alloc_kv_cache(ctx, total_seq)) return false;

    // Build splice mapping
    splice_data sd;
    for (int i = 0; i < n_prompt; i++) {
        if (prompt_ids[i] == (int32_t)l.image_token_id) {
            int img_idx = (int)sd.token_to_image.size();
            if (img_idx < n_image_tokens)
                sd.token_to_image[i] = img_idx;
        }
    }

    // Build image embed + splice mask for prefill
    // We need to set the img_embeds tensor data after the graph is allocated.
    // For simplicity, do host-side splice into prompt embeddings.
    // Actually — the splice happens in the graph. We need to pass image_embeds
    // via the img_embeds input tensor. Build the full img_data array.
    std::vector<float> img_data(D * n_prompt, 0.0f);
    std::vector<float> mask_f(D * n_prompt, 1.0f);
    for (auto &[tok_pos, img_idx] : sd.token_to_image) {
        for (int d = 0; d < D; d++) {
            mask_f[tok_pos * D + d] = 0.0f;
            img_data[tok_pos * D + d] = image_embeds[img_idx * embed_dim + d];
        }
    }

    // Prefill
    int T = n_prompt;
    auto lg = build_llm_graph(ctx, T, 0, true);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        ggml_free(lg.gctx);
        return false;
    }

    ggml_backend_tensor_set(lg.token_in, prompt_ids, 0, T * sizeof(int32_t));

    std::vector<int32_t> pos_data(T);
    for (int j = 0; j < T; j++) pos_data[j] = j;
    ggml_backend_tensor_set(lg.pos_in, pos_data.data(), 0, T * sizeof(int32_t));

    int Lk = T;
    std::vector<ggml_fp16_t> mask_data(Lk * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < Lk; ki++)
            mask_data[qi * Lk + ki] =
                ggml_fp32_to_fp16(ki > qi ? -INFINITY : 0.0f);
    ggml_backend_tensor_set(lg.mask_in, mask_data.data(), 0, Lk * T * sizeof(ggml_fp16_t));

    ggml_backend_tensor_set(lg.splice_mask, mask_f.data(), 0, D * T * sizeof(float));
    ggml_backend_tensor_set(lg.img_embeds, img_data.data(), 0, D * T * sizeof(float));

    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_gen_total).count();
        fprintf(stderr, "[got_ocr-bench] prefill: %lldms\n", (long long)ms);
    }

    int V = (int)l.vocab_size;
    std::vector<float> last_logits(V);
    size_t offset = (size_t)(T - 1) * V * sizeof(float);
    ggml_backend_tensor_get(lg.logits_out, last_logits.data(), offset, V * sizeof(float));
    ggml_free(lg.gctx);

    // Argmax with no-repeat-ngram blocking (ngram=3)
    const int no_repeat_ngram = 3;
    out.token_confidences.clear();
    int next_token = argmax_no_repeat_ngram(last_logits.data(), V, out.token_ids, no_repeat_ngram);

    // Confidence: numerically-stable softmax for winning token
    {
        float max_l = last_logits[next_token];
        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) sum_exp += expf(last_logits[v] - max_l);
        out.token_confidences.push_back(1.0f / sum_exp);
    }

    out.token_ids.push_back(next_token);
    ctx.kvc.n_past = T;

    if (ctx.verbosity >= 2) {
        fprintf(stderr, "got_ocr: prefill %d tokens, %zu image spliced, first=%d\n",
                T, sd.token_to_image.size(), next_token);
    }

    // Stop tokens: <|endoftext|> (eos) and <|im_end|> (151645)
    auto is_stop = [&](int tok_id) {
        return tok_id == (int)l.eos_token_id || tok_id == 151645;
    };

    if (is_stop(next_token)) {
        out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size());
        return true;
    }

    // Autoregressive decode
    long long decode_total_ms = 0;
    int decode_steps = 0;
    for (int step = 0; step < max_new_tokens - 1; step++) {
        auto t_step = std::chrono::steady_clock::now();
        int32_t tok = (int32_t)next_token;
        std::vector<float> logits;
        if (!run_cached_step(ctx, &tok, 1, ctx.kvc.n_past, logits))
            break;
        ctx.kvc.n_past += 1;

        next_token = argmax_no_repeat_ngram(logits.data(), (int)logits.size(),
                                             out.token_ids, no_repeat_ngram);

        {
            float max_l = logits[next_token];
            float sum_exp = 0.0f;
            for (int v = 0; v < (int)logits.size(); v++) sum_exp += expf(logits[v] - max_l);
            out.token_confidences.push_back(1.0f / sum_exp);
        }

        out.token_ids.push_back(next_token);

        if (bench) {
            auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_step).count();
            decode_total_ms += step_ms;
            decode_steps++;
            fprintf(stderr, "[got_ocr-bench] decode_step[%d]: %lldms\n", step + 1, (long long)step_ms);
        }

        if (is_stop(next_token)) break;
    }

    if (bench) {
        fprintf(stderr, "[got_ocr-bench] decode_total: %lldms (%d steps)\n", decode_total_ms, decode_steps);
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_gen_total).count();
        fprintf(stderr, "[got_ocr-bench] total: %lldms\n", (long long)total_ms);
    }

    out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size());
    return true;
}

// ---------------------------------------------------------------------------
// C ABI wrappers
// ---------------------------------------------------------------------------

struct got_ocr_context {
    got_ocr::context inner;
    std::string result;
    std::vector<float> char_confidences;
};

got_ocr_context * got_ocr_init(const char * model_path, int n_threads) {
    auto *c = new got_ocr_context;
    int verbosity = std::getenv("CRISPEMBED_GOT_OCR_DEBUG") ? 2 : 1;
    if (!got_ocr::load(c->inner, model_path, n_threads, verbosity)) {
        delete c;
        return nullptr;
    }
    return c;
}

void got_ocr_free(got_ocr_context * ctx) {
    if (!ctx) return;
    got_ocr::free_(ctx->inner);
    delete ctx;
}

const char * got_ocr_recognize_raw(got_ocr_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }

    auto &v = ctx->inner.m.vhp;
    auto &l = ctx->inner.m.lhp;
    int imgS = (int)v.image_size; // 1024

    // Resize to (imgS, imgS) with bilinear interpolation + CLIP normalize
    std::vector<float> pixels(3 * imgS * imgS);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < imgS; y++) {
            float fy = (float)y * h / imgS;
            int iy = std::min((int)fy, h - 1);
            for (int x = 0; x < imgS; x++) {
                float fx = (float)x * w / imgS;
                int ix = std::min((int)fx, w - 1);
                int ci = std::min(c, ch - 1);
                float val = (float)px[(iy * w + ix) * ch + ci] / 255.0f;
                pixels[c * imgS * imgS + y * imgS + x] =
                    (val - v.image_mean[c]) / v.image_std[c];
            }
        }
    }

    // Vision encode
    got_ocr::vision_result vr;
    if (!got_ocr::encode_vision(ctx->inner, pixels.data(), vr)) {
        fprintf(stderr, "got_ocr: vision encoding failed\n");
        if (out_len) *out_len = 0;
        return "";
    }

    // Build prompt with Qwen2 chat template:
    //   <|im_start|>system\n{system_msg}<|im_end|>\n
    //   <|im_start|>user\n<img><imgpad>*256</img>\nOCR: <|im_end|>\n
    //   <|im_start|>assistant\n
    int n_img_tokens = vr.n_tokens;
    const int32_t im_start = 151644;   // <|im_start|>
    const int32_t im_end   = 151645;   // <|im_end|>
    const int32_t newline  = 198;      // \n
    const int32_t img_open = (int32_t)l.image_start_token_id;  // <img>
    const int32_t img_close = (int32_t)l.image_end_token_id;   // </img>
    const int32_t imgpad   = (int32_t)l.image_token_id;        // <imgpad>
    int n_imgpad = (int)l.image_token_len;

    // System message tokens: "system" + \n + "You should follow the instructions carefully and explain your answers in detail."
    // Pre-encoded from Qwen2 tiktoken (these are stable BPE token IDs).
    const int32_t sys_role[] = {8948};
    const int32_t sys_text[] = {2610, 1265, 1795, 279, 11221, 15516, 323, 10339, 697, 11253, 304, 7716, 13};
    const int32_t user_role[] = {872};
    const int32_t ocr_task[] = {93495, 25, 220};  // "OCR: "
    const int32_t asst_role[] = {77091};

    std::vector<int32_t> prompt;
    prompt.reserve(n_imgpad + 50);

    // System turn (MPT/ChatML: no \n between <|im_end|> and <|im_start|>)
    prompt.push_back(im_start);
    prompt.insert(prompt.end(), std::begin(sys_role), std::end(sys_role));
    prompt.push_back(newline);
    prompt.insert(prompt.end(), std::begin(sys_text), std::end(sys_text));
    prompt.push_back(im_end);

    // User turn
    prompt.push_back(im_start);
    prompt.insert(prompt.end(), std::begin(user_role), std::end(user_role));
    prompt.push_back(newline);
    prompt.push_back(img_open);
    for (int i = 0; i < n_imgpad; i++) prompt.push_back(imgpad);
    prompt.push_back(img_close);
    prompt.push_back(newline);
    prompt.insert(prompt.end(), std::begin(ocr_task), std::end(ocr_task));
    prompt.push_back(im_end);

    // Assistant turn
    prompt.push_back(im_start);
    prompt.insert(prompt.end(), std::begin(asst_role), std::end(asst_role));
    prompt.push_back(newline);

    int prompt_len = (int)prompt.size();

    // Generate
    got_ocr::generate_result gen;
    bool ok = got_ocr::generate(ctx->inner,
        vr.hidden, n_img_tokens, (int)vr.hidden_dim,
        prompt.data(), prompt_len, 1024, gen);
    free(vr.hidden);

    if (!ok) {
        if (out_len) *out_len = 0;
        return "";
    }

    ctx->result = gen.text;
    ctx->char_confidences = std::move(gen.token_confidences);
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * got_ocr_recognize(got_ocr_context * ctx,
    const float * px, int w, int h, int * out_len) {
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }
    // Convert grayscale float [0,1] to RGB uint8
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, px[i] * 255.0f));
        rgb[i * 3 + 0] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    return got_ocr_recognize_raw(ctx, rgb.data(), w, h, 3, out_len);
}

const float * got_ocr_confidences(const got_ocr_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float got_ocr_mean_confidence(const got_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
