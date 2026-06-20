// lfm2_embed.cpp — LFM2.5 bidirectional encoder, CLS-pooled text embeddings.
//
// Same backbone graph as GLiNER-LFM (gliner_ner.cpp) without layer fuser /
// BiLSTM / GLiNER head.  Applies the embedding_norm RMSNorm after all layers
// (consistent with the GLiNER usage), extracts position-0 (CLS), L2-normalises.

#include "lfm2_embed.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/gguf_loader.h"
#include "core/bpe.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Hyperparameters
// ============================================================================

struct lfm2_hparams {
    uint32_t hidden_size  = 1024;
    uint32_t n_layers     = 16;
    uint32_t n_heads      = 16;
    uint32_t n_kv_heads   = 8;
    uint32_t head_dim     = 64;
    uint32_t ff_dim       = 4608;
    uint32_t conv_kernel  = 3;
    float    rope_theta   = 1000000.0f;
    float    norm_eps     = 1e-5f;
    std::string layer_types;  // e.g. "ccaccaccacacacac"
    uint32_t vocab_size   = 65536;
    uint32_t bos_id       = 1;
    uint32_t eos_id       = 7;
};

// ============================================================================
// Per-layer weights
// ============================================================================

struct lfm2_layer {
    ggml_tensor * operator_norm_w = nullptr;
    ggml_tensor * ffn_norm_w      = nullptr;
    ggml_tensor * ff_w1 = nullptr, * ff_w2 = nullptr, * ff_w3 = nullptr;
    bool is_attention = false;
    // Conv layers
    ggml_tensor * conv_conv_w     = nullptr;
    ggml_tensor * conv_in_proj_w  = nullptr;
    ggml_tensor * conv_out_proj_w = nullptr;
    // Attention layers
    ggml_tensor * attn_q_proj_w   = nullptr;
    ggml_tensor * attn_k_proj_w   = nullptr;
    ggml_tensor * attn_v_proj_w   = nullptr;
    ggml_tensor * attn_out_proj_w = nullptr;
    ggml_tensor * attn_q_ln_w    = nullptr;
    ggml_tensor * attn_k_ln_w    = nullptr;
};

// ============================================================================
// Model
// ============================================================================

struct lfm2_embed_model {
    lfm2_hparams hparams;
    ggml_tensor * embed_tokens_w   = nullptr;
    ggml_tensor * embedding_norm_w = nullptr;
    std::vector<lfm2_layer> layers;

    // ColBERT projection head: Linear(hidden→colbert_dim, no bias)
    ggml_tensor * colbert_proj_w = nullptr;
    int colbert_dim = 0;

    ggml_context       * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    // BPE tokenizer
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

// ============================================================================
// Context
// ============================================================================

// Bucket sequence length to reduce scheduler re-reserves (same as encoder path)
static int lfm2_bucket_seq_len(int T) {
    if (T <= 8)   return 8;
    if (T <= 16)  return 16;
    if (T <= 32)  return 32;
    if (T <= 64)  return 64;
    if (T <= 128) return 128;
    if (T <= 256) return 256;
    if (T <= 512) return 512;
    return T;
}

struct lfm2_embed_ctx {
    lfm2_embed_model model;
    ggml_backend_t   backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int reserved_T = 0;          // dense encode path bucket
    int reserved_T_colbert = 0;  // ColBERT path bucket
    std::vector<int32_t> pos_cache;
    bool bench = false;
};

// ============================================================================
// Load
// ============================================================================

lfm2_embed_ctx * lfm2_embed_load(const char * path, ggml_backend_t backend) {
    gguf_context * gctx = core_gguf::open_metadata(path);
    if (!gctx) {
        fprintf(stderr, "[lfm2_embed] failed to open GGUF: %s\n", path);
        return nullptr;
    }

    auto * ctx = new lfm2_embed_ctx;
    ctx->backend  = backend;
    ctx->bench = (std::getenv("CRISPEMBED_LFM2_EMBED_BENCH") != nullptr);
    auto & hp = ctx->model.hparams;

    hp.hidden_size = core_gguf::kv_u32(gctx, "lfm2.hidden_size", 1024);
    hp.n_layers    = core_gguf::kv_u32(gctx, "lfm2.n_layers",    16);
    hp.n_heads     = core_gguf::kv_u32(gctx, "lfm2.n_heads",     16);
    hp.n_kv_heads  = core_gguf::kv_u32(gctx, "lfm2.n_kv_heads",  8);
    hp.head_dim    = core_gguf::kv_u32(gctx, "lfm2.head_dim",    64);
    hp.ff_dim      = core_gguf::kv_u32(gctx, "lfm2.ff_dim",      4608);
    hp.conv_kernel = core_gguf::kv_u32(gctx, "lfm2.conv_kernel", 3);
    hp.rope_theta  = core_gguf::kv_f32(gctx, "lfm2.rope_theta",  1000000.0f);
    hp.norm_eps    = core_gguf::kv_f32(gctx, "lfm2.norm_eps",    1e-5f);
    hp.layer_types = core_gguf::kv_str(gctx, "lfm2.layer_types", "ccaccaccacacacac");
    hp.vocab_size  = core_gguf::kv_u32(gctx, "lfm2.vocab_size",  65536);
    hp.bos_id      = core_gguf::kv_u32(gctx, "tokenizer.ggml.bos_token_id", 1);
    hp.eos_id      = core_gguf::kv_u32(gctx, "tokenizer.ggml.eos_token_id", 7);

    // BPE vocabulary
    auto tokens_vec = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
    if (tokens_vec.empty()) {
        fprintf(stderr, "[lfm2_embed] no tokenizer tokens in GGUF\n");
        core_gguf::free_metadata(gctx);
        delete ctx;
        return nullptr;
    }
    for (size_t i = 0; i < tokens_vec.size(); i++)
        ctx->model.token_to_id[tokens_vec[i]] = (int32_t)i;

    // Merges: try array key first, then blob key
    {
        const int64_t mi = gguf_find_key(gctx, "tokenizer.ggml.merges");
        if (mi >= 0 && gguf_get_arr_type(gctx, mi) == GGUF_TYPE_STRING) {
            int nm = (int)gguf_get_arr_n(gctx, mi);
            int rank = 0;
            for (int i = 0; i < nm; i++) {
                std::string m = gguf_get_arr_str(gctx, mi, i);
                if (!m.empty()) ctx->model.merge_rank[m] = rank++;
            }
        } else {
            // Fallback: blob (space-separated entries, newline-delimited)
            std::string blob = core_gguf::kv_str(gctx, "tokenizer.merges_blob", "");
            int rank = 0;
            size_t pos = 0;
            while (pos < blob.size()) {
                size_t nl = blob.find('\n', pos);
                if (nl == std::string::npos) nl = blob.size();
                std::string m = blob.substr(pos, nl - pos);
                if (!m.empty()) ctx->model.merge_rank[m] = rank++;
                pos = nl + 1;
            }
        }
    }

    core_gguf::free_metadata(gctx);

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "lfm2", wl)) {
        fprintf(stderr, "[lfm2_embed] failed to load weights: %s\n", path);
        if (ctx->sched) ggml_backend_sched_free(ctx->sched);
        delete ctx;
        return nullptr;
    }
    ctx->model.ctx     = wl.ctx;
    ctx->model.buf     = wl.buf;
    ctx->model.tensors = wl.tensors;

    auto R = [&](const char * name) -> ggml_tensor * {
        return core_gguf::require(ctx->model.tensors, name, "lfm2_embed");
    };

    ctx->model.embed_tokens_w   = R("lfm.embed_tokens.weight");
    ctx->model.embedding_norm_w = R("lfm.embedding_norm.weight");

    ctx->model.layers.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        auto & l = ctx->model.layers[i];
        auto ln = [&](const char * suffix) {
            char name[128];
            snprintf(name, sizeof(name), "lfm.layers.%u.%s", i, suffix);
            return R(name);
        };
        l.operator_norm_w = ln("operator_norm.weight");
        l.ffn_norm_w      = ln("ffn_norm.weight");
        l.ff_w1 = ln("ff.w1.weight");
        l.ff_w2 = ln("ff.w2.weight");
        l.ff_w3 = ln("ff.w3.weight");
        l.is_attention = (i < hp.layer_types.size() && hp.layer_types[i] == 'a');
        if (l.is_attention) {
            l.attn_q_proj_w   = ln("attn.q_proj.weight");
            l.attn_k_proj_w   = ln("attn.k_proj.weight");
            l.attn_v_proj_w   = ln("attn.v_proj.weight");
            l.attn_out_proj_w = ln("attn.out_proj.weight");
            l.attn_q_ln_w     = ln("attn.q_layernorm.weight");
            l.attn_k_ln_w     = ln("attn.k_layernorm.weight");
        } else {
            l.conv_conv_w     = ln("conv.conv.weight");
            l.conv_in_proj_w  = ln("conv.in_proj.weight");
            l.conv_out_proj_w = ln("conv.out_proj.weight");
        }
    }

    // ColBERT projection head (optional — present in LFM2.5-ColBERT)
    ctx->model.colbert_proj_w = core_gguf::try_get(wl.tensors, "colbert.projection.weight");
    if (ctx->model.colbert_proj_w) {
        // Weight shape [colbert_dim, hidden] in PyTorch → ne[0]=hidden, ne[1]=colbert_dim in ggml
        ctx->model.colbert_dim = (int)ctx->model.colbert_proj_w->ne[1];
        fprintf(stderr, "[lfm2_embed] ColBERT head: %d → %d\n",
                hp.hidden_size, ctx->model.colbert_dim);
    }

    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 4096, false, false);
    if (!ctx->sched) {
        fprintf(stderr, "[lfm2_embed] failed to create backend scheduler\n");
        ggml_backend_buffer_free(ctx->model.buf);
        ggml_free(ctx->model.ctx);
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "[lfm2_embed] loaded: hidden=%u, layers=%u, heads=%u/%u, "
            "ff=%u, vocab=%u%s\n",
            hp.hidden_size, hp.n_layers, hp.n_heads, hp.n_kv_heads,
            hp.ff_dim, hp.vocab_size,
            ctx->model.colbert_dim > 0 ? ", ColBERT" : "");
    return ctx;
}

void lfm2_embed_free(lfm2_embed_ctx * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    // backend is owned by crispembed_context — do not free here
    delete ctx;
}

int lfm2_embed_n_embd(const lfm2_embed_ctx * ctx) {
    return ctx ? (int)ctx->model.hparams.hidden_size : 0;
}

// ============================================================================
// Tokenizer
// ============================================================================

static std::vector<int32_t> lfm2_tokenize(const lfm2_embed_model & m,
                                           const char * text) {
    // BPE-encode the text (whitespace pre-tokeniser, GPT-2 byte encoding)
    std::vector<int32_t> ids = core_bpe::tokenize_simple(
        m.token_to_id, m.merge_rank, std::string(text));

    // BOS prefix only (add_eos_token=False for this model)
    std::vector<int32_t> result;
    result.reserve(ids.size() + 1);
    result.push_back((int32_t)m.hparams.bos_id);
    for (int32_t id : ids) result.push_back(id);
    return result;
}

// ============================================================================
// Graph building blocks  (bidirectional LFM2 — matches gliner_ner.cpp exactly)
// ============================================================================

static ggml_tensor * lfm2_rms_norm(ggml_context * g, ggml_tensor * x,
                                    ggml_tensor * w, float eps) {
    // Metal ggml_mul requires src[1] to be F32; cast if stored as F16.
    if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
    return ggml_mul(g, ggml_rms_norm(g, x, eps), w);
}

static ggml_tensor * lfm2_swiglu(ggml_context * g, ggml_tensor * x,
                                  ggml_tensor * w1, ggml_tensor * w2,
                                  ggml_tensor * w3) {
    return ggml_mul_mat(g, w2,
               ggml_mul(g,
                   ggml_silu(g, ggml_mul_mat(g, w1, x)),
                   ggml_mul_mat(g, w3, x)));
}

// Bidirectional ShortConv (symmetric centre-padding, not causal).
static ggml_tensor * lfm2_short_conv(ggml_context * g, ggml_tensor * x,
                                      const lfm2_layer & w, int H, int T) {
    // in_proj: (H, T) → (3H, T)
    ggml_tensor * bcx = ggml_mul_mat(g, w.conv_in_proj_w, x);

    ggml_tensor * B  = ggml_cont(g, ggml_view_2d(g, bcx, H, T, bcx->nb[1], 0));
    ggml_tensor * C  = ggml_cont(g, ggml_view_2d(g, bcx, H, T, bcx->nb[1],
                                                   H * sizeof(float)));
    ggml_tensor * xi = ggml_cont(g, ggml_view_2d(g, bcx, H, T, bcx->nb[1],
                                                   2 * H * sizeof(float)));
    ggml_tensor * Bx = ggml_mul(g, ggml_cont(g, B), ggml_cont(g, xi));

    // Symmetric depthwise conv1d, kernel=3, pad=1 → T_out == T
    ggml_tensor * conv_w = ggml_cast(g, w.conv_conv_w, GGML_TYPE_F16);
    ggml_tensor * Bx_t   = ggml_cont(g, ggml_transpose(g, Bx));   // (T, H)
    ggml_tensor * co     = ggml_conv_1d_dw(g, conv_w, Bx_t, 1, 1, 1);
    int T_conv = (int)co->ne[0];
    if (T_conv > T)
        co = ggml_view_2d(g, co, T, H, co->nb[1], 0);
    co = ggml_cont(g, ggml_transpose(g, co));  // (H, T)

    ggml_tensor * y = ggml_mul(g, ggml_cont(g, C), ggml_cont(g, co));
    return ggml_mul_mat(g, w.conv_out_proj_w, y);
}

// Bidirectional GQA (no causal mask).
static ggml_tensor * lfm2_gqa(ggml_context * g, ggml_tensor * x,
                                const lfm2_layer & w,
                                int H, int nh, int nkv, int hd,
                                int T, float theta,
                                ggml_tensor * pos) {
    ggml_tensor * Q = ggml_mul_mat(g, w.attn_q_proj_w, x);
    ggml_tensor * K = ggml_mul_mat(g, w.attn_k_proj_w, x);
    ggml_tensor * V = ggml_mul_mat(g, w.attn_v_proj_w, x);

    Q = ggml_reshape_3d(g, Q, hd, nh,  T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);

    // Per-head QK RMSNorm — cast scale to F32 for Metal binary-op compatibility.
    auto f32 = [&](ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
    };
    Q = ggml_mul(g, ggml_rms_norm(g, Q, 1e-5f), f32(w.attn_q_ln_w));
    K = ggml_mul(g, ggml_rms_norm(g, K, 1e-5f), f32(w.attn_k_ln_w));

    Q = ggml_rope_ext(g, Q, pos, nullptr, hd,
                      GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos, nullptr, hd,
                      GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

    const float scale = 1.0f / sqrtf((float)hd);
    ggml_tensor * attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(g, attn, H, T);
    return ggml_mul_mat(g, w.attn_out_proj_w, attn);
}

// One LFM2 layer: norm → op → residual → norm → SwiGLU → residual.
static ggml_tensor * lfm2_layer_fwd(ggml_context * g, ggml_tensor * x,
                                     const lfm2_layer & w,
                                     int H, int nh, int nkv, int hd,
                                     int T, float eps, float theta,
                                     ggml_tensor * pos) {
    ggml_tensor * r = x;
    ggml_tensor * h = lfm2_rms_norm(g, x, w.operator_norm_w, eps);
    h = w.is_attention
        ? lfm2_gqa(g, h, w, H, nh, nkv, hd, T, theta, pos)
        : lfm2_short_conv(g, h, w, H, T);
    x = ggml_add(g, r, h);
    r = x;
    h = lfm2_rms_norm(g, x, w.ffn_norm_w, eps);
    h = lfm2_swiglu(g, h, w.ff_w1, w.ff_w2, w.ff_w3);
    return ggml_add(g, r, h);
}

// ============================================================================
// Encode
// ============================================================================

bool lfm2_embed_encode_to(lfm2_embed_ctx * ctx, const char * text, float * out) {
    if (!ctx || !text || !out) return false;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();
    auto t_tok0  = std::chrono::steady_clock::now();

    const auto & hp = ctx->model.hparams;
    const int  H   = (int)hp.hidden_size;
    const int  nh  = (int)hp.n_heads;
    const int  nkv = (int)hp.n_kv_heads;
    const int  hd  = (int)hp.head_dim;
    const float eps   = hp.norm_eps;
    const float theta = hp.rope_theta;

    std::vector<int32_t> ids = lfm2_tokenize(ctx->model, text);
    if (ids.empty()) return false;
    const int T = (int)ids.size();

    if (bench) {
        auto t_tok1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[lfm2_embed-bench] tokenize: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_tok1 - t_tok0).count());
    }

    // Build graph (no-alloc metadata context from a small heap buffer)
    // ~50 nodes/layer (ShortConv ~20 + GQA ~30, plus cast + cont nodes)
    const int max_nodes = 1024 + (int)hp.n_layers * 120;
    size_t meta_size = ggml_tensor_overhead() * (size_t)max_nodes
                     + ggml_graph_overhead_custom(max_nodes, false);
    struct ggml_init_params ip = { meta_size, /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    ggml_context * g = ggml_init(ip);
    if (!g) return false;

    // Input token IDs
    ggml_tensor * inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(inp, "input_ids");
    ggml_set_input(inp);

    // Embedding lookup
    ggml_tensor * cur = ggml_get_rows(g, ctx->model.embed_tokens_w, inp);

    ggml_tensor * pos = nullptr;
    if (hp.layer_types.find('a') != std::string::npos) {
        pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(pos, "positions");
        ggml_set_input(pos);
    }

    // Transformer layers
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = lfm2_layer_fwd(g, cur, ctx->model.layers[il],
                             H, nh, nkv, hd, T, eps, theta, pos);
    }

    // Final norm (embedding_norm applied after all layers, matching GLiNER usage)
    cur = lfm2_rms_norm(g, cur, ctx->model.embedding_norm_w, eps);

    // CLS token: column-0 of [H, T]
    ggml_tensor * cls = ggml_cont(g, ggml_view_1d(g, cur, H, 0));
    ggml_set_name(cls, "cls");
    ggml_set_output(cls);

    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_build_forward_expand(gf, cls);

    // Reserve scheduler for this T bucket (reuse across same-length inputs)
    const int T_bucket = lfm2_bucket_seq_len(T);
    if (ctx->reserved_T != T_bucket) {
        ggml_backend_sched_reserve(ctx->sched, gf);
        ctx->reserved_T = T_bucket;
        ctx->reserved_T_colbert = 0;  // invalidate ColBERT reservation
        // Rebuild graph for actual T (reserve used the bucket graph)
        ggml_free(g);
        g = ggml_init(ip);
        if (!g) return false;
        inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(inp, "input_ids"); ggml_set_input(inp);
        cur = ggml_get_rows(g, ctx->model.embed_tokens_w, inp);
        pos = nullptr;
        if (hp.layer_types.find('a') != std::string::npos) {
            pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
            ggml_set_name(pos, "positions"); ggml_set_input(pos);
        }
        for (uint32_t il = 0; il < hp.n_layers; il++)
            cur = lfm2_layer_fwd(g, cur, ctx->model.layers[il],
                                 H, nh, nkv, hd, T, eps, theta, pos);
        cur = lfm2_rms_norm(g, cur, ctx->model.embedding_norm_w, eps);
        cls = ggml_cont(g, ggml_view_1d(g, cur, H, 0));
        ggml_set_name(cls, "cls"); ggml_set_output(cls);
        gf = ggml_new_graph_custom(g, max_nodes, false);
        ggml_build_forward_expand(gf, cls);
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "[lfm2_embed] graph allocation failed (T=%d)\n", T);
        ggml_free(g);
        return false;
    }

    // Fill inputs (tensors are allocated — safe to set now)
    ggml_backend_tensor_set(inp, ids.data(), 0, T * sizeof(int32_t));
    if (pos) {
        ctx->pos_cache.resize(T);
        for (int i = 0; i < T; i++) ctx->pos_cache[i] = i;
        ggml_backend_tensor_set(pos, ctx->pos_cache.data(), 0, T * sizeof(int32_t));
    }

    {
        auto t_comp0 = std::chrono::steady_clock::now();
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        if (bench) {
            auto t_comp1 = std::chrono::steady_clock::now();
            fprintf(stderr, "[lfm2_embed-bench] graph compute: %.3f ms\n",
                    std::chrono::duration<double, std::milli>(t_comp1 - t_comp0).count());
        }
    }

    // Read CLS embedding
    auto t_post0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(cls, out, 0, H * sizeof(float));

    ggml_free(g);

    // L2 normalise
    float norm = 0.0f;
    for (int i = 0; i < H; i++) norm += out[i] * out[i];
    norm = sqrtf(std::max(norm, 1e-12f));
    for (int i = 0; i < H; i++) out[i] /= norm;

    if (bench) {
        auto t_post1 = std::chrono::steady_clock::now();
        auto t_total1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[lfm2_embed-bench] postprocess: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());
        fprintf(stderr, "[lfm2_embed-bench] total: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_total1 - t_total).count());
    }

    return true;
}

std::vector<float> lfm2_embed_encode(lfm2_embed_ctx * ctx, const char * text) {
    if (!ctx || !text) return {};
    const int H = (int)ctx->model.hparams.hidden_size;
    std::vector<float> out(H);
    if (!lfm2_embed_encode_to(ctx, text, out.data())) return {};
    return out;
}

int lfm2_embed_colbert_dim(const lfm2_embed_ctx * ctx) {
    return ctx ? ctx->model.colbert_dim : 0;
}

bool lfm2_embed_has_colbert(const lfm2_embed_ctx * ctx) {
    return ctx && ctx->model.colbert_dim > 0 && ctx->model.colbert_proj_w;
}

int lfm2_embed_encode_multivec(lfm2_embed_ctx * ctx, const char * text,
                                float * out, int max_tokens) {
    if (!ctx || !text || !out || !lfm2_embed_has_colbert(ctx)) return 0;

    const auto & hp = ctx->model.hparams;
    const int H = (int)hp.hidden_size;
    const int cd = ctx->model.colbert_dim;

    // Tokenize
    std::vector<int32_t> ids = lfm2_tokenize(ctx->model, text);
    int T = (int)ids.size();
    if (T <= 0) return 0;
    if (T > max_tokens) T = max_tokens;

    // Build graph — same as encode_to but output ALL tokens, not just CLS
    const int nh  = (int)hp.n_heads;
    const int nkv = (int)hp.n_kv_heads;
    const int hd  = H / nh;
    const float eps   = hp.norm_eps;
    const float theta = hp.rope_theta;
    const int max_nodes = 4096;

    ggml_init_params gp = {
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
        nullptr, true
    };
    ggml_context * g = ggml_init(gp);

    ggml_tensor * inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(inp, "input_ids"); ggml_set_input(inp);
    ggml_tensor * pos = nullptr;
    if (hp.layer_types.find('a') != std::string::npos) {
        pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(pos, "pos_ids"); ggml_set_input(pos);
    }

    // Token embedding
    ggml_tensor * cur = ggml_get_rows(g, ctx->model.embed_tokens_w, inp);

    // Encoder layers
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = lfm2_layer_fwd(g, cur, ctx->model.layers[il],
                             H, nh, nkv, hd, T, eps, theta, pos);
    }

    // Final norm
    cur = lfm2_rms_norm(g, cur, ctx->model.embedding_norm_w, eps);

    // ColBERT projection: [H, T] → matmul with proj [cd, H] → [cd, T]
    ggml_tensor * projected = ggml_mul_mat(g, ctx->model.colbert_proj_w, cur);
    ggml_set_name(projected, "colbert_out");
    ggml_set_output(projected);

    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_build_forward_expand(gf, projected);

    // Reserve scheduler for ColBERT bucket
    const int T_bucket = lfm2_bucket_seq_len(T);
    if (ctx->reserved_T_colbert != T_bucket) {
        ggml_backend_sched_reserve(ctx->sched, gf);
        ctx->reserved_T_colbert = T_bucket;
        ctx->reserved_T = 0;  // invalidate dense reservation
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "[lfm2_embed] ColBERT graph allocation failed (T=%d)\n", T);
        ggml_free(g);
        return 0;
    }

    // Fill inputs
    ggml_backend_tensor_set(inp, ids.data(), 0, T * sizeof(int32_t));
    if (pos) {
        ctx->pos_cache.resize(T);
        for (int i = 0; i < T; i++) ctx->pos_cache[i] = i;
        ggml_backend_tensor_set(pos, ctx->pos_cache.data(), 0, T * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read projected output: [cd, T] in ggml → read as [T, cd] row-major
    std::vector<float> raw(cd * T);
    ggml_backend_tensor_get(projected, raw.data(), 0, cd * T * sizeof(float));

    ggml_free(g);

    // Transpose from ggml [cd, T] (col-major per token) to [T, cd] row-major
    // and L2-normalize each token
    for (int t = 0; t < T; t++) {
        float norm = 0.0f;
        for (int d = 0; d < cd; d++) {
            float v = raw[d * T + t]; // ggml layout: fast dim is cd, stride T
            // Actually ggml [cd, T]: element [d, t] = data[t * cd + d] (ne[0]=cd is fast)
            // Wait — ggml_mul_mat output has ne[0] = cd (from proj), ne[1] = T
            // So data[t * cd + d] is correct for row t, column d
            out[t * cd + d] = raw[t * cd + d];
            norm += raw[t * cd + d] * raw[t * cd + d];
        }
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int d = 0; d < cd; d++) out[t * cd + d] /= norm;
    }

    return T;
}

// ============================================================================
// Dump mode — per-layer intermediate capture for crispembed_diff parity testing
// ============================================================================

std::vector<lfm2_dump_entry> lfm2_embed_encode_dump(lfm2_embed_ctx * ctx,
                                                      const char * text) {
    if (!ctx || !text) return {};

    const auto & hp = ctx->model.hparams;
    const int  H   = (int)hp.hidden_size;
    const int  nh  = (int)hp.n_heads;
    const int  nkv = (int)hp.n_kv_heads;
    const int  hd  = (int)hp.head_dim;
    const float eps   = hp.norm_eps;
    const float theta = hp.rope_theta;

    std::vector<int32_t> ids = lfm2_tokenize(ctx->model, text);
    if (ids.empty()) return {};
    const int T = (int)ids.size();

    // Build graph with extra output markers on every stage we want to dump.
    // max_nodes needs more headroom for the extra ggml_cont copies we'll add.
    const int max_nodes = 1024 + (int)hp.n_layers * 120;
    size_t meta_size = ggml_tensor_overhead() * (size_t)max_nodes
                     + ggml_graph_overhead_custom(max_nodes, false);
    struct ggml_init_params ip = { meta_size, nullptr, true };
    ggml_context * g = ggml_init(ip);
    if (!g) return {};

    ggml_tensor * inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(inp, "input_ids");
    ggml_set_input(inp);

    ggml_tensor * cur = ggml_get_rows(g, ctx->model.embed_tokens_w, inp);

    // post_embed: mark AFTER embedding lookup (shape: H x T in ggml = T rows of H)
    ggml_tensor * post_embed_out = ggml_cont(g, cur);
    ggml_set_name(post_embed_out, "post_embed");
    ggml_set_output(post_embed_out);
    cur = post_embed_out;

    // Per-layer outputs
    ggml_tensor * pos = nullptr;
    if (hp.layer_types.find('a') != std::string::npos) {
        pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(pos, "positions");
        ggml_set_input(pos);
    }

    std::vector<ggml_tensor *> layer_outs(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = lfm2_layer_fwd(g, cur, ctx->model.layers[il],
                             H, nh, nkv, hd, T, eps, theta, pos);
        layer_outs[il] = ggml_cont(g, cur);
        char lname[32];
        snprintf(lname, sizeof(lname), "layer_%u", il);
        ggml_set_name(layer_outs[il], lname);
        ggml_set_output(layer_outs[il]);
        cur = layer_outs[il];
    }

    // final_norm
    cur = lfm2_rms_norm(g, cur, ctx->model.embedding_norm_w, eps);
    ggml_tensor * final_norm_out = ggml_cont(g, cur);
    ggml_set_name(final_norm_out, "final_norm");
    ggml_set_output(final_norm_out);
    cur = final_norm_out;

    // cls_raw (position 0, before L2 norm)
    ggml_tensor * cls_raw = ggml_cont(g, ggml_view_1d(g, cur, H, 0));
    ggml_set_name(cls_raw, "cls_raw");
    ggml_set_output(cls_raw);

    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_build_forward_expand(gf, cls_raw);

    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "[lfm2_embed] dump graph alloc failed (T=%d)\n", T);
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(inp, ids.data(), 0, T * sizeof(int32_t));
    if (pos) {
        ctx->pos_cache.resize(T);
        for (int i = 0; i < T; i++) ctx->pos_cache[i] = i;
        ggml_backend_tensor_set(pos, ctx->pos_cache.data(), 0, T * sizeof(int32_t));
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // Collect results — all tensors are (H, T) in ggml = T rows of H in Python
    std::vector<lfm2_dump_entry> entries;
    auto collect2d = [&](ggml_tensor * t, const char * name) {
        lfm2_dump_entry e;
        e.name = name;
        e.H = H; e.T = T;
        e.data.resize((size_t)H * T);
        ggml_backend_tensor_get(t, e.data.data(), 0, H * T * sizeof(float));
        entries.push_back(std::move(e));
    };
    auto collect1d = [&](ggml_tensor * t, const char * name) {
        lfm2_dump_entry e;
        e.name = name;
        e.H = H; e.T = 1;
        e.data.resize(H);
        ggml_backend_tensor_get(t, e.data.data(), 0, H * sizeof(float));
        entries.push_back(std::move(e));
    };

    collect2d(post_embed_out, "post_embed");
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        char lname[32];
        snprintf(lname, sizeof(lname), "layer_%u", il);
        collect2d(layer_outs[il], lname);
    }
    collect2d(final_norm_out, "final_norm");

    // cls_raw
    collect1d(cls_raw, "cls_raw");

    // Also compute cls_norm (L2-normalized) from cls_raw
    {
        lfm2_dump_entry e;
        e.name = "cls_norm";
        e.H = H; e.T = 1;
        e.data = entries.back().data;  // copy cls_raw
        // Oops — cls_raw is the last, but let's find it safely
        for (auto & en : entries) {
            if (en.name == "cls_raw") { e.data = en.data; break; }
        }
        float n2 = 0.0f;
        for (float v : e.data) n2 += v * v;
        n2 = sqrtf(std::max(n2, 1e-12f));
        for (float & v : e.data) v /= n2;
        entries.push_back(std::move(e));
    }

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return entries;
}
