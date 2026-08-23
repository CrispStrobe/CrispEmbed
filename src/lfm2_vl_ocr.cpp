// lfm2_vl_ocr.cpp — LFM2.5-VL-3B vision-language OCR inference engine.
//
// Architecture (LiquidAI/LFM2.5-VL-3B):
//
// Vision encoder (SigLIP2 NaFlex, 400M):
//   patches → Linear(768, 1152) + learned pos embed
//   27 × pre-LayerNorm ViT block:
//     LayerNorm → QKV → bidirectional attention → residual
//     LayerNorm → GELU MLP (1152→4304→1152) → residual
//
// Projector:
//   pixel_unshuffle (2×) → Linear(4608, 2048) → GELU → Linear(2048, 2048)
//
// LLM decoder (LFM2.5, 2.6B, hybrid conv+attention):
//   embed_tokens(128000, 2048) → splice image_embeds at image_token positions
//   30 × pre-RMSNorm hybrid block:
//     Conv layers (22/30): in_proj(2048, 6144) → B*x gate → causal depthwise
//                          conv1d(k=3) → C gate → out_proj(2048, 2048) → residual
//     Attn layers (8/30):  QK RMSNorm → RoPE → GQA(32h/8kv) → residual
//     All layers:          RMSNorm → SwiGLU FFN (2048→10752→2048) → residual
//   embedding_norm(RMSNorm) → lm_head (tied) → greedy decode
//
// License: LFM-1.0 (revenue-capped; requires CRISPEMBED_ACCEPT_LFM_LICENSE=1)

#include "lfm2_vl_ocr.h"
#include "core/bpe.h"
#include "core/gguf_loader.h"
#include "core/env_gate.h"
#include "core/gpu_backend_pref.h"
#include "crispembed_diff.h"
#include "core/no_repeat_ngram.h"
#include "image_preprocess.h"
#include "imatrix.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Internal namespace
// ============================================================================

namespace lfm2_vl {

namespace {

using steady_clock = std::chrono::steady_clock;

static bool dbg() { return core_env::on("LFM2_VL_DBG"); }

static long long ms_since(steady_clock::time_point t0) {
    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
               steady_clock::now() - t0)
        .count();
}

// ============================================================================
// Hyperparameters
// ============================================================================

struct vision_hparams {
    uint32_t depth         = 27;
    uint32_t hidden_size   = 1152;
    uint32_t ff_size       = 4304;
    uint32_t num_heads     = 16;
    uint32_t head_dim      = 72;   // 1152 / 16
    uint32_t patch_size    = 16;
    uint32_t image_size    = 256;  // SigLIP2 base grid (16×16 patches)
    uint32_t tile_size     = 512;  // VL tile target for NaFlex resize
    float    norm_eps      = 1e-6f;
    float    image_mean[3] = { 0.5f, 0.5f, 0.5f };
    float    image_std[3]  = { 0.5f, 0.5f, 0.5f };
};

struct projector_hparams {
    uint32_t unshuffle_factor = 2;
    uint32_t in_dim  = 4608;  // hidden * factor^2
    uint32_t mid_dim = 2048;
    uint32_t out_dim = 2048;
};

struct llm_hparams {
    uint32_t vocab_size     = 128000;
    uint32_t hidden_size    = 2048;
    uint32_t ff_size        = 10752;
    uint32_t n_layers       = 30;
    uint32_t n_heads        = 32;
    uint32_t n_kv_heads     = 8;
    uint32_t head_dim       = 64;   // 2048 / 32
    uint32_t conv_kernel    = 3;
    float    rope_theta     = 1e6f;
    float    norm_eps       = 1e-5f;
    uint32_t bos_id         = 124894;
    uint32_t eos_id         = 124900;
    uint32_t pad_id         = 124893;
    uint32_t image_token_id = 124907;
    bool     tie_embeddings = true;
    // Layer type string: 'c' = conv, 'a' = attention.
    // Default: ccaccaccccacccaccccacccaccaccacc (22 conv, 8 attn)
    std::string layer_types = "ccaccaccccacccaccccacccaccaccacc";
};

// ============================================================================
// Per-layer weight structs
// ============================================================================

struct vision_layer_w {
    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    // Attention: fused QKV or separate
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;
    ggml_tensor * q_w = nullptr; ggml_tensor * q_b = nullptr;
    ggml_tensor * k_w = nullptr; ggml_tensor * k_b = nullptr;
    ggml_tensor * v_w = nullptr; ggml_tensor * v_b = nullptr;
    ggml_tensor * proj_w = nullptr;
    ggml_tensor * proj_b = nullptr;
    // MLP: GELU fc1/fc2
    ggml_tensor * fc1_w = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;
};

struct llm_layer_w {
    ggml_tensor * operator_norm_w = nullptr;
    ggml_tensor * ffn_norm_w      = nullptr;
    // SwiGLU FFN (all layers)
    ggml_tensor * ff_w1 = nullptr;  // gate
    ggml_tensor * ff_w2 = nullptr;  // down
    ggml_tensor * ff_w3 = nullptr;  // up
    bool is_attention = false;
    // Conv layers
    ggml_tensor * conv_in_proj_w  = nullptr;
    ggml_tensor * conv_conv_w     = nullptr;
    ggml_tensor * conv_out_proj_w = nullptr;
    // Attention layers
    ggml_tensor * attn_q_proj_w   = nullptr;
    ggml_tensor * attn_k_proj_w   = nullptr;
    ggml_tensor * attn_v_proj_w   = nullptr;
    ggml_tensor * attn_out_proj_w = nullptr;
    ggml_tensor * attn_q_ln_w     = nullptr;
    ggml_tensor * attn_k_ln_w     = nullptr;
};

// ============================================================================
// Model weights
// ============================================================================

struct model_weights {
    vision_hparams    vhp;
    projector_hparams php;
    llm_hparams       lhp;

    // Vision encoder
    ggml_tensor * v_patch_embed_w = nullptr;
    ggml_tensor * v_patch_embed_b = nullptr;
    ggml_tensor * v_pos_embed     = nullptr;  // learned position embeddings
    ggml_tensor * v_post_ln_w     = nullptr;
    ggml_tensor * v_post_ln_b     = nullptr;
    std::vector<vision_layer_w> v_layers;

    // Projector
    ggml_tensor * proj_fc1_w = nullptr;
    ggml_tensor * proj_fc1_b = nullptr;
    ggml_tensor * proj_fc2_w = nullptr;
    ggml_tensor * proj_fc2_b = nullptr;

    // LLM
    ggml_tensor * embed_tokens_w    = nullptr;
    ggml_tensor * embedding_norm_w  = nullptr;
    ggml_tensor * lm_head_w         = nullptr;
    std::vector<llm_layer_w> llm_layers;
};

// ============================================================================
// Context
// ============================================================================

struct ctx {
    model_weights m;

    // Weight storage
    ggml_context *          model_ctx  = nullptr;
    ggml_backend_buffer_t   model_buf  = nullptr;
    ggml_context *          mmproj_ctx = nullptr;
    ggml_backend_buffer_t   mmproj_buf = nullptr;

    // Backend
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for attention layers only
    struct {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *         k   = nullptr;  // [kv_dim, max_seq, n_attn_layers]
        ggml_tensor *         v   = nullptr;
        int max_seq       = 0;
        int n_attn_layers = 0;
        bool allocated    = false;
    } kvc;

    // Conv state cache: last (kernel_size - 1) = 2 columns per conv layer.
    // Layout: conv_state[conv_layer_index] → [hidden_size * 2] floats.
    std::vector<std::vector<float>> conv_state;
    int n_conv_layers = 0;

    // Tokenizer
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string>                 id_to_piece;
    std::unordered_map<std::string, int32_t> merge_rank;

    int n_threads = 4;
    int verbosity = 1;

    // Diff harness: loaded when LFM2_VL_DIFF_REF is set
    crispembed_diff::Ref diff_ref;
    bool has_diff_ref = false;
};

// Compare a tensor against the diff reference and print the result.
static void diff_stage(ctx & c, const char * name, const float * data, size_t n_elem) {
    if (!c.has_diff_ref) return;
    auto r = c.diff_ref.compare(name, data, n_elem);
    if (!r.found) {
        fprintf(stderr, "  DIFF %-25s (not in ref)\n", name);
        return;
    }
    fprintf(stderr, "  DIFF %-25s cos_min=%.6f max_abs=%.2e |mine|=%.4f |ref|=%.4f  %s\n",
            name, r.cos_min, r.max_abs, r.mine_norm, r.ref_norm,
            r.is_pass() ? "PASS" : "FAIL");
}

// ============================================================================
// KV cache management (attention layers only)
// ============================================================================

static void free_kv_cache(ctx & c) {
    if (c.kvc.buf) { ggml_backend_buffer_free(c.kvc.buf); c.kvc.buf = nullptr; }
    if (c.kvc.ctx) { ggml_free(c.kvc.ctx); c.kvc.ctx = nullptr; }
    c.kvc.k = nullptr;
    c.kvc.v = nullptr;
    c.kvc.max_seq = 0;
    c.kvc.allocated = false;
}

static bool alloc_kv_cache(ctx & c, int max_seq) {
    free_kv_cache(c);

    const auto & lhp = c.m.lhp;
    const int n_kv_heads = (int)lhp.n_kv_heads;
    const int head_dim   = (int)lhp.head_dim;
    const int kv_dim     = head_dim * n_kv_heads;

    // Count attention layers
    int n_attn = 0;
    for (uint32_t i = 0; i < lhp.n_layers; i++) {
        if (i < lhp.layer_types.size() && lhp.layer_types[i] == 'a') n_attn++;
    }
    c.kvc.n_attn_layers = n_attn;
    if (n_attn == 0) return true;

    ggml_init_params ip{ 2 * ggml_tensor_overhead() + 256, nullptr, true };
    c.kvc.ctx = ggml_init(ip);
    if (!c.kvc.ctx) return false;

    c.kvc.k = ggml_new_tensor_3d(c.kvc.ctx, GGML_TYPE_F32, kv_dim, max_seq, n_attn);
    c.kvc.v = ggml_new_tensor_3d(c.kvc.ctx, GGML_TYPE_F32, kv_dim, max_seq, n_attn);
    ggml_set_name(c.kvc.k, "lfm_kv_k");
    ggml_set_name(c.kvc.v, "lfm_kv_v");

    c.kvc.buf = ggml_backend_alloc_ctx_tensors(c.kvc.ctx, c.backend);
    if (!c.kvc.buf) {
        fprintf(stderr, "[lfm2_vl] KV cache alloc failed (max_seq=%d)\n", max_seq);
        free_kv_cache(c);
        return false;
    }
    ggml_backend_buffer_clear(c.kvc.buf, 0);
    c.kvc.max_seq = max_seq;
    c.kvc.allocated = true;

    if (c.verbosity >= 1) {
        size_t bytes = ggml_backend_buffer_get_size(c.kvc.buf);
        fprintf(stderr, "  KV cache: %d attn layers, max_seq=%d, %.1f MB\n",
                n_attn, max_seq, (float)bytes / (1024.0f * 1024.0f));
    }
    return true;
}

// Conv state management
static void init_conv_state(ctx & c) {
    const int D = (int)c.m.lhp.hidden_size;
    const int pad = (int)c.m.lhp.conv_kernel - 1;  // 2
    c.n_conv_layers = 0;
    for (uint32_t i = 0; i < c.m.lhp.n_layers; i++) {
        if (i < c.m.lhp.layer_types.size() && c.m.lhp.layer_types[i] == 'c')
            c.n_conv_layers++;
    }
    c.conv_state.resize(c.n_conv_layers);
    for (int i = 0; i < c.n_conv_layers; i++) {
        c.conv_state[i].assign((size_t)D * pad, 0.0f);
    }
}

static void reset_conv_state(ctx & c) {
    for (auto & s : c.conv_state)
        std::fill(s.begin(), s.end(), 0.0f);
}

// ============================================================================
// CPU scalar helpers
// ============================================================================

static std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> raw(nb);
    const void * src;
    if (t->buffer) {
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
        src = raw.data();
    } else {
        src = t->data;
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), src, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)src;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float)
            traits->to_float(src, out.data(), n);
        else
            memset(out.data(), 0, n * sizeof(float));
    }
    return out;
}

// ============================================================================
// Hparams loading
// ============================================================================

static bool load_hparams(ctx & c, const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char * k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32 = [&](const char * k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto & lhp = c.m.lhp;
    lhp.vocab_size     = u32("lfm2.vocab_size", u32("lfm2vl.vocab_size", lhp.vocab_size));
    lhp.hidden_size    = u32("lfm2.hidden_size", u32("lfm2.embedding_length", lhp.hidden_size));
    lhp.ff_size        = u32("lfm2.ff_dim", u32("lfm2.feed_forward_length", lhp.ff_size));
    lhp.n_layers       = u32("lfm2.n_layers", u32("lfm2.block_count", lhp.n_layers));
    lhp.n_heads        = u32("lfm2.n_heads", u32("lfm2.attention.head_count", lhp.n_heads));
    lhp.conv_kernel    = u32("lfm2.conv_kernel", u32("lfm2.shortconv.l_cache", lhp.conv_kernel));
    lhp.rope_theta     = f32("lfm2.rope_theta", f32("lfm2.rope.freq_base", lhp.rope_theta));
    lhp.norm_eps       = f32("lfm2.norm_eps", f32("lfm2.attention.layer_norm_rms_epsilon", lhp.norm_eps));

    // Read n_kv_heads (may be scalar or per-layer array)
    {
        int64_t k = gguf_find_key(g, "lfm2.n_kv_heads");
        if (k >= 0) {
            lhp.n_kv_heads = gguf_get_val_u32(g, k);
        } else {
            k = gguf_find_key(g, "lfm2.attention.head_count_kv");
            if (k >= 0) {
                if (gguf_get_kv_type(g, k) == GGUF_TYPE_ARRAY) {
                    int n = (int)gguf_get_arr_n(g, k);
                    uint32_t mx = 0;
                    for (int i = 0; i < n; i++) {
                        uint32_t v = ((const uint32_t *)gguf_get_arr_data(g, k))[i];
                        if (v > mx) mx = v;
                    }
                    if (mx > 0) lhp.n_kv_heads = mx;
                } else {
                    lhp.n_kv_heads = gguf_get_val_u32(g, k);
                }
            }
        }
    }

    lhp.head_dim = u32("lfm2.head_dim", 0);
    if (lhp.head_dim == 0 && lhp.n_heads > 0) lhp.head_dim = lhp.hidden_size / lhp.n_heads;

    // Layer types string
    lhp.layer_types = core_gguf::kv_str(g, "lfm2.layer_types", "");

    // Special tokens
    lhp.bos_id = u32("tokenizer.ggml.bos_token_id", lhp.bos_id);
    lhp.eos_id = u32("tokenizer.ggml.eos_token_id", lhp.eos_id);
    lhp.pad_id = u32("tokenizer.ggml.padding_token_id", lhp.pad_id);
    lhp.image_token_id = u32("lfm2vl.image_token_id", lhp.image_token_id);

    // Tied embeddings
    lhp.tie_embeddings = core_gguf::kv_bool(g, "lfm2.tie_word_embeddings", true);

    core_gguf::free_metadata(g);
    return true;
}

static bool load_vision_hparams(ctx & c, const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char * k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32v = [&](const char * k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto & vhp = c.m.vhp;
    // Try siglip2 prefix, then clip.vision prefix, then lfm2vl.vision prefix
    vhp.depth       = u32("siglip2.vision.depth", u32("clip.vision.block_count", u32("lfm2vl.vision.depth", vhp.depth)));
    vhp.hidden_size = u32("siglip2.vision.hidden_size", u32("clip.vision.embedding_length", u32("lfm2vl.vision.hidden_size", vhp.hidden_size)));
    vhp.ff_size     = u32("siglip2.vision.ff_size", u32("clip.vision.feed_forward_length", u32("lfm2vl.vision.ff_size", vhp.ff_size)));
    vhp.num_heads   = u32("siglip2.vision.num_heads", u32("clip.vision.attention.head_count", u32("lfm2vl.vision.num_heads", vhp.num_heads)));
    vhp.patch_size  = u32("siglip2.vision.patch_size", u32("clip.vision.patch_size", u32("lfm2vl.vision.patch_size", vhp.patch_size)));
    vhp.image_size  = u32("siglip2.vision.image_size", u32("clip.vision.image_size", u32("lfm2vl.vision.image_size", vhp.image_size)));
    vhp.norm_eps    = f32v("siglip2.vision.norm_eps", f32v("lfm2vl.vision.norm_eps", vhp.norm_eps));

    if (vhp.num_heads > 0) vhp.head_dim = vhp.hidden_size / vhp.num_heads;

    // Image mean/std
    int idx = gguf_find_key(g, "clip.vision.image_mean");
    if (idx < 0) idx = gguf_find_key(g, "siglip2.vision.image_mean");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto * d = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_mean[i] = d[i];
    }
    idx = gguf_find_key(g, "clip.vision.image_std");
    if (idx < 0) idx = gguf_find_key(g, "siglip2.vision.image_std");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto * d = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_std[i] = d[i];
    }

    // Projector
    auto & php = c.m.php;
    php.unshuffle_factor = u32("lfm2vl.projector.unshuffle_factor", php.unshuffle_factor);
    php.in_dim  = vhp.hidden_size * php.unshuffle_factor * php.unshuffle_factor;
    php.mid_dim = u32("lfm2vl.projector.mid_dim", php.mid_dim);
    php.out_dim = u32("lfm2vl.projector.out_dim", php.out_dim);

    core_gguf::free_metadata(g);
    return true;
}

// ============================================================================
// Weight loading
// ============================================================================

static bool load_llm_tensors(ctx & c, const char * path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c.backend, "lfm2_vl_llm", wl)) {
        return false;
    }
    c.model_ctx = wl.ctx;
    c.model_buf = wl.buf;

    auto get1 = [&](const std::string & name) -> ggml_tensor * {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };
    auto get2 = [&](const std::string & a, const std::string & b) -> ggml_tensor * {
        auto * t = get1(a);
        return t ? t : get1(b);
    };

    auto & m = c.m;
    m.embed_tokens_w   = get2("lfm.embed_tokens.weight", "token_embd.weight");
    m.embedding_norm_w = get2("lfm.embedding_norm.weight", "token_embd_norm.weight");
    m.lm_head_w        = get2("lfm.lm_head.weight", "output.weight");
    if (!m.lm_head_w && m.lhp.tie_embeddings) m.lm_head_w = m.embed_tokens_w;

    // Derive layer types from tensor presence if not in GGUF metadata
    if (m.lhp.layer_types.empty() || m.lhp.layer_types.size() < m.lhp.n_layers) {
        std::string lt(m.lhp.n_layers, 'c');
        for (uint32_t i = 0; i < m.lhp.n_layers; i++) {
            char a[128], b[128];
            snprintf(a, sizeof(a), "lfm.layers.%u.attn.q_proj.weight", i);
            snprintf(b, sizeof(b), "blk.%u.attn_q.weight", i);
            if (wl.tensors.count(a) || wl.tensors.count(b)) lt[i] = 'a';
        }
        m.lhp.layer_types = lt;
    }

    m.llm_layers.resize(m.lhp.n_layers);
    for (uint32_t i = 0; i < m.lhp.n_layers; i++) {
        auto & l = m.llm_layers[i];
        auto ln = [&](const char * our, const char * llama) -> ggml_tensor * {
            char a[128], b[128];
            snprintf(a, sizeof(a), "lfm.layers.%u.%s", i, our);
            snprintf(b, sizeof(b), "blk.%u.%s", i, llama);
            return get2(a, b);
        };
        l.operator_norm_w = ln("operator_norm.weight", "attn_norm.weight");
        l.ffn_norm_w      = ln("ffn_norm.weight", "ffn_norm.weight");
        l.ff_w1           = ln("ff.w1.weight", "ffn_gate.weight");
        l.ff_w2           = ln("ff.w2.weight", "ffn_down.weight");
        l.ff_w3           = ln("ff.w3.weight", "ffn_up.weight");
        l.is_attention = (i < m.lhp.layer_types.size() && m.lhp.layer_types[i] == 'a');
        if (l.is_attention) {
            l.attn_q_proj_w   = ln("attn.q_proj.weight", "attn_q.weight");
            l.attn_k_proj_w   = ln("attn.k_proj.weight", "attn_k.weight");
            l.attn_v_proj_w   = ln("attn.v_proj.weight", "attn_v.weight");
            l.attn_out_proj_w = ln("attn.out_proj.weight", "attn_output.weight");
            l.attn_q_ln_w     = ln("attn.q_layernorm.weight", "attn_q_norm.weight");
            l.attn_k_ln_w     = ln("attn.k_layernorm.weight", "attn_k_norm.weight");
        } else {
            l.conv_in_proj_w  = ln("conv.in_proj.weight", "shortconv.in_proj.weight");
            l.conv_conv_w     = ln("conv.conv.weight", "shortconv.conv.weight");
            l.conv_out_proj_w = ln("conv.out_proj.weight", "shortconv.out_proj.weight");
        }
    }

    return true;
}

static bool load_vision_tensors(ctx & c, const char * path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c.backend, "lfm2_vl_mmproj", wl)) {
        return false;
    }
    c.mmproj_ctx = wl.ctx;
    c.mmproj_buf = wl.buf;

    auto get1 = [&](const std::string & name) -> ggml_tensor * {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };
    auto get2 = [&](const std::string & a, const std::string & b) -> ggml_tensor * {
        auto * t = get1(a);
        return t ? t : get1(b);
    };

    auto & m = c.m;

    // Patch embedding
    m.v_patch_embed_w = get2("v.patch_embed.weight", "v.patch_embd.weight");
    m.v_patch_embed_b = get2("v.patch_embed.bias", "v.patch_embd.bias");
    // Position embeddings (learned)
    m.v_pos_embed = get1("v.pos_embed");
    if (!m.v_pos_embed) m.v_pos_embed = get1("v.pos_embed.weight");
    if (!m.v_pos_embed) m.v_pos_embed = get1("v.position_embed.weight");
    if (!m.v_pos_embed) m.v_pos_embed = get1("v.position_embd.weight");
    // Post-encoder layernorm
    m.v_post_ln_w = get2("v.post_layernorm.weight", "v.post_ln.weight");
    m.v_post_ln_b = get2("v.post_layernorm.bias", "v.post_ln.bias");

    m.v_layers.resize(m.vhp.depth);
    for (uint32_t i = 0; i < m.vhp.depth; i++) {
        auto & bl = m.v_layers[i];
        std::string p = "v.blk." + std::to_string(i) + ".";
        bl.ln1_w  = get2(p + "norm1.weight", p + "ln1.weight");
        bl.ln1_b  = get2(p + "norm1.bias", p + "ln1.bias");
        bl.ln2_w  = get2(p + "norm2.weight", p + "ln2.weight");
        bl.ln2_b  = get2(p + "norm2.bias", p + "ln2.bias");
        // Fused QKV
        bl.qkv_w  = get2(p + "attn_qkv.weight", p + "attn.qkv.weight");
        bl.qkv_b  = get2(p + "attn_qkv.bias", p + "attn.qkv.bias");
        // Separate Q/K/V
        bl.q_w    = get2(p + "attn_q.weight", p + "attn.q.weight");
        bl.q_b    = get2(p + "attn_q.bias", p + "attn.q.bias");
        bl.k_w    = get2(p + "attn_k.weight", p + "attn.k.weight");
        bl.k_b    = get2(p + "attn_k.bias", p + "attn.k.bias");
        bl.v_w    = get2(p + "attn_v.weight", p + "attn.v.weight");
        bl.v_b    = get2(p + "attn_v.bias", p + "attn.v.bias");
        bl.proj_w = get2(p + "attn_out.weight", p + "attn_proj.weight");
        bl.proj_b = get2(p + "attn_out.bias", p + "attn_proj.bias");
        // MLP
        bl.fc1_w  = get2(p + "ffn_fc1.weight", p + "ffn.fc1.weight");
        if (!bl.fc1_w) bl.fc1_w = get2(p + "ffn_up.weight", p + "ffn.up.weight");
        bl.fc1_b  = get2(p + "ffn_fc1.bias", p + "ffn.fc1.bias");
        if (!bl.fc1_b) bl.fc1_b = get2(p + "ffn_up.bias", p + "ffn.up.bias");
        bl.fc2_w  = get2(p + "ffn_fc2.weight", p + "ffn.fc2.weight");
        if (!bl.fc2_w) bl.fc2_w = get2(p + "ffn_down.weight", p + "ffn.down.weight");
        bl.fc2_b  = get2(p + "ffn_fc2.bias", p + "ffn.fc2.bias");
        if (!bl.fc2_b) bl.fc2_b = get2(p + "ffn_down.bias", p + "ffn.down.bias");
    }

    // Projector weights
    m.proj_fc1_w = get2("projector.fc1.weight", "mm.1.weight");
    m.proj_fc1_b = get2("projector.fc1.bias", "mm.1.bias");
    m.proj_fc2_w = get2("projector.fc2.weight", "mm.2.weight");
    m.proj_fc2_b = get2("projector.fc2.bias", "mm.2.bias");

    return true;
}

// ============================================================================
// Tokenizer loading
// ============================================================================

static bool load_tokenizer(ctx & c, const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto tokens_vec = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (tokens_vec.empty()) {
        core_gguf::free_metadata(g);
        return false;
    }
    c.id_to_piece.resize(tokens_vec.size());
    for (size_t i = 0; i < tokens_vec.size(); i++) {
        c.id_to_piece[i] = tokens_vec[i];
        c.token_to_id[tokens_vec[i]] = (int32_t)i;
    }

    // Merges
    int64_t mi = gguf_find_key(g, "tokenizer.ggml.merges");
    if (mi >= 0 && gguf_get_arr_type(g, mi) == GGUF_TYPE_STRING) {
        int nm = (int)gguf_get_arr_n(g, mi);
        int rank = 0;
        for (int i = 0; i < nm; i++) {
            std::string m = gguf_get_arr_str(g, mi, i);
            if (!m.empty()) c.merge_rank[m] = rank++;
        }
    }

    core_gguf::free_metadata(g);

    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] tokenizer: %zu vocab, %zu merges\n",
                c.id_to_piece.size(), c.merge_rank.size());
        // Check if expected tokens exist
        auto chk = [&](const char * s, int expected) {
            auto it = c.token_to_id.find(s);
            if (it != c.token_to_id.end()) {
                fprintf(stderr, "  '%s' → %d %s\n", s, it->second,
                        (it->second == expected) ? "✓" : "✗ (WRONG)");
            } else {
                fprintf(stderr, "  '%s' → NOT FOUND (expected %d)\n", s, expected);
            }
        };
        chk("O", 55);
        chk("CR", 6193);
        chk("Ġthis", 532);
        // Check merges
        auto mchk = [&](const char * m) {
            auto it = c.merge_rank.find(m);
            fprintf(stderr, "  merge '%s' → %s\n", m,
                    it != c.merge_rank.end() ? std::to_string(it->second).c_str() : "NOT FOUND");
        };
        mchk("C R");
        mchk("O CR");
        mchk("Ġ t");
        mchk("Ġt he");
        // Test tokenize (can't call here — function not yet declared)
        // Will test via build_token_ids debug output.
    }

    return true;
}

static std::vector<int32_t> tokenize(const ctx & c, const std::string & text) {
    // Debug: show pre-tokenized pieces
    if (dbg()) {
        auto pieces = core_bpe::lfm2_pretokenize(text);
        fprintf(stderr, "[lfm2_vl] pretokenize('%s'): %zu pieces\n", text.c_str(), pieces.size());
        for (auto & p : pieces) {
            auto enc = core_bpe::bytes_to_unicode(p.data(), p.size());
            // Tokenize this single piece
            std::vector<int32_t> ids;
            core_bpe::bpe_one(c.token_to_id, c.merge_rank, enc, ids);
            fprintf(stderr, "  '%s' → ", enc.c_str());
            for (auto id : ids) fprintf(stderr, "%d ", id);
            fprintf(stderr, "\n");
        }
    }
    return core_bpe::tokenize_lfm2(c.token_to_id, c.merge_rank, text);
}

static std::string decode_tokens(const ctx & c, const std::vector<int32_t> & ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int32_t)c.id_to_piece.size()) {
            out += c.id_to_piece[id];
        }
    }
    // GPT-2 byte-level BPE decode: convert unicode pieces to raw bytes
    return core_bpe::unicode_to_bytes(out);
}

// ============================================================================
// Image preprocessing (single 512x512 tile, SigLIP2 NaFlex style)
// ============================================================================

struct image_patches {
    std::vector<float> data;  // [n_patches, patch_dim] row-major
    int n_patches = 0;
    int patch_dim = 0;        // 3 * patch_size^2 = 768
    int h_patches = 0;        // patches per height
    int w_patches = 0;        // patches per width
};

static bool preprocess_image(const uint8_t * rgb, int height, int width, int channels,
                             const vision_hparams & vhp, image_patches & out) {
    const int P      = (int)vhp.patch_size;   // 16
    const int target = (int)vhp.tile_size;   // 512 (VL tile target)
    const int patch_dim = 3 * P * P;         // 768

    // HF NaFlex processor: aspect-preserving resize to target² total pixels,
    // with H and W rounded to multiples of P (=16).
    int rH, rW;
    image_preproc::smart_resize(height, width, P, target * target, target * target, &rH, &rW);

    const int gH = rH / P;
    const int gW = rW / P;
    const int n_patches = gH * gW;

    // Step 1: bilinear resize to rW × rH
    std::vector<float> resized((size_t)rH * rW * 3);
    for (int y = 0; y < rH; y++) {
        float sy = (float)y * (height - 1) / std::max(rH - 1, 1);
        int y0 = (int)sy, y1 = std::min(y0 + 1, height - 1);
        float fy = sy - y0;
        for (int x = 0; x < rW; x++) {
            float sx = (float)x * (width - 1) / std::max(rW - 1, 1);
            int x0 = (int)sx, x1 = std::min(x0 + 1, width - 1);
            float fx = sx - x0;
            for (int c = 0; c < 3; c++) {
                int ch = (channels >= 3) ? c : 0;
                float v00 = rgb[((size_t)y0 * width + x0) * channels + ch];
                float v01 = rgb[((size_t)y0 * width + x1) * channels + ch];
                float v10 = rgb[((size_t)y1 * width + x0) * channels + ch];
                float v11 = rgb[((size_t)y1 * width + x1) * channels + ch];
                float v = (1 - fy) * ((1 - fx) * v00 + fx * v01) +
                          fy       * ((1 - fx) * v10 + fx * v11);
                // Normalize: (v / 255 - mean) / std → [-1, 1]
                resized[((size_t)c * rH + y) * rW + x] =
                    (v / 255.0f - vhp.image_mean[c]) / vhp.image_std[c];
            }
        }
    }

    // Step 2: patchify to match the Conv2d patch embedding weight layout.
    //
    // The GGUF stores the Conv2d weight as [kW, kH, in_C, out_C] (ggml
    // column-major ne[0..3]). After reshape to 2D [patch_dim, hidden],
    // ne[0] = patch_dim with elements in (kW, kH, in_C) order — that is,
    // pixel-x varies fastest, then pixel-y, then channel.
    //
    // We must patchify into the SAME order so that ggml_mul_mat(weight, patches)
    // computes the correct dot product.
    out.data.resize((size_t)n_patches * patch_dim);
    for (int gy = 0; gy < gH; gy++) {
        for (int gx = 0; gx < gW; gx++) {
            float * dst = &out.data[(size_t)(gy * gW + gx) * patch_dim];
            // Fill in (px, py, c) order = x varies fastest
            for (int c = 0; c < 3; c++) {
                for (int py = 0; py < P; py++) {
                    for (int px = 0; px < P; px++) {
                        int iy = gy * P + py;
                        int ix = gx * P + px;
                        // Flat index in (kW, kH, in_C) order:
                        int flat = px + py * P + c * P * P;
                        dst[flat] = resized[((size_t)c * rH + iy) * rW + ix];
                    }
                }
            }
        }
    }

    out.n_patches = n_patches;
    out.patch_dim = patch_dim;
    out.h_patches = gH;
    out.w_patches = gW;

    fprintf(stderr, "[lfm2_vl] preproc: %dx%d → %dx%d, %d patches (%dx%d grid)\n",
            width, height, rW, rH, n_patches, gW, gH);
    return true;
}

// ============================================================================
// Vision encoder forward (SigLIP2 — 27 ViT layers, LayerNorm, bidirectional)
// ============================================================================

// Build the vision encoder graph as a ggml computation graph.
// Returns the output tensor pointer; the graph is built in the provided context.
static ggml_tensor * build_vision_graph(ctx & c, ggml_context * g, ggml_cgraph * gf,
                                        int n_patches) {
    const auto & vhp = c.m.vhp;
    const int H        = (int)vhp.hidden_size;    // 1152
    const int n_heads  = (int)vhp.num_heads;       // 16
    const int head_dim = (int)vhp.head_dim;        // 72
    const int ff       = (int)vhp.ff_size;         // 4304
    const float eps    = vhp.norm_eps;
    const int patch_dim = 3 * (int)vhp.patch_size * (int)vhp.patch_size;  // 768

    // Input: flattened patches [patch_dim, n_patches]
    ggml_tensor * pixel_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, patch_dim, n_patches);
    ggml_set_name(pixel_in, "pixel_in");
    ggml_set_input(pixel_in);

    // Patch embedding: Conv2d(3, 1152, 16, 16) → equivalent to Linear(768, 1152)
    // Weight is 4D [16, 16, 3, 1152] in GGUF; reshape to 2D [768, 1152] for mul_mat.
    ggml_tensor * pe_w = c.m.v_patch_embed_w;
    if (!pe_w) {
        fprintf(stderr, "[lfm2_vl] FATAL: v_patch_embed_w is null!\n");
        return nullptr;
    }
    if (dbg()) {
        fprintf(stderr, "[lfm2_vl] patch_embed_w: ndims=%d, ne=[%lld,%lld,%lld,%lld]\n",
                ggml_n_dims(pe_w), (long long)pe_w->ne[0], (long long)pe_w->ne[1],
                (long long)pe_w->ne[2], (long long)pe_w->ne[3]);
    }
    // Conv2d weight [16,16,3,1152] stored as 4D in GGUF. For mul_mat we need
    // it as 2D [768, 1152]. The weight tensor lives on the backend buffer so
    // we can't reshape it in-graph; instead we just override the shape in
    // place — the flat memory is identical (16*16*3 = 768, contiguous).
    if (ggml_n_dims(pe_w) > 2) {
        pe_w->ne[0] = patch_dim;  // 768
        pe_w->ne[1] = H;          // 1152
        pe_w->ne[2] = 1;
        pe_w->ne[3] = 1;
        pe_w->nb[1] = pe_w->nb[0] * pe_w->ne[0];
        pe_w->nb[2] = pe_w->nb[1] * pe_w->ne[1];
        pe_w->nb[3] = pe_w->nb[2];
        fprintf(stderr, "[lfm2_vl] after reshape: ne=[%lld,%lld,%lld,%lld], nb=[%lld,%lld,%lld,%lld], type=%d\n",
                (long long)pe_w->ne[0], (long long)pe_w->ne[1],
                (long long)pe_w->ne[2], (long long)pe_w->ne[3],
                (long long)pe_w->nb[0], (long long)pe_w->nb[1],
                (long long)pe_w->nb[2], (long long)pe_w->nb[3],
                (int)pe_w->type);
        fprintf(stderr, "[lfm2_vl] pixel_in: ne=[%lld,%lld], type=%d\n",
                (long long)pixel_in->ne[0], (long long)pixel_in->ne[1],
                (int)pixel_in->type);
    }
    ggml_tensor * x = ggml_mul_mat(g, pe_w, pixel_in);
    if (c.m.v_patch_embed_b) x = ggml_add(g, x, c.m.v_patch_embed_b);

    // Learned position embeddings — bilinear-interpolated on CPU.
    // The GGUF stores a [1152, 256] table (16×16 grid). For images whose
    // patch grid differs from 16×16, we interpolate to (h_patches, w_patches)
    // and pass the result as a graph input tensor.
    ggml_tensor * pos_emb_input = ggml_new_tensor_2d(g, GGML_TYPE_F32, H, n_patches);
    ggml_set_name(pos_emb_input, "v_pos_emb");
    ggml_set_input(pos_emb_input);
    x = ggml_add(g, x, pos_emb_input);

    // LayerNorm helper
    auto layernorm = [&](ggml_tensor * t, ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
        ggml_tensor * y = ggml_norm(g, t, eps);
        y = ggml_mul(g, y, w);
        if (b) y = ggml_add(g, y, b);
        return y;
    };

    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    // ViT blocks
    for (uint32_t il = 0; il < vhp.depth; il++) {
        const auto & bl = c.m.v_layers[il];
        if (!bl.ln1_w) {
            fprintf(stderr, "[lfm2_vl] FATAL: v_layers[%u].ln1_w is null!\n", il);
            return nullptr;
        }
        if (!bl.q_w && !bl.qkv_w) {
            fprintf(stderr, "[lfm2_vl] FATAL: v_layers[%u] has no Q weight!\n", il);
            return nullptr;
        }
        if (il == 0 && dbg()) {
            ggml_tensor * qw = bl.q_w ? bl.q_w : bl.qkv_w;
            fprintf(stderr, "[lfm2_vl] v_layer[0] q_w: ne=[%lld,%lld], type=%d\n",
                    (long long)qw->ne[0], (long long)qw->ne[1], (int)qw->type);
        }
        ggml_tensor * residual = x;

        // Pre-attention LayerNorm
        ggml_tensor * y = layernorm(x, bl.ln1_w, bl.ln1_b);

        // QKV
        ggml_tensor * Q, * K, * V;
        if (bl.qkv_w) {
            ggml_tensor * qkv = ggml_mul_mat(g, bl.qkv_w, y);
            if (bl.qkv_b) qkv = ggml_add(g, qkv, bl.qkv_b);
            // [3*H, n_patches] → [head_dim, n_heads, 3, n_patches]
            qkv = ggml_reshape_4d(g, qkv, head_dim, n_heads, 3, n_patches);
            Q = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], 0);
            K = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], qkv->nb[2]);
            V = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], 2 * qkv->nb[2]);
        } else {
            Q = ggml_mul_mat(g, bl.q_w, y);
            if (bl.q_b) Q = ggml_add(g, Q, bl.q_b);
            K = ggml_mul_mat(g, bl.k_w, y);
            if (bl.k_b) K = ggml_add(g, K, bl.k_b);
            V = ggml_mul_mat(g, bl.v_w, y);
            if (bl.v_b) V = ggml_add(g, V, bl.v_b);
            Q = ggml_reshape_3d(g, Q, head_dim, n_heads, n_patches);
            K = ggml_reshape_3d(g, K, head_dim, n_heads, n_patches);
            V = ggml_reshape_3d(g, V, head_dim, n_heads, n_patches);
        }

        // Bidirectional attention (no causal mask)
        // Permute for standard attention: (head_dim, n_patches, n_heads)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // Manual attention: QK^T * scale → softmax → V
        // This avoids flash_attn_ext mask issues for bidirectional attention.
        // Gate flash behind LFM2_VL_FLASH_ATTN env var.
        ggml_tensor * attn_out;
        if (core_env::on("LFM2_VL_FLASH_ATTN")) {
            attn_out = ggml_flash_attn_ext(g, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        } else {
            // Q, K, V: (head_dim, n_patches, n_heads) after permute
            // ggml_mul_mat(A, B) = B × A^T
            // scores = Q × K^T → ggml_mul_mat(K, Q) → (n_patches, n_patches, n_heads)
            ggml_tensor * scores = ggml_mul_mat(g, K, Q);
            scores = ggml_scale(g, scores, attn_scale);
            scores = ggml_soft_max(g, scores);
            // out = scores × V → need ggml_mul_mat with V permuted
            // V: (head_dim, n_patches, n_heads)
            // scores: (n_patches, n_patches, n_heads)
            // Want: out[d,t,h] = sum_s scores[s,t,h] * V[d,s,h]
            // = ggml_mul_mat(V, scores) → but V->ne[0]=head_dim ≠ scores->ne[0]=n_patches
            // Need V transposed to (n_patches, head_dim, n_heads), then mul_mat
            ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
            attn_out = ggml_mul_mat(g, Vt, scores);
        }

        // Permute back: (head_dim, n_heads, n_patches)
        attn_out = ggml_cont(g, ggml_permute(g, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(g, attn_out, H, n_patches);

        // Output projection
        attn_out = ggml_mul_mat(g, bl.proj_w, attn_out);
        if (bl.proj_b) attn_out = ggml_add(g, attn_out, bl.proj_b);

        x = ggml_add(g, residual, attn_out);

        // Pre-MLP LayerNorm
        residual = x;
        y = layernorm(x, bl.ln2_w, bl.ln2_b);

        // MLP: fc1 → gelu_tanh → fc2
        ggml_tensor * mlp = ggml_mul_mat(g, bl.fc1_w, y);
        if (bl.fc1_b) mlp = ggml_add(g, mlp, bl.fc1_b);
        mlp = ggml_gelu(g, mlp);
        mlp = ggml_mul_mat(g, bl.fc2_w, mlp);
        if (bl.fc2_b) mlp = ggml_add(g, mlp, bl.fc2_b);

        x = ggml_add(g, residual, mlp);
    }

    // Post-encoder LayerNorm
    if (c.m.v_post_ln_w) {
        x = layernorm(x, c.m.v_post_ln_w, c.m.v_post_ln_b);
    }

    ggml_set_name(x, "vision_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    return x;
}

// Run vision encoder on preprocessed patches.
// Returns float vector of shape [n_projected, proj_dim].
static bool encode_vision(ctx & c, const image_patches & patches,
                          std::vector<float> & out_embeds, int & out_n_tokens, int & out_dim) {
    auto t0 = steady_clock::now();
    const auto & vhp = c.m.vhp;
    const int H = (int)vhp.hidden_size;
    const int n_patches = patches.n_patches;

    // Build graph
    const int max_nodes = 4096;
    size_t meta_size = ggml_tensor_overhead() * max_nodes +
                       ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip{ meta_size, nullptr, true };
    ggml_context * g = ggml_init(ip);
    if (!g) return false;

    fprintf(stderr, "[lfm2_vl] vision: building graph for %d patches...\n", n_patches);

    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_tensor * vis_out = build_vision_graph(c, g, gf, n_patches);
    fprintf(stderr, "[lfm2_vl] vision: graph built, %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_backend_sched_reset(c.sched);
    fprintf(stderr, "[lfm2_vl] vision: allocating graph...\n");
    if (!ggml_backend_sched_alloc_graph(c.sched, gf)) {
        fprintf(stderr, "[lfm2_vl] vision graph alloc failed\n");
        ggml_free(g);
        return false;
    }
    fprintf(stderr, "[lfm2_vl] vision: graph allocated, setting inputs...\n");

    // Set input: pixel patches
    ggml_tensor * pixel_in = ggml_graph_get_tensor(gf, "pixel_in");
    if (!pixel_in) { fprintf(stderr, "[lfm2_vl] pixel_in tensor not found!\n"); ggml_free(g); return false; }
    fprintf(stderr, "[lfm2_vl] pixel_in: %lld x %lld, data size %zu\n",
            (long long)pixel_in->ne[0], (long long)pixel_in->ne[1],
            patches.data.size() * sizeof(float));
    ggml_backend_tensor_set(pixel_in, patches.data.data(), 0,
                            patches.data.size() * sizeof(float));
    // Debug: print first few patch values for parity
    fprintf(stderr, "[lfm2_vl] input patch 0 first 5: ");
    for (int i = 0; i < std::min(5, (int)patches.patch_dim); i++)
        fprintf(stderr, "%.6f ", patches.data[i]);
    fprintf(stderr, "\n");

    // Position embeddings: bilinear-interpolate from learned 16×16 grid.
    // v_pos_embed is [H, 256] in ggml (= [256, H] row-major = 16×16 grid of H-dim vectors).
    {
        ggml_tensor * pos_tensor = ggml_graph_get_tensor(gf, "v_pos_emb");
        const int hp = patches.h_patches;
        const int wp = patches.w_patches;
        // Read the raw 16×16 position embeddings from the weight tensor
        const int grid = 16;
        std::vector<float> pos_table((size_t)grid * grid * H);
        if (c.m.v_pos_embed) {
            ggml_backend_tensor_get(c.m.v_pos_embed, pos_table.data(), 0,
                                    pos_table.size() * sizeof(float));
        }
        // Bilinear interpolation: map (hp, wp) grid → (grid, grid) source
        std::vector<float> interp_pos((size_t)n_patches * H, 0.0f);
        for (int r = 0; r < hp; r++) {
            for (int col = 0; col < wp; col++) {
                // Map target (r, col) to source coordinates.
                // Use align_corners=False (PyTorch default for F.interpolate):
                // src = (dst + 0.5) * src_size / dst_size - 0.5
                float sy = ((float)r + 0.5f) * grid / hp - 0.5f;
                float sx = ((float)col + 0.5f) * grid / wp - 0.5f;
                sy = std::max(0.0f, std::min(sy, (float)(grid - 1)));
                sx = std::max(0.0f, std::min(sx, (float)(grid - 1)));
                int y0 = (int)sy; int y1 = std::min(y0 + 1, grid - 1);
                int x0 = (int)sx; int x1 = std::min(x0 + 1, grid - 1);
                float fy = sy - y0; float fx = sx - x0;
                float w00 = (1 - fy) * (1 - fx);
                float w01 = (1 - fy) * fx;
                float w10 = fy * (1 - fx);
                float w11 = fy * fx;
                int dst_idx = r * wp + col;
                const float * s00 = &pos_table[((size_t)y0 * grid + x0) * H];
                const float * s01 = &pos_table[((size_t)y0 * grid + x1) * H];
                const float * s10 = &pos_table[((size_t)y1 * grid + x0) * H];
                const float * s11 = &pos_table[((size_t)y1 * grid + x1) * H];
                float * dst = &interp_pos[(size_t)dst_idx * H];
                for (int d = 0; d < H; d++) {
                    dst[d] = w00 * s00[d] + w01 * s01[d] + w10 * s10[d] + w11 * s11[d];
                }
            }
        }
        ggml_backend_tensor_set(pos_tensor, interp_pos.data(), 0,
                                interp_pos.size() * sizeof(float));
    }

    // Compute
    if (ggml_backend_sched_graph_compute(c.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[lfm2_vl] vision compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read vision output: [H, n_patches] — ggml col-major, ne[0]=H
    std::vector<float> vis_data((size_t)H * n_patches);
    ggml_backend_tensor_get(vis_out, vis_data.data(), 0, vis_data.size() * sizeof(float));

    // Debug: print first few values and norm for parity checking
    {
        fprintf(stderr, "[lfm2_vl] vision output first 5 (patch 0): ");
        for (int i = 0; i < std::min(5, H); i++)
            fprintf(stderr, "%.6f ", vis_data[i]);  // vis_data[dim + patch*H]
        fprintf(stderr, "\n");
        double norm = 0;
        for (size_t i = 0; i < vis_data.size(); i++) norm += vis_data[i] * vis_data[i];
        fprintf(stderr, "[lfm2_vl] vision output full norm: %.4f\n", std::sqrt(norm));
    }

    ggml_free(g);

    if (c.verbosity >= 1) {
        fprintf(stderr, "  vision encoder: %d patches, %lld ms\n", n_patches, ms_since(t0));
    }

    // Diff: compare vision output against reference
    diff_stage(c, "vis_post_ln", vis_data.data(), vis_data.size());

    // ── Projector: pixel_unshuffle → Linear → GELU → Linear ──
    auto t1 = steady_clock::now();

    const int factor = (int)c.m.php.unshuffle_factor;  // 2
    // Python output layout: [W//f, H//f] where W=h_patches, H=w_patches
    const int w_out  = patches.h_patches / factor;   // W//f = 36/2 = 18
    const int h_out  = patches.w_patches / factor;   // H//f = 28/2 = 14
    const int n_proj = w_out * h_out;
    const int C_in   = H;                              // 1152
    const int C_us   = C_in * factor * factor;         // 4608

    // pixel_unshuffle on CPU — matches HF Lfm2VlMultiModalProjector exactly.
    //
    // Python (from transformers/models/lfm2_vl/modeling_lfm2_vl.py):
    //   input:  [B, W=h_patches, H=w_patches, C]     (channels-last)
    //   step1:  reshape(B, W, H//f, C*f)
    //   step2:  permute(0, 2, 1, 3)                   → [B, H//f, W, C*f]
    //   step3:  reshape(B, H//f, W//f, C*f*f)
    //   step4:  permute(0, 2, 1, 3)                   → [B, W//f, H//f, C*f²]
    //
    // Input: vis_data [C_in, n_patches] col-major, patches in row-major
    //        [h_patches, w_patches] order.
    // Output: us_data [C_us, n_proj] col-major, projected patches in
    //         row-major [w_out, h_out] order (matching Python's [W//f, H//f]).
    //
    // The key: Python's W dim = h_patches, H dim = w_patches (it reshapes
    // the flat sequence as [h_patches, w_patches] but calls them W, H).
    const int pW = patches.h_patches;  // 36 = h_patches (Python's W dim)
    const int pH = patches.w_patches;  // 28 = w_patches (Python's H dim)
    const int f = factor;              // 2

    std::vector<float> us_data((size_t)C_us * n_proj, 0.0f);
    // Iterate over the output layout [W//f, H//f] = [w_out=18, h_out=14]
    for (int ow = 0; ow < pW / f; ow++) {        // W//f = 18
        for (int oh = 0; oh < pH / f; oh++) {     // H//f = 14
            int out_idx = ow * (pH / f) + oh;     // row-major [W//f, H//f]
            for (int dw = 0; dw < f; dw++) {
                for (int dh = 0; dh < f; dh++) {
                    int src_w = ow * f + dw;  // index in W = h_patches
                    int src_h = oh * f + dh;  // index in H = w_patches
                    int src_patch = src_w * patches.w_patches + src_h;
                    // Channel offset: dh comes from step1's C*f split,
                    // dw comes from step3's C*f*f split.
                    int c_off = (dw * f + dh) * C_in;
                    for (int ch = 0; ch < C_in; ch++) {
                        // vis_data is ggml col-major [H, n_patches]:
                        // element (ch, patch) at flat offset patch * H + ch
                        us_data[(size_t)(c_off + ch) * n_proj + out_idx] =
                            vis_data[(size_t)src_patch * H + ch];
                    }
                }
            }
        }
    }

    // Debug: pixel_unshuffle token 0 first 5 values
    fprintf(stderr, "[lfm2_vl] unshuffle token 0 first 5: ");
    for (int i = 0; i < std::min(5, C_us); i++)
        fprintf(stderr, "%.6f ", us_data[(size_t)i * n_proj + 0]);  // column 0
    fprintf(stderr, "\n");

    // Diff: compare pixel_unshuffle output (before MLP)
    diff_stage(c, "projector_unshuffle", us_data.data(), us_data.size());

    // Project through MLP: Linear(4608→2048) → GELU → Linear(2048→2048)
    const int mid_dim = (int)c.m.php.mid_dim;
    const int out_d   = (int)c.m.php.out_dim;

    // fc1
    auto fc1_w = to_f32(c.m.proj_fc1_w);
    auto fc1_b = to_f32(c.m.proj_fc1_b);
    auto fc2_w = to_f32(c.m.proj_fc2_w);
    auto fc2_b = to_f32(c.m.proj_fc2_b);

    out_embeds.resize((size_t)out_d * n_proj);
    std::vector<float> mid_buf(mid_dim);

    for (int p = 0; p < n_proj; p++) {
        const float * in = us_data.data() + (size_t)p;  // column p

        // Gather column p from us_data [C_us, n_proj] layout
        std::vector<float> col_in(C_us);
        for (int i = 0; i < C_us; i++) col_in[i] = us_data[(size_t)i * n_proj + p];

        // fc1: [C_us → mid_dim]
        for (int o = 0; o < mid_dim; o++) {
            float s = fc1_b.empty() ? 0.0f : fc1_b[o];
            for (int i = 0; i < C_us; i++) s += col_in[i] * fc1_w[(size_t)o * C_us + i];
            mid_buf[o] = s;
        }
        // GELU (tanh approximation)
        for (int i = 0; i < mid_dim; i++) {
            float x = mid_buf[i];
            float t = std::tanh(0.7978845608f * (x + 0.044715f * x * x * x));
            mid_buf[i] = 0.5f * x * (1.0f + t);
        }
        // fc2: [mid_dim → out_d]
        for (int o = 0; o < out_d; o++) {
            float s = fc2_b.empty() ? 0.0f : fc2_b[o];
            for (int i = 0; i < mid_dim; i++) s += mid_buf[i] * fc2_w[(size_t)o * mid_dim + i];
            out_embeds[(size_t)o * n_proj + p] = s;
        }
    }

    // Transpose to [n_proj, out_d] row-major for splicing
    std::vector<float> out_row((size_t)n_proj * out_d);
    for (int p = 0; p < n_proj; p++) {
        for (int d = 0; d < out_d; d++) {
            out_row[(size_t)p * out_d + d] = out_embeds[(size_t)d * n_proj + p];
        }
    }
    out_embeds = std::move(out_row);
    out_n_tokens = n_proj;
    out_dim = out_d;

    if (c.verbosity >= 1) {
        fprintf(stderr, "  projector: %d tokens → %d dim, %lld ms\n",
                n_proj, out_d, ms_since(t1));
    }

    // Debug: first 5 values of projector output (token 0)
    fprintf(stderr, "[lfm2_vl] projector first token first 5: ");
    for (int i = 0; i < std::min(5, out_d); i++)
        fprintf(stderr, "%.6f ", out_embeds[i]);  // [p=0, d=0..4]
    fprintf(stderr, "\n");

    // Diff: compare projector output against reference
    diff_stage(c, "projector_out", out_embeds.data(), out_embeds.size());

    return true;
}

// ============================================================================
// LLM graph helpers
// ============================================================================

static ggml_tensor * lfm2_rms_norm(ggml_context * g, ggml_tensor * x,
                                    ggml_tensor * w, float eps) {
    if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
    return ggml_mul(g, ggml_rms_norm(g, x, eps), w);
}

static ggml_tensor * lfm2_swiglu(ggml_context * g, ggml_tensor * x,
                                  ggml_tensor * w1, ggml_tensor * w2,
                                  ggml_tensor * w3) {
    return ggml_mul_mat(g, w2,
        ggml_mul(g, ggml_silu(g, ggml_mul_mat(g, w1, x)),
                 ggml_mul_mat(g, w3, x)));
}

// ============================================================================
// LLM prefill graph (full sequence, splicing image embeddings)
// ============================================================================

// Build a prefill graph for the full token sequence with image embeddings
// spliced in at IMAGE token positions. The graph includes all 30 hybrid
// layers (conv + attn). Conv layers run causal depthwise conv1d; attention
// layers use full GQA over the prefill sequence. KV cache is populated
// during this pass.
//
// Returns the logits tensor for the last token position.
static ggml_tensor * build_prefill_graph(ctx & c, ggml_context * g, ggml_cgraph * gf,
                                          int n_tokens, int n_image_tokens,
                                          bool populate_kvc) {
    const auto & lhp = c.m.lhp;
    const int D          = (int)lhp.hidden_size;
    const int n_heads    = (int)lhp.n_heads;
    const int n_kv_heads = (int)lhp.n_kv_heads;
    const int head_dim   = (int)lhp.head_dim;
    const int n_layers   = (int)lhp.n_layers;
    const float eps      = lhp.norm_eps;
    const float theta    = lhp.rope_theta;
    const int kv_dim     = head_dim * n_kv_heads;
    const int conv_k     = (int)lhp.conv_kernel;
    const int pad        = conv_k - 1;  // 2

    // Inputs
    ggml_tensor * tok_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tok_ids, "tok_ids");
    ggml_set_input(tok_ids);

    // Image embeddings input [n_image_tokens, D] row-major → ggml [D, n_image_tokens]
    ggml_tensor * img_emb = nullptr;
    if (n_image_tokens > 0) {
        img_emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_image_tokens);
        ggml_set_name(img_emb, "img_emb");
        ggml_set_input(img_emb);
    }

    // Image token mask: 1 where token is IMAGE, 0 otherwise [n_tokens]
    ggml_tensor * img_mask = nullptr;
    if (n_image_tokens > 0) {
        img_mask = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
        ggml_set_name(img_mask, "img_mask");
        ggml_set_input(img_mask);
    }

    // Position IDs for RoPE
    ggml_tensor * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "positions");
    ggml_set_input(pos);

    // Embedding lookup
    ggml_tensor * x = ggml_get_rows(g, c.m.embed_tokens_w, tok_ids);  // [D, n_tokens]

    // Splice image embeddings at IMAGE token positions.
    // We do this by building a combined embedding on the CPU side (set after
    // graph alloc). So the graph just takes the pre-spliced input.
    // Actually, we'll use a simpler approach: pass the final spliced embedding
    // directly as a float input.
    // Re-architecture: use a float input for the full spliced sequence.
    // The tok_ids lookup + splice is complex in ggml; simpler to compute the
    // spliced embeddings on CPU and pass as a single input.

    // Alternative: pass pre-spliced embeddings
    ggml_tensor * emb_input = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_tokens);
    ggml_set_name(emb_input, "emb_input");
    ggml_set_input(emb_input);
    x = emb_input;

    // Track attn layer index for KV cache
    int attn_idx = 0;

    for (int il = 0; il < n_layers; il++) {
        const auto & l = c.m.llm_layers[il];
        ggml_tensor * residual = x;

        // Pre-operator RMSNorm
        ggml_tensor * h = lfm2_rms_norm(g, x, l.operator_norm_w, eps);

        if (l.is_attention) {
            // GQA attention with QK RMSNorm + RoPE
            ggml_tensor * Q = ggml_mul_mat(g, l.attn_q_proj_w, h);
            ggml_tensor * K = ggml_mul_mat(g, l.attn_k_proj_w, h);
            ggml_tensor * V = ggml_mul_mat(g, l.attn_v_proj_w, h);

            Q = ggml_reshape_3d(g, Q, head_dim, n_heads, n_tokens);
            K = ggml_reshape_3d(g, K, head_dim, n_kv_heads, n_tokens);
            V = ggml_reshape_3d(g, V, head_dim, n_kv_heads, n_tokens);

            // Per-head QK RMSNorm (before RoPE)
            auto f32_cast = [&](ggml_tensor * t) -> ggml_tensor * {
                return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
            };
            Q = ggml_mul(g, ggml_rms_norm(g, Q, 1e-5f), f32_cast(l.attn_q_ln_w));
            K = ggml_mul(g, ggml_rms_norm(g, K, 1e-5f), f32_cast(l.attn_k_ln_w));

            // RoPE (NEOX interleaved, n_dims=head_dim, theta=1e6)
            Q = ggml_rope_ext(g, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                              0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(g, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                              0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Write KV to cache
            if (populate_kvc && c.kvc.k && attn_idx < c.kvc.n_attn_layers) {
                ggml_tensor * K_flat = ggml_cont(g, ggml_reshape_2d(g, K, kv_dim, n_tokens));
                ggml_tensor * V_flat = ggml_cont(g, ggml_reshape_2d(g, V, kv_dim, n_tokens));
                ggml_tensor * k_wr = ggml_view_2d(g, c.kvc.k, kv_dim, n_tokens,
                    c.kvc.k->nb[1], (size_t)attn_idx * c.kvc.k->nb[2]);
                ggml_tensor * v_wr = ggml_view_2d(g, c.kvc.v, kv_dim, n_tokens,
                    c.kvc.v->nb[1], (size_t)attn_idx * c.kvc.v->nb[2]);
                ggml_build_forward_expand(gf, ggml_cpy(g, K_flat, k_wr));
                ggml_build_forward_expand(gf, ggml_cpy(g, V_flat, v_wr));
            }

            // Causal attention with causal mask (flash_attn_ext with mask=nullptr
            // is FULL attention — the dev guide warns this is NOT causal).
            Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
            K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
            V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

            const float scale = 1.0f / sqrtf((float)head_dim);

            // Causal mask for flash_attn_ext: [n_kv, n_q] F16
            // n_kv = n_tokens (full sequence), n_q = n_tokens (all queries)
            ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, n_tokens, n_tokens);
            {
                char mname[64];
                snprintf(mname, sizeof(mname), "causal_mask_%d", il);
                ggml_set_name(mask, mname);
                ggml_set_input(mask);
            }

            ggml_tensor * attn_out = ggml_flash_attn_ext(g, Q, K, V, mask,
                                                         scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            attn_out = ggml_reshape_2d(g, attn_out, D, n_tokens);

            h = ggml_mul_mat(g, l.attn_out_proj_w, attn_out);
            attn_idx++;
        } else {
            // ShortConv: causal depthwise conv1d
            // in_proj: (D, T) → (3D, T), splits B, C, x
            ggml_tensor * bcx = ggml_mul_mat(g, l.conv_in_proj_w, h);

            ggml_tensor * B  = ggml_cont(g, ggml_view_2d(g, bcx, D, n_tokens, bcx->nb[1], 0));
            ggml_tensor * C  = ggml_cont(g, ggml_view_2d(g, bcx, D, n_tokens, bcx->nb[1],
                                                          (size_t)D * sizeof(float)));
            ggml_tensor * xi = ggml_cont(g, ggml_view_2d(g, bcx, D, n_tokens, bcx->nb[1],
                                                          (size_t)2 * D * sizeof(float)));

            // Bx = B * x
            ggml_tensor * Bx = ggml_mul(g, B, xi);

            // Mark Bx for conv state extraction (last 2 columns needed for decode)
            if (populate_kvc) {
                char bx_name[64];
                snprintf(bx_name, sizeof(bx_name), "bx_%d", il);
                ggml_set_name(Bx, bx_name);
                ggml_set_output(Bx);
            }

            // Causal depthwise conv1d: left-pad by (kernel-1)=2
            // Create padded input: [D, pad + n_tokens]
            ggml_tensor * pad_zeros = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, pad);
            ggml_set_name(pad_zeros, ("conv_pad_" + std::to_string(il)).c_str());
            ggml_set_input(pad_zeros);

            ggml_tensor * Bx_padded = ggml_concat(g, pad_zeros, Bx, 1);  // [D, pad+T]

            // Transpose for conv1d: [pad+T, D]
            ggml_tensor * Bx_t = ggml_cont(g, ggml_transpose(g, Bx_padded));

            // Depthwise conv1d with kernel
            ggml_tensor * conv_w = ggml_cast(g, l.conv_conv_w, GGML_TYPE_F16);
            conv_w = ggml_reshape_3d(g, conv_w, conv_w->ne[0], 1, D);
            ggml_tensor * co = ggml_conv_1d_dw(g, conv_w, Bx_t, 1, 0, 1);

            // conv_1d_dw output: [T_out, D] where T_out = (pad+T) - K + 1 = T
            int T_conv = (int)co->ne[0];
            if (T_conv > n_tokens)
                co = ggml_view_2d(g, co, n_tokens, D, co->nb[1], 0);

            co = ggml_cont(g, ggml_transpose(g, co));  // [D, T]

            // y = C * conv_out
            ggml_tensor * y = ggml_mul(g, C, co);
            h = ggml_mul_mat(g, l.conv_out_proj_w, y);
        }

        x = ggml_add(g, residual, h);

        // FFN: RMSNorm → SwiGLU
        residual = x;
        h = lfm2_rms_norm(g, x, l.ffn_norm_w, eps);
        h = lfm2_swiglu(g, h, l.ff_w1, l.ff_w2, l.ff_w3);
        x = ggml_add(g, residual, h);

        // Mark first 4 layer outputs for diff comparison
        if (il < 4 && populate_kvc) {
            char lname[64];
            snprintf(lname, sizeof(lname), "llm_layer_%d", il);
            ggml_set_name(x, lname);
            ggml_set_output(x);
        }
    }

    // Final norm
    x = lfm2_rms_norm(g, x, c.m.embedding_norm_w, eps);

    // lm_head: logits for last token only
    ggml_tensor * last_tok = ggml_view_2d(g, x, D, 1, x->nb[1],
                                          (size_t)(n_tokens - 1) * x->nb[1]);
    last_tok = ggml_cont(g, last_tok);

    ggml_tensor * lm_w = c.m.lm_head_w;
    ggml_tensor * logits = ggml_mul_mat(g, lm_w, last_tok);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    return logits;
}

// ============================================================================
// LLM decode step (single token, KV cached attention + conv state)
// ============================================================================

static ggml_cgraph * build_decode_step_graph(ctx & c, ggml_context * g,
                                              int n_kv, int pos, int max_seq) {
    const auto & lhp = c.m.lhp;
    const int D          = (int)lhp.hidden_size;
    const int n_heads    = (int)lhp.n_heads;
    const int n_kv_heads = (int)lhp.n_kv_heads;
    const int head_dim   = (int)lhp.head_dim;
    const int n_layers   = (int)lhp.n_layers;
    const float eps      = lhp.norm_eps;
    const float theta    = lhp.rope_theta;
    const int kv_dim     = head_dim * n_kv_heads;
    const int conv_k     = (int)lhp.conv_kernel;

    ggml_cgraph * gf = ggml_new_graph_custom(g, 16384, false);

    // Input: single token embedding [D, 1]
    ggml_tensor * tok_emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, 1);
    ggml_set_name(tok_emb, "tok_emb");
    ggml_set_input(tok_emb);

    // Position for RoPE
    ggml_tensor * pos_t = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pos_t, "pos");
    ggml_set_input(pos_t);

    // KV mask removed: decode reads only n_kv+1 valid entries, no mask needed.

    ggml_tensor * x = tok_emb;
    int attn_idx = 0;
    int conv_idx = 0;

    for (int il = 0; il < n_layers; il++) {
        const auto & l = c.m.llm_layers[il];
        ggml_tensor * residual = x;

        ggml_tensor * h = lfm2_rms_norm(g, x, l.operator_norm_w, eps);

        if (l.is_attention) {
            // Single-token GQA with KV cache
            ggml_tensor * Q = ggml_mul_mat(g, l.attn_q_proj_w, h);
            ggml_tensor * K_new = ggml_mul_mat(g, l.attn_k_proj_w, h);
            ggml_tensor * V_new = ggml_mul_mat(g, l.attn_v_proj_w, h);

            Q     = ggml_reshape_3d(g, Q, head_dim, n_heads, 1);
            K_new = ggml_reshape_3d(g, K_new, head_dim, n_kv_heads, 1);
            V_new = ggml_reshape_3d(g, V_new, head_dim, n_kv_heads, 1);

            // QK RMSNorm
            auto f32_cast = [&](ggml_tensor * t) -> ggml_tensor * {
                return t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32);
            };
            Q     = ggml_mul(g, ggml_rms_norm(g, Q, 1e-5f), f32_cast(l.attn_q_ln_w));
            K_new = ggml_mul(g, ggml_rms_norm(g, K_new, 1e-5f), f32_cast(l.attn_k_ln_w));

            // RoPE
            Q     = ggml_rope_ext(g, Q, pos_t, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                                  0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K_new = ggml_rope_ext(g, K_new, pos_t, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                                  0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            K_new = ggml_cont(g, K_new);
            V_new = ggml_cont(g, V_new);

            ggml_tensor * K_flat = ggml_reshape_2d(g, K_new, kv_dim, 1);
            ggml_tensor * V_flat = ggml_reshape_2d(g, V_new, kv_dim, 1);

            // Write K/V to cache at position n_kv
            ggml_tensor * k_write = ggml_view_2d(g, c.kvc.k, kv_dim, 1,
                c.kvc.k->nb[1],
                (size_t)attn_idx * c.kvc.k->nb[2] + (size_t)n_kv * c.kvc.k->nb[1]);
            ggml_tensor * v_write = ggml_view_2d(g, c.kvc.v, kv_dim, 1,
                c.kvc.v->nb[1],
                (size_t)attn_idx * c.kvc.v->nb[2] + (size_t)n_kv * c.kvc.v->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(g, K_flat, k_write));
            ggml_build_forward_expand(gf, ggml_cpy(g, V_flat, v_write));

            // Read full KV cache for this layer (n_kv+1 valid entries)
            int n_kv_total = n_kv + 1;
            ggml_tensor * k_layer = ggml_view_2d(g, c.kvc.k, kv_dim, n_kv_total,
                c.kvc.k->nb[1], (size_t)attn_idx * c.kvc.k->nb[2]);
            ggml_tensor * v_layer = ggml_view_2d(g, c.kvc.v, kv_dim, n_kv_total,
                c.kvc.v->nb[1], (size_t)attn_idx * c.kvc.v->nb[2]);

            ggml_tensor * K_full = ggml_reshape_3d(g, k_layer, head_dim, n_kv_heads, n_kv_total);
            ggml_tensor * V_full = ggml_reshape_3d(g, v_layer, head_dim, n_kv_heads, n_kv_total);

            Q      = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
            K_full = ggml_permute(g, K_full, 0, 2, 1, 3);
            V_full = ggml_permute(g, V_full, 0, 2, 1, 3);

            // No mask needed: only n_kv+1 valid positions in the view
            const float scale = 1.0f / sqrtf((float)head_dim);
            ggml_tensor * attn_out = ggml_flash_attn_ext(g, Q, K_full, V_full,
                                                         nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            attn_out = ggml_reshape_2d(g, attn_out, D, 1);

            h = ggml_mul_mat(g, l.attn_out_proj_w, attn_out);
            attn_idx++;
        } else {
            // ShortConv decode: single token, use conv state cache
            // Conv state is [D, pad] = [D, 2] — maintained on CPU.
            // For decode, we pass the conv state + new token as input,
            // run a depthwise multiply, produce output.

            // in_proj: [D, 1] → [3D, 1]
            ggml_tensor * bcx = ggml_mul_mat(g, l.conv_in_proj_w, h);

            ggml_tensor * B  = ggml_cont(g, ggml_view_1d(g, bcx, D, 0));
            ggml_tensor * C  = ggml_cont(g, ggml_view_1d(g, bcx, D, (size_t)D * sizeof(float)));
            ggml_tensor * xi = ggml_cont(g, ggml_view_1d(g, bcx, D, (size_t)2 * D * sizeof(float)));

            // Bx = B * x
            ggml_tensor * Bx = ggml_mul(g, B, xi);

            // State columns [D, kernel_size-1] passed as input (conv cache)
            ggml_tensor * state_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, conv_k - 1);
            ggml_set_name(state_in, ("conv_state_" + std::to_string(il)).c_str());
            ggml_set_input(state_in);

            // Build window: concat state [D,2] with Bx [D,1] → [D, 3]
            ggml_tensor * Bx_2d = ggml_reshape_2d(g, Bx, D, 1);
            ggml_tensor * window = ggml_concat(g, state_in, Bx_2d, 1);  // [D, 3]

            // Depthwise conv: element-wise multiply with kernel [3, D] then sum over kernel dim
            // conv_conv_w is [kernel_size, D] or [kernel_size, 1, D]
            // We manually implement the pointwise depthwise conv:
            // output[d] = sum_k(window[d, k] * kernel[k, d])
            // Transpose window to [3, D] for element-wise
            ggml_tensor * wt = ggml_cont(g, ggml_transpose(g, window));  // [conv_k, D]

            // Kernel weight: reshape to [conv_k, D] if needed
            ggml_tensor * kern = l.conv_conv_w;
            if (kern->type != GGML_TYPE_F32) kern = ggml_cast(g, kern, GGML_TYPE_F32);
            kern = ggml_reshape_2d(g, kern, kern->ne[0], D);  // [conv_k, D]

            // Element-wise multiply
            ggml_tensor * prod = ggml_mul(g, wt, kern);  // [conv_k, D]

            // Sum over kernel dimension (dim 0) → [D]
            // ggml doesn't have a direct reduce_sum, so we sum manually using views
            // For kernel_size=3, just add the 3 rows
            ggml_tensor * r0 = ggml_view_1d(g, prod, D, 0);
            ggml_tensor * r1 = ggml_view_1d(g, prod, D, (size_t)D * sizeof(float));
            ggml_tensor * r2 = ggml_view_1d(g, prod, D, (size_t)2 * D * sizeof(float));
            ggml_tensor * conv_out = ggml_add(g, ggml_add(g, r0, r1), r2);

            // Also output Bx so we can update conv state after compute
            ggml_tensor * bx_out = ggml_reshape_1d(g, Bx, D);
            ggml_set_name(bx_out, ("bx_out_" + std::to_string(il)).c_str());
            ggml_set_output(bx_out);
            ggml_build_forward_expand(gf, bx_out);

            // y = C * conv_out
            ggml_tensor * y = ggml_mul(g, C, conv_out);
            h = ggml_mul_mat(g, l.conv_out_proj_w, y);
            conv_idx++;
        }

        x = ggml_add(g, residual, h);

        // FFN
        residual = x;
        h = lfm2_rms_norm(g, x, l.ffn_norm_w, eps);
        h = lfm2_swiglu(g, h, l.ff_w1, l.ff_w2, l.ff_w3);
        x = ggml_add(g, residual, h);
    }

    // Final norm + logits
    x = lfm2_rms_norm(g, x, c.m.embedding_norm_w, eps);
    ggml_tensor * lm_w = c.m.lm_head_w;
    x = ggml_mul_mat(g, lm_w, x);
    ggml_set_name(x, "logits");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    return gf;
}

// ============================================================================
// Generation (prefill + decode loop)
// ============================================================================

struct generate_result {
    std::vector<int32_t> token_ids;
    std::string text;
    std::vector<float> confidences;
};

static bool generate(ctx & c, const float * image_embeds, int n_image_tokens,
                     int embed_dim, const int32_t * prompt_ids, int n_prompt_tokens,
                     int max_new_tokens, generate_result & out) {
    const auto & lhp = c.m.lhp;
    const int D      = (int)lhp.hidden_size;
    const int V      = (int)lhp.vocab_size;
    const int n_layers = (int)lhp.n_layers;

    auto t_gen = steady_clock::now();

    // ── Step 1: Build spliced embeddings on CPU ──
    // Look up token embeddings, replace IMAGE tokens with projected image embeddings
    auto embed_w = to_f32(c.m.embed_tokens_w);
    std::vector<float> spliced((size_t)D * n_prompt_tokens);

    // Build spliced embeddings in ggml column-major layout: [D, n_tokens]
    // Element (d, t) at flat offset d + t * D.
    int img_pos = 0;
    for (int t = 0; t < n_prompt_tokens; t++) {
        int32_t tok = prompt_ids[t];
        if (tok == (int32_t)lhp.image_token_id && image_embeds && img_pos < n_image_tokens) {
            // Copy from image embeddings (row-major: [n_image_tokens, D])
            for (int d = 0; d < D; d++) {
                spliced[(size_t)t * D + d] =
                    image_embeds[(size_t)img_pos * embed_dim + d];
            }
            img_pos++;
        } else {
            // Token embedding: ggml embed_tokens [D, V], element (d, tok) = data[tok * D + d]
            if (tok >= 0 && tok < (int32_t)lhp.vocab_size) {
                for (int d = 0; d < D; d++) {
                    spliced[(size_t)t * D + d] =
                        embed_w[(size_t)tok * D + d];
                }
            }
        }
    }

    // ── Step 2: Allocate KV cache ──
    bool use_kv = !core_env::on("CRISPEMBED_NO_KV_CACHE");
    bool kv_ok = use_kv && alloc_kv_cache(c, n_prompt_tokens + max_new_tokens);

    // Init conv state
    init_conv_state(c);

    // ── Step 3: Prefill ──
    const int max_nodes = 16384;
    size_t meta_size = ggml_tensor_overhead() * max_nodes +
                       ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip{ meta_size, nullptr, true };
    ggml_context * g = ggml_init(ip);
    if (!g) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_tensor * logits_t = build_prefill_graph(c, g, gf, n_prompt_tokens,
                                                  n_image_tokens, kv_ok);

    ggml_backend_sched_reset(c.sched);
    if (!ggml_backend_sched_alloc_graph(c.sched, gf)) {
        fprintf(stderr, "[lfm2_vl] prefill graph alloc failed\n");
        ggml_free(g);
        return false;
    }

    // Set inputs
    ggml_tensor * emb_in = ggml_graph_get_tensor(gf, "emb_input");
    ggml_backend_tensor_set(emb_in, spliced.data(), 0, spliced.size() * sizeof(float));

    // Diff: spliced is now in ggml col-major layout [D, n_tokens] = flat[d + t*D]
    diff_stage(c, "llm_embed", spliced.data(), spliced.size());
    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] prompt: %d tokens (%d text + %d image), img_pos used=%d\n",
                n_prompt_tokens, n_prompt_tokens - n_image_tokens, n_image_tokens, img_pos);
    }

    // Position IDs
    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "positions");
    {
        std::vector<int32_t> pos_data(n_prompt_tokens);
        for (int i = 0; i < n_prompt_tokens; i++) pos_data[i] = i;
        ggml_backend_tensor_set(pos_in, pos_data.data(), 0,
                                n_prompt_tokens * sizeof(int32_t));
    }

    // Conv layer padding inputs (zeros)
    {
        const int pad = (int)lhp.conv_kernel - 1;
        std::vector<float> zeros((size_t)D * pad, 0.0f);
        for (int il = 0; il < n_layers; il++) {
            if (c.m.llm_layers[il].is_attention) continue;
            char name[64];
            snprintf(name, sizeof(name), "conv_pad_%d", il);
            ggml_tensor * pt = ggml_graph_get_tensor(gf, name);
            if (pt) ggml_backend_tensor_set(pt, zeros.data(), 0, zeros.size() * sizeof(float));
        }
    }

    // Causal attention masks: -inf above diagonal, 0 on and below (F16)
    {
        std::vector<uint16_t> mask_data((size_t)n_prompt_tokens * n_prompt_tokens, 0);
        uint16_t f16_neg_inf = 0xFC00;  // -inf in F16
        for (int r = 0; r < n_prompt_tokens; r++)
            for (int c2 = r + 1; c2 < n_prompt_tokens; c2++)
                mask_data[(size_t)r * n_prompt_tokens + c2] = f16_neg_inf;
        for (int il = 0; il < n_layers; il++) {
            if (!c.m.llm_layers[il].is_attention) continue;
            char mname[64];
            snprintf(mname, sizeof(mname), "causal_mask_%d", il);
            ggml_tensor * mt = ggml_graph_get_tensor(gf, mname);
            if (mt) ggml_backend_tensor_set(mt, mask_data.data(), 0,
                                             mask_data.size() * sizeof(uint16_t));
        }
    }

    // Compute prefill
    auto t_prefill = steady_clock::now();
    if (ggml_backend_sched_graph_compute(c.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[lfm2_vl] prefill compute failed\n");
        ggml_free(g);
        return false;
    }
    if (c.verbosity >= 1) {
        fprintf(stderr, "  prefill: %d tokens, %lld ms\n", n_prompt_tokens, ms_since(t_prefill));
    }

    // Read and diff per-layer intermediates
    for (int il = 0; il < std::min(4, n_layers); il++) {
        char lname[64];
        snprintf(lname, sizeof(lname), "llm_layer_%d", il);
        ggml_tensor * lt = ggml_graph_get_tensor(gf, lname);
        if (lt) {
            std::vector<float> layer_data((size_t)D * n_prompt_tokens);
            ggml_backend_tensor_get(lt, layer_data.data(), 0, layer_data.size() * sizeof(float));
            diff_stage(c, lname, layer_data.data(), layer_data.size());
        }
    }

    // Read logits and take argmax
    std::vector<float> logits_data(V);
    ggml_backend_tensor_get(logits_t, logits_data.data(), 0, V * sizeof(float));

    // Diff: compare logits against reference
    diff_stage(c, "llm_logits_last", logits_data.data(), logits_data.size());

    // Optionally skip conv state extraction (for debugging decode issues)
    bool skip_conv_state = core_env::on("LFM2_VL_ZERO_CONV_STATE");

    // Extract conv state from prefill: the last (kernel-1)=2 columns of Bx
    // at each conv layer are the conv state for decode.
    if (!skip_conv_state) {
        const int pad = (int)lhp.conv_kernel - 1;  // 2
        int conv_idx = 0;
        for (int il = 0; il < n_layers; il++) {
            if (c.m.llm_layers[il].is_attention) continue;
            char bx_name[64];
            snprintf(bx_name, sizeof(bx_name), "bx_%d", il);
            ggml_tensor * bx_t = ggml_graph_get_tensor(gf, bx_name);
            if (bx_t && conv_idx < (int)c.conv_state.size()) {
                // Bx is [D, n_tokens] in ggml col-major → element (d, t) at t*D+d
                // We need the last `pad` columns: t = n_prompt_tokens - pad .. n_prompt_tokens - 1
                std::vector<float> bx_data((size_t)D * n_prompt_tokens);
                ggml_backend_tensor_get(bx_t, bx_data.data(), 0, bx_data.size() * sizeof(float));
                for (int p = 0; p < pad; p++) {
                    int src_t = n_prompt_tokens - pad + p;
                    for (int d = 0; d < D; d++) {
                        c.conv_state[conv_idx][(size_t)p * D + d] = bx_data[(size_t)src_t * D + d];
                    }
                }
            }
            conv_idx++;
        }
        if (c.verbosity >= 1) {
            fprintf(stderr, "  conv state: extracted from %d layers\n", conv_idx);
        }
    } else {
        fprintf(stderr, "  conv state: ZEROED (LFM2_VL_ZERO_CONV_STATE=1)\n");
    }

    ggml_free(g);

    // Greedy argmax
    int no_repeat_ngram = 5;  // default no-repeat n-gram
    if (const char * e = getenv("LFM2_VL_NO_REPEAT_NGRAM")) no_repeat_ngram = atoi(e);

    int best_id = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++) {
        if (logits_data[v] > best_score) { best_score = logits_data[v]; best_id = v; }
    }
    // Confidence
    {
        float max_l = best_score;
        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) sum_exp += expf(logits_data[v] - max_l);
        out.confidences.push_back(expf(best_score - max_l) / sum_exp);
    }
    out.token_ids.push_back(best_id);

    int eos_id = (int)lhp.eos_id;
    if (best_id == eos_id || max_new_tokens <= 1) {
        if (c.verbosity >= 1) fprintf(stderr, "  gen: 1 token (hit EOS on prefill)\n");
        return true;
    }

    // ── Step 4: Decode loop ──
    int n_kv = n_prompt_tokens;
    int max_seq_kv = kv_ok ? c.kvc.max_seq : 0;

    // Attention mask
    std::vector<ggml_fp16_t> kv_mask_data;
    if (kv_ok) {
        kv_mask_data.resize((size_t)max_seq_kv, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i < n_prompt_tokens; i++)
            kv_mask_data[i] = ggml_fp32_to_fp16(0.0f);
    }

    for (int gen = 1; gen < max_new_tokens; gen++) {
        auto t_step = steady_clock::now();

        if (!kv_ok) {
            // Fallback: full recompute
            std::vector<int32_t> all_tokens(prompt_ids, prompt_ids + n_prompt_tokens);
            for (auto id : out.token_ids) all_tokens.push_back(id);

            // Re-splice embeddings
            int total = (int)all_tokens.size();
            std::vector<float> full_emb((size_t)D * total);
            int ip2 = 0;
            for (int t = 0; t < total; t++) {
                int32_t tok = all_tokens[t];
                if (tok == (int32_t)lhp.image_token_id && image_embeds && ip2 < n_image_tokens) {
                    for (int d = 0; d < D; d++)
                        full_emb[(size_t)d * total + t] = image_embeds[(size_t)ip2 * embed_dim + d];
                    ip2++;
                } else if (tok >= 0 && tok < (int32_t)lhp.vocab_size) {
                    for (int d = 0; d < D; d++)
                        full_emb[(size_t)d * total + t] = embed_w[(size_t)tok * D + d];
                }
            }

            ggml_context * g2 = ggml_init(ip);
            if (!g2) return false;
            ggml_cgraph * gf2 = ggml_new_graph_custom(g2, max_nodes, false);
            ggml_tensor * lt2 = build_prefill_graph(c, g2, gf2, total, n_image_tokens, false);

            ggml_backend_sched_reset(c.sched);
            if (!ggml_backend_sched_alloc_graph(c.sched, gf2)) {
                ggml_free(g2);
                return false;
            }

            ggml_tensor * e2 = ggml_graph_get_tensor(gf2, "emb_input");
            ggml_backend_tensor_set(e2, full_emb.data(), 0, full_emb.size() * sizeof(float));
            ggml_tensor * p2 = ggml_graph_get_tensor(gf2, "positions");
            {
                std::vector<int32_t> pd(total);
                for (int i = 0; i < total; i++) pd[i] = i;
                ggml_backend_tensor_set(p2, pd.data(), 0, total * sizeof(int32_t));
            }
            // Conv pads
            {
                const int pad = (int)lhp.conv_kernel - 1;
                std::vector<float> zeros((size_t)D * pad, 0.0f);
                for (int il = 0; il < n_layers; il++) {
                    if (c.m.llm_layers[il].is_attention) continue;
                    char name[64];
                    snprintf(name, sizeof(name), "conv_pad_%d", il);
                    ggml_tensor * pt = ggml_graph_get_tensor(gf2, name);
                    if (pt) ggml_backend_tensor_set(pt, zeros.data(), 0, zeros.size() * sizeof(float));
                }
            }

            if (ggml_backend_sched_graph_compute(c.sched, gf2) != GGML_STATUS_SUCCESS) {
                ggml_free(g2);
                return false;
            }
            ggml_backend_tensor_get(lt2, logits_data.data(), 0, V * sizeof(float));
            ggml_free(g2);
        } else {
            // KV-cached decode step
            ggml_context * g3 = ggml_init(ip);
            ggml_cgraph * gf3 = build_decode_step_graph(c, g3, n_kv, n_kv, max_seq_kv);

            ggml_backend_sched_reset(c.sched);
            if (!ggml_backend_sched_alloc_graph(c.sched, gf3)) {
                fprintf(stderr, "[lfm2_vl] decode step alloc failed\n");
                ggml_free(g3);
                return false;
            }

            // Set tok_emb: look up the token embedding for best_id
            if (gen == 1 && c.verbosity >= 1) {
                fprintf(stderr, "[lfm2_vl] decode step 0: input token=%d, n_kv=%d, pos=%d\n",
                        best_id, n_kv, n_kv);
                // Print conv state diagnostics
                if (!c.conv_state.empty() && c.conv_state[0].size() >= 5) {
                    fprintf(stderr, "[lfm2_vl] conv_state[0] first 5: ");
                    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", c.conv_state[0][i]);
                    // Also print norm and max
                    double norm = 0; float mx = 0;
                    for (float v : c.conv_state[0]) { norm += v*v; if (std::abs(v) > mx) mx = std::abs(v); }
                    fprintf(stderr, " norm=%.4f max=%.6f\n", std::sqrt(norm), mx);
                }
            }
            std::vector<float> tok_emb_data(D);
            for (int d = 0; d < D; d++)
                tok_emb_data[d] = embed_w[(size_t)best_id * D + d];
            ggml_tensor * te = ggml_graph_get_tensor(gf3, "tok_emb");
            ggml_backend_tensor_set(te, tok_emb_data.data(), 0, D * sizeof(float));

            // Position
            int32_t pos_val = n_kv;
            ggml_tensor * pi = ggml_graph_get_tensor(gf3, "pos");
            ggml_backend_tensor_set(pi, &pos_val, 0, sizeof(int32_t));

            // KV mask removed: decode reads only n_kv+1 valid entries

            // Conv state inputs
            int conv_layer = 0;
            for (int il = 0; il < n_layers; il++) {
                if (c.m.llm_layers[il].is_attention) continue;
                char name[64];
                snprintf(name, sizeof(name), "conv_state_%d", il);
                ggml_tensor * st = ggml_graph_get_tensor(gf3, name);
                if (st && conv_layer < c.n_conv_layers) {
                    ggml_backend_tensor_set(st, c.conv_state[conv_layer].data(), 0,
                                            c.conv_state[conv_layer].size() * sizeof(float));
                }
                conv_layer++;
            }

            // Compute
            if (ggml_backend_sched_graph_compute(c.sched, gf3) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[lfm2_vl] decode step compute failed\n");
                ggml_free(g3);
                return false;
            }

            // Read logits
            ggml_tensor * lt3 = ggml_graph_get_tensor(gf3, "logits");
            ggml_backend_tensor_get(lt3, logits_data.data(), 0, V * sizeof(float));

            if (gen == 1 && c.verbosity >= 1) {
                // Print decode step 0 argmax and top-3
                int am = 0; float amv = -INFINITY;
                for (int v = 0; v < V; v++) if (logits_data[v] > amv) { amv = logits_data[v]; am = v; }
                fprintf(stderr, "[lfm2_vl] decode step 0 argmax: %d (%.2f), expected 1870 ('son')\n", am, amv);
                // Check value at expected token
                fprintf(stderr, "[lfm2_vl] decode step 0 logit[1870]='son': %.2f\n", logits_data[1870]);
            }

            // Update conv state from bx_out tensors
            conv_layer = 0;
            for (int il = 0; il < n_layers; il++) {
                if (c.m.llm_layers[il].is_attention) continue;
                char name[64];
                snprintf(name, sizeof(name), "bx_out_%d", il);
                ggml_tensor * bx = ggml_graph_get_tensor(gf3, name);
                if (bx && conv_layer < c.n_conv_layers) {
                    const int pad = (int)lhp.conv_kernel - 1;
                    // Shift state left: col[0] = col[1], col[1] = new_bx
                    std::vector<float> new_bx(D);
                    ggml_backend_tensor_get(bx, new_bx.data(), 0, D * sizeof(float));
                    // state is [D * pad] laid out as [D, pad] column-major
                    // Shift: copy second column to first
                    if (pad >= 2) {
                        memcpy(c.conv_state[conv_layer].data(),
                               c.conv_state[conv_layer].data() + D,
                               (size_t)D * sizeof(float));
                    }
                    // Write new column
                    memcpy(c.conv_state[conv_layer].data() + (size_t)(pad - 1) * D,
                           new_bx.data(), (size_t)D * sizeof(float));
                }
                conv_layer++;
            }

            n_kv++;
            ggml_free(g3);
        }

        // Greedy argmax with no-repeat-ngram
        best_id = core_decode::argmax_no_repeat_ngram(logits_data.data(), V,
                                                       out.token_ids, no_repeat_ngram);
        best_score = logits_data[best_id];
        {
            float max_l = best_score;
            float sum_exp = 0.0f;
            for (int v = 0; v < V; v++) sum_exp += expf(logits_data[v] - max_l);
            out.confidences.push_back(expf(best_score - max_l) / sum_exp);
        }
        out.token_ids.push_back(best_id);

        if (c.verbosity >= 2) {
            fprintf(stderr, "  gen[%d]: token=%d (%.2f), %lld ms\n",
                    gen, best_id, best_score, ms_since(t_step));
        }

        if (best_id == eos_id) break;
    }

    if (c.verbosity >= 1) {
        fprintf(stderr, "  generate: %zu tokens, %lld ms total\n",
                out.token_ids.size(), ms_since(t_gen));
    }

    return true;
}

// ============================================================================
// Load + free
// ============================================================================

static bool load_model(ctx & c, const char * model_path, const char * mmproj_path,
                       int n_threads) {
    c.n_threads = n_threads;

    // License gate
    if (!core_env::on("CRISPEMBED_ACCEPT_LFM_LICENSE")) {
        fprintf(stderr,
                "[lfm2_vl] LFM2.5-VL is released under the LFM-1.0 license, which\n"
                "  includes a revenue cap for commercial use. Set the environment\n"
                "  variable CRISPEMBED_ACCEPT_LFM_LICENSE=1 to acknowledge this.\n"
                "  See: https://huggingface.co/LiquidAI/LFM2.5-VL-3B\n");
        return false;
    }

    // Load LLM hparams from the model GGUF
    if (!load_hparams(c, model_path)) {
        fprintf(stderr, "[lfm2_vl] failed to load LLM hparams from %s\n", model_path);
        return false;
    }

    // Load vision hparams from mmproj (or model if stacked)
    const char * vis_path = mmproj_path ? mmproj_path : model_path;
    if (!load_vision_hparams(c, vis_path)) {
        fprintf(stderr, "[lfm2_vl] failed to load vision hparams from %s\n", vis_path);
        return false;
    }

    if (c.verbosity >= 1) {
        const auto & vhp = c.m.vhp;
        const auto & lhp = c.m.lhp;
        fprintf(stderr, "[lfm2_vl] vision: %u layers, %ud, %u heads, patch=%u\n",
                vhp.depth, vhp.hidden_size, vhp.num_heads, vhp.patch_size);
        fprintf(stderr, "[lfm2_vl] llm: %u layers, %ud, %u/%u heads, ff=%u\n",
                lhp.n_layers, lhp.hidden_size, lhp.n_heads, lhp.n_kv_heads, lhp.ff_size);
        fprintf(stderr, "[lfm2_vl] layer types: %s\n", lhp.layer_types.c_str());
    }

    // Init backend
    bool force_cpu = core_env::on("LFM2_VL_FORCE_CPU");
    c.backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!c.backend) c.backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(c.backend))
        ggml_backend_cpu_set_n_threads(c.backend, n_threads);
    c.backend_cpu = ggml_backend_is_cpu(c.backend) ? nullptr : ggml_backend_cpu_init();
    if (c.backend_cpu) ggml_backend_cpu_set_n_threads(c.backend_cpu, n_threads);

    // Compute meta scratch
    constexpr int kGraphCapacity = 16384;
    c.compute_meta.resize(ggml_tensor_overhead() * kGraphCapacity +
                          ggml_graph_overhead_custom(kGraphCapacity, false));

    // Scheduler
    std::vector<ggml_backend_t> backends;
    backends.push_back(c.backend);
    if (c.backend_cpu && c.backend_cpu != c.backend)
        backends.push_back(c.backend_cpu);
    c.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                      (int)backends.size(), kGraphCapacity, false, false);

    // Load LLM tensors
    if (!load_llm_tensors(c, model_path)) {
        fprintf(stderr, "[lfm2_vl] failed to load LLM tensors\n");
        return false;
    }

    // Load vision tensors (from mmproj or same file)
    if (!load_vision_tensors(c, vis_path)) {
        fprintf(stderr, "[lfm2_vl] failed to load vision tensors\n");
        return false;
    }

    // Load tokenizer
    if (!load_tokenizer(c, model_path)) {
        fprintf(stderr, "[lfm2_vl] warning: tokenizer not loaded\n");
    }

    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] loaded successfully, vocab=%u, tokenizer=%s\n",
                c.m.lhp.vocab_size,
                c.id_to_piece.empty() ? "none" : "ok");
    }

    // Diff reference (parity loop)
    const char * diff_ref = getenv("LFM2_VL_DIFF_REF");
    if (diff_ref && diff_ref[0]) {
        if (c.diff_ref.load(diff_ref)) {
            c.has_diff_ref = true;
            fprintf(stderr, "[lfm2_vl] diff reference loaded: %s\n", diff_ref);
        } else {
            fprintf(stderr, "[lfm2_vl] WARNING: diff reference failed to load: %s\n", diff_ref);
        }
    }

    return true;
}

static void free_model(ctx & c) {
    free_kv_cache(c);
    if (c.sched) { ggml_backend_sched_free(c.sched); c.sched = nullptr; }
    core_gguf::release_weight_buffer(c.mmproj_buf);
    if (c.mmproj_ctx) { ggml_free(c.mmproj_ctx); c.mmproj_ctx = nullptr; }
    core_gguf::release_weight_buffer(c.model_buf);
    if (c.model_ctx) { ggml_free(c.model_ctx); c.model_ctx = nullptr; }
    if (c.backend_cpu) { ggml_backend_free(c.backend_cpu); c.backend_cpu = nullptr; }
    if (c.backend) { ggml_backend_free(c.backend); c.backend = nullptr; }
}

} // anonymous namespace
} // namespace lfm2_vl

// ============================================================================
// C API wrapper
// ============================================================================

struct lfm2_vl_ocr_context {
    lfm2_vl::ctx inner;
    std::string prompt = "OCR this image. Output the text content.";
    int max_tokens = 2048;
    std::string last_result;
    std::vector<float> char_confidences;
};

// ── Build chat-format token IDs ──

// Look up a special token by its string representation.
static int32_t special_tok(const lfm2_vl::ctx & c, const std::string & s) {
    auto it = c.token_to_id.find(s);
    return it != c.token_to_id.end() ? it->second : -1;
}

static std::vector<int32_t> build_token_ids(lfm2_vl_ocr_context * ctx, int n_image_tokens) {
    // LFM2.5-VL chat template:
    // <|startoftext|><|im_start|>user\n<image>...<image>PROMPT<|im_end|>\n
    // <|im_start|>assistant\n
    //
    // Special tokens (<|im_start|>, <|im_end|>, etc.) must be inserted as
    // single token IDs, NOT passed through the BPE tokenizer which would
    // split them character by character.
    auto & c = ctx->inner;
    const auto & lhp = c.m.lhp;

    int32_t bos_id      = (int32_t)lhp.bos_id;       // <|startoftext|> = 124894
    int32_t im_start_id = special_tok(c, "<|im_start|>");
    int32_t im_end_id   = special_tok(c, "<|im_end|>");
    int32_t image_id    = (int32_t)lhp.image_token_id;  // 124907
    int32_t nl_id       = special_tok(c, "\n");

    // LFM2.5-VL known special token IDs (from tokenizer.json added_tokens)
    if (im_start_id < 0) im_start_id = 124899;
    if (im_end_id < 0)   im_end_id   = 124900;
    if (nl_id < 0) {
        // newline might be a regular token
        auto nl_ids = lfm2_vl::tokenize(c, "\n");
        if (!nl_ids.empty()) nl_id = nl_ids[0];
    }

    // Tokenize just the plain text portions
    auto user_ids    = lfm2_vl::tokenize(c, "user");
    auto prompt_ids  = lfm2_vl::tokenize(c, ctx->prompt);
    auto assist_ids  = lfm2_vl::tokenize(c, "assistant");

    std::vector<int32_t> ids;
    ids.reserve(n_image_tokens + 30);

    // <|startoftext|>
    ids.push_back(bos_id);
    // <|im_start|>
    ids.push_back(im_start_id);
    // user\n
    for (auto id : user_ids) ids.push_back(id);
    if (nl_id >= 0) ids.push_back(nl_id);
    // <|image_start|> <image>*N <|image_end|>  (use_image_special_tokens=true)
    int32_t img_start_id = 125009;  // <|image_start|>
    int32_t img_end_id   = 125010;  // <|image_end|>
    ids.push_back(img_start_id);
    for (int i = 0; i < n_image_tokens; i++)
        ids.push_back(image_id);
    ids.push_back(img_end_id);
    // prompt_text
    for (auto id : prompt_ids) ids.push_back(id);
    // <|im_end|>\n
    ids.push_back(im_end_id);
    if (nl_id >= 0) ids.push_back(nl_id);
    // <|im_start|>assistant\n
    ids.push_back(im_start_id);
    for (auto id : assist_ids) ids.push_back(id);
    if (nl_id >= 0) ids.push_back(nl_id);

    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] token_ids: %zu total (%d image + %zu text)\n",
                ids.size(), n_image_tokens, ids.size() - n_image_tokens);
        fprintf(stderr, "[lfm2_vl] first 10 ids: ");
        for (int i = 0; i < std::min(10, (int)ids.size()); i++)
            fprintf(stderr, "%d ", ids[i]);
        fprintf(stderr, "...\n");
        fprintf(stderr, "[lfm2_vl] user=%zu prompt=%zu assist=%zu nl=%d\n",
                user_ids.size(), prompt_ids.size(), assist_ids.size(), nl_id);
        fprintf(stderr, "[lfm2_vl] prompt ids: ");
        for (auto id : prompt_ids) fprintf(stderr, "%d ", id);
        fprintf(stderr, "\n");
        // Quick tokenize test
        auto ocr_ids = lfm2_vl::tokenize(c, "OCR");
        fprintf(stderr, "[lfm2_vl] tokenize('OCR'): ");
        for (auto id : ocr_ids) fprintf(stderr, "%d ", id);
        fprintf(stderr, "(%zu tokens, expected: 55 6193)\n", ocr_ids.size());
    }

    return ids;
}

// ── Pipeline: preprocess → vision → generate ──

static const char * run_pipeline(lfm2_vl_ocr_context * ctx,
                                 const uint8_t * rgb, int width, int height,
                                 int channels, int * out_len) {
    auto & c = ctx->inner;
    auto t0 = lfm2_vl::steady_clock::now();

    // 1. Preprocess image
    lfm2_vl::image_patches patches;
    if (!lfm2_vl::preprocess_image(rgb, height, width, channels, c.m.vhp, patches)) {
        return nullptr;
    }
    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] %dx%d → %dx%d patches, %lld ms preprocess\n",
                width, height, patches.h_patches, patches.w_patches,
                lfm2_vl::ms_since(t0));
    }

    // 2. Vision encoder + projector
    std::vector<float> image_embeds;
    int n_image_tokens = 0, embed_dim = 0;
    if (!lfm2_vl::encode_vision(c, patches, image_embeds, n_image_tokens, embed_dim)) {
        fprintf(stderr, "[lfm2_vl] vision encoder failed\n");
        return nullptr;
    }

    // 3. Build token IDs
    auto token_ids = build_token_ids(ctx, n_image_tokens);

    // 4. Generate
    lfm2_vl::generate_result gen;
    if (!lfm2_vl::generate(c, image_embeds.data(), n_image_tokens, embed_dim,
                            token_ids.data(), (int)token_ids.size(),
                            ctx->max_tokens, gen)) {
        fprintf(stderr, "[lfm2_vl] generation failed\n");
        return nullptr;
    }

    ctx->char_confidences = std::move(gen.confidences);

    // 5. Decode token IDs to text
    std::vector<int32_t> decode_ids;
    decode_ids.reserve(gen.token_ids.size());
    int eos_id = (int)c.m.lhp.eos_id;
    for (int32_t id : gen.token_ids) {
        if (id == eos_id || id == (int32_t)c.m.lhp.bos_id ||
            id == (int32_t)c.m.lhp.pad_id)
            continue;
        decode_ids.push_back(id);
    }

    if (!c.id_to_piece.empty()) {
        ctx->last_result = lfm2_vl::decode_tokens(c, decode_ids);
    } else {
        ctx->last_result.clear();
        for (size_t i = 0; i < decode_ids.size(); i++) {
            if (i > 0) ctx->last_result += ",";
            ctx->last_result += std::to_string(decode_ids[i]);
        }
    }

    if (c.verbosity >= 1) {
        fprintf(stderr, "[lfm2_vl] total pipeline: %lld ms, output: %zu chars\n",
                lfm2_vl::ms_since(t0), ctx->last_result.size());
    }

    if (out_len) *out_len = (int)ctx->last_result.size();
    return ctx->last_result.c_str();
}

// ============================================================================
// C API functions
// ============================================================================

lfm2_vl_ocr_context * lfm2_vl_ocr_init_split(const char * model_path,
                                               const char * mmproj_path,
                                               int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new lfm2_vl_ocr_context();
    if (!lfm2_vl::load_model(ctx->inner, model_path, mmproj_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

lfm2_vl_ocr_context * lfm2_vl_ocr_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    // Try to auto-discover mmproj sibling file.
    // Convention: mmproj-<BaseName>-F16.gguf or mmproj-<BaseName>-Q8_0.gguf
    // in the same directory as the model.
    std::string path(model_path);
    std::string dir, base;
    auto slash = path.find_last_of("/\\");
    if (slash != std::string::npos) {
        dir = path.substr(0, slash + 1);
        base = path.substr(slash + 1);
    } else {
        dir = "";
        base = path;
    }
    // Try common mmproj patterns
    const char * mmproj_path = nullptr;
    std::string mmproj;
    for (const char * suffix : { "F16", "Q8_0", "BF16" }) {
        // Pattern: mmproj-<ModelBaseName>-<suffix>.gguf where ModelBaseName
        // is derived from the LLM filename (strip quant suffix)
        // e.g. LFM2.5-VL-3B-Q4_K_M.gguf → mmproj-LFM2.5-VL-3B-F16.gguf
        // Find the model base: strip from last dash-uppercase-quant pattern
        std::string model_base = base;
        auto dash = model_base.rfind('-');
        if (dash != std::string::npos) {
            model_base = model_base.substr(0, dash);
        }
        // Strip .gguf if present
        auto dot = model_base.rfind(".gguf");
        if (dot != std::string::npos) model_base = model_base.substr(0, dot);
        mmproj = dir + "mmproj-" + model_base + "-" + suffix + ".gguf";
        FILE * f = fopen(mmproj.c_str(), "rb");
        if (f) { fclose(f); mmproj_path = mmproj.c_str(); break; }
    }
    if (mmproj_path) {
        fprintf(stderr, "[lfm2_vl] auto-discovered mmproj: %s\n", mmproj_path);
    }
    return lfm2_vl_ocr_init_split(model_path, mmproj_path, n_threads);
}

void lfm2_vl_ocr_free(lfm2_vl_ocr_context * ctx) {
    if (ctx) {
        lfm2_vl::free_model(ctx->inner);
        delete ctx;
    }
}

void lfm2_vl_ocr_set_prompt(lfm2_vl_ocr_context * ctx, const char * prompt) {
    if (ctx && prompt) ctx->prompt = prompt;
}

void lfm2_vl_ocr_set_max_tokens(lfm2_vl_ocr_context * ctx, int max_tokens) {
    if (ctx && max_tokens > 0) ctx->max_tokens = max_tokens;
}

const char * lfm2_vl_ocr_recognize_raw(lfm2_vl_ocr_context * ctx,
                                        const uint8_t * pixel_bytes,
                                        int width, int height, int channels,
                                        int * out_len) {
    if (!ctx || !pixel_bytes || width <= 0 || height <= 0) return nullptr;
    return run_pipeline(ctx, pixel_bytes, width, height, channels, out_len);
}

const char * lfm2_vl_ocr_recognize(lfm2_vl_ocr_context * ctx,
                                    const float * pixels,
                                    int width, int height, int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    // Convert grayscale float [0,1] to uint8 RGB
    std::vector<uint8_t> rgb((size_t)width * height * 3);
    for (int i = 0; i < width * height; i++) {
        uint8_t v = (uint8_t)(pixels[i] * 255.0f + 0.5f);
        rgb[(size_t)i * 3 + 0] = v;
        rgb[(size_t)i * 3 + 1] = v;
        rgb[(size_t)i * 3 + 2] = v;
    }

    return run_pipeline(ctx, rgb.data(), width, height, 3, out_len);
}

const float * lfm2_vl_ocr_confidences(const lfm2_vl_ocr_context * ctx,
                                       int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float lfm2_vl_ocr_mean_confidence(const lfm2_vl_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float cf : ctx->char_confidences) sum += cf;
    return (float)(sum / ctx->char_confidences.size());
}
