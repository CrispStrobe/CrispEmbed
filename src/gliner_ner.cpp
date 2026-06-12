// gliner_ner.cpp — GLiNER zero-shot NER via LFM2.5 bidirectional backbone.
//
// Architecture (from SauerkrautLM-LFM2.5-GLiNER):
//   1. BPE tokenize: <<ENT>> label1 <<SEP>> label2 <<SEP>> ... <<SEP>> text
//   2. LFM2.5 bidirectional backbone (16L, hidden=1024, ShortConv+GQA)
//   3. Layer fusion: learned attention-weighted sum of all layer outputs
//   4. BiLSTM: 1-layer bidirectional (hidden=512) → 1024-dim output
//   5. GLiNER head:
//      - Extract entity type reps at <<ENT>> positions → prompt_rep MLP
//      - SpanMarkerV1: project start/end/first → concat → ReLU → project
//      - Score: dot(span_rep, prompt_rep) * exp(log_temperature)
//      - Sigmoid → threshold → greedy decode
//
// Ported from CrispASR/src/lfm2_audio.cpp (LFM2 backbone) with:
//   - Bidirectional attention (no causal mask)
//   - Symmetric conv padding (center-pad instead of left-pad)

#include "gliner_ner.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/gguf_loader.h"
#include "core/bpe.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Debug logging
static bool g_debug = false;
#define GDBG(fmt, ...) do { if (g_debug) fprintf(stderr, "[gliner] " fmt "\n", ##__VA_ARGS__); } while (0)

// ============================================================================
// Hyperparameters
// ============================================================================

struct gliner_hparams {
    uint32_t hidden_size   = 1024;
    uint32_t n_layers      = 16;
    uint32_t n_heads       = 16;
    uint32_t n_kv_heads    = 8;
    uint32_t head_dim      = 64;
    uint32_t ff_dim        = 4608;
    uint32_t conv_kernel   = 3;
    float    rope_theta    = 1000000.0f;
    std::string layer_types;  // "ccaccaccacacacac"
    uint32_t vocab_size    = 64404;

    // GLiNER-specific
    uint32_t max_width     = 12;
    uint32_t ent_token_id  = 64402;
    uint32_t sep_token_id  = 64403;
};

// ============================================================================
// Per-layer weights (same structure as CrispASR lfm2_layer_weights)
// ============================================================================

struct gliner_layer_weights {
    ggml_tensor * operator_norm_w = nullptr;
    ggml_tensor * ffn_norm_w      = nullptr;
    ggml_tensor * ff_w1 = nullptr, * ff_w2 = nullptr, * ff_w3 = nullptr;

    bool is_attention = false;

    // Conv layers
    ggml_tensor * conv_conv_w     = nullptr;  // [hidden, 1, kernel]
    ggml_tensor * conv_in_proj_w  = nullptr;  // [3*hidden, hidden]
    ggml_tensor * conv_out_proj_w = nullptr;  // [hidden, hidden]

    // Attention layers
    ggml_tensor * attn_q_proj_w   = nullptr;
    ggml_tensor * attn_k_proj_w   = nullptr;
    ggml_tensor * attn_v_proj_w   = nullptr;
    ggml_tensor * attn_out_proj_w = nullptr;
    ggml_tensor * attn_q_ln_w    = nullptr;
    ggml_tensor * attn_k_ln_w    = nullptr;
};

// ============================================================================
// Model weights
// ============================================================================

struct gliner_model {
    gliner_hparams hparams;

    // LFM2 backbone
    ggml_tensor * embed_tokens_w    = nullptr;  // [vocab, hidden]
    ggml_tensor * embedding_norm_w  = nullptr;  // [hidden]
    std::vector<gliner_layer_weights> layers;

    // Layer fuser (attention-weighted sum of all layer outputs)
    ggml_tensor * fuser_squeeze_w      = nullptr;  // [1, hidden]
    ggml_tensor * fuser_squeeze_b      = nullptr;  // [1]
    ggml_tensor * fuser_W1_w           = nullptr;  // [8, n_layers]
    ggml_tensor * fuser_W1_b           = nullptr;  // [8]
    ggml_tensor * fuser_W2_w           = nullptr;  // [n_layers, 8]
    ggml_tensor * fuser_W2_b           = nullptr;  // [n_layers]
    ggml_tensor * fuser_output_proj_w  = nullptr;  // [hidden, hidden]
    ggml_tensor * fuser_output_proj_b  = nullptr;  // [hidden]

    // BiLSTM (1-layer bidirectional, hidden=512)
    ggml_tensor * lstm_weight_ih_l0     = nullptr;  // [4*512, 1024]
    ggml_tensor * lstm_bias_ih_l0       = nullptr;  // [4*512]
    ggml_tensor * lstm_weight_hh_l0     = nullptr;  // [4*512, 512]
    ggml_tensor * lstm_bias_hh_l0       = nullptr;  // [4*512]
    ggml_tensor * lstm_weight_ih_l0_rev = nullptr;
    ggml_tensor * lstm_bias_ih_l0_rev   = nullptr;
    ggml_tensor * lstm_weight_hh_l0_rev = nullptr;
    ggml_tensor * lstm_bias_hh_l0_rev   = nullptr;

    // GLiNER span representation (markerV1)
    // Each is a 2-layer MLP: Linear(1024,4096) → ReLU → Linear(4096,1024)
    ggml_tensor * span_proj_start_0_w = nullptr, * span_proj_start_0_b = nullptr;
    ggml_tensor * span_proj_start_3_w = nullptr, * span_proj_start_3_b = nullptr;
    ggml_tensor * span_proj_end_0_w   = nullptr, * span_proj_end_0_b   = nullptr;
    ggml_tensor * span_proj_end_3_w   = nullptr, * span_proj_end_3_b   = nullptr;
    ggml_tensor * span_proj_first_0_w = nullptr, * span_proj_first_0_b = nullptr;
    ggml_tensor * span_proj_first_3_w = nullptr, * span_proj_first_3_b = nullptr;
    // out_project: Linear(3*1024, 4096) → ReLU → Linear(4096, 1024)
    ggml_tensor * span_out_proj_0_w   = nullptr, * span_out_proj_0_b   = nullptr;
    ggml_tensor * span_out_proj_3_w   = nullptr, * span_out_proj_3_b   = nullptr;

    // Prompt/entity representation MLP
    ggml_tensor * prompt_rep_0_w = nullptr, * prompt_rep_0_b = nullptr;
    ggml_tensor * prompt_rep_3_w = nullptr, * prompt_rep_3_b = nullptr;

    // Scorer
    ggml_tensor * scorer_log_temp = nullptr;  // scalar

    // Model memory
    ggml_context       * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    // BPE tokenizer maps (built from GGUF arrays)
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

// ============================================================================
// Context
// ============================================================================

struct gliner_context {
    gliner_model model;
    int n_threads    = 4;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    // Reusable compute buffer
    std::vector<uint8_t> compute_meta;

    // Output storage (valid until next call)
    std::vector<gliner_ner_entity> result_entities;
    std::vector<std::string>       result_texts;
    std::vector<std::string>       result_labels;
};

// ============================================================================
// Model loading
// ============================================================================

static bool load_model(gliner_model & model, const char * path, ggml_backend_t backend) {
    GDBG("loading model from %s", path);

    // --- Pass 1: metadata ---
    gguf_context * gctx = core_gguf::open_metadata(path);
    if (!gctx) {
        fprintf(stderr, "[gliner] failed to open GGUF: %s\n", path);
        return false;
    }

    auto & hp = model.hparams;
    hp.hidden_size  = core_gguf::kv_u32(gctx, "gliner.hidden_size", 1024);
    hp.n_layers     = core_gguf::kv_u32(gctx, "gliner.n_layers", 16);
    hp.n_heads      = core_gguf::kv_u32(gctx, "gliner.n_heads", 16);
    hp.n_kv_heads   = core_gguf::kv_u32(gctx, "gliner.n_kv_heads", 8);
    hp.head_dim     = core_gguf::kv_u32(gctx, "gliner.head_dim", 64);
    hp.ff_dim       = core_gguf::kv_u32(gctx, "gliner.ff_dim", 4608);
    hp.conv_kernel  = core_gguf::kv_u32(gctx, "gliner.conv_kernel", 3);
    hp.rope_theta   = core_gguf::kv_f32(gctx, "gliner.rope_theta", 1000000.0f);
    hp.layer_types  = core_gguf::kv_str(gctx, "gliner.layer_types", "ccaccaccacacacac");
    hp.vocab_size   = core_gguf::kv_u32(gctx, "gliner.vocab_size", 64404);
    hp.max_width    = core_gguf::kv_u32(gctx, "gliner.max_width", 12);
    hp.ent_token_id = core_gguf::kv_u32(gctx, "gliner.ent_token_id", 64402);
    hp.sep_token_id = core_gguf::kv_u32(gctx, "gliner.sep_token_id", 64403);

    GDBG("  hidden=%u, layers=%u, heads=%u/%u, ff=%u",
         hp.hidden_size, hp.n_layers, hp.n_heads, hp.n_kv_heads, hp.ff_dim);
    GDBG("  layer_types=%s, vocab=%u, max_width=%u",
         hp.layer_types.c_str(), hp.vocab_size, hp.max_width);

    // Load BPE tokenizer
    auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    if (tokens.empty()) {
        fprintf(stderr, "[gliner] no tokenizer.tokens in GGUF\n");
        core_gguf::free_metadata(gctx);
        return false;
    }
    // Build token_to_id map
    for (size_t i = 0; i < tokens.size(); i++)
        model.token_to_id[tokens[i]] = (int32_t)i;

    // Load merges from newline-separated blob string
    std::string merges_blob = core_gguf::kv_str(gctx, "tokenizer.merges_blob", "");
    if (!merges_blob.empty()) {
        int rank = 0;
        size_t pos = 0;
        while (pos < merges_blob.size()) {
            size_t nl = merges_blob.find('\n', pos);
            if (nl == std::string::npos) nl = merges_blob.size();
            std::string merge = merges_blob.substr(pos, nl - pos);
            if (!merge.empty())
                model.merge_rank[merge] = rank++;
            pos = nl + 1;
        }
        GDBG("  tokenizer: %zu tokens, %d merges", tokens.size(), rank);
    } else {
        GDBG("  tokenizer: %zu tokens, no merges", tokens.size());
    }

    core_gguf::free_metadata(gctx);

    // --- Pass 2: weights ---
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "gliner", wl)) {
        fprintf(stderr, "[gliner] failed to load weights from: %s\n", path);
        return false;
    }
    model.ctx     = wl.ctx;
    model.buf     = wl.buf;
    model.tensors = wl.tensors;

    // Convenience macro
    auto T = [&](const char * name) -> ggml_tensor * {
        return core_gguf::try_get(model.tensors, name);
    };
    auto R = [&](const char * name) -> ggml_tensor * {
        return core_gguf::require(model.tensors, name, "gliner");
    };

    // LFM2 backbone
    model.embed_tokens_w   = R("lfm.embed_tokens.weight");
    model.embedding_norm_w = R("lfm.embedding_norm.weight");

    model.layers.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        auto & l = model.layers[i];
        auto ln = [&](const char * suffix) {
            char name[128];
            snprintf(name, sizeof(name), "lfm.layers.%u.%s", i, suffix);
            return R(name);
        };
        auto lt = [&](const char * suffix) {
            char name[128];
            snprintf(name, sizeof(name), "lfm.layers.%u.%s", i, suffix);
            return T(name);
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

    // Layer fuser
    model.fuser_squeeze_w     = R("fuser.squeeze.weight");
    model.fuser_squeeze_b     = R("fuser.squeeze.bias");
    model.fuser_W1_w          = R("fuser.W1.weight");
    model.fuser_W1_b          = R("fuser.W1.bias");
    model.fuser_W2_w          = R("fuser.W2.weight");
    model.fuser_W2_b          = R("fuser.W2.bias");
    model.fuser_output_proj_w = R("fuser.output_projection.weight");
    model.fuser_output_proj_b = R("fuser.output_projection.bias");

    // BiLSTM
    model.lstm_weight_ih_l0     = R("lstm.weight_ih_l0");
    model.lstm_bias_ih_l0       = R("lstm.bias_ih_l0");
    model.lstm_weight_hh_l0     = R("lstm.weight_hh_l0");
    model.lstm_bias_hh_l0       = R("lstm.bias_hh_l0");
    model.lstm_weight_ih_l0_rev = R("lstm.weight_ih_l0_reverse");
    model.lstm_bias_ih_l0_rev   = R("lstm.bias_ih_l0_reverse");
    model.lstm_weight_hh_l0_rev = R("lstm.weight_hh_l0_reverse");
    model.lstm_bias_hh_l0_rev   = R("lstm.bias_hh_l0_reverse");

    // GLiNER span representation
    model.span_proj_start_0_w = R("span.project_start.0.weight");
    model.span_proj_start_0_b = R("span.project_start.0.bias");
    model.span_proj_start_3_w = R("span.project_start.3.weight");
    model.span_proj_start_3_b = R("span.project_start.3.bias");
    model.span_proj_end_0_w   = R("span.project_end.0.weight");
    model.span_proj_end_0_b   = R("span.project_end.0.bias");
    model.span_proj_end_3_w   = R("span.project_end.3.weight");
    model.span_proj_end_3_b   = R("span.project_end.3.bias");
    model.span_proj_first_0_w = R("span.project_first.0.weight");
    model.span_proj_first_0_b = R("span.project_first.0.bias");
    model.span_proj_first_3_w = R("span.project_first.3.weight");
    model.span_proj_first_3_b = R("span.project_first.3.bias");
    model.span_out_proj_0_w   = R("span.out_project.0.weight");
    model.span_out_proj_0_b   = R("span.out_project.0.bias");
    model.span_out_proj_3_w   = R("span.out_project.3.weight");
    model.span_out_proj_3_b   = R("span.out_project.3.bias");

    // Prompt representation
    model.prompt_rep_0_w = R("prompt_rep.0.weight");
    model.prompt_rep_0_b = R("prompt_rep.0.bias");
    model.prompt_rep_3_w = R("prompt_rep.3.weight");
    model.prompt_rep_3_b = R("prompt_rep.3.bias");

    // Scorer
    model.scorer_log_temp = R("scorer.log_temperature");

    GDBG("model loaded: %zu tensors", model.tensors.size());
    return true;
}

// ============================================================================
// LFM2 backbone building blocks (ported from CrispASR/src/lfm2_audio.cpp)
// Modified for bidirectional: no causal mask, symmetric conv padding
// ============================================================================

static ggml_tensor * gliner_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

static ggml_tensor * gliner_swiglu_ffn(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * w1, ggml_tensor * w2,
                                       ggml_tensor * w3) {
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w1, x));
    ggml_tensor * up   = ggml_mul_mat(ctx, w3, x);
    return ggml_mul_mat(ctx, w2, ggml_mul(ctx, gate, up));
}

// Bidirectional ShortConv (center-padded, not causal left-padded)
static ggml_tensor * gliner_short_conv(ggml_context * ctx, ggml_tensor * x,
                                       const gliner_layer_weights & w,
                                       int hidden, int T) {
    // x: (hidden, T)
    // in_proj: hidden → 3*hidden
    ggml_tensor * bcx = ggml_mul_mat(ctx, w.conv_in_proj_w, x);  // (3*hidden, T)

    // Split into B, C, x_inner
    ggml_tensor * B_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], 0);
    ggml_tensor * C_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1],
                                        hidden * sizeof(float));
    ggml_tensor * x_inner = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1],
                                         2 * hidden * sizeof(float));

    // Bx = B * x_inner (element-wise)
    ggml_tensor * Bx = ggml_mul(ctx, ggml_cont(ctx, B_part),
                                ggml_cont(ctx, x_inner));

    // Symmetric depthwise conv1d: kernel=3, pad=(K-1)/2=1 on each side
    // NOTE: ggml im2col (used by conv_1d_dw) requires F16 kernel weights.
    const int K = 3;
    ggml_tensor * conv_w = ggml_cast(ctx, w.conv_conv_w, GGML_TYPE_F16);
    ggml_tensor * Bx_t = ggml_cont(ctx, ggml_transpose(ctx, Bx));  // (T, hidden)
    ggml_tensor * conv_out = ggml_conv_1d_dw(ctx, conv_w, Bx_t,
                                             /*stride=*/1,
                                             /*pad=*/(K - 1) / 2,
                                             /*dilation=*/1);
    // For symmetric pad with K=3, pad=1: T_out = T + 2*1 - 3 + 1 = T
    // No trimming needed
    int T_conv = (int)conv_out->ne[0];
    if (T_conv > T) {
        // Safety: trim to T if needed
        conv_out = ggml_view_2d(ctx, conv_out, T, hidden, conv_out->nb[1], 0);
    }
    conv_out = ggml_cont(ctx, ggml_transpose(ctx, conv_out));  // (hidden, T)

    // y = C * conv_out
    ggml_tensor * y = ggml_mul(ctx, ggml_cont(ctx, C_part),
                               ggml_cont(ctx, conv_out));

    // out_proj
    return ggml_mul_mat(ctx, w.conv_out_proj_w, y);
}

// Track position tensors for filling after graph allocation
static thread_local std::vector<ggml_tensor *> g_pos_tensors;

// Bidirectional GQA attention (no causal mask)
static ggml_tensor * gliner_gqa_attention(ggml_context * ctx, ggml_tensor * x,
                                          const gliner_layer_weights & w,
                                          int hidden, int n_heads, int n_kv_heads,
                                          int head_dim, int T, float rope_theta) {
    // Q, K, V projections
    ggml_tensor * Q = ggml_mul_mat(ctx, w.attn_q_proj_w, x);
    ggml_tensor * K = ggml_mul_mat(ctx, w.attn_k_proj_w, x);
    ggml_tensor * V = ggml_mul_mat(ctx, w.attn_v_proj_w, x);

    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, T);
    V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, T);

    // Per-head QK RMSNorm
    Q = ggml_rms_norm(ctx, Q, 1e-5f);
    Q = ggml_mul(ctx, Q, w.attn_q_ln_w);
    K = ggml_rms_norm(ctx, K, 1e-5f);
    K = ggml_mul(ctx, K, w.attn_k_ln_w);

    // Positions [0, 1, ..., T-1]
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    g_pos_tensors.push_back(positions);

    // RoPE (NEOX = split-rotate, theta=1e6)
    Q = ggml_rope_ext(ctx, Q, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 0, rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr, head_dim,
                      GGML_ROPE_TYPE_NEOX, 0, rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute for flash_attn_ext: (hd, T, n_heads)
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // BIDIRECTIONAL: no mask (pass nullptr for full attention)
    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q, K, V,
                                             /*mask=*/nullptr,
                                             scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, hidden, T);

    return ggml_mul_mat(ctx, w.attn_out_proj_w, attn);
}

// One LFM2 bidirectional layer: RMSNorm → op → residual → RMSNorm → SwiGLU → residual
static ggml_tensor * gliner_build_layer(ggml_context * ctx, ggml_tensor * x,
                                        const gliner_layer_weights & w,
                                        int hidden, int n_heads, int n_kv_heads,
                                        int head_dim, int T, float norm_eps,
                                        float rope_theta) {
    ggml_tensor * residual = x;

    ggml_tensor * h = gliner_rms_norm(ctx, x, w.operator_norm_w, norm_eps);

    if (w.is_attention) {
        h = gliner_gqa_attention(ctx, h, w, hidden, n_heads, n_kv_heads,
                                 head_dim, T, rope_theta);
    } else {
        h = gliner_short_conv(ctx, h, w, hidden, T);
    }

    x = ggml_add(ctx, residual, h);

    residual = x;
    h = gliner_rms_norm(ctx, x, w.ffn_norm_w, norm_eps);
    h = gliner_swiglu_ffn(ctx, h, w.ff_w1, w.ff_w2, w.ff_w3);
    x = ggml_add(ctx, residual, h);

    return x;
}

// ============================================================================
// BiLSTM (CPU-side, not in ggml graph — simpler for 1-layer)
// ============================================================================

static void lstm_forward_one_dir(
    const float * input,    // (T, input_size)
    float * output,         // (T, hidden_size) — column to write
    int T, int input_size, int hidden_size,
    const float * W_ih,     // (4*hidden_size, input_size)
    const float * b_ih,     // (4*hidden_size)
    const float * W_hh,     // (4*hidden_size, hidden_size)
    const float * b_hh,     // (4*hidden_size)
    bool reverse)
{
    // LSTM gate order in PyTorch: i, f, g, o (each hidden_size)
    const int gate_size = 4 * hidden_size;
    std::vector<float> h(hidden_size, 0.0f);
    std::vector<float> c(hidden_size, 0.0f);
    std::vector<float> gates(gate_size);

    auto sigmoid = [](float x) -> float { return 1.0f / (1.0f + expf(-x)); };

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;
        const float * xt = input + t * input_size;

        // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
        for (int g = 0; g < gate_size; g++) {
            float val = b_ih[g] + b_hh[g];
            for (int j = 0; j < input_size; j++)
                val += W_ih[g * input_size + j] * xt[j];
            for (int j = 0; j < hidden_size; j++)
                val += W_hh[g * hidden_size + j] * h[j];
            gates[g] = val;
        }

        // Split into i, f, g, o gates
        for (int j = 0; j < hidden_size; j++) {
            float i_gate = sigmoid(gates[0 * hidden_size + j]);
            float f_gate = sigmoid(gates[1 * hidden_size + j]);
            float g_gate = tanhf(gates[2 * hidden_size + j]);
            float o_gate = sigmoid(gates[3 * hidden_size + j]);

            c[j] = f_gate * c[j] + i_gate * g_gate;
            h[j] = o_gate * tanhf(c[j]);
        }

        // Write h to output row t
        float * out_row = output + t * hidden_size;
        memcpy(out_row, h.data(), hidden_size * sizeof(float));
    }
}

// Run BiLSTM on CPU. Input: (T, input_size), Output: (T, 2*hidden_size)
static void bilstm_forward(
    const float * input, float * output,
    int T, int input_size, int lstm_hidden,
    const gliner_model & model, ggml_backend_t backend)
{
    // Read LSTM weights from backend
    auto read_tensor = [&](ggml_tensor * t) -> std::vector<float> {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    auto W_ih_fwd = read_tensor(model.lstm_weight_ih_l0);
    auto b_ih_fwd = read_tensor(model.lstm_bias_ih_l0);
    auto W_hh_fwd = read_tensor(model.lstm_weight_hh_l0);
    auto b_hh_fwd = read_tensor(model.lstm_bias_hh_l0);
    auto W_ih_rev = read_tensor(model.lstm_weight_ih_l0_rev);
    auto b_ih_rev = read_tensor(model.lstm_bias_ih_l0_rev);
    auto W_hh_rev = read_tensor(model.lstm_weight_hh_l0_rev);
    auto b_hh_rev = read_tensor(model.lstm_bias_hh_l0_rev);

    // Forward direction → first half of output
    std::vector<float> fwd_out(T * lstm_hidden);
    lstm_forward_one_dir(input, fwd_out.data(), T, input_size, lstm_hidden,
                         W_ih_fwd.data(), b_ih_fwd.data(),
                         W_hh_fwd.data(), b_hh_fwd.data(), false);

    // Reverse direction → second half of output
    std::vector<float> rev_out(T * lstm_hidden);
    lstm_forward_one_dir(input, rev_out.data(), T, input_size, lstm_hidden,
                         W_ih_rev.data(), b_ih_rev.data(),
                         W_hh_rev.data(), b_hh_rev.data(), true);

    // Concatenate: output[t] = [fwd[t], rev[t]]
    for (int t = 0; t < T; t++) {
        memcpy(output + t * 2 * lstm_hidden,
               fwd_out.data() + t * lstm_hidden,
               lstm_hidden * sizeof(float));
        memcpy(output + t * 2 * lstm_hidden + lstm_hidden,
               rev_out.data() + t * lstm_hidden,
               lstm_hidden * sizeof(float));
    }
}

// ============================================================================
// GLiNER head helpers (CPU-side post-processing)
// ============================================================================

// 2-layer MLP: Linear → ReLU → Linear (CPU)
static void mlp_2layer(
    const float * input,  // (N, in_dim)
    float * output,       // (N, out_dim)
    int N, int in_dim, int mid_dim, int out_dim,
    const float * w0, const float * b0,  // (mid_dim, in_dim), (mid_dim)
    const float * w3, const float * b3)  // (out_dim, mid_dim), (out_dim)
{
    std::vector<float> mid(N * mid_dim);

    // Layer 0: Linear + ReLU
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < mid_dim; j++) {
            float val = b0[j];
            for (int k = 0; k < in_dim; k++)
                val += w0[j * in_dim + k] * input[n * in_dim + k];
            mid[n * mid_dim + j] = val > 0.0f ? val : 0.0f;  // ReLU
        }
    }

    // Layer 3: Linear (no activation)
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < out_dim; j++) {
            float val = b3[j];
            for (int k = 0; k < mid_dim; k++)
                val += w3[j * mid_dim + k] * mid[n * mid_dim + k];
            output[n * out_dim + j] = val;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

void * gliner_ner_init(const char * model_path, int n_threads) {
    g_debug = (std::getenv("GLINER_DEBUG") != nullptr);

    auto * ctx = new gliner_context();
    ctx->n_threads = n_threads;

    // Init backend
    const char * force_cpu = std::getenv("CRISPEMBED_FORCE_CPU");
    if (force_cpu && force_cpu[0] && std::strcmp(force_cpu, "0") != 0) {
        ctx->backend = ggml_backend_cpu_init();
    } else {
        ctx->backend = ggml_backend_init_best();
    }
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
    }
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    }

    // Compute meta buffer (256 MB should be sufficient)
    ctx->compute_meta.resize(256 * 1024 * 1024);

    if (!load_model(ctx->model, model_path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void gliner_ner_free(void * ptr) {
    if (!ptr) return;
    auto * ctx = (gliner_context *)ptr;
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

int gliner_ner_extract(void * ptr,
                       const char * text,
                       const char ** labels, int n_labels,
                       float threshold,
                       gliner_ner_entity ** out_entities)
{
    if (!ptr || !text || !labels || n_labels <= 0) return 0;

    auto * ctx = (gliner_context *)ptr;
    auto & model = ctx->model;
    auto & hp = model.hparams;

    // Clear previous results
    ctx->result_entities.clear();
    ctx->result_texts.clear();
    ctx->result_labels.clear();

    // -----------------------------------------------------------------------
    // 1. Tokenize: BOS <<ENT>> label1 <<ENT>> label2 ... <<SEP>> text EOS
    // GLiNER format: BOS, then <<ENT>> before each label, single <<SEP>>
    // after all labels, then text tokens, then EOS.
    // -----------------------------------------------------------------------

    // Track which token positions correspond to <<ENT>> (entity type markers)
    std::vector<int> ent_positions;       // positions of <<ENT>> tokens
    std::vector<std::string> ent_labels;  // corresponding label strings

    std::vector<int32_t> input_ids;

    // BOS token
    input_ids.push_back(1);  // <|startoftext|>

    for (int i = 0; i < n_labels; i++) {
        ent_positions.push_back((int)input_ids.size());
        ent_labels.push_back(labels[i]);
        input_ids.push_back((int32_t)hp.ent_token_id);

        // Tokenize the label text
        {
            std::string label_str(labels[i]);
            std::string encoded = core_bpe::bytes_to_unicode(label_str.data(), label_str.size());
            std::vector<int32_t> label_tokens;
            core_bpe::bpe_one(model.token_to_id, model.merge_rank, encoded, label_tokens);
            for (auto t : label_tokens)
                input_ids.push_back(t);
        }
    }

    // Single <<SEP>> after all labels
    input_ids.push_back((int32_t)hp.sep_token_id);

    // Track where the actual text tokens start
    int text_token_start = (int)input_ids.size();

    // Tokenize the input text, tracking character offsets
    // Split by whitespace to get word-level token groups
    std::string input_text(text);
    std::vector<int> token_to_word;  // for each text token, which word index
    std::vector<int> word_char_start;  // character offset of each word
    std::vector<int> word_char_end;    // character end of each word
    std::vector<std::string> words;

    // Word splitting: matches GLiNER's WhitespaceTokenSplitter regex:
    //   r"\w+(?:[-_]\w+)*|\S"
    // This splits "Cupertino," into ["Cupertino", ","] — punctuation is
    // a separate word, not glued to the preceding word.
    {
        int pos = 0;
        int len = (int)input_text.size();
        while (pos < len) {
            // Skip whitespace
            while (pos < len && (input_text[pos] == ' ' || input_text[pos] == '\t'
                                 || input_text[pos] == '\n' || input_text[pos] == '\r'))
                pos++;
            if (pos >= len) break;

            int start = pos;
            unsigned char c = (unsigned char)input_text[pos];

            // Check if this is a word character (alphanumeric or underscore)
            auto is_word_char = [](unsigned char ch) -> bool {
                return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') || ch == '_' || ch >= 0x80;
            };

            if (is_word_char(c)) {
                // Match \w+(?:[-_]\w+)* — word chars, possibly with hyphens/underscores
                pos++;
                while (pos < len && is_word_char((unsigned char)input_text[pos]))
                    pos++;
                // Check for hyphen/underscore continuation
                while (pos < len && (input_text[pos] == '-' || input_text[pos] == '_') &&
                       pos + 1 < len && is_word_char((unsigned char)input_text[pos + 1])) {
                    pos++; // skip hyphen/underscore
                    while (pos < len && is_word_char((unsigned char)input_text[pos]))
                        pos++;
                }
            } else {
                // Match \S — single non-whitespace character (punctuation etc.)
                pos++;
            }

            std::string word = input_text.substr(start, pos - start);
            int word_idx = (int)words.size();
            words.push_back(word);
            word_char_start.push_back(start);
            word_char_end.push_back(pos);

            // BPE-tokenize this word
            {
                std::string encoded = core_bpe::bytes_to_unicode(word.data(), word.size());
                std::vector<int32_t> word_tokens;
                core_bpe::bpe_one(model.token_to_id, model.merge_rank, encoded, word_tokens);
                for (auto t : word_tokens) {
                    input_ids.push_back(t);
                    token_to_word.push_back(word_idx);
                }
            }
        }
    }

    // EOS token
    input_ids.push_back(7);  // <|im_end|>

    int T = (int)input_ids.size();
    int n_text_tokens = T - text_token_start - 1;  // exclude EOS

    if (n_text_tokens <= 0) {
        if (out_entities) *out_entities = nullptr;
        return 0;
    }

    GDBG("input: %d tokens (%d label prefix + %d text + BOS/EOS), %d words, %d entity types",
         T, text_token_start, n_text_tokens, (int)words.size(), n_labels);
    if (g_debug) {
        fprintf(stderr, "[gliner] token IDs:");
        for (int i = 0; i < T && i < 30; i++)
            fprintf(stderr, " %d", input_ids[i]);
        if (T > 30) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // 2. Run LFM2 backbone (ggml graph)
    // -----------------------------------------------------------------------

    const int hidden = (int)hp.hidden_size;
    const int n_heads = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const float norm_eps = 1e-5f;

    // Build graph
    g_pos_tensors.clear();
    size_t buf_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(65536, false);
    ggml_init_params ip = { buf_size, nullptr, true };
    ggml_context * g = ggml_init(ip);

    // Input: token IDs
    ggml_tensor * inp_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(inp_ids, "input_ids");
    ggml_set_input(inp_ids);

    // Embedding lookup
    ggml_tensor * x = ggml_get_rows(g, model.embed_tokens_w, inp_ids);  // (hidden, T)
    ggml_tensor * post_embed = ggml_dup(g, x);
    ggml_set_name(post_embed, "post_embed");
    ggml_set_output(post_embed);

    // Run all layers, saving each output for layer fusion
    std::vector<ggml_tensor *> layer_outputs(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        x = gliner_build_layer(g, x, model.layers[i], hidden, n_heads, n_kv,
                               hd, T, norm_eps, hp.rope_theta);
        layer_outputs[i] = x;

        // Mark intermediate outputs for layer fusion
        char name[32];
        snprintf(name, sizeof(name), "layer_%u", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }

    // Final embedding norm
    x = gliner_rms_norm(g, x, model.embedding_norm_w, norm_eps);
    ggml_set_name(x, "final_norm");
    ggml_set_output(x);

    // Build and compute graph
    ggml_cgraph * gf = ggml_new_graph_custom(g, 65536, false);
    ggml_build_forward_expand(gf, post_embed);
    for (auto * lo : layer_outputs)
        ggml_build_forward_expand(gf, lo);
    ggml_build_forward_expand(gf, x);

    // Allocate with gallocr
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "[gliner] graph allocation failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return 0;
    }

    // Set input: token IDs
    ggml_backend_tensor_set(inp_ids, input_ids.data(), 0, T * sizeof(int32_t));

    // Set position IDs for attention layers (collected during graph build)
    {
        std::vector<int32_t> pos(T);
        for (int i = 0; i < T; i++) pos[i] = i;
        for (auto * pt : g_pos_tensors)
            ggml_backend_tensor_set(pt, pos.data(), 0, T * sizeof(int32_t));
    }

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // --- Parity comparison against Python reference (if GLINER_DIFF_REF set) ---
    const char * diff_ref_path = std::getenv("GLINER_DIFF_REF");
    if (diff_ref_path) {
        crispembed_diff::Ref ref;
        if (ref.load(diff_ref_path)) {
            fprintf(stderr, "[gliner-diff] Loaded reference: %s\n", diff_ref_path);

            // Compare post_embed
            {
                std::vector<float> embed_data(T * hidden);
                ggml_backend_tensor_get(post_embed, embed_data.data(),
                                        0, T * hidden * sizeof(float));
                // ggml layout: (hidden, T) = column-major, same as numpy (T, hidden) row-major
                auto r = ref.compare("post_embed", embed_data.data(), T * hidden);
                fprintf(stderr, "[gliner-diff] post_embed: cos=%.6f max_abs=%.2e %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }

            // Compare per-layer outputs
            for (uint32_t i = 0; i < hp.n_layers; i++) {
                std::vector<float> ldata(T * hidden);
                ggml_backend_tensor_get(layer_outputs[i], ldata.data(),
                                        0, T * hidden * sizeof(float));
                char name[32];
                snprintf(name, sizeof(name), "layer_%u", i);
                auto r = ref.compare(name, ldata.data(), T * hidden);
                fprintf(stderr, "[gliner-diff] %s: cos=%.6f max_abs=%.2e %s\n",
                        name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                if (!r.is_pass()) {
                    fprintf(stderr, "[gliner-diff] *** FIRST FAILURE at %s — stopping layer comparison\n", name);
                    break;
                }
            }

            // Compare final_norm
            {
                std::vector<float> norm_data(T * hidden);
                ggml_backend_tensor_get(x, norm_data.data(), 0, T * hidden * sizeof(float));
                auto r = ref.compare("final_norm", norm_data.data(), T * hidden);
                fprintf(stderr, "[gliner-diff] final_norm: cos=%.6f max_abs=%.2e %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        } else {
            fprintf(stderr, "[gliner-diff] WARNING: failed to load reference: %s\n", diff_ref_path);
        }
    }

    // Read layer outputs for fusion
    std::vector<std::vector<float>> all_layer_outs(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        all_layer_outs[i].resize(T * hidden);
        ggml_backend_tensor_get(layer_outputs[i], all_layer_outs[i].data(),
                                0, T * hidden * sizeof(float));
    }

    // Read final normed output (not used directly — we use fused output)
    // But keep it for potential fallback

    ggml_gallocr_free(galloc);
    ggml_free(g);

    // -----------------------------------------------------------------------
    // 3. Layer fusion (CPU) — attention-weighted sum of all layer outputs
    // -----------------------------------------------------------------------

    // Read fuser weights
    auto read_tensor = [&](ggml_tensor * t) -> std::vector<float> {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    auto squeeze_w = read_tensor(model.fuser_squeeze_w);   // (1, hidden)
    auto squeeze_b = read_tensor(model.fuser_squeeze_b);   // (1)
    auto W1_w = read_tensor(model.fuser_W1_w);             // (8, n_layers)
    auto W1_b = read_tensor(model.fuser_W1_b);             // (8)
    auto W2_w = read_tensor(model.fuser_W2_w);             // (n_layers, 8)
    auto W2_b = read_tensor(model.fuser_W2_b);             // (n_layers)
    auto out_proj_w = read_tensor(model.fuser_output_proj_w);  // (hidden, hidden)
    auto out_proj_b = read_tensor(model.fuser_output_proj_b);  // (hidden)

    int NL = (int)hp.n_layers;

    // For each layer, compute a scalar attention weight:
    //   score_l = mean_over_tokens(squeeze_w @ layer_out[t] + squeeze_b) for layer l
    std::vector<float> layer_scores(NL);
    for (int l = 0; l < NL; l++) {
        float total = 0.0f;
        for (int t = 0; t < T; t++) {
            float s = squeeze_b[0];
            for (int d = 0; d < hidden; d++)
                s += squeeze_w[d] * all_layer_outs[l][t * hidden + d];
            total += s;
        }
        layer_scores[l] = total / T;
    }

    // Apply W1 → ReLU → W2 to get refined attention weights
    std::vector<float> mid8(8);
    for (int j = 0; j < 8; j++) {
        float val = W1_b[j];
        for (int l = 0; l < NL; l++)
            val += W1_w[j * NL + l] * layer_scores[l];
        mid8[j] = val > 0.0f ? val : 0.0f;  // ReLU
    }
    std::vector<float> attn_weights(NL);
    for (int l = 0; l < NL; l++) {
        float val = W2_b[l];
        for (int j = 0; j < 8; j++)
            val += W2_w[l * 8 + j] * mid8[j];
        attn_weights[l] = val;
    }

    // Sigmoid (NOT softmax!) — per-layer independent gates
    for (int l = 0; l < NL; l++)
        attn_weights[l] = 1.0f / (1.0f + expf(-attn_weights[l]));

    // Weighted sum of layer outputs
    std::vector<float> fused(T * hidden, 0.0f);
    for (int l = 0; l < NL; l++) {
        float w = attn_weights[l];
        for (int i = 0; i < T * hidden; i++)
            fused[i] += w * all_layer_outs[l][i];
    }

    // Output projection: fused = out_proj_w @ fused + out_proj_b
    std::vector<float> fused_proj(T * hidden);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < hidden; d++) {
            float val = out_proj_b[d];
            for (int k = 0; k < hidden; k++)
                val += out_proj_w[d * hidden + k] * fused[t * hidden + k];
            fused_proj[t * hidden + d] = val;
        }
    }

    // Free layer outputs (no longer needed)
    all_layer_outs.clear();

    // Compare fused output against reference
    if (diff_ref_path) {
        crispembed_diff::Ref ref;
        if (ref.load(diff_ref_path)) {
            auto r = ref.compare("fused", fused_proj.data(), T * hidden);
            fprintf(stderr, "[gliner-diff] fused: cos=%.6f max_abs=%.2e %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    GDBG("layer fusion done, running BiLSTM...");

    // -----------------------------------------------------------------------
    // 4. Extract entity reps + first-subtoken pooling + BiLSTM (word-level)
    // -----------------------------------------------------------------------

    // 4a. Extract entity type representations from fused output at <<ENT>> positions
    //     This happens BEFORE the BiLSTM — the entity reps come from the fused token embeddings.
    std::vector<float> ent_hidden_from_fused(n_labels * hidden);
    for (int i = 0; i < n_labels; i++) {
        int pos = ent_positions[i];
        memcpy(ent_hidden_from_fused.data() + i * hidden,
               fused_proj.data() + pos * hidden,
               hidden * sizeof(float));
    }

    // 4b. First-subtoken pooling: for each word, take the first subtoken's representation
    //     This converts from token-level to word-level before the BiLSTM.
    std::vector<int> word_first_token;  // index into text portion tokens
    {
        int prev_word = -1;
        for (int i = 0; i < n_text_tokens; i++) {
            if (token_to_word[i] != prev_word) {
                word_first_token.push_back(i);
                prev_word = token_to_word[i];
            }
        }
    }
    int n_words = (int)word_first_token.size();

    std::vector<float> word_reps(n_words * hidden);
    for (int w = 0; w < n_words; w++) {
        int tok_idx = text_token_start + word_first_token[w];
        memcpy(word_reps.data() + w * hidden,
               fused_proj.data() + tok_idx * hidden,
               hidden * sizeof(float));
    }

    // 4c. BiLSTM on word-level representations only
    const int lstm_hidden = 512;
    std::vector<float> lstm_out(n_words * 2 * lstm_hidden);
    bilstm_forward(word_reps.data(), lstm_out.data(),
                   n_words, hidden, lstm_hidden, model, ctx->backend);

    // lstm_out: (n_words, 1024) — word-level bidirectional hidden states
    // Compare lstm output against reference (now word-level, should match)
    if (diff_ref_path) {
        crispembed_diff::Ref ref;
        if (ref.load(diff_ref_path)) {
            auto r = ref.compare("lstm_out", lstm_out.data(), n_words * 2 * lstm_hidden);
            fprintf(stderr, "[gliner-diff] lstm_out: cos=%.6f max_abs=%.2e %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }

    GDBG("BiLSTM done, computing GLiNER head...");

    // -----------------------------------------------------------------------
    // 5. GLiNER head (CPU)
    // -----------------------------------------------------------------------

    // 5a. Entity type representations: from fused output at <<ENT>> positions → prompt_rep MLP
    //     (entity reps come from fused token embeddings, NOT BiLSTM output)
    auto prompt_rep_0_w = read_tensor(model.prompt_rep_0_w);
    auto prompt_rep_0_b = read_tensor(model.prompt_rep_0_b);
    auto prompt_rep_3_w = read_tensor(model.prompt_rep_3_w);
    auto prompt_rep_3_b = read_tensor(model.prompt_rep_3_b);

    std::vector<float> ent_reps(n_labels * hidden);
    mlp_2layer(ent_hidden_from_fused.data(), ent_reps.data(),
               n_labels, hidden, 4096, hidden,
               prompt_rep_0_w.data(), prompt_rep_0_b.data(),
               prompt_rep_3_w.data(), prompt_rep_3_b.data());

    // 5b. Span representations (markerV1)
    // For each valid span (start, end) in the text portion:
    //   span_rep = out_project(concat(proj_start[start], proj_end[end], proj_first[0]))
    // where proj_* are 2-layer MLPs on the lstm_out hidden states

    // Read span projection weights
    auto sp_start_0_w = read_tensor(model.span_proj_start_0_w);
    auto sp_start_0_b = read_tensor(model.span_proj_start_0_b);
    auto sp_start_3_w = read_tensor(model.span_proj_start_3_w);
    auto sp_start_3_b = read_tensor(model.span_proj_start_3_b);
    auto sp_end_0_w   = read_tensor(model.span_proj_end_0_w);
    auto sp_end_0_b   = read_tensor(model.span_proj_end_0_b);
    auto sp_end_3_w   = read_tensor(model.span_proj_end_3_w);
    auto sp_end_3_b   = read_tensor(model.span_proj_end_3_b);
    auto sp_first_0_w = read_tensor(model.span_proj_first_0_w);
    auto sp_first_0_b = read_tensor(model.span_proj_first_0_b);
    auto sp_first_3_w = read_tensor(model.span_proj_first_3_w);
    auto sp_first_3_b = read_tensor(model.span_proj_first_3_b);
    auto sp_out_0_w   = read_tensor(model.span_out_proj_0_w);
    auto sp_out_0_b   = read_tensor(model.span_out_proj_0_b);
    auto sp_out_3_w   = read_tensor(model.span_out_proj_3_w);
    auto sp_out_3_b   = read_tensor(model.span_out_proj_3_b);

    // Project all WORD-LEVEL representations through start, end, and first projections
    // (using BiLSTM output, which is word-level)
    std::vector<float> proj_start_all(n_words * hidden);
    std::vector<float> proj_end_all(n_words * hidden);
    mlp_2layer(lstm_out.data(), proj_start_all.data(),
               n_words, hidden, 4096, hidden,
               sp_start_0_w.data(), sp_start_0_b.data(),
               sp_start_3_w.data(), sp_start_3_b.data());
    mlp_2layer(lstm_out.data(), proj_end_all.data(),
               n_words, hidden, 4096, hidden,
               sp_end_0_w.data(), sp_end_0_b.data(),
               sp_end_3_w.data(), sp_end_3_b.data());

    // Project first token (CLS-like) — mean of all word representations
    std::vector<float> first_token(hidden, 0.0f);
    for (int w = 0; w < n_words; w++)
        for (int d = 0; d < hidden; d++)
            first_token[d] += lstm_out[w * hidden + d];
    for (int d = 0; d < hidden; d++)
        first_token[d] /= n_words;

    std::vector<float> proj_first(hidden);
    mlp_2layer(first_token.data(), proj_first.data(),
               1, hidden, 4096, hidden,
               sp_first_0_w.data(), sp_first_0_b.data(),
               sp_first_3_w.data(), sp_first_3_b.data());

    // -----------------------------------------------------------------------
    // 5c. Score all spans against all entity types
    //     Score = einsum("BLKD,BCD->BLKC") = dot product of span_rep and ent_rep
    //     No temperature scaling in forward — that's applied during decoding.
    // -----------------------------------------------------------------------

    // Read scorer temperature (applied during sigmoid scoring)
    float log_temp;
    ggml_backend_tensor_get(model.scorer_log_temp, &log_temp, 0, sizeof(float));
    float temperature = expf(log_temp);
    GDBG("scorer temperature: exp(%f) = %f", log_temp, temperature);

    struct ScoredSpan {
        int word_start, word_end;  // word-level indices (inclusive)
        int label_idx;
        float score;
    };
    std::vector<ScoredSpan> candidates;

    int max_w = (int)hp.max_width;
    for (int ws = 0; ws < n_words; ws++) {
        for (int we = ws; we < std::min(ws + max_w, n_words); we++) {
            // Span representation: concat(proj_start[ws], proj_end[we], proj_first)
            std::vector<float> span_input(3 * hidden);
            memcpy(span_input.data(), proj_start_all.data() + ws * hidden,
                   hidden * sizeof(float));
            memcpy(span_input.data() + hidden, proj_end_all.data() + we * hidden,
                   hidden * sizeof(float));
            memcpy(span_input.data() + 2 * hidden, proj_first.data(),
                   hidden * sizeof(float));

            // out_project: (3*hidden) → 4096 → hidden
            std::vector<float> span_rep(hidden);
            mlp_2layer(span_input.data(), span_rep.data(),
                       1, 3 * hidden, 4096, hidden,
                       sp_out_0_w.data(), sp_out_0_b.data(),
                       sp_out_3_w.data(), sp_out_3_b.data());

            // Score against each entity type: sigmoid(dot(span_rep, ent_rep))
            // Note: log_score_temperature exists in weights but is NOT used by
            // GLiNER's inference code — raw dot product → sigmoid.
            for (int e = 0; e < n_labels; e++) {
                float dot = 0.0f;
                for (int d = 0; d < hidden; d++)
                    dot += span_rep[d] * ent_reps[e * hidden + d];
                float score = 1.0f / (1.0f + expf(-dot));

                // Debug: show top scores
                if (g_debug && score > 0.01f) {
                    GDBG("  span [%d-%d] '%s..%s' vs '%s': dot=%.4f score=%.4f",
                         ws, we,
                         ws < (int)words.size() ? words[ws].c_str() : "?",
                         we < (int)words.size() ? words[we].c_str() : "?",
                         ent_labels[e].c_str(), dot, score);
                }

                if (score >= threshold) {
                    candidates.push_back({ws, we, e, score});
                }
            }
        }
    }

    GDBG("found %zu candidate spans above threshold %.2f", candidates.size(), threshold);

    // -----------------------------------------------------------------------
    // 6. Greedy non-overlapping decode
    // -----------------------------------------------------------------------

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const ScoredSpan & a, const ScoredSpan & b) {
                  return a.score > b.score;
              });

    // Greedy: accept highest-scoring spans, reject overlapping ones
    std::vector<bool> word_taken(n_words, false);
    std::vector<ScoredSpan> accepted;

    for (const auto & span : candidates) {
        bool overlap = false;
        for (int w = span.word_start; w <= span.word_end; w++) {
            if (word_taken[w]) { overlap = true; break; }
        }
        if (overlap) continue;

        accepted.push_back(span);
        for (int w = span.word_start; w <= span.word_end; w++)
            word_taken[w] = true;
    }

    // -----------------------------------------------------------------------
    // 7. Build output entities
    // -----------------------------------------------------------------------

    ctx->result_entities.clear();
    ctx->result_texts.clear();
    ctx->result_labels.clear();

    // Sort accepted by position
    std::sort(accepted.begin(), accepted.end(),
              [](const ScoredSpan & a, const ScoredSpan & b) {
                  return a.word_start < b.word_start;
              });

    for (const auto & span : accepted) {
        int char_start = word_char_start[span.word_start];
        int char_end   = word_char_end[span.word_end];

        ctx->result_texts.push_back(input_text.substr(char_start, char_end - char_start));
        ctx->result_labels.push_back(ent_labels[span.label_idx]);
    }

    ctx->result_entities.resize(accepted.size());
    for (size_t i = 0; i < accepted.size(); i++) {
        auto & ent = ctx->result_entities[i];
        ent.start_char = word_char_start[accepted[i].word_start];
        ent.end_char   = word_char_end[accepted[i].word_end];
        ent.text  = ctx->result_texts[i].c_str();
        ent.label = ctx->result_labels[i].c_str();
        ent.score = accepted[i].score;
    }

    if (out_entities)
        *out_entities = ctx->result_entities.data();

    GDBG("extracted %zu entities", ctx->result_entities.size());
    return (int)ctx->result_entities.size();
}
