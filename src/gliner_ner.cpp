// gliner_ner.cpp — GLiNER zero-shot NER with multiple backbone support.
//
// Backbone A: LFM2.5-350M (SauerkrautLM-LFM2.5-GLiNER)
//   BPE tokenizer, 16L ShortConv+GQA, layer fusion, markerV1 spans
//
// Backbone B: DeBERTa-v3-base (urchade/gliner_medium-v2.1)
//   SentencePiece tokenizer, 12L disentangled attention, 768→512 projection,
//   no layer fusion, markerV0 spans (start+end only)

#include "gliner_ner.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/gguf_loader.h"
#include "core/bpe.h"
#include "tokenizer.h"
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
#include <chrono>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Debug logging
static bool g_debug = false;
#define GDBG(fmt, ...) do { if (g_debug) fprintf(stderr, "[gliner] " fmt "\n", ##__VA_ARGS__); } while (0)

// ============================================================================
// Backbone types
// ============================================================================

enum gliner_backbone {
    GLINER_BACKBONE_LFM2    = 0,  // SauerkrautLM-LFM2.5-GLiNER
    GLINER_BACKBONE_DEBERTA = 1,  // urchade/gliner_medium-v2.1 (DeBERTa-v3-base)
};

// ============================================================================
// Hyperparameters
// ============================================================================

struct gliner_hparams {
    gliner_backbone backbone = GLINER_BACKBONE_LFM2;

    // Encoder (LFM2 or DeBERTa)
    uint32_t hidden_size   = 1024;  // encoder hidden (LFM: 1024, DeBERTa: 768)
    uint32_t n_layers      = 16;
    uint32_t n_heads       = 16;
    uint32_t n_kv_heads    = 8;     // LFM2 only (GQA)
    uint32_t head_dim      = 64;
    uint32_t ff_dim        = 4608;  // LFM2 SwiGLU dim
    uint32_t intermediate_size = 3072;  // DeBERTa FFN intermediate
    uint32_t conv_kernel   = 3;
    float    rope_theta    = 1000000.0f;
    std::string layer_types;  // LFM2: "ccaccaccacacacac"
    uint32_t vocab_size    = 64404;

    // DeBERTa-specific
    uint32_t position_buckets = 256;
    float    layer_norm_eps   = 1e-7f;

    // GLiNER head
    uint32_t gliner_hidden = 0;     // 0 = same as hidden_size (LFM), else projected (DeBERTa: 512)
    uint32_t max_width     = 12;
    uint32_t ent_token_id  = 64402;
    uint32_t sep_token_id  = 64403;
    std::string span_mode  = "markerV1";  // "markerV0" (start+end) or "markerV1" (start+end+first)
};

// ============================================================================
// Per-layer weights — LFM2 variant
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
// Per-layer weights — DeBERTa variant
// ============================================================================

struct deberta_layer_weights {
    // Attention
    ggml_tensor * q_w = nullptr, * q_b = nullptr;
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    // Post-attention LayerNorm
    ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    // FFN
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
    // Post-FFN LayerNorm
    ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
};

// ============================================================================
// Model weights
// ============================================================================

struct gliner_model {
    gliner_hparams hparams;

    // --- LFM2 backbone ---
    ggml_tensor * embed_tokens_w    = nullptr;  // [vocab, hidden]
    ggml_tensor * embedding_norm_w  = nullptr;  // [hidden] (LFM2 RMSNorm)
    std::vector<gliner_layer_weights> layers;   // LFM2 layers

    // Layer fuser (LFM2 only)
    ggml_tensor * fuser_squeeze_w      = nullptr;
    ggml_tensor * fuser_squeeze_b      = nullptr;
    ggml_tensor * fuser_W1_w           = nullptr;
    ggml_tensor * fuser_W1_b           = nullptr;
    ggml_tensor * fuser_W2_w           = nullptr;
    ggml_tensor * fuser_W2_b           = nullptr;
    ggml_tensor * fuser_output_proj_w  = nullptr;
    ggml_tensor * fuser_output_proj_b  = nullptr;

    // --- DeBERTa backbone ---
    ggml_tensor * token_embd_w     = nullptr;  // [vocab, hidden]
    ggml_tensor * embd_ln_w        = nullptr;  // embedding LayerNorm
    ggml_tensor * embd_ln_b        = nullptr;
    ggml_tensor * encoder_ln_w     = nullptr;  // encoder-level LN (for rel_embd)
    ggml_tensor * encoder_ln_b     = nullptr;
    ggml_tensor * rel_embd_w       = nullptr;  // [hidden, max_rel_pos]
    ggml_tensor * projection_w     = nullptr;  // [gliner_hidden, hidden] (768→512)
    ggml_tensor * projection_b     = nullptr;
    std::vector<deberta_layer_weights> deb_layers;

    // --- Shared: BiLSTM ---
    ggml_tensor * lstm_weight_ih_l0     = nullptr;
    ggml_tensor * lstm_bias_ih_l0       = nullptr;
    ggml_tensor * lstm_weight_hh_l0     = nullptr;
    ggml_tensor * lstm_bias_hh_l0       = nullptr;
    ggml_tensor * lstm_weight_ih_l0_rev = nullptr;
    ggml_tensor * lstm_bias_ih_l0_rev   = nullptr;
    ggml_tensor * lstm_weight_hh_l0_rev = nullptr;
    ggml_tensor * lstm_bias_hh_l0_rev   = nullptr;

    // --- Shared: GLiNER span representation ---
    // markerV1: start+end+first (LFM2), markerV0: start+end only (DeBERTa)
    ggml_tensor * span_proj_start_0_w = nullptr, * span_proj_start_0_b = nullptr;
    ggml_tensor * span_proj_start_3_w = nullptr, * span_proj_start_3_b = nullptr;
    ggml_tensor * span_proj_end_0_w   = nullptr, * span_proj_end_0_b   = nullptr;
    ggml_tensor * span_proj_end_3_w   = nullptr, * span_proj_end_3_b   = nullptr;
    ggml_tensor * span_proj_first_0_w = nullptr, * span_proj_first_0_b = nullptr;  // V1 only
    ggml_tensor * span_proj_first_3_w = nullptr, * span_proj_first_3_b = nullptr;  // V1 only
    ggml_tensor * span_out_proj_0_w   = nullptr, * span_out_proj_0_b   = nullptr;
    ggml_tensor * span_out_proj_3_w   = nullptr, * span_out_proj_3_b   = nullptr;

    // Prompt/entity representation MLP
    ggml_tensor * prompt_rep_0_w = nullptr, * prompt_rep_0_b = nullptr;
    ggml_tensor * prompt_rep_3_w = nullptr, * prompt_rep_3_b = nullptr;

    // Scorer (LFM2 only — DeBERTa uses raw dot product)
    ggml_tensor * scorer_log_temp = nullptr;

    // Model memory
    ggml_context       * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    // BPE tokenizer maps (LFM2)
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;

    // SPM tokenizer (DeBERTa)
    SentencePieceTokenizer spm_tokenizer;
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

    // Cached F32 weights for CPU-side operations (avoids per-call dequant)
    // BiLSTM weights (8 tensors)
    std::vector<float> lstm_W_ih_fwd, lstm_b_ih_fwd, lstm_W_hh_fwd, lstm_b_hh_fwd;
    std::vector<float> lstm_W_ih_rev, lstm_b_ih_rev, lstm_W_hh_rev, lstm_b_hh_rev;
    // Layer fuser weights (LFM2 only, 8 tensors)
    std::vector<float> fuser_squeeze_w, fuser_squeeze_b;
    std::vector<float> fuser_W1_w, fuser_W1_b, fuser_W2_w, fuser_W2_b;
    std::vector<float> fuser_out_proj_w, fuser_out_proj_b;
    bool weights_cached = false;

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

    // Detect backbone type
    std::string backbone_str = core_gguf::kv_str(gctx, "gliner.backbone", "lfm2");
    if (backbone_str == "deberta") {
        hp.backbone = GLINER_BACKBONE_DEBERTA;
    } else {
        hp.backbone = GLINER_BACKBONE_LFM2;
    }

    // GLiNER head params (shared)
    hp.max_width    = core_gguf::kv_u32(gctx, "gliner.max_width", 12);
    hp.ent_token_id = core_gguf::kv_u32(gctx, "gliner.ent_token_id", 64402);
    hp.sep_token_id = core_gguf::kv_u32(gctx, "gliner.sep_token_id", 64403);
    hp.span_mode    = core_gguf::kv_str(gctx, "gliner.span_mode", "markerV1");

    if (hp.backbone == GLINER_BACKBONE_DEBERTA) {
        // DeBERTa encoder params
        hp.hidden_size       = core_gguf::kv_u32(gctx, "bert.hidden_size", 768);
        hp.n_layers          = core_gguf::kv_u32(gctx, "bert.num_hidden_layers", 12);
        hp.n_heads           = core_gguf::kv_u32(gctx, "bert.num_attention_heads", 12);
        hp.head_dim          = hp.hidden_size / hp.n_heads;
        hp.intermediate_size = core_gguf::kv_u32(gctx, "bert.intermediate_size", 3072);
        hp.position_buckets  = core_gguf::kv_u32(gctx, "bert.position_buckets", 256);
        hp.layer_norm_eps    = core_gguf::kv_f32(gctx, "bert.layer_norm_eps", 1e-7f);
        hp.vocab_size        = core_gguf::kv_u32(gctx, "bert.vocab_size", 128004);
        hp.gliner_hidden     = core_gguf::kv_u32(gctx, "gliner.hidden_size", 512);

        GDBG("  backbone=DeBERTa, hidden=%u, layers=%u, heads=%u, gliner_hidden=%u",
             hp.hidden_size, hp.n_layers, hp.n_heads, hp.gliner_hidden);
    } else {
        // LFM2 encoder params
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
        hp.gliner_hidden = hp.hidden_size;  // no projection for LFM2

        GDBG("  backbone=LFM2, hidden=%u, layers=%u, heads=%u/%u",
             hp.hidden_size, hp.n_layers, hp.n_heads, hp.n_kv_heads);
    }

    // --- Load tokenizer ---
    auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
    if (tokens.empty()) {
        // Try old naming
        tokens = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    }
    if (tokens.empty()) {
        fprintf(stderr, "[gliner] no tokenizer tokens in GGUF\n");
        core_gguf::free_metadata(gctx);
        return false;
    }

    int tokenizer_type = (int)core_gguf::kv_u32(gctx, "tokenizer.ggml.type", 0);

    if (hp.backbone == GLINER_BACKBONE_DEBERTA || tokenizer_type == 2) {
        // SentencePiece tokenizer — read scores via raw GGUF API
        std::vector<float> scores;
        const int64_t si = gguf_find_key(gctx, "tokenizer.ggml.scores");
        if (si >= 0 && gguf_get_arr_type(gctx, si) == GGUF_TYPE_FLOAT32) {
            int sn = (int)gguf_get_arr_n(gctx, si);
            scores.resize(sn);
            const float * sd = reinterpret_cast<const float *>(gguf_get_arr_data(gctx, si));
            std::memcpy(scores.data(), sd, sn * sizeof(float));
        }
        int bos_id = (int)core_gguf::kv_u32(gctx, "tokenizer.ggml.bos_token_id", 1);
        int eos_id = (int)core_gguf::kv_u32(gctx, "tokenizer.ggml.eos_token_id", 2);
        int unk_id = (int)core_gguf::kv_u32(gctx, "tokenizer.ggml.unknown_token_id", 3);
        int pad_id = (int)core_gguf::kv_u32(gctx, "tokenizer.ggml.padding_token_id", 0);
        model.spm_tokenizer.load(tokens, scores, bos_id, eos_id, unk_id, pad_id, 512);
        GDBG("  tokenizer: SentencePiece (%zu tokens)", tokens.size());
    } else {
        // BPE tokenizer (LFM2)
        for (size_t i = 0; i < tokens.size(); i++)
            model.token_to_id[tokens[i]] = (int32_t)i;
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
            GDBG("  tokenizer: BPE (%zu tokens, %d merges)", tokens.size(), rank);
        }
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

    auto T = [&](const char * name) -> ggml_tensor * {
        return core_gguf::try_get(model.tensors, name);
    };
    auto R = [&](const char * name) -> ggml_tensor * {
        return core_gguf::require(model.tensors, name, "gliner");
    };

    if (hp.backbone == GLINER_BACKBONE_DEBERTA) {
        // --- DeBERTa encoder weights ---
        model.token_embd_w = R("token_embd.weight");
        model.embd_ln_w    = R("embd_ln.weight");
        model.embd_ln_b    = R("embd_ln.bias");
        model.encoder_ln_w = R("encoder_ln.weight");
        model.encoder_ln_b = R("encoder_ln.bias");
        model.rel_embd_w   = R("rel_embd.weight");
        model.projection_w = R("projection.weight");
        model.projection_b = R("projection.bias");

        model.deb_layers.resize(hp.n_layers);
        for (uint32_t i = 0; i < hp.n_layers; i++) {
            auto & l = model.deb_layers[i];
            auto rn = [&](const char * suffix) {
                char name[128];
                snprintf(name, sizeof(name), "enc.%u.%s", i, suffix);
                return R(name);
            };
            l.q_w   = rn("attn.q.weight"); l.q_b   = rn("attn.q.bias");
            l.k_w   = rn("attn.k.weight"); l.k_b   = rn("attn.k.bias");
            l.v_w   = rn("attn.v.weight"); l.v_b   = rn("attn.v.bias");
            l.o_w   = rn("attn.o.weight"); l.o_b   = rn("attn.o.bias");
            l.ln1_w = rn("ln1.weight");    l.ln1_b = rn("ln1.bias");
            l.fc1_w = rn("ffn.fc1.weight"); l.fc1_b = rn("ffn.fc1.bias");
            l.fc2_w = rn("ffn.fc2.weight"); l.fc2_b = rn("ffn.fc2.bias");
            l.ln2_w = rn("ln2.weight");    l.ln2_b = rn("ln2.bias");
        }
    } else {
        // --- LFM2 backbone weights ---
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

        // Layer fuser (LFM2 only)
        model.fuser_squeeze_w     = R("fuser.squeeze.weight");
        model.fuser_squeeze_b     = R("fuser.squeeze.bias");
        model.fuser_W1_w          = R("fuser.W1.weight");
        model.fuser_W1_b          = R("fuser.W1.bias");
        model.fuser_W2_w          = R("fuser.W2.weight");
        model.fuser_W2_b          = R("fuser.W2.bias");
        model.fuser_output_proj_w = R("fuser.output_projection.weight");
        model.fuser_output_proj_b = R("fuser.output_projection.bias");
    }

    // --- Shared: BiLSTM ---
    model.lstm_weight_ih_l0     = R("lstm.weight_ih_l0");
    model.lstm_bias_ih_l0       = R("lstm.bias_ih_l0");
    model.lstm_weight_hh_l0     = R("lstm.weight_hh_l0");
    model.lstm_bias_hh_l0       = R("lstm.bias_hh_l0");
    model.lstm_weight_ih_l0_rev = R("lstm.weight_ih_l0_reverse");
    model.lstm_bias_ih_l0_rev   = R("lstm.bias_ih_l0_reverse");
    model.lstm_weight_hh_l0_rev = R("lstm.weight_hh_l0_reverse");
    model.lstm_bias_hh_l0_rev   = R("lstm.bias_hh_l0_reverse");

    // --- Shared: GLiNER span representation ---
    model.span_proj_start_0_w = R("span.project_start.0.weight");
    model.span_proj_start_0_b = R("span.project_start.0.bias");
    model.span_proj_start_3_w = R("span.project_start.3.weight");
    model.span_proj_start_3_b = R("span.project_start.3.bias");
    model.span_proj_end_0_w   = R("span.project_end.0.weight");
    model.span_proj_end_0_b   = R("span.project_end.0.bias");
    model.span_proj_end_3_w   = R("span.project_end.3.weight");
    model.span_proj_end_3_b   = R("span.project_end.3.bias");
    if (hp.span_mode == "markerV1") {
        model.span_proj_first_0_w = R("span.project_first.0.weight");
        model.span_proj_first_0_b = R("span.project_first.0.bias");
        model.span_proj_first_3_w = R("span.project_first.3.weight");
        model.span_proj_first_3_b = R("span.project_first.3.bias");
    }
    model.span_out_proj_0_w   = R("span.out_project.0.weight");
    model.span_out_proj_0_b   = R("span.out_project.0.bias");
    model.span_out_proj_3_w   = R("span.out_project.3.weight");
    model.span_out_proj_3_b   = R("span.out_project.3.bias");

    // Prompt representation
    model.prompt_rep_0_w = R("prompt_rep.0.weight");
    model.prompt_rep_0_b = R("prompt_rep.0.bias");
    model.prompt_rep_3_w = R("prompt_rep.3.weight");
    model.prompt_rep_3_b = R("prompt_rep.3.bias");

    // Scorer (optional — DeBERTa models don't have it)
    model.scorer_log_temp = T("scorer.log_temperature");

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

// Forward declaration (defined below, after BiLSTM)
static std::vector<float> tensor_to_f32_backend(ggml_tensor * t, ggml_backend_t backend);

// ============================================================================
// DeBERTa encoder graph builder
// ============================================================================

// Build DeBERTa encoder graph. Returns the last hidden state tensor [H, T].
// Fills rel_pos_expanded_out for CPU-side position expansion after alloc.
static ggml_tensor * gliner_build_deberta_encoder(
    ggml_context * g, const gliner_model & model, int T,
    ggml_tensor ** rel_pos_expanded_out)
{
    const auto & hp = model.hparams;
    const int H = (int)hp.hidden_size;
    const int n_heads = (int)hp.n_heads;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;

    // Input: token IDs
    ggml_tensor * inp_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(inp_ids, "input_ids");
    ggml_set_input(inp_ids);

    // Embedding lookup + LayerNorm
    ggml_tensor * cur = ggml_get_rows(g, model.token_embd_w, inp_ids);
    cur = ggml_norm(g, cur, ln_eps);
    cur = ggml_mul(g, cur, model.embd_ln_w);
    cur = ggml_add(g, cur, model.embd_ln_b);

    // Pre-expanded position embeddings [H, T*T]
    ggml_tensor * rel_pos_expanded = ggml_new_tensor_2d(g, GGML_TYPE_F32, H, (int64_t)T * T);
    ggml_set_name(rel_pos_expanded, "rel_pos_expanded");
    ggml_set_input(rel_pos_expanded);
    *rel_pos_expanded_out = rel_pos_expanded;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto & L = model.deb_layers[il];
        ggml_tensor * inp = cur;

        // QKV projections
        ggml_tensor * Q = ggml_add(g, ggml_mul_mat(g, L.q_w, cur), L.q_b);
        ggml_tensor * K = ggml_add(g, ggml_mul_mat(g, L.k_w, cur), L.k_b);
        ggml_tensor * V = ggml_add(g, ggml_mul_mat(g, L.v_w, cur), L.v_b);

        // Reshape [H, T] → [hd, nh, T] and permute to [hd, T, nh]
        Q = ggml_reshape_3d(g, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(g, K, head_dim, n_heads, T);
        V = ggml_reshape_3d(g, V, head_dim, n_heads, T);

        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // Disentangled attention: c2c + c2p + p2c
        ggml_tensor * Qs = ggml_cont(g, ggml_reshape_3d(g, Q, head_dim, T, n_heads));
        ggml_tensor * Ks = ggml_cont(g, ggml_reshape_3d(g, K, head_dim, T, n_heads));
        ggml_tensor * Vs = ggml_cont(g, ggml_reshape_3d(g, V, head_dim, T, n_heads));

        // c2c: Q^T @ K → [T, T, nh]
        ggml_tensor * scores = ggml_mul_mat(g, Ks, Qs);

        ggml_tensor * P = rel_pos_expanded;  // [H, T*T]

        // c2p: project pos through K weights (with bias), dot with Q
        ggml_tensor * Pk = ggml_add(g, ggml_mul_mat(g, L.k_w, P), L.k_b);
        Pk = ggml_reshape_4d(g, Pk, head_dim, n_heads, T, T);
        ggml_tensor * Pk_b = ggml_cont(g, ggml_permute(g, Pk, 0, 2, 1, 3));
        Pk_b = ggml_reshape_3d(g, Pk_b, head_dim, T, (int64_t)n_heads * T);
        ggml_tensor * Qs_b = ggml_cont(g, ggml_permute(g, Qs, 0, 2, 1, 3));
        Qs_b = ggml_reshape_3d(g, Qs_b, head_dim, 1, (int64_t)n_heads * T);
        ggml_tensor * c2p = ggml_mul_mat(g, Pk_b, Qs_b);
        c2p = ggml_reshape_3d(g, c2p, T, n_heads, T);
        c2p = ggml_cont(g, ggml_permute(g, c2p, 0, 2, 1, 3));

        // p2c: transpose grid, project through Q weights, dot with K
        ggml_tensor * P_p2c = ggml_reshape_2d(g,
            ggml_cont(g, ggml_permute(g,
                ggml_reshape_3d(g, P, H, T, T),
                0, 2, 1, 3)),
            H, (int64_t)T * T);
        ggml_tensor * Pq = ggml_add(g, ggml_mul_mat(g, L.q_w, P_p2c), L.q_b);
        Pq = ggml_reshape_4d(g, Pq, head_dim, n_heads, T, T);
        ggml_tensor * Pq_b = ggml_cont(g, ggml_permute(g, Pq, 0, 2, 1, 3));
        Pq_b = ggml_reshape_3d(g, Pq_b, head_dim, T, (int64_t)n_heads * T);
        ggml_tensor * Ks_b2 = ggml_cont(g, ggml_permute(g, Ks, 0, 2, 1, 3));
        Ks_b2 = ggml_reshape_3d(g, Ks_b2, head_dim, 1, (int64_t)n_heads * T);
        ggml_tensor * p2c = ggml_mul_mat(g, Pq_b, Ks_b2);
        p2c = ggml_reshape_3d(g, p2c, T, n_heads, T);
        p2c = ggml_cont(g, ggml_permute(g, p2c, 1, 2, 0, 3));

        // Combine: (c2c + c2p + p2c) / sqrt(3 * head_dim)
        scores = ggml_add(g, scores, c2p);
        scores = ggml_add(g, scores, p2c);
        float scale = 1.0f / sqrtf(3.0f * (float)head_dim);
        scores = ggml_scale(g, scores, scale);
        scores = ggml_soft_max(g, scores);

        // V^T @ scores → attention output
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vs, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(g, Vt, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(g, ggml_cont(g, attn), H, T);

        // Output projection
        attn = ggml_add(g, ggml_mul_mat(g, L.o_w, attn), L.o_b);

        // Post-LN: residual + LN
        cur = ggml_add(g, inp, attn);
        cur = ggml_norm(g, cur, ln_eps);
        cur = ggml_mul(g, cur, L.ln1_w);
        cur = ggml_add(g, cur, L.ln1_b);

        // FFN: GELU
        ggml_tensor * ffn = ggml_add(g, ggml_mul_mat(g, L.fc1_w, cur), L.fc1_b);
        ffn = ggml_gelu(g, ffn);
        ffn = ggml_add(g, ggml_mul_mat(g, L.fc2_w, ffn), L.fc2_b);

        // Post-FFN LN
        cur = ggml_add(g, cur, ffn);
        cur = ggml_norm(g, cur, ln_eps);
        cur = ggml_mul(g, cur, L.ln2_w);
        cur = ggml_add(g, cur, L.ln2_b);

        char lname[32];
        snprintf(lname, sizeof(lname), "layer_%u", il);
        ggml_set_name(cur, lname);
        ggml_set_output(cur);
    }

    return cur;
}

// Expand DeBERTa relative position embeddings on CPU.
// Fills the [H, T*T] tensor with bucketed position embeddings.
static void fill_deberta_rel_pos(
    const gliner_model & model, ggml_tensor * rpe_t, int T,
    ggml_backend_t backend)
{
    const int H = (int)model.hparams.hidden_size;
    const int max_pos = (int)model.rel_embd_w->ne[1];
    const int pos_buckets = (int)model.hparams.position_buckets;

    // Read rel_embd from backend (handles quantized types via dequant)
    std::vector<float> embd_data = tensor_to_f32_backend(model.rel_embd_w, backend);

    // Apply encoder LayerNorm to relative embeddings
    if (model.encoder_ln_w && model.encoder_ln_b) {
        std::vector<float> ln_w = tensor_to_f32_backend(model.encoder_ln_w, backend);
        std::vector<float> ln_b = tensor_to_f32_backend(model.encoder_ln_b, backend);
        const float ln_eps = model.hparams.layer_norm_eps;
        for (int p = 0; p < max_pos; p++) {
            float * row = &embd_data[(size_t)p * H];
            double sum = 0.0, sum2 = 0.0;
            for (int d = 0; d < H; d++) { sum += row[d]; sum2 += (double)row[d] * row[d]; }
            float mean = (float)(sum / H);
            float var  = (float)(sum2 / H) - mean * mean;
            float inv_std = 1.0f / std::sqrt(var + ln_eps);
            for (int d = 0; d < H; d++)
                row[d] = (row[d] - mean) * inv_std * ln_w[d] + ln_b[d];
        }
    }

    // Expand with log-bucket indices
    std::vector<float> expanded((size_t)H * T * T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int bucket;
            if (pos_buckets > 0) {
                int rel = i - j;
                int sign_val = (rel > 0) ? 1 : ((rel < 0) ? -1 : 0);
                int abs_rel = std::abs(rel);
                int mid = pos_buckets / 2;
                int abs_pos = (rel < mid && rel > -mid) ? (mid - 1) : abs_rel;
                int signed_bucket;
                if (abs_pos <= mid) {
                    signed_bucket = rel;
                } else {
                    double log_ratio = std::log((double)abs_pos / mid)
                                     / std::log((double)(max_pos - 1) / mid);
                    int log_pos = (int)std::ceil(log_ratio * (mid - 1)) + mid;
                    signed_bucket = log_pos * sign_val;
                }
                bucket = signed_bucket + pos_buckets;
            } else {
                bucket = i - j + max_pos / 2;
            }
            if (bucket < 0) bucket = 0;
            if (bucket >= max_pos) bucket = max_pos - 1;
            memcpy(&expanded[(size_t)(i * T + j) * H],
                   &embd_data[(size_t)bucket * H],
                   H * sizeof(float));
        }
    }
    ggml_backend_tensor_set(rpe_t, expanded.data(), 0, expanded.size() * sizeof(float));
}

// ============================================================================
// Tensor dequantization helper — reads any ggml tensor (F32/F16/Q8_0/Q4_K/...)
// from a backend buffer and returns F32 data on CPU.
// ============================================================================

static std::vector<float> tensor_to_f32_backend(ggml_tensor * t, ggml_backend_t backend) {
    if (!t) return {};
    int64_t n = ggml_nelements(t);
    size_t raw_size = ggml_nbytes(t);
    std::vector<float> out(n);

    if (t->type == GGML_TYPE_F32) {
        // Direct copy
        ggml_backend_tensor_get(t, out.data(), 0, raw_size);
    } else {
        // Read raw quantized bytes, then dequantize
        std::vector<uint8_t> raw(raw_size);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_size);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = (const ggml_fp16_t *)raw.data();
            for (int64_t i = 0; i < n; i++)
                out[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            fprintf(stderr, "[gliner] WARNING: unsupported tensor type %d for %s, zeroing\n",
                    (int)t->type, t->name);
            memset(out.data(), 0, n * sizeof(float));
        }
    }
    return out;
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
// Uses pre-cached F32 weights from gliner_context.
static void bilstm_forward_cached(
    const float * input, float * output,
    int T, int input_size, int lstm_hidden,
    const gliner_context * ctx)
{
    const float * W_ih_fwd_p = ctx->lstm_W_ih_fwd.data();
    const float * b_ih_fwd_p = ctx->lstm_b_ih_fwd.data();
    const float * W_hh_fwd_p = ctx->lstm_W_hh_fwd.data();
    const float * b_hh_fwd_p = ctx->lstm_b_hh_fwd.data();
    const float * W_ih_rev_p = ctx->lstm_W_ih_rev.data();
    const float * b_ih_rev_p = ctx->lstm_b_ih_rev.data();
    const float * W_hh_rev_p = ctx->lstm_W_hh_rev.data();
    const float * b_hh_rev_p = ctx->lstm_b_hh_rev.data();

    // Forward direction → first half of output
    std::vector<float> fwd_out(T * lstm_hidden);
    lstm_forward_one_dir(input, fwd_out.data(), T, input_size, lstm_hidden,
                         W_ih_fwd_p, b_ih_fwd_p, W_hh_fwd_p, b_hh_fwd_p, false);

    // Reverse direction → second half of output
    std::vector<float> rev_out(T * lstm_hidden);
    lstm_forward_one_dir(input, rev_out.data(), T, input_size, lstm_hidden,
                         W_ih_rev_p, b_ih_rev_p, W_hh_rev_p, b_hh_rev_p, true);

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

    // Cache dequantized weights for CPU-side operations (BiLSTM + layer fusion)
    auto & m = ctx->model;
    ctx->lstm_W_ih_fwd = tensor_to_f32_backend(m.lstm_weight_ih_l0, ctx->backend);
    ctx->lstm_b_ih_fwd = tensor_to_f32_backend(m.lstm_bias_ih_l0, ctx->backend);
    ctx->lstm_W_hh_fwd = tensor_to_f32_backend(m.lstm_weight_hh_l0, ctx->backend);
    ctx->lstm_b_hh_fwd = tensor_to_f32_backend(m.lstm_bias_hh_l0, ctx->backend);
    ctx->lstm_W_ih_rev = tensor_to_f32_backend(m.lstm_weight_ih_l0_rev, ctx->backend);
    ctx->lstm_b_ih_rev = tensor_to_f32_backend(m.lstm_bias_ih_l0_rev, ctx->backend);
    ctx->lstm_W_hh_rev = tensor_to_f32_backend(m.lstm_weight_hh_l0_rev, ctx->backend);
    ctx->lstm_b_hh_rev = tensor_to_f32_backend(m.lstm_bias_hh_l0_rev, ctx->backend);
    if (m.hparams.backbone == GLINER_BACKBONE_LFM2) {
        ctx->fuser_squeeze_w  = tensor_to_f32_backend(m.fuser_squeeze_w, ctx->backend);
        ctx->fuser_squeeze_b  = tensor_to_f32_backend(m.fuser_squeeze_b, ctx->backend);
        ctx->fuser_W1_w       = tensor_to_f32_backend(m.fuser_W1_w, ctx->backend);
        ctx->fuser_W1_b       = tensor_to_f32_backend(m.fuser_W1_b, ctx->backend);
        ctx->fuser_W2_w       = tensor_to_f32_backend(m.fuser_W2_w, ctx->backend);
        ctx->fuser_W2_b       = tensor_to_f32_backend(m.fuser_W2_b, ctx->backend);
        ctx->fuser_out_proj_w = tensor_to_f32_backend(m.fuser_output_proj_w, ctx->backend);
        ctx->fuser_out_proj_b = tensor_to_f32_backend(m.fuser_output_proj_b, ctx->backend);
    }
    ctx->weights_cached = true;
    GDBG("cached BiLSTM weights, backbone=%s",
         m.hparams.backbone == GLINER_BACKBONE_DEBERTA ? "DeBERTa" : "LFM2");

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
    // -----------------------------------------------------------------------

    std::vector<int> ent_positions;       // positions of <<ENT>> tokens
    std::vector<std::string> ent_labels;  // corresponding label strings
    std::vector<int32_t> input_ids;

    std::string input_text(text);
    std::vector<int> token_to_word;  // for each text token, which word index
    std::vector<int> word_char_start;
    std::vector<int> word_char_end;
    std::vector<std::string> words;

    // Word splitting: matches GLiNER's WhitespaceTokenSplitter regex:
    //   r"\w+(?:[-_]\w+)*|\S"
    auto split_words = [&]() {
        int pos = 0;
        int len = (int)input_text.size();
        while (pos < len) {
            while (pos < len && (input_text[pos] == ' ' || input_text[pos] == '\t'
                                 || input_text[pos] == '\n' || input_text[pos] == '\r'))
                pos++;
            if (pos >= len) break;
            int start = pos;
            unsigned char c = (unsigned char)input_text[pos];
            auto is_word_char = [](unsigned char ch) -> bool {
                return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                       (ch >= '0' && ch <= '9') || ch == '_' || ch >= 0x80;
            };
            if (is_word_char(c)) {
                pos++;
                while (pos < len && is_word_char((unsigned char)input_text[pos])) pos++;
                while (pos < len && (input_text[pos] == '-' || input_text[pos] == '_') &&
                       pos + 1 < len && is_word_char((unsigned char)input_text[pos + 1])) {
                    pos++;
                    while (pos < len && is_word_char((unsigned char)input_text[pos])) pos++;
                }
            } else {
                pos++;
            }
            words.push_back(input_text.substr(start, pos - start));
            word_char_start.push_back(start);
            word_char_end.push_back(pos);
        }
    };
    split_words();

    int text_token_start = 0;

    if (hp.backbone == GLINER_BACKBONE_DEBERTA) {
        // DeBERTa/SPM tokenization:
        // [CLS] <<ENT>> label_subtokens <<ENT>> label_subtokens <<SEP_gliner>> text_subtokens [SEP]
        input_ids.push_back(1);  // [CLS]

        for (int i = 0; i < n_labels; i++) {
            ent_positions.push_back((int)input_ids.size());
            ent_labels.push_back(labels[i]);
            input_ids.push_back((int32_t)hp.ent_token_id);  // <<ENT>>

            // SPM-tokenize the label (encode adds ▁ prefix + BOS/EOS automatically)
            auto label_enc = model.spm_tokenizer.encode(labels[i]);
            // encode() adds BOS/EOS — strip them to get just the label tokens
            std::vector<int32_t> label_ids;
            for (size_t li = 1; li < label_enc.ids.size(); li++) {
                if (label_enc.attn_mask[li] == 0) break;
                if (label_enc.ids[li] == 2) break; // EOS
                label_ids.push_back(label_enc.ids[li]);
            }
            for (auto t : label_ids) input_ids.push_back(t);
        }
        input_ids.push_back((int32_t)hp.sep_token_id);  // <<SEP>> (GLiNER separator)

        text_token_start = (int)input_ids.size();

        // SPM-tokenize each word
        for (int wi = 0; wi < (int)words.size(); wi++) {
            auto word_enc = model.spm_tokenizer.encode(words[wi]);
            // Strip BOS/EOS
            std::vector<int32_t> word_ids;
            for (size_t ti = 1; ti < word_enc.ids.size(); ti++) {
                if (word_enc.attn_mask[ti] == 0) break;
                if (word_enc.ids[ti] == 2) break; // EOS
                word_ids.push_back(word_enc.ids[ti]);
            }
            for (auto t : word_ids) {
                input_ids.push_back(t);
                token_to_word.push_back(wi);
            }
        }
        input_ids.push_back(2);  // [SEP]
    } else {
        // LFM2/BPE tokenization
        input_ids.push_back(1);  // BOS
        for (int i = 0; i < n_labels; i++) {
            ent_positions.push_back((int)input_ids.size());
            ent_labels.push_back(labels[i]);
            input_ids.push_back((int32_t)hp.ent_token_id);
            std::string label_str(labels[i]);
            std::string encoded = core_bpe::bytes_to_unicode(label_str.data(), label_str.size());
            std::vector<int32_t> label_tokens;
            core_bpe::bpe_one(model.token_to_id, model.merge_rank, encoded, label_tokens);
            for (auto t : label_tokens) input_ids.push_back(t);
        }
        input_ids.push_back((int32_t)hp.sep_token_id);
        text_token_start = (int)input_ids.size();

        for (int wi = 0; wi < (int)words.size(); wi++) {
            std::string encoded = core_bpe::bytes_to_unicode(words[wi].data(), words[wi].size());
            std::vector<int32_t> word_tokens;
            core_bpe::bpe_one(model.token_to_id, model.merge_rank, encoded, word_tokens);
            for (auto t : word_tokens) {
                input_ids.push_back(t);
                token_to_word.push_back(wi);
            }
        }
        input_ids.push_back(7);  // EOS
    }

    int T = (int)input_ids.size();
    int n_text_tokens = T - text_token_start - 1;  // exclude EOS

    if (n_text_tokens <= 0) {
        if (out_entities) *out_entities = nullptr;
        return 0;
    }

    GDBG("input: %d tokens (%d label prefix + %d text + BOS/EOS), %d words, %d entity types",
         T, text_token_start, n_text_tokens, (int)words.size(), n_labels);

    auto t_start = std::chrono::steady_clock::now();
    auto t_prev = t_start;
    auto elapsed = [&]() -> double {
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - t_prev).count();
        t_prev = now;
        return ms;
    };
    if (g_debug) {
        fprintf(stderr, "[gliner] token IDs:");
        for (int i = 0; i < T && i < 30; i++)
            fprintf(stderr, " %d", input_ids[i]);
        if (T > 30) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    // -----------------------------------------------------------------------
    // 2. Run encoder backbone (ggml graph)
    // -----------------------------------------------------------------------

    const int enc_hidden = (int)hp.hidden_size;     // encoder output dim
    const int gl_hidden  = (int)hp.gliner_hidden;   // GLiNER head dim (may differ)
    const char * diff_ref_path = std::getenv("GLINER_DIFF_REF");

    // encoder_out: (T, enc_hidden) — per-token encoder outputs, row-major on CPU
    std::vector<float> encoder_out;

    if (hp.backbone == GLINER_BACKBONE_DEBERTA) {
        // --- DeBERTa encoder ---
        size_t buf_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(65536, false);
        ggml_init_params ip = { buf_size, nullptr, true };
        ggml_context * g = ggml_init(ip);

        ggml_tensor * rpe_tensor = nullptr;
        ggml_tensor * x = gliner_build_deberta_encoder(g, model, T, &rpe_tensor);

        // Projection: enc_hidden → gl_hidden
        ggml_tensor * proj = ggml_add(g, ggml_mul_mat(g, model.projection_w, x),
                                      model.projection_b);
        ggml_set_name(proj, "projection");
        ggml_set_output(proj);

        ggml_cgraph * gf = ggml_new_graph_custom(g, 65536, false);
        ggml_build_forward_expand(gf, proj);

        ggml_gallocr_t galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "[gliner] DeBERTa graph allocation failed\n");
            ggml_gallocr_free(galloc); ggml_free(g);
            return 0;
        }

        // Set inputs
        ggml_tensor * inp_ids = ggml_graph_get_tensor(gf, "input_ids");
        ggml_backend_tensor_set(inp_ids, input_ids.data(), 0, T * sizeof(int32_t));

        // Fill relative position embeddings
        ggml_tensor * rpe_t = ggml_graph_get_tensor(gf, "rel_pos_expanded");
        if (rpe_t) fill_deberta_rel_pos(model, rpe_t, T, ctx->backend);

        elapsed();
        ggml_backend_graph_compute(ctx->backend, gf);
        GDBG("DeBERTa encoder: %.1f ms", elapsed());

        // Diff comparison
        if (diff_ref_path) {
            crispembed_diff::Ref ref;
            if (ref.load(diff_ref_path)) {
                fprintf(stderr, "[gliner-diff] Loaded reference: %s\n", diff_ref_path);
                for (uint32_t i = 0; i < hp.n_layers; i++) {
                    char name[32];
                    snprintf(name, sizeof(name), "layer_%u", i);
                    ggml_tensor * lt = ggml_graph_get_tensor(gf, name);
                    if (!lt) continue;
                    std::vector<float> ldata(T * enc_hidden);
                    ggml_backend_tensor_get(lt, ldata.data(), 0, T * enc_hidden * sizeof(float));
                    auto r = ref.compare(name, ldata.data(), T * enc_hidden);
                    fprintf(stderr, "[gliner-diff] %s: cos=%.6f max_abs=%.2e %s\n",
                            name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    if (!r.is_pass()) break;
                }
            }
        }

        // Read projected output: [gl_hidden, T]
        encoder_out.resize(T * gl_hidden);
        ggml_backend_tensor_get(proj, encoder_out.data(), 0, T * gl_hidden * sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(g);

    } else {
        // --- LFM2 encoder + layer fusion ---
        const int n_heads = (int)hp.n_heads;
        const int n_kv = (int)hp.n_kv_heads;
        const int hd = (int)hp.head_dim;
        const float norm_eps = 1e-5f;

        g_pos_tensors.clear();
        size_t buf_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(65536, false);
        ggml_init_params ip = { buf_size, nullptr, true };
        ggml_context * g = ggml_init(ip);

        ggml_tensor * inp_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(inp_ids, "input_ids"); ggml_set_input(inp_ids);

        ggml_tensor * x = ggml_get_rows(g, model.embed_tokens_w, inp_ids);

        std::vector<ggml_tensor *> layer_outputs(hp.n_layers);
        for (uint32_t i = 0; i < hp.n_layers; i++) {
            x = gliner_build_layer(g, x, model.layers[i], enc_hidden, n_heads, n_kv,
                                   hd, T, norm_eps, hp.rope_theta);
            layer_outputs[i] = x;
            char name[32];
            snprintf(name, sizeof(name), "layer_%u", i);
            ggml_set_name(x, name);
            ggml_set_output(x);
        }

        x = gliner_rms_norm(g, x, model.embedding_norm_w, norm_eps);
        ggml_set_name(x, "final_norm");
        ggml_set_output(x);

        ggml_cgraph * gf = ggml_new_graph_custom(g, 65536, false);
        for (auto * lo : layer_outputs) ggml_build_forward_expand(gf, lo);
        ggml_build_forward_expand(gf, x);

        ggml_gallocr_t galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            fprintf(stderr, "[gliner] LFM2 graph allocation failed\n");
            ggml_gallocr_free(galloc); ggml_free(g);
            return 0;
        }

        ggml_backend_tensor_set(inp_ids, input_ids.data(), 0, T * sizeof(int32_t));
        {
            std::vector<int32_t> pos(T);
            for (int i = 0; i < T; i++) pos[i] = i;
            for (auto * pt : g_pos_tensors)
                ggml_backend_tensor_set(pt, pos.data(), 0, T * sizeof(int32_t));
        }

        elapsed();
        ggml_backend_graph_compute(ctx->backend, gf);
        GDBG("LFM2 backbone: %.1f ms", elapsed());

        // Read layer outputs for fusion
        std::vector<std::vector<float>> all_layer_outs(hp.n_layers);
        for (uint32_t i = 0; i < hp.n_layers; i++) {
            all_layer_outs[i].resize(T * enc_hidden);
            ggml_backend_tensor_get(layer_outputs[i], all_layer_outs[i].data(),
                                    0, T * enc_hidden * sizeof(float));
        }
        ggml_gallocr_free(galloc);
        ggml_free(g);

        // Layer fusion (CPU)
        int NL = (int)hp.n_layers;
        std::vector<float> layer_scores(NL);
        for (int l = 0; l < NL; l++) {
            float total = 0.0f;
            for (int t = 0; t < T; t++) {
                float s = ctx->fuser_squeeze_b[0];
                for (int d = 0; d < enc_hidden; d++)
                    s += ctx->fuser_squeeze_w[d] * all_layer_outs[l][t * enc_hidden + d];
                total += s;
            }
            layer_scores[l] = total / T;
        }
        std::vector<float> mid8(8);
        for (int j = 0; j < 8; j++) {
            float val = ctx->fuser_W1_b[j];
            for (int l = 0; l < NL; l++) val += ctx->fuser_W1_w[j * NL + l] * layer_scores[l];
            mid8[j] = val > 0.0f ? val : 0.0f;
        }
        std::vector<float> attn_weights(NL);
        for (int l = 0; l < NL; l++) {
            float val = ctx->fuser_W2_b[l];
            for (int j = 0; j < 8; j++) val += ctx->fuser_W2_w[l * 8 + j] * mid8[j];
            attn_weights[l] = 1.0f / (1.0f + expf(-val));
        }
        std::vector<float> fused(T * enc_hidden, 0.0f);
        for (int l = 0; l < NL; l++) {
            float w = attn_weights[l];
            for (int i = 0; i < T * enc_hidden; i++) fused[i] += w * all_layer_outs[l][i];
        }
        // Output projection
        encoder_out.resize(T * enc_hidden);
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < enc_hidden; d++) {
                float val = ctx->fuser_out_proj_b[d];
                for (int k = 0; k < enc_hidden; k++)
                    val += ctx->fuser_out_proj_w[d * enc_hidden + k] * fused[t * enc_hidden + k];
                encoder_out[t * enc_hidden + d] = val;
            }
        }
        GDBG("layer fusion: %.1f ms", elapsed());
    }

    // -----------------------------------------------------------------------
    // 3. Extract entity reps + first-subtoken pooling + BiLSTM (word-level)
    // -----------------------------------------------------------------------

    // The effective hidden dim for the GLiNER head
    const int head_dim_gl = gl_hidden;  // DeBERTa: 512 (after projection), LFM2: 1024

    // Extract entity type representations at <<ENT>> positions
    std::vector<float> ent_hidden_from_encoder(n_labels * head_dim_gl);
    for (int i = 0; i < n_labels; i++) {
        int pos = ent_positions[i];
        memcpy(ent_hidden_from_encoder.data() + i * head_dim_gl,
               encoder_out.data() + pos * head_dim_gl,
               head_dim_gl * sizeof(float));
    }

    // First-subtoken pooling: word-level representations
    std::vector<int> word_first_token;
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

    std::vector<float> word_reps(n_words * head_dim_gl);
    for (int w = 0; w < n_words; w++) {
        int tok_idx = text_token_start + word_first_token[w];
        memcpy(word_reps.data() + w * head_dim_gl,
               encoder_out.data() + tok_idx * head_dim_gl,
               head_dim_gl * sizeof(float));
    }

    // BiLSTM on word-level representations
    // DeBERTa: input=512, hidden=256 → output=512
    // LFM2:    input=1024, hidden=512 → output=1024
    // W_hh shape is [4*lstm_hidden, lstm_hidden], so nelements = 4*h*h → h = sqrt(n/4)
    const int lstm_hidden = (int)std::sqrt(ctx->lstm_W_hh_fwd.size() / 4.0);
    const int lstm_out_dim = 2 * lstm_hidden;
    std::vector<float> lstm_out(n_words * lstm_out_dim);
    bilstm_forward_cached(word_reps.data(), lstm_out.data(),
                          n_words, head_dim_gl, lstm_hidden, ctx);

    if (diff_ref_path) {
        crispembed_diff::Ref ref;
        if (ref.load(diff_ref_path)) {
            auto r = ref.compare("lstm_out", lstm_out.data(), n_words * lstm_out_dim);
            fprintf(stderr, "[gliner-diff] lstm_out: cos=%.6f max_abs=%.2e %s\n",
                    r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        }
    }
    GDBG("BiLSTM: %.1f ms (input=%d, hidden=%d, output=%d)", elapsed(),
         head_dim_gl, lstm_hidden, lstm_out_dim);

    // -----------------------------------------------------------------------
    // 4. GLiNER head — span scoring
    //    markerV0: concat(start, end) → out_project MLP → dot(prompt_rep)
    //    markerV1: concat(start, end, first) → out_project MLP → dot(prompt_rep)
    // -----------------------------------------------------------------------

    const bool is_v0 = (hp.span_mode == "markerV0");
    const int span_cat_dim = is_v0 ? 2 * head_dim_gl : 3 * head_dim_gl;

    auto build_mlp2 = [](ggml_context * gc, ggml_tensor * x,
                         ggml_tensor * w0, ggml_tensor * b0,
                         ggml_tensor * w3, ggml_tensor * b3) -> ggml_tensor * {
        ggml_tensor * h = ggml_mul_mat(gc, w0, x);
        h = ggml_add(gc, h, b0);
        h = ggml_relu(gc, h);
        h = ggml_mul_mat(gc, w3, h);
        h = ggml_add(gc, h, b3);
        return h;
    };

    int max_w = (int)hp.max_width;
    struct SpanIdx { int ws, we; };
    std::vector<SpanIdx> all_spans;
    for (int ws = 0; ws < n_words; ws++)
        for (int we = ws; we < std::min(ws + max_w, n_words); we++)
            all_spans.push_back({ws, we});
    int n_spans = (int)all_spans.size();
    GDBG("  %d spans x %d labels = %d scores, mode=%s", n_spans, n_labels,
         n_spans * n_labels, is_v0 ? "V0" : "V1");

    struct ScoredSpan { int word_start, word_end, label_idx; float score; };
    std::vector<ScoredSpan> candidates;

    // ---- Pass 1: proj_start/end [/first] + prompt_rep ----
    {
        size_t hbuf = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(2048, false);
        ggml_init_params hip = { hbuf, nullptr, true };
        ggml_context * hg = ggml_init(hip);

        ggml_tensor * inp_w = ggml_new_tensor_2d(hg, GGML_TYPE_F32, head_dim_gl, n_words);
        ggml_set_name(inp_w, "inp_words"); ggml_set_input(inp_w);

        ggml_tensor * inp_e = ggml_new_tensor_2d(hg, GGML_TYPE_F32, head_dim_gl, n_labels);
        ggml_set_name(inp_e, "inp_ent"); ggml_set_input(inp_e);

        ggml_tensor * ps = build_mlp2(hg, inp_w,
            model.span_proj_start_0_w, model.span_proj_start_0_b,
            model.span_proj_start_3_w, model.span_proj_start_3_b);
        ggml_set_name(ps, "proj_start"); ggml_set_output(ps);

        ggml_tensor * pe = build_mlp2(hg, inp_w,
            model.span_proj_end_0_w, model.span_proj_end_0_b,
            model.span_proj_end_3_w, model.span_proj_end_3_b);
        ggml_set_name(pe, "proj_end"); ggml_set_output(pe);

        // V1 only: proj_first (mean of words)
        ggml_tensor * pf = nullptr;
        ggml_tensor * inp_mean = nullptr;
        if (!is_v0) {
            inp_mean = ggml_new_tensor_2d(hg, GGML_TYPE_F32, head_dim_gl, 1);
            ggml_set_name(inp_mean, "inp_mean"); ggml_set_input(inp_mean);
            pf = build_mlp2(hg, inp_mean,
                model.span_proj_first_0_w, model.span_proj_first_0_b,
                model.span_proj_first_3_w, model.span_proj_first_3_b);
            ggml_set_name(pf, "proj_first"); ggml_set_output(pf);
        }

        ggml_tensor * er = build_mlp2(hg, inp_e,
            model.prompt_rep_0_w, model.prompt_rep_0_b,
            model.prompt_rep_3_w, model.prompt_rep_3_b);
        ggml_set_name(er, "ent_reps"); ggml_set_output(er);

        ggml_cgraph * gf1 = ggml_new_graph_custom(hg, 2048, false);
        ggml_build_forward_expand(gf1, ps);
        ggml_build_forward_expand(gf1, pe);
        if (pf) ggml_build_forward_expand(gf1, pf);
        ggml_build_forward_expand(gf1, er);

        ggml_gallocr_t ga1 = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(ga1, gf1)) {
            fprintf(stderr, "[gliner] pass1 graph alloc failed\n");
            ggml_gallocr_free(ga1); ggml_free(hg);
            return 0;
        }

        ggml_backend_tensor_set(inp_w, lstm_out.data(), 0,
                                n_words * head_dim_gl * sizeof(float));
        ggml_backend_tensor_set(inp_e, ent_hidden_from_encoder.data(), 0,
                                n_labels * head_dim_gl * sizeof(float));

        if (!is_v0 && inp_mean) {
            std::vector<float> mean_vec(head_dim_gl, 0.0f);
            for (int w = 0; w < n_words; w++)
                for (int d = 0; d < head_dim_gl; d++)
                    mean_vec[d] += lstm_out[w * head_dim_gl + d];
            for (int d = 0; d < head_dim_gl; d++) mean_vec[d] /= n_words;
            ggml_backend_tensor_set(inp_mean, mean_vec.data(), 0,
                                    head_dim_gl * sizeof(float));
        }

        elapsed();
        ggml_backend_graph_compute(ctx->backend, gf1);
        GDBG("head pass1: %.1f ms", elapsed());

        std::vector<float> ps_data(n_words * head_dim_gl), pe_data(n_words * head_dim_gl);
        std::vector<float> pf_data(is_v0 ? 0 : head_dim_gl);
        std::vector<float> er_data(n_labels * head_dim_gl);
        ggml_backend_tensor_get(ps, ps_data.data(), 0, ps_data.size() * sizeof(float));
        ggml_backend_tensor_get(pe, pe_data.data(), 0, pe_data.size() * sizeof(float));
        if (!is_v0 && pf)
            ggml_backend_tensor_get(pf, pf_data.data(), 0, pf_data.size() * sizeof(float));
        ggml_backend_tensor_get(er, er_data.data(), 0, er_data.size() * sizeof(float));

        ggml_gallocr_free(ga1);
        ggml_free(hg);

        // Assemble span concatenations
        std::vector<float> span_cat(n_spans * span_cat_dim);
        for (int s = 0; s < n_spans; s++) {
            float * dst = span_cat.data() + s * span_cat_dim;
            memcpy(dst, ps_data.data() + all_spans[s].ws * head_dim_gl,
                   head_dim_gl * sizeof(float));
            memcpy(dst + head_dim_gl, pe_data.data() + all_spans[s].we * head_dim_gl,
                   head_dim_gl * sizeof(float));
            if (!is_v0)
                memcpy(dst + 2 * head_dim_gl, pf_data.data(),
                       head_dim_gl * sizeof(float));
        }

        // ---- Pass 2: out_project + scoring ----
        size_t hbuf2 = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(1024, false);
        ggml_init_params hip2 = { hbuf2, nullptr, true };
        ggml_context * hg2 = ggml_init(hip2);

        ggml_tensor * inp_sp = ggml_new_tensor_2d(hg2, GGML_TYPE_F32, span_cat_dim, n_spans);
        ggml_set_name(inp_sp, "inp_spans"); ggml_set_input(inp_sp);

        ggml_tensor * inp_er = ggml_new_tensor_2d(hg2, GGML_TYPE_F32, head_dim_gl, n_labels);
        ggml_set_name(inp_er, "inp_ent_reps"); ggml_set_input(inp_er);

        ggml_tensor * sr = build_mlp2(hg2, inp_sp,
            model.span_out_proj_0_w, model.span_out_proj_0_b,
            model.span_out_proj_3_w, model.span_out_proj_3_b);

        ggml_tensor * scores = ggml_mul_mat(hg2, inp_er, sr);
        ggml_set_name(scores, "scores"); ggml_set_output(scores);

        ggml_cgraph * gf2 = ggml_new_graph_custom(hg2, 1024, false);
        ggml_build_forward_expand(gf2, scores);

        ggml_gallocr_t ga2 = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(ga2, gf2)) {
            fprintf(stderr, "[gliner] pass2 graph alloc failed\n");
            ggml_gallocr_free(ga2); ggml_free(hg2);
            return 0;
        }

        ggml_backend_tensor_set(inp_sp, span_cat.data(), 0, span_cat.size() * sizeof(float));
        ggml_backend_tensor_set(inp_er, er_data.data(), 0, er_data.size() * sizeof(float));

        ggml_backend_graph_compute(ctx->backend, gf2);
        GDBG("head pass2: %.1f ms", elapsed());

        std::vector<float> scores_raw(n_labels * n_spans);
        ggml_backend_tensor_get(scores, scores_raw.data(), 0, scores_raw.size() * sizeof(float));

        ggml_gallocr_free(ga2);
        ggml_free(hg2);

        // Threshold + collect candidates
        for (int s = 0; s < n_spans; s++) {
            for (int e = 0; e < n_labels; e++) {
                float dot = scores_raw[s * n_labels + e];
                float sc = 1.0f / (1.0f + expf(-dot));
                if (sc >= threshold)
                    candidates.push_back({all_spans[s].ws, all_spans[s].we, e, sc});
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

    {
        auto t_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        GDBG("total: %.1f ms, extracted %zu entities", total_ms, ctx->result_entities.size());
    }
    return (int)ctx->result_entities.size();
}
