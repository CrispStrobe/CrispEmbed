// mixtex_ocr.cpp — MixTex Chinese+English LaTeX OCR
//
// Swin-Tiny encoder + 4-layer RoBERTa decoder with cross-attention.
// CPU-scalar forward pass. BPE tokenizer from GGUF metadata.
//
// Swin encoder:
//   Patch embed: Conv2d(3, 96, 4×4, stride=4) + LayerNorm
//   4 stages: depths=[2,2,6,2], heads=[3,6,12,24]
//   Each block: LN → WindowMSHA(+RPB) → residual → LN → FFN → residual
//   Odd blocks use shifted windows (shift = window_size/2)
//   Downsample between stages: concat 2×2 patches + Linear + LN
//
// RoBERTa decoder:
//   Word embed + Position embed + TypeEmbed + LN
//   4 layers: self-attn → LN → cross-attn → LN → FFN → LN
//   LM head: Dense → GELU → LN → projection (tied to word_embed)
//
// Debug: set env MIXTEX_DUMP=1 for per-layer stats.

#include "mixtex_ocr.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
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
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// CPU math helpers
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
        out[i] = ((in[i] - (float)mean) * s) * w[i] + b[i];
}

static void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                        const float* weight, const float* bias) {
    for (int o = 0; o < out_dim; o++) {
        float s = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_dim; i++)
            s += in[i] * weight[o * in_dim + i];
        out[o] = s;
    }
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax(float* data, int n) {
    float mx = data[0];
    for (int i = 1; i < n; i++) if (data[i] > mx) mx = data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { data[i] = expf(data[i] - mx); sum += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= sum;
}

// Multi-head attention (single query position)
static void mha_1q_cpu(const float* q, const float* k, const float* v,
                        float* out, int n_kv, int D, int n_heads,
                        const float* attn_bias = nullptr) {
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_kv);
        for (int t = 0; t < n_kv; t++) {
            float dot = 0;
            for (int d = 0; d < hd; d++)
                dot += q[off + d] * k[t * D + off + d];
            scores[t] = dot * scale;
            if (attn_bias) scores[t] += attn_bias[h * n_kv + t]; // NOT USED for decoder
        }
        softmax(scores.data(), n_kv);
        for (int d = 0; d < hd; d++) {
            float sum = 0;
            for (int t = 0; t < n_kv; t++)
                sum += scores[t] * v[t * D + off + d];
            out[off + d] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Swin window attention helpers
// ---------------------------------------------------------------------------

// Window partition: [H, W, C] → [nWindows, window_size², C]
static void window_partition(const float* x, float* out,
                              int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    // out shape: [nH*nW, ws*ws, C]
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int y = 0; y < ws; y++) {
                for (int x_pos = 0; x_pos < ws; x_pos++) {
                    int src_y = wh * ws + y;
                    int src_x = ww * ws + x_pos;
                    int token_idx = y * ws + x_pos;
                    memcpy(out + (win_idx * ws * ws + token_idx) * C,
                           x + (src_y * W + src_x) * C,
                           C * sizeof(float));
                }
            }
        }
    }
}

// Window reverse: [nWindows, window_size², C] → [H, W, C]
static void window_reverse(const float* windows, float* out,
                            int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int y = 0; y < ws; y++) {
                for (int x_pos = 0; x_pos < ws; x_pos++) {
                    int dst_y = wh * ws + y;
                    int dst_x = ww * ws + x_pos;
                    int token_idx = y * ws + x_pos;
                    memcpy(out + (dst_y * W + dst_x) * C,
                           windows + (win_idx * ws * ws + token_idx) * C,
                           C * sizeof(float));
                }
            }
        }
    }
}

// Cyclic shift: shift [H, W, C] by (shift_h, shift_w) with wrap-around
static void cyclic_shift(const float* in, float* out,
                          int H, int W, int C, int shift_h, int shift_w) {
    for (int y = 0; y < H; y++) {
        int src_y = (y + shift_h + H) % H;
        for (int x = 0; x < W; x++) {
            int src_x = (x + shift_w + W) % W;
            memcpy(out + (y * W + x) * C,
                   in + (src_y * W + src_x) * C,
                   C * sizeof(float));
        }
    }
}

// Window multi-head self-attention with relative position bias
static void window_mhsa(const float* tokens, float* out,
                         int n_tokens, int D, int n_heads,
                         const float* q_w, const float* q_b,
                         const float* k_w, const float* k_b,
                         const float* v_w, const float* v_b,
                         const float* out_w, const float* out_b,
                         const float* rpb_table, const float* rpb_index,
                         int rpb_table_len) {
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);

    // Project Q, K, V: [n_tokens, D] → [n_tokens, D]
    std::vector<float> Q(n_tokens * D), K(n_tokens * D), V(n_tokens * D);
    for (int t = 0; t < n_tokens; t++) {
        linear_cpu(tokens + t * D, Q.data() + t * D, D, D, q_w, q_b);
        linear_cpu(tokens + t * D, K.data() + t * D, D, D, k_w, k_b);
        linear_cpu(tokens + t * D, V.data() + t * D, D, D, v_w, v_b);
    }

    // Attention per head
    std::vector<float> attn_out(n_tokens * D);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;

        // Compute attention scores [n_tokens, n_tokens]
        std::vector<float> scores(n_tokens * n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += Q[i * D + off + d] * K[j * D + off + d];
                scores[i * n_tokens + j] = dot * scale;

                // Add relative position bias
                if (rpb_table && rpb_index) {
                    int idx = (int)rpb_index[i * n_tokens + j];
                    if (idx >= 0 && idx < rpb_table_len)
                        scores[i * n_tokens + j] += rpb_table[idx * n_heads + h];
                }
            }
        }

        // Softmax per row
        for (int i = 0; i < n_tokens; i++)
            softmax(scores.data() + i * n_tokens, n_tokens);

        // Weighted sum of V
        for (int i = 0; i < n_tokens; i++) {
            for (int d = 0; d < hd; d++) {
                float sum = 0;
                for (int j = 0; j < n_tokens; j++)
                    sum += scores[i * n_tokens + j] * V[j * D + off + d];
                attn_out[i * D + off + d] = sum;
            }
        }
    }

    // Output projection
    for (int t = 0; t < n_tokens; t++) {
        linear_cpu(attn_out.data() + t * D, out + t * D, D, D, out_w, out_b);
    }
}

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------
struct swin_block_weights {
    ggml_tensor *ln1_w, *ln1_b;     // pre-attention LayerNorm
    ggml_tensor *q_w, *q_b;         // Q projection
    ggml_tensor *k_w, *k_b;         // K projection
    ggml_tensor *v_w, *v_b;         // V projection
    ggml_tensor *out_w, *out_b;     // output projection
    ggml_tensor *rpb_table;          // relative position bias [169, n_heads]
    ggml_tensor *rpb_index;          // relative position index [49, 49]
    ggml_tensor *ln2_w, *ln2_b;     // pre-FFN LayerNorm
    ggml_tensor *ffn_up_w, *ffn_up_b;   // FFN up [4*D, D]
    ggml_tensor *ffn_down_w, *ffn_down_b; // FFN down [D, 4*D]
};

struct swin_downsample_weights {
    ggml_tensor *norm_w, *norm_b;    // LayerNorm
    ggml_tensor *reduction_w;        // Linear [2*C, 4*C] (no bias)
};

struct dec_layer_weights {
    ggml_tensor *self_ln_w, *self_ln_b;
    ggml_tensor *self_q_w, *self_q_b;
    ggml_tensor *self_k_w, *self_k_b;
    ggml_tensor *self_v_w, *self_v_b;
    ggml_tensor *self_out_w, *self_out_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b;
    ggml_tensor *cross_k_w, *cross_k_b;
    ggml_tensor *cross_v_w, *cross_v_b;
    ggml_tensor *cross_out_w, *cross_out_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b;
    ggml_tensor *ffn_up_w, *ffn_up_b;
    ggml_tensor *ffn_down_w, *ffn_down_b;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct mixtex_ocr_context {
    mixtex_ocr_hparams hp;
    int n_threads;
    bool dump;

    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;

    // Encoder: patch embedding
    ggml_tensor *patch_w, *patch_b;
    ggml_tensor *patch_norm_w, *patch_norm_b;

    // Encoder: stages
    std::vector<swin_block_weights> stage_blocks[4]; // [stage][block]
    swin_downsample_weights downsample[3]; // between stages 0-1, 1-2, 2-3

    // Encoder: final LayerNorm
    ggml_tensor *enc_final_norm_w, *enc_final_norm_b;

    // Decoder: embeddings
    ggml_tensor *word_embed_w;
    ggml_tensor *pos_embed_w;
    ggml_tensor *type_embed_w;
    ggml_tensor *embed_ln_w, *embed_ln_b;

    // Decoder: layers
    dec_layer_weights dec_layers[4];

    // Decoder: LM head
    ggml_tensor *lm_dense_w, *lm_dense_b;
    ggml_tensor *lm_ln_w, *lm_ln_b;
    ggml_tensor *lm_bias; // output bias [vocab]

    // Tokenizer
    std::vector<std::string> vocab;

    // Output buffer
    std::string output_text;

    // KV cache for decoder
    std::vector<std::vector<float>> kv_cache_k; // [layer][step * D]
    std::vector<std::vector<float>> kv_cache_v;
};

// Debug helper
static void dump_stats(const char* name, const float* data, int n) {
    float mn = data[0], mx = data[0];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  %-30s: min=%8.4f max=%8.4f mean=%8.4f  [n=%d]\n",
            name, mn, mx, (float)(sum / n), n);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
static ggml_tensor* find(const std::map<std::string, ggml_tensor*>& m, const char* name) {
    return core_gguf::try_get(m, name);
}

mixtex_ocr_context * mixtex_ocr_init(const char * model_path, int n_threads) {
    auto* ctx = new mixtex_ocr_context{};
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->dump = (getenv("MIXTEX_DUMP") != nullptr);

    // Pass 1: metadata
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) { delete ctx; return nullptr; }

    auto& hp = ctx->hp;
    hp.patch_size = core_gguf::kv_u32(gctx, "mixtex.encoder.patch_size", 4);
    hp.window_size = core_gguf::kv_u32(gctx, "mixtex.encoder.window_size", 7);
    hp.embed_dim = core_gguf::kv_u32(gctx, "mixtex.encoder.embed_dim", 96);
    hp.enc_hidden = core_gguf::kv_u32(gctx, "mixtex.encoder.hidden_size", 768);
    hp.image_h = core_gguf::kv_u32(gctx, "mixtex.encoder.image_h", 400);
    hp.image_w = core_gguf::kv_u32(gctx, "mixtex.encoder.image_w", 500);

    hp.dec_hidden = core_gguf::kv_u32(gctx, "mixtex.decoder.hidden_size", 768);
    hp.dec_layers = core_gguf::kv_u32(gctx, "mixtex.decoder.num_layers", 4);
    hp.dec_heads = core_gguf::kv_u32(gctx, "mixtex.decoder.num_heads", 12);
    hp.dec_ffn = core_gguf::kv_u32(gctx, "mixtex.decoder.ffn_dim", 3072);
    hp.vocab_size = core_gguf::kv_u32(gctx, "mixtex.decoder.vocab_size", 25681);
    hp.max_position = core_gguf::kv_u32(gctx, "mixtex.decoder.max_position", 300);
    hp.sos_token = core_gguf::kv_u32(gctx, "mixtex.decoder.sos_token", 0);
    hp.eos_token = core_gguf::kv_u32(gctx, "mixtex.decoder.eos_token", 25678);

    // Depths/heads arrays
    hp.enc_depths[0] = 2; hp.enc_depths[1] = 2; hp.enc_depths[2] = 6; hp.enc_depths[3] = 2;
    hp.enc_heads[0] = 3; hp.enc_heads[1] = 6; hp.enc_heads[2] = 12; hp.enc_heads[3] = 24;

    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "mixtex_ocr: patch=%d win=%d embed=%d hidden=%d vocab=%d(%zu)\n",
            hp.patch_size, hp.window_size, hp.embed_dim, hp.enc_hidden,
            hp.vocab_size, ctx->vocab.size());

    // Pass 2: weights
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (!core_gguf::load_weights(model_path, ctx->backend, "mixtex_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    const auto& m = ctx->wl.tensors;
    char buf[256];
    auto T = [&](const char* fmt, ...) -> ggml_tensor* {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        return find(m, buf);
    };

    // Patch embedding
    ctx->patch_w = find(m, "enc.patch.weight");
    ctx->patch_b = find(m, "enc.patch.bias");
    ctx->patch_norm_w = find(m, "enc.patch_norm.weight");
    ctx->patch_norm_b = find(m, "enc.patch_norm.bias");

    // Swin stages
    for (int s = 0; s < 4; s++) {
        ctx->stage_blocks[s].resize(hp.enc_depths[s]);
        for (int b = 0; b < hp.enc_depths[s]; b++) {
            auto& blk = ctx->stage_blocks[s][b];
            blk.ln1_w = T("enc.stage%d.block%d.ln1.weight", s, b);
            blk.ln1_b = T("enc.stage%d.block%d.ln1.bias", s, b);
            blk.q_w = T("enc.stage%d.block%d.attn.q.weight", s, b);
            blk.q_b = T("enc.stage%d.block%d.attn.q.bias", s, b);
            blk.k_w = T("enc.stage%d.block%d.attn.k.weight", s, b);
            blk.k_b = T("enc.stage%d.block%d.attn.k.bias", s, b);
            blk.v_w = T("enc.stage%d.block%d.attn.v.weight", s, b);
            blk.v_b = T("enc.stage%d.block%d.attn.v.bias", s, b);
            blk.out_w = T("enc.stage%d.block%d.attn.out.weight", s, b);
            blk.out_b = T("enc.stage%d.block%d.attn.out.bias", s, b);
            blk.rpb_table = T("enc.stage%d.block%d.attn.rpb_table", s, b);
            blk.rpb_index = T("enc.stage%d.block%d.attn.rpb_index", s, b);
            blk.ln2_w = T("enc.stage%d.block%d.ln2.weight", s, b);
            blk.ln2_b = T("enc.stage%d.block%d.ln2.bias", s, b);
            blk.ffn_up_w = T("enc.stage%d.block%d.ffn.up.weight", s, b);
            blk.ffn_up_b = T("enc.stage%d.block%d.ffn.up.bias", s, b);
            blk.ffn_down_w = T("enc.stage%d.block%d.ffn.down.weight", s, b);
            blk.ffn_down_b = T("enc.stage%d.block%d.ffn.down.bias", s, b);
        }
        if (s < 3) {
            ctx->downsample[s].norm_w = T("enc.stage%d.downsample.norm.weight", s);
            ctx->downsample[s].norm_b = T("enc.stage%d.downsample.norm.bias", s);
            ctx->downsample[s].reduction_w = T("enc.stage%d.downsample.reduction.weight", s);
        }
    }

    ctx->enc_final_norm_w = find(m, "enc.final_norm.weight");
    ctx->enc_final_norm_b = find(m, "enc.final_norm.bias");

    // Decoder
    ctx->word_embed_w = find(m, "dec.word_embed.weight");
    ctx->pos_embed_w = find(m, "dec.pos_embed.weight");
    ctx->type_embed_w = find(m, "dec.type_embed.weight");
    ctx->embed_ln_w = find(m, "dec.embed_ln.weight");
    ctx->embed_ln_b = find(m, "dec.embed_ln.bias");

    for (int i = 0; i < 4; i++) {
        auto& l = ctx->dec_layers[i];
        l.self_ln_w = T("dec.layers.%d.self_ln.weight", i);
        l.self_ln_b = T("dec.layers.%d.self_ln.bias", i);
        l.self_q_w = T("dec.layers.%d.self_q.weight", i);
        l.self_q_b = T("dec.layers.%d.self_q.bias", i);
        l.self_k_w = T("dec.layers.%d.self_k.weight", i);
        l.self_k_b = T("dec.layers.%d.self_k.bias", i);
        l.self_v_w = T("dec.layers.%d.self_v.weight", i);
        l.self_v_b = T("dec.layers.%d.self_v.bias", i);
        l.self_out_w = T("dec.layers.%d.self_out.weight", i);
        l.self_out_b = T("dec.layers.%d.self_out.bias", i);
        l.cross_ln_w = T("dec.layers.%d.cross_ln.weight", i);
        l.cross_ln_b = T("dec.layers.%d.cross_ln.bias", i);
        l.cross_q_w = T("dec.layers.%d.cross_q.weight", i);
        l.cross_q_b = T("dec.layers.%d.cross_q.bias", i);
        l.cross_k_w = T("dec.layers.%d.cross_k.weight", i);
        l.cross_k_b = T("dec.layers.%d.cross_k.bias", i);
        l.cross_v_w = T("dec.layers.%d.cross_v.weight", i);
        l.cross_v_b = T("dec.layers.%d.cross_v.bias", i);
        l.cross_out_w = T("dec.layers.%d.cross_out.weight", i);
        l.cross_out_b = T("dec.layers.%d.cross_out.bias", i);
        l.ffn_ln_w = T("dec.layers.%d.ffn_ln.weight", i);
        l.ffn_ln_b = T("dec.layers.%d.ffn_ln.bias", i);
        l.ffn_up_w = T("dec.layers.%d.ffn.up.weight", i);
        l.ffn_up_b = T("dec.layers.%d.ffn.up.bias", i);
        l.ffn_down_w = T("dec.layers.%d.ffn.down.weight", i);
        l.ffn_down_b = T("dec.layers.%d.ffn.down.bias", i);
    }

    ctx->lm_dense_w = find(m, "dec.lm_head.dense.weight");
    ctx->lm_dense_b = find(m, "dec.lm_head.dense.bias");
    ctx->lm_ln_w = find(m, "dec.lm_head.ln.weight");
    ctx->lm_ln_b = find(m, "dec.lm_head.ln.bias");
    ctx->lm_bias = find(m, "dec.lm_head.bias");

    fprintf(stderr, "mixtex_ocr: loaded %s\n", model_path);
    return ctx;
}

void mixtex_ocr_free(mixtex_ocr_context * ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

const mixtex_ocr_hparams * mixtex_ocr_get_hparams(const mixtex_ocr_context * ctx) {
    return ctx ? &ctx->hp : nullptr;
}

// ---------------------------------------------------------------------------
// Swin encoder forward pass
// ---------------------------------------------------------------------------
static std::vector<float> run_swin_encoder(mixtex_ocr_context* ctx,
                                            const float* pixels_chw,
                                            int img_h, int img_w) {
    auto& hp = ctx->hp;
    int D = hp.embed_dim; // 96
    int ws = hp.window_size; // 7
    int ps = hp.patch_size; // 4

    // Patch embedding: Conv2d(3, 96, 4×4, stride=4) — manual implementation
    int pH = img_h / ps; // 100
    int pW = img_w / ps; // 125
    int N = pH * pW; // 12500

    auto patch_w = to_f32(ctx->patch_w); // [96, 3, 4, 4]
    auto patch_b = to_f32(ctx->patch_b); // [96]

    // Apply patch embedding
    std::vector<float> patches(N * D);
    for (int py = 0; py < pH; py++) {
        for (int px = 0; px < pW; px++) {
            int pos = py * pW + px;
            for (int oc = 0; oc < D; oc++) {
                float sum = patch_b[oc];
                for (int ic = 0; ic < 3; ic++) {
                    for (int ky = 0; ky < ps; ky++) {
                        for (int kx = 0; kx < ps; kx++) {
                            int iy = py * ps + ky;
                            int ix = px * ps + kx;
                            float pixel = pixels_chw[ic * img_h * img_w + iy * img_w + ix];
                            sum += pixel * patch_w[oc * 3 * ps * ps + ic * ps * ps + ky * ps + kx];
                        }
                    }
                }
                patches[pos * D + oc] = sum;
            }
        }
    }

    // LayerNorm after patch embedding
    auto pn_w = to_f32(ctx->patch_norm_w);
    auto pn_b = to_f32(ctx->patch_norm_b);
    for (int i = 0; i < N; i++)
        layernorm_cpu(patches.data() + i * D, patches.data() + i * D,
                      D, pn_w.data(), pn_b.data());

    if (ctx->dump) dump_stats("enc_embed", patches.data(), N * D);

    // Process each stage
    int H = pH, W = pW;
    std::vector<float> x = std::move(patches);

    for (int stage = 0; stage < 4; stage++) {
        int n_blocks = hp.enc_depths[stage];
        int n_heads = hp.enc_heads[stage];

        for (int bi = 0; bi < n_blocks; bi++) {
            auto& blk = ctx->stage_blocks[stage][bi];
            int HW = H * W;

            // Pre-attention LayerNorm
            std::vector<float> normed(HW * D);
            auto ln1_w = to_f32(blk.ln1_w), ln1_b = to_f32(blk.ln1_b);
            for (int i = 0; i < HW; i++)
                layernorm_cpu(x.data() + i * D, normed.data() + i * D,
                              D, ln1_w.data(), ln1_b.data());

            // Shifted window attention
            bool shifted = (bi % 2 == 1);
            int shift = shifted ? ws / 2 : 0;

            // Reshape to [H, W, D] for window operations
            // (already in this layout: x[pos * D + d] where pos = y * W + x_pos)

            // Cyclic shift
            std::vector<float> shifted_x(HW * D);
            if (shifted) {
                cyclic_shift(normed.data(), shifted_x.data(), H, W, D, -shift, -shift);
            } else {
                shifted_x = normed;
            }

            // Pad H and W to multiples of window_size
            int pad_h = (ws - H % ws) % ws;
            int pad_w = (ws - W % ws) % ws;
            int pH2 = H + pad_h, pW2 = W + pad_w;

            std::vector<float> padded;
            if (pad_h > 0 || pad_w > 0) {
                padded.resize(pH2 * pW2 * D, 0);
                for (int y = 0; y < H; y++)
                    memcpy(padded.data() + y * pW2 * D, shifted_x.data() + y * W * D, W * D * sizeof(float));
                shifted_x = std::move(padded);
            }

            // Window partition
            int nH = pH2 / ws, nW = pW2 / ws;
            int n_windows = nH * nW;
            int tokens_per_win = ws * ws;
            std::vector<float> windows(n_windows * tokens_per_win * D);
            window_partition(shifted_x.data(), windows.data(), pH2, pW2, D, ws);

            // Attention per window
            auto q_w = to_f32(blk.q_w), q_b = to_f32(blk.q_b);
            auto k_w = to_f32(blk.k_w), k_b = to_f32(blk.k_b);
            auto v_w = to_f32(blk.v_w), v_b = to_f32(blk.v_b);
            auto out_w = to_f32(blk.out_w), out_b = to_f32(blk.out_b);
            auto rpb_t = to_f32(blk.rpb_table);
            auto rpb_i = to_f32(blk.rpb_index);
            // rpb_table ggml shape: ne[0]=n_heads, ne[1]=num_entries (169 for ws=7)
            int rpb_len = blk.rpb_table ? (int)blk.rpb_table->ne[1] : 0;

            std::vector<float> attn_out(n_windows * tokens_per_win * D);
            for (int w = 0; w < n_windows; w++) {
                window_mhsa(
                    windows.data() + w * tokens_per_win * D,
                    attn_out.data() + w * tokens_per_win * D,
                    tokens_per_win, D, n_heads,
                    q_w.data(), q_b.data(),
                    k_w.data(), k_b.data(),
                    v_w.data(), v_b.data(),
                    out_w.data(), out_b.data(),
                    rpb_t.empty() ? nullptr : rpb_t.data(),
                    rpb_i.empty() ? nullptr : rpb_i.data(),
                    rpb_len);
            }

            // Window reverse
            std::vector<float> merged(pH2 * pW2 * D);
            window_reverse(attn_out.data(), merged.data(), pH2, pW2, D, ws);

            // Remove padding
            if (pad_h > 0 || pad_w > 0) {
                std::vector<float> unpadded(H * W * D);
                for (int y = 0; y < H; y++)
                    memcpy(unpadded.data() + y * W * D, merged.data() + y * pW2 * D, W * D * sizeof(float));
                merged = std::move(unpadded);
            }

            // Reverse cyclic shift
            if (shifted) {
                std::vector<float> unshifted(HW * D);
                cyclic_shift(merged.data(), unshifted.data(), H, W, D, shift, shift);
                merged = std::move(unshifted);
            }

            // Residual
            for (int i = 0; i < HW * D; i++) x[i] += merged[i];

            // FFN: LN → up → GELU → down → residual
            auto ln2_w = to_f32(blk.ln2_w), ln2_b = to_f32(blk.ln2_b);
            auto up_w = to_f32(blk.ffn_up_w), up_b = to_f32(blk.ffn_up_b);
            auto down_w = to_f32(blk.ffn_down_w), down_b = to_f32(blk.ffn_down_b);
            int ffn_dim = (int)blk.ffn_up_w->ne[1]; // output dim of up proj

            for (int i = 0; i < HW; i++) {
                std::vector<float> ln(D), up(ffn_dim), down(D);
                layernorm_cpu(x.data() + i * D, ln.data(), D, ln2_w.data(), ln2_b.data());
                linear_cpu(ln.data(), up.data(), D, ffn_dim, up_w.data(), up_b.data());
                for (int j = 0; j < ffn_dim; j++) up[j] = gelu(up[j]);
                linear_cpu(up.data(), down.data(), ffn_dim, D, down_w.data(), down_b.data());
                for (int d = 0; d < D; d++) x[i * D + d] += down[d];
            }
        }

        if (ctx->dump) {
            char name[64];
            snprintf(name, sizeof(name), "enc_stage_%d", stage);
            dump_stats(name, x.data(), H * W * D);
        }

        // Downsample (patch merging) between stages
        if (stage < 3) {
            auto& ds = ctx->downsample[stage];
            auto norm_w = to_f32(ds.norm_w), norm_b = to_f32(ds.norm_b);
            auto red_w = to_f32(ds.reduction_w); // [2*D, 4*D]
            int new_D = D * 2;

            // Pad H/W to even if odd (Swin PatchMerging pads before merge)
            int padH = H + (H % 2);
            int padW = W + (W % 2);
            std::vector<float> padded;
            const float* merge_src = x.data();
            if (padH != H || padW != W) {
                padded.resize(padH * padW * D, 0.0f);
                for (int y = 0; y < H; y++)
                    memcpy(padded.data() + y * padW * D, x.data() + y * W * D, W * D * sizeof(float));
                merge_src = padded.data();
            }

            int newH = padH / 2, newW = padW / 2;
            int newN = newH * newW;

            // Concat 2×2 patches → [newN, 4*D]
            std::vector<float> merged(newN * 4 * D);
            for (int y = 0; y < newH; y++) {
                for (int xi = 0; xi < newW; xi++) {
                    int dst = y * newW + xi;
                    int s0 = (2*y) * padW + (2*xi);
                    int s1 = (2*y) * padW + (2*xi+1);
                    int s2 = (2*y+1) * padW + (2*xi);
                    int s3 = (2*y+1) * padW + (2*xi+1);
                    memcpy(merged.data() + dst * 4 * D + 0 * D, merge_src + s0 * D, D * sizeof(float));
                    memcpy(merged.data() + dst * 4 * D + 1 * D, merge_src + s1 * D, D * sizeof(float));
                    memcpy(merged.data() + dst * 4 * D + 2 * D, merge_src + s2 * D, D * sizeof(float));
                    memcpy(merged.data() + dst * 4 * D + 3 * D, merge_src + s3 * D, D * sizeof(float));
                }
            }

            // LayerNorm on 4*D
            for (int i = 0; i < newN; i++)
                layernorm_cpu(merged.data() + i * 4 * D, merged.data() + i * 4 * D,
                              4 * D, norm_w.data(), norm_b.data());

            // Linear reduction: [newN, 4*D] → [newN, 2*D]
            std::vector<float> reduced(newN * new_D);
            for (int i = 0; i < newN; i++)
                linear_cpu(merged.data() + i * 4 * D, reduced.data() + i * new_D,
                           4 * D, new_D, red_w.data(), nullptr);

            x = std::move(reduced);
            H = newH; W = newW; D = new_D;
        }
    }

    // Final LayerNorm
    if (ctx->enc_final_norm_w) {
        auto fn_w = to_f32(ctx->enc_final_norm_w);
        auto fn_b = to_f32(ctx->enc_final_norm_b);
        int N_out = H * W;
        for (int i = 0; i < N_out; i++)
            layernorm_cpu(x.data() + i * D, x.data() + i * D, D, fn_w.data(), fn_b.data());
    }

    if (ctx->dump) dump_stats("enc_output", x.data(), H * W * D);

    return x; // [N, D] where N = H*W, D = 768
}

// ---------------------------------------------------------------------------
// Decoder forward pass (autoregressive greedy)
// ---------------------------------------------------------------------------
static std::string run_decoder(mixtex_ocr_context* ctx,
                                const float* enc_output, int enc_len, int enc_dim) {
    auto& hp = ctx->hp;
    int D = hp.dec_hidden;
    int n_layers = hp.dec_layers;

    auto word_w = to_f32(ctx->word_embed_w);   // [vocab, D]
    auto pos_w = to_f32(ctx->pos_embed_w);     // [max_pos, D]
    auto type_w = to_f32(ctx->type_embed_w);   // [1, D]
    auto eln_w = to_f32(ctx->embed_ln_w);
    auto eln_b = to_f32(ctx->embed_ln_b);

    // Pre-compute cross-attention K,V for each layer (constant across decode steps)
    struct cross_kv {
        std::vector<float> K, V;
    };
    std::vector<cross_kv> cross_kvs(n_layers);
    for (int li = 0; li < n_layers; li++) {
        auto& l = ctx->dec_layers[li];
        auto ck_w = to_f32(l.cross_k_w), ck_b = to_f32(l.cross_k_b);
        auto cv_w = to_f32(l.cross_v_w), cv_b = to_f32(l.cross_v_b);
        cross_kvs[li].K.resize(enc_len * D);
        cross_kvs[li].V.resize(enc_len * D);
        for (int t = 0; t < enc_len; t++) {
            linear_cpu(enc_output + t * enc_dim, cross_kvs[li].K.data() + t * D,
                       enc_dim, D, ck_w.data(), ck_b.data());
            linear_cpu(enc_output + t * enc_dim, cross_kvs[li].V.data() + t * D,
                       enc_dim, D, cv_w.data(), cv_b.data());
        }
    }

    // Self-attention KV cache
    ctx->kv_cache_k.resize(n_layers);
    ctx->kv_cache_v.resize(n_layers);
    for (int li = 0; li < n_layers; li++) {
        ctx->kv_cache_k[li].clear();
        ctx->kv_cache_v[li].clear();
    }

    // LM head weights
    auto lm_d_w = to_f32(ctx->lm_dense_w), lm_d_b = to_f32(ctx->lm_dense_b);
    auto lm_n_w = to_f32(ctx->lm_ln_w), lm_n_b = to_f32(ctx->lm_ln_b);
    auto lm_bias = to_f32(ctx->lm_bias);

    // Greedy decode
    std::vector<int> tokens;
    tokens.push_back(hp.sos_token);
    int max_len = hp.max_position;
    int vocab = hp.vocab_size;

    for (int step = 0; step < max_len; step++) {
        int token_id = tokens.back();
        int pos = step;

        // Token embedding: word + position + type + LN
        std::vector<float> hidden(D);
        for (int d = 0; d < D; d++) {
            hidden[d] = word_w[token_id * D + d]
                      + pos_w[pos * D + d]
                      + type_w[d]; // type_vocab_size=1, always type 0
        }
        std::vector<float> ln_out(D);
        layernorm_cpu(hidden.data(), ln_out.data(), D, eln_w.data(), eln_b.data(), 1e-12f);
        hidden = std::move(ln_out);

        // Decoder layers
        for (int li = 0; li < n_layers; li++) {
            auto& l = ctx->dec_layers[li];

            // Self-attention
            auto sq_w = to_f32(l.self_q_w), sq_b = to_f32(l.self_q_b);
            auto sk_w = to_f32(l.self_k_w), sk_b = to_f32(l.self_k_b);
            auto sv_w = to_f32(l.self_v_w), sv_b = to_f32(l.self_v_b);
            auto so_w = to_f32(l.self_out_w), so_b = to_f32(l.self_out_b);
            auto sln_w = to_f32(l.self_ln_w), sln_b = to_f32(l.self_ln_b);

            // Compute Q for current token, append K/V to cache
            std::vector<float> q(D), k(D), v(D);
            linear_cpu(hidden.data(), q.data(), D, D, sq_w.data(), sq_b.data());
            linear_cpu(hidden.data(), k.data(), D, D, sk_w.data(), sk_b.data());
            linear_cpu(hidden.data(), v.data(), D, D, sv_w.data(), sv_b.data());

            ctx->kv_cache_k[li].insert(ctx->kv_cache_k[li].end(), k.begin(), k.end());
            ctx->kv_cache_v[li].insert(ctx->kv_cache_v[li].end(), v.begin(), v.end());

            int n_kv = step + 1;
            std::vector<float> attn_out(D);
            mha_1q_cpu(q.data(), ctx->kv_cache_k[li].data(),
                       ctx->kv_cache_v[li].data(), attn_out.data(),
                       n_kv, D, hp.dec_heads);

            // Output projection + residual + LN
            std::vector<float> proj(D);
            linear_cpu(attn_out.data(), proj.data(), D, D, so_w.data(), so_b.data());
            for (int d = 0; d < D; d++) hidden[d] += proj[d];
            std::vector<float> ln(D);
            layernorm_cpu(hidden.data(), ln.data(), D, sln_w.data(), sln_b.data(), 1e-12f);
            hidden = std::move(ln);

            // Cross-attention
            auto cq_w = to_f32(l.cross_q_w), cq_b = to_f32(l.cross_q_b);
            auto co_w = to_f32(l.cross_out_w), co_b = to_f32(l.cross_out_b);
            auto cln_w = to_f32(l.cross_ln_w), cln_b = to_f32(l.cross_ln_b);

            std::vector<float> cq(D);
            linear_cpu(hidden.data(), cq.data(), D, D, cq_w.data(), cq_b.data());

            std::vector<float> cross_out(D);
            mha_1q_cpu(cq.data(), cross_kvs[li].K.data(),
                       cross_kvs[li].V.data(), cross_out.data(),
                       enc_len, D, hp.dec_heads);

            std::vector<float> cproj(D);
            linear_cpu(cross_out.data(), cproj.data(), D, D, co_w.data(), co_b.data());
            for (int d = 0; d < D; d++) hidden[d] += cproj[d];
            layernorm_cpu(hidden.data(), hidden.data(), D, cln_w.data(), cln_b.data(), 1e-12f);

            // FFN
            auto fu_w = to_f32(l.ffn_up_w), fu_b = to_f32(l.ffn_up_b);
            auto fd_w = to_f32(l.ffn_down_w), fd_b = to_f32(l.ffn_down_b);
            auto fln_w = to_f32(l.ffn_ln_w), fln_b = to_f32(l.ffn_ln_b);

            std::vector<float> up(hp.dec_ffn), down(D);
            linear_cpu(hidden.data(), up.data(), D, hp.dec_ffn, fu_w.data(), fu_b.data());
            for (int j = 0; j < hp.dec_ffn; j++) up[j] = gelu(up[j]);
            linear_cpu(up.data(), down.data(), hp.dec_ffn, D, fd_w.data(), fd_b.data());
            for (int d = 0; d < D; d++) hidden[d] += down[d];
            layernorm_cpu(hidden.data(), hidden.data(), D, fln_w.data(), fln_b.data(), 1e-12f);
        }

        // LM head: Dense → GELU → LN → project to vocab
        std::vector<float> lm_h(D);
        linear_cpu(hidden.data(), lm_h.data(), D, D, lm_d_w.data(), lm_d_b.data());
        for (int d = 0; d < D; d++) lm_h[d] = gelu(lm_h[d]);
        layernorm_cpu(lm_h.data(), lm_h.data(), D, lm_n_w.data(), lm_n_b.data(), 1e-12f);

        // Project to vocab (tied weights = word_embed_w transposed)
        std::vector<float> logits(vocab);
        for (int v = 0; v < vocab; v++) {
            float sum = lm_bias.empty() ? 0.0f : lm_bias[v];
            for (int d = 0; d < D; d++)
                sum += lm_h[d] * word_w[v * D + d];
            logits[v] = sum;
        }

        // Greedy: argmax
        int best = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[v] > logits[best]) best = v;

        if (best == hp.eos_token) break;
        tokens.push_back(best);
    }

    // Decode tokens to string
    std::string result;
    for (size_t i = 1; i < tokens.size(); i++) { // skip SOS
        int tid = tokens[i];
        if (tid >= 0 && tid < (int)ctx->vocab.size()) {
            result += ctx->vocab[tid];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------
static std::vector<float> preprocess_mixtex(const uint8_t* pixels, int w, int h, int ch,
                                             int target_h, int target_w) {
    // Resize to target_h × target_w, normalize with mean=0.5, std=0.5
    std::vector<float> out(3 * target_h * target_w);
    for (int c = 0; c < 3; c++) {
        for (int oy = 0; oy < target_h; oy++) {
            int iy = oy * h / target_h;
            for (int ox = 0; ox < target_w; ox++) {
                int ix = ox * w / target_w;
                float pixel;
                if (ch == 1) pixel = pixels[iy * w + ix] / 255.0f;
                else pixel = pixels[(iy * w + ix) * ch + c] / 255.0f;
                out[c * target_h * target_w + oy * target_w + ox] =
                    (pixel - 0.5f) / 0.5f;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
const char * mixtex_ocr_recognize(mixtex_ocr_context * ctx,
                                   const uint8_t * pixels,
                                   int width, int height, int channels,
                                   int * out_len) {
    if (!ctx || !pixels) { if (out_len) *out_len = 0; return nullptr; }

    auto& hp = ctx->hp;
    auto input = preprocess_mixtex(pixels, width, height, channels,
                                    hp.image_h, hp.image_w);

    if (ctx->dump) dump_stats("input", input.data(), 3 * hp.image_h * hp.image_w);

    auto t0 = std::chrono::steady_clock::now();
    auto enc_out = run_swin_encoder(ctx, input.data(), hp.image_h, hp.image_w);
    auto t1 = std::chrono::steady_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int enc_len = (int)enc_out.size() / hp.enc_hidden;
    fprintf(stderr, "mixtex_ocr: encoder %d tokens × %d dim (%.1f ms)\n",
            enc_len, hp.enc_hidden, enc_ms);

    auto t2 = std::chrono::steady_clock::now();
    ctx->output_text = run_decoder(ctx, enc_out.data(), enc_len, hp.enc_hidden);
    auto t3 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    fprintf(stderr, "mixtex_ocr: decoded \"%s\" (%.1f ms)\n",
            ctx->output_text.c_str(), dec_ms);

    if (out_len) *out_len = (int)ctx->output_text.size();
    return ctx->output_text.c_str();
}

const char * mixtex_ocr_recognize_gray(mixtex_ocr_context * ctx,
                                        const float * pixels,
                                        int width, int height,
                                        int * out_len) {
    if (!ctx || !pixels) { if (out_len) *out_len = 0; return nullptr; }

    // Convert gray float [0..1] to uint8 RGB
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        uint8_t v = (uint8_t)(pixels[i] * 255.0f);
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
    }
    return mixtex_ocr_recognize(ctx, rgb.data(), width, height, 3, out_len);
}
