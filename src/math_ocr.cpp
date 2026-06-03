// math_ocr.cpp — DeiT+TrOCR math OCR inference via ggml.
//
// VisionEncoderDecoderModel: DeiT encoder (12L) + TrOCR decoder (6L).
// Loads GGUF files produced by convert-pix2tex-to-gguf.py or
// convert-trocr-to-gguf.py.
//
// Uses core/gguf_loader.h for weight loading (same infra as the rest
// of CrispEmbed). Preprocessing uses 3-channel RGB input with
// mean=0.5, std=0.5 normalization (from DeiTImageProcessor config).

#include "math_ocr.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Layer structs
// ---------------------------------------------------------------------------

struct enc_layer {
    struct ggml_tensor *ln1_w, *ln1_b;        // layernorm_before
    struct ggml_tensor *q_w, *q_b;            // query
    struct ggml_tensor *k_w, *k_b;            // key
    struct ggml_tensor *v_w, *v_b;            // value
    struct ggml_tensor *attn_out_w, *attn_out_b; // attention output
    struct ggml_tensor *ln2_w, *ln2_b;        // layernorm_after
    struct ggml_tensor *ff_up_w, *ff_up_b;    // intermediate.dense
    struct ggml_tensor *ff_down_w, *ff_down_b; // output.dense
};

struct dec_layer {
    // Self-attention
    struct ggml_tensor *self_ln_w, *self_ln_b;
    struct ggml_tensor *self_q_w, *self_q_b;
    struct ggml_tensor *self_k_w, *self_k_b;
    struct ggml_tensor *self_v_w, *self_v_b;
    struct ggml_tensor *self_out_w, *self_out_b;
    // Cross-attention
    struct ggml_tensor *cross_ln_w, *cross_ln_b;
    struct ggml_tensor *cross_q_w, *cross_q_b;
    struct ggml_tensor *cross_k_w, *cross_k_b;
    struct ggml_tensor *cross_v_w, *cross_v_b;
    struct ggml_tensor *cross_out_w, *cross_out_b;
    // FFN
    struct ggml_tensor *ff_ln_w, *ff_ln_b;
    struct ggml_tensor *ff_up_w, *ff_up_b;
    struct ggml_tensor *ff_down_w, *ff_down_b;
};

struct math_ocr_context {
    math_ocr_hparams hparams;

    // Encoder
    struct ggml_tensor *cls_token, *dist_token;
    struct ggml_tensor *patch_proj_w, *patch_proj_b, *pos_embed;
    struct ggml_tensor *enc_ln_w, *enc_ln_b;
    std::vector<enc_layer> enc_layers;

    // Decoder
    struct ggml_tensor *tok_embed, *pos_embed_dec;
    struct ggml_tensor *dec_embed_ln_w, *dec_embed_ln_b;
    struct ggml_tensor *dec_final_ln_w, *dec_final_ln_b;
    struct ggml_tensor *lm_head_w, *lm_head_b;
    std::vector<dec_layer> dec_layers;

    // Tokenizer + state
    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    int n_threads;
    std::string result_buf;

    // Decoder KV cache + encoder output cache
    std::vector<std::vector<float>> kv_k, kv_v; // per layer
    std::vector<float> enc_out;
    int n_enc_tokens;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m,
                       const char* name) {
    auto it = m.find(name);
    if (it != m.end()) return it->second;
    // Dot→underscore fallback
    std::string alt(name);
    for (auto& c : alt) if (c == '.') c = '_';
    it = m.find(alt);
    return it != m.end() ? it->second : nullptr;
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b
static void layernorm(const float* x, float* y, int D,
                       const ggml_tensor* w, const ggml_tensor* b) {
    if (!w || !b) { memcpy(y, x, D * sizeof(float)); return; }
    const float* W = (const float*)w->data;
    const float* B = (const float*)b->data;
    float mean = 0, var = 0;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;
    for (int i = 0; i < D; i++) { float d = x[i] - mean; var += d * d; }
    float inv = 1.0f / sqrtf(var / D + 1e-6f);
    for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * W[i] + B[i];
}

// Linear: out = W @ inp + bias  (W is [out_dim, in_dim] row-major)
static void linear(const float* inp, float* out, int in_dim, int out_dim,
                    const ggml_tensor* w, const ggml_tensor* bias) {
    if (!w) { memset(out, 0, out_dim * sizeof(float)); return; }
    const float* W = (const float*)w->data;
    const float* B = bias ? (const float*)bias->data : nullptr;
    for (int o = 0; o < out_dim; o++) {
        float sum = B ? B[o] : 0.0f;
        for (int j = 0; j < in_dim; j++) sum += inp[j] * W[o * in_dim + j];
        out[o] = sum;
    }
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Multi-head attention: (n_q tokens, D) × (n_kv tokens, D) → (n_q, D)
static void mha(const float* Q, const float* K, const float* V,
                float* out, int n_q, int n_kv, int D, int n_heads,
                bool causal = false) {
    int head_dim = D / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int qi = 0; qi < n_q; qi++) {
        for (int h = 0; h < n_heads; h++) {
            std::vector<float> scores(n_kv);
            float max_s = -1e9f;
            for (int ki = 0; ki < n_kv; ki++) {
                if (causal && ki > qi) { scores[ki] = -1e9f; continue; }
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q[qi * D + h * head_dim + d] * K[ki * D + h * head_dim + d];
                scores[ki] = dot * scale;
                max_s = std::max(max_s, scores[ki]);
            }
            float sum_exp = 0;
            for (int ki = 0; ki < n_kv; ki++) {
                scores[ki] = expf(scores[ki] - max_s);
                sum_exp += scores[ki];
            }
            for (auto& s : scores) s /= sum_exp;
            for (int d = 0; d < head_dim; d++) {
                float sum = 0;
                for (int ki = 0; ki < n_kv; ki++)
                    sum += scores[ki] * V[ki * D + h * head_dim + d];
                out[qi * D + h * head_dim + d] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor mapping
// ---------------------------------------------------------------------------

static void map_tensors(math_ocr_context* ctx) {
    const auto& m = ctx->wl.tensors;
    const auto& hp = ctx->hparams;

    // Encoder embeddings
    ctx->cls_token    = T(m, "enc.embeddings.cls_token");
    ctx->dist_token   = T(m, "enc.embeddings.distillation_token");
    ctx->patch_proj_w = T(m, "enc.embeddings.patch_embeddings.projection.weight");
    ctx->patch_proj_b = T(m, "enc.embeddings.patch_embeddings.projection.bias");
    ctx->pos_embed    = T(m, "enc.embeddings.position_embeddings");
    ctx->enc_ln_w     = T(m, "enc.layernorm.weight");
    ctx->enc_ln_b     = T(m, "enc.layernorm.bias");

    // Encoder layers
    ctx->enc_layers.resize(hp.enc_layers);
    char buf[256];
    for (int i = 0; i < hp.enc_layers; i++) {
        auto& l = ctx->enc_layers[i];
        auto E = [&](const char* s) {
            snprintf(buf, sizeof(buf), "enc.encoder.layer.%d.%s", i, s);
            return T(m, buf);
        };
        l.ln1_w = E("layernorm_before.weight"); l.ln1_b = E("layernorm_before.bias");
        l.q_w = E("attention.attention.query.weight"); l.q_b = E("attention.attention.query.bias");
        l.k_w = E("attention.attention.key.weight"); l.k_b = E("attention.attention.key.bias");
        l.v_w = E("attention.attention.value.weight"); l.v_b = E("attention.attention.value.bias");
        l.attn_out_w = E("attention.output.dense.weight"); l.attn_out_b = E("attention.output.dense.bias");
        l.ln2_w = E("layernorm_after.weight"); l.ln2_b = E("layernorm_after.bias");
        l.ff_up_w = E("intermediate.dense.weight"); l.ff_up_b = E("intermediate.dense.bias");
        l.ff_down_w = E("output.dense.weight"); l.ff_down_b = E("output.dense.bias");
    }

    // Decoder embeddings — try shortened then full names
    auto D2 = [&](const char* short_n, const char* full_n) {
        auto* t = T(m, short_n);
        return t ? t : T(m, full_n);
    };
    ctx->tok_embed = D2("dec.d.embed_tokens.weight", "dec.decoder.model.decoder.embed_tokens.weight");
    ctx->pos_embed_dec = D2("dec.d.embed_positions.weight", "dec.decoder.model.decoder.embed_positions.weight");
    ctx->dec_embed_ln_w = D2("dec.d.layernorm_embedding.weight", "dec.decoder.model.decoder.layernorm_embedding.weight");
    ctx->dec_embed_ln_b = D2("dec.d.layernorm_embedding.bias", "dec.decoder.model.decoder.layernorm_embedding.bias");
    ctx->dec_final_ln_w = D2("dec.d.layer_norm.weight", "dec.decoder.model.decoder.layer_norm.weight");
    ctx->dec_final_ln_b = D2("dec.d.layer_norm.bias", "dec.decoder.model.decoder.layer_norm.bias");
    ctx->lm_head_w = T(m, "dec.lm_head.weight");
    ctx->lm_head_b = T(m, "dec.lm_head.bias");

    // Decoder layers
    ctx->dec_layers.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto& l = ctx->dec_layers[i];
        auto DL = [&](const char* s) {
            snprintf(buf, sizeof(buf), "dec.d.layers.%d.%s", i, s);
            auto* t = T(m, buf);
            if (t) return t;
            snprintf(buf, sizeof(buf), "dec.decoder.model.decoder.layers.%d.%s", i, s);
            return T(m, buf);
        };
        l.self_ln_w = DL("self_attn_layer_norm.weight"); l.self_ln_b = DL("self_attn_layer_norm.bias");
        l.self_q_w = DL("self_attn.q_proj.weight"); l.self_q_b = DL("self_attn.q_proj.bias");
        l.self_k_w = DL("self_attn.k_proj.weight"); l.self_k_b = DL("self_attn.k_proj.bias");
        l.self_v_w = DL("self_attn.v_proj.weight"); l.self_v_b = DL("self_attn.v_proj.bias");
        l.self_out_w = DL("self_attn.out_proj.weight"); l.self_out_b = DL("self_attn.out_proj.bias");
        auto* xaln_w = DL("xaln.weight");
        l.cross_ln_w = xaln_w ? xaln_w : DL("encoder_attn_layer_norm.weight");
        auto* xaln_b = DL("xaln.bias");
        l.cross_ln_b = xaln_b ? xaln_b : DL("encoder_attn_layer_norm.bias");
        l.cross_q_w = DL("encoder_attn.q_proj.weight"); l.cross_q_b = DL("encoder_attn.q_proj.bias");
        l.cross_k_w = DL("encoder_attn.k_proj.weight"); l.cross_k_b = DL("encoder_attn.k_proj.bias");
        l.cross_v_w = DL("encoder_attn.v_proj.weight"); l.cross_v_b = DL("encoder_attn.v_proj.bias");
        l.cross_out_w = DL("encoder_attn.out_proj.weight"); l.cross_out_b = DL("encoder_attn.out_proj.bias");
        l.ff_ln_w = DL("final_layer_norm.weight"); l.ff_ln_b = DL("final_layer_norm.bias");
        l.ff_up_w = DL("fc1.weight"); l.ff_up_b = DL("fc1.bias");
        l.ff_down_w = DL("fc2.weight"); l.ff_down_b = DL("fc2.bias");
    }
}

// ---------------------------------------------------------------------------
// Init / Free
// ---------------------------------------------------------------------------

math_ocr_context* math_ocr_init(const char* model_path, int n_threads) {
    auto ctx = std::make_unique<math_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    // Metadata pass
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) { fprintf(stderr, "math_ocr: can't open %s\n", model_path); return nullptr; }

    auto& hp = ctx->hparams;
    hp.enc_layers       = core_gguf::kv_u32(gctx, "encoder.num_hidden_layers", 12);
    hp.enc_heads        = core_gguf::kv_u32(gctx, "encoder.num_attention_heads", 6);
    hp.enc_hidden       = core_gguf::kv_u32(gctx, "encoder.hidden_size", 384);
    hp.enc_intermediate = core_gguf::kv_u32(gctx, "encoder.intermediate_size", 1536);
    hp.image_size       = core_gguf::kv_u32(gctx, "encoder.image_size", 384);
    hp.patch_size       = core_gguf::kv_u32(gctx, "encoder.patch_size", 16);
    hp.dec_layers       = core_gguf::kv_u32(gctx, "decoder.decoder_layers", 6);
    hp.dec_heads        = core_gguf::kv_u32(gctx, "decoder.decoder_attention_heads", 8);
    hp.dec_d_model      = core_gguf::kv_u32(gctx, "decoder.d_model", 256);
    hp.dec_ffn_dim      = core_gguf::kv_u32(gctx, "decoder.decoder_ffn_dim", 1024);
    hp.vocab_size       = core_gguf::kv_u32(gctx, "decoder.vocab_size", 1200);
    hp.max_seq_len      = core_gguf::kv_u32(gctx, "decoder.max_position_embeddings", 512);
    hp.cross_attn_dim   = core_gguf::kv_u32(gctx, "decoder.cross_attention_hidden_size", hp.enc_hidden);
    hp.bos_token        = core_gguf::kv_u32(gctx, "decoder.bos_token_id", 0);
    hp.eos_token        = core_gguf::kv_u32(gctx, "decoder.eos_token_id", 2);
    hp.pad_token        = core_gguf::kv_u32(gctx, "decoder.pad_token_id", 1);
    hp.decoder_start_token = core_gguf::kv_u32(gctx, "decoder.decoder_start_token_id", 2);
    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "math_ocr: enc=%dL/%dH/%d dec=%dL/%dH/%d vocab=%d(%zu)\n",
            hp.enc_layers, hp.enc_heads, hp.enc_hidden,
            hp.dec_layers, hp.dec_heads, hp.dec_d_model, hp.vocab_size, ctx->vocab.size());

    // Weight pass
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "math_ocr", ctx->wl)) {
        ggml_backend_free(backend);
        return nullptr;
    }

    map_tensors(ctx.get());

    int me = 0, md = 0;
    for (auto& l : ctx->enc_layers) if (l.q_w) me++;
    for (auto& l : ctx->dec_layers) if (l.self_q_w) md++;
    fprintf(stderr, "math_ocr: mapped %d/%d enc, %d/%d dec\n", me, hp.enc_layers, md, hp.dec_layers);

    ctx->kv_k.resize(hp.dec_layers);
    ctx->kv_v.resize(hp.dec_layers);
    return ctx.release();
}

void math_ocr_free(math_ocr_context* ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const math_ocr_hparams* math_ocr_get_hparams(const math_ocr_context* ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ---------------------------------------------------------------------------
// Encoder forward pass
// ---------------------------------------------------------------------------

static void run_encoder(math_ocr_context* ctx, const float* pixels_rgb,
                         int img_w, int img_h) {
    const auto& hp = ctx->hparams;
    const int P = hp.patch_size, H = hp.enc_hidden;
    const int npw = img_w / P, nph = img_h / P;
    const int n_patches = npw * nph;
    const int n_tokens = n_patches + 2; // CLS + distillation

    // Patch embedding: extract P×P patches from 3-channel image, project
    // patch_proj_w shape: (H, 3, P, P) = (H, 3*P*P) row-major
    const int patch_dim = 3 * P * P;
    std::vector<float> hidden(n_tokens * H, 0.0f);

    if (ctx->patch_proj_w) {
        const float* W = (const float*)ctx->patch_proj_w->data;
        const float* B = ctx->patch_proj_b ? (const float*)ctx->patch_proj_b->data : nullptr;
        for (int py = 0; py < nph; py++) {
            for (int px = 0; px < npw; px++) {
                float* out = hidden.data() + (py * npw + px + 2) * H; // +2 for CLS+dist
                for (int h = 0; h < H; h++) {
                    float sum = B ? B[h] : 0.0f;
                    // W[h, c, dy, dx] stored as W[h * 3*P*P + c*P*P + dy*P + dx]
                    for (int c = 0; c < 3; c++) {
                        for (int dy = 0; dy < P; dy++) {
                            for (int dx = 0; dx < P; dx++) {
                                int sy = py * P + dy, sx = px * P + dx;
                                float pixel = pixels_rgb[(c * img_h + sy) * img_w + sx]; // CHW layout
                                sum += pixel * W[h * patch_dim + c * P * P + dy * P + dx];
                            }
                        }
                    }
                    out[h] = sum;
                }
            }
        }
    }

    // CLS + distillation tokens
    if (ctx->cls_token) memcpy(hidden.data(), ctx->cls_token->data, H * sizeof(float));
    if (ctx->dist_token) memcpy(hidden.data() + H, ctx->dist_token->data, H * sizeof(float));

    // Positional embeddings
    if (ctx->pos_embed) {
        const float* pe = (const float*)ctx->pos_embed->data;
        int pe_size = std::min(n_tokens * H, (int)ggml_nelements(ctx->pos_embed));
        for (int i = 0; i < pe_size; i++) hidden[i] += pe[i];
    }

    // Transformer layers
    std::vector<float> normed(n_tokens * H);
    std::vector<float> Qbuf(n_tokens * H), Kbuf(n_tokens * H), Vbuf(n_tokens * H);
    std::vector<float> attn_out(n_tokens * H), proj_out(n_tokens * H);

    for (int li = 0; li < hp.enc_layers; li++) {
        const auto& l = ctx->enc_layers[li];
        if (!l.q_w) continue;

        // Pre-LN self-attention
        for (int t = 0; t < n_tokens; t++)
            layernorm(hidden.data() + t * H, normed.data() + t * H, H, l.ln1_w, l.ln1_b);

        for (int t = 0; t < n_tokens; t++) {
            linear(normed.data() + t * H, Qbuf.data() + t * H, H, H, l.q_w, l.q_b);
            linear(normed.data() + t * H, Kbuf.data() + t * H, H, H, l.k_w, l.k_b);
            linear(normed.data() + t * H, Vbuf.data() + t * H, H, H, l.v_w, l.v_b);
        }

        mha(Qbuf.data(), Kbuf.data(), Vbuf.data(), attn_out.data(),
            n_tokens, n_tokens, H, hp.enc_heads);

        for (int t = 0; t < n_tokens; t++)
            linear(attn_out.data() + t * H, proj_out.data() + t * H, H, H, l.attn_out_w, l.attn_out_b);

        for (int i = 0; i < n_tokens * H; i++) hidden[i] += proj_out[i];

        // Post-LN FFN
        for (int t = 0; t < n_tokens; t++)
            layernorm(hidden.data() + t * H, normed.data() + t * H, H, l.ln2_w, l.ln2_b);

        if (l.ff_up_w && l.ff_down_w) {
            const int I = hp.enc_intermediate;
            std::vector<float> inter(I);
            for (int t = 0; t < n_tokens; t++) {
                linear(normed.data() + t * H, inter.data(), H, I, l.ff_up_w, l.ff_up_b);
                for (int i = 0; i < I; i++) inter[i] = gelu(inter[i]);
                float tmp[1024]; // stack buffer for output dim <= H
                linear(inter.data(), tmp, I, H, l.ff_down_w, l.ff_down_b);
                for (int j = 0; j < H; j++) hidden[t * H + j] += tmp[j];
            }
        }
    }

    // Final encoder LN
    if (ctx->enc_ln_w)
        for (int t = 0; t < n_tokens; t++)
            layernorm(hidden.data() + t * H, hidden.data() + t * H, H, ctx->enc_ln_w, ctx->enc_ln_b);

    ctx->enc_out = std::move(hidden);
    ctx->n_enc_tokens = n_tokens;
}

// ---------------------------------------------------------------------------
// Decoder step + greedy decode
// ---------------------------------------------------------------------------

static std::vector<float> decoder_step(math_ocr_context* ctx,
                                         const std::vector<int>& tokens, int step) {
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model, E = hp.enc_hidden, V = hp.vocab_size;
    const int head_dim = D / hp.dec_heads;

    // Token embed + positional + LN
    std::vector<float> x(D, 0.0f);
    int tok = tokens.back();
    if (ctx->tok_embed && tok >= 0 && tok < V) {
        const float* emb = (const float*)ctx->tok_embed->data;
        float scale = sqrtf((float)D);
        for (int i = 0; i < D; i++) x[i] = emb[tok * D + i] * scale;
    }
    if (ctx->pos_embed_dec) {
        const float* pe = (const float*)ctx->pos_embed_dec->data;
        int pos = step + 2; // TrOCR offset
        if (pos < hp.max_seq_len)
            for (int i = 0; i < D; i++) x[i] += pe[pos * D + i];
    }
    std::vector<float> normed(D);
    layernorm(x.data(), x.data(), D, ctx->dec_embed_ln_w, ctx->dec_embed_ln_b);

    for (int li = 0; li < hp.dec_layers; li++) {
        const auto& l = ctx->dec_layers[li];
        if (!l.self_q_w) continue;

        // Self-attention with KV cache
        layernorm(x.data(), normed.data(), D, l.self_ln_w, l.self_ln_b);
        std::vector<float> q(D), k(D), v(D);
        linear(normed.data(), q.data(), D, D, l.self_q_w, l.self_q_b);
        linear(normed.data(), k.data(), D, D, l.self_k_w, l.self_k_b);
        linear(normed.data(), v.data(), D, D, l.self_v_w, l.self_v_b);

        ctx->kv_k[li].insert(ctx->kv_k[li].end(), k.begin(), k.end());
        ctx->kv_v[li].insert(ctx->kv_v[li].end(), v.begin(), v.end());
        int n_past = step + 1;

        // Single-query attention against cached KV
        std::vector<float> sa_out(D, 0.0f);
        mha(q.data(), ctx->kv_k[li].data(), ctx->kv_v[li].data(),
            sa_out.data(), 1, n_past, D, hp.dec_heads, true);

        std::vector<float> sa_proj(D);
        linear(sa_out.data(), sa_proj.data(), D, D, l.self_out_w, l.self_out_b);
        for (int i = 0; i < D; i++) x[i] += sa_proj[i];

        // Cross-attention
        if (l.cross_q_w && !ctx->enc_out.empty()) {
            layernorm(x.data(), normed.data(), D, l.cross_ln_w, l.cross_ln_b);
            std::vector<float> cq(D);
            linear(normed.data(), cq.data(), D, D, l.cross_q_w, l.cross_q_b);

            // Project encoder output → decoder dim for K/V
            int n_enc = ctx->n_enc_tokens;
            std::vector<float> ck(n_enc * D), cv(n_enc * D);
            for (int t = 0; t < n_enc; t++) {
                linear(ctx->enc_out.data() + t * E, ck.data() + t * D, E, D, l.cross_k_w, l.cross_k_b);
                linear(ctx->enc_out.data() + t * E, cv.data() + t * D, E, D, l.cross_v_w, l.cross_v_b);
            }

            std::vector<float> ca_out(D, 0.0f);
            mha(cq.data(), ck.data(), cv.data(), ca_out.data(), 1, n_enc, D, hp.dec_heads);

            std::vector<float> ca_proj(D);
            linear(ca_out.data(), ca_proj.data(), D, D, l.cross_out_w, l.cross_out_b);
            for (int i = 0; i < D; i++) x[i] += ca_proj[i];
        }

        // FFN
        if (l.ff_up_w && l.ff_down_w) {
            layernorm(x.data(), normed.data(), D, l.ff_ln_w, l.ff_ln_b);
            const int F = hp.dec_ffn_dim;
            std::vector<float> inter(F);
            linear(normed.data(), inter.data(), D, F, l.ff_up_w, l.ff_up_b);
            for (int i = 0; i < F; i++) inter[i] = inter[i] > 0 ? inter[i] : 0; // ReLU
            std::vector<float> ffn_out(D);
            linear(inter.data(), ffn_out.data(), F, D, l.ff_down_w, l.ff_down_b);
            for (int i = 0; i < D; i++) x[i] += ffn_out[i];
        }
    }

    // Final LN + lm_head
    layernorm(x.data(), x.data(), D, ctx->dec_final_ln_w, ctx->dec_final_ln_b);
    std::vector<float> logits(V, 0.0f);
    linear(x.data(), logits.data(), D, V, ctx->lm_head_w, ctx->lm_head_b);
    return logits;
}

static std::string greedy_decode(math_ocr_context* ctx) {
    const auto& hp = ctx->hparams;
    std::vector<int> tokens = {hp.decoder_start_token};
    for (auto& c : ctx->kv_k) c.clear();
    for (auto& c : ctx->kv_v) c.clear();

    for (int step = 0; step < hp.max_seq_len; step++) {
        auto logits = decoder_step(ctx, tokens, step);
        int best = 0; float best_s = logits[0];
        for (int v = 1; v < hp.vocab_size; v++)
            if (logits[v] > best_s) { best_s = logits[v]; best = v; }
        if (best == hp.eos_token || best == hp.pad_token) break;
        tokens.push_back(best);
    }

    std::string result;
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < (int)ctx->vocab.size()) result += ctx->vocab[tok];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const char* math_ocr_recognize(math_ocr_context* ctx, const float* pixels,
                                 int width, int height, int* out_len) {
    if (!ctx || !pixels) return nullptr;
    const int S = ctx->hparams.image_size;

    // Resize to S×S and expand gray→RGB in CHW format with (v-0.5)/0.5 normalization
    std::vector<float> rgb(3 * S * S);
    float sx = (float)width / S, sy = (float)height / S;
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            int ox = std::min((int)(x * sx), width - 1);
            int oy = std::min((int)(y * sy), height - 1);
            float v = (pixels[oy * width + ox] - 0.5f) / 0.5f; // normalize
            rgb[0 * S * S + y * S + x] = v; // R
            rgb[1 * S * S + y * S + x] = v; // G
            rgb[2 * S * S + y * S + x] = v; // B
        }
    }

    run_encoder(ctx, rgb.data(), S, S);
    ctx->result_buf = greedy_decode(ctx);
    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char* math_ocr_recognize_file(math_ocr_context*, const char*, int*) { return nullptr; }

const char* math_ocr_recognize_raw(math_ocr_context* ctx, const uint8_t* bytes,
                                     int w, int h, int ch, int* out_len) {
    if (!ctx || !bytes) return nullptr;
    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) {
        if (ch == 1) gray[i] = bytes[i] / 255.0f;
        else { int b = i * ch; gray[i] = (0.299f*bytes[b] + 0.587f*bytes[b+1] + 0.114f*bytes[b+2]) / 255.0f; }
    }
    return math_ocr_recognize(ctx, gray.data(), w, h, out_len);
}
