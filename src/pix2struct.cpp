// pix2struct.cpp -- Pix2Struct image-to-text (CPU-scalar).
//
// Encoder: patch_projection + row/col embeddings → 12 T5-style layers
//   (Pre-RMSNorm → QKVO self-attn → Pre-RMSNorm → SwiGLU FFN)
//   No relative attention bias in encoder; position from row/col embeddings.
//
// Decoder: token embed → 12 T5-style layers
//   (Pre-RMSNorm → causal self-attn + T5 relative bias →
//    Pre-RMSNorm → cross-attn → Pre-RMSNorm → SwiGLU FFN)
//   → final norm → LM head → greedy/beam decode.

#include "pix2struct.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Helpers ──

static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(t->data, buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static void rms_norm(const float * x, int n, const float * w, float eps, float * out) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// Linear: out[o] = sum_i(w[o*ic + i] * x[i])
static void linear(const float * x, int ic, const float * w, int oc, float * out) {
    for (int o = 0; o < oc; o++) {
        float s = 0;
        for (int i = 0; i < ic; i++) s += w[o * ic + i] * x[i];
        out[o] = s;
    }
}

// T5 relative position bias: maps distance → bucket → bias per head
static int t5_relative_bucket(int rel_pos, bool bidirectional, int n_buckets, int max_distance) {
    int bucket = 0;
    int n = -rel_pos;
    if (bidirectional) {
        n_buckets /= 2;
        bucket += (n < 0 ? n_buckets : 0);
        n = abs(n);
    } else {
        n = std::max(n, 0);
    }
    int max_exact = n_buckets / 2;
    if (n < max_exact) {
        bucket += n;
    } else {
        bucket += max_exact + (int)(logf((float)n / max_exact) / logf((float)max_distance / max_exact) * (n_buckets - max_exact));
        bucket = std::min(bucket, n_buckets - 1);
    }
    return bucket;
}

// ── Encoder/Decoder layer weights ──

struct enc_layer_wt {
    ggml_tensor * pre_attn_norm;
    ggml_tensor * q_w, * k_w, * v_w, * o_w;
    ggml_tensor * pre_mlp_norm;
    ggml_tensor * wi_0, * wi_1, * wo; // SwiGLU
};

struct dec_layer_wt {
    // Self-attention
    ggml_tensor * sa_norm;
    ggml_tensor * sa_q, * sa_k, * sa_v, * sa_o;
    ggml_tensor * sa_rel_bias; // only layer 0 (shared)
    // Cross-attention
    ggml_tensor * ca_norm;
    ggml_tensor * ca_q, * ca_k, * ca_v, * ca_o;
    // FFN
    ggml_tensor * ffn_norm;
    ggml_tensor * wi_0, * wi_1, * wo;
};

// ── Model context ──

struct pix2struct_context {
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;

    int enc_layers, dec_layers, hidden, n_heads, d_kv, d_ff;
    int vocab_size, patch_size, max_patches;
    int rel_buckets, rel_max_dist;
    float rms_eps;

    // Encoder
    ggml_tensor * patch_proj_w, * patch_proj_b;
    ggml_tensor * row_emb, * col_emb;
    std::vector<enc_layer_wt> enc;
    ggml_tensor * enc_final_norm;

    // Decoder
    ggml_tensor * tok_emb;
    std::vector<dec_layer_wt> dec;
    ggml_tensor * final_norm;
    ggml_tensor * lm_head;

    // Tokenizer
    int eos_id, pad_id;

    // Cached encoder output
    std::vector<float> enc_cache;
    int enc_cache_n;

    // Per-token confidence (softmax probability of greedy-selected token)
    std::vector<float> char_confidences;

    bool bench;
};

pix2struct_context * pix2struct_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;

    auto * ctx = new pix2struct_context;
    ctx->enc_layers = (int)core_gguf::kv_u32(meta, "pix2struct.enc_layers", 12);
    ctx->dec_layers = (int)core_gguf::kv_u32(meta, "pix2struct.dec_layers", 12);
    ctx->hidden = (int)core_gguf::kv_u32(meta, "pix2struct.hidden_size", 768);
    ctx->n_heads = (int)core_gguf::kv_u32(meta, "pix2struct.n_heads", 12);
    ctx->d_kv = (int)core_gguf::kv_u32(meta, "pix2struct.d_kv", 64);
    ctx->d_ff = (int)core_gguf::kv_u32(meta, "pix2struct.d_ff", 2048);
    ctx->vocab_size = (int)core_gguf::kv_u32(meta, "pix2struct.vocab_size", 50244);
    ctx->patch_size = (int)core_gguf::kv_u32(meta, "pix2struct.patch_size", 16);
    ctx->max_patches = (int)core_gguf::kv_u32(meta, "pix2struct.max_patches", 2048);
    ctx->rel_buckets = (int)core_gguf::kv_u32(meta, "pix2struct.rel_attn_buckets", 32);
    ctx->rel_max_dist = (int)core_gguf::kv_u32(meta, "pix2struct.rel_attn_max_dist", 128);
    ctx->eos_id = (int)core_gguf::kv_u32(meta, "tokenizer.eos_token_id", 1);
    ctx->pad_id = (int)core_gguf::kv_u32(meta, "tokenizer.pad_token_id", 0);
    ctx->rms_eps = 1e-6f;
    core_gguf::free_metadata(meta);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) { delete ctx; return nullptr; }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "pix2struct", wl)) {
        ggml_backend_free(backend); delete ctx; return nullptr;
    }
    ggml_backend_free(backend);
    ctx->gguf_ctx = wl.ctx;
    ctx->gguf_buf = wl.buf;

    auto g = [&](const char * name) { return core_gguf::try_get(wl.tensors, name); };

    ctx->patch_proj_w = g("enc_emb.patch_proj.weight");
    ctx->patch_proj_b = g("enc_emb.patch_proj.bias");
    ctx->row_emb = g("enc_emb.row_emb.weight");
    ctx->col_emb = g("enc_emb.col_emb.weight");

    ctx->enc.resize(ctx->enc_layers);
    for (int i = 0; i < ctx->enc_layers; i++) {
        char pfx[128];
        auto k = [&](const char * s) { snprintf(pfx, sizeof(pfx), "enc.%d.%s", i, s); return g(pfx); };
        ctx->enc[i].pre_attn_norm = k("pre_attn_ln.weight");
        ctx->enc[i].q_w = k("attention.query.weight");
        ctx->enc[i].k_w = k("attention.key.weight");
        ctx->enc[i].v_w = k("attention.value.weight");
        ctx->enc[i].o_w = k("attention.output.weight");
        ctx->enc[i].pre_mlp_norm = k("pre_mlp_ln.weight");
        ctx->enc[i].wi_0 = k("mlp.wi_0.weight");
        ctx->enc[i].wi_1 = k("mlp.wi_1.weight");
        ctx->enc[i].wo = k("mlp.wo.weight");
    }

    ctx->enc_final_norm = g("encoder.layernorm.weight");
    ctx->tok_emb = g("dec_emb.weight");
    ctx->final_norm = g("dec_final_ln.weight");
    ctx->lm_head = g("lm_head.weight");

    ctx->dec.resize(ctx->dec_layers);
    for (int i = 0; i < ctx->dec_layers; i++) {
        char pfx[128];
        auto k = [&](const char * s) { snprintf(pfx, sizeof(pfx), "dec.%d.%s", i, s); return g(pfx); };
        ctx->dec[i].sa_norm = k("sa_ln.weight");
        ctx->dec[i].sa_q = k("sattn.query.weight");
        ctx->dec[i].sa_k = k("sattn.key.weight");
        ctx->dec[i].sa_v = k("sattn.value.weight");
        ctx->dec[i].sa_o = k("sattn.output.weight");
        ctx->dec[i].sa_rel_bias = k("sattn.rel_bias.weight");
        ctx->dec[i].ca_norm = k("xa_ln.weight");
        ctx->dec[i].ca_q = k("xattn.query.weight");
        ctx->dec[i].ca_k = k("xattn.key.weight");
        ctx->dec[i].ca_v = k("xattn.value.weight");
        ctx->dec[i].ca_o = k("xattn.output.weight");
        ctx->dec[i].ffn_norm = k("ffn_ln.weight");
        ctx->dec[i].wi_0 = k("mlp.dense.wi_0.weight");
        ctx->dec[i].wi_1 = k("mlp.dense.wi_1.weight");
        ctx->dec[i].wo = k("mlp.dense.wo.weight");
    }

    ctx->enc_cache_n = 0;
    ctx->bench = (std::getenv("CRISPEMBED_PIX2STRUCT_BENCH") != nullptr);
    return ctx;
}

void pix2struct_free(pix2struct_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx; wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    delete ctx;
}

// ── Self-attention (multi-head, no causal mask for encoder) ──

static void self_attn(const float * x, int T, int H, int n_heads, int d_kv,
                       const float * q_w, const float * k_w,
                       const float * v_w, const float * o_w,
                       const float * rel_bias, int n_buckets, int max_dist,
                       bool bidirectional, bool causal,
                       float * out, std::vector<float> & scratch) {
    int head_dim = d_kv;
    int qkv_dim = n_heads * head_dim;

    // Q, K, V projections
    std::vector<float> Q(T * qkv_dim), K(T * qkv_dim), V(T * qkv_dim);
    for (int t = 0; t < T; t++) {
        linear(x + t * H, H, q_w, qkv_dim, Q.data() + t * qkv_dim);
        linear(x + t * H, H, k_w, qkv_dim, K.data() + t * qkv_dim);
        linear(x + t * H, H, v_w, qkv_dim, V.data() + t * qkv_dim);
    }

    // Compute attention per head
    std::vector<float> attn_out(T * qkv_dim, 0.0f);
    float scale = 1.0f; // T5 doesn't scale by sqrt(d_k) — it uses the raw dot product

    for (int h = 0; h < n_heads; h++) {
        // Attention scores
        scratch.resize(T * T);
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q[i * qkv_dim + h * head_dim + d] *
                           K[j * qkv_dim + h * head_dim + d];

                // Add relative bias
                if (rel_bias) {
                    int bucket = t5_relative_bucket(i - j, bidirectional, n_buckets, max_dist);
                    dot += rel_bias[bucket * n_heads + h];
                }

                // Causal mask
                if (causal && j > i) dot = -1e30f;

                scratch[i * T + j] = dot;
            }
        }

        // Softmax
        for (int i = 0; i < T; i++) {
            float mx = -1e30f;
            for (int j = 0; j < T; j++) mx = std::max(mx, scratch[i * T + j]);
            float sum = 0;
            for (int j = 0; j < T; j++) {
                scratch[i * T + j] = expf(scratch[i * T + j] - mx);
                sum += scratch[i * T + j];
            }
            for (int j = 0; j < T; j++) scratch[i * T + j] /= sum;
        }

        // attn @ V
        for (int i = 0; i < T; i++)
            for (int d = 0; d < head_dim; d++) {
                float s = 0;
                for (int j = 0; j < T; j++)
                    s += scratch[i * T + j] * V[j * qkv_dim + h * head_dim + d];
                attn_out[i * qkv_dim + h * head_dim + d] = s;
            }
    }

    // Output projection
    for (int t = 0; t < T; t++)
        linear(attn_out.data() + t * qkv_dim, qkv_dim, o_w, H, out + t * H);
}

// ── Cross-attention ──

static void cross_attn(const float * x, int T_q, int H,
                        const float * enc, int T_kv,
                        int n_heads, int d_kv,
                        const float * q_w, const float * k_w,
                        const float * v_w, const float * o_w,
                        float * out, std::vector<float> & scratch) {
    int head_dim = d_kv;
    int qkv_dim = n_heads * head_dim;

    std::vector<float> Q(T_q * qkv_dim), K(T_kv * qkv_dim), V(T_kv * qkv_dim);
    for (int t = 0; t < T_q; t++)
        linear(x + t * H, H, q_w, qkv_dim, Q.data() + t * qkv_dim);
    for (int t = 0; t < T_kv; t++) {
        linear(enc + t * H, H, k_w, qkv_dim, K.data() + t * qkv_dim);
        linear(enc + t * H, H, v_w, qkv_dim, V.data() + t * qkv_dim);
    }

    std::vector<float> attn_out(T_q * qkv_dim, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        scratch.resize(T_q * T_kv);
        for (int i = 0; i < T_q; i++)
            for (int j = 0; j < T_kv; j++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q[i * qkv_dim + h * head_dim + d] *
                           K[j * qkv_dim + h * head_dim + d];
                scratch[i * T_kv + j] = dot;
            }
        for (int i = 0; i < T_q; i++) {
            float mx = -1e30f;
            for (int j = 0; j < T_kv; j++) mx = std::max(mx, scratch[i * T_kv + j]);
            float sum = 0;
            for (int j = 0; j < T_kv; j++) {
                scratch[i * T_kv + j] = expf(scratch[i * T_kv + j] - mx);
                sum += scratch[i * T_kv + j];
            }
            for (int j = 0; j < T_kv; j++) scratch[i * T_kv + j] /= sum;
        }
        for (int i = 0; i < T_q; i++)
            for (int d = 0; d < head_dim; d++) {
                float s = 0;
                for (int j = 0; j < T_kv; j++)
                    s += scratch[i * T_kv + j] * V[j * qkv_dim + h * head_dim + d];
                attn_out[i * qkv_dim + h * head_dim + d] = s;
            }
    }
    for (int t = 0; t < T_q; t++)
        linear(attn_out.data() + t * qkv_dim, qkv_dim, o_w, H, out + t * H);
}

// ── SwiGLU FFN ──

static void swiglu_ffn(const float * x, int H, int d_ff,
                        const float * wi_0, const float * wi_1, const float * wo,
                        float * out) {
    std::vector<float> gate(d_ff), up(d_ff);
    linear(x, H, wi_0, d_ff, gate.data());
    linear(x, H, wi_1, d_ff, up.data());
    // GELU_new(gate) * up
    std::vector<float> hidden(d_ff);
    for (int i = 0; i < d_ff; i++) {
        // gelu_new = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = gate[i];
        float g = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        hidden[i] = g * up[i];
    }
    linear(hidden.data(), d_ff, wo, H, out);
}

// ── Encoder forward ──

const float * pix2struct_encode_patches(pix2struct_context * ctx,
                                         const float * patches, int n_patches,
                                         int * out_dim) {
    if (!ctx || !patches || n_patches <= 0) return nullptr;

    const int H = ctx->hidden;
    const int patch_dim = ctx->patch_size * ctx->patch_size * 3; // 768
    std::vector<float> dq1, dq2, scratch;

    // Embed patches: projection(pixel_values) + row_emb(row_id) + col_emb(col_id)
    std::vector<float> x(n_patches * H);
    const float * proj_w = to_f32(ctx->patch_proj_w, dq1);
    const float * proj_b = to_f32(ctx->patch_proj_b, dq2);
    const float * row_w = to_f32(ctx->row_emb, dq1);
    const float * col_w = to_f32(ctx->col_emb, dq2);

    for (int p = 0; p < n_patches; p++) {
        int row_id = (int)patches[p * (patch_dim + 2) + 0];
        int col_id = (int)patches[p * (patch_dim + 2) + 1];
        const float * pixels = &patches[p * (patch_dim + 2) + 2];

        // Linear projection of pixel values
        linear(pixels, patch_dim, proj_w, H, &x[p * H]);
        // Add bias
        if (proj_b) for (int i = 0; i < H; i++) x[p * H + i] += proj_b[i];
        // Add row/col embeddings (clamp to embedding table size)
        row_id = std::max(0, std::min(row_id, 4095));
        col_id = std::max(0, std::min(col_id, 4095));
        for (int i = 0; i < H; i++) {
            x[p * H + i] += row_w[row_id * H + i];
            x[p * H + i] += col_w[col_id * H + i];
        }
    }


    // Encoder layers
    std::vector<float> normed(n_patches * H), attn_out(n_patches * H), ffn_out(n_patches * H);
    for (int li = 0; li < ctx->enc_layers; li++) {
        const auto & L = ctx->enc[li];

        // Pre-attention RMSNorm
        for (int t = 0; t < n_patches; t++)
            rms_norm(&x[t * H], H, to_f32(L.pre_attn_norm, dq1), ctx->rms_eps, &normed[t * H]);

        // Self-attention (no relative bias, no causal)
        self_attn(normed.data(), n_patches, H, ctx->n_heads, ctx->d_kv,
                  to_f32(L.q_w, dq1), to_f32(L.k_w, dq1),
                  to_f32(L.v_w, dq1), to_f32(L.o_w, dq1),
                  nullptr, 0, 0, true, false,
                  attn_out.data(), scratch);

        for (int i = 0; i < n_patches * H; i++) x[i] += attn_out[i];


        // Pre-MLP RMSNorm
        for (int t = 0; t < n_patches; t++)
            rms_norm(&x[t * H], H, to_f32(L.pre_mlp_norm, dq1), ctx->rms_eps, &normed[t * H]);

        // SwiGLU FFN
        for (int t = 0; t < n_patches; t++)
            swiglu_ffn(&normed[t * H], H, ctx->d_ff,
                       to_f32(L.wi_0, dq1), to_f32(L.wi_1, dq1), to_f32(L.wo, dq1),
                       &ffn_out[t * H]);

        for (int i = 0; i < n_patches * H; i++) x[i] += ffn_out[i];
    }

    // Final encoder RMSNorm
    if (ctx->enc_final_norm) {
        const float * fn_w = to_f32(ctx->enc_final_norm, dq1);
        for (int t = 0; t < n_patches; t++)
            rms_norm(&x[t * H], H, fn_w, ctx->rms_eps, &x[t * H]);
    }

    // Cache encoder output
    ctx->enc_cache = std::move(x);
    ctx->enc_cache_n = n_patches;

    if (out_dim) *out_dim = H;
    return ctx->enc_cache.data();
}

// ── Decoder step (single token) ──

static void decoder_step(pix2struct_context * ctx,
                          const std::vector<std::vector<float>> & past_tokens,
                          int step, float * logits,
                          std::vector<float> & dq1, std::vector<float> & scratch) {
    const int H = ctx->hidden;
    const int T = step + 1; // number of tokens so far

    // Token embedding for current token
    const float * emb_w = to_f32(ctx->tok_emb, dq1);
    int tok_id = (step == 0) ? ctx->pad_id : 0; // decoder_start_token
    // Actually use the last generated token
    // past_tokens[step] has the embedding

    // Build full sequence of embeddings
    std::vector<float> x(T * H);
    for (int t = 0; t < T; t++)
        memcpy(&x[t * H], past_tokens[t].data(), H * sizeof(float));

    std::vector<float> normed(T * H), attn_out(T * H), ffn_out(T * H);

    // Get shared relative bias from layer 0
    const float * rel_bias_w = to_f32(ctx->dec[0].sa_rel_bias, dq1);

    for (int li = 0; li < ctx->dec_layers; li++) {
        const auto & L = ctx->dec[li];

        // Self-attention with causal mask + relative bias (shared from layer 0)
        for (int t = 0; t < T; t++)
            rms_norm(&x[t * H], H, to_f32(L.sa_norm, dq1), ctx->rms_eps, &normed[t * H]);

        self_attn(normed.data(), T, H, ctx->n_heads, ctx->d_kv,
                  to_f32(L.sa_q, dq1), to_f32(L.sa_k, dq1),
                  to_f32(L.sa_v, dq1), to_f32(L.sa_o, dq1),
                  rel_bias_w, ctx->rel_buckets, ctx->rel_max_dist,
                  false, true,
                  attn_out.data(), scratch);
        for (int i = 0; i < T * H; i++) x[i] += attn_out[i];

        // Cross-attention to encoder output
        for (int t = 0; t < T; t++)
            rms_norm(&x[t * H], H, to_f32(L.ca_norm, dq1), ctx->rms_eps, &normed[t * H]);

        cross_attn(normed.data(), T, H,
                   ctx->enc_cache.data(), ctx->enc_cache_n,
                   ctx->n_heads, ctx->d_kv,
                   to_f32(L.ca_q, dq1), to_f32(L.ca_k, dq1),
                   to_f32(L.ca_v, dq1), to_f32(L.ca_o, dq1),
                   attn_out.data(), scratch);
        for (int i = 0; i < T * H; i++) x[i] += attn_out[i];

        // FFN
        for (int t = 0; t < T; t++)
            rms_norm(&x[t * H], H, to_f32(L.ffn_norm, dq1), ctx->rms_eps, &normed[t * H]);
        for (int t = 0; t < T; t++)
            swiglu_ffn(&normed[t * H], H, ctx->d_ff,
                       to_f32(L.wi_0, dq1), to_f32(L.wi_1, dq1), to_f32(L.wo, dq1),
                       &ffn_out[t * H]);
        for (int i = 0; i < T * H; i++) x[i] += ffn_out[i];
    }

    // Final norm + LM head on last token
    std::vector<float> final_h(H);
    rms_norm(&x[(T - 1) * H], H, to_f32(ctx->final_norm, dq1), ctx->rms_eps, final_h.data());
    linear(final_h.data(), H, to_f32(ctx->lm_head, dq1), ctx->vocab_size, logits);
}

// ── Greedy decode from pre-encoded patches ──

static std::string greedy_decode(pix2struct_context * ctx, int max_tokens) {
    if (!ctx || ctx->enc_cache_n <= 0) return "";
    if (max_tokens <= 0) max_tokens = 256;

    const int H = ctx->hidden;
    std::vector<float> dq1, scratch;
    const float * emb_w = to_f32(ctx->tok_emb, dq1);

    // Start with decoder_start_token_id = 0
    std::vector<int32_t> generated = {0};
    std::vector<std::vector<float>> token_embs;

    // Embed start token
    std::vector<float> emb(H);
    for (int i = 0; i < H; i++) emb[i] = emb_w[0 * H + i]; // token 0
    token_embs.push_back(emb);

    std::vector<float> logits(ctx->vocab_size);
    ctx->char_confidences.clear();

    for (int step = 0; step < max_tokens; step++) {
        decoder_step(ctx, token_embs, step, logits.data(), dq1, scratch);

        // Argmax
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < ctx->vocab_size; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }

        if (best == ctx->eos_id) break;
        generated.push_back(best);
        { float se = 0; for (int i = 0; i < ctx->vocab_size; i++) se += expf(logits[i] - best_val); ctx->char_confidences.push_back(1.0f / se); }

        // Embed new token
        std::vector<float> new_emb(H);
        for (int i = 0; i < H; i++) new_emb[i] = emb_w[best * H + i];
        token_embs.push_back(new_emb);
    }

    // TODO: decode token ids to text using tokenizer
    // For now, return token ids as comma-separated string
    std::string result;
    for (size_t i = 1; i < generated.size(); i++) {
        if (i > 1) result += ",";
        result += std::to_string(generated[i]);
    }
    return result;
}

// ── Public decode API for parity testing ──

int pix2struct_decode_step0(pix2struct_context * ctx, float * out_logits) {
    if (!ctx || ctx->enc_cache_n <= 0 || !out_logits) return -1;

    const int H = ctx->hidden;
    std::vector<float> dq1, scratch;
    const float * emb_w = to_f32(ctx->tok_emb, dq1);

    // Single token: decoder_start_token_id = 0
    std::vector<std::vector<float>> token_embs;
    std::vector<float> emb(H);
    for (int i = 0; i < H; i++) emb[i] = emb_w[0 * H + i];
    token_embs.push_back(emb);

    decoder_step(ctx, token_embs, 0, out_logits, dq1, scratch);
    return 0;
}

// ── Image preprocessing: variable-resolution patching ──
// Mirrors HF Pix2StructImageProcessor.extract_flattened_patches

static std::vector<float> image_to_patches(const uint8_t * rgb, int W, int H,
                                            int max_patches, int patch_size,
                                            int * out_n_patches) {
    const int pH = patch_size, pW = patch_size, C = 3;
    float scale = sqrtf((float)max_patches * ((float)pH / H) * ((float)pW / W));
    int n_rows = std::max(1, std::min((int)floorf(scale * H / pH), max_patches));
    int n_cols = std::max(1, std::min((int)floorf(scale * W / pW), max_patches));
    while (n_rows * n_cols > max_patches) {
        if (n_rows > n_cols) n_rows--; else n_cols--;
    }
    int rH = n_rows * pH, rW = n_cols * pW;

    std::vector<float> resized(C * rH * rW);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < rH; y++)
            for (int x = 0; x < rW; x++) {
                float sy = ((float)y + 0.5f) * H / rH - 0.5f;
                float sx = ((float)x + 0.5f) * W / rW - 0.5f;
                sy = std::max(0.0f, std::min(sy, (float)(H - 1)));
                sx = std::max(0.0f, std::min(sx, (float)(W - 1)));
                int y0 = (int)sy, x0 = (int)sx;
                int y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
                float fy = sy - y0, fx = sx - x0;
                float v = (1-fy)*((1-fx)*(float)rgb[(y0*W+x0)*C+c] + fx*(float)rgb[(y0*W+x1)*C+c])
                        + fy*((1-fx)*(float)rgb[(y1*W+x0)*C+c] + fx*(float)rgb[(y1*W+x1)*C+c]);
                resized[c * rH * rW + y * rW + x] = v / 255.0f;
            }

    int total = C * rH * rW;
    float mean = 0;
    for (int i = 0; i < total; i++) mean += resized[i];
    mean /= total;
    float var = 0;
    for (int i = 0; i < total; i++) { float d = resized[i] - mean; var += d * d; }
    float adj_std = std::max(sqrtf(var / total), 1.0f / sqrtf((float)total));
    for (int i = 0; i < total; i++) resized[i] = (resized[i] - mean) / adj_std;

    int n_patches = n_rows * n_cols;
    int patch_dim = pH * pW * C;
    int feat_dim = patch_dim + 2;
    std::vector<float> patches(max_patches * feat_dim, 0.0f);
    for (int r = 0; r < n_rows; r++)
        for (int col = 0; col < n_cols; col++) {
            int pi = r * n_cols + col;
            patches[pi * feat_dim + 0] = (float)(r + 1);
            patches[pi * feat_dim + 1] = (float)(col + 1);
            for (int c = 0; c < C; c++)
                for (int py = 0; py < pH; py++)
                    for (int px = 0; px < pW; px++)
                        patches[pi * feat_dim + 2 + c * pH * pW + py * pW + px] =
                            resized[c * rH * rW + (r * pH + py) * rW + (col * pW + px)];
        }
    if (out_n_patches) *out_n_patches = n_patches;
    return patches;
}

// ── Generate ──

const char * pix2struct_generate(pix2struct_context * ctx,
                                 const uint8_t * image, int width, int height,
                                 int max_tokens) {
    if (!ctx || !image || width <= 0 || height <= 0) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    int n_patches = 0;
    auto patches = image_to_patches(image, width, height,
                                     ctx->max_patches, ctx->patch_size, &n_patches);
    if (n_patches <= 0) return nullptr;
    if (bench) fprintf(stderr, "[pix2struct-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    int out_dim = 0;
    pix2struct_encode_patches(ctx, patches.data(), n_patches, &out_dim);
    if (bench) fprintf(stderr, "[pix2struct-bench] encoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    static std::string result;
    result = greedy_decode(ctx, max_tokens > 0 ? max_tokens : 256);
    if (bench) fprintf(stderr, "[pix2struct-bench] decoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[pix2struct-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    return result.c_str();
}

void pix2struct_free_text(const char * text) {
    (void)text;
}

const float * pix2struct_confidences(const pix2struct_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) { if (n_tokens) *n_tokens = 0; return nullptr; }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float pix2struct_mean_confidence(const pix2struct_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double s = 0; for (float v : ctx->char_confidences) s += v;
    return (float)(s / ctx->char_confidences.size());
}
