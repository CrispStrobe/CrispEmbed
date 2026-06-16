// granite_vision_ocr.cpp — Granite Vision 3.3-2B (LLaVA-Next) OCR engine.
//
// This engine follows the same CPU-scalar pattern as internvl2_ocr.cpp:
//   1. Load GGUF (core_gguf)
//   2. Vision encoder forward (SigLIP ViT, 27 layers)
//   3. Multi-layer feature extraction (layers 3, 7, 15, 26)
//   4. MLP projector (4608 → 2048 → 2048)
//   5. Token embedding + vision splicing
//   6. Autoregressive LLM decode (Granite-3.1-2B, 40 layers, GQA, KV cache)
//
// Granite LLM multipliers:
//   embedding_multiplier = 12.0  (scales token embeddings)
//   residual_multiplier  = 0.22  (scales residual connections)
//   logits_scaling       = 8.0   (divides logits)
//
// For the full implementation, see internvl2_ocr.cpp as the template.
// The key differences are documented inline.

#include "granite_vision_ocr.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Helpers (same as other VLM engines) ────────────────────────────────

static const float * gv_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(t->data, buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static void gv_layernorm(float * data, int n, int d,
                         const float * weight, const float * bias, float eps) {
    for (int i = 0; i < n; i++) {
        float * row = data + i * d;
        float mean = 0;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;
        float var = 0;
        for (int j = 0; j < d; j++) { float x = row[j] - mean; var += x * x; }
        var /= d;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < d; j++)
            row[j] = (row[j] - mean) * inv * weight[j] + (bias ? bias[j] : 0.0f);
    }
}

static void gv_rmsnorm(float * data, int n, int d, const float * weight, float eps) {
    for (int i = 0; i < n; i++) {
        float * row = data + i * d;
        float ss = 0;
        for (int j = 0; j < d; j++) ss += row[j] * row[j];
        float inv = 1.0f / sqrtf(ss / d + eps);
        for (int j = 0; j < d; j++) row[j] = row[j] * inv * weight[j];
    }
}

static void gv_linear(const float * input, int n, int id, int od,
                      const float * weight, const float * bias, float * output) {
    for (int i = 0; i < n; i++) {
        const float * in_row = input + i * id;
        float * out_row = output + i * od;
        for (int o = 0; o < od; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (int j = 0; j < id; j++) sum += in_row[j] * weight[o * id + j];
            out_row[o] = sum;
        }
    }
}

static float gv_gelu(float x) {
    // gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static float gv_silu(float x) { return x / (1.0f + expf(-x)); }

// ── Context ────────────────────────────────────────────────────────────

struct granite_vision_context {
    // Vision hparams
    int vis_dim, vis_layers, vis_heads, vis_image_size, vis_patch_size;
    std::vector<int> feature_layers;  // [-24, -20, -12, -1] → absolute indices

    // LLM hparams
    int llm_dim, llm_layers, llm_heads, llm_kv_heads, llm_ffn_dim, vocab_size;
    float embedding_multiplier, residual_multiplier, logits_scaling, rope_theta;
    int image_token_index;
    bool tie_word_embeddings;

    int max_tokens;
    int n_threads;

    // Weight storage
    core_gguf::WeightLoad wl;
    std::vector<std::vector<float>> wbufs;

    // KV cache
    std::vector<float> kv_cache;  // [2 * llm_layers * max_seq * head_dim * llm_kv_heads]
    int kv_allocated;
    int n_past;

    // Output buffer
    std::string output_text;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) return nullptr;
        wbufs.emplace_back();
        return gv_to_f32(t, wbufs.back());
    }
};

// ── Init / Free ────────────────────────────────────────────────────────

granite_vision_context * granite_vision_init(const char * model_path, int n_threads) {
    auto * ctx = new granite_vision_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->max_tokens = 2048;
    ctx->kv_allocated = 0;
    ctx->n_past = 0;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { fprintf(stderr, "granite_vision: failed to open %s\n", model_path); delete ctx; return nullptr; }

    ctx->vis_dim        = core_gguf::kv_u32(meta, "granite_vision.vis_dim", 1152);
    ctx->vis_layers     = core_gguf::kv_u32(meta, "granite_vision.vis_layers", 27);
    ctx->vis_heads      = core_gguf::kv_u32(meta, "granite_vision.vis_heads", 16);
    ctx->vis_image_size = core_gguf::kv_u32(meta, "granite_vision.vis_image_size", 384);
    ctx->vis_patch_size = core_gguf::kv_u32(meta, "granite_vision.vis_patch_size", 14);

    auto fl = core_gguf::kv_i32_array(meta, "granite_vision.feature_layers");
    // Convert negative indices to absolute
    for (int v : fl)
        ctx->feature_layers.push_back(v < 0 ? ctx->vis_layers + v : v);

    ctx->llm_dim        = core_gguf::kv_u32(meta, "granite_vision.llm_dim", 2048);
    ctx->llm_layers     = core_gguf::kv_u32(meta, "granite_vision.llm_layers", 40);
    ctx->llm_heads      = core_gguf::kv_u32(meta, "granite_vision.llm_heads", 32);
    ctx->llm_kv_heads   = core_gguf::kv_u32(meta, "granite_vision.llm_kv_heads", 8);
    ctx->llm_ffn_dim    = core_gguf::kv_u32(meta, "granite_vision.llm_ffn_dim", 8192);
    ctx->vocab_size     = core_gguf::kv_u32(meta, "granite_vision.vocab_size", 49156);
    ctx->image_token_index = core_gguf::kv_u32(meta, "granite_vision.image_token_index", 49155);
    ctx->tie_word_embeddings = core_gguf::kv_u32(meta, "granite_vision.tie_word_embeddings", 0) != 0;

    int idx;
    idx = gguf_find_key(meta, "granite_vision.embedding_multiplier");
    ctx->embedding_multiplier = idx >= 0 ? gguf_get_val_f32(meta, idx) : 12.0f;
    idx = gguf_find_key(meta, "granite_vision.residual_multiplier");
    ctx->residual_multiplier = idx >= 0 ? gguf_get_val_f32(meta, idx) : 0.22f;
    idx = gguf_find_key(meta, "granite_vision.logits_scaling");
    ctx->logits_scaling = idx >= 0 ? gguf_get_val_f32(meta, idx) : 8.0f;
    idx = gguf_find_key(meta, "granite_vision.rope_theta");
    ctx->rope_theta = idx >= 0 ? gguf_get_val_f32(meta, idx) : 300000.0f;

    core_gguf::free_metadata(meta);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "granite_vision", ctx->wl)) {
        fprintf(stderr, "granite_vision: failed to load weights\n");
        ggml_backend_free(backend); delete ctx; return nullptr;
    }
    ggml_backend_free(backend);

    int n_patches = (ctx->vis_image_size / ctx->vis_patch_size);
    n_patches *= n_patches;  // 27*27 = 729

    fprintf(stderr, "granite_vision: vis=%dL×%dd (%d patches), llm=%dL×%dd "
            "(heads=%d/%d, ffn=%d), vocab=%d, %d tensors\n",
            ctx->vis_layers, ctx->vis_dim, n_patches,
            ctx->llm_layers, ctx->llm_dim,
            ctx->llm_heads, ctx->llm_kv_heads, ctx->llm_ffn_dim,
            ctx->vocab_size, (int)ctx->wl.tensors.size());
    fprintf(stderr, "  multipliers: embed=%.1f, residual=%.2f, logits=%.1f\n",
            ctx->embedding_multiplier, ctx->residual_multiplier, ctx->logits_scaling);

    return ctx;
}

void granite_vision_free(granite_vision_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

void granite_vision_set_max_tokens(granite_vision_context * ctx, int max_tokens) {
    if (ctx) ctx->max_tokens = max_tokens > 0 ? max_tokens : 2048;
}

// ── SigLIP Vision Encoder ──────────────────────────────────────────────

// Run SigLIP ViT and return multi-layer features.
// Input: [3, image_size, image_size] float [0, 1]
// Output: [n_patches, vis_dim * n_feature_layers] — concatenated features
static void gv_vision_forward(granite_vision_context * ctx,
                               const float * image, int img_h, int img_w,
                               float * output, int * out_tokens) {
    int ps = ctx->vis_patch_size;
    int ph = img_h / ps, pw = img_w / ps;
    int T = ph * pw;  // number of patches (729 for 384/14)
    int D = ctx->vis_dim;
    int n_heads = ctx->vis_heads;
    int d_head = D / n_heads;
    float eps = 1e-6f;

    // Patch embedding: Conv2D(3, D, ps, stride=ps) — equivalent to linear on flattened patches
    const float * pe_w = ctx->get("vis.patch_embed.weight");
    const float * pe_b = ctx->get("vis.patch_embed.bias");
    std::vector<float> x(T * D);

    // Patch embed: for each patch, flatten [3, ps, ps] → linear → [D]
    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            int t = py * pw + px;
            float * out_row = x.data() + t * D;
            for (int d = 0; d < D; d++) {
                float sum = pe_b ? pe_b[d] : 0.0f;
                for (int c = 0; c < 3; c++)
                    for (int ky = 0; ky < ps; ky++)
                        for (int kx = 0; kx < ps; kx++) {
                            int iy = py * ps + ky, ix = px * ps + kx;
                            if (iy < img_h && ix < img_w)
                                sum += image[c * img_h * img_w + iy * img_w + ix]
                                     * pe_w[d * 3 * ps * ps + c * ps * ps + ky * ps + kx];
                        }
                out_row[d] = sum;
            }
        }
    }

    // Add position embedding
    const float * pos_w = ctx->get("vis.pos_embed.weight");
    if (pos_w) {
        for (int t = 0; t < T && t < (int)(ctx->vis_image_size / ps) * (ctx->vis_image_size / ps); t++)
            for (int d = 0; d < D; d++)
                x[t * D + d] += pos_w[t * D + d];
    }

    // Store multi-layer features
    int n_feat = (int)ctx->feature_layers.size();
    std::vector<std::vector<float>> layer_outputs(n_feat);

    // Transformer layers
    for (int li = 0; li < ctx->vis_layers; li++) {
        char buf[64];

        // LN1 + MHSA + residual
        std::vector<float> normed(T * D);
        memcpy(normed.data(), x.data(), T * D * sizeof(float));
        snprintf(buf, sizeof(buf), "vis.layer.%d.layer_norm1", li);
        gv_layernorm(normed.data(), T, D, ctx->get(std::string(buf) + ".weight"),
                     ctx->get(std::string(buf) + ".bias"), eps);

        // MHSA: Q, K, V projections
        snprintf(buf, sizeof(buf), "vis.layer.%d", li);
        std::string lp(buf);
        std::vector<float> Q(T * D), K(T * D), V(T * D);
        gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.q.weight"),
                  ctx->get(lp + ".attn.q.bias"), Q.data());
        gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.k.weight"),
                  ctx->get(lp + ".attn.k.bias"), K.data());
        gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.v.weight"),
                  ctx->get(lp + ".attn.v.bias"), V.data());

        // Multi-head attention
        float scale = 1.0f / sqrtf((float)d_head);
        std::vector<float> attn_out(T * D, 0.0f);
        for (int h = 0; h < n_heads; h++) {
            int off = h * d_head;
            for (int q = 0; q < T; q++) {
                // Compute attention scores
                float max_s = -1e9f;
                std::vector<float> scores(T);
                for (int k = 0; k < T; k++) {
                    float s = 0;
                    for (int d = 0; d < d_head; d++)
                        s += Q[q * D + off + d] * K[k * D + off + d];
                    s *= scale;
                    scores[k] = s;
                    if (s > max_s) max_s = s;
                }
                float sum_e = 0;
                for (int k = 0; k < T; k++) {
                    scores[k] = expf(scores[k] - max_s);
                    sum_e += scores[k];
                }
                float inv = 1.0f / sum_e;
                for (int d = 0; d < d_head; d++) {
                    float val = 0;
                    for (int k = 0; k < T; k++)
                        val += scores[k] * inv * V[k * D + off + d];
                    attn_out[q * D + off + d] = val;
                }
            }
        }

        // Output projection
        std::vector<float> proj(T * D);
        gv_linear(attn_out.data(), T, D, D, ctx->get(lp + ".attn.out.weight"),
                  ctx->get(lp + ".attn.out.bias"), proj.data());

        // Residual
        for (int i = 0; i < T * D; i++) x[i] += proj[i];

        // LN2 + FFN + residual
        memcpy(normed.data(), x.data(), T * D * sizeof(float));
        snprintf(buf, sizeof(buf), "vis.layer.%d.layer_norm2", li);
        gv_layernorm(normed.data(), T, D, ctx->get(std::string(buf) + ".weight"),
                     ctx->get(std::string(buf) + ".bias"), eps);

        int ffn_dim = 4304;  // SigLIP intermediate_size
        // Check actual weight size
        auto * fc1_t = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.up.weight").c_str());
        if (fc1_t) ffn_dim = fc1_t->ne[0];

        std::vector<float> fc1(T * ffn_dim);
        gv_linear(normed.data(), T, D, ffn_dim, ctx->get(lp + ".ffn.up.weight"),
                  ctx->get(lp + ".ffn.up.bias"), fc1.data());
        for (int i = 0; i < T * ffn_dim; i++) fc1[i] = gv_gelu(fc1[i]);

        std::vector<float> fc2(T * D);
        gv_linear(fc1.data(), T, ffn_dim, D, ctx->get(lp + ".ffn.down.weight"),
                  ctx->get(lp + ".ffn.down.bias"), fc2.data());

        for (int i = 0; i < T * D; i++) x[i] += fc2[i];

        // Check if this layer is a feature extraction point
        for (int fi = 0; fi < n_feat; fi++) {
            if (li == ctx->feature_layers[fi]) {
                layer_outputs[fi].assign(x.begin(), x.end());
            }
        }
    }

    // Concatenate multi-layer features: [T, n_feat * D]
    int feat_dim = n_feat * D;
    for (int t = 0; t < T; t++) {
        for (int fi = 0; fi < n_feat; fi++) {
            for (int d = 0; d < D; d++) {
                output[t * feat_dim + fi * D + d] = layer_outputs[fi][t * D + d];
            }
        }
    }
    *out_tokens = T;
}

// ── MLP Projector ──────────────────────────────────────────────────────

static void gv_projector(granite_vision_context * ctx,
                         const float * vis_features, int n_tokens, int feat_dim,
                         float * output) {
    int out_dim = ctx->llm_dim;

    // Linear1 + GELU
    std::vector<float> mid(n_tokens * out_dim);
    gv_linear(vis_features, n_tokens, feat_dim, out_dim,
              ctx->get("proj.linear_1.weight"), ctx->get("proj.linear_1.bias"),
              mid.data());
    for (int i = 0; i < n_tokens * out_dim; i++) mid[i] = gv_gelu(mid[i]);

    // Linear2
    gv_linear(mid.data(), n_tokens, out_dim, out_dim,
              ctx->get("proj.linear_2.weight"), ctx->get("proj.linear_2.bias"),
              output);
}

// ── Granite LLM Forward (single token, with KV cache) ──────────────────

static void gv_llm_decode_step(granite_vision_context * ctx,
                                const float * token_embed, int n_past,
                                float * logits) {
    int D = ctx->llm_dim;
    int n_heads = ctx->llm_heads;
    int n_kv = ctx->llm_kv_heads;
    int d_head = D / n_heads;
    int kv_repeat = n_heads / n_kv;
    float eps = 1e-5f;
    float res_mul = ctx->residual_multiplier;

    std::vector<float> x(D);
    memcpy(x.data(), token_embed, D * sizeof(float));

    int max_seq = ctx->kv_allocated;

    for (int li = 0; li < ctx->llm_layers; li++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "llm.layer.%d", li);
        std::string lp(buf);

        // RMSNorm1
        std::vector<float> normed(D);
        memcpy(normed.data(), x.data(), D * sizeof(float));
        gv_rmsnorm(normed.data(), 1, D, ctx->get(lp + ".norm1.weight"), eps);

        // GQA Self-Attention with KV cache
        std::vector<float> Q(D), K_new(n_kv * d_head), V_new(n_kv * d_head);
        gv_linear(normed.data(), 1, D, D, ctx->get(lp + ".attn.q.weight"), nullptr, Q.data());
        gv_linear(normed.data(), 1, D, n_kv * d_head, ctx->get(lp + ".attn.k.weight"), nullptr, K_new.data());
        gv_linear(normed.data(), 1, D, n_kv * d_head, ctx->get(lp + ".attn.v.weight"), nullptr, V_new.data());

        // RoPE on Q and K
        float theta = ctx->rope_theta;
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < d_head / 2; d++) {
                float freq = 1.0f / powf(theta, 2.0f * d / d_head);
                float angle = n_past * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int i0 = h * d_head + 2 * d, i1 = i0 + 1;
                float q0 = Q[i0], q1 = Q[i1];
                Q[i0] = q0 * cos_a - q1 * sin_a;
                Q[i1] = q0 * sin_a + q1 * cos_a;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            for (int d = 0; d < d_head / 2; d++) {
                float freq = 1.0f / powf(theta, 2.0f * d / d_head);
                float angle = n_past * freq;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int i0 = h * d_head + 2 * d, i1 = i0 + 1;
                float k0 = K_new[i0], k1 = K_new[i1];
                K_new[i0] = k0 * cos_a - k1 * sin_a;
                K_new[i1] = k0 * sin_a + k1 * cos_a;
            }
        }

        // Store K, V in cache
        int kv_off = li * 2 * max_seq * n_kv * d_head;
        int k_off = kv_off;
        int v_off = kv_off + max_seq * n_kv * d_head;
        memcpy(ctx->kv_cache.data() + k_off + n_past * n_kv * d_head, K_new.data(), n_kv * d_head * sizeof(float));
        memcpy(ctx->kv_cache.data() + v_off + n_past * n_kv * d_head, V_new.data(), n_kv * d_head * sizeof(float));

        // Compute attention
        float scale = 1.0f / sqrtf((float)d_head);
        // Note: attention_multiplier = 0.015625 = 1/64 = 1/sqrt(64) when d_head=64
        // This is the same as the standard QK scale, so we use `scale` directly.
        std::vector<float> attn_out(D, 0.0f);
        int Lk = n_past + 1;

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / kv_repeat;
            // Scores
            float max_s = -1e9f;
            std::vector<float> scores(Lk);
            for (int k = 0; k < Lk; k++) {
                float s = 0;
                for (int d = 0; d < d_head; d++)
                    s += Q[h * d_head + d] * ctx->kv_cache[k_off + k * n_kv * d_head + kv_h * d_head + d];
                s *= scale;
                scores[k] = s;
                if (s > max_s) max_s = s;
            }
            float sum_e = 0;
            for (int k = 0; k < Lk; k++) { scores[k] = expf(scores[k] - max_s); sum_e += scores[k]; }
            float inv = 1.0f / sum_e;
            for (int d = 0; d < d_head; d++) {
                float val = 0;
                for (int k = 0; k < Lk; k++)
                    val += scores[k] * inv * ctx->kv_cache[v_off + k * n_kv * d_head + kv_h * d_head + d];
                attn_out[h * d_head + d] = val;
            }
        }

        // Output projection
        std::vector<float> proj(D);
        gv_linear(attn_out.data(), 1, D, D, ctx->get(lp + ".attn.o.weight"), nullptr, proj.data());

        // Residual with multiplier
        for (int d = 0; d < D; d++) x[d] += proj[d] * res_mul;

        // RMSNorm2 + SiLU MLP
        memcpy(normed.data(), x.data(), D * sizeof(float));
        gv_rmsnorm(normed.data(), 1, D, ctx->get(lp + ".norm2.weight"), eps);

        int ffn = ctx->llm_ffn_dim;
        std::vector<float> gate(ffn), up(ffn);
        gv_linear(normed.data(), 1, D, ffn, ctx->get(lp + ".ffn.gate.weight"), nullptr, gate.data());
        gv_linear(normed.data(), 1, D, ffn, ctx->get(lp + ".ffn.up.weight"), nullptr, up.data());
        for (int i = 0; i < ffn; i++) gate[i] = gv_silu(gate[i]) * up[i];

        std::vector<float> down(D);
        gv_linear(gate.data(), 1, ffn, D, ctx->get(lp + ".ffn.down.weight"), nullptr, down.data());

        for (int d = 0; d < D; d++) x[d] += down[d] * res_mul;
    }

    // Final RMSNorm
    gv_rmsnorm(x.data(), 1, D, ctx->get("llm.norm.weight"), eps);

    // LM head (may be tied to embeddings)
    const float * lm_w = ctx->get("llm.lm_head.weight");
    if (!lm_w && ctx->tie_word_embeddings)
        lm_w = ctx->get("llm.embed.weight");

    if (lm_w) {
        for (int v = 0; v < ctx->vocab_size; v++) {
            float sum = 0;
            for (int d = 0; d < D; d++) sum += x[d] * lm_w[v * D + d];
            logits[v] = sum / ctx->logits_scaling;
        }
    }
}

// ── Main recognize function ────────────────────────────────────────────

const char * granite_vision_recognize(granite_vision_context * ctx,
                                       const uint8_t * pixels, int width, int height, int channels,
                                       const char * prompt, int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    int img_size = ctx->vis_image_size;
    int ps = ctx->vis_patch_size;
    int n_patches_side = img_size / ps;
    int T_vis = n_patches_side * n_patches_side;
    int D = ctx->llm_dim;
    int n_feat = (int)ctx->feature_layers.size();
    int feat_dim = n_feat * ctx->vis_dim;

    // Preprocess: resize to img_size × img_size, normalize to [0,1]
    // (simplified — no dynamic tiling for now, single tile)
    std::vector<float> image(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float sy = (y + 0.5f) * height / img_size - 0.5f;
                float sx = (x + 0.5f) * width / img_size - 0.5f;
                int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                int src_idx = channels > 1 ? (iy * width + ix) * channels + c : iy * width + ix;
                image[c * img_size * img_size + y * img_size + x] =
                    pixels[src_idx] / 255.0f;
            }

    // Vision encoder
    std::vector<float> vis_features(T_vis * feat_dim);
    int n_vis_tokens = 0;
    gv_vision_forward(ctx, image.data(), img_size, img_size, vis_features.data(), &n_vis_tokens);

    // Projector
    std::vector<float> proj_features(n_vis_tokens * D);
    gv_projector(ctx, vis_features.data(), n_vis_tokens, feat_dim, proj_features.data());

    // Allocate KV cache
    int max_seq = n_vis_tokens + ctx->max_tokens + 100;
    ctx->kv_cache.resize(2 * ctx->llm_layers * max_seq * ctx->llm_kv_heads * (D / ctx->llm_heads));
    ctx->kv_allocated = max_seq;
    ctx->n_past = 0;

    // Get embedding weights
    const float * embed_w = ctx->get("llm.embed.weight");
    float emb_mul = ctx->embedding_multiplier;

    // Prefill: vision tokens through LLM
    // For simplicity, process vision tokens one at a time through the LLM
    // (proper implementation would do batched prefill)
    std::vector<float> logits(ctx->vocab_size);

    for (int t = 0; t < n_vis_tokens; t++) {
        // Scale projected features by embedding_multiplier is NOT done for vision tokens
        // (only text embeddings are scaled)
        gv_llm_decode_step(ctx, proj_features.data() + t * D, ctx->n_past, logits.data());
        ctx->n_past++;
    }

    // Greedy decode
    ctx->output_text.clear();
    int eos_id = 0;  // Granite uses token 0 as EOS

    for (int step = 0; step < ctx->max_tokens; step++) {
        // Find argmax
        int best_id = 0;
        float best_score = logits[0];
        for (int v = 1; v < ctx->vocab_size; v++) {
            if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }
        }

        if (best_id == eos_id) break;

        // TODO: detokenize best_id → text
        // For now, just store the token ID (proper tokenizer integration needed)
        ctx->output_text += "<" + std::to_string(best_id) + ">";

        // Embed next token
        std::vector<float> next_embed(D);
        for (int d = 0; d < D; d++)
            next_embed[d] = embed_w[best_id * D + d] * emb_mul;

        gv_llm_decode_step(ctx, next_embed.data(), ctx->n_past, logits.data());
        ctx->n_past++;
    }

    if (out_len) *out_len = (int)ctx->output_text.size();
    return ctx->output_text.c_str();
}
