// math_ocr.cpp — DeiT+TrOCR math OCR via ggml graph compute.
//
// Uses ggml's graph-based computation for SIMD + multi-threading,
// following the same pattern as crispembed.cpp's BERT encoder.

#include "math_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct enc_layer {
    ggml_tensor *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
    ggml_tensor *attn_out_w, *attn_out_b, *ln2_w, *ln2_b;
    ggml_tensor *ff_up_w, *ff_up_b, *ff_down_w, *ff_down_b;
};

struct dec_layer {
    ggml_tensor *self_ln_w, *self_ln_b;
    ggml_tensor *self_q_w, *self_q_b, *self_k_w, *self_k_b, *self_v_w, *self_v_b;
    ggml_tensor *self_out_w, *self_out_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b, *cross_k_w, *cross_k_b, *cross_v_w, *cross_v_b;
    ggml_tensor *cross_out_w, *cross_out_b;
    ggml_tensor *ff_ln_w, *ff_ln_b, *ff_up_w, *ff_up_b, *ff_down_w, *ff_down_b;
};

struct math_ocr_context {
    math_ocr_hparams hparams;

    // Encoder
    ggml_tensor *cls_token, *dist_token, *patch_proj_w, *patch_proj_b, *pos_embed;
    ggml_tensor *enc_ln_w, *enc_ln_b;
    std::vector<enc_layer> enc_layers;

    // Decoder
    ggml_tensor *tok_embed, *pos_embed_dec, *dec_embed_ln_w, *dec_embed_ln_b;
    ggml_tensor *dec_final_ln_w, *dec_final_ln_b, *lm_head_w, *lm_head_b;
    std::vector<dec_layer> dec_layers;

    // Infrastructure
    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads;
    std::string result_buf;

    // Cached encoder output + cross-attention K/V (precomputed once)
    std::vector<float> enc_out;
    int n_enc_tokens = 0;
    std::vector<std::vector<float>> cross_k_cache; // per layer [n_enc * D]
    std::vector<std::vector<float>> cross_v_cache;
};

// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

static ggml_tensor* F(const std::map<std::string, ggml_tensor*>& m, const char* n) {
    auto it = m.find(n);
    if (it != m.end()) return it->second;
    std::string alt(n);
    for (auto& c : alt) if (c == '.') c = '_';
    it = m.find(alt);
    return it != m.end() ? it->second : nullptr;
}

static void map_tensors(math_ocr_context* ctx) {
    const auto& m = ctx->wl.tensors;
    const auto& hp = ctx->hparams;
    char buf[256];

    ctx->cls_token    = F(m, "enc.embeddings.cls_token");
    ctx->dist_token   = F(m, "enc.embeddings.distillation_token");
    ctx->patch_proj_w = F(m, "enc.embeddings.patch_embeddings.projection.weight");
    ctx->patch_proj_b = F(m, "enc.embeddings.patch_embeddings.projection.bias");
    ctx->pos_embed    = F(m, "enc.embeddings.position_embeddings");
    ctx->enc_ln_w     = F(m, "enc.layernorm.weight");
    ctx->enc_ln_b     = F(m, "enc.layernorm.bias");

    ctx->enc_layers.resize(hp.enc_layers);
    for (int i = 0; i < hp.enc_layers; i++) {
        auto& l = ctx->enc_layers[i];
        auto E = [&](const char* s) { snprintf(buf, sizeof(buf), "enc.encoder.layer.%d.%s", i, s); return F(m, buf); };
        l.ln1_w = E("layernorm_before.weight"); l.ln1_b = E("layernorm_before.bias");
        l.q_w = E("attention.attention.query.weight"); l.q_b = E("attention.attention.query.bias");
        l.k_w = E("attention.attention.key.weight"); l.k_b = E("attention.attention.key.bias");
        l.v_w = E("attention.attention.value.weight"); l.v_b = E("attention.attention.value.bias");
        l.attn_out_w = E("attention.output.dense.weight"); l.attn_out_b = E("attention.output.dense.bias");
        l.ln2_w = E("layernorm_after.weight"); l.ln2_b = E("layernorm_after.bias");
        l.ff_up_w = E("intermediate.dense.weight"); l.ff_up_b = E("intermediate.dense.bias");
        l.ff_down_w = E("output.dense.weight"); l.ff_down_b = E("output.dense.bias");
    }

    auto D2 = [&](const char* s, const char* f) { auto* t = F(m, s); return t ? t : F(m, f); };
    ctx->tok_embed = D2("dec.d.embed_tokens.weight", "dec.decoder.model.decoder.embed_tokens.weight");
    ctx->pos_embed_dec = D2("dec.d.embed_positions.weight", "dec.decoder.model.decoder.embed_positions.weight");
    ctx->dec_embed_ln_w = D2("dec.d.layernorm_embedding.weight", "dec.decoder.model.decoder.layernorm_embedding.weight");
    ctx->dec_embed_ln_b = D2("dec.d.layernorm_embedding.bias", "dec.decoder.model.decoder.layernorm_embedding.bias");
    ctx->dec_final_ln_w = D2("dec.d.layer_norm.weight", "dec.decoder.model.decoder.layer_norm.weight");
    ctx->dec_final_ln_b = D2("dec.d.layer_norm.bias", "dec.decoder.model.decoder.layer_norm.bias");
    ctx->lm_head_w = F(m, "dec.lm_head.weight");
    ctx->lm_head_b = F(m, "dec.lm_head.bias");

    ctx->dec_layers.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto& l = ctx->dec_layers[i];
        auto DL = [&](const char* s) {
            snprintf(buf, sizeof(buf), "dec.d.layers.%d.%s", i, s);
            auto* t = F(m, buf);
            if (t) return t;
            snprintf(buf, sizeof(buf), "dec.decoder.model.decoder.layers.%d.%s", i, s);
            return F(m, buf);
        };
        l.self_ln_w = DL("self_attn_layer_norm.weight"); l.self_ln_b = DL("self_attn_layer_norm.bias");
        l.self_q_w = DL("self_attn.q_proj.weight"); l.self_q_b = DL("self_attn.q_proj.bias");
        l.self_k_w = DL("self_attn.k_proj.weight"); l.self_k_b = DL("self_attn.k_proj.bias");
        l.self_v_w = DL("self_attn.v_proj.weight"); l.self_v_b = DL("self_attn.v_proj.bias");
        l.self_out_w = DL("self_attn.out_proj.weight"); l.self_out_b = DL("self_attn.out_proj.bias");
        auto* xw = DL("xaln.weight"); l.cross_ln_w = xw ? xw : DL("encoder_attn_layer_norm.weight");
        auto* xb = DL("xaln.bias"); l.cross_ln_b = xb ? xb : DL("encoder_attn_layer_norm.bias");
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
// ggml graph helpers
// ---------------------------------------------------------------------------

// LayerNorm via ggml ops
static ggml_tensor* g_ln(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_norm(g, x, 1e-6f);
    x = ggml_mul(g, x, w);
    if (b) x = ggml_add(g, x, b);
    return x;
}

// Linear projection: out = W @ x + b  (W is [out, in], x is [in, T])
static ggml_tensor* g_linear(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, b);
    return x;
}

// Multi-head self-attention (non-causal)
static ggml_tensor* g_mha(ggml_context* g, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,
                            int n_heads, int T) {
    int H = Q->ne[0];
    int hd = H / n_heads;
    // Reshape [H, T] → [hd, nh, T] → permute [hd, T, nh]
    // Reshape [H, T] → [hd, nh, T] → permute → [hd, T, nh] → cont
    Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Q, hd, n_heads, T), 0, 2, 1, 3));
    K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, K, hd, n_heads, T), 0, 2, 1, 3));
    V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, V, hd, n_heads, T), 0, 2, 1, 3));
    // scores = Q^T @ K / sqrt(hd) → [T, T, nh]
    ggml_tensor* scores = ggml_mul_mat(g, K, Q);
    scores = ggml_scale(g, scores, 1.0f / sqrtf((float)hd));
    scores = ggml_soft_max(g, scores);
    // attn = V @ scores^T → we need [hd, T, nh]
    // V is [hd, T, nh], scores is [T, T, nh]
    // We want: for each head, output[d, qi] = Σ_ki scores[ki, qi] * V[d, ki]
    // That's V @ scores^T per head = ggml_mul_mat(V_transposed, scores)
    // But simpler: transpose scores → [T, T, nh], then ggml_mul_mat(V, scores_t) doesn't work.
    // Use the flash_attn pattern: scores @ V → need V as [hd, T, nh]
    // ggml_mul_mat(a, b) computes b @ a^T. We want scores @ V.
    // So: ggml_mul_mat(V_transposed, scores) where V_transposed is [T, hd, nh]
    ggml_tensor* Vt = ggml_cont(g, ggml_transpose(g, V)); // [T, hd, nh]
    ggml_tensor* attn = ggml_mul_mat(g, Vt, scores);       // [hd, T, nh]
    // → [hd, nh, T] → [H, T]
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    return ggml_reshape_2d(g, attn, H, T);
}

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

static ggml_cgraph* build_encoder_graph(math_ocr_context* ctx, ggml_context* g, int T) {
    const auto& hp = ctx->hparams;
    const int H = hp.enc_hidden;

    ggml_cgraph* gf = ggml_new_graph_custom(g, hp.enc_layers * 30 + 256, false);

    // Input: pre-embedded patches [H, T] as a float input tensor
    ggml_tensor* inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, H, T);
    ggml_set_name(inp, "enc_input");
    ggml_set_input(inp);

    ggml_tensor* cur = inp;

    // Log input tensor shape
    fprintf(stderr, "math_ocr: graph input ne=[%lld, %lld]\n",
            (long long)inp->ne[0], (long long)inp->ne[1]);

    // Transformer layers
    for (int il = 0; il < hp.enc_layers; il++) {
        const auto& L = ctx->enc_layers[il];
        if (!L.q_w) continue;

        ggml_tensor* residual = cur;

        // Pre-LN
        cur = g_ln(g, cur, L.ln1_w, L.ln1_b);

        // Self-attention
        if (il == 0 && L.q_w) {
            fprintf(stderr, "math_ocr: layer 0 q_w ne=[%lld, %lld], cur ne=[%lld, %lld]\n",
                    (long long)L.q_w->ne[0], (long long)L.q_w->ne[1],
                    (long long)cur->ne[0], (long long)cur->ne[1]);
        }
        ggml_tensor* Q = g_linear(g, cur, L.q_w, L.q_b);
        ggml_tensor* K = g_linear(g, cur, L.k_w, L.k_b);
        ggml_tensor* V = g_linear(g, cur, L.v_w, L.v_b);
        ggml_tensor* attn = g_mha(g, Q, K, V, hp.enc_heads, T);
        attn = g_linear(g, attn, L.attn_out_w, L.attn_out_b);
        cur = ggml_add(g, residual, attn);

        // Post-LN + FFN
        residual = cur;
        cur = g_ln(g, cur, L.ln2_w, L.ln2_b);
        ggml_tensor* up = g_linear(g, cur, L.ff_up_w, L.ff_up_b);
        up = ggml_gelu(g, up);
        cur = g_linear(g, up, L.ff_down_w, L.ff_down_b);
        cur = ggml_add(g, residual, cur);
    }

    // Final LN
    if (ctx->enc_ln_w)
        cur = g_ln(g, cur, ctx->enc_ln_w, ctx->enc_ln_b);

    ggml_set_name(cur, "enc_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ---------------------------------------------------------------------------
// Run encoder
// ---------------------------------------------------------------------------

static void run_encoder(math_ocr_context* ctx, const float* pixels_rgb, int img_w, int img_h) {
    const auto& hp = ctx->hparams;
    const int P = hp.patch_size, H = hp.enc_hidden;
    const int npw = img_w / P, nph = img_h / P;
    const int n_patches = npw * nph;
    const int T = n_patches + 2;

    // Step 1: CPU-side patch embedding (conv projection)
    // This is a small op — keep on CPU, feed result to graph as input
    std::vector<float> embedded(T * H, 0.0f);
    const int patch_dim = 3 * P * P;

    if (ctx->patch_proj_w) {
        const float* W = (const float*)ctx->patch_proj_w->data;
        const float* B = ctx->patch_proj_b ? (const float*)ctx->patch_proj_b->data : nullptr;
        for (int py = 0; py < nph; py++) {
            for (int px = 0; px < npw; px++) {
                // ggml stores in column-major: tensor[H, T] → data[t * H + h]
                int t = py * npw + px + 2;
                for (int h = 0; h < H; h++) {
                    float sum = B ? B[h] : 0.0f;
                    for (int c = 0; c < 3; c++)
                        for (int dy = 0; dy < P; dy++)
                            for (int dx = 0; dx < P; dx++) {
                                int sy = py * P + dy, sx = px * P + dx;
                                float v = pixels_rgb[(c * img_h + sy) * img_w + sx];
                                sum += v * W[h * patch_dim + c * P * P + dy * P + dx];
                            }
                    embedded[t * H + h] = sum;
                }
            }
        }
    }

    // CLS + distillation tokens
    if (ctx->cls_token) {
        const float* d = (const float*)ctx->cls_token->data;
        for (int h = 0; h < H; h++) embedded[0 * H + h] = d[h];
    }
    if (ctx->dist_token) {
        const float* d = (const float*)ctx->dist_token->data;
        for (int h = 0; h < H; h++) embedded[1 * H + h] = d[h];
    }

    // Positional embeddings
    if (ctx->pos_embed) {
        const float* pe = (const float*)ctx->pos_embed->data;
        int n = std::min(T * H, (int)ggml_nelements(ctx->pos_embed));
        for (int i = 0; i < n; i++) embedded[i] += pe[i];
    }

    // Step 2: Build and run ggml graph for transformer layers
    fprintf(stderr, "math_ocr: encoder start (T=%d, H=%d, %d layers)\n", T, H, hp.enc_layers);
    size_t meta_size = 256 * 1024 * 1024; // 256 MB metadata buffer
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params ip = { meta_size, meta.data(), true };
    ggml_context* g = ggml_init(ip);

    ggml_cgraph* gf = build_encoder_graph(ctx, g, T);

    // Allocate + set input
    fprintf(stderr, "math_ocr: graph built (%d nodes), allocating...\n", ggml_graph_n_nodes(gf));
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "math_ocr: encoder alloc failed\n");
        ggml_free(g);
        return;
    }

    ggml_tensor* inp = ggml_graph_get_tensor(gf, "enc_input");
    if (inp) ggml_backend_tensor_set(inp, embedded.data(), 0, T * H * sizeof(float));

    // Compute
    fprintf(stderr, "math_ocr: computing encoder...\n");
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    fprintf(stderr, "math_ocr: encoder done\n");

    // Read output
    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_output");
    ctx->enc_out.resize(T * H);
    if (out) ggml_backend_tensor_get(out, ctx->enc_out.data(), 0, T * H * sizeof(float));

    ctx->n_enc_tokens = T;

    // Debug: dump first few encoder output values for comparison with ONNX ref
    fprintf(stderr, "math_ocr: enc_out token 0 first 5: ");
    for (int i = 0; i < 5 && i < H; i++) fprintf(stderr, "%.6f ", ctx->enc_out[0 * H + i]);
    fprintf(stderr, "\nmath_ocr: enc_out token 2 first 5: ");
    for (int i = 0; i < 5 && i < H; i++) fprintf(stderr, "%.6f ", ctx->enc_out[2 * H + i]);
    fprintf(stderr, "\n");

    // Also dump pre-transformer embedded values for debugging
    fprintf(stderr, "math_ocr: embedded (pre-graph) token 0 first 5: ");
    for (int i = 0; i < 5 && i < H; i++) fprintf(stderr, "%.6f ", embedded[0 * H + i]);
    fprintf(stderr, "\nmath_ocr: embedded (pre-graph) token 2 first 5: ");
    for (int i = 0; i < 5 && i < H; i++) fprintf(stderr, "%.6f ", embedded[2 * H + i]);
    fprintf(stderr, "\n");

    ggml_free(g);
}

// ---------------------------------------------------------------------------
// Decoder (scalar — fast enough for autoregressive single-token steps)
// ---------------------------------------------------------------------------

static void layernorm_cpu(const float* x, float* y, int D, const ggml_tensor* w, const ggml_tensor* b) {
    if (!w || !b) { if (x != y) memcpy(y, x, D * sizeof(float)); return; }
    const float* W = (const float*)w->data;
    const float* B = (const float*)b->data;
    float mean = 0, var = 0;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;
    for (int i = 0; i < D; i++) { float d = x[i] - mean; var += d * d; }
    float inv = 1.0f / sqrtf(var / D + 1e-6f);
    for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * W[i] + B[i];
}

static void linear_cpu(const float* inp, float* out, int in_d, int out_d, const ggml_tensor* w, const ggml_tensor* b) {
    if (!w) { memset(out, 0, out_d * sizeof(float)); return; }
    const float* W = (const float*)w->data;
    const float* B = b ? (const float*)b->data : nullptr;
    for (int o = 0; o < out_d; o++) {
        float s = B ? B[o] : 0.0f;
        for (int j = 0; j < in_d; j++) s += inp[j] * W[o * in_d + j];
        out[o] = s;
    }
}

static void mha_1q(const float* q, const float* K, const float* V,
                    float* out, int n_kv, int D, int n_heads) {
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> sc(n_kv);
        float mx = -1e9f;
        for (int k = 0; k < n_kv; k++) {
            float dot = 0;
            for (int d = 0; d < hd; d++) dot += q[h*hd+d] * K[k*D+h*hd+d];
            sc[k] = dot * scale;
            mx = std::max(mx, sc[k]);
        }
        float se = 0;
        for (auto& s : sc) { s = expf(s - mx); se += s; }
        for (auto& s : sc) s /= se;
        for (int d = 0; d < hd; d++) {
            float sum = 0;
            for (int k = 0; k < n_kv; k++) sum += sc[k] * V[k*D+h*hd+d];
            out[h*hd+d] = sum;
        }
    }
}

static std::vector<float> decoder_step(math_ocr_context* ctx,
                                         int tok, int step,
                                         std::vector<std::vector<float>>& kv_k,
                                         std::vector<std::vector<float>>& kv_v) {
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model, E = hp.enc_hidden, V = hp.vocab_size;

    std::vector<float> x(D, 0.0f), normed(D);

    // Token + positional embedding
    if (ctx->tok_embed && tok >= 0 && tok < V) {
        const float* emb = (const float*)ctx->tok_embed->data;
        float sc = sqrtf((float)D);
        for (int i = 0; i < D; i++) x[i] = emb[tok * D + i] * sc;
    }
    if (ctx->pos_embed_dec) {
        const float* pe = (const float*)ctx->pos_embed_dec->data;
        int pos = step + 2;
        if (pos < hp.max_seq_len)
            for (int i = 0; i < D; i++) x[i] += pe[pos * D + i];
    }
    layernorm_cpu(x.data(), x.data(), D, ctx->dec_embed_ln_w, ctx->dec_embed_ln_b);
    if (step == 0) {
        fprintf(stderr, "math_ocr: dec embed+pos+ln x[:5]=[%.4f %.4f %.4f %.4f %.4f]\n",
                x[0],x[1],x[2],x[3],x[4]);
    }

    for (int li = 0; li < hp.dec_layers; li++) {
        const auto& l = ctx->dec_layers[li];
        if (!l.self_q_w) continue;

        // Self-attention with KV cache
        layernorm_cpu(x.data(), normed.data(), D, l.self_ln_w, l.self_ln_b);
        std::vector<float> q(D), k(D), v(D);
        linear_cpu(normed.data(), q.data(), D, D, l.self_q_w, l.self_q_b);
        linear_cpu(normed.data(), k.data(), D, D, l.self_k_w, l.self_k_b);
        linear_cpu(normed.data(), v.data(), D, D, l.self_v_w, l.self_v_b);
        kv_k[li].insert(kv_k[li].end(), k.begin(), k.end());
        kv_v[li].insert(kv_v[li].end(), v.begin(), v.end());

        std::vector<float> sa(D);
        mha_1q(q.data(), kv_k[li].data(), kv_v[li].data(), sa.data(), step+1, D, hp.dec_heads);
        std::vector<float> sa_proj(D);
        linear_cpu(sa.data(), sa_proj.data(), D, D, l.self_out_w, l.self_out_b);
        for (int i = 0; i < D; i++) x[i] += sa_proj[i];

        // Cross-attention (K/V cached after first call)
        if (l.cross_q_w && !ctx->enc_out.empty()) {
            layernorm_cpu(x.data(), normed.data(), D, l.cross_ln_w, l.cross_ln_b);
            std::vector<float> cq(D);
            linear_cpu(normed.data(), cq.data(), D, D, l.cross_q_w, l.cross_q_b);

            int n_enc = ctx->n_enc_tokens;
            // Use cached cross K/V (precomputed once after encoding)
            const float* ck = ctx->cross_k_cache[li].data();
            const float* cv = ctx->cross_v_cache[li].data();

            std::vector<float> ca(D);
            mha_1q(cq.data(), ck, cv, ca.data(), n_enc, D, hp.dec_heads);
            std::vector<float> ca_proj(D);
            linear_cpu(ca.data(), ca_proj.data(), D, D, l.cross_out_w, l.cross_out_b);
            for (int i = 0; i < D; i++) x[i] += ca_proj[i];
        }

        // FFN
        if (l.ff_up_w) {
            layernorm_cpu(x.data(), normed.data(), D, l.ff_ln_w, l.ff_ln_b);
            const int FF = hp.dec_ffn_dim;
            std::vector<float> inter(FF), ffn(D);
            linear_cpu(normed.data(), inter.data(), D, FF, l.ff_up_w, l.ff_up_b);
            for (int i = 0; i < FF; i++) inter[i] = inter[i] > 0 ? inter[i] : 0;
            linear_cpu(inter.data(), ffn.data(), FF, D, l.ff_down_w, l.ff_down_b);
            for (int i = 0; i < D; i++) x[i] += ffn[i];
        }
    }

    layernorm_cpu(x.data(), x.data(), D, ctx->dec_final_ln_w, ctx->dec_final_ln_b);
    std::vector<float> logits(V, 0.0f);
    linear_cpu(x.data(), logits.data(), D, V, ctx->lm_head_w, ctx->lm_head_b);
    return logits;
}

// ---------------------------------------------------------------------------
// Init / Free / API
// ---------------------------------------------------------------------------

math_ocr_context* math_ocr_init(const char* model_path, int n_threads) {
    auto ctx = std::make_unique<math_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

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

    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (!core_gguf::load_weights(model_path, ctx->backend, "math_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        return nullptr;
    }

    // Create scheduler
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 4096, false, false);

    map_tensors(ctx.get());

    int me = 0, md = 0;
    for (auto& l : ctx->enc_layers) if (l.q_w) me++;
    for (auto& l : ctx->dec_layers) if (l.self_q_w) md++;
    fprintf(stderr, "math_ocr: mapped %d/%d enc, %d/%d dec\n", me, hp.enc_layers, md, hp.dec_layers);

    return ctx.release();
}

void math_ocr_free(math_ocr_context* ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const math_ocr_hparams* math_ocr_get_hparams(const math_ocr_context* ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

const char* math_ocr_recognize(math_ocr_context* ctx, const float* pixels,
                                 int width, int height, int* out_len) {
    if (!ctx || !pixels) return nullptr;
    const int S = ctx->hparams.image_size;

    // Resize + expand gray→3ch CHW + normalize (mean=0.5, std=0.5)
    std::vector<float> rgb(3 * S * S);
    float sx = (float)width / S, sy_f = (float)height / S;
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++) {
            int ox = std::min((int)(x * sx), width - 1);
            int oy = std::min((int)(y * sy_f), height - 1);
            float v = (pixels[oy * width + ox] - 0.5f) / 0.5f;
            rgb[0*S*S + y*S + x] = v;
            rgb[1*S*S + y*S + x] = v;
            rgb[2*S*S + y*S + x] = v;
        }

    // Encoder (ggml graph — fast)
    run_encoder(ctx, rgb.data(), S, S);

    // Precompute cross-attention K/V (once, reused for all decoder steps)
    {
        const int n_enc = ctx->n_enc_tokens;
        const int D = ctx->hparams.dec_d_model;
        const int E = ctx->hparams.enc_hidden;
        ctx->cross_k_cache.resize(ctx->hparams.dec_layers);
        ctx->cross_v_cache.resize(ctx->hparams.dec_layers);
        for (int li = 0; li < ctx->hparams.dec_layers; li++) {
            const auto& l = ctx->dec_layers[li];
            ctx->cross_k_cache[li].resize(n_enc * D);
            ctx->cross_v_cache[li].resize(n_enc * D);
            for (int t = 0; t < n_enc; t++) {
                linear_cpu(ctx->enc_out.data() + t * E,
                           ctx->cross_k_cache[li].data() + t * D,
                           E, D, l.cross_k_w, l.cross_k_b);
                linear_cpu(ctx->enc_out.data() + t * E,
                           ctx->cross_v_cache[li].data() + t * D,
                           E, D, l.cross_v_w, l.cross_v_b);
            }
        }
        fprintf(stderr, "math_ocr: cross K/V cached (%d enc tokens × %d layers)\n",
                n_enc, ctx->hparams.dec_layers);
    }

    // Decoder (scalar — fast enough for autoregressive)
    fprintf(stderr, "math_ocr: starting decoder (vocab=%d, start_tok=%d)\n",
            ctx->hparams.vocab_size, ctx->hparams.decoder_start_token);
    const auto& hp = ctx->hparams;
    std::vector<int> tokens = {hp.decoder_start_token};
    std::vector<std::vector<float>> kk(hp.dec_layers), kv(hp.dec_layers);

    for (int step = 0; step < hp.max_seq_len; step++) {
        auto logits = decoder_step(ctx, tokens.back(), step, kk, kv);
        int best = 0; float best_s = logits[0];
        for (int v = 1; v < hp.vocab_size; v++)
            if (logits[v] > best_s) { best_s = logits[v]; best = v; }
        if (step < 5) {
            fprintf(stderr, "math_ocr: dec step %d: tok=%d logits[0..4]=[%.3f %.3f %.3f %.3f %.3f] best=%d\n",
                    step, tokens.back(), logits[0], logits[1], logits[2], logits[3], logits[4], best);
        }
        if (best == hp.eos_token || best == hp.pad_token) break;
        tokens.push_back(best);
    }

    ctx->result_buf.clear();
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < (int)ctx->vocab.size()) ctx->result_buf += ctx->vocab[tok];
    }

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float* math_ocr_get_encoder_output(const math_ocr_context* ctx, int* out_n, int* out_h) {
    if (!ctx || ctx->enc_out.empty()) return nullptr;
    if (out_n) *out_n = ctx->n_enc_tokens;
    if (out_h) *out_h = ctx->hparams.enc_hidden;
    return ctx->enc_out.data();
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
