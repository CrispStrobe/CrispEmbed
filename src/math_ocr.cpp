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
#include "core/cpu_ops.h"

// stb_image declarations (implementation lives in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
    void stbi_image_free(void *retval_from_stbi_load);
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FP16/quantized → F32 dequantization for CPU-side tensor access
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
        // Quantized types — use ggml's dequantize
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, out.data(), n);
        } else {
            memset(out.data(), 0, n * sizeof(float));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct math_ocr_enc_layer {
    ggml_tensor *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b;
    ggml_tensor *attn_out_w, *attn_out_b, *ln2_w, *ln2_b;
    ggml_tensor *ff_up_w, *ff_up_b, *ff_down_w, *ff_down_b;
};

struct math_ocr_dec_layer {
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
    std::vector<math_ocr_enc_layer> enc_layers;

    // Decoder
    ggml_tensor *tok_embed, *pos_embed_dec, *dec_embed_ln_w, *dec_embed_ln_b;
    ggml_tensor *dec_final_ln_w, *dec_final_ln_b, *lm_head_w, *lm_head_b;
    std::vector<math_ocr_dec_layer> dec_layers;

    // Infrastructure
    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;  // per-context (replaces global _deq_cache)
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads;
    std::string result_buf;
    std::vector<float> char_confidences; // per-token softmax probabilities

    bool scale_embedding = true; // TrOCR: scale tok embed by sqrt(d_model)

    // Cached encoder output + cross-attention K/V (precomputed once)
    std::vector<float> enc_out;
    int n_enc_tokens = 0;
    std::vector<std::vector<float>> cross_k_cache; // per layer [n_enc * D]
    std::vector<std::vector<float>> cross_v_cache;

    // Cached encoder ggml graph (built once per T, reused for all crops of same size)
    ggml_context * enc_graph_g = nullptr;
    ggml_cgraph  * enc_graph   = nullptr;
    int enc_graph_T = -1;

    // Batched encoder graph (cached per T×B)
    ggml_context * enc_batch_g = nullptr;
    ggml_cgraph  * enc_batch   = nullptr;
    int enc_batch_T = -1;
    int enc_batch_B = -1;

    // Per-crop encoder outputs from the last encode_batch call
    std::vector<std::vector<float>> batch_enc_outs; // [n_crops][T * H]

    // Pre-allocated decoder scratch (avoids per-step heap allocs)
    struct dec_scratch {
        std::vector<float> x, normed, q, k, v, sa, sa_proj;
        std::vector<float> cq, ca, ca_proj;
        std::vector<float> inter, ffn;
        std::vector<float> logits;
        std::vector<float> scores; // for mha_1q
        bool allocated = false;
    } ds;

    bool bench = false;
};

// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

static ggml_tensor* F(const std::unordered_map<std::string, ggml_tensor*>& m, const char* n) {
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
    // Tied embeddings: lm_head shares embed_tokens weight
    if (!ctx->lm_head_w && ctx->tok_embed)
        ctx->lm_head_w = ctx->tok_embed;

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
// Ensure tensor is F32 for binary ops (quantized models keep biases as F16)
static ggml_tensor* ensure_f32(ggml_context* g, ggml_tensor* t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

static ggml_tensor* g_ln(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_norm(g, x, 1e-6f);
    x = ggml_mul(g, x, ensure_f32(g, w));
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

// Linear projection: out = W @ x + b  (W is [out, in], x is [in, T])
// ggml_mul_mat handles mixed-precision natively; only bias add needs F32 cast.
static ggml_tensor* g_linear(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    if (!w) return x;
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

// Multi-head self-attention (non-causal, symmetric Q/K/V length)
static ggml_tensor* g_mha(ggml_context* g, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,
                            int n_heads, int T) {
    int H = Q->ne[0];
    int hd = H / n_heads;
    // Reshape [H, T] → [hd, nh, T] → permute [hd, T, nh]
    // Reshape [H, T] → [hd, nh, T] → permute → [hd, T, nh] → cont
    Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Q, hd, n_heads, T), 0, 2, 1, 3));
    K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, K, hd, n_heads, T), 0, 2, 1, 3));
    V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, V, hd, n_heads, T), 0, 2, 1, 3));
    // Q,K,V: [hd, T, nh]

    ggml_tensor* attn;
    // Manual matmul attention (more reliable than flash_attn_ext for some models)
    ggml_tensor* scores = ggml_mul_mat(g, K, Q); // [T, T, nh]
    scores = ggml_soft_max_ext(g, scores, nullptr, 1.0f / sqrtf((float)hd), 0.0f);
    ggml_tensor* V_t = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3)); // [T, hd, nh]
    attn = ggml_mul_mat(g, V_t, scores); // [hd, T, nh]

    // Output: [hd, T, nh] → [hd, nh, T] → [H, T]
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    return ggml_reshape_2d(g, attn, H, T);
}

// Multi-head self-attention with batch dimension.
// Q, K, V: [H, T, B]; returns [H, T, B]
static ggml_tensor* g_mha_batch(ggml_context* g, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,
                                  int n_heads, int T, int B) {
    int H = (int)Q->ne[0];
    int hd = H / n_heads;
    // [H, T, B] → [hd, nh, T, B] → permute → [hd, T, nh, B]
    Q = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, Q, hd, n_heads, T, B), 0, 2, 1, 3));
    K = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, K, hd, n_heads, T, B), 0, 2, 1, 3));
    V = ggml_cont(g, ggml_permute(g, ggml_reshape_4d(g, V, hd, n_heads, T, B), 0, 2, 1, 3));
    // flash_attn: Q[hd,T,nh,B] → result[hd,nh,T,B] (permuted)
    ggml_tensor* attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
    // [hd,nh,T,B] → permute → [hd,T,nh,B] → reshape → [H,T,B]
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    return ggml_reshape_3d(g, attn, H, T, B);
}

// Batched encoder graph: processes B images simultaneously.
// Input: [H, T, B] pre-embedded patches; output: [H, T, B] encoder features.
static ggml_cgraph* build_encoder_graph_batch(math_ocr_context* ctx, ggml_context* g, int T, int B) {
    const auto& hp = ctx->hparams;
    const int H = hp.enc_hidden;

    ggml_cgraph* gf = ggml_new_graph_custom(g, hp.enc_layers * 80 + 512, false);

    ggml_tensor* inp = ggml_new_tensor_3d(g, GGML_TYPE_F32, H, T, B);
    ggml_set_name(inp, "enc_batch_input");
    ggml_set_input(inp);

    ggml_tensor* cur = inp;

    for (int il = 0; il < hp.enc_layers; il++) {
        const auto& L = ctx->enc_layers[il];
        if (!L.q_w) continue;

        ggml_tensor* residual = cur;
        cur = g_ln(g, cur, L.ln1_w, L.ln1_b);

        ggml_tensor* Q = g_linear(g, cur, L.q_w, L.q_b);
        ggml_tensor* K = g_linear(g, cur, L.k_w, L.k_b);
        ggml_tensor* V = g_linear(g, cur, L.v_w, L.v_b);
        ggml_tensor* attn = g_mha_batch(g, Q, K, V, hp.enc_heads, T, B);
        attn = g_linear(g, attn, L.attn_out_w, L.attn_out_b);
        cur = ggml_add(g, residual, attn);

        residual = cur;
        cur = g_ln(g, cur, L.ln2_w, L.ln2_b);
        ggml_tensor* up = g_linear(g, cur, L.ff_up_w, L.ff_up_b);
        up = ggml_gelu(g, up);
        cur = g_linear(g, up, L.ff_down_w, L.ff_down_b);
        cur = ggml_add(g, residual, cur);
    }

    if (ctx->enc_ln_w)
        cur = g_ln(g, cur, ctx->enc_ln_w, ctx->enc_ln_b);

    ggml_set_name(cur, "enc_batch_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Multi-head attention with single-token query vs multi-token K/V (asymmetric).
// Q: [D, 1], K: [D, n_kv], V: [D, n_kv]
// Returns: [D, 1]
static ggml_tensor* g_mha_1q(ggml_context* g, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,
                               int n_heads, int n_kv) {
    int H = (int)Q->ne[0];
    int hd = H / n_heads;
    // Q: [H, 1] → [hd, nh, 1] → permute → [hd, 1, nh]
    Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Q, hd, n_heads, 1), 0, 2, 1, 3));
    // K: [H, n_kv] → [hd, nh, n_kv] → permute → [hd, n_kv, nh]
    K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, K, hd, n_heads, n_kv), 0, 2, 1, 3));
    // V: [H, n_kv] → [hd, nh, n_kv] → permute → [hd, n_kv, nh]
    V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, V, hd, n_heads, n_kv), 0, 2, 1, 3));

    // flash_attn_ext: Q [hd,1,nh], K [hd,n_kv,nh], V [hd,n_kv,nh] → output [hd,1,nh]
    ggml_tensor* attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3)); // [hd, nh, 1]
    return ggml_reshape_2d(g, attn, H, 1);
}

// ---------------------------------------------------------------------------
// Encoder graph
// ---------------------------------------------------------------------------

static ggml_cgraph* build_encoder_graph(math_ocr_context* ctx, ggml_context* g, int T) {
    const auto& hp = ctx->hparams;
    const int H = hp.enc_hidden;

    ggml_cgraph* gf = ggml_new_graph_custom(g, hp.enc_layers * 60 + 512, false);

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
// Cross-attention K/V precomputation helper
// (shared by single-image recognize and batch decode paths)
// ---------------------------------------------------------------------------

static void precompute_cross_kv(math_ocr_context* ctx) {
    const int n_enc = ctx->n_enc_tokens;
    const int D = ctx->hparams.dec_d_model;
    const int E = ctx->hparams.enc_hidden;
    const int n_dec = ctx->hparams.dec_layers;

    ctx->cross_k_cache.resize(n_dec);
    ctx->cross_v_cache.resize(n_dec);

    size_t meta_sz = 64 * 1024 * 1024;
    std::vector<uint8_t> meta(meta_sz);
    ggml_init_params ip = { meta_sz, meta.data(), true };
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, n_dec * 6 + 16, false);

    ggml_tensor* enc_inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, E, n_enc);
    ggml_set_name(enc_inp, "enc_for_cross");
    ggml_set_input(enc_inp);

    for (int li = 0; li < n_dec; li++) {
        const auto& l = ctx->dec_layers[li];
        char name[64];

        ggml_tensor* k = ggml_mul_mat(g, l.cross_k_w, enc_inp);
        if (l.cross_k_b) k = ggml_add(g, k, ensure_f32(g, l.cross_k_b));
        snprintf(name, sizeof(name), "cross_k_%d", li);
        ggml_set_name(k, name);
        ggml_set_output(k);

        ggml_tensor* v = ggml_mul_mat(g, l.cross_v_w, enc_inp);
        if (l.cross_v_b) v = ggml_add(g, v, ensure_f32(g, l.cross_v_b));
        snprintf(name, sizeof(name), "cross_v_%d", li);
        ggml_set_name(v, name);
        ggml_set_output(v);

        ggml_build_forward_expand(gf, k);
        ggml_build_forward_expand(gf, v);
    }

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "math_ocr: cross K/V alloc failed\n");
        ggml_free(g);
        return;
    }

    ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "enc_for_cross");
    if (inp_t) ggml_backend_tensor_set(inp_t, ctx->enc_out.data(), 0, n_enc * E * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    for (int li = 0; li < n_dec; li++) {
        ctx->cross_k_cache[li].resize(n_enc * D);
        ctx->cross_v_cache[li].resize(n_enc * D);
        char name[64];
        snprintf(name, sizeof(name), "cross_k_%d", li);
        ggml_tensor* kt = ggml_graph_get_tensor(gf, name);
        if (kt) ggml_backend_tensor_get(kt, ctx->cross_k_cache[li].data(), 0, n_enc * D * sizeof(float));
        snprintf(name, sizeof(name), "cross_v_%d", li);
        ggml_tensor* vt = ggml_graph_get_tensor(gf, name);
        if (vt) ggml_backend_tensor_get(vt, ctx->cross_v_cache[li].data(), 0, n_enc * D * sizeof(float));
    }
    ggml_free(g);
}

// ---------------------------------------------------------------------------
// Run encoder
// ---------------------------------------------------------------------------

static void run_encoder(math_ocr_context* ctx, const float* pixels_rgb, int img_w, int img_h) {
    const auto& hp = ctx->hparams;
    const int P = hp.patch_size, H = hp.enc_hidden;
    const int npw = img_w / P, nph = img_h / P;
    const int n_patches = npw * nph;
    // DeiT has CLS + distillation = +2; ViT has only CLS = +1
    const int n_special = (ctx->dist_token != nullptr) ? 2 : 1;
    const int T = n_patches + n_special;

    // Step 1: CPU-side patch embedding (conv projection)
    // This is a small op — keep on CPU, feed result to graph as input
    std::vector<float> embedded(T * H, 0.0f);

    // Derive channel count from patch embedding weight: shape [H, C*P*P]
    const int patch_weight_dim = ctx->patch_proj_w
        ? (int)(ggml_nelements(ctx->patch_proj_w) / H) : 3 * P * P;
    const int n_channels = patch_weight_dim / (P * P);

    if (ctx->patch_proj_w) {
        auto W_buf = to_f32(ctx->patch_proj_w);
        auto B_buf = to_f32(ctx->patch_proj_b);
        const float* W = W_buf.data();
        const float* B = B_buf.empty() ? nullptr : B_buf.data();
        for (int py = 0; py < nph; py++) {
            for (int px = 0; px < npw; px++) {
                // ggml stores in column-major: tensor[H, T] → data[t * H + h]
                int t = py * npw + px + n_special;
                for (int h = 0; h < H; h++) {
                    float sum = B ? B[h] : 0.0f;
                    for (int c = 0; c < n_channels; c++)
                        for (int dy = 0; dy < P; dy++)
                            for (int dx = 0; dx < P; dx++) {
                                int sy = py * P + dy, sx = px * P + dx;
                                float v = pixels_rgb[(c * img_h + sy) * img_w + sx];
                                sum += v * W[h * patch_weight_dim + c * P * P + dy * P + dx];
                            }
                    embedded[t * H + h] = sum;
                }
            }
        }
    }

    // CLS + distillation tokens (dequantize for FP16/quantized models)
    if (ctx->cls_token) {
        auto d = to_f32(ctx->cls_token);
        for (int h = 0; h < H && h < (int)d.size(); h++) embedded[0 * H + h] = d[h];
    }
    if (ctx->dist_token) {
        auto d = to_f32(ctx->dist_token);
        for (int h = 0; h < H && h < (int)d.size(); h++) embedded[1 * H + h] = d[h];
    }

    // Positional embeddings
    if (ctx->pos_embed) {
        auto pe = to_f32(ctx->pos_embed);
        int n = std::min(T * H, (int)pe.size());
        for (int i = 0; i < n; i++) embedded[i] += pe[i];
    }

    // Step 2: Build (or reuse) ggml graph for transformer layers
    if (ctx->enc_graph_T != T) {
        // Free old cached graph if T changed (different image size)
        if (ctx->enc_graph_g) { ggml_free(ctx->enc_graph_g); ctx->enc_graph_g = nullptr; }
        ctx->enc_graph = nullptr;

        size_t meta_size = 16 * 1024 * 1024;
        std::vector<uint8_t> meta(meta_size);
        ggml_init_params ip = { meta_size, meta.data(), true };
        ggml_context* g = ggml_init(ip);
        ctx->enc_graph = build_encoder_graph(ctx, g, T);
        ctx->enc_graph_g = g;
        ctx->enc_graph_T = T;

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, ctx->enc_graph)) {
            fprintf(stderr, "math_ocr: encoder alloc failed\n");
            ggml_free(ctx->enc_graph_g);
            ctx->enc_graph_g = nullptr; ctx->enc_graph = nullptr; ctx->enc_graph_T = -1;
            return;
        }
    } else {
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, ctx->enc_graph)) {
            fprintf(stderr, "math_ocr: encoder realloc failed\n");
            return;
        }
    }

    ggml_tensor* inp = ggml_graph_get_tensor(ctx->enc_graph, "enc_input");
    if (inp) ggml_backend_tensor_set(inp, embedded.data(), 0, T * H * sizeof(float));

    ggml_backend_sched_graph_compute(ctx->sched, ctx->enc_graph);

    ggml_tensor* out = ggml_graph_get_tensor(ctx->enc_graph, "enc_output");
    ctx->enc_out.resize(T * H);
    if (out) ggml_backend_tensor_get(out, ctx->enc_out.data(), 0, T * H * sizeof(float));

    ctx->n_enc_tokens = T;
}

// ---------------------------------------------------------------------------
// Decoder (scalar — fast enough for autoregressive single-token steps)
// ---------------------------------------------------------------------------

// Per-context dequant cache (was global static — thread-unsafe + leaked across contexts)
static const float* cached_f32(math_ocr_context * ctx, const ggml_tensor* t) {
    return ctx->dcache.get(t);
}

static void layernorm_cpu(math_ocr_context* ctx, const float* x, float* y, int D, const ggml_tensor* w, const ggml_tensor* b) {
    if (!w || !b) { if (x != y) memcpy(y, x, D * sizeof(float)); return; }
    const float* W = cached_f32(ctx, w);
    const float* B = cached_f32(ctx, b);
    float mean = 0, var = 0;
    for (int i = 0; i < D; i++) mean += x[i];
    mean /= D;
    for (int i = 0; i < D; i++) { float d = x[i] - mean; var += d * d; }
    float inv = 1.0f / sqrtf(var / D + 1e-6f);
    for (int i = 0; i < D; i++) y[i] = (x[i] - mean) * inv * W[i] + B[i];
}

static void linear_cpu(math_ocr_context* ctx, const float* inp, float* out, int in_d, int out_d, const ggml_tensor* w, const ggml_tensor* b) {
    if (!w) { memset(out, 0, out_d * sizeof(float)); return; }
    const float* W = cached_f32(ctx, w);
    const float* B = cached_f32(ctx, b);
    core_cpu::linear_cpu(inp, out, in_d, out_d, W, B);
}

// MHA with pre-allocated scores buffer (avoids per-head heap alloc)
static void mha_1q(const float* q, const float* K, const float* V,
                    float* out, int n_kv, int D, int n_heads, float* scores_buf) {
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        float mx = -1e9f;
        for (int k = 0; k < n_kv; k++) {
            scores_buf[k] = core_cpu::dot_product(q + off, &K[k*D + off], hd) * scale;
            if (scores_buf[k] > mx) mx = scores_buf[k];
        }
        float se = 0;
        for (int k = 0; k < n_kv; k++) {
            scores_buf[k] = expf(scores_buf[k] - mx);
            se += scores_buf[k];
        }
        float inv = 1.0f / se;
        for (int k = 0; k < n_kv; k++) scores_buf[k] *= inv;
        for (int d = 0; d < hd; d++) {
            float sum = 0;
            for (int k = 0; k < n_kv; k++) sum += scores_buf[k] * V[k*D + off + d];
            out[off + d] = sum;
        }
    }
}

static void ensure_dec_scratch(math_ocr_context* ctx, int max_kv) {
    if (ctx->ds.allocated) return;
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model, V = hp.vocab_size, FF = hp.dec_ffn_dim;
    int n_enc = ctx->n_enc_tokens;

    ctx->ds.x.resize(D);
    ctx->ds.q.resize(D);
    ctx->ds.k.resize(D);
    ctx->ds.v.resize(D);
    ctx->ds.sa.resize(D);
    ctx->ds.sa_proj.resize(D);
    ctx->ds.cq.resize(D);
    ctx->ds.ca.resize(D);
    ctx->ds.ca_proj.resize(D);
    ctx->ds.inter.resize(FF);
    ctx->ds.ffn.resize(D);
    ctx->ds.logits.resize(V);
    ctx->ds.scores.resize(std::max(max_kv, n_enc));
    ctx->ds.allocated = true;
}

static std::vector<float> decoder_step_scalar(math_ocr_context* ctx,
                                         int tok, int step,
                                         std::vector<std::vector<float>>& kv_k,
                                         std::vector<std::vector<float>>& kv_v) {
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model, V = hp.vocab_size;
    auto& ds = ctx->ds;

    memset(ds.x.data(), 0, D * sizeof(float));

    // Token + positional embedding
    if (ctx->tok_embed && tok >= 0 && tok < V) {
        auto emb = to_f32(ctx->tok_embed);
        float sc = sqrtf((float)D);
        for (int i = 0; i < D; i++) ds.x[i] = emb[tok * D + i] * sc;
    }
    if (ctx->pos_embed_dec) {
        auto pe = to_f32(ctx->pos_embed_dec);
        int pos = step + 2;
        if (pos < hp.max_seq_len)
            for (int i = 0; i < D && pos * D + i < (int)pe.size(); i++)
                ds.x[i] += pe[pos * D + i];
    }
    layernorm_cpu(ctx, ds.x.data(), ds.x.data(), D, ctx->dec_embed_ln_w, ctx->dec_embed_ln_b);
    if (step == 0) {
        fprintf(stderr, "math_ocr: dec embed+pos+ln x[:5]=[%.4f %.4f %.4f %.4f %.4f]\n",
                ds.x[0],ds.x[1],ds.x[2],ds.x[3],ds.x[4]);
    }

    for (int li = 0; li < hp.dec_layers; li++) {
        const auto& l = ctx->dec_layers[li];
        if (!l.self_q_w) continue;

        // Self-attention
        linear_cpu(ctx, ds.x.data(), ds.q.data(), D, D, l.self_q_w, l.self_q_b);
        linear_cpu(ctx, ds.x.data(), ds.k.data(), D, D, l.self_k_w, l.self_k_b);
        linear_cpu(ctx, ds.x.data(), ds.v.data(), D, D, l.self_v_w, l.self_v_b);
        kv_k[li].insert(kv_k[li].end(), ds.k.begin(), ds.k.end());
        kv_v[li].insert(kv_v[li].end(), ds.v.begin(), ds.v.end());

        mha_1q(ds.q.data(), kv_k[li].data(), kv_v[li].data(), ds.sa.data(),
               step+1, D, hp.dec_heads, ds.scores.data());
        linear_cpu(ctx, ds.sa.data(), ds.sa_proj.data(), D, D, l.self_out_w, l.self_out_b);
        for (int i = 0; i < D; i++) ds.x[i] += ds.sa_proj[i];
        layernorm_cpu(ctx, ds.x.data(), ds.x.data(), D, l.self_ln_w, l.self_ln_b);

        // Cross-attention
        if (l.cross_q_w && !ctx->enc_out.empty()) {
            linear_cpu(ctx, ds.x.data(), ds.cq.data(), D, D, l.cross_q_w, l.cross_q_b);

            int n_enc = ctx->n_enc_tokens;
            const float* ck = ctx->cross_k_cache[li].data();
            const float* cv = ctx->cross_v_cache[li].data();

            mha_1q(ds.cq.data(), ck, cv, ds.ca.data(), n_enc, D, hp.dec_heads, ds.scores.data());
            linear_cpu(ctx, ds.ca.data(), ds.ca_proj.data(), D, D, l.cross_out_w, l.cross_out_b);
            for (int i = 0; i < D; i++) ds.x[i] += ds.ca_proj[i];
            layernorm_cpu(ctx, ds.x.data(), ds.x.data(), D, l.cross_ln_w, l.cross_ln_b);
        }

        // FFN
        if (l.ff_up_w) {
            const int FF = hp.dec_ffn_dim;
            linear_cpu(ctx, ds.x.data(), ds.inter.data(), D, FF, l.ff_up_w, l.ff_up_b);
            for (int i = 0; i < FF; i++) ds.inter[i] = ds.inter[i] > 0 ? ds.inter[i] : 0;
            linear_cpu(ctx, ds.inter.data(), ds.ffn.data(), FF, D, l.ff_down_w, l.ff_down_b);
            for (int i = 0; i < D; i++) ds.x[i] += ds.ffn[i];
            layernorm_cpu(ctx, ds.x.data(), ds.x.data(), D, l.ff_ln_w, l.ff_ln_b);
        }
        if (step == 0 && li < 3) {
            fprintf(stderr, "math_ocr: dec L%d end x[:5]=[%.4f %.4f %.4f %.4f %.4f]\n",
                    li, ds.x[0],ds.x[1],ds.x[2],ds.x[3],ds.x[4]);
        }
    }

    layernorm_cpu(ctx, ds.x.data(), ds.x.data(), D, ctx->dec_final_ln_w, ctx->dec_final_ln_b);
    if (step == 0) {
        fprintf(stderr, "math_ocr: dec final x[:5]=[%.4f %.4f %.4f %.4f %.4f]\n",
                ds.x[0],ds.x[1],ds.x[2],ds.x[3],ds.x[4]);
    }
    std::vector<float> logits(V, 0.0f);
    linear_cpu(ctx, ds.x.data(), logits.data(), D, V, ctx->lm_head_w, ctx->lm_head_b);
    return logits;
}

// ---------------------------------------------------------------------------
// Beam search decoder (scalar — reuses decoder_step_scalar per beam)
// ---------------------------------------------------------------------------

struct MathOcrBeam {
    std::vector<int> tokens;
    float score;
    int prev_token;
    bool finished;
    std::vector<std::vector<float>> kv_k;
    std::vector<std::vector<float>> kv_v;
};

static std::vector<int> run_decoder_beam_scalar(math_ocr_context* ctx, int beam_width) {
    const auto& hp = ctx->hparams;
    const int V = hp.vocab_size;
    const int n_dec = hp.dec_layers;
    const int max_steps = std::min(hp.max_seq_len, 200);

    ctx->ds.allocated = false;
    ensure_dec_scratch(ctx, hp.max_seq_len);

    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, 1);

    std::vector<MathOcrBeam> beams(1);
    beams[0].score = 0.0f;
    beams[0].prev_token = hp.decoder_start_token;
    beams[0].finished = false;
    beams[0].kv_k.resize(n_dec);
    beams[0].kv_v.resize(n_dec);

    std::vector<MathOcrBeam> completed;

    for (int step = 0; step < max_steps; step++) {
        struct Cand { int bi; int tok; float score; };
        std::vector<Cand> cands;

        for (int bi = 0; bi < (int)beams.size(); bi++) {
            if (beams[bi].finished) continue;
            auto logits = decoder_step_scalar(ctx, beams[bi].prev_token, step,
                                              beams[bi].kv_k, beams[bi].kv_v);
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_e = 0.0f;
            for (int v = 0; v < V; v++) { logits[v] = expf(logits[v] - max_l); sum_e += logits[v]; }
            float log_sum = logf(sum_e) + max_l;
            for (int v = 0; v < V; v++) {
                float log_p = logits[v] > 0 ? logf(logits[v]) + max_l - log_sum : -100.0f;
                cands.push_back({bi, v, beams[bi].score + log_p});
            }
        }

        if (cands.empty()) break;

        int keep = std::min(beam_width, (int)cands.size());
        std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                          [](const Cand& a, const Cand& b) { return a.score > b.score; });

        std::vector<MathOcrBeam> new_beams;
        new_beams.reserve(keep);
        for (int i = 0; i < keep; i++) {
            const auto& c = cands[i];
            MathOcrBeam nb;
            nb.tokens   = beams[c.bi].tokens;
            nb.kv_k     = beams[c.bi].kv_k;
            nb.kv_v     = beams[c.bi].kv_v;
            nb.score    = c.score;
            if (c.tok == hp.eos_token || c.tok == hp.pad_token) {
                nb.finished = true;
                nb.prev_token = c.tok;
                float len = (float)nb.tokens.size();
                nb.score /= len > 0 ? len : 1.0f;
                completed.push_back(std::move(nb));
            } else {
                nb.tokens.push_back(c.tok);
                nb.prev_token = c.tok;
                nb.finished = false;
                new_beams.push_back(std::move(nb));
            }
        }

        beams = std::move(new_beams);
        if (beams.empty()) break;
    }

    for (auto& b : beams) {
        float len = (float)b.tokens.size();
        b.score /= len > 0 ? len : 1.0f;
        completed.push_back(std::move(b));
    }

    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (completed.empty()) return {};
    auto best = std::max_element(completed.begin(), completed.end(),
                                  [](const MathOcrBeam& a, const MathOcrBeam& b) { return a.score < b.score; });
    return best->tokens;
}

// ---------------------------------------------------------------------------
// Graph-based decoder (one step at a time, with external KV cache)
// ---------------------------------------------------------------------------

// Build a ggml graph for one autoregressive decoder step.
// Inputs (named tensors, set externally):
//   "dec_tok_emb"   — [D, 1] token+pos embedding (pre-computed on CPU)
//   "self_k_in_L"   — [D, n_kv] self-attn K cache for layer L (0..n_dec-1)
//   "self_v_in_L"   — [D, n_kv] self-attn V cache for layer L
//   "cross_k_L"     — [D, n_enc] cross-attn K for layer L
//   "cross_v_L"     — [D, n_enc] cross-attn V for layer L
// Outputs:
//   "logits"        — [V, 1] logits for next token
//   "self_k_out_L"  — [D, 1] new self-attn K for layer L (to append to cache)
//   "self_v_out_L"  — [D, 1] new self-attn V for layer L

static ggml_cgraph* build_decoder_step_graph(math_ocr_context* ctx, ggml_context* g,
                                              int n_kv, int n_enc) {
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model;
    const int V = hp.vocab_size;
    const int n_dec = hp.dec_layers;

    // Generous node budget: per layer ~40 ops, plus embeddings + final
    ggml_cgraph* gf = ggml_new_graph_custom(g, n_dec * 100 + 512, false);

    // Input: pre-computed token embedding (tok + pos + embed_ln), shape [D, 1]
    ggml_tensor* cur = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, 1);
    ggml_set_name(cur, "dec_tok_emb");
    ggml_set_input(cur);

    char name[64];

    for (int li = 0; li < n_dec; li++) {
        const auto& l = ctx->dec_layers[li];
        if (!l.self_q_w) continue;

        ggml_tensor* residual = cur;

        // ---- Self-attention ----
        // Project Q, K, V from current hidden state [D, 1]
        ggml_tensor* q = g_linear(g, cur, l.self_q_w, l.self_q_b);  // [D, 1]
        ggml_tensor* k_new = g_linear(g, cur, l.self_k_w, l.self_k_b);  // [D, 1]
        ggml_tensor* v_new = g_linear(g, cur, l.self_v_w, l.self_v_b);  // [D, 1]

        // Output the new K/V so we can append to cache externally
        snprintf(name, sizeof(name), "self_k_out_%d", li);
        ggml_set_name(k_new, name);
        ggml_set_output(k_new);
        snprintf(name, sizeof(name), "self_v_out_%d", li);
        ggml_set_name(v_new, name);
        ggml_set_output(v_new);

        // Build K_full / V_full: either just the new token (step 0) or
        // concatenation of cache + new token
        ggml_tensor* k_full;
        ggml_tensor* v_full;
        if (n_kv > 0) {
            // Input: accumulated K/V cache [D, n_kv] (previous tokens)
            ggml_tensor* k_cache = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_kv);
            snprintf(name, sizeof(name), "self_k_in_%d", li);
            ggml_set_name(k_cache, name);
            ggml_set_input(k_cache);

            ggml_tensor* v_cache = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_kv);
            snprintf(name, sizeof(name), "self_v_in_%d", li);
            ggml_set_name(v_cache, name);
            ggml_set_input(v_cache);

            // Concatenate: K_full = [k_cache ; k_new] → [D, n_kv+1]
            k_full = ggml_concat(g, k_cache, k_new, 1);
            v_full = ggml_concat(g, v_cache, v_new, 1);
        } else {
            // First step: no cache, just use the new token
            k_full = k_new;  // [D, 1]
            v_full = v_new;
        }

        // Single-query MHA: Q [D,1], K [D, n_kv+1], V [D, n_kv+1]
        ggml_tensor* sa = g_mha_1q(g, q, k_full, v_full, hp.dec_heads, n_kv + 1);
        sa = g_linear(g, sa, l.self_out_w, l.self_out_b);  // [D, 1]

        // Residual + post-LN
        cur = ggml_add(g, residual, sa);
        cur = g_ln(g, cur, l.self_ln_w, l.self_ln_b);

        // ---- Cross-attention ----
        if (l.cross_q_w) {
            residual = cur;

            ggml_tensor* cq = g_linear(g, cur, l.cross_q_w, l.cross_q_b);  // [D, 1]

            // Cross K/V are pre-computed and fixed
            ggml_tensor* ck = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_enc);
            snprintf(name, sizeof(name), "cross_k_%d", li);
            ggml_set_name(ck, name);
            ggml_set_input(ck);

            ggml_tensor* cv = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_enc);
            snprintf(name, sizeof(name), "cross_v_%d", li);
            ggml_set_name(cv, name);
            ggml_set_input(cv);

            ggml_tensor* ca = g_mha_1q(g, cq, ck, cv, hp.dec_heads, n_enc);
            ca = g_linear(g, ca, l.cross_out_w, l.cross_out_b);

            // Residual + post-LN
            cur = ggml_add(g, residual, ca);
            cur = g_ln(g, cur, l.cross_ln_w, l.cross_ln_b);
        }

        // ---- FFN (GELU) ----
        if (l.ff_up_w) {
            residual = cur;
            ggml_tensor* up = g_linear(g, cur, l.ff_up_w, l.ff_up_b);
            up = ggml_gelu(g, up);
            ggml_tensor* down = g_linear(g, up, l.ff_down_w, l.ff_down_b);

            // Residual + post-LN
            cur = ggml_add(g, residual, down);
            cur = g_ln(g, cur, l.ff_ln_w, l.ff_ln_b);
        }
    }

    // Final LayerNorm
    if (ctx->dec_final_ln_w)
        cur = g_ln(g, cur, ctx->dec_final_ln_w, ctx->dec_final_ln_b);

    // LM head → logits [V, 1]
    cur = g_linear(g, cur, ctx->lm_head_w, ctx->lm_head_b);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Cosine similarity between two float vectors
static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

// Run the full decoder loop using graph-based computation.
// Returns the generated tokens (excluding the start token).
static std::vector<int> run_decoder_graph(math_ocr_context* ctx) {
    ctx->char_confidences.clear();
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model;
    const int V = hp.vocab_size;
    const int n_dec = hp.dec_layers;
    const int n_enc = ctx->n_enc_tokens;

    // --- Optimization 1: Pre-cache dequantized embeddings ---
    // Dequantize once instead of every step (avoids full vocab table dequant per step)
    std::vector<float> tok_emb_f32;
    std::vector<float> pos_emb_f32;
    if (ctx->tok_embed)    tok_emb_f32 = to_f32(ctx->tok_embed);
    if (ctx->pos_embed_dec) pos_emb_f32 = to_f32(ctx->pos_embed_dec);

    // --- Optimization: Use 1 thread for decoder (small tensors, threading overhead dominates) ---
    // The decoder operates on [D,1] vectors where D=256, so matrix ops are tiny.
    // Multi-threading adds massive synchronization overhead for these small ops.
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, 1);

    // --- Optimization 2: Single metadata buffer (16MB, reused) ---
    const size_t meta_sz = 16 * 1024 * 1024;
    std::vector<uint8_t> meta_buf(meta_sz);
    ggml_init_params ip = { meta_sz, meta_buf.data(), true };

    // Self-attention KV cache: per layer, grows by [D] each step
    std::vector<std::vector<float>> self_k_cache(n_dec);
    std::vector<std::vector<float>> self_v_cache(n_dec);

    // --- Validation harness (scalar decoder for comparison) ---
#ifdef DECODER_VALIDATE
    std::vector<std::vector<float>> scalar_k_cache(n_dec);
    std::vector<std::vector<float>> scalar_v_cache(n_dec);
    std::vector<int> scalar_tokens = {hp.decoder_start_token};
    float min_cos = 1.0f;
    fprintf(stderr, "math_ocr: DECODER_VALIDATE enabled — comparing graph vs scalar\n");
    ctx->ds.allocated = false;
    ensure_dec_scratch(ctx, hp.max_seq_len);
#endif

    std::vector<int> tokens = {hp.decoder_start_token};
    int max_steps = std::min(hp.max_seq_len, 200);

    for (int step = 0; step < max_steps; step++) {
        int tok = tokens.back();
        int n_kv = step;  // number of previously cached K/V tokens (0 on first step)

        // 1. CPU-side: compute token + positional embedding + embed LN
        //    Uses pre-cached dequantized tables (Optimization 1)
        std::vector<float> emb(D, 0.0f);
        if (!tok_emb_f32.empty() && tok >= 0 && tok < V) {
            float sc = ctx->scale_embedding ? sqrtf((float)D) : 1.0f;
            for (int i = 0; i < D; i++) emb[i] = tok_emb_f32[tok * D + i] * sc;
        }
        if (!pos_emb_f32.empty()) {
            int pos = step + 2;
            if (pos < hp.max_seq_len)
                for (int i = 0; i < D && pos * D + i < (int)pos_emb_f32.size(); i++)
                    emb[i] += pos_emb_f32[pos * D + i];
        }
        layernorm_cpu(ctx, emb.data(), emb.data(), D, ctx->dec_embed_ln_w, ctx->dec_embed_ln_b);

        if (step == 0) {
            fprintf(stderr, "math_ocr: [graph] dec embed+pos+ln x[:5]=[%.4f %.4f %.4f %.4f %.4f]\n",
                    emb[0], emb[1], emb[2], emb[3], emb[4]);
        }

        // 2. Build the graph for this step (reuse metadata buffer — Optimization 2)
        ggml_context* g = ggml_init(ip);

        ggml_cgraph* gf = build_decoder_step_graph(ctx, g, n_kv, n_enc);

        // 3. Allocate graph
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "math_ocr: decoder graph alloc failed at step %d\n", step);
            ggml_free(g);
            break;
        }

        // 4. Set input tensors

        // Token embedding
        ggml_tensor* t_emb = ggml_graph_get_tensor(gf, "dec_tok_emb");
        if (t_emb) ggml_backend_tensor_set(t_emb, emb.data(), 0, D * sizeof(float));

        // Self-attention KV caches
        char name[64];
        for (int li = 0; li < n_dec; li++) {
            if (n_kv > 0) {
                snprintf(name, sizeof(name), "self_k_in_%d", li);
                ggml_tensor* kt = ggml_graph_get_tensor(gf, name);
                if (kt) ggml_backend_tensor_set(kt, self_k_cache[li].data(), 0,
                                                  n_kv * D * sizeof(float));

                snprintf(name, sizeof(name), "self_v_in_%d", li);
                ggml_tensor* vt = ggml_graph_get_tensor(gf, name);
                if (vt) ggml_backend_tensor_set(vt, self_v_cache[li].data(), 0,
                                                  n_kv * D * sizeof(float));
            }

            // Cross-attention K/V (constant, but must re-set after sched realloc)
            snprintf(name, sizeof(name), "cross_k_%d", li);
            ggml_tensor* ck = ggml_graph_get_tensor(gf, name);
            if (ck) ggml_backend_tensor_set(ck, ctx->cross_k_cache[li].data(), 0,
                                              n_enc * D * sizeof(float));

            snprintf(name, sizeof(name), "cross_v_%d", li);
            ggml_tensor* cv = ggml_graph_get_tensor(gf, name);
            if (cv) ggml_backend_tensor_set(cv, ctx->cross_v_cache[li].data(), 0,
                                              n_enc * D * sizeof(float));
        }

        // 5. Compute
        ggml_backend_sched_graph_compute(ctx->sched, gf);

        // 6. Read logits
        std::vector<float> logits(V);
        ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
        if (logits_t) ggml_backend_tensor_get(logits_t, logits.data(), 0, V * sizeof(float));

        // 7. Read new K/V and append to cache
        for (int li = 0; li < n_dec; li++) {
            std::vector<float> k_new(D), v_new(D);
            snprintf(name, sizeof(name), "self_k_out_%d", li);
            ggml_tensor* ko = ggml_graph_get_tensor(gf, name);
            if (ko) ggml_backend_tensor_get(ko, k_new.data(), 0, D * sizeof(float));

            snprintf(name, sizeof(name), "self_v_out_%d", li);
            ggml_tensor* vo = ggml_graph_get_tensor(gf, name);
            if (vo) ggml_backend_tensor_get(vo, v_new.data(), 0, D * sizeof(float));

            self_k_cache[li].insert(self_k_cache[li].end(), k_new.begin(), k_new.end());
            self_v_cache[li].insert(self_v_cache[li].end(), v_new.begin(), v_new.end());
        }

        ggml_free(g);

        // --- Validation: compare graph logits with scalar decoder ---
#ifdef DECODER_VALIDATE
        {
            int scalar_tok = scalar_tokens.back();
            auto scalar_logits = decoder_step_scalar(ctx, scalar_tok, step,
                                                      scalar_k_cache, scalar_v_cache);
            float cs = cosine_sim(logits.data(), scalar_logits.data(), V);
            if (cs < min_cos) min_cos = cs;
            if (step < 10 || cs < 0.99f) {
                int g_best = 0, s_best = 0;
                for (int v = 1; v < V; v++) {
                    if (logits[v] > logits[g_best]) g_best = v;
                    if (scalar_logits[v] > scalar_logits[s_best]) s_best = v;
                }
                fprintf(stderr, "math_ocr: [validate] step %d cosine=%.6f graph_best=%d scalar_best=%d\n",
                        step, cs, g_best, s_best);
            }
        }
#endif

        // 8. Greedy argmax
        int best = 0;
        float best_s = logits[0];
        for (int v = 1; v < V; v++)
            if (logits[v] > best_s) { best_s = logits[v]; best = v; }

        if (step < 5) {
            fprintf(stderr, "math_ocr: [graph] dec step %d: tok=%d logits[0..4]=[%.3f %.3f %.3f %.3f %.3f] best=%d\n",
                    step, tok, logits[0], logits[1], logits[2], logits[3], logits[4], best);
        }

        if (best == hp.eos_token || best == hp.pad_token) break;

        // Confidence: softmax of winning token
        {
            float max_l = logits[0];
            for (int v = 1; v < V; v++) if (logits[v] > max_l) max_l = logits[v];
            float sum_e = 0;
            for (int v = 0; v < V; v++) sum_e += expf(logits[v] - max_l);
            ctx->char_confidences.push_back(expf(logits[best] - max_l) / sum_e);
        }

        tokens.push_back(best);

#ifdef DECODER_VALIDATE
        // Mirror the scalar decoder's token sequence to match graph decoder
        scalar_tokens.push_back(best);
#endif
    }

#ifdef DECODER_VALIDATE
    fprintf(stderr, "math_ocr: [validate] DONE — min cosine similarity across all steps: %.6f %s\n",
            min_cos, min_cos >= 0.99f ? "PASS" : "FAIL");
#endif

    // Restore multi-threading for subsequent encoder runs
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    return tokens;
}

// ---------------------------------------------------------------------------
// Init / Free / API
// ---------------------------------------------------------------------------

math_ocr_context* math_ocr_init(const char* model_path, int n_threads) {
    auto ctx = std::make_unique<math_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 1;

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
    {
        int idx = gguf_find_key(gctx, "decoder.scale_embedding");
        ctx->scale_embedding = (idx < 0) ? true : gguf_get_val_bool(gctx, idx);
    }
    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "math_ocr: enc=%dL/%dH/%d dec=%dL/%dH/%d vocab=%d(%zu)\n",
            hp.enc_layers, hp.enc_heads, hp.enc_hidden,
            hp.dec_layers, hp.dec_heads, hp.dec_d_model, hp.vocab_size, ctx->vocab.size());

    // Init backend — prefer GPU when available
    fprintf(stderr, "math_ocr: init backend...\n");
    bool force_cpu = (getenv("MATH_OCR_FORCE_CPU") && atoi(getenv("MATH_OCR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    fprintf(stderr, "math_ocr: loading weights...\n");
    if (!core_gguf::load_weights(model_path, ctx->backend, "math_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        return nullptr;
    }
    fprintf(stderr, "math_ocr: weights loaded (%zu tensors)\n", ctx->wl.tensors.size());

    // Create scheduler — GPU + CPU fallback
    fprintf(stderr, "math_ocr: creating scheduler...\n");
    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx->backend);
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr,
                                        (int)backends.size(), 8192, false, false);
    fprintf(stderr, "math_ocr: scheduler created\n");

    map_tensors(ctx.get());
    fprintf(stderr, "math_ocr: tensors mapped\n");

    int me = 0, md = 0;
    for (int i = 0; i < hp.enc_layers; i++) if (ctx->enc_layers[i].q_w) me++;
    for (int i = 0; i < hp.dec_layers; i++) if (ctx->dec_layers[i].self_q_w) md++;
    fprintf(stderr, "math_ocr: mapped %d/%d enc, %d/%d dec\n", me, hp.enc_layers, md, hp.dec_layers);

    ctx->bench = (std::getenv("CRISPEMBED_MATH_OCR_BENCH") != nullptr);

    fprintf(stderr, "math_ocr: init complete, returning context\n");
    return ctx.release();
}

void math_ocr_free(math_ocr_context* ctx) {
    if (!ctx) return;
    if (ctx->enc_graph_g) ggml_free(ctx->enc_graph_g);
    if (ctx->enc_batch_g) ggml_free(ctx->enc_batch_g);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
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

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Derive channel count from patch embedding weight
    const int n_channels = ctx->patch_proj_w
        ? (int)(ggml_nelements(ctx->patch_proj_w) / ctx->hparams.enc_hidden)
              / (ctx->hparams.patch_size * ctx->hparams.patch_size)
        : 3;
    fprintf(stderr, "math_ocr: image_size S=%d, channels=%d, allocating buf(%d)\n",
            S, n_channels, n_channels*S*S);
    // Resize + expand gray→CHW + normalize (mean=0.5, std=0.5)
    auto tb0 = std::chrono::steady_clock::now();
    std::vector<float> rgb(n_channels * S * S);
    fprintf(stderr, "math_ocr: buf allocated\n");
    float fsx = (float)width / S, fsy = (float)height / S;
    for (int y = 0; y < S; y++) {
        float fy = y * fsy;
        int y0 = (int)fy, y1 = std::min(y0 + 1, height - 1);
        float wy = fy - y0;
        for (int x = 0; x < S; x++) {
            float fx = x * fsx;
            int x0 = (int)fx, x1 = std::min(x0 + 1, width - 1);
            float wx = fx - x0;
            float v = (1 - wy) * ((1 - wx) * pixels[y0*width+x0] + wx * pixels[y0*width+x1])
                    +      wy  * ((1 - wx) * pixels[y1*width+x0] + wx * pixels[y1*width+x1]);
            v = (v - 0.5f) / 0.5f;
            for (int c = 0; c < n_channels; c++)
                rgb[c*S*S + y*S + x] = v;
        }
    }
    if (bench) fprintf(stderr, "[math_ocr-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tb0).count());

    // Encoder (ggml graph — fast)
    fprintf(stderr, "math_ocr: about to run encoder S=%d\n", S);
    tb0 = std::chrono::steady_clock::now();
    run_encoder(ctx, rgb.data(), S, S);
    if (bench) fprintf(stderr, "[math_ocr-bench] encoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tb0).count());
    fprintf(stderr, "math_ocr: encoder done, n_enc=%d\n", ctx->n_enc_tokens);

    // Precompute cross-attention K/V via ggml graph (SIMD, fast)
    precompute_cross_kv(ctx);
    fprintf(stderr, "math_ocr: cross K/V cached (%d enc tokens × %d layers)\n",
            ctx->n_enc_tokens, ctx->hparams.dec_layers);

    // Decoder (ggml graph — SIMD-accelerated, with scalar fallback)
    fprintf(stderr, "math_ocr: starting decoder (vocab=%d, start_tok=%d) [graph mode]\n",
            ctx->hparams.vocab_size, ctx->hparams.decoder_start_token);
    const auto& hp = ctx->hparams;

    auto dec_t0 = std::chrono::steady_clock::now();
    std::vector<int> tokens = run_decoder_graph(ctx);
    auto dec_t1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(dec_t1 - dec_t0).count();
    fprintf(stderr, "math_ocr: decoder done in %.1f ms (%zu tokens)\n",
            dec_ms, tokens.size());
    if (bench) fprintf(stderr, "[math_ocr-bench] decoder: %.1f ms\n", dec_ms);

    ctx->result_buf.clear();
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            const auto& piece = ctx->vocab[tok];
            // Replace word-boundary markers with space:
            //   SentencePiece ▁ (U+2581) = 3 bytes: 0xE2 0x96 0x81
            //   GPT-2 BPE Ġ (U+0120) = 2 bytes: 0xC4 0xA0
            for (size_t j = 0; j < piece.size(); ) {
                if (j + 2 < piece.size() &&
                    (uint8_t)piece[j] == 0xE2 &&
                    (uint8_t)piece[j+1] == 0x96 &&
                    (uint8_t)piece[j+2] == 0x81) {
                    ctx->result_buf += ' ';
                    j += 3;
                } else if (j + 1 < piece.size() &&
                           (uint8_t)piece[j] == 0xC4 &&
                           (uint8_t)piece[j+1] == 0xA0) {
                    ctx->result_buf += ' ';
                    j += 2;
                } else {
                    ctx->result_buf += piece[j];
                    j++;
                }
            }
        }
    }
    // Trim leading/trailing whitespace
    while (!ctx->result_buf.empty() && ctx->result_buf.front() == ' ')
        ctx->result_buf.erase(ctx->result_buf.begin());
    while (!ctx->result_buf.empty() && ctx->result_buf.back() == ' ')
        ctx->result_buf.pop_back();

    if (bench) fprintf(stderr, "[math_ocr-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float* math_ocr_get_encoder_output(const math_ocr_context* ctx, int* out_n, int* out_h) {
    if (!ctx || ctx->enc_out.empty()) return nullptr;
    if (out_n) *out_n = ctx->n_enc_tokens;
    if (out_h) *out_h = ctx->hparams.enc_hidden;
    return ctx->enc_out.data();
}

const char* math_ocr_recognize_file(math_ocr_context* ctx, const char* path, int* out_len) {
    if (!ctx || !path) return nullptr;
    int w, h, c;
    unsigned char* img = stbi_load(path, &w, &h, &c, 3);
    if (!img) {
        fprintf(stderr, "math_ocr: cannot load image: %s\n", path);
        return nullptr;
    }
    const char* result = math_ocr_recognize_raw(ctx, img, w, h, 3, out_len);
    stbi_image_free(img);
    return result;
}

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

const char* math_ocr_recognize_beam(math_ocr_context* ctx, const float* pixels,
                                     int width, int height,
                                     int beam_width, int* out_len) {
    if (!ctx || !pixels) return nullptr;
    if (beam_width <= 1)
        return math_ocr_recognize(ctx, pixels, width, height, out_len);

    const int S = ctx->hparams.image_size;

    // Preprocess: resize + gray→3ch + normalize (mean=0.5, std=0.5)
    std::vector<float> rgb(3 * S * S);
    float fsx = (float)width / S, fsy = (float)height / S;
    for (int y = 0; y < S; y++) {
        float fy = y * fsy;
        int y0 = (int)fy, y1 = std::min(y0 + 1, height - 1);
        float wy = fy - y0;
        for (int x = 0; x < S; x++) {
            float fx = x * fsx;
            int x0 = (int)fx, x1 = std::min(x0 + 1, width - 1);
            float wx = fx - x0;
            float v = (1 - wy) * ((1 - wx) * pixels[y0*width+x0] + wx * pixels[y0*width+x1])
                    +      wy  * ((1 - wx) * pixels[y1*width+x0] + wx * pixels[y1*width+x1]);
            v = (v - 0.5f) / 0.5f;
            rgb[0*S*S + y*S + x] = rgb[1*S*S + y*S + x] = rgb[2*S*S + y*S + x] = v;
        }
    }

    run_encoder(ctx, rgb.data(), S, S);

    // Precompute cross-attention K/V
    precompute_cross_kv(ctx);

    std::vector<int> tokens = run_decoder_beam_scalar(ctx, beam_width);

    ctx->result_buf.clear();
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            const auto& piece = ctx->vocab[tok];
            for (size_t j = 0; j < piece.size(); ) {
                if (j + 2 < piece.size() &&
                    (uint8_t)piece[j] == 0xE2 && (uint8_t)piece[j+1] == 0x96 && (uint8_t)piece[j+2] == 0x81) {
                    ctx->result_buf += ' '; j += 3;
                } else if (j + 1 < piece.size() &&
                           (uint8_t)piece[j] == 0xC4 && (uint8_t)piece[j+1] == 0xA0) {
                    ctx->result_buf += ' '; j += 2;
                } else { ctx->result_buf += piece[j]; j++; }
            }
        }
    }
    while (!ctx->result_buf.empty() && ctx->result_buf.front() == ' ')
        ctx->result_buf.erase(ctx->result_buf.begin());
    while (!ctx->result_buf.empty() && ctx->result_buf.back() == ' ')
        ctx->result_buf.pop_back();

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char* math_ocr_recognize_raw_beam(math_ocr_context* ctx, const uint8_t* bytes,
                                          int w, int h, int ch,
                                          int beam_width, int* out_len) {
    if (!ctx || !bytes) return nullptr;
    std::vector<float> gray(w * h);
    for (int i = 0; i < w * h; i++) {
        if (ch == 1) gray[i] = bytes[i] / 255.0f;
        else { int b = i * ch; gray[i] = (0.299f*bytes[b] + 0.587f*bytes[b+1] + 0.114f*bytes[b+2]) / 255.0f; }
    }
    return math_ocr_recognize_beam(ctx, gray.data(), w, h, beam_width, out_len);
}

const float * math_ocr_confidences(const math_ocr_context * ctx, int * n_chars) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float math_ocr_mean_confidence(const math_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}

// ---------------------------------------------------------------------------
// Batch encoder API
// ---------------------------------------------------------------------------

bool math_ocr_encode_batch_raw(
    math_ocr_context* ctx,
    const uint8_t* const* crops,
    const int* widths, const int* heights,
    int n_crops)
{
    if (!ctx || !crops || n_crops <= 0) return false;

    const int S = ctx->hparams.image_size;
    const int P = ctx->hparams.patch_size;
    const int H = ctx->hparams.enc_hidden;
    const int npw = S / P, nph = S / P;
    const int n_patches = npw * nph;
    const int T = n_patches + 2;  // patches + CLS + dist tokens
    const int B = n_crops;

    // --- Step 1: preprocess all crops and compute patch embeddings ---
    // Stacked layout: embedded[b * T * H + t * H + h]  (batch-major)
    std::vector<float> embedded(B * T * H, 0.0f);

    auto cls_f32  = to_f32(ctx->cls_token);
    auto dist_f32 = to_f32(ctx->dist_token);
    auto pe_f32   = to_f32(ctx->pos_embed);

    std::vector<float> rgb(3 * S * S);
    for (int b = 0; b < B; b++) {
        const uint8_t* src = crops[b];
        int sw = widths[b], sh = heights[b];

        // Bilinear resize + gray→3ch CHW + normalize (mean=0.5, std=0.5)
        float fsx = (float)sw / S, fsy = (float)sh / S;
        for (int y = 0; y < S; y++) {
            float fy = y * fsy;
            int y0 = (int)fy, y1 = std::min(y0 + 1, sh - 1);
            float wy = fy - y0;
            for (int x = 0; x < S; x++) {
                float fx = x * fsx;
                int x0 = (int)fx, x1 = std::min(x0 + 1, sw - 1);
                float wx = fx - x0;
                // Assume 3-channel RGB; convert to grayscale-replicated CHW
                int p00 = (y0*sw + x0)*3, p01 = (y0*sw + x1)*3;
                int p10 = (y1*sw + x0)*3, p11 = (y1*sw + x1)*3;
                auto luma = [&](int p) {
                    return (0.299f*src[p] + 0.587f*src[p+1] + 0.114f*src[p+2]) / 255.0f;
                };
                float v = (1-wy)*((1-wx)*luma(p00) + wx*luma(p01))
                        +    wy *((1-wx)*luma(p10) + wx*luma(p11));
                v = (v - 0.5f) / 0.5f;
                rgb[0*S*S + y*S + x] = v;
                rgb[1*S*S + y*S + x] = v;
                rgb[2*S*S + y*S + x] = v;
            }
        }

        float* emb = embedded.data() + b * T * H;

        // CLS + dist tokens
        if (!cls_f32.empty())
            for (int h = 0; h < H && h < (int)cls_f32.size();  h++) emb[0*H+h] = cls_f32[h];
        if (!dist_f32.empty())
            for (int h = 0; h < H && h < (int)dist_f32.size(); h++) emb[1*H+h] = dist_f32[h];

        // Patch embeddings
        if (ctx->patch_proj_w) {
            const int patch_dim = 3 * P * P;
            auto W_buf = to_f32(ctx->patch_proj_w);
            auto B_buf = to_f32(ctx->patch_proj_b);
            const float* W = W_buf.data();
            const float* Bv = B_buf.empty() ? nullptr : B_buf.data();
            for (int py = 0; py < nph; py++) {
                for (int px = 0; px < npw; px++) {
                    int t = py * npw + px + 2;
                    for (int h = 0; h < H; h++) {
                        float sum = Bv ? Bv[h] : 0.0f;
                        for (int c = 0; c < 3; c++)
                            for (int dy = 0; dy < P; dy++)
                                for (int dx = 0; dx < P; dx++) {
                                    int sy = py*P+dy, sx = px*P+dx;
                                    float v = rgb[(c * S + sy) * S + sx];
                                    sum += v * W[h * patch_dim + c*P*P + dy*P + dx];
                                }
                        emb[t*H+h] = sum;
                    }
                }
            }
        }

        // Add positional embeddings
        if (!pe_f32.empty()) {
            int n = std::min(T * H, (int)pe_f32.size());
            for (int i = 0; i < n; i++) emb[i] += pe_f32[i];
        }
    }

    // --- Step 2: build/reuse batched encoder graph ---
    if (ctx->enc_batch_T != T || ctx->enc_batch_B != B) {
        if (ctx->enc_batch_g) { ggml_free(ctx->enc_batch_g); ctx->enc_batch_g = nullptr; }
        ctx->enc_batch = nullptr;

        size_t meta_size = (size_t)(ctx->hparams.enc_layers * 80 + 512) * ggml_tensor_overhead()
                           + ggml_graph_overhead_custom(ctx->hparams.enc_layers * 80 + 512, false)
                           + 4 * 1024 * 1024;
        std::vector<uint8_t> meta(meta_size);
        ggml_init_params ip = { meta_size, meta.data(), true };
        ggml_context* g = ggml_init(ip);
        ctx->enc_batch   = build_encoder_graph_batch(ctx, g, T, B);
        ctx->enc_batch_g = g;
        ctx->enc_batch_T = T;
        ctx->enc_batch_B = B;
    }

    // --- Step 3: run the batch encoder graph ---
    // Input tensor layout for ggml: [H, T, B] — innermost dim H, then T, outermost B.
    // Our `embedded` is [B, T, H] (B-major) — need to transpose to [H, T, B].
    std::vector<float> inp_ggml(H * T * B);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                inp_ggml[h + t*H + b*T*H] = embedded[b*T*H + t*H + h];

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, ctx->enc_batch)) {
        fprintf(stderr, "math_ocr: batch encoder alloc failed\n");
        return false;
    }

    ggml_tensor* inp_t = ggml_graph_get_tensor(ctx->enc_batch, "enc_batch_input");
    if (inp_t)
        ggml_backend_tensor_set(inp_t, inp_ggml.data(), 0, H * T * B * sizeof(float));

    ggml_backend_sched_graph_compute(ctx->sched, ctx->enc_batch);

    // --- Step 4: extract per-crop encoder outputs ---
    ggml_tensor* out_t = ggml_graph_get_tensor(ctx->enc_batch, "enc_batch_output");
    if (!out_t) {
        fprintf(stderr, "math_ocr: batch encoder output tensor missing\n");
        return false;
    }

    // Output shape: [H, T, B] (ggml layout) → read and split
    std::vector<float> out_ggml(H * T * B);
    ggml_backend_tensor_get(out_t, out_ggml.data(), 0, H * T * B * sizeof(float));

    ctx->batch_enc_outs.resize(B);
    for (int b = 0; b < B; b++) {
        ctx->batch_enc_outs[b].resize(T * H);
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                ctx->batch_enc_outs[b][t*H + h] = out_ggml[h + t*H + b*T*H];
    }

    if (ctx->bench)
        fprintf(stderr, "[math_ocr-bench] batch encode: B=%d T=%d H=%d\n", B, T, H);

    return true;
}

const char* math_ocr_decode_batch_crop(math_ocr_context* ctx, int crop_idx, int* out_len)
{
    if (!ctx || crop_idx < 0 || crop_idx >= (int)ctx->batch_enc_outs.size()) {
        if (out_len) *out_len = 0;
        return nullptr;
    }

    // Restore encoder output for this crop
    ctx->enc_out = ctx->batch_enc_outs[crop_idx];
    ctx->n_enc_tokens = (int)(ctx->enc_out.size() / ctx->hparams.enc_hidden);

    // Precompute cross-attention K/V for this crop's encoder output
    precompute_cross_kv(ctx);

    // Run decoder
    std::vector<int> tokens = run_decoder_graph(ctx);

    ctx->result_buf.clear();
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            const auto& piece = ctx->vocab[tok];
            for (size_t j = 0; j < piece.size(); ) {
                if (j + 2 < piece.size() &&
                    (uint8_t)piece[j] == 0xE2 &&
                    (uint8_t)piece[j+1] == 0x96 &&
                    (uint8_t)piece[j+2] == 0x81) {
                    ctx->result_buf += ' '; j += 3;
                } else if (j + 1 < piece.size() &&
                           (uint8_t)piece[j] == 0xC4 &&
                           (uint8_t)piece[j+1] == 0xA0) {
                    ctx->result_buf += ' '; j += 2;
                } else {
                    ctx->result_buf += piece[j]; j++;
                }
            }
        }
    }
    while (!ctx->result_buf.empty() && ctx->result_buf.front() == ' ')
        ctx->result_buf.erase(ctx->result_buf.begin());
    while (!ctx->result_buf.empty() && ctx->result_buf.back() == ' ')
        ctx->result_buf.pop_back();

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}
