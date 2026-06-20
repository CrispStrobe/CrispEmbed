// pix2struct.cpp -- Pix2Struct image-to-text (optimized).
//
// Phase 1: Encoder as ggml graph (SIMD / GPU-ready via ggml_backend_sched).
// Phase 2: Decoder KV cache — incremental single-token decode with cached
//          self-attn K/V and pre-computed cross-attn K/V.
// Phase 3: DequantCache for all remaining CPU-scalar weight access.
//
// Encoder: patch_projection + row/col embeddings → 12 T5-style layers
//   (Pre-RMSNorm → QKVO self-attn → Pre-RMSNorm → GeGLU FFN)
//   No relative attention bias in encoder; position from row/col embeddings.
//
// Decoder: token embed → 12 T5-style layers
//   (Pre-RMSNorm → causal self-attn + T5 relative bias →
//    Pre-RMSNorm → cross-attn → Pre-RMSNorm → GeGLU FFN)
//   → final norm → LM head → greedy decode.

#include "pix2struct.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── T5 relative position bias ──

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
    ggml_tensor * wi_0, * wi_1, * wo; // GeGLU
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
    // Weight storage
    core_gguf::WeightLoad wl;

    // ggml backend (CPU, kept alive for graph compute)
    ggml_backend_t backend;

    // Encoder scheduler (reusable)
    ggml_backend_sched_t enc_sched;

    int enc_layers, dec_layers, hidden, n_heads, d_kv, d_ff;
    int vocab_size, patch_size, max_patches;
    int rel_buckets, rel_max_dist;
    float rms_eps;

    // Encoder weights
    ggml_tensor * patch_proj_w, * patch_proj_b;
    ggml_tensor * row_emb, * col_emb;
    std::vector<enc_layer_wt> enc;
    ggml_tensor * enc_final_norm;

    // Decoder weights
    ggml_tensor * tok_emb;
    std::vector<dec_layer_wt> dec;
    ggml_tensor * final_norm;
    ggml_tensor * lm_head;

    // Tokenizer
    int eos_id, pad_id;

    // Cached encoder output [n_patches, hidden]
    std::vector<float> enc_cache;
    int enc_cache_n;

    // Pre-computed cross-attn K/V per decoder layer [n_patches, qkv_dim]
    std::vector<std::vector<float>> cross_k_cache;
    std::vector<std::vector<float>> cross_v_cache;

    // Self-attention KV cache per decoder layer [max_seq, qkv_dim]
    std::vector<std::vector<float>> sa_k_cache;
    std::vector<std::vector<float>> sa_v_cache;
    int sa_cache_len; // number of cached positions

    // Dequantization cache (avoids re-dequantizing immutable weights)
    core_cpu::DequantCache dc;

    // Per-token confidence (softmax probability of greedy-selected token)
    std::vector<float> char_confidences;

    bool bench;
};

// ── Helper: cast ggml tensor to f32 in graph ──

static ggml_tensor * cast_f32(ggml_context * g, ggml_tensor * t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

// ── Init ──

pix2struct_context * pix2struct_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;

    auto * ctx = new pix2struct_context;
    ctx->backend = nullptr;
    ctx->enc_sched = nullptr;

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

    // Keep backend alive for ggml graph compute
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) { delete ctx; return nullptr; }
    if (!core_gguf::load_weights(model_path, ctx->backend, "pix2struct", ctx->wl)) {
        ggml_backend_free(ctx->backend); delete ctx; return nullptr;
    }

    auto g = [&](const char * name) { return core_gguf::try_get(ctx->wl.tensors, name); };

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
    ctx->sa_cache_len = 0;

    // Create encoder scheduler (reusable across calls)
    {
        int max_nodes = ctx->enc_layers * 40 + 64;
        ggml_backend_t backends[1] = { ctx->backend };
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, 1, max_nodes, false, false);
    }

    ctx->bench = (std::getenv("CRISPEMBED_PIX2STRUCT_BENCH") != nullptr);
    return ctx;
}

void pix2struct_free(pix2struct_context * ctx) {
    if (!ctx) return;
    if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ── Phase 1: Encoder as ggml graph ──

const float * pix2struct_encode_patches(pix2struct_context * ctx,
                                         const float * patches, int n_patches,
                                         int * out_dim) {
    if (!ctx || !patches || n_patches <= 0) return nullptr;

    const int H = ctx->hidden;
    const int patch_dim = ctx->patch_size * ctx->patch_size * 3; // 768
    const int n_heads = ctx->n_heads;
    const int hd = ctx->d_kv;  // 64
    const int d_ff = ctx->d_ff;
    const float eps = ctx->rms_eps;

    // Step 1: Embed patches on CPU (projection + row/col embeddings)
    // This part remains CPU-scalar because patch_proj is a simple linear
    // and the row/col embedding lookup is inherently sequential per-patch.
    std::vector<float> embedded(n_patches * H);
    {
        const float * proj_w = ctx->dc.get(ctx->patch_proj_w);
        const float * proj_b = ctx->dc.get(ctx->patch_proj_b);
        const float * row_w = ctx->dc.get(ctx->row_emb);
        const float * col_w = ctx->dc.get(ctx->col_emb);

        for (int p = 0; p < n_patches; p++) {
            int row_id = (int)patches[p * (patch_dim + 2) + 0];
            int col_id = (int)patches[p * (patch_dim + 2) + 1];
            const float * pixels = &patches[p * (patch_dim + 2) + 2];

            core_cpu::linear_cpu(pixels, &embedded[p * H], patch_dim, H, proj_w, proj_b);

            row_id = std::max(0, std::min(row_id, 4095));
            col_id = std::max(0, std::min(col_id, 4095));
            for (int i = 0; i < H; i++) {
                embedded[p * H + i] += row_w[row_id * H + i];
                embedded[p * H + i] += col_w[col_id * H + i];
            }
        }
    }

    // Step 2: Build ggml graph for 12 encoder layers
    int max_nodes = ctx->enc_layers * 40 + 64;
    size_t meta_sz = ggml_tensor_overhead() * (max_nodes + 64)
                   + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> meta_buf(meta_sz);
    ggml_init_params ip = { meta_sz, meta_buf.data(), true };
    ggml_context * gc = ggml_init(ip);

    // Input: [H, n_patches]
    ggml_tensor * x = ggml_new_tensor_2d(gc, GGML_TYPE_F32, H, n_patches);
    ggml_set_name(x, "enc_input");
    ggml_set_input(x);

    for (int li = 0; li < ctx->enc_layers; li++) {
        const auto & L = ctx->enc[li];
        ggml_tensor * residual = x;

        // Pre-attention RMSNorm
        ggml_tensor * normed = ggml_rms_norm(gc, x, eps);
        normed = ggml_mul(gc, normed, cast_f32(gc, L.pre_attn_norm));

        // QKV projections: [H, T] → [qkv_dim, T]
        ggml_tensor * Q = ggml_mul_mat(gc, L.q_w, normed);
        ggml_tensor * K = ggml_mul_mat(gc, L.k_w, normed);
        ggml_tensor * V = ggml_mul_mat(gc, L.v_w, normed);

        // Reshape to [hd, nh, T] → permute to [hd, T, nh]
        Q = ggml_reshape_3d(gc, Q, hd, n_heads, n_patches);
        K = ggml_reshape_3d(gc, K, hd, n_heads, n_patches);
        V = ggml_reshape_3d(gc, V, hd, n_heads, n_patches);
        Q = ggml_permute(gc, Q, 0, 2, 1, 3);
        K = ggml_permute(gc, K, 0, 2, 1, 3);
        V = ggml_permute(gc, V, 0, 2, 1, 3);

        // Flash attention: T5 encoder uses scale=1.0 (no 1/sqrt(d) scaling)
        // No causal mask, no relative bias in encoder
        ggml_tensor * attn = ggml_flash_attn_ext(gc, Q, K, V, nullptr, 1.0f, 0.0f, 0.0f);
        attn = ggml_reshape_2d(gc, attn, H, n_patches);

        // Output projection + residual
        ggml_tensor * attn_proj = ggml_mul_mat(gc, L.o_w, attn);
        x = ggml_add(gc, residual, attn_proj);

        // Pre-MLP RMSNorm
        residual = x;
        normed = ggml_rms_norm(gc, x, eps);
        normed = ggml_mul(gc, normed, cast_f32(gc, L.pre_mlp_norm));

        // GeGLU FFN: gate = GELU(x @ wi_0), up = x @ wi_1, out = (gate * up) @ wo
        ggml_tensor * gate = ggml_mul_mat(gc, L.wi_0, normed);
        gate = ggml_gelu(gc, gate);
        ggml_tensor * up = ggml_mul_mat(gc, L.wi_1, normed);
        ggml_tensor * ffn_hidden = ggml_mul(gc, gate, up);
        ggml_tensor * ffn_out = ggml_mul_mat(gc, L.wo, ffn_hidden);

        x = ggml_add(gc, residual, ffn_out);
    }

    // Final encoder RMSNorm
    if (ctx->enc_final_norm) {
        x = ggml_rms_norm(gc, x, eps);
        x = ggml_mul(gc, x, cast_f32(gc, ctx->enc_final_norm));
    }

    ggml_set_name(x, "enc_output");
    ggml_set_output(x);

    ggml_cgraph * gf = ggml_new_graph_custom(gc, max_nodes, false);
    ggml_build_forward_expand(gf, x);

    // Execute via backend scheduler
    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "pix2struct: encoder graph alloc failed\n");
        ggml_free(gc);
        return nullptr;
    }

    ggml_tensor * inp_t = ggml_graph_get_tensor(gf, "enc_input");
    ggml_backend_tensor_set(inp_t, embedded.data(), 0, n_patches * H * sizeof(float));

    ggml_backend_sched_graph_compute(ctx->enc_sched, gf);

    // Read output
    ctx->enc_cache.resize(n_patches * H);
    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "enc_output");
    ggml_backend_tensor_get(out_t, ctx->enc_cache.data(), 0, n_patches * H * sizeof(float));
    ctx->enc_cache_n = n_patches;

    ggml_free(gc);

    if (out_dim) *out_dim = H;
    return ctx->enc_cache.data();
}

// ── Phase 2: Pre-compute cross-attention K/V ──

static void precompute_cross_kv(pix2struct_context * ctx) {
    const int n_enc = ctx->enc_cache_n;
    const int H = ctx->hidden;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    const int n_dec = ctx->dec_layers;

    ctx->cross_k_cache.resize(n_dec);
    ctx->cross_v_cache.resize(n_dec);

    // Build ggml graph: project encoder output through cross-attn K/V for all layers
    int max_nodes = n_dec * 6 + 16;
    size_t meta_sz = ggml_tensor_overhead() * (max_nodes + 16)
                   + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> meta_buf(meta_sz);
    ggml_init_params ip = { meta_sz, meta_buf.data(), true };
    ggml_context * gc = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gc, max_nodes, false);

    ggml_tensor * enc_inp = ggml_new_tensor_2d(gc, GGML_TYPE_F32, H, n_enc);
    ggml_set_name(enc_inp, "enc_for_cross");
    ggml_set_input(enc_inp);

    for (int li = 0; li < n_dec; li++) {
        const auto & L = ctx->dec[li];
        char name[64];

        ggml_tensor * k = ggml_mul_mat(gc, L.ca_k, enc_inp);
        snprintf(name, sizeof(name), "cross_k_%d", li);
        ggml_set_name(k, name);
        ggml_set_output(k);
        ggml_build_forward_expand(gf, k);

        ggml_tensor * v = ggml_mul_mat(gc, L.ca_v, enc_inp);
        snprintf(name, sizeof(name), "cross_v_%d", li);
        ggml_set_name(v, name);
        ggml_set_output(v);
        ggml_build_forward_expand(gf, v);
    }

    // Execute
    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "pix2struct: cross K/V alloc failed\n");
        ggml_free(gc);
        return;
    }

    ggml_tensor * inp_t = ggml_graph_get_tensor(gf, "enc_for_cross");
    ggml_backend_tensor_set(inp_t, ctx->enc_cache.data(), 0, n_enc * H * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->enc_sched, gf);

    for (int li = 0; li < n_dec; li++) {
        ctx->cross_k_cache[li].resize(n_enc * qkv_dim);
        ctx->cross_v_cache[li].resize(n_enc * qkv_dim);
        char name[64];

        snprintf(name, sizeof(name), "cross_k_%d", li);
        ggml_tensor * kt = ggml_graph_get_tensor(gf, name);
        if (kt) ggml_backend_tensor_get(kt, ctx->cross_k_cache[li].data(),
                                        0, n_enc * qkv_dim * sizeof(float));

        snprintf(name, sizeof(name), "cross_v_%d", li);
        ggml_tensor * vt = ggml_graph_get_tensor(gf, name);
        if (vt) ggml_backend_tensor_get(vt, ctx->cross_v_cache[li].data(),
                                        0, n_enc * qkv_dim * sizeof(float));
    }

    ggml_free(gc);
}

// ── Decoder: RMSNorm (CPU, single token) ──

static void rms_norm(const float * x, int n, const float * w, float eps, float * out) {
    float sum_sq = 0;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

// ── Decoder: T5 self-attention (single query, cached K/V) ──
// T5 uses raw dot products (no 1/sqrt(d) scaling).

static void t5_self_attn_1q(const float * q_proj,    // [qkv_dim]
                             const float * k_cache,   // [n_past+1, qkv_dim]
                             const float * v_cache,   // [n_past+1, qkv_dim]
                             int n_kv, int n_heads, int hd,
                             const float * rel_bias,  // [n_buckets, n_heads]
                             int n_buckets, int max_dist,
                             int q_pos,               // current position
                             float * out) {            // [qkv_dim]
    int D = n_heads * hd;
    std::vector<float> result(D, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_kv);

        // T5: no scaling, add relative bias
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = core_cpu::dot_product(q_proj + off, k_cache + ki * D + off, hd);
            if (rel_bias) {
                int bucket = t5_relative_bucket(q_pos - ki, false, n_buckets, max_dist);
                scores[ki] += rel_bias[bucket * n_heads + h];
            }
            // Causal mask: mask out future positions
            if (ki > q_pos) scores[ki] = -1e30f;
        }

        // Softmax
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_kv; ki++) scores[ki] *= inv_sum;

        // Weighted sum of values
        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_kv; ki++)
                s += scores[ki] * v_cache[ki * D + off + d];
            result[off + d] = s;
        }
    }
    memcpy(out, result.data(), D * sizeof(float));
}

// ── Decoder: T5 cross-attention (single query, pre-computed K/V) ──
// T5: no 1/sqrt(d) scaling.

static void t5_cross_attn_1q(const float * q_proj,    // [qkv_dim]
                              const float * k_cache,   // [n_enc, qkv_dim]
                              const float * v_cache,   // [n_enc, qkv_dim]
                              int n_enc, int n_heads, int hd,
                              float * out) {            // [qkv_dim]
    int D = n_heads * hd;
    std::vector<float> result(D, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_enc);

        for (int ki = 0; ki < n_enc; ki++)
            scores[ki] = core_cpu::dot_product(q_proj + off, k_cache + ki * D + off, hd);

        float maxs = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (int ki = 0; ki < n_enc; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_enc; ki++) scores[ki] *= inv_sum;

        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_enc; ki++)
                s += scores[ki] * v_cache[ki * D + off + d];
            result[off + d] = s;
        }
    }
    memcpy(out, result.data(), D * sizeof(float));
}

// ── Decoder: GeGLU FFN (single token, CPU) ──

static void geglu_ffn_1t(const float * x, int H, int d_ff,
                          const float * wi_0, const float * wi_1, const float * wo,
                          float * out) {
    std::vector<float> gate(d_ff), up(d_ff), hidden(d_ff);
    core_cpu::linear_cpu(x, gate.data(), H, d_ff, wi_0, nullptr);
    core_cpu::linear_cpu(x, up.data(), H, d_ff, wi_1, nullptr);
    for (int i = 0; i < d_ff; i++) {
        float g = gate[i];
        // GELU (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
        hidden[i] = gelu * up[i];
    }
    core_cpu::linear_cpu(hidden.data(), out, d_ff, H, wo, nullptr);
}

// ── Decoder step (single token, incremental KV cache) ──

static void decoder_step_cached(pix2struct_context * ctx, int step, int tok_id,
                                 float * logits) {
    const int H = ctx->hidden;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    const int n_enc = ctx->enc_cache_n;

    // Get token embedding
    const float * emb_w = ctx->dc.get(ctx->tok_emb);
    std::vector<float> x(H);
    memcpy(x.data(), emb_w + tok_id * H, H * sizeof(float));

    // Get shared relative bias from layer 0
    const float * rel_bias_w = ctx->dc.get(ctx->dec[0].sa_rel_bias);

    std::vector<float> normed(H), attn_out(qkv_dim), proj_out(H);

    for (int li = 0; li < ctx->dec_layers; li++) {
        const auto & L = ctx->dec[li];

        // ── Self-attention with incremental KV cache ──
        rms_norm(x.data(), H, ctx->dc.get(L.sa_norm), ctx->rms_eps, normed.data());

        // Compute Q, K, V for current token only
        std::vector<float> q_proj(qkv_dim), k_new(qkv_dim), v_new(qkv_dim);
        core_cpu::linear_cpu(normed.data(), q_proj.data(), H, qkv_dim,
                             ctx->dc.get(L.sa_q), nullptr);
        core_cpu::linear_cpu(normed.data(), k_new.data(), H, qkv_dim,
                             ctx->dc.get(L.sa_k), nullptr);
        core_cpu::linear_cpu(normed.data(), v_new.data(), H, qkv_dim,
                             ctx->dc.get(L.sa_v), nullptr);

        // Append K/V to cache
        auto & kc = ctx->sa_k_cache[li];
        auto & vc = ctx->sa_v_cache[li];
        memcpy(&kc[step * qkv_dim], k_new.data(), qkv_dim * sizeof(float));
        memcpy(&vc[step * qkv_dim], v_new.data(), qkv_dim * sizeof(float));

        // Attend to full cache (0..step)
        t5_self_attn_1q(q_proj.data(), kc.data(), vc.data(),
                        step + 1, ctx->n_heads, ctx->d_kv,
                        rel_bias_w, ctx->rel_buckets, ctx->rel_max_dist,
                        step, attn_out.data());

        // Output projection + residual
        core_cpu::linear_cpu(attn_out.data(), proj_out.data(), qkv_dim, H,
                             ctx->dc.get(L.sa_o), nullptr);
        for (int i = 0; i < H; i++) x[i] += proj_out[i];

        // ── Cross-attention (pre-computed K/V) ──
        rms_norm(x.data(), H, ctx->dc.get(L.ca_norm), ctx->rms_eps, normed.data());

        // Only compute Q from decoder hidden state
        core_cpu::linear_cpu(normed.data(), q_proj.data(), H, qkv_dim,
                             ctx->dc.get(L.ca_q), nullptr);

        t5_cross_attn_1q(q_proj.data(),
                         ctx->cross_k_cache[li].data(),
                         ctx->cross_v_cache[li].data(),
                         n_enc, ctx->n_heads, ctx->d_kv,
                         attn_out.data());

        core_cpu::linear_cpu(attn_out.data(), proj_out.data(), qkv_dim, H,
                             ctx->dc.get(L.ca_o), nullptr);
        for (int i = 0; i < H; i++) x[i] += proj_out[i];

        // ── FFN ──
        rms_norm(x.data(), H, ctx->dc.get(L.ffn_norm), ctx->rms_eps, normed.data());
        geglu_ffn_1t(normed.data(), H, ctx->d_ff,
                     ctx->dc.get(L.wi_0), ctx->dc.get(L.wi_1), ctx->dc.get(L.wo),
                     proj_out.data());
        for (int i = 0; i < H; i++) x[i] += proj_out[i];
    }

    // Final norm + LM head
    std::vector<float> final_h(H);
    rms_norm(x.data(), H, ctx->dc.get(ctx->final_norm), ctx->rms_eps, final_h.data());
    core_cpu::linear_cpu(final_h.data(), logits, H, ctx->vocab_size,
                         ctx->dc.get(ctx->lm_head), nullptr);
}

// ── Public decode API for parity testing ──

int pix2struct_decode_step0(pix2struct_context * ctx, float * out_logits) {
    if (!ctx || ctx->enc_cache_n <= 0 || !out_logits) return -1;

    // Pre-compute cross-attn K/V if not done
    if (ctx->cross_k_cache.empty()) precompute_cross_kv(ctx);

    // Allocate self-attn KV cache for single step
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    ctx->sa_k_cache.resize(ctx->dec_layers);
    ctx->sa_v_cache.resize(ctx->dec_layers);
    for (int li = 0; li < ctx->dec_layers; li++) {
        ctx->sa_k_cache[li].resize(qkv_dim, 0.0f);
        ctx->sa_v_cache[li].resize(qkv_dim, 0.0f);
    }
    ctx->sa_cache_len = 0;

    // Decode step 0: decoder_start_token_id = 0
    decoder_step_cached(ctx, 0, 0, out_logits);
    ctx->sa_cache_len = 1;
    return 0;
}

// ── Greedy decode (incremental) ──

static std::string greedy_decode(pix2struct_context * ctx, int max_tokens) {
    if (!ctx || ctx->enc_cache_n <= 0) return "";
    if (max_tokens <= 0) max_tokens = 256;

    const int H = ctx->hidden;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;

    // Pre-compute cross-attn K/V (once per encoder call)
    precompute_cross_kv(ctx);

    // Allocate self-attn KV cache
    ctx->sa_k_cache.resize(ctx->dec_layers);
    ctx->sa_v_cache.resize(ctx->dec_layers);
    for (int li = 0; li < ctx->dec_layers; li++) {
        ctx->sa_k_cache[li].resize((max_tokens + 1) * qkv_dim, 0.0f);
        ctx->sa_v_cache[li].resize((max_tokens + 1) * qkv_dim, 0.0f);
    }
    ctx->sa_cache_len = 0;

    std::vector<int32_t> generated = {0}; // start with decoder_start_token_id = 0
    std::vector<float> logits(ctx->vocab_size);
    ctx->char_confidences.clear();

    for (int step = 0; step < max_tokens; step++) {
        int tok_id = generated.back();
        decoder_step_cached(ctx, step, tok_id, logits.data());
        ctx->sa_cache_len = step + 1;

        // Argmax
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < ctx->vocab_size; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }

        if (best == ctx->eos_id) break;
        generated.push_back(best);

        // Confidence
        float se = 0;
        for (int i = 0; i < ctx->vocab_size; i++) se += expf(logits[i] - best_val);
        ctx->char_confidences.push_back(1.0f / se);
    }

    // Token ids as comma-separated string (same as original — no tokenizer yet)
    std::string result;
    for (size_t i = 1; i < generated.size(); i++) {
        if (i > 1) result += ",";
        result += std::to_string(generated[i]);
    }
    return result;
}

// ── Image preprocessing: variable-resolution patching ──

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
