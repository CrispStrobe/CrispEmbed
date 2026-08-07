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
#include "core/env_gate.h"
#include "core/gpu_backend_pref.h"

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
        bucket += max_exact +
                  (int)(logf((float)n / max_exact) / logf((float)max_distance / max_exact) * (n_buckets - max_exact));
        bucket = std::min(bucket, n_buckets - 1);
    }
    return bucket;
}

// ── Encoder/Decoder layer weights ──

struct enc_layer_wt {
    ggml_tensor * pre_attn_norm;
    ggml_tensor *q_w, *k_w, *v_w, *o_w;
    ggml_tensor * pre_mlp_norm;
    ggml_tensor *wi_0, *wi_1, *wo; // GeGLU
};

struct dec_layer_wt {
    // Self-attention
    ggml_tensor * sa_norm;
    ggml_tensor *sa_q, *sa_k, *sa_v, *sa_o;
    ggml_tensor * sa_rel_bias; // only layer 0 (shared)
    // Cross-attention
    ggml_tensor * ca_norm;
    ggml_tensor *ca_q, *ca_k, *ca_v, *ca_o;
    // FFN
    ggml_tensor * ffn_norm;
    ggml_tensor *wi_0, *wi_1, *wo;
};

// ── Model context ──

struct pix2struct_context {
    // Weight storage
    core_gguf::WeightLoad wl;

    // ggml backend (CPU by default on Metal/CPU-only boxes; auto-GPU when a
    // CUDA device is present, or CRISPEMBED_PIX2STRUCT_ENC_GPU=1 forces the
    // best GPU backend with backend_cpu as the sched fallback — O3 + N+4)
    ggml_backend_t backend;
    ggml_backend_t backend_cpu;

    // Encoder scheduler (reusable)
    ggml_backend_sched_t enc_sched;

    int enc_layers, dec_layers, hidden, n_heads, d_kv, d_ff;
    int vocab_size, patch_size, max_patches;
    int n_threads = 1;
    // T5 sentencepiece pieces from tokenizer.tokens (empty on old GGUFs).
    std::vector<std::string> vocab;
    int rel_buckets, rel_max_dist;
    float rms_eps;

    // Encoder weights
    ggml_tensor *patch_proj_w, *patch_proj_b;
    ggml_tensor *row_emb, *col_emb;
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

    // Pre-allocated decoder scratch buffers (avoid per-step/per-layer heap allocs)
    struct dec_scratch {
        std::vector<float> x, normed, attn_out, proj_out;
        std::vector<float> q_proj, k_new, v_new;
        std::vector<float> ffn_gate, ffn_up, ffn_hidden;
        std::vector<float> attn_result, attn_scores;
        std::vector<float> final_h;
        bool allocated = false;
    } ds;

    // Per-token confidence (softmax probability of greedy-selected token)
    std::vector<float> char_confidences;

    // Phase 4 (CRISPEMBED_PIX2STRUCT_GGML_DECODE; default ON when the weights
    // sit on a CUDA backend, opt-in elsewhere): decode step as a
    // single-backend ggml graph with device-resident self/cross KV.
    struct dec_ggml_state {
        ggml_context * kv_ctx = nullptr;
        ggml_backend_buffer_t kv_buf = nullptr;
        ggml_tensor * k = nullptr;  // F32 [qkv_dim, max_seq, n_layers]
        ggml_tensor * v = nullptr;  // F32 [qkv_dim, max_seq, n_layers]
        ggml_tensor * ck = nullptr; // F32 [qkv_dim, n_enc, n_layers]
        ggml_tensor * cv = nullptr; // F32 [qkv_dim, n_enc, n_layers]
        ggml_gallocr_t galloc = nullptr;
        std::vector<uint8_t> meta;
        std::vector<float> bias_host; // [n_kv * n_heads] staged per step
        int max_seq = 0, n_enc = 0;
        bool ready = false;
    } dg;

    bool bench;
};

// ── Helper: cast ggml tensor to f32 in graph ──

static ggml_tensor * cast_f32(ggml_context * g, ggml_tensor * t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

// ── Init ──

pix2struct_context * pix2struct_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;

    auto * ctx = new pix2struct_context;
    ctx->backend = nullptr;
    ctx->enc_sched = nullptr;
    ctx->backend_cpu = nullptr;

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
    ctx->vocab = core_gguf::kv_str_array(meta, "tokenizer.tokens");
    ctx->pad_id = (int)core_gguf::kv_u32(meta, "tokenizer.pad_token_id", 0);
    ctx->rms_eps = 1e-6f;
    core_gguf::free_metadata(meta);

    // Keep backend alive for ggml graph compute.
    // Ignored-n_threads bug class (O13b family): this engine is CPU-only —
    // its whole encoder graph runs here — yet the thread count was discarded
    // with (void)n_threads, so every caller's -t silently ran ggml's default.
    //
    // O3: CRISPEMBED_PIX2STRUCT_ENC_GPU puts the weights + encoder sched on
    // the best GPU backend (got_ocr/R4 pattern: GPU + CPU fallback, guarded
    // set_n_threads).
    //
    // Round N+4 per-backend-kind default (O11 pattern): a CUDA device present
    // => weights load on the GPU so the ggml decode graph runs there — the
    // P100 A/B (chr1s4/crispembed-pix2struct-decode-ab v1) measured decoder
    // 3640-3700 -> 376-431 ms (~9x, q8_0) / 4606 -> 360 ms (f16), decoded
    // text byte-identical in all arms, encoder within noise. Metal/CPU-only
    // boxes keep the CPU default (Metal enc measured no win; O3). =0 forces
    // CPU, =1 forces GPU (either kind), unset = the per-kind default.
    auto cuda_device_present = [] {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char * n = ggml_backend_dev_name(dev);
            if (n && (n[0] == 'C' || n[0] == 'c') && (n[1] == 'U' || n[1] == 'u')) return true; // "CUDA0"
        }
        return false;
    };
    bool enc_gpu;
    if (const char * eg = getenv("CRISPEMBED_PIX2STRUCT_ENC_GPU"); eg && eg[0]) {
        enc_gpu = eg[0] != '0';
    } else {
        enc_gpu = cuda_device_present();
    }
    ctx->backend = enc_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, std::max(1, n_threads));
    ctx->n_threads = std::max(1, n_threads);
    if (!core_gguf::load_weights(model_path, ctx->backend, "pix2struct", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    auto g = [&](const char * name) { return core_gguf::try_get(ctx->wl.tensors, name); };

    ctx->patch_proj_w = g("enc_emb.patch_proj.weight");
    ctx->patch_proj_b = g("enc_emb.patch_proj.bias");
    ctx->row_emb = g("enc_emb.row_emb.weight");
    ctx->col_emb = g("enc_emb.col_emb.weight");

    ctx->enc.resize(ctx->enc_layers);
    for (int i = 0; i < ctx->enc_layers; i++) {
        char pfx[128];
        auto k = [&](const char * s) {
            snprintf(pfx, sizeof(pfx), "enc.%d.%s", i, s);
            return g(pfx);
        };
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
        auto k = [&](const char * s) {
            snprintf(pfx, sizeof(pfx), "dec.%d.%s", i, s);
            return g(pfx);
        };
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

    // Create encoder scheduler (reusable across calls). sched_new asserts the
    // LAST backend is CPU, so a GPU primary gets a CPU fallback appended.
    {
        int max_nodes = ctx->enc_layers * 40 + 64;
        ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
        if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, std::max(1, n_threads));
        ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
        const int n_backends = ctx->backend_cpu ? 2 : 1;
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, n_backends, max_nodes, false, false);
    }

    ctx->bench = core_env::on("CRISPEMBED_PIX2STRUCT_BENCH");
    return ctx;
}

void pix2struct_free(pix2struct_context * ctx) {
    if (!ctx) return;
    if (ctx->dg.galloc) ggml_gallocr_free(ctx->dg.galloc);
    if (ctx->dg.kv_buf) ggml_backend_buffer_free(ctx->dg.kv_buf);
    if (ctx->dg.kv_ctx) ggml_free(ctx->dg.kv_ctx);
    if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// ── Phase 1: Encoder as ggml graph ──

const float * pix2struct_encode_patches(pix2struct_context * ctx, const float * patches, int n_patches, int * out_dim) {
    if (!ctx || !patches || n_patches <= 0) return nullptr;

    const int H = ctx->hidden;
    const int patch_dim = ctx->patch_size * ctx->patch_size * 3; // 768
    const int n_heads = ctx->n_heads;
    const int hd = ctx->d_kv; // 64
    const int d_ff = ctx->d_ff;
    const float eps = ctx->rms_eps;

    // Step 1: Prepare pixel data + position embeddings on CPU
    // Extract pixels into contiguous [patch_dim, n_patches] matrix
    // Gather row/col embeddings into [H, n_patches] matrix
    std::vector<float> pixels_flat(patch_dim * n_patches);
    std::vector<float> pos_emb(H * n_patches);
    {
        const float * row_w = ctx->dc.get(ctx->row_emb);
        const float * col_w = ctx->dc.get(ctx->col_emb);

        for (int p = 0; p < n_patches; p++) {
            int row_id = (int)patches[p * (patch_dim + 2) + 0];
            int col_id = (int)patches[p * (patch_dim + 2) + 1];
            const float * px = &patches[p * (patch_dim + 2) + 2];

            // Copy pixel values into contiguous column
            memcpy(&pixels_flat[p * patch_dim], px, patch_dim * sizeof(float));

            // Gather row + col embeddings
            row_id = std::max(0, std::min(row_id, 4095));
            col_id = std::max(0, std::min(col_id, 4095));
            for (int i = 0; i < H; i++) pos_emb[p * H + i] = row_w[row_id * H + i] + col_w[col_id * H + i];
        }
    }

    // Step 2: Build ggml graph: patch_proj + pos_emb + 12 encoder layers
    int max_nodes = ctx->enc_layers * 40 + 80;
    size_t meta_sz = ggml_tensor_overhead() * (max_nodes + 64) + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> meta_buf(meta_sz);
    ggml_init_params ip = { meta_sz, meta_buf.data(), true };
    ggml_context * gc = ggml_init(ip);

    // Patch projection: pixels [patch_dim, n_patches] @ proj_w → [H, n_patches]
    ggml_tensor * px_inp = ggml_new_tensor_2d(gc, GGML_TYPE_F32, patch_dim, n_patches);
    ggml_set_name(px_inp, "pixels");
    ggml_set_input(px_inp);

    ggml_tensor * x = ggml_mul_mat(gc, ctx->patch_proj_w, px_inp);
    if (ctx->patch_proj_b) x = ggml_add(gc, x, cast_f32(gc, ctx->patch_proj_b));

    // Add position embeddings (pre-gathered row+col on CPU)
    ggml_tensor * pe_inp = ggml_new_tensor_2d(gc, GGML_TYPE_F32, H, n_patches);
    ggml_set_name(pe_inp, "pos_emb");
    ggml_set_input(pe_inp);
    x = ggml_add(gc, x, pe_inp);
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

    // Feed pixel data and position embeddings
    ggml_tensor * px_t = ggml_graph_get_tensor(gf, "pixels");
    ggml_backend_tensor_set(px_t, pixels_flat.data(), 0, patch_dim * n_patches * sizeof(float));
    ggml_tensor * pe_t = ggml_graph_get_tensor(gf, "pos_emb");
    ggml_backend_tensor_set(pe_t, pos_emb.data(), 0, H * n_patches * sizeof(float));

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
    size_t meta_sz = ggml_tensor_overhead() * (max_nodes + 16) + ggml_graph_overhead_custom(max_nodes, false);
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
        if (kt) ggml_backend_tensor_get(kt, ctx->cross_k_cache[li].data(), 0, n_enc * qkv_dim * sizeof(float));

        snprintf(name, sizeof(name), "cross_v_%d", li);
        ggml_tensor * vt = ggml_graph_get_tensor(gf, name);
        if (vt) ggml_backend_tensor_get(vt, ctx->cross_v_cache[li].data(), 0, n_enc * qkv_dim * sizeof(float));
    }

    ggml_free(gc);
}

// ── Allocate decoder scratch buffers (once, reused across all decode steps) ──

static void ensure_dec_scratch(pix2struct_context * ctx, int max_seq) {
    if (ctx->ds.allocated) return;
    const int H = ctx->hidden;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    const int d_ff = ctx->d_ff;
    const int n_enc = ctx->enc_cache_n;
    int max_kv = std::max(max_seq, n_enc); // scores buffer must fit both

    ctx->ds.x.resize(H);
    ctx->ds.normed.resize(H);
    ctx->ds.attn_out.resize(qkv_dim);
    ctx->ds.proj_out.resize(H);
    ctx->ds.q_proj.resize(qkv_dim);
    ctx->ds.k_new.resize(qkv_dim);
    ctx->ds.v_new.resize(qkv_dim);
    ctx->ds.ffn_gate.resize(d_ff);
    ctx->ds.ffn_up.resize(d_ff);
    ctx->ds.ffn_hidden.resize(d_ff);
    ctx->ds.attn_result.resize(qkv_dim);
    ctx->ds.attn_scores.resize(max_kv);
    ctx->ds.final_h.resize(H);
    ctx->ds.allocated = true;
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

static void t5_self_attn_1q(const float * q_proj,  // [qkv_dim]
                            const float * k_cache, // [n_past+1, qkv_dim]
                            const float * v_cache, // [n_past+1, qkv_dim]
                            int n_kv, int n_heads, int hd,
                            const float * rel_bias, // [n_buckets, n_heads]
                            int n_buckets, int max_dist,
                            int q_pos,            // current position
                            float * out,          // [qkv_dim]
                            float * result_buf,   // [qkv_dim] pre-allocated
                            float * scores_buf) { // [>=n_kv] pre-allocated
    int D = n_heads * hd;
    memset(result_buf, 0, D * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;

        // T5: no scaling, add relative bias
        for (int ki = 0; ki < n_kv; ki++) {
            scores_buf[ki] = core_cpu::dot_product(q_proj + off, k_cache + ki * D + off, hd);
            if (rel_bias) {
                // t5_relative_bucket expects HF's raw relative_position =
                // memory_position - query_position (its body does
                // n = -rel_pos). This call passed q_pos - ki (query - memory),
                // so n = ki - q_pos <= 0 clamped to 0: EVERY history token
                // landed in bucket 0 and the decoder had no positional
                // discrimination at all — the repetition degeneration that
                // base-model babble had masked (textcaps + the HF reference
                // exposed it: HF captions fox correctly, the port looped).
                int bucket = t5_relative_bucket(ki - q_pos, false, n_buckets, max_dist);
                scores_buf[ki] += rel_bias[bucket * n_heads + h];
            }
            if (ki > q_pos) scores_buf[ki] = -1e30f;
        }

        // Softmax
        float maxs = scores_buf[0];
        for (int ki = 1; ki < n_kv; ki++) maxs = std::max(maxs, scores_buf[ki]);
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores_buf[ki] = expf(scores_buf[ki] - maxs);
            sum += scores_buf[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_kv; ki++) scores_buf[ki] *= inv_sum;

        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_kv; ki++) s += scores_buf[ki] * v_cache[ki * D + off + d];
            result_buf[off + d] = s;
        }
    }
    memcpy(out, result_buf, D * sizeof(float));
}

// ── Decoder: T5 cross-attention (single query, pre-computed K/V) ──
// T5: no 1/sqrt(d) scaling.

static void t5_cross_attn_1q(const float * q_proj,  // [qkv_dim]
                             const float * k_cache, // [n_enc, qkv_dim]
                             const float * v_cache, // [n_enc, qkv_dim]
                             int n_enc, int n_heads, int hd,
                             float * out,          // [qkv_dim]
                             float * result_buf,   // [qkv_dim] pre-allocated
                             float * scores_buf) { // [>=n_enc] pre-allocated
    int D = n_heads * hd;
    memset(result_buf, 0, D * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;

        for (int ki = 0; ki < n_enc; ki++)
            scores_buf[ki] = core_cpu::dot_product(q_proj + off, k_cache + ki * D + off, hd);

        float maxs = scores_buf[0];
        for (int ki = 1; ki < n_enc; ki++) maxs = std::max(maxs, scores_buf[ki]);
        float sum = 0;
        for (int ki = 0; ki < n_enc; ki++) {
            scores_buf[ki] = expf(scores_buf[ki] - maxs);
            sum += scores_buf[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_enc; ki++) scores_buf[ki] *= inv_sum;

        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_enc; ki++) s += scores_buf[ki] * v_cache[ki * D + off + d];
            result_buf[off + d] = s;
        }
    }
    memcpy(out, result_buf, D * sizeof(float));
}

// ── Decoder: GeGLU FFN (single token, CPU) ──

static void geglu_ffn_1t(const float * x, int H, int d_ff, const float * wi_0, const float * wi_1, const float * wo,
                         float * out, float * gate_buf, float * up_buf, float * hidden_buf) {
    core_cpu::linear_cpu(x, gate_buf, H, d_ff, wi_0, nullptr);
    core_cpu::linear_cpu(x, up_buf, H, d_ff, wi_1, nullptr);
    for (int i = 0; i < d_ff; i++) {
        float g = gate_buf[i];
        float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
        hidden_buf[i] = gelu * up_buf[i];
    }
    core_cpu::linear_cpu(hidden_buf, out, d_ff, H, wo, nullptr);
}

// ── Decoder step (single token, incremental KV cache) ──

static void decoder_step_cached(pix2struct_context * ctx, int step, int tok_id, float * logits) {
    const int H = ctx->hidden;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    const int n_enc = ctx->enc_cache_n;
    auto & ds = ctx->ds;

    // Get token embedding
    const float * emb_w = ctx->dc.get(ctx->tok_emb);
    memcpy(ds.x.data(), emb_w + tok_id * H, H * sizeof(float));

    // Get shared relative bias from layer 0
    const float * rel_bias_w = ctx->dc.get(ctx->dec[0].sa_rel_bias);

    for (int li = 0; li < ctx->dec_layers; li++) {
        const auto & L = ctx->dec[li];

        // ── Self-attention with incremental KV cache ──
        rms_norm(ds.x.data(), H, ctx->dc.get(L.sa_norm), ctx->rms_eps, ds.normed.data());

        core_cpu::linear_cpu(ds.normed.data(), ds.q_proj.data(), H, qkv_dim, ctx->dc.get(L.sa_q), nullptr);
        core_cpu::linear_cpu(ds.normed.data(), ds.k_new.data(), H, qkv_dim, ctx->dc.get(L.sa_k), nullptr);
        core_cpu::linear_cpu(ds.normed.data(), ds.v_new.data(), H, qkv_dim, ctx->dc.get(L.sa_v), nullptr);

        // Append K/V to cache
        auto & kc = ctx->sa_k_cache[li];
        auto & vc = ctx->sa_v_cache[li];
        memcpy(&kc[step * qkv_dim], ds.k_new.data(), qkv_dim * sizeof(float));
        memcpy(&vc[step * qkv_dim], ds.v_new.data(), qkv_dim * sizeof(float));

        // Attend to full cache (0..step)
        t5_self_attn_1q(ds.q_proj.data(), kc.data(), vc.data(), step + 1, ctx->n_heads, ctx->d_kv, rel_bias_w,
                        ctx->rel_buckets, ctx->rel_max_dist, step, ds.attn_out.data(), ds.attn_result.data(),
                        ds.attn_scores.data());

        // Output projection + residual
        core_cpu::linear_cpu(ds.attn_out.data(), ds.proj_out.data(), qkv_dim, H, ctx->dc.get(L.sa_o), nullptr);
        for (int i = 0; i < H; i++) ds.x[i] += ds.proj_out[i];

        // ── Cross-attention (pre-computed K/V) ──
        rms_norm(ds.x.data(), H, ctx->dc.get(L.ca_norm), ctx->rms_eps, ds.normed.data());

        core_cpu::linear_cpu(ds.normed.data(), ds.q_proj.data(), H, qkv_dim, ctx->dc.get(L.ca_q), nullptr);

        t5_cross_attn_1q(ds.q_proj.data(), ctx->cross_k_cache[li].data(), ctx->cross_v_cache[li].data(), n_enc,
                         ctx->n_heads, ctx->d_kv, ds.attn_out.data(), ds.attn_result.data(), ds.attn_scores.data());

        core_cpu::linear_cpu(ds.attn_out.data(), ds.proj_out.data(), qkv_dim, H, ctx->dc.get(L.ca_o), nullptr);
        for (int i = 0; i < H; i++) ds.x[i] += ds.proj_out[i];

        // ── FFN ──
        rms_norm(ds.x.data(), H, ctx->dc.get(L.ffn_norm), ctx->rms_eps, ds.normed.data());
        geglu_ffn_1t(ds.normed.data(), H, ctx->d_ff, ctx->dc.get(L.wi_0), ctx->dc.get(L.wi_1), ctx->dc.get(L.wo),
                     ds.proj_out.data(), ds.ffn_gate.data(), ds.ffn_up.data(), ds.ffn_hidden.data());
        for (int i = 0; i < H; i++) ds.x[i] += ds.proj_out[i];
    }

    // Final norm + LM head
    rms_norm(ds.x.data(), H, ctx->dc.get(ctx->final_norm), ctx->rms_eps, ds.final_h.data());
    // The 768x50244 lm_head is the single largest per-step matvec — row-split
    // it across the engine's threads (bitwise-identical to the 1t path).
    core_cpu::linear_cpu_mt(ds.final_h.data(), logits, H, ctx->vocab_size, ctx->dc.get(ctx->lm_head), nullptr,
                            ctx->n_threads);
}

// ── Phase 4: decode step as a ggml graph (opt-in) ──
//
// CRISPEMBED_PIX2STRUCT_GGML_DECODE=1 replaces the per-token CPU scalar loop
// with a single-backend ggml step graph: self/cross KV live device-resident
// in a persistent buffer (got_ocr alloc_kv_cache pattern), K/V for the new
// position are written in-graph via ggml_cpy into a position view, and a
// dedicated gallocr is reserved once at max KV length so every step's alloc
// takes the no-realloc fast path (GOT_OCR_DECODE_CACHE pattern; the graph is
// single-backend by construction — weights, KV, and compute all sit on
// ctx->backend — so no sched and no split_graph per token).
//
// The T5 relative-position bias depends on the step, so it enters as a
// per-step input tensor [n_kv, 1, n_heads] filled host-side from the same
// t5_relative_bucket the scalar path uses. n_kv == step+1 exactly (the KV
// read view never exposes future slots), so no causal mask is needed.
// T5 attention is unscaled (scale = 1.0) in both attentions, matching the
// scalar path.
//
// Default stays the scalar loop until the A/B wins speed AND quality
// (decoded-output identity gate). GPU decode additionally requires the
// weights on the GPU backend, i.e. CRISPEMBED_PIX2STRUCT_ENC_GPU=1.

struct dec_step_graph {
    ggml_cgraph * gf = nullptr;
    ggml_context * gctx = nullptr;
    ggml_tensor * tok_in = nullptr;  // I32 [1]
    ggml_tensor * bias_in = nullptr; // F32 [n_kv, 1, n_heads]
    ggml_tensor * logits = nullptr;  // F32 [vocab]
};

static dec_step_graph build_dec_step_graph(pix2struct_context * ctx, int step) {
    const int H = ctx->hidden;
    const int nh = ctx->n_heads;
    const int hd = ctx->d_kv;
    const int qkv_dim = nh * hd;
    const int n_kv = step + 1;
    const int n_enc = ctx->dg.n_enc;
    const float eps = ctx->rms_eps;
    auto & dg = ctx->dg;

    // Views/permutes/conts/casts all count as graph nodes: each attention is
    // ~15 nodes and a layer carries two of them plus QKV/O matmuls, KV cpys,
    // norms, and the FFN — ~80 nodes/layer measured; 128 leaves slack.
    const int max_nodes = ctx->dec_layers * 128 + 64;
    size_t meta_sz = ggml_tensor_overhead() * (max_nodes + 64) + ggml_graph_overhead_custom(max_nodes, false);
    if (dg.meta.size() < meta_sz) dg.meta.resize(meta_sz);

    dec_step_graph sg;
    ggml_init_params ip = { meta_sz, dg.meta.data(), true };
    sg.gctx = ggml_init(ip);
    auto * g = sg.gctx;
    sg.gf = ggml_new_graph_custom(g, max_nodes, false);

    sg.tok_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(sg.tok_in, "tok");
    ggml_set_input(sg.tok_in);

    sg.bias_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, n_kv, 1, nh);
    ggml_set_name(sg.bias_in, "sa_bias");
    ggml_set_input(sg.bias_in);

    // Token embedding (get_rows dequantizes the selected row to F32)
    ggml_tensor * x = ggml_get_rows(g, ctx->tok_emb, sg.tok_in); // [H, 1]

    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), cast_f32(g, w));
    };
    // Single-query attention over a [qkv_dim, n_kv_len] K/V view.
    // q2d: [qkv_dim, 1]. bias: nullptr or [n_kv_len, 1, nh]. Returns [qkv_dim, 1].
    auto attn_1q = [&](ggml_tensor * q2d, ggml_tensor * Kv, ggml_tensor * Vv, int n_kv_len, ggml_tensor * bias) {
        ggml_tensor * Q = ggml_permute(g, ggml_reshape_3d(g, q2d, hd, nh, 1), 0, 2, 1, 3); // [hd, 1, nh]
        ggml_tensor * K = ggml_permute(g, ggml_reshape_3d(g, Kv, hd, nh, n_kv_len), 0, 2, 1, 3);
        ggml_tensor * scores = ggml_mul_mat(g, ggml_cont(g, K), ggml_cont(g, Q)); // [n_kv_len, 1, nh]
        if (bias) scores = ggml_add(g, scores, bias);
        ggml_tensor * probs = ggml_soft_max(g, scores); // T5: unscaled
        // V^T layout [n_kv_len, hd, nh] so mul_mat gives [hd, 1, nh]
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Vv, hd, nh, n_kv_len), 1, 2, 0, 3));
        ggml_tensor * out = ggml_mul_mat(g, Vt, probs);       // [hd, 1, nh]
        out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)); // [hd, nh, 1]
        return ggml_reshape_2d(g, out, qkv_dim, 1);
    };

    for (int li = 0; li < ctx->dec_layers; li++) {
        const auto & L = ctx->dec[li];

        // ── Self-attention (incremental KV, written in-graph) ──
        ggml_tensor * normed = rmsnorm(x, L.sa_norm);
        ggml_tensor * q = ggml_mul_mat(g, L.sa_q, normed);
        ggml_tensor * k_new = ggml_mul_mat(g, L.sa_k, normed);
        ggml_tensor * v_new = ggml_mul_mat(g, L.sa_v, normed);

        const size_t k_off = (size_t)li * dg.k->nb[2] + (size_t)step * dg.k->nb[1];
        const size_t v_off = (size_t)li * dg.v->nb[2] + (size_t)step * dg.v->nb[1];
        ggml_tensor * k_dst = ggml_view_1d(g, dg.k, qkv_dim, k_off);
        ggml_tensor * v_dst = ggml_view_1d(g, dg.v, qkv_dim, v_off);
        ggml_build_forward_expand(sg.gf, ggml_cpy(g, ggml_reshape_1d(g, k_new, qkv_dim), k_dst));
        ggml_build_forward_expand(sg.gf, ggml_cpy(g, ggml_reshape_1d(g, v_new, qkv_dim), v_dst));

        ggml_tensor * Kv = ggml_view_2d(g, dg.k, qkv_dim, n_kv, dg.k->nb[1], (size_t)li * dg.k->nb[2]);
        ggml_tensor * Vv = ggml_view_2d(g, dg.v, qkv_dim, n_kv, dg.v->nb[1], (size_t)li * dg.v->nb[2]);
        ggml_tensor * attn = attn_1q(q, Kv, Vv, n_kv, sg.bias_in);
        x = ggml_add(g, x, ggml_mul_mat(g, L.sa_o, attn));

        // ── Cross-attention (pre-computed device K/V) ──
        normed = rmsnorm(x, L.ca_norm);
        q = ggml_mul_mat(g, L.ca_q, normed);
        ggml_tensor * CKv = ggml_view_2d(g, dg.ck, qkv_dim, n_enc, dg.ck->nb[1], (size_t)li * dg.ck->nb[2]);
        ggml_tensor * CVv = ggml_view_2d(g, dg.cv, qkv_dim, n_enc, dg.cv->nb[1], (size_t)li * dg.cv->nb[2]);
        attn = attn_1q(q, CKv, CVv, n_enc, nullptr);
        x = ggml_add(g, x, ggml_mul_mat(g, L.ca_o, attn));

        // ── GeGLU FFN (ggml_gelu is the tanh approximation, matching the
        //    scalar path's formula) ──
        normed = rmsnorm(x, L.ffn_norm);
        ggml_tensor * gate = ggml_gelu(g, ggml_mul_mat(g, L.wi_0, normed));
        ggml_tensor * up = ggml_mul_mat(g, L.wi_1, normed);
        x = ggml_add(g, x, ggml_mul_mat(g, L.wo, ggml_mul(g, gate, up)));
    }

    // Final norm + LM head
    x = rmsnorm(x, ctx->final_norm);
    sg.logits = ggml_mul_mat(g, ctx->lm_head, x); // [vocab, 1]
    ggml_set_name(sg.logits, "logits");
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);
    return sg;
}

// Allocate device KV, upload cross K/V (after precompute_cross_kv), reserve
// the gallocr at max shapes. Returns false on any failure (caller falls back
// to the scalar path).
static bool dec_ggml_prepare(pix2struct_context * ctx, int max_seq) {
    auto & dg = ctx->dg;
    const int qkv_dim = ctx->n_heads * ctx->d_kv;
    const int nl = ctx->dec_layers;
    const int n_enc = ctx->enc_cache_n;

    if (dg.kv_buf && (dg.max_seq < max_seq || dg.n_enc != n_enc)) {
        if (dg.galloc) ggml_gallocr_free(dg.galloc);
        ggml_backend_buffer_free(dg.kv_buf);
        ggml_free(dg.kv_ctx);
        dg.galloc = nullptr;
        dg.kv_buf = nullptr;
        dg.kv_ctx = nullptr;
        dg.ready = false;
    }

    if (!dg.kv_buf) {
        size_t ctx_size = 4 * ggml_tensor_overhead() + 1024;
        ggml_init_params ip = { ctx_size, nullptr, true };
        dg.kv_ctx = ggml_init(ip);
        dg.k = ggml_new_tensor_3d(dg.kv_ctx, GGML_TYPE_F32, qkv_dim, max_seq, nl);
        dg.v = ggml_new_tensor_3d(dg.kv_ctx, GGML_TYPE_F32, qkv_dim, max_seq, nl);
        dg.ck = ggml_new_tensor_3d(dg.kv_ctx, GGML_TYPE_F32, qkv_dim, n_enc, nl);
        dg.cv = ggml_new_tensor_3d(dg.kv_ctx, GGML_TYPE_F32, qkv_dim, n_enc, nl);
        dg.kv_buf = ggml_backend_alloc_ctx_tensors(dg.kv_ctx, ctx->backend);
        if (!dg.kv_buf) {
            ggml_free(dg.kv_ctx);
            dg.kv_ctx = nullptr;
            return false;
        }
        dg.max_seq = max_seq;
        dg.n_enc = n_enc;
    }
    ggml_backend_buffer_clear(dg.kv_buf, 0);

    // Upload the host cross K/V (already [n_enc, qkv_dim] row-major = the
    // [qkv_dim, n_enc] ggml layout of one layer slice).
    const size_t layer_bytes = (size_t)n_enc * qkv_dim * sizeof(float);
    for (int li = 0; li < nl; li++) {
        ggml_backend_tensor_set(dg.ck, ctx->cross_k_cache[li].data(), (size_t)li * dg.ck->nb[2], layer_bytes);
        ggml_backend_tensor_set(dg.cv, ctx->cross_v_cache[li].data(), (size_t)li * dg.cv->nb[2], layer_bytes);
    }

    if (!dg.galloc) {
        dg.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        dec_step_graph rg = build_dec_step_graph(ctx, max_seq - 1);
        const bool reserved = ggml_gallocr_reserve(dg.galloc, rg.gf);
        ggml_free(rg.gctx);
        if (!reserved) {
            fprintf(stderr, "pix2struct: decode gallocr reserve failed; falling back to scalar decode\n");
            ggml_gallocr_free(dg.galloc);
            dg.galloc = nullptr;
            return false;
        }
    }
    dg.ready = true;
    return true;
}

static bool decoder_step_ggml(pix2struct_context * ctx, int step, int tok_id, float * logits) {
    auto & dg = ctx->dg;
    const int nh = ctx->n_heads;
    const int n_kv = step + 1;

    dec_step_graph sg = build_dec_step_graph(ctx, step);
    if (!ggml_gallocr_alloc_graph(dg.galloc, sg.gf)) {
        ggml_free(sg.gctx);
        return false;
    }

    const int32_t tok = tok_id;
    ggml_backend_tensor_set(sg.tok_in, &tok, 0, sizeof(int32_t));

    // T5 relative bias for this step: bias[h][ki] over the [n_kv, 1, nh]
    // input; same bucket call as the scalar path (rel_pos = ki - q_pos <= 0).
    const float * rel_bias_w = ctx->dc.get(ctx->dec[0].sa_rel_bias);
    dg.bias_host.resize((size_t)n_kv * nh);
    for (int ki = 0; ki < n_kv; ki++) {
        const int bucket = t5_relative_bucket(ki - step, false, ctx->rel_buckets, ctx->rel_max_dist);
        for (int h = 0; h < nh; h++) dg.bias_host[(size_t)h * n_kv + ki] = rel_bias_w[bucket * nh + h];
    }
    ggml_backend_tensor_set(sg.bias_in, dg.bias_host.data(), 0, (size_t)n_kv * nh * sizeof(float));

    if (ggml_backend_graph_compute(ctx->backend, sg.gf) != GGML_STATUS_SUCCESS) {
        ggml_free(sg.gctx);
        return false;
    }
    ggml_backend_tensor_get(sg.logits, logits, 0, (size_t)ctx->vocab_size * sizeof(float));
    ggml_free(sg.gctx);
    return true;
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

    // Ensure decoder scratch buffers are allocated
    ctx->ds.allocated = false; // force re-alloc in case n_enc changed
    ensure_dec_scratch(ctx, 1);

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

    // Phase 4: ggml decode-step graph with device-resident KV. Per-backend-
    // kind default (P100 A/B: decoder ~9x q8_0 / ~12.8x f16, decoded text
    // byte-identical): weights on a CUDA backend => ggml decode; elsewhere
    // the scalar loop stays the default. =0 forces scalar, =1 forces the
    // ggml graph on whatever backend holds the weights (CPU/Metal proven
    // byte-identical locally), unset = the per-kind default.
    bool want_ggml;
    if (const char * ge = getenv("CRISPEMBED_PIX2STRUCT_GGML_DECODE"); ge && ge[0]) {
        want_ggml = ge[0] != '0';
    } else {
        const char * bn = ggml_backend_name(ctx->backend);
        want_ggml = bn && (bn[0] == 'C' || bn[0] == 'c') && (bn[1] == 'U' || bn[1] == 'u'); // "CUDA0"
    }
    const bool use_ggml = want_ggml && dec_ggml_prepare(ctx, max_tokens + 1);
    if (ctx->bench) fprintf(stderr, "[pix2struct-bench] decode path: %s\n", use_ggml ? "ggml" : "scalar");

    if (!use_ggml) {
        // Allocate self-attn KV cache
        ctx->sa_k_cache.resize(ctx->dec_layers);
        ctx->sa_v_cache.resize(ctx->dec_layers);
        for (int li = 0; li < ctx->dec_layers; li++) {
            ctx->sa_k_cache[li].resize((max_tokens + 1) * qkv_dim, 0.0f);
            ctx->sa_v_cache[li].resize((max_tokens + 1) * qkv_dim, 0.0f);
        }
        ctx->sa_cache_len = 0;

        // Pre-allocate decoder scratch buffers
        ctx->ds.allocated = false;
        ensure_dec_scratch(ctx, max_tokens + 1);
    }

    std::vector<int32_t> generated = { 0 }; // start with decoder_start_token_id = 0
    std::vector<float> logits(ctx->vocab_size);
    ctx->char_confidences.clear();

    for (int step = 0; step < max_tokens; step++) {
        int tok_id = generated.back();
        if (use_ggml) {
            if (!decoder_step_ggml(ctx, step, tok_id, logits.data())) {
                fprintf(stderr, "pix2struct: ggml decode step %d failed; output truncated\n", step);
                break;
            }
        } else {
            decoder_step_cached(ctx, step, tok_id, logits.data());
        }
        ctx->sa_cache_len = step + 1;

        // Argmax
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < ctx->vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }

        if (best == ctx->eos_id) break;
        generated.push_back(best);

        // Confidence
        float se = 0;
        for (int i = 0; i < ctx->vocab_size; i++) se += expf(logits[i] - best_val);
        ctx->char_confidences.push_back(1.0f / se);
    }

    // Detokenize via the T5 sentencepiece pieces the converter has always
    // written to tokenizer.tokens — the engine just never read them (output
    // was raw comma-separated ids). '\xE2\x96\x81' (U+2581) marks a word
    // start -> space; <0xNN> byte-fallback pieces decode to their raw byte;
    // other <...> specials are skipped. PIX2STRUCT_RAW_IDS=1 (or an old GGUF
    // without tokenizer.tokens) restores the id output.
    std::string result;
    const bool raw_ids = core_env::on("PIX2STRUCT_RAW_IDS") || ctx->vocab.empty();
    if (!raw_ids) {
        for (size_t i = 1; i < generated.size(); i++) {
            const int id = generated[i];
            if (id < 0 || id >= (int)ctx->vocab.size()) continue;
            const std::string & piece = ctx->vocab[id];
            if (piece.size() >= 2 && piece.front() == '<' && piece.back() == '>') {
                if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0) {
                    result += (char)strtol(piece.c_str() + 3, nullptr, 16);
                }
                continue; // <pad>, </s>, <unk>, extra_id specials
            }
            std::string p = piece;
            size_t pos = 0;
            while ((pos = p.find("\xE2\x96\x81", pos)) != std::string::npos) {
                p.replace(pos, 3, " ");
                pos += 1;
            }
            result += p;
        }
        if (!result.empty() && result.front() == ' ') result.erase(0, 1);
        return result;
    }
    for (size_t i = 1; i < generated.size(); i++) {
        if (i > 1) result += ",";
        result += std::to_string(generated[i]);
    }
    return result;
}

// ── Image preprocessing: variable-resolution patching ──

static std::vector<float> image_to_patches(const uint8_t * rgb, int W, int H, int max_patches, int patch_size,
                                           int * out_n_patches) {
    const int pH = patch_size, pW = patch_size, C = 3;
    float scale = sqrtf((float)max_patches * ((float)pH / H) * ((float)pW / W));
    int n_rows = std::max(1, std::min((int)floorf(scale * H / pH), max_patches));
    int n_cols = std::max(1, std::min((int)floorf(scale * W / pW), max_patches));
    while (n_rows * n_cols > max_patches) {
        if (n_rows > n_cols)
            n_rows--;
        else
            n_cols--;
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
                float v =
                    (1 - fy) * ((1 - fx) * (float)rgb[(y0 * W + x0) * C + c] + fx * (float)rgb[(y0 * W + x1) * C + c]) +
                    fy * ((1 - fx) * (float)rgb[(y1 * W + x0) * C + c] + fx * (float)rgb[(y1 * W + x1) * C + c]);
                resized[c * rH * rW + y * rW + x] = v / 255.0f;
            }

    int total = C * rH * rW;
    float mean = 0;
    for (int i = 0; i < total; i++) mean += resized[i];
    mean /= total;
    float var = 0;
    for (int i = 0; i < total; i++) {
        float d = resized[i] - mean;
        var += d * d;
    }
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
            // HF flattens each patch HWC (torch_extract_patches permutes to
            // (ph, pw, channels) before reshape) and enc_emb.patch_proj was
            // trained on that order. This loop packed CHW — a within-patch
            // PERMUTATION that is invisible on constant background patches
            // (cos 1.0, which is how it survived) and decorrelates every
            // structured patch (text patches measured cos 0.16 vs HF; the
            // encoder then smeared the damage everywhere). The caption drift
            // ('mummies over the lazy day') was this.
            for (int py = 0; py < pH; py++)
                for (int px = 0; px < pW; px++)
                    for (int c = 0; c < C; c++)
                        patches[pi * feat_dim + 2 + (py * pW + px) * C + c] =
                            resized[c * rH * rW + (r * pH + py) * rW + (col * pW + px)];
        }
    if (out_n_patches) *out_n_patches = n_patches;
    return patches;
}

// ── Generate ──

const char * pix2struct_generate(pix2struct_context * ctx, const uint8_t * image, int width, int height,
                                 int max_tokens) {
    if (!ctx || !image || width <= 0 || height <= 0) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    int n_patches = 0;
    auto patches = image_to_patches(image, width, height, ctx->max_patches, ctx->patch_size, &n_patches);
    if (n_patches <= 0) return nullptr;
    // Parity tracing: dump the packed patch tensor ([max_patches, 2+patch_flat])
    if (const char * dp = std::getenv("PIX2STRUCT_DUMP_PATCHES")) {
        FILE * f = fopen(dp, "wb");
        if (f) {
            fwrite(patches.data(), sizeof(float), patches.size(), f);
            fclose(f);
            fprintf(stderr, "[p2s-dump] patches %d x %d -> %s\n", n_patches, ctx->patch_size * ctx->patch_size * 3 + 2,
                    dp);
        }
    }
    if (bench)
        fprintf(stderr, "[pix2struct-bench] preprocess: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    t0 = std::chrono::steady_clock::now();
    int out_dim = 0;
    pix2struct_encode_patches(ctx, patches.data(), n_patches, &out_dim);
    // Parity tracing: dump the encoder output cache ([n_enc, hidden])
    if (const char * de = std::getenv("PIX2STRUCT_DUMP_ENC")) {
        FILE * f = fopen(de, "wb");
        if (f) {
            fwrite(ctx->enc_cache.data(), sizeof(float), (size_t)ctx->enc_cache_n * ctx->hidden, f);
            fclose(f);
            fprintf(stderr, "[p2s-dump] enc %d x %d -> %s\n", ctx->enc_cache_n, ctx->hidden, de);
        }
    }
    if (bench)
        fprintf(stderr, "[pix2struct-bench] encoder: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    t0 = std::chrono::steady_clock::now();
    static std::string result;
    result = greedy_decode(ctx, max_tokens > 0 ? max_tokens : 256);
    if (bench)
        fprintf(stderr, "[pix2struct-bench] decoder: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    if (bench)
        fprintf(stderr, "[pix2struct-bench] total: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count());

    return result.c_str();
}

void pix2struct_free_text(const char * text) {
    (void)text;
}

const float * pix2struct_confidences(const pix2struct_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float pix2struct_mean_confidence(const pix2struct_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double s = 0;
    for (float v : ctx->char_confidences) s += v;
    return (float)(s / ctx->char_confidences.size());
}
