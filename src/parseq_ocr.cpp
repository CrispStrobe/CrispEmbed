// parseq_ocr.cpp — PARSeq scene text recognition via ggml.
//
// Architecture (ECCV 2022, baudm/parseq, Apache-2.0):
//   Encoder: 12-layer pre-LN ViT (GELU FFN, fused QKV, learned pos embed)
//     Input: [3, 32, 128] RGB → Conv2d patch [4,8] → 128 tokens → 12 layers
//   Decoder: single-layer two-stream transformer
//     Self-attn: position queries (Q) attend to context tokens (K/V)
//     Cross-attn: attends to encoder output (image memory)
//     Pre-LN with norm_q, norm_c (self-attn), norm1 (cross-attn), norm2 (FFN)
//   Head: Linear(embed_dim, 95) → 94 printable ASCII chars + EOS
//   Inference: autoregressive (causal mask) + optional refinement (cloze mask)
//
// The encoder runs as a ggml graph (fast BLAS-backed matmuls).
// The decoder runs CPU-scalar (1 layer, overhead of graph setup not worth it).

#include "parseq_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct enc_layer {
    ggml_tensor * ln1_w, * ln1_b;
    ggml_tensor * qkv_w, * qkv_b;
    ggml_tensor * proj_w, * proj_b;
    ggml_tensor * ln2_w, * ln2_b;
    ggml_tensor * fc1_w, * fc1_b;
    ggml_tensor * fc2_w, * fc2_b;
};

struct parseq_ocr_context {
    parseq_ocr_hparams hp;

    // Encoder
    ggml_tensor * patch_proj_w;  // [D, 3*ph*pw] (flattened from [D, 3, ph, pw])
    ggml_tensor * patch_proj_b;  // [D]
    ggml_tensor * pos_embed;     // [D, 128] (ggml: [ne0=D, ne1=N])
    ggml_tensor * enc_norm_w, * enc_norm_b;
    std::vector<enc_layer> enc_layers;

    // Decoder
    ggml_tensor * pos_queries;   // [D, 26] (ggml: ne0=D, ne1=max_label_len)
    ggml_tensor * text_embed_w;  // [D, 97] (ggml: ne0=D, ne1=n_tokens)
    // Self-attention (in_proj fused QKV)
    ggml_tensor * sa_in_proj_w, * sa_in_proj_b;
    ggml_tensor * sa_out_proj_w, * sa_out_proj_b;
    ggml_tensor * norm_q_w, * norm_q_b;
    ggml_tensor * norm_c_w, * norm_c_b;
    // Cross-attention
    ggml_tensor * ca_in_proj_w, * ca_in_proj_b;
    ggml_tensor * ca_out_proj_w, * ca_out_proj_b;
    ggml_tensor * norm1_w, * norm1_b;
    // FFN
    ggml_tensor * linear1_w, * linear1_b;
    ggml_tensor * linear2_w, * linear2_b;
    ggml_tensor * norm2_w, * norm2_b;
    // Final norm + head
    ggml_tensor * dec_norm_w, * dec_norm_b;
    ggml_tensor * head_w, * head_b;

    // Charset (94 printable ASCII)
    std::vector<std::string> charset;

    core_gguf::WeightLoad wl;
    ggml_backend_t backend;
    int n_threads;

    std::string result_buf;
    std::vector<float> encoder_output;  // [N * D]
    std::vector<float> char_confidences; // per-character softmax probabilities

    // Dequant cache for quantized weights
    std::map<const void *, std::vector<float>> dequant_cache;

    // Pre-allocated decoder scratch (avoids per-step heap allocs)
    struct dec_scratch {
        std::vector<float> ctx_emb, ctx_norm;      // [max_T * D]
        std::vector<float> pq, pq_norm;             // [D]
        std::vector<float> sa_Q, sa_K, sa_V;        // Q=[D], K/V=[max_T * D]
        std::vector<float> sa_out, sa_proj, h;      // [D]
        std::vector<float> h_norm, ca_Q, ca_out, ca_proj; // [D]
        std::vector<float> h_norm2, ff_up, ff_down; // ff_up=[ffn]
        std::vector<float> h_final, logits;         // logits=[V]
        bool allocated = false;
    } ds;

    bool bench;

    ggml_gallocr_t galloc = nullptr;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const float * tf32(parseq_ocr_context * ctx, ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    auto it = ctx->dequant_cache.find(t->data);
    if (it != ctx->dequant_cache.end()) return it->second.data();
    int64_t n = ggml_nelements(t);
    auto & buf = ctx->dequant_cache[t->data];
    buf.resize(n);
    const auto * traits = ggml_get_type_traits(t->type);
    if (traits->to_float) traits->to_float(t->data, buf.data(), n);
    else std::fill(buf.begin(), buf.end(), 0.0f);
    return buf.data();
}

static void layer_norm(const float * x, const float * w, const float * b,
                       int D, int T, float * out, float eps = 1e-6f) {
    for (int t = 0; t < T; t++) {
        const float * xt = x + t * D;
        float * ot = out + t * D;
        float mean = 0, var = 0;
        for (int d = 0; d < D; d++) mean += xt[d];
        mean /= D;
        for (int d = 0; d < D; d++) { float v = xt[d] - mean; var += v * v; }
        var /= D;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int d = 0; d < D; d++)
            ot[d] = (xt[d] - mean) * inv * w[d] + b[d];
    }
}

static float gelu_scalar(float x) {
    return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

static void softmax_row(float * x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// mat_vec: out[M] = W[M,K] @ x[K] + b[M]
static void mat_vec(const float * W, const float * x, const float * b,
                    int M, int K, float * out) {
    for (int m = 0; m < M; m++) {
        float s = b ? b[m] : 0.0f;
        for (int k = 0; k < K; k++) s += W[m * K + k] * x[k];
        out[m] = s;
    }
}

// matmul: C[M,N] = A[M,K] @ B[K,N]
static void matmul(const float * A, const float * B, float * C,
                   int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
}

// Multi-head attention: Q[Tq,D] K[Tk,D] V[Tk,D] → out[Tq,D]
static void mha_cpu(const float * Q, const float * K, const float * V,
                    int Tq, int Tk, int D, int n_heads,
                    const float * mask, // [Tq, Tk] or null, additive
                    float * out) {
    int hd = D / n_heads;
    float scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> scores(Tq * Tk);
    std::vector<float> attn_v(Tq * D);

    for (int h = 0; h < n_heads; h++) {
        // Compute scores for this head
        for (int qi = 0; qi < Tq; qi++)
            for (int ki = 0; ki < Tk; ki++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += Q[qi * D + h * hd + d] * K[ki * D + h * hd + d];
                s *= scale;
                if (mask) s += mask[qi * Tk + ki];
                scores[qi * Tk + ki] = s;
            }

        // Softmax per query
        for (int qi = 0; qi < Tq; qi++)
            softmax_row(&scores[qi * Tk], Tk);

        // Weighted sum of V
        for (int qi = 0; qi < Tq; qi++)
            for (int d = 0; d < hd; d++) {
                float s = 0;
                for (int ki = 0; ki < Tk; ki++)
                    s += scores[qi * Tk + ki] * V[ki * D + h * hd + d];
                out[qi * D + h * hd + d] = s;
            }
    }
}

// Linear: out[T,M] = x[T,K] @ W^T[K,M] + b[M] (W stored as [M,K])
static void linear_batch(const float * x, const float * W, const float * b,
                         int T, int K, int M, float * out) {
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < M; m++) {
            float s = b ? b[m] : 0.0f;
            for (int k = 0; k < K; k++) s += W[m * K + k] * x[t * K + k];
            out[t * M + m] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Init / Free
// ---------------------------------------------------------------------------

parseq_ocr_context * parseq_ocr_init(const char * model_path, int n_threads) {
    auto * ctx = new parseq_ocr_context();
    ctx->n_threads = n_threads;
    bool force_cpu = (getenv("PARSEQ_OCR_FORCE_CPU") && atoi(getenv("PARSEQ_OCR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    // Pass 1: read metadata
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "parseq: failed to open metadata from %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    auto & hp = ctx->hp;
    hp.embed_dim     = core_gguf::kv_i32(meta, "parseq.encoder.embed_dim", 384);
    hp.enc_layers    = core_gguf::kv_i32(meta, "parseq.encoder.num_layers", 12);
    hp.enc_heads     = core_gguf::kv_i32(meta, "parseq.encoder.num_heads", 6);
    hp.ffn_dim       = core_gguf::kv_i32(meta, "parseq.encoder.ffn_dim", 1536);
    hp.patch_h       = core_gguf::kv_i32(meta, "parseq.encoder.patch_h", 4);
    hp.patch_w       = core_gguf::kv_i32(meta, "parseq.encoder.patch_w", 8);
    hp.img_h         = core_gguf::kv_i32(meta, "parseq.encoder.img_h", 32);
    hp.img_w         = core_gguf::kv_i32(meta, "parseq.encoder.img_w", 128);
    hp.n_patches     = core_gguf::kv_i32(meta, "parseq.encoder.n_patches", 128);
    hp.dec_heads     = core_gguf::kv_i32(meta, "parseq.decoder.num_heads", 12);
    hp.dec_ffn       = core_gguf::kv_i32(meta, "parseq.decoder.ffn_dim", 1536);
    hp.max_label_len = core_gguf::kv_i32(meta, "parseq.decoder.max_label_len", 26);
    hp.vocab_size    = core_gguf::kv_i32(meta, "parseq.vocab_size", 95);
    hp.n_tokens      = core_gguf::kv_i32(meta, "parseq.n_tokens", 97);
    hp.bos_token     = core_gguf::kv_i32(meta, "parseq.bos_token", 0);
    hp.eos_token     = core_gguf::kv_i32(meta, "parseq.eos_token", 95);
    hp.pad_token     = core_gguf::kv_i32(meta, "parseq.pad_token", 96);

    fprintf(stderr, "parseq: embed=%d enc_layers=%d heads=%d ffn=%d patch=[%d,%d] "
            "img=[%d,%d] patches=%d\n",
            hp.embed_dim, hp.enc_layers, hp.enc_heads, hp.ffn_dim,
            hp.patch_h, hp.patch_w, hp.img_h, hp.img_w, hp.n_patches);

    // Load tokenizer
    {
        auto tokens = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        if (!tokens.empty()) {
            ctx->charset = std::move(tokens);
        } else {
            // Fallback: hardcoded charset
            ctx->charset.push_back("[B]"); // BOS
            const char * cs = "0123456789abcdefghijklmnopqrstuvwxyz"
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
            for (int i = 0; cs[i]; i++) ctx->charset.push_back(std::string(1, cs[i]));
            ctx->charset.push_back("[E]"); // EOS
            ctx->charset.push_back("[P]"); // PAD
        }
        fprintf(stderr, "parseq: charset=%zu tokens\n", ctx->charset.size());
    }

    core_gguf::free_metadata(meta);

    // Pass 2: load weights
    if (!core_gguf::load_weights(model_path, ctx->backend, "parseq", ctx->wl)) {
        fprintf(stderr, "parseq: failed to load weights from %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    // Map tensors
    auto T = [&](const char * name) -> ggml_tensor * {
        return core_gguf::try_get(ctx->wl.tensors, name);
    };

    // Encoder
    ctx->patch_proj_w = T("encoder.patch_embed.proj.weight");
    ctx->patch_proj_b = T("encoder.patch_embed.proj.bias");
    ctx->pos_embed    = T("encoder.pos_embed");
    ctx->enc_norm_w   = T("encoder.norm.weight");
    ctx->enc_norm_b   = T("encoder.norm.bias");

    ctx->enc_layers.resize(hp.enc_layers);
    for (int i = 0; i < hp.enc_layers; i++) {
        char buf[128];
        auto P = [&](const char * suffix) -> ggml_tensor * {
            snprintf(buf, sizeof(buf), "encoder.blocks.%d.%s", i, suffix);
            return T(buf);
        };
        auto & L = ctx->enc_layers[i];
        L.ln1_w  = P("norm1.weight");  L.ln1_b  = P("norm1.bias");
        L.qkv_w  = P("attn.qkv.weight"); L.qkv_b = P("attn.qkv.bias");
        L.proj_w = P("attn.proj.weight"); L.proj_b = P("attn.proj.bias");
        L.ln2_w  = P("norm2.weight");  L.ln2_b  = P("norm2.bias");
        L.fc1_w  = P("mlp.fc1.weight"); L.fc1_b = P("mlp.fc1.bias");
        L.fc2_w  = P("mlp.fc2.weight"); L.fc2_b = P("mlp.fc2.bias");
    }

    // Decoder
    ctx->pos_queries    = T("pos_queries");
    ctx->text_embed_w   = T("text_embed.embedding.weight");
    ctx->sa_in_proj_w   = T("decoder.layers.0.self_attn.in_proj_weight");
    ctx->sa_in_proj_b   = T("decoder.layers.0.self_attn.in_proj_bias");
    ctx->sa_out_proj_w  = T("decoder.layers.0.self_attn.out_proj.weight");
    ctx->sa_out_proj_b  = T("decoder.layers.0.self_attn.out_proj.bias");
    ctx->norm_q_w       = T("decoder.layers.0.norm_q.weight");
    ctx->norm_q_b       = T("decoder.layers.0.norm_q.bias");
    ctx->norm_c_w       = T("decoder.layers.0.norm_c.weight");
    ctx->norm_c_b       = T("decoder.layers.0.norm_c.bias");
    ctx->ca_in_proj_w   = T("decoder.layers.0.cross_attn.in_proj_weight");
    ctx->ca_in_proj_b   = T("decoder.layers.0.cross_attn.in_proj_bias");
    ctx->ca_out_proj_w  = T("decoder.layers.0.cross_attn.out_proj.weight");
    ctx->ca_out_proj_b  = T("decoder.layers.0.cross_attn.out_proj.bias");
    ctx->norm1_w        = T("decoder.layers.0.norm1.weight");
    ctx->norm1_b        = T("decoder.layers.0.norm1.bias");
    ctx->linear1_w      = T("decoder.layers.0.linear1.weight");
    ctx->linear1_b      = T("decoder.layers.0.linear1.bias");
    ctx->linear2_w      = T("decoder.layers.0.linear2.weight");
    ctx->linear2_b      = T("decoder.layers.0.linear2.bias");
    ctx->norm2_w        = T("decoder.layers.0.norm2.weight");
    ctx->norm2_b        = T("decoder.layers.0.norm2.bias");
    ctx->dec_norm_w     = T("decoder.norm.weight");
    ctx->dec_norm_b     = T("decoder.norm.bias");
    ctx->head_w         = T("head.weight");
    ctx->head_b         = T("head.bias");

    ctx->bench = (std::getenv("CRISPEMBED_PARSEQ_BENCH") != nullptr);

    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));

    return ctx;
}

void parseq_ocr_free(parseq_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

const parseq_ocr_hparams * parseq_ocr_get_hparams(const parseq_ocr_context * ctx) {
    return &ctx->hp;
}

// ---------------------------------------------------------------------------
// Encoder (ggml graph)
// ---------------------------------------------------------------------------

static bool run_encoder(parseq_ocr_context * ctx, const float * pixels) {
    const auto & hp = ctx->hp;
    const int D = hp.embed_dim;
    const int N = hp.n_patches;  // 128
    const int nh = hp.enc_heads;
    const int hd = D / nh;
    const float eps = 1e-6f;

    // Patch embedding is Conv2d with non-square kernel [4,8] — not supported
    // by ggml_conv_2d (square only). Run it CPU-side, then use ggml graph
    // for the 12 transformer layers.
    const int ph = hp.patch_h, pw = hp.patch_w;
    const int oh = hp.img_h / ph;  // 8
    const int ow = hp.img_w / pw;  // 16
    const int ic = 3;

    // patch_proj_w is stored as [D, ic*ph*pw] (flattened from [D, 3, 4, 8])
    const float * pw_data = tf32(ctx, ctx->patch_proj_w);
    const float * pb_data = tf32(ctx, ctx->patch_proj_b);
    const int patch_len = ic * ph * pw;  // 3*4*8 = 96

    // CPU patch embedding: extract patches, matmul with weight
    std::vector<float> patch_out(N * D);  // [N, D] row-major
    for (int py = 0; py < oh; py++) {
        for (int px = 0; px < ow; px++) {
            int tok = py * ow + px;
            // Extract patch [ic, ph, pw] → flat [ic*ph*pw]
            float patch[96 * 4];  // max patch_len
            for (int c = 0; c < ic; c++)
                for (int y = 0; y < ph; y++)
                    for (int x = 0; x < pw; x++)
                        patch[c * ph * pw + y * pw + x] =
                            pixels[c * hp.img_h * hp.img_w +
                                   (py * ph + y) * hp.img_w + (px * pw + x)];

            // dot product with weight: out[d] = sum_k(W[d,k] * patch[k]) + b[d]
            for (int d = 0; d < D; d++) {
                float s = pb_data[d];
                for (int k = 0; k < patch_len; k++)
                    s += pw_data[d * patch_len + k] * patch[k];
                patch_out[tok * D + d] = s;
            }
        }
    }

    // Add position embedding
    const float * pe = tf32(ctx, ctx->pos_embed);
    // pos_embed in ggml is [D, 1, N] or [D, N] depending on squeeze.
    // In GGUF: stored as [1, 128, 384] → ggml ne=[384, 128, 1] → data is [N * D]
    // Row t of pos_embed = pe[t*D .. (t+1)*D]
    for (int t = 0; t < N; t++)
        for (int d = 0; d < D; d++)
            patch_out[t * D + d] += pe[t * D + d];

    // Build ggml graph for encoder transformer layers
    // Layout: x is [D, N] in ggml (ne[0]=D, ne[1]=N)
    const int n_tensors = hp.enc_layers * 30 + 20;
    const size_t buf_sz = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false);
    ggml_init_params ip = {buf_sz, nullptr, true};
    ggml_context * g = ggml_init(ip);

    // Input tensor
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, N);
    ggml_set_name(x, "enc_input");
    ggml_set_input(x);

    // Encoder layers
    for (int il = 0; il < hp.enc_layers; il++) {
        const auto & L = ctx->enc_layers[il];
        ggml_tensor * residual = x;

        // Pre-LN
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, L.ln1_w);
        x = ggml_add(g, x, L.ln1_b);

        // Fused QKV
        ggml_tensor * qkv = ggml_mul_mat(g, L.qkv_w, x);  // [3D, N]
        qkv = ggml_add(g, qkv, L.qkv_b);

        ggml_tensor * Q = ggml_cont(g, ggml_view_2d(g, qkv, D, N, qkv->nb[1], 0));
        ggml_tensor * K = ggml_cont(g, ggml_view_2d(g, qkv, D, N, qkv->nb[1], D * ggml_type_size(qkv->type)));
        ggml_tensor * V = ggml_cont(g, ggml_view_2d(g, qkv, D, N, qkv->nb[1], 2 * D * ggml_type_size(qkv->type)));

        // Reshape for MHA: [D, N] → [hd, nh, N] → permute to [hd, N, nh]
        Q = ggml_reshape_3d(g, Q, hd, nh, N);
        K = ggml_reshape_3d(g, K, hd, nh, N);
        V = ggml_reshape_3d(g, V, hd, nh, N);
        Q = ggml_permute(g, Q, 0, 2, 1, 3);
        K = ggml_permute(g, K, 0, 2, 1, 3);
        V = ggml_permute(g, V, 0, 2, 1, 3);

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor * attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, N);

        // Output projection
        attn = ggml_mul_mat(g, L.proj_w, attn);
        attn = ggml_add(g, attn, L.proj_b);

        x = ggml_add(g, residual, attn);

        // Pre-FFN LN
        residual = x;
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, L.ln2_w);
        x = ggml_add(g, x, L.ln2_b);

        // FFN: fc1 → GELU → fc2
        x = ggml_mul_mat(g, L.fc1_w, x);
        x = ggml_add(g, x, L.fc1_b);
        x = ggml_gelu(g, x);
        x = ggml_mul_mat(g, L.fc2_w, x);
        x = ggml_add(g, x, L.fc2_b);

        x = ggml_add(g, residual, x);
    }

    // Final LayerNorm
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, ctx->enc_norm_w);
    x = ggml_add(g, x, ctx->enc_norm_b);

    ggml_set_name(x, "enc_output");
    ggml_set_output(x);

    // Build and compute graph
    ggml_cgraph * gf_graph = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(gf_graph, x);

    ggml_gallocr_alloc_graph(ctx->galloc, gf_graph);

    // Set input
    ggml_tensor * inp = ggml_graph_get_tensor(gf_graph, "enc_input");
    // patch_out is [N, D] row-major. ggml expects [D, N] col-major = same layout.
    ggml_backend_tensor_set(inp, patch_out.data(), 0, N * D * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf_graph);

    // Read output
    ggml_tensor * out = ggml_graph_get_tensor(gf_graph, "enc_output");
    ctx->encoder_output.resize(N * D);
    ggml_backend_tensor_get(out, ctx->encoder_output.data(), 0, N * D * sizeof(float));

    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// Decoder (CPU scalar — single layer, not worth ggml graph overhead)
// ---------------------------------------------------------------------------

static std::string run_decoder_ar(parseq_ocr_context * ctx) {
    ctx->char_confidences.clear();
    const auto & hp = ctx->hp;
    const int D = hp.embed_dim;
    const int N = hp.n_patches;
    const int num_steps = hp.max_label_len;  // 26 (max_label_length + 1 for EOS)
    const int dec_heads = hp.dec_heads;
    const float embed_scale = std::sqrt((float)D);

    // Dequant all decoder weights upfront
    const float * pos_q     = tf32(ctx, ctx->pos_queries);
    const float * embed_w   = tf32(ctx, ctx->text_embed_w);
    const float * sa_w      = tf32(ctx, ctx->sa_in_proj_w);
    const float * sa_b      = tf32(ctx, ctx->sa_in_proj_b);
    const float * sa_ow     = tf32(ctx, ctx->sa_out_proj_w);
    const float * sa_ob     = tf32(ctx, ctx->sa_out_proj_b);
    const float * nq_w      = tf32(ctx, ctx->norm_q_w);
    const float * nq_b      = tf32(ctx, ctx->norm_q_b);
    const float * nc_w      = tf32(ctx, ctx->norm_c_w);
    const float * nc_b      = tf32(ctx, ctx->norm_c_b);
    const float * ca_w      = tf32(ctx, ctx->ca_in_proj_w);
    const float * ca_b      = tf32(ctx, ctx->ca_in_proj_b);
    const float * ca_ow     = tf32(ctx, ctx->ca_out_proj_w);
    const float * ca_ob     = tf32(ctx, ctx->ca_out_proj_b);
    const float * n1_w      = tf32(ctx, ctx->norm1_w);
    const float * n1_b      = tf32(ctx, ctx->norm1_b);
    const float * ff1_w     = tf32(ctx, ctx->linear1_w);
    const float * ff1_b     = tf32(ctx, ctx->linear1_b);
    const float * ff2_w     = tf32(ctx, ctx->linear2_w);
    const float * ff2_b     = tf32(ctx, ctx->linear2_b);
    const float * n2_w      = tf32(ctx, ctx->norm2_w);
    const float * n2_b      = tf32(ctx, ctx->norm2_b);
    const float * dn_w      = tf32(ctx, ctx->dec_norm_w);
    const float * dn_b      = tf32(ctx, ctx->dec_norm_b);
    const float * hd_w      = tf32(ctx, ctx->head_w);
    const float * hd_b      = tf32(ctx, ctx->head_b);
    const float * memory    = ctx->encoder_output.data();

    // PARSeq token ordering: EOS=0, chars=1..94, BOS=95, PAD=96
    // Head outputs 95 classes (0..94): class 0=EOS, classes 1..94=chars
    const int eos_head = 0;  // EOS is index 0 in head output
    const int bos_id = hp.bos_token;  // 95
    const int ffn = hp.dec_ffn;

    std::string result;

    // tgt_in stores token IDs in embedding space (0..96)
    std::vector<int> tgt_in(num_steps, hp.pad_token);
    tgt_in[0] = bos_id;  // 95

    // Pre-allocate decoder scratch buffers
    {
        auto & ds = ctx->ds;
        ds.ctx_emb.resize(num_steps * D);
        ds.ctx_norm.resize(num_steps * D);
        ds.pq.resize(D);
        ds.pq_norm.resize(D);
        ds.sa_Q.resize(D);
        ds.sa_K.resize(num_steps * D);
        ds.sa_V.resize(num_steps * D);
        ds.sa_out.resize(D);
        ds.sa_proj.resize(D);
        ds.h.resize(D);
        ds.h_norm.resize(D);
        ds.ca_Q.resize(D);
        ds.ca_out.resize(D);
        ds.ca_proj.resize(D);
        ds.h_norm2.resize(D);
        ds.ff_up.resize(ffn);
        ds.ff_down.resize(D);
        ds.h_final.resize(D);
        ds.logits.resize(hp.vocab_size);
        ds.allocated = true;
    }
    auto & ds = ctx->ds;

    // Pre-compute cross-attention K/V from memory (constant across steps)
    const bool bench = ctx->bench;
    auto t_dec_kv = std::chrono::steady_clock::now();
    std::vector<float> ca_K(N * D), ca_V(N * D);
    linear_batch(memory, ca_w + D * D, ca_b + D, N, D, D, ca_K.data());
    linear_batch(memory, ca_w + 2 * D * D, ca_b + 2 * D, N, D, D, ca_V.data());
    if (bench) fprintf(stderr, "[parseq-bench] decoder CA K/V: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_dec_kv).count());

    for (int i = 0; i < num_steps; i++) {
        const int j = i + 1;
        const int T = j;

        // Build context embeddings
        for (int t = 0; t < T; t++) {
            int tid = tgt_in[t];
            for (int d = 0; d < D; d++) {
                float e = embed_w[tid * D + d] * embed_scale;
                if (t > 0) e += pos_q[(t - 1) * D + d];
                ds.ctx_emb[t * D + d] = e;
            }
        }

        // Query: position i
        for (int d = 0; d < D; d++)
            ds.pq[d] = pos_q[i * D + d];

        // Self-attention: Q from norm_q(pq), K/V from norm_c(ctx_emb)
        layer_norm(ds.pq.data(), nq_w, nq_b, D, 1, ds.pq_norm.data());
        layer_norm(ds.ctx_emb.data(), nc_w, nc_b, D, T, ds.ctx_norm.data());

        linear_batch(ds.pq_norm.data(), sa_w, sa_b, 1, D, D, ds.sa_Q.data());
        linear_batch(ds.ctx_norm.data(), sa_w + D * D, sa_b + D, T, D, D, ds.sa_K.data());
        linear_batch(ds.ctx_norm.data(), sa_w + 2 * D * D, sa_b + 2 * D, T, D, D, ds.sa_V.data());

        mha_cpu(ds.sa_Q.data(), ds.sa_K.data(), ds.sa_V.data(), 1, T, D, dec_heads,
                nullptr, ds.sa_out.data());

        linear_batch(ds.sa_out.data(), sa_ow, sa_ob, 1, D, D, ds.sa_proj.data());

        // Residual: h = pq + sa_proj
        for (int d = 0; d < D; d++) ds.h[d] = ds.pq[d] + ds.sa_proj[d];

        // Cross-attention: Q from norm1(h), K/V from memory
        layer_norm(ds.h.data(), n1_w, n1_b, D, 1, ds.h_norm.data());
        linear_batch(ds.h_norm.data(), ca_w, ca_b, 1, D, D, ds.ca_Q.data());

        mha_cpu(ds.ca_Q.data(), ca_K.data(), ca_V.data(), 1, N, D, dec_heads,
                nullptr, ds.ca_out.data());

        linear_batch(ds.ca_out.data(), ca_ow, ca_ob, 1, D, D, ds.ca_proj.data());
        for (int d = 0; d < D; d++) ds.h[d] += ds.ca_proj[d];

        // FFN: norm2 → linear1 → GELU → linear2
        layer_norm(ds.h.data(), n2_w, n2_b, D, 1, ds.h_norm2.data());
        linear_batch(ds.h_norm2.data(), ff1_w, ff1_b, 1, D, ffn, ds.ff_up.data());
        for (int f = 0; f < ffn; f++) ds.ff_up[f] = gelu_scalar(ds.ff_up[f]);
        linear_batch(ds.ff_up.data(), ff2_w, ff2_b, 1, ffn, D, ds.ff_down.data());
        for (int d = 0; d < D; d++) ds.h[d] += ds.ff_down[d];

        // Final norm + head
        layer_norm(ds.h.data(), dn_w, dn_b, D, 1, ds.h_final.data());
        linear_batch(ds.h_final.data(), hd_w, hd_b, 1, D, hp.vocab_size, ds.logits.data());

        // Softmax + argmax
        float max_logit = ds.logits[0];
        for (int k = 1; k < hp.vocab_size; k++)
            if (ds.logits[k] > max_logit) max_logit = ds.logits[k];
        float sum_exp = 0;
        for (int k = 0; k < hp.vocab_size; k++) {
            ds.logits[k] = expf(ds.logits[k] - max_logit);
            sum_exp += ds.logits[k];
        }
        for (int k = 0; k < hp.vocab_size; k++) ds.logits[k] /= sum_exp;

        int pred = 0;
        float best_prob = ds.logits[0];
        for (int k = 1; k < hp.vocab_size; k++) {
            if (ds.logits[k] > best_prob) { best_prob = ds.logits[k]; pred = k; }
        }

        // Head output: 0=EOS, 1..94=chars
        if (pred == eos_head) break;

        // Map to character: pred 1..94 → charset index pred (in PARSEQ_TOKENS)
        // charset[0]="[E]", charset[1..94]=chars, charset[95]="[B]", charset[96]="[P]"
        if (pred >= 1 && pred <= 94 && pred < (int)ctx->charset.size()) {
            result += ctx->charset[pred];
            ctx->char_confidences.push_back(best_prob);
        }

        // Feed prediction back as next context token
        // Head output index maps directly to embedding index (both start with EOS=0)
        if (j < num_steps) {
            tgt_in[j] = pred;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

static void preprocess_rgb_to_input(const uint8_t * rgb, int w, int h, int ch,
                                    int target_w, int target_h,
                                    std::vector<float> & out) {
    // Resize to target_w × target_h, normalize to [-1, 1]
    out.resize(3 * target_h * target_w);

    for (int c = 0; c < 3; c++) {
        for (int ty = 0; ty < target_h; ty++) {
            for (int tx = 0; tx < target_w; tx++) {
                // Nearest-neighbor resize
                int sy = ty * h / target_h;
                int sx = tx * w / target_w;
                if (sy >= h) sy = h - 1;
                if (sx >= w) sx = w - 1;

                uint8_t val;
                if (ch == 1) {
                    val = rgb[sy * w + sx];
                } else if (ch == 3) {
                    val = rgb[(sy * w + sx) * 3 + c];
                } else if (ch == 4) {
                    val = rgb[(sy * w + sx) * 4 + c];
                } else {
                    val = 128;
                }

                // Normalize to [-1, 1]
                float f = (float)val / 255.0f;
                out[c * target_h * target_w + ty * target_w + tx] = f * 2.0f - 1.0f;
            }
        }
    }
}

static void preprocess_gray_to_input(const float * gray, int w, int h,
                                     int target_w, int target_h,
                                     std::vector<float> & out) {
    // gray is [0,1]. Resize and replicate to 3 channels, normalize to [-1,1].
    out.resize(3 * target_h * target_w);
    for (int c = 0; c < 3; c++) {
        for (int ty = 0; ty < target_h; ty++) {
            for (int tx = 0; tx < target_w; tx++) {
                int sy = ty * h / target_h;
                int sx = tx * w / target_w;
                if (sy >= h) sy = h - 1;
                if (sx >= w) sx = w - 1;
                float val = gray[sy * w + sx];
                out[c * target_h * target_w + ty * target_w + tx] = val * 2.0f - 1.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const char * parseq_ocr_recognize(
    parseq_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len
) {
    if (!ctx || !pixels) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> input;
    preprocess_gray_to_input(pixels, width, height,
                             ctx->hp.img_w, ctx->hp.img_h, input);
    if (bench) fprintf(stderr, "[parseq-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    if (!run_encoder(ctx, input.data())) return nullptr;
    if (bench) fprintf(stderr, "[parseq-bench] encoder graph: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    ctx->result_buf = run_decoder_ar(ctx);
    if (bench) fprintf(stderr, "[parseq-bench] decoder total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[parseq-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * parseq_ocr_recognize_raw(
    parseq_ocr_context * ctx,
    const uint8_t * pixel_bytes, int width, int height, int channels,
    int * out_len
) {
    if (!ctx || !pixel_bytes) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> input;
    preprocess_rgb_to_input(pixel_bytes, width, height, channels,
                            ctx->hp.img_w, ctx->hp.img_h, input);
    if (bench) fprintf(stderr, "[parseq-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    if (!run_encoder(ctx, input.data())) return nullptr;
    if (bench) fprintf(stderr, "[parseq-bench] encoder graph: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    ctx->result_buf = run_decoder_ar(ctx);
    if (bench) fprintf(stderr, "[parseq-bench] decoder total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[parseq-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float * parseq_ocr_confidences(const parseq_ocr_context * ctx, int * n_chars) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float parseq_ocr_mean_confidence(const parseq_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
