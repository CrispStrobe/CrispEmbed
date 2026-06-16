// lightonocr.cpp — LightOnOCR-2-1B inference engine.
//
// Pixtral ViT (24L, 2D RoPE) + Qwen3 decoder (28L, QK norm, GQA).
// Single GGUF, CPU-only via ggml_backend_sched.

#include "lightonocr.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// stbi_load/stbi_image_free are provided with C linkage by image_preprocess.cpp's
// STB_IMAGE_IMPLEMENTATION. Forward-declare them here (matching ocr_orchestrator.cpp)
// instead of including stb_image.h — `#define STB_IMAGE_STATIC` + include declared
// them static-without-definition in this TU → undefined-internal link error.
extern "C" {
unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file,
                         int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
}

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightonocr {

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

struct vis_layer {
    ggml_tensor *q_w, *k_w, *v_w, *o_w;        // 1024×1024 each
    ggml_tensor *attn_norm_w;                     // RMSNorm
    ggml_tensor *gate_w, *up_w, *down_w;         // SiLU FFN
    ggml_tensor *ffn_norm_w;                      // RMSNorm
};

struct lm_layer {
    ggml_tensor *q_w, *k_w, *v_w, *o_w;         // Q: 2048×1024, KV: 1024×1024
    ggml_tensor *q_norm_w, *k_norm_w;            // QK norm (head_dim)
    ggml_tensor *attn_norm_w;                     // RMSNorm
    ggml_tensor *gate_w, *up_w, *down_w;         // SwiGLU FFN
    ggml_tensor *ffn_norm_w;                      // RMSNorm
};

struct model {
    // Vision hparams
    int vis_layers, vis_dim, vis_heads, vis_head_dim, vis_inter;
    int patch_size, image_size;
    float vis_rope_theta;

    // LM hparams
    int lm_layers, lm_dim, lm_heads, lm_kv_heads, lm_head_dim, lm_inter;
    int vocab_size;
    float lm_rms_eps, lm_rope_theta;
    bool use_qk_norm;

    // General
    int spatial_merge_size;
    int image_token_id, eos_token_id, pad_token_id;

    // Vision weights
    ggml_tensor *patch_conv_w;          // [1024, 3, 14, 14]
    ggml_tensor *ln_pre_w;             // [1024]
    std::vector<vis_layer> vis;

    // Projection
    ggml_tensor *proj_merger_w;         // [1024, 4096]
    ggml_tensor *proj_linear1_w;        // [1024, 1024]
    ggml_tensor *proj_linear2_w;        // [1024, 1024]
    ggml_tensor *proj_norm_w;           // [1024]

    // LM decoder
    ggml_tensor *embed_tokens;          // [vocab, dim]
    ggml_tensor *lm_norm_w;            // [dim]
    std::vector<lm_layer> lm;
};

struct context {
    model m;
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    ggml_backend_sched_t sched = nullptr;
    std::vector<char> compute_meta;
    int n_threads = 4;
    int max_tokens = 2048;
    std::string last_text;

    // Tokenizer
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool load(context &ctx, const char *gguf_path, int n_threads) {
    ctx.n_threads = n_threads;

    // Pass 1: metadata
    gguf_context *gctx = core_gguf::open_metadata(gguf_path);
    if (!gctx) return false;

    auto &m = ctx.m;
    m.vis_layers    = core_gguf::kv_u32(gctx, "lightonocr.vision.num_hidden_layers", 24);
    m.vis_dim       = core_gguf::kv_u32(gctx, "lightonocr.vision.hidden_size", 1024);
    m.vis_heads     = core_gguf::kv_u32(gctx, "lightonocr.vision.num_attention_heads", 16);
    m.vis_head_dim  = core_gguf::kv_u32(gctx, "lightonocr.vision.head_dim", 64);
    m.vis_inter     = core_gguf::kv_u32(gctx, "lightonocr.vision.intermediate_size", 4096);
    m.patch_size    = core_gguf::kv_u32(gctx, "lightonocr.vision.patch_size", 14);
    m.image_size    = core_gguf::kv_u32(gctx, "lightonocr.vision.image_size", 1540);
    m.vis_rope_theta = core_gguf::kv_f32(gctx, "lightonocr.vision.rope_theta", 10000.0f);

    m.lm_layers     = core_gguf::kv_u32(gctx, "lightonocr.text.num_hidden_layers", 28);
    m.lm_dim        = core_gguf::kv_u32(gctx, "lightonocr.text.hidden_size", 1024);
    m.lm_heads      = core_gguf::kv_u32(gctx, "lightonocr.text.num_attention_heads", 16);
    m.lm_kv_heads   = core_gguf::kv_u32(gctx, "lightonocr.text.num_key_value_heads", 8);
    m.lm_head_dim   = core_gguf::kv_u32(gctx, "lightonocr.text.head_dim", 128);
    m.lm_inter      = core_gguf::kv_u32(gctx, "lightonocr.text.intermediate_size", 3072);
    m.vocab_size    = core_gguf::kv_u32(gctx, "lightonocr.text.vocab_size", 151936);
    m.lm_rms_eps    = core_gguf::kv_f32(gctx, "lightonocr.text.rms_norm_eps", 1e-6f);
    m.lm_rope_theta = core_gguf::kv_f32(gctx, "lightonocr.text.rope_theta", 1000000.0f);
    m.use_qk_norm   = core_gguf::kv_bool(gctx, "lightonocr.text.use_qk_norm", true);

    m.spatial_merge_size = core_gguf::kv_u32(gctx, "lightonocr.spatial_merge_size", 2);
    m.image_token_id = core_gguf::kv_u32(gctx, "lightonocr.image_token_id", 151655);
    m.eos_token_id   = core_gguf::kv_u32(gctx, "lightonocr.eos_token_id", 151645);
    m.pad_token_id   = core_gguf::kv_u32(gctx, "lightonocr.pad_token_id", 151643);

    // Read tokenizer vocab
    ctx.vocab = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
    for (int i = 0; i < (int)ctx.vocab.size(); i++)
        ctx.token_to_id[ctx.vocab[i]] = i;

    core_gguf::free_metadata(gctx);

    fprintf(stderr, "lightonocr: vis=%dL/%dd, lm=%dL/%dd, vocab=%d\n",
            m.vis_layers, m.vis_dim, m.lm_layers, m.lm_dim, m.vocab_size);

    // Pass 2: weights
    ctx.backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(gguf_path, ctx.backend, "lightonocr", ctx.wl)) {
        fprintf(stderr, "lightonocr: failed to load weights\n");
        ggml_backend_free(ctx.backend);
        return false;
    }

    auto get = [&](const char *name) -> ggml_tensor* {
        return core_gguf::try_get(ctx.wl.tensors, name);
    };

    // Vision
    m.patch_conv_w = get("vis.patch_conv.weight");
    m.ln_pre_w     = get("vis.ln_pre.weight");
    m.vis.resize(m.vis_layers);
    for (int i = 0; i < m.vis_layers; i++) {
        char pfx[32]; snprintf(pfx, sizeof(pfx), "vis.blk.%d.", i);
        auto k = [&](const char *s) { return get((std::string(pfx) + s).c_str()); };
        auto &L = m.vis[i];
        L.q_w = k("attn.q_proj.weight"); L.k_w = k("attn.k_proj.weight");
        L.v_w = k("attn.v_proj.weight"); L.o_w = k("attn.o_proj.weight");
        L.attn_norm_w = k("attn_norm.weight");
        L.gate_w = k("ffn.gate_proj.weight"); L.up_w = k("ffn.up_proj.weight");
        L.down_w = k("ffn.down_proj.weight");
        L.ffn_norm_w = k("ffn_norm.weight");
    }

    // Projection
    m.proj_merger_w  = get("proj.patch_merger.merging_layer.weight");
    m.proj_linear1_w = get("proj.linear_1.weight");
    m.proj_linear2_w = get("proj.linear_2.weight");
    m.proj_norm_w    = get("proj.norm.weight");

    // LM decoder
    m.embed_tokens = get("lm.embed.weight");
    m.lm_norm_w    = get("lm.norm.weight");
    m.lm.resize(m.lm_layers);
    for (int i = 0; i < m.lm_layers; i++) {
        char pfx[32]; snprintf(pfx, sizeof(pfx), "lm.blk.%d.", i);
        auto k = [&](const char *s) { return get((std::string(pfx) + s).c_str()); };
        auto &L = m.lm[i];
        L.q_w = k("attn.q_proj.weight"); L.k_w = k("attn.k_proj.weight");
        L.v_w = k("attn.v_proj.weight"); L.o_w = k("attn.o_proj.weight");
        L.q_norm_w = k("attn.q_norm.weight"); L.k_norm_w = k("attn.k_norm.weight");
        L.attn_norm_w = k("attn_norm.weight");
        L.gate_w = k("ffn.gate_proj.weight"); L.up_w = k("ffn.up_proj.weight");
        L.down_w = k("ffn.down_proj.weight");
        L.ffn_norm_w = k("ffn_norm.weight");
    }

    if (!m.patch_conv_w || !m.embed_tokens) {
        fprintf(stderr, "lightonocr: missing critical tensors\n");
        return false;
    }

    // Scheduler
    ctx.compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    ctx.sched = ggml_backend_sched_new(&ctx.backend, nullptr, 1, 16384, false, false);

    return true;
}

void free_(context &ctx) {
    if (ctx.sched) ggml_backend_sched_free(ctx.sched);
    core_gguf::free_weights(ctx.wl);
    if (ctx.backend) ggml_backend_free(ctx.backend);
}

// ---------------------------------------------------------------------------
// Image preprocessing (simplified — resize + normalize + patchify)
// ---------------------------------------------------------------------------

static std::vector<float> preprocess_image(const uint8_t *rgb, int w, int h,
                                             int patch_size, int max_size,
                                             int *out_ph, int *out_pw) {
    // Resize to fit max_size while preserving aspect ratio, then pad to patch grid
    float scale = std::min((float)max_size / w, (float)max_size / h);
    if (scale > 1.0f) scale = 1.0f;  // don't upscale
    int rw = (int)(w * scale);
    int rh = (int)(h * scale);
    // Round up to patch grid
    int pw = (rw + patch_size - 1) / patch_size;
    int ph = (rh + patch_size - 1) / patch_size;
    int tw = pw * patch_size;
    int th = ph * patch_size;

    // Bilinear resize + normalize (ImageNet mean/std)
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    std::vector<float> pixels(3 * th * tw, 0.0f);
    for (int y = 0; y < rh; y++) {
        float sy = (float)y * h / rh;
        int y0 = std::min((int)sy, h - 1);
        for (int x = 0; x < rw; x++) {
            float sx = (float)x * w / rw;
            int x0 = std::min((int)sx, w - 1);
            for (int c = 0; c < 3; c++) {
                float val = rgb[(y0 * w + x0) * 3 + c] / 255.0f;
                pixels[c * th * tw + y * tw + x] = (val - mean[c]) / std_[c];
            }
        }
    }

    *out_ph = ph;
    *out_pw = pw;
    return pixels;
}

// ---------------------------------------------------------------------------
// 2D RoPE: compute cos/sin for Pixtral's interleaved h/w frequencies
// ---------------------------------------------------------------------------

static void compute_2d_rope(int ph, int pw, int head_dim, float theta,
                             std::vector<float> &cos_out,
                             std::vector<float> &sin_out) {
    int T = ph * pw;
    int half = head_dim / 2;
    cos_out.resize(head_dim * T);
    sin_out.resize(head_dim * T);

    // Pixtral: even freq indices → height, odd → width
    // freq[i] = 1 / (theta^(i / dim))
    std::vector<float> freqs(half);
    for (int i = 0; i < half; i++)
        freqs[i] = 1.0f / powf(theta, (float)(2 * i) / head_dim);

    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            int t = y * pw + x;
            for (int i = 0; i < half; i++) {
                // Even indices: height position, odd: width position
                float angle_h = y * freqs[i];
                float angle_w = x * freqs[i];
                // Interleave: dim[2i] = height, dim[2i+1] = width
                cos_out[t * head_dim + 2 * i + 0] = cosf(angle_h);
                sin_out[t * head_dim + 2 * i + 0] = sinf(angle_h);
                cos_out[t * head_dim + 2 * i + 1] = cosf(angle_w);
                sin_out[t * head_dim + 2 * i + 1] = sinf(angle_w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vision encoder: Pixtral ViT with 2D RoPE
// ---------------------------------------------------------------------------

// Apply patch_conv via CPU (Conv2d 14x14 stride 14, no bias)
static std::vector<float> apply_patch_conv(context &ctx,
                                             const std::vector<float> &pixels,
                                             int th, int tw, int ph, int pw) {
    const auto &m = ctx.m;
    int D = m.vis_dim;
    int P = m.patch_size;
    int T = ph * pw;

    // patch_conv weight: [D, 3, P, P] in ggml = ne[0]=P, ne[1]=P, ne[2]=3, ne[3]=D
    // Read weight from backend
    std::vector<float> w(D * 3 * P * P);
    ggml_backend_tensor_get(m.patch_conv_w, w.data(), 0, w.size() * sizeof(float));

    // Conv2d with stride P: for each output patch (y,x) and channel d:
    // out[d, y, x] = sum over c,ky,kx of w[d,c,ky,kx] * pixels[c, y*P+ky, x*P+kx]
    std::vector<float> out(D * T, 0.0f);
    for (int d = 0; d < D; d++) {
        for (int py = 0; py < ph; py++) {
            for (int px = 0; px < pw; px++) {
                float sum = 0.0f;
                for (int c = 0; c < 3; c++) {
                    for (int ky = 0; ky < P; ky++) {
                        for (int kx = 0; kx < P; kx++) {
                            int iy = py * P + ky;
                            int ix = px * P + kx;
                            if (iy < th && ix < tw) {
                                float pix = pixels[c * th * tw + iy * tw + ix];
                                // Weight layout: ggml stores as [P, P, 3, D]
                                // Index: d * (3*P*P) + c * (P*P) + ky * P + kx
                                float wt = w[d * (3 * P * P) + c * (P * P) + ky * P + kx];
                                sum += pix * wt;
                            }
                        }
                    }
                }
                // ggml layout: (D, T) where fast axis = D
                out[py * pw * D + px * D + d] = sum;
            }
        }
    }

    // Reshape to (D, T) row-major for ggml
    std::vector<float> result(D * T);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            result[t * D + d] = out[t * D + d];
    return result;
}

// Build vision encoder ggml graph (24 transformer layers)
static bool run_vision_encoder(context &ctx,
                                const std::vector<float> &patch_embeds,
                                const std::vector<float> &rope_cos,
                                const std::vector<float> &rope_sin,
                                int n_patches,
                                std::vector<float> &out) {
    auto &m = ctx.m;
    const int D = m.vis_dim;
    const int NH = m.vis_heads;
    const int HD = m.vis_head_dim;
    const float attn_scale = 1.0f / sqrtf((float)HD);

    ggml_init_params ip{ctx.compute_meta.size(), ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Inputs
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_patches);
    ggml_set_name(x, "vis_input"); ggml_set_input(x);

    // (HD, 1, n_patches) — broadcasts over NH heads in ggml_mul
    ggml_tensor *cos_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, HD, 1, n_patches);
    ggml_tensor *sin_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, HD, 1, n_patches);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    // ln_pre (RMSNorm before transformer blocks)
    if (m.ln_pre_w) {
        x = ggml_rms_norm(g, x, 1e-6f);
        x = ggml_mul(g, x, m.ln_pre_w);
    }

    // RoPE helper (rotate-half)
    auto apply_rope = [&](ggml_tensor *t) -> ggml_tensor * {
        int half = HD / 2;
        ggml_tensor *h1 = ggml_view_3d(g, t, half, NH, n_patches,
                                         t->nb[1], t->nb[2], 0);
        ggml_tensor *h2 = ggml_view_3d(g, t, half, NH, n_patches,
                                         t->nb[1], t->nb[2],
                                         (size_t)half * t->nb[0]);
        ggml_tensor *h2_neg = ggml_scale(g, ggml_cont(g, h2), -1.0f);
        ggml_tensor *rot = ggml_concat(g, h2_neg, ggml_cont(g, h1), 0);
        // Broadcast cos/sin: (HD, n_patches) → needs reshaping to (HD, 1, n_patches)
        // for broadcasting over NH heads. But we compute as (HD, NH, n_patches) by repeat.
        // Simple approach: mul with (HD, n_patches) broadcasts along dim 1 in ggml.
        return ggml_add(g, ggml_mul(g, t, cos_in), ggml_mul(g, rot, sin_in));
    };

    // Transformer layers
    for (int il = 0; il < m.vis_layers; il++) {
        auto &L = m.vis[il];
        ggml_tensor *residual = x;

        // Pre-attn RMSNorm
        ggml_tensor *y = ggml_rms_norm(g, x, 1e-6f);
        y = ggml_mul(g, y, L.attn_norm_w);

        // Separate Q/K/V projections
        ggml_tensor *Q = ggml_mul_mat(g, L.q_w, y);
        ggml_tensor *K = ggml_mul_mat(g, L.k_w, y);
        ggml_tensor *V = ggml_mul_mat(g, L.v_w, y);

        Q = ggml_reshape_3d(g, Q, HD, NH, n_patches);
        K = ggml_reshape_3d(g, K, HD, NH, n_patches);
        V = ggml_reshape_3d(g, V, HD, NH, n_patches);
        Q = ggml_cont(g, Q);
        K = ggml_cont(g, K);

        // Apply 2D RoPE
        Q = apply_rope(Q);
        K = apply_rope(K);

        // Permute to (HD, T, NH) for attention
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // Attention: scores = Q @ K^T / sqrt(HD)
        ggml_tensor *scores = ggml_mul_mat(g, K, Q);
        scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

        // attn_out = scores @ V^T
        ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor *attn = ggml_mul_mat(g, V_perm, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(g, attn, D, n_patches);

        // Output projection
        attn = ggml_mul_mat(g, L.o_w, attn);

        x = ggml_add(g, residual, attn);

        // Pre-FFN RMSNorm
        residual = x;
        y = ggml_rms_norm(g, x, 1e-6f);
        y = ggml_mul(g, y, L.ffn_norm_w);

        // SiLU FFN (SwiGLU): gate * silu(up) → down
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, L.gate_w, y));
        ggml_tensor *up = ggml_mul_mat(g, L.up_w, y);
        ggml_tensor *ffn = ggml_mul(g, gate, up);
        ffn = ggml_mul_mat(g, L.down_w, ffn);

        x = ggml_add(g, residual, ffn);
    }

    ggml_set_name(x, "vis_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // Allocate and compute
    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "lightonocr: vision graph alloc failed\n");
        ggml_free(g);
        return false;
    }

    // Set inputs
    ggml_tensor *t_in = ggml_graph_get_tensor(gf, "vis_input");
    ggml_backend_tensor_set(t_in, patch_embeds.data(), 0, D * n_patches * sizeof(float));
    ggml_tensor *t_cos = ggml_graph_get_tensor(gf, "cos_in");
    ggml_backend_tensor_set(t_cos, rope_cos.data(), 0, HD * n_patches * sizeof(float));
    ggml_tensor *t_sin = ggml_graph_get_tensor(gf, "sin_in");
    ggml_backend_tensor_set(t_sin, rope_sin.data(), 0, HD * n_patches * sizeof(float));

    ggml_backend_cpu_set_n_threads(ctx.backend, ctx.n_threads);
    if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "lightonocr: vision graph compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read output
    ggml_tensor *t_out = ggml_graph_get_tensor(gf, "vis_out");
    out.resize(D * n_patches);
    ggml_backend_tensor_get(t_out, out.data(), 0, out.size() * sizeof(float));

    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// Projection: spatial merge (2×2) + MLP + RMSNorm
// ---------------------------------------------------------------------------

static bool run_projection(context &ctx,
                             const std::vector<float> &vis_out,
                             int ph, int pw,
                             std::vector<float> &proj_out) {
    auto &m = ctx.m;
    int D = m.vis_dim;
    int merge = m.spatial_merge_size;  // 2
    int mh = ph / merge;
    int mw = pw / merge;
    int n_merged = mh * mw;

    // Spatial merge on CPU: group 2×2 patches → concat features
    // Input: (D, ph*pw), output: (D*4, mh*mw)
    int merge_dim = D * merge * merge;  // 4096
    std::vector<float> merged(merge_dim * n_merged, 0.0f);
    for (int my = 0; my < mh; my++) {
        for (int mx = 0; mx < mw; mx++) {
            int out_idx = my * mw + mx;
            for (int dy = 0; dy < merge; dy++) {
                for (int dx = 0; dx < merge; dx++) {
                    int src_idx = (my * merge + dy) * pw + (mx * merge + dx);
                    int feat_offset = (dy * merge + dx) * D;
                    memcpy(merged.data() + out_idx * merge_dim + feat_offset,
                           vis_out.data() + src_idx * D, D * sizeof(float));
                }
            }
        }
    }

    // Build projection graph: merger → linear1 → GELU → linear2 → RMSNorm
    ggml_init_params ip{ctx.compute_meta.size(), ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 256, false);

    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, merge_dim, n_merged);
    ggml_set_name(x, "merge_in"); ggml_set_input(x);

    // patch_merger: (merge_dim → D)
    x = ggml_mul_mat(g, m.proj_merger_w, x);
    // linear1: (D → D)
    x = ggml_mul_mat(g, m.proj_linear1_w, x);
    x = ggml_gelu_erf(g, x);
    // linear2: (D → D)
    x = ggml_mul_mat(g, m.proj_linear2_w, x);
    // RMSNorm
    x = ggml_rms_norm(g, x, 1e-6f);
    x = ggml_mul(g, x, m.proj_norm_w);

    ggml_set_name(x, "proj_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        ggml_free(g); return false;
    }

    ggml_tensor *t_in = ggml_graph_get_tensor(gf, "merge_in");
    ggml_backend_tensor_set(t_in, merged.data(), 0, merged.size() * sizeof(float));

    ggml_backend_cpu_set_n_threads(ctx.backend, ctx.n_threads);
    if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(g); return false;
    }

    ggml_tensor *t_out = ggml_graph_get_tensor(gf, "proj_out");
    proj_out.resize(D * n_merged);
    ggml_backend_tensor_get(t_out, proj_out.data(), 0, proj_out.size() * sizeof(float));

    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// Qwen3 decoder: prefill + KV-cached single-token decode
// ---------------------------------------------------------------------------

// Build prefill graph for the full input sequence (text + image tokens).
// Returns logits for the last token + K/V cache tensors.
static bool run_decoder_prefill(context &ctx,
                                 const std::vector<float> &image_embeds,
                                 int n_image_tokens,
                                 int max_new_tokens,
                                 std::string &out_text) {
    auto &m = ctx.m;
    const int D = m.lm_dim;
    const int V = m.vocab_size;
    const int n_layers = m.lm_layers;
    const int n_heads = m.lm_heads;
    const int n_kv_heads = m.lm_kv_heads;
    const int head_dim = m.lm_head_dim;
    const int kv_dim = head_dim * n_kv_heads;
    const float rms_eps = m.lm_rms_eps;
    const float rope_theta = m.lm_rope_theta;
    const int eos_id = m.eos_token_id;

    // Build prompt sequence: [text_before_image | image_tokens | text_after_image]
    // Chat template: <|im_start|>user\n[image]OCR this document.<|im_end|>\n<|im_start|>assistant\n
    // For simplicity, use a minimal prompt: just image tokens bracketed by
    // special tokens that tell the model to start generating.
    //
    // Token IDs for Qwen3 chat template:
    //   <|im_start|> = 151644, user = ? , <|im_end|> = 151645, assistant = ?
    // We'll embed a few framing tokens + image + assistant prefix.

    // Build the full prompt embedding: text tokens with image features spliced in.
    // Chat template for LightOnOCR:
    //   <|im_start|>user\n<image>OCR this document.<|im_end|>\n<|im_start|>assistant\n
    // We construct: [prefix_tokens] + [image_features] + [suffix_tokens]
    // where prefix = <|im_start|>user\n  and suffix = OCR this.<|im_end|>\n<|im_start|>assistant\n

    // Qwen3 special tokens
    const int im_start = 151644;  // <|im_start|>
    const int im_end   = 151645;  // <|im_end|>
    const int nl_tok   = 198;     // \n

    // Simple prefix/suffix token IDs (hardcoded for now)
    std::vector<int32_t> prefix_ids = {im_start, 882 /*user*/, nl_tok};  // <|im_start|>user\n
    std::vector<int32_t> suffix_ids = {nl_tok, im_end, nl_tok, im_start, 78191 /*assistant*/, nl_tok};

    int n_prefix = (int)prefix_ids.size();
    int n_suffix = (int)suffix_ids.size();
    int n_prompt = n_prefix + n_image_tokens + n_suffix;

    // Embed prefix and suffix tokens via embed_tokens on CPU
    auto embed_tokens_cpu = [&](const std::vector<int32_t> &ids) -> std::vector<float> {
        int n = (int)ids.size();
        ggml_init_params eip{ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
        ggml_context *eg = ggml_init(eip);
        ggml_tensor *idx = ggml_new_tensor_1d(eg, GGML_TYPE_I32, n);
        ggml_set_input(idx);
        ggml_tensor *emb = ggml_get_rows(eg, m.embed_tokens, idx);
        ggml_set_output(emb);
        ggml_cgraph *egf = ggml_new_graph(eg);
        ggml_build_forward_expand(egf, emb);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, egf);
        ggml_backend_tensor_set(idx, ids.data(), 0, n * sizeof(int32_t));
        ggml_backend_sched_graph_compute(ctx.sched, egf);
        std::vector<float> result(D * n);
        ggml_backend_tensor_get(emb, result.data(), 0, result.size() * sizeof(float));
        ggml_free(eg);
        return result;
    };

    auto prefix_emb = embed_tokens_cpu(prefix_ids);
    auto suffix_emb = embed_tokens_cpu(suffix_ids);

    // Build combined embedding: [prefix_emb | image_embeds | suffix_emb]
    std::vector<float> full_emb(D * n_prompt);
    memcpy(full_emb.data(), prefix_emb.data(), D * n_prefix * sizeof(float));
    memcpy(full_emb.data() + D * n_prefix, image_embeds.data(), D * n_image_tokens * sizeof(float));
    memcpy(full_emb.data() + D * (n_prefix + n_image_tokens), suffix_emb.data(), D * n_suffix * sizeof(float));

    fprintf(stderr, "lightonocr: prompt = %d prefix + %d image + %d suffix = %d total\n",
            n_prefix, n_image_tokens, n_suffix, n_prompt);

    // Prefill: build decoder graph for n_prompt tokens
    ggml_init_params ip{ctx.compute_meta.size(), ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Input: pre-embedded sequence (D, n_prompt)
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_prompt);
    ggml_set_name(x, "lm_input"); ggml_set_input(x);

    // Causal mask: (n_prompt, n_prompt) — lower-triangular
    ggml_tensor *mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, n_prompt, n_prompt);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Position IDs: [0, 1, 2, ..., n_prompt-1]
    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_prompt);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    auto rmsnorm = [&rms_eps](ggml_context *gc, ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(gc, ggml_rms_norm(gc, t, rms_eps), w);
    };

    // Store K/V pointers per layer for cache extraction after compute
    std::vector<ggml_tensor*> kv_k_ptrs(n_layers, nullptr);
    std::vector<ggml_tensor*> kv_v_ptrs(n_layers, nullptr);

    // Decoder layers
    for (int il = 0; il < n_layers; il++) {
        auto &L = m.lm[il];
        ggml_tensor *residual = x;

        // Pre-attn RMSNorm
        ggml_tensor *normed = rmsnorm(g, x, L.attn_norm_w);

        // Q/K/V (no bias)
        ggml_tensor *Q = ggml_mul_mat(g, L.q_w, normed);
        ggml_tensor *K = ggml_mul_mat(g, L.k_w, normed);
        ggml_tensor *V = ggml_mul_mat(g, L.v_w, normed);

        // Reshape: Q (head_dim, n_heads, T), K/V (head_dim, n_kv, T)
        Q = ggml_reshape_3d(g, Q, head_dim, n_heads, n_prompt);
        K = ggml_reshape_3d(g, K, head_dim, n_kv_heads, n_prompt);
        V = ggml_reshape_3d(g, V, head_dim, n_kv_heads, n_prompt);

        // QK norm (Qwen3): RMSNorm per head on the head_dim axis
        // Applied AFTER reshape but BEFORE RoPE (matching HF Qwen3Attention)
        if (m.use_qk_norm && L.q_norm_w && L.k_norm_w) {
            // Flatten heads×tokens for RMSNorm, then reshape back
            Q = ggml_cont(g, ggml_reshape_2d(g, Q, head_dim, n_heads * n_prompt));
            Q = ggml_rms_norm(g, Q, rms_eps);
            Q = ggml_mul(g, Q, L.q_norm_w);
            Q = ggml_cont(g, ggml_reshape_3d(g, Q, head_dim, n_heads, n_prompt));

            K = ggml_cont(g, ggml_reshape_2d(g, K, head_dim, n_kv_heads * n_prompt));
            K = ggml_rms_norm(g, K, rms_eps);
            K = ggml_mul(g, K, L.k_norm_w);
            K = ggml_cont(g, ggml_reshape_3d(g, K, head_dim, n_kv_heads, n_prompt));
        }

        // RoPE (standard 1D, not mRoPE)
        Q = ggml_rope_ext(g, Q, pos_ids, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, pos_ids, nullptr,
                          head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // KV cache extraction disabled — ggml_set_output on 56 tensors
        // (28 K + 28 V) prevents memory reuse and causes OOM on 8GB.
        // The decode loop falls back to O(n²) full recompute.
        // TODO: implement streaming KV extraction or use the qwen2vl
        // approach (dedicated graph output pass).
        // kv_k_ptrs[il] = K;
        // kv_v_ptrs[il] = V;

        // GQA repeat K/V if needed
        if (n_kv_heads < n_heads) {
            int repeat = n_heads / n_kv_heads;
            K = ggml_reshape_4d(g, K, head_dim, 1, n_kv_heads, n_prompt);
            ggml_tensor *K_tgt = ggml_new_tensor_4d(g, K->type, head_dim, repeat, n_kv_heads, n_prompt);
            K = ggml_repeat(g, K, K_tgt);
            K = ggml_reshape_3d(g, K, head_dim, n_heads, n_prompt);

            V = ggml_reshape_4d(g, V, head_dim, 1, n_kv_heads, n_prompt);
            ggml_tensor *V_tgt = ggml_new_tensor_4d(g, V->type, head_dim, repeat, n_kv_heads, n_prompt);
            V = ggml_repeat(g, V, V_tgt);
            V = ggml_reshape_3d(g, V, head_dim, n_heads, n_prompt);
        }

        // Permute for attention
        float scale = 1.0f / sqrtf((float)head_dim);
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // Causal attention with mask
        ggml_tensor *scores = ggml_mul_mat(g, K, Q);
        scores = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);

        ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor *attn = ggml_mul_mat(g, V_perm, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        int q_total = n_heads * head_dim;  // 2048 for GQA where Q has more heads
        attn = ggml_reshape_2d(g, attn, q_total, n_prompt);

        // Output projection: (D, q_total) @ (q_total, T) → (D, T)
        attn = ggml_mul_mat(g, L.o_w, attn);

        x = ggml_add(g, residual, attn);

        // FFN: SwiGLU
        residual = x;
        normed = rmsnorm(g, x, L.ffn_norm_w);

        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, L.gate_w, normed));
        ggml_tensor *up = ggml_mul_mat(g, L.up_w, normed);
        ggml_tensor *ffn = ggml_mul(g, gate, up);
        ffn = ggml_mul_mat(g, L.down_w, ffn);

        x = ggml_add(g, residual, ffn);
    }

    // Final RMSNorm
    x = rmsnorm(g, x, m.lm_norm_w);

    // LM head: tied weights (embed_tokens transposed)
    // Only decode last token to save memory
    ggml_tensor *last_x = ggml_view_2d(g, x, D, 1, x->nb[1], (size_t)(n_prompt - 1) * x->nb[1]);
    last_x = ggml_cont(g, last_x);
    ggml_tensor *last_logits = ggml_mul_mat(g, m.embed_tokens, last_x);
    last_logits = ggml_reshape_1d(g, last_logits, V);
    ggml_set_name(last_logits, "last_logits"); ggml_set_output(last_logits);
    ggml_build_forward_expand(gf, last_logits);

    // K/V outputs are stored in kv_k_ptrs/kv_v_ptrs for extraction after compute

    // Allocate and compute
    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "lightonocr: decoder prefill alloc failed\n");
        ggml_free(g);
        return false;
    }

    // Set inputs
    ggml_tensor *t_in = ggml_graph_get_tensor(gf, "lm_input");
    ggml_backend_tensor_set(t_in, full_emb.data(), 0, D * n_prompt * sizeof(float));

    // Build causal mask (-inf above diagonal)
    std::vector<float> mask_data(n_prompt * n_prompt);
    for (int i = 0; i < n_prompt; i++)
        for (int j = 0; j < n_prompt; j++)
            mask_data[i * n_prompt + j] = (j <= i) ? 0.0f : -INFINITY;
    ggml_tensor *t_mask = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(t_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    // Position IDs: [0, 1, 2, ..., n_prompt-1]
    std::vector<int32_t> pos_data(n_prompt);
    for (int i = 0; i < n_prompt; i++) pos_data[i] = i;
    ggml_tensor *t_pos = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, n_prompt * sizeof(int32_t));

    ggml_backend_cpu_set_n_threads(ctx.backend, ctx.n_threads);
    if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "lightonocr: decoder prefill compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read last logits
    std::vector<float> logits_data(V);
    ggml_tensor *t_logits = ggml_graph_get_tensor(gf, "last_logits");
    ggml_backend_tensor_get(t_logits, logits_data.data(), 0, V * sizeof(float));
    // Extract KV cache from prefill graph outputs (using stored pointers)
    std::vector<std::vector<float>> k_cache(n_layers), v_cache(n_layers);
    bool kv_ok = true;
    for (int il = 0; il < n_layers; il++) {
        ggml_tensor *kt = kv_k_ptrs[il];
        ggml_tensor *vt = kv_v_ptrs[il];
        if (!kt || !vt) { kv_ok = false; break; }
        size_t sz = (size_t)kv_dim * n_prompt;
        k_cache[il].resize(sz);
        v_cache[il].resize(sz);
        ggml_backend_tensor_get(kt, k_cache[il].data(), 0, sz * sizeof(float));
        ggml_backend_tensor_get(vt, v_cache[il].data(), 0, sz * sizeof(float));
    }
    ggml_free(g);
    fprintf(stderr, "lightonocr: KV extraction: %s\n", kv_ok ? "OK" : "FAILED");
    if (kv_ok)
        fprintf(stderr, "lightonocr: KV cache: %d layers × %d tokens × %d dim = %.1f MB\n",
                n_layers, n_prompt, kv_dim,
                (float)n_layers * n_prompt * kv_dim * 2 * sizeof(float) / (1024*1024));

    std::vector<int32_t> generated;
    int best = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++)
        if (logits_data[v] > best_score) { best_score = logits_data[v]; best = v; }
    generated.push_back(best);

    fprintf(stderr, "lightonocr: prefill done, first token=%d (score=%.2f)\n", best, best_score);
    fprintf(stderr, "lightonocr: logits[0..4] = [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
            logits_data[0], logits_data[1], logits_data[2], logits_data[3], logits_data[4]);
    fprintf(stderr, "lightonocr: logits[eos=%d] = %.3f\n", eos_id, logits_data[eos_id]);

    if (best == eos_id) {
        out_text = "";
        return true;
    }

    // KV-cached decode: O(n) per step
    int n_kv = n_prompt;
    int kv_repeat = n_heads / n_kv_heads;

    for (int step = 1; step < max_new_tokens && best != eos_id; step++) {
        if (!kv_ok) break;  // fallback not implemented, stop

        int pos = n_kv;
        // Embed new token
        auto tok_emb = embed_tokens_cpu({(int32_t)best});

        // Build single-token decode graph
        ggml_init_params ip2{ctx.compute_meta.size(), ctx.compute_meta.data(), true};
        ggml_context *g2 = ggml_init(ip2);
        ggml_cgraph *gf2 = ggml_new_graph_custom(g2, 16384, false);

        ggml_tensor *x2 = ggml_new_tensor_2d(g2, GGML_TYPE_F32, D, 1);
        ggml_set_name(x2, "tok_emb"); ggml_set_input(x2);
        ggml_tensor *pos2 = ggml_new_tensor_1d(g2, GGML_TYPE_I32, 1);
        ggml_set_name(pos2, "pos_ids"); ggml_set_input(pos2);

        ggml_tensor *cur = x2;
        char name[64];
        for (int il = 0; il < n_layers; il++) {
            auto &L = m.lm[il];
            ggml_tensor *res = cur;
            ggml_tensor *normed = rmsnorm(g2, cur, L.attn_norm_w);

            ggml_tensor *Q = ggml_mul_mat(g2, L.q_w, normed);
            ggml_tensor *K_new = ggml_mul_mat(g2, L.k_w, normed);
            ggml_tensor *V_new = ggml_mul_mat(g2, L.v_w, normed);

            Q = ggml_reshape_3d(g2, Q, head_dim, n_heads, 1);
            K_new = ggml_reshape_3d(g2, K_new, head_dim, n_kv_heads, 1);
            V_new = ggml_reshape_3d(g2, V_new, head_dim, n_kv_heads, 1);

            // QK norm
            if (m.use_qk_norm && L.q_norm_w && L.k_norm_w) {
                Q = ggml_cont(g2, ggml_reshape_2d(g2, Q, head_dim, n_heads));
                Q = ggml_rms_norm(g2, Q, rms_eps); Q = ggml_mul(g2, Q, L.q_norm_w);
                Q = ggml_cont(g2, ggml_reshape_3d(g2, Q, head_dim, n_heads, 1));
                K_new = ggml_cont(g2, ggml_reshape_2d(g2, K_new, head_dim, n_kv_heads));
                K_new = ggml_rms_norm(g2, K_new, rms_eps); K_new = ggml_mul(g2, K_new, L.k_norm_w);
                K_new = ggml_cont(g2, ggml_reshape_3d(g2, K_new, head_dim, n_kv_heads, 1));
            }

            // RoPE at position `pos`
            Q = ggml_rope_ext(g2, Q, pos2, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                              rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K_new = ggml_rope_ext(g2, K_new, pos2, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                  rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Output new K/V for cache append
            snprintf(name, sizeof(name), "k_out_%d", il);
            ggml_tensor *K_new_flat = ggml_reshape_1d(g2, K_new, kv_dim);
            ggml_set_name(K_new_flat, name); ggml_set_output(K_new_flat);
            snprintf(name, sizeof(name), "v_out_%d", il);
            ggml_tensor *V_new_flat = ggml_reshape_1d(g2, V_new, kv_dim);
            ggml_set_name(V_new_flat, name); ggml_set_output(V_new_flat);

            // Load KV cache + concat new
            ggml_tensor *k_in = ggml_new_tensor_2d(g2, GGML_TYPE_F32, kv_dim, n_kv);
            snprintf(name, sizeof(name), "k_in_%d", il);
            ggml_set_name(k_in, name); ggml_set_input(k_in);
            ggml_tensor *v_in = ggml_new_tensor_2d(g2, GGML_TYPE_F32, kv_dim, n_kv);
            snprintf(name, sizeof(name), "v_in_%d", il);
            ggml_set_name(v_in, name); ggml_set_input(v_in);

            ggml_tensor *K_full = ggml_concat(g2,
                ggml_reshape_3d(g2, k_in, head_dim, n_kv_heads, n_kv), K_new, 2);
            ggml_tensor *V_full = ggml_concat(g2,
                ggml_reshape_3d(g2, v_in, head_dim, n_kv_heads, n_kv), V_new, 2);

            // GQA repeat
            int seq = n_kv + 1;
            if (kv_repeat > 1) {
                K_full = ggml_reshape_4d(g2, K_full, head_dim, 1, n_kv_heads, seq);
                K_full = ggml_repeat(g2, K_full,
                    ggml_new_tensor_4d(g2, K_full->type, head_dim, kv_repeat, n_kv_heads, seq));
                K_full = ggml_reshape_3d(g2, K_full, head_dim, n_heads, seq);
                V_full = ggml_reshape_4d(g2, V_full, head_dim, 1, n_kv_heads, seq);
                V_full = ggml_repeat(g2, V_full,
                    ggml_new_tensor_4d(g2, V_full->type, head_dim, kv_repeat, n_kv_heads, seq));
                V_full = ggml_reshape_3d(g2, V_full, head_dim, n_heads, seq);
            }

            // Attention: Q(hd,nh,1) vs K_full(hd,nh,seq)
            float sc = 1.0f / sqrtf((float)head_dim);
            Q = ggml_cont(g2, ggml_permute(g2, Q, 0, 2, 1, 3));
            K_full = ggml_cont(g2, ggml_permute(g2, K_full, 0, 2, 1, 3));
            V_full = ggml_cont(g2, ggml_permute(g2, V_full, 0, 2, 1, 3));
            ggml_tensor *scores = ggml_soft_max_ext(g2, ggml_mul_mat(g2, K_full, Q),
                                                     nullptr, sc, 0.0f);
            ggml_tensor *Vp = ggml_cont(g2, ggml_permute(g2, V_full, 1, 0, 2, 3));
            ggml_tensor *attn = ggml_mul_mat(g2, Vp, scores);
            attn = ggml_cont(g2, ggml_permute(g2, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(g2, attn, n_heads * head_dim, 1);
            attn = ggml_mul_mat(g2, L.o_w, attn);
            cur = ggml_add(g2, res, attn);

            // FFN
            res = cur;
            normed = rmsnorm(g2, cur, L.ffn_norm_w);
            ggml_tensor *gate = ggml_silu(g2, ggml_mul_mat(g2, L.gate_w, normed));
            ggml_tensor *up = ggml_mul_mat(g2, L.up_w, normed);
            cur = ggml_add(g2, res, ggml_mul_mat(g2, L.down_w, ggml_mul(g2, gate, up)));
        }

        cur = rmsnorm(g2, cur, m.lm_norm_w);
        ggml_tensor *ll = ggml_reshape_1d(g2, ggml_mul_mat(g2, m.embed_tokens, cur), V);
        ggml_set_name(ll, "last_logits"); ggml_set_output(ll);
        ggml_build_forward_expand(gf2, ll);

        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, gf2)) { ggml_free(g2); break; }

        // Set inputs
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf2, "tok_emb"),
                                tok_emb.data(), 0, D * sizeof(float));
        int32_t pos_val = pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf2, "pos_ids"),
                                &pos_val, 0, sizeof(int32_t));
        for (int il = 0; il < n_layers; il++) {
            snprintf(name, sizeof(name), "k_in_%d", il);
            ggml_tensor *ki = ggml_graph_get_tensor(gf2, name);
            if (ki) ggml_backend_tensor_set(ki, k_cache[il].data(), 0,
                                             k_cache[il].size() * sizeof(float));
            snprintf(name, sizeof(name), "v_in_%d", il);
            ggml_tensor *vi = ggml_graph_get_tensor(gf2, name);
            if (vi) ggml_backend_tensor_set(vi, v_cache[il].data(), 0,
                                             v_cache[il].size() * sizeof(float));
        }

        ggml_backend_cpu_set_n_threads(ctx.backend, ctx.n_threads);
        if (ggml_backend_sched_graph_compute(ctx.sched, gf2) != GGML_STATUS_SUCCESS) {
            ggml_free(g2); break;
        }

        ggml_backend_tensor_get(ggml_graph_get_tensor(gf2, "last_logits"),
                                logits_data.data(), 0, V * sizeof(float));

        // Append new K/V to cache
        for (int il = 0; il < n_layers; il++) {
            std::vector<float> k_new(kv_dim), v_new(kv_dim);
            snprintf(name, sizeof(name), "k_out_%d", il);
            ggml_tensor *ko = ggml_graph_get_tensor(gf2, name);
            if (ko) ggml_backend_tensor_get(ko, k_new.data(), 0, kv_dim * sizeof(float));
            snprintf(name, sizeof(name), "v_out_%d", il);
            ggml_tensor *vo = ggml_graph_get_tensor(gf2, name);
            if (vo) ggml_backend_tensor_get(vo, v_new.data(), 0, kv_dim * sizeof(float));
            k_cache[il].insert(k_cache[il].end(), k_new.begin(), k_new.end());
            v_cache[il].insert(v_cache[il].end(), v_new.begin(), v_new.end());
        }
        n_kv++;
        ggml_free(g2);

        best = 0; best_score = -INFINITY;
        for (int v = 0; v < V; v++)
            if (logits_data[v] > best_score) { best_score = logits_data[v]; best = v; }
        generated.push_back(best);
        fprintf(stderr, "  gen[%d] tok=%d%s\n", step, best, kv_ok ? " (cached)" : "");
    }

    // Decode generated token IDs to text using the vocab
    out_text = "";
    for (int id : generated) {
        if (id == eos_id) break;
        if (id >= 0 && id < (int)ctx.vocab.size()) {
            const std::string &tok = ctx.vocab[id];
            // GPT-2 BPE: Ġ → space
            if (!tok.empty() && tok[0] == '\xc4' && tok.size() >= 2 && tok[1] == '\xa0') {
                out_text += ' ';
                out_text += tok.substr(2);
            } else {
                out_text += tok;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Full inference: vision → projection → decoder (greedy)
// ---------------------------------------------------------------------------

std::string recognize_raw(context &ctx,
                           const uint8_t *pixels, int width, int height, int channels,
                           int max_tokens) {
    // Convert to RGB
    std::vector<uint8_t> rgb;
    const uint8_t *rgb_data = pixels;
    if (channels == 1) {
        rgb.resize(width * height * 3);
        for (int i = 0; i < width * height; i++)
            rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = pixels[i];
        rgb_data = rgb.data();
    } else if (channels == 4) {
        rgb.resize(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            rgb[i * 3 + 0] = pixels[i * 4 + 0];
            rgb[i * 3 + 1] = pixels[i * 4 + 1];
            rgb[i * 3 + 2] = pixels[i * 4 + 2];
        }
        rgb_data = rgb.data();
    }

    int ph = 0, pw = 0;
    auto img = preprocess_image(rgb_data, width, height,
                                 ctx.m.patch_size, ctx.m.image_size, &ph, &pw);
    int n_patches = ph * pw;
    fprintf(stderr, "lightonocr: image %dx%d → %dx%d patches (%d total)\n",
            width, height, pw, ph, n_patches);

    // Step 1: patch conv on CPU
    int th = ph * ctx.m.patch_size;
    int tw = pw * ctx.m.patch_size;
    auto patch_embeds = apply_patch_conv(ctx, img, th, tw, ph, pw);
    fprintf(stderr, "lightonocr: patch_conv done (%d patches × %d dim)\n",
            n_patches, ctx.m.vis_dim);

    // Step 2: 2D RoPE
    std::vector<float> rope_cos, rope_sin;
    compute_2d_rope(ph, pw, ctx.m.vis_head_dim, ctx.m.vis_rope_theta, rope_cos, rope_sin);

    // Step 3: vision encoder
    std::vector<float> vis_out;
    if (!run_vision_encoder(ctx, patch_embeds, rope_cos, rope_sin, n_patches, vis_out)) {
        return "";
    }
    fprintf(stderr, "lightonocr: vision encoder done\n");

    // Step 4: projection (spatial merge + MLP)
    std::vector<float> proj_out;
    if (!run_projection(ctx, vis_out, ph, pw, proj_out)) {
        return "";
    }
    int n_image_tokens = (ph / ctx.m.spatial_merge_size) * (pw / ctx.m.spatial_merge_size);
    fprintf(stderr, "lightonocr: projection done → %d image tokens\n", n_image_tokens);

    // Step 5: Qwen3 decoder — greedy generation from image tokens
    std::string gen_text;
    if (!run_decoder_prefill(ctx, proj_out, n_image_tokens, max_tokens, gen_text)) {
        fprintf(stderr, "lightonocr: decoder failed\n");
        return "";
    }
    fprintf(stderr, "lightonocr: generated: %s\n", gen_text.c_str());

    ctx.last_text = gen_text;
    return ctx.last_text;
}

std::string recognize_file(context &ctx, const char *image_path, int max_tokens) {
    int w, h, ch;
    unsigned char *data = stbi_load(image_path, &w, &h, &ch, 3);
    if (!data) {
        fprintf(stderr, "lightonocr: cannot load %s\n", image_path);
        return "";
    }
    auto result = recognize_raw(ctx, data, w, h, 3, max_tokens);
    stbi_image_free(data);
    return result;
}

} // namespace lightonocr

// ── C ABI ──

struct lightonocr_context {
    lightonocr::context ctx;
    int max_tokens = 2048;
};

extern "C" lightonocr_context * lightonocr_init(const char * model_path, int n_threads) {
    auto *c = new lightonocr_context;
    if (!lightonocr::load(c->ctx, model_path, n_threads)) {
        delete c;
        return nullptr;
    }
    return c;
}

extern "C" void lightonocr_free(lightonocr_context * c) {
    if (!c) return;
    lightonocr::free_(c->ctx);
    delete c;
}

extern "C" void lightonocr_set_max_tokens(lightonocr_context * c, int max_tokens) {
    if (c) c->max_tokens = max_tokens;
}

extern "C" const char * lightonocr_recognize_raw(
        lightonocr_context * c,
        const uint8_t * pixels, int width, int height, int channels,
        int * out_len) {
    if (!c) return nullptr;
    c->ctx.last_text = lightonocr::recognize_raw(c->ctx, pixels, width, height, channels,
                                                   c->max_tokens);
    if (out_len) *out_len = (int)c->ctx.last_text.size();
    return c->ctx.last_text.c_str();
}

extern "C" const char * lightonocr_recognize_file(
        lightonocr_context * c, const char * image_path, int * out_len) {
    if (!c || !image_path) return nullptr;
    c->ctx.last_text = lightonocr::recognize_file(c->ctx, image_path, c->max_tokens);
    if (out_len) *out_len = (int)c->ctx.last_text.size();
    return c->ctx.last_text.c_str();
}
