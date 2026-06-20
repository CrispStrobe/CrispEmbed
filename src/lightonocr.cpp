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
#include <chrono>

// Env var helpers for gating debug prints and feature paths
static bool env_flag(const char *name) {
    const char *v = std::getenv(name);
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}
#define LOCR_DEBUG env_flag("CRISPEMBED_LIGHTONOCR_DEBUG")
#define LOCR_NO_KV env_flag("CRISPEMBED_LIGHTONOCR_NO_KV")

// Env var gating: CRISPEMBED_LIGHTONOCR_FLASH_ATTN=1 to use flash attention
// Default: manual attention (confirmed working output)
static bool use_flash_attn() {
    static int cached = -1;
    if (cached < 0) {
        const char *v = std::getenv("CRISPEMBED_LIGHTONOCR_FLASH_ATTN");
        cached = (v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y')) ? 1 : 0;
    }
    return cached != 0;
}
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
    bool bench = false;
    std::string last_text;

    // Tokenizer
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;

    // Per-token confidence
    std::vector<float> char_confidences;
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

    ctx.bench = (std::getenv("CRISPEMBED_LIGHTONOCR_BENCH") != nullptr);

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
    // Allow runtime override of effective max_size for CPU-friendly inference
    int eff_max = max_size;
    if (const char *mp = getenv("CRISPEMBED_MAX_PIXELS")) {
        int max_px = atoi(mp);
        // Derive max_side from max_pixels assuming square aspect
        int max_side = (int)std::sqrt((float)max_px);
        eff_max = std::min(eff_max, max_side);
    }
    // Resize to fit max_size while preserving aspect ratio, then pad to patch grid
    float scale = std::min((float)eff_max / w, (float)eff_max / h);
    if (scale > 1.0f) scale = 1.0f;  // don't upscale
    int rw = (int)(w * scale);
    int rh = (int)(h * scale);
    // Round up to patch grid
    // Pixtral: ceiling division for patch count (matches _num_image_tokens)
    // num_tokens = (dim - 1) // patch_size + 1
    // Then resize image to num_tokens * patch_size
    int pw = (rw - 1) / patch_size + 1;
    int ph = (rh - 1) / patch_size + 1;
    // Ensure even for spatial merge (merge_size=2)
    if (pw % 2 != 0) pw++;
    if (ph % 2 != 0) ph++;
    int tw = pw * patch_size;
    int th = ph * patch_size;

    // Bilinear resize to (tw, th) + normalize (ImageNet mean/std).
    // The image must fill the entire target grid — Pixtral processor does
    // img.resize((tw, th), BILINEAR) which stretches the input to fit.
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float std_[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    std::vector<float> pixels(3 * th * tw, 0.0f);
    for (int y = 0; y < th; y++) {
        float sy = (float)y * (h - 1) / std::max(th - 1, 1);
        int y0 = std::min((int)sy, h - 1);
        int y1 = std::min(y0 + 1, h - 1);
        float fy = sy - y0;
        for (int x = 0; x < tw; x++) {
            float sx = (float)x * (w - 1) / std::max(tw - 1, 1);
            int x0 = std::min((int)sx, w - 1);
            int x1 = std::min(x0 + 1, w - 1);
            float fx = sx - x0;
            for (int c = 0; c < 3; c++) {
                // Bilinear interpolation
                float v00 = rgb[(y0 * w + x0) * 3 + c] / 255.0f;
                float v01 = rgb[(y0 * w + x1) * 3 + c] / 255.0f;
                float v10 = rgb[(y1 * w + x0) * 3 + c] / 255.0f;
                float v11 = rgb[(y1 * w + x1) * 3 + c] / 255.0f;
                float val = (1-fy)*((1-fx)*v00 + fx*v01) + fy*((1-fx)*v10 + fx*v11);
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
    int half = head_dim / 2;     // rotate-half pair stride
    int quarter = head_dim / 4;  // # of h freqs = # of w freqs
    cos_out.resize(head_dim * T);
    sin_out.resize(head_dim * T);

    // Pixtral 2D RoPE (transformers PixtralRotaryEmbedding): build the dim/2
    // base frequencies, then height uses freqs[::2] and width uses freqs[1::2].
    // The per-patch angle vector is [h_angles(dim/4) | w_angles(dim/4)] and the
    // attention applies it with rotate_half — i.e. cos/sin must repeat the
    // angle vector across the two halves: idx j and j+half share the same angle.
    //   freqs[k] = 1 / theta^(2k/dim),  h_freq[j] = freqs[2j], w_freq[j] = freqs[2j+1]
    std::vector<float> freqs(half);
    for (int k = 0; k < half; k++)
        freqs[k] = 1.0f / powf(theta, (float)(2 * k) / head_dim);

    for (int y = 0; y < ph; y++) {
        for (int x = 0; x < pw; x++) {
            int t = y * pw + x;
            float *cr = cos_out.data() + (size_t)t * head_dim;
            float *sr = sin_out.data() + (size_t)t * head_dim;
            for (int j = 0; j < quarter; j++) {
                float ah = y * freqs[2 * j];      // height, freqs[::2]
                float aw = x * freqs[2 * j + 1];  // width,  freqs[1::2]
                // first half: [h_angles | w_angles]; second half repeats it
                cr[j] = cr[j + half] = cosf(ah);
                sr[j] = sr[j + half] = sinf(ah);
                cr[quarter + j] = cr[quarter + j + half] = cosf(aw);
                sr[quarter + j] = sr[quarter + j + half] = sinf(aw);
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

    // cos/sin for 2D RoPE, broadcast over heads: (HD, 1, n_patches)
    ggml_tensor *cos_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, HD, 1, n_patches);
    ggml_tensor *sin_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, HD, 1, n_patches);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    // ln_pre (RMSNorm before transformer blocks)
    if (m.ln_pre_w) {
        x = ggml_rms_norm(g, x, 1e-6f);
        x = ggml_mul(g, x, m.ln_pre_w);
    }

    bool debug = LOCR_DEBUG;
    std::vector<ggml_tensor *> dbg_outputs;  // side outputs to expand into graph

    // Debug: mark ln_pre output for readback
    if (debug && m.ln_pre_w) {
        ggml_tensor *ln_pre_dbg = ggml_cont(g, x);
        ggml_set_name(ln_pre_dbg, "dbg_ln_pre");
        ggml_set_output(ln_pre_dbg);
        dbg_outputs.push_back(ln_pre_dbg);
    }

    // RoPE helper (rotate-half) — broadcasts cos/sin (HD, 1, T) over NH heads
    auto apply_rope = [&](ggml_tensor *t) -> ggml_tensor * {
        int half = HD / 2;
        ggml_tensor *h1 = ggml_view_3d(g, t, half, NH, n_patches,
                                         t->nb[1], t->nb[2], 0);
        ggml_tensor *h2 = ggml_view_3d(g, t, half, NH, n_patches,
                                         t->nb[1], t->nb[2],
                                         (size_t)half * t->nb[0]);
        ggml_tensor *h2_neg = ggml_scale(g, ggml_cont(g, h2), -1.0f);
        ggml_tensor *rot = ggml_concat(g, h2_neg, ggml_cont(g, h1), 0);
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

        // Debug: mark Q/K after RoPE and V for readback (before permute)
        if (debug && il == 0) {
            char nm[32];
            ggml_tensor *q_dbg = ggml_cont(g, ggml_reshape_2d(g, Q, D, n_patches));
            snprintf(nm, sizeof(nm), "dbg_q_rope_%d", il);
            ggml_set_name(q_dbg, nm); ggml_set_output(q_dbg);
            dbg_outputs.push_back(q_dbg);

            ggml_tensor *k_dbg = ggml_cont(g, ggml_reshape_2d(g, K, D, n_patches));
            snprintf(nm, sizeof(nm), "dbg_k_rope_%d", il);
            ggml_set_name(k_dbg, nm); ggml_set_output(k_dbg);
            dbg_outputs.push_back(k_dbg);

        }

        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // (HD, T, NH)
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        ggml_tensor *attn;
        if (use_flash_attn()) {
            attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        } else {
            ggml_tensor *scores = ggml_mul_mat(g, K, Q);
            scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);
            ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
            attn = ggml_mul_mat(g, V_perm, scores);
            attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        }
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

        // Debug: skip remaining layers after first 1 for fast parity testing
        if (debug && il == 0) {
            fprintf(stderr, "lightonocr: DEBUG — stopping vision encoder at layer %d\n", il);
            break;
        }
    }

    ggml_set_name(x, "vis_out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    for (auto *dbg_t : dbg_outputs)
        ggml_build_forward_expand(gf, dbg_t);

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

    // Debug: read named debug tensors for parity comparison
    if (LOCR_DEBUG) {
        auto dump_dbg = [&](const char *name) {
            ggml_tensor *t = ggml_graph_get_tensor(gf, name);
            if (!t) return;
            size_t nel = ggml_nelements(t);
            std::vector<float> d(nel);
            ggml_backend_tensor_get(t, d.data(), 0, nel * sizeof(float));
            int n = std::min((int)nel, 10);
            fprintf(stderr, "lightonocr: %s[0,:%d] =", name, n);
            for (int i = 0; i < n; i++) fprintf(stderr, " %.6f", d[i]);
            fprintf(stderr, "\n");
        };
        dump_dbg("dbg_ln_pre");
        dump_dbg("dbg_q_rope_0");
        dump_dbg("dbg_k_rope_0");
    }

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

    // Mistral3 projector applies norm(image_features) BEFORE the patch merger
    // (forward: norm → patch_merger → linear_1 → gelu → linear_2). The norm is
    // an RMSNorm over the D-dim vision features, per patch. Do it on CPU here,
    // before the 2×2 spatial merge, and drop the (wrong) trailing norm below.
    std::vector<float> vis_normed(vis_out.size());
    {
        std::vector<float> nw(D);
        ggml_backend_tensor_get(m.proj_norm_w, nw.data(), 0, D * sizeof(float));
        const float eps = ctx.m.lm_rms_eps;  // text_config.rms_norm_eps
        int n_patches = ph * pw;
        for (int p = 0; p < n_patches; p++) {
            const float *src = vis_out.data() + (size_t)p * D;
            float *dst = vis_normed.data() + (size_t)p * D;
            double ss = 0.0;
            for (int i = 0; i < D; i++) ss += (double)src[i] * src[i];
            float scale = 1.0f / std::sqrt((float)(ss / D) + eps);
            for (int i = 0; i < D; i++) dst[i] = src[i] * scale * nw[i];
        }
    }

    // Spatial merge on CPU: group 2×2 patches → concat features.
    // Mistral3PatchMerger uses F.unfold, which orders the merge_dim vector
    // CHANNEL-major: [c0·(k00,k01,k10,k11), c1·(...), ...] — i.e. for each of
    // the D channels, the merge*merge kernel positions. (NOT patch-major.)
    int msq = merge * merge;
    int merge_dim = D * msq;  // 4096
    std::vector<float> merged(merge_dim * n_merged, 0.0f);
    for (int my = 0; my < mh; my++) {
        for (int mx = 0; mx < mw; mx++) {
            int out_idx = my * mw + mx;
            float *dst = merged.data() + (size_t)out_idx * merge_dim;
            for (int dy = 0; dy < merge; dy++) {
                for (int dx = 0; dx < merge; dx++) {
                    int src_idx = (my * merge + dy) * pw + (mx * merge + dx);
                    int kpos = dy * merge + dx;
                    const float *src = vis_normed.data() + (size_t)src_idx * D;
                    for (int c = 0; c < D; c++)
                        dst[c * msq + kpos] = src[c];
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
    // linear2: (D → D). (The projector norm is applied to the vision features
    // BEFORE the merge above, matching Mistral3MultiModalProjector — not here.)
    x = ggml_mul_mat(g, m.proj_linear2_w, x);

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

// Persistent T=1 decode graph: fixed-shape graph reused across all decode steps.
// Avoids per-step ggml_init + sched_alloc_graph overhead.
// Gate with LOCR_DECODE_REBUILD=1 to revert to per-step rebuild.
struct LocPdGraph {
    std::vector<uint8_t>      meta;
    ggml_context*             gctx     = nullptr;
    ggml_cgraph*              gf       = nullptr;
    ggml_tensor*              t_tok_emb = nullptr;  // [D, 1] F32
    ggml_tensor*              t_pos     = nullptr;  // [1] I32
    ggml_tensor*              t_mask    = nullptr;  // [max_kv, 1] F16
    std::vector<ggml_tensor*> t_k_cache;            // [n_layers] each [kv_dim, max_kv-1] F32
    std::vector<ggml_tensor*> t_v_cache;
    ggml_tensor*              t_logits  = nullptr;  // [V] F32
    std::vector<ggml_tensor*> t_k_new;              // [n_layers] each [kv_dim] F32 output
    std::vector<ggml_tensor*> t_v_new;
    int max_kv = 0;
};

static LocPdGraph build_locr_pd_graph(context &ctx, int max_kv) {
    auto &m = ctx.m;
    const int D          = m.lm_dim;
    const int V          = m.vocab_size;
    const int n_layers   = m.lm_layers;
    const int n_heads    = m.lm_heads;
    const int n_kv_heads = m.lm_kv_heads;
    const int head_dim   = m.lm_head_dim;
    const int kv_dim     = head_dim * n_kv_heads;
    const float rms_eps  = m.lm_rms_eps;
    const float rope_theta = m.lm_rope_theta;
    const int max_kv_in  = max_kv - 1;  // past-token slots (current appended at max_kv-1)

    LocPdGraph pd;
    pd.max_kv = max_kv;
    pd.t_k_cache.resize(n_layers);
    pd.t_v_cache.resize(n_layers);
    pd.t_k_new.resize(n_layers);
    pd.t_v_new.resize(n_layers);

    pd.meta.resize(8 * 1024 * 1024);
    ggml_init_params ip = {pd.meta.size(), pd.meta.data(), true};
    pd.gctx = ggml_init(ip);
    auto *g = pd.gctx;
    pd.gf = ggml_new_graph_custom(g, 16384, false);

    auto rn = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(g, ggml_rms_norm(g, t, rms_eps), w);
    };

    pd.t_tok_emb = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, 1);
    ggml_set_name(pd.t_tok_emb, "pd_tok_emb"); ggml_set_input(pd.t_tok_emb);

    pd.t_pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pd.t_pos, "pd_pos"); ggml_set_input(pd.t_pos);

    // Causal mask [max_kv, 1] F16: 0 = attend, -INF = mask.
    // Positions 0..n_past-1 open, n_past..max_kv-2 masked (-INF), max_kv-1 always 0.
    pd.t_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, max_kv, 1);
    ggml_set_name(pd.t_mask, "pd_mask"); ggml_set_input(pd.t_mask);

    ggml_tensor *x = pd.t_tok_emb;
    char name[32];

    for (int il = 0; il < n_layers; il++) {
        auto &L = m.lm[il];

        snprintf(name, sizeof(name), "pd_kc%d", il);
        pd.t_k_cache[il] = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, max_kv_in);
        ggml_set_name(pd.t_k_cache[il], name); ggml_set_input(pd.t_k_cache[il]);
        snprintf(name, sizeof(name), "pd_vc%d", il);
        pd.t_v_cache[il] = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, max_kv_in);
        ggml_set_name(pd.t_v_cache[il], name); ggml_set_input(pd.t_v_cache[il]);

        ggml_tensor *res    = x;
        ggml_tensor *normed = rn(x, L.attn_norm_w);

        ggml_tensor *Q = ggml_mul_mat(g, L.q_w, normed);
        ggml_tensor *K = ggml_mul_mat(g, L.k_w, normed);
        ggml_tensor *V = ggml_mul_mat(g, L.v_w, normed);

        Q = ggml_reshape_3d(g, Q, head_dim, n_heads,    1);
        K = ggml_reshape_3d(g, K, head_dim, n_kv_heads, 1);
        V = ggml_reshape_3d(g, V, head_dim, n_kv_heads, 1);

        if (m.use_qk_norm && L.q_norm_w && L.k_norm_w) {
            Q = ggml_cont(g, ggml_reshape_2d(g, Q, head_dim, n_heads));
            Q = ggml_rms_norm(g, Q, rms_eps); Q = ggml_mul(g, Q, L.q_norm_w);
            Q = ggml_cont(g, ggml_reshape_3d(g, Q, head_dim, n_heads, 1));
            K = ggml_cont(g, ggml_reshape_2d(g, K, head_dim, n_kv_heads));
            K = ggml_rms_norm(g, K, rms_eps); K = ggml_mul(g, K, L.k_norm_w);
            K = ggml_cont(g, ggml_reshape_3d(g, K, head_dim, n_kv_heads, 1));
        }

        Q = ggml_rope_ext(g, Q, pd.t_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, pd.t_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor *Kc = ggml_cont(g, K);
        ggml_tensor *Vc = ggml_cont(g, V);

        snprintf(name, sizeof(name), "pd_kn%d", il);
        pd.t_k_new[il] = ggml_cont(g, ggml_reshape_1d(g, Kc, kv_dim));
        ggml_set_name(pd.t_k_new[il], name); ggml_set_output(pd.t_k_new[il]);
        snprintf(name, sizeof(name), "pd_vn%d", il);
        pd.t_v_new[il] = ggml_cont(g, ggml_reshape_1d(g, Vc, kv_dim));
        ggml_set_name(pd.t_v_new[il], name); ggml_set_output(pd.t_v_new[il]);

        // Full KV: concat(past_cache[max_kv-1], current[1]) → [hd, n_kv_heads, max_kv]
        ggml_tensor *kp    = ggml_reshape_3d(g, pd.t_k_cache[il], head_dim, n_kv_heads, max_kv_in);
        ggml_tensor *Kfull = ggml_concat(g, kp, Kc, 2);
        ggml_tensor *vp    = ggml_reshape_3d(g, pd.t_v_cache[il], head_dim, n_kv_heads, max_kv_in);
        ggml_tensor *Vfull = ggml_concat(g, vp, Vc, 2);

        float sc = 1.0f / sqrtf((float)head_dim);
        Q     = ggml_cont(g, ggml_permute(g, Q,     0, 2, 1, 3));  // [hd, 1,      nh]
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));  // [hd, max_kv, nkv]
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));  // [hd, max_kv, nkv]

        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, pd.t_mask, sc, 0.0f, 0.0f);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));    // [hd, nh, 1]
        attn = ggml_reshape_2d(g, attn, n_heads * head_dim, 1);
        attn = ggml_mul_mat(g, L.o_w, attn);
        x    = ggml_add(g, res, attn);

        ggml_tensor *res2 = x;
        ggml_tensor *nrm2 = rn(x, L.ffn_norm_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, L.gate_w, nrm2));
        ggml_tensor *up   = ggml_mul_mat(g, L.up_w, nrm2);
        x = ggml_add(g, res2, ggml_mul_mat(g, L.down_w, ggml_mul(g, gate, up)));
    }

    x = rn(x, m.lm_norm_w);
    ggml_tensor *logits = ggml_reshape_1d(g, ggml_mul_mat(g, m.embed_tokens, x), V);
    ggml_set_name(logits, "pd_logits"); ggml_set_output(logits);
    pd.t_logits = logits;
    ggml_build_forward_expand(pd.gf, logits);
    for (int il = 0; il < n_layers; il++) {
        ggml_build_forward_expand(pd.gf, pd.t_k_new[il]);
        ggml_build_forward_expand(pd.gf, pd.t_v_new[il]);
    }

    return pd;
}

// Build prefill graph for the full input sequence (text + image tokens).
// Returns logits for the last token + K/V cache tensors.
static bool run_decoder_prefill(context &ctx,
                                 const std::vector<float> &image_embeds,
                                 int n_image_tokens,
                                 int max_new_tokens,
                                 std::string &out_text) {
    const bool bench = ctx.bench;
    auto t_prefill_start = std::chrono::steady_clock::now();
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

    // Qwen3 chat template token IDs (verified with AutoTokenizer):
    //   <|im_start|>user\n  [IMAGE]  OCR this document.<|im_end|>\n<|im_start|>assistant\n
    // Prefix: before image. Suffix: after image + assistant trigger.
    std::vector<int32_t> prefix_ids = {151644, 872, 198};  // <|im_start|> user \n
    std::vector<int32_t> suffix_ids = {
        93495, 419, 2197, 13,  // OCR this document.
        151645, 198,           // <|im_end|> \n
        151644, 77091, 198     // <|im_start|> assistant \n
    };

    int n_prefix = (int)prefix_ids.size();
    int n_suffix = (int)suffix_ids.size();
    int n_prompt = n_prefix + n_image_tokens + n_suffix;

    // Per-row embed dequant: avoids building a ggml graph + alloc + compute cycle
    // just to look up a handful of token IDs.
    ggml_tensor* emb_t = m.embed_tokens;
    const auto* emb_tt = ggml_get_type_traits(emb_t->type);
    const size_t emb_row_bytes = ggml_row_size(emb_t->type, D);
    std::vector<uint8_t> emb_row_buf(emb_row_bytes);
    auto embed_tokens_cpu = [&](const std::vector<int32_t> &ids) -> std::vector<float> {
        int n = (int)ids.size();
        std::vector<float> result((size_t)D * n);
        for (int i = 0; i < n; i++) {
            float *dst = result.data() + (size_t)i * D;
            const size_t off = (size_t)ids[i] * emb_row_bytes;
            if (emb_t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(emb_t, dst, off, (size_t)D * sizeof(float));
            } else {
                ggml_backend_tensor_get(emb_t, emb_row_buf.data(), off, emb_row_bytes);
                emb_tt->to_float(emb_row_buf.data(), dst, D);
            }
        }
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

    // Causal mask: F16 for flash_attn, F32 for manual attention
    ggml_type mask_type = use_flash_attn() ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor *mask = ggml_new_tensor_2d(g, mask_type, n_prompt, n_prompt);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Position IDs: [0, 1, 2, ..., n_prompt-1]
    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_prompt);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    auto rmsnorm = [&rms_eps](ggml_context *gc, ggml_tensor *t, ggml_tensor *w) -> ggml_tensor* {
        return ggml_mul(gc, ggml_rms_norm(gc, t, rms_eps), w);
    };

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

        // Output post-RoPE K/V for KV cache extraction. These are read back via
        // ggml_backend_tensor_get, so they MUST be materialized: V is a
        // ggml_reshape_3d *view* of the projection, and a view marked as an
        // output has its (shared) buffer reused by the no-alloc scheduler →
        // garbage read-back. ggml_cont gives each its own concrete buffer.
        {
            char kname[32], vname[32];
            snprintf(kname, sizeof(kname), "k_out_%d", il);
            snprintf(vname, sizeof(vname), "v_out_%d", il);
            K = ggml_cont(g, K);
            V = ggml_cont(g, V);
            ggml_set_name(K, kname);
            ggml_set_output(K);
            ggml_set_name(V, vname);
            ggml_set_output(V);
        }

        float scale = 1.0f / sqrtf((float)head_dim);
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // (hd, T, nh)

        ggml_tensor *attn;
        if (use_flash_attn()) {
            // flash_attn_ext handles GQA natively — pass K/V with nkv heads directly.
            K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3)); // (hd, T, nkv)
            V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3)); // (hd, T, nkv)
            attn = ggml_flash_attn_ext(g, Q, K, V, mask, scale, 0.0f, 0.0f);
        } else {
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
            K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
            V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
            ggml_tensor *scores = ggml_mul_mat(g, K, Q);
            scores = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);
            ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
            attn = ggml_mul_mat(g, V_perm, scores);
            attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        }
        int q_total = n_heads * head_dim;
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

    // K/V outputs are named k_out_N/v_out_N — extracted after compute via ggml_graph_get_tensor

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
    ggml_tensor *t_mask = ggml_graph_get_tensor(gf, "causal_mask");
    if (use_flash_attn()) {
        std::vector<ggml_fp16_t> mask_data(n_prompt * n_prompt);
        for (int i = 0; i < n_prompt; i++)
            for (int j = 0; j < n_prompt; j++)
                mask_data[i * n_prompt + j] = ggml_fp32_to_fp16((j <= i) ? 0.0f : -INFINITY);
        ggml_backend_tensor_set(t_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<float> mask_data(n_prompt * n_prompt);
        for (int i = 0; i < n_prompt; i++)
            for (int j = 0; j < n_prompt; j++)
                mask_data[i * n_prompt + j] = (j <= i) ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(t_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

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
    // Extract KV cache from prefill graph (read BEFORE freeing g)
    // Disable KV cache with CRISPEMBED_LIGHTONOCR_NO_KV=1 for A/B testing
    std::vector<std::vector<float>> k_cache(n_layers), v_cache(n_layers);
    bool kv_ok = !LOCR_NO_KV;
    for (int il = 0; il < n_layers; il++) {
        char kname[32], vname[32];
        snprintf(kname, sizeof(kname), "k_out_%d", il);
        snprintf(vname, sizeof(vname), "v_out_%d", il);
        ggml_tensor *kt = ggml_graph_get_tensor(gf, kname);
        ggml_tensor *vt = ggml_graph_get_tensor(gf, vname);
        if (!kt || !vt) {
            if (il == 0) fprintf(stderr, "lightonocr: k_out_%d not found in graph\n", il);
            kv_ok = false; break;
        }
        // K/V are 3D (head_dim, n_kv_heads, n_prompt) — read as flat (kv_dim * n_prompt)
        size_t sz = (size_t)ggml_nelements(kt);
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

    ctx.char_confidences.clear();
    std::vector<int32_t> generated;
    int best = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++)
        if (logits_data[v] > best_score) { best_score = logits_data[v]; best = v; }
    generated.push_back(best);
    { float se = 0; for (int v = 0; v < V; v++) se += expf(logits_data[v] - best_score); ctx.char_confidences.push_back(1.0f / se); }

    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
        fprintf(stderr, "[lightonocr-bench] prefill: %lldms\n", (long long)ms);
    }
    fprintf(stderr, "lightonocr: prefill done, first token=%d (score=%.2f)\n", best, best_score);
    fprintf(stderr, "lightonocr: logits[0..4] = [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
            logits_data[0], logits_data[1], logits_data[2], logits_data[3], logits_data[4]);
    fprintf(stderr, "lightonocr: logits[eos=%d] = %.3f\n", eos_id, logits_data[eos_id]);

    if (best == eos_id) {
        out_text = "";
        return true;
    }

    // KV-cached decode: O(n) per step
    long long decode_total_ms = 0;
    int decode_steps = 0;
    int n_kv = n_prompt;
    int kv_repeat = n_heads / n_kv_heads;

    // Persistent decode graph: build once, reuse every step.
    static constexpr int PD_GEN_CAP = 256;
    bool use_pd = kv_ok && !getenv("LOCR_DECODE_REBUILD");
    int pd_max_kv = use_pd ? n_prompt + std::min(max_new_tokens, PD_GEN_CAP) : 0;
    LocPdGraph pd;
    std::vector<ggml_fp16_t> pd_mask_buf;
    std::vector<float> pd_k_tmp, pd_v_tmp;
    bool pd_ready = false;
    if (use_pd && pd_max_kv > n_prompt) {
        pd = build_locr_pd_graph(ctx, pd_max_kv);
        pd_mask_buf.assign(pd_max_kv, ggml_fp32_to_fp16(-INFINITY));
        pd_mask_buf[pd_max_kv - 1] = ggml_fp32_to_fp16(0.0f);
        pd_k_tmp.resize(kv_dim);
        pd_v_tmp.resize(kv_dim);
    } else {
        use_pd = false;
    }

    for (int step = 1; step < max_new_tokens && best != eos_id; step++) {
        if (!kv_ok) break;

        auto t_step_start = std::chrono::steady_clock::now();
        int pos = n_kv;
        // Embed new token
        auto tok_emb = embed_tokens_cpu({(int32_t)best});

        bool did_pd = false;

        // === Persistent decode path ===
        if (use_pd) {
            if (n_kv >= pd_max_kv) {
                use_pd = false;  // KV window full, fall back to per-step rebuild
            }
        }
        if (use_pd) {
            if (!pd_ready) {
                // First gen step: allocate the persistent graph once, upload prefill KV.
                ggml_backend_sched_reset(ctx.sched);
                if (!ggml_backend_sched_alloc_graph(ctx.sched, pd.gf)) {
                    fprintf(stderr, "lightonocr: pd alloc failed — falling back\n");
                    use_pd = false;
                } else {
                    // Unlock all prefill positions in the mask.
                    for (int ki = 0; ki < n_kv; ki++)
                        pd_mask_buf[ki] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, pd_mask_buf.data(), 0,
                                            (size_t)pd_max_kv * sizeof(ggml_fp16_t));
                    // Upload prefill KV (valid rows 0..n_kv-1 per layer).
                    size_t kv_b = (size_t)n_kv * kv_dim * sizeof(float);
                    for (int il = 0; il < n_layers; il++) {
                        ggml_backend_tensor_set(pd.t_k_cache[il], k_cache[il].data(), 0, kv_b);
                        ggml_backend_tensor_set(pd.t_v_cache[il], v_cache[il].data(), 0, kv_b);
                    }
                    pd_ready = true;
                }
            } else {
                // Subsequent gen steps: unlock the slot written last step (position n_kv-1)
                // and upload the one new KV row to that cache slot.
                int prev = n_kv - 1;
                if (prev < pd_max_kv - 1) {
                    pd_mask_buf[prev] = ggml_fp32_to_fp16(0.0f);
                    ggml_backend_tensor_set(pd.t_mask, &pd_mask_buf[prev],
                                            (size_t)prev * sizeof(ggml_fp16_t),
                                            sizeof(ggml_fp16_t));
                }
                size_t off = (size_t)prev * kv_dim * sizeof(float);
                size_t sz  = (size_t)kv_dim * sizeof(float);
                for (int il = 0; il < n_layers; il++) {
                    ggml_backend_tensor_set(pd.t_k_cache[il],
                        k_cache[il].data() + (size_t)prev * kv_dim, off, sz);
                    ggml_backend_tensor_set(pd.t_v_cache[il],
                        v_cache[il].data() + (size_t)prev * kv_dim, off, sz);
                }
            }
        }
        if (use_pd) {
            ggml_backend_tensor_set(pd.t_tok_emb, tok_emb.data(), 0, D * sizeof(float));
            int32_t pos_val = pos;
            ggml_backend_tensor_set(pd.t_pos, &pos_val, 0, sizeof(int32_t));

            ggml_backend_cpu_set_n_threads(ctx.backend, ctx.n_threads);
            ggml_backend_sched_graph_compute(ctx.sched, pd.gf);

            ggml_backend_tensor_get(pd.t_logits, logits_data.data(), 0, V * sizeof(float));
            for (int il = 0; il < n_layers; il++) {
                ggml_backend_tensor_get(pd.t_k_new[il], pd_k_tmp.data(), 0, kv_dim * sizeof(float));
                ggml_backend_tensor_get(pd.t_v_new[il], pd_v_tmp.data(), 0, kv_dim * sizeof(float));
                k_cache[il].insert(k_cache[il].end(), pd_k_tmp.begin(), pd_k_tmp.end());
                v_cache[il].insert(v_cache[il].end(), pd_v_tmp.begin(), pd_v_tmp.end());
            }
            n_kv++;
            did_pd = true;
        }

        // === Per-step rebuild fallback ===
        if (!did_pd) {
        ggml_init_params ip2{ctx.compute_meta.size(), ctx.compute_meta.data(), true};
        ggml_context *g2 = ggml_init(ip2);
        ggml_cgraph *gf2 = ggml_new_graph_custom(g2, 16384, false);

        ggml_tensor *x2 = ggml_new_tensor_2d(g2, GGML_TYPE_F32, D, 1);
        ggml_set_name(x2, "tok_emb"); ggml_set_input(x2);
        ggml_tensor *pos2 = ggml_new_tensor_1d(g2, GGML_TYPE_I32, 1);
        ggml_set_name(pos2, "pos_ids"); ggml_set_input(pos2);

        // side outputs (k_out/v_out) aren't ancestors of the logits — collect
        // so the graph builder doesn't prune them
        std::vector<ggml_tensor *> kv_out2;

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
            ggml_tensor *K_new_flat = ggml_cont(g2, ggml_reshape_1d(g2, K_new, kv_dim));
            ggml_set_name(K_new_flat, name); ggml_set_output(K_new_flat);
            kv_out2.push_back(K_new_flat);
            snprintf(name, sizeof(name), "v_out_%d", il);
            ggml_tensor *V_new_flat = ggml_cont(g2, ggml_reshape_1d(g2, V_new, kv_dim));
            ggml_set_name(V_new_flat, name); ggml_set_output(V_new_flat);
            kv_out2.push_back(V_new_flat);

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

            int seq = n_kv + 1;
            float sc = 1.0f / sqrtf((float)head_dim);
            Q = ggml_cont(g2, ggml_permute(g2, Q, 0, 2, 1, 3));
            ggml_tensor *attn;
            if (use_flash_attn()) {
                // flash_attn_ext handles GQA natively — pass K/V with nkv heads directly.
                K_full = ggml_cont(g2, ggml_permute(g2, K_full, 0, 2, 1, 3));
                V_full = ggml_cont(g2, ggml_permute(g2, V_full, 0, 2, 1, 3));
                attn = ggml_flash_attn_ext(g2, Q, K_full, V_full, nullptr, sc, 0.0f, 0.0f);
            } else {
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
                K_full = ggml_cont(g2, ggml_permute(g2, K_full, 0, 2, 1, 3));
                V_full = ggml_cont(g2, ggml_permute(g2, V_full, 0, 2, 1, 3));
                ggml_tensor *scores = ggml_mul_mat(g2, K_full, Q);
                scores = ggml_soft_max_ext(g2, scores, nullptr, sc, 0.0f);
                ggml_tensor *Vp = ggml_cont(g2, ggml_permute(g2, V_full, 1, 0, 2, 3));
                attn = ggml_mul_mat(g2, Vp, scores);
                attn = ggml_cont(g2, ggml_permute(g2, attn, 0, 2, 1, 3));
            }
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
        for (ggml_tensor *t : kv_out2) ggml_build_forward_expand(gf2, t);

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
        } // end fallback

        best = 0; best_score = -INFINITY;
        for (int v = 0; v < V; v++)
            if (logits_data[v] > best_score) { best_score = logits_data[v]; best = v; }
        if (best == eos_id) break;
        generated.push_back(best);
        { float se = 0; for (int v = 0; v < V; v++) se += expf(logits_data[v] - best_score); ctx.char_confidences.push_back(1.0f / se); }
        auto t_step_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        if (bench) {
            long long step_ms_ll = (long long)step_ms;
            decode_total_ms += step_ms_ll;
            decode_steps++;
            fprintf(stderr, "[lightonocr-bench] decode_step[%d]: %lldms\n", step, step_ms_ll);
        } else {
            fprintf(stderr, "  gen[%d] tok=%d (%.0fms)%s\n", step, best, step_ms, kv_ok ? " cached" : "");
        }
    }

    if (bench) {
        fprintf(stderr, "[lightonocr-bench] decode_total: %lldms (%d steps)\n", decode_total_ms, decode_steps);
    }

    if (pd.gctx) ggml_free(pd.gctx);

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
    const bool bench = ctx.bench;
    auto t_total = std::chrono::steady_clock::now();
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
    // Debug: print first patch values for parity check
    fprintf(stderr, "lightonocr: patch[0,:5] = [%.8f, %.8f, %.8f, %.8f, %.8f]\n",
            patch_embeds[0], patch_embeds[1], patch_embeds[2], patch_embeds[3], patch_embeds[4]);
    // (patch[7] debug removed — was used for image resize parity debugging)

    // Step 2: 2D RoPE
    std::vector<float> rope_cos, rope_sin;
    compute_2d_rope(ph, pw, ctx.m.vis_head_dim, ctx.m.vis_rope_theta, rope_cos, rope_sin);
    if (LOCR_DEBUG) {
        fprintf(stderr, "lightonocr: rope_cos[0,:5] = [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                rope_cos[0], rope_cos[1], rope_cos[2], rope_cos[3], rope_cos[4]);
        fprintf(stderr, "lightonocr: rope_sin[0,:5] = [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                rope_sin[0], rope_sin[1], rope_sin[2], rope_sin[3], rope_sin[4]);
    }

    // Step 3: vision encoder
    auto t_vis = std::chrono::steady_clock::now();
    std::vector<float> vis_out;
    if (!run_vision_encoder(ctx, patch_embeds, rope_cos, rope_sin, n_patches, vis_out)) {
        return "";
    }
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_vis).count();
        fprintf(stderr, "[lightonocr-bench] vision_encoder: %lldms\n", (long long)ms);
    }
    fprintf(stderr, "lightonocr: vision encoder done\n");
    if (LOCR_DEBUG) {
        fprintf(stderr, "lightonocr: vis_out[0,:5] = [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                vis_out[0], vis_out[1], vis_out[2], vis_out[3], vis_out[4]);
    }

    // Step 4: projection (spatial merge + MLP)
    auto t_proj = std::chrono::steady_clock::now();
    std::vector<float> proj_out;
    if (!run_projection(ctx, vis_out, ph, pw, proj_out)) {
        return "";
    }
    int n_image_tokens = (ph / ctx.m.spatial_merge_size) * (pw / ctx.m.spatial_merge_size);
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_proj).count();
        fprintf(stderr, "[lightonocr-bench] projection: %lldms\n", (long long)ms);
    }
    fprintf(stderr, "lightonocr: projection done → %d image tokens\n", n_image_tokens);

    if (const char *dp = getenv("CRISPEMBED_LIGHTON_DUMP")) {
        auto wr = [&](const char *suf, const std::vector<float> &v) {
            std::string p = std::string(dp) + suf;
            FILE *f = fopen(p.c_str(), "wb");
            if (f) { fwrite(v.data(), sizeof(float), v.size(), f); fclose(f); }
            fprintf(stderr, "lighton dump %s: %zu floats\n", suf, v.size());
        };
        wr("_vis.bin", vis_out);
        wr("_proj.bin", proj_out);
    }

    // Step 5: Qwen3 decoder — greedy generation from image tokens
    std::string gen_text;
    if (!run_decoder_prefill(ctx, proj_out, n_image_tokens, max_tokens, gen_text)) {
        fprintf(stderr, "lightonocr: decoder failed\n");
        return "";
    }
    if (bench) {
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[lightonocr-bench] total: %lldms\n", (long long)total_ms);
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

extern "C" const float * lightonocr_confidences(const lightonocr_context * c, int * n) {
    if (!c || c->ctx.char_confidences.empty()) { if (n) *n = 0; return nullptr; }
    if (n) *n = (int)c->ctx.char_confidences.size();
    return c->ctx.char_confidences.data();
}

extern "C" float lightonocr_mean_confidence(const lightonocr_context * c) {
    if (!c || c->ctx.char_confidences.empty()) return 0.0f;
    double s = 0; for (float v : c->ctx.char_confidences) s += v;
    return (float)(s / c->ctx.char_confidences.size());
}
