// lightonocr.cpp — LightOnOCR-2-1B inference engine.
//
// Pixtral ViT (24L, 2D RoPE) + Qwen3 decoder (28L, QK norm, GQA).
// Single GGUF, CPU-only via ggml_backend_sched.

#include "lightonocr.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#define STB_IMAGE_STATIC
#include "../../ggml/examples/stb_image.h"

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
// Stub: generation (to be completed with full ggml graph)
// ---------------------------------------------------------------------------

std::string recognize_raw(context &ctx,
                           const uint8_t *pixels, int width, int height, int channels,
                           int max_tokens) {
    // Convert to RGB if needed
    std::vector<uint8_t> rgb;
    const uint8_t *rgb_data = pixels;
    if (channels == 1) {
        rgb.resize(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = pixels[i];
        }
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

    // TODO: build ggml graph for vision encoder + projection + LM decoder
    // For now, return a placeholder
    ctx.last_text = "[lightonocr engine loaded — inference graph pending]";
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
