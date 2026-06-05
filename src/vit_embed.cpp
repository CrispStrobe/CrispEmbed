// vit_embed.cpp — Standalone ViT image encoder for SigLIP / CLIP.
//
// Standard ViT: patch_embed (conv2d) → + pos_embd → N × pre-LN blocks
// → post_ln → optional attention pooling head → embedding.
//
// This is intentionally simpler than bidirlm_vision.cpp which handles
// 2D RoPE, deepstack, and block-diagonal attention masks.

#include "vit_embed.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vit_embed {

struct layer {
    ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    ggml_tensor * q_w = nullptr, * q_b = nullptr;
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

struct attn_pool_head {
    ggml_tensor * probe = nullptr;       // [1, 1, H]
    ggml_tensor * in_proj_w = nullptr;   // [3H, H]
    ggml_tensor * in_proj_b = nullptr;   // [3H]
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * ln_w = nullptr, * ln_b = nullptr;
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

struct context {
    int hidden = 0;
    int n_layers = 0;
    int n_heads = 0;
    int intermediate = 0;
    int img_size = 0;
    int patch_size = 0;
    int n_patches = 0;
    int n_channels = 3;
    float ln_eps = 1e-6f;
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3]  = {0.5f, 0.5f, 0.5f};
    bool has_cls_token = false;
    bool has_attn_pool = false;
    bool has_visual_proj = false;

    // Weights
    ggml_tensor * patch_embed_w = nullptr;
    ggml_tensor * patch_embed_b = nullptr;
    ggml_tensor * pos_embd = nullptr;
    ggml_tensor * cls_token = nullptr;
    ggml_tensor * pre_ln_w = nullptr, * pre_ln_b = nullptr;
    ggml_tensor * post_ln_w = nullptr, * post_ln_b = nullptr;
    ggml_tensor * visual_proj_w = nullptr;
    std::vector<layer> layers;
    attn_pool_head head;

    // Backend
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;
};

bool load(context** out, const char* path, int n_threads) {
    auto* ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;

    // Read metadata
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) {
        fprintf(stderr, "vit_embed: cannot open %s\n", path);
        return false;
    }

    auto u32 = [&](const char* k, int d) -> int {
        int64_t i = gguf_find_key(g, k);
        return i >= 0 ? (int)gguf_get_val_u32(g, i) : d;
    };
    auto f32v = [&](const char* k, float d) -> float {
        int64_t i = gguf_find_key(g, k);
        return i >= 0 ? gguf_get_val_f32(g, i) : d;
    };
    auto boolv = [&](const char* k, bool d) -> bool {
        int64_t i = gguf_find_key(g, k);
        return i >= 0 ? gguf_get_val_bool(g, i) : d;
    };

    ctx->hidden       = u32("vit.hidden_size", 768);
    ctx->n_layers     = u32("vit.num_hidden_layers", 12);
    ctx->n_heads      = u32("vit.num_attention_heads", 12);
    ctx->intermediate = u32("vit.intermediate_size", 3072);
    ctx->img_size     = u32("vit.image_size", 384);
    ctx->patch_size   = u32("vit.patch_size", 16);
    ctx->n_patches    = u32("vit.num_patches", 576);
    ctx->n_channels   = u32("vit.num_channels", 3);
    ctx->ln_eps       = f32v("vit.layer_norm_eps", 1e-6f);
    ctx->has_cls_token   = boolv("vit.has_cls_token", false);
    ctx->has_attn_pool   = boolv("vit.has_attn_pool", false);
    ctx->has_visual_proj = boolv("vit.has_visual_proj", false);

    // Read per-channel image normalization from GGUF (falls back to SigLIP defaults)
    auto read_f32_arr3 = [&](const char* key, float* dst) {
        int64_t i = gguf_find_key(g, key);
        if (i >= 0 && gguf_get_arr_n(g, i) >= 3) {
            const float* src = (const float*)gguf_get_arr_data(g, i);
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
    };
    read_f32_arr3("vit.image_mean", ctx->image_mean);
    read_f32_arr3("vit.image_std",  ctx->image_std);

    core_gguf::free_metadata(g);

    fprintf(stderr, "vit_embed: hidden=%d layers=%d heads=%d patches=%d img=%d patch=%d"
            " mean=[%.3f,%.3f,%.3f] std=[%.3f,%.3f,%.3f]\n",
            ctx->hidden, ctx->n_layers, ctx->n_heads, ctx->n_patches,
            ctx->img_size, ctx->patch_size,
            ctx->image_mean[0], ctx->image_mean[1], ctx->image_mean[2],
            ctx->image_std[0], ctx->image_std[1], ctx->image_std[2]);

    // Load weights
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    if (!core_gguf::load_weights(path, ctx->backend, "vit", ctx->wl)) {
        fprintf(stderr, "vit_embed: failed to load weights\n");
        return false;
    }

    auto get = [&](const std::string& n) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(n);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    // Embeddings
    ctx->patch_embed_w = get("patch_embed.weight");
    ctx->patch_embed_b = get("patch_embed.bias");
    ctx->pos_embd      = get("position_embd.weight");
    ctx->cls_token     = get("cls_token");
    ctx->pre_ln_w      = get("pre_ln.weight");
    ctx->pre_ln_b      = get("pre_ln.bias");
    ctx->post_ln_w     = get("post_ln.weight");
    ctx->post_ln_b     = get("post_ln.bias");
    ctx->visual_proj_w = get("visual_proj.weight");

    if (!ctx->patch_embed_w || !ctx->pos_embd) {
        fprintf(stderr, "vit_embed: missing patch_embed or position_embd\n");
        return false;
    }

    // Encoder layers
    ctx->layers.resize(ctx->n_layers);
    for (int i = 0; i < ctx->n_layers; i++) {
        auto pfx = "enc." + std::to_string(i) + ".";
        auto& L = ctx->layers[i];
        L.ln1_w = get(pfx + "ln1.weight");
        L.ln1_b = get(pfx + "ln1.bias");
        L.q_w   = get(pfx + "attn.q.weight");
        L.q_b   = get(pfx + "attn.q.bias");
        L.k_w   = get(pfx + "attn.k.weight");
        L.k_b   = get(pfx + "attn.k.bias");
        L.v_w   = get(pfx + "attn.v.weight");
        L.v_b   = get(pfx + "attn.v.bias");
        L.o_w   = get(pfx + "attn.o.weight");
        L.o_b   = get(pfx + "attn.o.bias");
        L.ln2_w = get(pfx + "ln2.weight");
        L.ln2_b = get(pfx + "ln2.bias");
        L.fc1_w = get(pfx + "ffn.fc1.weight");
        L.fc1_b = get(pfx + "ffn.fc1.bias");
        L.fc2_w = get(pfx + "ffn.fc2.weight");
        L.fc2_b = get(pfx + "ffn.fc2.bias");

        if (!L.ln1_w || !L.q_w || !L.fc1_w) {
            fprintf(stderr, "vit_embed: missing tensors for layer %d\n", i);
            return false;
        }
    }

    // Attention pooling head (SigLIP)
    if (ctx->has_attn_pool) {
        ctx->head.probe      = get("head.probe");
        ctx->head.in_proj_w  = get("head.attn.in_proj.weight");
        ctx->head.in_proj_b  = get("head.attn.in_proj.bias");
        ctx->head.o_w        = get("head.attn.o.weight");
        ctx->head.o_b        = get("head.attn.o.bias");
        ctx->head.ln_w       = get("head.ln.weight");
        ctx->head.ln_b       = get("head.ln.bias");
        ctx->head.fc1_w      = get("head.mlp.fc1.weight");
        ctx->head.fc1_b      = get("head.mlp.fc1.bias");
        ctx->head.fc2_w      = get("head.mlp.fc2.weight");
        ctx->head.fc2_b      = get("head.mlp.fc2.bias");
    }

    fprintf(stderr, "vit_embed: loaded %d layers, %s pooling\n",
            ctx->n_layers, ctx->has_attn_pool ? "attention" : "mean");
    return true;
}

std::vector<float> encode(context* ctx, const float* pixels, int H, int W) {
    if (!ctx || H != ctx->img_size || W != ctx->img_size) {
        fprintf(stderr, "vit_embed: image must be %dx%d, got %dx%d\n",
                ctx->img_size, ctx->img_size, H, W);
        return {};
    }

    const int T = ctx->n_patches;  // number of patches
    const int D = ctx->hidden;
    const int nh = ctx->n_heads;
    const int hd = D / nh;
    const float eps = ctx->ln_eps;
    const int ps = ctx->patch_size;
    const int grid = ctx->img_size / ps;  // patches per side

    // Build ggml graph
    const int extra = ctx->has_attn_pool ? 60 : 0;
    size_t buf_size = ggml_tensor_overhead() * (ctx->n_layers * 40 + 100 + extra)
                    + ggml_graph_overhead_custom(ctx->n_layers * 30 + 100 + extra, false);
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context* g = ggml_init(p);

    // Input: pixel patches [D, T] — we do patch embedding manually
    // pixels input: [C, H, W] = [3, img_size, img_size]
    ggml_tensor* pixel_in = ggml_new_tensor_3d(g, GGML_TYPE_F32,
                                                W, H, ctx->n_channels);
    ggml_set_name(pixel_in, "pixels");
    ggml_set_input(pixel_in);

    // Patch embedding via conv2d
    // Input pixels: [W, H, C] in ggml layout (ne[0]=W, ne[1]=H, ne[2]=C)
    // Kernel: [kw, kh, C_in, C_out] in ggml
    // Output: [OW, OH, C_out] where OW = OH = grid
    ggml_tensor* x = ggml_conv_2d(g, ctx->patch_embed_w, pixel_in,
                                   ps, ps, 0, 0, 1, 1);
    // x shape: [grid, grid, D] = [24, 24, 768]

    if (ctx->patch_embed_b) {
        // bias [D] broadcasts over spatial dims (ggml_add repeats ne[0] match)
        x = ggml_add(g, x, ggml_reshape_3d(g, ctx->patch_embed_b, 1, 1, D));
    }

    // Reshape [OW, OH, D] → [D, T] where T = OH*OW
    // ggml ne: ne[0]=OW, ne[1]=OH, ne[2]=D
    // HF flattens (D, OH, OW) → (D, T) with row-major spatial order (OH then OW)
    // permute(2,1,0): [OW, OH, D] → [D, OH, OW] — matches HF's (D, H, W) layout
    x = ggml_cont(g, ggml_permute(g, x, 2, 1, 0, 3));  // [D, OH, OW]
    x = ggml_reshape_2d(g, x, D, T);                     // [D, T]

    // Position embeddings: pos_embd stored as [D, T] (from converter)
    x = ggml_add(g, x, ctx->pos_embd);

    // Pre-LN (CLIP only)
    if (ctx->pre_ln_w) {
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, ctx->pre_ln_w);
        if (ctx->pre_ln_b) x = ggml_add(g, x, ctx->pre_ln_b);
    }

    // Encoder layers (pre-LN ViT: LN → Attn → Add → LN → MLP → Add)
    for (int il = 0; il < ctx->n_layers; il++) {
        const auto& L = ctx->layers[il];
        ggml_tensor* residual = x;

        // Pre-attention LN
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, L.ln1_w);
        if (L.ln1_b) x = ggml_add(g, x, L.ln1_b);

        // Q/K/V projections
        ggml_tensor* Q = ggml_mul_mat(g, L.q_w, x);
        ggml_tensor* K = ggml_mul_mat(g, L.k_w, x);
        ggml_tensor* V = ggml_mul_mat(g, L.v_w, x);
        if (L.q_b) Q = ggml_add(g, Q, L.q_b);
        if (L.k_b) K = ggml_add(g, K, L.k_b);
        if (L.v_b) V = ggml_add(g, V, L.v_b);

        // Reshape [D, T] → [hd, nh, T] → permute to [hd, T, nh]
        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K = ggml_reshape_3d(g, K, hd, nh, T);
        V = ggml_reshape_3d(g, V, hd, nh, T);
        Q = ggml_permute(g, Q, 0, 2, 1, 3);  // [hd, T, nh]
        K = ggml_permute(g, K, 0, 2, 1, 3);
        V = ggml_permute(g, V, 0, 2, 1, 3);

        // Flash attention (no causal mask for ViT encoder)
        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor* attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // Result: [hd, nh, T] → reshape to [D, T]
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection
        attn = ggml_mul_mat(g, L.o_w, attn);
        if (L.o_b) attn = ggml_add(g, attn, L.o_b);

        // Residual add
        x = ggml_add(g, residual, attn);

        // Pre-FFN LN
        residual = x;
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, L.ln2_w);
        if (L.ln2_b) x = ggml_add(g, x, L.ln2_b);

        // MLP: fc1 → GELU → fc2
        x = ggml_mul_mat(g, L.fc1_w, x);
        if (L.fc1_b) x = ggml_add(g, x, L.fc1_b);
        x = ggml_gelu(g, x);
        x = ggml_mul_mat(g, L.fc2_w, x);
        if (L.fc2_b) x = ggml_add(g, x, L.fc2_b);

        // Residual add
        x = ggml_add(g, residual, x);
    }

    // Post-LayerNorm
    if (ctx->post_ln_w) {
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, ctx->post_ln_w);
        if (ctx->post_ln_b) x = ggml_add(g, x, ctx->post_ln_b);
    }

    ggml_tensor* pooled = nullptr;

    if (ctx->has_attn_pool && ctx->head.probe && ctx->head.in_proj_w) {
        // ── SigLIP attention pooling head ──
        // x is [D, T] after post_ln.
        // probe is [D, 1, 1] or [D, 1] or [D] — reshape to [D, 1]
        const auto& H = ctx->head;
        ggml_tensor* probe = ggml_reshape_2d(g, H.probe, D, 1);

        // Concatenate [probe; x] along token dim → [D, T+1]
        ggml_tensor* x_cat = ggml_concat(g, probe, x, 1);  // dim=1: concat along ne[1]
        const int S = T + 1;  // total sequence length

        // Fused QKV projection: in_proj_w [3D, D] @ x_cat [D, S] + bias → [3D, S]
        ggml_tensor* qkv = ggml_mul_mat(g, H.in_proj_w, x_cat);
        if (H.in_proj_b) qkv = ggml_add(g, qkv, H.in_proj_b);

        // Split Q, K, V — each [D, S]
        // Q from first D rows, K from next D, V from last D
        ggml_tensor* Qa = ggml_view_2d(g, qkv, D, S, qkv->nb[1], 0);
        ggml_tensor* Ka = ggml_view_2d(g, qkv, D, S, qkv->nb[1], D * sizeof(float));
        ggml_tensor* Va = ggml_view_2d(g, qkv, D, S, qkv->nb[1], 2 * D * sizeof(float));

        // Q: take only probe position (token 0) → [D, 1]
        Qa = ggml_view_2d(g, Qa, D, 1, Qa->nb[1], 0);

        // Reshape for multi-head attention:
        // Q [D, 1] → [hd, nh, 1], K [D, S] → [hd, nh, S], V [D, S] → [hd, nh, S]
        Qa = ggml_reshape_3d(g, Qa, hd, nh, 1);
        Ka = ggml_reshape_3d(g, Ka, hd, nh, S);
        Va = ggml_reshape_3d(g, Va, hd, nh, S);

        // Permute to [hd, seq, nh] for ggml_flash_attn_ext
        Qa = ggml_permute(g, Qa, 0, 2, 1, 3);  // [hd, 1, nh]
        Ka = ggml_permute(g, Ka, 0, 2, 1, 3);  // [hd, S, nh]
        Va = ggml_permute(g, Va, 0, 2, 1, 3);  // [hd, S, nh]

        // Scaled dot-product attention
        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor* attn_out = ggml_flash_attn_ext(g, Qa, Ka, Va, nullptr, scale, 0.0f, 0.0f);
        // attn_out: [hd, nh, 1] → reshape to [D, 1]
        attn_out = ggml_reshape_2d(g, attn_out, D, 1);

        // Output projection
        attn_out = ggml_mul_mat(g, H.o_w, attn_out);
        if (H.o_b) attn_out = ggml_add(g, attn_out, H.o_b);

        // Residual add with probe
        attn_out = ggml_add(g, probe, attn_out);

        // LayerNorm
        attn_out = ggml_norm(g, attn_out, eps);
        attn_out = ggml_mul(g, attn_out, H.ln_w);
        if (H.ln_b) attn_out = ggml_add(g, attn_out, H.ln_b);

        // MLP: fc1 → GELU → fc2
        ggml_tensor* mlp = ggml_mul_mat(g, H.fc1_w, attn_out);
        if (H.fc1_b) mlp = ggml_add(g, mlp, H.fc1_b);
        mlp = ggml_gelu(g, mlp);
        mlp = ggml_mul_mat(g, H.fc2_w, mlp);
        if (H.fc2_b) mlp = ggml_add(g, mlp, H.fc2_b);

        // Reshape to [D]
        pooled = ggml_reshape_1d(g, mlp, D);
    } else {
        // ── Mean pooling ──
        // x is [D, T].
        // ggml_sum_rows sums over ne[0] → [1, T].
        // We need sum over ne[1] (T dim). Transpose first.
        ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, x));  // [T, D]
        ggml_tensor* summed = ggml_sum_rows(g, xt);  // [1, D]
        pooled = ggml_reshape_1d(g, summed, D);
        pooled = ggml_scale(g, pooled, 1.0f / (float)T);
    }

    // Visual projection (CLIP)
    if (ctx->has_visual_proj && ctx->visual_proj_w) {
        pooled = ggml_mul_mat(g, ctx->visual_proj_w, pooled);
    }

    // L2 normalize
    // ggml doesn't have L2 norm — do it in post-processing

    ggml_set_name(pooled, "embedding");
    ggml_set_output(pooled);

    // Build and compute graph
    ggml_cgraph* gf = ggml_new_graph_custom(g, ctx->n_layers * 30 + 100 + extra, false);
    ggml_build_forward_expand(gf, pooled);

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    // Set input pixels
    ggml_tensor* px = ggml_graph_get_tensor(gf, "pixels");
    ggml_backend_tensor_set(px, pixels, 0, ctx->n_channels * H * W * sizeof(float));

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embedding");
    int out_dim = (int)ggml_nelements(out);
    std::vector<float> result(out_dim);
    ggml_backend_tensor_get(out, result.data(), 0, out_dim * sizeof(float));

    // L2 normalize
    float norm = 0.0f;
    for (float v : result) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (float& v : result) v /= norm;
    }

    ggml_gallocr_free(alloc);
    ggml_free(g);

    return result;
}

int dim(const context* ctx) {
    return ctx ? ctx->hidden : 0;
}

int image_size(const context* ctx) {
    return ctx ? ctx->img_size : 0;
}

void free(context* ctx) {
    if (ctx) {
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

} // namespace vit_embed (close before stb_image to avoid free() clash)

// stb_image — must be outside any namespace so STBI_FREE resolves to ::free
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

namespace vit_embed {
// ── Image file loading + preprocessing ──────────────────────────────────

// Bilinear resize from [src_h, src_w, 3] uint8 to [dst_h, dst_w, 3] float [0,1]
static void bilinear_resize(const unsigned char* src, int src_w, int src_h,
                            float* dst, int dst_w, int dst_h) {
    const float sx = (float)src_w / dst_w;
    const float sy = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)fy; if (y0 < 0) y0 = 0;
        int y1 = y0 + 1; if (y1 >= src_h) y1 = src_h - 1;
        float wy = fy - y0; if (wy < 0) wy = 0;

        for (int x = 0; x < dst_w; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = (int)fx; if (x0 < 0) x0 = 0;
            int x1 = x0 + 1; if (x1 >= src_w) x1 = src_w - 1;
            float wx = fx - x0; if (wx < 0) wx = 0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(y0 * src_w + x0) * 3 + c] / 255.0f;
                float v01 = src[(y0 * src_w + x1) * 3 + c] / 255.0f;
                float v10 = src[(y1 * src_w + x0) * 3 + c] / 255.0f;
                float v11 = src[(y1 * src_w + x1) * 3 + c] / 255.0f;
                float v = v00 * (1-wx) * (1-wy) + v01 * wx * (1-wy)
                        + v10 * (1-wx) * wy     + v11 * wx * wy;
                // Output in CHW order: [c, y, x]
                dst[c * dst_h * dst_w + y * dst_w + x] = v;
            }
        }
    }
}

std::vector<float> encode_file(context* ctx, const char* image_path) {
    if (!ctx || !image_path) return {};

    int w, h, channels;
    unsigned char* data = stbi_load(image_path, &w, &h, &channels, 3);
    if (!data) {
        fprintf(stderr, "vit_embed: cannot load image '%s'\n", image_path);
        return {};
    }

    int sz = ctx->img_size;
    std::vector<float> pixels(3 * sz * sz);

    // Resize to model's image size
    bilinear_resize(data, w, h, pixels.data(), sz, sz);
    stbi_image_free(data);

    // Normalize: (pixel - mean) / std using per-channel values from GGUF
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < sz * sz; i++) {
            pixels[c * sz * sz + i] = (pixels[c * sz * sz + i] - ctx->image_mean[c]) / ctx->image_std[c];
        }
    }

    return encode(ctx, pixels.data(), sz, sz);
}

} // namespace vit_embed
