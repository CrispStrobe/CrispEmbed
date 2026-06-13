// glm_ocr.cpp — GLM-OCR inference engine.
// See glm_ocr.h for architecture overview.

#include "glm_ocr.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace glm_ocr {

namespace {

// ── Hparams ─────────────────────────────────────────────────────────

bool load_hparams(context &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32v = [&](const char *k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto &vhp = ctx.m.vhp;
    auto &lhp = ctx.m.lhp;

    vhp.depth             = u32("glm_ocr.vision.depth", vhp.depth);
    vhp.hidden_size       = u32("glm_ocr.vision.hidden_size", vhp.hidden_size);
    vhp.intermediate_size = u32("glm_ocr.vision.intermediate_size", vhp.intermediate_size);
    vhp.num_heads         = u32("glm_ocr.vision.num_heads", vhp.num_heads);
    vhp.patch_size        = u32("glm_ocr.vision.patch_size", vhp.patch_size);
    vhp.image_size        = u32("glm_ocr.vision.image_size", vhp.image_size);
    vhp.temporal_patch_size = u32("glm_ocr.vision.temporal_patch_size", vhp.temporal_patch_size);
    vhp.spatial_merge_size  = u32("glm_ocr.vision.spatial_merge_size", vhp.spatial_merge_size);
    vhp.out_hidden_size   = u32("glm_ocr.vision.out_hidden_size", vhp.out_hidden_size);
    vhp.rms_norm_eps      = f32v("glm_ocr.vision.rms_norm_eps", vhp.rms_norm_eps);
    vhp.head_dim          = vhp.hidden_size / vhp.num_heads;

    int idx = gguf_find_key(g, "glm_ocr.vision.image_mean");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *d = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_mean[i] = d[i];
    }
    idx = gguf_find_key(g, "glm_ocr.vision.image_std");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *d = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_std[i] = d[i];
    }

    lhp.vocab_size          = u32("glm_ocr.vocab_size", lhp.vocab_size);
    lhp.hidden_size         = u32("glm_ocr.hidden_size", lhp.hidden_size);
    lhp.intermediate_size   = u32("glm_ocr.intermediate_size", lhp.intermediate_size);
    lhp.num_hidden_layers   = u32("glm_ocr.num_hidden_layers", lhp.num_hidden_layers);
    lhp.num_attention_heads = u32("glm_ocr.num_attention_heads", lhp.num_attention_heads);
    lhp.num_key_value_heads = u32("glm_ocr.num_key_value_heads", lhp.num_key_value_heads);
    lhp.head_dim            = u32("glm_ocr.head_dim", lhp.head_dim);
    lhp.max_position_embeddings = u32("glm_ocr.max_position_embeddings", lhp.max_position_embeddings);
    lhp.rms_norm_eps        = f32v("glm_ocr.rms_norm_eps", lhp.rms_norm_eps);
    lhp.rope_theta          = f32v("glm_ocr.rope_theta", lhp.rope_theta);
    lhp.image_token_id      = u32("glm_ocr.image_token_id", lhp.image_token_id);
    lhp.eos_token_id        = u32("glm_ocr.tokenizer.eos_id", lhp.eos_token_id);

    idx = gguf_find_key(g, "glm_ocr.rope_sections");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *d = (const uint32_t *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) lhp.rope_sections[i] = (int)d[i];
    }

    // Tokenizer
    ctx.tok.eos_id = (int)lhp.eos_token_id;
    int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        int n = gguf_get_arr_n(g, vocab_idx);
        ctx.tok.id_to_piece.resize(n);
        for (int i = 0; i < n; i++)
            ctx.tok.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
        ctx.tok.vocab_size = n;
    }

    core_gguf::free_metadata(g);
    return true;
}

// ── Tensor loading ──────────────────────────────────────────────────

bool load_tensors(context &ctx, const char *path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "glm_ocr", wl))
        return false;
    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;

    auto &m = ctx.m;
    auto get = [&](const std::string &name) -> ggml_tensor * {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Vision
    m.patch_embed_w = get("v.patch_embed.weight");
    m.patch_embed_b = get("v.patch_embed.bias");

    m.vis_blocks.resize(m.vhp.depth);
    for (uint32_t i = 0; i < m.vhp.depth; i++) {
        auto &b = m.vis_blocks[i];
        std::string p = "v.blk." + std::to_string(i) + ".";
        b.norm1_w    = get(p + "norm1.weight");
        b.norm2_w    = get(p + "norm2.weight");
        b.qkv_w      = get(p + "attn_qkv.weight");
        b.qkv_b      = get(p + "attn_qkv.bias");
        b.proj_w     = get(p + "attn_proj.weight");
        b.proj_b     = get(p + "attn_proj.bias");
        b.q_norm_w   = get(p + "attn_q_norm.weight");
        b.k_norm_w   = get(p + "attn_k_norm.weight");
        b.ffn_gate_w = get(p + "ffn_gate.weight");
        b.ffn_gate_b = get(p + "ffn_gate.bias");
        b.ffn_up_w   = get(p + "ffn_up.weight");
        b.ffn_up_b   = get(p + "ffn_up.bias");
        b.ffn_down_w = get(p + "ffn_down.weight");
        b.ffn_down_b = get(p + "ffn_down.bias");
    }

    m.post_layernorm_w = get("v.post_layernorm.weight");
    m.downsample_w = get("v.downsample.weight");
    m.downsample_b = get("v.downsample.bias");

    m.merger.proj_w = get("v.merger.proj.weight");
    m.merger.gate_w = get("v.merger.gate.weight");
    m.merger.up_w   = get("v.merger.up.weight");
    m.merger.down_w = get("v.merger.down.weight");
    m.merger.norm_w = get("v.merger.norm.weight");
    m.merger.norm_b = get("v.merger.norm.bias");

    // LLM
    m.embed_tokens = get("l.embed_tokens.weight");

    m.llm_layers.resize(m.lhp.num_hidden_layers);
    for (uint32_t i = 0; i < m.lhp.num_hidden_layers; i++) {
        auto &ly = m.llm_layers[i];
        std::string p = "l.blk." + std::to_string(i) + ".";
        ly.input_layernorm_w         = get(p + "input_layernorm.weight");
        ly.post_self_attn_layernorm_w = get(p + "post_self_attn_layernorm.weight");
        ly.post_attention_layernorm_w = get(p + "post_attention_layernorm.weight");
        ly.post_mlp_layernorm_w      = get(p + "post_mlp_layernorm.weight");
        ly.q_w = get(p + "attn_q.weight");
        ly.k_w = get(p + "attn_k.weight");
        ly.v_w = get(p + "attn_v.weight");
        ly.o_w = get(p + "attn_o.weight");
        ly.ffn_gate_w = get(p + "ffn_gate.weight");
        ly.ffn_up_w   = get(p + "ffn_up.weight");
        ly.ffn_down_w = get(p + "ffn_down.weight");
    }

    m.output_norm_w = get("l.output_norm.weight");
    m.lm_head_w     = get("l.lm_head.weight");

    return true;
}

// ── Vision encoder graph ────────────────────────────────────────────

struct vision_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *pixel_in = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<ggml_tensor *> layer_outputs;
};

vision_graph build_vision_graph(context &ctx, int n_patches) {
    vision_graph vg;
    const auto &vhp = ctx.m.vhp;
    const int D = (int)vhp.hidden_size;
    const int nh = (int)vhp.num_heads;
    const int hd = D / nh;
    const int n_layers = (int)vhp.depth;
    const float eps = vhp.rms_norm_eps;
    const int P = (int)vhp.patch_size;
    const int T_p = (int)vhp.temporal_patch_size;
    const int patch_flat = 3 * T_p * P * P;

    size_t meta_size = (size_t)(n_layers * 40 + 200) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(16384, false);
    ctx.compute_meta.resize(meta_size);
    ggml_init_params ip{meta_size, ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    vg.gctx = g;

    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Input: (patch_flat, n_patches)
    ggml_tensor *pixel_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, patch_flat, n_patches);
    ggml_set_name(pixel_in, "pixel_in");
    ggml_set_input(pixel_in);
    vg.pixel_in = pixel_in;

    // Patch embedding
    ggml_tensor *x = ggml_mul_mat(g, ctx.m.patch_embed_w, pixel_in);
    if (ctx.m.patch_embed_b) x = ggml_add(g, x, ctx.m.patch_embed_b);

    ggml_set_name(x, "vis_patch_embed");
    ggml_set_output(x);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    // CogViT transformer layers
    for (int i = 0; i < n_layers; i++) {
        auto &blk = ctx.m.vis_blocks[i];
        int T = n_patches;

        // Pre-norm
        ggml_tensor *h = rmsnorm(x, blk.norm1_w);

        // Fused QKV
        ggml_tensor *qkv = ggml_mul_mat(g, blk.qkv_w, h);
        if (blk.qkv_b) qkv = ggml_add(g, qkv, blk.qkv_b);

        ggml_tensor *Q = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor *K = ggml_view_2d(g, qkv, D, T, qkv->nb[1], D * sizeof(float));
        ggml_tensor *V = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 2 * D * sizeof(float));

        Q = ggml_reshape_3d(g, ggml_cont(g, Q), hd, nh, T);
        K = ggml_reshape_3d(g, ggml_cont(g, K), hd, nh, T);
        V = ggml_reshape_3d(g, ggml_cont(g, V), hd, nh, T);

        // Q/K RMSNorm (per-head, weight is (hd,))
        if (blk.q_norm_w) {
            Q = ggml_mul(g, ggml_rms_norm(g, Q, eps), blk.q_norm_w);
        }
        if (blk.k_norm_w) {
            K = ggml_mul(g, ggml_rms_norm(g, K, eps), blk.k_norm_w);
        }

        // Flash attention (bidirectional — no mask)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection
        attn = ggml_mul_mat(g, blk.proj_w, attn);
        if (blk.proj_b) attn = ggml_add(g, attn, blk.proj_b);
        x = ggml_add(g, x, attn);

        // Pre-norm SwiGLU FFN
        h = rmsnorm(x, blk.norm2_w);
        ggml_tensor *gate = ggml_mul_mat(g, blk.ffn_gate_w, h);
        if (blk.ffn_gate_b) gate = ggml_add(g, gate, blk.ffn_gate_b);
        gate = ggml_silu(g, gate);
        ggml_tensor *up = ggml_mul_mat(g, blk.ffn_up_w, h);
        if (blk.ffn_up_b) up = ggml_add(g, up, blk.ffn_up_b);
        ggml_tensor *ffn = ggml_mul_mat(g, blk.ffn_down_w, ggml_mul(g, gate, up));
        if (blk.ffn_down_b) ffn = ggml_add(g, ffn, blk.ffn_down_b);
        x = ggml_add(g, x, ffn);

        char name[64];
        snprintf(name, sizeof(name), "vis_layer_%d", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
        vg.layer_outputs.push_back(x);
    }

    // Post-layernorm
    if (ctx.m.post_layernorm_w) {
        x = rmsnorm(x, ctx.m.post_layernorm_w);
        ggml_set_name(x, "vis_post_norm");
        ggml_set_output(x);
    }

    ggml_build_forward_expand(gf, x);
    vg.gf = gf;
    vg.output = x;
    return vg;
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────

bool load(context &ctx, const char *gguf_path, int n_threads, int verbosity) {
    ctx.n_threads = n_threads;
    ctx.verbosity = verbosity;

    if (verbosity >= 1) printf("glm_ocr: loading %s\n", gguf_path);

    if (!load_hparams(ctx, gguf_path)) {
        fprintf(stderr, "glm_ocr: failed to load hparams\n");
        return false;
    }

    if (verbosity >= 1) {
        printf("  Vision: %uL, %ud, %uH, patch=%u, image=%u\n",
               ctx.m.vhp.depth, ctx.m.vhp.hidden_size,
               ctx.m.vhp.num_heads, ctx.m.vhp.patch_size,
               ctx.m.vhp.image_size);
        printf("  LLM: %uL, %ud, %uH/%uKV, hd=%u, inter=%u, vocab=%u\n",
               ctx.m.lhp.num_hidden_layers, ctx.m.lhp.hidden_size,
               ctx.m.lhp.num_attention_heads, ctx.m.lhp.num_key_value_heads,
               ctx.m.lhp.head_dim, ctx.m.lhp.intermediate_size,
               ctx.m.lhp.vocab_size);
    }

    ctx.backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);

    if (!load_tensors(ctx, gguf_path)) {
        fprintf(stderr, "glm_ocr: failed to load tensors\n");
        return false;
    }

    ggml_backend_t backends[] = {ctx.backend};
    ctx.sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);

    if (verbosity >= 1) printf("  Ready (CPU, %d threads)\n", n_threads);
    return true;
}

static void free_kv_cache(context &ctx);  // forward decl

void free_(context &ctx) {
    free_kv_cache(ctx);
    if (ctx.sched) { ggml_backend_sched_free(ctx.sched); ctx.sched = nullptr; }
    if (ctx.model_buf) { ggml_backend_buffer_free(ctx.model_buf); ctx.model_buf = nullptr; }
    if (ctx.model_ctx) { ggml_free(ctx.model_ctx); ctx.model_ctx = nullptr; }
    if (ctx.backend) { ggml_backend_free(ctx.backend); ctx.backend = nullptr; }
}

bool encode_vision(context &ctx, const float *pixels, vision_result &out) {
    const auto &vhp = ctx.m.vhp;
    const int D = (int)vhp.hidden_size;
    const int P = (int)vhp.patch_size;
    const int T_p = (int)vhp.temporal_patch_size;
    const int img_size = (int)vhp.image_size;
    const int n_ph = img_size / P;
    const int n_pw = n_ph;
    const int n_patches = n_ph * n_pw;
    const int patch_flat = 3 * T_p * P * P;

    // Extract patches from (3, H, W) with temporal duplication
    std::vector<float> patches(n_patches * patch_flat);
    int idx = 0;
    for (int ph = 0; ph < n_ph; ph++) {
        for (int pw = 0; pw < n_pw; pw++) {
            // Duplicate frame for temporal dim
            for (int t = 0; t < T_p; t++) {
                for (int c = 0; c < 3; c++) {
                    for (int py = 0; py < P; py++) {
                        for (int px = 0; px < P; px++) {
                            int y = ph * P + py;
                            int x = pw * P + px;
                            patches[idx * patch_flat + t * 3 * P * P + c * P * P + py * P + px] =
                                pixels[c * img_size * img_size + y * img_size + x];
                        }
                    }
                }
            }
            idx++;
        }
    }

    vision_graph vg = build_vision_graph(ctx, n_patches);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, vg.gf)) {
        fprintf(stderr, "glm_ocr: vision graph alloc failed\n");
        ggml_free(vg.gctx);
        return false;
    }

    ggml_backend_tensor_set(vg.pixel_in, patches.data(), 0,
                            n_patches * patch_flat * sizeof(float));
    ggml_backend_sched_graph_compute(ctx.sched, vg.gf);

    // Read vision encoder output: (D=1024, N=576)
    int vis_D = (int)vg.output->ne[0];
    int vis_N = (int)vg.output->ne[1];
    std::vector<float> vis_out(vis_D * vis_N);
    ggml_backend_tensor_get(vg.output, vis_out.data(), 0,
                            vis_D * vis_N * sizeof(float));

    // ── Spatial downsample: Conv2D [out_hidden, 1024, 2, 2] stride 2 ──
    // Host-side: vis_out (1024, 576) → reshape to (1024, 24, 24) → conv2d → (1536, 12, 12)
    const int merge = (int)vhp.spatial_merge_size;
    const int out_h = n_ph / merge;  // 12
    const int out_w = n_pw / merge;  // 12
    const int out_D = (int)vhp.out_hidden_size;  // 1536
    const int n_merged = out_h * out_w;  // 144

    // Read downsample weights
    std::vector<float> ds_w_buf, ds_b_buf;
    if (ctx.m.downsample_w) {
        size_t dsn = ggml_nelements(ctx.m.downsample_w);
        ds_w_buf.resize(dsn);
        ggml_backend_tensor_get(ctx.m.downsample_w, ds_w_buf.data(), 0, dsn * sizeof(float));
    }
    if (ctx.m.downsample_b) {
        ds_b_buf.resize(out_D);
        ggml_backend_tensor_get(ctx.m.downsample_b, ds_b_buf.data(), 0, out_D * sizeof(float));
    }

    // Conv2D stride 2: for each output position, gather 2x2 patch from input
    // ds_w: (out_D, D, 2, 2) stored as (D*2*2, out_D) in ggml col-major
    std::vector<float> ds_out(out_D * n_merged, 0.0f);
    int in_stride = D * 4;  // D * kH * kW flattened
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            int out_idx = oh * out_w + ow;
            // Gather 2x2 patch of D-dim vectors
            float patch[4096];  // D * 2 * 2 = 1024 * 4 = 4096
            for (int kh = 0; kh < 2; kh++) {
                for (int kw = 0; kw < 2; kw++) {
                    int in_h = oh * 2 + kh;
                    int in_w = ow * 2 + kw;
                    int in_idx = in_h * n_pw + in_w;
                    // vis_out is col-major (D, N): element [d, n] at d + n*D
                    for (int d = 0; d < D; d++) {
                        patch[d * 4 + kh * 2 + kw] = vis_out[d + in_idx * D];
                    }
                }
            }
            // Matmul: ds_w (out_D, D*4) @ patch (D*4,) → (out_D,)
            // ds_w in ggml is (D*4, out_D) col-major = (ne0=D*4, ne1=out_D)
            // So ds_w[i + j*D*4] = weight[j, i] in row-major
            for (int o = 0; o < out_D; o++) {
                float sum = 0;
                for (int k = 0; k < D * 4; k++) {
                    sum += ds_w_buf[k + o * D * 4] * patch[k];
                }
                if (!ds_b_buf.empty()) sum += ds_b_buf[o];
                ds_out[o + out_idx * out_D] = sum;  // col-major (out_D, n_merged)
            }
        }
    }

    // ── Merger: proj → SwiGLU → LayerNorm ──
    // Read merger weights
    auto read_w = [&](ggml_tensor *t) -> std::vector<float> {
        if (!t) return {};
        size_t n = ggml_nelements(t);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        return buf;
    };

    auto proj_w = read_w(ctx.m.merger.proj_w);
    auto gate_w = read_w(ctx.m.merger.gate_w);
    auto up_w   = read_w(ctx.m.merger.up_w);
    auto down_w = read_w(ctx.m.merger.down_w);
    auto norm_w = read_w(ctx.m.merger.norm_w);
    auto norm_b = read_w(ctx.m.merger.norm_b);
    int inter = (int)vhp.out_hidden_size * 3;  // 4608 — check from config
    // Actually merger inter = 4608 from the weight shapes
    if (ctx.m.merger.gate_w) inter = (int)ctx.m.merger.gate_w->ne[0];

    std::vector<float> merger_out(out_D * n_merged);
    // ggml_mul_mat(A, B) = B × A^T.
    // A has ne=(ne0, ne1). Output ne[0] = ne1 (columns of A).
    // Host: out[j] = sum_i(x[i] * A_flat[i + j * ne0])
    // where i ∈ [0, ne0) = inner/input dim, j ∈ [0, ne1) = output dim.
    auto host_matmul = [](const float *w, int ne0, int ne1,
                          const float *x, float *out) {
        for (int j = 0; j < ne1; j++) {
            float sum = 0;
            for (int i = 0; i < ne0; i++) {
                sum += w[i + j * ne0] * x[i];
            }
            out[j] = sum;
        }
    };

    // Weight shapes (ggml ne): proj (1536,1536), gate (1536,4608), up (1536,4608), down (4608,1536)
    // For host_matmul: ne0 = input dim, ne1 = output dim
    int proj_ne0 = ctx.m.merger.proj_w ? (int)ctx.m.merger.proj_w->ne[0] : out_D;
    int proj_ne1 = ctx.m.merger.proj_w ? (int)ctx.m.merger.proj_w->ne[1] : out_D;
    int gate_ne0 = ctx.m.merger.gate_w ? (int)ctx.m.merger.gate_w->ne[0] : out_D;
    int gate_ne1 = ctx.m.merger.gate_w ? (int)ctx.m.merger.gate_w->ne[1] : inter;
    int down_ne0 = ctx.m.merger.down_w ? (int)ctx.m.merger.down_w->ne[0] : inter;
    int down_ne1 = ctx.m.merger.down_w ? (int)ctx.m.merger.down_w->ne[1] : out_D;
    inter = gate_ne1;  // output dim of gate projection

    for (int t = 0; t < n_merged; t++) {
        // proj: ne=(1536,1536), input=1536, output=1536
        std::vector<float> x_proj(proj_ne1);
        host_matmul(proj_w.data(), proj_ne0, proj_ne1, &ds_out[t * out_D], x_proj.data());

        // SwiGLU: gate/up ne=(1536,4608), input=1536, output=4608
        std::vector<float> g(gate_ne1), u(gate_ne1), ffn(down_ne1);
        host_matmul(gate_w.data(), gate_ne0, gate_ne1, x_proj.data(), g.data());
        host_matmul(up_w.data(), gate_ne0, gate_ne1, x_proj.data(), u.data());
        for (int k = 0; k < inter; k++) {
            g[k] = g[k] / (1.0f + std::exp(-g[k]));  // silu
        }
        // down: ne=(4608,1536), input=4608, output=1536
        std::vector<float> gu(gate_ne1);
        for (int k = 0; k < gate_ne1; k++) gu[k] = g[k] * u[k];
        host_matmul(down_w.data(), down_ne0, down_ne1, gu.data(), ffn.data());
        // LayerNorm (not RMSNorm — has bias)
        float mean = 0;
        for (int d = 0; d < out_D; d++) mean += ffn[d];
        mean /= out_D;
        float var = 0;
        for (int d = 0; d < out_D; d++) var += (ffn[d] - mean) * (ffn[d] - mean);
        var /= out_D;
        for (int d = 0; d < out_D; d++) {
            float normed = (ffn[d] - mean) / std::sqrt(var + 1e-6f);
            merger_out[d + t * out_D] = normed * norm_w[d] + (norm_b.empty() ? 0 : norm_b[d]);
        }
    }

    out.n_tokens = n_merged;
    out.hidden_dim = out_D;
    out.hidden = (float *)malloc(out_D * n_merged * sizeof(float));
    std::memcpy(out.hidden, merger_out.data(), out_D * n_merged * sizeof(float));

    // Diff comparison
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            // Patch embed
            {
                ggml_tensor *pe = ggml_graph_get_tensor(vg.gf, "vis_patch_embed");
                if (pe) {
                    size_t n = ggml_nelements(pe);
                    float *buf = (float *)malloc(n * sizeof(float));
                    ggml_backend_tensor_get(pe, buf, 0, n * sizeof(float));
                    auto r = ref.compare("vis_patch_embed", buf, n);
                    printf("  vis_patch_embed: cos=%.6f max_abs=%.6f %s\n",
                           r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    free(buf);
                }
            }
            // Post-norm
            {
                ggml_tensor *pn = ggml_graph_get_tensor(vg.gf, "vis_post_norm");
                if (pn && ref.has("vis_post_norm")) {
                    size_t n = ggml_nelements(pn);
                    float *buf = (float *)malloc(n * sizeof(float));
                    ggml_backend_tensor_get(pn, buf, 0, n * sizeof(float));
                    auto r = ref.compare("vis_post_norm", buf, n);
                    printf("  vis_post_norm: cos=%.6f max_abs=%.6f %s\n",
                           r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    free(buf);
                }
            }
            // Downsample
            if (ref.has("vis_downsample")) {
                auto r = ref.compare("vis_downsample", ds_out.data(), ds_out.size());
                printf("  vis_downsample: cos=%.6f max_abs=%.6f %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            // Merger
            if (ref.has("vis_merger_output")) {
                auto r = ref.compare("vis_merger_output", merger_out.data(), merger_out.size());
                printf("  vis_merger_output: cos=%.6f max_abs=%.6f %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            // Layers
            for (size_t i = 0; i < vg.layer_outputs.size(); i++) {
                char name[64];
                snprintf(name, sizeof(name), "vis_layer_%zu", i);
                size_t n = ggml_nelements(vg.layer_outputs[i]);
                float *buf = (float *)malloc(n * sizeof(float));
                ggml_backend_tensor_get(vg.layer_outputs[i], buf, 0, n * sizeof(float));
                auto r = ref.compare(name, buf, n);
                printf("  %s: cos=%.6f max_abs=%.6f %s\n",
                       name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                free(buf);
            }
        }
    }

    ggml_free(vg.gctx);
    return true;
}

// ── Tokenizer decode ─────────────────────────────────────────────────

std::string tokenizer::decode(const int32_t *ids, int n) const {
    // GLM-OCR uses GPT-2 BPE via chatglm-bpe tokenizer.
    // For now, simple concatenation with byte-decode via core_bpe.
    std::string result;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= vocab_size) continue;
        if (id == eos_id) break;
        const std::string &piece = id_to_piece[id];
        if (piece.empty()) continue;
        if (piece[0] == '<' && piece.back() == '>' && piece.find("0x") == std::string::npos) continue;
        if (piece[0] == '[' && piece.back() == ']') continue;
        result += piece;
    }
    return result;
}

// ── KV cache ────────────────────────────────────────────────────────

static bool alloc_kv_cache(context &ctx, int max_seq) {
    const auto &lhp = ctx.m.lhp;
    const int hd = (int)lhp.head_dim;
    const int nkv = (int)lhp.num_key_value_heads;
    const int n_layers = (int)lhp.num_hidden_layers;

    size_t ctx_size = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params ip{ctx_size, nullptr, true};
    ctx.kvc.ctx = ggml_init(ip);
    ctx.kvc.k = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16, hd, max_seq, nkv, n_layers);
    ctx.kvc.v = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16, hd, max_seq, nkv, n_layers);
    ctx.kvc.buf = ggml_backend_alloc_ctx_tensors(ctx.kvc.ctx, ctx.backend);
    if (!ctx.kvc.buf) {
        fprintf(stderr, "glm_ocr: KV cache alloc failed\n");
        ggml_free(ctx.kvc.ctx); ctx.kvc.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(ctx.kvc.buf, 0);
    ctx.kvc.max_seq = max_seq;
    ctx.kvc.n_past = 0;
    ctx.kvc.allocated = true;
    if (ctx.verbosity >= 1) {
        printf("  KV cache: %d layers, max_seq=%d, %.1f MB\n",
               n_layers, max_seq,
               (float)ggml_backend_buffer_get_size(ctx.kvc.buf) / 1024 / 1024);
    }
    return true;
}

static void free_kv_cache(context &ctx) {
    if (ctx.kvc.buf) { ggml_backend_buffer_free(ctx.kvc.buf); ctx.kvc.buf = nullptr; }
    if (ctx.kvc.ctx) { ggml_free(ctx.kvc.ctx); ctx.kvc.ctx = nullptr; }
    ctx.kvc.allocated = false;
    ctx.kvc.n_past = 0;
}

// ── LLM graph builder (uncached + cached modes) ─────────────────────

struct llm_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *token_in = nullptr;
    ggml_tensor *output = nullptr;
    ggml_tensor *logits_out = nullptr;
    std::vector<ggml_tensor *> layer_outputs;
};

static llm_graph build_llm_graph(context &ctx, int n_tokens, int n_past,
                                  bool use_kv_cache) {
    llm_graph lg;
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int nh = (int)lhp.num_attention_heads;
    const int nkv = (int)lhp.num_key_value_heads;
    const int hd = (int)lhp.head_dim;
    const int q_dim = nh * hd;
    const int kv_dim = nkv * hd;
    const int V_sz = (int)lhp.vocab_size;
    const int T = n_tokens;
    const int n_layers = (int)lhp.num_hidden_layers;
    const float rms_eps = lhp.rms_norm_eps;
    const int kv_repeat = nh / nkv;
    const int Lk = use_kv_cache ? (n_past + T) : T;

    int sections[4] = { lhp.rope_sections[0], lhp.rope_sections[1], lhp.rope_sections[2], 0 };

    int tpl = use_kv_cache ? 80 : 60;
    size_t meta_size = (size_t)(n_layers * tpl + 300) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(32768, false);
    ctx.compute_meta.resize(meta_size);
    ggml_init_params ip{meta_size, ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 32768, false);

    ggml_tensor *tok_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(tok_in, "token_ids"); ggml_set_input(tok_in);
    lg.token_in = tok_in;

    ggml_tensor *x = ggml_get_rows(g, ctx.m.embed_tokens, tok_in);
    ggml_set_name(x, "llm_embed"); ggml_set_output(x);

    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T * 4);
    ggml_set_name(pos_ids, "pos_ids"); ggml_set_input(pos_ids);

    ggml_tensor *mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, rms_eps), w);
    };

    for (int i = 0; i < n_layers; i++) {
        auto &ly = ctx.m.llm_layers[i];

        // ── Attention ──
        ggml_tensor *h = rmsnorm(x, ly.input_layernorm_w);
        ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, h);
        ggml_tensor *K_new = ggml_mul_mat(g, ly.k_w, h);
        ggml_tensor *V_new = ggml_mul_mat(g, ly.v_w, h);

        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K_new = ggml_reshape_3d(g, K_new, hd, nkv, T);
        V_new = ggml_reshape_3d(g, V_new, hd, nkv, T);

        Q = ggml_rope_multi(g, Q, pos_ids, nullptr, hd, sections,
                            GGML_ROPE_TYPE_MROPE, 0, lhp.rope_theta,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_multi(g, K_new, pos_ids, nullptr, hd, sections,
                                GGML_ROPE_TYPE_MROPE, 0, lhp.rope_theta,
                                1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor *Kfull, *Vfull;

        if (use_kv_cache && ctx.kvc.allocated) {
            ggml_tensor *K_perm = ggml_permute(g, K_new, 0, 2, 1, 3);
            ggml_tensor *V_perm = ggml_permute(g, V_new, 0, 2, 1, 3);

            ggml_tensor *k_view = ggml_view_4d(g, ctx.kvc.k,
                hd, T, nkv, 1,
                ctx.kvc.k->nb[1], ctx.kvc.k->nb[2], ctx.kvc.k->nb[3],
                (size_t)i * ctx.kvc.k->nb[3] + (size_t)n_past * ctx.kvc.k->nb[1]);
            ggml_tensor *v_view = ggml_view_4d(g, ctx.kvc.v,
                hd, T, nkv, 1,
                ctx.kvc.v->nb[1], ctx.kvc.v->nb[2], ctx.kvc.v->nb[3],
                (size_t)i * ctx.kvc.v->nb[3] + (size_t)n_past * ctx.kvc.v->nb[1]);

            ggml_build_forward_expand(gf, ggml_cpy(g, K_perm, k_view));
            ggml_build_forward_expand(gf, ggml_cpy(g, V_perm, v_view));

            ggml_tensor *k_layer = ggml_view_3d(g, ctx.kvc.k,
                hd, Lk, nkv, ctx.kvc.k->nb[1], ctx.kvc.k->nb[2],
                (size_t)i * ctx.kvc.k->nb[3]);
            ggml_tensor *v_layer = ggml_view_3d(g, ctx.kvc.v,
                hd, Lk, nkv, ctx.kvc.v->nb[1], ctx.kvc.v->nb[2],
                (size_t)i * ctx.kvc.v->nb[3]);

            Kfull = ggml_cont(g, k_layer);
            Vfull = ggml_cont(g, v_layer);

            if (kv_repeat > 1) {
                Kfull = ggml_reshape_4d(g, Kfull, hd, Lk, 1, nkv);
                Kfull = ggml_repeat(g, Kfull, ggml_new_tensor_4d(g, Kfull->type, hd, Lk, kv_repeat, nkv));
                Kfull = ggml_reshape_3d(g, Kfull, hd, Lk, nh);
                Vfull = ggml_reshape_4d(g, Vfull, hd, Lk, 1, nkv);
                Vfull = ggml_repeat(g, Vfull, ggml_new_tensor_4d(g, Vfull->type, hd, Lk, kv_repeat, nkv));
                Vfull = ggml_reshape_3d(g, Vfull, hd, Lk, nh);
            }
        } else {
            if (kv_repeat > 1) {
                K_new = ggml_reshape_4d(g, K_new, hd, 1, nkv, T);
                K_new = ggml_repeat(g, K_new, ggml_new_tensor_4d(g, K_new->type, hd, kv_repeat, nkv, T));
                K_new = ggml_reshape_3d(g, K_new, hd, nh, T);
                V_new = ggml_reshape_4d(g, V_new, hd, 1, nkv, T);
                V_new = ggml_repeat(g, V_new, ggml_new_tensor_4d(g, V_new->type, hd, kv_repeat, nkv, T));
                V_new = ggml_reshape_3d(g, V_new, hd, nh, T);
            }
            Kfull = ggml_cont(g, ggml_permute(g, K_new, 0, 2, 1, 3));
            Vfull = ggml_cont(g, ggml_permute(g, V_new, 0, 2, 1, 3));
        }

        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, mask, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, q_dim, T);
        attn = ggml_mul_mat(g, ly.o_w, attn);

        // Post-norm: post_self_attn_layernorm → + residual
        attn = rmsnorm(attn, ly.post_self_attn_layernorm_w);
        x = ggml_add(g, x, attn);

        // ── FFN ──
        h = rmsnorm(x, ly.post_attention_layernorm_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);
        ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));

        // Post-norm: post_mlp_layernorm → + residual
        ffn = rmsnorm(ffn, ly.post_mlp_layernorm_w);
        x = ggml_add(g, x, ffn);

        char name[64];
        snprintf(name, sizeof(name), "llm_layer_%d", i);
        ggml_set_name(x, name); ggml_set_output(x);
        lg.layer_outputs.push_back(x);
    }

    x = rmsnorm(x, ctx.m.output_norm_w);

    if (ctx.m.lm_head_w) {
        ggml_tensor *logits = ggml_mul_mat(g, ctx.m.lm_head_w, x);
        ggml_set_name(logits, "logits"); ggml_set_output(logits);
        lg.logits_out = logits;
        ggml_build_forward_expand(gf, logits);
    } else {
        ggml_build_forward_expand(gf, x);
    }

    lg.gf = gf; lg.gctx = g; lg.output = x;
    return lg;
}

// ── run_llm_forward (uncached, for parity testing) ──────────────────

// ── Cached step helper ──────────────────────────────────────────────

static bool run_cached_step(context &ctx, const int32_t *token_ids, int n_tokens,
                            int n_past, std::vector<float> &last_logits_out) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int V = (int)lhp.vocab_size;
    const int T = n_tokens;
    const int Lk = n_past + T;

    llm_graph lg = build_llm_graph(ctx, T, n_past, true);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        fprintf(stderr, "glm_ocr: cached step alloc failed\n");
        ggml_free(lg.gctx);
        return false;
    }

    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // mRoPE positions
    std::vector<int32_t> pos_data(T * 4, 0);
    for (int j = 0; j < T; j++) {
        pos_data[j]       = n_past + j;
        pos_data[T + j]   = n_past + j;
        pos_data[2*T + j] = n_past + j;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(lg.gf, "pos_ids"),
                            pos_data.data(), 0, T * 4 * sizeof(int32_t));

    // Causal mask: (Lk, T)
    std::vector<ggml_fp16_t> mask_data((size_t)Lk * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < Lk; ki++)
            mask_data[(size_t)qi * Lk + ki] =
                ggml_fp32_to_fp16((ki > n_past + qi) ? -INFINITY : 0.0f);
    ggml_backend_tensor_set(ggml_graph_get_tensor(lg.gf, "causal_mask"),
                            mask_data.data(), 0, (size_t)Lk * T * sizeof(ggml_fp16_t));

    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    if (lg.logits_out) {
        last_logits_out.resize(V);
        ggml_backend_tensor_get(lg.logits_out, last_logits_out.data(),
                                (size_t)(T - 1) * V * sizeof(float), V * sizeof(float));
    }

    ggml_free(lg.gctx);
    return true;
}

// ── generate ────────────────────────────────────────────────────────

bool generate(context &ctx, const int32_t *prompt_ids, int n_prompt,
              int max_new_tokens, generate_result &out) {
    const auto &lhp = ctx.m.lhp;
    const int V = (int)lhp.vocab_size;
    const int eos_id = (int)lhp.eos_token_id;
    const int max_seq = n_prompt + max_new_tokens + 16;

    if (!ctx.kvc.allocated || ctx.kvc.max_seq < max_seq) {
        free_kv_cache(ctx);
        if (!alloc_kv_cache(ctx, max_seq)) return false;
    }
    ctx.kvc.n_past = 0;

    // Prefill
    std::vector<float> logits;
    if (!run_cached_step(ctx, prompt_ids, n_prompt, 0, logits)) return false;
    ctx.kvc.n_past = n_prompt;

    int best_id = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++)
        if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }
    out.token_ids.push_back(best_id);
    if (ctx.verbosity >= 1)
        fprintf(stderr, "  gen[0]: token=%d score=%.2f (prefill %d)\n", best_id, best_score, n_prompt);
    if (best_id == eos_id) { out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size()); return true; }

    // Decode
    for (int gen = 1; gen < max_new_tokens; gen++) {
        int32_t next = best_id;
        if (!run_cached_step(ctx, &next, 1, ctx.kvc.n_past, logits)) return false;
        ctx.kvc.n_past += 1;

        best_id = 0; best_score = -INFINITY;
        for (int v = 0; v < V; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }
        out.token_ids.push_back(best_id);
        if (ctx.verbosity >= 1)
            fprintf(stderr, "  gen[%d]: token=%d score=%.2f\n", gen, best_id, best_score);
        if (best_id == eos_id) break;
    }

    out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size());
    return true;
}

// ── run_llm_forward (uncached, for parity) ──────────────────────────

bool run_llm_forward(context &ctx, const int32_t *token_ids, int n_tokens,
                     llm_result &out) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int V_sz = (int)lhp.vocab_size;
    const int T = n_tokens;

    llm_graph lg = build_llm_graph(ctx, T, 0, false);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        fprintf(stderr, "glm_ocr: LLM graph alloc failed\n");
        ggml_free(lg.gctx);
        return false;
    }

    // Set inputs
    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // mRoPE positions (text-only: all dims = sequential)
    std::vector<int32_t> pos_data(T * 4, 0);
    for (int j = 0; j < T; j++) {
        pos_data[j] = j; pos_data[T+j] = j; pos_data[2*T+j] = j;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(lg.gf, "pos_ids"),
                            pos_data.data(), 0, T * 4 * sizeof(int32_t));

    // Causal mask (T, T)
    std::vector<ggml_fp16_t> mask_data(T * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < T; ki++)
            mask_data[qi * T + ki] = ggml_fp32_to_fp16((ki > qi) ? -INFINITY : 0.0f);
    ggml_backend_tensor_set(ggml_graph_get_tensor(lg.gf, "causal_mask"),
                            mask_data.data(), 0, T * T * sizeof(ggml_fp16_t));

    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    // Read output
    out.n_tokens = T;
    out.hidden_dim = D;
    out.hidden = (float *)malloc(T * D * sizeof(float));
    ggml_backend_tensor_get(lg.output, out.hidden, 0, T * D * sizeof(float));
    if (lg.logits_out) {
        out.vocab_size = V_sz;
        out.logits = (float *)malloc(T * V_sz * sizeof(float));
        ggml_backend_tensor_get(lg.logits_out, out.logits, 0, T * V_sz * sizeof(float));
    }

    // Diff comparison
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            {
                ggml_tensor *emb = ggml_graph_get_tensor(lg.gf, "llm_embed");
                if (emb && ref.has("llm_embed")) {
                    float *buf = (float *)malloc(T * D * sizeof(float));
                    ggml_backend_tensor_get(emb, buf, 0, T * D * sizeof(float));
                    auto r = ref.compare("llm_embed", buf, T * D);
                    printf("  llm_embed: cos=%.6f max_abs=%.6f %s\n",
                           r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    free(buf);
                }
            }
            for (size_t li = 0; li < lg.layer_outputs.size(); li++) {
                char name[64];
                snprintf(name, sizeof(name), "llm_layer_%zu", li);
                if (ref.has(name)) {
                    float *buf = (float *)malloc(T * D * sizeof(float));
                    ggml_backend_tensor_get(lg.layer_outputs[li], buf, 0, T * D * sizeof(float));
                    auto r = ref.compare(name, buf, T * D);
                    printf("  %s: cos=%.6f max_abs=%.6f %s\n",
                           name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    free(buf);
                }
            }
        }
    }

    ggml_free(lg.gctx);
    return true;
}

}  // namespace glm_ocr

// ── C ABI ───────────────────────────────────────────────────────────

struct glm_ocr_context {
    glm_ocr::context ctx;
    std::string last_text;
};

glm_ocr_context * glm_ocr_init(const char *model_path, int n_threads) {
    auto *c = new glm_ocr_context();
    if (!glm_ocr::load(c->ctx, model_path, n_threads, 1)) {
        delete c;
        return nullptr;
    }
    return c;
}

void glm_ocr_free(glm_ocr_context *ctx) {
    if (ctx) { glm_ocr::free_(ctx->ctx); delete ctx; }
}

const char * glm_ocr_recognize_raw(glm_ocr_context *ctx,
    const uint8_t *px, int w, int h, int ch, int *out_len) {
    if (out_len) *out_len = 0;
    return "";
}

const char * glm_ocr_recognize(glm_ocr_context *ctx,
    const float *px, int w, int h, int *out_len) {
    if (out_len) *out_len = 0;
    return "";
}
