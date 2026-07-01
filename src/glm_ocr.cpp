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
#include <chrono>
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

// ── 2D vision RoPE (host-side cos/sin) ──────────────────────────────
//
// GLM-4V's ViT applies 2D rotary position embedding to Q/K in every vision
// layer (transformers Glm4vVisionAttention.apply_rotary_pos_emb_vision). This
// is the identical scheme used by Qwen2-VL vision (see qwen2vl_ocr.cpp
// compute_vision_rope): a Glm4vVisionRotaryEmbedding of dim=head_dim/2 with
// theta=10000, producing per-patch h/w frequencies concatenated to head_dim/2
// and duplicated (emb = cat(rot, rot)) to head_dim. The rotate is NEOX
// (split-half) style: rotate_half(x) = cat(-x[d/2:], x[:d/2]).
//
// Layout of each per-patch row (length head_dim, quart = head_dim/4):
//   [ h_freqs(quart), w_freqs(quart), h_freqs(quart), w_freqs(quart) ]
// so the first/second halves are identical (cat(rot,rot)), which is what the
// split-half rotate expects.
//
// Patch order: CrispEmbed's encode_vision extracts patches in raster order
// (token = row*n_pw + col), and the downsample/merger consumes that same
// raster order, so the rope positions are assigned in raster order too. HF's
// spatial-merge-windowed pos ordering is only needed when the patch sequence
// itself is windowed; set merge_order=true (GLM_OCR_ROPE_MERGE_ORDER=1) to
// match that ordering if ever needed.
static void compute_glm_vision_rope(std::vector<float> &cos_buf,
                                    std::vector<float> &sin_buf,
                                    int n_ph, int n_pw, int head_dim,
                                    int spatial_merge, bool merge_order,
                                    float theta) {
    const int n_patches = n_ph * n_pw;
    cos_buf.resize((size_t)n_patches * head_dim);
    sin_buf.resize((size_t)n_patches * head_dim);

    const int quart = head_dim / 4;
    const float rot_dim = (float)(head_dim / 2);
    std::vector<float> inv_freq(quart);
    for (int j = 0; j < quart; j++)
        inv_freq[j] = 1.0f / std::pow(theta, (float)(2 * j) / rot_dim);

    int tok = 0;
    auto fill_one = [&](int row, int col) {
        float *cr = cos_buf.data() + (size_t)tok * head_dim;
        float *sr = sin_buf.data() + (size_t)tok * head_dim;
        for (int j = 0; j < quart; j++) {
            float vr = (float)row * inv_freq[j];
            float vc = (float)col * inv_freq[j];
            cr[j]             = std::cos(vr); sr[j]             = std::sin(vr);
            cr[j + quart]     = std::cos(vc); sr[j + quart]     = std::sin(vc);
            cr[j + 2 * quart] = std::cos(vr); sr[j + 2 * quart] = std::sin(vr);
            cr[j + 3 * quart] = std::cos(vc); sr[j + 3 * quart] = std::sin(vc);
        }
        tok++;
    };

    if (merge_order && spatial_merge > 1) {
        for (int mh = 0; mh < n_ph / spatial_merge; mh++)
            for (int mw = 0; mw < n_pw / spatial_merge; mw++)
                for (int ir = 0; ir < spatial_merge; ir++)
                    for (int ic = 0; ic < spatial_merge; ic++)
                        fill_one(mh * spatial_merge + ir, mw * spatial_merge + ic);
    } else {
        for (int row = 0; row < n_ph; row++)
            for (int col = 0; col < n_pw; col++)
                fill_one(row, col);
    }
}

// ── Vision encoder graph ────────────────────────────────────────────

// Build the monolithic vision encoder graph (24 CogViT layers + post-norm).
// The graph computes correctly internally — ggml_gallocr reuses input tensor
// memory after first consumer, but the computation reads inputs before freeing.
// Only the FINAL output tensor (vis_output) is reliable for readback.
struct vision_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *embed_in = nullptr;   // pre-computed patch embeddings input
    ggml_tensor *cos_in = nullptr;     // 2D RoPE cos (head_dim, 1, n_patches)
    ggml_tensor *sin_in = nullptr;     // 2D RoPE sin (head_dim, 1, n_patches)
    ggml_tensor *output = nullptr;     // final post-norm output
    ggml_tensor *dbg_q_rope0 = nullptr;
    ggml_tensor *dbg_k_rope0 = nullptr;
    std::vector<ggml_tensor *> layer_outputs;  // per-layer (diff only)
};

static vision_graph build_vision_graph(context &ctx, int n_patches) {
    vision_graph vg;
    const auto &vhp = ctx.m.vhp;
    const int D = (int)vhp.hidden_size;
    const int nh = (int)vhp.num_heads;
    const int hd = D / nh;
    const int n_layers = (int)vhp.depth;
    const float eps = vhp.rms_norm_eps;
    const int T = n_patches;

    size_t meta_size = (size_t)(n_layers * 64 + 200) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(16384, false);
    ctx.compute_meta.resize(meta_size);
    ggml_init_params ip{meta_size, ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    vg.gctx = g;

    const bool diff_mode = !ctx.diff_ref_path.empty();
    // GLM-OCR's glm_ocr_vision tower applies 2D RoPE to Q/K in every layer
    // (authoritative: transformers modeling_glm_ocr.py). ON by default;
    // GLM_OCR_VISION_ROPE=0 disables for A/B testing. NOTE: the uploaded
    // reference dumps were generated by the STALE no-rope dump script, so they
    // do NOT validate the rope path — regenerate refs with rope before using
    // per-layer diff as a gate.
    const char *rope_env = std::getenv("GLM_OCR_VISION_ROPE");
    const bool use_vision_rope = !(rope_env && atoi(rope_env) == 0);
    // Explicit F32 attention by default (flash-attn's F16 K/V loses precision on
    // this ViT's massive activations); GLM_OCR_VISION_FLASH=1 to A/B the flash path.
    const bool use_flash_attn = (std::getenv("GLM_OCR_VISION_FLASH") != nullptr);

    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Input: pre-computed patch embeddings (D, n_patches)
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "vis_embed_in");
    ggml_set_input(x);
    vg.embed_in = x;

    // 2D RoPE cos/sin inputs: (head_dim, 1, n_patches) — broadcast over heads.
    // Only materialized when the (opt-in, off-by-default) rope path is active,
    // so they never become unallocated dangling graph inputs.
    ggml_tensor *cos_in = nullptr, *sin_in = nullptr;
    if (use_vision_rope) {
        cos_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, 1, T);
        sin_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, 1, T);
        ggml_set_name(cos_in, "vis_cos_in"); ggml_set_input(cos_in);
        ggml_set_name(sin_in, "vis_sin_in"); ggml_set_input(sin_in);
        vg.cos_in = cos_in;
        vg.sin_in = sin_in;
    }

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    // ggml_mul_mat with an F16 weight converts the F32 activation to F16 for the
    // vec_dot. This ViT's residual stream has "massive activation" outliers
    // (max_abs ~1900 in late layers); the F16 round-trip destroys them and the
    // cosine collapses. Cast quantized/F16 weights to F32 so matmuls stay F32.
    // GLM_OCR_VISION_F16MM=1 restores the old (lossy) behavior for A/B testing.
    const bool f32_matmul = !(std::getenv("GLM_OCR_VISION_F16MM"));
    auto mm = [&](ggml_tensor *w, ggml_tensor *x) -> ggml_tensor * {
        if (f32_matmul && w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
        return ggml_mul_mat(g, w, x);
    };

    // NEOX split-half rotate: result = t*cos + rotate_half(t)*sin,
    // rotate_half(t) = [-t[hd/2:], t[:hd/2]].  t: (hd, nh, T) contiguous.
    auto apply_rope = [&](ggml_tensor *t) -> ggml_tensor * {
        const int half = hd / 2;
        ggml_tensor *h1 = ggml_view_3d(g, t, half, nh, T, t->nb[1], t->nb[2], 0);
        ggml_tensor *h2 = ggml_view_3d(g, t, half, nh, T, t->nb[1], t->nb[2],
                                       (size_t)half * t->nb[0]);
        ggml_tensor *h2_neg = ggml_scale(g, ggml_cont(g, h2), -1.0f);
        ggml_tensor *rot = ggml_concat(g, h2_neg, ggml_cont(g, h1), 0);
        return ggml_add(g, ggml_mul(g, t, cos_in), ggml_mul(g, rot, sin_in));
    };

    for (int i = 0; i < n_layers; i++) {
        auto &blk = ctx.m.vis_blocks[i];

        ggml_tensor *h = rmsnorm(x, blk.norm1_w);
        ggml_tensor *qkv = mm(blk.qkv_w, h);
        if (blk.qkv_b) qkv = ggml_add(g, qkv, blk.qkv_b);

        ggml_tensor *Q = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor *K = ggml_view_2d(g, qkv, D, T, qkv->nb[1], D * sizeof(float));
        ggml_tensor *V = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 2 * D * sizeof(float));
        Q = ggml_reshape_3d(g, ggml_cont(g, Q), hd, nh, T);
        K = ggml_reshape_3d(g, ggml_cont(g, K), hd, nh, T);
        V = ggml_reshape_3d(g, ggml_cont(g, V), hd, nh, T);
        if (blk.q_norm_w) Q = ggml_mul(g, ggml_rms_norm(g, Q, eps), blk.q_norm_w);
        if (blk.k_norm_w) K = ggml_mul(g, ggml_rms_norm(g, K, eps), blk.k_norm_w);

        // 2D vision RoPE (after QK-norm, before the attention permute), matching
        // transformers glm_ocr apply_rotary_pos_emb_vision. ON by default;
        // GLM_OCR_VISION_ROPE=0 disables for A/B testing.
        if (use_vision_rope) {
            Q = apply_rope(ggml_cont(g, Q));
            K = apply_rope(ggml_cont(g, K));
        }

        if (diff_mode && i == 0) {
            ggml_set_name(Q, "dbg_q_rope0");
            ggml_set_output(Q);
            vg.dbg_q_rope0 = Q;
            ggml_set_name(K, "dbg_k_rope0");
            ggml_set_output(K);
            vg.dbg_k_rope0 = K;
        }

        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));  // (hd, T, nh)
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor *attn;
        if (use_flash_attn) {
            attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(g, attn, D, T);
        } else {
            // Explicit F32 attention. This ViT has large "massive activation"
            // outliers (V/residual max_abs grows to ~1900 by the last layer);
            // flash-attn converts K/V to F16 and the resulting precision loss
            // makes the late-layer cosine collapse (verified: f32 numpy stays
            // 0.99999 to layer 23, C++ flash-attn drops to <0 at layer 15).
            // Keep everything F32 via explicit matmul + softmax.
            ggml_tensor *scores = ggml_mul_mat(g, K, Q);  // (T_k, T_q, nh)
            scores = ggml_soft_max_ext(g, scores, nullptr, scale, 0.0f);
            ggml_tensor *Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));  // (T, hd, nh)
            attn = ggml_mul_mat(g, Vt, scores);              // (hd, T_q, nh)
            attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));  // (hd, nh, T)
            attn = ggml_reshape_2d(g, attn, D, T);
        }
        attn = mm(blk.proj_w, attn);
        if (blk.proj_b) attn = ggml_add(g, attn, blk.proj_b);
        x = ggml_add(g, x, attn);

        h = rmsnorm(x, blk.norm2_w);
        ggml_tensor *gate = mm(blk.ffn_gate_w, h);
        if (blk.ffn_gate_b) gate = ggml_add(g, gate, blk.ffn_gate_b);
        gate = ggml_silu(g, gate);
        ggml_tensor *up = mm(blk.ffn_up_w, h);
        if (blk.ffn_up_b) up = ggml_add(g, up, blk.ffn_up_b);
        ggml_tensor *ffn = mm(blk.ffn_down_w, ggml_mul(g, gate, up));
        if (blk.ffn_down_b) ffn = ggml_add(g, ffn, blk.ffn_down_b);
        x = ggml_add(g, x, ffn);

        // Per-layer readback for diff bisection (diff mode only — set_output
        // keeps the tensor live, which costs memory in the monolithic graph).
        if (diff_mode) {
            char nm[32];
            snprintf(nm, sizeof(nm), "vis_layer_%d", i);
            ggml_set_name(x, nm);
            ggml_set_output(x);
            vg.layer_outputs.push_back(x);
        }
    }

    // Post-layernorm
    if (ctx.m.post_layernorm_w) {
        x = rmsnorm(x, ctx.m.post_layernorm_w);
    }

    ggml_set_name(x, "vis_output");
    ggml_set_output(x);
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

    // Prefer GPU backend when available
    bool force_cpu = (getenv("GLM_OCR_FORCE_CPU") && atoi(getenv("GLM_OCR_FORCE_CPU")));
    ctx.backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx.backend) ctx.backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx.backend))
        ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);

    if (!load_tensors(ctx, gguf_path)) {
        fprintf(stderr, "glm_ocr: failed to load tensors\n");
        return false;
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(), 16384, false, false);

    ctx.bench = (std::getenv("CRISPEMBED_GLM_OCR_BENCH") != nullptr);

    if (verbosity >= 1) {
        const char *bname = ggml_backend_is_cpu(ctx.backend) ? "CPU" : "GPU";
        printf("  Ready (%s, %d threads)\n", bname, n_threads);
    }
    return true;
}

static void free_kv_cache(context &ctx);  // forward decl

void free_(context &ctx) {
    free_kv_cache(ctx);
    if (ctx.sched) { ggml_backend_sched_free(ctx.sched); ctx.sched = nullptr; }
    if (ctx.model_buf) { ggml_backend_buffer_free(ctx.model_buf); ctx.model_buf = nullptr; }
    if (ctx.model_ctx) { ggml_free(ctx.model_ctx); ctx.model_ctx = nullptr; }
    if (ctx.backend_cpu) { ggml_backend_free(ctx.backend_cpu); ctx.backend_cpu = nullptr; }
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

    const bool bench = ctx.bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t_vis = std::chrono::steady_clock::now();

    // Phase 1: Compute patch embedding in a separate small graph.
    // The main vision graph (24 transformer layers) is too large for the
    // scheduler's allocator — it reuses the pixel_in buffer for compute
    // tensors after the first op consumes it, corrupting the input.
    const int vis_D = (int)vhp.hidden_size;
    std::vector<float> patch_embed_data(vis_D * n_patches);
    {
        size_t pe_meta = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        ggml_init_params pe_ip{pe_meta, nullptr, true};
        ggml_context *pe_ctx = ggml_init(pe_ip);

        ggml_tensor *px_in = ggml_new_tensor_2d(pe_ctx, GGML_TYPE_F32, patch_flat, n_patches);
        ggml_set_name(px_in, "px_in"); ggml_set_input(px_in);

        ggml_tensor *pe_out = ggml_mul_mat(pe_ctx, ctx.m.patch_embed_w, px_in);
        if (ctx.m.patch_embed_b) pe_out = ggml_add(pe_ctx, pe_out, ctx.m.patch_embed_b);
        ggml_set_name(pe_out, "pe_out"); ggml_set_output(pe_out);

        ggml_cgraph *pe_gf = ggml_new_graph(pe_ctx);
        ggml_build_forward_expand(pe_gf, pe_out);

        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, pe_gf)) {
            fprintf(stderr, "glm_ocr: patch_embed graph alloc failed\n");
            ggml_free(pe_ctx);
            return false;
        }
        ggml_backend_tensor_set(px_in, patches.data(), 0, n_patches * patch_flat * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, pe_gf);
        ggml_backend_tensor_get(pe_out, patch_embed_data.data(), 0,
                                vis_D * n_patches * sizeof(float));
        ggml_free(pe_ctx);
    }

    // Phase 2: Run all 24 CogViT layers + post-norm in one monolithic graph.
    // The graph computes correctly internally (ggml_gallocr reuses input
    // tensor memory after first consumer, but ops read before the reuse).
    // Only the final output is read back.
    vision_graph vg = build_vision_graph(ctx, n_patches);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, vg.gf)) {
        fprintf(stderr, "glm_ocr: vision graph alloc failed\n");
        ggml_free(vg.gctx);
        return false;
    }

    ggml_backend_tensor_set(vg.embed_in, patch_embed_data.data(), 0,
                            vis_D * n_patches * sizeof(float));

    // 2D vision RoPE (opt-in, off by default — glm_ocr_vision is a plain ViT).
    if (vg.cos_in && vg.sin_in) {
        const int hd_v = D / (int)vhp.num_heads;
        const bool rope_merge_order =
            (std::getenv("GLM_OCR_ROPE_MERGE_ORDER") != nullptr);
        std::vector<float> rope_cos, rope_sin;
        compute_glm_vision_rope(rope_cos, rope_sin, n_ph, n_pw, hd_v,
                                (int)vhp.spatial_merge_size, rope_merge_order,
                                10000.0f);
        if (std::getenv("GLM_OCR_ROPE_DEBUG")) {
            fprintf(stderr, "[rope] hd_v=%d n_ph=%d n_pw=%d size=%zu\n",
                    hd_v, n_ph, n_pw, rope_cos.size());
            fprintf(stderr, "[rope] tok0: cos[0,1,16,17]=%.4f %.4f %.4f %.4f\n",
                    rope_cos[0], rope_cos[1], rope_cos[16], rope_cos[17]);
            fprintf(stderr, "[rope] tok100(row=%d,col=%d): cos[0,1,16,17]=%.4f %.4f %.4f %.4f sin=%.4f %.4f\n",
                    100/n_pw, 100%n_pw,
                    rope_cos[100*hd_v+0], rope_cos[100*hd_v+1],
                    rope_cos[100*hd_v+16], rope_cos[100*hd_v+17],
                    rope_sin[100*hd_v+0], rope_sin[100*hd_v+16]);
        }
        ggml_backend_tensor_set(vg.cos_in, rope_cos.data(), 0,
                                rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(vg.sin_in, rope_sin.data(), 0,
                                rope_sin.size() * sizeof(float));
    }
    ggml_backend_sched_graph_compute(ctx.sched, vg.gf);

    if (std::getenv("GLM_OCR_ROPE_DEBUG") && vg.dbg_q_rope0) {
        std::vector<float> qd(64 * 16 * n_patches);
        ggml_backend_tensor_get(vg.dbg_q_rope0, qd.data(), 0, qd.size() * sizeof(float));
        FILE *f = fopen("/tmp/glm_cpp_q_rope0.bin", "wb");
        if (f) { fwrite(qd.data(), sizeof(float), qd.size(), f); fclose(f); }
        if (vg.dbg_k_rope0) {
            std::vector<float> kd(64 * 16 * n_patches);
            ggml_backend_tensor_get(vg.dbg_k_rope0, kd.data(), 0, kd.size() * sizeof(float));
            FILE *fk = fopen("/tmp/glm_cpp_k_rope0.bin", "wb");
            if (fk) { fwrite(kd.data(), sizeof(float), kd.size(), fk); fclose(fk); }
        }
        fprintf(stderr, "[rope] dumped layer0 Q/K-after-rope\n");
    }

    // Per-layer diff bisection (before the graph context is freed).
    if (!ctx.diff_ref_path.empty() && !vg.layer_outputs.empty()) {
        crispembed_diff::Ref lref;
        if (lref.load(ctx.diff_ref_path.c_str())) {
            std::vector<float> lbuf(vis_D * n_patches);
            for (size_t li = 0; li < vg.layer_outputs.size(); li++) {
                char nm[32];
                snprintf(nm, sizeof(nm), "vis_layer_%zu", li);
                if (!lref.has(nm)) continue;
                ggml_backend_tensor_get(vg.layer_outputs[li], lbuf.data(), 0,
                                        vis_D * n_patches * sizeof(float));
                auto r = lref.compare(nm, lbuf.data(), (size_t)vis_D * n_patches);
                printf("  %s: cos=%.6f max_abs=%.6f %s\n", nm, r.cos_min,
                       r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }
    }

    // Read only the final output (vis_output)
    std::vector<float> layer_data(vis_D * n_patches);
    ggml_backend_tensor_get(vg.output, layer_data.data(), 0,
                            vis_D * n_patches * sizeof(float));
    if (std::getenv("GLM_OCR_ROPE_DEBUG")) {
        FILE *f = fopen("/tmp/glm_cpp_postnorm.bin", "wb");
        if (f) { fwrite(layer_data.data(), sizeof(float), layer_data.size(), f); fclose(f); }
        fprintf(stderr, "[rope] dumped post_norm (%d x %d)\n", vis_D, n_patches);
    }
    ggml_free(vg.gctx);
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_vis).count();
        fprintf(stderr, "[glm_ocr-bench] vision_encoder: %lldms\n", (long long)ms);
    }

    // Vision encoder output is now in layer_data (vis_D × n_patches)
    // Debug: inject Python reference vision output if GLM_OCR_INJECT_VIS is set
    static const char *inject_vis = std::getenv("GLM_OCR_INJECT_VIS");
    if (inject_vis) {
        FILE *f = fopen(inject_vis, "rb");
        if (f) {
            fread(layer_data.data(), sizeof(float), layer_data.size(), f);
            fclose(f);
            fprintf(stderr, "[DEBUG] Injected reference vision output from %s\n", inject_vis);
        }
    }
    std::vector<float> &vis_out = layer_data;
    int vis_N = n_patches;

    // ── Downsample + Merger via ggml graph ──────────────────────
    // Gated: CRISPEMBED_GLM_OCR_SCALAR_MERGER=1 for CPU-scalar fallback
    auto t_ds = std::chrono::steady_clock::now();
    const int merge = (int)vhp.spatial_merge_size;
    const int out_h = n_ph / merge;  // 12
    const int out_w = n_pw / merge;  // 12
    const int out_D = (int)vhp.out_hidden_size;  // 1536
    const int n_merged = out_h * out_w;  // 144

    std::vector<float> merger_out(out_D * n_merged);
    static const bool scalar_merger = (std::getenv("CRISPEMBED_GLM_OCR_SCALAR_MERGER") != nullptr);

    if (scalar_merger) {
        // ── Scalar CPU fallback (original code) ──
        auto read_model_w = [&](ggml_tensor *t) -> std::vector<float> {
            if (!t) return {};
            size_t n = ggml_nelements(t);
            size_t nb = ggml_nbytes(t);
            std::vector<float> buf(n);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, buf.data(), 0, nb);
            } else {
                std::vector<uint8_t> raw(nb);
                ggml_backend_tensor_get(t, raw.data(), 0, nb);
                ggml_get_type_traits(t->type)->to_float(raw.data(), buf.data(), n);
            }
            return buf;
        };
        auto ds_w_buf = read_model_w(ctx.m.downsample_w);
        auto ds_b_buf = read_model_w(ctx.m.downsample_b);
        std::vector<float> ds_out(out_D * n_merged, 0.0f);
        for (int oh = 0; oh < out_h; oh++)
            for (int ow = 0; ow < out_w; ow++) {
                int out_idx = oh * out_w + ow;
                float patch[4096];
                for (int kh = 0; kh < 2; kh++)
                    for (int kw = 0; kw < 2; kw++) {
                        int in_idx = (oh * 2 + kh) * n_pw + (ow * 2 + kw);
                        for (int d = 0; d < D; d++)
                            patch[d * 4 + kh * 2 + kw] = vis_out[d + in_idx * D];
                    }
                for (int o = 0; o < out_D; o++) {
                    float sum = 0;
                    for (int k = 0; k < D * 4; k++)
                        sum += ds_w_buf[k + o * D * 4] * patch[k];
                    if (!ds_b_buf.empty()) sum += ds_b_buf[o];
                    ds_out[o + out_idx * out_D] = sum;
                }
            }
        auto proj_w = read_model_w(ctx.m.merger.proj_w);
        auto gate_w = read_model_w(ctx.m.merger.gate_w);
        auto up_w   = read_model_w(ctx.m.merger.up_w);
        auto down_w = read_model_w(ctx.m.merger.down_w);
        auto norm_w = read_model_w(ctx.m.merger.norm_w);
        auto norm_b = read_model_w(ctx.m.merger.norm_b);
        int inter = ctx.m.merger.gate_w ? (int)ctx.m.merger.gate_w->ne[0] : out_D * 3;
        for (int t = 0; t < n_merged; t++) {
            std::vector<float> x_proj(out_D), g(inter), u(inter), gu(inter), ffn(out_D);
            for (int j = 0; j < out_D; j++) { float s = 0; for (int i = 0; i < out_D; i++) s += proj_w[i + j * out_D] * ds_out[i + t * out_D]; x_proj[j] = s; }
            for (int j = 0; j < inter; j++) { float s = 0; for (int i = 0; i < out_D; i++) s += gate_w[i + j * out_D] * x_proj[i]; g[j] = s / (1.0f + std::exp(-s)); }
            for (int j = 0; j < inter; j++) { float s = 0; for (int i = 0; i < out_D; i++) s += up_w[i + j * out_D] * x_proj[i]; u[j] = s; }
            for (int k = 0; k < inter; k++) gu[k] = g[k] * u[k];
            for (int j = 0; j < out_D; j++) { float s = 0; for (int i = 0; i < inter; i++) s += down_w[i + j * inter] * gu[i]; ffn[j] = s; }
            float mean = 0; for (int d = 0; d < out_D; d++) mean += ffn[d]; mean /= out_D;
            float var = 0; for (int d = 0; d < out_D; d++) { float dd = ffn[d] - mean; var += dd * dd; } var /= out_D;
            for (int d = 0; d < out_D; d++)
                merger_out[d + t * out_D] = (ffn[d] - mean) / std::sqrt(var + 1e-6f) * norm_w[d] + (norm_b.empty() ? 0 : norm_b[d]);
        }
    } else {
        // ── ggml graph: downsample conv + merger (proj + SwiGLU + LN) ──
        // vis_out is (D, N) col-major = ggml (ne0=D, ne1=N). Reshape to (n_pw, n_ph, D) spatial.
        // ggml conv2d expects input as (W, H, C).
        // vis_out[d + tok*D] where tok = h*n_pw + w → need (W=n_pw, H=n_ph, C=D).
        // ggml col-major (D, N) already has spatial flattened correctly if we just reshape.

        size_t buf_sz = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(256, false);
        ggml_init_params mip{buf_sz, nullptr, true};
        ggml_context *mg = ggml_init(mip);

        // Input: vision encoder output as spatial (n_pw, n_ph, D)
        ggml_tensor *x = ggml_new_tensor_3d(mg, GGML_TYPE_F32, n_pw, n_ph, D);
        ggml_set_name(x, "vis_spatial"); ggml_set_input(x);

        // Downsample: Conv2D(D→out_D, 2×2, stride=2, bias)
        {
            ggml_tensor *w = ctx.m.downsample_w;
            // Reshape to 4D: (KW=2, KH=2, IC=D, OC=out_D)
            if (ggml_n_dims(w) <= 2) w = ggml_reshape_4d(mg, w, 2, 2, D, out_D);
            if (w->type != GGML_TYPE_F32) w = ggml_cast(mg, w, GGML_TYPE_F32);
            x = ggml_conv_2d_direct(mg, w, x, 2, 2, 0, 0, 1, 1);
            if (ctx.m.downsample_b) {
                ggml_tensor *b = ctx.m.downsample_b;
                if (b->type != GGML_TYPE_F32) b = ggml_cast(mg, b, GGML_TYPE_F32);
                x = ggml_add(mg, x, ggml_reshape_3d(mg, b, 1, 1, out_D));
            }
        }
        // x is now (out_w, out_h, out_D) = (12, 12, 1536)

        // Flatten spatial to tokens: (out_D, n_merged) = (1536, 144)
        x = ggml_cont(mg, ggml_permute(mg, x, 1, 2, 0, 3));  // (out_D, out_w, out_h)
        x = ggml_reshape_2d(mg, x, out_D, n_merged);           // (1536, 144)

        // Merger: proj → SwiGLU → LayerNorm (all batched matmuls)
        // proj: (out_D, out_D) × (out_D, n_merged) → (out_D, n_merged)
        x = ggml_mul_mat(mg, ctx.m.merger.proj_w, x);

        // SwiGLU: gate = silu(gate_w @ x), up = up_w @ x, down_w @ (gate * up)
        ggml_tensor *gate = ggml_silu(mg, ggml_mul_mat(mg, ctx.m.merger.gate_w, x));
        ggml_tensor *up = ggml_mul_mat(mg, ctx.m.merger.up_w, x);
        x = ggml_mul_mat(mg, ctx.m.merger.down_w, ggml_mul(mg, gate, up));

        // LayerNorm (with bias — use ggml_norm + mul + add)
        x = ggml_norm(mg, x, 1e-6f);
        if (ctx.m.merger.norm_w) x = ggml_mul(mg, x, ctx.m.merger.norm_w);
        if (ctx.m.merger.norm_b) x = ggml_add(mg, x, ctx.m.merger.norm_b);

        ggml_set_name(x, "merger_out"); ggml_set_output(x);

        ggml_cgraph *mgf = ggml_new_graph_custom(mg, 256, false);
        ggml_build_forward_expand(mgf, x);

        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, mgf)) {
            fprintf(stderr, "glm_ocr: merger graph alloc failed\n");
            ggml_free(mg);
            return false;
        }

        // Set input: reshape vis_out (D, N) to spatial (n_pw, n_ph, D)
        // vis_out is (D, N) flat. For ggml 3D (n_pw, n_ph, D), we need
        // the data laid out as [w + h*n_pw + c*n_pw*n_ph] = pixel at (w,h) channel c.
        // vis_out[d + tok*D] where tok = h*n_pw + w → vis_out[d + (h*n_pw+w)*D]
        // This is (D, n_pw*n_ph) which reshaped to (D, n_pw, n_ph) is NOT (n_pw, n_ph, D).
        // We need to permute: transpose vis_out from (D, N) to (N, D), then reshape to (n_pw, n_ph, D).
        // But vis_out data is already in the right order for ggml: ne[0]=D stride 1,
        // ne[1]=N stride D. Reshaped to (n_pw, n_ph, D): ne[0]=n_pw stride 1, ne[1]=n_ph stride n_pw,
        // ne[2]=D stride n_pw*n_ph. This is WRONG — we need spatial dims first.
        //
        // Fix: permute on CPU to (n_pw, n_ph, D) layout before upload.
        std::vector<float> spatial(D * n_ph * n_pw);
        for (int h = 0; h < n_ph; h++)
            for (int w = 0; w < n_pw; w++)
                for (int c = 0; c < D; c++)
                    spatial[c * n_ph * n_pw + h * n_pw + w] = vis_out[c + (h * n_pw + w) * D];

        ggml_backend_tensor_set(ggml_graph_get_tensor(mgf, "vis_spatial"),
                                spatial.data(), 0, D * n_ph * n_pw * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, mgf);

        ggml_backend_tensor_get(ggml_graph_get_tensor(mgf, "merger_out"),
                                merger_out.data(), 0, out_D * n_merged * sizeof(float));
        ggml_free(mg);
    }

    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_ds).count();
        fprintf(stderr, "[glm_ocr-bench] downsample+merger: %lldms\n", (long long)ms);
    }

    // Debug: inject Python reference merger output if GLM_OCR_INJECT_MERGER is set
    static const char *inject_merger = std::getenv("GLM_OCR_INJECT_MERGER");
    if (inject_merger) {
        FILE *f = fopen(inject_merger, "rb");
        if (f) {
            fread(merger_out.data(), sizeof(float), merger_out.size(), f);
            fclose(f);
            fprintf(stderr, "[DEBUG] Injected reference merger output from %s\n", inject_merger);
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
                auto r = ref.compare("vis_patch_embed", patch_embed_data.data(),
                                     patch_embed_data.size());
                printf("  vis_patch_embed: cos=%.6f max_abs=%.6f %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            // Post-norm (final vision output after all 24 layers + LN)
            if (ref.has("vis_post_norm")) {
                auto r = ref.compare("vis_post_norm", layer_data.data(), layer_data.size());
                printf("  vis_post_norm: cos=%.6f max_abs=%.6f %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            // Merger
            if (ref.has("vis_merger_output")) {
                auto r = ref.compare("vis_merger_output", merger_out.data(), merger_out.size());
                printf("  vis_merger_output: cos=%.6f max_abs=%.6f %s\n",
                       r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
            // Per-layer comparisons already printed in per-layer loop above
        }
    }
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

    // Vision-text splice (during prefill only)
    if (n_past == 0) {
        ggml_tensor *img_embeds = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(img_embeds, "image_embeds"); ggml_set_input(img_embeds);
        ggml_tensor *splice_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(splice_mask, "splice_mask"); ggml_set_input(splice_mask);
        x = ggml_add(g, ggml_mul(g, x, splice_mask), img_embeds);
    }

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

            // flash_attn_ext handles GQA natively — no need to expand heads
            Kfull = ggml_cont(g, k_layer);
            Vfull = ggml_cont(g, v_layer);
        } else {
            // No KV cache: permute to (hd, T, nkv) for flash_attn
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

struct splice_data {
    const float *image_embeds = nullptr;  // (D, n_image_tokens)
    int n_image_tokens = 0;
    const int *token_to_image = nullptr;  // (n_tokens,): image index or -1
};

static bool run_cached_step(context &ctx, const int32_t *token_ids, int n_tokens,
                            int n_past, std::vector<float> &last_logits_out,
                            const splice_data *splice = nullptr) {
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

    // mRoPE positions — must match transformers glm_ocr get_rope_index /
    // get_vision_position_ids exactly:
    //   text token: all 3 dims = current_pos; current_pos++
    //   image block (t=1, h=gh, w=gw after spatial_merge): for patch k
    //     (row = k/gw, col = k%gw), temporal = start, height = start+row,
    //     width = start+col, where start = current_pos at the block's first
    //     image token; after the block current_pos += max(gh, gw).
    // Because the image block advances current_pos by only max(gh,gw) (not by
    // the number of image tokens), the mRoPE position after prefill differs from
    // the token count — decode steps continue from ctx.mrope_next_pos.
    const int merge_size = (int)ctx.m.vhp.spatial_merge_size;
    const int img_h = (int)ctx.m.vhp.image_size / (int)ctx.m.vhp.patch_size / merge_size;
    const int img_w = img_h;
    const int img_token_id = (int)ctx.m.lhp.image_token_id;

    std::vector<int32_t> pos_data(T * 4, 0);
    if (n_past == 0) {
        int current_pos = 0;
        int img_idx = 0;
        for (int j = 0; j < T; j++) {
            if (token_ids[j] == img_token_id) {
                int start = current_pos;              // block start (same for all patches)
                int row = img_idx / img_w;
                int col = img_idx % img_w;
                pos_data[j]       = start;            // temporal (t grid = 1)
                pos_data[T + j]   = start + row;      // height
                pos_data[2*T + j] = start + col;      // width
                img_idx++;
                // Advance current_pos once the block ends (next token is text).
                if (j + 1 >= T || token_ids[j + 1] != img_token_id) {
                    current_pos = start + std::max(img_h, img_w);
                    img_idx = 0;
                }
            } else {
                pos_data[j]       = current_pos;
                pos_data[T + j]   = current_pos;
                pos_data[2*T + j] = current_pos;
                current_pos++;
            }
        }
        ctx.mrope_next_pos = current_pos;
    } else {
        // Decode: single (or few) new text tokens continue from mrope_next_pos.
        for (int j = 0; j < T; j++) {
            int p = ctx.mrope_next_pos + j;
            pos_data[j] = p; pos_data[T + j] = p; pos_data[2*T + j] = p;
        }
        ctx.mrope_next_pos += T;
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

    // Set splice inputs
    if (n_past == 0) {
        const int D = (int)ctx.m.lhp.hidden_size;
        ggml_tensor *sm = ggml_graph_get_tensor(lg.gf, "splice_mask");
        ggml_tensor *ie = ggml_graph_get_tensor(lg.gf, "image_embeds");
        if (splice && splice->token_to_image) {
            std::vector<float> img_data((size_t)D * T, 0.0f);
            std::vector<float> mask_f((size_t)D * T, 1.0f);
            for (int t = 0; t < T; t++) {
                int idx = splice->token_to_image[t];
                if (idx >= 0 && idx < splice->n_image_tokens) {
                    for (int d = 0; d < D; d++) {
                        img_data[(size_t)t * D + d] = splice->image_embeds[(size_t)idx * D + d];
                        mask_f[(size_t)t * D + d] = 0.0f;
                    }
                }
            }
            if (ie) ggml_backend_tensor_set(ie, img_data.data(), 0, (size_t)D * T * sizeof(float));
            if (sm) ggml_backend_tensor_set(sm, mask_f.data(), 0, (size_t)D * T * sizeof(float));
        } else {
            if (ie) { std::vector<float> z((size_t)D * T, 0.0f); ggml_backend_tensor_set(ie, z.data(), 0, z.size() * sizeof(float)); }
            if (sm) { std::vector<float> o((size_t)D * T, 1.0f); ggml_backend_tensor_set(sm, o.data(), 0, o.size() * sizeof(float)); }
        }
    }

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

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *prompt_ids, int n_prompt,
              int max_new_tokens, generate_result &out) {
    const auto &lhp = ctx.m.lhp;
    const int V = (int)lhp.vocab_size;
    const int eos_id = (int)lhp.eos_token_id;
    const int max_seq = n_prompt + max_new_tokens + 16;
    const int img_token_id = (int)lhp.image_token_id;

    if (!ctx.kvc.allocated || ctx.kvc.max_seq < max_seq) {
        free_kv_cache(ctx);
        if (!alloc_kv_cache(ctx, max_seq)) return false;
    }
    ctx.kvc.n_past = 0;

    // Build splice mapping
    splice_data sd = {};
    std::vector<int> token_to_image;
    if (image_embeds && n_image_tokens > 0) {
        sd.image_embeds = image_embeds;
        sd.n_image_tokens = n_image_tokens;
        token_to_image.resize(n_prompt, -1);
        int img_idx = 0;
        for (int t = 0; t < n_prompt && img_idx < n_image_tokens; t++) {
            if (prompt_ids[t] == img_token_id)
                token_to_image[t] = img_idx++;
        }
        sd.token_to_image = token_to_image.data();
        if (ctx.verbosity >= 1)
            fprintf(stderr, "  Spliced %d image tokens into %d prompt tokens\n",
                    img_idx, n_prompt);
    }

    const bool bench = ctx.bench;
    auto t_gen_total = std::chrono::steady_clock::now();

    // Prefill
    auto t_prefill = std::chrono::steady_clock::now();
    std::vector<float> logits;
    const splice_data *sd_ptr = (image_embeds && n_image_tokens > 0) ? &sd : nullptr;
    if (!run_cached_step(ctx, prompt_ids, n_prompt, 0, logits, sd_ptr)) return false;
    ctx.kvc.n_past = n_prompt;
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_prefill).count();
        fprintf(stderr, "[glm_ocr-bench] prefill: %lldms\n", (long long)ms);
    }

    out.token_confidences.clear();

    int best_id = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++)
        if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }

    // Confidence: numerically-stable softmax for winning token
    {
        float max_l = best_score;
        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) sum_exp += expf(logits[v] - max_l);
        out.token_confidences.push_back(expf(best_score - max_l) / sum_exp);
    }

    out.token_ids.push_back(best_id);
    if (ctx.verbosity >= 1)
        fprintf(stderr, "  gen[0]: token=%d score=%.2f (prefill %d)\n", best_id, best_score, n_prompt);
    if (best_id == eos_id) { out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size()); return true; }

    // Decode
    long long decode_total_ms = 0;
    int decode_steps = 0;
    for (int gen = 1; gen < max_new_tokens; gen++) {
        auto t_step = std::chrono::steady_clock::now();
        int32_t next = best_id;
        if (!run_cached_step(ctx, &next, 1, ctx.kvc.n_past, logits)) return false;
        ctx.kvc.n_past += 1;
        if (bench) {
            auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_step).count();
            decode_total_ms += step_ms;
            decode_steps++;
            fprintf(stderr, "[glm_ocr-bench] decode_step[%d]: %lldms\n", gen, (long long)step_ms);
        }

        best_id = 0; best_score = -INFINITY;
        for (int v = 0; v < V; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }

        {
            float max_l = best_score;
            float sum_exp = 0.0f;
            for (int v = 0; v < V; v++) sum_exp += expf(logits[v] - max_l);
            out.token_confidences.push_back(expf(best_score - max_l) / sum_exp);
        }

        out.token_ids.push_back(best_id);
        if (ctx.verbosity >= 1)
            fprintf(stderr, "  gen[%d]: token=%d score=%.2f\n", gen, best_id, best_score);
        if (best_id == eos_id) break;
    }

    out.text = ctx.tok.decode(out.token_ids.data(), (int)out.token_ids.size());
    if (bench) {
        fprintf(stderr, "[glm_ocr-bench] decode_total: %lldms (%d steps)\n",
                decode_total_ms, decode_steps);
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_gen_total).count();
        fprintf(stderr, "[glm_ocr-bench] total: %lldms\n", (long long)total_ms);
    }
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

    // Set splice to identity (no image in parity test)
    ggml_tensor *sm = ggml_graph_get_tensor(lg.gf, "splice_mask");
    ggml_tensor *ie = ggml_graph_get_tensor(lg.gf, "image_embeds");
    if (sm) { std::vector<float> o((size_t)D * T, 1.0f); ggml_backend_tensor_set(sm, o.data(), 0, o.size() * sizeof(float)); }
    if (ie) { std::vector<float> z((size_t)D * T, 0.0f); ggml_backend_tensor_set(ie, z.data(), 0, z.size() * sizeof(float)); }

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
    std::vector<float> char_confidences;
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
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }

    auto &v = ctx->ctx.m.vhp;
    auto &l = ctx->ctx.m.lhp;
    int imgS = (int)v.image_size; // 336

    // Resize to (imgS, imgS) with bilinear interpolation + CLIP normalize
    std::vector<float> pixels(3 * imgS * imgS);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < imgS; y++) {
            float fy = (float)y * h / imgS;
            int iy = std::min((int)fy, h - 1);
            for (int x = 0; x < imgS; x++) {
                float fx = (float)x * w / imgS;
                int ix = std::min((int)fx, w - 1);
                int ci = std::min(c, ch - 1);
                float val = (float)px[(iy * w + ix) * ch + ci] / 255.0f;
                pixels[c * imgS * imgS + y * imgS + x] =
                    (val - v.image_mean[c]) / v.image_std[c];
            }
        }
    }

    // Vision encode
    glm_ocr::vision_result vr;
    if (!glm_ocr::encode_vision(ctx->ctx, pixels.data(), vr)) {
        fprintf(stderr, "glm_ocr: vision encoding failed\n");
        if (out_len) *out_len = 0;
        return "";
    }

    // Build prompt from GLM-OCR's chat template (chat_template.jinja) applied to
    // the recommended single-turn OCR message
    //   [{role:user, content:[{image}, {text:"Text Recognition:"}]}]
    // with add_generation_prompt=True. Rendered (trim/lstrip blocks) as:
    //   [gMASK]<sop><|user|>\n<|begin_of_image|><|image|>*N<|end_of_image|>Text Recognition:<|assistant|>\n
    // NOTE: there is NO system turn, the leading [gMASK] is required, and the
    // instruction text sits between <|end_of_image|> and <|assistant|> with no
    // surrounding newlines. (The previous prompt dropped [gMASK] and the
    // instruction and injected a spurious empty <|system|> block → garbage.)
    int n_img_tokens = vr.n_tokens;
    const int32_t gmask_id = 59248;
    const int32_t sop_id   = 59250;
    const int32_t user_id  = 59253;
    const int32_t asst_id  = 59254;
    const int32_t boi_id   = 59256;
    const int32_t eoi_id   = 59257;
    const int32_t img_id   = (int32_t)l.image_token_id;  // 59280
    const int32_t nl_id    = 10;     // '\n' ('Ċ') in chatglm-bpe tokenizer
    // "Text Recognition:" → BPE ids (Text / ĠRec / ognition / :)
    const int32_t instr_ids[] = {3649, 7404, 49600, 58};

    std::vector<int32_t> prompt;
    prompt.push_back(gmask_id);
    prompt.push_back(sop_id);
    prompt.push_back(user_id);
    prompt.push_back(nl_id);
    // <|begin_of_image|> <|image|>*N <|end_of_image|>
    prompt.push_back(boi_id);
    for (int i = 0; i < n_img_tokens; i++)
        prompt.push_back(img_id);
    prompt.push_back(eoi_id);
    // Instruction text (no surrounding newlines)
    for (int32_t t : instr_ids) prompt.push_back(t);
    // <|assistant|>\n
    prompt.push_back(asst_id);
    prompt.push_back(nl_id);

    // Generate
    glm_ocr::generate_result gen;
    bool ok = glm_ocr::generate(ctx->ctx,
        vr.hidden, n_img_tokens, (int)vr.hidden_dim,
        prompt.data(), (int)prompt.size(), 1024, gen);
    free(vr.hidden);

    if (!ok) {
        if (out_len) *out_len = 0;
        return "";
    }

    ctx->last_text = gen.text;
    ctx->char_confidences = std::move(gen.token_confidences);
    if (out_len) *out_len = (int)ctx->last_text.size();
    return ctx->last_text.c_str();
}

const char * glm_ocr_recognize(glm_ocr_context *ctx,
    const float *px, int w, int h, int *out_len) {
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, px[i] * 255.0f));
        rgb[i * 3 + 0] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    return glm_ocr_recognize_raw(ctx, rgb.data(), w, h, 3, out_len);
}

const float * glm_ocr_confidences(const glm_ocr_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float glm_ocr_mean_confidence(const glm_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
