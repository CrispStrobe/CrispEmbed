// qwen2vl_ocr.cpp — Qwen2.5-VL OCR inference engine.
//
// Architecture (Qwen2.5-VL-3B-Instruct):
//
// Vision encoder:
//   patches → Conv3D (flattened matmul) → + 2D RoPE pos embed
//   32 × pre-RMSNorm ViT block:
//     RMSNorm → fused QKV + RoPE → attention (windowed or full) → residual
//     RMSNorm → SwiGLU FFN (gate + up + down) → residual
//   LayerNorm → spatial merge (4:1) → FC1 → GELU → FC2 → (n_merged, 2048)
//
// LLM decoder:
//   embed_tokens → splice image_embeds at image_token positions
//   36 × pre-RMSNorm Qwen2 block:
//     RMSNorm → Q/K/V + bias (GQA 16/2) + mRoPE → attention → residual
//     RMSNorm → SwiGLU FFN → residual
//   RMSNorm → lm_head → logits → greedy decode
//
// Per-layer diff comparison via crispembed_diff.h when ctx.diff_ref_path is set.

#include "qwen2vl_ocr.h"
#include "crispembed_diff.h"
#include "image_preprocess.h"
#include "tokenizer.h"
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
#include <map>
#include <string>
#include <vector>

namespace qwen2vl_ocr {

namespace {

// ── Hparams loading ──────────────────────────────────────────────────

bool load_hparams(context &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *k, uint32_t d) {
        return core_gguf::kv_u32(g, k, d);
    };
    auto f32v = [&](const char *k, float d) {
        return core_gguf::kv_f32(g, k, d);
    };

    auto &vhp = ctx.m.vhp;
    auto &lhp = ctx.m.lhp;

    // Vision
    vhp.depth               = u32("qwen2vl.vision.depth", vhp.depth);
    vhp.hidden_size         = u32("qwen2vl.vision.hidden_size", vhp.hidden_size);
    vhp.intermediate_size   = u32("qwen2vl.vision.intermediate_size", vhp.intermediate_size);
    vhp.num_heads           = u32("qwen2vl.vision.num_heads", vhp.num_heads);
    vhp.in_channels         = u32("qwen2vl.vision.in_channels", vhp.in_channels);
    vhp.spatial_patch_size  = u32("qwen2vl.vision.spatial_patch_size", vhp.spatial_patch_size);
    vhp.spatial_merge_size  = u32("qwen2vl.vision.spatial_merge_size", vhp.spatial_merge_size);
    vhp.temporal_patch_size = u32("qwen2vl.vision.temporal_patch_size", vhp.temporal_patch_size);
    vhp.out_hidden_size     = u32("qwen2vl.vision.out_hidden_size", vhp.out_hidden_size);
    vhp.window_size         = u32("qwen2vl.vision.window_size", vhp.window_size);
    vhp.min_pixels          = u32("qwen2vl.vision.min_pixels", vhp.min_pixels);
    vhp.max_pixels          = u32("qwen2vl.vision.max_pixels", vhp.max_pixels);

    // Fullatt block indexes
    int idx = gguf_find_key(g, "qwen2vl.vision.fullatt_block_indexes");
    if (idx >= 0) {
        int n = gguf_get_arr_n(g, idx);
        vhp.fullatt_block_indexes.resize(n);
        for (int i = 0; i < n; i++) {
            vhp.fullatt_block_indexes[i] =
                (int)((const uint32_t *)gguf_get_arr_data(g, idx))[i];
        }
    }

    // Image preprocessor arrays
    idx = gguf_find_key(g, "qwen2vl.vision.image_mean");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *data = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_mean[i] = data[i];
    }
    idx = gguf_find_key(g, "qwen2vl.vision.image_std");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *data = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_std[i] = data[i];
    }

    // LLM — CrispEmbed keys, then llama.cpp keys as fallback
    lhp.vocab_size              = u32("qwen2vl.vocab_size", lhp.vocab_size);
    lhp.hidden_size             = u32("qwen2vl.hidden_size",
                                      u32("qwen2vl.embedding_length", lhp.hidden_size));
    lhp.intermediate_size       = u32("qwen2vl.intermediate_size",
                                      u32("qwen2vl.feed_forward_length", lhp.intermediate_size));
    lhp.num_hidden_layers       = u32("qwen2vl.num_hidden_layers",
                                      u32("qwen2vl.block_count", lhp.num_hidden_layers));
    lhp.num_attention_heads     = u32("qwen2vl.num_attention_heads",
                                      u32("qwen2vl.attention.head_count", lhp.num_attention_heads));
    lhp.num_key_value_heads     = u32("qwen2vl.num_key_value_heads",
                                      u32("qwen2vl.attention.head_count_kv", lhp.num_key_value_heads));
    lhp.max_position_embeddings = u32("qwen2vl.max_position_embeddings",
                                      u32("qwen2vl.context_length", lhp.max_position_embeddings));
    lhp.rms_norm_eps            = f32v("qwen2vl.rms_norm_eps",
                                       f32v("qwen2vl.attention.layer_norm_rms_epsilon", lhp.rms_norm_eps));
    lhp.rope_theta              = f32v("qwen2vl.rope_theta",
                                       f32v("qwen2vl.rope.freq_base", lhp.rope_theta));
    lhp.image_token_id          = u32("qwen2vl.image_token_id", lhp.image_token_id);
    lhp.video_token_id          = u32("qwen2vl.video_token_id", lhp.video_token_id);
    lhp.vision_start_token_id   = u32("qwen2vl.vision_start_token_id", lhp.vision_start_token_id);
    lhp.vision_end_token_id     = u32("qwen2vl.vision_end_token_id", lhp.vision_end_token_id);

    // mRoPE sections (try qwen3vl prefix first, then qwen2vl)
    idx = gguf_find_key(g, "qwen3vl.rope_sections");
    if (idx < 0) idx = gguf_find_key(g, "qwen2vl.rope_sections");
    if (idx >= 0) {
        int n = std::min(4, (int)gguf_get_arr_n(g, idx));
        auto *data = (const uint32_t *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < n; i++) lhp.rope_sections[i] = (int)data[i];
    }

    // Qwen3-VL: interleaved mRoPE + QK norms
    int interleaved_idx = gguf_find_key(g, "qwen3vl.mrope_interleaved");
    if (interleaved_idx >= 0) lhp.mrope_interleaved = gguf_get_val_u32(g, interleaved_idx) != 0;
    int qknorm_idx = gguf_find_key(g, "qwen3vl.has_qk_norm");
    if (qknorm_idx >= 0) lhp.has_qk_norm = gguf_get_val_u32(g, qknorm_idx) != 0;

    // Qwen3-VL: deepstack visual indexes
    idx = gguf_find_key(g, "qwen3vl.vision.deepstack_indexes");
    if (idx >= 0) {
        int n = (int)gguf_get_arr_n(g, idx);
        auto *data = (const int32_t *)gguf_get_arr_data(g, idx);
        vhp.deepstack_indexes.resize(n);
        for (int i = 0; i < n; i++) vhp.deepstack_indexes[i] = data[i];
    }

    // Also try qwen3vl prefix for all config keys
    auto u32_3 = [&](const char *k, uint32_t d) {
        std::string qwen3_key = std::string("qwen3vl") + (strchr(k, '.') ? strchr(k, '.') : "");
        int i = gguf_find_key(g, qwen3_key.c_str());
        return i >= 0 ? (uint32_t)gguf_get_val_u32(g, i) : d;
    };
    // Re-read with qwen3vl prefix (overrides qwen2vl if present)
    vhp.depth           = u32_3("qwen3vl.vision.depth", vhp.depth);
    vhp.hidden_size     = u32_3("qwen3vl.vision.hidden_size", vhp.hidden_size);
    vhp.num_heads       = u32_3("qwen3vl.vision.num_heads", vhp.num_heads);
    vhp.spatial_patch_size = u32_3("qwen3vl.vision.patch_size", vhp.spatial_patch_size);
    vhp.spatial_merge_size = u32_3("qwen3vl.vision.spatial_merge_size", vhp.spatial_merge_size);
    vhp.temporal_patch_size = u32_3("qwen3vl.vision.temporal_patch_size", vhp.temporal_patch_size);
    vhp.out_hidden_size = u32_3("qwen3vl.vision.out_hidden_size", vhp.out_hidden_size);
    lhp.vocab_size      = u32_3("qwen3vl.vocab_size", lhp.vocab_size);
    lhp.hidden_size     = u32_3("qwen3vl.hidden_size", lhp.hidden_size);
    lhp.intermediate_size = u32_3("qwen3vl.intermediate_size", lhp.intermediate_size);
    lhp.num_hidden_layers = u32_3("qwen3vl.num_hidden_layers", lhp.num_hidden_layers);
    lhp.num_attention_heads = u32_3("qwen3vl.num_attention_heads", lhp.num_attention_heads);
    lhp.num_key_value_heads = u32_3("qwen3vl.num_key_value_heads", lhp.num_key_value_heads);
    lhp.image_token_id  = u32_3("qwen3vl.image_token_id", lhp.image_token_id);

    auto f32_3 = [&](const char *k, float d) {
        int i = gguf_find_key(g, k);
        return i >= 0 ? gguf_get_val_f32(g, i) : d;
    };
    lhp.rope_theta = f32_3("qwen3vl.rope_theta", lhp.rope_theta);

    // Tie embeddings
    int tie_idx = gguf_find_key(g, "qwen3vl.tie_word_embeddings");
    if (tie_idx < 0) tie_idx = gguf_find_key(g, "qwen2vl.tie_word_embeddings");
    if (tie_idx >= 0) {
        // May be stored as BOOL or UINT32 depending on converter version
        auto tie_type = gguf_get_kv_type(g, tie_idx);
        if (tie_type == GGUF_TYPE_BOOL)
            lhp.tie_word_embeddings = gguf_get_val_bool(g, tie_idx);
        else
            lhp.tie_word_embeddings = gguf_get_val_u32(g, tie_idx) != 0;
    }

    core_gguf::free_metadata(g);
    return true;
}

// ── Tensor loading ───────────────────────────────────────────────────

bool load_tensors(context &ctx, const char *path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "qwen2vl_ocr", wl)) {
        return false;
    }
    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;

    auto &m = ctx.m;
    auto get = [&](const std::string &name) -> ggml_tensor * {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };
    // Try first name, fall back to alternative (for llama.cpp GGUF compat)
    auto get2 = [&](const std::string &a, const std::string &b) -> ggml_tensor * {
        auto *t = get(a);
        return t ? t : get(b);
    };

    // Vision encoder (CrispEmbed names; mmproj names as fallback)
    m.patch_embed_w = get2("v.patch_embed.weight", "v.patch_embd.weight");
    m.patch_embed_b = get("v.patch_embed.bias");

    m.vis_blocks.resize(m.vhp.depth);
    for (uint32_t i = 0; i < m.vhp.depth; i++) {
        auto &blk = m.vis_blocks[i];
        std::string p = "v.blk." + std::to_string(i) + ".";
        // Norms: CrispEmbed "norm1" vs llama.cpp mmproj "ln1"
        blk.norm1_w    = get2(p + "norm1.weight", p + "ln1.weight");
        blk.norm1_b    = get2(p + "norm1.bias",   p + "ln1.bias");
        blk.norm2_w    = get2(p + "norm2.weight", p + "ln2.weight");
        blk.norm2_b    = get2(p + "norm2.bias",   p + "ln2.bias");
        // Attention: fused QKV or separate Q/K/V (mmproj uses separate)
        blk.qkv_w      = get(p + "attn_qkv.weight");
        blk.qkv_b      = get(p + "attn_qkv.bias");
        blk.q_w        = get(p + "attn_q.weight");
        blk.q_b        = get(p + "attn_q.bias");
        blk.k_w        = get(p + "attn_k.weight");
        blk.k_b        = get(p + "attn_k.bias");
        blk.v_w        = get(p + "attn_v.weight");
        blk.v_b        = get(p + "attn_v.bias");
        blk.proj_w     = get2(p + "attn_proj.weight", p + "attn_out.weight");
        blk.proj_b     = get2(p + "attn_proj.bias",   p + "attn_out.bias");
        // SwiGLU FFN (Qwen2.5-VL)
        blk.ffn_gate_w = get(p + "ffn_gate.weight");
        blk.ffn_gate_b = get(p + "ffn_gate.bias");
        blk.ffn_up_w   = get(p + "ffn_up.weight");
        blk.ffn_up_b   = get(p + "ffn_up.bias");
        blk.ffn_down_w = get(p + "ffn_down.weight");
        blk.ffn_down_b = get(p + "ffn_down.bias");
        // GELU fc1/fc2 FFN (Qwen2-VL)
        blk.ffn_fc1_w  = get(p + "ffn_fc1.weight");
        blk.ffn_fc1_b  = get(p + "ffn_fc1.bias");
        blk.ffn_fc2_w  = get(p + "ffn_fc2.weight");
        blk.ffn_fc2_b  = get(p + "ffn_fc2.bias");
        // mmproj uses ffn_up/ffn_down for GELU too — map to fc1/fc2 if needed
        if (!blk.ffn_fc1_w && !blk.ffn_gate_w && blk.ffn_up_w) {
            blk.ffn_fc1_w = blk.ffn_up_w;   blk.ffn_fc1_b = blk.ffn_up_b;
            blk.ffn_fc2_w = blk.ffn_down_w;  blk.ffn_fc2_b = blk.ffn_down_b;
        }
    }

    // Auto-detect Qwen2-VL vs Qwen2.5-VL from loaded weights:
    // Qwen2-VL has norm bias + fc1/fc2; Qwen2.5-VL has no norm bias + gate/up/down
    if (m.vhp.depth > 0 && m.vis_blocks[0].norm1_b != nullptr) {
        m.vhp.is_qwen2_vl = true;
        if (ctx.verbosity >= 1) {
            fprintf(stderr, "  vision variant: Qwen2-VL (LayerNorm, GELU fc1/fc2)\n");
        }
    } else {
        if (ctx.verbosity >= 1) {
            fprintf(stderr, "  vision variant: Qwen2.5-VL (RMSNorm, SwiGLU)\n");
        }
    }

    m.merger.norm_w = get("v.merger.norm.weight");
    m.merger.norm_b = get("v.merger.norm.bias");
    m.merger.fc1_w  = get2("v.merger.fc1.weight", "mm.0.weight");
    m.merger.fc1_b  = get2("v.merger.fc1.bias",   "mm.0.bias");
    m.merger.fc2_w  = get2("v.merger.fc2.weight", "mm.2.weight");
    m.merger.fc2_b  = get2("v.merger.fc2.bias",   "mm.2.bias");

    // LLM decoder (CrispEmbed "l.blk.*" or llama.cpp "blk.*")
    m.embed_tokens = get2("l.embed_tokens.weight", "token_embd.weight");

    m.llm_layers.resize(m.lhp.num_hidden_layers);
    for (uint32_t i = 0; i < m.lhp.num_hidden_layers; i++) {
        auto &ly = m.llm_layers[i];
        std::string p1 = "l.blk." + std::to_string(i) + ".";  // CrispEmbed
        std::string p2 = "blk." + std::to_string(i) + ".";    // llama.cpp
        ly.attn_norm_w = get2(p1 + "attn_norm.weight", p2 + "attn_norm.weight");
        ly.ffn_norm_w  = get2(p1 + "ffn_norm.weight",  p2 + "ffn_norm.weight");
        ly.q_w         = get2(p1 + "attn_q.weight",    p2 + "attn_q.weight");
        ly.q_b         = get2(p1 + "attn_q.bias",      p2 + "attn_q.bias");
        ly.k_w         = get2(p1 + "attn_k.weight",    p2 + "attn_k.weight");
        ly.k_b         = get2(p1 + "attn_k.bias",      p2 + "attn_k.bias");
        ly.v_w         = get2(p1 + "attn_v.weight",    p2 + "attn_v.weight");
        ly.v_b         = get2(p1 + "attn_v.bias",      p2 + "attn_v.bias");
        ly.o_w         = get2(p1 + "attn_o.weight",    p2 + "attn_output.weight");
        ly.ffn_gate_w  = get2(p1 + "ffn_gate.weight",  p2 + "ffn_gate.weight");
        ly.ffn_up_w    = get2(p1 + "ffn_up.weight",    p2 + "ffn_up.weight");
        ly.ffn_down_w  = get2(p1 + "ffn_down.weight",  p2 + "ffn_down.weight");

        // Qwen3-VL: QK norms (try both naming conventions)
        std::string p3 = "llm.blk." + std::to_string(i) + ".";
        ly.q_norm_w = get(p1 + "attn_q_norm.weight");
        if (!ly.q_norm_w) ly.q_norm_w = get(p3 + "attn.q_norm.weight");
        ly.k_norm_w = get(p1 + "attn_k_norm.weight");
        if (!ly.k_norm_w) ly.k_norm_w = get(p3 + "attn.k_norm.weight");

        // Also try qwen3vl naming for Q/K/V/O if CrispEmbed/llama.cpp names not found
        if (!ly.q_w) ly.q_w = get(p3 + "attn.q.weight");
        if (!ly.k_w) ly.k_w = get(p3 + "attn.k.weight");
        if (!ly.v_w) ly.v_w = get(p3 + "attn.v.weight");
        if (!ly.o_w) ly.o_w = get(p3 + "attn.o.weight");
        if (!ly.attn_norm_w) ly.attn_norm_w = get(p3 + "norm1.weight");
        if (!ly.ffn_norm_w) ly.ffn_norm_w = get(p3 + "norm2.weight");
        if (!ly.ffn_gate_w) ly.ffn_gate_w = get(p3 + "ffn.gate.weight");
        if (!ly.ffn_up_w) ly.ffn_up_w = get(p3 + "ffn.up.weight");
        if (!ly.ffn_down_w) ly.ffn_down_w = get(p3 + "ffn.down.weight");
    }

    m.output_norm_w = get2("l.output_norm.weight", "output_norm.weight");
    if (!m.output_norm_w) m.output_norm_w = get("llm.norm.weight");
    m.lm_head_w     = get2("l.lm_head.weight", "output.weight");
    if (!m.lm_head_w) m.lm_head_w = get("llm.lm_head.weight");
    if (!m.lm_head_w) m.lm_head_w = get("llm.embed.weight");  // tied

    return true;
}

// ── 2D RoPE computation (host-side) ──────────────────────────────────

struct host_rope {
    std::vector<float> cos_buf;  // (n_patches, head_dim)
    std::vector<float> sin_buf;  // (n_patches, head_dim)
};

void compute_vision_rope(host_rope &out, const int32_t *grid_thw,
                         int n_patches, int head_dim, float theta = 10000.0f) {
    out.cos_buf.resize((size_t)n_patches * head_dim);
    out.sin_buf.resize((size_t)n_patches * head_dim);

    const int quart = head_dim / 4;
    std::vector<float> inv_freq(quart);
    for (int j = 0; j < quart; j++) {
        inv_freq[j] = 1.0f / std::pow(theta, (float)(2 * j) / (float)head_dim);
    }

    int t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
    int tok = 0;
    for (int f = 0; f < t; f++) {
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                float *cr = out.cos_buf.data() + (size_t)tok * head_dim;
                float *sr = out.sin_buf.data() + (size_t)tok * head_dim;
                for (int j = 0; j < quart; j++) {
                    float vr = (float)row * inv_freq[j];
                    float vc = (float)col * inv_freq[j];
                    cr[j]                = std::cos(vr);
                    sr[j]                = std::sin(vr);
                    cr[j + quart]        = std::cos(vc);
                    sr[j + quart]        = std::sin(vc);
                    cr[j + 2 * quart]    = std::cos(vr);
                    sr[j + 2 * quart]    = std::sin(vr);
                    cr[j + 3 * quart]    = std::cos(vc);
                    sr[j + 3 * quart]    = std::sin(vc);
                }
                tok++;
            }
        }
    }
}

// ── Vision encoder graph ─────────────────────────────────────────────

struct vision_graph_result {
    ggml_cgraph *gf = nullptr;
    ggml_tensor *output = nullptr;
    int h_patches = 0;
    int w_patches = 0;
};

vision_graph_result build_vision_graph(context &ctx, int n_patches,
                                        const int32_t *grid_thw) {
    const auto &vhp = ctx.m.vhp;
    const int H = (int)vhp.hidden_size;
    const int n_heads = (int)vhp.num_heads;
    const int head_dim = H / n_heads;
    const int merge = (int)vhp.spatial_merge_size;
    const int merge_unit = merge * merge;
    const int merger_in_dim = H * merge_unit;
    const int patch_flat_dim = (int)vhp.in_channels *
                               (int)vhp.temporal_patch_size *
                               (int)vhp.spatial_patch_size *
                               (int)vhp.spatial_patch_size;

    const int h_patches = grid_thw[1];
    const int w_patches = grid_thw[2];
    const int n_merged = (h_patches / merge) * (w_patches / merge);

    ggml_init_params ip{
        ctx.compute_meta.size(),
        ctx.compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // ── Inputs ──
    ggml_tensor *pixel_in = ggml_new_tensor_2d(g, GGML_TYPE_F32,
                                                patch_flat_dim, n_patches);
    ggml_set_name(pixel_in, "pixel_in");
    ggml_set_input(pixel_in);

    // RoPE cos/sin: (head_dim, 1, n_patches) — broadcasts over n_heads
    ggml_tensor *cos_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, head_dim, 1, n_patches);
    ggml_tensor *sin_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, head_dim, 1, n_patches);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    // ── Patch embedding ──
    // Use model tensor directly — no reshape (already ne=(patch_flat_dim, H))
    ggml_tensor *x = ggml_mul_mat(g, ctx.m.patch_embed_w, pixel_in);
    if (ctx.m.patch_embed_b) {
        x = ggml_add(g, x, ctx.m.patch_embed_b);
    }
    ggml_set_name(x, "patch_embed_out");
    if (!ctx.diff_ref_path.empty()) ggml_set_output(x);

    // ── RMSNorm helper ──
    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        ggml_tensor *y = ggml_rms_norm(g, t, 1e-6f);
        return ggml_mul(g, y, w);
    };

    // ── Rotate-half RoPE (matches bidirlm_vision.cpp) ──
    // Standard neghalf: result = x * cos + rotate_half(x) * sin
    // where rotate_half(x) = [-x[half:], x[:half]]
    auto apply_rope = [&](ggml_tensor *t) -> ggml_tensor * {
        // t: (head_dim, n_heads, n_patches), contiguous
        // cos_in/sin_in: (head_dim, n_patches)
        int half = head_dim / 2;
        ggml_tensor *h1 = ggml_view_3d(g, t, half, n_heads, n_patches,
                                         t->nb[1], t->nb[2], 0);
        ggml_tensor *h2 = ggml_view_3d(g, t, half, n_heads, n_patches,
                                         t->nb[1], t->nb[2],
                                         (size_t)half * t->nb[0]);
        ggml_tensor *h2_neg = ggml_scale(g, ggml_cont(g, h2), -1.0f);
        ggml_tensor *rot = ggml_concat(g, h2_neg, ggml_cont(g, h1), 0);
        return ggml_add(g,
            ggml_mul(g, t, cos_in),
            ggml_mul(g, rot, sin_in));
    };

    // ── Norm helper (variant-aware) ──
    const bool is_qwen2 = vhp.is_qwen2_vl;
    auto vnorm = [&](ggml_tensor *t, ggml_tensor *w, ggml_tensor *b) -> ggml_tensor * {
        if (b) {
            // LayerNorm (Qwen2-VL)
            ggml_tensor *y = ggml_norm(g, t, 1e-6f);
            y = ggml_mul(g, y, w);
            return ggml_add(g, y, b);
        } else {
            // RMSNorm (Qwen2.5-VL)
            return rmsnorm(t, w);
        }
    };

    // ── ViT blocks ──
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    for (uint32_t il = 0; il < vhp.depth; il++) {
        const auto &blk = ctx.m.vis_blocks[il];
        ggml_tensor *residual = x;

        // Pre-attn norm (LayerNorm or RMSNorm)
        ggml_tensor *y = vnorm(x, blk.norm1_w, blk.norm1_b);

        // QKV: fused (CrispEmbed/Qwen2.5-VL) or separate (llama.cpp mmproj)
        ggml_tensor *Q, *K, *V;
        if (blk.qkv_w) {
            ggml_tensor *qkv = ggml_mul_mat(g, blk.qkv_w, y);
            if (blk.qkv_b) qkv = ggml_add(g, qkv, blk.qkv_b);
            qkv = ggml_reshape_4d(g, qkv, head_dim, n_heads, 3, n_patches);
            Q = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], 0 * qkv->nb[2]);
            K = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], 1 * qkv->nb[2]);
            V = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                             qkv->nb[1], qkv->nb[3], 2 * qkv->nb[2]);
        } else {
            // Separate Q/K/V (llama.cpp mmproj)
            Q = ggml_mul_mat(g, blk.q_w, y);
            if (blk.q_b) Q = ggml_add(g, Q, blk.q_b);
            K = ggml_mul_mat(g, blk.k_w, y);
            if (blk.k_b) K = ggml_add(g, K, blk.k_b);
            V = ggml_mul_mat(g, blk.v_w, y);
            if (blk.v_b) V = ggml_add(g, V, blk.v_b);
            Q = ggml_reshape_3d(g, Q, head_dim, n_heads, n_patches);
            K = ggml_reshape_3d(g, K, head_dim, n_heads, n_patches);
            V = ggml_reshape_3d(g, V, head_dim, n_heads, n_patches);
        }
        Q = ggml_cont(g, Q);
        K = ggml_cont(g, K);
        V = ggml_cont(g, V);

        // Apply 2D RoPE
        Q = apply_rope(Q);
        K = apply_rope(K);

        // Permute to (head_dim, n_patches, n_heads) for attention
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        // Attention
        ggml_tensor *scores = ggml_mul_mat(g, K, Q);
        scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor *attn_out = ggml_mul_mat(g, V_perm, scores);
        attn_out = ggml_cont(g, ggml_permute(g, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(g, attn_out, H, n_patches);

        attn_out = ggml_mul_mat(g, blk.proj_w, attn_out);
        if (blk.proj_b) attn_out = ggml_add(g, attn_out, blk.proj_b);

        x = ggml_add(g, residual, attn_out);

        // Pre-FFN norm (LayerNorm or RMSNorm)
        residual = x;
        y = vnorm(x, blk.norm2_w, blk.norm2_b);

        // FFN: variant-aware
        ggml_tensor *ffn;
        if (is_qwen2 && blk.ffn_fc1_w) {
            // Qwen2-VL: exact erf GELU fc1 → fc2 (nn.GELU(), not tanh approx)
            ffn = ggml_mul_mat(g, blk.ffn_fc1_w, y);
            if (blk.ffn_fc1_b) ffn = ggml_add(g, ffn, blk.ffn_fc1_b);
            ffn = ggml_gelu_erf(g, ffn);
            ffn = ggml_mul_mat(g, blk.ffn_fc2_w, ffn);
            if (blk.ffn_fc2_b) ffn = ggml_add(g, ffn, blk.ffn_fc2_b);
        } else {
            // Qwen2.5-VL: SwiGLU gate * silu(up) → down
            ggml_tensor *gate = ggml_mul_mat(g, blk.ffn_gate_w, y);
            if (blk.ffn_gate_b) gate = ggml_add(g, gate, blk.ffn_gate_b);
            gate = ggml_silu(g, gate);
            ggml_tensor *up = ggml_mul_mat(g, blk.ffn_up_w, y);
            if (blk.ffn_up_b) up = ggml_add(g, up, blk.ffn_up_b);
            ffn = ggml_mul(g, gate, up);
            ffn = ggml_mul_mat(g, blk.ffn_down_w, ffn);
            if (blk.ffn_down_b) ffn = ggml_add(g, ffn, blk.ffn_down_b);
        }

        x = ggml_add(g, residual, ffn);

        char name[64];
        std::snprintf(name, sizeof(name), "vis_layer_%u", il);
        ggml_set_name(x, name);
        if (!ctx.diff_ref_path.empty()) ggml_set_output(x);
    }

    // ── Merger (LayerNorm → spatial merge on CPU → FC1 → GELU → FC2) ──
    // The spatial merge permutation (2×2 patch grouping) is hard to express
    // in ggml's 4D tensor ops, so we:
    //   1. Compute LayerNorm in-graph
    //   2. Mark as output, compute graph
    //   3. Do spatial rearrangement on CPU
    //   4. Build a second graph for FC1→GELU→FC2
    {
        ggml_tensor *normed = x;
        if (ctx.m.merger.norm_w) {
            if (ctx.m.merger.norm_b) {
                // LayerNorm (has bias)
                normed = ggml_norm(g, x, 1e-6f);
                normed = ggml_mul(g, normed, ctx.m.merger.norm_w);
                normed = ggml_add(g, normed, ctx.m.merger.norm_b);
            } else {
                // RMSNorm (no bias) — Qwen2.5-VL uses this
                normed = ggml_rms_norm(g, x, 1e-6f);
                normed = ggml_mul(g, normed, ctx.m.merger.norm_w);
            }
        }

        ggml_set_name(normed, "vis_pre_merger");
        ggml_set_output(normed);
        ggml_build_forward_expand(gf, normed);
    }

    vision_graph_result out;
    out.gf = gf;
    out.output = x;
    out.h_patches = h_patches;
    out.w_patches = w_patches;
    return out;
}

}  // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────

bool load(context &ctx, const char *gguf_path, int n_threads, int verbosity,
          const char *mmproj_path) {
    ctx.n_threads = n_threads;
    ctx.verbosity = verbosity;

    if (verbosity >= 1) {
        fprintf(stderr, "qwen2vl_ocr: loading %s\n", gguf_path);
    }

    if (!load_hparams(ctx, gguf_path)) {
        fprintf(stderr, "qwen2vl_ocr: failed to load hparams\n");
        return false;
    }

    if (verbosity >= 1) {
        const auto &vhp = ctx.m.vhp;
        const auto &lhp = ctx.m.lhp;
        fprintf(stderr, "  vision: %u layers, %ud, %u heads, patch=%u, merge=%u\n",
                vhp.depth, vhp.hidden_size, vhp.num_heads,
                vhp.spatial_patch_size, vhp.spatial_merge_size);
        fprintf(stderr, "  llm: %u layers, %ud, %u/%u heads, inter=%u\n",
                lhp.num_hidden_layers, lhp.hidden_size,
                lhp.num_attention_heads, lhp.num_key_value_heads,
                lhp.intermediate_size);
    }

    // Init backend — prefer GPU when available
    bool force_cpu = (getenv("QWEN2VL_OCR_FORCE_CPU") && atoi(getenv("QWEN2VL_OCR_FORCE_CPU")));
    ctx.backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx.backend) ctx.backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx.backend))
        ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);

    // Compute-meta scratch: 32 layers × ~30 ops + merger.
    constexpr int kGraphCapacity = 32768;
    ctx.compute_meta.resize(
        ggml_tensor_overhead() * kGraphCapacity +
        ggml_graph_overhead_custom(kGraphCapacity, false));

    // Create scheduler
    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu && ctx.backend_cpu != ctx.backend) {
        backends.push_back(ctx.backend_cpu);
    }
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(),
                                       kGraphCapacity, false, false);

    if (!load_tensors(ctx, gguf_path)) {
        fprintf(stderr, "qwen2vl_ocr: failed to load tensors\n");
        return false;
    }

    // ── Load vision encoder from separate mmproj GGUF (llama.cpp split) ──
    if (mmproj_path) {
        gguf_context *mg = core_gguf::open_metadata(mmproj_path);
        if (mg) {
            auto mu = [&](const char *k, uint32_t d) {
                return core_gguf::kv_u32(mg, k, d);
            };
            ctx.m.vhp.depth               = mu("clip.vision.block_count", ctx.m.vhp.depth);
            ctx.m.vhp.hidden_size         = mu("clip.vision.embedding_length", ctx.m.vhp.hidden_size);
            ctx.m.vhp.intermediate_size   = mu("clip.vision.feed_forward_length", ctx.m.vhp.intermediate_size);
            ctx.m.vhp.num_heads           = mu("clip.vision.attention.head_count", ctx.m.vhp.num_heads);
            ctx.m.vhp.spatial_patch_size  = mu("clip.vision.patch_size", ctx.m.vhp.spatial_patch_size);
            // Read image mean/std
            int mi = gguf_find_key(mg, "clip.vision.image_mean");
            if (mi >= 0 && gguf_get_arr_n(mg, mi) >= 3) {
                auto *d = (const float *)gguf_get_arr_data(mg, mi);
                for (int i = 0; i < 3; i++) ctx.m.vhp.image_mean[i] = d[i];
            }
            int si = gguf_find_key(mg, "clip.vision.image_std");
            if (si >= 0 && gguf_get_arr_n(mg, si) >= 3) {
                auto *d = (const float *)gguf_get_arr_data(mg, si);
                for (int i = 0; i < 3; i++) ctx.m.vhp.image_std[i] = d[i];
            }
            core_gguf::free_metadata(mg);
            if (verbosity >= 1) {
                fprintf(stderr, "  mmproj: %u layers, %ud, %u heads, patch=%u\n",
                        ctx.m.vhp.depth, ctx.m.vhp.hidden_size,
                        ctx.m.vhp.num_heads, ctx.m.vhp.spatial_patch_size);
            }
        }

        core_gguf::WeightLoad vwl;
        if (!core_gguf::load_weights(mmproj_path, ctx.backend, "qwen2vl_mmproj", vwl)) {
            fprintf(stderr, "qwen2vl_ocr: failed to load mmproj tensors\n");
            return false;
        }

        auto vget = [&](const std::string &nm) -> ggml_tensor * {
            auto it = vwl.tensors.find(nm);
            return it != vwl.tensors.end() ? it->second : nullptr;
        };
        auto vget2 = [&](const std::string &a, const std::string &b) -> ggml_tensor * {
            auto *t = vget(a); return t ? t : vget(b);
        };

        // Patch embed (may be 4D — flatten if needed)
        ctx.m.patch_embed_w = vget2("v.patch_embd.weight", "v.patch_embed.weight");
        // 4D patch embed (14,14,3,1280) needs to be treated as 2D for mul_mat.
        // ggml will handle this if ne[0]*ne[1]*ne[2] matches the flat dim.
        // The tensor data is already flat in memory — just fix the shape metadata.
        if (ctx.m.patch_embed_w && ggml_n_dims(ctx.m.patch_embed_w) > 2) {
            int64_t flat = 1;
            for (int d = 0; d < ggml_n_dims(ctx.m.patch_embed_w) - 1; d++) {
                flat *= ctx.m.patch_embed_w->ne[d];
            }
            int64_t out_ch = ctx.m.patch_embed_w->ne[ggml_n_dims(ctx.m.patch_embed_w) - 1];
            // Reshape in-place: (flat, out_ch)
            ctx.m.patch_embed_w->ne[0] = flat;
            ctx.m.patch_embed_w->ne[1] = out_ch;
            ctx.m.patch_embed_w->ne[2] = 1;
            ctx.m.patch_embed_w->ne[3] = 1;
            ctx.m.patch_embed_w->nb[1] = flat * ggml_type_size(ctx.m.patch_embed_w->type);
            ctx.m.patch_embed_w->nb[2] = ctx.m.patch_embed_w->nb[1] * out_ch;
            ctx.m.patch_embed_w->nb[3] = ctx.m.patch_embed_w->nb[2];
            // Detect temporal_patch_size from flat dim:
            // Qwen2-VL: 14*14*3 = 588 (T=1, Conv2D)
            // Qwen2.5-VL: 14*14*3*2 = 1176 (T=2, Conv3D)
            int P = (int)ctx.m.vhp.spatial_patch_size;
            int C = 3;
            int expected_t1 = P * P * C;
            if ((int)flat == expected_t1) {
                ctx.m.vhp.temporal_patch_size = 1;
            }
            if (verbosity >= 1) {
                fprintf(stderr, "  mmproj: flattened patch_embed %lldx%lld (T=%u)\n",
                        (long long)flat, (long long)out_ch,
                        ctx.m.vhp.temporal_patch_size);
            }
        }

        // Vision blocks
        ctx.m.vis_blocks.resize(ctx.m.vhp.depth);
        for (uint32_t i = 0; i < ctx.m.vhp.depth; i++) {
            auto &blk = ctx.m.vis_blocks[i];
            std::string p = "v.blk." + std::to_string(i) + ".";
            blk.norm1_w    = vget(p + "ln1.weight");
            blk.norm1_b    = vget(p + "ln1.bias");
            blk.norm2_w    = vget(p + "ln2.weight");
            blk.norm2_b    = vget(p + "ln2.bias");
            blk.qkv_w      = vget(p + "attn_qkv.weight");
            blk.qkv_b      = vget(p + "attn_qkv.bias");
            blk.proj_w     = vget(p + "attn_out.weight");
            blk.proj_b     = vget(p + "attn_out.bias");
            // mmproj uses ffn_up/ffn_down (GELU, not SwiGLU)
            blk.ffn_fc1_w  = vget(p + "ffn_up.weight");
            blk.ffn_fc1_b  = vget(p + "ffn_up.bias");
            blk.ffn_fc2_w  = vget(p + "ffn_down.weight");
            blk.ffn_fc2_b  = vget(p + "ffn_down.bias");
            // Also try explicit attn_q/k/v (some mmproj variants)
            if (!blk.qkv_w) {
                // Fuse separate Q/K/V into qkv_w if all present
                // (our graph expects fused QKV — the parallel agent's code)
                // For now just store as-is; the graph handles both
                blk.qkv_w = vget(p + "attn_q.weight");  // will be null if separate
            }
        }
        // Detect Qwen2-VL from norm bias presence
        if (ctx.m.vhp.depth > 0 && ctx.m.vis_blocks[0].norm1_b) {
            ctx.m.vhp.is_qwen2_vl = true;
        }

        // Merger: mm.0 + mm.2
        ctx.m.merger.fc1_w = vget("mm.0.weight");
        ctx.m.merger.fc1_b = vget("mm.0.bias");
        ctx.m.merger.fc2_w = vget("mm.2.weight");
        ctx.m.merger.fc2_b = vget("mm.2.bias");

        // Keep mmproj buffer alive
        ctx.mmproj_ctx = vwl.ctx;
        ctx.mmproj_buf = vwl.buf;

        fprintf(stderr, "  mmproj: loaded %zu vision tensors (%s)\n",
                vwl.tensors.size(),
                ctx.m.vhp.is_qwen2_vl ? "Qwen2-VL" : "Qwen2.5-VL");
    }

    if (verbosity >= 1) {
        fprintf(stderr, "qwen2vl_ocr: loaded successfully\n");
    }

    return true;
}

void free_(context &ctx) {
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    if (ctx.mmproj_buf) {
        ggml_backend_buffer_free(ctx.mmproj_buf);
        ctx.mmproj_buf = nullptr;
    }
    if (ctx.mmproj_ctx) {
        ggml_free(ctx.mmproj_ctx);
        ctx.mmproj_ctx = nullptr;
    }
    if (ctx.model_buf) {
        ggml_backend_buffer_free(ctx.model_buf);
        ctx.model_buf = nullptr;
    }
    if (ctx.model_ctx) {
        ggml_free(ctx.model_ctx);
        ctx.model_ctx = nullptr;
    }
    if (ctx.backend_cpu) {
        ggml_backend_free(ctx.backend_cpu);
        ctx.backend_cpu = nullptr;
    }
    if (ctx.backend) {
        ggml_backend_free(ctx.backend);
        ctx.backend = nullptr;
    }
}

bool encode_vision(context &ctx,
                   const float *patches, int n_patches,
                   const int32_t *grid_thw,
                   vision_result &out) {
    // Debug: verify weight tensor is loaded correctly
    if (ctx.verbosity >= 2 && ctx.m.patch_embed_w) {
        float w5[5];
        ggml_backend_tensor_get(ctx.m.patch_embed_w, w5, 0, 5 * sizeof(float));
        fprintf(stderr, "  patch_embed_w ne=[%lld,%lld] buffer=%p data=%p first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                (long long)ctx.m.patch_embed_w->ne[0],
                (long long)ctx.m.patch_embed_w->ne[1],
                (void*)ctx.m.patch_embed_w->buffer,
                ctx.m.patch_embed_w->data,
                w5[0], w5[1], w5[2], w5[3], w5[4]);
    }

    // Compute 2D RoPE tables
    const int head_dim = (int)ctx.m.vhp.hidden_size / (int)ctx.m.vhp.num_heads;
    host_rope rope;
    compute_vision_rope(rope, grid_thw, n_patches, head_dim);

    // Build graph
    auto gr = build_vision_graph(ctx, n_patches, grid_thw);
    if (!gr.gf) return false;

    // Allocate via scheduler (handles model weights + compute buffers)
    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gr.gf)) {
        fprintf(stderr, "qwen2vl_ocr: graph allocation failed\n");
        return false;
    }

    // Set input tensors
    const int patch_flat_dim = (int)ctx.m.vhp.in_channels *
                               (int)ctx.m.vhp.temporal_patch_size *
                               (int)ctx.m.vhp.spatial_patch_size *
                               (int)ctx.m.vhp.spatial_patch_size;

    auto set_in = [&](const char *name, const void *data, size_t bytes) -> bool {
        ggml_tensor *t = ggml_graph_get_tensor(gr.gf, name);
        if (!t) {
            fprintf(stderr, "qwen2vl_ocr: input tensor '%s' not found\n", name);
            return false;
        }
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_in("pixel_in", patches,
                (size_t)n_patches * patch_flat_dim * sizeof(float)))
        return false;
    if (!set_in("cos_in", rope.cos_buf.data(),
                rope.cos_buf.size() * sizeof(float)))
        return false;
    if (!set_in("sin_in", rope.sin_buf.data(),
                rope.sin_buf.size() * sizeof(float)))
        return false;

    // Compute
    if (ggml_backend_sched_graph_compute(ctx.sched, gr.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen2vl_ocr: graph compute failed\n");
        return false;
    }

    // Read pre-merger output (normed ViT features)
    ggml_tensor *pre_merger = ggml_graph_get_tensor(gr.gf, "vis_pre_merger");
    if (!pre_merger) {
        fprintf(stderr, "qwen2vl_ocr: vis_pre_merger not found\n");
        return false;
    }

    const int H = (int)ctx.m.vhp.hidden_size;
    const int merge = (int)ctx.m.vhp.spatial_merge_size;
    const int h_p = gr.h_patches;
    const int w_p = gr.w_patches;
    const int merged_h = h_p / merge;
    const int merged_w = w_p / merge;
    const int n_merged = merged_h * merged_w;
    const int merger_in_dim = H * merge * merge;

    // Read normed features: ne=(H, n_patches) in column-major
    // Flat layout: [dim0_patch0, dim1_patch0, ..., dimH_patch0, dim0_patch1, ...]
    std::vector<float> normed_data((size_t)n_patches * H);
    ggml_backend_tensor_get(pre_merger, normed_data.data(), 0,
                            normed_data.size() * sizeof(float));

    // CPU spatial merge: group 2×2 patches, concatenate features
    // normed_data layout: patch_idx * H + dim_idx (column-major ggml)
    // Patch index = row * w_p + col
    std::vector<float> merged_data((size_t)n_merged * merger_in_dim);

    int midx = 0;
    for (int mh = 0; mh < merged_h; mh++) {
        for (int mw = 0; mw < merged_w; mw++) {
            float *dst = merged_data.data() + (size_t)midx * merger_in_dim;
            int off = 0;
            for (int ir = 0; ir < merge; ir++) {
                for (int ic = 0; ic < merge; ic++) {
                    int row = mh * merge + ir;
                    int col = mw * merge + ic;
                    int patch_idx = row * w_p + col;
                    const float *src = normed_data.data() + (size_t)patch_idx * H;
                    std::memcpy(dst + off, src, H * sizeof(float));
                    off += H;
                }
            }
            midx++;
        }
    }

    if (ctx.verbosity >= 2) {
        fprintf(stderr, "  merger: %d merged tokens, in_dim=%d, first5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                n_merged, merger_in_dim,
                merged_data[0], merged_data[1], merged_data[2],
                merged_data[3], merged_data[4]);
    }

    // FC1 → GELU → FC2 via a second ggml graph
    {
        ggml_init_params ip2{
            ctx.compute_meta.size(),
            ctx.compute_meta.data(),
            true,
        };
        ggml_context *g2 = ggml_init(ip2);
        ggml_cgraph *gf2 = ggml_new_graph(g2);

        ggml_tensor *merged_in = ggml_new_tensor_2d(g2, GGML_TYPE_F32,
                                                      merger_in_dim, n_merged);
        ggml_set_name(merged_in, "merged_in");
        ggml_set_input(merged_in);

        ggml_tensor *m = ggml_mul_mat(g2, ctx.m.merger.fc1_w, merged_in);
        if (ctx.m.merger.fc1_b) m = ggml_add(g2, m, ctx.m.merger.fc1_b);
        m = ggml_gelu_erf(g2, m);  // Qwen2.5-VL merger uses exact GELU (nn.GELU())
        m = ggml_mul_mat(g2, ctx.m.merger.fc2_w, m);
        if (ctx.m.merger.fc2_b) m = ggml_add(g2, m, ctx.m.merger.fc2_b);

        ggml_set_name(m, "merger_output");
        ggml_set_output(m);
        ggml_build_forward_expand(gf2, m);

        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, gf2)) {
            fprintf(stderr, "qwen2vl_ocr: merger graph allocation failed\n");
            ggml_free(g2);
            return false;
        }

        ggml_tensor *min_t = ggml_graph_get_tensor(gf2, "merged_in");
        ggml_backend_tensor_set(min_t, merged_data.data(), 0,
                                merged_data.size() * sizeof(float));

        if (ggml_backend_sched_graph_compute(ctx.sched, gf2) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "qwen2vl_ocr: merger graph compute failed\n");
            ggml_free(g2);
            return false;
        }

        ggml_tensor *mout = ggml_graph_get_tensor(gf2, "merger_output");
        int out_dim = (int)ctx.m.vhp.out_hidden_size;
        out.n_merged = n_merged;
        out.embed_dim = out_dim;
        out.image_embeds = (float *)malloc((size_t)n_merged * out_dim * sizeof(float));
        ggml_backend_tensor_get(mout, out.image_embeds, 0,
                                (size_t)n_merged * out_dim * sizeof(float));

        ggml_free(g2);
    }

    // Diff comparison if enabled
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path)) {
            // Compare patch embedding first
            ggml_tensor *pe_out = ggml_graph_get_tensor(gr.gf, "patch_embed_out");
            if (pe_out && ref.has("vis_patch_embed")) {
                std::vector<float> pe_data((size_t)n_patches * H);
                ggml_backend_tensor_get(pe_out, pe_data.data(), 0,
                                        pe_data.size() * sizeof(float));
                auto r = ref.compare("vis_patch_embed", pe_data.data(),
                                     pe_data.size());
                fprintf(stderr, "  diff vis_patch_embed: cos_min=%.6f max_abs=%.2e %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                // Print first 5 values from C++ and reference
                fprintf(stderr, "    C++: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                        pe_data[0], pe_data[1], pe_data[2], pe_data[3], pe_data[4]);
                auto [ref_pe, ref_n] = ref.get_f32("vis_patch_embed");
                if (ref_pe) {
                    fprintf(stderr, "    Ref: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                            ref_pe[0], ref_pe[1], ref_pe[2], ref_pe[3], ref_pe[4]);
                }
            }

            // Compare per-layer
            for (uint32_t il = 0; il < ctx.m.vhp.depth; il++) {
                char name[64];
                std::snprintf(name, sizeof(name), "vis_layer_%u", il);
                ggml_tensor *layer_out = ggml_graph_get_tensor(gr.gf, name);
                if (layer_out && ref.has(name)) {
                    std::vector<float> layer_data((size_t)n_patches * H);
                    ggml_backend_tensor_get(layer_out, layer_data.data(), 0,
                                            layer_data.size() * sizeof(float));
                    auto r = ref.compare(name, layer_data.data(),
                                         layer_data.size());
                    fprintf(stderr, "  diff %s: cos_min=%.6f max_abs=%.2e %s\n",
                            name, r.cos_min, r.max_abs,
                            r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }
    }

    return true;
}

void vision_result_free(vision_result &r) {
    if (r.image_embeds) {
        free(r.image_embeds);
        r.image_embeds = nullptr;
    }
}

// ── LLM decoder forward pass (text-only, no KV cache) ───────────────

bool run_llm_forward(context &ctx,
                     const int32_t *token_ids, int n_tokens,
                     llm_result &out,
                     const image_input *img) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int n_heads = (int)lhp.num_attention_heads;
    const int n_kv_heads = (int)lhp.num_key_value_heads;
    const int head_dim = D / n_heads;
    const int n_layers = (int)lhp.num_hidden_layers;
    const float rms_eps = lhp.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    const int kv_repeat = n_heads / n_kv_heads;

    ggml_init_params ip{
        ctx.compute_meta.size(),
        ctx.compute_meta.data(),
        true,
    };
    ggml_context *g = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Token IDs input
    ggml_tensor *ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);

    // Token embedding lookup
    ggml_tensor *x = ggml_get_rows(g, ctx.m.embed_tokens, ids);
    // x: ne=(D, n_tokens)

    // Vision-text splicing: replace image_pad token embeddings with
    // image embeddings from the vision encoder
    bool has_image = (img && img->image_embeds && img->n_image_tokens > 0);
    if (has_image) {
        // keep_mask: (1, n_tokens) — 1.0 for text, 0.0 for image positions
        ggml_tensor *keep_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, n_tokens);
        ggml_set_name(keep_mask, "keep_mask");
        ggml_set_input(keep_mask);

        // image_patches: (D, n_tokens) — image embeds at image positions, 0 elsewhere
        ggml_tensor *image_patches = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_tokens);
        ggml_set_name(image_patches, "image_patches");
        ggml_set_input(image_patches);

        // x = x * keep_mask + image_patches
        x = ggml_add(g, ggml_mul(g, x, keep_mask), image_patches);
    }

    ggml_set_name(x, "llm_embed");
    if (!ctx.diff_ref_path.empty()) ggml_set_output(x);

    // Causal mask: (n_tokens, n_tokens) with -inf for future positions
    ggml_tensor *causal_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32,
                                                    n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // mRoPE position IDs: (n_tokens * 4) I32
    // Layout: [t0..t_{n-1}, h0..h_{n-1}, w0..w_{n-1}, 0..0]
    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens * 4);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // mRoPE sections
    int sections[4] = {
        lhp.rope_sections[0],  // 16 (temporal)
        lhp.rope_sections[1],  // 24 (height)
        lhp.rope_sections[2],  // 24 (width)
        lhp.rope_sections[3],  // 0
    };

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        ggml_tensor *y = ggml_rms_norm(g, t, rms_eps);
        return ggml_mul(g, y, w);
    };

    // Decoder layers
    for (int il = 0; il < n_layers; il++) {
        const auto &ly = ctx.m.llm_layers[il];
        ggml_tensor *residual = x;

        // Pre-attn RMSNorm
        ggml_tensor *normed = rmsnorm(x, ly.attn_norm_w);

        // Q/K/V projections (separate, not fused)
        ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, normed);
        if (ly.q_b) Q = ggml_add(g, Q, ly.q_b);
        ggml_tensor *K = ggml_mul_mat(g, ly.k_w, normed);
        if (ly.k_b) K = ggml_add(g, K, ly.k_b);
        ggml_tensor *V = ggml_mul_mat(g, ly.v_w, normed);
        if (ly.v_b) V = ggml_add(g, V, ly.v_b);

        // Reshape for multi-head: Q (head_dim, n_heads, T), K/V (head_dim, n_kv, T)
        Q = ggml_reshape_3d(g, Q, head_dim, n_heads, n_tokens);
        K = ggml_reshape_3d(g, K, head_dim, n_kv_heads, n_tokens);
        V = ggml_reshape_3d(g, V, head_dim, n_kv_heads, n_tokens);

        // Qwen3-VL: per-head QK RMSNorm (applied after reshape, before RoPE)
        if (ly.q_norm_w) {
            Q = ggml_rms_norm(g, Q, lhp.rms_norm_eps);
            Q = ggml_mul(g, Q, ly.q_norm_w);
        }
        if (ly.k_norm_w) {
            K = ggml_rms_norm(g, K, lhp.rms_norm_eps);
            K = ggml_mul(g, K, ly.k_norm_w);
        }

        // Apply mRoPE (multi-dimensional rotary position embedding)
        Q = ggml_rope_multi(g, Q, pos_ids, nullptr,
                            head_dim, sections,
                            GGML_ROPE_TYPE_MROPE,
                            0,  // n_ctx_orig
                            lhp.rope_theta,
                            1.0f,  // freq_scale
                            0.0f,  // ext_factor
                            1.0f,  // attn_factor
                            0.0f,  // beta_fast
                            0.0f); // beta_slow
        K = ggml_rope_multi(g, K, pos_ids, nullptr,
                            head_dim, sections,
                            GGML_ROPE_TYPE_MROPE,
                            0,
                            lhp.rope_theta,
                            1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Output post-RoPE K/V for KV cache extraction
        // Shape: (head_dim, n_kv_heads, n_tokens) → flatten to (kv_dim, n_tokens)
        {
            char kname[64], vname[64];
            std::snprintf(kname, sizeof(kname), "k_out_%d", il);
            std::snprintf(vname, sizeof(vname), "v_out_%d", il);
            ggml_tensor *K_flat = ggml_reshape_2d(g, ggml_cont(g, K),
                                                    head_dim * n_kv_heads, n_tokens);
            ggml_tensor *V_flat = ggml_reshape_2d(g, ggml_cont(g, V),
                                                    head_dim * n_kv_heads, n_tokens);
            ggml_set_name(K_flat, kname);
            ggml_set_name(V_flat, vname);
            ggml_set_output(K_flat);
            ggml_set_output(V_flat);
        }

        // GQA: interleave KV heads to match Q heads
        // K: (head_dim, n_kv, T) → (head_dim, n_heads, T)
        // Each KV head serves kv_repeat Q heads (interleave, not tile)
        // ggml_repeat tiles [0,1,0,1,...] but we need [0,0,...,1,1,...]
        // Use reshape + repeat on the right axis:
        // K(hd, n_kv, T) → K(hd, 1, n_kv, T) → repeat → K(hd, kv_repeat, n_kv, T)
        // → reshape → K(hd, n_heads, T)
        if (kv_repeat > 1) {
            // Reshape to add repeat dimension
            K = ggml_reshape_4d(g, K, head_dim, 1, n_kv_heads, n_tokens);
            V = ggml_reshape_4d(g, V, head_dim, 1, n_kv_heads, n_tokens);
            // Create target shape for repeat
            ggml_tensor *K_tgt = ggml_new_tensor_4d(g, K->type, head_dim, kv_repeat, n_kv_heads, n_tokens);
            ggml_tensor *V_tgt = ggml_new_tensor_4d(g, V->type, head_dim, kv_repeat, n_kv_heads, n_tokens);
            K = ggml_repeat(g, K, K_tgt);
            V = ggml_repeat(g, V, V_tgt);
            // Reshape back to (head_dim, n_heads, T)
            K = ggml_reshape_3d(g, K, head_dim, n_heads, n_tokens);
            V = ggml_reshape_3d(g, V, head_dim, n_heads, n_tokens);
        }

        // Attention (same pattern as vision encoder)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));  // (hd, T, nh)
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        ggml_tensor *scores = ggml_mul_mat(g, K, Q);  // (T, T, nh)
        scores = ggml_add(g, scores, causal_mask);
        scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor *attn_out = ggml_mul_mat(g, V_perm, scores);
        attn_out = ggml_cont(g, ggml_permute(g, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(g, attn_out, D, n_tokens);

        // Output projection
        attn_out = ggml_mul_mat(g, ly.o_w, attn_out);
        x = ggml_add(g, residual, attn_out);

        // Pre-FFN RMSNorm
        residual = x;
        normed = rmsnorm(x, ly.ffn_norm_w);

        // SwiGLU FFN
        ggml_tensor *gate = ggml_mul_mat(g, ly.ffn_gate_w, normed);
        gate = ggml_silu(g, gate);
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, normed);
        ggml_tensor *ffn = ggml_mul(g, gate, up);
        ffn = ggml_mul_mat(g, ly.ffn_down_w, ffn);
        x = ggml_add(g, residual, ffn);

        char name[64];
        std::snprintf(name, sizeof(name), "llm_layer_%d", il);
        ggml_set_name(x, name);
        if (!ctx.diff_ref_path.empty()) ggml_set_output(x);
    }

    // Final RMSNorm
    if (ctx.m.output_norm_w) {
        x = rmsnorm(x, ctx.m.output_norm_w);
        ggml_set_name(x, "llm_final_norm");
        if (!ctx.diff_ref_path.empty()) ggml_set_output(x);
    }

    // LM head (logits) — compute in graph to handle quantized weights
    ggml_tensor *lm_head = ctx.m.lm_head_w;
    if (!lm_head && lhp.tie_word_embeddings && ctx.m.embed_tokens) {
        lm_head = ctx.m.embed_tokens;
    }
    if (lm_head) {
        ggml_tensor *logits = ggml_mul_mat(g, lm_head, x);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        x = logits;
    }

    ggml_build_forward_expand(gf, x);

    // Allocate and compute
    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "qwen2vl_ocr: LLM graph allocation failed\n");
        ggml_free(g);
        return false;
    }

    // Set inputs
    ggml_tensor *ids_t = ggml_graph_get_tensor(gf, "token_ids");
    ggml_backend_tensor_set(ids_t, token_ids, 0, n_tokens * sizeof(int32_t));

    // Build causal mask on CPU
    std::vector<float> mask_data((size_t)n_tokens * n_tokens, -INFINITY);
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j <= i; j++) {
            mask_data[(size_t)i * n_tokens + j] = 0.0f;
        }
    }
    ggml_tensor *mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_t, mask_data.data(), 0,
                            mask_data.size() * sizeof(float));

    // Build mRoPE position IDs and image splicing data
    std::vector<int32_t> pos_data(n_tokens * 4, 0);
    const int img_tok_id = (int)lhp.image_token_id;
    const int spatial_merge = (int)ctx.m.vhp.spatial_merge_size;

    if (has_image) {
        // Build keep_mask and image_patches
        std::vector<float> keep_data(n_tokens, 1.0f);
        std::vector<float> patch_data((size_t)n_tokens * D, 0.0f);

        int img_idx = 0;
        for (int t = 0; t < n_tokens && img_idx < img->n_image_tokens; t++) {
            if (token_ids[t] != img_tok_id) continue;
            keep_data[t] = 0.0f;
            std::memcpy(patch_data.data() + (size_t)t * D,
                        img->image_embeds + (size_t)img_idx * D,
                        (size_t)D * sizeof(float));
            img_idx++;
        }

        ggml_tensor *km = ggml_graph_get_tensor(gf, "keep_mask");
        ggml_backend_tensor_set(km, keep_data.data(), 0,
                                keep_data.size() * sizeof(float));
        ggml_tensor *ip = ggml_graph_get_tensor(gf, "image_patches");
        ggml_backend_tensor_set(ip, patch_data.data(), 0,
                                patch_data.size() * sizeof(float));

        // Build mRoPE positions with image awareness
        // Text tokens: pos_t = pos_h = pos_w = sequential
        // Image tokens: pos_t = 0, pos_h = row, pos_w = col (within merged grid)
        int last_max = -1;
        int img_consumed = 0;
        int st = 0;
        while (st < n_tokens) {
            // Find next image_pad token
            int ed = n_tokens;
            if (img_consumed < img->n_images) {
                for (int p = st; p < n_tokens; p++) {
                    if (token_ids[p] == img_tok_id) { ed = p; break; }
                }
            }

            // Text segment [st, ed)
            int text_len = ed - st;
            int st_idx = last_max + 1;
            for (int i = 0; i < text_len; i++) {
                pos_data[st + i]                = st_idx + i;
                pos_data[n_tokens + st + i]     = st_idx + i;
                pos_data[2 * n_tokens + st + i] = st_idx + i;
            }
            if (text_len > 0) last_max = st_idx + text_len - 1;

            if (ed >= n_tokens) break;

            // Image segment: count consecutive image_pad tokens
            int img_start = ed;
            while (ed < n_tokens && token_ids[ed] == img_tok_id) ed++;
            int n_img_tokens = ed - img_start;

            // Get grid for this image
            int grid_t = img->grid_thw[img_consumed * 3 + 0];
            int grid_h = img->grid_thw[img_consumed * 3 + 1] / spatial_merge;
            int grid_w = img->grid_thw[img_consumed * 3 + 2] / spatial_merge;
            int t_idx = last_max + 1;

            int tok = 0;
            for (int ft = 0; ft < grid_t && tok < n_img_tokens; ft++) {
                for (int fh = 0; fh < grid_h && tok < n_img_tokens; fh++) {
                    for (int fw = 0; fw < grid_w && tok < n_img_tokens; fw++) {
                        int pos = img_start + tok;
                        pos_data[pos]                = t_idx;
                        pos_data[n_tokens + pos]     = fh;
                        pos_data[2 * n_tokens + pos] = fw;
                        tok++;
                    }
                }
            }

            last_max = t_idx + std::max(grid_h, grid_w) - 1;
            img_consumed++;
            st = ed;
        }
    } else {
        // Text-only: all 3 dims = sequential position
        for (int i = 0; i < n_tokens; i++) {
            pos_data[i]                = i;
            pos_data[n_tokens + i]     = i;
            pos_data[2 * n_tokens + i] = i;
        }
    }

    ggml_tensor *pos_t_tensor = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(pos_t_tensor, pos_data.data(), 0,
                            pos_data.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "qwen2vl_ocr: LLM graph compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read outputs
    out.n_tokens = n_tokens;
    out.hidden_dim = D;

    // Read logits if available (for generation)
    ggml_tensor *logits_t = ggml_graph_get_tensor(gf, "logits");
    if (logits_t) {
        int V = (int)logits_t->ne[0];
        out.vocab_size = V;
        out.logits = (float *)malloc((size_t)n_tokens * V * sizeof(float));
        ggml_backend_tensor_get(logits_t, out.logits, 0,
                                (size_t)n_tokens * V * sizeof(float));
    }

    // Read hidden states
    out.hidden = (float *)malloc((size_t)n_tokens * D * sizeof(float));
    ggml_tensor *final_out = ggml_graph_get_tensor(gf, "llm_final_norm");
    if (final_out) {
        ggml_backend_tensor_get(final_out, out.hidden, 0,
                                (size_t)n_tokens * D * sizeof(float));
    }

    // Diff comparison
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path)) {
            // Compare embed
            ggml_tensor *emb = ggml_graph_get_tensor(gf, "llm_embed");
            if (emb && ref.has("llm_embed")) {
                std::vector<float> emb_data((size_t)n_tokens * D);
                ggml_backend_tensor_get(emb, emb_data.data(), 0,
                                        emb_data.size() * sizeof(float));
                auto r = ref.compare("llm_embed", emb_data.data(), emb_data.size());
                fprintf(stderr, "  diff llm_embed: cos_min=%.6f max_abs=%.2e %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }

            // Compare per-layer
            for (int il = 0; il < n_layers; il++) {
                char name[64];
                std::snprintf(name, sizeof(name), "llm_layer_%d", il);
                ggml_tensor *lt = ggml_graph_get_tensor(gf, name);
                if (lt && ref.has(name)) {
                    std::vector<float> ld((size_t)n_tokens * D);
                    ggml_backend_tensor_get(lt, ld.data(), 0, ld.size() * sizeof(float));
                    auto r = ref.compare(name, ld.data(), ld.size());
                    fprintf(stderr, "  diff %s: cos_min=%.6f max_abs=%.2e %s\n",
                            name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                    if (!r.is_pass()) {
                        auto [rd, rn] = ref.get_f32(name);
                        fprintf(stderr, "    C++: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                                ld[0], ld[1], ld[2], ld[3], ld[4]);
                        fprintf(stderr, "    Ref: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                                rd[0], rd[1], rd[2], rd[3], rd[4]);
                    }
                }
            }

            // Compare final norm
            if (ref.has("llm_final_norm") && out.hidden) {
                auto r = ref.compare("llm_final_norm", out.hidden,
                                     (size_t)n_tokens * D);
                fprintf(stderr, "  diff llm_final_norm: cos_min=%.6f max_abs=%.2e %s\n",
                        r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
            }
        }
    }

    // Keep graph alive for KV cache extraction by generate()
    out.kv_graph = gf;
    out.kv_graph_ctx = g;
    // Caller must ggml_free(out.kv_graph_ctx) after extracting KV

    return true;
}

// ── Decode-step graph (single token, with KV cache) ──────────────────

static ggml_cgraph * build_decode_step_graph(
        context &ctx, ggml_context *g,
        int n_kv,       // number of cached KV tokens (0 on first step)
        int pos) {      // position of the new token

    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int n_heads = (int)lhp.num_attention_heads;
    const int n_kv_heads = (int)lhp.num_key_value_heads;
    const int head_dim = D / n_heads;
    const int n_layers = (int)lhp.num_hidden_layers;
    const float rms_eps = lhp.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);
    const int kv_repeat = n_heads / n_kv_heads;
    const int kv_dim = head_dim * n_kv_heads;  // KV projection dim

    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Input: token embedding (pre-computed, D × 1)
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, 1);
    ggml_set_name(x, "tok_emb");
    ggml_set_input(x);

    // mRoPE position IDs for this single token (4 values: t, h, w, 0)
    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, 4);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    int sections[4] = {
        lhp.rope_sections[0], lhp.rope_sections[1],
        lhp.rope_sections[2], lhp.rope_sections[3],
    };

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, rms_eps), w);
    };

    char name[64];
    for (int il = 0; il < n_layers; il++) {
        const auto &ly = ctx.m.llm_layers[il];
        ggml_tensor *residual = x;

        ggml_tensor *normed = rmsnorm(x, ly.attn_norm_w);

        // Q/K/V for the single new token
        ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, normed);
        if (ly.q_b) Q = ggml_add(g, Q, ly.q_b);
        ggml_tensor *K_new = ggml_mul_mat(g, ly.k_w, normed);
        if (ly.k_b) K_new = ggml_add(g, K_new, ly.k_b);
        ggml_tensor *V_new = ggml_mul_mat(g, ly.v_w, normed);
        if (ly.v_b) V_new = ggml_add(g, V_new, ly.v_b);

        // Reshape: Q (head_dim, n_heads, 1), K/V (head_dim, n_kv_heads, 1)
        Q = ggml_reshape_3d(g, Q, head_dim, n_heads, 1);
        K_new = ggml_reshape_3d(g, K_new, head_dim, n_kv_heads, 1);
        V_new = ggml_reshape_3d(g, V_new, head_dim, n_kv_heads, 1);

        // Apply mRoPE to Q and K
        Q = ggml_rope_multi(g, Q, pos_ids, nullptr,
                            head_dim, sections, GGML_ROPE_TYPE_MROPE,
                            0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_multi(g, K_new, pos_ids, nullptr,
                                head_dim, sections, GGML_ROPE_TYPE_MROPE,
                                0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Output new K/V for cache append
        std::snprintf(name, sizeof(name), "k_out_%d", il);
        ggml_set_name(K_new, name);
        ggml_set_output(K_new);
        std::snprintf(name, sizeof(name), "v_out_%d", il);
        ggml_set_name(V_new, name);
        ggml_set_output(V_new);

        // Load KV cache and concatenate
        ggml_tensor *K_full, *V_full;
        if (n_kv > 0) {
            ggml_tensor *k_cache = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, n_kv);
            std::snprintf(name, sizeof(name), "k_in_%d", il);
            ggml_set_name(k_cache, name);
            ggml_set_input(k_cache);

            ggml_tensor *v_cache = ggml_new_tensor_2d(g, GGML_TYPE_F32, kv_dim, n_kv);
            std::snprintf(name, sizeof(name), "v_in_%d", il);
            ggml_set_name(v_cache, name);
            ggml_set_input(v_cache);

            // Reshape cache to 3D for concat
            k_cache = ggml_reshape_3d(g, k_cache, head_dim, n_kv_heads, n_kv);
            v_cache = ggml_reshape_3d(g, v_cache, head_dim, n_kv_heads, n_kv);

            K_full = ggml_concat(g, k_cache, K_new, 2);  // concat on seq dim
            V_full = ggml_concat(g, v_cache, V_new, 2);
        } else {
            K_full = K_new;
            V_full = V_new;
        }

        // GQA: repeat KV heads
        if (kv_repeat > 1) {
            int seq_len = n_kv + 1;
            K_full = ggml_reshape_4d(g, K_full, head_dim, 1, n_kv_heads, seq_len);
            ggml_tensor *K_tgt = ggml_new_tensor_4d(g, K_full->type, head_dim, kv_repeat, n_kv_heads, seq_len);
            K_full = ggml_repeat(g, K_full, K_tgt);
            K_full = ggml_reshape_3d(g, K_full, head_dim, n_heads, seq_len);

            V_full = ggml_reshape_4d(g, V_full, head_dim, 1, n_kv_heads, seq_len);
            ggml_tensor *V_tgt = ggml_new_tensor_4d(g, V_full->type, head_dim, kv_repeat, n_kv_heads, seq_len);
            V_full = ggml_repeat(g, V_full, V_tgt);
            V_full = ggml_reshape_3d(g, V_full, head_dim, n_heads, seq_len);
        }

        // Attention: Q(hd, nh, 1) @ K_full^T(hd, nh, seq) → scores(seq, 1, nh)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));  // (hd, 1, nh)
        K_full = ggml_cont(g, ggml_permute(g, K_full, 0, 2, 1, 3)); // (hd, seq, nh)
        V_full = ggml_cont(g, ggml_permute(g, V_full, 0, 2, 1, 3));

        ggml_tensor *scores = ggml_mul_mat(g, K_full, Q); // (seq, 1, nh)
        // No causal mask needed — single query always attends to all cached KV
        scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor *V_perm = ggml_cont(g, ggml_permute(g, V_full, 1, 0, 2, 3));
        ggml_tensor *attn_out = ggml_mul_mat(g, V_perm, scores); // (hd, 1, nh)
        attn_out = ggml_cont(g, ggml_permute(g, attn_out, 0, 2, 1, 3)); // (hd, nh, 1)
        attn_out = ggml_reshape_2d(g, attn_out, D, 1);

        attn_out = ggml_mul_mat(g, ly.o_w, attn_out);
        x = ggml_add(g, residual, attn_out);

        // FFN
        residual = x;
        normed = rmsnorm(x, ly.ffn_norm_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, normed));
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, normed);
        ggml_tensor *ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
        x = ggml_add(g, residual, ffn);
    }

    // Final norm + logits
    if (ctx.m.output_norm_w) {
        x = rmsnorm(x, ctx.m.output_norm_w);
    }
    ggml_tensor *lm_head = ctx.m.lm_head_w;
    if (!lm_head && lhp.tie_word_embeddings && ctx.m.embed_tokens) {
        lm_head = ctx.m.embed_tokens;
    }
    if (lm_head) {
        x = ggml_mul_mat(g, lm_head, x);
    }
    ggml_set_name(x, "logits");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    return gf;
}

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *grid_thw,  // actual image grid (t,h,w) for mRoPE
              const int32_t *prompt_token_ids, int n_prompt_tokens,
              int max_new_tokens,
              generate_result &out) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int V = (int)lhp.vocab_size;
    const int n_layers = (int)lhp.num_hidden_layers;
    const int n_kv_heads = (int)lhp.num_key_value_heads;
    const int head_dim = D / (int)lhp.num_attention_heads;
    const int kv_dim = head_dim * n_kv_heads;

    // ── Step 1: Prefill — full forward pass to get logits + KV cache ──
    image_input img_in = {};
    if (image_embeds && n_image_tokens > 0) {
        img_in.image_embeds = image_embeds;
        img_in.n_image_tokens = n_image_tokens;
        img_in.grid_thw = grid_thw;  // actual spatial grid for mRoPE
        img_in.n_images = 1;
    }

    // ── Prefill: full forward pass, extract KV cache ──
    llm_result prefill = {};
    bool ok = run_llm_forward(ctx, prompt_token_ids, n_prompt_tokens,
                               prefill, (img_in.image_embeds) ? &img_in : nullptr);
    if (!ok || !prefill.logits) {
        if (prefill.hidden) free(prefill.hidden);
        if (prefill.logits) free(prefill.logits);
        fprintf(stderr, "qwen2vl_ocr: prefill failed\n");
        return false;
    }

    // Extract per-layer KV cache from prefill graph outputs
    // k_out_N / v_out_N tensors: (kv_dim, n_prompt_tokens) each
    std::vector<std::vector<float>> k_cache(n_layers);
    std::vector<std::vector<float>> v_cache(n_layers);
    bool kv_ok = (prefill.kv_graph != nullptr);

    if (kv_ok) {
        for (int il = 0; il < n_layers; il++) {
            char name[64];
            std::snprintf(name, sizeof(name), "k_out_%d", il);
            ggml_tensor *kt = ggml_graph_get_tensor(prefill.kv_graph, name);
            std::snprintf(name, sizeof(name), "v_out_%d", il);
            ggml_tensor *vt = ggml_graph_get_tensor(prefill.kv_graph, name);

            if (!kt || !vt) { kv_ok = false; break; }

            size_t sz = (size_t)kv_dim * n_prompt_tokens;
            k_cache[il].resize(sz);
            v_cache[il].resize(sz);
            ggml_backend_tensor_get(kt, k_cache[il].data(), 0, sz * sizeof(float));
            ggml_backend_tensor_get(vt, v_cache[il].data(), 0, sz * sizeof(float));
        }
    }

    if (kv_ok && ctx.verbosity >= 1) {
        fprintf(stderr, "  KV cache: %d layers × %d tokens × %d dim = %.1f MB\n",
                n_layers, n_prompt_tokens, kv_dim,
                (float)n_layers * n_prompt_tokens * kv_dim * 2 * sizeof(float) / (1024*1024));
    }

    // Greedy: argmax prefill logits at last position
    out.token_confidences.clear();
    const float *last_logits = prefill.logits + (size_t)(n_prompt_tokens - 1) * V;
    int best_id = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++) {
        if (last_logits[v] > best_score) {
            best_score = last_logits[v];
            best_id = v;
        }
    }

    // Confidence: numerically-stable softmax for winning token
    {
        float max_l = best_score;
        float sum_exp = 0.0f;
        for (int v = 0; v < V; v++) sum_exp += expf(last_logits[v] - max_l);
        out.token_confidences.push_back(expf(best_score - max_l) / sum_exp);
    }

    if (prefill.hidden) free(prefill.hidden);
    free(prefill.logits);
    // Free prefill graph context (KV already extracted to k_cache/v_cache)
    if (prefill.kv_graph_ctx) ggml_free(prefill.kv_graph_ctx);

    out.token_ids.push_back(best_id);
    if (ctx.verbosity >= 1) {
        fprintf(stderr, "  gen[0]: token=%d score=%.2f (prefill)\n", best_id, best_score);
    }

    int eos_id = 151645;
    if (best_id == eos_id || max_new_tokens <= 1) return true;

    // ── Decode: single-token steps with KV cache ──
    int n_kv = n_prompt_tokens;  // KV cache size grows each step

    for (int gen = 1; gen < max_new_tokens; gen++) {
        if (!kv_ok) {
            // Fallback: recompute full sequence (no KV cache)
            std::vector<int32_t> all_tokens(prompt_token_ids,
                                             prompt_token_ids + n_prompt_tokens);
            for (auto id : out.token_ids) all_tokens.push_back(id);

            llm_result fwd = {};
            ok = run_llm_forward(ctx, all_tokens.data(), (int)all_tokens.size(), fwd);
            if (!ok || !fwd.logits) {
                if (fwd.hidden) free(fwd.hidden);
                if (fwd.logits) free(fwd.logits);
                return false;
            }
            last_logits = fwd.logits + (size_t)((int)all_tokens.size() - 1) * V;
            best_id = 0; best_score = -INFINITY;
            for (int v = 0; v < V; v++) {
                if (last_logits[v] > best_score) { best_score = last_logits[v]; best_id = v; }
            }
            {
                float max_l = best_score;
                float sum_exp = 0.0f;
                for (int v = 0; v < V; v++) sum_exp += expf(last_logits[v] - max_l);
                out.token_confidences.push_back(expf(best_score - max_l) / sum_exp);
            }
            if (fwd.hidden) free(fwd.hidden);
            free(fwd.logits);
        } else {
            // KV-cached decode step
            int pos = n_kv;  // position of the new token

            ggml_init_params ip{
                ctx.compute_meta.size(),
                ctx.compute_meta.data(),
                true,
            };
            ggml_context *g = ggml_init(ip);
            ggml_cgraph *gf = build_decode_step_graph(ctx, g, n_kv, pos);

            ggml_backend_sched_reset(ctx.sched);
            if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
                fprintf(stderr, "qwen2vl_ocr: decode step alloc failed\n");
                ggml_free(g);
                return false;
            }

            // Set token embedding input
            // Look up the embedding for best_id from embed_tokens
            std::vector<float> tok_emb(D);
            {
                // Read one row from embed_tokens (may be quantized)
                // Use a tiny ggml graph: get_rows(embed_tokens, [best_id])
                ggml_init_params eip{4096, nullptr, true};
                ggml_context *eg = ggml_init(eip);
                ggml_tensor *idx = ggml_new_tensor_1d(eg, GGML_TYPE_I32, 1);
                ggml_set_input(idx);
                ggml_tensor *emb = ggml_get_rows(eg, ctx.m.embed_tokens, idx);
                ggml_set_output(emb);
                ggml_cgraph *egf = ggml_new_graph(eg);
                ggml_build_forward_expand(egf, emb);
                ggml_backend_sched_reset(ctx.sched);
                ggml_backend_sched_alloc_graph(ctx.sched, egf);
                ggml_backend_tensor_set(ggml_graph_get_tensor(egf, idx->name),
                                        &best_id, 0, sizeof(int32_t));
                ggml_backend_sched_graph_compute(ctx.sched, egf);
                ggml_backend_tensor_get(emb, tok_emb.data(), 0, D * sizeof(float));
                ggml_free(eg);
            }

            // Re-alloc decode graph (sched was reset by embed lookup)
            ggml_backend_sched_reset(ctx.sched);
            if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
                ggml_free(g);
                return false;
            }

            ggml_tensor *te = ggml_graph_get_tensor(gf, "tok_emb");
            ggml_backend_tensor_set(te, tok_emb.data(), 0, D * sizeof(float));

            // Set mRoPE position (all 3 dims = pos for text tokens)
            int32_t pos_data[4] = {pos, pos, pos, 0};
            ggml_tensor *pi = ggml_graph_get_tensor(gf, "pos_ids");
            ggml_backend_tensor_set(pi, pos_data, 0, 4 * sizeof(int32_t));

            // Set KV cache inputs
            char name[64];
            for (int il = 0; il < n_layers; il++) {
                if (n_kv > 0) {
                    std::snprintf(name, sizeof(name), "k_in_%d", il);
                    ggml_tensor *ki = ggml_graph_get_tensor(gf, name);
                    if (ki) ggml_backend_tensor_set(ki, k_cache[il].data(), 0,
                                                    k_cache[il].size() * sizeof(float));
                    std::snprintf(name, sizeof(name), "v_in_%d", il);
                    ggml_tensor *vi = ggml_graph_get_tensor(gf, name);
                    if (vi) ggml_backend_tensor_set(vi, v_cache[il].data(), 0,
                                                    v_cache[il].size() * sizeof(float));
                }
            }

            // Compute
            if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "qwen2vl_ocr: decode step compute failed\n");
                ggml_free(g);
                return false;
            }

            // Read logits
            ggml_tensor *logits_t = ggml_graph_get_tensor(gf, "logits");
            std::vector<float> logits_data(V);
            ggml_backend_tensor_get(logits_t, logits_data.data(), 0, V * sizeof(float));

            best_id = 0; best_score = -INFINITY;
            for (int v = 0; v < V; v++) {
                if (logits_data[v] > best_score) { best_score = logits_data[v]; best_id = v; }
            }

            {
                float max_l = best_score;
                float sum_exp = 0.0f;
                for (int v = 0; v < V; v++) sum_exp += expf(logits_data[v] - max_l);
                out.token_confidences.push_back(expf(best_score - max_l) / sum_exp);
            }

            // Append new K/V to cache
            for (int il = 0; il < n_layers; il++) {
                std::vector<float> k_new(kv_dim), v_new(kv_dim);
                std::snprintf(name, sizeof(name), "k_out_%d", il);
                ggml_tensor *ko = ggml_graph_get_tensor(gf, name);
                if (ko) ggml_backend_tensor_get(ko, k_new.data(), 0, kv_dim * sizeof(float));
                std::snprintf(name, sizeof(name), "v_out_%d", il);
                ggml_tensor *vo = ggml_graph_get_tensor(gf, name);
                if (vo) ggml_backend_tensor_get(vo, v_new.data(), 0, kv_dim * sizeof(float));

                k_cache[il].insert(k_cache[il].end(), k_new.begin(), k_new.end());
                v_cache[il].insert(v_cache[il].end(), v_new.begin(), v_new.end());
            }
            n_kv++;

            ggml_free(g);
        }

        out.token_ids.push_back(best_id);
        if (ctx.verbosity >= 1) {
            fprintf(stderr, "  gen[%d]: token=%d score=%.2f%s\n",
                    gen, best_id, best_score, kv_ok ? " (cached)" : "");
        }
        if (best_id == eos_id) break;
    }

    return true;
}

}  // namespace qwen2vl_ocr

// ── GPT-2 BPE byte decoder ───────────────────────────────────────────

namespace {

// Build the inverse of GPT-2's bytes_to_unicode() table.
// Maps unicode codepoints back to byte values [0..255].
std::unordered_map<int, uint8_t> build_byte_decoder() {
    std::unordered_map<int, uint8_t> dec;
    // The standard GPT-2 byte_encoder: printable ASCII + Latin-1 supplement
    // map to themselves; other bytes map to 256+ codepoints.
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            dec[b] = (uint8_t)b;
        }
    }
    n = 0;
    for (int b = 0; b < 256; b++) {
        if (dec.find(b) == dec.end()) {
            dec[256 + n] = (uint8_t)b;
            n++;
        }
    }
    return dec;
}

// Decode a sequence of GPT-2 BPE token IDs to UTF-8 text.
std::string gpt2_bpe_decode(const std::vector<int32_t> & ids,
                             const std::vector<std::string> & vocab) {
    static auto byte_dec = build_byte_decoder();

    std::string merged;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int32_t)vocab.size()) {
            merged += vocab[id];
        }
    }

    // Convert GPT-2 unicode codepoints back to bytes
    std::string result;
    size_t i = 0;
    while (i < merged.size()) {
        // Decode one UTF-8 codepoint
        uint32_t cp = 0;
        int len = 1;
        uint8_t c = (uint8_t)merged[i];
        if (c < 0x80) {
            cp = c; len = 1;
        } else if (c < 0xE0) {
            cp = c & 0x1F; len = 2;
        } else if (c < 0xF0) {
            cp = c & 0x0F; len = 3;
        } else {
            cp = c & 0x07; len = 4;
        }
        for (int j = 1; j < len && i + j < merged.size(); j++) {
            cp = (cp << 6) | ((uint8_t)merged[i + j] & 0x3F);
        }
        i += len;

        auto it = byte_dec.find((int)cp);
        if (it != byte_dec.end()) {
            result += (char)it->second;
        } else {
            // Unknown codepoint — pass through as UTF-8
            for (int j = 0; j < len && i - len + j < merged.size(); j++) {
                result += merged[i - len + j];
            }
        }
    }
    return result;
}

}  // namespace

// ── C ABI wrapper ────────────────────────────────────────────────────

struct qwen2vl_ocr_context {
    qwen2vl_ocr::context inner;
    BPETokenizer tokenizer;
    bool has_tokenizer = false;
    std::string prompt = "Describe this image.";
    std::vector<int32_t> prompt_ids;  // cached tokenized prompt
    int max_tokens = 512;
    std::string last_result;
    std::vector<float> char_confidences;

    // Special token IDs (from GGUF metadata)
    int32_t im_start_id = 151644;
    int32_t im_end_id   = 151645;
    int32_t system_id   = 8948;    // "system" token
    int32_t user_id     = 872;     // "user" token
    int32_t assistant_id = 77091;  // "assistant" token
    int32_t newline_id  = 198;     // "\n" token
    int32_t vision_start_id = 151652;
    int32_t image_pad_id    = 151655;
    int32_t vision_end_id   = 151653;

    // Tokenize a text string via BPE. Falls back to hardcoded IDs if no tokenizer.
    std::vector<int32_t> tokenize(const std::string & text) {
        if (has_tokenizer) {
            auto enc = tokenizer.encode(text);
            // BPETokenizer.encode() adds BOS/EOS — strip them for raw text
            // We just want the raw token IDs without special tokens
            return enc.ids;
        }
        // Fallback: return empty (caller uses hardcoded defaults)
        return {};
    }

    // Build full chat-format token sequence with image placeholders
    std::vector<int32_t> build_token_ids(int n_image_tokens) {
        std::vector<int32_t> ids;

        // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        ids.push_back(im_start_id);
        ids.push_back(system_id);
        ids.push_back(newline_id);
        // "You are a helpful assistant."
        auto sys_tokens = tokenize("You are a helpful assistant.");
        if (sys_tokens.empty()) {
            ids.insert(ids.end(), {2610, 525, 264, 10950, 17847, 13});
        } else {
            ids.insert(ids.end(), sys_tokens.begin(), sys_tokens.end());
        }
        ids.push_back(im_end_id);
        ids.push_back(newline_id);

        // <|im_start|>user\n<|vision_start|>
        ids.push_back(im_start_id);
        ids.push_back(user_id);
        ids.push_back(newline_id);
        ids.push_back(vision_start_id);

        // <|image_pad|> × n_image_tokens
        for (int i = 0; i < n_image_tokens; i++) ids.push_back(image_pad_id);

        // <|vision_end|>
        ids.push_back(vision_end_id);

        // User prompt text
        if (!prompt_ids.empty()) {
            ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
        } else {
            auto toks = tokenize(prompt);
            if (toks.empty()) {
                // Fallback: "Describe this image."
                ids.insert(ids.end(), {41215, 419, 2168, 13});
            } else {
                ids.insert(ids.end(), toks.begin(), toks.end());
            }
        }

        // <|im_end|>\n<|im_start|>assistant\n
        ids.push_back(im_end_id);
        ids.push_back(newline_id);
        ids.push_back(im_start_id);
        ids.push_back(assistant_id);
        ids.push_back(newline_id);

        return ids;
    }
};

// Shared post-load init: load tokenizer from GGUF, set special IDs
static void post_load_init(qwen2vl_ocr_context * ctx, const char * gguf_path) {
    // Load BPE tokenizer from GGUF metadata
    gguf_context * g = core_gguf::open_metadata(gguf_path);
    if (g) {
        int tok_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tok_idx >= 0) {
            int n = gguf_get_arr_n(g, tok_idx);
            std::vector<std::string> vocab(n);
            for (int i = 0; i < n; i++) {
                vocab[i] = gguf_get_arr_str(g, tok_idx, i);
            }

            // Load merges
            std::vector<std::string> merges;
            int merge_idx = gguf_find_key(g, "tokenizer.ggml.merges");
            if (merge_idx >= 0) {
                int nm = gguf_get_arr_n(g, merge_idx);
                merges.resize(nm);
                for (int i = 0; i < nm; i++) {
                    merges[i] = gguf_get_arr_str(g, merge_idx, i);
                }
            }

            int eos_id = (int)core_gguf::kv_u32(g, "tokenizer.ggml.eos_token_id",
                                                  ctx->im_end_id);
            int pad_id = (int)core_gguf::kv_u32(g, "tokenizer.ggml.padding_token_id",
                                                  ctx->im_end_id);

            // GPT-2 BPE: no BOS, no suffix, not SPM style
            ctx->tokenizer.load(vocab, merges, eos_id, pad_id, -1, -1, false, 8192);
            ctx->has_tokenizer = true;
            fprintf(stderr, "qwen2vl_ocr: loaded BPE tokenizer (%d tokens, %zu merges)\n",
                    n, merges.size());
        }

        // Read special token IDs
        ctx->vision_start_id = (int32_t)core_gguf::kv_u32(g, "qwen2vl.vision_start_token_id",
                                                            ctx->vision_start_id);
        ctx->image_pad_id    = (int32_t)core_gguf::kv_u32(g, "qwen2vl.image_token_id",
                                                            ctx->image_pad_id);
        ctx->vision_end_id   = (int32_t)core_gguf::kv_u32(g, "qwen2vl.vision_end_token_id",
                                                            ctx->vision_end_id);

        core_gguf::free_metadata(g);
    }

    // Pre-tokenize default prompt
    if (ctx->has_tokenizer) {
        ctx->prompt_ids = ctx->tokenize(ctx->prompt);
    }
}

qwen2vl_ocr_context * qwen2vl_ocr_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new qwen2vl_ocr_context();
    if (!qwen2vl_ocr::load(ctx->inner, model_path, n_threads, 1)) {
        delete ctx;
        return nullptr;
    }
    post_load_init(ctx, model_path);
    return ctx;
}

qwen2vl_ocr_context * qwen2vl_ocr_init_split(
        const char * llm_path, const char * mmproj_path, int n_threads) {
    if (!llm_path) return nullptr;
    auto * ctx = new qwen2vl_ocr_context();
    if (!qwen2vl_ocr::load(ctx->inner, llm_path, n_threads, 1, mmproj_path)) {
        delete ctx;
        return nullptr;
    }
    post_load_init(ctx, llm_path);
    return ctx;
}

void qwen2vl_ocr_free(qwen2vl_ocr_context * ctx) {
    if (ctx) {
        qwen2vl_ocr::free_(ctx->inner);
        delete ctx;
    }
}

void qwen2vl_ocr_set_prompt(qwen2vl_ocr_context * ctx, const char * prompt) {
    if (ctx && prompt) {
        ctx->prompt = prompt;
        if (ctx->has_tokenizer) {
            ctx->prompt_ids = ctx->tokenize(prompt);
            fprintf(stderr, "qwen2vl_ocr: prompt tokenized to %zu tokens\n",
                    ctx->prompt_ids.size());
        } else {
            ctx->prompt_ids.clear();
        }
    }
}

void qwen2vl_ocr_set_max_tokens(qwen2vl_ocr_context * ctx, int max_tokens) {
    if (ctx) ctx->max_tokens = max_tokens;
}

// Internal: run full pipeline (preprocess → vision → tokenize → generate)
static const char * run_pipeline(qwen2vl_ocr_context * ctx,
                                  const image_preproc::result & pp,
                                  int * out_len) {
    // 1. Run vision encoder
    qwen2vl_ocr::vision_result vis = {};
    if (!qwen2vl_ocr::encode_vision(ctx->inner,
                                     pp.patches.data(), pp.n_patches,
                                     pp.grid_thw, vis)) {
        fprintf(stderr, "qwen2vl_ocr: vision encoder failed\n");
        return nullptr;
    }

    // 2. Build token IDs with image_pad placeholders
    auto token_ids = ctx->build_token_ids(vis.n_merged);

    // 3. Generate text
    qwen2vl_ocr::generate_result gen = {};
    // Pass grid_thw for mRoPE position computation
    qwen2vl_ocr::image_input img_in = {};
    img_in.image_embeds = vis.image_embeds;
    img_in.n_image_tokens = vis.n_merged;
    img_in.grid_thw = pp.grid_thw;
    img_in.n_images = 1;

    bool ok = qwen2vl_ocr::generate(ctx->inner,
                                     vis.image_embeds, vis.n_merged,
                                     vis.embed_dim,
                                     pp.grid_thw,  // pass actual grid for mRoPE
                                     token_ids.data(), (int)token_ids.size(),
                                     ctx->max_tokens, gen);

    qwen2vl_ocr::vision_result_free(vis);

    if (!ok) {
        fprintf(stderr, "qwen2vl_ocr: generation failed\n");
        return nullptr;
    }

    ctx->char_confidences = std::move(gen.token_confidences);

    // 4. Decode token IDs to text
    if (ctx->has_tokenizer) {
        ctx->last_result = gpt2_bpe_decode(gen.token_ids, ctx->tokenizer.get_vocab());
    } else {
        // Fallback: raw token IDs as comma-separated string
        ctx->last_result.clear();
        for (size_t i = 0; i < gen.token_ids.size(); i++) {
            if (i > 0) ctx->last_result += ",";
            ctx->last_result += std::to_string(gen.token_ids[i]);
        }
    }

    if (out_len) *out_len = (int)ctx->last_result.size();
    return ctx->last_result.c_str();
}

const char * qwen2vl_ocr_recognize_raw(
        qwen2vl_ocr_context * ctx,
        const uint8_t * pixel_bytes,
        int width, int height, int channels,
        int * out_len) {
    if (!ctx || !pixel_bytes) return nullptr;

    // Preprocess image via image_preprocess.cpp
    const auto & vhp = ctx->inner.m.vhp;
    image_preproc::config cfg;
    cfg.patch_size          = (int)vhp.spatial_patch_size;  // 14
    cfg.temporal_patch_size = (int)vhp.temporal_patch_size;  // 2
    cfg.merge_size          = (int)vhp.spatial_merge_size;   // 2
    cfg.min_pixels          = (int)vhp.min_pixels;
    cfg.max_pixels          = (int)vhp.max_pixels;
    for (int i = 0; i < 3; i++) {
        cfg.mean[i] = vhp.image_mean[i];
        cfg.std[i]  = vhp.image_std[i];
    }

    image_preproc::result pp;
    if (!image_preproc::preprocess_rgb(pixel_bytes, height, width, channels, cfg, pp)) {
        fprintf(stderr, "qwen2vl_ocr: image preprocessing failed\n");
        return nullptr;
    }

    fprintf(stderr, "qwen2vl_ocr: %dx%d → %dx%d, %d patches (%dx%d)\n",
            width, height, pp.resized_w, pp.resized_h,
            pp.n_patches, pp.grid_thw[1], pp.grid_thw[2]);

    return run_pipeline(ctx, pp, out_len);
}

const char * qwen2vl_ocr_recognize(
        qwen2vl_ocr_context * ctx,
        const float * pixels,
        int width, int height,
        int * out_len) {
    if (!ctx || !pixels) return nullptr;

    // Convert grayscale float [0,1] to uint8 RGB for preprocess_rgb
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        uint8_t v = (uint8_t)(pixels[i] * 255.0f + 0.5f);
        rgb[i*3+0] = rgb[i*3+1] = rgb[i*3+2] = v;
    }

    return qwen2vl_ocr_recognize_raw(ctx, rgb.data(), width, height, 3, out_len);
}

const float * qwen2vl_ocr_confidences(const qwen2vl_ocr_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float qwen2vl_ocr_mean_confidence(const qwen2vl_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}

