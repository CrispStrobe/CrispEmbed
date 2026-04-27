// bidirlm_vision.cpp — BidirLM-Omni vision tower forward pass.
//
// Architecture (from modeling_bidirlm_omni.py BidirLMOmniVisionModel):
//   PIL image → Python preprocessor → (n_patches, 3, T_patch, 16, 16) flat tensor
//   patch_embed (Conv3D == matmul of (1536, hidden)) → (n_patches, hidden)
//   + bilinear-interpolated learned pos_embed (host-precomputed corner gather)
//   2D rotate-half RoPE (host-precomputed cos/sin tables)
//   N × pre-LN ViT block (fused QKV + bias, gelu_pytorch_tanh MLP)
//   final patch merger: norm(hidden) → reshape(-,hidden*4) → fc1 → GELU → fc2 → (-, out_dim)
//   deepstack hooks at layers in cfg.deepstack_visual_indexes:
//     reshape(-,hidden*4) → norm(hidden*4) → fc1 → GELU → fc2 → (-, out_dim)
//
// Host-side preparation per encode():
//   * Position-embed bilinear interp baked into 4 corner index/weight buffers,
//     already in the merge-permuted order the merger expects.
//   * RoPE 2D (row, col) cos/sin tables, head_dim slots per token.
//   * Block-diagonal attn mask: each frame is its own attention block, with
//     boundaries set by cu_seqlens = repeat_interleave(h*w, t).cumsum().
//
// Graph itself: standard ggml-backed pre-LN ViT with manual rotate_half RoPE
// (we don't use ggml_rope_multi VISION mode here; the freq layout in HF is the
//  doubled `[row, col, row, col]` pattern, easier to apply directly).

#include "bidirlm_vision.h"

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

namespace bidirlm_vision {

namespace {

constexpr float kLayerNormEps = 1e-6f;
constexpr float kRopeTheta = 10000.0f;

bool load_hparams(context& ctx, const char* path) {
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) return false;
    auto u = [&](const char* k, uint32_t d) {
        return core_gguf::kv_u32(g, (std::string("bidirlm.vision.") + k).c_str(), d);
    };
    auto& hp = ctx.m.hp;
    if (gguf_find_key(g, "bidirlm.vision.depth") < 0) {
        // Not a vision-enabled GGUF — silently bail.
        core_gguf::free_metadata(g);
        return false;
    }
    hp.depth                   = u("depth", hp.depth);
    hp.hidden_size             = u("hidden_size", hp.hidden_size);
    hp.intermediate_size       = u("intermediate_size", hp.intermediate_size);
    hp.num_heads               = u("num_heads", hp.num_heads);
    hp.in_channels             = u("in_channels", hp.in_channels);
    hp.patch_size              = u("patch_size", hp.patch_size);
    hp.spatial_merge_size      = u("spatial_merge_size", hp.spatial_merge_size);
    hp.temporal_patch_size     = u("temporal_patch_size", hp.temporal_patch_size);
    hp.out_hidden_size         = u("out_hidden_size", hp.out_hidden_size);
    hp.num_position_embeddings = u("num_position_embeddings", hp.num_position_embeddings);

    int idx = gguf_find_key(g, "bidirlm.vision.deepstack_visual_indexes");
    if (idx >= 0) {
        const int n = gguf_get_arr_n(g, idx);
        hp.deepstack_indexes.resize(n);
        for (int i = 0; i < n; i++) {
            hp.deepstack_indexes[i] = (int)((const uint32_t*)gguf_get_arr_data(g, idx))[i];
        }
    }
    core_gguf::free_metadata(g);
    return true;
}

bool load_tensors_from_path(context& ctx, const char* path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "bidirlm_vision", wl)) {
        return false;
    }
    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;

    auto& m = ctx.m;
    auto get_t = [&](const std::string& name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string& name) -> ggml_tensor* {
        auto* t = get_t(name);
        if (!t) fprintf(stderr, "bidirlm_vision: required tensor missing: %s\n", name.c_str());
        return t;
    };

    m.patch_embed_w = require("visual.patch_embed.weight");
    m.patch_embed_b = require("visual.patch_embed.bias");
    m.pos_embed_w   = require("visual.pos_embed.weight");
    if (!m.patch_embed_w || !m.pos_embed_w) return false;

    m.blocks.resize(m.hp.depth);
    char buf[160];
    for (uint32_t i = 0; i < m.hp.depth; i++) {
        auto& b = m.blocks[i];
        auto rq = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "visual.blk.%u.%s", i, suf);
            return require(buf);
        };
        b.norm1_w = rq("norm1.weight");
        b.norm1_b = rq("norm1.bias");
        b.norm2_w = rq("norm2.weight");
        b.norm2_b = rq("norm2.bias");
        b.qkv_w   = rq("attn_qkv.weight");
        b.qkv_b   = rq("attn_qkv.bias");
        b.proj_w  = rq("attn_proj.weight");
        b.proj_b  = rq("attn_proj.bias");
        b.fc1_w   = rq("ffn_fc1.weight");
        b.fc1_b   = rq("ffn_fc1.bias");
        b.fc2_w   = rq("ffn_fc2.weight");
        b.fc2_b   = rq("ffn_fc2.bias");
        if (!b.qkv_w) return false;
    }

    auto load_merger = [&](merger_weights& mw, const std::string& pfx) {
        mw.norm_w = require(pfx + "norm.weight");
        mw.norm_b = require(pfx + "norm.bias");
        mw.fc1_w  = require(pfx + "fc1.weight");
        mw.fc1_b  = require(pfx + "fc1.bias");
        mw.fc2_w  = require(pfx + "fc2.weight");
        mw.fc2_b  = require(pfx + "fc2.bias");
        return mw.fc2_w != nullptr;
    };
    if (!load_merger(m.merger, "visual.merger.")) return false;

    m.deepstack.resize(m.hp.deepstack_indexes.size());
    for (size_t i = 0; i < m.hp.deepstack_indexes.size(); i++) {
        char p[80];
        std::snprintf(p, sizeof(p), "visual.deepstack.%zu.", i);
        if (!load_merger(m.deepstack[i], p)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host-side preparation: pos_embed bilinear corners + RoPE cos/sin + attn mask,
// all baked in merge-permuted token order.
// ---------------------------------------------------------------------------

struct host_prep {
    int n_patches = 0;
    int n_merged = 0;

    std::vector<int32_t> pe_idx[4];          // each (n_patches,)
    std::vector<float>   pe_w[4];            // each (n_patches,)

    std::vector<float>   cos_buf;            // (head_dim, n_patches) ne-flat
    std::vector<float>   sin_buf;
    std::vector<float>   mask_buf;           // (n_patches, n_patches)
};

void compute_host_inputs(const hparams& hp,
                          const int32_t* grid_thw, int n_images,
                          host_prep& hp_out) {
    const int merge = (int)hp.spatial_merge_size;
    const int merge_unit = merge * merge;
    const int head_dim = (int)(hp.hidden_size / hp.num_heads);
    const int half = head_dim / 2;
    const int quart = head_dim / 4;
    const int side = (int)std::lround(std::sqrt((float)hp.num_position_embeddings));

    // Total patches across all images: sum(t * h * w).
    int n_patches = 0;
    for (int i = 0; i < n_images; i++) {
        n_patches += grid_thw[i*3 + 0] * grid_thw[i*3 + 1] * grid_thw[i*3 + 2];
    }
    hp_out.n_patches = n_patches;
    hp_out.n_merged = n_patches / merge_unit;

    for (int c = 0; c < 4; c++) {
        hp_out.pe_idx[c].assign((size_t)n_patches, 0);
        hp_out.pe_w[c].assign((size_t)n_patches, 0.0f);
    }
    hp_out.cos_buf.assign((size_t)head_dim * n_patches, 0.0f);
    hp_out.sin_buf.assign((size_t)head_dim * n_patches, 0.0f);
    hp_out.mask_buf.assign((size_t)n_patches * n_patches,
                            -std::numeric_limits<float>::infinity());

    // Precompute inv_freq[j] = 1 / theta^(2j/half) for j in [0, quart).
    // (BidirLMOmniVisionRotaryEmbedding uses dim=head_dim/2 internally.)
    std::vector<float> inv_freq(quart);
    for (int j = 0; j < quart; j++) {
        inv_freq[j] = std::pow(kRopeTheta, -(float)(2 * j) / (float)half);
    }

    int token_offset = 0;        // global token index across all images
    int seq_offset = 0;           // start of current attention block (frame)
    for (int img = 0; img < n_images; img++) {
        const int t = grid_thw[img*3 + 0];
        const int h = grid_thw[img*3 + 1];
        const int w = grid_thw[img*3 + 2];
        const int tokens_per_frame = h * w;

        // Pos-embed grid samples for this image (h_idxs, w_idxs in [0, side-1]).
        // Mirrors torch.linspace(0, side-1, length).
        std::vector<float> h_idxs(h), w_idxs(w);
        for (int i = 0; i < h; i++) {
            h_idxs[i] = (h == 1) ? 0.0f
                                  : (float)i * ((float)(side - 1) / (float)(h - 1));
        }
        for (int i = 0; i < w; i++) {
            w_idxs[i] = (w == 1) ? 0.0f
                                  : (float)i * ((float)(side - 1) / (float)(w - 1));
        }

        const int merged_h = h / merge;
        const int merged_w = w / merge;

        for (int frame = 0; frame < t; frame++) {
            // Mark this frame's intra-block attention as 0 (allowed) in the mask.
            const int block_start = seq_offset;
            const int block_end = seq_offset + tokens_per_frame;
            for (int i = block_start; i < block_end; i++) {
                for (int j = block_start; j < block_end; j++) {
                    hp_out.mask_buf[(size_t)i * n_patches + j] = 0.0f;
                }
            }

            // Iterate in merge-permuted order: (mh, mw, ir, ic).
            for (int mh = 0; mh < merged_h; mh++) {
                for (int mw = 0; mw < merged_w; mw++) {
                    for (int ir = 0; ir < merge; ir++) {
                        for (int ic = 0; ic < merge; ic++) {
                            const int row = mh * merge + ir;
                            const int col = mw * merge + ic;
                            const float fh = h_idxs[row];
                            const float fw = w_idxs[col];
                            int hf = (int)fh;
                            int hc = std::min(hf + 1, side - 1);
                            int wf = (int)fw;
                            int wc = std::min(wf + 1, side - 1);
                            const float dh = fh - (float)hf;
                            const float dw = fw - (float)wf;

                            const int idx0 = hf * side + wf;
                            const int idx1 = hf * side + wc;
                            const int idx2 = hc * side + wf;
                            const int idx3 = hc * side + wc;
                            const float w0 = (1.0f - dh) * (1.0f - dw);
                            const float w1 = (1.0f - dh) * dw;
                            const float w2 = dh * (1.0f - dw);
                            const float w3 = dh * dw;

                            const int tok = token_offset++;
                            hp_out.pe_idx[0][tok] = idx0;
                            hp_out.pe_idx[1][tok] = idx1;
                            hp_out.pe_idx[2][tok] = idx2;
                            hp_out.pe_idx[3][tok] = idx3;
                            hp_out.pe_w[0][tok] = w0;
                            hp_out.pe_w[1][tok] = w1;
                            hp_out.pe_w[2][tok] = w2;
                            hp_out.pe_w[3][tok] = w3;

                            // RoPE: emb[k] for k in [0, head_dim) is
                            //   k in [0,        quart):  row * inv_freq[k]
                            //   k in [quart,    half):   col * inv_freq[k-quart]
                            //   k in [half,     3quart): row * inv_freq[k-half]
                            //   k in [3quart,   head):   col * inv_freq[k-3quart]
                            float* cos_row = hp_out.cos_buf.data() + (size_t)tok * head_dim;
                            float* sin_row = hp_out.sin_buf.data() + (size_t)tok * head_dim;
                            for (int j = 0; j < quart; j++) {
                                const float vr = (float)row * inv_freq[j];
                                const float vc = (float)col * inv_freq[j];
                                const float cr = std::cos(vr), sr_ = std::sin(vr);
                                const float cc = std::cos(vc), sc_ = std::sin(vc);
                                cos_row[j]                   = cr;
                                sin_row[j]                   = sr_;
                                cos_row[j + quart]           = cc;
                                sin_row[j + quart]           = sc_;
                                cos_row[j + half]            = cr;
                                sin_row[j + half]            = sr_;
                                cos_row[j + half + quart]    = cc;
                                sin_row[j + half + quart]    = sc_;
                            }
                        }
                    }
                }
            }
            seq_offset += tokens_per_frame;
        }
    }
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

struct graph_outputs {
    ggml_cgraph* gf = nullptr;
    std::vector<int> deepstack_layer_index;   // which layer each deepstack out came from
};

graph_outputs build_graph(context& ctx, int n_patches) {
    const auto& hp = ctx.m.hp;
    const int H = (int)hp.hidden_size;
    const int n_heads = (int)hp.num_heads;
    const int head_dim = H / n_heads;
    const int half = head_dim / 2;
    const int merge_unit = (int)(hp.spatial_merge_size * hp.spatial_merge_size);
    const int merger_in_dim = H * merge_unit;
    const int patch_flat_dim = (int)hp.in_channels *
                                (int)hp.temporal_patch_size *
                                (int)hp.patch_size *
                                (int)hp.patch_size;
    const int n_merged = n_patches / merge_unit;

    ggml_init_params ip{
        ctx.compute_meta.size(),
        ctx.compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);

    // ---- Inputs ----
    ggml_tensor* pixel_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, patch_flat_dim, n_patches);
    ggml_set_name(pixel_in, "pixel_in");
    ggml_set_input(pixel_in);

    ggml_tensor* pe_idx[4];
    ggml_tensor* pe_w[4];
    for (int c = 0; c < 4; c++) {
        char nm[32];
        pe_idx[c] = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_patches);
        std::snprintf(nm, sizeof(nm), "pe_idx_%d", c);
        ggml_set_name(pe_idx[c], nm); ggml_set_input(pe_idx[c]);

        pe_w[c] = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, n_patches);
        std::snprintf(nm, sizeof(nm), "pe_w_%d", c);
        ggml_set_name(pe_w[c], nm); ggml_set_input(pe_w[c]);
    }

    // cos/sin: (head_dim, 1, n_patches) — broadcasts over n_heads in attention
    ggml_tensor* cos_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, head_dim, 1, n_patches);
    ggml_tensor* sin_in = ggml_new_tensor_3d(g, GGML_TYPE_F32, head_dim, 1, n_patches);
    ggml_set_name(cos_in, "cos_in"); ggml_set_input(cos_in);
    ggml_set_name(sin_in, "sin_in"); ggml_set_input(sin_in);

    ggml_tensor* mask_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, n_patches, n_patches);
    ggml_set_name(mask_in, "attn_mask"); ggml_set_input(mask_in);

    // ---- Patch embed ----
    // Converter stores patch_embed_w as 2D (in_flat, H); reshape is a no-op
    // here, kept defensive in case future GGUFs change layout.
    ggml_tensor* W_pe = ggml_reshape_2d(g, ctx.m.patch_embed_w, patch_flat_dim, H);
    ggml_tensor* x = ggml_mul_mat(g, W_pe, pixel_in);
    if (ctx.m.patch_embed_b) x = ggml_add(g, x, ctx.m.patch_embed_b);

    // ---- Position embed: 4-corner gather + weighted sum ----
    {
        ggml_tensor* pe_sum = nullptr;
        for (int c = 0; c < 4; c++) {
            // pos_embed_w ne=(H, num_pos). get_rows -> (H, n_patches).
            ggml_tensor* g_c = ggml_get_rows(g, ctx.m.pos_embed_w, pe_idx[c]);
            // pe_w[c] ne=(1, n_patches) — broadcasts over H dim.
            g_c = ggml_mul(g, g_c, pe_w[c]);
            pe_sum = (c == 0) ? g_c : ggml_add(g, pe_sum, g_c);
        }
        x = ggml_add(g, x, pe_sum);
    }

    // ---- LayerNorm helper (with bias) ----
    auto ln = [&](ggml_tensor* t, ggml_tensor* w, ggml_tensor* b) -> ggml_tensor* {
        ggml_tensor* y = ggml_norm(g, t, kLayerNormEps);
        y = ggml_mul(g, y, w);
        if (b) y = ggml_add(g, y, b);
        return y;
    };

    // ---- ViT blocks ----
    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    graph_outputs out_meta;
    out_meta.gf = gf;

    for (uint32_t il = 0; il < hp.depth; il++) {
        const auto& blk = ctx.m.blocks[il];
        ggml_tensor* residual = x;

        // Attention path
        ggml_tensor* y = ln(x, blk.norm1_w, blk.norm1_b);
        ggml_tensor* qkv = ggml_mul_mat(g, blk.qkv_w, y);
        if (blk.qkv_b) qkv = ggml_add(g, qkv, blk.qkv_b);
        // qkv ne=(3*H, n_patches); flat HF order is (n_patches, 3, n_heads, head_dim)
        // → ggml ne=(head_dim, n_heads, 3, n_patches).
        qkv = ggml_reshape_4d(g, qkv, head_dim, n_heads, 3, n_patches);

        // View q/k/v slices (each ne=(head_dim, n_heads, 1, n_patches), squeeze "1" via reshape).
        ggml_tensor* Q = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                                       qkv->nb[1], qkv->nb[3], 0 * qkv->nb[2]);
        ggml_tensor* K = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                                       qkv->nb[1], qkv->nb[3], 1 * qkv->nb[2]);
        ggml_tensor* V = ggml_view_3d(g, qkv, head_dim, n_heads, n_patches,
                                       qkv->nb[1], qkv->nb[3], 2 * qkv->nb[2]);
        Q = ggml_cont(g, Q);
        K = ggml_cont(g, K);
        V = ggml_cont(g, V);

        // 2D rotate-half RoPE
        auto apply_rope = [&](ggml_tensor* t) {
            ggml_tensor* h1 = ggml_view_3d(g, t, half, n_heads, n_patches,
                                            t->nb[1], t->nb[2], 0);
            ggml_tensor* h2 = ggml_view_3d(g, t, half, n_heads, n_patches,
                                            t->nb[1], t->nb[2], (size_t)half * t->nb[0]);
            ggml_tensor* h2_neg = ggml_scale(g, ggml_cont(g, h2), -1.0f);
            ggml_tensor* rot = ggml_concat(g, h2_neg, ggml_cont(g, h1), 0);
            return ggml_add(g,
                ggml_mul(g, t, cos_in),
                ggml_mul(g, rot, sin_in));
        };
        Q = apply_rope(Q);
        K = apply_rope(K);

        // Permute to (head_dim, n_patches, n_heads): treat n_heads as batch.
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(g, K, Q);  // (n_patches_k, n_patches_q, n_heads)
        scores = ggml_add(g, scores, mask_in);
        scores = ggml_soft_max_ext(g, scores, nullptr, attn_scale, 0.0f);

        ggml_tensor* V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(g, V_perm, scores);             // (head_dim, n_patches, n_heads)
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));          // (head_dim, n_heads, n_patches)
        attn = ggml_reshape_2d(g, attn, H, n_patches);

        attn = ggml_mul_mat(g, blk.proj_w, attn);
        if (blk.proj_b) attn = ggml_add(g, attn, blk.proj_b);
        x = ggml_add(g, residual, attn);

        // FFN (gelu_pytorch_tanh)
        residual = x;
        y = ln(x, blk.norm2_w, blk.norm2_b);
        y = ggml_mul_mat(g, blk.fc1_w, y);
        if (blk.fc1_b) y = ggml_add(g, y, blk.fc1_b);
        y = ggml_gelu(g, y);
        y = ggml_mul_mat(g, blk.fc2_w, y);
        if (blk.fc2_b) y = ggml_add(g, y, blk.fc2_b);
        x = ggml_add(g, residual, y);

        // DeepStack hook — fires when this layer index appears in deepstack_indexes.
        for (size_t k = 0; k < ctx.m.deepstack.size(); k++) {
            if ((int)il != hp.deepstack_indexes[k]) continue;
            // PostShuffle merger: reshape first (4 patches → one merged token),
            // then norm over the merged H*4 dim.
            ggml_tensor* shuffled = ggml_reshape_2d(g, ggml_cont(g, x),
                                                    merger_in_dim, n_merged);
            ggml_tensor* h = ln(shuffled,
                                ctx.m.deepstack[k].norm_w,
                                ctx.m.deepstack[k].norm_b);
            h = ggml_mul_mat(g, ctx.m.deepstack[k].fc1_w, h);
            if (ctx.m.deepstack[k].fc1_b) h = ggml_add(g, h, ctx.m.deepstack[k].fc1_b);
            h = ggml_gelu_erf(g, h);  // patch merger uses exact GELU
            h = ggml_mul_mat(g, ctx.m.deepstack[k].fc2_w, h);
            if (ctx.m.deepstack[k].fc2_b) h = ggml_add(g, h, ctx.m.deepstack[k].fc2_b);

            char nm[40];
            std::snprintf(nm, sizeof(nm), "deepstack_%zu", out_meta.deepstack_layer_index.size());
            ggml_set_name(h, nm);
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);
            out_meta.deepstack_layer_index.push_back((int)il);
            break;
        }
    }

    // ---- Final patch merger (use_postshuffle_norm=False) ----
    // Norm over hidden first, THEN reshape. norm.weight has shape (H,).
    ggml_tensor* m1 = ggml_norm(g, x, kLayerNormEps);
    m1 = ggml_mul(g, m1, ctx.m.merger.norm_w);
    if (ctx.m.merger.norm_b) m1 = ggml_add(g, m1, ctx.m.merger.norm_b);
    m1 = ggml_reshape_2d(g, ggml_cont(g, m1), merger_in_dim, n_merged);
    m1 = ggml_mul_mat(g, ctx.m.merger.fc1_w, m1);
    if (ctx.m.merger.fc1_b) m1 = ggml_add(g, m1, ctx.m.merger.fc1_b);
    m1 = ggml_gelu_erf(g, m1);
    m1 = ggml_mul_mat(g, ctx.m.merger.fc2_w, m1);
    if (ctx.m.merger.fc2_b) m1 = ggml_add(g, m1, ctx.m.merger.fc2_b);

    ggml_set_name(m1, "image_embeds");
    ggml_set_output(m1);
    ggml_build_forward_expand(gf, m1);

    ggml_free(g);
    return out_meta;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool load(context& ctx, const char* gguf_path, ggml_backend_t shared_backend,
          int n_threads, int verbosity) {
    ctx.n_threads = n_threads > 0 ? n_threads : 4;
    ctx.verbosity = verbosity;

    if (!load_hparams(ctx, gguf_path)) return false;

    // Reuse parent context's backend if shared_backend is provided; otherwise
    // pick the best available. We don't free shared_backend in free_().
    if (shared_backend) {
        ctx.backend = shared_backend;
    } else {
        ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ctx.backend = gdev ? ggml_backend_dev_init(gdev, nullptr) : ggml_backend_cpu_init();
    }
    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, ctx.n_threads);

    if (!load_tensors_from_path(ctx, gguf_path)) {
        free_(ctx);
        return false;
    }

    // Compute-meta scratch: 32 layers × ~30 ops + 3 deepstack hooks × ~8 ops +
    // ~30 setup ops. 32768 nodes is comfortable headroom.
    constexpr int kGraphCapacity = 32768;
    ctx.compute_meta.resize(
        ggml_tensor_overhead() * kGraphCapacity +
        ggml_graph_overhead_custom(kGraphCapacity, false));

    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(),
                                       kGraphCapacity, false, false);

    if (ctx.verbosity >= 1) {
        fprintf(stderr,
                "bidirlm_vision: loaded depth=%u hidden=%u out=%u "
                "patch=%u merge=%u num_pos=%u deepstack_at=[",
                ctx.m.hp.depth, ctx.m.hp.hidden_size, ctx.m.hp.out_hidden_size,
                ctx.m.hp.patch_size, ctx.m.hp.spatial_merge_size,
                ctx.m.hp.num_position_embeddings);
        for (size_t i = 0; i < ctx.m.hp.deepstack_indexes.size(); i++) {
            fprintf(stderr, "%s%d", (i ? "," : ""), ctx.m.hp.deepstack_indexes[i]);
        }
        fprintf(stderr, "]\n");
    }
    return true;
}

void free_(context& ctx) {
    if (ctx.sched) { ggml_backend_sched_free(ctx.sched); ctx.sched = nullptr; }
    if (ctx.model_buf) { ggml_backend_buffer_free(ctx.model_buf); ctx.model_buf = nullptr; }
    if (ctx.model_ctx) { ggml_free(ctx.model_ctx); ctx.model_ctx = nullptr; }
    if (ctx.backend_cpu) { ggml_backend_free(ctx.backend_cpu); ctx.backend_cpu = nullptr; }
    // ctx.backend is shared with parent — do not free here.
    ctx.backend = nullptr;
}

bool encode(context& ctx,
            const float* pixel_patches, int n_patches,
            const int32_t* grid_thw, int n_images,
            encode_result& out) {
    if (!grid_thw || n_images <= 0 || n_patches <= 0 || !pixel_patches) return false;

    const auto& hp = ctx.m.hp;
    const int merge_unit = (int)(hp.spatial_merge_size * hp.spatial_merge_size);

    // Validate that grid_thw and n_patches agree.
    int total = 0;
    for (int i = 0; i < n_images; i++) {
        total += grid_thw[i*3 + 0] * grid_thw[i*3 + 1] * grid_thw[i*3 + 2];
    }
    if (total != n_patches) {
        fprintf(stderr, "bidirlm_vision: grid_thw token count (%d) != n_patches (%d)\n",
                total, n_patches);
        return false;
    }
    if (n_patches % merge_unit != 0) {
        fprintf(stderr, "bidirlm_vision: n_patches=%d not divisible by merge_unit=%d\n",
                n_patches, merge_unit);
        return false;
    }

    const int patch_flat_dim = (int)hp.in_channels *
                                (int)hp.temporal_patch_size *
                                (int)hp.patch_size *
                                (int)hp.patch_size;

    host_prep prep;
    compute_host_inputs(hp, grid_thw, n_images, prep);

    graph_outputs gout = build_graph(ctx, n_patches);
    ggml_cgraph* gf = gout.gf;

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "bidirlm_vision: failed to allocate compute graph\n");
        return false;
    }

    // Set inputs.
    auto set_in = [&](const char* name, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            fprintf(stderr, "bidirlm_vision: graph input '%s' not found\n", name);
            return false;
        }
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_in("pixel_in", pixel_patches,
                (size_t)patch_flat_dim * n_patches * sizeof(float)))
        return false;
    char nm[32];
    for (int c = 0; c < 4; c++) {
        std::snprintf(nm, sizeof(nm), "pe_idx_%d", c);
        if (!set_in(nm, prep.pe_idx[c].data(),
                    prep.pe_idx[c].size() * sizeof(int32_t))) return false;
        std::snprintf(nm, sizeof(nm), "pe_w_%d", c);
        if (!set_in(nm, prep.pe_w[c].data(),
                    prep.pe_w[c].size() * sizeof(float))) return false;
    }
    if (!set_in("cos_in", prep.cos_buf.data(),
                prep.cos_buf.size() * sizeof(float))) return false;
    if (!set_in("sin_in", prep.sin_buf.data(),
                prep.sin_buf.size() * sizeof(float))) return false;
    if (!set_in("attn_mask", prep.mask_buf.data(),
                prep.mask_buf.size() * sizeof(float))) return false;

    if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "bidirlm_vision: graph compute failed\n");
        return false;
    }

    // Read out.
    ggml_tensor* img_t = ggml_graph_get_tensor(gf, "image_embeds");
    if (!img_t) return false;
    const int dim = (int)img_t->ne[0];
    const int n_merged = (int)img_t->ne[1];
    const int n_ds = (int)gout.deepstack_layer_index.size();

    out.n_merged = n_merged;
    out.output_dim = dim;
    out.n_deepstack = n_ds;
    out.image_embeds = (float*)std::calloc((size_t)n_merged * dim, sizeof(float));
    out.deepstack    = n_ds > 0
        ? (float*)std::calloc((size_t)n_ds * n_merged * dim, sizeof(float))
        : nullptr;
    if (!out.image_embeds || (n_ds > 0 && !out.deepstack)) {
        encode_result_free(out);
        return false;
    }

    ggml_backend_tensor_get(img_t, out.image_embeds, 0,
                            (size_t)n_merged * dim * sizeof(float));
    for (int k = 0; k < n_ds; k++) {
        char dnm[32];
        std::snprintf(dnm, sizeof(dnm), "deepstack_%d", k);
        ggml_tensor* dt = ggml_graph_get_tensor(gf, dnm);
        if (!dt) {
            fprintf(stderr, "bidirlm_vision: deepstack output '%s' missing\n", dnm);
            continue;
        }
        ggml_backend_tensor_get(dt, out.deepstack + (size_t)k * n_merged * dim, 0,
                                (size_t)n_merged * dim * sizeof(float));
    }
    return true;
}

void encode_result_free(encode_result& r) {
    if (r.image_embeds) { std::free(r.image_embeds); r.image_embeds = nullptr; }
    if (r.deepstack)    { std::free(r.deepstack); r.deepstack = nullptr; }
    r.n_merged = r.output_dim = r.n_deepstack = 0;
}

}  // namespace bidirlm_vision
