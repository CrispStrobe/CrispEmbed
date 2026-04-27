// bidirlm_vision.cpp — BidirLM-Omni vision tower forward pass.
//
// Architecture (from modeling_bidirlm_omni.py BidirLMOmniVisionModel):
//   PIL image → Python preprocessor → (n_patches, 3, T_patch, 16, 16) flat tensor
//   patch_embed (Conv3D == matmul of (1536, hidden)) → (n_patches, hidden=1024)
//   + bilinear-interpolated learned pos_embed (precomputed CPU-side)
//   2D RoPE (precomputed cos/sin tables, head_dim split y/x halves)
//   24 × pre-LN ViT block (fused QKV + bias, GELU MLP)
//   final patch merger: reshape (-, hidden*4) → LN → fc1 → GELU → fc2 → (-, 2048)
//   deepstack hooks at layers [8,16,24]: same merger shape but use_postshuffle_norm=True
//
// At extraction-start this file is a working scaffold with the loader done and
// encode() returning zeros. The forward pass is filled in incrementally; when
// encode() returns real data the parity test in tests/test_bidirlm_vision.py
// will go green.

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

bool load_tensors(context& ctx) {
    auto& m = ctx.m;
    auto get_t = [&](const std::map<std::string, ggml_tensor*>& tt,
                     const std::string& name) -> ggml_tensor* {
        auto it = tt.find(name);
        return it != tt.end() ? it->second : nullptr;
    };
    auto require = [&](const std::map<std::string, ggml_tensor*>& tt,
                       const std::string& name) -> ggml_tensor* {
        auto* t = get_t(tt, name);
        if (!t) fprintf(stderr, "bidirlm_vision: required tensor missing: %s\n", name.c_str());
        return t;
    };

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(/*path=*/nullptr, /*backend=*/ctx.backend,
                                 /*tag=*/"bidirlm_vision", wl)) {
        // load_weights with null path would fail — we need the path. Reload.
        // (Loader contract is that path is mandatory; that branch is unreachable
        //  from real callers.)
        return false;
    }
    return true;
}

// Variant of the above that takes the GGUF path. We keep load_tensors_from_path
// separate from load_hparams so callers can short-circuit when the GGUF has no
// vision tensors.
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

    // Compute-meta scratch: vision graph has ~depth*15 + ~30 ops, so 16384 nodes
    // is comfortable headroom. The same sizing crisp_audio uses.
    constexpr int kGraphCapacity = 16384;
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
                "patch=%u merge=%u deepstack_at=[%d,%d,%d]\n",
                ctx.m.hp.depth, ctx.m.hp.hidden_size, ctx.m.hp.out_hidden_size,
                ctx.m.hp.patch_size, ctx.m.hp.spatial_merge_size,
                ctx.m.hp.deepstack_indexes.size() > 0 ? ctx.m.hp.deepstack_indexes[0] : -1,
                ctx.m.hp.deepstack_indexes.size() > 1 ? ctx.m.hp.deepstack_indexes[1] : -1,
                ctx.m.hp.deepstack_indexes.size() > 2 ? ctx.m.hp.deepstack_indexes[2] : -1);
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

// ---------------------------------------------------------------------------
// Forward pass — filled in step by step. Below is a working stub that returns
// zeros so the rest of the system (C ABI, Python wrapper, parity test) can be
// built and integrated; replace the body with the real graph next.
// ---------------------------------------------------------------------------

bool encode(context& ctx,
            const float* /*pixel_patches*/, int n_patches,
            const int32_t* grid_thw, int n_images,
            encode_result& out) {
    if (!grid_thw || n_images <= 0 || n_patches <= 0) return false;
    const int merge = (int)ctx.m.hp.spatial_merge_size;
    const int merge_unit = merge * merge;

    int n_merged = 0;
    for (int i = 0; i < n_images; i++) {
        const int t = grid_thw[i*3 + 0];
        const int h = grid_thw[i*3 + 1];
        const int w = grid_thw[i*3 + 2];
        n_merged += (t * h * w) / merge_unit;
    }
    const int dim = (int)ctx.m.hp.out_hidden_size;
    const int n_ds = (int)ctx.m.deepstack.size();

    out.n_merged = n_merged;
    out.output_dim = dim;
    out.n_deepstack = n_ds;
    out.image_embeds = (float*)std::calloc((size_t)n_merged * dim, sizeof(float));
    out.deepstack    = (float*)std::calloc((size_t)n_ds * n_merged * dim, sizeof(float));
    if (!out.image_embeds || !out.deepstack) {
        encode_result_free(out);
        return false;
    }
    // STUB: returns zero embeddings. Replace with real graph.
    fprintf(stderr, "bidirlm_vision: STUB encode() returning zeros (graph pending)\n");
    return true;
}

void encode_result_free(encode_result& r) {
    if (r.image_embeds) { std::free(r.image_embeds); r.image_embeds = nullptr; }
    if (r.deepstack)    { std::free(r.deepstack); r.deepstack = nullptr; }
    r.n_merged = r.output_dim = r.n_deepstack = 0;
}

}  // namespace bidirlm_vision
