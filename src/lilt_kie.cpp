// lilt_kie.cpp — LiLT dual-stream encoder with BiACM for token classification.

#include "lilt_kie.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace lilt_kie {

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct lilt_layer {
    // Text stream attention
    ggml_tensor *q_w, *q_b;
    ggml_tensor *k_w, *k_b;
    ggml_tensor *v_w, *v_b;
    ggml_tensor *o_w, *o_b;
    ggml_tensor *attn_ln_w, *attn_ln_b;

    // Layout stream attention
    ggml_tensor *lq_w, *lq_b;
    ggml_tensor *lk_w, *lk_b;
    ggml_tensor *lv_w, *lv_b;
    ggml_tensor *lo_w, *lo_b;
    ggml_tensor *lattn_ln_w, *lattn_ln_b;

    // Text FFN
    ggml_tensor *fc1_w, *fc1_b;
    ggml_tensor *fc2_w, *fc2_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b;

    // Layout FFN
    ggml_tensor *lfc1_w, *lfc1_b;
    ggml_tensor *lfc2_w, *lfc2_b;
    ggml_tensor *lffn_ln_w, *lffn_ln_b;
};

struct lilt_model {
    // Hyperparams
    int n_vocab, n_layer, n_head, n_embd, n_inter;
    int max_pos, max_2d_pos, shrink, layout_dim;
    float ln_eps;
    int n_labels;
    std::map<int, std::string> id2label;

    // Text embeddings
    ggml_tensor *token_embd;   // [n_vocab, n_embd]
    ggml_tensor *pos_embd;     // [max_pos, n_embd]
    ggml_tensor *type_embd;    // [1, n_embd] (optional)
    ggml_tensor *embd_ln_w, *embd_ln_b;

    // Layout embeddings
    ggml_tensor *x_embd;       // [max_2d, n_embd/6]  (128 for base)
    ggml_tensor *y_embd;       // [max_2d, n_embd/6]
    ggml_tensor *h_embd;       // [max_2d, n_embd/6]
    ggml_tensor *w_embd;       // [max_2d, n_embd/6]
    ggml_tensor *box_proj_w, *box_proj_b;  // [n_embd, layout_dim]
    ggml_tensor *box_pos_embd; // [max_pos, layout_dim]
    ggml_tensor *layout_ln_w, *layout_ln_b;

    // Classifier head
    ggml_tensor *cls_w, *cls_b;  // [n_embd, n_labels]

    std::vector<lilt_layer> layers;
};

struct context {
    lilt_model model;
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    ggml_backend_sched_t sched = nullptr;
    std::vector<char> compute_meta;
    int n_threads = 4;
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool load(context** out, const char* model_path, int n_threads) {
    if (!out || !model_path) return false;

    auto* ctx = new context;
    ctx->n_threads = n_threads;

    // Init CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) { delete ctx; return false; }

    // Pass 1: read metadata
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "lilt_kie: failed to open %s\n", model_path);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return false;
    }

    auto& m = ctx->model;
    m.n_vocab    = core_gguf::kv_u32(gctx, "lilt.vocab_size", 50265);
    m.n_embd     = core_gguf::kv_u32(gctx, "lilt.hidden_size", 768);
    m.n_layer    = core_gguf::kv_u32(gctx, "lilt.num_hidden_layers", 12);
    m.n_head     = core_gguf::kv_u32(gctx, "lilt.num_attention_heads", 12);
    m.n_inter    = core_gguf::kv_u32(gctx, "lilt.intermediate_size", 3072);
    m.max_pos    = core_gguf::kv_u32(gctx, "lilt.max_position_embeddings", 514);
    m.max_2d_pos = core_gguf::kv_u32(gctx, "lilt.max_2d_position_embeddings", 1024);
    m.shrink     = core_gguf::kv_u32(gctx, "lilt.channel_shrink_ratio", 4);
    m.layout_dim = core_gguf::kv_u32(gctx, "lilt.layout_dim", m.n_embd / m.shrink);
    m.ln_eps     = core_gguf::kv_f32(gctx, "lilt.layer_norm_eps", 1e-5f);
    m.n_labels   = core_gguf::kv_u32(gctx, "lilt.num_labels", 0);

    // Read label names
    for (int i = 0; i < m.n_labels; i++) {
        char key[64];
        snprintf(key, sizeof(key), "lilt.label.%d", i);
        std::string val = core_gguf::kv_str(gctx, key, "");
        if (!val.empty()) m.id2label[i] = val;
    }

    core_gguf::free_metadata(gctx);

    // Pass 2: load tensor weights
    if (!core_gguf::load_weights(model_path, ctx->backend, "lilt", ctx->wl)) {
        fprintf(stderr, "lilt_kie: failed to load weights from %s\n", model_path);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return false;
    }

    auto get = [&](const char* key) -> ggml_tensor* {
        return core_gguf::try_get(ctx->wl.tensors, key);
    };

    fprintf(stderr, "lilt_kie: %d layers, %d heads, %d hidden, %d layout_dim, %d labels\n",
            m.n_layer, m.n_head, m.n_embd, m.layout_dim, m.n_labels);

    // Text embeddings
    m.token_embd = get("token_embd.weight");
    m.pos_embd   = get("position_embd.weight");
    m.type_embd  = get("token_type_embd.weight");
    m.embd_ln_w  = get("embd_ln.weight");
    m.embd_ln_b  = get("embd_ln.bias");

    // Layout embeddings
    m.x_embd       = get("layout.x_embd.weight");
    m.y_embd       = get("layout.y_embd.weight");
    m.h_embd       = get("layout.h_embd.weight");
    m.w_embd       = get("layout.w_embd.weight");
    m.box_proj_w   = get("layout.box_proj.weight");
    m.box_proj_b   = get("layout.box_proj.bias");
    m.box_pos_embd = get("layout.pos_embd.weight");
    m.layout_ln_w  = get("layout.ln.weight");
    m.layout_ln_b  = get("layout.ln.bias");

    // Classifier
    m.cls_w = get("classifier.weight");
    m.cls_b = get("classifier.bias");

    // Per-layer weights
    m.layers.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; i++) {
        char pfx[32];
        snprintf(pfx, sizeof(pfx), "blk.%d.", i);
        auto k = [&](const char* suffix) {
            std::string s = std::string(pfx) + suffix;
            return get(s.c_str());
        };
        auto& L = m.layers[i];

        // Text attention
        L.q_w = k("attn_q.weight"); L.q_b = k("attn_q.bias");
        L.k_w = k("attn_k.weight"); L.k_b = k("attn_k.bias");
        L.v_w = k("attn_v.weight"); L.v_b = k("attn_v.bias");
        L.o_w = k("attn_o.weight"); L.o_b = k("attn_o.bias");
        L.attn_ln_w = k("attn_ln.weight"); L.attn_ln_b = k("attn_ln.bias");

        // Layout attention
        L.lq_w = k("layout_q.weight"); L.lq_b = k("layout_q.bias");
        L.lk_w = k("layout_k.weight"); L.lk_b = k("layout_k.bias");
        L.lv_w = k("layout_v.weight"); L.lv_b = k("layout_v.bias");
        L.lo_w = k("layout_attn_o.weight"); L.lo_b = k("layout_attn_o.bias");
        L.lattn_ln_w = k("layout_attn_ln.weight"); L.lattn_ln_b = k("layout_attn_ln.bias");

        // Text FFN
        L.fc1_w = k("ffn_up.weight"); L.fc1_b = k("ffn_up.bias");
        L.fc2_w = k("ffn_down.weight"); L.fc2_b = k("ffn_down.bias");
        L.ffn_ln_w = k("ffn_ln.weight"); L.ffn_ln_b = k("ffn_ln.bias");

        // Layout FFN
        L.lfc1_w = k("layout_ffn_up.weight"); L.lfc1_b = k("layout_ffn_up.bias");
        L.lfc2_w = k("layout_ffn_down.weight"); L.lfc2_b = k("layout_ffn_down.bias");
        L.lffn_ln_w = k("layout_ffn_ln.weight"); L.lffn_ln_b = k("layout_ffn_ln.bias");
    }

    // Verify critical tensors
    if (!m.token_embd || !m.pos_embd || !m.x_embd) {
        fprintf(stderr, "lilt_kie: missing critical tensors\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return false;
    }

    // Create scheduler
    ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 8192, false, false);

    *out = ctx;
    return true;
}

// ---------------------------------------------------------------------------
// Helper: LayerNorm
// ---------------------------------------------------------------------------
static ggml_tensor* layer_norm(ggml_context* g, ggml_tensor* x,
                                ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, w);
    if (b) x = ggml_add(g, x, b);
    return x;
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

static ggml_cgraph* build_graph(context* ctx, int T, bool dump = false) {
    auto& m = ctx->model;
    const int H    = m.n_embd;
    const int LD   = m.layout_dim;
    const int NH   = m.n_head;
    const int HD   = H / NH;
    const int LHD  = LD / NH;  // layout head dim
    const float eps = m.ln_eps;

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context* g = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);

    // --- Inputs ---
    ggml_tensor* tok_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(tok_ids, "tok_ids");
    ggml_set_input(tok_ids);

    ggml_tensor* pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // bbox: [T, 4] as int32, but we pass 6 values: x0, y0, x1, y1, w, h
    // Actually we pass [T*6] int32 for the 6 position lookups
    ggml_tensor* bbox_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T * 6);
    ggml_set_name(bbox_ids, "bbox_ids");
    ggml_set_input(bbox_ids);

    // --- Text embeddings ---
    ggml_tensor* cur = ggml_get_rows(g, m.token_embd, tok_ids);
    ggml_tensor* pos = ggml_get_rows(g, m.pos_embd, pos_ids);
    cur = ggml_add(g, cur, pos);
    if (m.type_embd) {
        // All zeros for single-segment — just add the type=0 embedding
        ggml_tensor* type_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_name(type_ids, "type_ids");
        ggml_set_input(type_ids);
        cur = ggml_add(g, cur, ggml_get_rows(g, m.type_embd, type_ids));
    }
    cur = layer_norm(g, cur, m.embd_ln_w, m.embd_ln_b, eps);

    if (dump) {
        ggml_set_name(cur, "text_embed");
        ggml_set_output(cur);
    }

    // --- Layout embeddings ---
    // bbox_ids layout: [x0_0..x0_T, y0_0..y0_T, x1_0..x1_T, y1_0..y1_T, w_0..w_T, h_0..h_T]
    auto bbox_slice = [&](int offset) -> ggml_tensor* {
        return ggml_view_1d(g, bbox_ids, T, offset * T * sizeof(int32_t));
    };

    // x_embd is [max_2d, emb_dim_per_coord] where emb_dim_per_coord = H/6 ≈ 128
    ggml_tensor* x0_e = ggml_get_rows(g, m.x_embd, bbox_slice(0));
    ggml_tensor* y0_e = ggml_get_rows(g, m.y_embd, bbox_slice(1));
    ggml_tensor* x1_e = ggml_get_rows(g, m.x_embd, bbox_slice(2));
    ggml_tensor* y1_e = ggml_get_rows(g, m.y_embd, bbox_slice(3));
    ggml_tensor* w_e  = ggml_get_rows(g, m.w_embd, bbox_slice(4));
    ggml_tensor* h_e  = ggml_get_rows(g, m.h_embd, bbox_slice(5));

    // Concatenate: [T, 128] × 6 → [T, 768]  (along dim 0 which is the fast axis)
    // Order must match Python: left(x0), upper(y0), right(x1), lower(y1), h, w
    ggml_tensor* layout_cat = ggml_concat(g, x0_e, y0_e, 0);
    layout_cat = ggml_concat(g, layout_cat, x1_e, 0);
    layout_cat = ggml_concat(g, layout_cat, y1_e, 0);
    layout_cat = ggml_concat(g, layout_cat, h_e, 0);
    layout_cat = ggml_concat(g, layout_cat, w_e, 0);
    // layout_cat: [768, T]

    // Project 768 → layout_dim (192)
    ggml_tensor* lcur = ggml_mul_mat(g, m.box_proj_w, layout_cat);
    if (m.box_proj_b) lcur = ggml_add(g, lcur, m.box_proj_b);

    // Add layout position embedding
    ggml_tensor* lpos = ggml_get_rows(g, m.box_pos_embd, pos_ids);
    lcur = ggml_add(g, lcur, lpos);
    lcur = layer_norm(g, lcur, m.layout_ln_w, m.layout_ln_b, eps);

    if (dump) {
        ggml_set_name(lcur, "layout_embed");
        ggml_set_output(lcur);
    }

    // --- Encoder layers ---
    for (int il = 0; il < m.n_layer; il++) {
        const auto& L = m.layers[il];
        ggml_tensor* text_inp = cur;
        ggml_tensor* layout_inp = lcur;

        // === Text QKV ===
        ggml_tensor* Q = ggml_mul_mat(g, L.q_w, cur);
        if (L.q_b) Q = ggml_add(g, Q, L.q_b);
        ggml_tensor* K = ggml_mul_mat(g, L.k_w, cur);
        if (L.k_b) K = ggml_add(g, K, L.k_b);
        ggml_tensor* V = ggml_mul_mat(g, L.v_w, cur);
        if (L.v_b) V = ggml_add(g, V, L.v_b);

        // === Layout QKV ===
        ggml_tensor* LQ = ggml_mul_mat(g, L.lq_w, lcur);
        if (L.lq_b) LQ = ggml_add(g, LQ, L.lq_b);
        ggml_tensor* LK = ggml_mul_mat(g, L.lk_w, lcur);
        if (L.lk_b) LK = ggml_add(g, LK, L.lk_b);
        ggml_tensor* LV = ggml_mul_mat(g, L.lv_w, lcur);
        if (L.lv_b) LV = ggml_add(g, LV, L.lv_b);

        // === Reshape for multi-head: [H, T] → [HD, NH, T] ===
        Q = ggml_reshape_3d(g, Q, HD, NH, T);
        K = ggml_reshape_3d(g, K, HD, NH, T);
        V = ggml_reshape_3d(g, V, HD, NH, T);
        LQ = ggml_reshape_3d(g, LQ, LHD, NH, T);
        LK = ggml_reshape_3d(g, LK, LHD, NH, T);
        LV = ggml_reshape_3d(g, LV, LHD, NH, T);

        // === Permute to [HD, T, NH] for matmul ===
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
        LQ = ggml_cont(g, ggml_permute(g, LQ, 0, 2, 1, 3));
        LK = ggml_cont(g, ggml_permute(g, LK, 0, 2, 1, 3));
        LV = ggml_cont(g, ggml_permute(g, LV, 0, 2, 1, 3));

        // === BiACM: compute attention scores for both streams ===
        // text_scores = Q @ K^T / sqrt(HD)  → [T, T, NH]
        float text_scale = 1.0f / sqrtf((float)HD);
        ggml_tensor* text_scores = ggml_mul_mat(g, K, Q);
        text_scores = ggml_scale(g, text_scores, text_scale);

        // layout_scores = LQ @ LK^T / sqrt(LHD)  → [T, T, NH]
        float layout_scale = 1.0f / sqrtf((float)LHD);
        ggml_tensor* layout_scores = ggml_mul_mat(g, LK, LQ);
        layout_scores = ggml_scale(g, layout_scores, layout_scale);

        // BiACM: combine scores
        ggml_tensor* combined = ggml_add(g, text_scores, layout_scores);
        // Both streams use the same combined scores
        ggml_tensor* text_attn_probs = ggml_soft_max(g, combined);
        ggml_tensor* layout_attn_probs = ggml_soft_max(g, ggml_add(g, layout_scores, text_scores));

        // === Text attention output ===
        // V is [HD, T, NH], permute to [T, HD, NH] for matmul with probs [T, T, NH]
        ggml_tensor* V_perm = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
        ggml_tensor* text_attn = ggml_mul_mat(g, V_perm, text_attn_probs);
        // text_attn: [HD, T, NH] → permute to [HD, NH, T] → reshape [H, T]
        text_attn = ggml_cont(g, ggml_permute(g, text_attn, 0, 2, 1, 3));
        text_attn = ggml_reshape_2d(g, text_attn, H, T);

        // === Layout attention output ===
        ggml_tensor* LV_perm = ggml_cont(g, ggml_permute(g, LV, 1, 0, 2, 3));
        ggml_tensor* layout_attn = ggml_mul_mat(g, LV_perm, layout_attn_probs);
        layout_attn = ggml_cont(g, ggml_permute(g, layout_attn, 0, 2, 1, 3));
        layout_attn = ggml_reshape_2d(g, layout_attn, LD, T);

        // === Output projections ===
        text_attn = ggml_mul_mat(g, L.o_w, text_attn);
        if (L.o_b) text_attn = ggml_add(g, text_attn, L.o_b);

        layout_attn = ggml_mul_mat(g, L.lo_w, layout_attn);
        if (L.lo_b) layout_attn = ggml_add(g, layout_attn, L.lo_b);

        // === Residual + LayerNorm ===
        cur = ggml_add(g, text_inp, text_attn);
        cur = layer_norm(g, cur, L.attn_ln_w, L.attn_ln_b, eps);

        lcur = ggml_add(g, layout_inp, layout_attn);
        lcur = layer_norm(g, lcur, L.lattn_ln_w, L.lattn_ln_b, eps);

        // === Text FFN ===
        ggml_tensor* text_ffn_inp = cur;
        ggml_tensor* ffn = ggml_mul_mat(g, L.fc1_w, cur);
        if (L.fc1_b) ffn = ggml_add(g, ffn, L.fc1_b);
        ffn = ggml_gelu_erf(g, ffn);
        ffn = ggml_mul_mat(g, L.fc2_w, ffn);
        if (L.fc2_b) ffn = ggml_add(g, ffn, L.fc2_b);
        cur = ggml_add(g, text_ffn_inp, ffn);
        cur = layer_norm(g, cur, L.ffn_ln_w, L.ffn_ln_b, eps);

        // === Layout FFN ===
        ggml_tensor* layout_ffn_inp = lcur;
        ggml_tensor* lffn = ggml_mul_mat(g, L.lfc1_w, lcur);
        if (L.lfc1_b) lffn = ggml_add(g, lffn, L.lfc1_b);
        lffn = ggml_gelu(g, lffn);
        lffn = ggml_mul_mat(g, L.lfc2_w, lffn);
        if (L.lfc2_b) lffn = ggml_add(g, lffn, L.lfc2_b);
        lcur = ggml_add(g, layout_ffn_inp, lffn);
        lcur = layer_norm(g, lcur, L.lffn_ln_w, L.lffn_ln_b, eps);

        if (dump) {
            char name[32];
            snprintf(name, sizeof(name), "layer_%d_text", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            snprintf(name, sizeof(name), "layer_%d_layout", il);
            ggml_set_name(lcur, name);
            ggml_set_output(lcur);
        }
    }

    // --- Classifier head ---
    if (m.cls_w) {
        cur = ggml_mul_mat(g, m.cls_w, cur);
        if (m.cls_b) cur = ggml_add(g, cur, m.cls_b);
    }

    ggml_set_name(cur, "output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ---------------------------------------------------------------------------
// Classify
// ---------------------------------------------------------------------------

std::vector<token_result> classify(context* ctx,
                                    const int32_t* input_ids,
                                    const int32_t* bbox,  // [T * 4]
                                    int T) {
    std::vector<token_result> results;
    if (!ctx || !input_ids || !bbox || T <= 0) return results;

    auto& m = ctx->model;

    // Build graph
    ggml_cgraph* gf = build_graph(ctx, T);
    if (!gf) return results;

    // Reserve scheduler
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "lilt_kie: failed to allocate graph\n");
        return results;
    }

    // Set inputs
    // tok_ids
    ggml_tensor* t_tok = ggml_graph_get_tensor(gf, "tok_ids");
    ggml_backend_tensor_set(t_tok, input_ids, 0, T * sizeof(int32_t));

    // pos_ids: RoBERTa positions start at padding_idx + 1 = 2
    std::vector<int32_t> pos_ids(T);
    for (int i = 0; i < T; i++) pos_ids[i] = i + 2;  // RoBERTa: padding_idx=1, pos starts at 2
    ggml_tensor* t_pos = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(t_pos, pos_ids.data(), 0, T * sizeof(int32_t));

    // type_ids: all zeros
    if (m.type_embd) {
        std::vector<int32_t> type_ids(T, 0);
        ggml_tensor* t_type = ggml_graph_get_tensor(gf, "type_ids");
        ggml_backend_tensor_set(t_type, type_ids.data(), 0, T * sizeof(int32_t));
    }

    // bbox_ids: [x0_0..x0_T, y0_0..y0_T, x1_0..x1_T, y1_0..y1_T, w_0..w_T, h_0..h_T]
    std::vector<int32_t> bbox_flat(T * 6);
    for (int i = 0; i < T; i++) {
        int x0 = bbox[i * 4 + 0];
        int y0 = bbox[i * 4 + 1];
        int x1 = bbox[i * 4 + 2];
        int y1 = bbox[i * 4 + 3];
        // Clamp to [0, max_2d_pos - 1]
        x0 = std::clamp(x0, 0, m.max_2d_pos - 1);
        y0 = std::clamp(y0, 0, m.max_2d_pos - 1);
        x1 = std::clamp(x1, 0, m.max_2d_pos - 1);
        y1 = std::clamp(y1, 0, m.max_2d_pos - 1);
        int w = std::clamp(x1 - x0, 0, m.max_2d_pos - 1);
        int h = std::clamp(y1 - y0, 0, m.max_2d_pos - 1);

        bbox_flat[0 * T + i] = x0;
        bbox_flat[1 * T + i] = y0;
        bbox_flat[2 * T + i] = x1;
        bbox_flat[3 * T + i] = y1;
        bbox_flat[4 * T + i] = w;
        bbox_flat[5 * T + i] = h;
    }
    ggml_tensor* t_bbox = ggml_graph_get_tensor(gf, "bbox_ids");
    ggml_backend_tensor_set(t_bbox, bbox_flat.data(), 0, T * 6 * sizeof(int32_t));

    // Compute
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "lilt_kie: graph compute failed\n");
        return results;
    }

    // Read output
    ggml_tensor* t_out = ggml_graph_get_tensor(gf, "output");
    if (!t_out) return results;

    int out_dim = (m.cls_w) ? m.n_labels : m.n_embd;
    std::vector<float> logits(T * out_dim);
    ggml_backend_tensor_get(t_out, logits.data(), 0, logits.size() * sizeof(float));

    // Convert logits to predictions
    results.resize(T);
    for (int i = 0; i < T; i++) {
        const float* row = logits.data() + i * out_dim;
        int best = 0;
        float best_val = row[0];
        for (int j = 1; j < out_dim; j++) {
            if (row[j] > best_val) { best_val = row[j]; best = j; }
        }

        // Softmax for score
        float max_val = *std::max_element(row, row + out_dim);
        float sum = 0.0f;
        for (int j = 0; j < out_dim; j++) sum += expf(row[j] - max_val);
        float score = expf(row[best] - max_val) / sum;

        results[i].token_id = input_ids[i];
        results[i].label_id = best;
        results[i].score    = score;
        auto it = m.id2label.find(best);
        results[i].label = (it != m.id2label.end()) ? it->second : "LABEL_" + std::to_string(best);
    }

    return results;
}

std::vector<dump_tensor> classify_dump(context* ctx,
                                        const int32_t* input_ids,
                                        const int32_t* bbox,
                                        int T) {
    std::vector<dump_tensor> dumps;
    if (!ctx || !input_ids || !bbox || T <= 0) return dumps;

    auto& m = ctx->model;

    // Build graph with dump=true
    ggml_cgraph* gf = build_graph(ctx, T, /*dump=*/true);
    if (!gf) return dumps;

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return dumps;

    // Set inputs (same as classify)
    ggml_tensor* t_tok = ggml_graph_get_tensor(gf, "tok_ids");
    ggml_backend_tensor_set(t_tok, input_ids, 0, T * sizeof(int32_t));

    std::vector<int32_t> pos_ids(T);
    for (int i = 0; i < T; i++) pos_ids[i] = i + 2;
    ggml_tensor* t_pos = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(t_pos, pos_ids.data(), 0, T * sizeof(int32_t));

    if (m.type_embd) {
        std::vector<int32_t> type_ids(T, 0);
        ggml_tensor* t_type = ggml_graph_get_tensor(gf, "type_ids");
        ggml_backend_tensor_set(t_type, type_ids.data(), 0, T * sizeof(int32_t));
    }

    std::vector<int32_t> bbox_flat(T * 6);
    for (int i = 0; i < T; i++) {
        int x0 = std::clamp(bbox[i*4+0], 0, m.max_2d_pos-1);
        int y0 = std::clamp(bbox[i*4+1], 0, m.max_2d_pos-1);
        int x1 = std::clamp(bbox[i*4+2], 0, m.max_2d_pos-1);
        int y1 = std::clamp(bbox[i*4+3], 0, m.max_2d_pos-1);
        bbox_flat[0*T+i] = x0; bbox_flat[1*T+i] = y0;
        bbox_flat[2*T+i] = x1; bbox_flat[3*T+i] = y1;
        bbox_flat[4*T+i] = std::clamp(x1-x0, 0, m.max_2d_pos-1);
        bbox_flat[5*T+i] = std::clamp(y1-y0, 0, m.max_2d_pos-1);
    }
    ggml_tensor* t_bbox = ggml_graph_get_tensor(gf, "bbox_ids");
    ggml_backend_tensor_set(t_bbox, bbox_flat.data(), 0, T * 6 * sizeof(int32_t));

    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        return dumps;
    }

    // Read all named dump tensors
    const char* dump_names[] = {
        "text_embed", "layout_embed", "output",
    };
    for (auto name : dump_names) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) continue;
        int n = (int)ggml_nelements(t);
        dump_tensor dt;
        dt.name = name;
        dt.n_elem = n;
        dt.data.resize(n);
        ggml_backend_tensor_get(t, dt.data.data(), 0, n * sizeof(float));
        dumps.push_back(std::move(dt));
    }
    // Per-layer dumps
    for (int il = 0; il < m.n_layer; il++) {
        char name[32];
        for (const char* suffix : {"text", "layout"}) {
            snprintf(name, sizeof(name), "layer_%d_%s", il, suffix);
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t) continue;
            int n = (int)ggml_nelements(t);
            dump_tensor dt;
            dt.name = name;
            dt.n_elem = n;
            dt.data.resize(n);
            ggml_backend_tensor_get(t, dt.data.data(), 0, n * sizeof(float));
            dumps.push_back(std::move(dt));
        }
    }

    return dumps;
}

const char* label_name(context* ctx, int label_id) {
    if (!ctx) return "";
    auto it = ctx->model.id2label.find(label_id);
    return (it != ctx->model.id2label.end()) ? it->second.c_str() : "";
}

int num_labels(context* ctx) {
    return ctx ? ctx->model.n_labels : 0;
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

} // namespace lilt_kie
