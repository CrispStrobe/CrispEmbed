// decoder_embed.cpp — Qwen3/LLaMA/Gemma3 decoder embedding graph via ggml.
// Uses ggml_backend_sched for GPU dispatch (same pattern as encoder).

#include "decoder_embed_internal.h"
#include "crispembed.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

bool load_decoder_model(dec_model & m, core_gguf::WeightLoad & wl,
                         const char * path, ggml_backend_t backend) {
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;

    auto u32 = [&](const char * key, int def) -> int {
        int k = gguf_find_key(g, key);
        return k >= 0 ? (int)gguf_get_val_u32(g, k) : def;
    };
    auto f32v = [&](const char * key, float def) -> float {
        int k = gguf_find_key(g, key);
        return k >= 0 ? gguf_get_val_f32(g, k) : def;
    };

    m.n_vocab = u32("decoder.vocab_size", 151669);
    m.n_embd = u32("decoder.hidden_size", 1024);
    m.n_head = u32("decoder.num_attention_heads", 16);
    m.n_kv_head = u32("decoder.num_key_value_heads", m.n_head);
    m.n_layer = u32("decoder.num_hidden_layers", 28);
    m.n_intermediate = u32("decoder.intermediate_size", 3072);
    m.n_max_pos = u32("decoder.max_position_embeddings", 8192);
    m.rms_norm_eps = f32v("decoder.rms_norm_eps", 1e-6f);
    m.rope_theta = f32v("decoder.rope_theta", 10000.0f);
    m.is_bidirectional = u32("decoder.is_bidirectional", 0) != 0;
    m.pooling_method = u32("decoder.pooling_method", 2);
    m.activation = u32("decoder.activation", 0);
    m.head_dim = u32("decoder.head_dim", 0);
    m.attn_scale = f32v("decoder.attn_scale", 0.0f);
    m.embed_scale = f32v("decoder.embed_scale", 1.0f);
    m.gemma_norm = u32("decoder.gemma_norm", 0) != 0;

    gguf_free(g);

    if (!core_gguf::load_weights(path, backend, "decoder_embed", wl))
        return false;

    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = wl.tensors.find(n);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    m.token_embd = get("token_embd.weight");
    m.output_norm = get("output_norm.weight");

    m.layers.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; i++) {
        auto p = "dec." + std::to_string(i) + ".";
        auto & L = m.layers[i];
        L.attn_norm_w = get(p + "attn_norm.weight");
        L.q_w = get(p + "attn.q.weight"); L.q_b = get(p + "attn.q.bias");
        L.k_w = get(p + "attn.k.weight"); L.k_b = get(p + "attn.k.bias");
        L.v_w = get(p + "attn.v.weight"); L.v_b = get(p + "attn.v.bias");
        L.o_w = get(p + "attn.o.weight"); L.o_b = get(p + "attn.o.bias");
        L.q_norm_w = get(p + "attn.q_norm.weight");
        L.k_norm_w = get(p + "attn.k_norm.weight");
        L.ffn_norm_w = get(p + "ffn_norm.weight");
        L.gate_w = get(p + "ffn.gate.weight");
        L.up_w = get(p + "ffn.up.weight");
        L.down_w = get(p + "ffn.down.weight");
        L.pre_ffn_norm_w = get(p + "pre_ffn_norm.weight");
        L.post_ffn_norm_w = get(p + "post_ffn_norm.weight");
    }

    const char * pool_str = (m.pooling_method == 1) ? "mean" : "last-token";
    fprintf(stderr, "decoder_embed: loaded %d layers, %d dims, %d vocab, %d heads (%d kv), rope_theta=%.0f, pool=%s%s\n",
            m.n_layer, m.n_embd, m.n_vocab, m.n_head, m.n_kv_head, m.rope_theta,
            pool_str, m.is_bidirectional ? ", bidirectional" : "");
    return true;
}

// ---------------------------------------------------------------------------
// Full-graph decoder with scheduler support (GPU + CPU)
// ---------------------------------------------------------------------------

std::vector<float> decoder_encode_tokens(
    const dec_model & m, ggml_backend_t backend,
    const embed_tokens & tokens, int n_threads,
    ggml_backend_sched_t sched,
    std::vector<uint8_t> * compute_meta) {

    const int T = (int)tokens.ids.size();
    const int H = m.n_embd;
    const int n_heads = m.n_head;
    const int n_kv_heads = m.n_kv_head;
    int q_dim = m.layers[0].q_w ? (int)m.layers[0].q_w->ne[1] : H;
    const int head_dim = (m.head_dim > 0) ? m.head_dim : (q_dim / n_heads);
    q_dim = n_heads * head_dim;
    const float eps = m.rms_norm_eps;

    // Graph context: no_alloc=true when using scheduler, false otherwise
    bool use_sched = (sched != nullptr && compute_meta != nullptr);
    int graph_size = std::max(4096, m.n_layer * 50 + 256);

    size_t mem;
    std::vector<uint8_t> local_buf;
    uint8_t * buf_ptr;

    if (use_sched) {
        mem = compute_meta->size();
        buf_ptr = compute_meta->data();
    } else {
        size_t per_layer = (size_t)H * T * 4 * 30
                         + (size_t)T * T * n_heads * 4 * 2
                         + (size_t)m.n_intermediate * T * 4 * 3;
        mem = per_layer * m.n_layer
            + (size_t)H * T * 4 * 10
            + ggml_tensor_overhead() * (size_t)(m.n_layer * 50 + 200)
            + ggml_graph_overhead_custom(graph_size, false)
            + 64 * 1024 * 1024;
        local_buf.resize(mem);
        buf_ptr = local_buf.data();
    }

    ggml_init_params ip = { mem, buf_ptr, use_sched };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // --- Token embedding ---
    ggml_tensor * ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(ids_t, "tok_ids");
    ggml_set_input(ids_t);

    ggml_tensor * cur = ggml_get_rows(gctx, m.token_embd, ids_t);

    if (m.embed_scale != 1.0f) {
        cur = ggml_scale(gctx, cur, m.embed_scale);
    }

    // Gemma3 ones tensor for (1 + weight) RMSNorm
    ggml_tensor * ones_h = nullptr;
    ggml_tensor * ones_hd = nullptr;
    if (m.gemma_norm) {
        ones_h = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, H);
        ggml_set_name(ones_h, "ones_h");
        ggml_set_input(ones_h);  // will set to 1.0f
        if (m.layers[0].q_norm_w) {
            ones_hd = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, head_dim);
            ggml_set_name(ones_hd, "ones_hd");
            ggml_set_input(ones_hd);
        }
    }

    auto rms_norm = [&](ggml_tensor * x, ggml_tensor * w) -> ggml_tensor * {
        x = ggml_rms_norm(gctx, x, eps);
        if (m.gemma_norm) {
            ggml_tensor * ones = (w->ne[0] == H) ? ones_h : ones_hd;
            return ggml_mul(gctx, x, ggml_add(gctx, w, ones));
        }
        return ggml_mul(gctx, x, w);
    };

    // --- Position IDs for RoPE ---
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    // --- Transformer layers ---
    for (int il = 0; il < m.n_layer; il++) {
        const auto & L = m.layers[il];
        ggml_tensor * residual = cur;

        if (L.attn_norm_w) cur = rms_norm(cur, L.attn_norm_w);

        ggml_tensor * Q = ggml_mul_mat(gctx, L.q_w, cur);
        if (L.q_b) Q = ggml_add(gctx, Q, L.q_b);
        ggml_tensor * K = ggml_mul_mat(gctx, L.k_w, cur);
        if (L.k_b) K = ggml_add(gctx, K, L.k_b);
        ggml_tensor * V = ggml_mul_mat(gctx, L.v_w, cur);
        if (L.v_b) V = ggml_add(gctx, V, L.v_b);

        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, T);
        V = ggml_reshape_3d(gctx, V, head_dim, n_kv_heads, T);

        if (L.q_norm_w) Q = rms_norm(Q, L.q_norm_w);
        if (L.k_norm_w) K = rms_norm(K, L.k_norm_w);

        int rope_mode = 2;
        Q = ggml_rope_ext(gctx, Q, pos, nullptr,
                           head_dim, rope_mode, 0,
                           m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(gctx, K, pos, nullptr,
                           head_dim, rope_mode, 0,
                           m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Attention: permute → scores → mask → softmax → value
        Q = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

        float scale = (m.attn_scale > 0.0f)
                        ? (1.0f / sqrtf(m.attn_scale))
                        : (1.0f / sqrtf((float)head_dim));
        ggml_tensor * scores = ggml_mul_mat(gctx, K, Q);
        scores = ggml_scale(gctx, scores, scale);
        if (!m.is_bidirectional) {
            scores = ggml_diag_mask_inf(gctx, scores, 0);
        }
        scores = ggml_soft_max(gctx, scores);

        ggml_tensor * V_perm = ggml_cont(gctx, ggml_permute(gctx, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(gctx, V_perm, scores);
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(gctx, attn, q_dim, T);

        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);

        if (L.pre_ffn_norm_w) {
            if (L.ffn_norm_w) attn = rms_norm(attn, L.ffn_norm_w);
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            cur = rms_norm(cur, L.pre_ffn_norm_w);
        } else {
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            if (L.ffn_norm_w) cur = rms_norm(cur, L.ffn_norm_w);
        }

        if (L.gate_w && L.up_w && L.down_w) {
            ggml_tensor * gate = ggml_mul_mat(gctx, L.gate_w, cur);
            if (m.activation == 2 || m.activation == 1) {
                gate = ggml_gelu(gctx, gate);
            } else {
                gate = ggml_silu(gctx, gate);
            }
            ggml_tensor * up = ggml_mul_mat(gctx, L.up_w, cur);
            ggml_tensor * ffn = ggml_mul(gctx, gate, up);
            ffn = ggml_mul_mat(gctx, L.down_w, ffn);
            if (L.post_ffn_norm_w) ffn = rms_norm(ffn, L.post_ffn_norm_w);
            cur = ggml_add(gctx, residual, ffn);
        } else {
            cur = residual;
        }
    }

    if (m.output_norm) cur = rms_norm(cur, m.output_norm);

    ggml_set_name(cur, "decoder_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // --- Set inputs and compute ---
    if (use_sched) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "decoder: failed to allocate graph\n");
            ggml_free(gctx);
            return {};
        }

        // Set input data
        std::vector<int32_t> tok_data(tokens.ids.begin(), tokens.ids.end());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_ids"),
                                tok_data.data(), 0, T * sizeof(int32_t));

        std::vector<int32_t> pos_data(T);
        for (int t = 0; t < T; t++) pos_data[t] = t;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                                pos_data.data(), 0, T * sizeof(int32_t));

        if (m.gemma_norm) {
            std::vector<float> ones(H, 1.0f);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_h"),
                                    ones.data(), 0, H * sizeof(float));
            if (ones_hd) {
                std::vector<float> ones_hd_data(head_dim, 1.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_hd"),
                                        ones_hd_data.data(), 0, head_dim * sizeof(float));
            }
        }

        // Compute via scheduler
        ggml_backend_sched_graph_compute(sched, gf);
    } else {
        // CPU fallback (set data directly)
        int32_t * id = (int32_t *)ids_t->data;
        for (int t = 0; t < T; t++) id[t] = tokens.ids[t];

        int32_t * pd = (int32_t *)pos->data;
        for (int t = 0; t < T; t++) pd[t] = t;

        if (ones_h) {
            float * d = (float *)ones_h->data;
            for (int i = 0; i < H; i++) d[i] = 1.0f;
        }
        if (ones_hd) {
            float * d = (float *)ones_hd->data;
            for (int i = 0; i < head_dim; i++) d[i] = 1.0f;
        }

        struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, NULL);
        std::vector<uint8_t> work;
        if (cplan.work_size > 0) {
            work.resize(cplan.work_size);
            cplan.work_data = work.data();
        }
        ggml_graph_compute(gf, &cplan);
    }

    // --- Read output ---
    ggml_tensor * out = ggml_graph_get_tensor(gf, "decoder_out");
    std::vector<float> hidden(H * T);
    if (use_sched) {
        ggml_backend_tensor_get(out, hidden.data(), 0, H * T * sizeof(float));
    } else {
        memcpy(hidden.data(), out->data, H * T * sizeof(float));
    }

    ggml_free(gctx);

    std::vector<float> pooled(H, 0.0f);

    if (m.pooling_method == 1) {
        // Mean pooling over non-padding tokens (BidirLM-style)
        int n_valid = 0;
        for (int t = 0; t < T; t++) {
            if (!tokens.attn_mask[t]) continue;
            const float * row = hidden.data() + (size_t)t * H;
            for (int i = 0; i < H; i++) pooled[i] += row[i];
            n_valid++;
        }
        if (n_valid > 0) {
            const float inv = 1.0f / (float)n_valid;
            for (int i = 0; i < H; i++) pooled[i] *= inv;
        }
    } else {
        // Last-token pooling (Qwen3/Gemma3)
        int last_t = 0;
        for (int t = T - 1; t >= 0; t--) {
            if (tokens.attn_mask[t]) { last_t = t; break; }
        }
        memcpy(pooled.data(), hidden.data() + (size_t)last_t * H, H * sizeof(float));
    }

    // L2 normalize
    float norm = 0;
    for (int i = 0; i < H; i++) norm += pooled[i] * pooled[i];
    norm = sqrtf(std::max(norm, 1e-12f));
    for (int i = 0; i < H; i++) pooled[i] /= norm;

    return pooled;
}
