// decoder_embed.cpp — Qwen3/LLaMA decoder embedding graph via ggml.

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
    // Read metadata
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

    gguf_free(g);

    // Load weights
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
    }

    fprintf(stderr, "decoder_embed: loaded %d layers, %d dims, %d vocab, %d heads (%d kv), rope_theta=%.0f\n",
            m.n_layer, m.n_embd, m.n_vocab, m.n_head, m.n_kv_head, m.rope_theta);
    return true;
}

// ---------------------------------------------------------------------------
// Encode tokens via decoder graph (layer-by-layer for memory efficiency)
// ---------------------------------------------------------------------------

std::vector<float> decoder_encode_tokens(
    const dec_model & m, ggml_backend_t backend,
    const embed_tokens & tokens, int n_threads) {

    const int T = (int)tokens.ids.size();
    const int H = m.n_embd;
    const int n_heads = m.n_head;
    const int n_kv_heads = m.n_kv_head;
    // Detect head_dim from Q weight shape
    // q_w in ggml has ne[0]=H (input), ne[1]=q_dim (output)
    // q_dim = n_heads * head_dim
    int q_dim = m.layers[0].q_w ? (int)m.layers[0].q_w->ne[1] : H;
    const int head_dim = q_dim / n_heads;
    const int kv_dim = n_kv_heads * head_dim;
    const float eps = m.rms_norm_eps;

    // Allocate hidden state [T, H]
    std::vector<float> hidden(T * H, 0.0f);

    // Token embedding lookup (manual, from backend tensor)
    for (int t = 0; t < T; t++) {
        int tid = tokens.ids[t];
        if (tid >= 0 && tid < m.n_vocab) {
            ggml_backend_tensor_get(m.token_embd, hidden.data() + t * H,
                                     (size_t)tid * H * sizeof(float), H * sizeof(float));
        }
    }

    // Debug: dump embedding output
    fprintf(stderr, "decoder: embed[0,:4]: %.6f %.6f %.6f %.6f\n",
            hidden[0], hidden[1], hidden[2], hidden[3]);
    fprintf(stderr, "decoder: token IDs:");
    for (int t = 0; t < std::min(T, 5); t++) fprintf(stderr, " %d", tokens.ids[t]);
    fprintf(stderr, "\n");

    // Layer-by-layer processing (reusing CrispASR pattern)
    size_t layer_mem = (size_t)H * T * 4 * 30
                     + (size_t)T * T * n_heads * 4 * 2
                     + (size_t)m.n_intermediate * T * 4
                     + ggml_tensor_overhead() * 300
                     + ggml_graph_overhead_custom(2048, false)
                     + 32 * 1024 * 1024;

    std::vector<uint8_t> lbuf(layer_mem);
    std::vector<uint8_t> work_buf;

    for (int il = 0; il < m.n_layer; il++) {
        const auto & L = m.layers[il];

        ggml_init_params lip = { layer_mem, lbuf.data(), false };
        ggml_context * lctx = ggml_init(lip);
        ggml_cgraph * lgf = ggml_new_graph_custom(lctx, 2048, false);

        // Input [H, T] (ggml layout = C's [T][H] row-major)
        ggml_tensor * cur = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, H, T);
        memcpy(cur->data, hidden.data(), H * T * sizeof(float));

        ggml_tensor * residual = cur;

        // Debug: save tensor pointers for post-compute inspection
        ggml_tensor * dbg_input_ln = nullptr;
        ggml_tensor * dbg_q_proj = nullptr;
        ggml_tensor * dbg_k_proj = nullptr;
        ggml_tensor * dbg_q_norm = nullptr;
        ggml_tensor * dbg_k_norm = nullptr;
        ggml_tensor * dbg_o_proj = nullptr;

        // RMSNorm
        if (L.attn_norm_w) {
            cur = ggml_rms_norm(lctx, cur, eps);
            cur = ggml_mul(lctx, cur, L.attn_norm_w);
        }
        if (il == 0) dbg_input_ln = cur;

        // Q/K/V projections
        ggml_tensor * Q = ggml_mul_mat(lctx, L.q_w, cur);
        if (L.q_b) Q = ggml_add(lctx, Q, L.q_b);
        if (il == 0) dbg_q_proj = Q;
        ggml_tensor * K = ggml_mul_mat(lctx, L.k_w, cur);
        if (L.k_b) K = ggml_add(lctx, K, L.k_b);
        if (il == 0) dbg_k_proj = K;
        ggml_tensor * V = ggml_mul_mat(lctx, L.v_w, cur);
        if (L.v_b) V = ggml_add(lctx, V, L.v_b);

        // Reshape for multi-head
        Q = ggml_reshape_3d(lctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(lctx, K, head_dim, n_kv_heads, T);
        V = ggml_reshape_3d(lctx, V, head_dim, n_kv_heads, T);

        // QK norm (Qwen3)
        if (L.q_norm_w) {
            Q = ggml_rms_norm(lctx, Q, eps);
            Q = ggml_mul(lctx, Q, L.q_norm_w);
        }
        if (il == 0) dbg_q_norm = Q;
        if (L.k_norm_w) {
            K = ggml_rms_norm(lctx, K, eps);
            K = ggml_mul(lctx, K, L.k_norm_w);
        }
        if (il == 0) dbg_k_norm = K;

        // RoPE on [head_dim, n_heads, T] — ne[2]=T matches ggml_rope_ext's
        // position dimension requirement. Applied BEFORE permute.
        // HF's equivalent: transpose(1,2) puts T in dim 2 of [B,n_heads,T,hd],
        // and ggml's layout [hd,n_heads,T] has T as ne[2] — same dimension.
        {
            ggml_tensor * pos = ggml_new_tensor_1d(lctx, GGML_TYPE_I32, T);
            for (int t = 0; t < T; t++)
                ((int32_t *)pos->data)[t] = t;

            int rope_mode = 2;  // GGML_ROPE_TYPE_NEOX
            Q = ggml_rope_ext(lctx, Q, pos, nullptr,
                               head_dim, rope_mode, 0,
                               m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(lctx, K, pos, nullptr,
                               head_dim, rope_mode, 0,
                               m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // Permute: [head_dim, n_heads, T] → [head_dim, T, n_heads]
        Q = ggml_cont(lctx, ggml_permute(lctx, Q, 0, 2, 1, 3));
        K = ggml_cont(lctx, ggml_permute(lctx, K, 0, 2, 1, 3));
        V = ggml_cont(lctx, ggml_permute(lctx, V, 0, 2, 1, 3));

        // GQA: repeat K/V heads to match Q heads
        // After permute: K/V are [head_dim, T, n_kv_heads]
        // Need: [head_dim, T, n_heads]
        // GQA: ggml_mul_mat broadcasts ne[2] automatically when
        // Q has n_heads and K/V have n_kv_heads (n_heads % n_kv_heads == 0).
        // No explicit repeat needed — mul_mat handles it.

        // Attention scores with causal mask
        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * scores = ggml_mul_mat(lctx, K, Q);
        scores = ggml_scale(lctx, scores, scale);

        // Causal mask: mask future tokens with -inf
        scores = ggml_diag_mask_inf(lctx, scores, 0);

        scores = ggml_soft_max(lctx, scores);

        // V @ scores
        ggml_tensor * V_perm = ggml_cont(lctx, ggml_permute(lctx, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(lctx, V_perm, scores);

        // Reshape back to [q_dim, T] (q_dim = n_heads * head_dim, may differ from H for GQA)
        attn = ggml_cont(lctx, ggml_permute(lctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(lctx, attn, q_dim, T);

        // Output projection
        attn = ggml_mul_mat(lctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(lctx, attn, L.o_b);
        if (il == 0) dbg_o_proj = attn;

        // Residual add
        cur = ggml_add(lctx, residual, attn);

        // FFN: RMSNorm → SwiGLU → residual
        residual = cur;
        if (L.ffn_norm_w) {
            cur = ggml_rms_norm(lctx, cur, eps);
            cur = ggml_mul(lctx, cur, L.ffn_norm_w);
        }

        if (L.gate_w && L.up_w && L.down_w) {
            // SwiGLU: down(silu(gate(x)) * up(x))
            ggml_tensor * gate = ggml_mul_mat(lctx, L.gate_w, cur);
            gate = ggml_silu(lctx, gate);
            ggml_tensor * up = ggml_mul_mat(lctx, L.up_w, cur);
            ggml_tensor * ffn = ggml_mul(lctx, gate, up);
            ffn = ggml_mul_mat(lctx, L.down_w, ffn);
            cur = ggml_add(lctx, residual, ffn);
        } else {
            cur = residual;
        }

        ggml_set_name(cur, "layer_out");
        ggml_build_forward_expand(lgf, cur);

        struct ggml_cplan cplan = ggml_graph_plan(lgf, n_threads, NULL);
        if (cplan.work_size > 0) {
            if (work_buf.size() < cplan.work_size) work_buf.resize(cplan.work_size);
            cplan.work_data = work_buf.data();
        }
        ggml_graph_compute(lgf, &cplan);

        ggml_tensor * lout = ggml_graph_get_tensor(lgf, "layer_out");
        memcpy(hidden.data(), lout->data, H * T * sizeof(float));

        if (il == 0) {
            auto dump = [](const char * label, ggml_tensor * t) {
                if (!t || !t->data) { fprintf(stderr, "  %-16s: (null)\n", label); return; }
                const float * d = (const float *)t->data;
                fprintf(stderr, "  %-16s:", label);
                for (int i = 0; i < std::min(8, (int)(t->ne[0])); i++)
                    fprintf(stderr, " %.6f", d[i]);
                fprintf(stderr, "\n");
            };
            fprintf(stderr, "decoder L0 intermediates (t=0):\n");
            dump("input_ln", dbg_input_ln);
            dump("q_proj", dbg_q_proj);
            dump("k_proj", dbg_k_proj);
            dump("q_norm", dbg_q_norm);
            dump("k_norm", dbg_k_norm);
            dump("o_proj", dbg_o_proj);
            fprintf(stderr, "  %-16s: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    "layer_out",
                    hidden[0], hidden[1], hidden[2], hidden[3],
                    hidden[4], hidden[5], hidden[6], hidden[7]);
        }

        ggml_free(lctx);
    }

    // Final RMSNorm
    if (m.output_norm) {
        // Manual RMSNorm since we're outside a graph
        const float * nw = nullptr;
        std::vector<float> norm_w(H);
        ggml_backend_tensor_get(m.output_norm, norm_w.data(), 0, H * sizeof(float));

        for (int t = 0; t < T; t++) {
            float * h = hidden.data() + t * H;
            float ss = 0;
            for (int i = 0; i < H; i++) ss += h[i] * h[i];
            ss = 1.0f / sqrtf(ss / H + eps);
            for (int i = 0; i < H; i++) h[i] = h[i] * ss * norm_w[i];
        }
    }

    // Last-token pooling (take the last non-padding token)
    int last_t = 0;
    for (int t = T - 1; t >= 0; t--) {
        if (tokens.attn_mask[t]) { last_t = t; break; }
    }

    std::vector<float> pooled(H);
    memcpy(pooled.data(), hidden.data() + last_t * H, H * sizeof(float));

    // L2 normalize
    float norm = 0;
    for (int i = 0; i < H; i++) norm += pooled[i] * pooled[i];
    norm = sqrtf(std::max(norm, 1e-12f));
    for (int i = 0; i < H; i++) pooled[i] /= norm;

    return pooled;
}
