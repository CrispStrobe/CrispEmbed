// clip_text_embed.cpp — CLIP text encoder.
//
// Pre-LN transformer with causal attention mask, EOS-position pooling,
// and text_projection. Uses quick_gelu activation (x * sigmoid(1.702x)).
//
// Architecture (same as CLIP vision but with text input):
//   token_embd + pos_embd → N × (LN→CausalAttn→LN→MLP) →
//   final_ln → pool[EOS] → text_proj → L2 normalize

#include "clip_text_embed.h"
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
#include <cstring>
#include <string>
#include <vector>

namespace clip_text {

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

struct context {
    int hidden = 0;
    int n_layers = 0;
    int n_heads = 0;
    int intermediate = 0;
    int vocab_size = 0;
    int max_pos = 77;
    int proj_dim = 0;
    float ln_eps = 1e-5f;
    bool use_quick_gelu = true;
    bool causal = true;       // CLIP=causal, SigLIP=bidirectional
    bool has_head = false;     // SigLIP has head projection

    // Tokenizer (BPE for CLIP, SentencePiece for SigLIP)
    BPETokenizer bpe_tokenizer;
    SentencePieceTokenizer sp_tokenizer;
    bool use_sp = false;
    int bos_id = 49406;
    int eos_id = 49407;

    // Weights
    ggml_tensor * token_embd = nullptr;
    ggml_tensor * pos_embd = nullptr;
    ggml_tensor * final_ln_w = nullptr, * final_ln_b = nullptr;
    ggml_tensor * text_proj_w = nullptr;
    ggml_tensor * head_w = nullptr, * head_b = nullptr;  // SigLIP head
    std::vector<layer> layers;

    // Backend
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    int n_threads = 4;
};

bool load(context** out, const char* path, int n_threads) {
    auto* ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;

    gguf_context* g = core_gguf::open_metadata(path);
    if (!g) { fprintf(stderr, "clip_text: cannot open %s\n", path); return false; }

    auto u32 = [&](const char* k, int d) -> int {
        int64_t i = gguf_find_key(g, k); return i >= 0 ? (int)gguf_get_val_u32(g, i) : d;
    };
    auto f32v = [&](const char* k, float d) -> float {
        int64_t i = gguf_find_key(g, k); return i >= 0 ? gguf_get_val_f32(g, i) : d;
    };
    auto str_val = [&](const char* k, const char* d) -> std::string {
        int64_t i = gguf_find_key(g, k); return i >= 0 ? gguf_get_val_str(g, i) : d;
    };

    ctx->hidden       = u32("clip_text.hidden_size", 512);
    ctx->n_layers     = u32("clip_text.num_hidden_layers", 12);
    ctx->n_heads      = u32("clip_text.num_attention_heads", 8);
    ctx->intermediate = u32("clip_text.intermediate_size", 2048);
    ctx->vocab_size   = u32("clip_text.vocab_size", 49408);
    ctx->max_pos      = u32("clip_text.max_position_embeddings", 77);
    ctx->proj_dim     = u32("clip_text.projection_dim", 512);
    ctx->ln_eps       = f32v("clip_text.layer_norm_eps", 1e-5f);
    ctx->bos_id       = u32("clip_text.bos_token_id", 49406);
    ctx->eos_id       = u32("clip_text.eos_token_id", 49407);
    ctx->use_quick_gelu = (str_val("clip_text.hidden_act", "quick_gelu") == "quick_gelu");

    // CLIP uses causal attention, SigLIP uses bidirectional
    {   auto bv = [&](const char* k, bool d) -> bool {
            int64_t i = gguf_find_key(g, k);
            return i >= 0 ? gguf_get_val_bool(g, i) : d;
        };
        ctx->causal = bv("clip_text.causal", true);
        ctx->has_head = bv("clip_text.has_head", false);
    }

    // Read tokenizer vocab and merges from GGUF metadata
    std::vector<std::string> vocab;
    std::vector<std::string> merges;

    int64_t tok_key = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = (int)gguf_get_arr_n(g, tok_key);
        vocab.resize(n);
        for (int i = 0; i < n; i++) {
            vocab[i] = gguf_get_arr_str(g, tok_key, i);
        }
    }

    int64_t merge_key = gguf_find_key(g, "tokenizer.ggml.merges");
    if (merge_key >= 0) {
        int n = (int)gguf_get_arr_n(g, merge_key);
        merges.resize(n);
        for (int i = 0; i < n; i++) {
            merges[i] = gguf_get_arr_str(g, merge_key, i);
        }
    }

    // Detect tokenizer type (before freeing metadata)
    int tok_type = 1; // default BPE
    {   int64_t i = gguf_find_key(g, "tokenizer.ggml.type");
        if (i >= 0) tok_type = (int)gguf_get_val_u32(g, i);
    }
    ctx->use_sp = (tok_type == 2) || (!ctx->causal);

    // Read scores
    std::vector<float> scores;
    {   int64_t sk = gguf_find_key(g, "tokenizer.ggml.scores");
        if (sk >= 0) {
            int n = (int)gguf_get_arr_n(g, sk);
            scores.resize(n);
            const float* src = (const float*)gguf_get_arr_data(g, sk);
            for (int i = 0; i < n; i++) scores[i] = src[i];
        }
    }

    core_gguf::free_metadata(g);

    // Initialize tokenizer
    if (ctx->use_sp && !vocab.empty()) {
        ctx->sp_tokenizer.load(vocab, scores, -1, ctx->eos_id, 2, 0, ctx->max_pos);
        fprintf(stderr, "clip_text: SentencePiece tokenizer loaded (%zu vocab)\n", vocab.size());
    } else if (!vocab.empty() && !merges.empty()) {
        ctx->bpe_tokenizer.load(vocab, merges, ctx->eos_id, ctx->eos_id, -1,
                               ctx->bos_id, false, ctx->max_pos);
        fprintf(stderr, "clip_text: BPE tokenizer loaded (%zu vocab, %zu merges)\n",
                vocab.size(), merges.size());
    } else {
        fprintf(stderr, "clip_text: WARNING — no tokenizer in GGUF\n");
    }

    fprintf(stderr, "clip_text: hidden=%d layers=%d heads=%d vocab=%d max_pos=%d proj=%d\n",
            ctx->hidden, ctx->n_layers, ctx->n_heads, ctx->vocab_size,
            ctx->max_pos, ctx->proj_dim);

    // Load weights
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);

    if (!core_gguf::load_weights(path, ctx->backend, "clip_text", ctx->wl)) {
        fprintf(stderr, "clip_text: failed to load weights\n");
        return false;
    }

    auto get = [&](const std::string& n) -> ggml_tensor* {
        auto it = ctx->wl.tensors.find(n);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    ctx->token_embd  = get("token_embd.weight");
    ctx->pos_embd    = get("position_embd.weight");
    ctx->final_ln_w  = get("final_ln.weight");
    ctx->final_ln_b  = get("final_ln.bias");
    ctx->text_proj_w = get("text_proj.weight");
    ctx->head_w      = get("head.weight");
    ctx->head_b      = get("head.bias");

    if (!ctx->token_embd || !ctx->pos_embd) {
        fprintf(stderr, "clip_text: missing token_embd or position_embd\n");
        return false;
    }

    ctx->layers.resize(ctx->n_layers);
    for (int i = 0; i < ctx->n_layers; i++) {
        auto pfx = "enc." + std::to_string(i) + ".";
        auto& L = ctx->layers[i];
        L.ln1_w = get(pfx + "ln1.weight"); L.ln1_b = get(pfx + "ln1.bias");
        L.q_w = get(pfx + "attn.q.weight"); L.q_b = get(pfx + "attn.q.bias");
        L.k_w = get(pfx + "attn.k.weight"); L.k_b = get(pfx + "attn.k.bias");
        L.v_w = get(pfx + "attn.v.weight"); L.v_b = get(pfx + "attn.v.bias");
        L.o_w = get(pfx + "attn.o.weight"); L.o_b = get(pfx + "attn.o.bias");
        L.ln2_w = get(pfx + "ln2.weight"); L.ln2_b = get(pfx + "ln2.bias");
        L.fc1_w = get(pfx + "ffn.fc1.weight"); L.fc1_b = get(pfx + "ffn.fc1.bias");
        L.fc2_w = get(pfx + "ffn.fc2.weight"); L.fc2_b = get(pfx + "ffn.fc2.bias");
        if (!L.ln1_w || !L.q_w || !L.fc1_w) {
            fprintf(stderr, "clip_text: missing tensors for layer %d\n", i);
            return false;
        }
    }

    fprintf(stderr, "clip_text: loaded %d layers\n", ctx->n_layers);
    return true;
}

std::vector<float> encode(context* ctx, const char* text) {
    if (!ctx || !text) return {};

    // Tokenize
    embed_tokens toks;
    if (ctx->use_sp) {
        toks = ctx->sp_tokenizer.encode(text);
        // Remove invalid BOS token if present (SigLIP has no BOS)
        if (!toks.ids.empty() && toks.ids[0] < 0) {
            toks.ids.erase(toks.ids.begin());
            if (!toks.type_ids.empty()) toks.type_ids.erase(toks.type_ids.begin());
        }
    } else {
        toks = ctx->bpe_tokenizer.encode(text);
    }
    int T = (int)toks.ids.size();
    if (T == 0) return {};
    if (T > ctx->max_pos) T = ctx->max_pos;


    // Find EOS position for CLIP (last occurrence of eos_id)
    int eos_pos = T - 1;
    if (ctx->causal) {
        for (int i = T - 1; i >= 0; i--) {
            if (toks.ids[i] == ctx->eos_id) { eos_pos = i; break; }
        }
    }

    const int D = ctx->hidden;
    const int nh = ctx->n_heads;
    const int hd = D / nh;
    const float eps = ctx->ln_eps;

    // Build ggml graph
    const int ops_per_layer = ctx->use_quick_gelu ? 50 : 40;
    const int total_nodes = ctx->n_layers * ops_per_layer + 200;
    size_t buf_size = ggml_tensor_overhead() * total_nodes
                    + ggml_graph_overhead_custom(total_nodes, false);
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context* g = ggml_init(p);

    // Input token IDs
    ggml_tensor* ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);

    // Token embedding lookup
    ggml_tensor* x = ggml_get_rows(g, ctx->token_embd, ids);  // [D, T]

    // Position embedding: pos_embd is [D, max_pos], take first T positions
    ggml_tensor* pos = ggml_view_2d(g, ctx->pos_embd, D, T,
                                     ctx->pos_embd->nb[1], 0);
    x = ggml_add(g, x, pos);

    // Causal mask for CLIP (nullptr for SigLIP bidirectional)
    ggml_tensor* causal_mask = nullptr;
    if (ctx->causal) {
        causal_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Encoder layers (pre-LN: LN → CausalAttn → Add → LN → MLP → Add)
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

        // Reshape and permute for attention
        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K = ggml_reshape_3d(g, K, hd, nh, T);
        V = ggml_reshape_3d(g, V, hd, nh, T);
        Q = ggml_permute(g, Q, 0, 2, 1, 3);  // [hd, T, nh]
        K = ggml_permute(g, K, 0, 2, 1, 3);
        V = ggml_permute(g, V, 0, 2, 1, 3);

        // Flash attention with causal mask
        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor* attn = ggml_flash_attn_ext(g, Q, K, V, causal_mask,
                                                  scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection
        attn = ggml_mul_mat(g, L.o_w, attn);
        if (L.o_b) attn = ggml_add(g, attn, L.o_b);

        x = ggml_add(g, residual, attn);

        // Pre-FFN LN
        residual = x;
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, L.ln2_w);
        if (L.ln2_b) x = ggml_add(g, x, L.ln2_b);

        // MLP: fc1 → quick_gelu → fc2
        x = ggml_mul_mat(g, L.fc1_w, x);
        if (L.fc1_b) x = ggml_add(g, x, L.fc1_b);
        if (ctx->use_quick_gelu) {
            x = ggml_mul(g, x, ggml_sigmoid(g, ggml_scale(g, ggml_dup(g, x), 1.702f)));
        } else {
            x = ggml_gelu(g, x);
        }
        x = ggml_mul_mat(g, L.fc2_w, x);
        if (L.fc2_b) x = ggml_add(g, x, L.fc2_b);

        x = ggml_add(g, residual, x);
    }

    // Final layer norm
    if (ctx->final_ln_w) {
        x = ggml_norm(g, x, eps);
        x = ggml_mul(g, x, ctx->final_ln_w);
        if (ctx->final_ln_b) x = ggml_add(g, x, ctx->final_ln_b);
    }

    // Pool: CLIP at EOS position, SigLIP at last token (T-1)
    int pool_pos = ctx->causal ? eos_pos : (T - 1);
    ggml_tensor* pooled = ggml_view_1d(g, x, D, pool_pos * x->nb[1]);
    pooled = ggml_cont(g, pooled);

    // CLIP: text_projection, SigLIP: head (linear+bias)
    if (ctx->text_proj_w) {
        pooled = ggml_mul_mat(g, ctx->text_proj_w, pooled);
    }
    if (ctx->head_w) {
        pooled = ggml_mul_mat(g, ctx->head_w, pooled);
        if (ctx->head_b) pooled = ggml_add(g, pooled, ctx->head_b);
    }

    ggml_set_name(pooled, "embedding");
    ggml_set_output(pooled);

    // Build and compute graph
    ggml_cgraph* gf = ggml_new_graph_custom(g, total_nodes, false);
    ggml_build_forward_expand(gf, pooled);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    // Set input: token IDs
    ggml_tensor* inp_ids = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(inp_ids, toks.ids.data(), 0, T * sizeof(int32_t));

    // Set causal mask (CLIP only)
    if (ctx->causal) {
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<uint16_t> mask_data(T * T);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                mask_data[j + i * T] = (j <= i) ? (uint16_t)0 : (uint16_t)0xFC00;
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, T * T * sizeof(uint16_t));
    }

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
    return ctx ? ctx->proj_dim : 0;
}

void free(context* ctx) {
    if (ctx) {
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

} // namespace clip_text
