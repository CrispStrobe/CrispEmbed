// crispembed.cpp — BERT/MiniLM encoder via ggml graph.

#include "crispembed.h"
#include "tokenizer.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model structure
// ---------------------------------------------------------------------------

struct embed_layer {
    // Pre-attention LayerNorm
    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    // Attention Q/K/V/O
    ggml_tensor * q_w = nullptr, * q_b = nullptr;
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    // Pre-merged QKV (created at load time for fused projection)
    ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    // Post-attention LayerNorm
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    // FFN
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

struct embed_model {
    crispembed_hparams hparams;

    // Embeddings
    ggml_tensor * token_embd   = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * pos_embd     = nullptr;  // [n_embd, n_max_tokens]
    ggml_tensor * type_embd    = nullptr;  // [n_embd, 2] (optional)
    ggml_tensor * embd_ln_w    = nullptr;  // LayerNorm after embedding sum
    ggml_tensor * embd_ln_b    = nullptr;

    // Encoder layers
    std::vector<embed_layer> layers;

    // Optional pooler / projection
    ggml_tensor * pooler_w     = nullptr;
    ggml_tensor * pooler_b     = nullptr;
};

#include "decoder_embed_internal.h"

struct crispembed_context {
    embed_model model;
    std::unique_ptr<dec_model> dec;  // non-null for decoder models
    bool is_decoder = false;
    WordPieceTokenizer wp_tokenizer;
    SentencePieceTokenizer sp_tokenizer;
    BPETokenizer bpe_tokenizer;
    bool use_sentencepiece = false;
    bool use_bpe = false;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
    int pool_method = 0;  // 0=mean, 1=cls, 2=last-token
    int pos_offset = 0;   // position embedding offset (2 for RoBERTa/XLM-R)
    int matryoshka_dim = 0;  // 0 = use model default
    std::vector<float> last_output;     // reused buffer
    std::vector<uint8_t> compute_meta;  // graph metadata buffer (no_alloc=true)
    std::vector<uint8_t> graph_buf;    // decoder graph buffer (TODO: migrate to scheduler)
    std::vector<uint8_t> work_buf;     // decoder work buffer
    ggml_context * qkv_ctx = nullptr;   // pre-merged QKV weights
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

static bool load_model(crispembed_context * ctx, const char * path) {
    auto & m = ctx->model;
    auto & hp = m.hparams;

    // Load GGUF metadata first
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "crispembed: failed to open '%s'\n", path);
        return false;
    }

    auto u32 = [&](const char * key, int def) -> int {
        int k = gguf_find_key(g, key);
        return k >= 0 ? (int)gguf_get_val_u32(g, k) : def;
    };
    auto f32 = [&](const char * key, float def) -> float {
        int k = gguf_find_key(g, key);
        return k >= 0 ? gguf_get_val_f32(g, k) : def;
    };

    hp.n_vocab         = u32("bert.vocab_size", 30522);
    hp.n_max_tokens    = u32("bert.max_position_embeddings", 512);
    hp.n_embd          = u32("bert.hidden_size", 384);
    hp.n_head          = u32("bert.num_attention_heads", 12);
    hp.n_layer         = u32("bert.num_hidden_layers", 6);
    hp.n_intermediate  = u32("bert.intermediate_size", 1536);
    hp.n_output        = u32("bert.output_dim", hp.n_embd);
    hp.layer_norm_eps  = f32("bert.layer_norm_eps", 1e-12f);

    // Pooling method: 0=mean (default), 1=cls, 2=last-token
    ctx->pool_method   = u32("bert.pooling_method", 0);
    // Position embedding offset: 0 for BERT, 2 for RoBERTa/XLM-R
    ctx->pos_offset    = u32("bert.position_offset", 0);
    // fprintf(stderr, "crispembed: pool_method=%d pos_offset=%d\n", ctx->pool_method, ctx->pos_offset);

    // Load tokenizer vocab from GGUF metadata
    int ki = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (ki >= 0) {
        int n = gguf_get_arr_n(g, ki);
        std::vector<std::string> vocab(n);
        for (int i = 0; i < n; i++)
            vocab[i] = gguf_get_arr_str(g, ki, i);

        // Load scores if available (SentencePiece models)
        std::vector<float> scores;
        int si = gguf_find_key(g, "tokenizer.ggml.scores");
        if (si >= 0 && gguf_get_arr_type(g, si) == GGUF_TYPE_FLOAT32) {
            int sn = (int)gguf_get_arr_n(g, si);
            scores.resize(sn);
            const float * sd = (const float *)gguf_get_arr_data(g, si);
            std::memcpy(scores.data(), sd, sn * sizeof(float));
        }

        // Detect tokenizer type: SentencePiece for XLM-R (vocab > 100K)
        int tokenizer_type = u32("tokenizer.ggml.type", 0); // 0=WordPiece, 1=BPE, 2=SentencePiece
        if (tokenizer_type == 2 || n > 100000) {
            // SentencePiece / XLM-RoBERTa
            int bos_id = u32("tokenizer.ggml.bos_token_id", 0);
            int eos_id = u32("tokenizer.ggml.eos_token_id", 2);
            int unk_id = u32("tokenizer.ggml.unknown_token_id", 3);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 1);
            ctx->sp_tokenizer.load(vocab, scores, bos_id, eos_id, unk_id, pad_id, hp.n_max_tokens);
            ctx->use_sentencepiece = true;
            fprintf(stderr, "crispembed: using SentencePiece tokenizer (%d tokens, %zu scores)\n",
                    n, scores.size());
        } else {
            // WordPiece / BERT
            int cls_id = u32("tokenizer.ggml.cls_token_id", 101);
            int sep_id = u32("tokenizer.ggml.sep_token_id", 102);
            int unk_id = u32("tokenizer.ggml.unknown_token_id", 100);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 0);
            ctx->wp_tokenizer.load(vocab, cls_id, sep_id, unk_id, pad_id, hp.n_max_tokens);
            fprintf(stderr, "crispembed: using WordPiece tokenizer (%d tokens)\n", n);
        }
    }

    gguf_free(g);

    // Initialize backends: try GPU first, CPU always as fallback
    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) {
        fprintf(stderr, "crispembed: failed to init backend\n");
        return false;
    }
    ctx->backends.push_back(ctx->backend);

    bool have_gpu = !ggml_backend_is_cpu(ctx->backend);
    if (have_gpu) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, ctx->n_threads);
        ctx->backends.push_back(cpu);
        fprintf(stderr, "crispembed: using %s backend with CPU fallback\n",
                ggml_backend_name(ctx->backend));
    } else {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        fprintf(stderr, "crispembed: using CPU backend (%d threads)\n", ctx->n_threads);
    }

    // Create scheduler for graph dispatch (handles GPU/CPU allocation)
    int graph_nodes = 16384;
    ctx->sched = ggml_backend_sched_new(
        ctx->backends.data(), nullptr, (int)ctx->backends.size(),
        graph_nodes, false, false);

    // Allocate metadata buffer for graph building (no_alloc=true pattern)
    ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes
                           + ggml_graph_overhead_custom(graph_nodes, false));

    if (!core_gguf::load_weights(path, ctx->backend, "crispembed", ctx->wl)) {
        fprintf(stderr, "crispembed: failed to load weights\n");
        return false;
    }

    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = ctx->wl.tensors.find(n);
        return it != ctx->wl.tensors.end() ? it->second : nullptr;
    };

    // Embeddings
    m.token_embd = get("token_embd.weight");
    m.pos_embd   = get("position_embd.weight");
    m.type_embd  = get("token_type_embd.weight");
    m.embd_ln_w  = get("embd_ln.weight");
    m.embd_ln_b  = get("embd_ln.bias");

    if (!m.token_embd || !m.pos_embd) {
        fprintf(stderr, "crispembed: missing embedding tensors\n");
        return false;
    }

    // Encoder layers
    m.layers.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        auto pfx = "enc." + std::to_string(il) + ".";
        auto & L = m.layers[il];
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
    }

    // Pooler (optional)
    m.pooler_w = get("pooler.weight");
    m.pooler_b = get("pooler.bias");

    // Pre-merge QKV weights for fused projection (1 matmul instead of 3)
    // Allocate a separate context for merged weights
    {
        const int H = hp.n_embd;
        size_t qkv_mem = (size_t)hp.n_layer * (3 * H * H + 3 * H) * sizeof(float)
                       + hp.n_layer * 2 * ggml_tensor_overhead() + 1024;
        ggml_init_params qkv_ip = { qkv_mem, nullptr, false };
        ctx->qkv_ctx = ggml_init(qkv_ip);
        for (int i = 0; i < hp.n_layer; i++) {
            auto & L = m.layers[i];
            if (!L.q_w || !L.k_w || !L.v_w) continue;
            // Create merged weight [H, 3H] = concat(Q_w, K_w, V_w) along dim 1
            L.qkv_w = ggml_new_tensor_2d(ctx->qkv_ctx, GGML_TYPE_F32, H, 3 * H);
            ggml_backend_tensor_get(L.q_w, (char *)L.qkv_w->data, 0, H * H * sizeof(float));
            ggml_backend_tensor_get(L.k_w, (char *)L.qkv_w->data + H * H * sizeof(float), 0, H * H * sizeof(float));
            ggml_backend_tensor_get(L.v_w, (char *)L.qkv_w->data + 2 * H * H * sizeof(float), 0, H * H * sizeof(float));
            if (L.q_b && L.k_b && L.v_b) {
                L.qkv_b = ggml_new_tensor_1d(ctx->qkv_ctx, GGML_TYPE_F32, 3 * H);
                ggml_backend_tensor_get(L.q_b, (char *)L.qkv_b->data, 0, H * sizeof(float));
                ggml_backend_tensor_get(L.k_b, (char *)L.qkv_b->data + H * sizeof(float), 0, H * sizeof(float));
                ggml_backend_tensor_get(L.v_b, (char *)L.qkv_b->data + 2 * H * sizeof(float), 0, H * sizeof(float));
            }
        }
    }

    fprintf(stderr, "crispembed: loaded %d layers, %d dims, %d vocab\n",
            hp.n_layer, hp.n_embd, hp.n_vocab);
    return true;
}

// ---------------------------------------------------------------------------
// Graph: build fresh each call (no_alloc=true), scheduler handles allocation
// ---------------------------------------------------------------------------

static ggml_cgraph * build_encoder_graph(crispembed_context * ctx, int T) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int H = hp.n_embd;
    const int n_heads = hp.n_head;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;

    int graph_size = std::max(2048, hp.n_layer * 30 + 256);

    // Metadata-only context (no_alloc=true — scheduler allocates buffers)
    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // Input tensors (marked so scheduler keeps them on CPU for easy data set)
    ggml_tensor * tok_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(tok_ids, "tok_ids");
    ggml_set_input(tok_ids);
    ggml_tensor * pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    ggml_tensor * embd = ggml_get_rows(gctx, m.token_embd, tok_ids);
    ggml_tensor * pos_embd = ggml_get_rows(gctx, m.pos_embd, pos_ids);
    embd = ggml_add(gctx, embd, pos_embd);

    if (m.type_embd) {
        ggml_tensor * type_ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
        ggml_set_name(type_ids_t, "type_ids");
        ggml_set_input(type_ids_t);
        ggml_tensor * type_embd = ggml_get_rows(gctx, m.type_embd, type_ids_t);
        embd = ggml_add(gctx, embd, type_embd);
    }

    ggml_tensor * cur = embd;
    if (m.embd_ln_w) {
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, m.embd_ln_w);
        cur = ggml_add(gctx, cur, m.embd_ln_b);
    }

    for (int il = 0; il < hp.n_layer; il++) {
        const auto & L = m.layers[il];

        // QKV projection (fused or separate)
        ggml_tensor * Q, * K, * V;
        if (L.qkv_w) {
            ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, cur);
            if (L.qkv_b) qkv = ggml_add(gctx, qkv, L.qkv_b);
            Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, T, 3 * H * sizeof(float), 0));
            K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, T, 3 * H * sizeof(float), H * sizeof(float)));
            V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, T, 3 * H * sizeof(float), 2 * H * sizeof(float)));
        } else {
            Q = ggml_add(gctx, ggml_mul_mat(gctx, L.q_w, cur), L.q_b);
            K = ggml_add(gctx, ggml_mul_mat(gctx, L.k_w, cur), L.k_b);
            V = ggml_add(gctx, ggml_mul_mat(gctx, L.v_w, cur), L.v_b);
        }

        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(gctx, K, head_dim, n_heads, T);
        V = ggml_reshape_3d(gctx, V, head_dim, n_heads, T);

        // Flash attention
        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * attn = ggml_flash_attn_ext(gctx,
            ggml_permute(gctx, Q, 0, 2, 1, 3),
            ggml_permute(gctx, K, 0, 2, 1, 3),
            ggml_permute(gctx, V, 0, 2, 1, 3),
            nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(gctx, attn, H, T);

        attn = ggml_add(gctx, ggml_mul_mat(gctx, L.o_w, attn), L.o_b);

        cur = ggml_add(gctx, cur, attn);
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, L.ln1_w);
        cur = ggml_add(gctx, cur, L.ln1_b);

        ggml_tensor * ffn = ggml_add(gctx, ggml_mul_mat(gctx, L.fc1_w, cur), L.fc1_b);
        ffn = ggml_gelu(gctx, ffn);
        ffn = ggml_add(gctx, ggml_mul_mat(gctx, L.fc2_w, ffn), L.fc2_b);

        cur = ggml_add(gctx, cur, ffn);
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, L.ln2_w);
        cur = ggml_add(gctx, cur, L.ln2_b);
    }

    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Set thread count on all backends (like CrispASR's cohere_sched_graph_compute)
static bool sched_graph_compute(ggml_backend_sched_t sched, ggml_cgraph * gf, int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
        ggml_backend_t be = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto * fn = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) fn(be, n_threads);
        }
    }
    return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
}

static std::vector<float> encode_tokens(crispembed_context * ctx,
                                         const embed_tokens & tokens) {
    const auto & hp = ctx->model.hparams;
    const int T = (int)tokens.ids.size();
    const int H = hp.n_embd;

    // Build graph (fresh each call, metadata-only — scheduler allocates buffers)
    ggml_cgraph * gf = build_encoder_graph(ctx, T);

    // Allocate graph on backends (GPU + CPU)
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate encoder graph\n");
        return {};
    }

    // Set input data via backend API (works for both CPU and GPU tensors)
    std::vector<int32_t> tok_data(tokens.ids.begin(), tokens.ids.end());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_ids"),
                            tok_data.data(), 0, T * sizeof(int32_t));

    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = t + ctx->pos_offset;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                            pos_data.data(), 0, T * sizeof(int32_t));

    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(tokens.type_ids.begin(), tokens.type_ids.end());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "type_ids"),
                                type_data.data(), 0, T * sizeof(int32_t));
    }

    // Compute (scheduler dispatches to GPU or CPU)
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: encoder compute failed\n");
        return {};
    }

    // Read output (works whether tensor is on GPU or CPU)
    // Read encoder output [H, T] via backend API (works for GPU and CPU)
    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    std::vector<float> out_buf(H * T);
    ggml_backend_tensor_get(out, out_buf.data(), 0, H * T * sizeof(float));
    float * out_data = out_buf.data();

    // Pooling — method determined by model metadata or default
    int dim = hp.n_output > 0 ? hp.n_output : H;
    std::vector<float> pooled(dim, 0.0f);

    // Check pooling method from model hparams (0=mean, 1=cls, 2=last)
    int pool_method = ctx->pool_method;  // set during load from metadata

    if (pool_method == 1) {
        // CLS pooling: take the first token (position 0 = [CLS])
        for (int h = 0; h < std::min(H, dim); h++) {
            pooled[h] = out_data[h + 0 * H];  // token 0 = [CLS]
        }
    } else if (pool_method == 2) {
        // Last-token pooling (decoder models)
        int last_t = 0;
        for (int t = T - 1; t >= 0; t--) {
            if (tokens.attn_mask[t]) { last_t = t; break; }
        }
        for (int h = 0; h < std::min(H, dim); h++) {
            pooled[h] = out_data[h + last_t * H];
        }
    } else {
        // Mean pooling (default)
        int n_real = 0;
        for (int t = 0; t < T; t++) {
            if (tokens.attn_mask[t]) n_real++;
        }
        if (n_real > 0) {
            for (int t = 0; t < T; t++) {
                if (!tokens.attn_mask[t]) continue;
                for (int h = 0; h < std::min(H, dim); h++) {
                    pooled[h] += out_data[h + t * H];
                }
            }
            for (int h = 0; h < dim; h++) pooled[h] /= n_real;
        }
    }

    // L2 normalize
    float norm = 0;
    for (int h = 0; h < dim; h++) norm += pooled[h] * pooled[h];
    norm = sqrtf(std::max(norm, 1e-12f));
    for (int h = 0; h < dim; h++) pooled[h] /= norm;

    return pooled;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" crispembed_context * crispembed_init(const char * model_path, int n_threads) {
    auto * ctx = new crispembed_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    // Detect model type from GGUF metadata
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    bool is_dec = false;
    if (g) {
        is_dec = gguf_find_key(g, "decoder.hidden_size") >= 0;
        gguf_free(g);
    }

    if (is_dec) {
        ctx->is_decoder = true;
        ctx->dec = std::make_unique<dec_model>();
        // Initialize backends for decoder
        ctx->backend = ggml_backend_init_best();
        ctx->backends.push_back(ctx->backend);
        if (!ggml_backend_is_cpu(ctx->backend)) {
            ggml_backend_t cpu = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(cpu, ctx->n_threads);
            ctx->backends.push_back(cpu);
            fprintf(stderr, "crispembed: using %s backend with CPU fallback\n",
                    ggml_backend_name(ctx->backend));
        } else {
            ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        }
        if (!load_decoder_model(*ctx->dec, ctx->wl, model_path, ctx->backend)) {
            delete ctx;
            return nullptr;
        }
        ctx->model.hparams.n_embd = ctx->dec->n_embd;
        ctx->model.hparams.n_layer = ctx->dec->n_layer;
        ctx->model.hparams.n_vocab = ctx->dec->n_vocab;
        ctx->model.hparams.n_output = ctx->dec->n_embd;

        // Load BPE tokenizer from GGUF
        gguf_init_params gp2 = { true, nullptr };
        gguf_context * g2 = gguf_init_from_file(model_path, gp2);
        if (g2) {
            int ki2 = gguf_find_key(g2, "tokenizer.ggml.tokens");
            int mi2 = gguf_find_key(g2, "tokenizer.ggml.merges");
            if (ki2 >= 0) {
                int nv = (int)gguf_get_arr_n(g2, ki2);
                std::vector<std::string> vocab(nv);
                for (int i = 0; i < nv; i++)
                    vocab[i] = gguf_get_arr_str(g2, ki2, i);

                std::vector<std::string> merges;
                if (mi2 >= 0) {
                    int nm = (int)gguf_get_arr_n(g2, mi2);
                    merges.resize(nm);
                    for (int i = 0; i < nm; i++)
                        merges[i] = gguf_get_arr_str(g2, mi2, i);
                }

                auto u32g = [&](const char * key, int def) -> int {
                    int k = gguf_find_key(g2, key);
                    return k >= 0 ? (int)gguf_get_val_u32(g2, k) : def;
                };
                int eos_id = u32g("tokenizer.ggml.eos_token_id", 151645);
                int pad_id = u32g("tokenizer.ggml.padding_token_id", 151643);
                int bos_id = u32g("tokenizer.ggml.bos_token_id", -1);
                int ki_sfx = gguf_find_key(g2, "tokenizer.ggml.suffix_token_id");
                int suffix_id = ki_sfx >= 0 ? (int)gguf_get_val_i32(g2, ki_sfx) : pad_id;
                bool is_spm_bpe = u32g("tokenizer.ggml.is_spm_bpe", 0) != 0;

                ctx->bpe_tokenizer.load(vocab, merges, eos_id, pad_id,
                                         suffix_id, bos_id, is_spm_bpe,
                                         ctx->dec->n_max_pos);
                ctx->use_bpe = true;
                fprintf(stderr, "crispembed: %s BPE tokenizer (%d tokens, %zu merges)\n",
                        is_spm_bpe ? "SentencePiece" : "GPT-2",
                        nv, merges.size());
            }
            gguf_free(g2);
        }
    } else {
        if (!load_model(ctx, model_path)) {
            delete ctx;
            return nullptr;
        }
    }
    return ctx;
}

extern "C" const crispembed_hparams * crispembed_get_hparams(const crispembed_context * ctx) {
    return ctx ? &ctx->model.hparams : nullptr;
}

extern "C" const float * crispembed_encode(crispembed_context * ctx,
                                            const char * text,
                                            int * out_n_dim) {
    if (!ctx || !text) return nullptr;
    embed_tokens tokens;
    if (ctx->use_bpe) {
        tokens = ctx->bpe_tokenizer.encode(text);
    } else if (ctx->use_sentencepiece) {
        tokens = ctx->sp_tokenizer.encode(text);
    } else {
        tokens = ctx->wp_tokenizer.encode(text);
    }
    // Trim padding: only keep tokens where attn_mask == 1
    {
        int actual_len = 0;
        for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
            if (tokens.attn_mask[i]) { actual_len = i + 1; break; }
        }
        if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
            tokens.ids.resize(actual_len);
            tokens.type_ids.resize(actual_len);
            tokens.attn_mask.resize(actual_len);
        }
    }

    if (ctx->is_decoder && ctx->dec) {
        ctx->last_output = decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads,
                                                  &ctx->graph_buf, &ctx->work_buf);
    } else {
        ctx->last_output = encode_tokens(ctx, tokens);
    }

    // Matryoshka dimension truncation: truncate + re-normalize
    if (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < (int)ctx->last_output.size()) {
        ctx->last_output.resize(ctx->matryoshka_dim);
        float norm = 0;
        for (int i = 0; i < ctx->matryoshka_dim; i++)
            norm += ctx->last_output[i] * ctx->last_output[i];
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int i = 0; i < ctx->matryoshka_dim; i++)
            ctx->last_output[i] /= norm;
    }

    if (out_n_dim) *out_n_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_set_dim(crispembed_context * ctx, int dim) {
    if (ctx) ctx->matryoshka_dim = dim;
}

extern "C" const float * crispembed_encode_batch(crispembed_context * ctx,
                                                   const char ** texts,
                                                   int n_texts,
                                                   int * out_n_dim) {
    if (!ctx || !texts || n_texts <= 0) return nullptr;

    // Pre-tokenize all texts, sort by length for graph cache efficiency
    struct text_entry { int idx; embed_tokens tokens; };
    std::vector<text_entry> entries(n_texts);
    for (int i = 0; i < n_texts; i++) {
        entries[i].idx = i;
        if (ctx->use_bpe) {
            entries[i].tokens = ctx->bpe_tokenizer.encode(texts[i]);
        } else if (ctx->use_sentencepiece) {
            entries[i].tokens = ctx->sp_tokenizer.encode(texts[i]);
        } else {
            entries[i].tokens = ctx->wp_tokenizer.encode(texts[i]);
        }
    }
    // Sort by token count (minimizes graph rebuilds)
    std::sort(entries.begin(), entries.end(), [](const text_entry & a, const text_entry & b) {
        return a.tokens.ids.size() < b.tokens.ids.size();
    });

    // Determine output dim from first encoding
    int dim = 0;
    {
        auto & tokens = entries[0].tokens;
        // Trim padding
        int actual_len = (int)tokens.attn_mask.size();
        for (int i = actual_len - 1; i >= 0; i--) {
            if (tokens.attn_mask[i]) { actual_len = i + 1; break; }
        }
        if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
            tokens.ids.resize(actual_len);
            tokens.type_ids.resize(actual_len);
            tokens.attn_mask.resize(actual_len);
        }
        if (ctx->is_decoder && ctx->dec) {
            auto vec = decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads,
                                              &ctx->graph_buf, &ctx->work_buf);
            dim = (int)vec.size();
        } else {
            auto vec = encode_tokens(ctx, tokens);
            dim = (int)vec.size();
        }
    }
    if (dim == 0) return nullptr;

    // Apply Matryoshka
    int out_dim = (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < dim) ? ctx->matryoshka_dim : dim;
    ctx->last_output.resize(n_texts * out_dim);

    // Encode all texts (sorted order maximizes graph cache hits)
    for (int i = 0; i < n_texts; i++) {
        auto & tokens = entries[i].tokens;
        // Trim padding
        int actual_len = (int)tokens.attn_mask.size();
        for (int j = actual_len - 1; j >= 0; j--) {
            if (tokens.attn_mask[j]) { actual_len = j + 1; break; }
        }
        if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
            tokens.ids.resize(actual_len);
            tokens.type_ids.resize(actual_len);
            tokens.attn_mask.resize(actual_len);
        }

        std::vector<float> vec;
        if (ctx->is_decoder && ctx->dec) {
            vec = decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads,
                                         &ctx->graph_buf, &ctx->work_buf);
        } else {
            vec = encode_tokens(ctx, tokens);
        }

        // Matryoshka truncation + L2 normalize
        int d = std::min((int)vec.size(), out_dim);
        float norm = 0;
        for (int j = 0; j < d; j++) norm += vec[j] * vec[j];
        norm = sqrtf(std::max(norm, 1e-12f));
        float * dst = ctx->last_output.data() + entries[i].idx * out_dim;
        for (int j = 0; j < d; j++) dst[j] = vec[j] / norm;
    }
    if (out_n_dim) *out_n_dim = dim;
    return ctx->last_output.data();
}

extern "C" void crispembed_free(crispembed_context * ctx) {
    if (!ctx) return;
    if (ctx->qkv_ctx) { ggml_free(ctx->qkv_ctx); ctx->qkv_ctx = nullptr; }
    core_gguf::free_weights(ctx->wl);
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
        ctx->sched = nullptr;
    }
    for (auto b : ctx->backends) {
        if (b) ggml_backend_free(b);
    }
    ctx->backends.clear();
    ctx->backend = nullptr;
    delete ctx;
}
