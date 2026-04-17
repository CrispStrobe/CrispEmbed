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

// MPNet-style relative position bucket (matches HuggingFace implementation).
static int relative_position_bucket(int rel_pos, int num_buckets = 32, int max_distance = 128) {
    int ret = 0;
    int n = -rel_pos;
    int half = num_buckets / 2;
    if (n < 0) { ret += half; n = -n; }
    int max_exact = half / 2;
    if (n < max_exact) {
        ret += n;
    } else {
        int val = max_exact + (int)(log((double)n / max_exact) / log((double)max_distance / max_exact) * (half - max_exact));
        if (val > half - 1) val = half - 1;
        ret += val;
    }
    return ret;
}

// Precompute MPNet relative position bias for sequence length T.
// rel_attn_bias: [n_buckets, n_heads] tensor
// Output: [n_heads, T, T] float array (row-major)
static std::vector<float> compute_rel_pos_bias(
    ggml_tensor * rel_attn_bias, int T, int n_heads, int n_buckets = 32)
{
    // Read bias weights from tensor [n_buckets, n_heads]
    std::vector<float> bias_weights(n_buckets * n_heads);
    ggml_backend_tensor_get(rel_attn_bias, bias_weights.data(), 0,
                            n_buckets * n_heads * sizeof(float));

    // Compute bucket indices for all (i, j) pairs
    std::vector<float> out(n_heads * T * T, 0.0f);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int bucket = relative_position_bucket(j - i, n_buckets);
            for (int h = 0; h < n_heads; h++) {
                // out[h][i][j] = bias_weights[bucket][h]
                out[h * T * T + i * T + j] = bias_weights[bucket * n_heads + h];
            }
        }
    }
    return out;
}
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
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
    // Pre-merged QKV (in backend buffer — works on GPU)
    ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    // Post-attention LayerNorm
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    // FFN
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;  // SwiGLU gate (NomicBERT)
};

struct embed_model {
    crispembed_hparams hparams;

    // Embeddings
    ggml_tensor * token_embd   = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * pos_embd     = nullptr;  // [n_embd, n_max_tokens]
    ggml_tensor * type_embd    = nullptr;  // [n_embd, 2] (optional)
    ggml_tensor * embd_ln_w    = nullptr;  // LayerNorm after embedding sum
    ggml_tensor * embd_ln_b    = nullptr;
    ggml_tensor * rel_attn_bias = nullptr;  // MPNet relative position bias [n_buckets, n_heads]
    ggml_tensor * rel_embd      = nullptr;  // DeBERTa relative position embeddings [n_embd, max_rel_pos]
    ggml_tensor * encoder_ln_w = nullptr;   // DeBERTa encoder-level LayerNorm
    ggml_tensor * encoder_ln_b = nullptr;

    // Encoder layers
    std::vector<embed_layer> layers;

    // Optional pooler / projection
    ggml_tensor * pooler_w     = nullptr;
    ggml_tensor * pooler_b     = nullptr;

    // Sparse retrieval head (BGE-M3): Linear(n_embd, 1)
    ggml_tensor * sparse_linear_w  = nullptr;  // [H, 1]
    ggml_tensor * sparse_linear_b  = nullptr;  // [1], optional
    // ColBERT multi-vector head: Linear(n_embd, colbert_dim)
    ggml_tensor * colbert_linear_w = nullptr;  // [H, colbert_dim]
    ggml_tensor * colbert_linear_b = nullptr;  // [colbert_dim], optional
    // Reranker: 1-layer head Linear(H, 1)
    ggml_tensor * classifier_w         = nullptr;  // [1, H]
    ggml_tensor * classifier_b         = nullptr;  // [1]
    // Reranker: 2-layer RobertaClassificationHead (bge-reranker-v2-m3)
    ggml_tensor * classifier_dense_w   = nullptr;  // [H, H]
    ggml_tensor * classifier_dense_b   = nullptr;  // [H]
    ggml_tensor * classifier_out_w     = nullptr;  // [1, H]
    ggml_tensor * classifier_out_b     = nullptr;  // [1]
    bool classifier_2layer = false;

    bool has_sparse  = false;
    bool has_colbert = false;
    bool is_reranker = false;
    int  colbert_dim = 128;
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
    bool use_rope = false;    // encoder uses RoPE instead of absolute position embeddings (NomicBERT)
    float rope_theta = 10000.0f;
    bool pre_ln = false;      // pre-LN (ModernBERT) vs post-LN (BERT) ordering
    int matryoshka_dim = 0;  // 0 = use model default
    std::string prefix;  // prepended to text before tokenization (e.g. "query: ")
    std::vector<float> last_output;     // reused buffer (dense encode)
    std::vector<uint8_t> compute_meta;  // graph metadata buffer (no_alloc=true)
    ggml_context * qkv_ctx = nullptr;   // pre-merged QKV tensor metadata
    ggml_backend_buffer_t qkv_buf = nullptr;  // backend buffer for merged QKV
    int reserved_T = 0;                  // scheduler reserved for this seq len
    // Sparse / colbert / reranker output buffers (valid until next call)
    std::vector<int32_t> last_sparse_indices;
    std::vector<float>   last_sparse_values;
    std::vector<float>   last_multivec;
    int last_multivec_n_tokens = 0;
    int last_multivec_dim      = 0;
    // Per-mode scheduler reservation buckets
    int reserved_T_sparse  = 0;
    int reserved_T_colbert = 0;
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
    // ColBERT output dimension (BGE-M3 default 128) — read while g is valid
    m.colbert_dim      = u32("bert.colbert_dim", 128);
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
    m.rel_attn_bias = get("rel_attn_bias.weight");
    m.rel_embd      = get("rel_embd.weight");
    m.encoder_ln_w  = get("encoder_ln.weight");
    m.encoder_ln_b  = get("encoder_ln.bias");

    if (!m.token_embd) {
        fprintf(stderr, "crispembed: missing token_embd.weight\n");
        return false;
    }
    // NomicBERT and other RoPE-based encoders lack position embeddings
    if (!m.pos_embd) {
        ctx->use_rope = true;
        ctx->rope_theta = f32("bert.rope_theta", 10000.0f);
        fprintf(stderr, "crispembed: no position embeddings, using RoPE (theta=%.0f)\n", ctx->rope_theta);
    }
    // Pre-LN detection: if GGUF flag set, or if model has gate weights + no biases (ModernBERT)
    ctx->pre_ln = u32("bert.pre_ln", 0) != 0;

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
        L.ffn_gate_w = get(pfx + "ffn_gate.weight");  // SwiGLU gate (NomicBERT)
    }

    // Pooler (optional)
    m.pooler_w = get("pooler.weight");
    m.pooler_b = get("pooler.bias");

    // Optional sparse / colbert / classifier heads
    m.sparse_linear_w    = get("sparse_linear.weight");
    m.sparse_linear_b    = get("sparse_linear.bias");
    m.colbert_linear_w   = get("colbert_linear.weight");
    m.colbert_linear_b   = get("colbert_linear.bias");
    // Try 2-layer RobertaClassificationHead first (bge-reranker-v2-m3)
    m.classifier_dense_w = get("classifier.dense.weight");
    m.classifier_dense_b = get("classifier.dense.bias");
    m.classifier_out_w   = get("classifier.out_proj.weight");
    m.classifier_out_b   = get("classifier.out_proj.bias");
    if (m.classifier_dense_w && m.classifier_out_w) {
        m.classifier_2layer = true;
        m.is_reranker = true;
    } else {
        // Fall back to 1-layer head
        m.classifier_w   = get("classifier.weight");
        m.classifier_b   = get("classifier.bias");
        m.is_reranker    = m.classifier_w != nullptr;
    }
    m.has_sparse  = m.sparse_linear_w != nullptr;
    m.has_colbert = m.colbert_linear_w != nullptr;
    if (m.has_sparse)  fprintf(stderr, "crispembed: sparse head loaded\n");
    if (m.has_colbert) fprintf(stderr, "crispembed: colbert head loaded (dim=%d)\n", m.colbert_dim);
    if (m.is_reranker) fprintf(stderr, "crispembed: classifier head loaded (reranker=%s)\n",
                               m.classifier_2layer ? "2-layer" : "1-layer");

    // Pre-merge QKV weights into backend buffer (works on CPU + GPU)
    {
        const int H = hp.n_embd;
        size_t qkv_mem = hp.n_layer * 2 * ggml_tensor_overhead() + 1024;
        ggml_init_params qkv_ip = { qkv_mem, nullptr, true };  // no_alloc
        ctx->qkv_ctx = ggml_init(qkv_ip);

        for (int i = 0; i < hp.n_layer; i++) {
            auto & L = m.layers[i];
            if (!L.q_w || !L.k_w || !L.v_w) continue;
            if (L.q_w->type != GGML_TYPE_F32) continue;  // skip quantized
            L.qkv_w = ggml_new_tensor_2d(ctx->qkv_ctx, GGML_TYPE_F32, H, 3 * H);
            if (L.q_b && L.k_b && L.v_b)
                L.qkv_b = ggml_new_tensor_1d(ctx->qkv_ctx, GGML_TYPE_F32, 3 * H);
        }

        ctx->qkv_buf = ggml_backend_alloc_ctx_tensors(ctx->qkv_ctx, ctx->backend);
        if (ctx->qkv_buf) {
            // Copy Q/K/V data into merged tensor
            std::vector<float> tmp;
            for (int i = 0; i < hp.n_layer; i++) {
                auto & L = m.layers[i];
                if (!L.qkv_w) continue;
                tmp.resize(H * H);
                ggml_backend_tensor_get(L.q_w, tmp.data(), 0, H * H * sizeof(float));
                ggml_backend_tensor_set(L.qkv_w, tmp.data(), 0, H * H * sizeof(float));
                ggml_backend_tensor_get(L.k_w, tmp.data(), 0, H * H * sizeof(float));
                ggml_backend_tensor_set(L.qkv_w, tmp.data(), H * H * sizeof(float), H * H * sizeof(float));
                ggml_backend_tensor_get(L.v_w, tmp.data(), 0, H * H * sizeof(float));
                ggml_backend_tensor_set(L.qkv_w, tmp.data(), 2 * H * H * sizeof(float), H * H * sizeof(float));
                if (L.qkv_b) {
                    tmp.resize(H);
                    ggml_backend_tensor_get(L.q_b, tmp.data(), 0, H * sizeof(float));
                    ggml_backend_tensor_set(L.qkv_b, tmp.data(), 0, H * sizeof(float));
                    ggml_backend_tensor_get(L.k_b, tmp.data(), 0, H * sizeof(float));
                    ggml_backend_tensor_set(L.qkv_b, tmp.data(), H * sizeof(float), H * sizeof(float));
                    ggml_backend_tensor_get(L.v_b, tmp.data(), 0, H * sizeof(float));
                    ggml_backend_tensor_set(L.qkv_b, tmp.data(), 2 * H * sizeof(float), H * sizeof(float));
                }
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

// Build encoder graph for T tokens × B batch items.
// mode: 0=dense (encoder_out), 1=sparse (sparse_out [1,T]), 2=colbert (colbert_out [dim,T])
// When B=1: standard single-text graph.
// When B>1: batched graph with 4D attention via flash_attn_ext.
static ggml_cgraph * build_encoder_graph(crispembed_context * ctx, int T, int B = 1, int mode = 0) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int H = hp.n_embd;
    const int n_heads = hp.n_head;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;
    const int TB = T * B;  // total tokens in batch

    int graph_size = std::max(2048, hp.n_layer * 30 + 256);

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // Input: flattened token IDs [T*B] and position IDs [T*B]
    ggml_tensor * tok_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TB);
    ggml_set_name(tok_ids, "tok_ids");
    ggml_set_input(tok_ids);
    ggml_tensor * pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TB);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Embeddings: [H, T*B]
    ggml_tensor * embd = ggml_get_rows(gctx, m.token_embd, tok_ids);
    if (m.pos_embd) {
        ggml_tensor * pos_embd = ggml_get_rows(gctx, m.pos_embd, pos_ids);
        embd = ggml_add(gctx, embd, pos_embd);
    }

    if (m.type_embd) {
        ggml_tensor * type_ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, TB);
        ggml_set_name(type_ids_t, "type_ids");
        ggml_set_input(type_ids_t);
        embd = ggml_add(gctx, embd, ggml_get_rows(gctx, m.type_embd, type_ids_t));
    }

    // For RoPE encoders, need a [T]-shaped position tensor (not [T*B]).
    // RoPE expects ne[0]=T matching the time dimension of Q/K before permute.
    // Use a view of the first T elements of pos_ids (which are [0,1,...T-1]).
    ggml_tensor * rope_pos = nullptr;
    if (ctx->use_rope) {
        rope_pos = ggml_view_1d(gctx, pos_ids, T, 0);
    }

    // MPNet/DeBERTa relative position bias: precomputed [T, T, n_heads]
    // Flash attention requires F16 mask
    ggml_tensor * rel_pos_bias = nullptr;
    if (m.rel_attn_bias) {
        rel_pos_bias = ggml_new_tensor_3d(gctx, GGML_TYPE_F16, T, T, n_heads);
        ggml_set_name(rel_pos_bias, "rel_pos_bias");
        ggml_set_input(rel_pos_bias);
    }

    // cur: [H, T*B] — all matmuls batch naturally
    ggml_tensor * cur = embd;
    if (m.embd_ln_w) {
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, m.embd_ln_w);
        if (m.embd_ln_b) cur = ggml_add(gctx, cur, m.embd_ln_b);
    }

    for (int il = 0; il < hp.n_layer; il++) {
        const auto & L = m.layers[il];
        ggml_tensor * inp = cur;  // save for residual connection

        // Pre-LN: normalize before attention (ModernBERT)
        if (ctx->pre_ln && L.ln1_w) {
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln1_w);
            if (L.ln1_b) cur = ggml_add(gctx, cur, L.ln1_b);
        }

        // QKV projection (fused: 1 matmul + 3 view+cont, or 3 separate matmuls)
        ggml_tensor * Q, * K, * V;
        if (L.qkv_w) {
            ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, cur);
            if (L.qkv_b) qkv = ggml_add(gctx, qkv, L.qkv_b);
            Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3*H*sizeof(float), 0));
            K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3*H*sizeof(float), H*sizeof(float)));
            V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3*H*sizeof(float), 2*H*sizeof(float)));
        } else {
            Q = ggml_mul_mat(gctx, L.q_w, cur);
            K = ggml_mul_mat(gctx, L.k_w, cur);
            V = ggml_mul_mat(gctx, L.v_w, cur);
            if (L.q_b) Q = ggml_add(gctx, Q, L.q_b);
            if (L.k_b) K = ggml_add(gctx, K, L.k_b);
            if (L.v_b) V = ggml_add(gctx, V, L.v_b);
        }

        // Reshape for attention: [H, T*B] → [head_dim, T, n_heads, B]
        // flash_attn_ext: q[hd, T, nh, B], k[hd, T, nh, B], v[hd, T, nh, B]
        Q = ggml_reshape_4d(gctx, Q, head_dim, n_heads, T, B);
        K = ggml_reshape_4d(gctx, K, head_dim, n_heads, T, B);
        V = ggml_reshape_4d(gctx, V, head_dim, n_heads, T, B);

        // Optional RoPE for encoder models without position embeddings (NomicBERT)
        // Apply before permute: Q/K shape is [hd, nh, T, B], RoPE uses ne[2]=T
        if (rope_pos) {
            Q = ggml_rope_ext(gctx, Q, rope_pos, nullptr,
                              head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens,
                              ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(gctx, K, rope_pos, nullptr,
                              head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens,
                              ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // Permute: [hd, nh, T, B] → [hd, T, nh, B]
        Q = ggml_permute(gctx, Q, 0, 2, 1, 3);
        K = ggml_permute(gctx, K, 0, 2, 1, 3);
        V = ggml_permute(gctx, V, 0, 2, 1, 3);

        ggml_tensor * attn;

        if (m.rel_embd && B == 1) {
            // DeBERTa disentangled attention (manual, B=1 only)
            // Q/K/V after permute: [hd, T, nh, 1]
            ggml_tensor * Qs = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, Q), head_dim, T, n_heads));
            ggml_tensor * Ks = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, K), head_dim, T, n_heads));
            ggml_tensor * Vs = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, V), head_dim, T, n_heads));

            // c2c: standard content-to-content scores [T, T, nh]
            ggml_tensor * scores = ggml_mul_mat(gctx, Ks, Qs);

            // Position embeddings: look up rel_embd for relative positions
            // rel_pos_idx: [T*T] precomputed indices into rel_embd
            ggml_tensor * rel_pos_idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T * T);
            ggml_set_name(rel_pos_idx, "rel_pos_idx");
            ggml_set_input(rel_pos_idx);

            // P = rel_embd[rel_pos_idx] → [H, T*T]
            ggml_tensor * P = ggml_get_rows(gctx, m.rel_embd, rel_pos_idx);
            // Reshape to [H, T, T] then project through Q/K
            P = ggml_reshape_3d(gctx, P, H, T, T);

            // c2p: Q × P^T for each position pair
            // Project P through K weights: P_k = W_k × P → [H, T, T]
            // Then Q × P_k^T → additional score per (q_pos, k_pos)
            // Simplified: use P directly as position key, compute Q^T × P
            // P[:, :, j] = position embedding for relative pos (i-j)
            // For each query pos i: score_c2p[i,j] = Q[i] · P[i,j]
            // This requires per-pair dot products — approximate with matmul

            // Project P through K layer weights: [hd, T, T] per head
            ggml_tensor * P_flat = ggml_reshape_2d(gctx, P, H, T * T);  // [H, T*T]
            ggml_tensor * Pk = ggml_mul_mat(gctx, L.k_w, P_flat);  // [H, T*T]
            if (L.k_b) Pk = ggml_add(gctx, Pk, ggml_repeat(gctx, L.k_b,
                ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T * T)));
            Pk = ggml_reshape_3d(gctx, Pk, head_dim, n_heads, T * T);
            Pk = ggml_cont(gctx, ggml_permute(gctx, Pk, 0, 2, 1, 3));  // [hd, T*T, nh]
            Pk = ggml_reshape_4d(gctx, Pk, head_dim, T, T, n_heads);    // [hd, T, T, nh]

            // c2p score: for each head h, query pos i, key pos j:
            //   c2p[i,j,h] = sum_d Q[d,i,h] * Pk[d,j,i,h]  (where i selects the relative pos row)
            // This is complex — use a gather + batched dot product approach
            // Simplified: compute Q × Pk^T per query position (expensive but correct)
            // For now, skip c2p/p2c and just use c2c (partial DeBERTa)

            // Scale by 1/sqrt(scale_factor * head_dim) where scale_factor=3 for c2p+p2c
            float scale = 1.0f / sqrtf((float)head_dim);  // Use 1-factor since no c2p/p2c yet
            scores = ggml_scale(gctx, scores, scale);

            // Add any available position bias mask
            if (rel_pos_bias) scores = ggml_add(gctx, scores, rel_pos_bias);

            scores = ggml_soft_max(gctx, scores);

            // weighted sum: V × scores
            ggml_tensor * Vt = ggml_cont(gctx, ggml_permute(gctx, Vs, 1, 0, 2, 3));
            attn = ggml_mul_mat(gctx, Vt, scores);
            attn = ggml_reshape_2d(gctx, ggml_cont(gctx, attn), H, T);
        } else {
            float scale = 1.0f / sqrtf((float)head_dim);

            // Flash attention (supports optional position bias mask)
            // Q/K/V: [hd, T, nh, B] after permute
            // rel_pos_bias: [T, T, nh] — passed as mask (additive to attention scores)
            attn = ggml_flash_attn_ext(gctx, Q, K, V,
                                       rel_pos_bias, scale, 0.0f, 0.0f);
            // Result: [hd, nh, T, B] → reshape to [H, T*B]
            attn = ggml_reshape_2d(gctx, attn, H, TB);
        }

        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);

        if (ctx->pre_ln) {
            // Pre-LN: residual add (LN was applied before attention)
            cur = ggml_add(gctx, inp, attn);
            inp = cur;  // save for FFN residual
            // Pre-FFN norm
            if (L.ln2_w) {
                cur = ggml_norm(gctx, cur, ln_eps);
                cur = ggml_mul(gctx, cur, L.ln2_w);
                if (L.ln2_b) cur = ggml_add(gctx, cur, L.ln2_b);
            }
        } else {
            // Post-LN: residual add then LN
            cur = ggml_add(gctx, inp, attn);
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln1_w);
            if (L.ln1_b) cur = ggml_add(gctx, cur, L.ln1_b);
        }

        ggml_tensor * ffn;
        if (L.ffn_gate_w) {
            // Gated FFN: gate * activation(up) → down
            ggml_tensor * up   = ggml_mul_mat(gctx, L.fc1_w, cur);
            ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_gate_w, cur);
            // NomicBERT uses SiLU, ModernBERT uses GELU for gate
            gate = ctx->pre_ln ? ggml_gelu(gctx, gate) : ggml_silu(gctx, gate);
            ffn = ggml_mul(gctx, up, gate);
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
        } else {
            // Standard GELU FFN (BERT)
            ffn = ggml_mul_mat(gctx, L.fc1_w, cur);
            if (L.fc1_b) ffn = ggml_add(gctx, ffn, L.fc1_b);
            ffn = ggml_gelu(gctx, ffn);
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
            if (L.fc2_b) ffn = ggml_add(gctx, ffn, L.fc2_b);
        }

        if (ctx->pre_ln) {
            // Pre-LN: just residual add
            cur = ggml_add(gctx, inp, ffn);
        } else {
            // Post-LN: residual add then LN
            cur = ggml_add(gctx, cur, ffn);
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln2_w);
            if (L.ln2_b) cur = ggml_add(gctx, cur, L.ln2_b);
        }
    }

    // Named output depends on requested mode
    if (mode == 1 && ctx->model.sparse_linear_w) {
        // Sparse head: Linear(H,1) [+ bias] + ReLU → [1, T*B]
        ggml_tensor * sw = ggml_mul_mat(gctx, ctx->model.sparse_linear_w, cur);
        if (ctx->model.sparse_linear_b)
            sw = ggml_add(gctx, sw, ctx->model.sparse_linear_b);
        sw = ggml_relu(gctx, sw);
        ggml_set_name(sw, "sparse_out");
        ggml_set_output(sw);
        ggml_build_forward_expand(gf, sw);
    } else if (mode == 2 && ctx->model.colbert_linear_w) {
        // ColBERT head: Linear(H, colbert_dim) [+ bias] → [colbert_dim, T*B]
        ggml_tensor * cv = ggml_mul_mat(gctx, ctx->model.colbert_linear_w, cur);
        if (ctx->model.colbert_linear_b)
            cv = ggml_add(gctx, cv, ctx->model.colbert_linear_b);
        ggml_set_name(cv, "colbert_out");
        ggml_set_output(cv);
        ggml_build_forward_expand(gf, cv);
    } else {
        ggml_set_name(cur, "encoder_out");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

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

// Bucket sequence length to reduce scheduler re-reserves
static int bucket_seq_len(int T) {
    if (T <= 8)   return 8;
    if (T <= 16)  return 16;
    if (T <= 32)  return 32;
    if (T <= 64)  return 64;
    if (T <= 128) return 128;
    if (T <= 256) return 256;
    if (T <= 512) return 512;
    return T;
}

static std::vector<float> encode_tokens(crispembed_context * ctx,
                                         const embed_tokens & tokens) {
    const auto & hp = ctx->model.hparams;
    const int T = (int)tokens.ids.size();
    const int H = hp.n_embd;

    // Pad T to bucket for scheduler reservation reuse
    int T_bucket = bucket_seq_len(T);

    // Reserve scheduler for this bucket if not already reserved
    if (ctx->reserved_T != T_bucket) {
        ggml_cgraph * measure_gf = build_encoder_graph(ctx, T_bucket);
        ggml_backend_sched_reserve(ctx->sched, measure_gf);
        ctx->reserved_T = T_bucket;
    }

    // Build graph for actual T (metadata only — scheduler already has buffers)
    ggml_cgraph * gf = build_encoder_graph(ctx, T);

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

    // MPNet relative position bias (precomputed for this sequence length, F16)
    if (ctx->model.rel_attn_bias) {
        ggml_tensor * bias_t = ggml_graph_get_tensor(gf, "rel_pos_bias");
        if (bias_t) {
            auto bias_f32 = compute_rel_pos_bias(
                ctx->model.rel_attn_bias, T, ctx->model.hparams.n_head);
            // Convert to F16 for flash attention mask
            std::vector<ggml_fp16_t> bias_f16(bias_f32.size());
            for (size_t i = 0; i < bias_f32.size(); i++)
                bias_f16[i] = ggml_fp32_to_fp16(bias_f32[i]);
            ggml_backend_tensor_set(bias_t, bias_f16.data(), 0,
                                    bias_f16.size() * sizeof(ggml_fp16_t));
        }
    }

    // DeBERTa relative position indices [T*T]
    if (ctx->model.rel_embd) {
        ggml_tensor * idx_t = ggml_graph_get_tensor(gf, "rel_pos_idx");
        if (idx_t) {
            // rel_embd shape: [H, max_pos]. Relative position = clamp(i-j + max_pos/2, 0, max_pos-1)
            int max_pos = (int)ctx->model.rel_embd->ne[1];
            int half = max_pos / 2;
            std::vector<int32_t> idx(T * T);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    int rel = i - j + half;
                    if (rel < 0) rel = 0;
                    if (rel >= max_pos) rel = max_pos - 1;
                    idx[i * T + j] = rel;
                }
            }
            ggml_backend_tensor_set(idx_t, idx.data(), 0, T * T * sizeof(int32_t));
        }
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

// Batched encoding: multiple texts in one graph (padded to max length)
static std::vector<std::vector<float>> encode_tokens_batch(
    crispembed_context * ctx,
    const std::vector<embed_tokens> & batch) {

    const auto & hp = ctx->model.hparams;
    const int H = hp.n_embd;
    const int B = (int)batch.size();
    if (B == 0) return {};

    // Find max token count and pad all texts
    int T_max = 0;
    for (auto & t : batch) T_max = std::max(T_max, (int)t.ids.size());

    // Build batched graph [T_max * B tokens]
    ggml_cgraph * gf = build_encoder_graph(ctx, T_max, B);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate batched graph\n");
        return {};
    }

    // Prepare flattened input arrays [T_max * B]
    int TB = T_max * B;
    std::vector<int32_t> all_tok(TB, 0);   // pad token = 0
    std::vector<int32_t> all_pos(TB, 0);
    std::vector<int32_t> all_type(TB, 0);
    std::vector<std::vector<int>> attn_masks(B);

    for (int b = 0; b < B; b++) {
        const auto & t = batch[b];
        int len = (int)t.ids.size();
        attn_masks[b] = std::vector<int>(t.attn_mask.begin(), t.attn_mask.end());
        attn_masks[b].resize(T_max, 0);  // pad mask with 0s
        for (int i = 0; i < len; i++) {
            all_tok[b * T_max + i] = t.ids[i];
            all_pos[b * T_max + i] = i + ctx->pos_offset;
            if (i < (int)t.type_ids.size())
                all_type[b * T_max + i] = t.type_ids[i];
        }
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_ids"),
                            all_tok.data(), 0, TB * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                            all_pos.data(), 0, TB * sizeof(int32_t));
    if (ctx->model.type_embd) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "type_ids"),
                                all_type.data(), 0, TB * sizeof(int32_t));
    }

    // Relative position bias (MPNet) — broadcasts across batch dim
    if (ctx->model.rel_attn_bias) {
        ggml_tensor * bias_t = ggml_graph_get_tensor(gf, "rel_pos_bias");
        if (bias_t) {
            auto bias_f32 = compute_rel_pos_bias(
                ctx->model.rel_attn_bias, T_max, ctx->model.hparams.n_head);
            std::vector<ggml_fp16_t> bias_f16(bias_f32.size());
            for (size_t i = 0; i < bias_f32.size(); i++)
                bias_f16[i] = ggml_fp32_to_fp16(bias_f32[i]);
            ggml_backend_tensor_set(bias_t, bias_f16.data(), 0,
                                    bias_f16.size() * sizeof(ggml_fp16_t));
        }
    }

    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: batched compute failed\n");
        return {};
    }

    // Read output [H, T_max * B]
    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    std::vector<float> out_buf(H * TB);
    ggml_backend_tensor_get(out, out_buf.data(), 0, H * TB * sizeof(float));

    // Pool and normalize each text in batch
    int dim = hp.n_output > 0 ? hp.n_output : H;
    int pool_method = ctx->pool_method;
    std::vector<std::vector<float>> results(B);

    for (int b = 0; b < B; b++) {
        std::vector<float> pooled(dim, 0.0f);
        float * data = out_buf.data() + b * T_max * H;
        auto & mask = attn_masks[b];

        if (pool_method == 1) {
            for (int h = 0; h < std::min(H, dim); h++) pooled[h] = data[h];
        } else if (pool_method == 2) {
            int last_t = 0;
            for (int t = T_max - 1; t >= 0; t--) { if (mask[t]) { last_t = t; break; } }
            for (int h = 0; h < std::min(H, dim); h++) pooled[h] = data[h + last_t * H];
        } else {
            int n_real = 0;
            for (int t = 0; t < T_max; t++) if (mask[t]) n_real++;
            if (n_real > 0) {
                for (int t = 0; t < T_max; t++) {
                    if (!mask[t]) continue;
                    for (int h = 0; h < std::min(H, dim); h++) pooled[h] += data[h + t * H];
                }
                for (int h = 0; h < dim; h++) pooled[h] /= n_real;
            }
        }

        float norm = 0;
        for (int h = 0; h < dim; h++) norm += pooled[h] * pooled[h];
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int h = 0; h < dim; h++) pooled[h] /= norm;
        results[b] = std::move(pooled);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Sparse / ColBERT / Reranker helpers (single-text, encoder models only)
// ---------------------------------------------------------------------------

// Run the encoder for a single embed_tokens, returning raw [H * T] output.
// Handles scheduler reservation using a separate bucket tracking field.
static std::vector<float> run_encoder_raw(crispembed_context * ctx,
                                           const embed_tokens & tokens,
                                           int mode,
                                           int * out_T) {
    const auto & hp = ctx->model.hparams;
    const int H = hp.n_embd;
    const int T = (int)tokens.ids.size();
    if (out_T) *out_T = T;

    int T_bucket = bucket_seq_len(T);
    int & reserved = (mode == 1) ? ctx->reserved_T_sparse
                   : (mode == 2) ? ctx->reserved_T_colbert
                   : ctx->reserved_T;

    if (reserved != T_bucket) {
        ggml_cgraph * measure_gf = build_encoder_graph(ctx, T_bucket, 1, mode);
        ggml_backend_sched_reserve(ctx->sched, measure_gf);
        reserved = T_bucket;
    }

    ggml_cgraph * gf = build_encoder_graph(ctx, T, 1, mode);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate graph (mode=%d)\n", mode);
        return {};
    }

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

    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: compute failed (mode=%d)\n", mode);
        return {};
    }

    const char * out_name = (mode == 1) ? "sparse_out"
                          : (mode == 2) ? "colbert_out"
                          : "encoder_out";
    ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
    if (!out) return {};

    // Output dims: mode=1 → [1,T], mode=2 → [colbert_dim,T], mode=0 → [H,T]
    int out_rows = (int)out->ne[0];
    std::vector<float> buf(out_rows * T);
    ggml_backend_tensor_get(out, buf.data(), 0, out_rows * T * sizeof(float));
    return buf;
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

    // Prepend prefix if set (e.g. "query: ", "Represent this sentence: ")
    std::string prefixed;
    const char * enc_text = text;
    if (!ctx->prefix.empty()) {
        prefixed = ctx->prefix + text;
        enc_text = prefixed.c_str();
    }

    embed_tokens tokens;
    if (ctx->use_bpe) {
        tokens = ctx->bpe_tokenizer.encode(enc_text);
    } else if (ctx->use_sentencepiece) {
        tokens = ctx->sp_tokenizer.encode(enc_text);
    } else {
        tokens = ctx->wp_tokenizer.encode(enc_text);
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
                                                  ctx->sched, &ctx->compute_meta);
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

extern "C" void crispembed_set_prefix(crispembed_context * ctx, const char * prefix) {
    if (ctx) ctx->prefix = prefix ? prefix : "";
}

extern "C" const char * crispembed_get_prefix(const crispembed_context * ctx) {
    return ctx ? ctx->prefix.c_str() : "";
}

extern "C" const float * crispembed_encode_batch(crispembed_context * ctx,
                                                   const char ** texts,
                                                   int n_texts,
                                                   int * out_n_dim) {
    if (!ctx || !texts || n_texts <= 0) return nullptr;

    // Tokenize all texts (with prefix if set)
    std::vector<embed_tokens> all_tokens(n_texts);
    for (int i = 0; i < n_texts; i++) {
        const char * inp = texts[i];
        std::string prefixed;
        if (!ctx->prefix.empty()) {
            prefixed = ctx->prefix + inp;
            inp = prefixed.c_str();
        }
        if (ctx->use_bpe)
            all_tokens[i] = ctx->bpe_tokenizer.encode(inp);
        else if (ctx->use_sentencepiece)
            all_tokens[i] = ctx->sp_tokenizer.encode(inp);
        else
            all_tokens[i] = ctx->wp_tokenizer.encode(inp);

        // Trim padding
        auto & t = all_tokens[i];
        int actual_len = (int)t.attn_mask.size();
        for (int j = actual_len - 1; j >= 0; j--) {
            if (t.attn_mask[j]) { actual_len = j + 1; break; }
        }
        if (actual_len > 0 && actual_len < (int)t.ids.size()) {
            t.ids.resize(actual_len);
            t.type_ids.resize(actual_len);
            t.attn_mask.resize(actual_len);
        }
    }

    // For encoder models: true batched inference (one graph, all texts)
    const auto & hp = ctx->model.hparams;
    int dim = hp.n_output > 0 ? hp.n_output : hp.n_embd;
    std::vector<std::vector<float>> batch_results;

    if (!ctx->is_decoder) {
        batch_results = encode_tokens_batch(ctx, all_tokens);
    } else {
        // Decoder: sequential (batched decoding more complex)
        for (int i = 0; i < n_texts; i++) {
            auto vec = decoder_encode_tokens(*ctx->dec, ctx->backend, all_tokens[i],
                                              ctx->n_threads, ctx->sched, &ctx->compute_meta);
            batch_results.push_back(std::move(vec));
        }
    }

    if (batch_results.empty() || batch_results[0].empty()) return nullptr;
    dim = (int)batch_results[0].size();

    // Apply Matryoshka and copy results
    int out_dim = (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < dim) ? ctx->matryoshka_dim : dim;
    ctx->last_output.resize(n_texts * out_dim);

    for (int i = 0; i < n_texts; i++) {
        auto & vec = batch_results[i];
        int d = std::min((int)vec.size(), out_dim);
        // Already L2-normalized from encode_tokens_batch / encode_tokens
        // But may need re-normalize after Matryoshka truncation
        if (out_dim < dim) {
            float norm = 0;
            for (int j = 0; j < d; j++) norm += vec[j] * vec[j];
            norm = sqrtf(std::max(norm, 1e-12f));
            float * dst = ctx->last_output.data() + i * out_dim;
            for (int j = 0; j < d; j++) dst[j] = vec[j] / norm;
        } else {
            memcpy(ctx->last_output.data() + i * out_dim, vec.data(), d * sizeof(float));
        }
    }
    if (out_n_dim) *out_n_dim = out_dim;
    return ctx->last_output.data();
}

// ---------------------------------------------------------------------------
// Capability queries
// ---------------------------------------------------------------------------

extern "C" int crispembed_has_sparse(const crispembed_context * ctx) {
    return (ctx && ctx->model.has_sparse) ? 1 : 0;
}

extern "C" int crispembed_has_colbert(const crispembed_context * ctx) {
    return (ctx && ctx->model.has_colbert) ? 1 : 0;
}

extern "C" int crispembed_is_reranker(const crispembed_context * ctx) {
    return (ctx && ctx->model.is_reranker) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Sparse encode (BGE-M3 sparse head)
// ---------------------------------------------------------------------------

extern "C" int crispembed_encode_sparse(crispembed_context * ctx,
                                         const char        * text,
                                         const int32_t    ** out_indices,
                                         const float      ** out_values) {
    if (!ctx || !text || !ctx->model.has_sparse || ctx->is_decoder) return 0;

    embed_tokens tokens;
    if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(text);
    else                        tokens = ctx->wp_tokenizer.encode(text);

    // Trim to actual (non-padded) length
    int T = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { T = i + 1; break; }
    }
    if (T == 0) return 0;
    tokens.ids.resize(T);
    tokens.type_ids.resize(T);
    tokens.attn_mask.resize(T);

    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 1, &raw_T);
    if (raw.empty()) return 0;

    // Detect sparse style from weight output dimension (ne[1] after shape reversal in gguf):
    //   BGE-M3: sparse_linear is Linear(H, 1)  → weight stored [H,1] in ggml → ne[1]=1
    //   SPLADE: sparse_linear is Linear(H, V)  → weight stored [H,V] in ggml → ne[1]=V
    // The graph output is [out_dim, T] → out_dim = sparse_linear_w->ne[1].
    int out_dim = (int)ctx->model.sparse_linear_w->ne[1];

    ctx->last_sparse_indices.clear();
    ctx->last_sparse_values.clear();

    if (out_dim == 1) {
        // BGE-M3 style: raw is [1, T] — one scalar per token.
        // Scatter to vocab positions via input_ids, take max per vocab id.
        std::unordered_map<int32_t, float> vocab_weights;
        for (int t = 0; t < raw_T; t++) {
            if (!tokens.attn_mask[t]) continue;
            float weight = raw[t];  // element [0, t]
            if (weight <= 0.0f) continue;
            int32_t vid = tokens.ids[t];
            auto it = vocab_weights.find(vid);
            if (it == vocab_weights.end() || it->second < weight)
                vocab_weights[vid] = weight;
        }
        for (auto & kv : vocab_weights) {
            ctx->last_sparse_indices.push_back(kv.first);
            ctx->last_sparse_values.push_back(kv.second);
        }
    } else {
        // SPLADE style: raw is [V, T] where V = vocab_size.
        // Max-pool over T → [V], apply log(1+x), filter zeros.
        // raw layout: element [v, t] at offset v + t * out_dim
        for (int v = 0; v < out_dim; v++) {
            float max_w = 0.0f;
            for (int t = 0; t < raw_T; t++) {
                if (!tokens.attn_mask[t]) continue;
                float w = raw[v + t * out_dim];
                if (w > max_w) max_w = w;
            }
            if (max_w <= 0.0f) continue;
            ctx->last_sparse_indices.push_back((int32_t)v);
            ctx->last_sparse_values.push_back(logf(1.0f + max_w));  // SPLADE uses log(1+ReLU)
        }
    }

    int n = (int)ctx->last_sparse_indices.size();
    if (out_indices) *out_indices = ctx->last_sparse_indices.data();
    if (out_values)  *out_values  = ctx->last_sparse_values.data();
    return n;
}

// ---------------------------------------------------------------------------
// Multi-vector encode (ColBERT head)
// ---------------------------------------------------------------------------

extern "C" const float * crispembed_encode_multivec(crispembed_context * ctx,
                                                      const char         * text,
                                                      int                * out_n_tokens,
                                                      int                * out_dim) {
    if (!ctx || !text || !ctx->model.has_colbert || ctx->is_decoder) return nullptr;

    embed_tokens tokens;
    if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(text);
    else                        tokens = ctx->wp_tokenizer.encode(text);

    // Count real tokens (non-padded)
    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { T_real = i + 1; break; }
    }
    if (T_real == 0) return nullptr;
    tokens.ids.resize(T_real);
    tokens.type_ids.resize(T_real);
    tokens.attn_mask.resize(T_real);

    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 2, &raw_T);
    if (raw.empty()) return nullptr;

    const int dim = ctx->model.colbert_dim;
    // raw is [colbert_dim, T_real] — L2 normalize each token vector
    ctx->last_multivec.resize(dim * raw_T);
    for (int t = 0; t < raw_T; t++) {
        float * vec = raw.data() + t * dim;
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) norm += vec[d] * vec[d];
        norm = sqrtf(std::max(norm, 1e-12f));
        float * out = ctx->last_multivec.data() + t * dim;
        for (int d = 0; d < dim; d++) out[d] = vec[d] / norm;
    }
    ctx->last_multivec_n_tokens = raw_T;
    ctx->last_multivec_dim      = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim)      *out_dim      = dim;
    return ctx->last_multivec.data();
}

// ---------------------------------------------------------------------------
// Reranker (cross-encoder score)
// ---------------------------------------------------------------------------

extern "C" float crispembed_rerank(crispembed_context * ctx,
                                    const char         * query,
                                    const char         * document) {
    if (!ctx || !query || !document || !ctx->model.is_reranker || ctx->is_decoder)
        return 0.0f;

    embed_tokens tokens;
    if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode_pair(query, document);
    else
        tokens = ctx->wp_tokenizer.encode_pair(query, document);

    // Trim to real tokens
    int T = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { T = i + 1; break; }
    }
    if (T == 0) return 0.0f;
    tokens.ids.resize(T);
    tokens.type_ids.resize(T);
    tokens.attn_mask.resize(T);

    int raw_T = 0;
    // Run dense encoder (mode=0), we read CLS token ourselves
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
    if (raw.empty()) return 0.0f;

    const int H = ctx->model.hparams.n_embd;
    // CLS token is position 0 in encoder_out [H, T]
    const float * cls_vec = raw.data();  // first H floats = token 0

    float score = 0.0f;
    if (ctx->model.classifier_2layer) {
        // 2-layer RobertaClassificationHead: cls → dense[H,H] → tanh → out_proj[1,H]
        std::vector<float> dw(H * H), db(H), ow(H);
        ggml_backend_tensor_get(ctx->model.classifier_dense_w, dw.data(), 0, H*H*sizeof(float));
        ggml_backend_tensor_get(ctx->model.classifier_dense_b, db.data(), 0, H*sizeof(float));
        ggml_backend_tensor_get(ctx->model.classifier_out_w,   ow.data(), 0, H*sizeof(float));
        std::vector<float> hidden(H);
        for (int i = 0; i < H; i++) {
            float acc = db[i];
            for (int j = 0; j < H; j++) acc += cls_vec[j] * dw[i * H + j];
            hidden[i] = std::tanh(acc);
        }
        for (int i = 0; i < H; i++) score += hidden[i] * ow[i];
        if (ctx->model.classifier_out_b) {
            float bias = 0.0f;
            ggml_backend_tensor_get(ctx->model.classifier_out_b, &bias, 0, sizeof(float));
            score += bias;
        }
    } else {
        // 1-layer: score = cls_vec · classifier_w + bias
        std::vector<float> cw(H);
        ggml_backend_tensor_get(ctx->model.classifier_w, cw.data(), 0, H * sizeof(float));
        for (int h = 0; h < H; h++) score += cls_vec[h] * cw[h];
        if (ctx->model.classifier_b) {
            float bias = 0.0f;
            ggml_backend_tensor_get(ctx->model.classifier_b, &bias, 0, sizeof(float));
            score += bias;
        }
    }
    return score;
}

extern "C" void crispembed_free(crispembed_context * ctx) {
    if (!ctx) return;
    if (ctx->qkv_buf) { ggml_backend_buffer_free(ctx->qkv_buf); ctx->qkv_buf = nullptr; }
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
