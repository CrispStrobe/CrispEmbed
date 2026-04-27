// crispembed.cpp — BERT/MiniLM encoder via ggml graph.

#include "crispembed.h"
#include "model_mgr.h"
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
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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

static ggml_backend_t crispembed_init_backend(int n_threads) {
    const char * force_cpu = std::getenv("CRISPEMBED_FORCE_CPU");
    if (force_cpu && force_cpu[0] && std::strcmp(force_cpu, "0") != 0) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        if (cpu) {
            ggml_backend_cpu_set_n_threads(cpu, n_threads);
            fprintf(stderr, "crispembed: forcing CPU backend via CRISPEMBED_FORCE_CPU\n");
        }
        return cpu;
    }
    return ggml_backend_init_best();
}

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
    ggml_tensor * ffn_gate_w = nullptr;     // SwiGLU gate (NomicBERT, separate)
    ggml_tensor * ffn_up_gate_w = nullptr; // Fused gate+up [2*inter, H] for ggml_geglu
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
    ggml_tensor * final_norm_w = nullptr;  // ModernBERT final norm (pre-LN models)

    // Encoder layers
    std::vector<embed_layer> layers;

    // Optional pooler / projection
    ggml_tensor * pooler_w     = nullptr;
    ggml_tensor * pooler_b     = nullptr;

    // Sparse retrieval head (BGE-M3): Linear(n_embd, 1)
    ggml_tensor * sparse_linear_w  = nullptr;  // [H, 1]
    ggml_tensor * sparse_linear_b  = nullptr;  // [1], optional
    // SPLADE/MLM head: transform(H→H) + LN + decode(H→V) → sparse
    ggml_tensor * mlm_transform_w  = nullptr;  // [H, H]
    ggml_tensor * mlm_transform_b  = nullptr;  // [H]
    ggml_tensor * mlm_ln_w         = nullptr;  // [H]
    ggml_tensor * mlm_ln_b         = nullptr;  // [H]
    ggml_tensor * mlm_bias         = nullptr;  // [V] (decoder bias; weight tied to token_embd)
    bool has_mlm_head = false;
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

static bool validate_encoder_model(const embed_model & m, bool pre_ln) {
    bool ok = true;
    for (size_t il = 0; il < m.layers.size(); il++) {
        const auto & L = m.layers[il];
        auto require = [&](bool cond, const char * name) {
            if (!cond) {
                fprintf(stderr, "crispembed: missing required tensor layer=%zu name=%s\n", il, name);
                ok = false;
            }
        };

        require(L.q_w || L.qkv_w, "attn.q.weight");
        require(L.k_w || L.qkv_w, "attn.k.weight");
        require(L.v_w || L.qkv_w, "attn.v.weight");
        require(L.o_w, "attn.o.weight");
        require(L.fc2_w, "ffn.fc2.weight");
        require(L.ffn_up_gate_w || L.fc1_w, "ffn input weights");

        if (pre_ln) {
            require(L.ln1_w, "ln1.weight");
            require(L.ln2_w, "ln2.weight");
        } else {
            require(L.ln1_w, "ln1.weight");
            require(L.ln2_w, "ln2.weight");
            require(L.fc1_w || L.ffn_up_gate_w, "ffn.fc1.weight");
        }
    }
    return ok;
}

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
    float rope_theta = 10000.0f;       // default/sliding theta
    float rope_theta_global = 0.0f;    // global attention theta (ModernBERT, 0 = same as rope_theta)
    int   global_attn_every_n = 0;     // ModernBERT: every Nth layer uses global attention (0 = all same)
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
    // Audio path — opaque pointer into bidirlm_audio.cpp (lazily inited on
    // first crispembed_encode_audio call). Built only when CRISPEMBED_HAS_CRISP_AUDIO.
    void * audio_ctx = nullptr;
    std::string model_path_for_audio;
    // Vision path — opaque pointer into bidirlm_vision.cpp (lazily inited on
    // first encode_image call). Always compiled in (no sibling-lib dependency).
    void * vision_ctx = nullptr;
    int    vision_load_attempted = 0;  // avoid re-loading after a failed open
    std::vector<float> last_vision_out;  // owned buffer for the last encode_image* call
    int last_vision_dim = 0;
    int last_vision_n_merged = 0;
    int last_vision_n_deepstack = 0;
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
        const int64_t k = gguf_find_key(g, key);
        return k >= 0 ? (int)gguf_get_val_u32(g, k) : def;
    };
    auto f32 = [&](const char * key, float def) -> float {
        const int64_t k = gguf_find_key(g, key);
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
    // RoPE and pre-LN flags — MUST be read before gguf_free(g)
    ctx->rope_theta         = f32("bert.rope_theta", 10000.0f);
    ctx->rope_theta_global  = f32("bert.rope_theta_global", 0.0f);
    ctx->global_attn_every_n = u32("bert.global_attn_every_n", 0);
    ctx->pre_ln             = u32("bert.pre_ln", 0) != 0;

    // Load tokenizer vocab from GGUF metadata
    const int64_t ki = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (ki >= 0) {
        const int n = (int)gguf_get_arr_n(g, ki);
        std::vector<std::string> vocab(n);
        for (int i = 0; i < n; i++)
            vocab[i] = gguf_get_arr_str(g, ki, i);

        // Load scores if available (SentencePiece models)
        std::vector<float> scores;
        const int64_t si = gguf_find_key(g, "tokenizer.ggml.scores");
        if (si >= 0 && gguf_get_arr_type(g, si) == GGUF_TYPE_FLOAT32) {
            int sn = (int)gguf_get_arr_n(g, si);
            scores.resize(sn);
            const float * sd = reinterpret_cast<const float *>(gguf_get_arr_data(g, si));
            std::memcpy(scores.data(), sd, sn * sizeof(float));
        }

        // Detect tokenizer type: 0=WordPiece, 1=BPE, 2=SentencePiece
        int tokenizer_type = u32("tokenizer.ggml.type", 0);
        if (tokenizer_type == 2 || (tokenizer_type == 0 && n > 100000)) {
            // SentencePiece / XLM-RoBERTa
            int bos_id = u32("tokenizer.ggml.bos_token_id", 0);
            int eos_id = u32("tokenizer.ggml.eos_token_id", 2);
            int unk_id = u32("tokenizer.ggml.unknown_token_id", 3);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 1);
            ctx->sp_tokenizer.load(vocab, scores, bos_id, eos_id, unk_id, pad_id, hp.n_max_tokens);
            ctx->use_sentencepiece = true;
            fprintf(stderr, "crispembed: using SentencePiece tokenizer (%d tokens, %zu scores)\n",
                    n, scores.size());
        } else if (tokenizer_type == 1) {
            // BPE (GPT-2 style, ModernBERT, etc.)
            int cls_id = u32("tokenizer.ggml.cls_token_id", 0);
            int sep_id = u32("tokenizer.ggml.sep_token_id", 2);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 1);

            // BPE merges stored as tensor (newline-separated blob)
            // Merges will be loaded after weight loading (from tensor "tokenizer.merges")
            std::vector<std::string> empty_merges;
            // For encoder BPE: eos=SEP, suffix=-1 (handled by encode), bos=CLS
            ctx->bpe_tokenizer.load(vocab, empty_merges, sep_id, pad_id, -1,
                                     cls_id, false, hp.n_max_tokens);
            ctx->use_bpe = true;
            fprintf(stderr, "crispembed: using BPE tokenizer (%d tokens)\n", n);
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
    ctx->backend = crispembed_init_backend(ctx->n_threads);
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
    auto get_any = [&](std::initializer_list<std::string> names) -> ggml_tensor * {
        for (const auto & name : names) {
            if (ggml_tensor * tensor = get(name)) {
                return tensor;
            }
        }
        return nullptr;
    };

    // Embeddings
    m.token_embd = get("token_embd.weight");
    m.pos_embd   = get("position_embd.weight");
    m.type_embd  = get_any({"token_type_embd.weight", "token_types.weight"});
    m.embd_ln_w  = get_any({"embd_ln.weight", "token_embd_norm.weight"});
    m.embd_ln_b  = get_any({"embd_ln.bias", "token_embd_norm.bias"});
    m.rel_attn_bias = get("rel_attn_bias.weight");
    m.rel_embd      = get("rel_embd.weight");
    m.encoder_ln_w  = get("encoder_ln.weight");
    m.encoder_ln_b  = get("encoder_ln.bias");
    m.final_norm_w  = get("final_norm.weight");

    if (!m.token_embd) {
        fprintf(stderr, "crispembed: missing token_embd.weight\n");
        return false;
    }
    // NomicBERT/ModernBERT: RoPE-based encoders lack position embeddings
    if (!m.pos_embd) {
        ctx->use_rope = true;
        fprintf(stderr, "crispembed: no position embeddings, using RoPE (theta=%.0f%s)\n",
                ctx->rope_theta, ctx->pre_ln ? ", pre-LN" : "");
    }

    // Encoder layers
    m.layers.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        auto pfx = "enc." + std::to_string(il) + ".";
        auto blk = "blk." + std::to_string(il) + ".";
        auto & L = m.layers[il];
        L.ln1_w = get_any({pfx + "ln1.weight", blk + "attn_output_norm.weight"});
        L.ln1_b = get_any({pfx + "ln1.bias", blk + "attn_output_norm.bias"});
        L.q_w   = get_any({pfx + "attn.q.weight", blk + "attn_q.weight"});
        L.q_b   = get_any({pfx + "attn.q.bias", blk + "attn_q.bias"});
        L.k_w   = get_any({pfx + "attn.k.weight", blk + "attn_k.weight"});
        L.k_b   = get_any({pfx + "attn.k.bias", blk + "attn_k.bias"});
        L.v_w   = get_any({pfx + "attn.v.weight", blk + "attn_v.weight"});
        L.v_b   = get_any({pfx + "attn.v.bias", blk + "attn_v.bias"});
        L.o_w   = get_any({pfx + "attn.o.weight", blk + "attn_output.weight"});
        L.o_b   = get_any({pfx + "attn.o.bias", blk + "attn_output.bias"});
        L.ln2_w = get_any({pfx + "ln2.weight", blk + "layer_output_norm.weight"});
        L.ln2_b = get_any({pfx + "ln2.bias", blk + "layer_output_norm.bias"});
        L.fc1_w = get_any({pfx + "ffn.fc1.weight", blk + "ffn_up.weight"});
        L.fc1_b = get_any({pfx + "ffn.fc1.bias", blk + "ffn_up.bias"});
        L.fc2_w = get_any({pfx + "ffn.fc2.weight", blk + "ffn_down.weight"});
        L.fc2_b = get_any({pfx + "ffn.fc2.bias", blk + "ffn_down.bias"});
        L.ffn_gate_w    = get(pfx + "ffn_gate.weight");     // SwiGLU gate (NomicBERT)
        L.ffn_up_gate_w = get(pfx + "ffn_up_gate.weight"); // Fused gate+up (ModernBERT/GTE v1.5)
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
    // SPLADE/MLM head
    m.mlm_transform_w = get("mlm_transform.weight");
    m.mlm_transform_b = get("mlm_transform.bias");
    m.mlm_ln_w        = get("mlm_ln.weight");
    m.mlm_ln_b        = get("mlm_ln.bias");
    m.mlm_bias        = get("mlm_bias");
    m.has_mlm_head    = m.mlm_transform_w != nullptr;
    if (m.has_mlm_head) fprintf(stderr, "crispembed: MLM/SPLADE head loaded\n");

    m.has_sparse  = m.sparse_linear_w != nullptr || m.has_mlm_head;
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

    // Load BPE merges from tensor (stored as newline-separated UTF-8 blob)
    if (ctx->use_bpe) {
        ggml_tensor * merge_t = get("tokenizer.merges");
        if (merge_t) {
            size_t nbytes = ggml_nbytes(merge_t);
            std::vector<uint8_t> blob(nbytes);
            ggml_backend_tensor_get(merge_t, blob.data(), 0, nbytes);
            // Parse newline-separated merges
            std::vector<std::string> merges;
            std::string current;
            for (size_t i = 0; i < nbytes; i++) {
                if (blob[i] == '\n') {
                    if (!current.empty()) merges.push_back(current);
                    current.clear();
                } else {
                    current += (char)blob[i];
                }
            }
            if (!current.empty()) merges.push_back(current);
            // Re-load BPE tokenizer with merges (preserve suffix_id=-1 for encoder)
            int cls_id = ctx->bpe_tokenizer.bos_id();
            int sep_id = ctx->bpe_tokenizer.eos_id();
            int pad_id = ctx->bpe_tokenizer.pad_id();
            ctx->bpe_tokenizer.load(ctx->bpe_tokenizer.get_vocab(), merges,
                                     sep_id, pad_id, -1, cls_id, false, hp.n_max_tokens);
            fprintf(stderr, "crispembed: loaded %zu BPE merges from tensor\n", merges.size());
        }
    }

    fprintf(stderr, "crispembed: loaded %d layers, %d dims, %d vocab\n",
    // Temp debug: will be removed
            hp.n_layer, hp.n_embd, hp.n_vocab);
    if (!validate_encoder_model(m, ctx->pre_ln)) {
        fprintf(stderr, "crispembed: model validation failed\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph: build fresh each call (no_alloc=true), scheduler handles allocation
// ---------------------------------------------------------------------------

// Build encoder graph for T tokens × B batch items.
// mode: 0=dense (encoder_out), 1=sparse (sparse_out [1,T]), 2=colbert (colbert_out [dim,T])
// When B=1: standard single-text graph.
// When B>1: batched graph with 4D attention via flash_attn_ext.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
static ggml_cgraph * build_encoder_graph(crispembed_context * ctx, int T, int B = 1, int mode = 0) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int H = hp.n_embd;
    const int n_heads = hp.n_head;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;
    const int TB = T * B;  // total tokens in batch

    int graph_size = std::max(4096, hp.n_layer * 40 + 512);

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

        // Optional RoPE for encoder models without position embeddings (NomicBERT/ModernBERT)
        // Apply before permute: Q/K shape is [hd, nh, T, B], RoPE uses ne[2]=T
        if (rope_pos) {
            // Per-layer theta: ModernBERT alternates sliding/global attention
            float layer_theta = ctx->rope_theta;
            if (ctx->global_attn_every_n > 0 && ctx->rope_theta_global > 0.0f) {
                bool is_global = (il % ctx->global_attn_every_n == 0);
                layer_theta = is_global ? ctx->rope_theta_global : ctx->rope_theta;
            }
            Q = ggml_rope_ext(gctx, Q, rope_pos, nullptr,
                              head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens,
                              layer_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(gctx, K, rope_pos, nullptr,
                              head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens,
                              layer_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
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
            ggml_tensor * rel_pos_idx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)T * T);
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
            ggml_tensor * P_flat = ggml_reshape_2d(gctx, P, H, (int64_t)T * T);  // [H, T*T]
            ggml_tensor * Pk = ggml_mul_mat(gctx, L.k_w, P_flat);  // [H, T*T]
            if (L.k_b) Pk = ggml_add(gctx, Pk, ggml_repeat(gctx, L.k_b,
                ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, (int64_t)T * T)));
            Pk = ggml_reshape_3d(gctx, Pk, head_dim, n_heads, (int64_t)T * T);
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
        if (L.ffn_up_gate_w) {
            // Fused GeGLU (ModernBERT/GTE v1.5): single matmul → ggml_geglu → down
            ggml_tensor * up_gate = ggml_mul_mat(gctx, L.ffn_up_gate_w, cur);  // [2*inter, T]
            ffn = ggml_geglu(gctx, up_gate);  // fused: gelu(first_half) * second_half → [inter, T]
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
        } else if (L.ffn_gate_w) {
            // Separate SwiGLU (NomicBERT)
            ggml_tensor * up   = ggml_mul_mat(gctx, L.fc1_w, cur);
            ggml_tensor * gate = ggml_mul_mat(gctx, L.ffn_gate_w, cur);
            gate = ggml_silu(gctx, gate);
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
        // Apply final norm for pre-LN models (ModernBERT)
        if (m.final_norm_w) {
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, m.final_norm_w);
        }
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

static ggml_tensor * graph_tensor_or_log(ggml_cgraph * gf, const char * name) {
    ggml_tensor * tensor = ggml_graph_get_tensor(gf, name);
    if (!tensor) {
        fprintf(stderr, "crispembed: missing graph tensor '%s'\n", name);
    }
    return tensor;
}

static bool crispembed_debug_encode_enabled() {
    const char * value = std::getenv("CRISPEMBED_DEBUG_ENCODE");
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static void debug_encode_stage(const char * stage, int T, int B, int mode) {
    if (crispembed_debug_encode_enabled()) {
        fprintf(stderr, "crispembed: encode debug stage=%s T=%d B=%d mode=%d\n", stage, T, B, mode);
    }
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
    debug_encode_stage("encode_tokens:start", T, 1, 0);

    // Reserve scheduler for this bucket if not already reserved
    if (ctx->reserved_T != T_bucket) {
        debug_encode_stage("encode_tokens:reserve-build", T_bucket, 1, 0);
        ggml_cgraph * measure_gf = build_encoder_graph(ctx, T_bucket);
        debug_encode_stage("encode_tokens:reserve", T_bucket, 1, 0);
        ggml_backend_sched_reserve(ctx->sched, measure_gf);
        ctx->reserved_T = T_bucket;
    }

    // Build graph for actual T (metadata only — scheduler already has buffers)
    debug_encode_stage("encode_tokens:graph-build", T, 1, 0);
    ggml_cgraph * gf = build_encoder_graph(ctx, T);

    debug_encode_stage("encode_tokens:alloc-reset", T, 1, 0);
    ggml_backend_sched_reset(ctx->sched);
    debug_encode_stage("encode_tokens:alloc", T, 1, 0);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate encoder graph\n");
        return {};
    }

    // Set input data via backend API (works for both CPU and GPU tensors)
    std::vector<int32_t> tok_data(tokens.ids.begin(), tokens.ids.end());
    ggml_tensor * tok_ids = graph_tensor_or_log(gf, "tok_ids");
    if (!tok_ids) return {};
    debug_encode_stage("encode_tokens:set-tok", T, 1, 0);
    ggml_backend_tensor_set(tok_ids, tok_data.data(), 0, T * sizeof(int32_t));

    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = t + ctx->pos_offset;
    ggml_tensor * pos_ids = graph_tensor_or_log(gf, "pos_ids");
    if (!pos_ids) return {};
    debug_encode_stage("encode_tokens:set-pos", T, 1, 0);
    ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T * sizeof(int32_t));

    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(tokens.type_ids.begin(), tokens.type_ids.end());
        ggml_tensor * type_ids = graph_tensor_or_log(gf, "type_ids");
        if (!type_ids) return {};
        debug_encode_stage("encode_tokens:set-type", T, 1, 0);
        ggml_backend_tensor_set(type_ids, type_data.data(), 0, T * sizeof(int32_t));
    }

    // MPNet relative position bias (precomputed for this sequence length, F16)
    if (ctx->model.rel_attn_bias) {
        ggml_tensor * bias_t = ggml_graph_get_tensor(gf, "rel_pos_bias");
        if (bias_t) {
            debug_encode_stage("encode_tokens:set-rel-bias", T, 1, 0);
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
            debug_encode_stage("encode_tokens:set-rel-idx", T, 1, 0);
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
    debug_encode_stage("encode_tokens:compute", T, 1, 0);
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: encoder compute failed\n");
        return {};
    }

    // Read output (works whether tensor is on GPU or CPU)
    // Read encoder output [H, T] via backend API (works for GPU and CPU)
    ggml_tensor * out = graph_tensor_or_log(gf, "encoder_out");
    if (!out) return {};
    debug_encode_stage("encode_tokens:get-output", T, 1, 0);
    std::vector<float> out_buf(H * T);
    ggml_backend_tensor_get(out, out_buf.data(), 0, H * T * sizeof(float));
    const float * out_data = out_buf.data();

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
    const int B = (int)batch.size();
    if (B == 0) return {};
    std::vector<std::vector<float>> results;
    results.reserve(B);

    // The previous fused batch path padded sequences but did not mask padded
    // tokens inside attention, so shorter items diverged from single-item
    // encoding. Prefer correctness here until a masked fused path exists.
    for (const auto & tokens : batch) {
        results.push_back(encode_tokens(ctx, tokens));
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
    const int T = (int)tokens.ids.size();
    if (out_T) *out_T = T;

    int T_bucket = bucket_seq_len(T);
    int & reserved = (mode == 1) ? ctx->reserved_T_sparse
                   : (mode == 2) ? ctx->reserved_T_colbert
                   : ctx->reserved_T;
    debug_encode_stage("run_encoder_raw:start", T, 1, mode);

    if (reserved != T_bucket) {
        debug_encode_stage("run_encoder_raw:reserve-build", T_bucket, 1, mode);
        ggml_cgraph * measure_gf = build_encoder_graph(ctx, T_bucket, 1, mode);
        debug_encode_stage("run_encoder_raw:reserve", T_bucket, 1, mode);
        ggml_backend_sched_reserve(ctx->sched, measure_gf);
        reserved = T_bucket;
    }

    debug_encode_stage("run_encoder_raw:graph-build", T, 1, mode);
    ggml_cgraph * gf = build_encoder_graph(ctx, T, 1, mode);
    debug_encode_stage("run_encoder_raw:alloc-reset", T, 1, mode);
    ggml_backend_sched_reset(ctx->sched);
    debug_encode_stage("run_encoder_raw:alloc", T, 1, mode);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate graph (mode=%d)\n", mode);
        return {};
    }

    std::vector<int32_t> tok_data(tokens.ids.begin(), tokens.ids.end());
    ggml_tensor * tok_ids = graph_tensor_or_log(gf, "tok_ids");
    if (!tok_ids) return {};
    debug_encode_stage("run_encoder_raw:set-tok", T, 1, mode);
    ggml_backend_tensor_set(tok_ids, tok_data.data(), 0, T * sizeof(int32_t));
    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = t + ctx->pos_offset;
    ggml_tensor * pos_ids = graph_tensor_or_log(gf, "pos_ids");
    if (!pos_ids) return {};
    debug_encode_stage("run_encoder_raw:set-pos", T, 1, mode);
    ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T * sizeof(int32_t));
    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(tokens.type_ids.begin(), tokens.type_ids.end());
        ggml_tensor * type_ids = graph_tensor_or_log(gf, "type_ids");
        if (!type_ids) return {};
        debug_encode_stage("run_encoder_raw:set-type", T, 1, mode);
        ggml_backend_tensor_set(type_ids, type_data.data(), 0, T * sizeof(int32_t));
    }

    debug_encode_stage("run_encoder_raw:compute", T, 1, mode);
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: compute failed (mode=%d)\n", mode);
        return {};
    }

    const char * out_name = (mode == 1) ? "sparse_out"
                          : (mode == 2) ? "colbert_out"
                          : "encoder_out";
    ggml_tensor * out = graph_tensor_or_log(gf, out_name);
    if (!out) return {};
    debug_encode_stage("run_encoder_raw:get-output", T, 1, mode);

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
    if (model_path) ctx->model_path_for_audio = model_path;

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
        ctx->backend = crispembed_init_backend(ctx->n_threads);
        ctx->backends.push_back(ctx->backend);
        if (!ctx->backend) {
            delete ctx;
            return nullptr;
        }
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

        const int graph_nodes = std::max(4096, ctx->dec->n_layer * 50 + 256);
        ctx->sched = ggml_backend_sched_new(
            ctx->backends.data(), nullptr, (int)ctx->backends.size(),
            graph_nodes, false, false);
        ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes
                               + ggml_graph_overhead_custom(graph_nodes, false));

        // Load BPE tokenizer from GGUF
        gguf_init_params gp2 = { true, nullptr };
        gguf_context * g2 = gguf_init_from_file(model_path, gp2);
        if (g2) {
            const int64_t ki2 = gguf_find_key(g2, "tokenizer.ggml.tokens");
            const int64_t mi2 = gguf_find_key(g2, "tokenizer.ggml.merges");
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
                    const int64_t k = gguf_find_key(g2, key);
                    return k >= 0 ? (int)gguf_get_val_u32(g2, k) : def;
                };
                int eos_id = u32g("tokenizer.ggml.eos_token_id", 151645);
                int pad_id = u32g("tokenizer.ggml.padding_token_id", 151643);
                int bos_id = u32g("tokenizer.ggml.bos_token_id", -1);
                const int64_t ki_sfx = gguf_find_key(g2, "tokenizer.ggml.suffix_token_id");
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

extern "C" const char * crispembed_cache_dir(void) {
    static std::string value;
    value = crispembed_mgr::cache_dir();
    return value.c_str();
}

extern "C" const char * crispembed_resolve_model(const char * arg, int auto_download) {
    static std::string value;
    value.clear();
    if (!arg) return value.c_str();
    value = crispembed_mgr::resolve_model(arg, auto_download != 0);
    return value.c_str();
}

extern "C" int crispembed_n_models(void) {
    return crispembed_mgr::n_models();
}

extern "C" const char * crispembed_model_name(int index) {
    const char * value = crispembed_mgr::model_name(index);
    return value ? value : "";
}

extern "C" const char * crispembed_model_desc(int index) {
    const char * value = crispembed_mgr::model_desc(index);
    return value ? value : "";
}

extern "C" const char * crispembed_model_filename(int index) {
    const char * value = crispembed_mgr::model_filename(index);
    return value ? value : "";
}

extern "C" const char * crispembed_model_size(int index) {
    const char * value = crispembed_mgr::model_size(index);
    return value ? value : "";
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

    // NOLINTNEXTLINE(bugprone-branch-clone)
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
    const int dim = (int)batch_results[0].size();

    // Apply Matryoshka and copy results
    int out_dim = (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < dim) ? ctx->matryoshka_dim : dim;
    ctx->last_output.resize(n_texts * out_dim);

    for (int i = 0; i < n_texts; i++) {
        const auto & vec = batch_results[i];
        if ((int)vec.size() != dim) {
            fprintf(stderr, "crispembed: batch encode failed for item %d\n", i);
            return nullptr;
        }
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

    // SPLADE via MLM head: compute sparse from per-token encoder hidden states
    if (ctx->model.has_mlm_head) {
        const int H = ctx->model.hparams.n_embd;
        const int V = ctx->model.hparams.n_vocab;
        const float ln_eps = ctx->model.hparams.layer_norm_eps;

        // Get per-token encoder output [H, T] via mode=0 (dense) graph
        int raw_T = 0;
        std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
        if (raw.empty() || raw_T == 0) return 0;

        // Read MLM head weights from GPU/CPU backend
        std::vector<float> tw(H * H), tb(H), lnw(H), lnb(H);
        ggml_backend_tensor_get(ctx->model.mlm_transform_w, tw.data(), 0, H * H * sizeof(float));
        ggml_backend_tensor_get(ctx->model.mlm_transform_b, tb.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ctx->model.mlm_ln_w, lnw.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ctx->model.mlm_ln_b, lnb.data(), 0, H * sizeof(float));
        std::vector<float> emb_w(V * H);
        ggml_backend_tensor_get(ctx->model.token_embd, emb_w.data(), 0, V * H * sizeof(float));
        std::vector<float> mlm_b(V, 0.0f);
        if (ctx->model.mlm_bias)
            ggml_backend_tensor_get(ctx->model.mlm_bias, mlm_b.data(), 0, V * sizeof(float));

        // SPLADE: for each token, compute MLM logits, apply log(1+ReLU), max-pool
        std::vector<float> max_logits(V, 0.0f);

        for (int t = 0; t < std::min(raw_T, T); t++) {
            if (!tokens.attn_mask[t]) continue;
            const float * ht = raw.data() + t * H;

            // MLM transform: h' = GELU(W*h + b)
            std::vector<float> h(H);
            for (int i = 0; i < H; i++) {
                float v = tb[i];
                for (int j = 0; j < H; j++) v += tw[i * H + j] * ht[j];
                v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                h[i] = v;
            }

            // LayerNorm
            float mean = 0, var = 0;
            for (int i = 0; i < H; i++) mean += h[i];
            mean /= H;
            for (int i = 0; i < H; i++) { float d = h[i] - mean; var += d * d; }
            var = 1.0f / sqrtf(var / H + ln_eps);
            for (int i = 0; i < H; i++) h[i] = (h[i] - mean) * var * lnw[i] + lnb[i];

            // Decode to vocab logits + SPLADE activation
            for (int v = 0; v < V; v++) {
                float logit = mlm_b[v];
                for (int j = 0; j < H; j++) logit += emb_w[v * H + j] * h[j];
                if (logit > 0.0f) {
                    float sv = logf(1.0f + logit);
                    if (sv > max_logits[v]) max_logits[v] = sv;
                }
            }
        }

        // Collect non-zero entries (skip special tokens)
        ctx->last_sparse_indices.clear();
        ctx->last_sparse_values.clear();
        for (int v = 0; v < V; v++) {
            if (max_logits[v] > 0.0f && v != 0 && v != 101 && v != 102) {
                ctx->last_sparse_indices.push_back(v);
                ctx->last_sparse_values.push_back(max_logits[v]);
            }
        }

        if (out_indices) *out_indices = ctx->last_sparse_indices.data();
        if (out_values) *out_values = ctx->last_sparse_values.data();
        return (int)ctx->last_sparse_indices.size();
    }

    // BGE-M3 sparse path (mode=1 graph with sparse_linear head)
    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 1, &raw_T);
    if (raw.empty()) return 0;

    if (!ctx->model.sparse_linear_w) return 0;
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
        for (const auto & kv : vocab_weights) {
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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
        const float * vec = raw.data() + t * dim;
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

// ---------------------------------------------------------------------------
// Audio encoding via crisp_audio (BidirLM-Omni and similar)
// ---------------------------------------------------------------------------
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
namespace bidirlm_audio {
struct context;
context* open(const char* gguf_path, int n_threads, bool use_gpu);
const float* encode(context* ctx, const float* pcm, int n_samples, int* out_dim);
void close(context* ctx);
}

static bidirlm_audio::context * audio_lazy_open(crispembed_context * ctx) {
    if (!ctx) return nullptr;
    if (ctx->audio_ctx) return (bidirlm_audio::context *)ctx->audio_ctx;
    if (ctx->model_path_for_audio.empty()) return nullptr;
    bool use_gpu = ctx->backend && !ggml_backend_is_cpu(ctx->backend);
    auto * a = bidirlm_audio::open(ctx->model_path_for_audio.c_str(),
                                    ctx->n_threads, use_gpu);
    ctx->audio_ctx = a;
    return a;
}
#endif

extern "C" int crispembed_has_audio(const crispembed_context * /*ctx*/) {
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
    // Only the loader knows for sure (a GGUF either has the audio tower or
    // doesn't). We could prefetch the metadata here, but doing it lazily
    // matches the rest of the API: callers check the return of
    // crispembed_encode_audio() instead.
    return 1;
#else
    return 0;
#endif
}

extern "C" const float * crispembed_encode_audio(crispembed_context * ctx,
                                                  const float * pcm_samples,
                                                  int n_samples,
                                                  int * out_dim) {
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
    auto * a = audio_lazy_open(ctx);
    if (!a) return nullptr;
    return bidirlm_audio::encode(a, pcm_samples, n_samples, out_dim);
#else
    (void)ctx; (void)pcm_samples; (void)n_samples;
    if (out_dim) *out_dim = 0;
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Vision encoding via bidirlm_vision (BidirLM-Omni)
// ---------------------------------------------------------------------------
#include "bidirlm_vision.h"

static bidirlm_vision::context * vision_lazy_open(crispembed_context * ctx) {
    if (!ctx) return nullptr;
    if (ctx->vision_ctx) return (bidirlm_vision::context *)ctx->vision_ctx;
    if (ctx->vision_load_attempted) return nullptr;
    ctx->vision_load_attempted = 1;
    if (ctx->model_path_for_audio.empty()) return nullptr;
    auto * v = new bidirlm_vision::context();
    if (!bidirlm_vision::load(*v, ctx->model_path_for_audio.c_str(),
                               /*shared_backend=*/ctx->backend,
                               ctx->n_threads, /*verbosity=*/1)) {
        delete v;
        return nullptr;
    }
    ctx->vision_ctx = v;
    return v;
}

extern "C" int crispembed_has_vision(const crispembed_context * ctx) {
    if (!ctx) return 0;
    if (ctx->vision_ctx) return 1;
    if (ctx->vision_load_attempted) return 0;
    return 1;  // unknown — caller should attempt encode and check return.
}

namespace {

// Run the vision tower and stage results into ctx->last_vision_out.
// Layout: [image_embeds (n_merged*dim), deepstack_0, deepstack_1, ...].
bool vision_run_and_stage(crispembed_context * ctx,
                          const float * pixel_patches, int n_patches,
                          const int32_t * grid_thw, int n_images,
                          bool include_deepstack) {
    auto * v = vision_lazy_open(ctx);
    if (!v) return false;
    bidirlm_vision::encode_result r;
    if (!bidirlm_vision::encode(*v, pixel_patches, n_patches,
                                 grid_thw, n_images, r, include_deepstack)) {
        return false;
    }
    const size_t per_slab = (size_t)r.n_merged * r.output_dim;
    const size_t total = per_slab * (1 + r.n_deepstack);
    ctx->last_vision_out.resize(total);
    std::memcpy(ctx->last_vision_out.data(), r.image_embeds, per_slab * sizeof(float));
    if (r.n_deepstack > 0 && r.deepstack) {
        std::memcpy(ctx->last_vision_out.data() + per_slab,
                    r.deepstack,
                    (size_t)r.n_deepstack * per_slab * sizeof(float));
    }
    ctx->last_vision_dim = r.output_dim;
    ctx->last_vision_n_merged = r.n_merged;
    ctx->last_vision_n_deepstack = r.n_deepstack;
    bidirlm_vision::encode_result_free(r);
    return true;
}

}  // namespace

extern "C" const float * crispembed_encode_image(crispembed_context * ctx,
                                                  const float * pixel_patches,
                                                  int n_patches,
                                                  const int32_t * grid_thw,
                                                  int n_images,
                                                  int * out_dim) {
    if (!vision_run_and_stage(ctx, pixel_patches, n_patches, grid_thw, n_images,
                              /*include_deepstack=*/false)) {
        if (out_dim) *out_dim = 0;
        return nullptr;
    }
    const int dim = ctx->last_vision_dim;
    const int n_merged = ctx->last_vision_n_merged;

    // Mean-pool image_embeds over the n_merged tokens.
    std::vector<float> pooled(dim, 0.0f);
    const float * src = ctx->last_vision_out.data();
    for (int t = 0; t < n_merged; t++) {
        const float * row = src + (size_t)t * dim;
        for (int i = 0; i < dim; i++) pooled[i] += row[i];
    }
    if (n_merged > 0) {
        const float inv = 1.0f / (float)n_merged;
        for (int i = 0; i < dim; i++) pooled[i] *= inv;
    }
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += pooled[i] * pooled[i];
    const float norm = std::sqrt(std::max(norm_sq, 1e-12f));
    for (int i = 0; i < dim; i++) pooled[i] /= norm;

    // Stage the pooled vector at the front of last_vision_out so the returned
    // pointer remains valid until the next call. We reuse the buffer by
    // resizing it to dim and copying — the raw layout is gone after this.
    ctx->last_vision_out.assign(pooled.begin(), pooled.end());
    if (out_dim) *out_dim = dim;
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_encode_image_raw(crispembed_context * ctx,
                                                      const float * pixel_patches,
                                                      int n_patches,
                                                      const int32_t * grid_thw,
                                                      int n_images,
                                                      int * out_n_merged,
                                                      int * out_dim,
                                                      int * out_n_deepstack) {
    if (!vision_run_and_stage(ctx, pixel_patches, n_patches, grid_thw, n_images,
                              /*include_deepstack=*/true)) {
        if (out_n_merged) *out_n_merged = 0;
        if (out_dim) *out_dim = 0;
        if (out_n_deepstack) *out_n_deepstack = 0;
        return nullptr;
    }
    if (out_n_merged) *out_n_merged = ctx->last_vision_n_merged;
    if (out_dim) *out_dim = ctx->last_vision_dim;
    if (out_n_deepstack) *out_n_deepstack = ctx->last_vision_n_deepstack;
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_encode_text_with_image(
        crispembed_context * ctx,
        const char * text,
        const float * pixel_patches, int n_patches,
        const int32_t * grid_thw, int n_images,
        int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !text || !pixel_patches || !grid_thw || n_images <= 0) return nullptr;
    if (!ctx->is_decoder || !ctx->dec) {
        fprintf(stderr, "crispembed_encode_text_with_image: model is not a "
                        "multimodal decoder.\n");
        return nullptr;
    }
    if (ctx->dec->image_token_id < 0) {
        fprintf(stderr, "crispembed_encode_text_with_image: model GGUF has no "
                        "decoder.image_token_id — re-export with vision metadata.\n");
        return nullptr;
    }

    // 1. Run the vision tower into a local buffer (we can't reuse ctx->last_vision_out
    //    because the decoder will overwrite ctx->last_output via the same bookkeeping).
    if (!vision_run_and_stage(ctx, pixel_patches, n_patches, grid_thw, n_images,
                              /*include_deepstack=*/true)) {
        return nullptr;
    }
    const int v_dim = ctx->last_vision_dim;
    const int v_merged = ctx->last_vision_n_merged;
    const int v_nds = ctx->last_vision_n_deepstack;
    if (v_dim != ctx->dec->n_embd) {
        fprintf(stderr, "crispembed_encode_text_with_image: vision tower output "
                        "dim %d != decoder hidden_size %d — model mismatch.\n",
                        v_dim, ctx->dec->n_embd);
        return nullptr;
    }
    std::vector<float> vision_buf;
    vision_buf.swap(ctx->last_vision_out);
    const float * image_embeds = vision_buf.data();
    const float * deepstack    = (v_nds > 0)
        ? vision_buf.data() + (size_t)v_merged * v_dim
        : nullptr;

    // 2. Tokenize.
    std::string prefixed;
    const char * enc_text = text;
    if (!ctx->prefix.empty()) {
        prefixed = ctx->prefix + text;
        enc_text = prefixed.c_str();
    }
    embed_tokens tokens;
    if (ctx->use_bpe)              tokens = ctx->bpe_tokenizer.encode(enc_text);
    else if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(enc_text);
    else                            tokens = ctx->wp_tokenizer.encode(enc_text);

    // Trim padding: only keep tokens where attn_mask == 1.
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

    // 3. Validate placeholder count.
    int placeholder_count = 0;
    for (int id : tokens.ids) {
        if (id == ctx->dec->image_token_id) placeholder_count++;
    }
    if (placeholder_count != v_merged) {
        fprintf(stderr,
                "crispembed_encode_text_with_image: text has %d image_token_id "
                "placeholders but vision tower produced %d merged tokens.\n",
                placeholder_count, v_merged);
        return nullptr;
    }

    // 4. Run decoder with image conditioning.
    dec_image_input dimg;
    dimg.image_embeds = image_embeds;
    dimg.deepstack    = deepstack;
    dimg.n_image_tokens = v_merged;
    dimg.n_deepstack  = v_nds;
    dimg.grid_thw    = grid_thw;
    dimg.n_images    = n_images;

    auto vec = decoder_encode_tokens(*ctx->dec, ctx->backend, tokens,
                                      ctx->n_threads, ctx->sched,
                                      &ctx->compute_meta, &dimg);
    if (vec.empty()) return nullptr;

    // 5. Matryoshka truncation + re-normalize.
    if (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < (int)vec.size()) {
        vec.resize(ctx->matryoshka_dim);
        float n = 0;
        for (int i = 0; i < ctx->matryoshka_dim; i++) n += vec[i] * vec[i];
        n = std::sqrt(std::max(n, 1e-12f));
        for (int i = 0; i < ctx->matryoshka_dim; i++) vec[i] /= n;
    }

    ctx->last_output = std::move(vec);
    if (out_dim) *out_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_free(crispembed_context * ctx) {
    if (!ctx) return;
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
    if (ctx->audio_ctx) {
        bidirlm_audio::close((bidirlm_audio::context *)ctx->audio_ctx);
        ctx->audio_ctx = nullptr;
    }
#endif
    if (ctx->vision_ctx) {
        auto * v = (bidirlm_vision::context *)ctx->vision_ctx;
        bidirlm_vision::free_(*v);
        delete v;
        ctx->vision_ctx = nullptr;
    }
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
