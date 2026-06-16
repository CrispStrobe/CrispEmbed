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
#include <cstdlib>
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
    // MoE (Mixture of Experts) FFN — present on MoE layers only
    ggml_tensor * moe_gate_w    = nullptr; // Router: [H, N_experts]
    ggml_tensor * expert_fc1_w  = nullptr; // Expert up: [H, inter, N_experts]
    ggml_tensor * expert_fc2_w  = nullptr; // Expert down: [inter, H, N_experts]
    ggml_tensor * moe_ffn_bias  = nullptr; // Output bias: [H]
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

        bool is_moe = L.moe_gate_w != nullptr;
        if (is_moe) {
            require(L.expert_fc1_w, "ffn.expert_fc1.weight");
            require(L.expert_fc2_w, "ffn.expert_fc2.weight");
        } else {
            require(L.fc2_w, "ffn.fc2.weight");
            require(L.ffn_up_gate_w || L.fc1_w, "ffn input weights");
        }

        if (pre_ln) {
            // ln1 optional for ModernBERT (no pre-attention norm, only pre-FFN ln2)
            require(L.ln1_w || L.ln2_w, "ln1.weight or ln2.weight");
        } else {
            require(L.ln1_w, "ln1.weight");
            require(L.ln2_w, "ln2.weight");
            if (!is_moe) require(L.fc1_w || L.ffn_up_gate_w, "ffn.fc1.weight");
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
    bool dump_layers = false; // dump per-layer intermediates (CRISPEMBED_DUMP_LAYERS=1)
    int  position_buckets = 0;  // DeBERTa log-bucket count (0 = linear positions)
    int matryoshka_dim = 0;  // 0 = use model default
    std::string prefix;  // prepended to text before tokenization (e.g. "query: ")
    // ColBERT self-describing metadata (read from GGUF, empty = not set)
    std::string colbert_query_prefix;
    std::string colbert_doc_prefix;
    std::string colbert_similarity_fn;
    int colbert_query_length = 0;
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
    // Per-token encoder embeddings (encode_tokens): raw final-hidden-state
    // output, L2-normalized, plus the token ids those vectors correspond
    // to. Valid until the next encode_tokens / encode_multivec / sparse /
    // dense encode call.
    std::vector<float>   last_token_embeddings;
    std::vector<int32_t> last_token_ids;
    int last_token_n   = 0;
    int last_token_dim = 0;
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
    // LoRA adapter name cache for list_lora API
    std::vector<std::string> lora_name_strings;
    std::vector<const char *> lora_name_ptrs;
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
    auto strv = [&](const char * key) -> std::string {
        const int64_t k = gguf_find_key(g, key);
        return k >= 0 ? std::string(gguf_get_val_str(g, k)) : std::string();
    };

    // Hyperparams — check CrispEmbed, Ollama bert.*, and Ollama xlmr.* keys.
    // CrispEmbed: bert.hidden_size, bert.num_hidden_layers, ...
    // Ollama:     {arch}.embedding_length, {arch}.block_count, ...
    //             where arch = "bert" or "xlmr"
    hp.n_vocab         = u32("bert.vocab_size", 30522);
    hp.n_max_tokens    = u32("bert.max_position_embeddings",
                         u32("bert.context_length",
                         u32("xlmr.context_length", 512)));
    hp.n_embd          = u32("bert.hidden_size",
                         u32("bert.embedding_length",
                         u32("xlmr.embedding_length", 384)));
    hp.n_head          = u32("bert.num_attention_heads",
                         u32("bert.attention.head_count",
                         u32("xlmr.attention.head_count", 12)));
    hp.n_layer         = u32("bert.num_hidden_layers",
                         u32("bert.block_count",
                         u32("xlmr.block_count", 6)));
    hp.n_intermediate  = u32("bert.intermediate_size",
                         u32("bert.feed_forward_length",
                         u32("xlmr.feed_forward_length", 1536)));
    hp.n_output        = u32("bert.output_dim", hp.n_embd);
    hp.layer_norm_eps  = f32("bert.layer_norm_eps",
                         f32("bert.attention.layer_norm_epsilon",
                         f32("xlmr.attention.layer_norm_epsilon", 1e-12f)));

    // Pooling method: 0=mean (default), 1=cls, 2=last-token
    // CrispEmbed format: bert.pooling_method (0=mean, 1=cls, 2=last)
    // Ollama format:     bert.pooling_type   (0=none, 1=mean, 2=cls, 3=last)
    {
        int pm = u32("bert.pooling_method", -1);
        if (pm < 0) {
            // Try Ollama format and convert: Ollama{1=mean,2=cls,3=last} → CE{0,1,2}
            int pt = u32("bert.pooling_type", -1);
            // Also check arch-prefixed key (xlmr.pooling_type, bert.pooling_type)
            if (pt < 0) pt = u32("xlmr.pooling_type", -1);
            if (pt > 0) pm = pt - 1;  // Ollama 1→0(mean), 2→1(cls), 3→2(last)
            else pm = 0;              // default mean
        }
        ctx->pool_method = pm;
    }
    // Position embedding offset: 0 for BERT, 2 for RoBERTa/XLM-R
    ctx->pos_offset    = u32("bert.position_offset", u32("xlmr.position_offset", 0));
    // ColBERT output dimension (BGE-M3 default 128) — read while g is valid
    m.colbert_dim      = u32("bert.colbert_dim", 128);
    // ColBERT self-describing metadata (from config_sentence_transformers.json)
    ctx->colbert_query_prefix  = strv("colbert.query_prefix");
    ctx->colbert_doc_prefix    = strv("colbert.document_prefix");
    ctx->colbert_similarity_fn = strv("colbert.similarity_fn_name");
    ctx->colbert_query_length  = u32("colbert.query_length", 0);
    // RoPE and pre-LN flags — MUST be read before gguf_free(g)
    ctx->rope_theta         = f32("bert.rope_theta", 10000.0f);
    ctx->rope_theta_global  = f32("bert.rope_theta_global", 0.0f);
    ctx->global_attn_every_n = u32("bert.global_attn_every_n", 0);
    ctx->pre_ln             = u32("bert.pre_ln", 0) != 0;
    ctx->position_buckets   = u32("bert.position_buckets", 0);
    hp.n_experts            = u32("bert.num_experts", 0);
    hp.n_experts_per_tok    = u32("bert.num_experts_per_tok", 0);

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
            // Detect casing: if vocab contains uppercase letters like "A", it's cased
            bool do_lower_case = true;
            for (const auto& t : vocab) {
                if (t.size() == 1 && t[0] >= 'A' && t[0] <= 'Z') {
                    do_lower_case = false;
                    break;
                }
            }
            ctx->wp_tokenizer.load(vocab, cls_id, sep_id, unk_id, pad_id, hp.n_max_tokens, do_lower_case);
            fprintf(stderr, "crispembed: using WordPiece tokenizer (%d tokens, %s)\n",
                    n, do_lower_case ? "uncased" : "cased");
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
    // Infer hparams from tensor shapes when metadata was missing (Ollama format).
    // token_embd.weight is [n_embd, n_vocab].
    {
        int64_t tensor_vocab = m.token_embd->ne[1];
        int64_t tensor_embd  = m.token_embd->ne[0];
        if (tensor_vocab > 0 && tensor_vocab != hp.n_vocab) {
            hp.n_vocab = (int)tensor_vocab;
        }
        if (tensor_embd > 0 && tensor_embd != hp.n_embd) {
            hp.n_embd = (int)tensor_embd;
            hp.n_output = hp.n_embd;
        }
        // Count actual encoder layers from loaded tensors
        int counted = 0;
        for (const auto& kv : ctx->wl.tensors) {
            // Match enc.N. or blk.N. prefix
            const auto& name = kv.first;
            int layer_id = -1;
            if (sscanf(name.c_str(), "enc.%d.", &layer_id) == 1 ||
                sscanf(name.c_str(), "blk.%d.", &layer_id) == 1) {
                if (layer_id + 1 > counted) counted = layer_id + 1;
            }
        }
        if (counted > 0 && counted != hp.n_layer) {
            hp.n_layer = counted;
        }
    }
    // NomicBERT/ModernBERT: RoPE-based encoders lack absolute position embeddings.
    // DeBERTa uses rel_embd for relative positions instead — do NOT apply RoPE in that case.
    if (!m.pos_embd && !m.rel_embd) {
        ctx->use_rope = true;
        fprintf(stderr, "crispembed: no position embeddings, using RoPE (theta=%.0f%s)\n",
                ctx->rope_theta, ctx->pre_ln ? ", pre-LN" : "");
    } else if (!m.pos_embd && m.rel_embd) {
        fprintf(stderr, "crispembed: DeBERTa disentangled relative-position attention\n");
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
        L.ffn_gate_w    = get_any({pfx + "ffn_gate.weight", blk + "ffn_gate.weight"});     // SwiGLU gate (NomicBERT)
        L.ffn_up_gate_w = get_any({pfx + "ffn_up_gate.weight", blk + "ffn_up_gate.weight"}); // Fused gate+up (ModernBERT/GTE v1.5)
        // MoE expert tensors (present only on MoE layers)
        L.moe_gate_w    = get(pfx + "ffn.moe_gate.weight");
        L.expert_fc1_w  = get(pfx + "ffn.expert_fc1.weight");
        L.expert_fc2_w  = get(pfx + "ffn.expert_fc2.weight");
        L.moe_ffn_bias  = get(pfx + "ffn.moe_bias");
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
    if (hp.n_experts > 0) {
        int moe_count = 0;
        for (int i = 0; i < hp.n_layer; i++)
            if (m.layers[i].moe_gate_w) moe_count++;
        fprintf(stderr, "crispembed: MoE encoder (%d experts, top-%d, %d/%d MoE layers)\n",
                hp.n_experts, hp.n_experts_per_tok, moe_count, hp.n_layer);
    }

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

    // MPNet relative position bias: precomputed [T, T, n_heads]
    ggml_tensor * rel_pos_bias = nullptr;
    if (m.rel_attn_bias) {
        rel_pos_bias = ggml_new_tensor_3d(gctx, GGML_TYPE_F16, T, T, n_heads);
        ggml_set_name(rel_pos_bias, "rel_pos_bias");
        ggml_set_input(rel_pos_bias);
    }

    // DeBERTa: pre-expanded position embeddings [H, T*T] (filled on CPU)
    ggml_tensor * rel_pos_expanded = nullptr;
    if (m.rel_embd) {
        rel_pos_expanded = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, (int64_t)T * T);
        ggml_set_name(rel_pos_expanded, "rel_pos_expanded");
        ggml_set_input(rel_pos_expanded);
    }

    // cur: [H, T*B] — all matmuls batch naturally
    ggml_tensor * cur = embd;
    if (m.embd_ln_w) {
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, m.embd_ln_w);
        if (m.embd_ln_b) cur = ggml_add(gctx, cur, m.embd_ln_b);
    }

    if (ctx->dump_layers) {
        ggml_set_name(cur, "emb_ln_out");
        ggml_set_output(cur);
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
            // DeBERTa-v2 disentangled attention: c2c + c2p + p2c
            ggml_tensor * Qs = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, Q), head_dim, T, n_heads));
            ggml_tensor * Ks = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, K), head_dim, T, n_heads));
            ggml_tensor * Vs = ggml_cont(gctx, ggml_reshape_3d(gctx, ggml_cont(gctx, V), head_dim, T, n_heads));

            // c2c: Q^T @ K → [T, T, nh]
            ggml_tensor * scores = ggml_mul_mat(gctx, Ks, Qs);

            // Expand position embeddings by bucket indices (shared tensor, zero-initialized)
            ggml_tensor * P = rel_pos_expanded;  // [H, T*T]

            // c2p: project pos through K weights (with bias), dot with Q
            ggml_tensor * Pk = ggml_mul_mat(gctx, L.k_w, P);  // [H, T*T]
            if (L.k_b) Pk = ggml_add(gctx, Pk, L.k_b);
            // Pk after reshape: [hd, nh, j, i] (j=ne[2] fast, i=ne[3] slow)
            Pk = ggml_reshape_4d(gctx, Pk, head_dim, n_heads, T, T);
            // c2p needs batch=(h,i) to match Qs batch=(h,i_q)
            // permute(0,2,1,3) → [hd, j, nh, i] → cont → batch = h+nh*i ✓
            ggml_tensor * Pk_b = ggml_cont(gctx, ggml_permute(gctx, Pk, 0, 2, 1, 3));
            Pk_b = ggml_reshape_3d(gctx, Pk_b, head_dim, T, (int64_t)n_heads * T);
            ggml_tensor * Qs_b = ggml_cont(gctx, ggml_permute(gctx, Qs, 0, 2, 1, 3));
            Qs_b = ggml_reshape_3d(gctx, Qs_b, head_dim, 1, (int64_t)n_heads * T);
            ggml_tensor * c2p = ggml_mul_mat(gctx, Pk_b, Qs_b);  // [j, 1, nh*i]
            // [T_j, 1, nh*T_i] → reshape [T_j, nh, T_i] → permute → [T_j, T_i, nh]
            c2p = ggml_reshape_3d(gctx, c2p, T, n_heads, T);
            c2p = ggml_cont(gctx, ggml_permute(gctx, c2p, 0, 2, 1, 3));

            // p2c: project pos through Q weights (with bias), dot with K
            // HF: p2c[q,k] = K[k] · Q_proj(rel_embd[bucket(q-k) + att_span])
            // This is the SAME position index as c2p (bucket(q-k)), NOT the mirror bucket(k-q).
            // Our pre-expanded P has P[:,i*T+j] = rel_embd[bucket(i-j)].
            // With the current batching (batch=t_key=i, row=j), indexing P gives bucket(i-j)=bucket(k-q).
            // To get bucket(q-k) instead, we transpose the TxT grid so P_p2c[:,i*T+j]=rel_embd[bucket(j-i)]:
            //   with batch=t_key=i, row=t_query=j: bucket(j-i) = bucket(t_query - t_key) = bucket(q-k) ✓
            // Transpose: reshape P→[H,T_j,T_i], permute→[H,T_i,T_j], reshape→[H,T*T]
            ggml_tensor * P_p2c = ggml_reshape_2d(gctx,
                ggml_cont(gctx, ggml_permute(gctx,
                    ggml_reshape_3d(gctx, P, H, T, T),  // [H, T_j, T_i]
                    0, 2, 1, 3)),                         // → [H, T_i, T_j]
                H, (int64_t)T * T);
            ggml_tensor * Pq = ggml_mul_mat(gctx, L.q_w, P_p2c);
            if (L.q_b) Pq = ggml_add(gctx, Pq, L.q_b);
            // Pq after reshape: [hd, nh, T_j, T_i]
            // with batch=(h, t_key=T_i), row=t_query=T_j:
            //   result[t_q, 0, h*T+t_key] = K[t_key] · Q_proj(rel_embd[bucket(t_q - t_key)]) ✓
            Pq = ggml_reshape_4d(gctx, Pq, head_dim, n_heads, T, T);
            // permute(0,2,1,3): [hd,nh,T_j,T_i] → [hd,T_j,nh,T_i]
            ggml_tensor * Pq_b = ggml_cont(gctx, ggml_permute(gctx, Pq, 0, 2, 1, 3));
            Pq_b = ggml_reshape_3d(gctx, Pq_b, head_dim, T, (int64_t)n_heads * T);
            ggml_tensor * Ks_b = ggml_cont(gctx, ggml_permute(gctx, Ks, 0, 2, 1, 3));
            Ks_b = ggml_reshape_3d(gctx, Ks_b, head_dim, 1, (int64_t)n_heads * T);
            ggml_tensor * p2c = ggml_mul_mat(gctx, Pq_b, Ks_b);  // [T_q, 1, nh*T_k]
            // [T_q, 1, nh*T_k] → reshape [T_q, nh, T_k] → permute → [T_k, T_q, nh]
            p2c = ggml_reshape_3d(gctx, p2c, T, n_heads, T);
            p2c = ggml_cont(gctx, ggml_permute(gctx, p2c, 1, 2, 0, 3));

            // Combine: (c2c + c2p + p2c) / sqrt(3 * head_dim)
            scores = ggml_add(gctx, scores, c2p);
            scores = ggml_add(gctx, scores, p2c);
            float scale = 1.0f / sqrtf(3.0f * (float)head_dim);
            scores = ggml_scale(gctx, scores, scale);

            scores = ggml_soft_max(gctx, scores);

            // Vt: [T_k, hd, nh] so mul_mat contracts over T_k, giving [hd, T_q, nh]
            ggml_tensor * Vt = ggml_cont(gctx, ggml_permute(gctx, Vs, 1, 0, 2, 3));
            attn = ggml_mul_mat(gctx, Vt, scores);
            // attn: [hd, T_q, nh] → need [H, T] = [hd*nh, T]
            // Must permute [hd, T, nh] → [hd, nh, T] so that hd and nh are contiguous,
            // then reshape to [H, T]. Without this permute, reshape produces wrong values.
            attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));  // [hd, nh, T]
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

        if (ctx->dump_layers && il == 0) {
            ggml_set_name(attn, "attn_out_0");
            ggml_set_output(attn);
        }

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
        if (L.moe_gate_w) {
            // MoE FFN (Nomic v2): router → top-K → expert dispatch → weighted combine
            const int n_exp = hp.n_experts;
            const int K     = hp.n_experts_per_tok;

            // Router logits: gate_w [H, n_exp] @ cur [H, TB] → [n_exp, TB]
            ggml_tensor * logits = ggml_mul_mat(gctx, L.moe_gate_w, cur);

            // Softmax over experts (ne[0] = n_exp) per token
            ggml_tensor * probs = ggml_soft_max(gctx, logits);

            // Top-K expert selection: [K, TB] I32
            ggml_tensor * ids = ggml_top_k(gctx, probs, K);

            // Gather top-K weights from softmax probs via get_rows
            // probs_3d [1, n_exp, TB]: get_rows selects K from n_exp per token
            ggml_tensor * probs_3d  = ggml_reshape_3d(gctx, probs, 1, n_exp, TB);
            ggml_tensor * top_w     = ggml_get_rows(gctx, probs_3d, ids);  // [1, K, TB]
            top_w = ggml_reshape_2d(gctx, top_w, K, TB);                   // [K, TB]

            // Expand input for K expert slots: [H, TB] → [H, K, TB]
            ggml_tensor * cur_3d  = ggml_reshape_3d(gctx, cur, H, 1, TB);
            ggml_tensor * rep_tgt = ggml_new_tensor_3d(gctx, cur->type, H, K, TB);
            ggml_tensor * cur_exp = ggml_repeat(gctx, cur_3d, rep_tgt);

            // Expert up projection: expert_fc1 [H, inter, n_exp] × [H, K, TB] → [inter, K, TB]
            ggml_tensor * up = ggml_mul_mat_id(gctx, L.expert_fc1_w, cur_exp, ids);

            // Activation: exact erf-GELU (NomicBERT v2 uses nn.GELU(approximate='none'))
            up = ggml_gelu_erf(gctx, up);

            // Expert down projection: expert_fc2 [inter, H, n_exp] × [inter, K, TB] → [H, K, TB]
            ggml_tensor * down = ggml_mul_mat_id(gctx, L.expert_fc2_w, up, ids);

            // Weighted combination: sum over K experts per token
            // down [H, K, TB] → permute to [K, H, TB], mul by weights [K, 1, TB], matmul sums K
            ggml_tensor * down_p = ggml_cont(gctx, ggml_permute(gctx, down, 1, 0, 2, 3));  // [K, H, TB]
            ggml_tensor * w_col  = ggml_reshape_3d(gctx, top_w, K, 1, TB);                  // [K, 1, TB]
            ffn = ggml_mul_mat(gctx, w_col, down_p);  // [1, H, TB]
            ffn = ggml_reshape_2d(gctx, ffn, H, TB);  // [H, TB]

            // MoE output bias
            if (L.moe_ffn_bias) ffn = ggml_add(gctx, ffn, L.moe_ffn_bias);

        } else if (L.ffn_up_gate_w) {
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
            // Standard GELU FFN (BERT / NomicBERT dense layers)
            ffn = ggml_mul_mat(gctx, L.fc1_w, cur);
            if (L.fc1_b) ffn = ggml_add(gctx, ffn, L.fc1_b);
            // HuggingFace/PyTorch GELU is erf-exact; use it for all BERT models
            // to match reference outputs (tanh-approx causes argmax flips in
            // token classification on borderline tokens).
            ffn = ggml_gelu_erf(gctx, ffn);
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

        // Per-layer dump for diff harness (activated by env var)
        if (ctx->dump_layers) {
            char lname[32];
            snprintf(lname, sizeof(lname), "layer_%d", il);
            ggml_set_name(cur, lname);
            ggml_set_output(cur);
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
    // CRISPEMBED_DEBUG_TOKENS=1 prints the final token-id sequence to stderr.
    // Used by tests/parity_layers_bert.py to diff against an HF tokenizer
    // without exposing a tokenize-only public API.
    if (const char * v = std::getenv("CRISPEMBED_DEBUG_TOKENS");
        v && v[0] && std::strcmp(v, "0") != 0) {
        fprintf(stderr, "crispembed: token_ids (n=%d):", T);
        for (int i = 0; i < T; i++) fprintf(stderr, " %d", tok_data[i]);
        fprintf(stderr, "\n");
    }
    ggml_backend_tensor_set(tok_ids, tok_data.data(), 0, T * sizeof(int32_t));

    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = t + ctx->pos_offset;
    // pos_ids is only connected to the graph when absolute pos_embd or RoPE is used.
    // DeBERTa models use rel_embd instead and don't wire pos_ids into the graph.
    ggml_tensor * pos_ids = ggml_graph_get_tensor(gf, "pos_ids");
    if (!pos_ids && (ctx->model.pos_embd || ctx->use_rope)) {
        fprintf(stderr, "crispembed: missing graph tensor 'pos_ids'\n");
        return {};
    }
    if (pos_ids) {
        debug_encode_stage("encode_tokens:set-pos", T, 1, 0);
        ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T * sizeof(int32_t));
    }

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

    // DeBERTa: expand position embeddings on CPU using bucket indices
    if (ctx->model.rel_embd) {
        ggml_tensor * rpe_t = ggml_graph_get_tensor(gf, "rel_pos_expanded");
        if (rpe_t) {
            debug_encode_stage("encode_tokens:set-rel-pos", T, 1, 0);
            int max_pos = (int)ctx->model.rel_embd->ne[1];
            int H_emb = (int)ctx->model.rel_embd->ne[0];
            int pos_buckets = ctx->position_buckets;

            // Read rel_embd data from backend
            std::vector<float> embd_data((size_t)H_emb * max_pos);
            ggml_backend_tensor_get(ctx->model.rel_embd, embd_data.data(), 0,
                                    embd_data.size() * sizeof(float));

            // Apply encoder LayerNorm to relative embeddings before expansion.
            // HF DeBERTa-v2: encoder.get_rel_embedding() does
            //   rel_embd = self.LayerNorm(self.rel_embeddings.weight)
            // when norm_rel_ebd == "layer_norm" (the default for DeBERTa-v2).
            // encoder_ln_w/b correspond to encoder.LayerNorm in HF.
            if (ctx->model.encoder_ln_w && ctx->model.encoder_ln_b) {
                std::vector<float> ln_w(H_emb), ln_b(H_emb);
                ggml_backend_tensor_get(ctx->model.encoder_ln_w, ln_w.data(), 0, H_emb * sizeof(float));
                ggml_backend_tensor_get(ctx->model.encoder_ln_b, ln_b.data(), 0, H_emb * sizeof(float));
                const float ln_eps = ctx->model.hparams.layer_norm_eps;
                for (int p = 0; p < max_pos; p++) {
                    float * row = &embd_data[(size_t)p * H_emb];
                    // Compute mean and variance
                    double sum = 0.0, sum2 = 0.0;
                    for (int d = 0; d < H_emb; d++) { sum += row[d]; sum2 += (double)row[d] * row[d]; }
                    float mean = (float)(sum / H_emb);
                    float var  = (float)(sum2 / H_emb) - mean * mean;
                    float inv_std = 1.0f / std::sqrt(var + ln_eps);
                    for (int d = 0; d < H_emb; d++) {
                        row[d] = (row[d] - mean) * inv_std * ln_w[d] + ln_b[d];
                    }
                }
            }

            // Expand: for each (i,j) pair, look up the position embedding
            std::vector<float> expanded((size_t)H_emb * T * T);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    int bucket;
                    if (pos_buckets > 0) {
                        // Log-bucket encoding matching HF make_log_bucket_position
                        int rel = i - j;
                        int sign_val = (rel > 0) ? 1 : ((rel < 0) ? -1 : 0);
                        int abs_rel = std::abs(rel);
                        int mid = pos_buckets / 2;

                        // HF: abs_pos = (|rel| < mid) ? mid-1 : |rel|
                        int abs_pos = (rel < mid && rel > -mid) ? (mid - 1) : abs_rel;

                        int signed_bucket;
                        if (abs_pos <= mid) {
                            // Inner region: use signed relative position directly
                            signed_bucket = rel;
                        } else {
                            // Outer region: log-scaled bucket
                            double log_ratio = std::log((double)abs_pos / mid)
                                             / std::log((double)(max_pos - 1) / mid);
                            int log_pos = (int)std::ceil(log_ratio * (mid - 1)) + mid;
                            signed_bucket = log_pos * sign_val;
                        }
                        // gather_index = signed_bucket + att_span (att_span = pos_buckets)
                        bucket = signed_bucket + pos_buckets;
                    } else {
                        bucket = i - j + max_pos / 2;
                    }
                    if (bucket < 0) bucket = 0;
                    if (bucket >= max_pos) bucket = max_pos - 1;
                    // Copy embedding row: embd_data[d + bucket*H_emb] → expanded[d + (i*T+j)*H_emb]
                    memcpy(&expanded[(size_t)(i * T + j) * H_emb],
                           &embd_data[(size_t)bucket * H_emb],
                           H_emb * sizeof(float));
                }
            }
            ggml_backend_tensor_set(rpe_t, expanded.data(), 0, expanded.size() * sizeof(float));
        }
    }

    // Compute (scheduler dispatches to GPU or CPU)
    debug_encode_stage("encode_tokens:compute", T, 1, 0);
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: encoder compute failed\n");
        return {};
    }

    // Dump per-layer intermediates for diff harness
    if (ctx->dump_layers) {
        auto dump_tensor = [&](const char * name) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, name);
            if (!t) return;
            int64_t n = ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            fprintf(stderr, "DUMP %s shape=[%lld,%lld] data=", name, (long long)t->ne[0], (long long)t->ne[1]);
            int show = n < 6 ? (int)n : 6;
            for (int i = 0; i < show; i++) fprintf(stderr, " %.6f", buf[i]);
            fprintf(stderr, " ...\n");
        };
        dump_tensor("emb_ln_out");
        dump_tensor("attn_out_0");
        for (int il = 0; il < hp.n_layer; il++) {
            char lname[32];
            snprintf(lname, sizeof(lname), "layer_%d", il);
            dump_tensor(lname);
        }
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
    // pos_ids is only wired into the graph when pos_embd or RoPE is active.
    // DeBERTa models use rel_embd instead, so pos_ids won't be in the graph.
    ggml_tensor * pos_ids = ggml_graph_get_tensor(gf, "pos_ids");
    if (!pos_ids && (ctx->model.pos_embd || ctx->use_rope)) {
        fprintf(stderr, "crispembed: missing graph tensor 'pos_ids'\n");
        return {};
    }
    if (pos_ids) {
        debug_encode_stage("run_encoder_raw:set-pos", T, 1, mode);
        ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T * sizeof(int32_t));
    }
    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(tokens.type_ids.begin(), tokens.type_ids.end());
        ggml_tensor * type_ids = graph_tensor_or_log(gf, "type_ids");
        if (!type_ids) return {};
        debug_encode_stage("run_encoder_raw:set-type", T, 1, mode);
        ggml_backend_tensor_set(type_ids, type_data.data(), 0, T * sizeof(int32_t));
    }

    // DeBERTa: expand position embeddings on CPU using bucket indices
    if (ctx->model.rel_embd) {
        ggml_tensor * rpe_t = ggml_graph_get_tensor(gf, "rel_pos_expanded");
        if (rpe_t) {
            int max_pos = (int)ctx->model.rel_embd->ne[1];
            int H_emb = (int)ctx->model.rel_embd->ne[0];
            int pos_buckets = ctx->position_buckets;

            std::vector<float> embd_data((size_t)H_emb * max_pos);
            ggml_backend_tensor_get(ctx->model.rel_embd, embd_data.data(), 0,
                                    embd_data.size() * sizeof(float));

            // Apply encoder LayerNorm to relative embeddings before expansion.
            // HF DeBERTa-v2: encoder.get_rel_embedding() does
            //   rel_embd = self.LayerNorm(self.rel_embeddings.weight)
            // when norm_rel_ebd == "layer_norm" (the default for DeBERTa-v2).
            if (ctx->model.encoder_ln_w && ctx->model.encoder_ln_b) {
                std::vector<float> ln_w(H_emb), ln_b(H_emb);
                ggml_backend_tensor_get(ctx->model.encoder_ln_w, ln_w.data(), 0, H_emb * sizeof(float));
                ggml_backend_tensor_get(ctx->model.encoder_ln_b, ln_b.data(), 0, H_emb * sizeof(float));
                const float ln_eps = ctx->model.hparams.layer_norm_eps;
                for (int p = 0; p < max_pos; p++) {
                    float * row = &embd_data[(size_t)p * H_emb];
                    double sum = 0.0, sum2 = 0.0;
                    for (int d = 0; d < H_emb; d++) { sum += row[d]; sum2 += (double)row[d] * row[d]; }
                    float mean = (float)(sum / H_emb);
                    float var  = (float)(sum2 / H_emb) - mean * mean;
                    float inv_std = 1.0f / std::sqrt(var + ln_eps);
                    for (int d = 0; d < H_emb; d++) {
                        row[d] = (row[d] - mean) * inv_std * ln_w[d] + ln_b[d];
                    }
                }
            }

            std::vector<float> expanded((size_t)H_emb * T * T);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    int bucket;
                    if (pos_buckets > 0) {
                        int rel = i - j;
                        int sign_val = (rel > 0) ? 1 : ((rel < 0) ? -1 : 0);
                        int abs_rel = std::abs(rel);
                        int mid = pos_buckets / 2;
                        int abs_pos = (rel < mid && rel > -mid) ? (mid - 1) : abs_rel;
                        int signed_bucket;
                        if (abs_pos <= mid) {
                            signed_bucket = rel;
                        } else {
                            double log_ratio = std::log((double)abs_pos / mid)
                                             / std::log((double)(max_pos - 1) / mid);
                            int log_pos = (int)std::ceil(log_ratio * (mid - 1)) + mid;
                            signed_bucket = log_pos * sign_val;
                        }
                        bucket = signed_bucket + pos_buckets;
                    } else {
                        bucket = i - j + max_pos / 2;
                    }
                    if (bucket < 0) bucket = 0;
                    if (bucket >= max_pos) bucket = max_pos - 1;
                    memcpy(&expanded[(size_t)(i * T + j) * H_emb],
                           &embd_data[(size_t)bucket * H_emb],
                           H_emb * sizeof(float));
                }
            }
            ggml_backend_tensor_set(rpe_t, expanded.data(), 0, expanded.size() * sizeof(float));
        }
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
    ctx->dump_layers = (std::getenv("CRISPEMBED_DUMP_LAYERS") != nullptr);

    // Detect model type from GGUF metadata.
    // Decoder models have either decoder.hidden_size (CrispEmbed-native) or
    // general.architecture in {qwen3, gemma3, llama, ...} (Ollama-format).
    // Encoder models (BERT/XLM-R) have bert.* keys and enc.N.* tensor names.
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    bool is_dec = false;
    if (g) {
        is_dec = gguf_find_key(g, "decoder.hidden_size") >= 0;
        if (!is_dec) {
            int64_t ki = gguf_find_key(g, "general.architecture");
            if (ki >= 0) {
                std::string arch = gguf_get_val_str(g, ki);
                is_dec = (arch == "qwen3" || arch == "gemma3" || arch == "llama"
                       || arch == "qwen2" || arch == "mistral" || arch == "phi3");
            }
        }
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
                // Respect add_bos_token=false: if the flag is explicitly false,
                // don't prepend BOS even when bos_token_id is set.
                {
                    const int64_t ki_add_bos = gguf_find_key(g2, "tokenizer.ggml.add_bos_token");
                    if (ki_add_bos >= 0) {
                        auto type = gguf_get_kv_type(g2, ki_add_bos);
                        bool add_bos = true;
                        if (type == GGUF_TYPE_BOOL)        add_bos = gguf_get_val_bool(g2, ki_add_bos);
                        else if (type == GGUF_TYPE_UINT32) add_bos = gguf_get_val_u32(g2, ki_add_bos) != 0;
                        else if (type == GGUF_TYPE_INT32)  add_bos = gguf_get_val_i32(g2, ki_add_bos) != 0;
                        if (!add_bos) bos_id = -1;
                    }
                }
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

extern "C" const char * crispembed_query_prefix(const char * model_name) {
    return crispembed_mgr::get_query_prefix(model_name);
}
extern "C" const char * crispembed_passage_prefix(const char * model_name) {
    return crispembed_mgr::get_passage_prefix(model_name);
}

extern "C" const char * crispembed_ctx_query_prefix(const crispembed_context * ctx) {
    if (!ctx) return nullptr;
    return ctx->colbert_query_prefix.empty() ? nullptr : ctx->colbert_query_prefix.c_str();
}
extern "C" const char * crispembed_ctx_passage_prefix(const crispembed_context * ctx) {
    if (!ctx) return nullptr;
    return ctx->colbert_doc_prefix.empty() ? nullptr : ctx->colbert_doc_prefix.c_str();
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

extern "C" int crispembed_set_lora(crispembed_context * ctx, const char * adapter_name) {
    if (!ctx || !ctx->is_decoder || !ctx->dec) return 0;
    if (ctx->dec->lora_adapters.empty()) return 0;
    std::string name = adapter_name ? adapter_name : "";
    return decoder_set_lora(*ctx->dec, ctx->backend, name) ? 1 : 0;
}

extern "C" const char * crispembed_get_lora(const crispembed_context * ctx) {
    if (!ctx || !ctx->is_decoder || !ctx->dec) return "";
    return ctx->dec->active_lora.c_str();
}

extern "C" int crispembed_list_lora(const crispembed_context * ctx,
                                     const char *** out_names, int * out_count) {
    if (!ctx || !ctx->is_decoder || !ctx->dec || ctx->dec->lora_adapters.empty()) {
        if (out_count) *out_count = 0;
        if (out_names) *out_names = nullptr;
        return 0;
    }
    // Build name pointer cache (const_cast is safe — ctx owns the strings)
    auto * mctx = const_cast<crispembed_context *>(ctx);
    mctx->lora_name_strings.clear();
    mctx->lora_name_ptrs.clear();
    for (const auto & a : ctx->dec->lora_adapters) {
        mctx->lora_name_strings.push_back(a.name);
    }
    for (const auto & s : mctx->lora_name_strings) {
        mctx->lora_name_ptrs.push_back(s.c_str());
    }
    mctx->lora_name_ptrs.push_back(nullptr);  // null-terminated
    if (out_names) *out_names = mctx->lora_name_ptrs.data();
    if (out_count) *out_count = (int)ctx->dec->lora_adapters.size();
    return 1;
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
        // Decoder: batched graph (falls back to sequential for B=1 or multimodal)
        batch_results = decoder_encode_tokens_batch(*ctx->dec, ctx->backend, all_tokens,
                                                     ctx->n_threads, ctx->sched, &ctx->compute_meta);
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
// Per-token contextual embeddings (any encoder model)
// ---------------------------------------------------------------------------
//
// Unlike encode_multivec, which is gated on the ColBERT projection head,
// encode_tokens returns the encoder's raw final hidden states for every
// non-padded token. This is what SimAlign-style word aligners want:
// pairwise cosine similarity over contextual token embeddings.

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
extern "C" const float * crispembed_encode_tokens(crispembed_context * ctx,
                                                    const char         * text,
                                                    int                * out_n_tokens,
                                                    int                * out_dim) {
    if (!ctx || !text || ctx->is_decoder) return nullptr;

    // Apply the configured prefix (e.g. "query: ") for consistency with
    // the dense encode path.
    std::string enc_text = ctx->prefix.empty() ? std::string(text) : ctx->prefix + text;

    embed_tokens tokens;
    if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(enc_text);
    else                        tokens = ctx->wp_tokenizer.encode(enc_text);

    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { T_real = i + 1; break; }
    }
    if (T_real == 0) return nullptr;
    tokens.ids.resize(T_real);
    tokens.type_ids.resize(T_real);
    tokens.attn_mask.resize(T_real);

    // mode=0: dense encoder graph. Returns [n_embd, T_real] raw output.
    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
    if (raw.empty() || raw_T == 0) return nullptr;

    const int dim = ctx->model.hparams.n_embd;
    ctx->last_token_embeddings.resize((size_t)dim * (size_t)raw_T);
    for (int t = 0; t < raw_T; t++) {
        const float * vec = raw.data() + (size_t)t * (size_t)dim;
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) norm += vec[d] * vec[d];
        norm = std::sqrt(std::max(norm, 1e-12f));
        float * out = ctx->last_token_embeddings.data() + (size_t)t * (size_t)dim;
        for (int d = 0; d < dim; d++) out[d] = vec[d] / norm;
    }

    ctx->last_token_ids.assign(tokens.ids.begin(), tokens.ids.begin() + raw_T);
    ctx->last_token_n   = raw_T;
    ctx->last_token_dim = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim)      *out_dim      = dim;
    return ctx->last_token_embeddings.data();
}

extern "C" const float * crispembed_encode_tokens_raw(crispembed_context * ctx,
                                                        const char         * text,
                                                        int                * out_n_tokens,
                                                        int                * out_dim) {
    if (!ctx || !text || ctx->is_decoder) return nullptr;

    std::string enc_text = ctx->prefix.empty() ? std::string(text) : ctx->prefix + text;

    embed_tokens tokens;
    if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(enc_text);
    else                        tokens = ctx->wp_tokenizer.encode(enc_text);

    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { T_real = i + 1; break; }
    }
    if (T_real == 0) return nullptr;
    tokens.ids.resize(T_real);
    tokens.type_ids.resize(T_real);
    tokens.attn_mask.resize(T_real);

    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
    if (raw.empty() || raw_T == 0) return nullptr;

    const int dim = ctx->model.hparams.n_embd;
    // Store raw (unnormalized) hidden states
    ctx->last_token_embeddings.resize((size_t)dim * (size_t)raw_T);
    std::memcpy(ctx->last_token_embeddings.data(), raw.data(),
                (size_t)dim * (size_t)raw_T * sizeof(float));

    ctx->last_token_ids.assign(tokens.ids.begin(), tokens.ids.begin() + raw_T);
    ctx->last_token_n   = raw_T;
    ctx->last_token_dim = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim)      *out_dim      = dim;
    return ctx->last_token_embeddings.data();
}

extern "C" const int32_t * crispembed_last_token_ids(const crispembed_context * ctx) {
    if (!ctx || ctx->last_token_n == 0) return nullptr;
    return ctx->last_token_ids.data();
}

extern "C" const char * crispembed_token_str(const crispembed_context * ctx, int32_t id) {
    if (!ctx || ctx->is_decoder) return nullptr;
    const std::string & s = ctx->use_sentencepiece
        ? ctx->sp_tokenizer.token_str((int)id)
        : ctx->wp_tokenizer.token_str((int)id);
    return s.c_str();
}

extern "C" int crispembed_tokenizer_kind(const crispembed_context * ctx) {
    // 0 = unknown, 1 = WordPiece (## continuation marker),
    // 2 = SentencePiece (▁ word-start marker), 3 = BPE.
    if (!ctx) return 0;
    if (ctx->use_bpe) return 3;
    if (ctx->use_sentencepiece) return 2;
    return 1;
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

    // Debug: dump CLS vector when CRISPEMBED_DEBUG_CLS=1
    if (const char * dcls = std::getenv("CRISPEMBED_DEBUG_CLS")) {
        if (dcls[0] && dcls[0] != '0') {
            fprintf(stderr, "CLS[0:8]:");
            for (int i = 0; i < std::min(8, H); i++) fprintf(stderr, " %.4f", cls_vec[i]);
            fprintf(stderr, "\n");
        }
    }

    // Apply ContextPooler if present (DeBERTa-v2 reranker):
    //   pooled = GELU(dense_w @ cls + dense_b)
    // The classifier then operates on pooled rather than raw CLS.
    std::vector<float> pooled_buf;
    if (ctx->model.pooler_w && ctx->model.pooler_b) {
        std::vector<float> pw(H * H), pb(H);
        ggml_backend_tensor_get(ctx->model.pooler_w, pw.data(), 0, H*H*sizeof(float));
        ggml_backend_tensor_get(ctx->model.pooler_b, pb.data(), 0, H*sizeof(float));
        pooled_buf.resize(H);
        for (int i = 0; i < H; i++) {
            float acc = pb[i];
            for (int j = 0; j < H; j++) acc += cls_vec[j] * pw[i * H + j];
            // GELU activation (standard pooler_hidden_act for DeBERTa)
            // Using the same polynomial approximation as ggml_gelu
            const float x = acc;
            pooled_buf[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
        cls_vec = pooled_buf.data();
    }

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

namespace {

// Shared tail of the image-conditioned encoders: run the vision tower, validate
// dims/placeholder count, build dec_image_input, run the decoder graph,
// apply matryoshka truncation, and stage the L2-normalized output into
// ctx->last_output. `tokens` is consumed by-move; on success the returned
// pointer is owned by ctx and valid until the next call.
const float * encode_image_conditioned(
        crispembed_context * ctx,
        embed_tokens && tokens,
        const float * pixel_patches, int n_patches,
        const int32_t * grid_thw, int n_images,
        int * out_dim,
        const char * caller) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !pixel_patches || !grid_thw || n_images <= 0) return nullptr;
    if (!ctx->is_decoder || !ctx->dec) {
        fprintf(stderr, "%s: model is not a multimodal decoder.\n", caller);
        return nullptr;
    }
    if (ctx->dec->image_token_id < 0) {
        fprintf(stderr, "%s: model GGUF has no decoder.image_token_id — "
                        "re-export with vision metadata.\n", caller);
        return nullptr;
    }

    // 1. Run vision tower into a local buffer (the decoder will reuse
    //    ctx->last_output, so we can't keep both pointing at last_vision_out).
    if (!vision_run_and_stage(ctx, pixel_patches, n_patches, grid_thw, n_images,
                              /*include_deepstack=*/true)) {
        return nullptr;
    }
    const int v_dim = ctx->last_vision_dim;
    const int v_merged = ctx->last_vision_n_merged;
    const int v_nds = ctx->last_vision_n_deepstack;
    if (v_dim != ctx->dec->n_embd) {
        fprintf(stderr, "%s: vision tower output dim %d != decoder "
                        "hidden_size %d — model mismatch.\n",
                        caller, v_dim, ctx->dec->n_embd);
        return nullptr;
    }
    std::vector<float> vision_buf;
    vision_buf.swap(ctx->last_vision_out);
    const float * image_embeds = vision_buf.data();
    const float * deepstack    = (v_nds > 0)
        ? vision_buf.data() + (size_t)v_merged * v_dim
        : nullptr;

    // 2. Validate placeholder count.
    int placeholder_count = 0;
    for (int id : tokens.ids) {
        if (id == ctx->dec->image_token_id) placeholder_count++;
    }
    if (placeholder_count != v_merged) {
        fprintf(stderr,
                "%s: input has %d image_token_id placeholders but vision "
                "tower produced %d merged tokens.\n",
                caller, placeholder_count, v_merged);
        return nullptr;
    }

    // 3. Run decoder with image conditioning.
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

    // 4. Matryoshka truncation + re-normalize.
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

}  // namespace

extern "C" const float * crispembed_encode_text_with_image(
        crispembed_context * ctx,
        const char * text,
        const float * pixel_patches, int n_patches,
        const int32_t * grid_thw, int n_images,
        int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !text) return nullptr;

    // Tokenize (with optional prefix).
    std::string prefixed;
    const char * enc_text = text;
    if (!ctx->prefix.empty()) {
        prefixed = ctx->prefix + text;
        enc_text = prefixed.c_str();
    }
    embed_tokens tokens;
    if (ctx->use_bpe)               tokens = ctx->bpe_tokenizer.encode(enc_text);
    else if (ctx->use_sentencepiece) tokens = ctx->sp_tokenizer.encode(enc_text);
    else                             tokens = ctx->wp_tokenizer.encode(enc_text);

    // Trim padding: only keep tokens where attn_mask == 1.
    int actual_len = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) { actual_len = i + 1; break; }
    }
    if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
        tokens.ids.resize(actual_len);
        tokens.type_ids.resize(actual_len);
        tokens.attn_mask.resize(actual_len);
    }

    return encode_image_conditioned(
        ctx, std::move(tokens),
        pixel_patches, n_patches, grid_thw, n_images,
        out_dim, "crispembed_encode_text_with_image");
}

extern "C" const float * crispembed_encode_with_image_ids(
        crispembed_context * ctx,
        const int32_t * token_ids, int n_tokens,
        const float * pixel_patches, int n_patches,
        const int32_t * grid_thw, int n_images,
        int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !token_ids || n_tokens <= 0) return nullptr;

    embed_tokens tokens;
    tokens.ids.assign(token_ids, token_ids + n_tokens);
    tokens.type_ids.assign((size_t)n_tokens, 0);
    tokens.attn_mask.assign((size_t)n_tokens, 1);

    return encode_image_conditioned(
        ctx, std::move(tokens),
        pixel_patches, n_patches, grid_thw, n_images,
        out_dim, "crispembed_encode_with_image_ids");
}

// ---------------------------------------------------------------------------
// In-process image preprocessor (file-based)
// ---------------------------------------------------------------------------
#include "image_preprocess.h"

extern "C" const float * crispembed_preprocess_image(
        crispembed_context * ctx,
        const char * image_path,
        int * out_n_patches, int * out_row_dim,
        int32_t out_grid_thw[3]) {
    if (out_n_patches) *out_n_patches = 0;
    if (out_row_dim) *out_row_dim = 0;
    if (!ctx || !image_path) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec) {
        // BidirLM-Omni: patch_size=16, merge_size=2 by default. Encoder vision
        // tower's spatial_merge_size lives on the dec_model side; trust it.
        if (ctx->dec->spatial_merge_size > 0) {
            cfg.merge_size = ctx->dec->spatial_merge_size;
        }
    }
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) {
        return nullptr;
    }
    // Stash into ctx->last_vision_out so the returned pointer remains valid
    // until the next preprocessor call (mirrors encode_image's contract).
    ctx->last_vision_out = std::move(r.patches);
    if (out_n_patches) *out_n_patches = r.n_patches;
    if (out_row_dim)   *out_row_dim   = r.row_dim;
    if (out_grid_thw) {
        out_grid_thw[0] = r.grid_thw[0];
        out_grid_thw[1] = r.grid_thw[1];
        out_grid_thw[2] = r.grid_thw[2];
    }
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_preprocess_image_rgb(
        crispembed_context * ctx,
        const uint8_t * rgb, int height, int width, int channels,
        int * out_n_patches, int * out_row_dim,
        int32_t out_grid_thw[3]) {
    if (out_n_patches) *out_n_patches = 0;
    if (out_row_dim) *out_row_dim = 0;
    if (!ctx || !rgb || height <= 0 || width <= 0
        || (channels != 3 && channels != 4)) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    image_preproc::result r;
    if (!image_preproc::preprocess_rgb(rgb, height, width, channels, cfg, r)) {
        return nullptr;
    }
    ctx->last_vision_out = std::move(r.patches);
    if (out_n_patches) *out_n_patches = r.n_patches;
    if (out_row_dim)   *out_row_dim   = r.row_dim;
    if (out_grid_thw) {
        out_grid_thw[0] = r.grid_thw[0];
        out_grid_thw[1] = r.grid_thw[1];
        out_grid_thw[2] = r.grid_thw[2];
    }
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_encode_image_file(
        crispembed_context * ctx,
        const char * image_path,
        int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !image_path) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) return nullptr;

    return crispembed_encode_image(ctx,
                                    r.patches.data(), r.n_patches,
                                    r.grid_thw, /*n_images=*/1,
                                    out_dim);
}

extern "C" const float * crispembed_encode_text_with_image_file(
        crispembed_context * ctx,
        const char * text,
        const char * image_path,
        int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !text || !image_path) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) return nullptr;

    return crispembed_encode_text_with_image(ctx, text,
                                              r.patches.data(), r.n_patches,
                                              r.grid_thw, /*n_images=*/1,
                                              out_dim);
}

// ---------------------------------------------------------------------------
// Standalone ViT image embedding C API (SigLIP, CLIP)
// ---------------------------------------------------------------------------

#include "vit_embed.h"

struct crispembed_vit_context {
    vit_embed::context * vit = nullptr;
    std::vector<float> last_output;
};

extern "C" crispembed_vit_context * crispembed_vit_init(
        const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new crispembed_vit_context();
    if (!vit_embed::load(&ctx->vit, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" int crispembed_vit_dim(const crispembed_vit_context * ctx) {
    return ctx ? vit_embed::dim(ctx->vit) : 0;
}

extern "C" const float * crispembed_vit_encode_file(
        crispembed_vit_context * ctx,
        const char * image_path,
        int * out_dim) {
    if (!ctx || !image_path || !out_dim) {
        if (out_dim) *out_dim = 0;
        return nullptr;
    }
    ctx->last_output = vit_embed::encode_file(ctx->vit, image_path);
    if (ctx->last_output.empty()) { *out_dim = 0; return nullptr; }
    *out_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_vit_free(crispembed_vit_context * ctx) {
    if (!ctx) return;
    if (ctx->vit) vit_embed::free(ctx->vit);
    delete ctx;
}

// ---------------------------------------------------------------------------
// CLIP text encoding C API
// ---------------------------------------------------------------------------

#include "clip_text_embed.h"

struct crispembed_clip_text_context {
    clip_text::context * ct = nullptr;
    std::vector<float> last_output;
};

extern "C" crispembed_clip_text_context * crispembed_clip_text_init(
        const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new crispembed_clip_text_context();
    if (!clip_text::load(&ctx->ct, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" int crispembed_clip_text_dim(const crispembed_clip_text_context * ctx) {
    return ctx && ctx->ct ? clip_text::dim(ctx->ct) : 0;
}

extern "C" const float * crispembed_clip_text_encode(
        crispembed_clip_text_context * ctx,
        const char * text,
        int * out_dim) {
    if (!ctx || !ctx->ct || !text || !out_dim) return nullptr;
    ctx->last_output = clip_text::encode(ctx->ct, text);
    if (ctx->last_output.empty()) { *out_dim = 0; return nullptr; }
    *out_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_clip_text_free(crispembed_clip_text_context * ctx) {
    if (!ctx) return;
    if (ctx->ct) clip_text::free(ctx->ct);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Face detection & recognition C API
// ---------------------------------------------------------------------------

#include "cnn_embed.h"

struct crispembed_face_context {
    cnn_embed::context * cnn = nullptr;
    // Scratch buffers for returning results (valid until next call)
    std::vector<crispembed_face_detection> det_buf;
    std::vector<crispembed_face_result> result_buf;
    std::vector<std::vector<float>> emb_buf;  // owns embedding data
    std::vector<float> single_emb;
};

extern "C" crispembed_face_context * crispembed_face_init(
        const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new crispembed_face_context();
    if (!cnn_embed::load(&ctx->cnn, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" int crispembed_face_dim(const crispembed_face_context * ctx) {
    return ctx ? cnn_embed::dim(ctx->cnn) : 0;
}

extern "C" const char * crispembed_face_type(const crispembed_face_context * ctx) {
    return ctx ? cnn_embed::model_type(ctx->cnn) : "";
}

extern "C" const crispembed_face_detection * crispembed_detect_faces(
        crispembed_face_context * ctx,
        const char * image_path,
        float conf_threshold,
        int det_size,
        int * out_n_faces) {
    if (!ctx || !image_path || !out_n_faces) { if (out_n_faces) *out_n_faces = 0; return nullptr; }

    auto dets = cnn_embed::detect_file(ctx->cnn, image_path, conf_threshold,
                                        det_size > 0 ? det_size : 640);
    ctx->det_buf.resize(dets.size());
    for (size_t i = 0; i < dets.size(); i++) {
        auto & d = ctx->det_buf[i];
        d.x = dets[i].x; d.y = dets[i].y;
        d.w = dets[i].w; d.h = dets[i].h;
        d.confidence = dets[i].confidence;
        memcpy(d.landmarks, dets[i].landmarks, sizeof(d.landmarks));
    }
    *out_n_faces = (int)dets.size();
    return ctx->det_buf.empty() ? nullptr : ctx->det_buf.data();
}

extern "C" const float * crispembed_encode_face(
        crispembed_face_context * ctx,
        const char * image_path,
        const float * landmarks_10,
        int * out_dim) {
    if (!ctx || !image_path || !landmarks_10 || !out_dim) {
        if (out_dim) *out_dim = 0;
        return nullptr;
    }

    ctx->single_emb = cnn_embed::encode_face_file(ctx->cnn, image_path, landmarks_10);

    if (ctx->single_emb.empty()) { *out_dim = 0; return nullptr; }
    *out_dim = (int)ctx->single_emb.size();
    return ctx->single_emb.data();
}

extern "C" const crispembed_face_result * crispembed_face_pipeline(
        crispembed_face_context * det_ctx,
        crispembed_face_context * rec_ctx,
        const char * image_path,
        float conf_threshold,
        int det_size,
        int * out_n_faces) {
    if (!det_ctx || !rec_ctx || !image_path || !out_n_faces) {
        if (out_n_faces) *out_n_faces = 0;
        return nullptr;
    }

    auto results = cnn_embed::face_pipeline(det_ctx->cnn, rec_ctx->cnn,
                                             image_path, conf_threshold,
                                             det_size > 0 ? det_size : 640);
    // Store results in det_ctx scratch buffers
    det_ctx->result_buf.resize(results.size());
    det_ctx->emb_buf.resize(results.size());
    for (size_t i = 0; i < results.size(); i++) {
        auto & r = det_ctx->result_buf[i];
        r.det.x = results[i].det.x; r.det.y = results[i].det.y;
        r.det.w = results[i].det.w; r.det.h = results[i].det.h;
        r.det.confidence = results[i].det.confidence;
        memcpy(r.det.landmarks, results[i].det.landmarks, sizeof(r.det.landmarks));
        det_ctx->emb_buf[i] = std::move(results[i].embedding);
        r.embedding = det_ctx->emb_buf[i].data();
        r.embedding_dim = (int)det_ctx->emb_buf[i].size();
    }
    *out_n_faces = (int)results.size();
    return det_ctx->result_buf.empty() ? nullptr : det_ctx->result_buf.data();
}

extern "C" void crispembed_face_free(crispembed_face_context * ctx) {
    if (!ctx) return;
    if (ctx->cnn) cnn_embed::free(ctx->cnn);
    delete ctx;
}

// ---------------------------------------------------------------------------
// ColBERT MaxSim scoring
// ---------------------------------------------------------------------------

extern "C" float crispembed_colbert_score(
    const float * query_vecs,  int n_query,
    const float * doc_vecs,    int n_doc,
    int dim
) {
    // MaxSim: score = sum_i(max_j(dot(Q[i], D[j])))
    // Q and D are already L2-normalized, so dot = cosine.
    float score = 0.0f;
    for (int qi = 0; qi < n_query; qi++) {
        const float * q = query_vecs + qi * dim;
        float max_sim = -1e30f;
        for (int di = 0; di < n_doc; di++) {
            const float * d = doc_vecs + di * dim;
            float dot = 0.0f;
            for (int k = 0; k < dim; k++) dot += q[k] * d[k];
            if (dot > max_sim) max_sim = dot;
        }
        score += max_sim;
    }
    return score;
}

extern "C" int crispembed_colbert_score_batch(
    const float * query_vecs,  int n_query,
    const float ** doc_vecs_list, const int * doc_n_tokens,
    int n_docs, int dim,
    float * out_scores
) {
    if (!query_vecs || !doc_vecs_list || !doc_n_tokens || !out_scores) return -1;

    #pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < n_docs; d++) {
        out_scores[d] = crispembed_colbert_score(
            query_vecs, n_query,
            doc_vecs_list[d], doc_n_tokens[d],
            dim);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Unified Math OCR — auto-detects model architecture from GGUF metadata.
// Supports: pix2tex_mfr (DeiT+TrOCR), hmer (DenseNet+GRU), bttr (DenseNet+Transformer)
// ---------------------------------------------------------------------------

#include "math_ocr.h"
#include "hmer_ocr.h"
#include "bttr_ocr.h"
#include "ppformulanet_ocr.h"
#include "ppformulanet_l_ocr.h"
#include "posformer_ocr.h"
#include "mixtex_ocr.h"
#include "qwen2vl_ocr.h"
#include "internvl2_ocr.h"
#include "parseq_ocr.h"
#include "glm_ocr.h"
#include "got_ocr.h"
#include "pix2struct.h"
#include "tesseract_lstm.h"
#include "granite_vision_ocr.h"
#include "core/gguf_loader.h"

enum math_ocr_type { MATH_OCR_PIX2TEX, MATH_OCR_HMER, MATH_OCR_BTTR, MATH_OCR_PPFORMULANET, MATH_OCR_PPFORMULANET_L, MATH_OCR_POSFORMER, MATH_OCR_MIXTEX, MATH_OCR_QWEN2VL, MATH_OCR_INTERNVL2, MATH_OCR_PARSEQ, MATH_OCR_GLM_OCR, MATH_OCR_GOT_OCR, MATH_OCR_TESSERACT_LSTM, MATH_OCR_GRANITE_VISION };

struct unified_math_ocr {
    math_ocr_type type;
    void * ctx;
};

static math_ocr_type detect_arch(const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return MATH_OCR_PIX2TEX;
    std::string arch = core_gguf::kv_str(g, "general.architecture", "pix2tex_mfr");
    core_gguf::free_metadata(g);
    if (arch == "hmer") return MATH_OCR_HMER;
    if (arch == "bttr") return MATH_OCR_BTTR;
    if (arch == "ppformulanet") return MATH_OCR_PPFORMULANET;
    if (arch == "ppformulanet_l") return MATH_OCR_PPFORMULANET_L;
    if (arch == "posformer") return MATH_OCR_POSFORMER;
    if (arch == "mixtex") return MATH_OCR_MIXTEX;
    if (arch == "qwen2vl" || arch == "qwen3vl") return MATH_OCR_QWEN2VL;
    if (arch == "internvl2") return MATH_OCR_INTERNVL2;
if (arch == "parseq") return MATH_OCR_PARSEQ;
if (arch == "glm_ocr") return MATH_OCR_GLM_OCR;
if (arch == "got_ocr") return MATH_OCR_GOT_OCR;
if (arch == "tesseract_lstm") return MATH_OCR_TESSERACT_LSTM;
if (arch == "granite_vision") return MATH_OCR_GRANITE_VISION;
    return MATH_OCR_PIX2TEX;
}

extern "C" void * crispembed_math_ocr_init(const char * path, int n_threads) {
    auto type = detect_arch(path);
    void * inner = nullptr;
    switch (type) {
        case MATH_OCR_PIX2TEX:      inner = math_ocr_init(path, n_threads); break;
        case MATH_OCR_HMER:         inner = hmer_ocr_init(path, n_threads); break;
        case MATH_OCR_BTTR:         inner = bttr_ocr_init(path, n_threads); break;
        case MATH_OCR_PPFORMULANET: inner = ppformulanet_ocr_init(path, n_threads); break;
        case MATH_OCR_PPFORMULANET_L: inner = ppformulanet_l_ocr_init(path, n_threads); break;
        case MATH_OCR_POSFORMER:      inner = posformer_ocr_init(path, n_threads); break;
        case MATH_OCR_MIXTEX:         inner = mixtex_ocr_init(path, n_threads); break;
        case MATH_OCR_QWEN2VL:        inner = qwen2vl_ocr_init(path, n_threads); break;
        case MATH_OCR_INTERNVL2:      inner = internvl2_ocr_init(path, n_threads); break;
case MATH_OCR_PARSEQ:         inner = parseq_ocr_init(path, n_threads); break;
case MATH_OCR_GLM_OCR:        inner = glm_ocr_init(path, n_threads); break;
case MATH_OCR_GOT_OCR:        inner = got_ocr_init(path, n_threads); break;
case MATH_OCR_TESSERACT_LSTM: inner = tesseract_lstm_init(path, n_threads); break;
case MATH_OCR_GRANITE_VISION: inner = granite_vision_init(path, n_threads); break;
    }
    if (!inner) return nullptr;
    auto * u = new unified_math_ocr{type, inner};
    return u;
}

extern "C" void crispembed_math_ocr_free(void * ctx) {
    if (!ctx) return;
    auto * u = (unified_math_ocr *)ctx;
    switch (u->type) {
        case MATH_OCR_PIX2TEX:      math_ocr_free((math_ocr_context *)u->ctx); break;
        case MATH_OCR_HMER:         hmer_ocr_free((hmer_ocr_context *)u->ctx); break;
        case MATH_OCR_BTTR:         bttr_ocr_free((bttr_ocr_context *)u->ctx); break;
        case MATH_OCR_PPFORMULANET: ppformulanet_ocr_free((ppformulanet_ocr_context *)u->ctx); break;
        case MATH_OCR_PPFORMULANET_L: ppformulanet_l_ocr_free((ppformulanet_l_ocr_context *)u->ctx); break;
        case MATH_OCR_POSFORMER:      posformer_ocr_free((posformer_ocr_context *)u->ctx); break;
        case MATH_OCR_MIXTEX:         mixtex_ocr_free((mixtex_ocr_context *)u->ctx); break;
        case MATH_OCR_QWEN2VL:        qwen2vl_ocr_free((qwen2vl_ocr_context *)u->ctx); break;
        case MATH_OCR_INTERNVL2:      internvl2_ocr_free((internvl2_ocr_context *)u->ctx); break;
case MATH_OCR_PARSEQ:         parseq_ocr_free((parseq_ocr_context *)u->ctx); break;
case MATH_OCR_GLM_OCR:        glm_ocr_free((glm_ocr_context *)u->ctx); break;
case MATH_OCR_GOT_OCR:        got_ocr_free((got_ocr_context *)u->ctx); break;
case MATH_OCR_TESSERACT_LSTM: tesseract_lstm_free((tesseract_lstm_context *)u->ctx); break;
case MATH_OCR_GRANITE_VISION: granite_vision_free((granite_vision_context *)u->ctx); break;
    }
    delete u;
}

extern "C" const char * crispembed_math_ocr_recognize(
    void * ctx, const uint8_t * px, int w, int h, int ch, int * ol
) {
    if (!ctx) return nullptr;
    auto * u = (unified_math_ocr *)ctx;
    switch (u->type) {
        case MATH_OCR_PIX2TEX:      return math_ocr_recognize_raw((math_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_HMER:         return hmer_ocr_recognize_raw((hmer_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_BTTR:         return bttr_ocr_recognize_raw((bttr_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_PPFORMULANET: return ppformulanet_ocr_recognize_raw((ppformulanet_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_PPFORMULANET_L: return ppformulanet_l_ocr_recognize_raw((ppformulanet_l_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_POSFORMER:      return posformer_ocr_recognize_raw((posformer_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_MIXTEX:         return mixtex_ocr_recognize((mixtex_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_QWEN2VL:        return qwen2vl_ocr_recognize_raw((qwen2vl_ocr_context *)u->ctx, px, w, h, ch, ol);
        case MATH_OCR_INTERNVL2:      return internvl2_ocr_recognize_raw((internvl2_ocr_context *)u->ctx, px, w, h, ch, ol);
case MATH_OCR_PARSEQ:         return parseq_ocr_recognize_raw((parseq_ocr_context *)u->ctx, px, w, h, ch, ol);
case MATH_OCR_GLM_OCR:        return glm_ocr_recognize_raw((glm_ocr_context *)u->ctx, px, w, h, ch, ol);
case MATH_OCR_GOT_OCR:        return got_ocr_recognize_raw((got_ocr_context *)u->ctx, px, w, h, ch, ol);
case MATH_OCR_TESSERACT_LSTM: {
            // tesseract_lstm_recognize takes grayscale uint8 — convert if needed
            if (ch == 1) {
                return tesseract_lstm_recognize((tesseract_lstm_context *)u->ctx, px, w, h, ol);
            }
            std::vector<uint8_t> gray(w * h);
            for (int i = 0; i < w * h; i++) {
                int r = px[i * ch], g = px[i * ch + 1], b = px[i * ch + 2];
                gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
            }
            return tesseract_lstm_recognize((tesseract_lstm_context *)u->ctx, gray.data(), w, h, ol);
        }
case MATH_OCR_GRANITE_VISION: return granite_vision_recognize((granite_vision_context *)u->ctx, px, w, h, ch, nullptr, ol);
    }
    return nullptr;
}

extern "C" const char * crispembed_math_ocr_recognize_gray(
    void * ctx, const float * px, int w, int h, int * ol
) {
    if (!ctx) return nullptr;
    auto * u = (unified_math_ocr *)ctx;
    switch (u->type) {
        case MATH_OCR_PIX2TEX:      return math_ocr_recognize((math_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_HMER:         return hmer_ocr_recognize((hmer_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_BTTR:         return bttr_ocr_recognize((bttr_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_PPFORMULANET: return ppformulanet_ocr_recognize((ppformulanet_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_PPFORMULANET_L: return ppformulanet_l_ocr_recognize((ppformulanet_l_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_POSFORMER:      return posformer_ocr_recognize((posformer_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_MIXTEX:         return mixtex_ocr_recognize_gray((mixtex_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_QWEN2VL:        return qwen2vl_ocr_recognize((qwen2vl_ocr_context *)u->ctx, px, w, h, ol);
        case MATH_OCR_INTERNVL2:      return internvl2_ocr_recognize((internvl2_ocr_context *)u->ctx, px, w, h, ol);
case MATH_OCR_PARSEQ:         return parseq_ocr_recognize((parseq_ocr_context *)u->ctx, px, w, h, ol);
case MATH_OCR_GLM_OCR:        return glm_ocr_recognize((glm_ocr_context *)u->ctx, px, w, h, ol);
case MATH_OCR_GOT_OCR:        return got_ocr_recognize((got_ocr_context *)u->ctx, px, w, h, ol);
case MATH_OCR_TESSERACT_LSTM: {
            // Convert float [0,1] grayscale → uint8
            std::vector<uint8_t> gray(w * h);
            for (int i = 0; i < w * h; i++)
                gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
            return tesseract_lstm_recognize((tesseract_lstm_context *)u->ctx, gray.data(), w, h, ol);
        }
case MATH_OCR_GRANITE_VISION: {
            // Convert float gray → uint8 RGB for granite_vision_recognize
            std::vector<uint8_t> rgb(w * h * 3);
            for (int i = 0; i < w * h; i++) {
                uint8_t v = (uint8_t)(px[i] * 255.0f + 0.5f);
                rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
            }
            return granite_vision_recognize((granite_vision_context *)u->ctx, rgb.data(), w, h, 3, nullptr, ol);
        }
    }
    return nullptr;
}

extern "C" const float * crispembed_math_ocr_confidences(const void * ctx, int * n_tokens) {
    if (n_tokens) *n_tokens = 0;
    if (!ctx) return nullptr;
    auto * u = (const unified_math_ocr *)ctx;
    switch (u->type) {
        case MATH_OCR_PIX2TEX:        return math_ocr_confidences((const math_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_HMER:           return hmer_ocr_confidences((const hmer_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_BTTR:           return bttr_ocr_confidences((const bttr_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_PPFORMULANET:   return ppformulanet_ocr_confidences((const ppformulanet_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_PPFORMULANET_L: return ppformulanet_l_ocr_confidences((const ppformulanet_l_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_POSFORMER:      return posformer_ocr_confidences((const posformer_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_MIXTEX:         return mixtex_ocr_confidences((const mixtex_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_QWEN2VL:        return qwen2vl_ocr_confidences((const qwen2vl_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_INTERNVL2:      return internvl2_ocr_confidences((const internvl2_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_PARSEQ:         return parseq_ocr_confidences((const parseq_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_GLM_OCR:        return glm_ocr_confidences((const glm_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_GOT_OCR:        return got_ocr_confidences((const got_ocr_context *)u->ctx, n_tokens);
        case MATH_OCR_TESSERACT_LSTM: return tesseract_lstm_confidences((const tesseract_lstm_context *)u->ctx, n_tokens);
        default: return nullptr;
    }
}

extern "C" float crispembed_math_ocr_mean_confidence(const void * ctx) {
    if (!ctx) return 0.0f;
    int n = 0;
    const float * c = crispembed_math_ocr_confidences(ctx, &n);
    if (!c || n <= 0) return 0.0f;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += c[i];
    return (float)(sum / n);
}

// Also expose individual APIs for direct use
extern "C" void * crispembed_hmer_ocr_init(const char * p, int t) { return hmer_ocr_init(p, t); }
extern "C" void crispembed_hmer_ocr_free(void * c) { hmer_ocr_free((hmer_ocr_context*)c); }
extern "C" const char * crispembed_hmer_ocr_recognize(void * c, const uint8_t * px, int w, int h, int ch, int * ol) {
    return hmer_ocr_recognize_raw((hmer_ocr_context*)c, px, w, h, ch, ol);
}
extern "C" const char * crispembed_hmer_ocr_recognize_gray(void * c, const float * px, int w, int h, int * ol) {
    return hmer_ocr_recognize((hmer_ocr_context*)c, px, w, h, ol);
}

extern "C" void * crispembed_bttr_ocr_init(const char * p, int t) { return bttr_ocr_init(p, t); }
extern "C" void crispembed_bttr_ocr_free(void * c) { bttr_ocr_free((bttr_ocr_context*)c); }
extern "C" const char * crispembed_bttr_ocr_recognize(void * c, const uint8_t * px, int w, int h, int ch, int * ol) {
    return bttr_ocr_recognize_raw((bttr_ocr_context*)c, px, w, h, ch, ol);
}
extern "C" const char * crispembed_bttr_ocr_recognize_gray(void * c, const float * px, int w, int h, int * ol) {
    return bttr_ocr_recognize((bttr_ocr_context*)c, px, w, h, ol);
}

// ---------------------------------------------------------------------------
// Pix2Struct — wrappers around pix2struct.h
// ---------------------------------------------------------------------------

extern "C" crispembed_pix2struct_context * crispembed_pix2struct_init(const char * model_path, int n_threads) {
    return (crispembed_pix2struct_context *)pix2struct_init(model_path, n_threads);
}

extern "C" void crispembed_pix2struct_free(crispembed_pix2struct_context * ctx) {
    if (ctx) pix2struct_free((pix2struct_context *)ctx);
}

extern "C" const char * crispembed_pix2struct_generate(
        crispembed_pix2struct_context * ctx,
        const uint8_t * image, int width, int height,
        int max_tokens) {
    if (!ctx) return nullptr;
    return pix2struct_generate((pix2struct_context *)ctx, image, width, height, max_tokens);
}

extern "C" void crispembed_pix2struct_free_text(const char * text) {
    pix2struct_free_text(text);
}

extern "C" const float * crispembed_pix2struct_encode_patches(
        crispembed_pix2struct_context * ctx,
        const float * patches, int n_patches,
        int * out_dim) {
    if (!ctx) return nullptr;
    return pix2struct_encode_patches((pix2struct_context *)ctx, patches, n_patches, out_dim);
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

// ---------------------------------------------------------------------------
// General OCR Pipeline C API
// ---------------------------------------------------------------------------

#include "ocr_pipeline.h"
#include "ocr_orchestrator.h"
#include "layout_detect.h"

struct ocr_pipeline_wrapper {
    ocr_pipeline::context * ctx = nullptr;
    std::vector<ocr_pipeline::ocr_result> results;
    std::vector<crispembed_ocr_result> c_results;
    std::string rec_buf;
};

extern "C" void * crispembed_ocr_init(const char * det_path, const char * rec_path, int n_threads) {
    auto * w = new ocr_pipeline_wrapper();
    if (!ocr_pipeline::load(&w->ctx, det_path, rec_path, n_threads)) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C" void crispembed_ocr_free(void * ctx) {
    if (!ctx) return;
    auto * w = (ocr_pipeline_wrapper *)ctx;
    if (w->ctx) ocr_pipeline::free(w->ctx);
    delete w;
}

extern "C" const crispembed_ocr_result * crispembed_ocr(
        void * ctx, const char * image_path, int * out_n) {
    if (!ctx || !image_path) { if (out_n) *out_n = 0; return nullptr; }
    auto * w = (ocr_pipeline_wrapper *)ctx;
    w->results = ocr_pipeline::run_file(w->ctx, image_path);
    w->c_results.resize(w->results.size());
    for (size_t i = 0; i < w->results.size(); i++) {
        auto & r = w->results[i];
        auto & c = w->c_results[i];
        c.x = r.box.x; c.y = r.box.y;
        c.w = r.box.w; c.h = r.box.h;
        c.confidence = r.confidence;
        c.text = r.text.c_str();
        c.text_len = (int)r.text.size();
    }
    if (out_n) *out_n = (int)w->c_results.size();
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

extern "C" const char * crispembed_ocr_recognize(
        void * ctx, const char * image_path, int * out_len) {
    if (!ctx || !image_path) { if (out_len) *out_len = 0; return nullptr; }
    auto * w = (ocr_pipeline_wrapper *)ctx;
    w->rec_buf = ocr_pipeline::recognize_file(w->ctx, image_path);
    if (out_len) *out_len = (int)w->rec_buf.size();
    return w->rec_buf.empty() ? nullptr : w->rec_buf.c_str();
}

// ---------------------------------------------------------------------------
// OCR Pipeline (orchestrator) — see ocr_orchestrator.{h,cpp}
// ---------------------------------------------------------------------------

struct ocr_pipeline_orch_wrapper {
    ocr_orchestrator::context * ctx = nullptr;
    ocr_orchestrator::result    last;
    std::vector<crispembed_ocr_result> c_results;
    std::string full_text;
    void * punct = nullptr;   // optional post-OCR punctuation/spacing restorer
};

extern "C" crispembed_ocr_pipeline_params crispembed_ocr_pipeline_defaults(void) {
    crispembed_ocr_pipeline_params p;
    p.router          = 1;
    p.cleanup_enabled = 1;
    p.min_chars       = 8;
    p.min_confidence  = 0.5f;
    p.det_model       = nullptr;
    p.rec_model       = nullptr;
    p.nafnet_model    = nullptr;
    p.sr_model        = nullptr;
    p.vlm_model       = nullptr;
    p.vlm_engine      = 0;
    p.punct_model     = nullptr;
    p.lid_model       = nullptr;
    p.truecase_model  = nullptr;
    p.tess_model_dir  = nullptr;
    return p;
}

extern "C" void * crispembed_ocr_pipeline_init(
        const crispembed_ocr_pipeline_params * params, int n_threads) {
    if (!params) return nullptr;
    ocr_orchestrator::config cfg = ocr_orchestrator::default_config();
    cfg.router = params->router != 0;
    if (params->nafnet_model && *params->nafnet_model) {
        cfg.nafnet_model = params->nafnet_model;
    }
    if (params->sr_model && *params->sr_model) {
        cfg.sr_model = params->sr_model;
    }
    // Apply the flat models / accept-gate / cleanup toggle to every stage of
    // every chain (per-stage config lands in a later slice via JSON).
    // Map the optional VLM escalation engine id.
    ocr_orchestrator::engine vlm_eng = ocr_orchestrator::engine::got;
    switch (params->vlm_engine) {
        case 1: vlm_eng = ocr_orchestrator::engine::glm;       break;
        case 2: vlm_eng = ocr_orchestrator::engine::qwen2vl;   break;
        case 3: vlm_eng = ocr_orchestrator::engine::internvl2; break;
        default: vlm_eng = ocr_orchestrator::engine::got;      break;
    }
    const bool has_vlm = params->vlm_model && *params->vlm_model;
    for (auto & ch : cfg.chains) {
        for (auto & st : ch.stages) {
            if (params->det_model) st.model_a = params->det_model;
            if (params->rec_model) st.model_b = params->rec_model;
            st.accept.min_chars      = params->min_chars;
            st.accept.min_confidence = params->min_confidence;
            if (!params->cleanup_enabled) st.cleanup.enabled = false;
        }
        // Append a single-shot VLM escalation stage: the chain tries the fast
        // DBNet+TrOCR first and falls back to the VLM when the accept-gate fails.
        if (has_vlm) {
            ocr_orchestrator::stage vs;
            vs.eng             = vlm_eng;
            vs.enabled         = true;
            vs.model_a         = params->vlm_model;
            vs.accept.min_chars      = params->min_chars;
            vs.accept.min_confidence = 0.0f;  // VLM has no per-region confidence
            vs.cleanup.enabled = params->cleanup_enabled != 0;
            vs.cleanup.params  = scan_cleanup_defaults();
            vs.cleanup.params.binarize = 0;   // never binarize for a VLM
            vs.cleanup.denoise = false;
            ch.stages.push_back(vs);
        }
    }
    // LID + truecasing + Tesseract auto-select
    if (params->lid_model && *params->lid_model)
        cfg.lid_model = params->lid_model;
    if (params->truecase_model && *params->truecase_model)
        cfg.truecase_model = params->truecase_model;
    if (params->tess_model_dir && *params->tess_model_dir)
        cfg.tess_model_dir = params->tess_model_dir;

    // Enable verbose logging via environment variable
    if (const char * v = std::getenv("CRISPEMBED_VERBOSE_OCR"))
        cfg.verbose = (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');

    auto * w = new ocr_pipeline_orch_wrapper();
    if (!ocr_orchestrator::load(&w->ctx, cfg, n_threads)) {
        delete w;
        return nullptr;
    }
    if (params->punct_model && *params->punct_model) {
        w->punct = crispembed_punct_init(params->punct_model, n_threads);
    }
    return w;
}

extern "C" const crispembed_ocr_result * crispembed_ocr_pipeline_run(
        void * ctx, const char * image_path, int * out_n,
        const char ** out_full_text, float * out_mean_conf) {
    if (out_n)         *out_n         = 0;
    if (out_full_text) *out_full_text = nullptr;
    if (out_mean_conf) *out_mean_conf = 0.0f;
    if (!ctx || !image_path) return nullptr;
    auto * w  = (ocr_pipeline_orch_wrapper *)ctx;
    w->last      = ocr_orchestrator::run_file(w->ctx, image_path);
    w->full_text = w->last.full_text;
    // Optional post-OCR restore: punctuation / capitalization / spacing.
    if (w->punct && !w->full_text.empty()) {
        const char * restored = crispembed_punct_process(w->punct, w->full_text.c_str());
        if (restored && *restored) w->full_text = restored;
    }
    w->c_results.resize(w->last.regions.size());
    for (size_t i = 0; i < w->last.regions.size(); i++) {
        auto & r = w->last.regions[i];
        auto & c = w->c_results[i];
        c.x = r.box.x; c.y = r.box.y;
        c.w = r.box.w; c.h = r.box.h;
        c.confidence = r.confidence;
        c.text       = r.text.c_str();
        c.text_len   = (int)r.text.size();
    }
    if (out_n)         *out_n         = (int)w->c_results.size();
    if (out_full_text) *out_full_text = w->full_text.c_str();
    if (out_mean_conf) *out_mean_conf = w->last.mean_confidence;
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

static ocr_orchestrator::engine map_engine(int e) {
    using E = ocr_orchestrator::engine;
    switch (e) {
        case 1:  return E::surya;
        case 2:  return E::got;
        case 3:  return E::glm;
        case 4:  return E::qwen2vl;
        case 5:  return E::internvl2;
        case 6:  return E::tesseract;
        default: return E::dbnet_trocr;
    }
}

static ocr_orchestrator::source_type map_source(int s) {
    using S = ocr_orchestrator::source_type;
    switch (s) {
        case 1:  return S::screenshot;
        case 2:  return S::scanned_doc;
        case 3:  return S::photo;
        default: return S::auto_detect;
    }
}

// crispembed_scan_cleanup_params (C API) and scan_cleanup_params (internal)
// share the same field layout; copy field-by-field across the type boundary.
static scan_cleanup_params to_cleanup(const crispembed_scan_cleanup_params & p) {
    scan_cleanup_params o;
    o.deskew            = p.deskew;
    o.crop_borders      = p.crop_borders;
    o.whiten_background  = p.whiten_background;
    o.binarize          = p.binarize;
    o.binarize_method   = p.binarize_method;
    o.sauvola_k         = p.sauvola_k;
    o.sauvola_window    = p.sauvola_window;
    o.morph_kernel      = p.morph_kernel;
    o.border_threshold  = p.border_threshold;
    o.deskew_max_angle  = p.deskew_max_angle;
    return o;
}

extern "C" void * crispembed_ocr_pipeline_init_stages(
        int router, const char * nafnet_model, const char * sr_model,
        const char * punct_model,
        const char * lid_model, const char * truecase_model,
        const char * tess_model_dir,
        const crispembed_ocr_stage * stages, int n_stages, int n_threads) {
    if (!stages || n_stages <= 0) return nullptr;
    ocr_orchestrator::config cfg;
    cfg.router = router != 0;
    if (nafnet_model && *nafnet_model) cfg.nafnet_model = nafnet_model;
    if (sr_model && *sr_model) cfg.sr_model = sr_model;
    // LID + truecasing + Tesseract auto-select (mirrors the params-struct path).
    if (lid_model && *lid_model) cfg.lid_model = lid_model;
    if (truecase_model && *truecase_model) cfg.truecase_model = truecase_model;
    if (tess_model_dir && *tess_model_dir) cfg.tess_model_dir = tess_model_dir;

    // Group stages into per-source-type chains, preserving array order.
    for (int i = 0; i < n_stages; i++) {
        const crispembed_ocr_stage & s = stages[i];
        ocr_orchestrator::stage st;
        st.eng     = map_engine(s.engine);
        st.enabled = true;
        if (s.model_a) st.model_a = s.model_a;
        if (s.model_b) st.model_b = s.model_b;
        st.cleanup.enabled = s.cleanup_enabled != 0;
        st.cleanup.params  = to_cleanup(s.cleanup);
        st.cleanup.denoise = s.denoise != 0;
        st.params.det_prob_threshold = s.det_prob_threshold;
        st.params.det_box_threshold  = s.det_box_threshold;
        st.params.det_target_short   = s.det_target_short > 0 ? s.det_target_short : 736;
        st.params.vlm_max_tokens     = s.vlm_max_tokens;
        if (s.vlm_prompt && *s.vlm_prompt) st.params.vlm_prompt = s.vlm_prompt;
        st.accept.min_chars      = s.min_chars;
        st.accept.min_confidence = s.min_confidence;

        const ocr_orchestrator::source_type type = map_source(s.source_type);
        ocr_orchestrator::chain * chain = nullptr;
        for (auto & c : cfg.chains) {
            if (c.type == type) { chain = &c; break; }
        }
        if (!chain) {
            ocr_orchestrator::chain c;
            c.type = type;
            cfg.chains.push_back(c);
            chain = &cfg.chains.back();
        }
        chain->stages.push_back(st);
    }

    auto * w = new ocr_pipeline_orch_wrapper();
    if (!ocr_orchestrator::load(&w->ctx, cfg, n_threads)) {
        delete w;
        return nullptr;
    }
    if (punct_model && *punct_model) {
        w->punct = crispembed_punct_init(punct_model, n_threads);
    }
    return w;
}

extern "C" const char * crispembed_ocr_pipeline_detected_lang(void * ctx, float * out_confidence) {
    if (!ctx) { if (out_confidence) *out_confidence = 0.0f; return ""; }
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (out_confidence) *out_confidence = w->last.lang_confidence;
    return w->last.detected_lang.c_str();
}

// Per-region recognition confidence (mean per-char softmax) from the last run.
// Returns 0 for an out-of-range index or recognizer that yields no confidence.
extern "C" float crispembed_ocr_pipeline_region_rec_confidence(void * ctx, int region_idx) {
    if (!ctx) return 0.0f;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (region_idx < 0 || (size_t)region_idx >= w->last.regions.size()) return 0.0f;
    return w->last.regions[region_idx].rec_confidence;
}

// Per-character confidence for a region from the last run. Returns a pointer to
// `*out_len` floats (owned by ctx, valid until the next run / free), or NULL
// when the recognizer doesn't expose per-character confidence.
extern "C" const float * crispembed_ocr_pipeline_region_char_conf(
        void * ctx, int region_idx, int * out_len) {
    if (out_len) *out_len = 0;
    if (!ctx) return nullptr;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (region_idx < 0 || (size_t)region_idx >= w->last.regions.size()) return nullptr;
    auto & cc = w->last.regions[region_idx].char_conf;
    if (cc.empty()) return nullptr;
    if (out_len) *out_len = (int)cc.size();
    return cc.data();
}

extern "C" void crispembed_ocr_pipeline_free(void * ctx) {
    if (!ctx) return;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (w->ctx) ocr_orchestrator::free(w->ctx);
    if (w->punct) crispembed_punct_free(w->punct);
    delete w;
}

// ---------------------------------------------------------------------------
// Standalone Text LID
// ---------------------------------------------------------------------------

#if __has_include("text_lid_dispatch.h")
#include "text_lid_dispatch.h"
#define HAS_LID 1
#else
#define HAS_LID 0
#endif

extern "C" void * crispembed_lid_init(const char * model_path, int n_threads) {
#if HAS_LID
    return text_lid_init_from_file(model_path, n_threads);
#else
    (void)model_path; (void)n_threads;
    fprintf(stderr, "crispembed: LID not available (crisp_lid not linked)\n");
    return nullptr;
#endif
}

extern "C" void crispembed_lid_free(void * ctx) {
#if HAS_LID
    if (ctx) text_lid_free((text_lid_context *)ctx);
#else
    (void)ctx;
#endif
}

extern "C" const char * crispembed_lid_predict(void * ctx, const char * text, float * out_confidence) {
#if HAS_LID
    if (!ctx || !text) { if (out_confidence) *out_confidence = 0; return ""; }
    return text_lid_predict((text_lid_context *)ctx, text, out_confidence);
#else
    (void)ctx; (void)text;
    if (out_confidence) *out_confidence = 0;
    return "";
#endif
}

extern "C" int crispembed_lid_predict_topk(void * ctx, const char * text, int k,
                                            const char ** out_labels, float * out_confidences) {
#if HAS_LID
    if (!ctx || !text) return 0;
    return text_lid_predict_topk((text_lid_context *)ctx, text, k, out_labels, out_confidences);
#else
    (void)ctx; (void)text; (void)k; (void)out_labels; (void)out_confidences;
    return 0;
#endif
}

extern "C" int crispembed_lid_n_labels(const void * ctx) {
#if HAS_LID
    return ctx ? text_lid_n_labels((const text_lid_context *)ctx) : 0;
#else
    (void)ctx; return 0;
#endif
}

// ---------------------------------------------------------------------------
// Layout Detection (RT-DETRv2)
// ---------------------------------------------------------------------------

struct layout_wrapper {
    layout_detect::context * ctx = nullptr;
    std::vector<crispembed_layout_region> c_results;
};

extern "C" void * crispembed_layout_init(const char * model_path, int n_threads) {
    auto * w = new layout_wrapper();
    if (!layout_detect::load(&w->ctx, model_path, n_threads)) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C" void crispembed_layout_free(void * ctx) {
    if (!ctx) return;
    auto * w = (layout_wrapper *)ctx;
    layout_detect::free(w->ctx);
    delete w;
}

extern "C" const crispembed_layout_region * crispembed_layout_detect(
        void * ctx, const char * image_path, float score_threshold, int * out_n) {
    if (!ctx || !image_path) { if (out_n) *out_n = 0; return nullptr; }
    auto * w = (layout_wrapper *)ctx;
    auto regions = layout_detect::detect_file(w->ctx, image_path, score_threshold);
    w->c_results.resize(regions.size());
    for (size_t i = 0; i < regions.size(); i++) {
        w->c_results[i].x1 = regions[i].x1;
        w->c_results[i].y1 = regions[i].y1;
        w->c_results[i].x2 = regions[i].x2;
        w->c_results[i].y2 = regions[i].y2;
        w->c_results[i].score = regions[i].score;
        w->c_results[i].label = (int)regions[i].label;
        w->c_results[i].label_name = regions[i].label_name;
    }
    if (out_n) *out_n = (int)regions.size();
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

// ---------------------------------------------------------------------------
// Surya Text Detection (EfficientViT segformer)
// ---------------------------------------------------------------------------

#include "surya_det.h"

struct surya_det_wrapper {
    surya_det_context * ctx = nullptr;
    std::vector<crispembed_text_det_result> c_results;
};

extern "C" void * crispembed_text_det_init(const char * model_path, int n_threads) {
    auto * w = new surya_det_wrapper();
    w->ctx = surya_det_init(model_path, n_threads);
    if (!w->ctx) { delete w; return nullptr; }
    return w;
}

extern "C" void crispembed_text_det_free(void * ctx) {
    if (!ctx) return;
    auto * w = (surya_det_wrapper *)ctx;
    surya_det_free(w->ctx);
    delete w;
}

extern "C" const crispembed_text_det_result * crispembed_text_det(
        void * ctx, const uint8_t * pixels, int width, int height, int channels,
        float text_threshold, float low_threshold, int * out_n) {
    if (!ctx || !pixels) { if (out_n) *out_n = 0; return nullptr; }
    auto * w = (surya_det_wrapper *)ctx;

    // Run detection
    int hm_h = 0, hm_w = 0;
    surya_det_detect(w->ctx, pixels, width, height, channels, &hm_h, &hm_w);

    // Extract boxes
    int n_boxes = 0;
    const surya_det_bbox * boxes = surya_det_get_boxes(w->ctx, width, height,
                                                        text_threshold, low_threshold,
                                                        &n_boxes);
    w->c_results.resize(n_boxes);
    for (int i = 0; i < n_boxes; i++) {
        w->c_results[i].x0 = boxes[i].x0;
        w->c_results[i].y0 = boxes[i].y0;
        w->c_results[i].x1 = boxes[i].x1;
        w->c_results[i].y1 = boxes[i].y1;
        w->c_results[i].confidence = boxes[i].confidence;
    }
    if (out_n) *out_n = n_boxes;
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

extern "C" const float * crispembed_text_det_heatmap(void * ctx, int * out_h, int * out_w) {
    if (!ctx) return nullptr;
    auto * w = (surya_det_wrapper *)ctx;
    return surya_det_get_heatmap(w->ctx, out_h, out_w);
}

// ===========================================================================
// Named Entity Recognition (GLiNER + BERT NER auto-detect)
// ===========================================================================

#include "gliner_ner.h"
#include "bert_ner.h"

// Dispatch wrapper: holds either a GLiNER or BERT NER context.
struct ner_dispatch {
    enum { GLINER, BERT_NER } backend;
    void * gliner_ctx = nullptr;
    bert_ner::context * bert_ctx = nullptr;

    // BERT NER: last result storage for C API lifetime
    std::vector<bert_ner::entity> last_entities;
    std::vector<crispembed_ner_entity> last_c_entities;
    std::vector<std::string> last_texts;   // keep strings alive
    std::vector<std::string> last_labels;
};

extern "C" void * crispembed_ner_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;

    // Peek at GGUF metadata to decide backend.
    gguf_context * gctx = core_gguf::open_metadata(model_path);
    if (!gctx) return nullptr;

    uint32_t ner_labels = core_gguf::kv_u32(gctx, "ner.num_labels", 0);
    core_gguf::free_metadata(gctx);

    auto * d = new ner_dispatch;

    if (ner_labels > 0) {
        // BERT NER path (fixed-label token classification)
        d->backend = ner_dispatch::BERT_NER;
        if (!bert_ner::load(&d->bert_ctx, model_path, n_threads)) {
            delete d;
            return nullptr;
        }
        fprintf(stderr, "crispembed_ner: using BERT NER backend (%d labels)\n", ner_labels);
    } else {
        // GLiNER path (zero-shot)
        d->backend = ner_dispatch::GLINER;
        d->gliner_ctx = gliner_ner_init(model_path, n_threads);
        if (!d->gliner_ctx) {
            delete d;
            return nullptr;
        }
    }

    return d;
}

extern "C" void crispembed_ner_free(void * ctx) {
    if (!ctx) return;
    auto * d = (ner_dispatch *)ctx;
    if (d->backend == ner_dispatch::GLINER && d->gliner_ctx)
        gliner_ner_free(d->gliner_ctx);
    if (d->backend == ner_dispatch::BERT_NER && d->bert_ctx)
        bert_ner::free(d->bert_ctx);
    delete d;
}

extern "C" int crispembed_ner_extract(void * ctx, const char * text,
                                      const char ** labels, int n_labels,
                                      float threshold,
                                      crispembed_ner_entity ** out_entities) {
    if (!ctx || !text) return 0;
    auto * d = (ner_dispatch *)ctx;

    if (d->backend == ner_dispatch::GLINER) {
        gliner_ner_entity * ents = nullptr;
        int n = gliner_ner_extract(d->gliner_ctx, text, labels, n_labels, threshold, &ents);
        if (out_entities)
            *out_entities = (crispembed_ner_entity *)ents;
        return n;
    }

    // BERT NER: fixed labels (ignore user-supplied labels/threshold)
    d->last_entities = bert_ner::extract(d->bert_ctx, text);

    // Convert to C API structs
    d->last_texts.clear();
    d->last_labels.clear();
    d->last_c_entities.clear();
    d->last_texts.reserve(d->last_entities.size());
    d->last_labels.reserve(d->last_entities.size());
    d->last_c_entities.reserve(d->last_entities.size());

    for (const auto & e : d->last_entities) {
        d->last_texts.push_back(e.text);
        d->last_labels.push_back(e.label);
    }
    for (size_t i = 0; i < d->last_entities.size(); i++) {
        crispembed_ner_entity ce;
        ce.start_char = d->last_entities[i].start_char;
        ce.end_char   = d->last_entities[i].end_char;
        ce.text       = d->last_texts[i].c_str();
        ce.label      = d->last_labels[i].c_str();
        ce.score      = d->last_entities[i].score;
        d->last_c_entities.push_back(ce);
    }

    if (out_entities)
        *out_entities = d->last_c_entities.data();
    return (int)d->last_c_entities.size();
}

// ===========================================================================
// LiLT — Language-independent Layout Transformer
// ===========================================================================

#include "lilt_kie.h"

struct crispembed_lilt_ctx {
    lilt_kie::context* pipe = nullptr;
    std::vector<lilt_kie::token_result> last_results;
    std::vector<crispembed_lilt_token> last_tokens;
};

extern "C" void * crispembed_lilt_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto* ctx = new crispembed_lilt_ctx;
    if (!lilt_kie::load(&ctx->pipe, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" void crispembed_lilt_free(void * ptr) {
    if (!ptr) return;
    auto* ctx = (crispembed_lilt_ctx*)ptr;
    if (ctx->pipe) lilt_kie::free(ctx->pipe);
    delete ctx;
}

extern "C" const crispembed_lilt_token * crispembed_lilt_classify(
        void * ptr, const int32_t * input_ids, const int32_t * bbox,
        int n_tokens, int * out_n) {
    if (out_n) *out_n = 0;
    if (!ptr || !input_ids || !bbox || n_tokens <= 0) return nullptr;

    auto* ctx = (crispembed_lilt_ctx*)ptr;
    ctx->last_results = lilt_kie::classify(ctx->pipe, input_ids, bbox, n_tokens);

    ctx->last_tokens.clear();
    ctx->last_tokens.reserve(ctx->last_results.size());
    for (const auto& r : ctx->last_results) {
        crispembed_lilt_token t;
        t.token_id = r.token_id;
        t.label_id = r.label_id;
        t.label    = r.label.c_str();
        t.score    = r.score;
        ctx->last_tokens.push_back(t);
    }

    if (out_n) *out_n = (int)ctx->last_tokens.size();
    return ctx->last_tokens.data();
}

extern "C" int crispembed_lilt_num_labels(void * ptr) {
    if (!ptr) return 0;
    return lilt_kie::num_labels(((crispembed_lilt_ctx*)ptr)->pipe);
}

extern "C" const char * crispembed_lilt_label_name(void * ptr, int label_id) {
    if (!ptr) return "";
    return lilt_kie::label_name(((crispembed_lilt_ctx*)ptr)->pipe, label_id);
}

// ===========================================================================
// Key Information Extraction (KIE)
// ===========================================================================

#include "kie_pipeline.h"

// Internal state for the C API — holds the pipeline context plus the last
// result's strings/fields so they stay alive for the caller.
struct crispembed_kie_ctx {
    kie_pipeline::context* pipe = nullptr;

    // Last result storage (valid until next extract call or free).
    kie_pipeline::result last_result;
    std::vector<crispembed_kie_field> last_fields;
    std::string last_ocr_text;
};

extern "C" void * crispembed_kie_init(
        const char * ocr_det_model, const char * ocr_rec_model,
        const char * ner_model, int n_threads) {
    if (!ocr_det_model || !ocr_rec_model || !ner_model) return nullptr;

    kie_pipeline::config cfg;
    cfg.ocr = ocr_orchestrator::default_config();

    // Wire in the provided OCR models to the first stage of the default chain.
    for (auto& chain : cfg.ocr.chains) {
        for (auto& stage : chain.stages) {
            if (stage.eng == ocr_orchestrator::engine::dbnet_trocr) {
                stage.model_a = ocr_det_model;
                stage.model_b = ocr_rec_model;
            }
        }
    }
    cfg.ner_model = ner_model;
    cfg.threshold = 0.5f;

    auto* kctx = new crispembed_kie_ctx;
    if (!kie_pipeline::load(&kctx->pipe, cfg, n_threads)) {
        delete kctx;
        return nullptr;
    }
    return kctx;
}

// LiLT-aware KIE: same as crispembed_kie_init but also wires a LiLT GGUF so the
// pipeline runs layout-aware token classification (Phase 2). ner_model may be
// empty when relying on LiLT alone.
extern "C" void * crispembed_kie_init_lilt(
        const char * ocr_det_model, const char * ocr_rec_model,
        const char * ner_model, const char * lilt_model, int n_threads) {
    if (!ocr_det_model || !ocr_rec_model) return nullptr;

    kie_pipeline::config cfg;
    cfg.ocr = ocr_orchestrator::default_config();
    for (auto& chain : cfg.ocr.chains) {
        for (auto& stage : chain.stages) {
            if (stage.eng == ocr_orchestrator::engine::dbnet_trocr) {
                stage.model_a = ocr_det_model;
                stage.model_b = ocr_rec_model;
            }
        }
    }
    if (ner_model && *ner_model) cfg.ner_model = ner_model;
    if (lilt_model && *lilt_model) cfg.lilt_model = lilt_model;
    cfg.threshold = 0.5f;

    auto* kctx = new crispembed_kie_ctx;
    if (!kie_pipeline::load(&kctx->pipe, cfg, n_threads)) {
        delete kctx;
        return nullptr;
    }
    return kctx;
}

extern "C" crispembed_kie_result crispembed_kie_extract(
        void * ptr, const char * image_path,
        const char ** labels, int n_labels,
        float threshold) {
    crispembed_kie_result out;
    std::memset(&out, 0, sizeof(out));
    if (!ptr || !image_path) return out;

    auto* kctx = (crispembed_kie_ctx*)ptr;

    kctx->last_result = kie_pipeline::extract(kctx->pipe, image_path, labels, n_labels, threshold);
    kctx->last_ocr_text = kctx->last_result.ocr_full_text;

    // Build flat C field array.
    kctx->last_fields.clear();
    kctx->last_fields.reserve(kctx->last_result.fields.size());
    for (const auto& f : kctx->last_result.fields) {
        crispembed_kie_field cf;
        cf.label = f.label.c_str();
        cf.value = f.value.c_str();
        cf.score = f.score;
        cf.x = f.x;
        cf.y = f.y;
        cf.w = f.w;
        cf.h = f.h;
        kctx->last_fields.push_back(cf);
    }

    out.fields         = kctx->last_fields.data();
    out.n_fields       = (int)kctx->last_fields.size();
    out.ocr_text       = kctx->last_ocr_text.c_str();
    out.ocr_confidence = kctx->last_result.ocr_confidence;
    out.n_ocr_regions  = kctx->last_result.n_ocr_regions;
    return out;
}

extern "C" void crispembed_kie_free(void * ptr) {
    if (!ptr) return;
    auto* kctx = (crispembed_kie_ctx*)ptr;
    if (kctx->pipe) kie_pipeline::free(kctx->pipe);
    delete kctx;
}

// ===========================================================================
// Scan Cleanup
// ===========================================================================

#include "scan_cleanup.h"

extern "C" crispembed_scan_cleanup_params crispembed_scan_cleanup_defaults(void) {
    scan_cleanup_params p = scan_cleanup_defaults();
    crispembed_scan_cleanup_params cp;
    cp.deskew            = p.deskew;
    cp.crop_borders      = p.crop_borders;
    cp.whiten_background = p.whiten_background;
    cp.binarize          = p.binarize;
    cp.binarize_method   = p.binarize_method;
    cp.sauvola_k         = p.sauvola_k;
    cp.sauvola_window    = p.sauvola_window;
    cp.morph_kernel      = p.morph_kernel;
    cp.border_threshold  = p.border_threshold;
    cp.deskew_max_angle  = p.deskew_max_angle;
    return cp;
}

extern "C" void * crispembed_scan_cleanup_init(const char * model_path, int n_threads) {
    return scan_cleanup_init(model_path, n_threads);
}

extern "C" void crispembed_scan_cleanup_free(void * ctx) {
    scan_cleanup_free((scan_cleanup_ctx *)ctx);
}

extern "C" int crispembed_scan_cleanup_process(
        void * ctx,
        const uint8_t * pixels, int width, int height, int channels,
        crispembed_scan_cleanup_params params,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    scan_cleanup_params p;
    p.deskew            = params.deskew;
    p.crop_borders      = params.crop_borders;
    p.whiten_background = params.whiten_background;
    p.binarize          = params.binarize;
    p.binarize_method   = params.binarize_method;
    p.sauvola_k         = params.sauvola_k;
    p.sauvola_window    = params.sauvola_window;
    p.morph_kernel      = params.morph_kernel;
    p.border_threshold  = params.border_threshold;
    p.deskew_max_angle  = params.deskew_max_angle;
    return scan_cleanup_process((scan_cleanup_ctx *)ctx, pixels, width, height, channels,
                                p, out_pixels, out_width, out_height);
}

extern "C" void crispembed_scan_cleanup_free_image(uint8_t * pixels) {
    scan_cleanup_free_image(pixels);
}

extern "C" int crispembed_scan_cleanup_process_simple(
        void * ctx,
        const uint8_t * pixels, int width, int height, int channels,
        int deskew, int crop_borders, int whiten_background, int binarize,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    scan_cleanup_params p = scan_cleanup_defaults();
    p.deskew = deskew;
    p.crop_borders = crop_borders;
    p.whiten_background = whiten_background;
    p.binarize = binarize;
    return scan_cleanup_process((scan_cleanup_ctx *)ctx, pixels, width, height, channels,
                                p, out_pixels, out_width, out_height);
}

// ---------------------------------------------------------------------------
// Text super-resolution
// ---------------------------------------------------------------------------

#include "text_sr.h"

extern "C" void * crispembed_text_sr_init(const char * model_path, int n_threads) {
    return text_sr_init(model_path, n_threads);
}

extern "C" void crispembed_text_sr_free(void * ctx) {
    text_sr_free((text_sr_context *)ctx);
}

extern "C" int crispembed_text_sr_upscale_factor(const void * ctx) {
    return text_sr_upscale_factor((const text_sr_context *)ctx);
}

extern "C" int crispembed_text_sr_process(
        void * ctx,
        const uint8_t * pixels, int width, int height,
        int tile_size, int tile_overlap,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return text_sr_process((text_sr_context *)ctx, pixels, width, height,
                           tile_size, tile_overlap, out_pixels, out_width, out_height);
}

extern "C" void crispembed_text_sr_free_image(uint8_t * pixels) {
    text_sr_free_image(pixels);
}

// ---------------------------------------------------------------------------
// TBSRN text-line super-resolution
// ---------------------------------------------------------------------------

#include "tbsrn_sr.h"

extern "C" void * crispembed_tbsrn_sr_init(const char * model_path, int n_threads) {
    return tbsrn_sr_init(model_path, n_threads);
}

extern "C" void crispembed_tbsrn_sr_free(void * ctx) {
    tbsrn_sr_free((tbsrn_sr_context *)ctx);
}

extern "C" int crispembed_tbsrn_sr_process(
        void * ctx,
        const uint8_t * pixels, int width, int height,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return tbsrn_sr_process((tbsrn_sr_context *)ctx, pixels, width, height,
                            out_pixels, out_width, out_height);
}

extern "C" void crispembed_tbsrn_sr_free_image(uint8_t * pixels) {
    tbsrn_sr_free_image(pixels);
}

// ---------------------------------------------------------------------------
// PAN whole-image super-resolution
// ---------------------------------------------------------------------------

#include "pan_sr.h"

extern "C" void * crispembed_pan_sr_init(const char * model_path, int n_threads) {
    return pan_sr_init(model_path, n_threads);
}
extern "C" void crispembed_pan_sr_free(void * ctx) {
    pan_sr_free((pan_sr_context *)ctx);
}
extern "C" int crispembed_pan_sr_scale(const void * ctx) {
    return pan_sr_scale((const pan_sr_context *)ctx);
}
extern "C" int crispembed_pan_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int tile_size, int tile_overlap,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return pan_sr_process((pan_sr_context *)ctx, pixels, width, height,
                          tile_size, tile_overlap, out_pixels, out_width, out_height);
}
extern "C" void crispembed_pan_sr_free_image(uint8_t * pixels) {
    pan_sr_free_image(pixels);
}

// ---------------------------------------------------------------------------
// DAT (Dual Aggregation Transformer) super-resolution
// ---------------------------------------------------------------------------

#include "dat_sr.h"

extern "C" void * crispembed_dat_sr_init(const char * model_path, int n_threads) {
    return dat_sr_init(model_path, n_threads);
}
extern "C" void crispembed_dat_sr_free(void * ctx) {
    dat_sr_free((dat_sr_context *)ctx);
}
extern "C" int crispembed_dat_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int tile_w, int tile_h,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return dat_sr_process((dat_sr_context *)ctx, pixels, width, height,
                          tile_w, tile_h, out_pixels, out_width, out_height);
}
extern "C" void crispembed_dat_sr_free_image(uint8_t * pixels) {
    dat_sr_free_image(pixels);
}

// ---------------------------------------------------------------------------
// SAFMN whole-image super-resolution
// ---------------------------------------------------------------------------

#include "safmn_sr.h"

extern "C" void * crispembed_safmn_sr_init(const char * model_path, int n_threads) {
    return safmn_init(model_path, n_threads);
}
extern "C" void crispembed_safmn_sr_free(void * ctx) {
    safmn_free((safmn_context *)ctx);
}
extern "C" int crispembed_safmn_sr_scale(const void * ctx) {
    return safmn_get_scale((const safmn_context *)ctx);
}
extern "C" int crispembed_safmn_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int /*tile_size*/, int /*tile_overlap*/,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    int scale = safmn_get_scale((const safmn_context *)ctx);
    int ow = width * scale;
    int oh = height * scale;
    uint8_t * out = (uint8_t *)malloc((size_t)ow * oh * 3);
    if (!out) return -1;
    int rc = safmn_process((safmn_context *)ctx, pixels, width, height, out);
    if (rc != 0) { free(out); return rc; }
    *out_pixels = out;
    *out_width  = ow;
    *out_height = oh;
    return 0;
}
extern "C" void crispembed_safmn_sr_free_image(uint8_t * pixels) {
    free(pixels);
}

// ---------------------------------------------------------------------------
// SwinIR-light whole-image super-resolution
// ---------------------------------------------------------------------------

#include "swinir_sr.h"

extern "C" void * crispembed_swinir_sr_init(const char * model_path, int n_threads) {
    return swinir_sr_init(model_path, n_threads);
}
extern "C" void crispembed_swinir_sr_free(void * ctx) {
    swinir_sr_free((swinir_sr_context *)ctx);
}
extern "C" int crispembed_swinir_sr_scale(const void * ctx) {
    return swinir_sr_scale((const swinir_sr_context *)ctx);
}
extern "C" int crispembed_swinir_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int tile_size, int tile_overlap,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return swinir_sr_process((swinir_sr_context *)ctx, pixels, width, height,
                             tile_size, tile_overlap, out_pixels, out_width, out_height);
}
extern "C" void crispembed_swinir_sr_free_image(uint8_t * pixels) {
    swinir_sr_free_image(pixels);
}

// ---------------------------------------------------------------------------
// Real-ESRGAN whole-image super-resolution
// ---------------------------------------------------------------------------

#include "esrgan_sr.h"

extern "C" void * crispembed_esrgan_sr_init(const char * model_path, int n_threads) {
    return esrgan_init(model_path, n_threads);
}
extern "C" void crispembed_esrgan_sr_free(void * ctx) {
    esrgan_free((esrgan_context *)ctx);
}
extern "C" int crispembed_esrgan_sr_scale(const void * ctx) {
    return esrgan_get_scale((const esrgan_context *)ctx);
}
extern "C" int crispembed_esrgan_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int /*tile_size*/, int /*tile_overlap*/,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    int scale = esrgan_get_scale((const esrgan_context *)ctx);
    int ow = width * scale;
    int oh = height * scale;
    uint8_t * out = (uint8_t *)malloc((size_t)ow * oh * 3);
    if (!out) return -1;
    int rc = esrgan_process((esrgan_context *)ctx, pixels, width, height, out);
    if (rc != 0) { free(out); return rc; }
    *out_pixels = out;
    *out_width  = ow;
    *out_height = oh;
    return 0;
}
extern "C" void crispembed_esrgan_sr_free_image(uint8_t * pixels) {
    free(pixels);
}

// ---------------------------------------------------------------------------
// Restormer image restoration
// ---------------------------------------------------------------------------

#include "restormer.h"

extern "C" void * crispembed_restormer_init(const char * model_path, int n_threads) {
    return restormer_init(model_path, n_threads);
}
extern "C" void crispembed_restormer_free(void * ctx) {
    restormer_free((restormer_context *)ctx);
}
extern "C" int crispembed_restormer_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int tile_size, int tile_overlap, uint8_t ** out_pixels) {
    return restormer_process((restormer_context *)ctx, pixels, width, height,
                             tile_size, tile_overlap, out_pixels);
}
extern "C" void crispembed_restormer_free_image(uint8_t * pixels) {
    restormer_free_image(pixels);
}

// ---------------------------------------------------------------------------
// SCUNet image denoising
// ---------------------------------------------------------------------------

#include "scunet_denoise.h"

extern "C" void * crispembed_scunet_init(const char * model_path, int n_threads) {
    return scunet_init(model_path, n_threads);
}
extern "C" void crispembed_scunet_free(void * ctx) {
    scunet_free((scunet_context *)ctx);
}
extern "C" int crispembed_scunet_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = scunet_process((scunet_context *)ctx, pixels, width, height, out);
    if (rc != 0) { free(out); return rc; }
    *out_pixels = out;
    return 0;
}
extern "C" void crispembed_scunet_free_image(uint8_t * pixels) {
    free(pixels);
}

// ---------------------------------------------------------------------------
// InstructIR all-in-one image restoration
// ---------------------------------------------------------------------------

#include "instructir.h"

extern "C" void * crispembed_instructir_init(const char * model_path, int n_threads) {
    return instructir_init(model_path, n_threads);
}
extern "C" void crispembed_instructir_free(void * ctx) {
    instructir_free((instructir_context *)ctx);
}
extern "C" int crispembed_instructir_n_tasks(const void * ctx) {
    return instructir_get_n_tasks((const instructir_context *)ctx);
}
extern "C" int crispembed_instructir_process(
        void * ctx, int task,
        const uint8_t * pixels, int width, int height,
        uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = instructir_process((instructir_context *)ctx, task, pixels, width, height, out);
    if (rc != 0) { free(out); return rc; }
    *out_pixels = out;
    return 0;
}
extern "C" void crispembed_instructir_free_image(uint8_t * pixels) {
    free(pixels);
}

// ---------------------------------------------------------------------------
// AdaIR all-in-one image restoration
// ---------------------------------------------------------------------------

#include "adair.h"

extern "C" void * crispembed_adair_init(const char * model_path, int n_threads) {
    return adair_init(model_path, n_threads);
}
extern "C" void crispembed_adair_free(void * ctx) {
    adair_free((adair_context *)ctx);
}
extern "C" int crispembed_adair_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = adair_process((adair_context *)ctx, pixels, width, height, out);
    if (rc != 0) { free(out); return rc; }
    *out_pixels = out;
    return 0;
}
extern "C" void crispembed_adair_free_image(uint8_t * pixels) {
    free(pixels);
}

// ---------------------------------------------------------------------------
// Punctuation restoration — FireRedPunc / PCS
// ---------------------------------------------------------------------------

#include "fireredpunc.h"
#include "pcs.h"

enum punct_type { PUNCT_FIREREDPUNC, PUNCT_PCS };

struct punct_wrapper {
    punct_type type;
    void * ctx;
    std::string result_buf;
};

extern "C" void * crispembed_punct_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    std::string arch = core_gguf::kv_str(meta, "general.architecture", "fireredpunc");
    core_gguf::free_metadata(meta);

    auto * w = new punct_wrapper();
    if (arch == "pcs" || arch == "pcs_xlmr") {
        w->type = PUNCT_PCS;
        w->ctx = pcs_init(model_path);
    } else {
        w->type = PUNCT_FIREREDPUNC;
        w->ctx = fireredpunc_init(model_path);
    }
    if (!w->ctx) { delete w; return nullptr; }
    return w;
}

extern "C" void crispembed_punct_free(void * ctx) {
    if (!ctx) return;
    auto * w = (punct_wrapper *)ctx;
    if (w->type == PUNCT_PCS) pcs_free((pcs_context *)w->ctx);
    else fireredpunc_free((fireredpunc_context *)w->ctx);
    delete w;
}

extern "C" const char * crispembed_punct_process(void * ctx, const char * text) {
    if (!ctx || !text) return text;
    auto * w = (punct_wrapper *)ctx;
    char * result = nullptr;
    if (w->type == PUNCT_PCS)
        result = pcs_process((pcs_context *)w->ctx, text);
    else
        result = fireredpunc_process((fireredpunc_context *)w->ctx, text);
    if (!result) return text;
    w->result_buf = result;
    free(result);
    return w->result_buf.c_str();
}

// ---------------------------------------------------------------------------
// OCR Result Renderers
// ---------------------------------------------------------------------------

#include "ocr_render.h"

extern "C" char * crispembed_ocr_render(
    const crispembed_ocr_result * results, int n_results,
    int page_width, int page_height,
    const char * format)
{
    if (!results || n_results <= 0 || !format) return nullptr;

    // Determine format
    ocr_render_format fmt = OCR_RENDER_TEXT;
    if (strcmp(format, "hocr") == 0) fmt = OCR_RENDER_HOCR;
    else if (strcmp(format, "alto") == 0) fmt = OCR_RENDER_ALTO;
    else if (strcmp(format, "pdf") == 0) fmt = OCR_RENDER_PDF;

    // Convert crispembed_ocr_result to ocr_render structures.
    // Each result becomes a single-word line (since we don't have
    // line-level grouping from the pipeline results).
    std::vector<ocr_render_word> words(n_results);
    std::vector<ocr_render_line> lines(n_results);
    for (int i = 0; i < n_results; i++) {
        words[i] = {results[i].text,
                    (int)results[i].x, (int)results[i].y,
                    (int)results[i].w, (int)results[i].h,
                    results[i].confidence};
        lines[i] = {&words[i], 1,
                    (int)results[i].x, (int)results[i].y,
                    (int)results[i].w, (int)results[i].h};
    }
    ocr_render_page page = {lines.data(), n_results,
                            page_width, page_height, nullptr};

    ocr_renderer * r = ocr_render_create(fmt);
    ocr_render_begin(r);
    ocr_render_add_page(r, &page);
    ocr_render_end(r);

    int size = ocr_render_output_size(r);
    char * out = (char *)malloc(size + 1);
    if (out) {
        memcpy(out, ocr_render_output(r), size);
        out[size] = '\0';
    }
    ocr_render_free(r);
    return out;
}

// ---------------------------------------------------------------------------
// Classical Preprocessing — C API wrappers
// ---------------------------------------------------------------------------

#include "dewarp.h"
#include "tps_warp.h"
#include "cc_detect.h"
#include "classical_preproc.h"
#include "pdf_info.h"

extern "C" int crispembed_pdf_page_dpi(
    const char * pdf_path, int page,
    float * out_dpi, int * out_n_images)
{
    pdf_page_dpi_result r = {};
    int ret = pdf_page_dpi(pdf_path, page, &r);
    if (out_dpi) *out_dpi = r.dpi;
    if (out_n_images) *out_n_images = r.n_images;
    return ret;
}

extern "C" int crispembed_dewarp(
    const uint8_t * gray, int w, int h,
    uint8_t * out, int * out_w, int * out_h)
{
    return dewarp_page(gray, w, h, out, out_w, out_h);
}

extern "C" int crispembed_tps_dewarp(
    const uint8_t * gray, int w, int h,
    const float * src_x, const float * src_y,
    const float * dst_x, const float * dst_y, int n,
    uint8_t * out)
{
    return tps_dewarp(gray, w, h, src_x, src_y, dst_x, dst_y, n, out);
}

extern "C" int crispembed_tps_auto_dewarp(
    const uint8_t * gray, int w, int h,
    const char * model_path,
    uint8_t * out)
{
    return tps_auto_dewarp(gray, w, h, model_path, out);
}

extern "C" crispembed_ocr_result * crispembed_cc_detect(
    const uint8_t * gray, int w, int h, int * out_n)
{
    int n = 0;
    cc_text_region * regions = cc_detect_lines(gray, w, h, &n);
    if (!regions || n <= 0) {
        if (out_n) *out_n = 0;
        cc_detect_free(regions);
        return nullptr;
    }
    // Convert to crispembed_ocr_result
    auto * results = (crispembed_ocr_result *)malloc(n * sizeof(crispembed_ocr_result));
    for (int i = 0; i < n; i++) {
        results[i].x = (float)regions[i].x;
        results[i].y = (float)regions[i].y;
        results[i].w = (float)regions[i].w;
        results[i].h = (float)regions[i].h;
        results[i].confidence = 1.0f;
        results[i].text = nullptr;
        results[i].text_len = 0;
    }
    cc_detect_free(regions);
    if (out_n) *out_n = n;
    return results;
}

extern "C" int crispembed_find_skew(
    const uint8_t * gray, int w, int h,
    float * angle, float * confidence)
{
    return find_skew_angle(gray, w, h, angle, confidence);
}

extern "C" void crispembed_adaptive_binarize(
    const uint8_t * gray, int w, int h, uint8_t * out)
{
    adaptive_otsu(gray, w, h, 0, 0, 0, out);
}

extern "C" void crispembed_background_norm(
    const uint8_t * gray, int w, int h, uint8_t * out)
{
    background_norm(gray, w, h, 0, 0, out);
}

extern "C" void crispembed_despeckle(
    const uint8_t * gray, int w, int h,
    int max_w, int max_h, uint8_t * out)
{
    despeckle_gray(gray, w, h, max_w, max_h, out);
}

// ---------------------------------------------------------------------------
// Table structure recognition
// ---------------------------------------------------------------------------

#include "table_parse.h"

extern "C" void * crispembed_table_parse_init(const char * ocr_model_path, int n_threads) {
    return table_parse_init(ocr_model_path, n_threads);
}

extern "C" void crispembed_table_parse_free(void * ctx) {
    table_parse_free((table_parse_context *)ctx);
}

extern "C" char * crispembed_table_parse_to_html(
        void * ctx, const uint8_t * gray, int width, int height) {
    return table_parse_to_html((table_parse_context *)ctx, gray, width, height);
}

extern "C" void crispembed_table_parse_free_string(char * str) {
    table_parse_free_string(str);
}

extern "C" int crispembed_table_parse_detect_grid(
        const uint8_t * gray, int width, int height,
        int * out_n_rows, int * out_n_cols) {
    return table_parse_detect_grid(gray, width, height, out_n_rows, out_n_cols);
}

// ---------------------------------------------------------------------------
// HAT super-resolution
// ---------------------------------------------------------------------------

#include "hat_sr.h"

extern "C" void * crispembed_hat_sr_init(const char * model_path, int n_threads) {
    return hat_sr_init(model_path, n_threads);
}
extern "C" void crispembed_hat_sr_free(void * ctx) {
    hat_sr_free((hat_sr_context *)ctx);
}
extern "C" int crispembed_hat_sr_scale(const void * ctx) {
    return hat_sr_scale((const hat_sr_context *)ctx);
}
extern "C" int crispembed_hat_sr_process(
        void * ctx, const uint8_t * pixels, int width, int height,
        int tile_size, int tile_overlap,
        uint8_t ** out_pixels, int * out_width, int * out_height) {
    return hat_sr_process((hat_sr_context *)ctx, pixels, width, height,
                          tile_size, tile_overlap, out_pixels, out_width, out_height);
}
extern "C" void crispembed_hat_sr_free_image(uint8_t * pixels) {
    hat_sr_free_image(pixels);
}
