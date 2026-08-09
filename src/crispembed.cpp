// crispembed.cpp — BERT/MiniLM encoder via ggml graph.

#include "crispembed.h"
#include "model_mgr.h"
#include "tokenizer.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "core/hparam_keys.h"
#include "core/imatrix_alias.h"
#include "core/init_bench.h"
#include "core/metal_pipeline_cache_policy.h"
#include "imatrix.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "ocr_pipeline.h"
#include "core/env_gate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    if (n < 0) {
        ret += half;
        n = -n;
    }
    int max_exact = half / 2;
    if (n < max_exact) {
        ret += n;
    } else {
        int val =
            max_exact + (int)(log((double)n / max_exact) / log((double)max_distance / max_exact) * (half - max_exact));
        if (val > half - 1) val = half - 1;
        ret += val;
    }
    return ret;
}

// Precompute MPNet relative position bias for sequence length T.
// rel_attn_bias: [n_buckets, n_heads] tensor
// Output: [n_heads, T, T] float array (row-major)
static std::vector<float> compute_rel_pos_bias(ggml_tensor * rel_attn_bias, int T, int n_heads, int n_buckets = 32) {
    // Read bias weights from tensor [n_buckets, n_heads]
    std::vector<float> bias_weights(n_buckets * n_heads);
    ggml_backend_tensor_get(rel_attn_bias, bias_weights.data(), 0, n_buckets * n_heads * sizeof(float));

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
    // T18: bound the ggml-metal MTLBinaryArchive open cost before the device is
    // created (see core/metal_pipeline_cache_policy.h — it was 680 of the 820 ms
    // fixed init here). CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB=0 restores the
    // pre-T18 behaviour. Skipped when `--gpu-backend cpu` already means no GPU
    // device will be created, so its diagnostic does not fire spuriously.
    {
        const std::string pref = crispasr_get_gpu_backend_pref();
        const bool pref_is_cpu = !pref.empty() && pref.size() <= 3 && ci_starts_with("cpu", pref.c_str());
        if (!pref_is_cpu) {
            core_metal_cache::apply();
        }
    }
    return crispasr_init_gpu_backend();
}

// ---------------------------------------------------------------------------
// Model structure
// ---------------------------------------------------------------------------

struct embed_layer {
    // Pre-attention LayerNorm
    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    // Attention Q/K/V/O
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    // Pre-merged QKV (in backend buffer — works on GPU)
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    // Post-attention LayerNorm
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    // FFN
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;    // SwiGLU gate (NomicBERT, separate)
    ggml_tensor * ffn_up_gate_w = nullptr; // Fused gate+up [2*inter, H] for ggml_geglu
    // MoE (Mixture of Experts) FFN — present on MoE layers only
    ggml_tensor * moe_gate_w = nullptr;   // Router: [H, N_experts]
    ggml_tensor * expert_fc1_w = nullptr; // Expert up: [H, inter, N_experts]
    ggml_tensor * expert_fc2_w = nullptr; // Expert down: [inter, H, N_experts]
    ggml_tensor * moe_ffn_bias = nullptr; // Output bias: [H]
};

struct embed_model {
    crispembed_hparams hparams;

    // Embeddings
    ggml_tensor * token_embd = nullptr; // [n_embd, n_vocab]
    ggml_tensor * pos_embd = nullptr;   // [n_embd, n_max_tokens]
    ggml_tensor * type_embd = nullptr;  // [n_embd, 2] (optional)
    ggml_tensor * embd_ln_w = nullptr;  // LayerNorm after embedding sum
    ggml_tensor * embd_ln_b = nullptr;
    ggml_tensor * rel_attn_bias = nullptr; // MPNet relative position bias [n_buckets, n_heads]
    ggml_tensor * rel_embd = nullptr;      // DeBERTa relative position embeddings [n_embd, max_rel_pos]
    ggml_tensor * encoder_ln_w = nullptr;  // DeBERTa encoder-level LayerNorm
    ggml_tensor * encoder_ln_b = nullptr;
    ggml_tensor * final_norm_w = nullptr; // ModernBERT final norm (pre-LN models)

    // Encoder layers
    std::vector<embed_layer> layers;

    // Optional pooler / projection
    ggml_tensor * pooler_w = nullptr;
    ggml_tensor * pooler_b = nullptr;

    // Sparse retrieval head (BGE-M3): Linear(n_embd, 1)
    ggml_tensor * sparse_linear_w = nullptr; // [H, 1]
    ggml_tensor * sparse_linear_b = nullptr; // [1], optional
    // SPLADE/MLM head: transform(H→H) + LN + decode(H→V) → sparse
    ggml_tensor * mlm_transform_w = nullptr; // [H, H]
    ggml_tensor * mlm_transform_b = nullptr; // [H]
    ggml_tensor * mlm_ln_w = nullptr;        // [H]
    ggml_tensor * mlm_ln_b = nullptr;        // [H]
    ggml_tensor * mlm_bias = nullptr;        // [V] (decoder bias; weight tied to token_embd)
    bool has_mlm_head = false;
    // ColBERT multi-vector head: Linear(n_embd, colbert_dim)
    ggml_tensor * colbert_linear_w = nullptr; // [H, colbert_dim]
    ggml_tensor * colbert_linear_b = nullptr; // [colbert_dim], optional
    // Reranker: 1-layer head Linear(H, 1)
    ggml_tensor * classifier_w = nullptr; // [1, H]
    ggml_tensor * classifier_b = nullptr; // [1]
    // Reranker: 2-layer RobertaClassificationHead (bge-reranker-v2-m3)
    ggml_tensor * classifier_dense_w = nullptr; // [H, H]
    ggml_tensor * classifier_dense_b = nullptr; // [H]
    ggml_tensor * classifier_out_w = nullptr;   // [1, H]
    ggml_tensor * classifier_out_b = nullptr;   // [1]
    bool classifier_2layer = false;

    bool has_sparse = false;
    bool has_colbert = false;
    bool is_reranker = false;
    int colbert_dim = 128;
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
#include "lfm2_embed.h"

struct crispembed_context {
    embed_model model;
    std::unique_ptr<dec_model> dec; // non-null for decoder models
    bool is_decoder = false;
    // C4 — cross-call prefix KV cache for the decoder-embedding path. Reuses a
    // shared instruction prefix's per-layer K/V + hidden across consecutive
    // encode() calls. prev_dec_tokens holds the previous call's ids (for LCP
    // detection). Opt out via CRISPEMBED_DECODER_PREFIX_CACHE=0.
    dec_prefix_cache dec_prefix;
    std::vector<int32_t> prev_dec_tokens;
    int prefix_cache_enabled = -1; // -1 = unresolved, 0 = off, 1 = on
    // LFM2.5 bidirectional embedding (arch="lfm2")
    lfm2_embed_ctx * lfm2_ctx = nullptr;
    bool is_lfm2 = false;
    WordPieceTokenizer wp_tokenizer;
    SentencePieceTokenizer sp_tokenizer;
    BPETokenizer bpe_tokenizer;
    bool use_sentencepiece = false;
    bool use_bpe = false;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 1;
    int pool_method = 0;            // 0=mean, 1=cls, 2=last-token
    int pos_offset = 0;             // position embedding offset (2 for RoBERTa/XLM-R)
    bool use_rope = false;          // encoder uses RoPE instead of absolute position embeddings (NomicBERT)
    float rope_theta = 10000.0f;    // default/sliding theta
    float rope_theta_global = 0.0f; // global attention theta (ModernBERT, 0 = same as rope_theta)
    int global_attn_every_n = 0;    // ModernBERT: every Nth layer uses global attention (0 = all same)
    int local_attention_window = 0; // ModernBERT: sliding-window size for local layers (0 = no window)
    bool pre_ln = false;            // pre-LN (ModernBERT) vs post-LN (BERT) ordering
    bool geglu_erf = false;         // gated FFN uses exact-erf gelu instead of the tanh approximation
    bool ffn_swiglu = false;        // gated FFN uses silu (SwiGLU) instead of any gelu flavour
    bool dump_layers = false;       // dump per-layer intermediates (CRISPEMBED_DUMP_LAYERS=1)
    int position_buckets = 0;       // DeBERTa log-bucket count (0 = linear positions)
    int matryoshka_dim = 0;         // 0 = use model default
    int image_deskew = 0;           // optional scan deskew on the file/RGB image paths
    float image_deskew_max_angle = 15.0f;
    int unk_id = -1;         // unknown-token id (for E6 UNK-ratio warning); -1 = not set
    bool unk_warned = false; // one-shot: suppress after first warning
    std::string prefix;      // prepended to text before tokenization (e.g. "query: ")
    // ColBERT self-describing metadata (read from GGUF, empty = not set)
    std::string colbert_query_prefix;
    std::string colbert_doc_prefix;
    std::string colbert_similarity_fn;
    int colbert_query_length = 0;
    std::vector<float> last_output;          // reused buffer (dense encode)
    std::vector<uint8_t> compute_meta;       // graph metadata buffer (no_alloc=true)
    ggml_context * qkv_ctx = nullptr;        // pre-merged QKV tensor metadata
    ggml_backend_buffer_t qkv_buf = nullptr; // backend buffer for merged QKV
    int reserved_T = 0;                      // scheduler reserved for this seq len
    // Sparse / colbert / reranker output buffers (valid until next call)
    std::vector<int32_t> last_sparse_indices;
    std::vector<float> last_sparse_values;
    std::vector<float> last_multivec;
    int last_multivec_n_tokens = 0;
    int last_multivec_dim = 0;
    // Per-token encoder embeddings (encode_tokens): raw final-hidden-state
    // output, L2-normalized, plus the token ids those vectors correspond
    // to. Valid until the next encode_tokens / encode_multivec / sparse /
    // dense encode call.
    std::vector<float> last_token_embeddings;
    std::vector<int32_t> last_token_ids;
    int last_token_n = 0;
    int last_token_dim = 0;
    // Per-mode scheduler reservation buckets
    int reserved_T_sparse = 0;
    int reserved_T_colbert = 0;
    int reserved_T_packed = 0; // packed block-diagonal batch graph (C3)
    // Reranker classifier weight cache (avoids 4MB GPU→CPU transfer per call)
    bool rerank_cache_valid = false;
    std::vector<float> rerank_dw; // dense_w [H*H]
    std::vector<float> rerank_db; // dense_b [H]
    std::vector<float> rerank_ow; // out_w [H]
    float rerank_out_bias = 0.0f;
    bool rerank_out_has_bias = false;
    std::vector<float> rerank_pw; // pooler_w [H*H] (DeBERTa)
    std::vector<float> rerank_pb; // pooler_b [H]
    bool rerank_has_pooler = false;
    // Audio path — opaque pointer into bidirlm_audio.cpp (lazily inited on
    // first crispembed_encode_audio call). Built only when CRISPEMBED_HAS_CRISP_AUDIO.
    void * audio_ctx = nullptr;
    std::string model_path_for_audio;
    // Vision path — opaque pointer into bidirlm_vision.cpp (lazily inited on
    // first encode_image call). Always compiled in (no sibling-lib dependency).
    void * vision_ctx = nullptr;
    int vision_load_attempted = 0;      // avoid re-loading after a failed open
    std::vector<float> last_vision_out; // owned buffer for the last encode_image* call
    int last_vision_dim = 0;
    int last_vision_n_merged = 0;
    int last_vision_n_deepstack = 0;
    // LoRA adapter name cache for list_lora API
    std::vector<std::string> lora_name_strings;
    std::vector<const char *> lora_name_ptrs;
    bool bench = false;
};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

// `pre_g` (optional): a gguf_context the CALLER already parsed for this exact
// path. crispembed_init() has to open the GGUF once anyway to tell an encoder
// from a decoder model, and re-parsing it here cost a measured 29 ms on
// multilingual-e5-small (250k-entry vocab KV array) — 23% of the whole CPU-path
// init (T18). When pre_g is given we borrow it and the caller keeps ownership.
// CRISPEMBED_GGUF_REPARSE=1 restores the pre-T18 second parse for A/B.
static bool load_model(crispembed_context * ctx, const char * path, gguf_context * pre_g = nullptr) {
    auto & m = ctx->model;
    auto & hp = m.hparams;
    core_initbench::timer ib("load_model");

    // Load GGUF metadata first
    gguf_init_params gp = { true, nullptr };
    if (const char * rp = std::getenv("CRISPEMBED_GGUF_REPARSE"); rp && rp[0] && std::strcmp(rp, "0") != 0) {
        pre_g = nullptr;
    }
    const bool own_g = (pre_g == nullptr);
    gguf_context * g = own_g ? gguf_init_from_file(path, gp) : pre_g;
    ib.mark(own_g ? "gguf_init_from_file" : "gguf_reuse_caller_parse");
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

    // Hyperparams — CrispEmbed (bert.hidden_size, ...), Ollama/llama.cpp
    // ({arch}.embedding_length, ...), plus the GGUF's OWN declared architecture.
    //
    // llama.cpp/Ollama always write <general.architecture>.<field>, so deriving
    // the prefix from general.architecture resolves any community GGUF — e.g.
    // nomic-embed-text-v2-moe's "nomic-bert-moe.*" keys (issue #33) — without a
    // per-model alias list. See src/core/hparam_keys.h.
    //
    // Gates: CRISPEMBED_ARCH_HPARAMS=0 disables the arch-derived candidates
    // (leaving exactly the legacy bert.*/xlmr.* behaviour, for A/B);
    // CRISPEMBED_STRICT_HPARAMS=1 hard-fails instead of silently defaulting.
    const std::string gguf_arch = strv("general.architecture");
    const bool arch_hp_on = core_hparams::arch_keys_enabled();
    auto ak = [&](const char * field) { return core_hparams::arch_key(gguf_arch, field, arch_hp_on); };

    auto look_u32 = [&](const std::string & key, int & v) -> bool {
        const int64_t k = gguf_find_key(g, key.c_str());
        if (k < 0) return false;
        v = (int)gguf_get_val_u32(g, k);
        return true;
    };
    auto look_f32 = [&](const std::string & key, float & v) -> bool {
        const int64_t k = gguf_find_key(g, key.c_str());
        if (k < 0) return false;
        v = gguf_get_val_f32(g, k);
        return true;
    };

    // Required hparams: a wrong default here silently yields a garbage embedding,
    // so record misses for the strict-mode report.
    std::vector<std::string> missing_hp;
    auto req_u32 = [&](const std::vector<std::string> & keys, int def, const char * what) -> int {
        int v = def;
        if (!core_hparams::resolve(look_u32, keys, v)) missing_hp.push_back(what);
        return v;
    };
    auto opt_u32 = [&](const std::vector<std::string> & keys, int def) -> int {
        int v = def;
        core_hparams::resolve(look_u32, keys, v);
        return v;
    };
    auto opt_f32 = [&](const std::vector<std::string> & keys, float def) -> float {
        float v = def;
        core_hparams::resolve(look_f32, keys, v);
        return v;
    };

    hp.n_vocab = opt_u32({ "bert.vocab_size", ak("vocab_size") }, 30522);
    hp.n_max_tokens = opt_u32(
        { "bert.max_position_embeddings", "bert.context_length", "xlmr.context_length", ak("context_length") }, 512);
    hp.n_embd = req_u32({ "bert.hidden_size", "bert.embedding_length", "xlmr.embedding_length", ak("embedding_length"),
                          ak("hidden_size") },
                        384, "embedding_length");
    hp.n_head = req_u32({ "bert.num_attention_heads", "bert.attention.head_count", "xlmr.attention.head_count",
                          ak("attention.head_count") },
                        12, "attention.head_count");
    hp.n_layer = req_u32({ "bert.num_hidden_layers", "bert.block_count", "xlmr.block_count", ak("block_count") }, 6,
                         "block_count");
    hp.n_intermediate = req_u32(
        { "bert.intermediate_size", "bert.feed_forward_length", "xlmr.feed_forward_length", ak("feed_forward_length") },
        1536, "feed_forward_length");
    hp.n_output = opt_u32({ "bert.output_dim" }, hp.n_embd);
    hp.layer_norm_eps = opt_f32({ "bert.layer_norm_eps", "bert.attention.layer_norm_epsilon",
                                  "xlmr.attention.layer_norm_epsilon", ak("attention.layer_norm_epsilon") },
                                1e-12f);

    // Pooling method: 0=mean (default), 1=cls, 2=last-token
    // CrispEmbed format: bert.pooling_method (0=mean, 1=cls, 2=last)
    // Ollama format:     bert.pooling_type   (0=none, 1=mean, 2=cls, 3=last)
    {
        int pm = u32("bert.pooling_method", -1);
        if (pm < 0) {
            // Try Ollama format and convert: Ollama{1=mean,2=cls,3=last} → CE{0,1,2}
            // Arch-derived key covers any community GGUF (nomic-bert-moe → 1=mean).
            const int pt = opt_u32({ "bert.pooling_type", "xlmr.pooling_type", ak("pooling_type") }, -1);
            if (pt > 0)
                pm = pt - 1; // Ollama 1→0(mean), 2→1(cls), 3→2(last)
            else
                pm = 0; // default mean
        }
        ctx->pool_method = pm;
    }
    // Position embedding offset: 0 for BERT, 2 for RoBERTa/XLM-R
    ctx->pos_offset = u32("bert.position_offset", u32("xlmr.position_offset", 0));
    // ColBERT output dimension (BGE-M3 default 128) — read while g is valid
    m.colbert_dim = u32("bert.colbert_dim", 128);
    // ColBERT self-describing metadata (from config_sentence_transformers.json)
    ctx->colbert_query_prefix = strv("colbert.query_prefix");
    ctx->colbert_doc_prefix = strv("colbert.document_prefix");
    ctx->colbert_similarity_fn = strv("colbert.similarity_fn_name");
    ctx->colbert_query_length = u32("colbert.query_length", 0);
    // RoPE and pre-LN flags — MUST be read before gguf_free(g)
    // Community GGUFs write the RoPE base as <arch>.rope.freq_base.
    ctx->rope_theta = opt_f32({ "bert.rope_theta", ak("rope.freq_base") }, 10000.0f);
    ctx->rope_theta_global = f32("bert.rope_theta_global", 0.0f);
    ctx->global_attn_every_n = u32("bert.global_attn_every_n", 0);
    ctx->local_attention_window = u32("bert.local_attention", 0);
    ctx->pre_ln = u32("bert.pre_ln", 0) != 0;
    ctx->position_buckets = u32("bert.position_buckets", 0);

    // Community `modern-bert` GGUFs (llama.cpp arch) name their metadata
    // differently from CrispEmbed's own bert.* keys, and their RoPE theta is
    // INVERTED vs our naming: `rope.freq_base` is the GLOBAL theta, while
    // `rope.freq_base_swa` is the LOCAL (sliding-window) theta. The generic
    // ak("rope.freq_base") read above therefore loaded the GLOBAL base into
    // rope_theta — correct it here. ModernBERT is architecturally pre-LN and
    // uses exact-erf GeGLU (see the graph builder). Layer 0's attn norm is
    // Identity in HF so ln1 is per-layer optional (guarded downstream).
    if (gguf_arch == "modern-bert") {
        ctx->rope_theta = opt_f32({ ak("rope.freq_base_swa") }, 10000.0f);     // local / sliding
        ctx->rope_theta_global = opt_f32({ ak("rope.freq_base") }, 160000.0f); // global
        ctx->global_attn_every_n = opt_u32({ ak("attention.sliding_window_pattern") }, 3);
        ctx->local_attention_window = opt_u32({ ak("attention.sliding_window") }, 128);
        ctx->pre_ln = true;
        ctx->geglu_erf = true;
    }

    // Self-describing gated-FFN activation (`bert.ffn_act`, written by
    // convert-bert-to-gguf.py from config.hidden_act / hidden_activation).
    // Absent = keep the historical per-arch default set above, so already
    // published GGUFs are byte-for-byte unaffected.
    {
        const std::string ffn_act = strv("bert.ffn_act");
        if (ffn_act == "silu" || ffn_act == "swish") {
            ctx->ffn_swiglu = true;
        } else if (ffn_act == "gelu") {
            ctx->geglu_erf = true; // HF ACT2FN["gelu"] is the exact erf GELU
        } else if (ffn_act == "gelu_pytorch_tanh" || ffn_act == "gelu_new") {
            ctx->geglu_erf = false;
        }
    }

    hp.n_experts = opt_u32({ "bert.num_experts", ak("expert_count") }, 0);
    hp.n_experts_per_tok = opt_u32({ "bert.num_experts_per_tok", ak("expert_used_count") }, 0);

    // Strict mode: a missing REQUIRED hparam means we are about to run with a
    // fabricated default (384-dim / 6-layer / ...) and emit a silently-garbage
    // embedding with exit code 0. Opt-in hard-fail instead (see hparam_keys.h).
    if (!missing_hp.empty()) {
        std::string joined;
        for (size_t i = 0; i < missing_hp.size(); i++) joined += (i ? ", " : "") + missing_hp[i];
        if (core_hparams::strict_hparams_enabled()) {
            fprintf(stderr,
                    "crispembed: missing required hyperparameter(s) [%s] for architecture '%s' — refusing to load "
                    "with fabricated defaults (CRISPEMBED_STRICT_HPARAMS=1). Unset it to load anyway.\n",
                    joined.c_str(), gguf_arch.empty() ? "(unset)" : gguf_arch.c_str());
            if (own_g) gguf_free(g);
            return false;
        }
        fprintf(stderr,
                "crispembed: warning: hyperparameter(s) [%s] not found for architecture '%s' — using defaults; "
                "embeddings may be wrong. Set CRISPEMBED_STRICT_HPARAMS=1 to make this fatal.\n",
                joined.c_str(), gguf_arch.empty() ? "(unset)" : gguf_arch.c_str());
    }

    // BPE merges may live in the `tokenizer.ggml.merges` KV STRING ARRAY
    // (community gpt2/modern-bert GGUFs) instead of the `tokenizer.merges`
    // TENSOR (CrispEmbed's own converter). Read the KV array here while `g`
    // is live (gguf_free use-after-free landmine); it is consumed after weight
    // loading, only if the tensor form is absent.
    std::vector<std::string> kv_merges;
    {
        const int64_t mki = gguf_find_key(g, "tokenizer.ggml.merges");
        if (mki >= 0 && gguf_get_arr_type(g, mki) == GGUF_TYPE_STRING) {
            const int nm = (int)gguf_get_arr_n(g, mki);
            kv_merges.resize(nm);
            for (int i = 0; i < nm; i++) kv_merges[i] = gguf_get_arr_str(g, mki, i);
        }
    }

    ib.mark("hparams+kv_merges");

    // Load tokenizer vocab from GGUF metadata
    const int64_t ki = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (ki >= 0) {
        const int n = (int)gguf_get_arr_n(g, ki);
        std::vector<std::string> vocab(n);
        for (int i = 0; i < n; i++) vocab[i] = gguf_get_arr_str(g, ki, i);
        ib.mark("vocab_read");

        // Load scores if available (SentencePiece models)
        std::vector<float> scores;
        const int64_t si = gguf_find_key(g, "tokenizer.ggml.scores");
        if (si >= 0 && gguf_get_arr_type(g, si) == GGUF_TYPE_FLOAT32) {
            int sn = (int)gguf_get_arr_n(g, si);
            scores.resize(sn);
            const float * sd = reinterpret_cast<const float *>(gguf_get_arr_data(g, si));
            std::memcpy(scores.data(), sd, sn * sizeof(float));
        }

        // Detect tokenizer type: 0=WordPiece, 1=BPE, 2=SentencePiece.
        // CrispEmbed's own GGUFs write the numeric `tokenizer.ggml.type`,
        // which when present is FINAL — an explicit WordPiece (0) is honoured
        // even for a >100k vocab (LaBSE, 501k; before this the legacy
        // heuristic below routed it into the SPM tokenizer, which wrapped
        // with bos=0/eos=2 instead of [CLS]/[SEP] and emitted the literal
        // "▁" vocab token for every space — 0/20 on the HF id battery).
        // Community/llama.cpp GGUFs instead write the STRING `tokenizer.ggml.model`
        // (gpt2 / bert / t5 / llama / unigram) with NO numeric type — for those the
        // model string is AUTHORITATIVE over the old vocab-size heuristic. Without
        // this a gpt2/modern-bert BPE GGUF (50368 vocab, no type) fell through to
        // WordPiece and produced garbage embeddings from token 0.
        // The decision table lives in resolve_tokenizer_family (tokenizer.h),
        // hermetically tested by tests/test_bert_pretokenize.cpp.
        const int tokenizer_type =
            resolve_tokenizer_family(gguf_find_key(g, "tokenizer.ggml.type") >= 0, (int)u32("tokenizer.ggml.type", 0),
                                     strv("tokenizer.ggml.model"), n);
        // C2 behavior flags (llama.cpp convention, BOOL-typed; absent or
        // non-BOOL → default true = the historical wrap behavior, so every
        // shipped GGUF is byte-identical). Read while `g` is live (gguf_free
        // use-after-free landmine).
        const bool tok_add_bos = core_gguf::kv_bool(g, "tokenizer.ggml.add_bos_token", true);
        const bool tok_add_eos = core_gguf::kv_bool(g, "tokenizer.ggml.add_eos_token", true);

        if (tokenizer_type == 2) {
            // SentencePiece / XLM-RoBERTa
            int bos_id = u32("tokenizer.ggml.bos_token_id", 0);
            int eos_id = u32("tokenizer.ggml.eos_token_id", 2);
            int unk_id = u32("tokenizer.ggml.unknown_token_id", 3);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 1);
            ctx->sp_tokenizer.load(vocab, scores, bos_id, eos_id, unk_id, pad_id, hp.n_max_tokens);
            ctx->unk_id = unk_id;
            ctx->sp_tokenizer.set_add_flags(tok_add_bos, tok_add_eos);
            // HF's `Precompiled` (nmt_nfkc charsmap) normalizer, which every
            // XLM-R-family Unigram embedder declares and which we implemented
            // nowhere: `…` tokenized to three <unk> instead of `...`, and
            // every fullwidth form / U+3000 went through unnormalized. The
            // charsmap is byte-identical across all six shipped multilingual
            // embedders, so one table serves them and no GGUF needs
            // re-converting. Measured in tests/embed_tokenizer_parity.py;
            // CRISPEMBED_SPM_HF_NORM=0 restores the historical path.
            ctx->sp_tokenizer.set_hf_normalize(true);
            ctx->use_sentencepiece = true;
            fprintf(stderr, "crispembed: using SentencePiece tokenizer (%d tokens, %zu scores)\n", n, scores.size());
        } else if (tokenizer_type == 1) {
            // BPE (GPT-2 style, ModernBERT, etc.). Community gpt2 encoder GGUFs
            // (e.g. modern-bert) declare CLS/SEP via bos/eos_token_id, not the
            // cls/sep keys — fall back to bos/eos so the [CLS]…[SEP] wrap is right.
            int cls_id = u32("tokenizer.ggml.cls_token_id", u32("tokenizer.ggml.bos_token_id", 0));
            int sep_id = u32("tokenizer.ggml.sep_token_id", u32("tokenizer.ggml.eos_token_id", 2));
            int pad_id = u32("tokenizer.ggml.padding_token_id", 1);
            // add_bos/add_eos=false disable the CLS/SEP wrap via the -1 id
            // convention (encode() only wraps ids that are >= 0); the merges
            // reload below re-reads bos_id()/eos_id(), so this persists.
            if (!tok_add_bos) cls_id = -1;
            if (!tok_add_eos) sep_id = -1;

            // BPE merges stored as tensor (newline blob) OR the tokenizer.ggml.merges
            // KV array (kv_merges, read above) — applied after weight loading.
            std::vector<std::string> empty_merges;
            // A BPE vocab whose tokenizer.json declares the SentencePiece
            // normalizer (space → ▁) is a SPM-BPE, not a byte-level one. The
            // converter says so with `tokenizer.ggml.is_spm_bpe` (the same key
            // the decoder-embedder path already reads); an ABSENT key keeps the
            // historical byte-level behavior, so every published GGUF is
            // unaffected. granite-embedding-311m-multilingual-r2 needs it.
            const bool is_spm_bpe = u32("tokenizer.ggml.is_spm_bpe", 0) != 0;
            // For encoder BPE: eos=SEP, suffix=-1 (handled by encode), bos=CLS
            ctx->bpe_tokenizer.load(vocab, empty_merges, sep_id, pad_id, -1, cls_id, is_spm_bpe, hp.n_max_tokens);
            ctx->use_bpe = true;
            // Pre-tokenizer selection, self-described by `tokenizer.ggml.pre`:
            // ModernBERT tokenizes with the GPT-2 ByteLevel regex pre-tokenizer
            // ("modern-bert"); granite-embedding-97m-multilingual-r2 with the
            // o200k_base split ("o200k"). The default whitespace-split
            // pre-tokenizer mis-splits punctuation/digits for both.
            const std::string tok_pre = strv("tokenizer.ggml.pre");
            if (tok_pre == "o200k")
                ctx->bpe_tokenizer.set_o200k_regex_pretok(true,
                                                          core_gguf::kv_bool(g, "tokenizer.ggml.ignore_merges", true));
            else if (gguf_arch == "modern-bert" || tok_pre == "modern-bert")
                ctx->bpe_tokenizer.set_gpt2_regex_pretok(true);
            fprintf(stderr, "crispembed: using %s BPE tokenizer (%d tokens, pre=%s)\n",
                    is_spm_bpe ? "SentencePiece" : "GPT-2", n, tok_pre.empty() ? "default" : tok_pre.c_str());
        } else {
            // WordPiece / BERT
            int cls_id = u32("tokenizer.ggml.cls_token_id", 101);
            int sep_id = u32("tokenizer.ggml.sep_token_id", 102);
            int unk_id = u32("tokenizer.ggml.unknown_token_id", 100);
            int pad_id = u32("tokenizer.ggml.padding_token_id", 0);
            // Detect casing: if vocab contains uppercase letters like "A", it's cased
            bool do_lower_case = true;
            for (const auto & t : vocab) {
                if (t.size() == 1 && t[0] >= 'A' && t[0] <= 'Z') {
                    do_lower_case = false;
                    break;
                }
            }
            ctx->wp_tokenizer.load(vocab, cls_id, sep_id, unk_id, pad_id, hp.n_max_tokens, do_lower_case);
            ctx->unk_id = unk_id;
            // `tokenizer.ggml.pre = "bert"` selects the HF BertNormalizer +
            // BertPreTokenizer path (core/bert_pretok.h) — written by the
            // converter when tokenizer.json itself declares it (LaBSE class).
            // ABSENT key = the historical per-byte splitter, so every shipped
            // WordPiece GGUF tokenizes byte-identically.
            const bool bert_pre = strv("tokenizer.ggml.pre") == "bert";
            ctx->wp_tokenizer.set_bert_pretok(bert_pre);
            fprintf(stderr, "crispembed: using WordPiece tokenizer (%d tokens, %s%s)\n", n,
                    do_lower_case ? "uncased" : "cased", bert_pre ? ", pre=bert" : "");
        }
        ib.mark("tokenizer_build");
    }

    if (own_g) gguf_free(g);
    g = nullptr; // borrowed context stays alive in the caller; do not touch it again
    ib.mark("gguf_free");

    // Initialize backends: try GPU first, CPU always as fallback
    ctx->backend = crispembed_init_backend(ctx->n_threads);
    ib.mark("backend_init");
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
        fprintf(stderr, "crispembed: using %s backend with CPU fallback\n", ggml_backend_name(ctx->backend));
    } else {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        fprintf(stderr, "crispembed: using CPU backend (%d threads)\n", ctx->n_threads);
    }

    // Create scheduler for graph dispatch (handles GPU/CPU allocation)
    int graph_nodes = 16384;
    ctx->sched =
        ggml_backend_sched_new(ctx->backends.data(), nullptr, (int)ctx->backends.size(), graph_nodes, false, false);
    crispembed_imatrix_install(ctx->sched);

    // Allocate metadata buffer for graph building (no_alloc=true pattern)
    ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false));
    ib.mark("sched+meta");

    if (!core_gguf::load_weights(path, ctx->backend, "crispembed", ctx->wl)) {
        fprintf(stderr, "crispembed: failed to load weights\n");
        return false;
    }
    ib.mark("weights_load");

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
    m.pos_embd = get("position_embd.weight");
    m.type_embd = get_any({ "token_type_embd.weight", "token_types.weight" });
    m.embd_ln_w = get_any({ "embd_ln.weight", "token_embd_norm.weight" });
    m.embd_ln_b = get_any({ "embd_ln.bias", "token_embd_norm.bias" });
    m.rel_attn_bias = get("rel_attn_bias.weight");
    m.rel_embd = get("rel_embd.weight");
    m.encoder_ln_w = get("encoder_ln.weight");
    m.encoder_ln_b = get("encoder_ln.bias");
    m.final_norm_w = get_any({ "final_norm.weight", "output_norm.weight" });

    if (!m.token_embd) {
        fprintf(stderr, "crispembed: missing token_embd.weight\n");
        return false;
    }
    // Infer hparams from tensor shapes when metadata was missing (Ollama format).
    // token_embd.weight is [n_embd, n_vocab].
    {
        int64_t tensor_vocab = m.token_embd->ne[1];
        int64_t tensor_embd = m.token_embd->ne[0];
        if (tensor_vocab > 0 && tensor_vocab != hp.n_vocab) {
            hp.n_vocab = (int)tensor_vocab;
        }
        if (tensor_embd > 0 && tensor_embd != hp.n_embd) {
            hp.n_embd = (int)tensor_embd;
            hp.n_output = hp.n_embd;
        }
        // Count actual encoder layers from loaded tensors
        int counted = 0;
        for (const auto & kv : ctx->wl.tensors) {
            // Match enc.N. or blk.N. prefix
            const auto & name = kv.first;
            int layer_id = -1;
            if (sscanf(name.c_str(), "enc.%d.", &layer_id) == 1 || sscanf(name.c_str(), "blk.%d.", &layer_id) == 1) {
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
        fprintf(stderr, "crispembed: no position embeddings, using RoPE (theta=%.0f%s)\n", ctx->rope_theta,
                ctx->pre_ln ? ", pre-LN" : "");
    } else if (!m.pos_embd && m.rel_embd) {
        fprintf(stderr, "crispembed: DeBERTa disentangled relative-position attention\n");
    }

    // Encoder layers
    m.layers.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; il++) {
        auto pfx = "enc." + std::to_string(il) + ".";
        auto blk = "blk." + std::to_string(il) + ".";
        auto & L = m.layers[il];
        L.ln1_w = get_any({ pfx + "ln1.weight", blk + "attn_output_norm.weight", blk + "attn_norm.weight" });
        L.ln1_b = get_any({ pfx + "ln1.bias", blk + "attn_output_norm.bias", blk + "attn_norm.bias" });
        // Pre-fused QKV (nomic-bert-moe exports blk.N.attn_qkv.weight, rows q|k|v,
        // possibly quantized). Consumed directly by the qkv graph path (issue #33).
        L.qkv_w = get_any({ pfx + "attn.qkv.weight", blk + "attn_qkv.weight" });
        L.qkv_b = get_any({ pfx + "attn.qkv.bias", blk + "attn_qkv.bias" });
        L.q_w = get_any({ pfx + "attn.q.weight", blk + "attn_q.weight" });
        L.q_b = get_any({ pfx + "attn.q.bias", blk + "attn_q.bias" });
        L.k_w = get_any({ pfx + "attn.k.weight", blk + "attn_k.weight" });
        L.k_b = get_any({ pfx + "attn.k.bias", blk + "attn_k.bias" });
        L.v_w = get_any({ pfx + "attn.v.weight", blk + "attn_v.weight" });
        L.v_b = get_any({ pfx + "attn.v.bias", blk + "attn_v.bias" });
        L.o_w = get_any({ pfx + "attn.o.weight", blk + "attn_output.weight" });
        L.o_b = get_any({ pfx + "attn.o.bias", blk + "attn_output.bias" });
        L.ln2_w = get_any({ pfx + "ln2.weight", blk + "layer_output_norm.weight", blk + "ffn_norm.weight" });
        L.ln2_b = get_any({ pfx + "ln2.bias", blk + "layer_output_norm.bias", blk + "ffn_norm.bias" });
        L.fc1_w = get_any({ pfx + "ffn.fc1.weight", blk + "ffn_up.weight" });
        L.fc1_b = get_any({ pfx + "ffn.fc1.bias", blk + "ffn_up.bias" });
        L.fc2_w = get_any({ pfx + "ffn.fc2.weight", blk + "ffn_down.weight" });
        L.fc2_b = get_any({ pfx + "ffn.fc2.bias", blk + "ffn_down.bias" });
        L.ffn_gate_w = get_any({ pfx + "ffn_gate.weight", blk + "ffn_gate.weight" }); // SwiGLU gate (NomicBERT)
        L.ffn_up_gate_w =
            get_any({ pfx + "ffn_up_gate.weight", blk + "ffn_up_gate.weight" }); // Fused gate+up (ModernBERT/GTE v1.5)
        // Community modern-bert GGUFs name the FUSED GeGLU weight `blk.N.ffn_up`
        // (same name as a PLAIN up-proj) — detect it by shape: [H, 2*inter]
        // instead of [H, inter]. Route it to ffn_up_gate_w so the graph takes the
        // GeGLU path, and drop the plain fc1 so it isn't double-used.
        if (!L.ffn_up_gate_w && L.fc1_w && L.fc1_w->ne[1] == 2 * (int64_t)hp.n_intermediate) {
            L.ffn_up_gate_w = L.fc1_w;
            L.fc1_w = nullptr;
            L.fc1_b = nullptr;
        }
        // MoE expert tensors (present only on MoE layers). nomic-bert-moe uses the
        // standard llama.cpp names: router ffn_gate_inp, stacked experts
        // ffn_up_exps [H,inter,n_exp] / ffn_down_exps [inter,H,n_exp] (issue #33).
        L.moe_gate_w = get_any({ pfx + "ffn.moe_gate.weight", blk + "ffn_gate_inp.weight" });
        L.expert_fc1_w = get_any({ pfx + "ffn.expert_fc1.weight", blk + "ffn_up_exps.weight" });
        L.expert_fc2_w = get_any({ pfx + "ffn.expert_fc2.weight", blk + "ffn_down_exps.weight" });
        L.moe_ffn_bias = get(pfx + "ffn.moe_bias");
    }

    // Pooler (optional)
    m.pooler_w = get("pooler.weight");
    m.pooler_b = get("pooler.bias");

    // Optional sparse / colbert / classifier heads
    m.sparse_linear_w = get("sparse_linear.weight");
    m.sparse_linear_b = get("sparse_linear.bias");
    m.colbert_linear_w = get("colbert_linear.weight");
    m.colbert_linear_b = get("colbert_linear.bias");
    // Try 2-layer RobertaClassificationHead first (bge-reranker-v2-m3)
    m.classifier_dense_w = get("classifier.dense.weight");
    m.classifier_dense_b = get("classifier.dense.bias");
    m.classifier_out_w = get("classifier.out_proj.weight");
    m.classifier_out_b = get("classifier.out_proj.bias");
    if (m.classifier_dense_w && m.classifier_out_w) {
        m.classifier_2layer = true;
        m.is_reranker = true;
    } else {
        // Fall back to 1-layer head
        m.classifier_w = get("classifier.weight");
        m.classifier_b = get("classifier.bias");
        m.is_reranker = m.classifier_w != nullptr;
    }
    // SPLADE/MLM head
    m.mlm_transform_w = get("mlm_transform.weight");
    m.mlm_transform_b = get("mlm_transform.bias");
    m.mlm_ln_w = get("mlm_ln.weight");
    m.mlm_ln_b = get("mlm_ln.bias");
    m.mlm_bias = get("mlm_bias");
    m.has_mlm_head = m.mlm_transform_w != nullptr;
    if (m.has_mlm_head) fprintf(stderr, "crispembed: MLM/SPLADE head loaded\n");

    m.has_sparse = m.sparse_linear_w != nullptr || m.has_mlm_head;
    m.has_colbert = m.colbert_linear_w != nullptr;
    if (m.has_sparse) fprintf(stderr, "crispembed: sparse head loaded\n");
    if (m.has_colbert) fprintf(stderr, "crispembed: colbert head loaded (dim=%d)\n", m.colbert_dim);
    if (m.is_reranker)
        fprintf(stderr, "crispembed: classifier head loaded (reranker=%s)\n",
                m.classifier_2layer ? "2-layer" : "1-layer");
    if (hp.n_experts > 0) {
        int moe_count = 0;
        for (int i = 0; i < hp.n_layer; i++)
            if (m.layers[i].moe_gate_w) moe_count++;
        fprintf(stderr, "crispembed: MoE encoder (%d experts, top-%d, %d/%d MoE layers)\n", hp.n_experts,
                hp.n_experts_per_tok, moe_count, hp.n_layer);
    }

    // Pre-merge QKV weights into backend buffer (works on CPU + GPU)
    {
        const int H = hp.n_embd;
        size_t qkv_mem = hp.n_layer * 2 * ggml_tensor_overhead() + 1024;
        ggml_init_params qkv_ip = { qkv_mem, nullptr, true }; // no_alloc
        ctx->qkv_ctx = ggml_init(qkv_ip);

        for (int i = 0; i < hp.n_layer; i++) {
            auto & L = m.layers[i];
            if (!L.q_w || !L.k_w || !L.v_w) continue;
            if (L.q_w->type != GGML_TYPE_F32) continue; // skip quantized
            L.qkv_w = ggml_new_tensor_2d(ctx->qkv_ctx, GGML_TYPE_F32, H, 3 * H);
            // Name the merged tensor so the imatrix collector files its matmul
            // statistics under a stable per-layer key instead of ggml's auto
            // "leaf_N" (which matches nothing at quantize time). The matmul
            // input width equals the shared QKV input (n_embd), so the
            // collected vector applies verbatim to attn.{q,k,v}.weight. The
            // name contract lives in core/imatrix_alias.h, shared with the
            // quantizer's alias lookup and guarded by tests/test_imatrix_alias.cpp.
            ggml_set_name(L.qkv_w, core_imatrix::qkv_merged_name(i).c_str());
            if (L.q_b && L.k_b && L.v_b) L.qkv_b = ggml_new_tensor_1d(ctx->qkv_ctx, GGML_TYPE_F32, 3 * H);
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

    // Load BPE merges. Two on-disk forms: CrispEmbed's converter writes the
    // `tokenizer.merges` TENSOR (newline-separated UTF-8 blob); community
    // gpt2/modern-bert GGUFs write the `tokenizer.ggml.merges` KV STRING ARRAY
    // (kv_merges, read above while `g` was live). Prefer the tensor, fall back
    // to the KV array. The reload preserves the gpt2-regex pre-tokenizer flag
    // (set_gpt2_regex_pretok is not a load() parameter, so it survives).
    if (ctx->use_bpe) {
        std::vector<std::string> merges;
        const char * src = nullptr;
        // `tokenizer.merges_nul` is the NUL-separated blob. It exists only for
        // vocabs whose merges contain newlines (SentencePiece-BPE), which the
        // newline-separated `tokenizer.merges` cannot encode; prefer it when
        // present, otherwise the historical tensor, so published GGUFs that
        // lack the key behave exactly as before.
        ggml_tensor * merge_t = get("tokenizer.merges_nul");
        const char sep = merge_t ? '\0' : '\n';
        if (!merge_t) merge_t = get("tokenizer.merges");
        if (merge_t) {
            size_t nbytes = ggml_nbytes(merge_t);
            std::vector<uint8_t> blob(nbytes);
            ggml_backend_tensor_get(merge_t, blob.data(), 0, nbytes);
            std::string current;
            for (size_t i = 0; i < nbytes; i++) {
                if ((char)blob[i] == sep) {
                    if (!current.empty()) merges.push_back(current);
                    current.clear();
                } else {
                    current += (char)blob[i];
                }
            }
            if (!current.empty()) merges.push_back(current);
            src = sep == '\0' ? "tensor (NUL-separated)" : "tensor";
        } else if (!kv_merges.empty()) {
            merges = std::move(kv_merges);
            src = "KV array";
        }
        if (!merges.empty()) {
            // Re-load BPE tokenizer with merges (preserve suffix_id=-1 for encoder)
            int cls_id = ctx->bpe_tokenizer.bos_id();
            int sep_id = ctx->bpe_tokenizer.eos_id();
            int pad_id = ctx->bpe_tokenizer.pad_id();
            // load() resets spm_style, so re-apply it after the reload (the
            // pre-tokenizer flags are setters and survive on their own).
            const bool is_spm_bpe = ctx->bpe_tokenizer.spm_style();
            ctx->bpe_tokenizer.load(ctx->bpe_tokenizer.get_vocab(), merges, sep_id, pad_id, -1, cls_id, is_spm_bpe,
                                    hp.n_max_tokens);
            fprintf(stderr, "crispembed: loaded %zu BPE merges from %s\n", merges.size(), src);
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

// ModernBERT local sliding-window attention active? (model has a window + alternating
// global layers, and not disabled for A/B via CRISPEMBED_ENCODER_NO_SWA). When off,
// local layers fall back to global attention — the pre-fix behavior, kept as a
// regression-bisection lever.
static bool modernbert_swa_enabled(const crispembed_context * ctx) {
    if (ctx->local_attention_window <= 0 || ctx->global_attn_every_n <= 0) return false;
    const char * v = std::getenv("CRISPEMBED_ENCODER_NO_SWA");
    if (v && v[0] && std::strcmp(v, "0") != 0) return false; // opt-out for A/B
    return true;
}

// Build encoder graph for T tokens × B batch items.
// mode: 0=dense (encoder_out), 1=sparse (sparse_out [1,T]), 2=colbert (colbert_out [dim,T])
// When B=1: standard single-text graph.
// When B>1: batched graph with 4D attention via flash_attn_ext.
// packed_mask (C3): B sequences are packed end-to-end into a single T=T_total token
//   stream (B stays 1); attention is restricted to each sequence's own tokens via a
//   host-built block-diagonal F16 mask input "seg_mask" [T,T] fed to flash_attn_ext.
//   Numerically identical to encoding each sequence alone (full bidirectional within a
//   segment, -inf across), but in one graph. Only used for absolute-position encoders
//   (no MPNet rel-bias / DeBERTa rel-embd / RoPE), so rel_pos_bias is null here.
// item_mask (C3): rectangular 4D batch — B sequences padded to T tokens, kept as
//   separate 4D batch items [hd,T,nh,B], with a per-item F16 mask "pad_mask" [T,T,1,B]
//   (−inf on padded key columns per item). Attention is O(B·T²) not O((B·T)²), so it
//   scales far better than packing for many short sequences. ggml flash_attn_ext accepts
//   the per-batch mask (q->ne[3] % mask->ne[3] == 0) and Metal indexes it per iq3.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
static ggml_cgraph * build_encoder_graph(crispembed_context * ctx, int T, int B = 1, int mode = 0,
                                         bool packed_mask = false, bool item_mask = false) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int H = hp.n_embd;
    const int n_heads = hp.n_head;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;
    const int TB = T * B; // total tokens in batch

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

    // Packed batch: block-diagonal segment mask [T, T] (F16), -inf across segments,
    // 0 within a segment. Fed to flash_attn_ext so packed sequences don't cross-attend.
    ggml_tensor * seg_mask = nullptr;
    if (packed_mask) {
        seg_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T, T);
        ggml_set_name(seg_mask, "seg_mask");
        ggml_set_input(seg_mask);
    }

    // Rectangular 4D batch: per-item padding mask [T, T, 1, B] (F16). For item b,
    // key column k is 0 if k is a real token, −inf if padding; broadcast over heads
    // (ne[2]=1) and applied per batch item (ne[3]=B). Filled by fill_item_pad_mask().
    ggml_tensor * pad_mask = nullptr;
    if (item_mask && !packed_mask && B > 1) {
        pad_mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, T, T, 1, B);
        ggml_set_name(pad_mask, "pad_mask");
        ggml_set_input(pad_mask);
    }

    // ModernBERT sliding-window (local attention) mask [T, T] (F16): local layers
    // attend only within ±local_attention/2; global layers (every Nth) use no mask.
    // Filled host-side by fill_local_window_mask(). Only created when the model has
    // a local window and alternating global layers.
    ggml_tensor * swa_mask = nullptr;
    if (!packed_mask && modernbert_swa_enabled(ctx)) {
        swa_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T, T);
        ggml_set_name(swa_mask, "swa_mask");
        ggml_set_input(swa_mask);
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
        ggml_tensor * inp = cur; // save for residual connection

        // Pre-LN: normalize before attention (ModernBERT)
        if (ctx->pre_ln && L.ln1_w) {
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln1_w);
            if (L.ln1_b) cur = ggml_add(gctx, cur, L.ln1_b);
        }

        // QKV projection (fused: 1 matmul + 3 view+cont, or 3 separate matmuls)
        ggml_tensor *Q, *K, *V;
        if (L.qkv_w) {
            ggml_tensor * qkv = ggml_mul_mat(gctx, L.qkv_w, cur);
            if (L.qkv_b) qkv = ggml_add(gctx, qkv, L.qkv_b);
            Q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3 * H * sizeof(float), 0));
            K = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3 * H * sizeof(float), H * sizeof(float)));
            V = ggml_cont(gctx, ggml_view_2d(gctx, qkv, H, TB, 3 * H * sizeof(float), 2 * H * sizeof(float)));
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
            Q = ggml_rope_ext(gctx, Q, rope_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens, layer_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(gctx, K, rope_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, hp.n_max_tokens, layer_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
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
            ggml_tensor * P = rel_pos_expanded; // [H, T*T]

            // c2p: project pos through K weights (with bias), dot with Q
            ggml_tensor * Pk = ggml_mul_mat(gctx, L.k_w, P); // [H, T*T]
            if (L.k_b) Pk = ggml_add(gctx, Pk, L.k_b);
            // Pk after reshape: [hd, nh, j, i] (j=ne[2] fast, i=ne[3] slow)
            Pk = ggml_reshape_4d(gctx, Pk, head_dim, n_heads, T, T);
            // c2p needs batch=(h,i) to match Qs batch=(h,i_q)
            // permute(0,2,1,3) → [hd, j, nh, i] → cont → batch = h+nh*i ✓
            ggml_tensor * Pk_b = ggml_cont(gctx, ggml_permute(gctx, Pk, 0, 2, 1, 3));
            Pk_b = ggml_reshape_3d(gctx, Pk_b, head_dim, T, (int64_t)n_heads * T);
            ggml_tensor * Qs_b = ggml_cont(gctx, ggml_permute(gctx, Qs, 0, 2, 1, 3));
            Qs_b = ggml_reshape_3d(gctx, Qs_b, head_dim, 1, (int64_t)n_heads * T);
            ggml_tensor * c2p = ggml_mul_mat(gctx, Pk_b, Qs_b); // [j, 1, nh*i]
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
            ggml_tensor * P_p2c =
                ggml_reshape_2d(gctx,
                                ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, P, H, T, T), // [H, T_j, T_i]
                                                             0, 2, 1, 3)), // → [H, T_i, T_j]
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
            ggml_tensor * p2c = ggml_mul_mat(gctx, Pq_b, Ks_b); // [T_q, 1, nh*T_k]
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
            attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3)); // [hd, nh, T]
            attn = ggml_reshape_2d(gctx, ggml_cont(gctx, attn), H, T);
        } else {
            float scale = 1.0f / sqrtf((float)head_dim);

            // Flash attention (supports optional position bias / segment mask)
            // Q/K/V: [hd, T, nh, B] after permute
            // rel_pos_bias: [T, T, nh] — MPNet additive bias; seg_mask: [T, T] F16
            // block-diagonal for packing; swa_mask: [T, T] F16 sliding window for
            // ModernBERT local layers. At most one applies per layer.
            const bool is_local_layer = (ctx->global_attn_every_n > 0) && (il % ctx->global_attn_every_n != 0);
            ggml_tensor * attn_mask;
            if (packed_mask)
                attn_mask = seg_mask;
            else if (pad_mask)
                attn_mask = pad_mask; // rectangular 4D per-item padding
            else if (swa_mask && is_local_layer)
                attn_mask = swa_mask; // ModernBERT local
            else
                attn_mask = rel_pos_bias; // global / MPNet / none
            attn = ggml_flash_attn_ext(gctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);
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
            inp = cur; // save for FFN residual
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
            const int K = hp.n_experts_per_tok;

            // Router logits: gate_w [H, n_exp] @ cur [H, TB] → [n_exp, TB]
            ggml_tensor * logits = ggml_mul_mat(gctx, L.moe_gate_w, cur);

            // Softmax over experts (ne[0] = n_exp) per token
            ggml_tensor * probs = ggml_soft_max(gctx, logits);

            // Top-K expert selection: [K, TB] I32
            ggml_tensor * ids = ggml_top_k(gctx, probs, K);

            // Gather top-K weights from softmax probs via get_rows
            // probs_3d [1, n_exp, TB]: get_rows selects K from n_exp per token
            ggml_tensor * probs_3d = ggml_reshape_3d(gctx, probs, 1, n_exp, TB);
            ggml_tensor * top_w = ggml_get_rows(gctx, probs_3d, ids); // [1, K, TB]
            top_w = ggml_reshape_2d(gctx, top_w, K, TB);              // [K, TB]

            // Input for the K expert slots: [H, TB] → [H, 1, TB].
            ggml_tensor * cur_3d = ggml_reshape_3d(gctx, cur, H, 1, TB);
            // The explicit ggml_repeat to [H, K, TB] is (hypothesized) redundant:
            // ggml_mul_mat_id broadcasts b's singleton slot dim over the K experts
            // in `ids` (llama.cpp's canonical build_moe_ffn pattern). Gate the
            // broadcast path behind CRISPEMBED_MOE_NO_REPEAT=1; default keeps the
            // repeat until byte-identity + latency are validated (env-gate rule).
            static const bool moe_no_repeat = [] {
                const char * e = std::getenv("CRISPEMBED_MOE_NO_REPEAT");
                return e && e[0] == '1';
            }();
            ggml_tensor * cur_slots = cur_3d;
            if (!moe_no_repeat) {
                ggml_tensor * rep_tgt = ggml_new_tensor_3d(gctx, cur->type, H, K, TB);
                cur_slots = ggml_repeat(gctx, cur_3d, rep_tgt); // [H, K, TB]
            }

            // Expert up projection: expert_fc1 [H, inter, n_exp] × [H, {K|1}, TB] → [inter, K, TB]
            ggml_tensor * up = ggml_mul_mat_id(gctx, L.expert_fc1_w, cur_slots, ids);

            // Activation: exact erf-GELU (NomicBERT v2 uses nn.GELU(approximate='none'))
            up = ggml_gelu_erf(gctx, up);

            // Expert down projection: expert_fc2 [inter, H, n_exp] × [inter, K, TB] → [H, K, TB]
            ggml_tensor * down = ggml_mul_mat_id(gctx, L.expert_fc2_w, up, ids);

            // Weighted combination: sum over K experts per token
            // down [H, K, TB] → permute to [K, H, TB], mul by weights [K, 1, TB], matmul sums K
            ggml_tensor * down_p = ggml_cont(gctx, ggml_permute(gctx, down, 1, 0, 2, 3)); // [K, H, TB]
            ggml_tensor * w_col = ggml_reshape_3d(gctx, top_w, K, 1, TB);                 // [K, 1, TB]
            ffn = ggml_mul_mat(gctx, w_col, down_p);                                      // [1, H, TB]
            ffn = ggml_reshape_2d(gctx, ffn, H, TB);                                      // [H, TB]

            // MoE output bias
            if (L.moe_ffn_bias) ffn = ggml_add(gctx, ffn, L.moe_ffn_bias);

        } else if (L.ffn_up_gate_w) {
            // Fused gated FFN (ModernBERT / GTE v1.5): one matmul → GLU → down.
            // All flavours use act(first_half)*second_half (non-swapped) layout
            // and differ only in the activation, which `bert.ffn_act` now names
            // explicitly (silu → SwiGLU, gelu → exact erf, gelu_pytorch_tanh →
            // tanh approximation). Without that key the historical per-arch
            // default applies, so published GGUFs keep their exact behaviour.
            ggml_tensor * up_gate = ggml_mul_mat(gctx, L.ffn_up_gate_w, cur); // [2*inter, T]
            if (ctx->ffn_swiglu)
                ffn = ggml_swiglu(gctx, up_gate);
            else
                ffn = ctx->geglu_erf ? ggml_geglu_erf(gctx, up_gate) : ggml_geglu(gctx, up_gate); // → [inter, T]
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
            // GTE v1.5's GteGatedMLP.down_proj is nn.Linear(..., bias=True) and the
            // converter emits ffn.fc2.bias; ModernBERT's mlp.Wo has no bias, so a
            // missing add here was invisible on ModernBERT but silently dropped a
            // whole per-layer bias vector on every GTE v1.5 model.
            if (L.fc2_b) ffn = ggml_add(gctx, ffn, L.fc2_b);
        } else if (L.ffn_gate_w) {
            // Separate SwiGLU (NomicBERT)
            ggml_tensor * up = ggml_mul_mat(gctx, L.fc1_w, cur);
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
        if (ctx->model.sparse_linear_b) sw = ggml_add(gctx, sw, ctx->model.sparse_linear_b);
        sw = ggml_relu(gctx, sw);
        ggml_set_name(sw, "sparse_out");
        ggml_set_output(sw);
        ggml_build_forward_expand(gf, sw);
    } else if (mode == 2 && ctx->model.colbert_linear_w) {
        // ColBERT head: Linear(H, colbert_dim) [+ bias] → [colbert_dim, T*B]
        ggml_tensor * cv = ggml_mul_mat(gctx, ctx->model.colbert_linear_w, cur);
        if (ctx->model.colbert_linear_b) cv = ggml_add(gctx, cv, ctx->model.colbert_linear_b);
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
            auto * fn =
                (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
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
    if (T <= 8) return 8;
    if (T <= 16) return 16;
    if (T <= 32) return 32;
    if (T <= 64) return 64;
    if (T <= 128) return 128;
    if (T <= 256) return 256;
    if (T <= 512) return 512;
    return T;
}

// ModernBERT sliding-window (local attention) mask. Fills the "swa_mask" input if
// the graph has one: mask[i][j] = 0 iff |i-j| <= local_attention/2, else -inf (F16).
// No-op unless the model defines a local window with alternating global layers.
static void fill_local_window_mask(crispembed_context * ctx, ggml_cgraph * gf, int T) {
    if (ctx->local_attention_window <= 0 || ctx->global_attn_every_n <= 0) return;
    ggml_tensor * swa = ggml_graph_get_tensor(gf, "swa_mask");
    if (!swa) return;
    const int radius = ctx->local_attention_window / 2;
    std::vector<ggml_fp16_t> md((size_t)T * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; i++) {
        ggml_fp16_t * row = md.data() + (size_t)i * T;
        for (int j = 0; j < T; j++) row[j] = (std::abs(i - j) <= radius) ? zero : ninf;
    }
    ggml_backend_tensor_set(swa, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
}

static std::vector<float> encode_tokens(crispembed_context * ctx, const embed_tokens & tokens) {
    const auto & hp = ctx->model.hparams;
    const int T = (int)tokens.ids.size();
    const int H = hp.n_embd;
    const bool bench = ctx->bench;
    auto t_encode_total = std::chrono::steady_clock::now();

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
    if (const char * v = std::getenv("CRISPEMBED_DEBUG_TOKENS"); v && v[0] && std::strcmp(v, "0") != 0) {
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
            auto bias_f32 = compute_rel_pos_bias(ctx->model.rel_attn_bias, T, ctx->model.hparams.n_head);
            // Convert to F16 for flash attention mask
            std::vector<ggml_fp16_t> bias_f16(bias_f32.size());
            for (size_t i = 0; i < bias_f32.size(); i++) bias_f16[i] = ggml_fp32_to_fp16(bias_f32[i]);
            ggml_backend_tensor_set(bias_t, bias_f16.data(), 0, bias_f16.size() * sizeof(ggml_fp16_t));
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

            // Read rel_embd data from backend (dequant-safe). The quantizer stores
            // rel_embd as Q8_0/Q4_K (it's a 2-D weight), so a raw n*sizeof(float) get
            // would overrun ggml_nbytes and abort — DeBERTa rerankers (mxbai-rerank-*)
            // and NER (gliner-deberta) ship a quantized rel_embd. to_f32 dequantizes.
            std::vector<float> embd_data = core_cpu::to_f32(ctx->model.rel_embd);

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
                    for (int d = 0; d < H_emb; d++) {
                        sum += row[d];
                        sum2 += (double)row[d] * row[d];
                    }
                    float mean = (float)(sum / H_emb);
                    float var = (float)(sum2 / H_emb) - mean * mean;
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
                            double log_ratio = std::log((double)abs_pos / mid) / std::log((double)(max_pos - 1) / mid);
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
                    memcpy(&expanded[(size_t)(i * T + j) * H_emb], &embd_data[(size_t)bucket * H_emb],
                           H_emb * sizeof(float));
                }
            }
            ggml_backend_tensor_set(rpe_t, expanded.data(), 0, expanded.size() * sizeof(float));
        }
    }

    // ModernBERT local sliding-window mask (no-op for other encoders)
    fill_local_window_mask(ctx, gf, T);

    // Compute (scheduler dispatches to GPU or CPU)
    debug_encode_stage("encode_tokens:compute", T, 1, 0);
    auto t_compute = std::chrono::steady_clock::now();
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: encoder compute failed\n");
        return {};
    }
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_compute).count();
        fprintf(stderr, "[crispembed-bench] encode_tokens graph compute (T=%d): %.1f ms\n", T, ms);
    }

    // Dump per-layer intermediates for diff harness
    if (ctx->dump_layers) {
        // CRISPEMBED_DUMP_LAYERS=1 prints a 6-float peek per stage (eyeballing).
        // CRISPEMBED_DUMP_LAYERS_GGUF=<path> writes the FULL tensors to a GGUF so
        // they can be diffed against the Python reference per stage — a peek
        // cannot be compared, and a final-embedding cosine only tells you THAT
        // something diverged, not WHERE. Stage names match
        // tools/dump_encoder_reference.py: emb_ln_out == HF hidden_states[0],
        // layer_i == HF hidden_states[i+1].
        const char * dump_gguf = std::getenv("CRISPEMBED_DUMP_LAYERS_GGUF");

        std::vector<std::string> names;
        names.push_back("emb_ln_out");
        names.push_back("attn_out_0");
        for (int il = 0; il < hp.n_layer; il++) names.push_back("layer_" + std::to_string(il));
        // The LAST block's output is renamed "encoder_out" above, so "layer_{n-1}"
        // does not exist in the graph. Without this the final block — the one that
        // actually feeds pooling — would be silently absent from the dump.
        names.push_back("encoder_out");

        struct ggml_context * dctx = nullptr;
        struct gguf_context * dg = nullptr;
        if (dump_gguf) {
            // no_alloc=false: tensors own their data so gguf_add_tensor can copy it.
            struct ggml_init_params dip = { (names.size() + 2) * ggml_tensor_overhead() +
                                                (size_t)hp.n_embd * 4096 * sizeof(float) * (names.size() + 1),
                                            nullptr, false };
            dctx = ggml_init(dip);
            if (dctx) {
                dg = gguf_init_empty();
                gguf_set_val_str(dg, "general.architecture", "crispembed-encoder-dump");
                gguf_set_val_u32(dg, "dump.n_layer", (uint32_t)hp.n_layer);
                gguf_set_val_u32(dg, "dump.n_embd", (uint32_t)hp.n_embd);
            }
        }

        for (const std::string & name : names) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, name.c_str());
            if (!t) continue;
            const int64_t n = ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));

            fprintf(stderr, "DUMP %s shape=[%lld,%lld] data=", name.c_str(), (long long)t->ne[0], (long long)t->ne[1]);
            const int show = n < 6 ? (int)n : 6;
            for (int i = 0; i < show; i++) fprintf(stderr, " %.6f", buf[i]);
            fprintf(stderr, " ...\n");

            if (dg && dctx) {
                // Keep ggml's [H, T] layout: ne[0] is the fast axis, so the flat
                // memory is already row-major (T, H) — the same as HF's
                // hidden_states[layer][0]. No transpose on either side.
                ggml_tensor * d = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
                if (!d) continue;
                ggml_set_name(d, name.c_str());
                memcpy(d->data, buf.data(), n * sizeof(float));
                gguf_add_tensor(dg, d);
            }
        }

        if (dg) {
            if (!gguf_write_to_file(dg, dump_gguf, /*only_meta*/ false)) {
                fprintf(stderr, "crispembed: failed to write layer dump '%s'\n", dump_gguf);
            } else {
                fprintf(stderr, "crispembed: wrote layer dump -> %s\n", dump_gguf);
            }
            gguf_free(dg);
        }
        if (dctx) ggml_free(dctx);
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
    int pool_method = ctx->pool_method; // set during load from metadata

    if (pool_method == 1) {
        // CLS pooling: take the first token (position 0 = [CLS])
        for (int h = 0; h < std::min(H, dim); h++) {
            pooled[h] = out_data[h + 0 * H]; // token 0 = [CLS]
        }
    } else if (pool_method == 2) {
        // Last-token pooling (decoder models)
        int last_t = 0;
        for (int t = T - 1; t >= 0; t--) {
            if (tokens.attn_mask[t]) {
                last_t = t;
                break;
            }
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
    auto t_pool = std::chrono::steady_clock::now();
    float norm = 0;
    for (int h = 0; h < dim; h++) norm += pooled[h] * pooled[h];
    norm = sqrtf(std::max(norm, 1e-12f));
    // Diagnostic: the pre-normalization magnitude is the only scale signal a
    // caller can see (the returned vector is unit-length by contract), so a
    // uniform scale error is invisible to cosine parity. Print it on request —
    // same lever as CRISPEMBED_DECODER_EMBED_RAW_NORM on the decoder path.
    if (std::getenv("CRISPEMBED_EMBED_RAW_NORM")) fprintf(stderr, "[crispembed-rawnorm] %.6f\n", norm);
    for (int h = 0; h < dim; h++) pooled[h] /= norm;
    if (bench) {
        double ms_pool = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pool).count();
        double ms_total =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_encode_total).count();
        fprintf(stderr, "[crispembed-bench] encode_tokens pool+normalize: %.2f ms\n", ms_pool);
        fprintf(stderr, "[crispembed-bench] encode_tokens total: %.1f ms\n", ms_total);
    }

    return pooled;
}

// C3: is the packed block-diagonal batch path eligible + enabled?
// Only for absolute-position encoders (no MPNet rel-bias, no DeBERTa rel-embd,
// no RoPE — those carry T×T / per-position structure that packing would need to
// re-index per segment). Default: ON when the primary backend is a GPU
// (A/B-proven on Metal 2026-07-12: 5.2–7.4× vs sequential on uniform AND
// mixed-length batches, parity cos 1.0 — see PLAN C3), OFF on CPU (measured
// unstable 0.46×–2.07× there). CRISPEMBED_ENCODER_PACKED=1/0 overrides the
// default in either direction.
static bool packed_batch_enabled(const crispembed_context * ctx) {
    if (ctx->is_decoder) return false;
    if (ctx->model.rel_attn_bias || ctx->model.rel_embd || ctx->use_rope) return false;
    const char * v = std::getenv("CRISPEMBED_ENCODER_PACKED");
    if (v && v[0]) return std::strcmp(v, "0") != 0;            // explicit override, both ways
    return ctx->backend && !ggml_backend_is_cpu(ctx->backend); // default: GPU on, CPU off
}

// Packed batched encoding (C3): pack all B sequences end-to-end into one graph of
// T_total = Σ T_i tokens, with a block-diagonal F16 mask restricting attention to
// each sequence's own tokens. Output is bit-parity with per-sequence encode_tokens
// (full bidirectional attention within a segment; positions restart per segment),
// but runs the whole batch in a single graph compute.
static std::vector<std::vector<float>> encode_tokens_packed(crispembed_context * ctx,
                                                            const std::vector<embed_tokens> & batch) {
    const auto & hp = ctx->model.hparams;
    const int B = (int)batch.size();
    const int H = hp.n_embd;
    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Segment offsets (dense — tokens were trimmed to real length upstream).
    std::vector<int> seg_start(B), seg_len(B);
    int T_total = 0;
    for (int b = 0; b < B; b++) {
        seg_start[b] = T_total;
        seg_len[b] = (int)batch[b].ids.size();
        T_total += seg_len[b];
    }
    if (T_total == 0) return {};

    // Reserve scheduler on a bucketed T_total (dedicated packed bucket).
    int T_bucket = bucket_seq_len(T_total);
    if (ctx->reserved_T_packed != T_bucket) {
        ggml_cgraph * measure_gf = build_encoder_graph(ctx, T_bucket, 1, 0, /*packed_mask=*/true);
        ggml_backend_sched_reserve(ctx->sched, measure_gf);
        ctx->reserved_T_packed = T_bucket;
    }

    ggml_cgraph * gf = build_encoder_graph(ctx, T_total, 1, 0, /*packed_mask=*/true);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate packed encoder graph\n");
        return {};
    }

    // Flatten token / position / type ids across segments.
    std::vector<int32_t> tok_data(T_total), pos_data(T_total);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < seg_len[b]; t++) {
            tok_data[seg_start[b] + t] = batch[b].ids[t];
            pos_data[seg_start[b] + t] = t + ctx->pos_offset; // positions restart per segment
        }
    }
    ggml_tensor * tok_ids = graph_tensor_or_log(gf, "tok_ids");
    if (!tok_ids) return {};
    ggml_backend_tensor_set(tok_ids, tok_data.data(), 0, T_total * sizeof(int32_t));

    if (ggml_tensor * pos_ids = ggml_graph_get_tensor(gf, "pos_ids")) {
        ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, T_total * sizeof(int32_t));
    } else if (ctx->model.pos_embd) {
        fprintf(stderr, "crispembed: missing graph tensor 'pos_ids' (packed)\n");
        return {};
    }

    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(T_total, 0);
        for (int b = 0; b < B; b++) {
            const auto & tids = batch[b].type_ids;
            for (int t = 0; t < seg_len[b] && t < (int)tids.size(); t++) {
                type_data[seg_start[b] + t] = tids[t];
            }
        }
        ggml_tensor * type_ids = graph_tensor_or_log(gf, "type_ids");
        if (!type_ids) return {};
        ggml_backend_tensor_set(type_ids, type_data.data(), 0, T_total * sizeof(int32_t));
    }

    // Block-diagonal F16 mask: -inf everywhere, 0 within each segment's block.
    ggml_tensor * seg_mask = graph_tensor_or_log(gf, "seg_mask");
    if (!seg_mask) return {};
    std::vector<ggml_fp16_t> mask_data((size_t)T_total * T_total, ggml_fp32_to_fp16(-INFINITY));
    const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
    for (int b = 0; b < B; b++) {
        const int s = seg_start[b], e = seg_start[b] + seg_len[b];
        for (int i = s; i < e; i++) {
            ggml_fp16_t * row = mask_data.data() + (size_t)i * T_total;
            for (int j = s; j < e; j++) row[j] = zero_f16;
        }
    }
    ggml_backend_tensor_set(seg_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    auto t_compute = std::chrono::steady_clock::now();
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: packed encoder compute failed\n");
        return {};
    }
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_compute).count();
        fprintf(stderr, "[crispembed-bench] packed encode (B=%d, T_total=%d): %.1f ms\n", B, T_total, ms);
    }

    ggml_tensor * out = graph_tensor_or_log(gf, "encoder_out");
    if (!out) return {};
    std::vector<float> out_buf((size_t)H * T_total);
    ggml_backend_tensor_get(out, out_buf.data(), 0, (size_t)H * T_total * sizeof(float));

    const int dim = hp.n_output > 0 ? hp.n_output : H;
    const int pool_method = ctx->pool_method;

    std::vector<std::vector<float>> results(B);
    for (int b = 0; b < B; b++) {
        const int s = seg_start[b], n = seg_len[b];
        std::vector<float> pooled(dim, 0.0f);
        if (pool_method == 1) { // CLS = first token of segment
            for (int h = 0; h < std::min(H, dim); h++) pooled[h] = out_buf[h + (size_t)s * H];
        } else if (pool_method == 2) { // last token of segment
            int last_t = s + n - 1;
            for (int h = 0; h < std::min(H, dim); h++) pooled[h] = out_buf[h + (size_t)last_t * H];
        } else { // mean over segment tokens
            if (n > 0) {
                for (int t = 0; t < n; t++)
                    for (int h = 0; h < std::min(H, dim); h++) pooled[h] += out_buf[h + (size_t)(s + t) * H];
                for (int h = 0; h < dim; h++) pooled[h] /= n;
            }
        }
        float norm = 0;
        for (int h = 0; h < dim; h++) norm += pooled[h] * pooled[h];
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int h = 0; h < dim; h++) pooled[h] /= norm;
        results[b] = std::move(pooled);
    }

    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[crispembed-bench] packed encode total (B=%d): %.1f ms\n", B, ms);
    }
    return results;
}

// Packed-group token budget. Packing collapses B sequences into one graph, which
// amortizes per-graph build/dispatch overhead and enlarges matmuls, but makes
// attention O(T_total^2) (the block-diagonal mask still computes the masked cells).
// So we pack GREEDILY into groups bounded by this token budget rather than one
// giant sequence — capping keeps attention bounded while still amortizing overhead.
static int packed_group_maxtok(const crispembed_context * ctx) {
    if (const char * v = std::getenv("CRISPEMBED_ENCODER_PACK_MAXTOK")) {
        int n = atoi(v);
        if (n > 0) return n;
    }
    // Longer sequences amortize better and lose relatively less to the quadratic
    // term; scale the budget with the model's typical single-seq bucket. Default
    // chosen empirically (see PLAN C3 A/B).
    (void)ctx;
    return 384;
}

// Pool + L2-normalize one sequence's [H, *] rows into a dim-vector.
// rows point at the H-strided hidden states; `n` valid tokens starting at row 0.
static std::vector<float> pool_and_norm(const float * rows, int n, int H, int dim, int pool_method) {
    std::vector<float> pooled(dim, 0.0f);
    if (pool_method == 1) { // CLS = first token
        for (int h = 0; h < std::min(H, dim); h++) pooled[h] = rows[h];
    } else if (pool_method == 2) { // last token
        const float * last = rows + (size_t)(n - 1) * H;
        for (int h = 0; h < std::min(H, dim); h++) pooled[h] = last[h];
    } else { // mean
        if (n > 0) {
            for (int t = 0; t < n; t++)
                for (int h = 0; h < std::min(H, dim); h++) pooled[h] += rows[(size_t)t * H + h];
            for (int h = 0; h < dim; h++) pooled[h] /= n;
        }
    }
    float norm = 0;
    for (int h = 0; h < dim; h++) norm += pooled[h] * pooled[h];
    norm = sqrtf(std::max(norm, 1e-12f));
    // See encode_tokens: pre-normalization magnitude, on request.
    if (std::getenv("CRISPEMBED_EMBED_RAW_NORM")) fprintf(stderr, "[crispembed-rawnorm] %.6f\n", norm);
    for (int h = 0; h < dim; h++) pooled[h] /= norm;
    return pooled;
}

// Rectangular 4D batch (C3): one group of B sequences padded to T_max, kept as
// separate 4D batch items with a per-item padding mask. Attention is O(B·T_max²)
// (vs packing's O((B·T)²)). Output is bit-parity with per-sequence encoding —
// padded keys are masked to −inf so real tokens never attend to padding, and
// padded query rows are discarded in pooling.
static std::vector<std::vector<float>> encode_tokens_group_4d(crispembed_context * ctx,
                                                              const std::vector<embed_tokens> & group) {
    const auto & hp = ctx->model.hparams;
    const int B = (int)group.size();
    const int H = hp.n_embd;
    int T_max = 0;
    for (const auto & t : group) T_max = std::max(T_max, (int)t.ids.size());
    if (T_max == 0) return {};
    const int TB = T_max * B;

    ggml_cgraph * gf = build_encoder_graph(ctx, T_max, B, 0, /*packed_mask=*/false, /*item_mask=*/true);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "crispembed: failed to allocate 4D batch graph\n");
        return {};
    }

    // Item-major flat layout: index (t + T_max*b) — t (position) fast, b (item) slow.
    std::vector<int32_t> tok_data(TB, 0), pos_data(TB, ctx->pos_offset);
    for (int b = 0; b < B; b++) {
        const int len = (int)group[b].ids.size();
        for (int t = 0; t < len; t++) {
            tok_data[(size_t)b * T_max + t] = group[b].ids[t];
            pos_data[(size_t)b * T_max + t] = t + ctx->pos_offset;
        }
    }
    ggml_tensor * tok_ids = graph_tensor_or_log(gf, "tok_ids");
    if (!tok_ids) return {};
    ggml_backend_tensor_set(tok_ids, tok_data.data(), 0, TB * sizeof(int32_t));
    if (ggml_tensor * pos_ids = ggml_graph_get_tensor(gf, "pos_ids")) {
        ggml_backend_tensor_set(pos_ids, pos_data.data(), 0, TB * sizeof(int32_t));
    } else if (ctx->model.pos_embd) {
        fprintf(stderr, "crispembed: missing graph tensor 'pos_ids' (4D)\n");
        return {};
    }
    if (ctx->model.type_embd) {
        std::vector<int32_t> type_data(TB, 0);
        for (int b = 0; b < B; b++) {
            const auto & tids = group[b].type_ids;
            const int len = std::min((int)group[b].ids.size(), (int)tids.size());
            for (int t = 0; t < len; t++) type_data[(size_t)b * T_max + t] = tids[t];
        }
        ggml_tensor * type_ids = graph_tensor_or_log(gf, "type_ids");
        if (!type_ids) return {};
        ggml_backend_tensor_set(type_ids, type_data.data(), 0, TB * sizeof(int32_t));
    }

    // Per-item padding mask [T_max, T_max, 1, B]: mask(k, q, 0, b) = 0 if key k is a
    // real token in item b, else −inf. Independent of q (bidirectional).
    ggml_tensor * pad_mask = graph_tensor_or_log(gf, "pad_mask");
    if (!pad_mask) return {};
    std::vector<ggml_fp16_t> mask_data((size_t)TB * T_max);
    const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf_f16 = ggml_fp32_to_fp16(-INFINITY);
    for (int b = 0; b < B; b++) {
        const int len = (int)group[b].ids.size();
        ggml_fp16_t * slab = mask_data.data() + (size_t)b * T_max * T_max; // item b
        for (int q = 0; q < T_max; q++) {
            ggml_fp16_t * row = slab + (size_t)q * T_max; // keys for query q
            for (int k = 0; k < T_max; k++) row[k] = (k < len) ? zero_f16 : ninf_f16;
        }
    }
    ggml_backend_tensor_set(pad_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: 4D batch compute failed\n");
        return {};
    }

    ggml_tensor * out = graph_tensor_or_log(gf, "encoder_out");
    if (!out) return {};
    std::vector<float> out_buf((size_t)H * TB);
    ggml_backend_tensor_get(out, out_buf.data(), 0, (size_t)H * TB * sizeof(float));

    const int dim = hp.n_output > 0 ? hp.n_output : H;
    std::vector<std::vector<float>> results(B);
    for (int b = 0; b < B; b++) {
        const int len = (int)group[b].ids.size();
        results[b] = pool_and_norm(out_buf.data() + (size_t)b * T_max * H, len, H, dim, ctx->pool_method);
    }
    return results;
}

// 4D batch group size (sequences per rectangular graph). Sequences are length-sorted
// then chunked so each group pads to a similar T_max, minimizing wasted compute.
static int four_d_group_size(const crispembed_context * ctx) {
    (void)ctx;
    if (const char * v = std::getenv("CRISPEMBED_ENCODER_4D_GROUP")) {
        int n = atoi(v);
        if (n > 0) return n;
    }
    return 32;
}

static bool four_d_batch_enabled(const crispembed_context * ctx) {
    if (ctx->is_decoder) return false;
    if (ctx->model.rel_attn_bias || ctx->model.rel_embd || ctx->use_rope) return false;
    const char * v = std::getenv("CRISPEMBED_ENCODER_4D");
    return v && v[0] && std::strcmp(v, "0") != 0; // opt-in until A/B-proven
}

// Rectangular 4D batch over the whole batch: length-sort, chunk into groups, run
// each as one padded 4D graph, then restore original order.
static std::vector<std::vector<float>> encode_tokens_4d(crispembed_context * ctx,
                                                        const std::vector<embed_tokens> & batch) {
    const int B = (int)batch.size();
    std::vector<int> order(B);
    for (int i = 0; i < B; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return batch[a].ids.size() < batch[b].ids.size(); });

    const int G = four_d_group_size(ctx);
    std::vector<std::vector<float>> results(B);
    for (int start = 0; start < B; start += G) {
        const int end = std::min(start + G, B);
        std::vector<embed_tokens> group;
        group.reserve(end - start);
        for (int i = start; i < end; i++) group.push_back(batch[order[i]]);
        auto part = (group.size() == 1) ? std::vector<std::vector<float>>{ encode_tokens(ctx, group[0]) }
                                        : encode_tokens_group_4d(ctx, group);
        if ((int)part.size() != (int)group.size()) return {}; // signal failure → caller falls back
        for (int i = start; i < end; i++) results[order[i]] = std::move(part[i - start]);
    }
    return results;
}

// Batched encoding: multiple texts. Uses the packed block-diagonal graph (C3)
// when eligible + enabled, greedily grouped under a token budget; otherwise
// encodes each sequence individually.
static std::vector<std::vector<float>> encode_tokens_batch(crispembed_context * ctx,
                                                           const std::vector<embed_tokens> & batch) {
    const int B = (int)batch.size();
    if (B == 0) return {};

    // Rectangular 4D per-item-mask batch (C3 follow-up): O(B·T²), preferred when enabled.
    if (B > 1 && four_d_batch_enabled(ctx)) {
        auto r = encode_tokens_4d(ctx, batch);
        if ((int)r.size() == B) return r;
        fprintf(stderr, "crispembed: 4D batch failed, falling back\n");
    }

    if (B > 1 && packed_batch_enabled(ctx)) {
        const int maxtok = packed_group_maxtok(ctx);
        std::vector<std::vector<float>> results;
        results.reserve(B);
        bool ok = true;
        int i = 0;
        while (i < B && ok) {
            // Greedily accumulate sequences until the token budget is hit.
            // A single over-budget sequence still forms its own (size-1) group.
            int j = i, tsum = 0;
            while (j < B) {
                int len = (int)batch[j].ids.size();
                if (j > i && tsum + len > maxtok) break;
                tsum += len;
                j++;
            }
            std::vector<embed_tokens> group(batch.begin() + i, batch.begin() + j);
            auto part = (group.size() == 1) ? std::vector<std::vector<float>>{ encode_tokens(ctx, group[0]) }
                                            : encode_tokens_packed(ctx, group);
            if ((int)part.size() != (int)group.size()) {
                ok = false;
                break;
            }
            for (auto & v : part) results.push_back(std::move(v));
            i = j;
        }
        if (ok && (int)results.size() == B) return results;
        fprintf(stderr, "crispembed: packed batch failed, falling back to sequential\n");
    }

    std::vector<std::vector<float>> results;
    results.reserve(B);
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
static std::vector<float> run_encoder_raw(crispembed_context * ctx, const embed_tokens & tokens, int mode,
                                          int * out_T) {
    const int T = (int)tokens.ids.size();
    if (out_T) *out_T = T;

    int T_bucket = bucket_seq_len(T);
    int & reserved = (mode == 1) ? ctx->reserved_T_sparse : (mode == 2) ? ctx->reserved_T_colbert : ctx->reserved_T;
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

            // Dequant-safe: rel_embd is a 2-D weight the quantizer stores as Q8_0/Q4_K,
            // so a raw n*sizeof(float) get overruns ggml_nbytes and aborts. DeBERTa
            // rerankers (mxbai-rerank-*) and NER (gliner-deberta) ship a quantized
            // rel_embd; to_f32 dequantizes it. (See the twin path for encode_tokens.)
            std::vector<float> embd_data = core_cpu::to_f32(ctx->model.rel_embd);

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
                    for (int d = 0; d < H_emb; d++) {
                        sum += row[d];
                        sum2 += (double)row[d] * row[d];
                    }
                    float mean = (float)(sum / H_emb);
                    float var = (float)(sum2 / H_emb) - mean * mean;
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
                            double log_ratio = std::log((double)abs_pos / mid) / std::log((double)(max_pos - 1) / mid);
                            int log_pos = (int)std::ceil(log_ratio * (mid - 1)) + mid;
                            signed_bucket = log_pos * sign_val;
                        }
                        bucket = signed_bucket + pos_buckets;
                    } else {
                        bucket = i - j + max_pos / 2;
                    }
                    if (bucket < 0) bucket = 0;
                    if (bucket >= max_pos) bucket = max_pos - 1;
                    memcpy(&expanded[(size_t)(i * T + j) * H_emb], &embd_data[(size_t)bucket * H_emb],
                           H_emb * sizeof(float));
                }
            }
            ggml_backend_tensor_set(rpe_t, expanded.data(), 0, expanded.size() * sizeof(float));
        }
    }

    // ModernBERT local sliding-window mask (no-op for other encoders)
    fill_local_window_mask(ctx, gf, T);

    debug_encode_stage("run_encoder_raw:compute", T, 1, mode);
    auto t_raw_compute = std::chrono::steady_clock::now();
    if (!sched_graph_compute(ctx->sched, gf, ctx->n_threads)) {
        fprintf(stderr, "crispembed: compute failed (mode=%d)\n", mode);
        return {};
    }
    if (ctx->bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_raw_compute).count();
        fprintf(stderr, "[crispembed-bench] run_encoder_raw graph compute (T=%d, mode=%d): %.1f ms\n", T, mode, ms);
    }

    const char * out_name = (mode == 1) ? "sparse_out" : (mode == 2) ? "colbert_out" : "encoder_out";
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
    core_initbench::timer ib_init("crispembed_init");
    auto * ctx = new crispembed_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    if (model_path) ctx->model_path_for_audio = model_path;
    ctx->dump_layers = (std::getenv("CRISPEMBED_DUMP_LAYERS") != nullptr);
    ctx->bench = core_env::on("CRISPEMBED_CRISPEMBED_BENCH");

    // Detect model type from GGUF metadata.
    // Decoder models have either decoder.hidden_size (CrispEmbed-native) or
    // general.architecture in {qwen3, gemma3, llama, ...} (Ollama-format).
    // Encoder models (BERT/XLM-R) have bert.* keys and enc.N.* tensor names.
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    bool is_dec = false;
    bool is_lfm2 = false;
    if (g) {
        is_dec = gguf_find_key(g, "decoder.hidden_size") >= 0;
        if (!is_dec) {
            int64_t ki = gguf_find_key(g, "general.architecture");
            if (ki >= 0) {
                std::string arch = gguf_get_val_str(g, ki);
                is_dec = (arch == "qwen3" || arch == "gemma3" || arch == "gemma-embedding" || arch == "llama" ||
                          arch == "qwen2" || arch == "mistral" || arch == "phi3");
                is_lfm2 = (arch == "lfm2");
            }
        }
    }
    // T18: keep this parse alive and hand it to load_model() instead of parsing
    // the same metadata a second time (29 ms on a 250k-vocab GGUF). RAII so the
    // several `delete ctx; return nullptr;` exits below cannot leak it.
    struct gguf_holder {
        gguf_context * g = nullptr;
        ~gguf_holder() {
            if (g) gguf_free(g);
        }
    } gg{ g };
    ib_init.mark("arch_detect_gguf_open");

    if (is_lfm2) {
        ctx->is_lfm2 = true;
        ctx->backend = crispembed_init_backend(ctx->n_threads);
        ctx->backends.push_back(ctx->backend);
        if (!ctx->backend) {
            delete ctx;
            return nullptr;
        }
        if (ggml_backend_is_cpu(ctx->backend)) {
            ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
        }
        ctx->lfm2_ctx = lfm2_embed_load(model_path, ctx->backend);
        if (!ctx->lfm2_ctx) {
            delete ctx;
            return nullptr;
        }
        ctx->model.hparams.n_embd = (uint32_t)lfm2_embed_n_embd(ctx->lfm2_ctx);
        ctx->model.hparams.n_output = ctx->model.hparams.n_embd;
        // ColBERT multi-vector support
        if (lfm2_embed_has_colbert(ctx->lfm2_ctx)) {
            ctx->model.has_colbert = true;
            ctx->model.colbert_dim = lfm2_embed_colbert_dim(ctx->lfm2_ctx);
        }
    } else if (is_dec) {
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
            fprintf(stderr, "crispembed: using %s backend with CPU fallback\n", ggml_backend_name(ctx->backend));
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
        ctx->sched =
            ggml_backend_sched_new(ctx->backends.data(), nullptr, (int)ctx->backends.size(), graph_nodes, false, false);
        crispembed_imatrix_install(ctx->sched);
        ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false));

        // Load BPE tokenizer from GGUF. T18: reuse the arch-detect parse rather
        // than parsing the same metadata again (CRISPEMBED_GGUF_REPARSE=1 keeps
        // the pre-T18 second parse). `own_g2` decides who frees it.
        gguf_init_params gp2 = { true, nullptr };
        const char * reparse2 = std::getenv("CRISPEMBED_GGUF_REPARSE");
        const bool own_g2 = !gg.g || (reparse2 && reparse2[0] && std::strcmp(reparse2, "0") != 0);
        gguf_context * g2 = own_g2 ? gguf_init_from_file(model_path, gp2) : gg.g;
        if (g2) {
            const int64_t ki2 = gguf_find_key(g2, "tokenizer.ggml.tokens");
            const int64_t mi2 = gguf_find_key(g2, "tokenizer.ggml.merges");
            if (ki2 >= 0) {
                int nv = (int)gguf_get_arr_n(g2, ki2);
                std::vector<std::string> vocab(nv);
                for (int i = 0; i < nv; i++) vocab[i] = gguf_get_arr_str(g2, ki2, i);

                std::vector<std::string> merges;
                if (mi2 >= 0) {
                    int nm = (int)gguf_get_arr_n(g2, mi2);
                    merges.resize(nm);
                    for (int i = 0; i < nm; i++) merges[i] = gguf_get_arr_str(g2, mi2, i);
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
                        if (type == GGUF_TYPE_BOOL)
                            add_bos = gguf_get_val_bool(g2, ki_add_bos);
                        else if (type == GGUF_TYPE_UINT32)
                            add_bos = gguf_get_val_u32(g2, ki_add_bos) != 0;
                        else if (type == GGUF_TYPE_INT32)
                            add_bos = gguf_get_val_i32(g2, ki_add_bos) != 0;
                        if (!add_bos) bos_id = -1;
                    }
                }
                const int64_t ki_sfx = gguf_find_key(g2, "tokenizer.ggml.suffix_token_id");
                int suffix_id = ki_sfx >= 0 ? (int)gguf_get_val_i32(g2, ki_sfx) : pad_id;
                bool is_spm_bpe = u32g("tokenizer.ggml.is_spm_bpe", 0) != 0;

                // Community/official llama.cpp SPM exports (e.g. gemma-embedding,
                // tokenizer.ggml.model="llama") store `scores` and NO `merges`.
                // A real BPE always has merges; loading such a GGUF as
                // BPE-with-empty-merges char-tokenizes every input → garbage
                // embeddings. Route merge-less + scored vocabs to the
                // SentencePiece tokenizer (which tokenizes from scores).
                const int64_t si2 = gguf_find_key(g2, "tokenizer.ggml.scores");
                const bool has_scores = (si2 >= 0 && gguf_get_arr_type(g2, si2) == GGUF_TYPE_FLOAT32);
                const bool is_spm = merges.empty() && has_scores;

                if (is_spm) {
                    std::vector<float> scores((size_t)gguf_get_arr_n(g2, si2));
                    std::memcpy(scores.data(), gguf_get_arr_data(g2, si2), scores.size() * sizeof(float));
                    const int unk_id = u32g("tokenizer.ggml.unknown_token_id", 3);
                    // `bos_id` above may be forced to -1 by add_bos_token=false;
                    // sp_tokenizer gates the wrap via set_add_flags, so pass the raw id.
                    const int sp_bos = u32g("tokenizer.ggml.bos_token_id", 2);
                    const bool add_bos = core_gguf::kv_bool(g2, "tokenizer.ggml.add_bos_token", true);
                    const bool add_eos = core_gguf::kv_bool(g2, "tokenizer.ggml.add_eos_token", true);
                    // llama.cpp SPM (tokenizer.ggml.model="llama"/"gemma") uses
                    // BPE-style merge ranks; T5-style unigram uses Viterbi.
                    std::string tk_model;
                    if (const int64_t km = gguf_find_key(g2, "tokenizer.ggml.model");
                        km >= 0 && gguf_get_kv_type(g2, km) == GGUF_TYPE_STRING)
                        tk_model = gguf_get_val_str(g2, km);
                    const bool bpe_merge = (tk_model == "llama" || tk_model == "gemma");
                    // Gemma sets add_space_prefix=false; llama.cpp default is true.
                    const bool add_space_prefix = core_gguf::kv_bool(g2, "tokenizer.ggml.add_space_prefix", true);
                    ctx->sp_tokenizer.load(vocab, scores, sp_bos, eos_id, unk_id, pad_id, ctx->dec->n_max_pos);
                    ctx->sp_tokenizer.set_add_flags(add_bos, add_eos);
                    ctx->sp_tokenizer.set_spm_mode(bpe_merge, add_space_prefix);
                    ctx->use_sentencepiece = true;
                    fprintf(stderr, "crispembed: using SentencePiece tokenizer (%d tokens, %zu scores, %s)\n", nv,
                            scores.size(), bpe_merge ? "bpe-merge" : "unigram");
                } else {
                    ctx->bpe_tokenizer.load(vocab, merges, eos_id, pad_id, suffix_id, bos_id, is_spm_bpe,
                                            ctx->dec->n_max_pos);
                    ctx->use_bpe = true;
                    fprintf(stderr, "crispembed: %s BPE tokenizer (%d tokens, %zu merges)\n",
                            is_spm_bpe ? "SentencePiece" : "GPT-2", nv, merges.size());
                }
            }
            if (own_g2) gguf_free(g2);
        }
    } else {
        if (!load_model(ctx, model_path, gg.g)) {
            delete ctx;
            return nullptr;
        }
    }
    ib_init.mark("model_load");
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

extern "C" const char * crispembed_model_license(int index) {
    const char * value = crispembed_mgr::model_license(index);
    return value ? value : "";
}

extern "C" const char * crispembed_model_card_url(int index) {
    const char * value = crispembed_mgr::model_card_url(index);
    return value ? value : "";
}

extern "C" const float * crispembed_encode(crispembed_context * ctx, const char * text, int * out_n_dim) {
    if (!ctx || !text) return nullptr;
    auto t_enc_start = std::chrono::steady_clock::now();
    // T18 instrument: the FIRST encode is where a GPU backend compiles its
    // compute pipelines, so it is part of the one-shot fixed cost even though
    // it is not in crispembed_init(). Subsequent encodes report warm compute.
    core_initbench::timer ib_enc("crispembed_encode");

    // Prepend prefix if set (e.g. "query: ", "Represent this sentence: ")
    std::string prefixed;
    const char * enc_text = text;
    if (!ctx->prefix.empty()) {
        prefixed = ctx->prefix + text;
        enc_text = prefixed.c_str();
    }

    if (ctx->is_lfm2 && ctx->lfm2_ctx) {
        const int dim = lfm2_embed_n_embd(ctx->lfm2_ctx);
        ctx->last_output.resize(dim);
        if (!lfm2_embed_encode_to(ctx->lfm2_ctx, enc_text, ctx->last_output.data())) {
            return nullptr;
        }

        if (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < dim) {
            ctx->last_output.resize(ctx->matryoshka_dim);
            float norm = 0;
            for (int i = 0; i < ctx->matryoshka_dim; i++) norm += ctx->last_output[i] * ctx->last_output[i];
            norm = sqrtf(std::max(norm, 1e-12f));
            for (int i = 0; i < ctx->matryoshka_dim; i++) ctx->last_output[i] /= norm;
        }

        if (out_n_dim) *out_n_dim = (int)ctx->last_output.size();
        return ctx->last_output.data();
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
            if (tokens.attn_mask[i]) {
                actual_len = i + 1;
                break;
            }
        }
        if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
            tokens.ids.resize(actual_len);
            tokens.type_ids.resize(actual_len);
            tokens.attn_mask.resize(actual_len);
        }
    }
    ib_enc.mark("tokenize");

    // CRISPEMBED_DEBUG_TOKENS=1 dumps the final token-id sequence (single encode).
    if (const char * dv = std::getenv("CRISPEMBED_DEBUG_TOKENS"); dv && dv[0] && std::strcmp(dv, "0") != 0) {
        fprintf(stderr, "crispembed: token_ids (n=%zu):", tokens.ids.size());
        for (int32_t id : tokens.ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    // E6: one-shot warning when ≥50% of content tokens are [UNK] — signals a
    // script/vocabulary mismatch (e.g. Japanese text on an English-only model).
    // Silenced by CRISPEMBED_WARN_UNK=0.
    if (ctx->unk_id >= 0 && !ctx->unk_warned) {
        // Count content tokens: skip first (CLS/BOS) and last active (SEP/EOS)
        int last_active = -1;
        for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
            if (tokens.attn_mask[i]) {
                last_active = i;
                break;
            }
        }
        int n_content = 0, n_unk = 0;
        for (int i = 1; i < last_active; i++) { // skip pos 0 (CLS/BOS) and last_active (SEP/EOS)
            if (!tokens.attn_mask[i]) continue;
            n_content++;
            if (tokens.ids[i] == ctx->unk_id) n_unk++;
        }
        if (n_content > 0 && n_unk * 100 / n_content >= 50) {
            if (!core_env::explicitly_off("CRISPEMBED_WARN_UNK")) {
                fprintf(stderr,
                        "crispembed: warning: %d%% of input tokens are [UNK] — this model's vocabulary "
                        "may not cover this script; see docs/LANGUAGES.md for models that do "
                        "(silence with CRISPEMBED_WARN_UNK=0)\n",
                        n_unk * 100 / n_content);
                ctx->unk_warned = true;
            }
        }
    }

    if (ctx->is_decoder && ctx->dec) {
        if (ctx->prefix_cache_enabled < 0) {
            const char * e = std::getenv("CRISPEMBED_DECODER_PREFIX_CACHE");
            ctx->prefix_cache_enabled = (e && e[0] == '0') ? 0 : 1; // default ON
        }
        if (ctx->prefix_cache_enabled) {
            ctx->last_output = decoder_encode_tokens_cached(*ctx->dec, ctx->backend, tokens, ctx->n_threads, ctx->sched,
                                                            &ctx->compute_meta, ctx->dec_prefix, ctx->prev_dec_tokens);
        } else {
            ctx->last_output =
                decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads, ctx->sched, &ctx->compute_meta);
        }
    } else {
        ctx->last_output = encode_tokens(ctx, tokens);
    }

    // Matryoshka dimension truncation: truncate + re-normalize
    if (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < (int)ctx->last_output.size()) {
        ctx->last_output.resize(ctx->matryoshka_dim);
        float norm = 0;
        for (int i = 0; i < ctx->matryoshka_dim; i++) norm += ctx->last_output[i] * ctx->last_output[i];
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int i = 0; i < ctx->matryoshka_dim; i++) ctx->last_output[i] /= norm;
    }

    if (ctx->bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc_start).count();
        fprintf(stderr, "[crispembed-bench] crispembed_encode total: %.1f ms\n", ms);
    }
    if (out_n_dim) *out_n_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_set_gpu_backend(const char * name) {
    crispasr_set_gpu_backend_pref(name);
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
    // Weights change under the prefix cache — invalidate it (its cached K/V
    // were computed against the old weights).
    ctx->dec_prefix.clear();
    ctx->prev_dec_tokens.clear();
    return decoder_set_lora(*ctx->dec, ctx->backend, name) ? 1 : 0;
}

extern "C" const char * crispembed_get_lora(const crispembed_context * ctx) {
    if (!ctx || !ctx->is_decoder || !ctx->dec) return "";
    return ctx->dec->active_lora.c_str();
}

extern "C" int crispembed_list_lora(const crispembed_context * ctx, const char *** out_names, int * out_count) {
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
    mctx->lora_name_ptrs.push_back(nullptr); // null-terminated
    if (out_names) *out_names = mctx->lora_name_ptrs.data();
    if (out_count) *out_count = (int)ctx->dec->lora_adapters.size();
    return 1;
}

extern "C" const float * crispembed_encode_batch(crispembed_context * ctx, const char ** texts, int n_texts,
                                                 int * out_n_dim) {
    if (!ctx || !texts || n_texts <= 0) return nullptr;
    auto t_batch_start = std::chrono::steady_clock::now();

    if (ctx->is_lfm2 && ctx->lfm2_ctx) {
        const int dim = lfm2_embed_n_embd(ctx->lfm2_ctx);
        const int out_dim = (ctx->matryoshka_dim > 0 && ctx->matryoshka_dim < dim) ? ctx->matryoshka_dim : dim;
        ctx->last_output.resize(n_texts * out_dim);
        std::vector<float> tmp;
        if (out_dim < dim) tmp.resize(dim);

        for (int i = 0; i < n_texts; i++) {
            const char * inp = texts[i] ? texts[i] : "";
            std::string prefixed;
            if (!ctx->prefix.empty()) {
                prefixed = ctx->prefix + inp;
                inp = prefixed.c_str();
            }

            float * dst = ctx->last_output.data() + i * out_dim;
            float * enc_dst = (out_dim < dim) ? tmp.data() : dst;
            if (!lfm2_embed_encode_to(ctx->lfm2_ctx, inp, enc_dst)) {
                fprintf(stderr, "crispembed: LFM2 batch encode failed for item %d\n", i);
                return nullptr;
            }
            if (out_dim < dim) {
                float norm = 0;
                for (int j = 0; j < out_dim; j++) norm += tmp[j] * tmp[j];
                norm = sqrtf(std::max(norm, 1e-12f));
                for (int j = 0; j < out_dim; j++) dst[j] = tmp[j] / norm;
            }
        }

        if (out_n_dim) *out_n_dim = out_dim;
        return ctx->last_output.data();
    }

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
            if (t.attn_mask[j]) {
                actual_len = j + 1;
                break;
            }
        }
        if (actual_len > 0 && actual_len < (int)t.ids.size()) {
            t.ids.resize(actual_len);
            t.type_ids.resize(actual_len);
            t.attn_mask.resize(actual_len);
        }

        // CRISPEMBED_DEBUG_TOKENS=1 dumps the final token-id sequence (decoder path).
        if (const char * dv = std::getenv("CRISPEMBED_DEBUG_TOKENS"); dv && dv[0] && std::strcmp(dv, "0") != 0) {
            fprintf(stderr, "crispembed: token_ids[%d] (n=%zu):", i, t.ids.size());
            for (int32_t id : t.ids) fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }

        // E6: one-shot UNK-ratio warning (same logic as single-encode path)
        if (ctx->unk_id >= 0 && !ctx->unk_warned) {
            int last_active = -1;
            for (int j = (int)t.attn_mask.size() - 1; j >= 0; j--) {
                if (t.attn_mask[j]) {
                    last_active = j;
                    break;
                }
            }
            int n_content = 0, n_unk = 0;
            for (int j = 1; j < last_active; j++) {
                if (!t.attn_mask[j]) continue;
                n_content++;
                if (t.ids[j] == ctx->unk_id) n_unk++;
            }
            if (n_content > 0 && n_unk * 100 / n_content >= 50) {
                const char * env = std::getenv("CRISPEMBED_WARN_UNK");
                if (!env || std::strcmp(env, "0") != 0) {
                    fprintf(stderr,
                            "crispembed: warning: %d%% of input tokens are [UNK] — this model's vocabulary "
                            "may not cover this script; see docs/LANGUAGES.md for models that do "
                            "(silence with CRISPEMBED_WARN_UNK=0)\n",
                            n_unk * 100 / n_content);
                    ctx->unk_warned = true;
                }
            }
        }
    }

    // For encoder models: true batched inference (one graph, all texts)
    std::vector<std::vector<float>> batch_results;

    if (!ctx->is_decoder) {
        batch_results = encode_tokens_batch(ctx, all_tokens);
    } else {
        // Decoder: batched graph (falls back to sequential for B=1 or multimodal)
        batch_results = decoder_encode_tokens_batch(*ctx->dec, ctx->backend, all_tokens, ctx->n_threads, ctx->sched,
                                                    &ctx->compute_meta);
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
    if (ctx->bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_batch_start).count();
        fprintf(stderr, "[crispembed-bench] crispembed_encode_batch total (%d texts): %.1f ms\n", n_texts, ms);
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

extern "C" int crispembed_encode_sparse(crispembed_context * ctx, const char * text, const int32_t ** out_indices,
                                        const float ** out_values) {
    if (!ctx || !text || !ctx->model.has_sparse || ctx->is_decoder) return 0;
    auto t_sparse_start = std::chrono::steady_clock::now();

    embed_tokens tokens;
    if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode(text);
    else
        tokens = ctx->wp_tokenizer.encode(text);

    // Trim to actual (non-padded) length
    int T = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            T = i + 1;
            break;
        }
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

        // Read MLM head weights from GPU/CPU backend. mlm_transform_w and token_embd
        // are 2-D weight matrices that the quantizer may store as Q8_0/F16/Q4_K, so
        // read them via to_f32 (dequant-safe) — a raw n*sizeof(float) get would overrun
        // ggml_nbytes and abort. The 1-D norm/bias tensors stay F32.
        std::vector<float> tb(H), lnw(H), lnb(H);
        std::vector<float> tw = core_cpu::to_f32(ctx->model.mlm_transform_w);
        ggml_backend_tensor_get(ctx->model.mlm_transform_b, tb.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ctx->model.mlm_ln_w, lnw.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ctx->model.mlm_ln_b, lnb.data(), 0, H * sizeof(float));
        std::vector<float> emb_w = core_cpu::to_f32(ctx->model.token_embd);
        std::vector<float> mlm_b(V, 0.0f);
        if (ctx->model.mlm_bias) ggml_backend_tensor_get(ctx->model.mlm_bias, mlm_b.data(), 0, V * sizeof(float));

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
            for (int i = 0; i < H; i++) {
                float d = h[i] - mean;
                var += d * d;
            }
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
        if (ctx->bench) {
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sparse_start).count();
            fprintf(stderr, "[crispembed-bench] crispembed_encode_sparse total: %.1f ms\n", ms);
        }
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
            float weight = raw[t]; // element [0, t]
            if (weight <= 0.0f) continue;
            int32_t vid = tokens.ids[t];
            auto it = vocab_weights.find(vid);
            if (it == vocab_weights.end() || it->second < weight) vocab_weights[vid] = weight;
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
            ctx->last_sparse_values.push_back(logf(1.0f + max_w)); // SPLADE uses log(1+ReLU)
        }
    }

    int n = (int)ctx->last_sparse_indices.size();
    if (out_indices) *out_indices = ctx->last_sparse_indices.data();
    if (out_values) *out_values = ctx->last_sparse_values.data();
    if (ctx->bench) {
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sparse_start).count();
        fprintf(stderr, "[crispembed-bench] crispembed_encode_sparse total: %.1f ms\n", ms);
    }
    return n;
}

// ---------------------------------------------------------------------------
// Multi-vector encode (ColBERT head)
// ---------------------------------------------------------------------------

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
extern "C" const float * crispembed_encode_multivec(crispembed_context * ctx, const char * text, int * out_n_tokens,
                                                    int * out_dim) {
    if (!ctx || !text || !ctx->model.has_colbert || ctx->is_decoder) return nullptr;
    auto t_multivec_start = std::chrono::steady_clock::now();

    // LFM2 ColBERT path — uses its own tokenizer + encoder
    if (ctx->is_lfm2 && ctx->lfm2_ctx) {
        const int cd = lfm2_embed_colbert_dim(ctx->lfm2_ctx);
        const int max_tok = 512;
        ctx->last_multivec.resize(max_tok * cd);
        int n = lfm2_embed_encode_multivec(ctx->lfm2_ctx, text, ctx->last_multivec.data(), max_tok);
        if (n <= 0) return nullptr;
        ctx->last_multivec_n_tokens = n;
        ctx->last_multivec_dim = cd;
        if (out_n_tokens) *out_n_tokens = n;
        if (out_dim) *out_dim = cd;
        if (ctx->bench) {
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_multivec_start).count();
            fprintf(stderr, "[crispembed-bench] crispembed_encode_multivec total: %.1f ms\n", ms);
        }
        return ctx->last_multivec.data();
    }

    embed_tokens tokens;
    if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode(text);
    else
        tokens = ctx->wp_tokenizer.encode(text);

    // Count real tokens (non-padded)
    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            T_real = i + 1;
            break;
        }
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
    ctx->last_multivec_dim = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim) *out_dim = dim;
    if (ctx->bench) {
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_multivec_start).count();
        fprintf(stderr, "[crispembed-bench] crispembed_encode_multivec total: %.1f ms\n", ms);
    }
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
extern "C" const float * crispembed_encode_tokens(crispembed_context * ctx, const char * text, int * out_n_tokens,
                                                  int * out_dim) {
    if (!ctx || !text || ctx->is_decoder) return nullptr;
    auto t_tokens_start = std::chrono::steady_clock::now();

    // Apply the configured prefix (e.g. "query: ") for consistency with
    // the dense encode path.
    std::string enc_text = ctx->prefix.empty() ? std::string(text) : ctx->prefix + text;

    embed_tokens tokens;
    if (ctx->use_bpe)
        tokens = ctx->bpe_tokenizer.encode(enc_text);
    else if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode(enc_text);
    else
        tokens = ctx->wp_tokenizer.encode(enc_text);

    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            T_real = i + 1;
            break;
        }
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
    ctx->last_token_n = raw_T;
    ctx->last_token_dim = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim) *out_dim = dim;
    if (ctx->bench) {
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tokens_start).count();
        fprintf(stderr, "[crispembed-bench] crispembed_encode_tokens total: %.1f ms\n", ms);
    }
    return ctx->last_token_embeddings.data();
}

extern "C" const float * crispembed_encode_tokens_raw(crispembed_context * ctx, const char * text, int * out_n_tokens,
                                                      int * out_dim) {
    if (!ctx || !text || ctx->is_decoder) return nullptr;

    std::string enc_text = ctx->prefix.empty() ? std::string(text) : ctx->prefix + text;

    embed_tokens tokens;
    if (ctx->use_bpe)
        tokens = ctx->bpe_tokenizer.encode(enc_text);
    else if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode(enc_text);
    else
        tokens = ctx->wp_tokenizer.encode(enc_text);

    int T_real = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            T_real = i + 1;
            break;
        }
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
    std::memcpy(ctx->last_token_embeddings.data(), raw.data(), (size_t)dim * (size_t)raw_T * sizeof(float));

    ctx->last_token_ids.assign(tokens.ids.begin(), tokens.ids.begin() + raw_T);
    ctx->last_token_n = raw_T;
    ctx->last_token_dim = dim;

    if (out_n_tokens) *out_n_tokens = raw_T;
    if (out_dim) *out_dim = dim;
    return ctx->last_token_embeddings.data();
}

extern "C" const int32_t * crispembed_last_token_ids(const crispembed_context * ctx) {
    if (!ctx || ctx->last_token_n == 0) return nullptr;
    return ctx->last_token_ids.data();
}

extern "C" const char * crispembed_token_str(const crispembed_context * ctx, int32_t id) {
    if (!ctx || ctx->is_decoder) return nullptr;
    const std::string & s =
        ctx->use_sentencepiece ? ctx->sp_tokenizer.token_str((int)id) : ctx->wp_tokenizer.token_str((int)id);
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

// Forward declaration — defined below, shared by single + batch rerank.
static float crispembed_apply_classifier(crispembed_context * ctx, const float * encoder_out, int T);

extern "C" float crispembed_rerank(crispembed_context * ctx, const char * query, const char * document) {
    if (!ctx || !query || !document || !ctx->model.is_reranker || ctx->is_decoder) return 0.0f;
    auto t_rerank_start = std::chrono::steady_clock::now();

    embed_tokens tokens;
    if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode_pair(query, document);
    else
        tokens = ctx->wp_tokenizer.encode_pair(query, document);

    // Trim to real tokens
    int T = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            T = i + 1;
            break;
        }
    }
    if (T == 0) return 0.0f;
    tokens.ids.resize(T);
    tokens.type_ids.resize(T);
    tokens.attn_mask.resize(T);

    int raw_T = 0;
    std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
    if (raw.empty()) return 0.0f;

    return crispembed_apply_classifier(ctx, raw.data(), raw_T);
}

// Apply classifier head to CLS vector (shared between single + batch rerank).
// Uses cached weights to avoid GPU→CPU transfer per call.
static float crispembed_apply_classifier(crispembed_context * ctx, const float * encoder_out, int T) {
    const int H = ctx->model.hparams.n_embd;
    const float * cls_vec = encoder_out; // first H floats = token 0

    // Cache classifier weights on first call (avoids 4MB transfer per rerank)
    if (!ctx->rerank_cache_valid) {
        // NOTE: the classifier/pooler *weights* are 2-D and the quantizer stores them
        // Q8_0/Q4_K, so read them via core_cpu::to_f32 (dequant-safe) — a raw
        // H*H*sizeof(float) get overruns ggml_nbytes and aborts ("tensor read out of
        // bounds"), which crashed reranking on every quantized GGUF (jina-reranker-v2
        // etc.). Biases stay F32.
        if (ctx->model.classifier_2layer) {
            ctx->rerank_db.resize(H);
            ctx->rerank_dw = core_cpu::to_f32(ctx->model.classifier_dense_w); // [H,H]
            ggml_backend_tensor_get(ctx->model.classifier_dense_b, ctx->rerank_db.data(), 0, H * sizeof(float));
            ctx->rerank_ow = core_cpu::to_f32(ctx->model.classifier_out_w); // [H]
            ctx->rerank_out_has_bias = ctx->model.classifier_out_b != nullptr;
            if (ctx->rerank_out_has_bias) {
                ggml_backend_tensor_get(ctx->model.classifier_out_b, &ctx->rerank_out_bias, 0, sizeof(float));
            }
        } else if (ctx->model.classifier_w) {
            ctx->rerank_ow = core_cpu::to_f32(ctx->model.classifier_w); // [H] or [H,1]
            ctx->rerank_out_has_bias = ctx->model.classifier_b != nullptr;
            if (ctx->rerank_out_has_bias) {
                ggml_backend_tensor_get(ctx->model.classifier_b, &ctx->rerank_out_bias, 0, sizeof(float));
            }
        }
        // Pooler (DeBERTa)
        ctx->rerank_has_pooler = ctx->model.pooler_w && ctx->model.pooler_b;
        if (ctx->rerank_has_pooler) {
            ctx->rerank_pb.resize(H);
            ctx->rerank_pw = core_cpu::to_f32(ctx->model.pooler_w); // [H,H]
            ggml_backend_tensor_get(ctx->model.pooler_b, ctx->rerank_pb.data(), 0, H * sizeof(float));
        }
        ctx->rerank_cache_valid = true;
    }

    // Apply ContextPooler if present (DeBERTa-v2 reranker).
    //
    // HF's ContextPooler applies ACT2FN[config.pooler_hidden_act]; for the
    // mxbai-rerank DeBERTa pair that is "gelu", which in HF/PyTorch is the
    // *erf-exact* GELU, not the tanh approximation used here historically
    // (same class as the granite projector finding). Default is erf-exact:
    // the A/B in tests/results/mxbai-gelu/ shows it collapses the f16
    // residual vs the ONNX reference 12-96x at no cost, and no shipped
    // artifact carries pooler tensors, so the flip perturbs nothing shipped.
    // CRISPEMBED_RERANK_POOLER_GELU_ERF=0 (value-parsed) restores tanh-approx.
    std::vector<float> pooled_buf;
    if (ctx->rerank_has_pooler) {
        static const bool pooler_gelu_erf = [] {
            const char * v = std::getenv("CRISPEMBED_RERANK_POOLER_GELU_ERF");
            return !(v && v[0] && std::strcmp(v, "0") == 0);
        }();
        pooled_buf.resize(H);
        for (int i = 0; i < H; i++) {
            float acc = ctx->rerank_pb[i];
            for (int j = 0; j < H; j++) acc += cls_vec[j] * ctx->rerank_pw[i * H + j];
            const float x = acc;
            if (pooler_gelu_erf) {
                pooled_buf[i] = 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
            } else {
                pooled_buf[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            }
        }
        cls_vec = pooled_buf.data();
    }

    float score = 0.0f;
    if (ctx->model.classifier_2layer) {
        std::vector<float> hidden(H);
        for (int i = 0; i < H; i++) {
            float acc = ctx->rerank_db[i];
            for (int j = 0; j < H; j++) acc += cls_vec[j] * ctx->rerank_dw[i * H + j];
            hidden[i] = std::tanh(acc);
        }
        for (int i = 0; i < H; i++) score += hidden[i] * ctx->rerank_ow[i];
        if (ctx->rerank_out_has_bias) score += ctx->rerank_out_bias;
    } else {
        for (int h = 0; h < H; h++) score += cls_vec[h] * ctx->rerank_ow[h];
        if (ctx->rerank_out_has_bias) score += ctx->rerank_out_bias;
    }
    return score;
}

// Batch rerank: score multiple documents against the same query.
// Runs the encoder for each pair sequentially (same as single rerank)
// but caches classifier weights so only the encoder forward pass repeats.
extern "C" int crispembed_rerank_batch(crispembed_context * ctx, const char * query, const char ** documents,
                                       int n_docs, float * out_scores) {
    if (!ctx || !query || !documents || !out_scores || n_docs <= 0) return 0;
    if (!ctx->model.is_reranker || ctx->is_decoder) return 0;

    // The classifier/pooler cache is populated lazily by
    // crispembed_apply_classifier() below, which dequantises the 2-D weights
    // via core_cpu::to_f32. This entry point used to duplicate that block
    // with raw H*H*sizeof(float) ggml_backend_tensor_get() calls, which
    // overrun ggml_nbytes on a Q8_0/Q4_K weight and abort the whole process
    // ("tensor read out of bounds") — the crash the single-document path was
    // already fixed for. The quantizer does quantise both classifier.dense
    // (2-layer heads: jina-reranker-v2) and pooler.weight (DeBERTa
    // ContextPooler: mxbai-rerank), so the duplicate was live on the server's
    // /rerank endpoint. Removed; there is nothing to warm here.

    int scored = 0;
    for (int d = 0; d < n_docs; d++) {
        if (!documents[d]) {
            out_scores[d] = 0.0f;
            scored++;
            continue;
        }

        embed_tokens tokens;
        if (ctx->use_sentencepiece)
            tokens = ctx->sp_tokenizer.encode_pair(query, documents[d]);
        else
            tokens = ctx->wp_tokenizer.encode_pair(query, documents[d]);

        int T = 0;
        for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
            if (tokens.attn_mask[i]) {
                T = i + 1;
                break;
            }
        }
        if (T == 0) {
            out_scores[d] = 0.0f;
            scored++;
            continue;
        }
        tokens.ids.resize(T);
        tokens.type_ids.resize(T);
        tokens.attn_mask.resize(T);

        int raw_T = 0;
        std::vector<float> raw = run_encoder_raw(ctx, tokens, 0, &raw_T);
        if (raw.empty()) {
            out_scores[d] = 0.0f;
            scored++;
            continue;
        }

        out_scores[d] = crispembed_apply_classifier(ctx, raw.data(), raw_T);
        scored++;
    }
    return scored;
}

// ---------------------------------------------------------------------------
// Audio encoding via crisp_audio (BidirLM-Omni and similar)
// ---------------------------------------------------------------------------
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
namespace bidirlm_audio {
struct context;
context * open(const char * gguf_path, int n_threads, bool use_gpu);
const float * encode(context * ctx, const float * pcm, int n_samples, int * out_dim);
void close(context * ctx);
} // namespace bidirlm_audio

static bidirlm_audio::context * audio_lazy_open(crispembed_context * ctx) {
    if (!ctx) return nullptr;
    if (ctx->audio_ctx) return (bidirlm_audio::context *)ctx->audio_ctx;
    if (ctx->model_path_for_audio.empty()) return nullptr;
    bool use_gpu = ctx->backend && !ggml_backend_is_cpu(ctx->backend);
    auto * a = bidirlm_audio::open(ctx->model_path_for_audio.c_str(), ctx->n_threads, use_gpu);
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

extern "C" const float * crispembed_encode_audio(crispembed_context * ctx, const float * pcm_samples, int n_samples,
                                                 int * out_dim) {
#ifdef CRISPEMBED_HAS_CRISP_AUDIO
    auto * a = audio_lazy_open(ctx);
    if (!a) return nullptr;
    return bidirlm_audio::encode(a, pcm_samples, n_samples, out_dim);
#else
    (void)ctx;
    (void)pcm_samples;
    (void)n_samples;
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
                              /*shared_backend=*/ctx->backend, ctx->n_threads, /*verbosity=*/1)) {
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
    return 1; // unknown — caller should attempt encode and check return.
}

namespace {

// Run the vision tower and stage results into ctx->last_vision_out.
// Layout: [image_embeds (n_merged*dim), deepstack_0, deepstack_1, ...].
bool vision_run_and_stage(crispembed_context * ctx, const float * pixel_patches, int n_patches,
                          const int32_t * grid_thw, int n_images, bool include_deepstack) {
    auto * v = vision_lazy_open(ctx);
    if (!v) return false;
    bidirlm_vision::encode_result r;
    if (!bidirlm_vision::encode(*v, pixel_patches, n_patches, grid_thw, n_images, r, include_deepstack)) {
        return false;
    }
    const size_t per_slab = (size_t)r.n_merged * r.output_dim;
    const size_t total = per_slab * (1 + r.n_deepstack);
    ctx->last_vision_out.resize(total);
    std::memcpy(ctx->last_vision_out.data(), r.image_embeds, per_slab * sizeof(float));
    if (r.n_deepstack > 0 && r.deepstack) {
        std::memcpy(ctx->last_vision_out.data() + per_slab, r.deepstack,
                    (size_t)r.n_deepstack * per_slab * sizeof(float));
    }
    ctx->last_vision_dim = r.output_dim;
    ctx->last_vision_n_merged = r.n_merged;
    ctx->last_vision_n_deepstack = r.n_deepstack;
    bidirlm_vision::encode_result_free(r);
    return true;
}

} // namespace

extern "C" const float * crispembed_encode_image(crispembed_context * ctx, const float * pixel_patches, int n_patches,
                                                 const int32_t * grid_thw, int n_images, int * out_dim) {
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

extern "C" const float * crispembed_encode_image_raw(crispembed_context * ctx, const float * pixel_patches,
                                                     int n_patches, const int32_t * grid_thw, int n_images,
                                                     int * out_n_merged, int * out_dim, int * out_n_deepstack) {
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
const float * encode_image_conditioned(crispembed_context * ctx, embed_tokens && tokens, const float * pixel_patches,
                                       int n_patches, const int32_t * grid_thw, int n_images, int * out_dim,
                                       const char * caller) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !pixel_patches || !grid_thw || n_images <= 0) return nullptr;
    if (!ctx->is_decoder || !ctx->dec) {
        fprintf(stderr, "%s: model is not a multimodal decoder.\n", caller);
        return nullptr;
    }
    if (ctx->dec->image_token_id < 0) {
        fprintf(stderr,
                "%s: model GGUF has no decoder.image_token_id — "
                "re-export with vision metadata.\n",
                caller);
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
        fprintf(stderr,
                "%s: vision tower output dim %d != decoder "
                "hidden_size %d — model mismatch.\n",
                caller, v_dim, ctx->dec->n_embd);
        return nullptr;
    }
    std::vector<float> vision_buf;
    vision_buf.swap(ctx->last_vision_out);
    const float * image_embeds = vision_buf.data();
    const float * deepstack = (v_nds > 0) ? vision_buf.data() + (size_t)v_merged * v_dim : nullptr;

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
    dimg.deepstack = deepstack;
    dimg.n_image_tokens = v_merged;
    dimg.n_deepstack = v_nds;
    dimg.grid_thw = grid_thw;
    dimg.n_images = n_images;

    auto vec =
        decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads, ctx->sched, &ctx->compute_meta, &dimg);
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

} // namespace

extern "C" const float * crispembed_encode_text_with_image(crispembed_context * ctx, const char * text,
                                                           const float * pixel_patches, int n_patches,
                                                           const int32_t * grid_thw, int n_images, int * out_dim) {
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
    if (ctx->use_bpe)
        tokens = ctx->bpe_tokenizer.encode(enc_text);
    else if (ctx->use_sentencepiece)
        tokens = ctx->sp_tokenizer.encode(enc_text);
    else
        tokens = ctx->wp_tokenizer.encode(enc_text);

    // Trim padding: only keep tokens where attn_mask == 1.
    int actual_len = 0;
    for (int i = (int)tokens.attn_mask.size() - 1; i >= 0; i--) {
        if (tokens.attn_mask[i]) {
            actual_len = i + 1;
            break;
        }
    }
    if (actual_len > 0 && actual_len < (int)tokens.ids.size()) {
        tokens.ids.resize(actual_len);
        tokens.type_ids.resize(actual_len);
        tokens.attn_mask.resize(actual_len);
    }

    return encode_image_conditioned(ctx, std::move(tokens), pixel_patches, n_patches, grid_thw, n_images, out_dim,
                                    "crispembed_encode_text_with_image");
}

extern "C" const float * crispembed_encode_with_image_ids(crispembed_context * ctx, const int32_t * token_ids,
                                                          int n_tokens, const float * pixel_patches, int n_patches,
                                                          const int32_t * grid_thw, int n_images, int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !token_ids || n_tokens <= 0) return nullptr;

    embed_tokens tokens;
    tokens.ids.assign(token_ids, token_ids + n_tokens);
    tokens.type_ids.assign((size_t)n_tokens, 0);
    tokens.attn_mask.assign((size_t)n_tokens, 1);

    return encode_image_conditioned(ctx, std::move(tokens), pixel_patches, n_patches, grid_thw, n_images, out_dim,
                                    "crispembed_encode_with_image_ids");
}

// ---------------------------------------------------------------------------
// In-process image preprocessor (file-based)
// ---------------------------------------------------------------------------
#include "image_preprocess.h"

extern "C" const float * crispembed_preprocess_image(crispembed_context * ctx, const char * image_path,
                                                     int * out_n_patches, int * out_row_dim, int32_t out_grid_thw[3]) {
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
    cfg.deskew = ctx->image_deskew;
    cfg.deskew_max_angle = ctx->image_deskew_max_angle;
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) {
        return nullptr;
    }
    // Stash into ctx->last_vision_out so the returned pointer remains valid
    // until the next preprocessor call (mirrors encode_image's contract).
    ctx->last_vision_out = std::move(r.patches);
    if (out_n_patches) *out_n_patches = r.n_patches;
    if (out_row_dim) *out_row_dim = r.row_dim;
    if (out_grid_thw) {
        out_grid_thw[0] = r.grid_thw[0];
        out_grid_thw[1] = r.grid_thw[1];
        out_grid_thw[2] = r.grid_thw[2];
    }
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_preprocess_image_rgb(crispembed_context * ctx, const uint8_t * rgb, int height,
                                                         int width, int channels, int * out_n_patches,
                                                         int * out_row_dim, int32_t out_grid_thw[3]) {
    if (out_n_patches) *out_n_patches = 0;
    if (out_row_dim) *out_row_dim = 0;
    if (!ctx || !rgb || height <= 0 || width <= 0 || (channels != 3 && channels != 4)) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    cfg.deskew = ctx->image_deskew;
    cfg.deskew_max_angle = ctx->image_deskew_max_angle;
    image_preproc::result r;
    if (!image_preproc::preprocess_rgb(rgb, height, width, channels, cfg, r)) {
        return nullptr;
    }
    ctx->last_vision_out = std::move(r.patches);
    if (out_n_patches) *out_n_patches = r.n_patches;
    if (out_row_dim) *out_row_dim = r.row_dim;
    if (out_grid_thw) {
        out_grid_thw[0] = r.grid_thw[0];
        out_grid_thw[1] = r.grid_thw[1];
        out_grid_thw[2] = r.grid_thw[2];
    }
    return ctx->last_vision_out.data();
}

extern "C" const float * crispembed_encode_image_file(crispembed_context * ctx, const char * image_path,
                                                      int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !image_path) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    cfg.deskew = ctx->image_deskew;
    cfg.deskew_max_angle = ctx->image_deskew_max_angle;
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) return nullptr;

    return crispembed_encode_image(ctx, r.patches.data(), r.n_patches, r.grid_thw, /*n_images=*/1, out_dim);
}

extern "C" const float * crispembed_encode_text_with_image_file(crispembed_context * ctx, const char * text,
                                                                const char * image_path, int * out_dim) {
    if (out_dim) *out_dim = 0;
    if (!ctx || !text || !image_path) return nullptr;

    image_preproc::config cfg;
    if (ctx->dec && ctx->dec->spatial_merge_size > 0) {
        cfg.merge_size = ctx->dec->spatial_merge_size;
    }
    cfg.deskew = ctx->image_deskew;
    cfg.deskew_max_angle = ctx->image_deskew_max_angle;
    image_preproc::result r;
    if (!image_preproc::preprocess_file(image_path, cfg, r)) return nullptr;

    return crispembed_encode_text_with_image(ctx, text, r.patches.data(), r.n_patches, r.grid_thw, /*n_images=*/1,
                                             out_dim);
}

extern "C" void crispembed_set_image_deskew(crispembed_context * ctx, int enable, float max_angle_deg) {
    if (!ctx) return;
    ctx->image_deskew = enable;
    if (max_angle_deg > 0.0f) ctx->image_deskew_max_angle = max_angle_deg;
}

// ---------------------------------------------------------------------------
// Standalone ViT image embedding C API (SigLIP, CLIP)
// ---------------------------------------------------------------------------

#include "vit_embed.h"

struct crispembed_vit_context {
    vit_embed::context * vit = nullptr;
    std::vector<float> last_output;
};

extern "C" crispembed_vit_context * crispembed_vit_init(const char * model_path, int n_threads) {
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

extern "C" const float * crispembed_vit_encode_file(crispembed_vit_context * ctx, const char * image_path,
                                                    int * out_dim) {
    if (!ctx || !image_path || !out_dim) {
        if (out_dim) *out_dim = 0;
        return nullptr;
    }
    ctx->last_output = vit_embed::encode_file(ctx->vit, image_path);
    if (ctx->last_output.empty()) {
        *out_dim = 0;
        return nullptr;
    }
    *out_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" void crispembed_vit_set_deskew(crispembed_vit_context * ctx, int enable, float max_angle_deg) {
    if (!ctx || !ctx->vit) return;
    vit_embed::set_deskew(ctx->vit, enable != 0, max_angle_deg);
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

extern "C" crispembed_clip_text_context * crispembed_clip_text_init(const char * model_path, int n_threads) {
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

extern "C" const float * crispembed_clip_text_encode(crispembed_clip_text_context * ctx, const char * text,
                                                     int * out_dim) {
    if (!ctx || !ctx->ct || !text || !out_dim) return nullptr;
    ctx->last_output = clip_text::encode(ctx->ct, text);
    if (ctx->last_output.empty()) {
        *out_dim = 0;
        return nullptr;
    }
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
    std::vector<std::vector<float>> emb_buf; // owns embedding data
    std::vector<float> single_emb;
};

// Biometric acknowledgement, process-wide. The CLI and the server set this
// after their own (interactive or flag-driven) acknowledgement, so the gate
// fires exactly once per process however CrispEmbed was entered.
static std::atomic<bool> g_biometric_accepted{ false };

extern "C" void crispembed_accept_biometric_use(void) {
    g_biometric_accepted.store(true, std::memory_order_relaxed);
}

static bool biometric_use_acknowledged() {
    if (g_biometric_accepted.load(std::memory_order_relaxed)) return true;
    const char * env = std::getenv("CRISPEMBED_ACCEPT_BIOMETRIC");
    return env && *env && strcmp(env, "0") != 0;
}

extern "C" crispembed_face_context * crispembed_face_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new crispembed_face_context();
    if (!cnn_embed::load(&ctx->cnn, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }

    // Gate on the model's own declared type, so a recognition model is caught
    // however it was named. This is the single chokepoint every binding funnels
    // through (Python ctypes, Rust, Dart FFI) — the CLI/server flag alone would
    // leave all three ungated. No interactive prompt here: a library must
    // not read stdin. Callers that want to ask a human do it themselves and
    // then call crispembed_accept_biometric_use().
    const char * type = cnn_embed::model_type(ctx->cnn);
    if (type && strcmp(type, "recognition") == 0 && !biometric_use_acknowledged()) {
        fprintf(stderr,
                "crispembed: '%s' is a FACE RECOGNITION model. Its output is a biometric\n"
                "template — special-category personal data under GDPR Art. 9, which generally\n"
                "needs an Art. 9(2) basis (e.g. explicit consent) before you process it.\n"
                "Searching a gallery of templates (1:N identification) builds a biometric\n"
                "identification system: high-risk under EU AI Act Annex III from 2 December\n"
                "2027, and prohibited outright in some settings (Art. 5). See POLICY.md.\n"
                "\n"
                "Refusing to load. Acknowledge with crispembed_accept_biometric_use() or set\n"
                "CRISPEMBED_ACCEPT_BIOMETRIC=1.\n",
                model_path);
        cnn_embed::free(ctx->cnn);
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

extern "C" const crispembed_face_detection * crispembed_detect_faces(crispembed_face_context * ctx,
                                                                     const char * image_path, float conf_threshold,
                                                                     int det_size, int * out_n_faces) {
    if (!ctx || !image_path || !out_n_faces) {
        if (out_n_faces) *out_n_faces = 0;
        return nullptr;
    }

    auto dets = cnn_embed::detect_file(ctx->cnn, image_path, conf_threshold, det_size > 0 ? det_size : 640);
    ctx->det_buf.resize(dets.size());
    for (size_t i = 0; i < dets.size(); i++) {
        auto & d = ctx->det_buf[i];
        d.x = dets[i].x;
        d.y = dets[i].y;
        d.w = dets[i].w;
        d.h = dets[i].h;
        d.confidence = dets[i].confidence;
        memcpy(d.landmarks, dets[i].landmarks, sizeof(d.landmarks));
    }
    *out_n_faces = (int)dets.size();
    return ctx->det_buf.empty() ? nullptr : ctx->det_buf.data();
}

extern "C" const float * crispembed_encode_face(crispembed_face_context * ctx, const char * image_path,
                                                const float * landmarks_10, int * out_dim) {
    if (!ctx || !image_path || !landmarks_10 || !out_dim) {
        if (out_dim) *out_dim = 0;
        return nullptr;
    }

    ctx->single_emb = cnn_embed::encode_face_file(ctx->cnn, image_path, landmarks_10);

    if (ctx->single_emb.empty()) {
        *out_dim = 0;
        return nullptr;
    }
    *out_dim = (int)ctx->single_emb.size();
    return ctx->single_emb.data();
}

extern "C" const crispembed_face_result * crispembed_face_pipeline(crispembed_face_context * det_ctx,
                                                                   crispembed_face_context * rec_ctx,
                                                                   const char * image_path, float conf_threshold,
                                                                   int det_size, int * out_n_faces) {
    if (!det_ctx || !rec_ctx || !image_path || !out_n_faces) {
        if (out_n_faces) *out_n_faces = 0;
        return nullptr;
    }

    auto results =
        cnn_embed::face_pipeline(det_ctx->cnn, rec_ctx->cnn, image_path, conf_threshold, det_size > 0 ? det_size : 640);
    // Store results in det_ctx scratch buffers
    det_ctx->result_buf.resize(results.size());
    det_ctx->emb_buf.resize(results.size());
    for (size_t i = 0; i < results.size(); i++) {
        auto & r = det_ctx->result_buf[i];
        r.det.x = results[i].det.x;
        r.det.y = results[i].det.y;
        r.det.w = results[i].det.w;
        r.det.h = results[i].det.h;
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

extern "C" float crispembed_colbert_score(const float * query_vecs, int n_query, const float * doc_vecs, int n_doc,
                                          int dim) {
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

extern "C" int crispembed_colbert_score_batch(const float * query_vecs, int n_query, const float ** doc_vecs_list,
                                              const int * doc_n_tokens, int n_docs, int dim, float * out_scores) {
    if (!query_vecs || !doc_vecs_list || !doc_n_tokens || !out_scores) return -1;

#pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < n_docs; d++) {
        out_scores[d] = crispembed_colbert_score(query_vecs, n_query, doc_vecs_list[d], doc_n_tokens[d], dim);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// OCR model dispatcher — auto-detects model architecture from GGUF metadata and
// routes to the matching backend. Covers both math (pix2tex_mfr, hmer, bttr,
// ppformulanet, posformer, mixtex) and general text/document OCR (qwen2vl,
// internvl2, glm_ocr, got_ocr, parseq, tesseract_lstm, granite_vision,
// lightonocr, deepseek_ocr2). Public API: crispembed_ocr_model_* (the
// crispembed_math_ocr_* names are kept as deprecated forwarding aliases).
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
#include "lightonocr.h"
#include "deepseek_ocr2.h"
#include "smoldocling_ocr.h"
#include "unlimited_ocr.h"
#include "smt_ocr.h"
#include "tromr_ocr.h"
#include "flova_ocr.h"
#include "transcoda_ocr.h"
#include "ppocrv6_ocr.h"
#include "core/gguf_loader.h"

enum ocr_model_type {
    OCR_MODEL_PIX2TEX,
    OCR_MODEL_HMER,
    OCR_MODEL_BTTR,
    OCR_MODEL_PPFORMULANET,
    OCR_MODEL_PPFORMULANET_L,
    OCR_MODEL_POSFORMER,
    OCR_MODEL_MIXTEX,
    OCR_MODEL_QWEN2VL,
    OCR_MODEL_INTERNVL2,
    OCR_MODEL_PARSEQ,
    OCR_MODEL_GLM_OCR,
    OCR_MODEL_GOT_OCR,
    OCR_MODEL_TESSERACT_LSTM,
    OCR_MODEL_GRANITE_VISION,
    OCR_MODEL_LIGHTONOCR,
    OCR_MODEL_DEEPSEEK_OCR2,
    OCR_MODEL_SMOLDOCLING,
    OCR_MODEL_UNLIMITED_OCR,
    OCR_MODEL_SMT,
    OCR_MODEL_TROMR,
    OCR_MODEL_FLOVA,
    OCR_MODEL_TRANSCODA,
    OCR_MODEL_PPOCRV6
};

struct ocr_model {
    ocr_model_type type;
    void * ctx;
};

static ocr_model_type detect_arch(const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return OCR_MODEL_PIX2TEX;
    std::string arch = core_gguf::kv_str(g, "general.architecture", "pix2tex_mfr");
    core_gguf::free_metadata(g);
    if (arch == "hmer") return OCR_MODEL_HMER;
    if (arch == "bttr") return OCR_MODEL_BTTR;
    if (arch == "ppformulanet") return OCR_MODEL_PPFORMULANET;
    if (arch == "ppformulanet_l") return OCR_MODEL_PPFORMULANET_L;
    if (arch == "posformer") return OCR_MODEL_POSFORMER;
    if (arch == "mixtex") return OCR_MODEL_MIXTEX;
    if (arch == "qwen2vl" || arch == "qwen3vl") return OCR_MODEL_QWEN2VL;
    if (arch == "internvl2") return OCR_MODEL_INTERNVL2;
    if (arch == "parseq") return OCR_MODEL_PARSEQ;
    if (arch == "glm_ocr") return OCR_MODEL_GLM_OCR;
    if (arch == "got_ocr") return OCR_MODEL_GOT_OCR;
    if (arch == "tesseract_lstm") return OCR_MODEL_TESSERACT_LSTM;
    if (arch == "granite_vision") return OCR_MODEL_GRANITE_VISION;
    if (arch == "lightonocr") return OCR_MODEL_LIGHTONOCR;
    if (arch == "deepseek_ocr2") return OCR_MODEL_DEEPSEEK_OCR2;
    if (arch == "smoldocling") return OCR_MODEL_SMOLDOCLING;
    if (arch == "math_ocr") return OCR_MODEL_PIX2TEX;
    if (arch == "unlimited_ocr") return OCR_MODEL_UNLIMITED_OCR;
    if (arch == "smt_ocr") return OCR_MODEL_SMT;
    if (arch == "tromr_ocr") return OCR_MODEL_TROMR;
    if (arch == "flova_ocr") return OCR_MODEL_FLOVA;
    if (arch == "transcoda_ocr") return OCR_MODEL_TRANSCODA;
    if (arch == "ppocrv6") return OCR_MODEL_PPOCRV6;
    return OCR_MODEL_PIX2TEX;
}

extern "C" void * crispembed_ocr_model_init(const char * path, int n_threads) {
    if (ocr_pipeline::is_dangerous_q4_recognizer_path(path) && !ocr_pipeline::dangerous_q4_override_enabled()) {
        fprintf(stderr,
                "crispembed_ocr_model: refusing TrOCR Q4_K model '%s'; use Q8_0 or explicitly set "
                "CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1\n",
                path ? path : "(null)");
        return nullptr;
    }
    auto type = detect_arch(path);
    void * inner = nullptr;
    switch (type) {
    case OCR_MODEL_PIX2TEX:
        inner = math_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_HMER:
        inner = hmer_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_BTTR:
        inner = bttr_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_PPFORMULANET:
        inner = ppformulanet_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_PPFORMULANET_L:
        inner = ppformulanet_l_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_POSFORMER:
        inner = posformer_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_MIXTEX:
        inner = mixtex_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_QWEN2VL:
        inner = qwen2vl_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_INTERNVL2:
        inner = internvl2_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_PARSEQ:
        inner = parseq_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_GLM_OCR:
        inner = glm_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_GOT_OCR:
        inner = got_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_TESSERACT_LSTM:
        inner = tesseract_lstm_init(path, n_threads);
        break;
    case OCR_MODEL_GRANITE_VISION:
        inner = granite_vision_init(path, n_threads);
        break;
    case OCR_MODEL_LIGHTONOCR:
        inner = lightonocr_init(path, n_threads);
        break;
    case OCR_MODEL_DEEPSEEK_OCR2:
        inner = deepseek_ocr2_init(path, n_threads);
        break;
    case OCR_MODEL_SMOLDOCLING:
        inner = smoldocling_init(path, n_threads);
        break;
    case OCR_MODEL_UNLIMITED_OCR:
        inner = unlimited_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_SMT:
        inner = smt_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_TROMR:
        inner = tromr_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_FLOVA:
        inner = flova_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_TRANSCODA:
        inner = transcoda_ocr_init(path, n_threads);
        break;
    case OCR_MODEL_PPOCRV6:
        inner = ppocrv6_ocr_init(path, n_threads);
        break;
    }
    if (!inner) return nullptr;
    auto * u = new ocr_model{ type, inner };
    return u;
}

extern "C" void crispembed_ocr_model_free(void * ctx) {
    if (!ctx) return;
    auto * u = (ocr_model *)ctx;
    switch (u->type) {
    case OCR_MODEL_PIX2TEX:
        math_ocr_free((math_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_HMER:
        hmer_ocr_free((hmer_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_BTTR:
        bttr_ocr_free((bttr_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_PPFORMULANET:
        ppformulanet_ocr_free((ppformulanet_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_PPFORMULANET_L:
        ppformulanet_l_ocr_free((ppformulanet_l_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_POSFORMER:
        posformer_ocr_free((posformer_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_MIXTEX:
        mixtex_ocr_free((mixtex_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_QWEN2VL:
        qwen2vl_ocr_free((qwen2vl_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_INTERNVL2:
        internvl2_ocr_free((internvl2_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_PARSEQ:
        parseq_ocr_free((parseq_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_GLM_OCR:
        glm_ocr_free((glm_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_GOT_OCR:
        got_ocr_free((got_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_TESSERACT_LSTM:
        tesseract_lstm_free((tesseract_lstm_context *)u->ctx);
        break;
    case OCR_MODEL_GRANITE_VISION:
        granite_vision_free((granite_vision_context *)u->ctx);
        break;
    case OCR_MODEL_LIGHTONOCR:
        lightonocr_free((lightonocr_context *)u->ctx);
        break;
    case OCR_MODEL_DEEPSEEK_OCR2:
        deepseek_ocr2_free((deepseek_ocr2_context *)u->ctx);
        break;
    case OCR_MODEL_SMOLDOCLING:
        smoldocling_free((smoldocling_context *)u->ctx);
        break;
    case OCR_MODEL_UNLIMITED_OCR:
        unlimited_ocr_free((unlimited_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_SMT:
        smt_ocr_free((smt_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_FLOVA:
        flova_ocr_free((flova_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_TROMR:
        tromr_ocr_free((tromr_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_TRANSCODA:
        transcoda_ocr_free((transcoda_ocr_context *)u->ctx);
        break;
    case OCR_MODEL_PPOCRV6:
        ppocrv6_ocr_free((ppocrv6_ocr_context *)u->ctx);
        break;
    }
    delete u;
}

extern "C" const char * crispembed_ocr_model_recognize(void * ctx, const uint8_t * px, int w, int h, int ch, int * ol) {
    if (!ctx) return nullptr;
    auto * u = (ocr_model *)ctx;
    switch (u->type) {
    case OCR_MODEL_PIX2TEX:
        return math_ocr_recognize_raw((math_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_HMER:
        return hmer_ocr_recognize_raw((hmer_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_BTTR:
        return bttr_ocr_recognize_raw((bttr_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_PPFORMULANET:
        return ppformulanet_ocr_recognize_raw((ppformulanet_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_PPFORMULANET_L:
        return ppformulanet_l_ocr_recognize_raw((ppformulanet_l_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_POSFORMER:
        return posformer_ocr_recognize_raw((posformer_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_MIXTEX:
        return mixtex_ocr_recognize((mixtex_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_QWEN2VL:
        return qwen2vl_ocr_recognize_raw((qwen2vl_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_INTERNVL2:
        return internvl2_ocr_recognize_raw((internvl2_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_PARSEQ:
        return parseq_ocr_recognize_raw((parseq_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_GLM_OCR:
        return glm_ocr_recognize_raw((glm_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_GOT_OCR:
        return got_ocr_recognize_raw((got_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_TESSERACT_LSTM: {
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
    case OCR_MODEL_GRANITE_VISION:
        return granite_vision_recognize((granite_vision_context *)u->ctx, px, w, h, ch, nullptr, ol);
    case OCR_MODEL_LIGHTONOCR:
        return lightonocr_recognize_raw((lightonocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_DEEPSEEK_OCR2:
        return deepseek_ocr2_recognize_raw((deepseek_ocr2_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_SMOLDOCLING:
        return smoldocling_recognize_raw((smoldocling_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_UNLIMITED_OCR:
        return unlimited_ocr_recognize_raw((unlimited_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_SMT:
        return smt_ocr_recognize_raw((smt_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_TROMR:
        return tromr_ocr_recognize_raw((tromr_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_FLOVA:
        return flova_ocr_recognize_raw((flova_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_TRANSCODA:
        return transcoda_ocr_recognize_raw((transcoda_ocr_context *)u->ctx, px, w, h, ch, ol);
    case OCR_MODEL_PPOCRV6:
        return ppocrv6_ocr_recognize_raw((ppocrv6_ocr_context *)u->ctx, px, w, h, ch, ol);
    }
    return nullptr;
}

extern "C" const char * crispembed_ocr_model_recognize_gray(void * ctx, const float * px, int w, int h, int * ol) {
    if (!ctx) return nullptr;
    auto * u = (ocr_model *)ctx;
    switch (u->type) {
    case OCR_MODEL_PIX2TEX:
        return math_ocr_recognize((math_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_HMER:
        return hmer_ocr_recognize((hmer_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_BTTR:
        return bttr_ocr_recognize((bttr_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_PPFORMULANET:
        return ppformulanet_ocr_recognize((ppformulanet_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_PPFORMULANET_L:
        return ppformulanet_l_ocr_recognize((ppformulanet_l_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_POSFORMER:
        return posformer_ocr_recognize((posformer_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_MIXTEX:
        return mixtex_ocr_recognize_gray((mixtex_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_QWEN2VL:
        return qwen2vl_ocr_recognize((qwen2vl_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_INTERNVL2:
        return internvl2_ocr_recognize((internvl2_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_PARSEQ:
        return parseq_ocr_recognize((parseq_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_GLM_OCR:
        return glm_ocr_recognize((glm_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_GOT_OCR:
        return got_ocr_recognize((got_ocr_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_DEEPSEEK_OCR2:
        return deepseek_ocr2_recognize((deepseek_ocr2_context *)u->ctx, px, w, h, ol);
    case OCR_MODEL_TESSERACT_LSTM: {
        // Convert float [0,1] grayscale → uint8
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return tesseract_lstm_recognize((tesseract_lstm_context *)u->ctx, gray.data(), w, h, ol);
    }
    case OCR_MODEL_GRANITE_VISION: {
        // Convert float gray → uint8 RGB for granite_vision_recognize
        std::vector<uint8_t> rgb(w * h * 3);
        for (int i = 0; i < w * h; i++) {
            uint8_t v = (uint8_t)(px[i] * 255.0f + 0.5f);
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
        }
        return granite_vision_recognize((granite_vision_context *)u->ctx, rgb.data(), w, h, 3, nullptr, ol);
    }
    case OCR_MODEL_LIGHTONOCR: {
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return lightonocr_recognize_raw((lightonocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    case OCR_MODEL_SMOLDOCLING: {
        std::vector<uint8_t> rgb(w * h * 3);
        for (int i = 0; i < w * h; i++) {
            uint8_t v = (uint8_t)(px[i] * 255.0f + 0.5f);
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = v;
        }
        return smoldocling_recognize_raw((smoldocling_context *)u->ctx, rgb.data(), w, h, 3, ol);
    }
    case OCR_MODEL_SMT: {
        // SMT preprocessing (invert+resize) happens in recognize_raw; hand it a
        // 1-channel uint8 image built from the [0,1] grayscale floats.
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return smt_ocr_recognize_raw((smt_ocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    case OCR_MODEL_TROMR: {
        // TrOMR preprocessing (resize+gray+normalize) happens in recognize_raw;
        // hand it a 1-channel uint8 image built from the [0,1] grayscale floats.
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return tromr_ocr_recognize_raw((tromr_ocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    case OCR_MODEL_FLOVA: {
        // Flova's Donut preprocessing runs in recognize_raw; hand it a 1-channel
        // uint8 image built from the [0,1] grayscale floats.
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return flova_ocr_recognize_raw((flova_ocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    case OCR_MODEL_TRANSCODA: {
        // Transcoda's full-page preprocessing (resize/pad + [-1,1]) runs in
        // recognize_raw; hand it a 1-channel uint8 image (broadcast to RGB there).
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)(px[i] * 255.0f + 0.5f);
        return transcoda_ocr_recognize_raw((transcoda_ocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    case OCR_MODEL_PPOCRV6: {
        std::vector<uint8_t> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = (uint8_t)std::clamp(int(px[i] * 255.0f + 0.5f), 0, 255);
        return ppocrv6_ocr_recognize_raw((ppocrv6_ocr_context *)u->ctx, gray.data(), w, h, 1, ol);
    }
    }
    return nullptr;
}

extern "C" const float * crispembed_ocr_model_confidences(const void * ctx, int * n_tokens) {
    if (n_tokens) *n_tokens = 0;
    if (!ctx) return nullptr;
    auto * u = (const ocr_model *)ctx;
    switch (u->type) {
    case OCR_MODEL_PIX2TEX:
        return math_ocr_confidences((const math_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_HMER:
        return hmer_ocr_confidences((const hmer_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_BTTR:
        return bttr_ocr_confidences((const bttr_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_PPFORMULANET:
        return ppformulanet_ocr_confidences((const ppformulanet_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_PPFORMULANET_L:
        return ppformulanet_l_ocr_confidences((const ppformulanet_l_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_POSFORMER:
        return posformer_ocr_confidences((const posformer_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_MIXTEX:
        return mixtex_ocr_confidences((const mixtex_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_QWEN2VL:
        return qwen2vl_ocr_confidences((const qwen2vl_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_INTERNVL2:
        return internvl2_ocr_confidences((const internvl2_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_PARSEQ:
        return parseq_ocr_confidences((const parseq_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_GLM_OCR:
        return glm_ocr_confidences((const glm_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_GOT_OCR:
        return got_ocr_confidences((const got_ocr_context *)u->ctx, n_tokens);
    case OCR_MODEL_DEEPSEEK_OCR2:
        return deepseek_ocr2_confidences((const deepseek_ocr2_context *)u->ctx, n_tokens);
    case OCR_MODEL_TESSERACT_LSTM:
        return tesseract_lstm_confidences((const tesseract_lstm_context *)u->ctx, n_tokens);
    case OCR_MODEL_GRANITE_VISION:
        return granite_vision_confidences((const granite_vision_context *)u->ctx, n_tokens);
    case OCR_MODEL_LIGHTONOCR:
        return lightonocr_confidences((const lightonocr_context *)u->ctx, n_tokens);
    default:
        return nullptr;
    }
}

extern "C" float crispembed_ocr_model_mean_confidence(const void * ctx) {
    if (!ctx) return 0.0f;
    int n = 0;
    const float * c = crispembed_ocr_model_confidences(ctx, &n);
    if (!c || n <= 0) return 0.0f;
    double sum = 0;
    for (int i = 0; i < n; i++) sum += c[i];
    return (float)(sum / n);
}

extern "C" void crispembed_ocr_model_set_max_tokens(void * ctx, int max_tokens) {
    if (!ctx || max_tokens <= 0) return;
    auto * u = (ocr_model *)ctx;
    switch (u->type) {
    case OCR_MODEL_QWEN2VL:
        qwen2vl_ocr_set_max_tokens((qwen2vl_ocr_context *)u->ctx, max_tokens);
        break;
    case OCR_MODEL_INTERNVL2:
        internvl2_ocr_set_max_tokens((internvl2_ocr_context *)u->ctx, max_tokens);
        break;
    case OCR_MODEL_GRANITE_VISION:
        granite_vision_set_max_tokens((granite_vision_context *)u->ctx, max_tokens);
        break;
    case OCR_MODEL_LIGHTONOCR:
        lightonocr_set_max_tokens((lightonocr_context *)u->ctx, max_tokens);
        break;
    case OCR_MODEL_SMOLDOCLING:
        smoldocling_set_max_tokens((smoldocling_context *)u->ctx, max_tokens);
        break;
    default:
        break; // formula OCR engines: no-op
    }
}

// --- Deprecated aliases -----------------------------------------------------
// The dispatcher was originally named crispembed_math_ocr_* but now handles
// general text/document OCR as well as math. The crispembed_ocr_model_* names
// above are canonical; these thin forwarders preserve ABI compatibility for
// existing callers and will be removed in a future major release.
extern "C" void * crispembed_math_ocr_init(const char * path, int n_threads) {
    return crispembed_ocr_model_init(path, n_threads);
}
extern "C" void crispembed_math_ocr_free(void * ctx) {
    crispembed_ocr_model_free(ctx);
}
extern "C" const char * crispembed_math_ocr_recognize(void * ctx, const uint8_t * px, int w, int h, int ch, int * ol) {
    return crispembed_ocr_model_recognize(ctx, px, w, h, ch, ol);
}
extern "C" const char * crispembed_math_ocr_recognize_gray(void * ctx, const float * px, int w, int h, int * ol) {
    return crispembed_ocr_model_recognize_gray(ctx, px, w, h, ol);
}
extern "C" const float * crispembed_math_ocr_confidences(const void * ctx, int * n_tokens) {
    return crispembed_ocr_model_confidences(ctx, n_tokens);
}
extern "C" float crispembed_math_ocr_mean_confidence(const void * ctx) {
    return crispembed_ocr_model_mean_confidence(ctx);
}

// Also expose individual APIs for direct use
extern "C" void * crispembed_hmer_ocr_init(const char * p, int t) {
    return hmer_ocr_init(p, t);
}
extern "C" void crispembed_hmer_ocr_free(void * c) {
    hmer_ocr_free((hmer_ocr_context *)c);
}
extern "C" const char * crispembed_hmer_ocr_recognize(void * c, const uint8_t * px, int w, int h, int ch, int * ol) {
    return hmer_ocr_recognize_raw((hmer_ocr_context *)c, px, w, h, ch, ol);
}
extern "C" const char * crispembed_hmer_ocr_recognize_gray(void * c, const float * px, int w, int h, int * ol) {
    return hmer_ocr_recognize((hmer_ocr_context *)c, px, w, h, ol);
}

extern "C" void * crispembed_bttr_ocr_init(const char * p, int t) {
    return bttr_ocr_init(p, t);
}
extern "C" void crispembed_bttr_ocr_free(void * c) {
    bttr_ocr_free((bttr_ocr_context *)c);
}
extern "C" const char * crispembed_bttr_ocr_recognize(void * c, const uint8_t * px, int w, int h, int ch, int * ol) {
    return bttr_ocr_recognize_raw((bttr_ocr_context *)c, px, w, h, ch, ol);
}
extern "C" const char * crispembed_bttr_ocr_recognize_gray(void * c, const float * px, int w, int h, int * ol) {
    return bttr_ocr_recognize((bttr_ocr_context *)c, px, w, h, ol);
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

extern "C" const char * crispembed_pix2struct_generate(crispembed_pix2struct_context * ctx, const uint8_t * image,
                                                       int width, int height, int max_tokens) {
    if (!ctx) return nullptr;
    return pix2struct_generate((pix2struct_context *)ctx, image, width, height, max_tokens);
}

extern "C" void crispembed_pix2struct_free_text(const char * text) {
    pix2struct_free_text(text);
}

extern "C" const float * crispembed_pix2struct_confidences(const crispembed_pix2struct_context * ctx, int * n_tokens) {
    if (!ctx) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    return pix2struct_confidences((const pix2struct_context *)ctx, n_tokens);
}

extern "C" float crispembed_pix2struct_mean_confidence(const crispembed_pix2struct_context * ctx) {
    if (!ctx) return 0.0f;
    return pix2struct_mean_confidence((const pix2struct_context *)ctx);
}

extern "C" const float * crispembed_pix2struct_encode_patches(crispembed_pix2struct_context * ctx,
                                                              const float * patches, int n_patches, int * out_dim) {
    if (!ctx) return nullptr;
    return pix2struct_encode_patches((pix2struct_context *)ctx, patches, n_patches, out_dim);
}

// ---------------------------------------------------------------------------
// Granite Vision OCR wrappers
// ---------------------------------------------------------------------------

extern "C" crispembed_granite_vision_context * crispembed_granite_vision_init(const char * model_path, int n_threads) {
    return (crispembed_granite_vision_context *)granite_vision_init(model_path, n_threads);
}

extern "C" void crispembed_granite_vision_free(crispembed_granite_vision_context * ctx) {
    if (ctx) granite_vision_free((granite_vision_context *)ctx);
}

extern "C" const char * crispembed_granite_vision_recognize(crispembed_granite_vision_context * ctx,
                                                            const uint8_t * pixels, int width, int height, int channels,
                                                            const char * prompt, int * out_len) {
    if (!ctx) return nullptr;
    return granite_vision_recognize((granite_vision_context *)ctx, pixels, width, height, channels, prompt, out_len);
}

// ---------------------------------------------------------------------------
// LightOnOCR wrappers
// ---------------------------------------------------------------------------

extern "C" crispembed_lightonocr_context * crispembed_lightonocr_init(const char * model_path, int n_threads) {
    return (crispembed_lightonocr_context *)lightonocr_init(model_path, n_threads);
}

extern "C" void crispembed_lightonocr_free(crispembed_lightonocr_context * ctx) {
    if (ctx) lightonocr_free((lightonocr_context *)ctx);
}

extern "C" const char * crispembed_lightonocr_recognize(crispembed_lightonocr_context * ctx, const uint8_t * pixels,
                                                        int width, int height, int channels, int * out_len) {
    if (!ctx) return nullptr;
    return lightonocr_recognize_raw((lightonocr_context *)ctx, pixels, width, height, channels, out_len);
}

extern "C" void crispembed_free(crispembed_context * ctx) {
    if (!ctx) return;
    // Flush the importance matrix here (not just at atexit): one-shot binaries
    // exit via core_util::clean_exit() -> _exit(), which bypasses atexit handlers.
    // No-op unless CRISPEMBED_IMATRIX_OUT is set; idempotent within a process.
    crispembed_imatrix_flush();
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
    if (ctx->lfm2_ctx) {
        lfm2_embed_free(ctx->lfm2_ctx);
        ctx->lfm2_ctx = nullptr;
    }
    if (ctx->qkv_buf) {
        ggml_backend_buffer_free(ctx->qkv_buf);
        ctx->qkv_buf = nullptr;
    }
    if (ctx->qkv_ctx) {
        ggml_free(ctx->qkv_ctx);
        ctx->qkv_ctx = nullptr;
    }
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
#include "ocr_pipeline_pool.h"
#include "ocr_orchestrator.h"
#include "layout_detect.h"

struct ocr_pipeline_wrapper {
    ocr_pipeline_pool::context * pool = nullptr;
    ocr_orchestrator::context * pp_ctx = nullptr;
    bool is_ppocrv6 = false;
    std::vector<ocr_pipeline::ocr_result> results;
    std::vector<crispembed_ocr_result> c_results;
    std::vector<crispembed_ocr_stage_metric> c_stage_metrics;
    std::vector<std::string> text_storage;
    std::string rec_buf;
};

extern "C" void * crispembed_ocr_init(const char * det_path, const char * rec_path, int n_threads) {
    auto * w = new ocr_pipeline_wrapper();
    auto * meta = core_gguf::open_metadata(det_path);
    const bool pp = meta && core_gguf::kv_str(meta, "ppocrv6.kind", "") == "det";
    if (meta) core_gguf::free_metadata(meta);
    // The rec slot dispatches on metadata too: a Tesseract-LSTM GGUF routes
    // to the orchestrator's tesseract stage (per-crop line recognition).
    // Before this it fell through to the flat pipeline's math_ocr loader,
    // which mis-dispatches the arch (vocab=1200) and crashes on region 1
    // (the T11 finding). Combined with the ppocrv6-capable det slot this is
    // the page-level CJK path: ppocrv6 det line boxes + tesseract-jpn crops.
    auto * rmeta = core_gguf::open_metadata(rec_path);
    const bool rec_is_tesseract = rmeta && core_gguf::kv_str(rmeta, "general.architecture", "") == "tesseract_lstm";
    if (rmeta) core_gguf::free_metadata(rmeta);
    if (pp || rec_is_tesseract) {
        ocr_orchestrator::config cfg;
        cfg.router = false;
        ocr_orchestrator::chain ch;
        ch.type = ocr_orchestrator::source_type::auto_detect;
        ocr_orchestrator::stage st;
        st.eng = rec_is_tesseract ? ocr_orchestrator::engine::tesseract : ocr_orchestrator::engine::ppocrv6;
        st.cleanup.enabled = false;
        st.model_a = det_path;
        st.model_b = rec_path;
        ch.stages.push_back(std::move(st));
        cfg.chains.push_back(std::move(ch));
        if (!ocr_orchestrator::load(&w->pp_ctx, cfg, n_threads)) {
            delete w;
            return nullptr;
        }
        w->is_ppocrv6 = true;
        return w;
    }
    int pool_size = 1;
    if (const char * env = std::getenv("CRISPEMBED_OCR_POOL_SIZE")) {
        char * end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && parsed >= 1 && parsed <= 64) pool_size = (int)parsed;
    }
    if (!ocr_pipeline_pool::load(&w->pool, det_path, rec_path, pool_size, n_threads)) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C" void crispembed_ocr_free(void * ctx) {
    if (!ctx) return;
    auto * w = (ocr_pipeline_wrapper *)ctx;
    if (w->pp_ctx) ocr_orchestrator::free(w->pp_ctx);
    if (w->pool) ocr_pipeline_pool::free(w->pool);
    delete w;
}

extern "C" const crispembed_ocr_result * crispembed_ocr(void * ctx, const char * image_path, int * out_n) {
    if (!ctx || !image_path) {
        if (out_n) *out_n = 0;
        return nullptr;
    }
    auto * w = (ocr_pipeline_wrapper *)ctx;
    if (w->is_ppocrv6) {
        auto result = ocr_orchestrator::run_file(w->pp_ctx, image_path);
        w->c_results.resize(result.regions.size());
        w->text_storage.resize(result.regions.size());
        for (size_t i = 0; i < result.regions.size(); ++i) {
            const auto & r = result.regions[i];
            auto & c = w->c_results[i];
            c.x = r.box.x;
            c.y = r.box.y;
            c.w = r.box.w;
            c.h = r.box.h;
            c.confidence = r.confidence;
            w->text_storage[i] = r.text;
            c.text = w->text_storage[i].c_str();
            c.text_len = (int)w->text_storage[i].size();
            c.orientation_corrected = r.orientation_corrected ? 1 : 0;
            c.orientation_angle = r.orientation_angle;
            c.orientation_confidence = r.orientation_confidence;
        }
        if (out_n) *out_n = (int)w->c_results.size();
        return w->c_results.empty() ? nullptr : w->c_results.data();
    }
    w->results = ocr_pipeline_pool::run_file(w->pool, image_path);
    w->c_results.resize(w->results.size());
    for (size_t i = 0; i < w->results.size(); i++) {
        auto & r = w->results[i];
        auto & c = w->c_results[i];
        c.x = r.box.x;
        c.y = r.box.y;
        c.w = r.box.w;
        c.h = r.box.h;
        c.confidence = r.confidence;
        c.text = r.text.c_str();
        c.text_len = (int)r.text.size();
        c.orientation_corrected = r.orientation_corrected ? 1 : 0;
        c.orientation_angle = r.orientation_angle;
        c.orientation_confidence = r.orientation_confidence;
    }
    if (out_n) *out_n = (int)w->c_results.size();
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

extern "C" const char * crispembed_ocr_recognize(void * ctx, const char * image_path, int * out_len) {
    if (!ctx || !image_path) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    auto * w = (ocr_pipeline_wrapper *)ctx;
    if (w->is_ppocrv6) {
        auto result = ocr_orchestrator::run_file(w->pp_ctx, image_path);
        w->rec_buf = result.full_text;
        if (out_len) *out_len = (int)w->rec_buf.size();
        return w->rec_buf.empty() ? nullptr : w->rec_buf.c_str();
    } else {
        w->rec_buf = ocr_pipeline_pool::recognize_file(w->pool, image_path);
    }
    if (out_len) *out_len = (int)w->rec_buf.size();
    return w->rec_buf.empty() ? nullptr : w->rec_buf.c_str();
}

// ---------------------------------------------------------------------------
// OCR Pipeline (orchestrator) — see ocr_orchestrator.{h,cpp}
// ---------------------------------------------------------------------------

struct ocr_pipeline_orch_wrapper {
    ocr_orchestrator::context * ctx = nullptr;
    ocr_orchestrator::result last;
    std::vector<crispembed_ocr_result> c_results;
    std::vector<crispembed_ocr_stage_metric> c_stage_metrics;
    std::string full_text;
    std::string markdown;
    void * punct = nullptr; // optional post-OCR punctuation/spacing restorer
};

extern "C" crispembed_ocr_pipeline_params crispembed_ocr_pipeline_defaults(void) {
    crispembed_ocr_pipeline_params p;
    p.router = 1;
    p.cleanup_enabled = 1;
    p.min_chars = 8;
    p.min_confidence = 0.5f;
    p.det_model = nullptr;
    p.rec_model = nullptr;
    p.nafnet_model = nullptr;
    p.sr_model = nullptr;
    p.vlm_model = nullptr;
    p.vlm_engine = 0;
    p.punct_model = nullptr;
    p.lid_model = nullptr;
    p.truecase_model = nullptr;
    p.tess_model_dir = nullptr;
    p.layout_model = nullptr;
    p.table_model = nullptr;
    p.formula_model = nullptr;
    p.route_tables = 0;
    p.route_formulas = 0;
    p.image_text_fallback = 1;
    return p;
}

extern "C" void * crispembed_ocr_pipeline_init(const crispembed_ocr_pipeline_params * params, int n_threads) {
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
    case 1:
        vlm_eng = ocr_orchestrator::engine::glm;
        break;
    case 2:
        vlm_eng = ocr_orchestrator::engine::qwen2vl;
        break;
    case 3:
        vlm_eng = ocr_orchestrator::engine::internvl2;
        break;
    case 4:
        vlm_eng = ocr_orchestrator::engine::qwen3vl;
        break;
    default:
        vlm_eng = ocr_orchestrator::engine::got;
        break;
    }
    const bool has_vlm = params->vlm_model && *params->vlm_model;
    for (auto & ch : cfg.chains) {
        for (auto & st : ch.stages) {
            if (params->det_model) st.model_a = params->det_model;
            if (params->rec_model) st.model_b = params->rec_model;
            st.accept.min_chars = params->min_chars;
            st.accept.min_confidence = params->min_confidence;
            if (!params->cleanup_enabled) st.cleanup.enabled = false;
        }
        // Append a single-shot VLM escalation stage: the chain tries the fast
        // DBNet+TrOCR first and falls back to the VLM when the accept-gate fails.
        if (has_vlm) {
            ocr_orchestrator::stage vs;
            vs.eng = vlm_eng;
            vs.enabled = true;
            vs.model_a = params->vlm_model;
            vs.accept.min_chars = params->min_chars;
            vs.accept.min_confidence = 0.0f; // VLM has no per-region confidence
            vs.cleanup.enabled = params->cleanup_enabled != 0;
            vs.cleanup.params = scan_cleanup_defaults();
            vs.cleanup.params.binarize = 0; // never binarize for a VLM
            vs.cleanup.denoise = false;
            ch.stages.push_back(vs);
        }
    }
    // LID + truecasing + Tesseract auto-select
    if (params->lid_model && *params->lid_model) cfg.lid_model = params->lid_model;
    if (params->truecase_model && *params->truecase_model) cfg.truecase_model = params->truecase_model;
    if (params->tess_model_dir && *params->tess_model_dir) cfg.tess_model_dir = params->tess_model_dir;
    if (params->layout_model && *params->layout_model) cfg.layout_model = params->layout_model;
    if (params->table_model && *params->table_model) cfg.table_model = params->table_model;
    if (params->formula_model && *params->formula_model) cfg.formula_model = params->formula_model;
    cfg.route_tables = params->route_tables != 0;
    cfg.route_formulas = params->route_formulas != 0;
    cfg.image_text_fallback = params->image_text_fallback != 0;

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

extern "C" const crispembed_ocr_result * crispembed_ocr_pipeline_run(void * ctx, const char * image_path, int * out_n,
                                                                     const char ** out_full_text,
                                                                     float * out_mean_conf) {
    if (out_n) *out_n = 0;
    if (out_full_text) *out_full_text = nullptr;
    if (out_mean_conf) *out_mean_conf = 0.0f;
    if (!ctx || !image_path) return nullptr;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    w->last = ocr_orchestrator::run_file(w->ctx, image_path);
    w->full_text = w->last.full_text;
    w->markdown = w->last.markdown;
    // Optional post-OCR restore: punctuation / capitalization / spacing.
    if (w->punct && !w->full_text.empty()) {
        const char * restored = crispembed_punct_process(w->punct, w->full_text.c_str());
        if (restored && *restored) w->full_text = restored;
    }
    w->c_results.resize(w->last.regions.size());
    for (size_t i = 0; i < w->last.regions.size(); i++) {
        auto & r = w->last.regions[i];
        auto & c = w->c_results[i];
        c.x = r.box.x;
        c.y = r.box.y;
        c.w = r.box.w;
        c.h = r.box.h;
        c.confidence = r.confidence;
        c.text = r.text.c_str();
        c.text_len = (int)r.text.size();
        c.orientation_corrected = r.orientation_corrected ? 1 : 0;
        c.orientation_angle = r.orientation_angle;
        c.orientation_confidence = r.orientation_confidence;
    }
    if (out_n) *out_n = (int)w->c_results.size();
    if (out_full_text) *out_full_text = w->full_text.c_str();
    if (out_mean_conf) *out_mean_conf = w->last.mean_confidence;
    return w->c_results.empty() ? nullptr : w->c_results.data();
}

extern "C" const int * crispembed_ocr_pipeline_reading_order(void * ctx, int * out_n) {
    if (out_n) *out_n = 0;
    if (!ctx) return nullptr;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (out_n) *out_n = (int)w->last.reading_order.size();
    return w->last.reading_order.empty() ? nullptr : w->last.reading_order.data();
}

extern "C" const crispembed_ocr_stage_metric * crispembed_ocr_pipeline_stage_metrics(void * ctx, int * out_n) {
    if (out_n) *out_n = 0;
    if (!ctx) return nullptr;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    w->c_stage_metrics.clear();
    w->c_stage_metrics.reserve(w->last.stage_metrics.size());
    for (const auto & m : w->last.stage_metrics) {
        w->c_stage_metrics.push_back({ m.index, m.engine.c_str(), m.elapsed_ms, m.cleanup_applied ? 1 : 0,
                                       m.accepted ? 1 : 0, m.text_chars, m.mean_confidence });
    }
    if (out_n) *out_n = (int)w->c_stage_metrics.size();
    return w->c_stage_metrics.empty() ? nullptr : w->c_stage_metrics.data();
}

extern "C" const char * crispembed_ocr_pipeline_markdown(void * ctx, int * out_len) {
    if (out_len) *out_len = 0;
    if (!ctx) return nullptr;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (out_len) *out_len = (int)w->markdown.size();
    return w->markdown.empty() ? nullptr : w->markdown.c_str();
}

static ocr_orchestrator::engine map_engine(int e) {
    using E = ocr_orchestrator::engine;
    switch (e) {
    case 0:
        return E::dbnet_trocr;
    case 1:
        return E::surya;
    case 2:
        return E::got;
    case 3:
        return E::glm;
    case 4:
        return E::qwen2vl;
    case 5:
        return E::internvl2;
    case 6:
        return E::tesseract;
    case 7:
        return E::parseq;
    case 8:
        return E::deepseek_ocr2;
    case 9:
        return E::pix2struct;
    case 10:
        return E::granite_vision;
    case 11:
        return E::lightonocr;
    case 12:
        return E::qwen3vl;
    case 13:
        return E::unlimited_ocr;
    case 14:
        return E::unified;
    case 15:
        return E::tesseract_fraktur;
    case 16:
        return E::ppocrv6;
    case 17:
        return E::easyocr;
    case 18:
        return E::olmocr;
    default:
        return E::dbnet_trocr;
    }
}

static ocr_orchestrator::source_type map_source(int s) {
    using S = ocr_orchestrator::source_type;
    switch (s) {
    case 1:
        return S::screenshot;
    case 2:
        return S::scanned_doc;
    case 3:
        return S::photo;
    default:
        return S::auto_detect;
    }
}

// crispembed_scan_cleanup_params (C API) and scan_cleanup_params (internal)
// share the same field layout; copy field-by-field across the type boundary.
static scan_cleanup_params to_cleanup(const crispembed_scan_cleanup_params & p) {
    scan_cleanup_params o = scan_cleanup_defaults(); // start from defaults so any
                                                     // fields not in the C API struct are sane
    o.deskew = p.deskew;
    o.crop_borders = p.crop_borders;
    o.whiten_background = p.whiten_background;
    o.binarize = p.binarize;
    o.binarize_method = p.binarize_method;
    o.sauvola_k = p.sauvola_k;
    o.sauvola_window = p.sauvola_window;
    o.morph_kernel = p.morph_kernel;
    o.border_threshold = p.border_threshold;
    o.deskew_max_angle = p.deskew_max_angle;
    o.despeckle = p.despeckle;
    o.despeckle_thresh = p.despeckle_thresh;
    o.blackfilter = p.blackfilter;
    o.blackfilter_thresh = p.blackfilter_thresh;
    o.deskew_consensus = p.deskew_consensus;
    return o;
}

extern "C" void * crispembed_ocr_pipeline_init_stages(int router, const char * nafnet_model, const char * sr_model,
                                                      const char * punct_model, const char * lid_model,
                                                      const char * truecase_model, const char * tess_model_dir,
                                                      const crispembed_ocr_stage * stages, int n_stages,
                                                      int n_threads) {
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
        st.eng = map_engine(s.engine);
        st.enabled = true;
        if (s.model_a) st.model_a = s.model_a;
        if (s.model_b) st.model_b = s.model_b;
        if (s.model_c) st.model_c = s.model_c;
        st.cleanup.enabled = s.cleanup_enabled != 0;
        st.cleanup.params = to_cleanup(s.cleanup);
        st.cleanup.denoise = s.denoise != 0;
        st.params.det_prob_threshold = s.det_prob_threshold;
        st.params.det_box_threshold = s.det_box_threshold;
        st.params.det_target_short = s.det_target_short > 0 ? s.det_target_short : 736;
        st.params.det_max_side = s.det_max_side > 0 ? s.det_max_side : 2000;
        st.params.det_min_height = s.det_min_height > 0 ? s.det_min_height : 30;
        st.params.det_width_height_ratio = s.det_width_height_ratio == 0.0f ? 8.0f : s.det_width_height_ratio;
        st.params.det_max_candidates = s.det_max_candidates == 0 ? 1000 : s.det_max_candidates;
        st.params.det_dilation = s.det_dilation == 0 ? 1 : s.det_dilation;
        st.params.det_scoring = s.det_score_mode == 1 ? ocr_detect::score_mode::accurate : ocr_detect::score_mode::fast;
        st.params.vlm_max_tokens = s.vlm_max_tokens;
        st.params.page_segmentation = s.page_segmentation;
        if (s.vlm_prompt && *s.vlm_prompt) st.params.vlm_prompt = s.vlm_prompt;
        st.accept.min_chars = s.min_chars;
        st.accept.min_confidence = s.min_confidence;

        const ocr_orchestrator::source_type type = map_source(s.source_type);
        ocr_orchestrator::chain * chain = nullptr;
        for (auto & c : cfg.chains) {
            if (c.type == type) {
                chain = &c;
                break;
            }
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
    if (!ctx) {
        if (out_confidence) *out_confidence = 0.0f;
        return "";
    }
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    if (out_confidence) *out_confidence = w->last.lang_confidence;
    return w->last.detected_lang.c_str();
}

extern "C" int crispembed_ocr_pipeline_capabilities(void * ctx, crispembed_ocr_capabilities * out) {
    if (!out) return 0;
    *out = {};
    if (!ctx) return 0;
    auto * w = (ocr_pipeline_orch_wrapper *)ctx;
    const auto caps = ocr_orchestrator::get_capabilities(w->ctx);
    out->layout = caps.layout;
    out->tables = caps.tables;
    out->formulas = caps.formulas;
    out->image_text_fallback = caps.image_text_fallback;
    return 1;
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
extern "C" const float * crispembed_ocr_pipeline_region_char_conf(void * ctx, int region_idx, int * out_len) {
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
    (void)model_path;
    (void)n_threads;
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
    if (!ctx || !text) {
        if (out_confidence) *out_confidence = 0;
        return "";
    }
    return text_lid_predict((text_lid_context *)ctx, text, out_confidence);
#else
    (void)ctx;
    (void)text;
    if (out_confidence) *out_confidence = 0;
    return "";
#endif
}

extern "C" int crispembed_lid_predict_topk(void * ctx, const char * text, int k, const char ** out_labels,
                                           float * out_confidences) {
#if HAS_LID
    if (!ctx || !text) return 0;
    return text_lid_predict_topk((text_lid_context *)ctx, text, k, out_labels, out_confidences);
#else
    (void)ctx;
    (void)text;
    (void)k;
    (void)out_labels;
    (void)out_confidences;
    return 0;
#endif
}

extern "C" int crispembed_lid_n_labels(const void * ctx) {
#if HAS_LID
    return ctx ? text_lid_n_labels((const text_lid_context *)ctx) : 0;
#else
    (void)ctx;
    return 0;
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

extern "C" const crispembed_layout_region * crispembed_layout_detect(void * ctx, const char * image_path,
                                                                     float score_threshold, int * out_n) {
    if (!ctx || !image_path) {
        if (out_n) *out_n = 0;
        return nullptr;
    }
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
    if (!w->ctx) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C" void crispembed_text_det_free(void * ctx) {
    if (!ctx) return;
    auto * w = (surya_det_wrapper *)ctx;
    surya_det_free(w->ctx);
    delete w;
}

extern "C" const crispembed_text_det_result * crispembed_text_det(void * ctx, const uint8_t * pixels, int width,
                                                                  int height, int channels, float text_threshold,
                                                                  float low_threshold, int * out_n) {
    if (!ctx || !pixels) {
        if (out_n) *out_n = 0;
        return nullptr;
    }
    auto * w = (surya_det_wrapper *)ctx;

    // Run detection
    int hm_h = 0, hm_w = 0;
    surya_det_detect(w->ctx, pixels, width, height, channels, &hm_h, &hm_w);

    // Extract boxes
    int n_boxes = 0;
    const surya_det_bbox * boxes = surya_det_get_boxes(w->ctx, width, height, text_threshold, low_threshold, &n_boxes);
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
    std::vector<std::string> last_texts; // keep strings alive
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
    if (d->backend == ner_dispatch::GLINER && d->gliner_ctx) gliner_ner_free(d->gliner_ctx);
    if (d->backend == ner_dispatch::BERT_NER && d->bert_ctx) bert_ner::free(d->bert_ctx);
    delete d;
}

extern "C" int crispembed_ner_extract(void * ctx, const char * text, const char ** labels, int n_labels,
                                      float threshold, crispembed_ner_entity ** out_entities) {
    if (!ctx || !text) return 0;
    auto * d = (ner_dispatch *)ctx;

    if (d->backend == ner_dispatch::GLINER) {
        gliner_ner_entity * ents = nullptr;
        int n = gliner_ner_extract(d->gliner_ctx, text, labels, n_labels, threshold, &ents);
        if (out_entities) *out_entities = (crispembed_ner_entity *)ents;
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
        ce.end_char = d->last_entities[i].end_char;
        ce.text = d->last_texts[i].c_str();
        ce.label = d->last_labels[i].c_str();
        ce.score = d->last_entities[i].score;
        d->last_c_entities.push_back(ce);
    }

    if (out_entities) *out_entities = d->last_c_entities.data();
    return (int)d->last_c_entities.size();
}

// ===========================================================================
// LiLT — Language-independent Layout Transformer
// ===========================================================================

#include "lilt_kie.h"

struct crispembed_lilt_ctx {
    lilt_kie::context * pipe = nullptr;
    std::vector<lilt_kie::token_result> last_results;
    std::vector<crispembed_lilt_token> last_tokens;
};

extern "C" void * crispembed_lilt_init(const char * model_path, int n_threads) {
    if (!model_path) return nullptr;
    auto * ctx = new crispembed_lilt_ctx;
    if (!lilt_kie::load(&ctx->pipe, model_path, n_threads)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" void crispembed_lilt_free(void * ptr) {
    if (!ptr) return;
    auto * ctx = (crispembed_lilt_ctx *)ptr;
    if (ctx->pipe) lilt_kie::free(ctx->pipe);
    delete ctx;
}

extern "C" const crispembed_lilt_token * crispembed_lilt_classify(void * ptr, const int32_t * input_ids,
                                                                  const int32_t * bbox, int n_tokens, int * out_n) {
    if (out_n) *out_n = 0;
    if (!ptr || !input_ids || !bbox || n_tokens <= 0) return nullptr;

    auto * ctx = (crispembed_lilt_ctx *)ptr;
    ctx->last_results = lilt_kie::classify(ctx->pipe, input_ids, bbox, n_tokens);

    ctx->last_tokens.clear();
    ctx->last_tokens.reserve(ctx->last_results.size());
    for (const auto & r : ctx->last_results) {
        crispembed_lilt_token t;
        t.token_id = r.token_id;
        t.label_id = r.label_id;
        t.label = r.label.c_str();
        t.score = r.score;
        ctx->last_tokens.push_back(t);
    }

    if (out_n) *out_n = (int)ctx->last_tokens.size();
    return ctx->last_tokens.data();
}

extern "C" int crispembed_lilt_num_labels(void * ptr) {
    if (!ptr) return 0;
    return lilt_kie::num_labels(((crispembed_lilt_ctx *)ptr)->pipe);
}

extern "C" const char * crispembed_lilt_label_name(void * ptr, int label_id) {
    if (!ptr) return "";
    return lilt_kie::label_name(((crispembed_lilt_ctx *)ptr)->pipe, label_id);
}

// ===========================================================================
// Key Information Extraction (KIE)
// ===========================================================================

#include "kie_pipeline.h"

// Internal state for the C API — holds the pipeline context plus the last
// result's strings/fields so they stay alive for the caller.
struct crispembed_kie_ctx {
    kie_pipeline::context * pipe = nullptr;

    // Last result storage (valid until next extract call or free).
    kie_pipeline::result last_result;
    std::vector<crispembed_kie_field> last_fields;
    std::string last_ocr_text;
};

extern "C" void * crispembed_kie_init(const char * ocr_det_model, const char * ocr_rec_model, const char * ner_model,
                                      int n_threads) {
    if (!ocr_det_model || !ocr_rec_model || !ner_model) return nullptr;

    kie_pipeline::config cfg;
    cfg.ocr = ocr_orchestrator::default_config();

    // Wire in the provided OCR models to the first stage of the default chain.
    for (auto & chain : cfg.ocr.chains) {
        for (auto & stage : chain.stages) {
            if (stage.eng == ocr_orchestrator::engine::dbnet_trocr) {
                stage.model_a = ocr_det_model;
                stage.model_b = ocr_rec_model;
            }
        }
    }
    cfg.ner_model = ner_model;
    cfg.threshold = 0.5f;

    auto * kctx = new crispembed_kie_ctx;
    if (!kie_pipeline::load(&kctx->pipe, cfg, n_threads)) {
        delete kctx;
        return nullptr;
    }
    return kctx;
}

// LiLT-aware KIE: same as crispembed_kie_init but also wires a LiLT GGUF so the
// pipeline runs layout-aware token classification (Phase 2). ner_model may be
// empty when relying on LiLT alone.
extern "C" void * crispembed_kie_init_lilt(const char * ocr_det_model, const char * ocr_rec_model,
                                           const char * ner_model, const char * lilt_model, int n_threads) {
    if (!ocr_det_model || !ocr_rec_model) return nullptr;

    kie_pipeline::config cfg;
    cfg.ocr = ocr_orchestrator::default_config();
    for (auto & chain : cfg.ocr.chains) {
        for (auto & stage : chain.stages) {
            if (stage.eng == ocr_orchestrator::engine::dbnet_trocr) {
                stage.model_a = ocr_det_model;
                stage.model_b = ocr_rec_model;
            }
        }
    }
    if (ner_model && *ner_model) cfg.ner_model = ner_model;
    if (lilt_model && *lilt_model) cfg.lilt_model = lilt_model;
    cfg.threshold = 0.5f;

    auto * kctx = new crispembed_kie_ctx;
    if (!kie_pipeline::load(&kctx->pipe, cfg, n_threads)) {
        delete kctx;
        return nullptr;
    }
    return kctx;
}

extern "C" crispembed_kie_result crispembed_kie_extract(void * ptr, const char * image_path, const char ** labels,
                                                        int n_labels, float threshold) {
    crispembed_kie_result out;
    std::memset(&out, 0, sizeof(out));
    if (!ptr || !image_path) return out;

    auto * kctx = (crispembed_kie_ctx *)ptr;

    kctx->last_result = kie_pipeline::extract(kctx->pipe, image_path, labels, n_labels, threshold);
    kctx->last_ocr_text = kctx->last_result.ocr_full_text;

    // Build flat C field array.
    kctx->last_fields.clear();
    kctx->last_fields.reserve(kctx->last_result.fields.size());
    for (const auto & f : kctx->last_result.fields) {
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

    out.fields = kctx->last_fields.data();
    out.n_fields = (int)kctx->last_fields.size();
    out.ocr_text = kctx->last_ocr_text.c_str();
    out.ocr_confidence = kctx->last_result.ocr_confidence;
    out.n_ocr_regions = kctx->last_result.n_ocr_regions;
    return out;
}

extern "C" void crispembed_kie_free(void * ptr) {
    if (!ptr) return;
    auto * kctx = (crispembed_kie_ctx *)ptr;
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
    cp.deskew = p.deskew;
    cp.crop_borders = p.crop_borders;
    cp.whiten_background = p.whiten_background;
    cp.binarize = p.binarize;
    cp.binarize_method = p.binarize_method;
    cp.sauvola_k = p.sauvola_k;
    cp.sauvola_window = p.sauvola_window;
    cp.morph_kernel = p.morph_kernel;
    cp.border_threshold = p.border_threshold;
    cp.deskew_max_angle = p.deskew_max_angle;
    cp.despeckle = p.despeckle;
    cp.despeckle_thresh = p.despeckle_thresh;
    cp.blackfilter = p.blackfilter;
    cp.blackfilter_thresh = p.blackfilter_thresh;
    cp.deskew_consensus = p.deskew_consensus;
    return cp;
}

extern "C" int crispembed_scan_cleanup_detect_page_split(const uint8_t * pixels, int width, int height, int channels) {
    return scan_cleanup_detect_page_split(pixels, width, height, channels);
}

extern "C" int crispembed_scan_cleanup_content_bbox(const uint8_t * pixels, int width, int height, int channels,
                                                    int * x0, int * y0, int * x1, int * y1) {
    return scan_cleanup_content_bbox(pixels, width, height, channels, x0, y0, x1, y1);
}

extern "C" void * crispembed_scan_cleanup_init(const char * model_path, int n_threads) {
    return scan_cleanup_init(model_path, n_threads);
}

extern "C" void crispembed_scan_cleanup_free(void * ctx) {
    scan_cleanup_free((scan_cleanup_ctx *)ctx);
}

extern "C" int crispembed_scan_cleanup_process(void * ctx, const uint8_t * pixels, int width, int height, int channels,
                                               crispembed_scan_cleanup_params params, uint8_t ** out_pixels,
                                               int * out_width, int * out_height) {
    scan_cleanup_params p = to_cleanup(params);
    return scan_cleanup_process((scan_cleanup_ctx *)ctx, pixels, width, height, channels, p, out_pixels, out_width,
                                out_height);
}

extern "C" void crispembed_scan_cleanup_free_image(uint8_t * pixels) {
    scan_cleanup_free_image(pixels);
}

extern "C" int crispembed_scan_cleanup_process_simple(void * ctx, const uint8_t * pixels, int width, int height,
                                                      int channels, int deskew, int crop_borders, int whiten_background,
                                                      int binarize, uint8_t ** out_pixels, int * out_width,
                                                      int * out_height) {
    scan_cleanup_params p = scan_cleanup_defaults();
    p.deskew = deskew;
    p.crop_borders = crop_borders;
    p.whiten_background = whiten_background;
    p.binarize = binarize;
    return scan_cleanup_process((scan_cleanup_ctx *)ctx, pixels, width, height, channels, p, out_pixels, out_width,
                                out_height);
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

extern "C" int crispembed_text_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_size,
                                          int tile_overlap, uint8_t ** out_pixels, int * out_width, int * out_height) {
    return text_sr_process((text_sr_context *)ctx, pixels, width, height, tile_size, tile_overlap, out_pixels,
                           out_width, out_height);
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

extern "C" int crispembed_tbsrn_sr_process(void * ctx, const uint8_t * pixels, int width, int height,
                                           uint8_t ** out_pixels, int * out_width, int * out_height) {
    return tbsrn_sr_process((tbsrn_sr_context *)ctx, pixels, width, height, out_pixels, out_width, out_height);
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
extern "C" int crispembed_pan_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_size,
                                         int tile_overlap, uint8_t ** out_pixels, int * out_width, int * out_height) {
    return pan_sr_process((pan_sr_context *)ctx, pixels, width, height, tile_size, tile_overlap, out_pixels, out_width,
                          out_height);
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
extern "C" int crispembed_dat_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_w,
                                         int tile_h, uint8_t ** out_pixels, int * out_width, int * out_height) {
    return dat_sr_process((dat_sr_context *)ctx, pixels, width, height, tile_w, tile_h, out_pixels, out_width,
                          out_height);
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
extern "C" int crispembed_safmn_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int /*tile_size*/,
                                           int /*tile_overlap*/, uint8_t ** out_pixels, int * out_width,
                                           int * out_height) {
    int scale = safmn_get_scale((const safmn_context *)ctx);
    int ow = width * scale;
    int oh = height * scale;
    uint8_t * out = (uint8_t *)malloc((size_t)ow * oh * 3);
    if (!out) return -1;
    int rc = safmn_process((safmn_context *)ctx, pixels, width, height, out);
    if (rc != 0) {
        free(out);
        return rc;
    }
    *out_pixels = out;
    *out_width = ow;
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
extern "C" int crispembed_swinir_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_size,
                                            int tile_overlap, uint8_t ** out_pixels, int * out_width,
                                            int * out_height) {
    return swinir_sr_process((swinir_sr_context *)ctx, pixels, width, height, tile_size, tile_overlap, out_pixels,
                             out_width, out_height);
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
extern "C" int crispembed_esrgan_sr_process(void * ctx, const uint8_t * pixels, int width, int height,
                                            int /*tile_size*/, int /*tile_overlap*/, uint8_t ** out_pixels,
                                            int * out_width, int * out_height) {
    int scale = esrgan_get_scale((const esrgan_context *)ctx);
    int ow = width * scale;
    int oh = height * scale;
    uint8_t * out = (uint8_t *)malloc((size_t)ow * oh * 3);
    if (!out) return -1;
    int rc = esrgan_process((esrgan_context *)ctx, pixels, width, height, out);
    if (rc != 0) {
        free(out);
        return rc;
    }
    *out_pixels = out;
    *out_width = ow;
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
extern "C" int crispembed_restormer_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_size,
                                            int tile_overlap, uint8_t ** out_pixels) {
    return restormer_process((restormer_context *)ctx, pixels, width, height, tile_size, tile_overlap, out_pixels);
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
extern "C" int crispembed_scunet_process(void * ctx, const uint8_t * pixels, int width, int height,
                                         uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = scunet_process((scunet_context *)ctx, pixels, width, height, out);
    if (rc != 0) {
        free(out);
        return rc;
    }
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
extern "C" int crispembed_instructir_process(void * ctx, int task, const uint8_t * pixels, int width, int height,
                                             uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = instructir_process((instructir_context *)ctx, task, pixels, width, height, out);
    if (rc != 0) {
        free(out);
        return rc;
    }
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
extern "C" int crispembed_adair_process(void * ctx, const uint8_t * pixels, int width, int height,
                                        uint8_t ** out_pixels) {
    uint8_t * out = (uint8_t *)malloc((size_t)width * height * 3);
    if (!out) return -1;
    int rc = adair_process((adair_context *)ctx, pixels, width, height, out);
    if (rc != 0) {
        free(out);
        return rc;
    }
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
    if (!w->ctx) {
        delete w;
        return nullptr;
    }
    return w;
}

extern "C" void crispembed_punct_free(void * ctx) {
    if (!ctx) return;
    auto * w = (punct_wrapper *)ctx;
    if (w->type == PUNCT_PCS)
        pcs_free((pcs_context *)w->ctx);
    else
        fireredpunc_free((fireredpunc_context *)w->ctx);
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

extern "C" char * crispembed_ocr_render(const crispembed_ocr_result * results, int n_results, int page_width,
                                        int page_height, const char * format) {
    if (!results || n_results <= 0 || !format) return nullptr;

    // Determine format
    ocr_render_format fmt = OCR_RENDER_TEXT;
    if (strcmp(format, "hocr") == 0)
        fmt = OCR_RENDER_HOCR;
    else if (strcmp(format, "alto") == 0)
        fmt = OCR_RENDER_ALTO;
    else if (strcmp(format, "pdf") == 0)
        fmt = OCR_RENDER_PDF;

    // Convert crispembed_ocr_result to ocr_render structures.
    // Each result becomes a single-word line (since we don't have
    // line-level grouping from the pipeline results).
    std::vector<ocr_render_word> words(n_results);
    std::vector<ocr_render_line> lines(n_results);
    for (int i = 0; i < n_results; i++) {
        words[i] = { results[i].text,   (int)results[i].x, (int)results[i].y,
                     (int)results[i].w, (int)results[i].h, results[i].confidence };
        lines[i] = { &words[i], 1, (int)results[i].x, (int)results[i].y, (int)results[i].w, (int)results[i].h };
    }
    ocr_render_page page = { lines.data(), n_results, page_width, page_height, nullptr };

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

extern "C" int crispembed_pdf_page_dpi(const char * pdf_path, int page, float * out_dpi, int * out_n_images) {
    pdf_page_dpi_result r = {};
    int ret = pdf_page_dpi(pdf_path, page, &r);
    if (out_dpi) *out_dpi = r.dpi;
    if (out_n_images) *out_n_images = r.n_images;
    return ret;
}

extern "C" const crispembed_pdf_page_dpi_result * crispembed_pdf_all_pages_dpi(const char * pdf_path,
                                                                               int * out_n_pages) {
    if (out_n_pages) *out_n_pages = 0;
    int n_pages = 0;
    pdf_page_dpi_result * source = pdf_all_pages_dpi(pdf_path, &n_pages);
    if (!source || n_pages <= 0) {
        pdf_dpi_free(source);
        return nullptr;
    }
    auto * results = (crispembed_pdf_page_dpi_result *)calloc((size_t)n_pages, sizeof(crispembed_pdf_page_dpi_result));
    if (!results) {
        pdf_dpi_free(source);
        return nullptr;
    }
    for (int i = 0; i < n_pages; i++) {
        results[i].dpi = source[i].dpi;
        results[i].dpi_min = source[i].dpi_min;
        results[i].dpi_max = source[i].dpi_max;
        results[i].n_images = source[i].n_images;
        results[i].page_width_pt = source[i].page_width_pt;
        results[i].page_height_pt = source[i].page_height_pt;
    }
    pdf_dpi_free(source);
    if (out_n_pages) *out_n_pages = n_pages;
    return results;
}

extern "C" void crispembed_pdf_all_pages_dpi_free(const crispembed_pdf_page_dpi_result * results) {
    free((void *)results);
}

extern "C" int crispembed_dewarp(const uint8_t * gray, int w, int h, uint8_t * out, int * out_w, int * out_h) {
    return dewarp_page(gray, w, h, out, out_w, out_h);
}

extern "C" int crispembed_tps_dewarp(const uint8_t * gray, int w, int h, const float * src_x, const float * src_y,
                                     const float * dst_x, const float * dst_y, int n, uint8_t * out) {
    return tps_dewarp(gray, w, h, src_x, src_y, dst_x, dst_y, n, out);
}

extern "C" int crispembed_tps_auto_dewarp(const uint8_t * gray, int w, int h, const char * model_path, uint8_t * out) {
    return tps_auto_dewarp(gray, w, h, model_path, out);
}

extern "C" crispembed_ocr_result * crispembed_cc_detect(const uint8_t * gray, int w, int h, int * out_n) {
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
        results[i].orientation_corrected = 0;
        results[i].orientation_angle = 0;
        results[i].orientation_confidence = 0.0f;
    }
    cc_detect_free(regions);
    if (out_n) *out_n = n;
    return results;
}

extern "C" int crispembed_find_skew(const uint8_t * gray, int w, int h, float * angle, float * confidence) {
    return find_skew_angle(gray, w, h, angle, confidence);
}

extern "C" int crispembed_detect_page_orientation(const uint8_t * gray, int w, int h, float * confidence) {
    return detect_page_orientation(gray, w, h, confidence);
}

extern "C" void crispembed_adaptive_binarize(const uint8_t * gray, int w, int h, uint8_t * out) {
    adaptive_otsu(gray, w, h, 0, 0, 0, out);
}

extern "C" void crispembed_background_norm(const uint8_t * gray, int w, int h, uint8_t * out) {
    background_norm(gray, w, h, 0, 0, out);
}

extern "C" void crispembed_despeckle(const uint8_t * gray, int w, int h, int max_w, int max_h, uint8_t * out) {
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

extern "C" char * crispembed_table_parse_to_html(void * ctx, const uint8_t * gray, int width, int height) {
    return table_parse_to_html((table_parse_context *)ctx, gray, width, height);
}

extern "C" void crispembed_table_parse_free_string(char * str) {
    table_parse_free_string(str);
}

extern "C" int crispembed_table_parse_detect_grid(const uint8_t * gray, int width, int height, int * out_n_rows,
                                                  int * out_n_cols) {
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
extern "C" int crispembed_hat_sr_process(void * ctx, const uint8_t * pixels, int width, int height, int tile_size,
                                         int tile_overlap, uint8_t ** out_pixels, int * out_width, int * out_height) {
    return hat_sr_process((hat_sr_context *)ctx, pixels, width, height, tile_size, tile_overlap, out_pixels, out_width,
                          out_height);
}
extern "C" void crispembed_hat_sr_free_image(uint8_t * pixels) {
    hat_sr_free_image(pixels);
}
