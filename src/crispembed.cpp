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
    std::vector<float> last_output;  // reused buffer
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

    // Initialize backends: try GPU first, CPU fallback
    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) {
        fprintf(stderr, "crispembed: failed to init ggml backend\n");
        return false;
    }
    ctx->backends.push_back(ctx->backend);

    // If best backend is not CPU, add CPU as fallback
    if (!ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, ctx->n_threads);
        ctx->backends.push_back(cpu);
        fprintf(stderr, "crispembed: using %s backend with CPU fallback\n",
                ggml_backend_name(ctx->backend));
    } else {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }

    // Create scheduler
    ctx->sched = ggml_backend_sched_new(
        ctx->backends.data(), nullptr, (int)ctx->backends.size(),
        GGML_DEFAULT_GRAPH_SIZE, false, false);

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

    fprintf(stderr, "crispembed: loaded %d layers, %d dims, %d vocab\n",
            hp.n_layer, hp.n_embd, hp.n_vocab);
    return true;
}

// ---------------------------------------------------------------------------
// Graph: build + compute one encoding pass
// ---------------------------------------------------------------------------

static std::vector<float> encode_tokens(crispembed_context * ctx,
                                         const embed_tokens & tokens) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int T = (int)tokens.ids.size();
    const int H = hp.n_embd;
    const int n_heads = hp.n_head;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;

    // Allocate context for one-shot graph
    bool use_sched = (ctx->sched != nullptr);
    int graph_size = std::max(2048, hp.n_layer * 30 + 256);
    size_t per_layer = (size_t)H * T * 4 * 20
                     + (size_t)T * T * n_heads * 4
                     + (size_t)hp.n_intermediate * T * 4 * 2;
    size_t mem = per_layer * hp.n_layer
               + (size_t)H * T * 4 * 10
               + ggml_tensor_overhead() * (size_t)(hp.n_layer * 30 + 100)
               + ggml_graph_overhead_custom(graph_size, false)
               + 64 * 1024 * 1024;

    std::vector<uint8_t> buf(mem);
    // Note: we always use no_alloc=false so tensor data is pre-allocated in buf.
    // The scheduler still works — it sees CPU-allocated tensors and dispatches ops
    // to the appropriate backend. For full GPU offload, no_alloc=true would be
    // needed, but that requires restructuring how inputs are set.
    ggml_init_params ip = { mem, buf.data(), false };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // Token embeddings via ggml_get_rows (supports quantized weights)
    ggml_tensor * tok_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(tok_ids, "tok_ids");
    ggml_set_input(tok_ids);
    ggml_tensor * pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    ggml_tensor * embd = ggml_get_rows(gctx, m.token_embd, tok_ids);
    ggml_tensor * pos_embd = ggml_get_rows(gctx, m.pos_embd, pos_ids);
    embd = ggml_add(gctx, embd, pos_embd);

    // Add type embeddings if present
    ggml_tensor * type_ids_t = nullptr;
    if (m.type_embd) {
        type_ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
        ggml_set_name(type_ids_t, "type_ids");
        ggml_set_input(type_ids_t);
        ggml_tensor * type_embd = ggml_get_rows(gctx, m.type_embd, type_ids_t);
        embd = ggml_add(gctx, embd, type_embd);
    }

    // Embedding LayerNorm
    ggml_tensor * cur = embd;
    if (m.embd_ln_w) {
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, m.embd_ln_w);
        cur = ggml_add(gctx, cur, m.embd_ln_b);
    }

    // Transformer layers (BERT post-LN architecture):
    //   attn_out = MHA(cur)
    //   cur = LN1(cur + attn_out)     ← LayerNorm AFTER residual add
    //   ffn_out = FFN(cur)
    //   cur = LN2(cur + ffn_out)      ← LayerNorm AFTER residual add
    for (int il = 0; il < hp.n_layer; il++) {
        const auto & L = m.layers[il];

        // Self-attention
        ggml_tensor * Q = ggml_add(gctx, ggml_mul_mat(gctx, L.q_w, cur), L.q_b);
        ggml_tensor * K = ggml_add(gctx, ggml_mul_mat(gctx, L.k_w, cur), L.k_b);
        ggml_tensor * V = ggml_add(gctx, ggml_mul_mat(gctx, L.v_w, cur), L.v_b);

        // Multi-head attention
        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(gctx, K, head_dim, n_heads, T);
        V = ggml_reshape_3d(gctx, V, head_dim, n_heads, T);
        Q = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * scores = ggml_mul_mat(gctx, K, Q);
        scores = ggml_scale(gctx, scores, scale);
        scores = ggml_soft_max(gctx, scores);

        ggml_tensor * V_perm = ggml_cont(gctx, ggml_permute(gctx, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(gctx, V_perm, scores);
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(gctx, attn, H, T);

        // Output projection
        attn = ggml_add(gctx, ggml_mul_mat(gctx, L.o_w, attn), L.o_b);

        // Post-attention: residual add → LayerNorm (BERT post-LN)
        cur = ggml_add(gctx, cur, attn);
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, L.ln1_w);
        cur = ggml_add(gctx, cur, L.ln1_b);

        // FFN
        ggml_tensor * ffn = ggml_add(gctx, ggml_mul_mat(gctx, L.fc1_w, cur), L.fc1_b);
        ffn = ggml_gelu(gctx, ffn);
        ffn = ggml_add(gctx, ggml_mul_mat(gctx, L.fc2_w, ffn), L.fc2_b);

        // Post-FFN: residual add → LayerNorm (BERT post-LN)
        cur = ggml_add(gctx, cur, ffn);
        cur = ggml_norm(gctx, cur, ln_eps);
        cur = ggml_mul(gctx, cur, L.ln2_w);
        cur = ggml_add(gctx, cur, L.ln2_b);
    }

    ggml_set_name(cur, "encoder_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Compute
    // Set input tensor data (before graph compute)
    {
        int32_t * tok_data = (int32_t *)tok_ids->data;
        int32_t * pos_data = (int32_t *)pos_ids->data;
        for (int t = 0; t < T; t++) {
            tok_data[t] = tokens.ids[t];
            pos_data[t] = t + ctx->pos_offset;
        }
        if (type_ids_t) {
            int32_t * type_data = (int32_t *)type_ids_t->data;
            for (int t = 0; t < T; t++)
                type_data[t] = tokens.type_ids[t];
        }
    }

    if (use_sched) {
        // Backend scheduler dispatch (GPU if available)
        ggml_backend_sched_graph_compute(ctx->sched, gf);
    } else {
        struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->n_threads, NULL);
        std::vector<uint8_t> work;
        if (cplan.work_size > 0) {
            work.resize(cplan.work_size);
            cplan.work_data = work.data();
        }
        ggml_graph_compute(gf, &cplan);
    }

    // Read encoder output [H, T]
    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_out");
    float * out_data = (float *)out->data;

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

    ggml_free(gctx);
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
        ctx->last_output = decoder_encode_tokens(*ctx->dec, ctx->backend, tokens, ctx->n_threads);
    } else {
        ctx->last_output = encode_tokens(ctx, tokens);
    }
    if (out_n_dim) *out_n_dim = (int)ctx->last_output.size();
    return ctx->last_output.data();
}

extern "C" const float * crispembed_encode_batch(crispembed_context * ctx,
                                                   const char ** texts,
                                                   int n_texts,
                                                   int * out_n_dim) {
    if (!ctx || !texts || n_texts <= 0) return nullptr;
    int dim = ctx->model.hparams.n_output > 0
        ? ctx->model.hparams.n_output
        : ctx->model.hparams.n_embd;
    ctx->last_output.resize(n_texts * dim);
    for (int i = 0; i < n_texts; i++) {
        auto tokens = ctx->use_sentencepiece
            ? ctx->sp_tokenizer.encode(texts[i])
            : ctx->wp_tokenizer.encode(texts[i]);
        auto vec = encode_tokens(ctx, tokens);
        std::memcpy(ctx->last_output.data() + i * dim, vec.data(), dim * sizeof(float));
    }
    if (out_n_dim) *out_n_dim = dim;
    return ctx->last_output.data();
}

extern "C" void crispembed_free(crispembed_context * ctx) {
    if (!ctx) return;
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
