// granite_vision_ocr.cpp — Granite Vision 3.3-2B (LLaVA-Next) OCR engine.
//
// This engine follows the same CPU-scalar pattern as internvl2_ocr.cpp:
//   1. Load GGUF (core_gguf)
//   2. Vision encoder forward (SigLIP ViT, 27 layers)
//   3. Multi-layer feature extraction (layers 3, 7, 15, 26)
//   4. MLP projector (4608 → 2048 → 2048)
//   5. Token embedding + vision splicing
//   6. Autoregressive LLM decode (Granite-3.1-2B, 40 layers, GQA, KV cache)
//
// Granite LLM multipliers:
//   embedding_multiplier = 12.0  (scales token embeddings)
//   residual_multiplier  = 0.22  (scales residual connections)
//   logits_scaling       = 8.0   (divides logits)
//
// For the full implementation, see internvl2_ocr.cpp as the template.
// The key differences are documented inline.

#include "granite_vision_ocr.h"
#include "core/gguf_loader.h"
#include "core/vlm_attention.h"
#include "core/cpu_ops.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// GPU-safe dequantization via ggml_backend_tensor_get (works for Metal too).
static const float * gv_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
    int64_t n = ggml_nelements(t);
    // Only take the zero-copy fast path when the buffer is a plain CPU buffer
    // (t->data is a valid host pointer). For Metal buffers t->data is a GPU
    // handle and must not be dereferenced directly.
    if (t->type == GGML_TYPE_F32 && t->data && ggml_backend_buffer_is_host(t->buffer))
        return (const float *)t->data;
    buf.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        size_t raw = ggml_nbytes(t);
        std::vector<uint8_t> rb(raw);
        ggml_backend_tensor_get(t, rb.data(), 0, raw);
        const auto * tr = ggml_get_type_traits(t->type);
        if (tr && tr->to_float) tr->to_float(rb.data(), buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static void gv_layernorm(float * data, int n, int d,
                         const float * weight, const float * bias, float eps) {
    for (int i = 0; i < n; i++) {
        float * row = data + i * d;
        float mean = 0;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;
        float var = 0;
        for (int j = 0; j < d; j++) { float x = row[j] - mean; var += x * x; }
        var /= d;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < d; j++)
            row[j] = (row[j] - mean) * inv * weight[j] + (bias ? bias[j] : 0.0f);
    }
}

static void gv_rmsnorm(float * data, int n, int d, const float * weight, float eps) {
    for (int i = 0; i < n; i++) {
        float * row = data + i * d;
        float ss = 0;
        for (int j = 0; j < d; j++) ss += row[j] * row[j];
        float inv = 1.0f / sqrtf(ss / d + eps);
        for (int j = 0; j < d; j++) row[j] = row[j] * inv * weight[j];
    }
}

// gv_linear: delegate to SIMD-accelerated core_cpu::linear_cpu
static void gv_linear(const float * input, int n, int id, int od,
                      const float * weight, const float * bias, float * output) {
    for (int i = 0; i < n; i++)
        core_cpu::linear_cpu(input + i * id, output + i * od, id, od, weight, bias);
}

static float gv_gelu(float x) {
    return core_cpu::gelu(x);
}

static float gv_silu(float x) { return core_cpu::silu(x); }

// ── GPT-2 byte-level BPE tokenizer ──────────────────────────────────────
// Granite uses the StarCoder byte-level BPE (same byte<->unicode mapping as
// GPT-2). Mirrors the proven smoldocling_ocr.cpp implementation.

static const std::vector<int> & gv_byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool init = false;
    if (init) return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++) printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++) printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++) printable.push_back(b);
    int next = 256;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int p : printable) if (p == b) { found = true; break; }
        bs[b] = found ? b : next++;
    }
    init = true;
    return bs;
}

static const std::unordered_map<uint32_t, uint8_t> & gv_byte_decoder() {
    static std::unordered_map<uint32_t, uint8_t> table;
    static bool init = false;
    if (init) return table;
    auto & enc = gv_byte_encoder();
    for (int b = 0; b < 256; b++) table[(uint32_t)enc[b]] = (uint8_t)b;
    init = true;
    return table;
}

static void gv_utf8_encode(uint32_t cp, std::string & out) {
    if (cp < 0x80) { out.push_back((char)cp); }
    else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
    else if (cp < 0x10000) { out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
    else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
}

static std::string gv_bytes_to_unicode(const char * bytes, size_t n) {
    auto & enc = gv_byte_encoder();
    std::string out;
    for (size_t i = 0; i < n; i++) gv_utf8_encode((uint32_t)enc[(unsigned char)bytes[i]], out);
    return out;
}

static std::string gv_token_to_bytes(const std::string & token) {
    auto & dec = gv_byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < token.size()) {
        unsigned char c = (unsigned char)token[i];
        uint32_t cp; size_t len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0 && i+1 < token.size()) { cp = ((c&0x1F)<<6)|((unsigned char)token[i+1]&0x3F); len = 2; }
        else if ((c & 0xF0) == 0xE0 && i+2 < token.size()) { cp = ((c&0x0F)<<12)|(((unsigned char)token[i+1]&0x3F)<<6)|((unsigned char)token[i+2]&0x3F); len = 3; }
        else if ((c & 0xF8) == 0xF0 && i+3 < token.size()) { cp = ((c&0x07)<<18)|(((unsigned char)token[i+1]&0x3F)<<12)|(((unsigned char)token[i+2]&0x3F)<<6)|((unsigned char)token[i+3]&0x3F); len = 4; }
        else { i++; continue; }
        i += len;
        auto it = dec.find(cp);
        if (it != dec.end()) out.push_back((char)it->second);
    }
    return out;
}

struct gv_tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;
    int image_id = 49155;

    bool load(gguf_context * meta) {
        vocab = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        if (vocab.empty()) vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        if (vocab.empty()) return false;
        for (int i = 0; i < (int)vocab.size(); i++) token_to_id[vocab[i]] = i;
        auto merges = core_gguf::kv_str_array(meta, "tokenizer.merges");
        if (merges.empty()) merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) merge_rank[merges[i]] = i;
        return true;
    }

    // Byte-level BPE on a raw substring (no special tokens inside).
    void encode_piece(const std::string & text, std::vector<int> & ids) const {
        if (text.empty()) return;
        std::string uni = gv_bytes_to_unicode(text.data(), text.size());
        std::vector<std::string> syms;
        size_t i = 0;
        while (i < uni.size()) {
            unsigned char c = (unsigned char)uni[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            syms.push_back(uni.substr(i, len));
            i += len;
        }
        while (syms.size() > 1) {
            int best_rank = INT_MAX, best_i = -1;
            for (int k = 0; k + 1 < (int)syms.size(); k++) {
                auto it = merge_rank.find(syms[k] + " " + syms[k + 1]);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second; best_i = k;
                }
            }
            if (best_i < 0) break;
            syms[best_i] += syms[best_i + 1];
            syms.erase(syms.begin() + best_i + 1);
        }
        for (auto & s : syms) {
            auto it = token_to_id.find(s);
            if (it != token_to_id.end()) ids.push_back(it->second);
        }
    }

    // Encode a prompt containing a single literal "<image>" marker. The marker
    // is emitted as image_id; everything else is byte-level BPE text.
    std::vector<int> encode_prompt(const std::string & text) const {
        std::vector<int> ids;
        const std::string marker = "<image>";
        size_t pos = 0, m;
        while ((m = text.find(marker, pos)) != std::string::npos) {
            encode_piece(text.substr(pos, m - pos), ids);
            ids.push_back(image_id);
            pos = m + marker.size();
        }
        encode_piece(text.substr(pos), ids);
        return ids;
    }

    // Detokenize generated ids → UTF-8. Control/special tokens are skipped.
    std::string decode_one(int id) const {
        if (id < 0 || id >= (int)vocab.size()) return "";
        if (id <= 18 || id >= 49152) return "";   // <|end_of_text|>, <fim_*>, <image>, ...
        return gv_token_to_bytes(vocab[id]);
    }
};

// ── Constants ──────────────────────────────────────────────────────────
// kVisGraphCap: 27 vis layers × ~50 ops/layer + inputs + feature outputs.
// kLlmGraphCap: 40 llm layers × ~46 ops/layer + extras. 4096 fits both.
static constexpr int kVisGraphCap = 4096;
static constexpr int kLlmGraphCap = 4096;

// ── Context ────────────────────────────────────────────────────────────

struct granite_vision_context {
    ggml_backend_t backend = nullptr;
    ggml_backend_t vis_backend_cpu = nullptr;  // CPU fallback for ops Metal can't run
    ggml_backend_sched_t vis_sched = nullptr;
    std::vector<uint8_t> vis_compute_meta;     // pre-allocated ggml graph metadata buffer (shared)

    // Vision hparams
    int vis_dim, vis_layers, vis_heads, vis_image_size, vis_patch_size;
    std::vector<int> feature_layers;  // [-24, -20, -12, -1] → absolute indices

    // LLM hparams
    int llm_dim, llm_layers, llm_heads, llm_kv_heads, llm_ffn_dim, vocab_size;
    float embedding_multiplier, residual_multiplier, logits_scaling, rope_theta;
    float attention_multiplier, rms_eps;
    int image_token_index;
    int eos_id;
    bool tie_word_embeddings;

    gv_tokenizer tokenizer;
    bool have_tokenizer = false;

    int max_tokens;
    int n_threads;
    bool bench = false;

    // Weight storage
    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;   // caches dequantized weights (replaces wcache/wbufs)

    // RoPE frequency table (precomputed once at init, used by scalar fallback)
    core_vlm::RoPEFreqTable rope_freq;

    // F16 KV cache — Metal-resident (replaces std::vector<float> kv_cache for fast path)
    ggml_context * kvc_ctx = nullptr;
    ggml_tensor  * kvc_k   = nullptr;  // [head_dim, max_seq, n_kv_heads, n_layers] F16
    ggml_tensor  * kvc_v   = nullptr;
    ggml_backend_buffer_t kvc_buf = nullptr;
    int kvc_max_seq = 0;

    // Scalar fallback KV cache (used by dump_llm / when ggml path unavailable)
    std::vector<float> kv_cache;  // [2 * llm_layers * max_seq * head_dim * llm_kv_heads]
    int kv_allocated = 0;
    int n_past = 0;

    // Output buffer
    std::string output_text;
    std::vector<float> char_confidences;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) return nullptr;
        return dcache.get(t);
    }
};

// ── Init / Free ────────────────────────────────────────────────────────

granite_vision_context * granite_vision_init(const char * model_path, int n_threads) {
    auto * ctx = new granite_vision_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->max_tokens = 1024;
    ctx->kv_allocated = 0;
    ctx->n_past = 0;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { fprintf(stderr, "granite_vision: failed to open %s\n", model_path); delete ctx; return nullptr; }

    ctx->vis_dim        = core_gguf::kv_u32(meta, "granite_vision.vis_dim", 1152);
    ctx->vis_layers     = core_gguf::kv_u32(meta, "granite_vision.vis_layers", 27);
    ctx->vis_heads      = core_gguf::kv_u32(meta, "granite_vision.vis_heads", 16);
    ctx->vis_image_size = core_gguf::kv_u32(meta, "granite_vision.vis_image_size", 384);
    ctx->vis_patch_size = core_gguf::kv_u32(meta, "granite_vision.vis_patch_size", 14);

    auto fl = core_gguf::kv_i32_array(meta, "granite_vision.feature_layers");
    // Convert negative indices to absolute
    for (int v : fl)
        ctx->feature_layers.push_back(v < 0 ? ctx->vis_layers + v : v);

    ctx->llm_dim        = core_gguf::kv_u32(meta, "granite_vision.llm_dim", 2048);
    ctx->llm_layers     = core_gguf::kv_u32(meta, "granite_vision.llm_layers", 40);
    ctx->llm_heads      = core_gguf::kv_u32(meta, "granite_vision.llm_heads", 32);
    ctx->llm_kv_heads   = core_gguf::kv_u32(meta, "granite_vision.llm_kv_heads", 8);
    ctx->llm_ffn_dim    = core_gguf::kv_u32(meta, "granite_vision.llm_ffn_dim", 8192);
    ctx->vocab_size     = core_gguf::kv_u32(meta, "granite_vision.vocab_size", 49156);
    ctx->image_token_index = core_gguf::kv_u32(meta, "granite_vision.image_token_index", 49155);
    ctx->tie_word_embeddings = core_gguf::kv_u32(meta, "granite_vision.tie_word_embeddings", 0) != 0;

    int idx;
    idx = gguf_find_key(meta, "granite_vision.embedding_multiplier");
    ctx->embedding_multiplier = idx >= 0 ? gguf_get_val_f32(meta, idx) : 12.0f;
    idx = gguf_find_key(meta, "granite_vision.residual_multiplier");
    ctx->residual_multiplier = idx >= 0 ? gguf_get_val_f32(meta, idx) : 0.22f;
    idx = gguf_find_key(meta, "granite_vision.logits_scaling");
    ctx->logits_scaling = idx >= 0 ? gguf_get_val_f32(meta, idx) : 8.0f;
    idx = gguf_find_key(meta, "granite_vision.rope_theta");
    ctx->rope_theta = idx >= 0 ? gguf_get_val_f32(meta, idx) : 300000.0f;
    // Granite scales attention by an explicit attention_multiplier (= 1/64 for
    // the 2B), NOT 1/sqrt(head_dim). Older GGUFs predate this key; default to
    // the config value so they still decode correctly.
    idx = gguf_find_key(meta, "granite_vision.attention_multiplier");
    ctx->attention_multiplier = idx >= 0 ? gguf_get_val_f32(meta, idx) : 0.015625f;
    idx = gguf_find_key(meta, "granite_vision.rms_eps");
    ctx->rms_eps = idx >= 0 ? gguf_get_val_f32(meta, idx) : 1e-5f;
    ctx->eos_id = core_gguf::kv_u32(meta, "granite_vision.eos_token_id", 0);

    // Pre-compute RoPE frequency table (eliminates powf per element per step)
    int head_dim = ctx->llm_dim / ctx->llm_heads;
    ctx->rope_freq.precompute(head_dim, ctx->rope_theta);

    // Tokenizer (optional: older GGUFs have none → engine falls back to
    // emitting raw token-id markers so it still runs).
    ctx->have_tokenizer = ctx->tokenizer.load(meta);
    if (ctx->have_tokenizer) ctx->tokenizer.image_id = ctx->image_token_index;

    core_gguf::free_metadata(meta);

    {
        // Diagnostic/fallback lever: CRISPEMBED_GRANITE_CPU=1 keeps the whole
        // engine (weights + ggml graphs) on the ggml CPU backend, bypassing the
        // Metal ggml-alloc buffer-reuse bug that corrupts the vis/LLM graphs.
        ggml_backend_dev_t gdev = std::getenv("CRISPEMBED_GRANITE_CPU")
            ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ctx->backend = gdev ? ggml_backend_dev_init(gdev, nullptr) : nullptr;
        if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    }
    if (!core_gguf::load_weights(model_path, ctx->backend, "granite_vision", ctx->wl)) {
        fprintf(stderr, "granite_vision: failed to load weights\n");
        ggml_backend_free(ctx->backend); delete ctx; return nullptr;
    }

    // Backend scheduler shared by vis, projector, and LLM ggml graphs.
    // The meta buffer is sized for the largest graph (LLM: ~1850 nodes ≤ kLlmGraphCap=4096).
    int meta_cap = (kVisGraphCap > kLlmGraphCap) ? kVisGraphCap : kLlmGraphCap;
    ctx->vis_compute_meta.resize(
        (size_t)meta_cap * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(meta_cap, false));
    {
        std::vector<ggml_backend_t> bends;
        bends.push_back(ctx->backend);
        if (!ggml_backend_is_cpu(ctx->backend)) {
            ctx->vis_backend_cpu = ggml_backend_cpu_init();
            if (ctx->vis_backend_cpu) {
                ggml_backend_cpu_set_n_threads(ctx->vis_backend_cpu, ctx->n_threads);
                bends.push_back(ctx->vis_backend_cpu);
            }
        }
        ctx->vis_sched = ggml_backend_sched_new(
            bends.data(), nullptr, (int)bends.size(), meta_cap, false, false);
    }

    int n_patches = (ctx->vis_image_size / ctx->vis_patch_size);
    n_patches *= n_patches;  // 27*27 = 729

    fprintf(stderr, "granite_vision: vis=%dL×%dd (%d patches), llm=%dL×%dd "
            "(heads=%d/%d, ffn=%d), vocab=%d, %d tensors\n",
            ctx->vis_layers, ctx->vis_dim, n_patches,
            ctx->llm_layers, ctx->llm_dim,
            ctx->llm_heads, ctx->llm_kv_heads, ctx->llm_ffn_dim,
            ctx->vocab_size, (int)ctx->wl.tensors.size());
    fprintf(stderr, "  multipliers: embed=%.1f, residual=%.2f, logits=%.1f, "
            "attn=%.6f; tokenizer=%s (%d tokens)\n",
            ctx->embedding_multiplier, ctx->residual_multiplier, ctx->logits_scaling,
            ctx->attention_multiplier,
            ctx->have_tokenizer ? "embedded" : "MISSING",
            ctx->have_tokenizer ? (int)ctx->tokenizer.vocab.size() : 0);

    ctx->bench = (std::getenv("CRISPEMBED_GRANITE_OCR_BENCH") != nullptr);

    return ctx;
}

void granite_vision_free(granite_vision_context * ctx) {
    if (ctx) {
        if (ctx->kvc_buf) { ggml_backend_buffer_free(ctx->kvc_buf); ctx->kvc_buf = nullptr; }
        if (ctx->kvc_ctx) { ggml_free(ctx->kvc_ctx); ctx->kvc_ctx = nullptr; }
        if (ctx->vis_sched)       ggml_backend_sched_free(ctx->vis_sched);
        if (ctx->vis_backend_cpu) ggml_backend_free(ctx->vis_backend_cpu);
        core_gguf::free_weights(ctx->wl);
        if (ctx->backend)         ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

void granite_vision_set_max_tokens(granite_vision_context * ctx, int max_tokens) {
    if (ctx) ctx->max_tokens = max_tokens > 0 ? max_tokens : 2048;
}

// ── ggml graph for 27-layer SigLIP ViT ────────────────────────────────
//
// Input x_in: [T*D] row-major (T rows of D floats, same layout as the scalar x).
// Output feat_outs: one vector<float> per feature_layer, each T*D floats.
// Uses ctx->vis_sched (Metal + CPU backends set up in granite_vision_init).
static void gv_run_vit_graph(granite_vision_context * ctx,
                              const float * x_in, int T, int D,
                              std::vector<std::vector<float>> & feat_outs) {
    const int   n_heads = ctx->vis_heads;
    const int   d_head  = D / n_heads;
    const int   n_feat  = (int)ctx->feature_layers.size();
    const float eps     = 1e-6f;
    const float scale   = 1.0f / sqrtf((float)d_head);

    feat_outs.assign(n_feat, {});
    if (!ctx->vis_sched) return;  // scheduler not init'd → caller falls back to scalar

    // Each call rebuilds the graph into the pre-allocated compute_meta buffer,
    // which is reused across calls (reset implicitly by ggml_init overwriting it).
    ggml_init_params ip{ctx->vis_compute_meta.size(), ctx->vis_compute_meta.data(), /*no_alloc=*/true};
    ggml_context * g  = ggml_init(ip);
    ggml_cgraph  * gf = ggml_new_graph_custom(g, kVisGraphCap, false);

    // Input: [D, T] in ggml (ne[0]=D fast-dim, ne[1]=T).
    // Matches the scalar x buffer layout: x[t*D+d] = element (d, t).
    ggml_tensor * x_t = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x_t, "vit_x");
    ggml_set_input(x_t);
    ggml_tensor * x = x_t;

    // LayerNorm helper: norm(t, eps) → mul(w) → add(b)
    auto ln_g = [&](ggml_tensor * t, ggml_tensor * w, ggml_tensor * b) -> ggml_tensor * {
        ggml_tensor * y = ggml_norm(g, t, eps);
        if (w) y = ggml_mul(g, y, w);
        if (b) y = ggml_add(g, y, b);
        return y;
    };
    auto wt = [&](const std::string & name) -> ggml_tensor * {
        return core_gguf::try_get(ctx->wl.tensors, name.c_str());
    };
    // On the ggml-CPU backend, mul_mat against a Q8_0 weight quantizes the F32
    // activation to Q8_0 for the dot product — lower precision than Metal's mul_mm
    // (which casts activations to F16) — and that error accumulates over the 27
    // ViT layers (cos ~0.90 vs Metal's 0.96). Dequantizing the (square) attention
    // weights to F32 on CPU forces an F32 activation dot. No-op on GPU (raw Q8_0 +
    // mul_mm is already accurate and faster, and avoids the extra F32 weight copy).
    const bool cpu_backend = ggml_backend_is_cpu(ctx->backend);
    auto castw = [&](ggml_tensor * w) -> ggml_tensor * {
        return (cpu_backend && w && ggml_is_quantized(w->type)) ? ggml_cast(g, w, GGML_TYPE_F32) : w;
    };

    std::vector<ggml_tensor *> feat_ptrs(n_feat, nullptr);

    for (int li = 0; li < ctx->vis_layers; li++) {
        std::string lp = "vis.layer." + std::to_string(li);

        ggml_tensor * ln1_w = wt(lp + ".layer_norm1.weight");
        ggml_tensor * ln1_b = wt(lp + ".layer_norm1.bias");
        ggml_tensor * q_w   = wt(lp + ".attn.q.weight");
        ggml_tensor * q_b   = wt(lp + ".attn.q.bias");
        ggml_tensor * k_w   = wt(lp + ".attn.k.weight");
        ggml_tensor * k_b   = wt(lp + ".attn.k.bias");
        ggml_tensor * v_w   = wt(lp + ".attn.v.weight");
        ggml_tensor * v_b   = wt(lp + ".attn.v.bias");
        ggml_tensor * o_w   = wt(lp + ".attn.out.weight");
        ggml_tensor * o_b   = wt(lp + ".attn.out.bias");
        ggml_tensor * ln2_w = wt(lp + ".layer_norm2.weight");
        ggml_tensor * ln2_b = wt(lp + ".layer_norm2.bias");
        ggml_tensor * fc1_w = wt(lp + ".ffn.up.weight");
        ggml_tensor * fc1_b = wt(lp + ".ffn.up.bias");
        ggml_tensor * fc2_w = wt(lp + ".ffn.down.weight");
        ggml_tensor * fc2_b = wt(lp + ".ffn.down.bias");

        if (!q_w || !k_w || !v_w || !o_w || !fc1_w || !fc2_w) {
            fprintf(stderr, "granite_vision: missing weights at layer %d, aborting graph\n", li);
            ggml_free(g);
            return;
        }

        // ── Attention ──
        ggml_tensor * resid = x;
        ggml_tensor * y = ln_g(x, ln1_w, ln1_b);

        // QKV projections: [D, T]
        ggml_tensor * Q = ggml_mul_mat(g, castw(q_w), y); if (q_b) Q = ggml_add(g, Q, q_b);
        ggml_tensor * K = ggml_mul_mat(g, castw(k_w), y); if (k_b) K = ggml_add(g, K, k_b);
        ggml_tensor * V = ggml_mul_mat(g, castw(v_w), y); if (v_b) V = ggml_add(g, V, v_b);

        // [D, T] → [d_head, n_heads, T] → permute → [d_head, T, n_heads]
        Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Q, d_head, n_heads, T), 0, 2, 1, 3));
        K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, K, d_head, n_heads, T), 0, 2, 1, 3));
        V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, V, d_head, n_heads, T), 0, 2, 1, 3));

        // flash_attn_ext returns [d_head, n_heads, T] → reshape directly to [D, T],
        // matching the validated bidirlm_vision ViT (no extra permute). The square
        // Q8_0 attn weights are used raw (no reshape) — correct because their
        // ne[0]=in axis already carries the 32-blocks; only the non-square FFN
        // weights need the reshape/dequant handling below.
        ggml_tensor * attn = ggml_flash_attn_ext(g, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);

        attn = ggml_mul_mat(g, castw(o_w), attn);
        if (o_b) attn = ggml_add(g, attn, o_b);
        x = ggml_add(g, resid, attn);

        // ── FFN ──
        // The converter stores 2D weights in PyTorch [out,in] order, so the
        // non-square FFN weights have transposed ggml ne (ne[0]=out) and must be
        // relabelled to ggml_mul_mat's [in,out] with a reshape. BUT reshaping a
        // *quantized* tensor only stays valid when the target ne[0] is a multiple
        // of the quant block size: vis.ffn.down is Q8_0 with ne[0] 1152→4304 and
        // 4304 % 32 != 0, so the reshaped view's blocks are mis-strided → garbage
        // that corrupts the residual stream from layer 0 (was misdiagnosed as a
        // ggml-alloc reuse bug). Dequantizing to F32 first makes the reshape a
        // plain element relabel. F16/F32 weights reshape safely as-is.
        // See LEARNINGS "granite_vision: ggml graph divergence was a Q8_0 reshape".
        auto reshape_w = [&](ggml_tensor * w) -> ggml_tensor * {
            // Cast to F32 when quantized (required: the Q8_0 down reshape is only
            // valid in F32) or, on the CPU backend, when F16 (for dot precision —
            // see castw). Then reshape to ggml_mul_mat's [in,out].
            const bool need_cast = ggml_is_quantized(w->type) ||
                                   (cpu_backend && w->type != GGML_TYPE_F32);
            ggml_tensor * t = need_cast ? ggml_cast(g, w, GGML_TYPE_F32) : w;
            return ggml_reshape_2d(g, t, w->ne[1], w->ne[0]);
        };
        resid = x;
        y = ln_g(x, ln2_w, ln2_b);
        y = ggml_mul_mat(g, reshape_w(fc1_w), y);
        if (fc1_b) y = ggml_add(g, y, fc1_b);
        // tanh-GELU (gelu_pytorch_tanh) in explicit F32 ops. ggml_gelu routes
        // through an F16 lookup table on the ggml-CPU backend (input quantized to
        // F16); over the 27 ViT layers that drifts to cos ~0.84 vs scalar, while
        // Metal computes gelu in F32. ggml_tanh is direct tanhf, so this keeps
        // the CPU graph at parity. 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³))).
        {
            ggml_tensor * x3 = ggml_mul(g, ggml_mul(g, y, y), y);
            ggml_tensor * inner = ggml_scale(g, ggml_add(g, y, ggml_scale(g, x3, 0.044715f)),
                                             0.7978845608028654f);
            ggml_tensor * th = ggml_tanh(g, inner);
            y = ggml_scale(g, ggml_add(g, y, ggml_mul(g, y, th)), 0.5f);
        }
        y = ggml_mul_mat(g, reshape_w(fc2_w), y);
        if (fc2_b) y = ggml_add(g, y, fc2_b);
        x = ggml_add(g, resid, y);

        // ── Feature extraction ──
        for (int fi = 0; fi < n_feat; fi++) {
            if (li != ctx->feature_layers[fi]) continue;
            ggml_tensor * fc = ggml_cont(g, x);
            char nm[32]; snprintf(nm, sizeof(nm), "feat_%d", fi);
            ggml_set_name(fc, nm);
            ggml_set_output(fc);
            ggml_build_forward_expand(gf, fc);
            feat_ptrs[fi] = fc;
        }
    }

    // Ensure the full graph is traced (last feature layer may not be the last ViT layer).
    ggml_build_forward_expand(gf, x);

    const bool gv_dbg = std::getenv("CRISPEMBED_GRANITE_VIS_DBG") != nullptr;
    if (gv_dbg) {
        fprintf(stderr, "[vis-dbg] n_nodes=%d cap=%d n_feat=%d x_t->buffer(pre)=%p\n",
                ggml_graph_n_nodes(gf), kVisGraphCap, n_feat, (void*)x_t->buffer);
        for (int fi = 0; fi < n_feat; fi++)
            fprintf(stderr, "[vis-dbg] feat_ptrs[%d]=%p name=%s\n", fi,
                    (void*)feat_ptrs[fi], feat_ptrs[fi] ? feat_ptrs[fi]->name : "(null)");
    }

    // ── Schedule + compute ──
    ggml_backend_sched_reset(ctx->vis_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->vis_sched, gf)) {
        fprintf(stderr, "granite_vision: vis graph alloc failed\n");
        ggml_free(g); return;
    }

    // Upload input (x_in layout matches ggml [D,T]: x_in[t*D+d] = (d,t))
    ggml_backend_tensor_set(x_t, x_in, 0, (size_t)T * D * sizeof(float));

    if (gv_dbg) {
        fprintf(stderr, "[vis-dbg] post-alloc x_t->buffer=%p feat0->buffer=%p\n",
                (void*)x_t->buffer, n_feat ? (void*)feat_ptrs[0]->buffer : nullptr);
        std::vector<float> chk(8);
        ggml_backend_tensor_get(x_t, chk.data(), 0, 8 * sizeof(float));
        fprintf(stderr, "[vis-dbg] x_t[0..3] before set = %.4f %.4f %.4f %.4f\n",
                chk[0], chk[1], chk[2], chk[3]);
    }

    if (ggml_backend_sched_graph_compute(ctx->vis_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "granite_vision: vis graph compute failed\n");
        ggml_free(g); return;
    }
    if (gv_dbg) {
        std::vector<float> chk(8);
        ggml_backend_tensor_get(x_t, chk.data(), 0, 8 * sizeof(float));
        fprintf(stderr, "[vis-dbg] x_t[0..3] after compute = %.4f %.4f %.4f %.4f\n",
                chk[0], chk[1], chk[2], chk[3]);
    }

    // Read back feature tensors (same [D,T] layout as scalar layer_outputs)
    for (int fi = 0; fi < n_feat; fi++) {
        if (!feat_ptrs[fi]) continue;
        feat_outs[fi].resize((size_t)T * D);
        ggml_backend_tensor_get(feat_ptrs[fi], feat_outs[fi].data(), 0, (size_t)T * D * sizeof(float));
    }

    ggml_free(g);
}

// ── F16 KV Cache (Metal-resident) ─────────────────────────────────────

static void gv_free_kv_cache(granite_vision_context * ctx) {
    if (ctx->kvc_buf) { ggml_backend_buffer_free(ctx->kvc_buf); ctx->kvc_buf = nullptr; }
    if (ctx->kvc_ctx) { ggml_free(ctx->kvc_ctx); ctx->kvc_ctx = nullptr; }
    ctx->kvc_k = ctx->kvc_v = nullptr;
    ctx->kvc_max_seq = 0;
}

static bool gv_alloc_kv_cache(granite_vision_context * ctx, int max_seq) {
    gv_free_kv_cache(ctx);
    const int hd  = ctx->llm_dim / ctx->llm_heads;
    const int nkv = ctx->llm_kv_heads;
    const int nl  = ctx->llm_layers;

    size_t meta_sz = 2 * ggml_tensor_overhead() + 256;
    ggml_init_params ip{meta_sz, nullptr, /*no_alloc=*/true};
    ctx->kvc_ctx = ggml_init(ip);
    ctx->kvc_k = ggml_new_tensor_4d(ctx->kvc_ctx, GGML_TYPE_F16, hd, max_seq, nkv, nl);
    ctx->kvc_v = ggml_new_tensor_4d(ctx->kvc_ctx, GGML_TYPE_F16, hd, max_seq, nkv, nl);
    ggml_set_name(ctx->kvc_k, "gv_kv_k");
    ggml_set_name(ctx->kvc_v, "gv_kv_v");

    ctx->kvc_buf = ggml_backend_alloc_ctx_tensors(ctx->kvc_ctx, ctx->backend);
    if (!ctx->kvc_buf) {
        ggml_free(ctx->kvc_ctx); ctx->kvc_ctx = nullptr;
        ctx->kvc_k = ctx->kvc_v = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(ctx->kvc_buf, 0);
    ctx->kvc_max_seq = max_seq;
    float kv_mb = (float)ggml_backend_buffer_get_size(ctx->kvc_buf) / 1048576.0f;
    fprintf(stderr, "granite_vision: KV cache %.1f MB (max_seq=%d, F16)\n", kv_mb, max_seq);
    return true;
}

// ── MLP Projector ggml Graph ───────────────────────────────────────────
// Runs Linear(feat_dim→llm_dim) + GELU + Linear(llm_dim→llm_dim) on Metal.
// Returns false if sched is unavailable; caller falls back to scalar.
static bool gv_run_projector_graph(granite_vision_context * ctx,
                                   const float * vis_feat, int n_tokens, int feat_dim,
                                   float * proj_out) {
    if (!ctx->vis_sched) return false;
    const int out_dim = ctx->llm_dim;

    ggml_init_params ip{ctx->vis_compute_meta.size(), ctx->vis_compute_meta.data(), true};
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, 32, false);

    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, feat_dim, n_tokens);
    ggml_set_name(x, "proj_in"); ggml_set_input(x);

    ggml_tensor * w1 = core_gguf::try_get(ctx->wl.tensors, "proj.linear_1.weight");
    ggml_tensor * b1 = core_gguf::try_get(ctx->wl.tensors, "proj.linear_1.bias");
    ggml_tensor * w2 = core_gguf::try_get(ctx->wl.tensors, "proj.linear_2.weight");
    ggml_tensor * b2 = core_gguf::try_get(ctx->wl.tensors, "proj.linear_2.bias");
    if (!w1 || !w2) { ggml_free(g); return false; }

    // Reshape weights from the converter's [out,in] ggml ne to ggml_mul_mat's
    // expected [in,out] (no-op for the square w2). See granite-converter-
    // transposed-ne; mirrors the vision/LLM graph fix.
    ggml_tensor * h = ggml_mul_mat(g, ggml_reshape_2d(g, w1, w1->ne[1], w1->ne[0]), x);
    if (b1) h = ggml_add(g, h, b1);
    // projector_hidden_act = "gelu" (exact erf), NOT the vision tower's
    // "gelu_pytorch_tanh". HF F.gelu() with no approximate= is erf. Using the
    // tanh approx here measurably lowered projector parity (cos 0.954 vs HF).
    h = ggml_gelu_erf(g, h);
    h = ggml_mul_mat(g, ggml_reshape_2d(g, w2, w2->ne[1], w2->ne[0]), h);
    if (b2) h = ggml_add(g, h, b2);

    ggml_tensor * out_t = ggml_cont(g, h);
    ggml_set_output(out_t);
    ggml_build_forward_expand(gf, out_t);

    ggml_backend_sched_reset(ctx->vis_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->vis_sched, gf)) {
        ggml_free(g); return false;
    }
    ggml_backend_tensor_set(x, vis_feat, 0, (size_t)n_tokens * feat_dim * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->vis_sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(g); return false;
    }
    ggml_backend_tensor_get(out_t, proj_out, 0, (size_t)n_tokens * out_dim * sizeof(float));
    ggml_free(g);
    return true;
}

// ── Granite LLM Body ggml Graph ───────────────────────────────────────
// Runs T tokens through all llm_layers of Granite-3.1 with the F16 KV cache.
//   embeds: [T × D] float (pre-assembled: text scaled, vision unscaled)
//   n_past: number of KV entries already in cache (0 for prefill)
//   hidden_out: [D] float — hidden state of the last token after final RMSNorm
//   logits_out: if non-null, the LM head (tied embedding) + logits_scaling is run
//     IN-GRAPH on Metal for the last token and read back as [vocab_size] — avoids
//     the per-token CPU vocab matmul + hidden GPU→CPU readback. Pass null (diff
//     path) to keep just hidden_out and apply the LM head on CPU.
// Returns false on error.
static bool gv_run_llm_body(granite_vision_context * ctx,
                             const float * embeds, int T, int n_past,
                             float * hidden_out,
                             gv_dump_cb dump_cb = nullptr, void * dump_ud = nullptr,
                             float * logits_out = nullptr) {
    if (!ctx->vis_sched || !ctx->kvc_buf) return false;
    // Per-layer last-token capture for crispembed-diff (only when dumping).
    std::vector<std::pair<std::string, ggml_tensor*>> dbg_dumps;

    const int D        = ctx->llm_dim;
    const int n_heads  = ctx->llm_heads;
    const int n_kv     = ctx->llm_kv_heads;
    const int d_head   = D / n_heads;
    // kv_rep not needed: ggml_flash_attn_ext handles GQA natively
    const int Lk       = n_past + T;
    const float eps    = ctx->rms_eps;
    const float res_mul = ctx->residual_multiplier;
    const float attn_mul = ctx->attention_multiplier;

    ggml_init_params ip{ctx->vis_compute_meta.size(), ctx->vis_compute_meta.data(), true};
    ggml_context * g  = ggml_init(ip);
    ggml_cgraph  * gf = ggml_new_graph_custom(g, kLlmGraphCap, false);

    // ── Inputs ──
    ggml_tensor * x_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x_in, "llm_in"); ggml_set_input(x_in);
    ggml_tensor * x = x_in;  // working variable; x_in preserved for data upload

    ggml_tensor * rope_pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(rope_pos, "rope_pos"); ggml_set_input(rope_pos);

    // Causal mask: [Lk, T] F16 — added to pre-softmax scores (0=attend, -inf=block)
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    auto wt = [&](const std::string & name) -> ggml_tensor * {
        return core_gguf::try_get(ctx->wl.tensors, name.c_str());
    };
    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };
    // The converter stores 2D weights in PyTorch [out,in] order, so non-square
    // weights have transposed ggml ne and ggml_mul_mat asserts (ne[0] mismatch).
    // Reshape relabels the contiguous data to ggml's expected [in,out] — a no-op
    // for the square q/o weights, valid for Q4_K because its 256-blocks lie along
    // the in axis. Mirrors the existing vision FFN fix; see the
    // granite-converter-transposed-ne note.
    auto sw = [&](ggml_tensor * w) -> ggml_tensor * {
        return ggml_reshape_2d(g, w, w->ne[1], w->ne[0]);
    };

    for (int li = 0; li < ctx->llm_layers; li++) {
        std::string lp = "llm.layer." + std::to_string(li);

        ggml_tensor * n1w = wt(lp + ".norm1.weight");
        ggml_tensor * n2w = wt(lp + ".norm2.weight");
        ggml_tensor * qw  = wt(lp + ".attn.q.weight");
        ggml_tensor * kw  = wt(lp + ".attn.k.weight");
        ggml_tensor * vw  = wt(lp + ".attn.v.weight");
        ggml_tensor * ow  = wt(lp + ".attn.o.weight");
        ggml_tensor * gw  = wt(lp + ".ffn.gate.weight");
        ggml_tensor * uw  = wt(lp + ".ffn.up.weight");
        ggml_tensor * dw  = wt(lp + ".ffn.down.weight");
        if (!n1w || !qw || !kw || !vw || !ow || !gw || !uw || !dw) {
            fprintf(stderr, "granite_vision: missing LLM weights at layer %d\n", li);
            ggml_free(g); return false;
        }

        // ── Self-attention ──
        ggml_tensor * resid = x;
        ggml_tensor * h = rmsnorm(x, n1w);

        // Q: [D, T], K_new: [n_kv*d_head, T], V_new: [n_kv*d_head, T]
        ggml_tensor * Q     = ggml_mul_mat(g, qw, h);
        ggml_tensor * K_new = ggml_mul_mat(g, sw(kw), h);
        ggml_tensor * V_new = ggml_mul_mat(g, sw(vw), h);

        // Reshape to [d_head, n_*, T]
        Q     = ggml_reshape_3d(g, Q,     d_head, n_heads, T);
        K_new = ggml_reshape_3d(g, K_new, d_head, n_kv,    T);
        V_new = ggml_reshape_3d(g, V_new, d_head, n_kv,    T);

        // RoPE (NEOX = split-half style, matches Granite/Llama HF)
        Q     = ggml_rope_ext(g, Q,     rope_pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0,
                              ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_ext(g, K_new, rope_pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0,
                              ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Write K/V to cache at [layer li, positions n_past..n_past+T)
        // K_new_perm: [d_head, n_kv, T] → [d_head, T, n_kv]
        ggml_tensor * Kp = ggml_cont(g, ggml_permute(g, K_new, 0, 2, 1, 3));
        ggml_tensor * Vp = ggml_cont(g, ggml_permute(g, V_new, 0, 2, 1, 3));

        size_t kh_nb1 = ctx->kvc_k->nb[1];
        size_t kh_nb2 = ctx->kvc_k->nb[2];
        size_t kh_nb3 = ctx->kvc_k->nb[3];
        ggml_tensor * kv_view = ggml_view_4d(g, ctx->kvc_k,
            d_head, T, n_kv, 1, kh_nb1, kh_nb2, kh_nb3,
            (size_t)li * kh_nb3 + (size_t)n_past * kh_nb1);
        ggml_tensor * vv_view = ggml_view_4d(g, ctx->kvc_v,
            d_head, T, n_kv, 1, kh_nb1, kh_nb2, kh_nb3,
            (size_t)li * ctx->kvc_v->nb[3] + (size_t)n_past * ctx->kvc_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(g, Kp, kv_view));
        ggml_build_forward_expand(gf, ggml_cpy(g, Vp, vv_view));

        // Read full K/V history [0..Lk) for this layer
        // K/V history views [d_head, Lk, n_kv] straight into the cache — passed to
        // flash_attn_ext as-is (it reads via nb strides), avoiding a full-history
        // ggml_cont copy on every layer/step (was ~64 MB/step at len ~800).
        ggml_tensor * k_lay = ggml_view_3d(g, ctx->kvc_k,
            d_head, Lk, n_kv, kh_nb1, kh_nb2, (size_t)li * kh_nb3);
        ggml_tensor * v_lay = ggml_view_3d(g, ctx->kvc_v,
            d_head, Lk, n_kv, ctx->kvc_v->nb[1], ctx->kvc_v->nb[2],
            (size_t)li * ctx->kvc_v->nb[3]);

        // flash_attn_ext handles GQA natively via rk2=n_heads/n_kv broadcast —
        // no explicit KV head repeat needed. Q [d_head, T, n_heads], K/V [d_head, Lk, n_kv]
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));  // [d_head, T, n_heads]
        ggml_tensor * attn = ggml_flash_attn_ext(g, Q, k_lay, v_lay, mask, attn_mul, 0.0f, 0.0f);
        // Output: [d_head, n_heads, T] → reshape → [D, T]
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection + scaled residual
        attn = ggml_mul_mat(g, ow, attn);
        x = ggml_add(g, resid, ggml_scale(g, attn, res_mul));

        // ── FFN (SwiGLU) ──
        // Metal's batched matmul (mul_mm, used for T>8 prefill) casts activations
        // to F16 for the simdgroup matrix units. Granite carries "massive
        // activations" (a few residual dims ~1e4, amplified ~×12 by the image
        // feature embeddings), so the SwiGLU gate*up product can exceed F16 range
        // (65504) → inf → NaN cascade from the first layer whose weights are large
        // enough (empirically layer 8). Scale the down-projection activation down
        // before the matmul and back up after: a lossless exponent shift that keeps
        // the F16 cast in range. (mul_mv / T=1 decode and the CPU backend use F32
        // and are unaffected.) See LEARNINGS "Metal mul_mm F16 activation overflow".
        resid = x;
        h = rmsnorm(x, n2w);
        ggml_tensor * gate = ggml_silu(g, ggml_mul_mat(g, sw(gw), h));
        ggml_tensor * up   = ggml_mul_mat(g, sw(uw), h);
        ggml_tensor * gu   = ggml_mul(g, gate, up);
        // The ÷256/×256 exponent shift only guards the F16 mul_mm cast used for the
        // batched prefill (T > ne11_mm_min = 8). T=1 decode uses mul_mv in F32, so
        // skip the two scale kernels entirely — saves ~2 dispatches/layer/token.
        ggml_tensor * down = (T > 8)
            ? ggml_scale(g, ggml_mul_mat(g, sw(dw), ggml_scale(g, gu, 1.0f / 256.0f)), 256.0f)
            : ggml_mul_mat(g, sw(dw), gu);
        x = ggml_add(g, resid, ggml_scale(g, down, res_mul));

        if (dump_cb) {
            ggml_tensor * col = ggml_cont(g, ggml_view_2d(g, x, D, 1, x->nb[1],
                                          (size_t)(T - 1) * x->nb[1]));
            char nm[32]; snprintf(nm, sizeof(nm), "llm_layer_%d_out", li);
            ggml_set_name(col, nm); ggml_set_output(col);
            ggml_build_forward_expand(gf, col);
            dbg_dumps.emplace_back(nm, col);
        }
    }

    // Final RMSNorm (all T tokens; caller reads only the last one)
    x = rmsnorm(x, wt("llm.norm.weight"));
    ggml_tensor * out_t = ggml_cont(g, x);
    ggml_set_output(out_t);
    ggml_build_forward_expand(gf, out_t);

    // In-graph LM head (tied embedding) for the last token — keeps the ~49156×2048
    // vocab projection on the GPU instead of a per-token CPU matmul + hidden
    // readback. The last-token hidden is normalized (O(1)) so the T=1 mul_mv is
    // F32-safe. logits = (W_lm · h_last) / logits_scaling.
    ggml_tensor * logit_t = nullptr;
    if (logits_out) {
        ggml_tensor * lm_w = wt("llm.lm_head.weight");
        if (!lm_w && ctx->tie_word_embeddings) lm_w = wt("llm.embed.weight");
        if (lm_w) {
            // Converter stores embed/lm_head with transposed ne (ne[0]=vocab);
            // relabel to mul_mat's [in=D, out=vocab] (cast quantized first so the
            // reshape stays block-valid — see reshape_w in the ViT).
            if (lm_w->ne[0] != D) {
                ggml_tensor * t = ggml_is_quantized(lm_w->type) ? ggml_cast(g, lm_w, GGML_TYPE_F32) : lm_w;
                lm_w = ggml_reshape_2d(g, t, lm_w->ne[1], lm_w->ne[0]);
            }
            ggml_tensor * h_last = ggml_view_2d(g, out_t, D, 1, out_t->nb[1],
                                                (size_t)(T - 1) * out_t->nb[1]);
            logit_t = ggml_mul_mat(g, lm_w, ggml_cont(g, h_last));   // [vocab, 1]
            logit_t = ggml_scale(g, logit_t, 1.0f / ctx->logits_scaling);
            ggml_set_output(logit_t);
            ggml_build_forward_expand(gf, logit_t);
        }
    }
    if (dump_cb) {
        ggml_tensor * col = ggml_cont(g, ggml_view_2d(g, out_t, D, 1, out_t->nb[1],
                                      (size_t)(T - 1) * out_t->nb[1]));
        ggml_set_name(col, "llm_final_norm"); ggml_set_output(col);
        ggml_build_forward_expand(gf, col);
        dbg_dumps.emplace_back("llm_final_norm", col);
    }

    // ── Schedule + compute ──
    ggml_backend_sched_reset(ctx->vis_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->vis_sched, gf)) {
        fprintf(stderr, "granite_vision: LLM graph alloc failed (T=%d, Lk=%d)\n", T, Lk);
        ggml_free(g); return false;
    }

    ggml_backend_tensor_set(x_in, embeds, 0, (size_t)T * D * sizeof(float));

    // Position IDs: [n_past, n_past+1, ..., n_past+T-1]
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++) pos[i] = n_past + i;
    ggml_backend_tensor_set(rope_pos, pos.data(), 0, T * sizeof(int32_t));

    // Causal mask — F16, shape [Lk, T]:
    //   mask[k, q] = 0 if k <= n_past+q (causal attend), else -inf
    std::vector<ggml_fp16_t> mask_data((size_t)Lk * T);
    const ggml_fp16_t h0  = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t hni = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < Lk; k++)
            mask_data[(size_t)q * Lk + k] = (k <= n_past + q) ? h0 : hni;
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->vis_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "granite_vision: LLM graph compute failed\n");
        ggml_free(g); return false;
    }

    // Read back last token's hidden state (offset = (T-1)*D floats)
    ggml_backend_tensor_get(out_t, hidden_out,
                            (size_t)(T - 1) * D * sizeof(float), D * sizeof(float));
    if (logit_t)
        ggml_backend_tensor_get(logit_t, logits_out, 0, (size_t)ctx->vocab_size * sizeof(float));

    if (dump_cb) {
        std::vector<float> tmp(D);
        for (auto & pr : dbg_dumps) {
            ggml_backend_tensor_get(pr.second, tmp.data(), 0, D * sizeof(float));
            dump_cb(pr.first.c_str(), tmp.data(), D, dump_ud);
        }
    }
    ggml_free(g);
    return true;
}

// ── SigLIP Vision Encoder ──────────────────────────────────────────────

// Run SigLIP ViT and return multi-layer features.
// Input: [3, image_size, image_size] float [0, 1]
// Output: [n_patches, vis_dim * n_feature_layers] — concatenated features
static void gv_vision_forward(granite_vision_context * ctx,
                               const float * image, int img_h, int img_w,
                               float * output, int * out_tokens,
                               gv_dump_cb cb = nullptr, void * ud = nullptr) {
    int ps = ctx->vis_patch_size;
    int ph = img_h / ps, pw = img_w / ps;
    int T = ph * pw;  // number of patches (729 for 384/14)
    int D = ctx->vis_dim;
    int n_heads = ctx->vis_heads;
    int d_head = D / n_heads;
    float eps = 1e-6f;

    // Patch embedding: Conv2D(3, D, ps, stride=ps) — equivalent to linear on flattened patches
    const float * pe_w = ctx->get("vis.patch_embed.weight");
    const float * pe_b = ctx->get("vis.patch_embed.bias");
    std::vector<float> x(T * D);

    // Patch embed: for each patch, flatten [3, ps, ps] → linear → [D]
    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            int t = py * pw + px;
            float * out_row = x.data() + t * D;
            for (int d = 0; d < D; d++) {
                float sum = pe_b ? pe_b[d] : 0.0f;
                for (int c = 0; c < 3; c++)
                    for (int ky = 0; ky < ps; ky++)
                        for (int kx = 0; kx < ps; kx++) {
                            int iy = py * ps + ky, ix = px * ps + kx;
                            if (iy < img_h && ix < img_w)
                                sum += image[c * img_h * img_w + iy * img_w + ix]
                                     * pe_w[d * 3 * ps * ps + c * ps * ps + ky * ps + kx];
                        }
                out_row[d] = sum;
            }
        }
    }

    // Add position embedding
    const float * pos_w = ctx->get("vis.pos_embed.weight");
    if (pos_w) {
        for (int t = 0; t < T && t < (int)(ctx->vis_image_size / ps) * (ctx->vis_image_size / ps); t++)
            for (int d = 0; d < D; d++)
                x[t * D + d] += pos_w[t * D + d];
    }

    // crispembed-diff: patch+pos embed is shared by both ViT paths, so this
    // localises where divergence begins (vs the ref's "vis_patch_embed").
    if (cb) cb("vis_patch_embed", x.data(), T * D, ud);

    // ── 27-layer SigLIP ViT (Metal-accelerated via ggml graph) ──
    int n_feat = (int)ctx->feature_layers.size();
    std::vector<std::vector<float>> layer_outputs(n_feat);

    // The ggml ViT graph (gv_run_vit_graph) is NUMERICALLY CORRECT on both Metal
    // and the ggml-CPU backend: per-layer crispembed-diff matches the scalar ViT
    // to 3-4 decimals on each (layer 3/7/15 cos 0.9996-0.99987; layer 26 cos 0.958
    // = the scalar's eps ref-artifact, not a bug). Metal runs ~3 s vs ~100 s
    // scalar; the CPU graph is also faster than the scalar loop. DEFAULT ON;
    // opt out with CRISPEMBED_GRANITE_VIS_SCALAR=1. (The CPU late-layer drift —
    // F16-table gelu + Q8_0-quantized activations in the attn matmuls — is fixed
    // inside gv_run_vit_graph via explicit F32 gelu + CPU F32 weight casts.)
    const bool use_vis_graph = !std::getenv("CRISPEMBED_GRANITE_VIS_SCALAR");
    if (ctx->vis_sched && use_vis_graph) {
        gv_run_vit_graph(ctx, x.data(), T, D, layer_outputs);
    }

    if (n_feat > 0 && layer_outputs[0].empty()) {
        // Fallback: scalar loop (used when vis_sched is null or graph failed).
        for (int li = 0; li < ctx->vis_layers; li++) {
            char buf[64];
            std::vector<float> normed(T * D);
            memcpy(normed.data(), x.data(), T * D * sizeof(float));
            snprintf(buf, sizeof(buf), "vis.layer.%d.layer_norm1", li);
            gv_layernorm(normed.data(), T, D, ctx->get(std::string(buf) + ".weight"),
                         ctx->get(std::string(buf) + ".bias"), eps);

            std::string lp = std::string("vis.layer.") + std::to_string(li);
            std::vector<float> Q(T * D), K(T * D), V(T * D);
            gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.q.weight"),
                      ctx->get(lp + ".attn.q.bias"), Q.data());
            gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.k.weight"),
                      ctx->get(lp + ".attn.k.bias"), K.data());
            gv_linear(normed.data(), T, D, D, ctx->get(lp + ".attn.v.weight"),
                      ctx->get(lp + ".attn.v.bias"), V.data());

            float scale = 1.0f / sqrtf((float)d_head);
            std::vector<float> attn_out(T * D, 0.0f);
            for (int h = 0; h < n_heads; h++) {
                int off = h * d_head;
                for (int q = 0; q < T; q++) {
                    float max_s = -1e9f;
                    std::vector<float> scores(T);
                    for (int k = 0; k < T; k++) {
                        float s = core_cpu::dot_product(Q.data() + q * D + off,
                                                        K.data() + k * D + off, d_head) * scale;
                        scores[k] = s;
                        if (s > max_s) max_s = s;
                    }
                    float sum_e = 0;
                    for (int k = 0; k < T; k++) { scores[k] = expf(scores[k] - max_s); sum_e += scores[k]; }
                    float inv = 1.0f / sum_e;
                    for (int d = 0; d < d_head; d++) {
                        float val = 0;
                        for (int k = 0; k < T; k++) val += scores[k] * inv * V[k * D + off + d];
                        attn_out[q * D + off + d] = val;
                    }
                }
            }

            std::vector<float> proj(T * D);
            gv_linear(attn_out.data(), T, D, D, ctx->get(lp + ".attn.out.weight"),
                      ctx->get(lp + ".attn.out.bias"), proj.data());
            for (int i = 0; i < T * D; i++) x[i] += proj[i];

            memcpy(normed.data(), x.data(), T * D * sizeof(float));
            snprintf(buf, sizeof(buf), "vis.layer.%d.layer_norm2", li);
            gv_layernorm(normed.data(), T, D, ctx->get(std::string(buf) + ".weight"),
                         ctx->get(std::string(buf) + ".bias"), eps);

            auto * fc1_t = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.up.weight").c_str());
            int ffn_dim = fc1_t ? (int)fc1_t->ne[0] : 4304;

            std::vector<float> fc1(T * ffn_dim);
            gv_linear(normed.data(), T, D, ffn_dim, ctx->get(lp + ".ffn.up.weight"),
                      ctx->get(lp + ".ffn.up.bias"), fc1.data());
            for (int i = 0; i < T * ffn_dim; i++) fc1[i] = gv_gelu(fc1[i]);

            std::vector<float> fc2(T * D);
            gv_linear(fc1.data(), T, ffn_dim, D, ctx->get(lp + ".ffn.down.weight"),
                      ctx->get(lp + ".ffn.down.bias"), fc2.data());
            for (int i = 0; i < T * D; i++) x[i] += fc2[i];

            for (int fi = 0; fi < n_feat; fi++) {
                if (li == ctx->feature_layers[fi])
                    layer_outputs[fi].assign(x.begin(), x.end());
            }
        }
    }

    // crispembed-diff: emit each captured feature layer ("vis_layer_<N>") so the
    // first diverging layer pinpoints the bug (works for BOTH the Metal graph and
    // the scalar loop, since both fill layer_outputs).
    if (cb) {
        for (int fi = 0; fi < n_feat; fi++) {
            if (layer_outputs[fi].empty()) continue;
            char nm[32];
            snprintf(nm, sizeof(nm), "vis_layer_%d", ctx->feature_layers[fi]);
            cb(nm, layer_outputs[fi].data(), T * D, ud);
        }
    }

    // Concatenate multi-layer features: output[t * feat_dim + fi * D + d]
    int feat_dim = n_feat * D;
    for (int t = 0; t < T; t++)
        for (int fi = 0; fi < n_feat; fi++)
            for (int d = 0; d < D; d++)
                output[t * feat_dim + fi * D + d] = layer_outputs[fi][t * D + d];
    *out_tokens = T;
}

// ── MLP Projector ──────────────────────────────────────────────────────

static void gv_projector(granite_vision_context * ctx,
                         const float * vis_features, int n_tokens, int feat_dim,
                         float * output) {
    // Fast path: ggml graph (Metal-accelerated matmul)
    if (gv_run_projector_graph(ctx, vis_features, n_tokens, feat_dim, output))
        return;

    // Scalar fallback
    int out_dim = ctx->llm_dim;
    std::vector<float> mid(n_tokens * out_dim);
    gv_linear(vis_features, n_tokens, feat_dim, out_dim,
              ctx->get("proj.linear_1.weight"), ctx->get("proj.linear_1.bias"),
              mid.data());
    // projector_hidden_act = "gelu" (exact erf); see gv_run_projector_graph note.
    for (int i = 0; i < n_tokens * out_dim; i++) mid[i] = core_cpu::gelu_erf(mid[i]);
    gv_linear(mid.data(), n_tokens, out_dim, out_dim,
              ctx->get("proj.linear_2.weight"), ctx->get("proj.linear_2.bias"),
              output);
}

// ── Vision dump for crispembed-diff ─────────────────────────────────────

void granite_vision_dump_vision(granite_vision_context * ctx,
                                 const float * image_f32, int img_h, int img_w,
                                 gv_dump_cb cb, void * ud) {
    if (!ctx || !image_f32 || !cb) return;

    int T_vis = 0;
    int n_feat = (int)ctx->feature_layers.size();
    int feat_dim = n_feat * ctx->vis_dim;

    // Run vision encoder (cb dumps vis_patch_embed + per-feature-layer stages)
    std::vector<float> vis_out(729 * feat_dim);  // max patches
    gv_vision_forward(ctx, image_f32, img_h, img_w, vis_out.data(), &T_vis, cb, ud);

    // Emit concatenated features
    cb("vis_features_concat", vis_out.data(), T_vis * feat_dim, ud);

    // Run projector
    std::vector<float> proj_out(T_vis * ctx->llm_dim);
    gv_projector(ctx, vis_out.data(), T_vis, feat_dim, proj_out.data());
    cb("projector", proj_out.data(), T_vis * ctx->llm_dim, ud);
}

// ── Granite LLM Forward (single token, with KV cache) ──────────────────

static void gv_llm_decode_step(granite_vision_context * ctx,
                                const float * token_embed, int n_past,
                                float * logits,
                                gv_dump_cb dump_cb = nullptr, void * dump_ud = nullptr,
                                bool want_logits = true) {
    int D = ctx->llm_dim;
    int n_heads = ctx->llm_heads;
    int n_kv = ctx->llm_kv_heads;
    int d_head = D / n_heads;
    int kv_repeat = n_heads / n_kv;
    (void)kv_repeat;
    float eps = ctx->rms_eps;
    float res_mul = ctx->residual_multiplier;

    std::vector<float> x(D);
    memcpy(x.data(), token_embed, D * sizeof(float));

    int max_seq = ctx->kv_allocated;

    if (dump_cb) dump_cb("llm_embed_in", x.data(), D, dump_ud);

    for (int li = 0; li < ctx->llm_layers; li++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "llm.layer.%d", li);
        std::string lp(buf);

        // RMSNorm1
        std::vector<float> normed(D);
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D,
                              ctx->get(lp + ".norm1.weight"), eps);

        // GQA Self-Attention with KV cache
        std::vector<float> Q(D), K_new(n_kv * d_head), V_new(n_kv * d_head);
        gv_linear(normed.data(), 1, D, D, ctx->get(lp + ".attn.q.weight"), nullptr, Q.data());
        gv_linear(normed.data(), 1, D, n_kv * d_head, ctx->get(lp + ".attn.k.weight"), nullptr, K_new.data());
        gv_linear(normed.data(), 1, D, n_kv * d_head, ctx->get(lp + ".attn.v.weight"), nullptr, V_new.data());

        // RoPE on Q and K — precomputed frequency table, NEGHALF style
        // (Granite uses rotate_half = split-half, same as HF Llama)
        ctx->rope_freq.apply(Q.data(), n_heads, n_past,
                             core_vlm::RoPEStyle::NEGHALF);
        ctx->rope_freq.apply(K_new.data(), n_kv, n_past,
                             core_vlm::RoPEStyle::NEGHALF);

        // GQA attention with KV cache. Granite scales scores by
        // attention_multiplier (= 1/64), not 1/sqrt(head_dim).
        std::vector<float> attn_out(D, 0.0f);
        core_vlm::gqa_attn_step(Q.data(), K_new.data(), V_new.data(),
                                ctx->kv_cache.data(),
                                n_heads, n_kv, d_head,
                                max_seq, n_past,
                                li, ctx->llm_layers,
                                attn_out.data(),
                                ctx->attention_multiplier);

        // Output projection
        std::vector<float> proj(D);
        gv_linear(attn_out.data(), 1, D, D, ctx->get(lp + ".attn.o.weight"), nullptr, proj.data());

        // Residual with multiplier
        for (int d = 0; d < D; d++) x[d] += proj[d] * res_mul;

        // RMSNorm2 + SiLU MLP
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D,
                              ctx->get(lp + ".norm2.weight"), eps);

        int ffn = ctx->llm_ffn_dim;
        std::vector<float> down(D);
        core_vlm::swiglu_ffn(normed.data(), down.data(), D, ffn,
                             ctx->get(lp + ".ffn.gate.weight"),
                             ctx->get(lp + ".ffn.up.weight"),
                             ctx->get(lp + ".ffn.down.weight"));

        for (int d = 0; d < D; d++) x[d] += down[d] * res_mul;

        if (dump_cb) {
            char nb[48];
            snprintf(nb, sizeof(nb), "llm_layer_%d_out", li);
            dump_cb(nb, x.data(), D, dump_ud);
        }
    }

    // Final RMSNorm
    {
        std::vector<float> tmp(D);
        core_cpu::rmsnorm_cpu(x.data(), tmp.data(), D,
                              ctx->get("llm.norm.weight"), eps);
        memcpy(x.data(), tmp.data(), D * sizeof(float));
    }
    if (dump_cb) dump_cb("llm_final_norm", x.data(), D, dump_ud);

    // LM head (may be tied to embeddings)
    const float * lm_w = ctx->get("llm.lm_head.weight");
    if (!lm_w && ctx->tie_word_embeddings)
        lm_w = ctx->get("llm.embed.weight");

    if (lm_w && (want_logits || dump_cb)) {
        // SIMD-accelerated LM head matmul (49156 × 2048). Skipped during scalar
        // prefill for all but the last token — only the final position seeds generation.
        core_cpu::linear_cpu(x.data(), logits, D, ctx->vocab_size, lm_w, nullptr);
        float inv_scale = 1.0f / ctx->logits_scaling;
        for (int v = 0; v < ctx->vocab_size; v++) logits[v] *= inv_scale;
    }
    if (dump_cb) dump_cb("llm_logits", logits, ctx->vocab_size, dump_ud);
}

// ── LLM decode dump for crispembed-diff ─────────────────────────────────
// Runs a fixed text-only token sequence (matching
// tools/dump_granite_llm_reference.py) through the decoder and emits per-layer
// hidden states + final logits for the last token, so the LLM decode path can
// be validated layer-by-layer against the Python reference.
void granite_vision_dump_llm(granite_vision_context * ctx,
                             const int * tokens, int n_tokens,
                             gv_dump_cb cb, void * ud) {
    if (!ctx || !tokens || n_tokens <= 0 || !cb) return;
    int D = ctx->llm_dim;
    int max_seq = n_tokens + 4;
    ctx->kv_cache.assign((size_t)2 * ctx->llm_layers * max_seq * ctx->llm_kv_heads *
                         (D / ctx->llm_heads), 0.0f);
    ctx->kv_allocated = max_seq;
    ctx->n_past = 0;

    const float * embed_w = ctx->get("llm.embed.weight");
    float emb_mul = ctx->embedding_multiplier;
    std::vector<float> logits(ctx->vocab_size);
    std::vector<float> emb(D);

    // Graph path: validate gv_run_llm_body (the Metal/ggml LLM body) against the
    // same reference. Runs all n_tokens as one batched prefill and dumps the
    // last token's per-layer hidden states — the gv_llm_decode_step scalar path
    // does NOT exercise the graph, so this is the only way to diff it.
    if (std::getenv("CRISPEMBED_GRANITE_LLM_GRAPH") && gv_alloc_kv_cache(ctx, n_tokens + 4)) {
        std::vector<float> embeds((size_t)n_tokens * D);
        for (int t = 0; t < n_tokens; t++)
            for (int d = 0; d < D; d++)
                embeds[(size_t)t * D + d] = embed_w[(size_t)tokens[t] * D + d] * emb_mul;
        // llm_embed_in = last token's scaled embedding (matches the scalar dump).
        cb("llm_embed_in", embeds.data() + (size_t)(n_tokens - 1) * D, D, ud);
        std::vector<float> hidden(D);
        if (gv_run_llm_body(ctx, embeds.data(), n_tokens, 0, hidden.data(), cb, ud)) {
            const float * lm_w = ctx->get("llm.lm_head.weight");
            if (!lm_w && ctx->tie_word_embeddings) lm_w = embed_w;
            if (lm_w) {
                core_cpu::linear_cpu(hidden.data(), logits.data(), D, ctx->vocab_size, lm_w, nullptr);
                float inv = 1.0f / ctx->logits_scaling;
                for (int v = 0; v < ctx->vocab_size; v++) logits[v] *= inv;
                cb("llm_logits", logits.data(), ctx->vocab_size, ud);
            }
            return;
        }
        fprintf(stderr, "granite_vision: LLM graph dump failed, falling back to scalar\n");
    }

    for (int t = 0; t < n_tokens; t++) {
        int id = tokens[t];
        for (int d = 0; d < D; d++) emb[d] = embed_w[(size_t)id * D + d] * emb_mul;
        bool last = (t == n_tokens - 1);
        gv_llm_decode_step(ctx, emb.data(), ctx->n_past, logits.data(),
                           last ? cb : nullptr, last ? ud : nullptr);
        ctx->n_past++;
    }
}

// ── Main recognize function ────────────────────────────────────────────

const char * granite_vision_recognize(granite_vision_context * ctx,
                                       const uint8_t * pixels, int width, int height, int channels,
                                       const char * prompt, int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    int img_size = ctx->vis_image_size;
    int ps = ctx->vis_patch_size;
    int n_patches_side = img_size / ps;
    int T_vis = n_patches_side * n_patches_side;
    int D = ctx->llm_dim;
    int n_feat = (int)ctx->feature_layers.size();
    int feat_dim = n_feat * ctx->vis_dim;

    // Preprocess: resize to img_size × img_size, then SigLIP normalization
    // (rescale 1/255 then (x-mean)/std with mean=std=0.5 → range [-1, 1]).
    // Feeding [0,1] makes the vision tower hallucinate — the parity test fed an
    // already-ranged reference, so this step is not exercised there.
    // (single tile; LLaVA-Next anyres tiling not yet implemented)
    std::vector<float> image(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float sy = (y + 0.5f) * height / img_size - 0.5f;
                float sx = (x + 0.5f) * width / img_size - 0.5f;
                int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                int src_idx = channels > 1 ? (iy * width + ix) * channels + c : iy * width + ix;
                image[c * img_size * img_size + y * img_size + x] =
                    (pixels[src_idx] / 255.0f - 0.5f) / 0.5f;
            }

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Vision encoder
    auto t_vis = std::chrono::steady_clock::now();
    std::vector<float> vis_features(T_vis * feat_dim);
    int n_vis_tokens = 0;
    gv_vision_forward(ctx, image.data(), img_size, img_size, vis_features.data(), &n_vis_tokens);
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_vis).count();
        fprintf(stderr, "[granite_ocr-bench] vision_encoder: %lldms\n", (long long)ms);
    }

    // Projector
    auto t_proj = std::chrono::steady_clock::now();
    std::vector<float> proj_features(n_vis_tokens * D);
    gv_projector(ctx, vis_features.data(), n_vis_tokens, feat_dim, proj_features.data());
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_proj).count();
        fprintf(stderr, "[granite_ocr-bench] projector: %lldms\n", (long long)ms);
    }

    const float * embed_w = ctx->get("llm.embed.weight");
    float emb_mul = ctx->embedding_multiplier;
    std::vector<float> logits(ctx->vocab_size);

    // ── Build the LLaVA-Next chat prompt and tokenize it ────────────────
    // Matches ibm-granite/granite-vision-3.3-2b's chat template:
    //   <|system|>\n<sys text>\n<|user|>\n<image>\n<instruction>\n<|assistant|>\n
    // add_bos_token=False, add_generation_prompt=True. The single <image>
    // marker expands to the n_vis_tokens projected vision rows.
    const char * instruction =
        (prompt && prompt[0]) ? prompt : "Convert this image to text.";
    std::string sys =
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.";
    std::string full =
        "<|system|>\n" + sys + "\n<|user|>\n<image>\n" + std::string(instruction) +
        "\n<|assistant|>\n";

    std::vector<int> prompt_ids;
    if (ctx->have_tokenizer) {
        prompt_ids = ctx->tokenizer.encode_prompt(full);
    } else {
        // No tokenizer embedded → image-only prefill (legacy behaviour).
        prompt_ids.push_back(ctx->image_token_index);
    }

    // ── Sequence sizes ────────────────────────────────────────────────────
    int n_text = 0;
    for (int id : prompt_ids) if (id != ctx->image_token_index) n_text++;
    // prefill_len: total tokens in the prompt after image expansion
    int prefill_len = n_text + n_vis_tokens;
    int max_seq = prefill_len + ctx->max_tokens + 8;

    // ── Fast path: ggml batched prefill + T=1 decode ────────────────────
    // gv_run_llm_body is diff-validated against granite-llm-ref (cos 0.9999
    // through logits) AND end-to-end correct on both Metal and ggml-CPU. DEFAULT
    // ON — ~22 s total on Metal vs the scalar decode's ~8 min prefill; the CPU
    // graph is likewise far faster than the per-call-dequantizing scalar decode.
    // The scalar decode (gv_llm_decode_step, cos=1.0) is the opt-out / fallback if
    // KV alloc fails. Force with CRISPEMBED_GRANITE_LLM_SCALAR=1.
    bool use_graph = false;
    if (!std::getenv("CRISPEMBED_GRANITE_LLM_SCALAR"))
        use_graph = gv_alloc_kv_cache(ctx, max_seq);

    auto t_prefill = std::chrono::steady_clock::now();
    ctx->n_past = 0;
    std::vector<float> emb(D);

    if (use_graph) {
        // Build the full prefill embedding sequence in one contiguous buffer.
        // All tokens (text and vision) are scaled by embedding_multiplier —
        // HF applies it to the entire inputs_embeds tensor after splicing.
        std::vector<float> prefill_embeds((size_t)prefill_len * D);
        int pos = 0;
        for (int id : prompt_ids) {
            if (id == ctx->image_token_index) {
                for (int t = 0; t < n_vis_tokens; t++, pos++) {
                    const float * vr = proj_features.data() + (size_t)t * D;
                    for (int d = 0; d < D; d++)
                        prefill_embeds[(size_t)pos * D + d] = vr[d] * emb_mul;
                }
            } else {
                for (int d = 0; d < D; d++)
                    prefill_embeds[(size_t)pos * D + d] = embed_w[(size_t)id * D + d] * emb_mul;
                pos++;
            }
        }

        // One ggml call for the entire prefill (replaces prefill_len serial decode
        // steps). LM head runs in-graph → logits filled directly (no CPU matmul).
        std::vector<float> hidden(D);
        if (!gv_run_llm_body(ctx, prefill_embeds.data(), prefill_len, 0, hidden.data(),
                             nullptr, nullptr, logits.data())) {
            use_graph = false;  // fall through to scalar path
        } else {
            ctx->n_past = prefill_len;
        }
    }

    if (!use_graph) {
        // Scalar fallback: allocate F32 KV cache and process token-by-token
        ctx->kv_cache.assign((size_t)2 * ctx->llm_layers * max_seq * ctx->llm_kv_heads *
                             (D / ctx->llm_heads), 0.0f);
        ctx->kv_allocated = max_seq;
        ctx->n_past = 0;
        for (size_t pi = 0; pi < prompt_ids.size(); pi++) {
            int id = prompt_ids[pi];
            bool last_prompt = (pi + 1 == prompt_ids.size());
            if (id == ctx->image_token_index) {
                for (int t = 0; t < n_vis_tokens; t++) {
                    const float * vr = proj_features.data() + (size_t)t * D;
                    for (int d = 0; d < D; d++) emb[d] = vr[d] * emb_mul;
                    gv_llm_decode_step(ctx, emb.data(), ctx->n_past, logits.data(),
                                       nullptr, nullptr, /*want_logits=*/false);
                    ctx->n_past++;
                }
            } else {
                for (int d = 0; d < D; d++) emb[d] = embed_w[(size_t)id * D + d] * emb_mul;
                gv_llm_decode_step(ctx, emb.data(), ctx->n_past, logits.data(),
                                   nullptr, nullptr, /*want_logits=*/last_prompt);
                ctx->n_past++;
            }
        }
    }

    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_prefill).count();
        fprintf(stderr, "[granite_ocr-bench] prefill (%s, %d tokens): %lldms\n",
                use_graph ? "ggml-batch" : "scalar", ctx->n_past, (long long)ms);
    }

    // ── Greedy decode → detokenized UTF-8 text ──────────────────────────
    ctx->output_text.clear();
    ctx->char_confidences.clear();

    long long decode_total_ms = 0;
    int decode_steps = 0;
    for (int step = 0; step < ctx->max_tokens; step++) {
        auto t_step = std::chrono::steady_clock::now();
        int best_id = 0;
        float best_score = logits[0];
        for (int v = 1; v < ctx->vocab_size; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }

        if (best_id == ctx->eos_id) break;

        float sum_e = 0;
        for (int v = 0; v < ctx->vocab_size; v++)
            sum_e += expf(logits[v] - best_score);
        ctx->char_confidences.push_back(1.0f / sum_e);

        if (ctx->have_tokenizer)
            ctx->output_text += ctx->tokenizer.decode_one(best_id);
        else
            ctx->output_text += "<" + std::to_string(best_id) + ">";

        for (int d = 0; d < D; d++) emb[d] = embed_w[(size_t)best_id * D + d] * emb_mul;

        if (use_graph) {
            std::vector<float> hidden(D);
            if (!gv_run_llm_body(ctx, emb.data(), 1, ctx->n_past, hidden.data(),
                                 nullptr, nullptr, logits.data())) {
                use_graph = false;
                // fall back to scalar for remaining steps
                ctx->kv_cache.assign((size_t)2 * ctx->llm_layers * max_seq *
                                     ctx->llm_kv_heads * (D / ctx->llm_heads), 0.0f);
                ctx->kv_allocated = max_seq;
                // Note: n_past is already correct; scalar cache starts cold but
                // worst case is slightly wrong KV — acceptable for a mid-run fallback.
                gv_llm_decode_step(ctx, emb.data(), ctx->n_past, logits.data());
            }
            // else: logits filled in-graph by the LM head
        } else {
            gv_llm_decode_step(ctx, emb.data(), ctx->n_past, logits.data());
        }

        ctx->n_past++;
        if (bench) {
            auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_step).count();
            decode_total_ms += step_ms;
            decode_steps++;
        }
    }
    if (bench) {
        fprintf(stderr, "[granite_ocr-bench] decode: %lldms (%d steps, %.1f ms/tok)\n",
                decode_total_ms, decode_steps,
                decode_steps ? (float)decode_total_ms / decode_steps : 0.0f);
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[granite_ocr-bench] total: %lldms\n", (long long)total_ms);
    }

    if (out_len) *out_len = (int)ctx->output_text.size();
    return ctx->output_text.c_str();
}

const float * granite_vision_confidences(const granite_vision_context * ctx, int * n) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n) *n = 0;
        return nullptr;
    }
    if (n) *n = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float granite_vision_mean_confidence(const granite_vision_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
