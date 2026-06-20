// smoldocling_ocr.cpp — SmolDocling OCR engine (SigLIP ViT + SmolLM2-135M).
//
// Architecture:
//   1. Load GGUF (core_gguf)
//   2. Vision encoder forward (SigLIP ViT, 12 layers, 768d)
//   3. Pixel shuffle connector (scale=4): [1024, 768] -> [64, 12288]
//   4. Linear projection: [64, 12288] -> [64, 576]
//   5. BPE tokenizer for prompt + output detokenization
//   6. Token embedding + vision splicing (masked_scatter at image_token_id)
//   7. Autoregressive LLM decode (SmolLM2-135M, 30 layers, GQA 9/3, KV cache)
//
// Follows the same CPU-scalar pattern as granite_vision_ocr.cpp.

#include "smoldocling_ocr.h"
#include "core/gguf_loader.h"
#include "core/vlm_attention.h"
#include "ggml-cpu.h"

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
// stb_image for file loading
extern "C" {
    unsigned char * stbi_load(const char * filename, int * x, int * y, int * comp, int req_comp);
    void stbi_image_free(void * retval_from_stbi_load);
}

// ── Helpers ───────────────────────────────────────────────────────────

static const float * sd_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(t->data, buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

// sd_linear: delegate to SIMD-accelerated core_cpu::linear_cpu
static void sd_linear(const float * input, int n, int id, int od,
                      const float * weight, const float * bias, float * output) {
    for (int i = 0; i < n; i++)
        core_cpu::linear_cpu(input + i * id, output + i * od, id, od, weight, bias);
}

// ── GPT-2 byte-level BPE tables ──────────────────────────────────────
// Same as CrispASR core/bpe.h — maps raw bytes ↔ printable unicode
// codepoints so the BPE vocab can survive JSON roundtrips.

static const std::vector<int> & sd_byte_encoder() {
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

static const std::unordered_map<uint32_t, uint8_t> & sd_byte_decoder() {
    static std::unordered_map<uint32_t, uint8_t> table;
    static bool init = false;
    if (init) return table;
    auto & enc = sd_byte_encoder();
    for (int b = 0; b < 256; b++) table[(uint32_t)enc[b]] = (uint8_t)b;
    init = true;
    return table;
}

// Encode a codepoint as UTF-8
static void sd_utf8_encode(uint32_t cp, std::string & out) {
    if (cp < 0x80) { out.push_back((char)cp); }
    else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
    else if (cp < 0x10000) { out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
    else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
}

// Encode raw bytes → GPT-2 unicode string
static std::string sd_bytes_to_unicode(const char * bytes, size_t n) {
    auto & enc = sd_byte_encoder();
    std::string out;
    for (size_t i = 0; i < n; i++) sd_utf8_encode((uint32_t)enc[(unsigned char)bytes[i]], out);
    return out;
}

// Decode a BPE token string (GPT-2 unicode codepoints) back to raw bytes
static std::string sd_token_to_bytes(const std::string & token) {
    auto & dec = sd_byte_decoder();
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

// ── BPE Tokenizer ─────────────────────────────────────────────────────

struct sd_tokenizer {
    std::vector<std::string> vocab;                       // id -> token string
    std::unordered_map<std::string, int> token_to_id;     // token string -> id
    std::unordered_map<std::string, int> merge_rank;      // "left right" -> rank
    int eos_id = 0;
    int bos_id = 1;
    int pad_id = 2;

    bool load(gguf_context * meta) {
        // Try both key conventions
        vocab = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        if (vocab.empty())
            vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        if (vocab.empty()) {
            fprintf(stderr, "smoldocling: no tokenizer.tokens in GGUF\n");
            return false;
        }
        for (int i = 0; i < (int)vocab.size(); i++)
            token_to_id[vocab[i]] = i;

        auto merge_strs = core_gguf::kv_str_array(meta, "tokenizer.merges");
        if (merge_strs.empty())
            merge_strs = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merge_strs.size(); i++) {
            merge_rank[merge_strs[i]] = i;
        }

        return true;
    }

    // GPT-2 byte-level BPE encode
    std::vector<int> encode(const std::string & text) const {
        if (text.empty()) return {};

        // Step 1: convert raw bytes to GPT-2 unicode string
        std::string unicode = sd_bytes_to_unicode(text.data(), text.size());

        // Step 2: split into per-byte unicode symbols
        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < unicode.size()) {
            unsigned char c = (unsigned char)unicode[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            symbols.push_back(unicode.substr(i, len));
            i += len;
        }

        // Step 3: greedy BPE merge (lowest rank first)
        while (symbols.size() > 1) {
            int best_rank = INT_MAX, best_i = -1;
            for (int k = 0; k + 1 < (int)symbols.size(); k++) {
                std::string pair = symbols[k] + " " + symbols[k + 1];
                auto it = merge_rank.find(pair);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = k;
                }
            }
            if (best_i < 0) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }

        // Step 4: map to IDs
        std::vector<int> ids;
        for (auto & s : symbols) {
            auto it = token_to_id.find(s);
            if (it != token_to_id.end()) ids.push_back(it->second);
        }
        return ids;
    }

    // Decode token IDs to UTF-8 string (reverses GPT-2 byte mapping)
    // Added/special tokens (IDs 0-16 and ≥49152) are literal strings.
    // Base BPE tokens (IDs 17-49151) use GPT-2 byte encoding.
    std::string decode(const std::vector<int> & ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= (int)vocab.size()) continue;
            // Skip control tokens
            if (id <= 2) continue;  // BOS=1, EOS=2, endoftext=0
            if (id == 49279) continue;  // <end_of_utterance>
            const std::string & piece = vocab[id];
            // Added tokens are literal (not GPT-2 byte encoded)
            if (id <= 16 || id >= 49152) {
                result += piece;
            } else {
                // Base BPE vocab: reverse GPT-2 byte mapping
                result += sd_token_to_bytes(piece);
            }
        }
        return result;
    }
};

// ── Context ───────────────────────────────────────────────────────────

static constexpr int kSdLlmGraphCap = 4096;

struct smoldocling_context {
    // Vision hparams
    int vis_dim, vis_layers, vis_heads, vis_image_size, vis_patch_size;
    int vis_intermediate;

    // Connector
    int connector_scale;

    // LLM hparams
    int llm_dim, llm_layers, llm_heads, llm_kv_heads, llm_ffn_dim;
    int head_dim, vocab_size, image_token_id;
    float rms_eps, rope_theta;

    int max_tokens;
    int n_threads;
    bool bench = false;

    // Tokenizer
    sd_tokenizer tokenizer;

    // Weight storage
    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;   // caches dequantized weights (replaces wbufs)

    // ggml backend (shared: vis encoder + LLM decoder)
    ggml_backend_t backend = nullptr;

    // LLM decoder: reusable scheduler + pre-allocated metadata buffer
    ggml_backend_sched_t llm_sched = nullptr;
    std::vector<uint8_t> llm_compute_meta;

    // F16 KV cache on the backend (re-allocated per image call)
    ggml_context        * kvc_ctx    = nullptr;
    ggml_tensor         * kvc_k      = nullptr;  // [kv_dim, max_seq, n_layers] F16
    ggml_tensor         * kvc_v      = nullptr;
    ggml_backend_buffer_t kvc_buf    = nullptr;
    int                   kvc_max_seq = 0;

    // RoPE frequency table (precomputed once at init)
    core_vlm::RoPEFreqTable rope_freq;

    // Scalar fallback KV cache (used when ggml graph path is unavailable)
    std::vector<float> kv_cache;
    int kv_allocated = 0;
    int n_past       = 0;

    // Output buffer
    std::string output_text;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) return nullptr;
        return dcache.get(t);
    }
};

// ── Init / Free ───────────────────────────────────────────────────────

smoldocling_context * smoldocling_init(const char * model_path, int n_threads) {
    auto * ctx = new smoldocling_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->max_tokens = 128;  // TODO: increase after parity is verified
    ctx->kv_allocated = 0;
    ctx->n_past = 0;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "smoldocling: failed to open %s\n", model_path);
        delete ctx; return nullptr;
    }

    // Vision hparams
    ctx->vis_dim        = core_gguf::kv_u32(meta, "smoldocling.vision.hidden_size", 768);
    ctx->vis_heads      = core_gguf::kv_u32(meta, "smoldocling.vision.num_heads", 12);
    ctx->vis_layers     = core_gguf::kv_u32(meta, "smoldocling.vision.num_layers", 12);
    ctx->vis_patch_size = core_gguf::kv_u32(meta, "smoldocling.vision.patch_size", 16);
    ctx->vis_image_size = core_gguf::kv_u32(meta, "smoldocling.vision.image_size", 512);
    ctx->vis_intermediate = core_gguf::kv_u32(meta, "smoldocling.vision.intermediate_size", 3072);

    // Connector
    ctx->connector_scale = core_gguf::kv_u32(meta, "smoldocling.connector.scale_factor", 4);

    // LLM hparams
    ctx->llm_dim        = core_gguf::kv_u32(meta, "smoldocling.hidden_size", 576);
    ctx->llm_heads      = core_gguf::kv_u32(meta, "smoldocling.num_attention_heads", 9);
    ctx->llm_kv_heads   = core_gguf::kv_u32(meta, "smoldocling.num_key_value_heads", 3);
    ctx->llm_layers     = core_gguf::kv_u32(meta, "smoldocling.num_hidden_layers", 30);
    ctx->llm_ffn_dim    = core_gguf::kv_u32(meta, "smoldocling.intermediate_size", 1536);
    ctx->head_dim       = core_gguf::kv_u32(meta, "smoldocling.head_dim", 64);
    ctx->vocab_size     = core_gguf::kv_u32(meta, "smoldocling.vocab_size", 49280);
    ctx->image_token_id = core_gguf::kv_u32(meta, "smoldocling.image_token_id", 49190);

    int idx;
    idx = gguf_find_key(meta, "smoldocling.rms_norm_eps");
    ctx->rms_eps = idx >= 0 ? gguf_get_val_f32(meta, idx) : 1e-5f;
    idx = gguf_find_key(meta, "smoldocling.rope_theta");
    ctx->rope_theta = idx >= 0 ? gguf_get_val_f32(meta, idx) : 100000.0f;
    ctx->rope_freq.precompute(ctx->head_dim, ctx->rope_theta);

    // Tokenizer
    if (!ctx->tokenizer.load(meta)) {
        fprintf(stderr, "smoldocling: failed to load tokenizer\n");
        core_gguf::free_metadata(meta);
        delete ctx; return nullptr;
    }

    core_gguf::free_metadata(meta);

    // Load weights — keep backend for ggml graph compute
    ctx->backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, ctx->backend, "smoldocling", ctx->wl)) {
        fprintf(stderr, "smoldocling: failed to load weights\n");
        ggml_backend_free(ctx->backend); ctx->backend = nullptr;
        delete ctx; return nullptr;
    }

    int n_patches = (ctx->vis_image_size / ctx->vis_patch_size);
    n_patches *= n_patches;
    int S = ctx->connector_scale;
    int connector_out = n_patches / (S * S);

    fprintf(stderr, "smoldocling: vis=%dL x %dd (%d patches -> %d after shuffle), "
            "llm=%dL x %dd (heads=%d/%d, ffn=%d), vocab=%d, %d tensors\n",
            ctx->vis_layers, ctx->vis_dim, n_patches, connector_out,
            ctx->llm_layers, ctx->llm_dim,
            ctx->llm_heads, ctx->llm_kv_heads, ctx->llm_ffn_dim,
            ctx->vocab_size, (int)ctx->wl.tensors.size());

    ctx->bench = (std::getenv("CRISPEMBED_SMOLDOCLING_BENCH") != nullptr);

    // LLM scheduler: reuse the same CPU backend (weights already in CPU memory)
    {
        size_t meta_sz = ggml_tensor_overhead() * kSdLlmGraphCap
                       + ggml_graph_overhead_custom(kSdLlmGraphCap, false);
        ctx->llm_compute_meta.resize(meta_sz);
        ggml_backend_t backends[1] = { ctx->backend };
        ctx->llm_sched = ggml_backend_sched_new(backends, nullptr, 1, kSdLlmGraphCap, false, false);
        if (!ctx->llm_sched) {
            fprintf(stderr, "smoldocling: failed to create LLM scheduler — scalar fallback only\n");
        }
    }

    return ctx;
}

void smoldocling_free(smoldocling_context * ctx) {
    if (ctx) {
        if (ctx->kvc_buf) ggml_backend_buffer_free(ctx->kvc_buf);
        if (ctx->kvc_ctx) ggml_free(ctx->kvc_ctx);
        if (ctx->llm_sched) ggml_backend_sched_free(ctx->llm_sched);
        core_gguf::free_weights(ctx->wl);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

// ── SigLIP Vision Encoder (ggml graph — BLAS accelerated) ────────────

// Run SigLIP ViT via ggml graph. Builds a compute graph with all
// 12 layers, then runs it in one shot. Much faster than CPU-scalar
// for T=1024 tokens (uses BLAS for matmuls, flash_attn_ext for attention).
//
// Input: [3, img_h, img_w] float, normalized to [-1, 1]
// Output: [n_patches, vis_dim]
static void sd_vision_forward(smoldocling_context * ctx,
                               const float * image, int img_h, int img_w,
                               float * output, int * out_tokens) {
    int ps = ctx->vis_patch_size;
    int ph = img_h / ps, pw = img_w / ps;
    int T = ph * pw;  // 1024
    int D = ctx->vis_dim;  // 768
    int nh = ctx->vis_heads;  // 12
    int hd = D / nh;  // 64
    float eps = 1e-6f;

    // ── Patch embedding: im2col + ggml matmul ──
    // Gated: CRISPEMBED_SMOLDOCLING_SCALAR_PATCH=1 for scalar fallback
    auto * pe_t = core_gguf::try_get(ctx->wl.tensors, "vis.patch_embed.weight");
    auto * pb_t = core_gguf::try_get(ctx->wl.tensors, "vis.patch_embed.bias");
    int patch_dim = 3 * ps * ps;  // 588

    // im2col: extract non-overlapping patches → [T, patch_dim]
    std::vector<float> im2col(T * patch_dim, 0.0f);
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            int t = py * pw + px;
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < ps; ky++) {
                    int iy = py * ps + ky;
                    if (iy >= img_h) continue;
                    for (int kx = 0; kx < ps; kx++) {
                        int ix = px * ps + kx;
                        if (ix >= img_w) continue;
                        im2col[t * patch_dim + c * ps * ps + ky * ps + kx] =
                            image[c * img_h * img_w + iy * img_w + ix];
                    }
                }
        }

    std::vector<float> patch_embed(T * D);
    static const bool scalar_patch = (std::getenv("CRISPEMBED_SMOLDOCLING_SCALAR_PATCH") != nullptr);
    if (scalar_patch) {
        std::vector<float> pe_buf, pb_buf;
        const float * pe_w = sd_to_f32(pe_t, pe_buf);
        const float * pe_b = pb_t ? sd_to_f32(pb_t, pb_buf) : nullptr;
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++) {
                float s = pe_b ? pe_b[d] : 0.0f;
                for (int k = 0; k < patch_dim; k++)
                    s += im2col[t * patch_dim + k] * pe_w[d * patch_dim + k];
                patch_embed[t * D + d] = s;
            }
    } else {
        // ggml graph: matmul weight × im2col → [D, T]
        size_t buf_sz = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
        ggml_init_params eip{buf_sz, nullptr, true};
        ggml_context *eg = ggml_init(eip);
        ggml_tensor *w = ggml_reshape_2d(eg, pe_t, patch_dim, D);
        ggml_tensor *inp = ggml_new_tensor_2d(eg, GGML_TYPE_F32, patch_dim, T);
        ggml_set_name(inp, "im2col"); ggml_set_input(inp);
        ggml_tensor *out = ggml_mul_mat(eg, w, inp);
        if (pb_t) {
            ggml_tensor *b = pb_t;
            if (b->type != GGML_TYPE_F32) b = ggml_cast(eg, b, GGML_TYPE_F32);
            out = ggml_add(eg, out, b);
        }
        ggml_set_name(out, "pe_out"); ggml_set_output(out);
        ggml_cgraph *egf = ggml_new_graph(eg);
        ggml_build_forward_expand(egf, out);
        ggml_backend_sched_t pe_sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 16, false, false);
        ggml_backend_sched_reset(pe_sched);
        if (!ggml_backend_sched_alloc_graph(pe_sched, egf)) {
            fprintf(stderr, "smoldocling: patch_embed graph alloc failed\n");
            ggml_backend_sched_free(pe_sched);
            ggml_free(eg);
            return;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(egf, "im2col"),
                                im2col.data(), 0, T * patch_dim * sizeof(float));
        ggml_backend_sched_graph_compute(pe_sched, egf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(egf, "pe_out"),
                                patch_embed.data(), 0, T * D * sizeof(float));
        ggml_backend_sched_free(pe_sched);
        ggml_free(eg);
    }

    // Add position embedding
    auto * pos_t = core_gguf::try_get(ctx->wl.tensors, "vis.pos_embed.weight");
    if (pos_t) {
        std::vector<float> pos_buf;
        const float * pos_w = sd_to_f32(pos_t, pos_buf);
        for (int i = 0; i < T * D; i++)
            patch_embed[i] += pos_w[i];
    }

    // ── Build ggml graph for transformer layers ──
    const int max_nodes = 2048;
    size_t ctx_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead();
    ggml_init_params ip = { ctx_size, nullptr, true };
    ggml_context * g_ctx = ggml_init(ip);

    // Input tensor (set from CPU data)
    ggml_tensor * x = ggml_new_tensor_2d(g_ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "vis_input");
    ggml_set_input(x);

    // Helper: cast to f32 if needed (norm weights are often f16 in GGUF)
    auto cast_f32 = [&](ggml_tensor * t) -> ggml_tensor * {
        if (!t || t->type == GGML_TYPE_F32) return t;
        return ggml_cast(g_ctx, t, GGML_TYPE_F32);
    };

    // Transformer layers
    for (int li = 0; li < ctx->vis_layers; li++) {
        char buf[64];
        ggml_tensor * residual = x;

        // LN1
        snprintf(buf, sizeof(buf), "vis.layers.%d.ln1.weight", li);
        auto * ln1_w = cast_f32(core_gguf::try_get(ctx->wl.tensors, buf));
        snprintf(buf, sizeof(buf), "vis.layers.%d.ln1.bias", li);
        auto * ln1_b = cast_f32(core_gguf::try_get(ctx->wl.tensors, buf));
        x = ggml_norm(g_ctx, x, eps);
        x = ggml_mul(g_ctx, x, ln1_w);
        if (ln1_b) x = ggml_add(g_ctx, x, ln1_b);

        // MHSA with separate Q, K, V projections
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.q.weight", li);
        auto * q_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.q.bias", li);
        auto * q_b = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.k.weight", li);
        auto * k_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.k.bias", li);
        auto * k_b = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.v.weight", li);
        auto * v_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.v.bias", li);
        auto * v_b = core_gguf::try_get(ctx->wl.tensors, buf);

        ggml_tensor * Q = ggml_mul_mat(g_ctx, q_w, x);
        if (q_b) Q = ggml_add(g_ctx, Q, cast_f32(q_b));
        ggml_tensor * K = ggml_mul_mat(g_ctx, k_w, x);
        if (k_b) K = ggml_add(g_ctx, K, cast_f32(k_b));
        ggml_tensor * V = ggml_mul_mat(g_ctx, v_w, x);
        if (v_b) V = ggml_add(g_ctx, V, cast_f32(v_b));

        // Reshape [D, T] → [hd, nh, T] → permute to [hd, T, nh]
        Q = ggml_reshape_3d(g_ctx, Q, hd, nh, T);
        K = ggml_reshape_3d(g_ctx, K, hd, nh, T);
        V = ggml_reshape_3d(g_ctx, V, hd, nh, T);
        Q = ggml_permute(g_ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(g_ctx, K, 0, 2, 1, 3);
        V = ggml_permute(g_ctx, V, 0, 2, 1, 3);

        // Flash attention (bidirectional — no causal mask)
        float scale = 1.0f / sqrtf((float)hd);
        ggml_tensor * attn = ggml_flash_attn_ext(g_ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g_ctx, attn, D, T);

        // Output projection
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.out.weight", li);
        auto * o_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.attn.out.bias", li);
        auto * o_b = core_gguf::try_get(ctx->wl.tensors, buf);
        attn = ggml_mul_mat(g_ctx, o_w, attn);
        if (o_b) attn = ggml_add(g_ctx, attn, cast_f32(o_b));

        // Residual
        x = ggml_add(g_ctx, residual, attn);

        // LN2
        residual = x;
        snprintf(buf, sizeof(buf), "vis.layers.%d.ln2.weight", li);
        auto * ln2_w = cast_f32(core_gguf::try_get(ctx->wl.tensors, buf));
        snprintf(buf, sizeof(buf), "vis.layers.%d.ln2.bias", li);
        auto * ln2_b = cast_f32(core_gguf::try_get(ctx->wl.tensors, buf));
        x = ggml_norm(g_ctx, x, eps);
        x = ggml_mul(g_ctx, x, ln2_w);
        if (ln2_b) x = ggml_add(g_ctx, x, ln2_b);

        // MLP: fc1 → GELU → fc2
        snprintf(buf, sizeof(buf), "vis.layers.%d.mlp.fc1.weight", li);
        auto * fc1_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.mlp.fc1.bias", li);
        auto * fc1_b = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.mlp.fc2.weight", li);
        auto * fc2_w = core_gguf::try_get(ctx->wl.tensors, buf);
        snprintf(buf, sizeof(buf), "vis.layers.%d.mlp.fc2.bias", li);
        auto * fc2_b = core_gguf::try_get(ctx->wl.tensors, buf);

        x = ggml_mul_mat(g_ctx, fc1_w, x);
        if (fc1_b) x = ggml_add(g_ctx, x, cast_f32(fc1_b));
        x = ggml_gelu(g_ctx, x);
        x = ggml_mul_mat(g_ctx, fc2_w, x);
        if (fc2_b) x = ggml_add(g_ctx, x, cast_f32(fc2_b));

        // Residual
        x = ggml_add(g_ctx, residual, x);
    }

    // Post-layernorm
    auto * pln_w = cast_f32(core_gguf::try_get(ctx->wl.tensors, "vis.post_ln.weight"));
    auto * pln_b = cast_f32(core_gguf::try_get(ctx->wl.tensors, "vis.post_ln.bias"));
    x = ggml_norm(g_ctx, x, eps);
    x = ggml_mul(g_ctx, x, pln_w);
    if (pln_b) x = ggml_add(g_ctx, x, pln_b);

    ggml_set_name(x, "vis_output");
    ggml_set_output(x);

    // Build and compute graph
    ggml_cgraph * gf = ggml_new_graph_custom(g_ctx, max_nodes, false);
    ggml_build_forward_expand(gf, x);

    // Use backend scheduler with model weights buffer
    ggml_backend_sched_t sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, max_nodes, false, false);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    // Set input data
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "vis_input");
    ggml_backend_tensor_set(inp, patch_embed.data(), 0, T * D * sizeof(float));

    // Compute
    ggml_backend_sched_graph_compute(sched, gf);

    // Read output
    ggml_tensor * out = ggml_graph_get_tensor(gf, "vis_output");
    ggml_backend_tensor_get(out, output, 0, T * D * sizeof(float));
    *out_tokens = T;

    ggml_backend_sched_free(sched);
    ggml_free(g_ctx);
}

// ── Pixel Shuffle Connector ───────────────────────────────────────────

// Pixel shuffle: reshapes [B, H*W, D] -> [B, H*W/S^2, D*S^2]
// then linear projection to LLM dim.
//
// Steps (for S=4, H=W=32, D=768):
//   1. view as (H=32, W=32, D=768)
//   2. reshape to (H=32, W/S=8, D*S=3072)
//   3. transpose to (W/S=8, H=32, D*S=3072)
//   4. reshape to (W/S=8, H/S=8, D*S*S=12288)
//   5. transpose to (H/S=8, W/S=8, D*S*S=12288)
//   6. flatten to (H*W/S^2=64, D*S^2=12288)
static void sd_pixel_shuffle(const float * input, int H, int W, int D, int S,
                              float * output) {
    // output shape: (H/S, W/S, D*S*S)
    int Ho = H / S, Wo = W / S;
    int Do = D * S * S;

    // Direct computation following the 6-step algorithm:
    // output[ho, wo, :] gathers from input arranged as (H, W, D)
    // The pixel shuffle groups S consecutive rows and S consecutive columns,
    // concatenating their features.
    //
    // Mapping: for output position (ho, wo), the feature vector is the
    // concatenation of input[(ho*S + sh), (wo*S + sw), :] for all (sw, sh)
    // pairs — but with the specific ordering from the algorithm above.
    //
    // Following the exact transpose sequence:
    // Step 1: view as (H, W, D)
    // Step 2: reshape (H, W/S, S, D) -> merge last two -> (H, W/S, D*S)
    //   intermediate[h, wo, d*S + sw] = input[h, wo*S + sw, d]
    // Step 3: transpose dims 0,1 -> (W/S, H, D*S)
    //   trans1[wo, h, :] = intermediate[h, wo, :]
    // Step 4: reshape (W/S, H/S, S, D*S) -> merge last two -> (W/S, H/S, D*S*S)
    //   inter2[wo, ho, d_s*S + sh] = trans1[wo, ho*S + sh, d_s]
    //     where d_s = d*S + sw
    // Step 5: transpose dims 0,1 -> (H/S, W/S, D*S*S)
    //   result[ho, wo, :] = inter2[wo, ho, :]
    //
    // Combined: result[ho, wo, sh*D*S + sw*D + d] = input[ho*S + sh, wo*S + sw, d]
    // The output groups by (sh, sw) blocks, each block is D values contiguous.

    for (int ho = 0; ho < Ho; ho++) {
        for (int wo = 0; wo < Wo; wo++) {
            float * out_row = output + (ho * Wo + wo) * Do;
            for (int sh = 0; sh < S; sh++) {
                for (int sw = 0; sw < S; sw++) {
                    int src_h = ho * S + sh;
                    int src_w = wo * S + sw;
                    const float * src_row = input + (src_h * W + src_w) * D;
                    int dst_off = sh * D * S + sw * D;
                    memcpy(out_row + dst_off, src_row, D * sizeof(float));
                }
            }
        }
    }
}

static void sd_connector(smoldocling_context * ctx,
                          const float * vis_features, int n_tokens,
                          float * output, int * out_n) {
    int D = ctx->vis_dim;    // 768
    int S = ctx->connector_scale;  // 4
    int H = (int)sqrtf((float)n_tokens);  // 32
    int W = H;

    int Ho = H / S, Wo = W / S;  // 8, 8
    int Do = D * S * S;           // 12288
    int n_out = Ho * Wo;          // 64

    // Pixel shuffle
    std::vector<float> shuffled(n_out * Do);
    sd_pixel_shuffle(vis_features, H, W, D, S, shuffled.data());

    // Linear projection: [12288] -> [576], no bias
    int llm_dim = ctx->llm_dim;
    sd_linear(shuffled.data(), n_out, Do, llm_dim,
              ctx->get("connector.proj.weight"), nullptr, output);

    *out_n = n_out;
}

// ── F16 KV cache management ───────────────────────────────────────────

static void sd_free_kv_cache(smoldocling_context *ctx) {
    if (ctx->kvc_buf) { ggml_backend_buffer_free(ctx->kvc_buf); ctx->kvc_buf = nullptr; }
    if (ctx->kvc_ctx) { ggml_free(ctx->kvc_ctx); ctx->kvc_ctx = nullptr; }
    ctx->kvc_k = nullptr;
    ctx->kvc_v = nullptr;
    ctx->kvc_max_seq = 0;
}

static bool sd_alloc_kv_cache(smoldocling_context *ctx, int max_seq) {
    sd_free_kv_cache(ctx);

    const int n_layers = ctx->llm_layers;
    const int kv_dim   = ctx->llm_kv_heads * ctx->head_dim;

    ggml_init_params ip{2 * ggml_tensor_overhead() + 256, nullptr, true};
    ctx->kvc_ctx = ggml_init(ip);
    if (!ctx->kvc_ctx) return false;

    ctx->kvc_k = ggml_new_tensor_3d(ctx->kvc_ctx, GGML_TYPE_F16, kv_dim, max_seq, n_layers);
    ctx->kvc_v = ggml_new_tensor_3d(ctx->kvc_ctx, GGML_TYPE_F16, kv_dim, max_seq, n_layers);
    ggml_set_name(ctx->kvc_k, "sd_kv_k");
    ggml_set_name(ctx->kvc_v, "sd_kv_v");

    ctx->kvc_buf = ggml_backend_alloc_ctx_tensors(ctx->kvc_ctx, ctx->backend);
    if (!ctx->kvc_buf) {
        fprintf(stderr, "smoldocling: KV cache allocation failed (max_seq=%d)\n", max_seq);
        sd_free_kv_cache(ctx);
        return false;
    }
    ggml_backend_buffer_clear(ctx->kvc_buf, 0);
    ctx->kvc_max_seq = max_seq;

    size_t bytes = ggml_backend_buffer_get_size(ctx->kvc_buf);
    fprintf(stderr, "  smoldocling KV cache: %d layers, max_seq=%d, %.1f MB\n",
            n_layers, max_seq, (float)bytes / (1024.0f * 1024.0f));
    return true;
}

// ── LLM body via ggml graph (batched — handles prefill T>1 and decode T=1) ─

// embeds[T*D]: token embeddings (F32). n_past: tokens already in KV cache.
// Writes K/V into kvc_k/kvc_v at positions [n_past..n_past+T-1].
// Reads hidden state of last token into hidden_out[D].
// Returns false on allocation or compute failure.
static bool sd_run_llm_body(smoldocling_context *ctx,
                             const float *embeds, int T, int n_past,
                             float *hidden_out) {
    const int D        = ctx->llm_dim;         // 576
    const int n_heads  = ctx->llm_heads;       // 9
    const int n_kv     = ctx->llm_kv_heads;    // 3
    const int hd       = ctx->head_dim;        // 64
    const int n_layers = ctx->llm_layers;      // 30
    const int kv_dim   = n_kv * hd;            // 192
    const float eps    = ctx->rms_eps;
    const float scale  = 1.0f / sqrtf((float)hd);
    const int kv_total = n_past + T;

    ggml_init_params ip{ctx->llm_compute_meta.size(),
                        ctx->llm_compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    if (!g) return false;

    ggml_cgraph *gf = ggml_new_graph_custom(g, kSdLlmGraphCap, false);

    // Inputs
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "llm_embeds");
    ggml_set_input(x);

    ggml_tensor *pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Causal mask [kv_total, T] F16: mask[k,q]=0 if k<=n_past+q else -inf
    ggml_tensor *causal_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, kv_total, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    for (int il = 0; il < n_layers; il++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "llm.layers.%d", il);
        std::string lp(buf);

        ggml_tensor *attn_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn_norm.weight").c_str());
        ggml_tensor *q_w    = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.q.weight").c_str());
        ggml_tensor *k_w    = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.k.weight").c_str());
        ggml_tensor *v_w    = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.v.weight").c_str());
        ggml_tensor *o_w    = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.o.weight").c_str());
        ggml_tensor *ffn_w  = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn_norm.weight").c_str());
        ggml_tensor *gate_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.gate.weight").c_str());
        ggml_tensor *up_w   = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.up.weight").c_str());
        ggml_tensor *down_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.down.weight").c_str());

        if (!attn_w || !q_w || !k_w || !v_w || !o_w || !ffn_w || !gate_w || !up_w || !down_w) {
            fprintf(stderr, "smoldocling: missing weights for layer %d\n", il);
            ggml_free(g);
            return false;
        }

        ggml_tensor *residual = x;
        x = rmsnorm(x, attn_w);

        // QKV projections
        ggml_tensor *Q = ggml_mul_mat(g, q_w, x);  // [n_heads*hd, T]
        ggml_tensor *K = ggml_mul_mat(g, k_w, x);  // [kv_dim, T]
        ggml_tensor *V = ggml_mul_mat(g, v_w, x);  // [kv_dim, T]

        // Reshape for attention: head_dim × n_heads × T
        Q = ggml_reshape_3d(g, Q, hd, n_heads, T);
        K = ggml_reshape_3d(g, K, hd, n_kv,    T);
        V = ggml_reshape_3d(g, V, hd, n_kv,    T);

        // RoPE (NEOX = GPT-NeoX split-half, matches SmolLM2)
        Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0,
                          ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Materialize K/V before writing to cache
        K = ggml_cont(g, K);
        V = ggml_cont(g, V);

        // Write [kv_dim, T] into kvc_k/kvc_v at layer il, position n_past
        {
            ggml_tensor *K_flat = ggml_reshape_2d(g, K, kv_dim, T);
            ggml_tensor *V_flat = ggml_reshape_2d(g, V, kv_dim, T);

            ggml_tensor *k_wr = ggml_view_2d(g, ctx->kvc_k, kv_dim, T,
                ctx->kvc_k->nb[1],
                (size_t)il * ctx->kvc_k->nb[2] + (size_t)n_past * ctx->kvc_k->nb[1]);
            ggml_tensor *v_wr = ggml_view_2d(g, ctx->kvc_v, kv_dim, T,
                ctx->kvc_v->nb[1],
                (size_t)il * ctx->kvc_v->nb[2] + (size_t)n_past * ctx->kvc_v->nb[1]);

            ggml_build_forward_expand(gf, ggml_cpy(g, K_flat, k_wr));
            ggml_build_forward_expand(gf, ggml_cpy(g, V_flat, v_wr));
        }

        // Read full [kv_dim, kv_total] from cache for this layer
        ggml_tensor *K_cache = ggml_reshape_3d(g,
            ggml_view_2d(g, ctx->kvc_k, kv_dim, kv_total,
                ctx->kvc_k->nb[1], (size_t)il * ctx->kvc_k->nb[2]),
            hd, n_kv, kv_total);
        ggml_tensor *V_cache = ggml_reshape_3d(g,
            ggml_view_2d(g, ctx->kvc_v, kv_dim, kv_total,
                ctx->kvc_v->nb[1], (size_t)il * ctx->kvc_v->nb[2]),
            hd, n_kv, kv_total);

        // Permute for flash_attn_ext:
        // Q: [hd, n_heads, T] → [hd, T, n_heads]
        // K/V: [hd, n_kv, kv_total] → [hd, kv_total, n_kv]
        // flash_attn_ext handles GQA natively (n_heads != n_kv)
        Q       = ggml_cont(g, ggml_permute(g, Q,       0, 2, 1, 3));
        K_cache =            ggml_permute(g, K_cache, 0, 2, 1, 3);
        V_cache =            ggml_permute(g, V_cache, 0, 2, 1, 3);

        ggml_tensor *attn = ggml_flash_attn_ext(g, Q, K_cache, V_cache,
                                                 causal_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        // Output: [hd, n_heads, T] → [D, T]
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection + residual
        x = ggml_add(g, residual, ggml_mul_mat(g, o_w, attn));

        // FFN: RMSNorm → SwiGLU → residual
        residual = x;
        x = rmsnorm(x, ffn_w);
        ggml_tensor *gate = ggml_silu(g, ggml_mul_mat(g, gate_w, x));
        ggml_tensor *up   = ggml_mul_mat(g, up_w, x);
        x = ggml_add(g, residual, ggml_mul_mat(g, down_w, ggml_mul(g, gate, up)));
    }

    // Final RMSNorm
    ggml_tensor *norm_w = core_gguf::try_get(ctx->wl.tensors, "llm.norm.weight");
    if (norm_w) x = rmsnorm(x, norm_w);

    ggml_set_name(x, "llm_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // Allocate and compute
    ggml_backend_sched_reset(ctx->llm_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->llm_sched, gf)) {
        fprintf(stderr, "smoldocling: LLM graph alloc failed\n");
        ggml_free(g);
        return false;
    }

    // Upload inputs
    ggml_tensor *emb_t = ggml_graph_get_tensor(gf, "llm_embeds");
    ggml_backend_tensor_set(emb_t, embeds, 0, (size_t)T * D * sizeof(float));

    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = n_past + t;
    ggml_tensor *pos_t = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, T * sizeof(int32_t));

    // Causal mask: mask[k,q]=0 if k<=n_past+q else -inf; shape [kv_total, T]
    std::vector<ggml_fp16_t> mask_data((size_t)kv_total * T);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < kv_total; k++)
            mask_data[(size_t)q * kv_total + k] =
                ggml_fp32_to_fp16(k <= n_past + q ? 0.0f : -INFINITY);
    ggml_tensor *mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_t, mask_data.data(), 0,
                            mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->llm_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "smoldocling: LLM graph compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read last token's hidden state
    ggml_tensor *out_t = ggml_graph_get_tensor(gf, "llm_output");
    ggml_backend_tensor_get(out_t, hidden_out,
                            (size_t)(T - 1) * D * sizeof(float),
                            (size_t)D * sizeof(float));

    ggml_free(g);
    return true;
}

// ── LLM Decode Step (SmolLM2-135M, single token with KV cache) ───────

// skip_logits=true skips the expensive lm_head matmul (49280×576).
// Use during prefill for all tokens except the last.
static void sd_llm_decode_step(smoldocling_context * ctx,
                                const float * token_embed, int n_past,
                                float * logits, bool skip_logits = false) {
    int D = ctx->llm_dim;          // 576
    int n_heads = ctx->llm_heads;  // 9
    int n_kv = ctx->llm_kv_heads;  // 3
    int d_head = ctx->head_dim;    // 64
    int kv_repeat = n_heads / n_kv; // 3
    float eps = ctx->rms_eps;

    std::vector<float> x(D);
    memcpy(x.data(), token_embed, D * sizeof(float));

    int max_seq = ctx->kv_allocated;

    for (int li = 0; li < ctx->llm_layers; li++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "llm.layers.%d", li);
        std::string lp(buf);

        // RMSNorm (attention) — uses core_cpu for SIMD-benefitting downstream
        std::vector<float> normed(D);
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D,
                              ctx->get(lp + ".attn_norm.weight"), eps);

        // GQA: Q [n_heads * d_head], K [n_kv * d_head], V [n_kv * d_head]
        int q_dim = n_heads * d_head;
        int kv_dim = n_kv * d_head;
        std::vector<float> Q(q_dim), K_new(kv_dim), V_new(kv_dim);
        sd_linear(normed.data(), 1, D, q_dim,
                  ctx->get(lp + ".attn.q.weight"), nullptr, Q.data());
        sd_linear(normed.data(), 1, D, kv_dim,
                  ctx->get(lp + ".attn.k.weight"), nullptr, K_new.data());
        sd_linear(normed.data(), 1, D, kv_dim,
                  ctx->get(lp + ".attn.v.weight"), nullptr, V_new.data());

        // RoPE (neghalf) — uses precomputed frequency table (no powf per element)
        ctx->rope_freq.apply(Q.data(), n_heads, n_past,
                             core_vlm::RoPEStyle::NEGHALF);
        ctx->rope_freq.apply(K_new.data(), n_kv, n_past,
                             core_vlm::RoPEStyle::NEGHALF);

        // GQA attention with KV cache
        std::vector<float> attn_out(q_dim, 0.0f);
        core_vlm::gqa_attn_step(Q.data(), K_new.data(), V_new.data(),
                                ctx->kv_cache.data(),
                                n_heads, n_kv, d_head,
                                max_seq, n_past,
                                li, ctx->llm_layers,
                                attn_out.data());

        // Output projection
        std::vector<float> proj(D);
        sd_linear(attn_out.data(), 1, q_dim, D,
                  ctx->get(lp + ".attn.o.weight"), nullptr, proj.data());

        // Residual (no multiplier for SmolLM2)
        for (int d = 0; d < D; d++) x[d] += proj[d];

        // RMSNorm (FFN)
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D,
                              ctx->get(lp + ".ffn_norm.weight"), eps);

        // SwiGLU FFN
        int ffn = ctx->llm_ffn_dim;
        std::vector<float> down(D);
        core_vlm::swiglu_ffn(normed.data(), down.data(), D, ffn,
                             ctx->get(lp + ".ffn.gate.weight"),
                             ctx->get(lp + ".ffn.up.weight"),
                             ctx->get(lp + ".ffn.down.weight"));

        // Residual
        for (int d = 0; d < D; d++) x[d] += down[d];

        // DequantCache: weights stay cached across layers and calls
    }

    // Final RMSNorm
    {
        std::vector<float> tmp(D);
        core_cpu::rmsnorm_cpu(x.data(), tmp.data(), D,
                              ctx->get("llm.norm.weight"), eps);
        memcpy(x.data(), tmp.data(), D * sizeof(float));
    }

    // LM head (separate, NOT tied) — skip during prefill for speed
    // Uses SIMD-accelerated linear_cpu for the (49280 × 576) matmul
    if (!skip_logits) {
        const float * lm_w = ctx->get("llm.lm_head.weight");
        if (lm_w)
            core_cpu::linear_cpu(x.data(), logits, D, ctx->vocab_size, lm_w, nullptr);
    }
}

// ── Main recognize (from raw pixels) ──────────────────────────────────

const char * smoldocling_recognize_raw(smoldocling_context * ctx,
                                        const uint8_t * pixels,
                                        int width, int height, int channels,
                                        int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    int img_size = ctx->vis_image_size;  // 512
    int ps = ctx->vis_patch_size;
    int n_patches_side = img_size / ps;  // 32
    int T_vis = n_patches_side * n_patches_side;  // 1024
    int D = ctx->llm_dim;

    // Preprocess: resize to img_size x img_size, normalize to [-1, 1]
    // Formula: pixel / 127.5 - 1.0
    std::vector<float> image(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float sy = (y + 0.5f) * height / img_size - 0.5f;
                float sx = (x + 0.5f) * width / img_size - 0.5f;
                int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                int src_c = c;
                if (channels == 1) src_c = 0;  // grayscale
                int src_idx = channels >= 3 ? (iy * width + ix) * channels + src_c
                                            : iy * width + ix;
                image[c * img_size * img_size + y * img_size + x] =
                    pixels[src_idx] / 127.5f - 1.0f;
            }

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Vision encoder
    fprintf(stderr, "smoldocling: running vision encoder...\n");
    auto t_vis = std::chrono::steady_clock::now();
    std::vector<float> vis_features(T_vis * ctx->vis_dim);
    int n_vis_tokens = 0;
    sd_vision_forward(ctx, image.data(), img_size, img_size,
                      vis_features.data(), &n_vis_tokens);
    fprintf(stderr, "smoldocling: vision done, %d tokens, first4=[%.4f,%.4f,%.4f,%.4f]\n",
            n_vis_tokens, vis_features[0], vis_features[1], vis_features[2], vis_features[3]);
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_vis).count();
        fprintf(stderr, "[smoldocling-bench] vision_encoder: %lldms\n", (long long)ms);
    }

    // Connector: pixel shuffle + projection
    auto t_connector = std::chrono::steady_clock::now();
    int n_connector_tokens = 0;
    std::vector<float> connector_out(n_vis_tokens * D);  // will be <= n_vis_tokens
    sd_connector(ctx, vis_features.data(), n_vis_tokens,
                 connector_out.data(), &n_connector_tokens);
    fprintf(stderr, "smoldocling: connector done, %d tokens\n", n_connector_tokens);
    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_connector).count();
        fprintf(stderr, "[smoldocling-bench] connector: %lldms\n", (long long)ms);
    }

    // Build input token sequence following SmolDocling chat template:
    //   <|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:
    // Special tokens must be inserted by ID, not BPE-encoded.
    // BOS=1 (<|im_start|>), EOS=2 (<|im_end|>), end_of_utterance=49279
    std::vector<int> input_ids;

    // <|im_start|> (BOS, id=1)
    input_ids.push_back(1);

    // "User:" — BPE encode
    auto user_ids = ctx->tokenizer.encode("User:");
    input_ids.insert(input_ids.end(), user_ids.begin(), user_ids.end());

    // Image token placeholders
    for (int i = 0; i < n_connector_tokens; i++)
        input_ids.push_back(ctx->image_token_id);

    // "Convert this page to docling." — BPE encode
    auto prompt_ids = ctx->tokenizer.encode("Convert this page to docling.");
    input_ids.insert(input_ids.end(), prompt_ids.begin(), prompt_ids.end());

    // <end_of_utterance> (id=49279)
    input_ids.push_back(49279);

    // "\nAssistant:" — BPE encode
    auto asst_ids = ctx->tokenizer.encode("\nAssistant:");
    input_ids.insert(input_ids.end(), asst_ids.begin(), asst_ids.end());

    fprintf(stderr, "smoldocling: prompt has %d tokens (%d image + %d text)\n",
            (int)input_ids.size(), n_connector_tokens, (int)input_ids.size() - n_connector_tokens);

    // Get embedding weights — DequantCache keeps the pointer stable across calls
    const float * embed_w = ctx->get("llm.embed.weight");
    const float * lm_head_w = ctx->get("llm.lm_head.weight");

    const int prefill_len = (int)input_ids.size();
    int max_seq = prefill_len + ctx->max_tokens + 4;

    // Build flat prefill embedding matrix [prefill_len × D]
    std::vector<float> prefill_embeds((size_t)prefill_len * D);
    {
        int vis_idx = 0;
        for (int t = 0; t < prefill_len; t++) {
            float *dst = prefill_embeds.data() + (size_t)t * D;
            if (input_ids[t] == ctx->image_token_id && vis_idx < n_connector_tokens) {
                memcpy(dst, connector_out.data() + (size_t)vis_idx * D, D * sizeof(float));
                vis_idx++;
            } else {
                memcpy(dst, embed_w + (size_t)input_ids[t] * D, D * sizeof(float));
            }
        }
    }

    // Try ggml batched prefill path
    bool use_ggml = (ctx->llm_sched != nullptr)
                 && sd_alloc_kv_cache(ctx, max_seq);

    std::vector<float> logits(ctx->vocab_size);
    std::vector<float> hidden(D);
    ctx->n_past = 0;

    fprintf(stderr, "smoldocling: starting prefill of %d tokens (%s)...\n",
            prefill_len, use_ggml ? "ggml batched" : "scalar");

    auto t_prefill = std::chrono::steady_clock::now();

    if (use_ggml) {
        if (!sd_run_llm_body(ctx, prefill_embeds.data(), prefill_len, 0, hidden.data())) {
            fprintf(stderr, "smoldocling: ggml prefill failed, falling back to scalar\n");
            sd_free_kv_cache(ctx);
            use_ggml = false;
        } else {
            ctx->n_past = prefill_len;
            if (lm_head_w)
                core_cpu::linear_cpu(hidden.data(), logits.data(), D, ctx->vocab_size, lm_head_w, nullptr);
        }
    }

    if (!use_ggml) {
        // Scalar fallback: token-by-token with F32 KV cache
        int kv_dim = ctx->llm_kv_heads * ctx->head_dim;
        ctx->kv_cache.assign((size_t)2 * ctx->llm_layers * max_seq * kv_dim, 0.0f);
        ctx->kv_allocated = max_seq;
        ctx->n_past = 0;

        for (int t = 0; t < prefill_len; t++) {
            bool is_last = (t == prefill_len - 1);
            sd_llm_decode_step(ctx, prefill_embeds.data() + (size_t)t * D,
                               ctx->n_past, logits.data(), /*skip_logits=*/!is_last);
            ctx->n_past++;
        }
    }

    if (bench) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_prefill).count();
        fprintf(stderr, "[smoldocling-bench] prefill: %lldms\n", (long long)ms);
    }

    // Greedy decode
    ctx->output_text.clear();
    std::vector<int> output_ids;
    const int eos_id = 2;    // <|im_end|>
    const int eou_id = 49279; // <end_of_utterance>

    auto t_decode_start = std::chrono::steady_clock::now();
    for (int step = 0; step < ctx->max_tokens; step++) {
        auto t_step = std::chrono::steady_clock::now();

        int best_id = 0;
        float best_score = logits[0];
        for (int v = 1; v < ctx->vocab_size; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best_id = v; }

        if (best_id == eos_id || best_id == eou_id) break;
        output_ids.push_back(best_id);

        const float *next_emb = embed_w + (size_t)best_id * D;

        if (use_ggml) {
            if (!sd_run_llm_body(ctx, next_emb, 1, ctx->n_past, hidden.data())) {
                fprintf(stderr, "smoldocling: ggml decode step failed at step %d\n", step);
                break;
            }
            ctx->n_past++;
            if (lm_head_w)
                core_cpu::linear_cpu(hidden.data(), logits.data(), D, ctx->vocab_size, lm_head_w, nullptr);
        } else {
            std::vector<float> next_embed(next_emb, next_emb + D);
            sd_llm_decode_step(ctx, next_embed.data(), ctx->n_past, logits.data());
            ctx->n_past++;
        }

        if (bench) {
            auto step_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_step).count();
            if (step == 0 || step == 1)
                fprintf(stderr, "[smoldocling-bench] decode_step[%d]: %lldms\n", step, (long long)step_ms);
        }
    }
    if (bench) {
        auto t_decode_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[smoldocling-bench] decode (%d steps): %.1f ms\n",
                (int)output_ids.size(),
                std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count());
        fprintf(stderr, "[smoldocling-bench] total: %.1f ms\n",
                std::chrono::duration<double, std::milli>(t_decode_end - t_total).count());
    }

    // Detokenize
    ctx->output_text = ctx->tokenizer.decode(output_ids);

    if (out_len) *out_len = (int)ctx->output_text.size();
    return ctx->output_text.c_str();
}

// ── Recognize from image file ─────────────────────────────────────────

const char * smoldocling_recognize(smoldocling_context * ctx,
                                    const char * image_path, int * out_len) {
    if (!ctx || !image_path) return nullptr;

    int w, h, c;
    unsigned char * img = stbi_load(image_path, &w, &h, &c, 3);
    if (!img) {
        fprintf(stderr, "smoldocling: failed to load image: %s\n", image_path);
        return nullptr;
    }

    const char * result = smoldocling_recognize_raw(ctx, img, w, h, 3, out_len);
    stbi_image_free(img);
    return result;
}

// ── Debug: vision encoder only ───────────────────────────────────────

static void sd_preprocess_image(const uint8_t * pixels, int width, int height, int channels,
                                 int img_size, std::vector<float> & image) {
    image.resize(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float sy = (y + 0.5f) * height / img_size - 0.5f;
                float sx = (x + 0.5f) * width / img_size - 0.5f;
                int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                int src_c = (channels >= 3) ? c : 0;
                int src_idx = (channels >= 3) ? (iy * width + ix) * channels + src_c
                                              : iy * width + ix;
                image[c * img_size * img_size + y * img_size + x] =
                    pixels[src_idx] / 127.5f - 1.0f;
            }
}

float * smoldocling_debug_vision(smoldocling_context * ctx,
                                  const uint8_t * pixels,
                                  int w, int h, int ch,
                                  int * out_n_tokens, int * out_dim) {
    if (!ctx || !pixels) return nullptr;
    int img_size = ctx->vis_image_size;
    std::vector<float> image;
    sd_preprocess_image(pixels, w, h, ch, img_size, image);

    int T = (img_size / ctx->vis_patch_size) * (img_size / ctx->vis_patch_size);
    int D = ctx->vis_dim;
    float * output = (float *)malloc(T * D * sizeof(float));
    int n_tokens = 0;
    sd_vision_forward(ctx, image.data(), img_size, img_size, output, &n_tokens);
    if (out_n_tokens) *out_n_tokens = n_tokens;
    if (out_dim) *out_dim = D;
    return output;
}

float * smoldocling_debug_connector(smoldocling_context * ctx,
                                     const uint8_t * pixels,
                                     int w, int h, int ch,
                                     int * out_n_tokens, int * out_dim) {
    if (!ctx || !pixels) return nullptr;
    int img_size = ctx->vis_image_size;
    std::vector<float> image;
    sd_preprocess_image(pixels, w, h, ch, img_size, image);

    int T_vis = (img_size / ctx->vis_patch_size) * (img_size / ctx->vis_patch_size);
    int D = ctx->vis_dim;
    std::vector<float> vis_features(T_vis * D);
    int n_vis = 0;
    sd_vision_forward(ctx, image.data(), img_size, img_size, vis_features.data(), &n_vis);

    int n_conn = 0;
    int llm_dim = ctx->llm_dim;
    float * output = (float *)malloc(n_vis * llm_dim * sizeof(float));
    sd_connector(ctx, vis_features.data(), n_vis, output, &n_conn);
    if (out_n_tokens) *out_n_tokens = n_conn;
    if (out_dim) *out_dim = llm_dim;
    return output;
}
