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
// Residency (G1/F4): the SigLIP vision graphs run on the GPU backend when one
// is available (vis.* weights GPU-resident via core_gguf::load_weights_split);
// the connector, LLM decode and LM head stay CPU-resident.
// SMOLDOCLING_FORCE_CPU=1 restores the historical all-CPU engine.

#include "smoldocling_ocr.h"
#include "core/bpe.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/vlm_attention.h"
#include "ggml-cpu.h"
#include "core/env_gate.h"

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
    // A weight resident on CUDA/Vulkan/SYCL/HIP has a DEVICE pointer in t->data:
    // it must NOT be returned/dereferenced on the host. Keep the zero-copy fast
    // path only for host-visible buffers (CPU / Metal unified memory); otherwise
    // copy through the backend buffer via ggml_backend_tensor_get.
    const bool host = !t->buffer || ggml_backend_buffer_is_host(t->buffer);
    if (t->type == GGML_TYPE_F32 && host) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    std::vector<uint8_t> raw;
    const void * src_bytes;
    if (t->buffer) {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        src_bytes = raw.data();
    } else {
        src_bytes = t->data;
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf.data(), src_bytes, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)src_bytes;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float)
            traits->to_float(src_bytes, buf.data(), n);
        else
            memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

// sd_linear: delegate to SIMD-accelerated core_cpu::linear_cpu
static void sd_linear(const float * input, int n, int id, int od, const float * weight, const float * bias,
                      float * output) {
    for (int i = 0; i < n; i++) core_cpu::linear_cpu(input + i * id, output + i * od, id, od, weight, bias);
}

// ── GPT-2 byte-level BPE tables ──────────────────────────────────────
// GPT-2 byte-level BPE byte<->unicode mapping is shared in core/bpe.h
// (core_bpe::bytes_to_unicode for encode, core_bpe::unicode_to_bytes for
// decode) — this engine used to carry private copies (sd_byte_encoder etc.).

// ── BPE Tokenizer ─────────────────────────────────────────────────────

struct sd_tokenizer {
    std::vector<std::string> vocab;                   // id -> token string
    std::unordered_map<std::string, int> token_to_id; // token string -> id
    std::unordered_map<std::string, int> merge_rank;  // "left right" -> rank
    int eos_id = 0;
    int bos_id = 1;
    int pad_id = 2;

    bool load(gguf_context * meta) {
        // Try both key conventions
        vocab = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        if (vocab.empty()) vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        if (vocab.empty()) {
            fprintf(stderr, "smoldocling: no tokenizer.tokens in GGUF\n");
            return false;
        }
        for (int i = 0; i < (int)vocab.size(); i++) token_to_id[vocab[i]] = i;

        auto merge_strs = core_gguf::kv_str_array(meta, "tokenizer.merges");
        if (merge_strs.empty()) merge_strs = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merge_strs.size(); i++) {
            merge_rank[merge_strs[i]] = i;
        }

        return true;
    }

    // GPT-2 byte-level BPE encode
    std::vector<int> encode(const std::string & text) const {
        if (text.empty()) return {};

        // Step 1: convert raw bytes to GPT-2 unicode string
        std::string unicode = core_bpe::bytes_to_unicode(text.data(), text.size());

        // Step 2: split into per-byte unicode symbols
        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < unicode.size()) {
            unsigned char c = (unsigned char)unicode[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;
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
            if (id <= 2) continue;     // BOS=1, EOS=2, endoftext=0
            if (id == 49279) continue; // <end_of_utterance>
            const std::string & piece = vocab[id];
            // Added tokens are literal (not GPT-2 byte encoded)
            if (id <= 16 || id >= 49152) {
                result += piece;
            } else {
                // Base BPE vocab: reverse GPT-2 byte mapping
                result += core_bpe::unicode_to_bytes(piece);
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
    core_cpu::DequantCache dcache; // caches dequantized weights (replaces wbufs)

    // ggml backends — split residency (G1/F4): `backend` is always the CPU
    // backend and owns the LLM weights, KV cache and decode graphs (the 135M
    // per-token decode is CPU-shaped). `gpu_backend` is non-null only when a
    // GPU device is available and not disabled; it owns the vis.* weights and
    // runs the SigLIP vision graphs (compute-bound, GPU-shaped).
    ggml_backend_t backend = nullptr;
    ggml_backend_t gpu_backend = nullptr;

    // LLM decoder: reusable scheduler + pre-allocated metadata buffer
    ggml_backend_sched_t llm_sched = nullptr;
    std::vector<uint8_t> llm_compute_meta;

    // F16 KV cache on the backend (re-allocated per image call)
    ggml_context * kvc_ctx = nullptr;
    ggml_tensor * kvc_k = nullptr; // [kv_dim, max_seq, n_layers] F16
    ggml_tensor * kvc_v = nullptr;
    ggml_backend_buffer_t kvc_buf = nullptr;
    int kvc_max_seq = 0;

    // RoPE frequency table (precomputed once at init)
    core_vlm::RoPEFreqTable rope_freq;

    // Scalar fallback KV cache (used when ggml graph path is unavailable)
    std::vector<float> kv_cache;
    int kv_allocated = 0;
    int n_past = 0;

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
    ctx->max_tokens = 1024;
    ctx->kv_allocated = 0;
    ctx->n_past = 0;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "smoldocling: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    // Vision hparams
    ctx->vis_dim = core_gguf::kv_u32(meta, "smoldocling.vision.hidden_size", 768);
    ctx->vis_heads = core_gguf::kv_u32(meta, "smoldocling.vision.num_heads", 12);
    ctx->vis_layers = core_gguf::kv_u32(meta, "smoldocling.vision.num_layers", 12);
    ctx->vis_patch_size = core_gguf::kv_u32(meta, "smoldocling.vision.patch_size", 16);
    ctx->vis_image_size = core_gguf::kv_u32(meta, "smoldocling.vision.image_size", 512);
    ctx->vis_intermediate = core_gguf::kv_u32(meta, "smoldocling.vision.intermediate_size", 3072);

    // Connector
    ctx->connector_scale = core_gguf::kv_u32(meta, "smoldocling.connector.scale_factor", 4);

    // LLM hparams
    ctx->llm_dim = core_gguf::kv_u32(meta, "smoldocling.hidden_size", 576);
    ctx->llm_heads = core_gguf::kv_u32(meta, "smoldocling.num_attention_heads", 9);
    ctx->llm_kv_heads = core_gguf::kv_u32(meta, "smoldocling.num_key_value_heads", 3);
    ctx->llm_layers = core_gguf::kv_u32(meta, "smoldocling.num_hidden_layers", 30);
    ctx->llm_ffn_dim = core_gguf::kv_u32(meta, "smoldocling.intermediate_size", 1536);
    ctx->head_dim = core_gguf::kv_u32(meta, "smoldocling.head_dim", 64);
    ctx->vocab_size = core_gguf::kv_u32(meta, "smoldocling.vocab_size", 49280);
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
        delete ctx;
        return nullptr;
    }

    core_gguf::free_metadata(meta);

    // Load weights — split residency (G1/F4). SMOLDOCLING_FORCE_CPU=1 (value-
    // parsed: =0 is off) restores the historical all-CPU engine; `--gpu-backend
    // cpu` reaches the same state through the pref helper's CPU short-circuit.
    ctx->backend = ggml_backend_cpu_init();
    const char * fc = std::getenv("SMOLDOCLING_FORCE_CPU");
    const bool force_cpu = fc && atoi(fc) != 0;
    if (!force_cpu) {
        ggml_backend_t gpu = crispasr_init_gpu_backend();
        if (gpu && !ggml_backend_is_cpu(gpu)) {
            ctx->gpu_backend = gpu;
        } else if (gpu) {
            ggml_backend_free(gpu); // --gpu-backend cpu / no GPU device: all-CPU
        }
    } else {
        fprintf(stderr, "smoldocling: SMOLDOCLING_FORCE_CPU=1 — CPU backend\n");
    }

    bool loaded;
    if (ctx->gpu_backend) {
        auto is_vis = [](const char * name, void *) { return strncmp(name, "vis.", 4) == 0; };
        loaded = core_gguf::load_weights_split(model_path, ctx->gpu_backend, ctx->backend, is_vis, nullptr,
                                               "smoldocling", ctx->wl);
    } else {
        loaded = core_gguf::load_weights(model_path, ctx->backend, "smoldocling", ctx->wl);
    }
    if (!loaded) {
        fprintf(stderr, "smoldocling: failed to load weights\n");
        if (ctx->gpu_backend) ggml_backend_free(ctx->gpu_backend);
        ctx->gpu_backend = nullptr;
        ggml_backend_free(ctx->backend);
        ctx->backend = nullptr;
        delete ctx;
        return nullptr;
    }

    int n_patches = (ctx->vis_image_size / ctx->vis_patch_size);
    n_patches *= n_patches;
    int S = ctx->connector_scale;
    int connector_out = n_patches / (S * S);

    fprintf(stderr,
            "smoldocling: vis=%dL x %dd (%d patches -> %d after shuffle), "
            "llm=%dL x %dd (heads=%d/%d, ffn=%d), vocab=%d, %d tensors\n",
            ctx->vis_layers, ctx->vis_dim, n_patches, connector_out, ctx->llm_layers, ctx->llm_dim, ctx->llm_heads,
            ctx->llm_kv_heads, ctx->llm_ffn_dim, ctx->vocab_size, (int)ctx->wl.tensors.size());

    ctx->bench = core_env::on("CRISPEMBED_SMOLDOCLING_BENCH");

    // LLM scheduler: reuse the same CPU backend (weights already in CPU memory)
    {
        size_t meta_sz = ggml_tensor_overhead() * kSdLlmGraphCap + ggml_graph_overhead_custom(kSdLlmGraphCap, false);
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
        if (ctx->gpu_backend) ggml_backend_free(ctx->gpu_backend);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

void smoldocling_set_max_tokens(smoldocling_context * ctx, int max_tokens) {
    if (ctx && max_tokens > 0) ctx->max_tokens = max_tokens;
}

// ── SigLIP Vision Encoder (ggml graph — BLAS accelerated) ────────────

// Run SigLIP ViT via ggml graph. Builds a compute graph with all
// 12 layers, then runs it in one shot. Much faster than CPU-scalar
// for T=1024 tokens (uses BLAS for matmuls, flash_attn_ext for attention).
//
// Input: [3, img_h, img_w] float, normalized to [-1, 1]
// Output: [n_patches, vis_dim]
static void sd_vision_forward(smoldocling_context * ctx, const float * image, int img_h, int img_w, float * output,
                              int * out_tokens) {
    int ps = ctx->vis_patch_size;
    int ph = img_h / ps, pw = img_w / ps;
    int T = ph * pw;         // 1024
    int D = ctx->vis_dim;    // 768
    int nh = ctx->vis_heads; // 12
    int hd = D / nh;         // 64
    float eps = 1e-6f;

    // ── Patch embedding: im2col + ggml matmul ──
    // Gated: CRISPEMBED_SMOLDOCLING_SCALAR_PATCH=1 for scalar fallback
    auto * pe_t = core_gguf::try_get(ctx->wl.tensors, "vis.patch_embed.weight");
    auto * pb_t = core_gguf::try_get(ctx->wl.tensors, "vis.patch_embed.bias");
    int patch_dim = 3 * ps * ps; // 588

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
                        im2col[t * patch_dim + c * ps * ps + ky * ps + kx] = image[c * img_h * img_w + iy * img_w + ix];
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
                for (int k = 0; k < patch_dim; k++) s += im2col[t * patch_dim + k] * pe_w[d * patch_dim + k];
                patch_embed[t * D + d] = s;
            }
    } else {
        // ggml graph: matmul weight × im2col → [D, T]
        size_t buf_sz = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
        ggml_init_params eip{ buf_sz, nullptr, true };
        ggml_context * eg = ggml_init(eip);
        ggml_tensor * w = ggml_reshape_2d(eg, pe_t, patch_dim, D);
        ggml_tensor * inp = ggml_new_tensor_2d(eg, GGML_TYPE_F32, patch_dim, T);
        ggml_set_name(inp, "im2col");
        ggml_set_input(inp);
        ggml_tensor * out = ggml_mul_mat(eg, w, inp);
        if (pb_t) {
            ggml_tensor * b = pb_t;
            if (b->type != GGML_TYPE_F32) b = ggml_cast(eg, b, GGML_TYPE_F32);
            out = ggml_add(eg, out, b);
        }
        ggml_set_name(out, "pe_out");
        ggml_set_output(out);
        ggml_cgraph * egf = ggml_new_graph(eg);
        ggml_build_forward_expand(egf, out);
        // Vision graphs follow the vis.* weight residency: GPU-first with the
        // CPU backend as per-op fallback when split residency is active.
        ggml_backend_t vis_be[2] = { ctx->gpu_backend ? ctx->gpu_backend : ctx->backend, ctx->backend };
        const int n_vis_be = ctx->gpu_backend ? 2 : 1;
        ggml_backend_sched_t pe_sched = ggml_backend_sched_new(vis_be, nullptr, n_vis_be, 16, false, false);
        ggml_backend_sched_reset(pe_sched);
        if (!ggml_backend_sched_alloc_graph(pe_sched, egf)) {
            fprintf(stderr, "smoldocling: patch_embed graph alloc failed\n");
            ggml_backend_sched_free(pe_sched);
            ggml_free(eg);
            return;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(egf, "im2col"), im2col.data(), 0, T * patch_dim * sizeof(float));
        ggml_backend_sched_graph_compute(pe_sched, egf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(egf, "pe_out"), patch_embed.data(), 0, T * D * sizeof(float));
        ggml_backend_sched_free(pe_sched);
        ggml_free(eg);
    }

    // Add position embedding
    auto * pos_t = core_gguf::try_get(ctx->wl.tensors, "vis.pos_embed.weight");
    if (pos_t) {
        std::vector<float> pos_buf;
        const float * pos_w = sd_to_f32(pos_t, pos_buf);
        for (int i = 0; i < T * D; i++) patch_embed[i] += pos_w[i];
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

    // Use backend scheduler with model weights buffer. With split residency
    // the vis.* weights live on the GPU backend, so the transformer's matmuls
    // and attention run there; the CPU backend covers any op the GPU backend
    // does not support.
    ggml_backend_t vis_be[2] = { ctx->gpu_backend ? ctx->gpu_backend : ctx->backend, ctx->backend };
    const int n_vis_be = ctx->gpu_backend ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(vis_be, nullptr, n_vis_be, max_nodes, false, false);
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
static void sd_pixel_shuffle(const float * input, int H, int W, int D, int S, float * output) {
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

static void sd_connector(smoldocling_context * ctx, const float * vis_features, int n_tokens, float * output,
                         int * out_n) {
    int D = ctx->vis_dim;                // 768
    int S = ctx->connector_scale;        // 4
    int H = (int)sqrtf((float)n_tokens); // 32
    int W = H;

    int Ho = H / S, Wo = W / S; // 8, 8
    int Do = D * S * S;         // 12288
    int n_out = Ho * Wo;        // 64

    // Pixel shuffle
    std::vector<float> shuffled(n_out * Do);
    sd_pixel_shuffle(vis_features, H, W, D, S, shuffled.data());

    // Linear projection: [12288] -> [576], no bias
    int llm_dim = ctx->llm_dim;
    sd_linear(shuffled.data(), n_out, Do, llm_dim, ctx->get("connector.proj.weight"), nullptr, output);

    *out_n = n_out;
}

// ── F16 KV cache management ───────────────────────────────────────────

static void sd_free_kv_cache(smoldocling_context * ctx) {
    if (ctx->kvc_buf) {
        ggml_backend_buffer_free(ctx->kvc_buf);
        ctx->kvc_buf = nullptr;
    }
    if (ctx->kvc_ctx) {
        ggml_free(ctx->kvc_ctx);
        ctx->kvc_ctx = nullptr;
    }
    ctx->kvc_k = nullptr;
    ctx->kvc_v = nullptr;
    ctx->kvc_max_seq = 0;
}

static bool sd_alloc_kv_cache(smoldocling_context * ctx, int max_seq) {
    sd_free_kv_cache(ctx);

    const int n_layers = ctx->llm_layers;
    const int kv_dim = ctx->llm_kv_heads * ctx->head_dim;

    ggml_init_params ip{ 2 * ggml_tensor_overhead() + 256, nullptr, true };
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
    fprintf(stderr, "  smoldocling KV cache: %d layers, max_seq=%d, %.1f MB\n", n_layers, max_seq,
            (float)bytes / (1024.0f * 1024.0f));
    return true;
}

// ── LLM body via ggml graph (batched — handles prefill T>1 and decode T=1) ─

// embeds[T*D]: token embeddings (F32). n_past: tokens already in KV cache.
// Writes K/V into kvc_k/kvc_v at positions [n_past..n_past+T-1].
// Reads hidden state of last token into hidden_out[D].
// Returns false on allocation or compute failure.
static bool sd_run_llm_body(smoldocling_context * ctx, const float * embeds, int T, int n_past, float * hidden_out) {
    const int D = ctx->llm_dim;           // 576
    const int n_heads = ctx->llm_heads;   // 9
    const int n_kv = ctx->llm_kv_heads;   // 3
    const int hd = ctx->head_dim;         // 64
    const int n_layers = ctx->llm_layers; // 30
    const int kv_dim = n_kv * hd;         // 192
    const float eps = ctx->rms_eps;
    const float scale = 1.0f / sqrtf((float)hd);
    const int kv_total = n_past + T;

    ggml_init_params ip{ ctx->llm_compute_meta.size(), ctx->llm_compute_meta.data(), true };
    ggml_context * g = ggml_init(ip);
    if (!g) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(g, kSdLlmGraphCap, false);

    // Inputs
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "llm_embeds");
    ggml_set_input(x);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Causal mask [kv_total, T] F16: mask[k,q]=0 if k<=n_past+q else -inf
    ggml_tensor * causal_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, kv_total, T);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) -> ggml_tensor * {
        if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
        return ggml_mul(g, ggml_rms_norm(g, t, eps), w);
    };

    for (int il = 0; il < n_layers; il++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "llm.layers.%d", il);
        std::string lp(buf);

        ggml_tensor * attn_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn_norm.weight").c_str());
        ggml_tensor * q_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.q.weight").c_str());
        ggml_tensor * k_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.k.weight").c_str());
        ggml_tensor * v_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.v.weight").c_str());
        ggml_tensor * o_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".attn.o.weight").c_str());
        ggml_tensor * ffn_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn_norm.weight").c_str());
        ggml_tensor * gate_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.gate.weight").c_str());
        ggml_tensor * up_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.up.weight").c_str());
        ggml_tensor * down_w = core_gguf::try_get(ctx->wl.tensors, (lp + ".ffn.down.weight").c_str());

        if (!attn_w || !q_w || !k_w || !v_w || !o_w || !ffn_w || !gate_w || !up_w || !down_w) {
            fprintf(stderr, "smoldocling: missing weights for layer %d\n", il);
            ggml_free(g);
            return false;
        }

        ggml_tensor * residual = x;
        x = rmsnorm(x, attn_w);

        // QKV projections
        ggml_tensor * Q = ggml_mul_mat(g, q_w, x); // [n_heads*hd, T]
        ggml_tensor * K = ggml_mul_mat(g, k_w, x); // [kv_dim, T]
        ggml_tensor * V = ggml_mul_mat(g, v_w, x); // [kv_dim, T]

        // Reshape for attention: head_dim × n_heads × T
        Q = ggml_reshape_3d(g, Q, hd, n_heads, T);
        K = ggml_reshape_3d(g, K, hd, n_kv, T);
        V = ggml_reshape_3d(g, V, hd, n_kv, T);

        // RoPE (NEOX = GPT-NeoX split-half, matches SmolLM2)
        Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, ctx->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        // Materialize K/V before writing to cache
        K = ggml_cont(g, K);
        V = ggml_cont(g, V);

        // Write [kv_dim, T] into kvc_k/kvc_v at layer il, position n_past
        {
            ggml_tensor * K_flat = ggml_reshape_2d(g, K, kv_dim, T);
            ggml_tensor * V_flat = ggml_reshape_2d(g, V, kv_dim, T);

            ggml_tensor * k_wr = ggml_view_2d(g, ctx->kvc_k, kv_dim, T, ctx->kvc_k->nb[1],
                                              (size_t)il * ctx->kvc_k->nb[2] + (size_t)n_past * ctx->kvc_k->nb[1]);
            ggml_tensor * v_wr = ggml_view_2d(g, ctx->kvc_v, kv_dim, T, ctx->kvc_v->nb[1],
                                              (size_t)il * ctx->kvc_v->nb[2] + (size_t)n_past * ctx->kvc_v->nb[1]);

            ggml_build_forward_expand(gf, ggml_cpy(g, K_flat, k_wr));
            ggml_build_forward_expand(gf, ggml_cpy(g, V_flat, v_wr));
        }

        // Read full [kv_dim, kv_total] from cache for this layer
        ggml_tensor * K_cache = ggml_reshape_3d(
            g, ggml_view_2d(g, ctx->kvc_k, kv_dim, kv_total, ctx->kvc_k->nb[1], (size_t)il * ctx->kvc_k->nb[2]), hd,
            n_kv, kv_total);
        ggml_tensor * V_cache = ggml_reshape_3d(
            g, ggml_view_2d(g, ctx->kvc_v, kv_dim, kv_total, ctx->kvc_v->nb[1], (size_t)il * ctx->kvc_v->nb[2]), hd,
            n_kv, kv_total);

        // Permute for flash_attn_ext:
        // Q: [hd, n_heads, T] → [hd, T, n_heads]
        // K/V: [hd, n_kv, kv_total] → [hd, kv_total, n_kv]
        // flash_attn_ext handles GQA natively (n_heads != n_kv)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
        K_cache = ggml_permute(g, K_cache, 0, 2, 1, 3);
        V_cache = ggml_permute(g, V_cache, 0, 2, 1, 3);

        ggml_tensor * attn = ggml_flash_attn_ext(g, Q, K_cache, V_cache, causal_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        // Output: [hd, n_heads, T] → [D, T]
        attn = ggml_reshape_2d(g, attn, D, T);

        // Output projection + residual
        x = ggml_add(g, residual, ggml_mul_mat(g, o_w, attn));

        // FFN: RMSNorm → SwiGLU → residual
        residual = x;
        x = rmsnorm(x, ffn_w);
        ggml_tensor * gate = ggml_silu(g, ggml_mul_mat(g, gate_w, x));
        ggml_tensor * up = ggml_mul_mat(g, up_w, x);
        x = ggml_add(g, residual, ggml_mul_mat(g, down_w, ggml_mul(g, gate, up)));
    }

    // Final RMSNorm
    ggml_tensor * norm_w = core_gguf::try_get(ctx->wl.tensors, "llm.norm.weight");
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
    ggml_tensor * emb_t = ggml_graph_get_tensor(gf, "llm_embeds");
    ggml_backend_tensor_set(emb_t, embeds, 0, (size_t)T * D * sizeof(float));

    std::vector<int32_t> pos_data(T);
    for (int t = 0; t < T; t++) pos_data[t] = n_past + t;
    ggml_tensor * pos_t = ggml_graph_get_tensor(gf, "pos_ids");
    ggml_backend_tensor_set(pos_t, pos_data.data(), 0, T * sizeof(int32_t));

    // Causal mask: mask[k,q]=0 if k<=n_past+q else -inf; shape [kv_total, T]
    std::vector<ggml_fp16_t> mask_data((size_t)kv_total * T);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < kv_total; k++)
            mask_data[(size_t)q * kv_total + k] = ggml_fp32_to_fp16(k <= n_past + q ? 0.0f : -INFINITY);
    ggml_tensor * mask_t = ggml_graph_get_tensor(gf, "causal_mask");
    ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->llm_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "smoldocling: LLM graph compute failed\n");
        ggml_free(g);
        return false;
    }

    // Read last token's hidden state
    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "llm_output");
    ggml_backend_tensor_get(out_t, hidden_out, (size_t)(T - 1) * D * sizeof(float), (size_t)D * sizeof(float));

    ggml_free(g);
    return true;
}

// ── LLM Decode Step (SmolLM2-135M, single token with KV cache) ───────

// skip_logits=true skips the expensive lm_head matmul (49280×576).
// Use during prefill for all tokens except the last.
static void sd_llm_decode_step(smoldocling_context * ctx, const float * token_embed, int n_past, float * logits,
                               bool skip_logits = false) {
    int D = ctx->llm_dim;           // 576
    int n_heads = ctx->llm_heads;   // 9
    int n_kv = ctx->llm_kv_heads;   // 3
    int d_head = ctx->head_dim;     // 64
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
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D, ctx->get(lp + ".attn_norm.weight"), eps);

        // GQA: Q [n_heads * d_head], K [n_kv * d_head], V [n_kv * d_head]
        int q_dim = n_heads * d_head;
        int kv_dim = n_kv * d_head;
        std::vector<float> Q(q_dim), K_new(kv_dim), V_new(kv_dim);
        sd_linear(normed.data(), 1, D, q_dim, ctx->get(lp + ".attn.q.weight"), nullptr, Q.data());
        sd_linear(normed.data(), 1, D, kv_dim, ctx->get(lp + ".attn.k.weight"), nullptr, K_new.data());
        sd_linear(normed.data(), 1, D, kv_dim, ctx->get(lp + ".attn.v.weight"), nullptr, V_new.data());

        // RoPE (neghalf) — uses precomputed frequency table (no powf per element)
        ctx->rope_freq.apply(Q.data(), n_heads, n_past, core_vlm::RoPEStyle::NEGHALF);
        ctx->rope_freq.apply(K_new.data(), n_kv, n_past, core_vlm::RoPEStyle::NEGHALF);

        // GQA attention with KV cache
        std::vector<float> attn_out(q_dim, 0.0f);
        core_vlm::gqa_attn_step(Q.data(), K_new.data(), V_new.data(), ctx->kv_cache.data(), n_heads, n_kv, d_head,
                                max_seq, n_past, li, ctx->llm_layers, attn_out.data());

        // Output projection
        std::vector<float> proj(D);
        sd_linear(attn_out.data(), 1, q_dim, D, ctx->get(lp + ".attn.o.weight"), nullptr, proj.data());

        // Residual (no multiplier for SmolLM2)
        for (int d = 0; d < D; d++) x[d] += proj[d];

        // RMSNorm (FFN)
        core_cpu::rmsnorm_cpu(x.data(), normed.data(), D, ctx->get(lp + ".ffn_norm.weight"), eps);

        // SwiGLU FFN
        int ffn = ctx->llm_ffn_dim;
        std::vector<float> down(D);
        core_vlm::swiglu_ffn(normed.data(), down.data(), D, ffn, ctx->get(lp + ".ffn.gate.weight"),
                             ctx->get(lp + ".ffn.up.weight"), ctx->get(lp + ".ffn.down.weight"));

        // Residual
        for (int d = 0; d < D; d++) x[d] += down[d];

        // DequantCache: weights stay cached across layers and calls
    }

    // Final RMSNorm
    {
        std::vector<float> tmp(D);
        core_cpu::rmsnorm_cpu(x.data(), tmp.data(), D, ctx->get("llm.norm.weight"), eps);
        memcpy(x.data(), tmp.data(), D * sizeof(float));
    }

    // LM head (separate, NOT tied) — skip during prefill for speed
    // Uses SIMD-accelerated linear_cpu for the (49280 × 576) matmul
    if (!skip_logits) {
        const float * lm_w = ctx->get("llm.lm_head.weight");
        if (lm_w) core_cpu::linear_cpu(x.data(), logits, D, ctx->vocab_size, lm_w, nullptr);
    }
}

// ── Reference-pipeline preprocessing (aspect resize + tiling) ─────────
//
// The document-VLM reference pipeline does NOT feed one squashed square
// image: it (1) rescales the longest edge to 2048 preserving aspect,
// (2) rounds both dims up to multiples of the vision size (512),
// (3) splits into 512x512 tiles PLUS a squashed 512x512 global view, and
// (4) lays the prompt out per tile with <row_r_col_c> markers. Feeding a
// single squashed image instead makes the decoder hallucinate duplicated
// text regions (payload CER 0.86 on the fox fixture vs 0.0 for the
// reference on the same page).

static inline float sd_lanczos3(float x) {
    if (x <= -3.0f || x >= 3.0f) return 0.0f;
    if (x == 0.0f) return 1.0f;
    const float pi = 3.14159265358979323846f;
    float a = pi * x;
    return 3.0f * sinf(a) * sinf(a / 3.0f) / (a * a);
}

// One separable resampling pass along x for a single channel plane.
// Lanczos-3 with support widened by the scale factor on downscale (the
// convention of the reference pipeline's image library).
static void sd_resample_pass_x(const float * src, int sw, int sh, float * dst, int dw) {
    float scale = (float)sw / dw;
    float fscale = scale > 1.0f ? scale : 1.0f;
    float support = 3.0f * fscale;
    std::vector<int> xmins(dw), xlens(dw);
    std::vector<std::vector<float>> weights(dw);
    for (int xo = 0; xo < dw; xo++) {
        float center = (xo + 0.5f) * scale;
        int xmin = (int)floorf(center - support);
        int xmax = (int)ceilf(center + support);
        if (xmin < 0) xmin = 0;
        if (xmax > sw) xmax = sw;
        xmins[xo] = xmin;
        xlens[xo] = xmax - xmin;
        auto & w = weights[xo];
        w.resize(xlens[xo]);
        float wsum = 0.0f;
        for (int x = xmin; x < xmax; x++) {
            float ww = sd_lanczos3((x + 0.5f - center) / fscale);
            w[x - xmin] = ww;
            wsum += ww;
        }
        if (wsum != 0.0f)
            for (auto & ww : w) ww /= wsum;
    }
    for (int y = 0; y < sh; y++) {
        const float * row = src + (size_t)y * sw;
        float * orow = dst + (size_t)y * dw;
        for (int xo = 0; xo < dw; xo++) {
            float acc = 0.0f;
            const float * w = weights[xo].data();
            const float * s = row + xmins[xo];
            for (int k = 0; k < xlens[xo]; k++) acc += s[k] * w[k];
            orow[xo] = acc;
        }
    }
}

// Full 2-D Lanczos resize of one plane [sh x sw] -> [dh x dw].
static void sd_lanczos_resize_plane(const float * src, int sw, int sh, float * dst, int dw, int dh) {
    std::vector<float> tmp((size_t)sh * dw);
    sd_resample_pass_x(src, sw, sh, tmp.data(), dw);
    // vertical pass = horizontal pass on the transposed plane
    std::vector<float> tmp_t((size_t)dw * sh), out_t((size_t)dw * dh);
    for (int y = 0; y < sh; y++)
        for (int x = 0; x < dw; x++) tmp_t[(size_t)x * sh + y] = tmp[(size_t)y * dw + x];
    sd_resample_pass_x(tmp_t.data(), sh, dw, out_t.data(), dh);
    for (int x = 0; x < dw; x++)
        for (int y = 0; y < dh; y++) dst[(size_t)y * dw + x] = out_t[(size_t)x * dh + y];
}

// Longest edge -> max_len, aspect preserved, odd result bumped to even.
static void sd_rescale_to_max_len(int h, int w, int max_len, int * oh, int * ow) {
    float aspect = (float)w / (float)h;
    if (w >= h) {
        w = max_len;
        h = (int)(w / aspect);
        if (h % 2 != 0) h += 1;
    } else {
        h = max_len;
        w = (int)(h * aspect);
        if (w % 2 != 0) w += 1;
    }
    *oh = h > 1 ? h : 1;
    *ow = w > 1 ? w : 1;
}

// Round both dims up to multiples of `vis` recomputing the short side from
// the aspect ratio first (reference resize_for_vision_encoder).
static void sd_round_up_to_vis(int h, int w, int vis, int * oh, int * ow) {
    float aspect = (float)w / (float)h;
    if (w >= h) {
        w = ((w + vis - 1) / vis) * vis;
        h = (int)(w / aspect);
        h = ((h + vis - 1) / vis) * vis;
    } else {
        h = ((h + vis - 1) / vis) * vis;
        w = (int)(h * aspect);
        w = ((w + vis - 1) / vis) * vis;
    }
    *oh = h;
    *ow = w;
}

struct sd_preproc_result {
    // Normalized CHW float images (3 * vis * vis each): tiles row-major,
    // then the squashed global view last. rows==cols==0 => single image.
    std::vector<std::vector<float>> images;
    int rows = 0, cols = 0;
};

static void sd_preprocess_reference(const uint8_t * pixels, int width, int height, int channels, int vis,
                                    sd_preproc_result & out) {
    // Planar float RGB [0,255] at source size.
    std::vector<float> plane[3];
    for (int c = 0; c < 3; c++) plane[c].resize((size_t)height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            size_t si = ((size_t)y * width + x) * (channels >= 3 ? channels : 1);
            for (int c = 0; c < 3; c++) {
                int sc = channels >= 3 ? c : 0;
                plane[c][(size_t)y * width + x] = (float)pixels[si + sc];
            }
        }

    // Pass 1: longest edge -> 2048 (upscales small pages too — that is the
    // reference behavior; tiling below is what recovers detail).
    int h1, w1;
    sd_rescale_to_max_len(height, width, 2048, &h1, &w1);
    // Pass 2: round up to multiples of the vision size.
    int h2, w2;
    sd_round_up_to_vis(h1, w1, vis, &h2, &w2);

    std::vector<float> resized[3];
    for (int c = 0; c < 3; c++) {
        std::vector<float> mid((size_t)h1 * w1);
        sd_lanczos_resize_plane(plane[c].data(), width, height, mid.data(), w1, h1);
        resized[c].resize((size_t)h2 * w2);
        sd_lanczos_resize_plane(mid.data(), w1, h1, resized[c].data(), w2, h2);
    }

    auto normalize_into = [&](const float * ch_planes[3], int sw, std::vector<float> & img, int x0, int y0) {
        img.resize((size_t)3 * vis * vis);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < vis; y++)
                for (int x = 0; x < vis; x++) {
                    float v = ch_planes[c][(size_t)(y0 + y) * sw + (x0 + x)];
                    if (v < 0.0f) v = 0.0f;
                    if (v > 255.0f) v = 255.0f;
                    img[(size_t)c * vis * vis + (size_t)y * vis + x] = v / 127.5f - 1.0f;
                }
    };

    out.images.clear();
    if (h2 > vis || w2 > vis) {
        out.rows = h2 / vis;
        out.cols = w2 / vis;
        const float * rp[3] = { resized[0].data(), resized[1].data(), resized[2].data() };
        for (int r = 0; r < out.rows; r++)
            for (int cc = 0; cc < out.cols; cc++) {
                out.images.emplace_back();
                normalize_into(rp, w2, out.images.back(), cc * vis, r * vis);
            }
        // Global view: the tiled image squashed to vis x vis.
        std::vector<float> glob[3];
        for (int c = 0; c < 3; c++) {
            glob[c].resize((size_t)vis * vis);
            sd_lanczos_resize_plane(resized[c].data(), w2, h2, glob[c].data(), vis, vis);
        }
        const float * gp[3] = { glob[0].data(), glob[1].data(), glob[2].data() };
        out.images.emplace_back();
        normalize_into(gp, vis, out.images.back(), 0, 0);
    } else {
        out.rows = out.cols = 0;
        const float * rp[3] = { resized[0].data(), resized[1].data(), resized[2].data() };
        out.images.emplace_back();
        normalize_into(rp, w2, out.images.back(), 0, 0);
    }
}

// ── Main recognize (from raw pixels) ──────────────────────────────────

const char * smoldocling_recognize_raw(smoldocling_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    int img_size = ctx->vis_image_size; // 512
    int ps = ctx->vis_patch_size;
    int n_patches_side = img_size / ps;          // 32
    int T_vis = n_patches_side * n_patches_side; // 1024
    int D = ctx->llm_dim;

    // SMOLDOCLING_LEGACY_PREPROC=1 restores the pre-tiling single squashed
    // 512x512 input (regression-bisection gate; known to hallucinate
    // duplicated regions on non-square pages).
    const char * legacy_env = getenv("SMOLDOCLING_LEGACY_PREPROC");
    const bool legacy = legacy_env && legacy_env[0] == '1';

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    sd_preproc_result prep;
    if (legacy) {
        // Single squashed image, nearest-neighbor (historical behavior).
        prep.rows = prep.cols = 0;
        prep.images.emplace_back();
        auto & image = prep.images.back();
        image.resize((size_t)3 * img_size * img_size);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < img_size; y++)
                for (int x = 0; x < img_size; x++) {
                    float sy = (y + 0.5f) * height / img_size - 0.5f;
                    float sx = (x + 0.5f) * width / img_size - 0.5f;
                    int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                    int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                    int src_c = c;
                    if (channels == 1) src_c = 0; // grayscale
                    int src_idx = channels >= 3 ? (iy * width + ix) * channels + src_c : iy * width + ix;
                    image[(size_t)c * img_size * img_size + (size_t)y * img_size + x] = pixels[src_idx] / 127.5f - 1.0f;
                }
    } else {
        sd_preprocess_reference(pixels, width, height, channels, img_size, prep);
    }
    const int n_imgs = (int)prep.images.size();

    // Vision encoder + connector per sub-image (tiles then global view).
    fprintf(stderr, "smoldocling: running vision encoder on %d sub-image(s) (%dx%d grid)...\n", n_imgs, prep.rows,
            prep.cols);
    auto t_vis = std::chrono::steady_clock::now();
    std::vector<float> vis_features((size_t)T_vis * ctx->vis_dim);
    std::vector<float> connector_all; // [n_imgs * seq_per_img, D]
    int seq_per_img = 0;
    for (int i = 0; i < n_imgs; i++) {
        int n_vis_tokens = 0;
        sd_vision_forward(ctx, prep.images[i].data(), img_size, img_size, vis_features.data(), &n_vis_tokens);
        int n_conn = 0;
        std::vector<float> conn((size_t)n_vis_tokens * D);
        sd_connector(ctx, vis_features.data(), n_vis_tokens, conn.data(), &n_conn);
        if (seq_per_img == 0) {
            seq_per_img = n_conn;
            connector_all.reserve((size_t)n_imgs * n_conn * D);
        }
        connector_all.insert(connector_all.end(), conn.begin(), conn.begin() + (size_t)n_conn * D);
    }
    const int n_connector_tokens = n_imgs * seq_per_img;
    fprintf(stderr, "smoldocling: vision+connector done, %d tokens (%d per sub-image)\n", n_connector_tokens,
            seq_per_img);
    if (bench) {
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_vis).count();
        fprintf(stderr, "[smoldocling-bench] vision_encoder+connector: %lldms\n", (long long)ms);
    }

    // Build input token sequence following the SmolDocling chat template:
    //   <|im_start|>User:<IMAGE-PROMPT>Convert this page to docling.<end_of_utterance>\nAssistant:
    // where <IMAGE-PROMPT> is, per split image,
    //   (<fake_token_around_image><row_r_col_c><image>*seq)+ "\n" per row,
    //   then "\n" <fake_token_around_image><global-img><image>*seq<fake_token_around_image>
    // and for a single image just the global wrapper. Text runs between
    // special tokens must be BPE-encoded as ONE chunk (the row-final "\n"
    // and the global-prefix "\n" merge into a single "\n\n" token).
    // Special tokens are inserted by ID, never BPE-encoded.
    std::vector<int> input_ids;
    std::string pending_text;
    auto flush_text = [&]() {
        if (pending_text.empty()) return;
        auto ids = ctx->tokenizer.encode(pending_text);
        input_ids.insert(input_ids.end(), ids.begin(), ids.end());
        pending_text.clear();
    };
    auto push_special = [&](int id) {
        flush_text();
        input_ids.push_back(id);
    };
    auto special_id = [&](const std::string & name, int fallback) {
        auto it = ctx->tokenizer.token_to_id.find(name);
        if (it != ctx->tokenizer.token_to_id.end()) return it->second;
        return fallback;
    };
    const int fake_id = special_id("<fake_token_around_image>", 49189);
    const int global_id = special_id("<global-img>", 49152);

    input_ids.push_back(1); // <|im_start|> (BOS)
    pending_text = "User:";
    if (prep.rows > 0) {
        for (int r = 0; r < prep.rows; r++) {
            for (int c = 0; c < prep.cols; c++) {
                push_special(fake_id);
                char rowcol[32];
                snprintf(rowcol, sizeof(rowcol), "<row_%d_col_%d>", r + 1, c + 1);
                push_special(special_id(rowcol, -1));
                for (int i = 0; i < seq_per_img; i++) input_ids.push_back(ctx->image_token_id);
            }
            pending_text += "\n";
        }
        pending_text += "\n";
        push_special(fake_id);
        push_special(global_id);
        for (int i = 0; i < seq_per_img; i++) input_ids.push_back(ctx->image_token_id);
        push_special(fake_id);
    } else if (!legacy) {
        push_special(fake_id);
        push_special(global_id);
        for (int i = 0; i < seq_per_img; i++) input_ids.push_back(ctx->image_token_id);
        push_special(fake_id);
    } else {
        // Historical prompt: bare <image> run with no wrapper tokens.
        flush_text();
        for (int i = 0; i < n_connector_tokens; i++) input_ids.push_back(ctx->image_token_id);
    }
    pending_text += "Convert this page to docling.";
    push_special(49279); // <end_of_utterance>
    pending_text = "\nAssistant:";
    flush_text();

    if (input_ids.end() != std::find(input_ids.begin(), input_ids.end(), -1)) {
        fprintf(stderr, "smoldocling: missing <row_r_col_c> special token in vocab (stale GGUF without added "
                        "tokens?) — falling back may produce degraded output\n");
    }

    fprintf(stderr, "smoldocling: prompt has %d tokens (%d image + %d text)\n", (int)input_ids.size(),
            n_connector_tokens, (int)input_ids.size() - n_connector_tokens);

    // Prompt-contract debug: dump the prefill ids for parity checks against
    // the reference processor (one id per line).
    if (const char * dump_path = getenv("SMOLDOCLING_DEBUG_PROMPT")) {
        if (FILE * f = fopen(dump_path, "w")) {
            for (int id : input_ids) fprintf(f, "%d\n", id);
            fclose(f);
        }
    }

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
            float * dst = prefill_embeds.data() + (size_t)t * D;
            if (input_ids[t] == ctx->image_token_id && vis_idx < n_connector_tokens) {
                memcpy(dst, connector_all.data() + (size_t)vis_idx * D, D * sizeof(float));
                vis_idx++;
            } else {
                memcpy(dst, embed_w + (size_t)input_ids[t] * D, D * sizeof(float));
            }
        }
    }

    // Try ggml batched prefill path
    bool use_ggml = (ctx->llm_sched != nullptr) && sd_alloc_kv_cache(ctx, max_seq);

    std::vector<float> logits(ctx->vocab_size);
    std::vector<float> hidden(D);
    ctx->n_past = 0;

    fprintf(stderr, "smoldocling: starting prefill of %d tokens (%s)...\n", prefill_len,
            use_ggml ? "ggml batched" : "scalar");

    auto t_prefill = std::chrono::steady_clock::now();

    if (use_ggml) {
        if (!sd_run_llm_body(ctx, prefill_embeds.data(), prefill_len, 0, hidden.data())) {
            fprintf(stderr, "smoldocling: ggml prefill failed, falling back to scalar\n");
            sd_free_kv_cache(ctx);
            use_ggml = false;
        } else {
            ctx->n_past = prefill_len;
            if (lm_head_w) core_cpu::linear_cpu(hidden.data(), logits.data(), D, ctx->vocab_size, lm_head_w, nullptr);
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
            sd_llm_decode_step(ctx, prefill_embeds.data() + (size_t)t * D, ctx->n_past, logits.data(),
                               /*skip_logits=*/!is_last);
            ctx->n_past++;
        }
    }

    if (bench) {
        auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_prefill).count();
        fprintf(stderr, "[smoldocling-bench] prefill: %lldms\n", (long long)ms);
    }

    // Greedy decode
    ctx->output_text.clear();
    std::vector<int> output_ids;
    const int eos_id = 2;     // <|im_end|>
    const int eou_id = 49279; // <end_of_utterance>

    auto t_decode_start = std::chrono::steady_clock::now();
    for (int step = 0; step < ctx->max_tokens; step++) {
        auto t_step = std::chrono::steady_clock::now();

        int best_id = 0;
        float best_score = logits[0];
        for (int v = 1; v < ctx->vocab_size; v++)
            if (logits[v] > best_score) {
                best_score = logits[v];
                best_id = v;
            }

        if (best_id == eos_id || best_id == eou_id) break;
        output_ids.push_back(best_id);

        const float * next_emb = embed_w + (size_t)best_id * D;

        if (use_ggml) {
            if (!sd_run_llm_body(ctx, next_emb, 1, ctx->n_past, hidden.data())) {
                fprintf(stderr, "smoldocling: ggml decode step failed at step %d\n", step);
                break;
            }
            ctx->n_past++;
            if (lm_head_w) core_cpu::linear_cpu(hidden.data(), logits.data(), D, ctx->vocab_size, lm_head_w, nullptr);
        } else {
            std::vector<float> next_embed(next_emb, next_emb + D);
            sd_llm_decode_step(ctx, next_embed.data(), ctx->n_past, logits.data());
            ctx->n_past++;
        }

        if (bench) {
            auto step_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_step)
                    .count();
            if (step == 0 || step == 1)
                fprintf(stderr, "[smoldocling-bench] decode_step[%d]: %lldms\n", step, (long long)step_ms);
        }
    }
    if (bench) {
        auto t_decode_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[smoldocling-bench] decode (%d steps): %.1f ms\n", (int)output_ids.size(),
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

const char * smoldocling_recognize(smoldocling_context * ctx, const char * image_path, int * out_len) {
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

static void sd_preprocess_image(const uint8_t * pixels, int width, int height, int channels, int img_size,
                                std::vector<float> & image) {
    image.resize(3 * img_size * img_size);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < img_size; y++)
            for (int x = 0; x < img_size; x++) {
                float sy = (y + 0.5f) * height / img_size - 0.5f;
                float sx = (x + 0.5f) * width / img_size - 0.5f;
                int iy = std::max(0, std::min(height - 1, (int)(sy + 0.5f)));
                int ix = std::max(0, std::min(width - 1, (int)(sx + 0.5f)));
                int src_c = (channels >= 3) ? c : 0;
                int src_idx = (channels >= 3) ? (iy * width + ix) * channels + src_c : iy * width + ix;
                image[c * img_size * img_size + y * img_size + x] = pixels[src_idx] / 127.5f - 1.0f;
            }
}

float * smoldocling_debug_vision(smoldocling_context * ctx, const uint8_t * pixels, int w, int h, int ch,
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

float * smoldocling_debug_connector(smoldocling_context * ctx, const uint8_t * pixels, int w, int h, int ch,
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
