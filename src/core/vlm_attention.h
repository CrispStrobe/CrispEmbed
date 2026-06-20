// src/core/vlm_attention.h — Shared scalar VLM decoder building blocks.
//
// Header-only. All functions live in namespace core_vlm with static inline
// linkage to avoid ODR violations when included from multiple TUs.
//
// Extracted from smoldocling_ocr.cpp and granite_vision_ocr.cpp which had
// identical ~200-line decode loops. This is Phase 2 of the refactoring
// started by core/cpu_ops.h (Phase 1).
//
// Building blocks provided:
//   - apply_rope()      — scalar RoPE in NEGHALF or INTERLEAVED style
//   - alloc_kv_cache()  — allocate flat F32 KV cache
//   - kv_k_offset()     — index into K portion of flat cache
//   - kv_v_offset()     — index into V portion of flat cache
//   - gqa_attn_step()   — single-token GQA self-attention with KV cache
//   - swiglu_ffn()      — SwiGLU feed-forward network step
//
// Usage:
//   #include "core/vlm_attention.h"
//   using core_vlm::apply_rope;
//   using core_vlm::gqa_attn_step;
//   // ... etc.

#pragma once

#include "core/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace core_vlm {

// ---------------------------------------------------------------------------
// RoPE styles
// ---------------------------------------------------------------------------

enum class RoPEStyle {
    NEGHALF,      // GPT-NeoX / SmolLM2: pairs (d, d+half), rotation [-x_hi, x_lo]
    INTERLEAVED   // Llama / Granite: pairs (2d, 2d+1), adjacent rotation
};

// ---------------------------------------------------------------------------
// RoPE frequency table — precompute once, reuse every step
// ---------------------------------------------------------------------------
// freqs[d] = 1.0 / theta^(2d / head_dim), for d in [0, head_dim/2).
// Call precompute() once at init, then use apply() instead of apply_rope().

struct RoPEFreqTable {
    std::vector<float> freqs;   // [head_dim/2]
    int head_dim = 0;

    void precompute(int hd, float theta) {
        head_dim = hd;
        int half = hd / 2;
        freqs.resize(half);
        for (int d = 0; d < half; d++)
            freqs[d] = 1.0f / powf(theta, 2.0f * d / hd);
    }

    void apply(float* qk, int n_heads, int position, RoPEStyle style) const {
        int half = head_dim / 2;
        for (int h = 0; h < n_heads; h++) {
            float* head = qk + h * head_dim;
            for (int d = 0; d < half; d++) {
                float angle = position * freqs[d];
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                if (style == RoPEStyle::NEGHALF) {
                    float lo = head[d];
                    float hi = head[d + half];
                    head[d]        = lo * cos_a - hi * sin_a;
                    head[d + half] = hi * cos_a + lo * sin_a;
                } else {
                    int i0 = 2 * d;
                    float v0 = head[i0];
                    float v1 = head[i0 + 1];
                    head[i0]     = v0 * cos_a - v1 * sin_a;
                    head[i0 + 1] = v0 * sin_a + v1 * cos_a;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Scalar RoPE — apply rotary position embedding in-place
// ---------------------------------------------------------------------------
// Operates on a [n_heads * head_dim] Q or K vector.
//
// NEGHALF (SmolLM2/GPT-NeoX):
//   For each head, pair indices (d, d+half) where half = head_dim/2.
//   x[d]      = x[d] * cos - x[d+half] * sin
//   x[d+half] = x[d+half] * cos + x[d] * sin
//
// INTERLEAVED (Llama/Granite):
//   For each head, pair indices (2d, 2d+1).
//   x[2d]   = x[2d] * cos - x[2d+1] * sin
//   x[2d+1] = x[2d] * sin + x[2d+1] * cos

static inline void apply_rope(float* qk, int n_heads, int head_dim,
                               int position, float theta, RoPEStyle style) {
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float* head = qk + h * head_dim;

        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, 2.0f * d / head_dim);
            float angle = position * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            if (style == RoPEStyle::NEGHALF) {
                float lo = head[d];
                float hi = head[d + half];
                head[d]        = lo * cos_a - hi * sin_a;
                head[d + half] = hi * cos_a + lo * sin_a;
            } else {  // INTERLEAVED
                int i0 = 2 * d;
                int i1 = i0 + 1;
                float v0 = head[i0];
                float v1 = head[i1];
                head[i0] = v0 * cos_a - v1 * sin_a;
                head[i1] = v0 * sin_a + v1 * cos_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache allocation and indexing
// ---------------------------------------------------------------------------
// Flat layout: [layer][K_or_V][seq_pos][kv_dim]
// where kv_dim = n_kv_heads * head_dim.
// Total size: 2 * n_layers * max_seq * kv_dim floats.

static inline std::vector<float> alloc_kv_cache(int n_layers, int max_seq,
                                                 int n_kv_heads, int head_dim) {
    size_t kv_dim = (size_t)n_kv_heads * head_dim;
    size_t total = 2 * (size_t)n_layers * max_seq * kv_dim;
    return std::vector<float>(total, 0.0f);
}

// Offset to K[layer][seq_pos] in the flat cache.
static inline size_t kv_k_offset(int layer, int seq_pos, int kv_dim,
                                  int max_seq, int n_layers) {
    (void)n_layers;
    return (size_t)layer * 2 * max_seq * kv_dim + (size_t)seq_pos * kv_dim;
}

// Offset to V[layer][seq_pos] in the flat cache.
static inline size_t kv_v_offset(int layer, int seq_pos, int kv_dim,
                                  int max_seq, int n_layers) {
    (void)n_layers;
    return (size_t)layer * 2 * max_seq * kv_dim + (size_t)max_seq * kv_dim
           + (size_t)seq_pos * kv_dim;
}

// ---------------------------------------------------------------------------
// Scalar GQA self-attention step with KV cache
// ---------------------------------------------------------------------------
// Writes new K,V into flat cache at position n_past.
// Computes attention over all past K,V positions [0..n_past].
// Supports GQA: n_kv_heads <= n_heads, kv_repeat = n_heads / n_kv_heads.
//
// q:       [n_heads * head_dim]      — query vector (already RoPE'd)
// k_new:   [n_kv_heads * head_dim]   — new key (already RoPE'd)
// v_new:   [n_kv_heads * head_dim]   — new value
// kv_cache: flat F32 buffer from alloc_kv_cache
// output:  [n_heads * head_dim]      — attention output

static inline void gqa_attn_step(
        const float* q,
        const float* k_new,
        const float* v_new,
        float* kv_cache,
        int n_heads, int n_kv_heads, int head_dim,
        int max_seq, int n_past,
        int layer_idx, int n_layers,
        float* output,
        float attn_scale = -1.0f) {
    int kv_dim = n_kv_heads * head_dim;
    int kv_repeat = n_heads / n_kv_heads;
    int Lk = n_past + 1;

    // Write new K, V into cache at position n_past
    size_t k_base = kv_k_offset(layer_idx, 0, kv_dim, max_seq, n_layers);
    size_t v_base = kv_v_offset(layer_idx, 0, kv_dim, max_seq, n_layers);

    memcpy(kv_cache + k_base + (size_t)n_past * kv_dim,
           k_new, kv_dim * sizeof(float));
    memcpy(kv_cache + v_base + (size_t)n_past * kv_dim,
           v_new, kv_dim * sizeof(float));

    // Per-head attention. Most models use 1/sqrt(head_dim); Granite passes an
    // explicit attention_multiplier via attn_scale.
    float scale = attn_scale >= 0.0f ? attn_scale : 1.0f / sqrtf((float)head_dim);

    // Thread-local scores buffer (avoids per-head heap alloc)
    static thread_local std::vector<float> tl_scores;
    if ((int)tl_scores.size() < Lk) tl_scores.resize(Lk);
    float* scores = tl_scores.data();

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_repeat;

        // Compute attention scores: Q·K^T (SIMD-accelerated via dot_product)
        float max_s = -1e9f;
        for (int k = 0; k < Lk; k++) {
            float s = core_cpu::dot_product(
                q + h * head_dim,
                kv_cache + k_base + k * kv_dim + kv_h * head_dim,
                head_dim) * scale;
            scores[k] = s;
            if (s > max_s) max_s = s;
        }

        // Softmax
        float sum_e = 0;
        for (int k = 0; k < Lk; k++) {
            scores[k] = expf(scores[k] - max_s);
            sum_e += scores[k];
        }
        float inv = 1.0f / sum_e;

        // Weighted sum of values
        for (int d = 0; d < head_dim; d++) {
            float val = 0;
            for (int k = 0; k < Lk; k++)
                val += scores[k] * inv
                     * kv_cache[v_base + k * kv_dim + kv_h * head_dim + d];
            output[h * head_dim + d] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU FFN step
// ---------------------------------------------------------------------------
// out = down_proj(silu(gate_proj(x)) * up_proj(x))
// Uses core_cpu::linear_cpu and core_cpu::silu internally.
//
// x:    [hidden_dim]        — input (typically after RMSNorm)
// out:  [hidden_dim]        — output (same dim as input)
// gate_w: [intermediate_dim * hidden_dim]  — gate projection weights
// up_w:   [intermediate_dim * hidden_dim]  — up projection weights
// down_w: [hidden_dim * intermediate_dim]  — down projection weights

static inline void swiglu_ffn(const float* x, float* out,
                                int hidden_dim, int intermediate_dim,
                                const float* gate_w, const float* up_w,
                                const float* down_w) {
    // Thread-local buffers (avoids per-call heap alloc for intermediate_dim vectors)
    static thread_local std::vector<float> tl_gate, tl_up;
    if ((int)tl_gate.size() < intermediate_dim) {
        tl_gate.resize(intermediate_dim);
        tl_up.resize(intermediate_dim);
    }

    core_cpu::linear_cpu(x, tl_gate.data(), hidden_dim, intermediate_dim,
                         gate_w, nullptr);
    core_cpu::linear_cpu(x, tl_up.data(), hidden_dim, intermediate_dim,
                         up_w, nullptr);

    // silu(gate) * up
    for (int i = 0; i < intermediate_dim; i++)
        tl_gate[i] = core_cpu::silu(tl_gate[i]) * tl_up[i];

    core_cpu::linear_cpu(tl_gate.data(), out, intermediate_dim, hidden_dim,
                         down_w, nullptr);
}

}  // namespace core_vlm
