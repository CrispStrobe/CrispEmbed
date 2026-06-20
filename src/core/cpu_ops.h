// src/core/cpu_ops.h — Shared CPU-scalar helper functions for VLM/OCR engines.
//
// Header-only. All functions live in namespace core_cpu with static inline
// linkage to avoid ODR violations when included from multiple TUs.
//
// Extracted from the ~7 engine files that copy-pasted identical helpers:
//   surya_det, got_ocr, ppformulanet_l_ocr, ppformulanet_ocr,
//   deepseek_ocr2, mixtex_ocr, math_ocr.
//
// Usage:
//   #include "core/cpu_ops.h"
//   using core_cpu::to_f32;
//   using core_cpu::layernorm_cpu;
//   // ... etc.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace core_cpu {

// ---------------------------------------------------------------------------
// FP16/quantized → F32 dequantization (GPU-safe)
// ---------------------------------------------------------------------------
// Uses ggml_backend_tensor_get() so this works whether the weight lives in a
// CPU buffer or a GPU (Metal/CUDA) buffer where t->data is not a valid host
// pointer.

static inline std::vector<float> to_f32(const ggml_tensor* t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Quantized: read raw bytes then dequantize via type traits
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            memset(out.data(), 0, n * sizeof(float));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dequantization cache — avoids re-dequantizing the same immutable weights
// ---------------------------------------------------------------------------
// Per-context cache: call cache.get(tensor) instead of to_f32(tensor).
// The returned pointer is valid for the lifetime of the cache. Thread-safety:
// each inference context should own its own DequantCache instance.

struct DequantCache {
    std::unordered_map<const void*, std::vector<float>> cache_;

    const float* get(const ggml_tensor* t) {
        if (!t) return nullptr;
        auto it = cache_.find(t->data);
        if (it != cache_.end()) return it->second.data();
        auto& v = cache_[t->data];
        v = to_f32(t);
        return v.data();
    }

    void clear() { cache_.clear(); }
};

// ---------------------------------------------------------------------------
// Dot product helper (SIMD-accelerated)
// ---------------------------------------------------------------------------
// Used by linear_cpu and mha_1q_cpu. AVX2+FMA (x86-64), NEON (ARM), scalar fallback.

static inline float dot_product(const float* a, const float* b, int n) {
    float s = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 7 < n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
#elif defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; i + 3 < n; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    s = vaddvq_f32(acc0);
    for (; i < n; i++) s += a[i] * b[i];
#else
    for (int i = 0; i < n; i++) s += a[i] * b[i];
#endif
    return s;
}

// ---------------------------------------------------------------------------
// LayerNorm (raw float pointers)
// ---------------------------------------------------------------------------
// Standard LayerNorm: mean/var over D, then scale+shift.
// eps has no default — callers must be explicit to avoid silent behavior changes
// across engines that historically used different defaults (1e-5, 1e-6, 1e-12).

static inline void layernorm_cpu(const float* in, float* out, int D,
                                 const float* w, const float* b, float eps) {
    double mean = 0;
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= D;
    double var = 0;
    for (int i = 0; i < D; i++) { double d = in[i] - mean; var += d * d; }
    var /= D;
    float s = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < D; i++)
        out[i] = ((in[i] - (float)mean) * s) * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

// ---------------------------------------------------------------------------
// LayerNorm (ggml_tensor overload — dequantizes w/b via to_f32)
// ---------------------------------------------------------------------------

static inline void layernorm_cpu(const float* in, float* out, int D,
                                 const ggml_tensor* w, const ggml_tensor* b, float eps) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    layernorm_cpu(in, out, D,
                  wv.empty() ? nullptr : wv.data(),
                  bv.empty() ? nullptr : bv.data(), eps);
}

// ---------------------------------------------------------------------------
// LayerNorm2d — normalize over channel dim for NCHW tensors
// ---------------------------------------------------------------------------
// Input/output shape: (C, H, W), normalize over C at each spatial position.

static inline void layernorm2d_cpu(const float* in, float* out,
                                   int C, int H, int W,
                                   const float* w, const float* b, float eps) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double mean = 0;
            for (int c = 0; c < C; c++)
                mean += in[c * H * W + y * W + x];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; c++) {
                double d = in[c * H * W + y * W + x] - mean;
                var += d * d;
            }
            var /= C;
            float s = 1.0f / sqrtf((float)var + eps);
            for (int c = 0; c < C; c++) {
                float v = (in[c * H * W + y * W + x] - (float)mean) * s;
                out[c * H * W + y * W + x] = v * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm — root-mean-square normalization (no mean subtraction)
// ---------------------------------------------------------------------------

static inline void rmsnorm_cpu(const float* in, float* out, int D,
                               const float* w, float eps) {
    double ss = 0;
    for (int i = 0; i < D; i++) ss += (double)in[i] * in[i];
    float s = 1.0f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = in[i] * s * (w ? w[i] : 1.0f);
}

// ---------------------------------------------------------------------------
// Linear (matrix-vector multiply)
// ---------------------------------------------------------------------------
// Convention: out[o] = sum_i(w[o*in_dim+i] * in[i]) + b[o]

static inline void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                               const float* w, const float* b) {
    for (int o = 0; o < out_dim; o++) {
        float s = dot_product(in, w + o * in_dim, in_dim);
        out[o] = s + (b ? b[o] : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Linear (ggml_tensor overload — dequantizes w/b via to_f32)
// ---------------------------------------------------------------------------

static inline void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                               const ggml_tensor* w, const ggml_tensor* b) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    linear_cpu(in, out, in_dim, out_dim, wv.data(), bv.empty() ? nullptr : bv.data());
}

// ---------------------------------------------------------------------------
// Conv2d (NCHW layout) with groups, padding, stride
// ---------------------------------------------------------------------------
// Weights: [OC, IC/groups, KH, KW]. groups=1 for standard convolution.

static inline void conv2d_cpu(const float* in, float* out,
                               const float* weight, const float* bias,
                               int in_ch, int out_ch, int H, int W,
                               int kh, int kw, int stride, int pad,
                               int groups = 1) {
    int out_H = (H + 2 * pad - kh) / stride + 1;
    int out_W = (W + 2 * pad - kw) / stride + 1;
    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;

    for (int oc = 0; oc < out_ch; oc++) {
        int g = oc / ch_per_group_out;
        float b = bias ? bias[oc] : 0.0f;

        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                float sum = b;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    int actual_ic = g * ch_per_group_in + ic;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * stride - pad + ky;
                            int ix = ox * stride - pad + kx;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                float pixel = in[actual_ic * H * W + iy * W + ix];
                                int w_idx = oc * (ch_per_group_in * kh * kw)
                                            + ic * kh * kw + ky * kw + kx;
                                sum += pixel * weight[w_idx];
                            }
                        }
                    }
                }
                out[oc * out_H * out_W + oy * out_W + ox] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

// GELU (tanh approximation) — matches PyTorch nn.GELU(approximate='tanh')
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// GELU (exact, erf-based) — matches PyTorch nn.GELU() default
static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

// SiLU (Swish): x * sigmoid(x)
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// In-place SiLU over a buffer
static inline void silu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) data[i] = data[i] / (1.0f + expf(-data[i]));
}

// In-place softmax
static inline void softmax(float* data, int n) {
    float mx = data[0];
    for (int i = 1; i < n; i++) if (data[i] > mx) mx = data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { data[i] = expf(data[i] - mx); sum += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= sum;
}

// HardSwish: x * min(max(x+3, 0), 6) / 6
static inline void hardswish_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        float x = data[i];
        if (x <= -3.0f) data[i] = 0.0f;
        else if (x >= 3.0f) { /* keep x */ }
        else data[i] = x * (x + 3.0f) / 6.0f;
    }
}

// ReLU6: clamp to [0, 6]
static inline void relu6_inplace(float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
        else if (data[i] > 6.0f) data[i] = 6.0f;
    }
}

// ReLU: max(0, x)
static inline void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Pooling helpers (NCHW layout, explicit output dims for ceil_mode compat)
// ---------------------------------------------------------------------------

// MaxPool2d: sliding window max, out[c,oy,ox] = max of window starting at (oy*s, ox*s).
// Caller pre-computes oh/ow (handles ceil_mode by expanding them before calling).
static inline void maxpool2d_cpu(const float* in, int ch, int ih, int iw,
                                  int k, int s, float* out, int oh, int ow) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float mx = -1e30f;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int iy = y * s + ky, ix = x * s + kx;
                        if (iy < ih && ix < iw) {
                            float v = in[c * ih * iw + iy * iw + ix];
                            if (v > mx) mx = v;
                        }
                    }
                out[c * oh * ow + y * ow + x] = mx;
            }
}

// AvgPool2d: counts only valid (in-bounds) pixels in the denominator.
static inline void avgpool2d_cpu(const float* in, int ch, int ih, int iw,
                                  int k, int s, float* out, int oh, int ow) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float sum = 0; int cnt = 0;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int iy = y * s + ky, ix = x * s + kx;
                        if (iy < ih && ix < iw) {
                            sum += in[c * ih * iw + iy * iw + ix];
                            cnt++;
                        }
                    }
                out[c * oh * ow + y * ow + x] = cnt > 0 ? sum / cnt : 0.0f;
            }
}

// ---------------------------------------------------------------------------
// BatchNorm (folded, i.e. scale+offset applied in-place)
// ---------------------------------------------------------------------------
// data layout: (ch, sp) = CHW flattened. Applies data[c,i] = data[c,i]*scale[c] + offset[c].

static inline void apply_bn_cpu(float* data, int ch, int sp,
                                 const float* scale, const float* offset) {
    for (int c = 0; c < ch; c++) {
        float s = scale[c], o = offset[c];
        float* row = data + c * sp;
        for (int i = 0; i < sp; i++) row[i] = row[i] * s + o;
    }
}

// Multi-head attention (single query position)
// q: [D], k: [n_kv, D], v: [n_kv, D], out: [D]
static inline void mha_1q_cpu(const float* q, const float* k, const float* v,
                               float* out, int n_kv, int D, int n_heads) {
    int hd = D / n_heads;
    std::vector<float> result(D, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_kv);
        float scale = 1.0f / sqrtf((float)hd);
        for (int ki = 0; ki < n_kv; ki++)
            scores[ki] = dot_product(q + off, k + ki * D + off, hd) * scale;
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_kv; ki++) scores[ki] *= inv_sum;
        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_kv; ki++)
                s += scores[ki] * v[ki * D + off + d];
            result[off + d] = s;
        }
    }
    memcpy(out, result.data(), D * sizeof(float));
}

} // namespace core_cpu
