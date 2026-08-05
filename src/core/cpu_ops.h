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

#include <cstdlib>

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#include "env_gate.h"

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

static inline std::vector<float> to_f32(const ggml_tensor * t) {
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
        const auto * traits = ggml_get_type_traits(t->type);
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
    // Backend-resident tensors commonly have t->data == nullptr.  Use the
    // tensor object as the identity so F16/Metal/CPU-buffer weights cannot
    // alias one another in the cache.
    std::unordered_map<const ggml_tensor *, std::vector<float>> cache_;

    const float * get(const ggml_tensor * t) {
        if (!t) return nullptr;
        auto it = cache_.find(t);
        if (it != cache_.end()) return it->second.data();
        auto & v = cache_[t];
        v = to_f32(t);
        return v.data();
    }

    void clear() { cache_.clear(); }
};

// ---------------------------------------------------------------------------
// Dot product helper (SIMD-accelerated)
// ---------------------------------------------------------------------------
// Used by linear_cpu and mha_1q_cpu. AVX2+FMA (x86-64), NEON (ARM), scalar fallback.

// Opt-in wider accumulator count for dot_product: CRISPEMBED_DOT_WIDE=1.
//
// Why this exists. Profiling the PP-OCRv6 scalar detector on both machines
// showed the SAME generic conv2d_cpu running its 1x1 layers at ~1.2 GF/s on an
// M1 and 5.8-10.8 GF/s on a Xeon Skylake -- a 5-8x gap from identical C++. The
// arithmetic below explains it. dot_product keeps two accumulators, so it has
// two loop-carried FMA dependency chains. On the M1 an FP FMA has ~4-cycle
// latency against 4 NEON pipes, so two chains can retire 2 FMAs per 4 cycles,
// about 0.5/cycle against a ~4/cycle peak -- an ~8x shortfall that matches the
// measured gap. The AVX2 arm has the same chain depth but each FMA is twice as
// wide, so it loses proportionally less.
//
// The fix is more INDEPENDENT accumulators, not wider ones: four chains give
// the scheduler enough to cover the latency. This changes the summation order,
// so it is numerically different from the two-accumulator form (both are
// equally "correct" -- neither is the exact real sum) and therefore gated.
//
// Declared as a C++17 inline variable rather than a function-local static on
// purpose: a function-local static with a non-constant initialiser needs a
// guard-variable check on every call, and this function is called millions of
// times per page. This is initialised once before main and reads as a plain
// load, so the branch is perfectly predicted.
inline const bool g_dot_wide = std::getenv("CRISPEMBED_DOT_WIDE") != nullptr;

static inline float dot_product_wide(const float * a, const float * b, int n) {
    float s = 0.0f;
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 31 < n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0), _mm256_loadu_ps(b + i + 0), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    a0 = _mm256_add_ps(a0, a1);
    a2 = _mm256_add_ps(a2, a3);
    a0 = _mm256_add_ps(a0, a2);
    for (; i + 7 < n; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    __m128 lo = _mm256_castps256_ps128(a0);
    __m128 hi = _mm256_extractf128_ps(a0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    s = _mm_cvtss_f32(lo);
#elif defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (; i + 15 < n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    a0 = vaddq_f32(a0, a1);
    a2 = vaddq_f32(a2, a3);
    a0 = vaddq_f32(a0, a2);
    for (; i + 3 < n; i += 4) a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    s = vaddvq_f32(a0);
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float dot_product(const float * a, const float * b, int n) {
    if (g_dot_wide) return dot_product_wide(a, b, n);
    float s = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 7 < n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
#elif defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    acc0 = vaddq_f32(acc0, acc1);
    for (; i + 3 < n; i += 4) acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
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

static inline void layernorm_cpu(const float * in, float * out, int D, const float * w, const float * b, float eps) {
    // Mean (SIMD-accelerated sum)
    float fmean;
    {
        float sum = 0.0f;
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < D; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(in + i));
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        sum = _mm_cvtss_f32(lo);
        for (; i < D; i++) sum += in[i];
#else
        for (int i = 0; i < D; i++) sum += in[i];
#endif
        fmean = sum / D;
    }
    // Variance (SIMD-accelerated sum of squared differences)
    float fvar;
    {
        float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vmean = _mm256_set1_ps(fmean);
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < D; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(in + i), vmean);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        ss = _mm_cvtss_f32(lo);
        for (; i < D; i++) {
            float d = in[i] - fmean;
            ss += d * d;
        }
#else
        for (int i = 0; i < D; i++) {
            float d = in[i] - fmean;
            ss += d * d;
        }
#endif
        fvar = ss / D;
    }
    // Scale + shift (SIMD-accelerated)
    float s = 1.0f / sqrtf(fvar + eps);
#if defined(__AVX2__) && defined(__FMA__)
    {
        __m256 vs = _mm256_set1_ps(s);
        __m256 vm = _mm256_set1_ps(fmean);
        int i = 0;
        if (w && b) {
            for (; i + 7 < D; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(in + i), vm), vs);
                v = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + i), _mm256_loadu_ps(b + i));
                _mm256_storeu_ps(out + i, v);
            }
        } else if (w) {
            for (; i + 7 < D; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(in + i), vm), vs);
                _mm256_storeu_ps(out + i, _mm256_mul_ps(v, _mm256_loadu_ps(w + i)));
            }
        }
        for (; i < D; i++) out[i] = ((in[i] - fmean) * s) * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
#else
    for (int i = 0; i < D; i++) out[i] = ((in[i] - fmean) * s) * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
#endif
}

// ---------------------------------------------------------------------------
// LayerNorm (ggml_tensor overload — dequantizes w/b via to_f32)
// ---------------------------------------------------------------------------

static inline void layernorm_cpu(const float * in, float * out, int D, const ggml_tensor * w, const ggml_tensor * b,
                                 float eps) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    layernorm_cpu(in, out, D, wv.empty() ? nullptr : wv.data(), bv.empty() ? nullptr : bv.data(), eps);
}

// ---------------------------------------------------------------------------
// LayerNorm2d — normalize over channel dim for NCHW tensors
// ---------------------------------------------------------------------------
// Input/output shape: (C, H, W), normalize over C at each spatial position.

static inline void layernorm2d_cpu(const float * in, float * out, int C, int H, int W, const float * w, const float * b,
                                   float eps) {
    // Thread-local gather buffer (avoids per-call heap alloc)
    static thread_local std::vector<float> buf;
    if ((int)buf.size() < C) buf.resize(C);
    int HW = H * W;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int pos = y * W + x;
            // Gather: strided → contiguous
            for (int c = 0; c < C; c++) buf[c] = in[c * HW + pos];
            // Compute layernorm on contiguous buffer
            layernorm_cpu(buf.data(), buf.data(), C, w, b, eps);
            // Scatter: contiguous → strided
            for (int c = 0; c < C; c++) out[c * HW + pos] = buf[c];
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm — root-mean-square normalization (no mean subtraction)
// ---------------------------------------------------------------------------

static inline void rmsnorm_cpu(const float * in, float * out, int D, const float * w, float eps) {
    // Sum of squares (SIMD-accelerated)
    float ss = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < D; i += 8) {
            __m256 v = _mm256_loadu_ps(in + i);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        ss = _mm_cvtss_f32(lo);
        for (; i < D; i++) ss += in[i] * in[i];
    }
#else
    for (int i = 0; i < D; i++) ss += in[i] * in[i];
#endif
    float s = 1.0f / sqrtf(ss / D + eps);
    // Scale (SIMD-accelerated)
#if defined(__AVX2__)
    {
        __m256 vs = _mm256_set1_ps(s);
        int i = 0;
        if (w) {
            for (; i + 7 < D; i += 8)
                _mm256_storeu_ps(out + i,
                                 _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in + i), vs), _mm256_loadu_ps(w + i)));
        } else {
            for (; i + 7 < D; i += 8) _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(in + i), vs));
        }
        for (; i < D; i++) out[i] = in[i] * s * (w ? w[i] : 1.0f);
    }
#else
    for (int i = 0; i < D; i++) out[i] = in[i] * s * (w ? w[i] : 1.0f);
#endif
}

// ---------------------------------------------------------------------------
// Linear (matrix-vector multiply)
// ---------------------------------------------------------------------------
// Convention: out[o] = sum_i(w[o*in_dim+i] * in[i]) + b[o]

static inline void linear_cpu(const float * in, float * out, int in_dim, int out_dim, const float * w,
                              const float * b) {
    for (int o = 0; o < out_dim; o++) {
        float s = dot_product(in, w + o * in_dim, in_dim);
        out[o] = s + (b ? b[o] : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Linear batched (matrix-matrix multiply): N tokens at once
// ---------------------------------------------------------------------------
// in:  [N, in_dim] row-major    (N rows of in_dim)
// out: [N, out_dim] row-major   (N rows of out_dim)
// w:   [out_dim, in_dim] row-major
// Equivalent to: for (i=0..N-1) linear_cpu(in+i*in_dim, out+i*out_dim, ...)

static inline void linear_batch_cpu(const float * in, float * out, int N, int in_dim, int out_dim, const float * w,
                                    const float * b) {
    for (int i = 0; i < N; i++) linear_cpu(in + i * in_dim, out + i * out_dim, in_dim, out_dim, w, b);
}

// ---------------------------------------------------------------------------
// Linear (ggml_tensor overload — dequantizes w/b via to_f32)
// ---------------------------------------------------------------------------

static inline void linear_cpu(const float * in, float * out, int in_dim, int out_dim, const ggml_tensor * w,
                              const ggml_tensor * b) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    linear_cpu(in, out, in_dim, out_dim, wv.data(), bv.empty() ? nullptr : bv.data());
}

// ---------------------------------------------------------------------------
// Conv2d (NCHW layout) with groups, padding, stride
// ---------------------------------------------------------------------------
// Weights: [OC, IC/groups, KH, KW]. groups=1 for standard convolution.
//
// Gather each spatial patch into a contiguous buffer then call dot_product
// (AVX2+FMA / NEON) for each output channel. This keeps the inner loop
// SIMD-friendly without a full im2col allocation (only one patch at a time).
// Boundary check is hoisted above the gather: most interior positions take
// the fast path that avoids per-element range tests.

// A 1x1 convolution is a channel matmul, not a windowed op: the generic
// conv2d_cpu below gathers a "patch" per output pixel, which for kh=kw=1 is a
// pure copy of ch_per_group_in floats repeated H*W times before the dot. This
// takes the axpy form instead -- contiguous in the pixel axis and trivially
// vectorisable.
//
// The pixel axis is blocked into tiles sized so a tile's input slab stays in
// L2, and four output channels are computed at once so each loaded input
// element feeds four FMAs from registers rather than one. Both matter: the
// first version of this path did neither, streaming the WHOLE output plane
// once per (oc, ic) pair -- for a 480x630 plane that is 1.2 MB
// read-modify-written ch_per_group_in times per output channel, so nothing but
// the weights stayed resident, and it was worth only ~6% CPU on the PP-OCRv6
// detector (7.59 vs 8.27/7.93 CPU-seconds median-of-3, 1920x2518 page).
//
// Callers reach this through conv2d_cpu's CRISPEMBED_CONV1X1_FAST gate; it is
// exposed separately so tests can compare it against the generic path inside a
// single process (the gate is a read-once static).
//
// Preconditions: kh == kw == 1, stride == 1, pad == 0.
static inline void conv2d_1x1_cpu(const float * in, float * out, const float * weight, const float * bias, int in_ch,
                                  int out_ch, int H, int W, int groups = 1) {
    const int ch_per_group_in = in_ch / groups;
    const int ch_per_group_out = out_ch / groups;
    const int kernel_size = ch_per_group_in;
    const size_t plane = (size_t)H * W;
    // 8192 floats = 32 KB per channel plane slice. With the ~100-256 input
    // channels these necks use, the resident slab is 3-8 MB, inside the M1's
    // 12 MB shared L2 and far below the 116 MB a full-plane traversal touches.
    constexpr size_t tile = 8192;

    for (int g = 0; g < groups; g++) {
        const int ic_off = g * ch_per_group_in;
        const int oc_off = g * ch_per_group_out;
        const float * w_g = weight + (size_t)oc_off * kernel_size;
        for (size_t p0 = 0; p0 < plane; p0 += tile) {
            const size_t len = std::min(tile, plane - p0);
            int oc = 0;
            for (; oc + 4 <= ch_per_group_out; oc += 4) {
                float * o0 = out + (size_t)(oc_off + oc + 0) * plane + p0;
                float * o1 = out + (size_t)(oc_off + oc + 1) * plane + p0;
                float * o2 = out + (size_t)(oc_off + oc + 2) * plane + p0;
                float * o3 = out + (size_t)(oc_off + oc + 3) * plane + p0;
                const float b0 = bias ? bias[oc_off + oc + 0] : 0.0f;
                const float b1 = bias ? bias[oc_off + oc + 1] : 0.0f;
                const float b2 = bias ? bias[oc_off + oc + 2] : 0.0f;
                const float b3 = bias ? bias[oc_off + oc + 3] : 0.0f;
                for (size_t q = 0; q < len; ++q) {
                    o0[q] = b0;
                    o1[q] = b1;
                    o2[q] = b2;
                    o3[q] = b3;
                }
                const float * w0 = w_g + (size_t)(oc + 0) * kernel_size;
                const float * w1 = w_g + (size_t)(oc + 1) * kernel_size;
                const float * w2 = w_g + (size_t)(oc + 2) * kernel_size;
                const float * w3 = w_g + (size_t)(oc + 3) * kernel_size;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    const float v0 = w0[ic], v1 = w1[ic], v2 = w2[ic], v3 = w3[ic];
                    const float * srow = in + (size_t)(ic_off + ic) * plane + p0;
                    for (size_t q = 0; q < len; ++q) {
                        const float s = srow[q];
                        o0[q] += v0 * s;
                        o1[q] += v1 * s;
                        o2[q] += v2 * s;
                        o3[q] += v3 * s;
                    }
                }
            }
            for (; oc < ch_per_group_out; oc++) {
                float * o = out + (size_t)(oc_off + oc) * plane + p0;
                const float bv = bias ? bias[oc_off + oc] : 0.0f;
                for (size_t q = 0; q < len; ++q) o[q] = bv;
                const float * wrow = w_g + (size_t)oc * kernel_size;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    const float wv = wrow[ic];
                    const float * srow = in + (size_t)(ic_off + ic) * plane + p0;
                    for (size_t q = 0; q < len; ++q) o[q] += wv * srow[q];
                }
            }
        }
    }
}

// Depthwise convolution (groups == in_ch == out_ch), the k>1 counterpart to
// conv2d_1x1_cpu.
//
// The generic path is at its worst here. With one input and one output channel
// per group there is nothing to amortise the patch gather against: it gathers a
// kh*kw window and then consumes it in a single dot_product, per output pixel.
// Measured on the PP-OCRv6 detector via CRISPEMBED_PPOCRV6_DET_PROFILE=1, the
// depthwise layers run at 0.02-0.19 GF/s against ~1.2 GF/s for the pointwise
// ones, and account for 20.4% of all detector convolution time -- the 7x7
// stage at 240x184 alone is the single most expensive layer in the network
// (13.7%).
//
// This inverts the loop nest instead: for each channel and output row, walk the
// kh*kw taps and accumulate a whole output row per tap. Each tap is a contiguous
// axpy over the row, the input row stays in L1 across all taps, the gather
// disappears entirely, and the boundary test moves out of the pixel loop into a
// per-tap column range computed in closed form.
//
// The interior column range for tap kx is the set of ox where
// ox*stride - pad + kx lies in [0, W), which is
// ox in [ceil((pad - kx)/stride), floor((W - 1 - kx + pad)/stride)].
// Computing it up front means the inner loop has no branches at all, which also
// lets it vectorise.
//
// Preconditions: groups == in_ch == out_ch.
static inline void conv2d_depthwise_cpu(const float * in, float * out, const float * weight, const float * bias,
                                        int channels, int H, int W, int kh, int kw, int stride, int pad) {
    const int out_H = (H + 2 * pad - kh) / stride + 1;
    const int out_W = (W + 2 * pad - kw) / stride + 1;
    if (out_H <= 0 || out_W <= 0) return;
    const int taps = kh * kw;

    for (int c = 0; c < channels; c++) {
        const float * src = in + (size_t)c * H * W;
        float * dst = out + (size_t)c * out_H * out_W;
        const float * wc = weight + (size_t)c * taps;
        const float bv = bias ? bias[c] : 0.0f;

        for (int oy = 0; oy < out_H; oy++) {
            float * orow = dst + (size_t)oy * out_W;
            for (int ox = 0; ox < out_W; ox++) orow[ox] = bv;

            const int iy_base = oy * stride - pad;
            for (int ky = 0; ky < kh; ky++) {
                const int iy = iy_base + ky;
                if (iy < 0 || iy >= H) continue;
                const float * irow = src + (size_t)iy * W;
                for (int kx = 0; kx < kw; kx++) {
                    const float wv = wc[ky * kw + kx];
                    if (wv == 0.0f) continue;
                    // Columns where this tap lands inside the input row:
                    //   0 <= ox*stride - pad + kx < W
                    // Both bounds are computed with explicit floor/ceil rather
                    // than C division, which truncates toward zero -- for a
                    // kernel wider than the padded input the numerator goes
                    // negative and (-1)/2 would round UP to 0, admitting an
                    // out-of-range column.
                    const int lo_num = pad - kx;
                    int lo = lo_num <= 0 ? 0 : (lo_num + stride - 1) / stride;
                    const int hi_num = W - 1 - kx + pad;
                    if (hi_num < 0) continue;
                    int hi = hi_num / stride;
                    if (hi > out_W - 1) hi = out_W - 1;
                    if (hi < lo) continue;
                    const int ix0 = lo * stride - pad + kx; // >= 0 by construction of lo
                    const float * s = irow + ix0;
                    if (stride == 1) {
                        for (int ox = lo; ox <= hi; ++ox) orow[ox] += wv * s[ox - lo];
                    } else {
                        for (int ox = lo; ox <= hi; ++ox) orow[ox] += wv * s[(ox - lo) * stride];
                    }
                }
            }
        }
    }
}

// im2col-tile variant of the generic path below (R6). Two changes, both
// output-preserving:
//
// 1. Loop interchange over a tile of output positions. The generic path
//    computes every output channel for one position before moving on, which
//    streams the ENTIRE weight matrix [ch_per_group_out, K] once per output
//    pixel -- for a 256ch 3x3 conv that is 2.3 MB re-read thousands of times,
//    so on any core whose L2 doesn't hold the weights the kernel is
//    memory-bound on weight traffic. Here a tile of output positions is
//    gathered into a [tile, K] column buffer sized to stay cache-resident,
//    and the oc loop runs OUTSIDE the position loop: each weight row is
//    loaded once per tile instead of once per pixel.
//
// 2. Fork-join multithreading over tiles. Threads own disjoint position
//    ranges (and their own column buffers), so every output element is still
//    computed exactly once by exactly the same arithmetic.
//
// Each output element remains `bias + dot_product(patch, w_row, K)` on a
// patch gathered in the same [ic, ky, kx] order as the generic path, so the
// result is BITWISE IDENTICAL to it at any thread count -- deliberately: a
// register-blocked GEMM micro-kernel would change the accumulation order and
// turn every engine A/B into a quality argument instead of a byte compare.
// That further step stays open until this one's win is banked.
//
// Callers reach this through conv2d_cpu's CRISPEMBED_CONV2D_GEMM gate
// (threads via CRISPEMBED_CONV2D_THREADS); exposed separately so tests can
// compare both paths in one process (the gates are read-once statics).
static inline void conv2d_im2col_cpu(const float * in, float * out, const float * weight, const float * bias, int in_ch,
                                     int out_ch, int H, int W, int kh, int kw, int stride, int pad, int groups = 1,
                                     int n_threads = 1) {
    const int out_H = (H + 2 * pad - kh) / stride + 1;
    const int out_W = (W + 2 * pad - kw) / stride + 1;
    if (out_H <= 0 || out_W <= 0) return;
    const int ch_per_group_in = in_ch / groups;
    const int ch_per_group_out = out_ch / groups;
    const int K = ch_per_group_in * kh * kw;
    const size_t n_pos = (size_t)out_H * out_W;

    // Tile length: keep the [tile, K] column buffer around 512 KB so it stays
    // L2-resident while the oc loop streams over it (M1 shares 12 MB of L2;
    // x86 boxes have 0.5-1 MB private L2 + a large L3 behind it).
    const int tile =
        (int)std::clamp((size_t)(512 * 1024 / sizeof(float)) / (size_t)std::max(K, 1), (size_t)16, (size_t)256);
    const size_t n_tiles = (n_pos + tile - 1) / tile;

    auto run_tiles = [&](int g, size_t t0, size_t t1, float * col) {
        const int ic_off = g * ch_per_group_in;
        const int oc_off = g * ch_per_group_out;
        const float * w_g = weight + (size_t)oc_off * K;
        for (size_t t = t0; t < t1; t++) {
            const size_t p0 = t * (size_t)tile;
            const int len = (int)std::min((size_t)tile, n_pos - p0);

            // Gather: same per-position patch layout and boundary handling as
            // the generic path, written into the column buffer instead of a
            // single reused patch.
            for (int i = 0; i < len; i++) {
                const size_t p = p0 + (size_t)i;
                const int oy = (int)(p / out_W);
                const int ox = (int)(p % out_W);
                const int top = oy * stride - pad;
                const int left = ox * stride - pad;
                float * dst = col + (size_t)i * K;
                const bool full = (top >= 0 && top + kh <= H && left >= 0 && left + kw <= W);
                if (full) {
                    int k = 0;
                    for (int ic = 0; ic < ch_per_group_in; ic++) {
                        const float * src = in + ((size_t)(ic_off + ic) * H + top) * W + left;
                        for (int ky = 0; ky < kh; ky++, k += kw)
                            memcpy(dst + k, src + (size_t)ky * W, kw * sizeof(float));
                    }
                } else {
                    int k = 0;
                    for (int ic = 0; ic < ch_per_group_in; ic++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                const int iy = top + ky, ix = left + kx;
                                dst[k++] = (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                               ? in[((size_t)(ic_off + ic) * H + iy) * W + ix]
                                               : 0.0f;
                            }
                        }
                    }
                }
            }

            // Consume: oc outside the position loop -- each weight row is
            // read once per tile and the streamed operand is the L2-resident
            // column buffer.
            for (int oc = 0; oc < ch_per_group_out; oc++) {
                const float b = bias ? bias[oc_off + oc] : 0.0f;
                const float * wrow = w_g + (size_t)oc * K;
                float * orow = out + (size_t)(oc_off + oc) * n_pos + p0;
                for (int i = 0; i < len; i++) orow[i] = b + dot_product(col + (size_t)i * K, wrow, K);
            }
        }
    };

    const int nt = (int)std::min((size_t)std::max(n_threads, 1), n_tiles);
    for (int g = 0; g < groups; g++) {
        if (nt <= 1) {
            std::vector<float> col((size_t)tile * K);
            run_tiles(g, 0, n_tiles, col.data());
            continue;
        }
        std::vector<std::thread> pool;
        std::vector<std::vector<float>> cols(nt, std::vector<float>((size_t)tile * K));
        const size_t chunk = (n_tiles + nt - 1) / nt;
        for (int th = 0; th < nt; th++) {
            const size_t t0 = (size_t)th * chunk, t1 = std::min(n_tiles, t0 + chunk);
            if (t0 < t1) pool.emplace_back(run_tiles, g, t0, t1, cols[th].data());
        }
        for (auto & thr : pool) thr.join();
    }
}

static inline void conv2d_cpu(const float * in, float * out, const float * weight, const float * bias, int in_ch,
                              int out_ch, int H, int W, int kh, int kw, int stride, int pad, int groups = 1) {
    int out_H = (H + 2 * pad - kh) / stride + 1;
    int out_W = (W + 2 * pad - kw) / stride + 1;
    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;
    int kernel_size = ch_per_group_in * kh * kw;

    // Opt-in until A/B'd per engine, since this helper is shared by 15 files
    // and the generic path is better than it looks -- it reuses one gathered
    // patch across every output channel in the group. Wall-clock A/B on this
    // box was useless (contradictory 1.8x-slower / 2x-faster readings at load
    // 40-52); only CPU time was stable. See conv2d_1x1_cpu above.
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0) {
        static const bool fast_1x1 = std::getenv("CRISPEMBED_CONV1X1_FAST") != nullptr;
        if (fast_1x1) {
            conv2d_1x1_cpu(in, out, weight, bias, in_ch, out_ch, out_H, out_W, groups);
            return;
        }
    }

    // Same disposition as the 1x1 gate above: opt-in until A/B'd per engine.
    // See conv2d_depthwise_cpu for why this shape is the generic path's worst
    // case. CRISPEMBED_CONVDW_FAST=1.
    if (groups > 1 && groups == in_ch && groups == out_ch) {
        static const bool fast_dw = std::getenv("CRISPEMBED_CONVDW_FAST") != nullptr;
        if (fast_dw) {
            conv2d_depthwise_cpu(in, out, weight, bias, in_ch, H, W, kh, kw, stride, pad);
            return;
        }
    }

    // R6: opt-in im2col-tile path, bitwise-identical to the loop below (see
    // conv2d_im2col_cpu). CRISPEMBED_CONV2D_GEMM=1 enables it;
    // CRISPEMBED_CONV2D_THREADS=N (default 1) adds fork-join threading so the
    // two levers can be A/B'd separately. Ordered after the shape-specific
    // gates above so their opt-ins keep precedence.
    {
        static const bool im2col = core_env::on("CRISPEMBED_CONV2D_GEMM");
        if (im2col) {
            static const int nt = [] {
                const char * e = std::getenv("CRISPEMBED_CONV2D_THREADS");
                const int v = e ? atoi(e) : 1;
                return v < 1 ? 1 : v;
            }();
            conv2d_im2col_cpu(in, out, weight, bias, in_ch, out_ch, H, W, kh, kw, stride, pad, groups, nt);
            return;
        }
    }

    static thread_local std::vector<float> patch_buf;
    if ((int)patch_buf.size() < kernel_size) patch_buf.resize(kernel_size);
    float * patch = patch_buf.data();

    for (int g = 0; g < groups; g++) {
        int ic_off = g * ch_per_group_in;
        int oc_off = g * ch_per_group_out;
        const float * w_g = weight + oc_off * kernel_size;

        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                // Gather [ch_per_group_in, kh, kw] input patch into patch[].
                // Hoist boundary check: if the entire kernel window fits inside
                // the input, skip per-element if-guards.
                int top = oy * stride - pad;
                int left = ox * stride - pad;
                bool full = (top >= 0 && top + kh <= H && left >= 0 && left + kw <= W);
                int k = 0;
                if (full) {
                    for (int ic = 0; ic < ch_per_group_in; ic++) {
                        const float * src = in + (ic_off + ic) * H * W + top * W + left;
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++) patch[k++] = src[ky * W + kx];
                    }
                } else {
                    for (int ic = 0; ic < ch_per_group_in; ic++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = top + ky, ix = left + kx;
                                patch[k++] = (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                                 ? in[(ic_off + ic) * H * W + iy * W + ix]
                                                 : 0.0f;
                            }
                        }
                    }
                }

                // SIMD dot product with each output-channel filter row.
                int p = oy * out_W + ox;
                for (int oc = 0; oc < ch_per_group_out; oc++) {
                    float b = bias ? bias[oc_off + oc] : 0.0f;
                    out[(oc_off + oc) * out_H * out_W + p] =
                        b + dot_product(patch, w_g + oc * kernel_size, kernel_size);
                }
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
static inline void silu_inplace(float * data, int n) {
    for (int i = 0; i < n; i++) data[i] = data[i] / (1.0f + expf(-data[i]));
}

// In-place softmax (SIMD-accelerated max, normalize)
static inline void softmax(float * data, int n) {
    // Max (SIMD-accelerated)
    float mx = data[0];
#if defined(__AVX2__)
    if (n >= 8) {
        __m256 vmx = _mm256_loadu_ps(data);
        int i = 8;
        for (; i + 7 < n; i += 8) vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(data + i));
        __m128 lo = _mm256_castps256_ps128(vmx);
        __m128 hi = _mm256_extractf128_ps(vmx, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        mx = _mm_cvtss_f32(lo);
        for (; i < n; i++)
            if (data[i] > mx) mx = data[i];
    } else {
        for (int i = 1; i < n; i++)
            if (data[i] > mx) mx = data[i];
    }
#else
    for (int i = 1; i < n; i++)
        if (data[i] > mx) mx = data[i];
#endif
    // Exp + sum (expf is not SIMD-friendly, keep scalar)
    float sum = 0;
    for (int i = 0; i < n; i++) {
        data[i] = expf(data[i] - mx);
        sum += data[i];
    }
    // Normalize (SIMD-accelerated)
    float inv_sum = 1.0f / sum;
#if defined(__AVX2__)
    {
        __m256 vs = _mm256_set1_ps(inv_sum);
        int i = 0;
        for (; i + 7 < n; i += 8) _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
        for (; i < n; i++) data[i] *= inv_sum;
    }
#else
    for (int i = 0; i < n; i++) data[i] *= inv_sum;
#endif
}

// HardSwish: x * min(max(x+3, 0), 6) / 6
static inline void hardswish_inplace(float * data, int n) {
    for (int i = 0; i < n; i++) {
        float x = data[i];
        if (x <= -3.0f)
            data[i] = 0.0f;
        else if (x >= 3.0f) { /* keep x */
        } else
            data[i] = x * (x + 3.0f) / 6.0f;
    }
}

// ReLU6: clamp to [0, 6]
static inline void relu6_inplace(float * data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] < 0.0f)
            data[i] = 0.0f;
        else if (data[i] > 6.0f)
            data[i] = 6.0f;
    }
}

// ReLU: max(0, x)
static inline void relu_inplace(float * data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0.0f) data[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Pooling helpers (NCHW layout, explicit output dims for ceil_mode compat)
// ---------------------------------------------------------------------------

// MaxPool2d: sliding window max, out[c,oy,ox] = max of window starting at (oy*s, ox*s).
// Caller pre-computes oh/ow (handles ceil_mode by expanding them before calling).
static inline void maxpool2d_cpu(const float * in, int ch, int ih, int iw, int k, int s, float * out, int oh, int ow) {
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
static inline void avgpool2d_cpu(const float * in, int ch, int ih, int iw, int k, int s, float * out, int oh, int ow) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float sum = 0;
                int cnt = 0;
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

static inline void apply_bn_cpu(float * data, int ch, int sp, const float * scale, const float * offset) {
    for (int c = 0; c < ch; c++) {
        float s = scale[c], o = offset[c];
        float * row = data + c * sp;
        for (int i = 0; i < sp; i++) row[i] = row[i] * s + o;
    }
}

// Multi-head attention (single query position)
// q: [D], k: [n_kv, D], v: [n_kv, D], out: [D]
// scores_buf: optional pre-allocated buffer [>=n_kv] to avoid per-head heap alloc.
//             If nullptr, uses a thread-local buffer (safe, no per-call alloc).
static inline void mha_1q_cpu(const float * q, const float * k, const float * v, float * out, int n_kv, int D,
                              int n_heads, float * scores_buf = nullptr) {
    int hd = D / n_heads;
    // Write directly to out (avoids separate result vector + memcpy)
    memset(out, 0, D * sizeof(float));

    // Scores buffer: use external if provided, else thread-local (no per-call alloc)
    static thread_local std::vector<float> tl_scores;
    float * scores;
    if (scores_buf) {
        scores = scores_buf;
    } else {
        if ((int)tl_scores.size() < n_kv) tl_scores.resize(n_kv);
        scores = tl_scores.data();
    }

    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        // Q·K scores (SIMD via dot_product)
        float maxs = -1e30f;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = dot_product(q + off, k + ki * D + off, hd) * scale;
            if (scores[ki] > maxs) maxs = scores[ki];
        }
        // Softmax
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_kv; ki++) scores[ki] *= inv_sum;
        // Weighted V accumulation: out[off:off+hd] = sum(scores[ki] * V[ki][off:off+hd])
        float * dst = out + off;
        for (int ki = 0; ki < n_kv; ki++) {
            float s = scores[ki];
            const float * vrow = v + ki * D + off;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 vs = _mm256_set1_ps(s);
            int d = 0;
            for (; d + 7 < hd; d += 8)
                _mm256_storeu_ps(dst + d, _mm256_fmadd_ps(vs, _mm256_loadu_ps(vrow + d), _mm256_loadu_ps(dst + d)));
            for (; d < hd; d++) dst[d] += s * vrow[d];
#elif defined(__aarch64__)
            float32x4_t vs = vdupq_n_f32(s);
            int d = 0;
            for (; d + 3 < hd; d += 4) vst1q_f32(dst + d, vfmaq_f32(vld1q_f32(dst + d), vs, vld1q_f32(vrow + d)));
            for (; d < hd; d++) dst[d] += s * vrow[d];
#else
            for (int d = 0; d < hd; d++) dst[d] += s * vrow[d];
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Otsu threshold — interclass variance maximization
// ---------------------------------------------------------------------------
// Returns the optimal uint8 threshold for binarizing a grayscale image.
// Shared across table_parse, cc_detect, classical_preproc, dewarp.

static inline uint8_t otsu_threshold(const uint8_t * gray, int n) {
    int hist[256] = {};
    for (int i = 0; i < n; i++) hist[gray[i]]++;
    double sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];
    double sumB = 0;
    int wB = 0;
    double max_var = 0;
    int best_t = 128;
    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = n - wB;
        if (wF == 0) break;
        sumB += t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double var = (double)wB * wF * (mB - mF) * (mB - mF);
        if (var > max_var) {
            max_var = var;
            best_t = t;
        }
    }
    return (uint8_t)best_t;
}

} // namespace core_cpu
