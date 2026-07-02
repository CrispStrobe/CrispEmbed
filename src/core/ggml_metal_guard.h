// src/core/ggml_metal_guard.h — shared ggml-graph safety helpers for Metal.
//
// Header-only. All functions live in namespace core_ggml with static inline
// linkage to avoid ODR violations when included from multiple TUs. Unlike
// core/vlm_attention.h (scalar CPU building blocks), these operate on the ggml
// *graph* (ggml_context* / ggml_tensor*) and codify two Metal-specific pitfalls
// that have each caused silent, hard-to-debug output corruption in this repo:
//
//   1. Metal's batched simdgroup GEMM (mul_mm) casts activations to F16.
//      "Massive activations" overflow F16 (>65504) → NaN cascade. C7 guard:
//      mul_mat_f16_guarded() / metal_mul_mm_f16_cast_active().
//      (memory: metal-mul-mm-f16-overflow)
//
//   2. ggml_flash_attn_ext already returns [head_dim, n_heads, T] (it permutes
//      internally); a *trailing* permute(0,2,1,3) on its result double-permutes
//      and scrambles output. C6 guard: assert_fa_layout().
//      (memory: flashattn-ext-already-permutes)
//
// Usage:
//   #include "core/ggml_metal_guard.h"
//   using core_ggml::mul_mat_f16_guarded;
//   using core_ggml::assert_fa_layout;

#pragma once

#include "ggml.h"

namespace core_ggml {

// ---------------------------------------------------------------------------
// C7 — Metal mul_mm F16 activation-overflow guard
// ---------------------------------------------------------------------------
//
// Diagnostic rule (symptom → cause):
//   NaN / garbage output with MANY tokens (prefill, T>8) but CLEAN single-token
//   decode (T=1)  ⇒  you are on the Metal mul_mm path and an activation
//   overflowed the F16 cast. set_prec(F32) does NOT help — Metal picks mul_mm
//   purely by shape and ignores the precision hint for the GEMM.
//
// Threshold: ggml-metal-ops.cpp selects the mul_mm (F16-casting simdgroup GEMM)
// kernel over mul_mv when
//     has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min   (ne11_mm_min = 8)
// i.e. when the RHS operand's ne[1] (the token/batch count, "T") exceeds 8.
// T=1 decode and short (<=8) batches use mul_mv, which stays F32.
static constexpr int64_t METAL_MUL_MM_NE11_MIN = 8;

// True when Metal will (shape permitting) route ggml_mul_mat through the
// F16-casting mul_mm kernel for a right-hand operand with the given token count.
// ne00 (the contracted / inner dim) must also be >= 64 for mul_mm; callers that
// know ne00 can pass it to avoid a false positive on small inner dims.
static inline bool metal_mul_mm_f16_cast_active(int64_t ne11, int64_t ne00 = 64) {
    return ne11 > METAL_MUL_MM_NE11_MIN && ne00 >= 64;
}

// Guarded matmul: ggml_mul_mat(g, w, act) with a lossless exponent shift that
// keeps the F16 mul_mm cast in range for activations that can exceed F16.
//
// When the token count `n_tokens` (== act's ne[1]) is on the mul_mm path
// (> METAL_MUL_MM_NE11_MIN), scale the activation down by `guard` before the
// matmul and back up by `guard` after — an exact power-of-two exponent shift
// that changes no mantissa bits (guard should be a power of two, default 256).
// On the mul_mv / T<=8 / CPU path (F32, no overflow) the scales are skipped so
// no extra kernels are dispatched.
//
// This is the reusable form of the granite_vision_ocr SwiGLU down-projection
// fix. Use it for any matmul whose activation input can carry massive values
// (image-feature-amplified residuals, embedding-multiplier paths, etc).
static inline ggml_tensor * mul_mat_f16_guarded(ggml_context * g, ggml_tensor * w, ggml_tensor * act, int64_t n_tokens,
                                                float guard = 256.0f) {
    if (metal_mul_mm_f16_cast_active(n_tokens)) {
        return ggml_scale(g, ggml_mul_mat(g, w, ggml_scale(g, act, 1.0f / guard)), guard);
    }
    return ggml_mul_mat(g, w, act);
}

// ---------------------------------------------------------------------------
// C6 — flash_attn_ext output-layout guard
// ---------------------------------------------------------------------------
//
// ggml_flash_attn_ext(g, Q, K, V, mask, scale, ...) returns a tensor already
// in [head_dim, n_heads, T (, batch)] layout — it does the head/token permute
// internally. The CORRECT epilogue is a DIRECT reshape to [head_dim*n_heads, T]
// (ggml_reshape_2d/3d). A trailing ggml_permute(attn, 0, 2, 1, 3) on the RESULT
// is a double-permute bug: it swaps ne[1]/ne[2] so the reshape then interleaves
// heads and tokens → scrambled output (silent; no crash).
//
// Call this IMMEDIATELY after ggml_flash_attn_ext, before the reshape:
//
//   attn = core_ggml::assert_fa_layout(
//              ggml_flash_attn_ext(g, Q, K, V, mask, scale, 0.0f, 0.0f),
//              head_dim, n_heads);
//   attn = ggml_reshape_2d(g, attn, head_dim * n_heads, T);
//
// It only validates the layout invariant that a spurious permute violates
// (ne[0]==head_dim && ne[1]==n_heads) and returns the tensor unchanged, so it
// composes with any downstream reshape (2D, 3D, or batched [.,.,T,B]). The
// shape is a pure function of the graph, never of input data, so the assert can
// only fire on a programming error (a reintroduced permute) — which is exactly
// the point: it craters the model's diff test at graph-build time instead of
// letting the regression ship silently.
static inline ggml_tensor * assert_fa_layout(ggml_tensor * attn, int64_t head_dim, int64_t n_heads) {
    GGML_ASSERT(attn->ne[0] == head_dim && attn->ne[1] == n_heads &&
                "flash_attn_ext output must be [head_dim, n_heads, ...]; a "
                "trailing permute(0,2,1,3) on the result is a double-permute "
                "bug (see memory flashattn-ext-already-permutes)");
    return attn;
}

} // namespace core_ggml
