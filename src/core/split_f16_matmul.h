// split_f16_matmul.h — near-F32 matrix products from half-precision GPU kernels.
//
// Why: ggml-metal's fast mul_mm kernels stage BOTH operands as half in
// threadgroup memory (F32 accumulation). The fork's GGML_PREC_F32 `_hp` kernel
// keeps F32 tiles but measured ~90 GFLOP/s on an Apple M1 against ~1700 GFLOP/s
// for the half kernel (tools/metal: K=1152 M=14400 N=128). This helper emulates
// an F32 product with half-staged kernels by splitting each F32 operand into a
// high and a low half part (x = hi(x) + lo(x), both representable in F16):
//
//     a.b ~= hi(a).hi(b) + hi(a).lo(b) + lo(a).hi(b)        (lo.lo dropped)
//
// Each term runs on the ordinary half kernel with F32 accumulation, so the
// result carries ~22 mantissa bits instead of 11. An F16 operand has no low
// part (its term drops out). Limits: values beyond the F16 range (|x| > 65504)
// overflow, and the low part loses precision in the F16 subnormal range, so
// every use is env-gated, default off, and A/B'd against ground truth.
//
// mul_mat(a, b) semantics are unchanged: contracts ne[0], broadcasting as
// ggml_mul_mat. The products are created WITHOUT GGML_PREC_F32 (which would
// route them back to the slow _hp kernel).
#pragma once

#include "ggml.h"

namespace core_split_f16 {

inline ggml_tensor * contiguous(ggml_context * g, ggml_tensor * t) {
    return ggml_is_contiguous(t) ? t : ggml_cont(g, t);
}

// Returns {hi as F16, lo as F16 or nullptr}.
inline void split(ggml_context * g, ggml_tensor * t, ggml_tensor ** hi, ggml_tensor ** lo) {
    if (t->type == GGML_TYPE_F16) {
        *hi = t;
        *lo = nullptr;
        return;
    }
    t = contiguous(g, t->type == GGML_TYPE_F32 ? t : ggml_cast(g, t, GGML_TYPE_F32));
    *hi = ggml_cast(g, t, GGML_TYPE_F16);
    *lo = ggml_cast(g, ggml_sub(g, t, ggml_cast(g, *hi, GGML_TYPE_F32)), GGML_TYPE_F16);
}

inline ggml_tensor * mul_mat(ggml_context * g, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor *ah, *al;
    split(g, a, &ah, &al);
    // src1 is passed as F32 tensors holding hi(b) and lo(b) explicitly, so the
    // result is the same whether the backend stages src1 as half (mul_mm) or
    // reads it as float (small-batch mul_mv).
    b = contiguous(g, b->type == GGML_TYPE_F32 ? b : ggml_cast(g, b, GGML_TYPE_F32));
    ggml_tensor * b_hi = ggml_cast(g, ggml_cast(g, b, GGML_TYPE_F16), GGML_TYPE_F32);
    ggml_tensor * b_lo = ggml_sub(g, b, b_hi);
    ggml_tensor * out = ggml_add(g, ggml_mul_mat(g, ah, b_hi), ggml_mul_mat(g, ah, b_lo));
    if (al) out = ggml_add(g, out, ggml_mul_mat(g, al, b_hi));
    return out;
}

} // namespace core_split_f16
