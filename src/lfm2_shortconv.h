#pragma once

// lfm2_shortconv.h — LFM2 ShortConv single-token decode step, weight-free.
//
// Extracted out of src/lfm2_vl_ocr.cpp so it can be unit-tested hermetically
// against a scalar reference (tests/test_lfm2_shortconv.cpp). The prefill path
// runs the same math through ggml_conv_1d_dw over a left-padded sequence; the
// decode path has to reproduce it from a K-1 column state cache, and the two
// must agree exactly or the KV-cached decode diverges from full recompute.

#include "ggml.h"

#include <cstddef>

namespace lfm2_shortconv {

// Depthwise causal conv1d for a SINGLE decode step.
//
//   window : [D, K] F32 — the K most recent Bx columns, oldest in column 0,
//                         the current token in column K-1.
//   kern   : [K, D] F32 — conv weight; kern(k, d) is tap k of channel d.
//   returns  [D]    F32 — out[d] = sum_k window(d, k) * kern(k, d)
//
// This is the cross-correlation ggml_conv_1d_dw computes (im2col + mul_mat,
// no kernel flip): with the sequence left-padded by K-1,
//   out[t] = sum_k w[k] * Bx[t + k - (K-1)]
// so window column k holds Bx[t + k - (K-1)] and tap k multiplies it.
inline ggml_tensor * step(ggml_context * g, ggml_tensor * window, ggml_tensor * kern, int D, int K) {
    // Bring the kernel into the window's [D, K] layout rather than the window
    // into the kernel's [K, D] layout. Both products have the same shape, so
    // this choice is invisible until the reduction: the tap axis must be the
    // SLOW axis (ne[1]) so that each tap's D channel values sit in one
    // contiguous run and ggml_view_1d can address them.
    //
    // Reducing a [K, D] product with ggml_view_1d instead sums flat indices
    // {i, i+D, i+2D}, which straddles taps and channels (flat index there is
    // d*K + k) and silently returns a scrambled vector of the right shape.
    ggml_tensor * kern_t = ggml_cont(g, ggml_transpose(g, kern)); // [D, K]
    ggml_tensor * prod = ggml_mul(g, window, kern_t);             // [D, K]

    // Reduce over the tap axis: prod is contiguous with flat index k * D + d,
    // so tap k occupies [k * D, (k + 1) * D).
    ggml_tensor * acc = ggml_view_1d(g, prod, D, 0);
    for (int k = 1; k < K; k++) {
        acc = ggml_add(g, acc, ggml_view_1d(g, prod, D, (size_t)k * D * sizeof(float)));
    }
    return acc;
}

} // namespace lfm2_shortconv
