#pragma once
#include "ggml.h"

namespace oxr_graph_ops {
// GGML's generic CPU GELU uses an F16 lookup table even for F32 tensors in
// the pinned backend. Preserve PyTorch's gelu_pytorch_tanh arithmetic with
// explicit F32 graph operations; do not introduce hidden activation rounding.
inline ggml_tensor * gelu_tanh_f32(ggml_context * ctx, ggml_tensor * x) {
    auto * cube = ggml_mul(ctx, ggml_mul(ctx, x, x), x);
    auto * inner = ggml_add(ctx, x, ggml_scale(ctx, cube, 0.044715f));
    inner = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.7978845608028654f));
    return ggml_mul(ctx, ggml_scale(ctx, x, 0.5f), ggml_scale_bias(ctx, inner, 1.0f, 1.0f));
}
} // namespace oxr_graph_ops
