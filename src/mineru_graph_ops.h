// SPDX-License-Identifier: Apache-2.0
// MinerU/Qwen2-VL QuickGELU arithmetic. This helper is independent of OXR.
#pragma once
#include "ggml.h"

namespace mineru_graph_ops {
// The pinned GGML CPU quick-GELU operator uses an F16 lookup table even
// with F32 tensors. Keep the reference x*sigmoid(1.702*x) in F32 throughout.
inline ggml_tensor * quick_gelu_f32(ggml_context * ctx, ggml_tensor * x) {
    return ggml_mul(ctx, x, ggml_sigmoid(ctx, ggml_scale(ctx, x, 1.702f)));
}
} // namespace mineru_graph_ops
