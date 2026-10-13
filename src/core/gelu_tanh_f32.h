#pragma once
#include "ggml.h"
#include <algorithm>
#include <cmath>

namespace core_cpu {
// Match gelu_pytorch_tanh without ggml's CPU FP16 input/output LUT.
static inline void gelu_tanh_f32_compute(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * src = dst->src[0];
    const int64_t n = ggml_nelements(src), chunk = (n + nth - 1) / nth;
    const auto * x = static_cast<const float *>(src->data);
    auto * y = static_cast<float *>(dst->data);
    for (int64_t i = ith * chunk; i < std::min(n, (ith + 1) * chunk); ++i) {
        const float v = x[i];
        y[i] = 0.5f * v * (1 + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}
static inline ggml_tensor * gelu_tanh_f32(ggml_context * g, ggml_tensor * x) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    ggml_tensor * args[] = { x };
    return ggml_custom_4d(g, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], args, 1, gelu_tanh_f32_compute,
                          GGML_N_TASKS_MAX, nullptr);
}
} // namespace core_cpu
