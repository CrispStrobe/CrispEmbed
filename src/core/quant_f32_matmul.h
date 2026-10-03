#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace core_cpu {

// CPU accuracy path: dequantize one weight row per worker, retaining F32
// activations. Unlike casting a complete matrix, scratch is O(columns*threads).
static inline void quant_f32_matmul_compute(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t columns = w->ne[0];
    const auto dequant = ggml_get_type_traits(w->type)->to_float;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
    std::vector<float> row(columns);
    const int64_t chunk = (w->ne[1] + nth - 1) / nth;
    for (int64_t r = ith * chunk; r < std::min(w->ne[1], (ith + 1) * chunk); ++r) {
        dequant(static_cast<const uint8_t *>(w->data) + r * w->nb[1], row.data(), columns);
        for (int64_t token = 0; token < x->ne[1]; ++token) {
            float result;
            dot(columns, &result, 0, row.data(), 0, static_cast<const uint8_t *>(x->data) + token * x->nb[1], 0, 1);
            *reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + r * dst->nb[0] + token * dst->nb[1]) =
                result;
        }
    }
}

static inline ggml_tensor * quant_f32_matmul(ggml_context * g, ggml_tensor * w, ggml_tensor * x) {
    GGML_ASSERT(ggml_is_quantized(w->type) && ggml_get_type_traits(w->type)->to_float);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && w->ne[0] == x->ne[0]);
    GGML_ASSERT(w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_is_contiguous(x));
    ggml_tensor * args[] = { w, x };
    return ggml_custom_4d(g, GGML_TYPE_F32, w->ne[1], x->ne[1], 1, 1, args, 2, quant_f32_matmul_compute,
                          GGML_N_TASKS_MAX, nullptr);
}

} // namespace core_cpu
