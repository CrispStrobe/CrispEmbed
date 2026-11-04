#pragma once

#include "qwen35_delta.h"

struct ggml_tensor;

namespace qwen35_ocr {
class converted_weight_cache;
// Actual checkpoint weights for one linear-attention block. The legacy entry
// admits equal K/V heads; the separate grouped entry takes an explicit K count.
struct linear_weights {
    ggml_tensor * qkv;
    ggml_tensor * z;
    ggml_tensor * a;
    ggml_tensor * b;
    ggml_tensor * out;
    ggml_tensor * conv;
    ggml_tensor * a_log;
    ggml_tensor * dt_bias;
    ggml_tensor * norm;
};
struct projected_linear_outputs {
    std::vector<float> qkv, z, a, b, attention;
    linear_outputs core;
};
// Reference CPU path: materialize checkpoint matrices as F32 to avoid FP16
// activation-dot rounding. Quantized matrices require the separate quant gate;
// this establishes storage/quality behavior, not an optimized quantized matmul.
bool linear_cpu(const linear_weights & weights, const float * hidden, int batch, int tokens, int threads,
                const float * conv_history, const float * state, projected_linear_outputs & output,
                converted_weight_cache * weights_cache = nullptr);
// Default-off component path for compact grouped Q/K with the same K/V width.
// Requires QWEN35_GDN_GROUPED_HEADS and QWEN35_OCR; F32/F16 weights only.
// Value heads/width are read from A_log/norm, key_heads is explicit. Cache
// layouts follow linear_grouped_core_cpu. Equal heads delegate unchanged.
bool linear_grouped_cpu(const linear_weights & weights, int key_heads, const float * hidden, int batch, int tokens,
                        int threads, const float * conv_history, const float * state, projected_linear_outputs & output,
                        converted_weight_cache * weights_cache = nullptr);
} // namespace qwen35_ocr
