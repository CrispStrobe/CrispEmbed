#pragma once
#include <vector>
struct ggml_tensor;
namespace qwen35_ocr {
struct attention_weights {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * out;
    ggml_tensor * q_norm;
    ggml_tensor * k_norm;
};
struct attention_config {
    int query_heads = 8, kv_heads = 2, head_dim = 256, rotary_dim = 64;
    int mrope_section[3] = { 11, 11, 10 };
    float theta = 10000000.0f, eps = 1e-6f;
};
// Cache uses official [batch, kv_heads, sequence, head_dim] layout.
struct attention_cache {
    int batch = 0, heads = 0, tokens = 0, width = 0;
    std::vector<float> keys, values;
};
struct attention_outputs {
    std::vector<float> q, k, v, q_norm, k_norm, pooled, attention;
    attention_cache cache;
};
// Unpadded causal reference path. Positions are [3,batch,tokens] multimodal
// coordinates, independent of causal sequence chronology. Cache is immutable.
bool attention_cpu(const attention_weights & weights, const attention_config & config, const float * hidden,
                   const float * positions, int batch, int tokens, int threads, const attention_cache * previous,
                   attention_outputs & output);
} // namespace qwen35_ocr
