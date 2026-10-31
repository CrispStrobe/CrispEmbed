#pragma once
#include "core/gguf_loader.h"
#include <vector>
namespace qwen35_ocr {
struct vision_layer_weights {
    ggml_tensor * norm1 = nullptr;
    ggml_tensor * norm1_bias = nullptr;
    ggml_tensor * norm2 = nullptr;
    ggml_tensor * norm2_bias = nullptr;
    ggml_tensor * qkv = nullptr;
    ggml_tensor * qkv_bias = nullptr;
    ggml_tensor * out = nullptr;
    ggml_tensor * out_bias = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * down_bias = nullptr;
};
struct vision_weights {
    ggml_tensor * patch = nullptr;
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position = nullptr;
    ggml_tensor * merger_norm = nullptr;
    ggml_tensor * merger_norm_bias = nullptr;
    ggml_tensor * merger_up = nullptr;
    ggml_tensor * merger_up_bias = nullptr;
    ggml_tensor * merger_down = nullptr;
    ggml_tensor * merger_down_bias = nullptr;
    std::vector<vision_layer_weights> layers;
};
struct vision_config {
    int width = 768, heads = 12, intermediate = 3072, output = 1024;
    int patch_width = 1536, merge = 2, position_side = 48;
    // Unmerged patches this call may run. Default preserves the ordinary bound;
    // a validated profile may pass its own frontend limit instead.
    int max_patches = 1024;
    float eps = 1e-6f, theta = 10000.0f;
};
struct vision_layer_outputs {
    std::vector<float> norm1, qkv, pooled, attention, norm2, up, activation, down;
};
struct vision_outputs {
    std::vector<float> patch, position, embedded, rotary_cos, rotary_sin;
    std::vector<std::vector<float>> hidden;
    // Sensitive trained blocks 5/11, for earliest-stage numerical probes.
    std::vector<vision_layer_outputs> probes;
    std::vector<float> merger_norm, merger_up, merger_activation, merged;
};
bool map_small_vision(const core_gguf::tensor_map & tensors, vision_weights & weights);
// Packed patches [sum(t*h*w),C*T*P*P]; grid rows [t,h,w]. Attention is
// non-causal per frame, never across images or frames. CPU reference path.
bool vision_cpu(const vision_weights & weights, const vision_config & config, const float * patches, int tokens,
                const std::vector<int> & grid_thw, int threads, vision_outputs & output);
} // namespace qwen35_ocr
