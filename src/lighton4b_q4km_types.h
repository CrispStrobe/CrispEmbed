#pragma once
#include "qwen35_decoder.h"
#include "lighton4b_q6_policy.h"
namespace qwen35_ocr {
// Pure type descriptor only: callers must independently gate and verify their
// coordinate/provenance profile. This never admits a file or enables inference.
inline bool lighton4b_q4km_types_match(const decoder_weights & decoder, ggml_type head_type) {
    if ((head_type != GGML_TYPE_Q6_K && head_type != GGML_TYPE_Q8_0) || decoder.layers.size() != 32 ||
        !decoder.embedding || decoder.head != decoder.embedding || decoder.embedding->type != head_type)
        return false;
    auto type = [](const ggml_tensor * t, ggml_type expected) { return t && t->type == expected; };
    for (int i = 0; i < 32; ++i) {
        const auto & layer = decoder.layers[size_t(i)];
        const auto selected = lighton4b_q6_layer(i) ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
        if (layer.full_attention != (i % 4 == 3) || !type(layer.gate, GGML_TYPE_Q4_K) ||
            !type(layer.up, GGML_TYPE_Q4_K) || !type(layer.down, selected))
            return false;
        if (layer.full_attention) {
            if (!type(layer.attention.v, selected)) return false;
            for (auto * t : { layer.attention.q, layer.attention.k, layer.attention.out })
                if (!type(t, GGML_TYPE_Q4_K)) return false;
        } else {
            if (!type(layer.linear.qkv, selected)) return false;
            for (auto * t : { layer.linear.z, layer.linear.a, layer.linear.b, layer.linear.out })
                if (!type(t, GGML_TYPE_Q4_K)) return false;
        }
    }
    return true;
}
} // namespace qwen35_ocr
