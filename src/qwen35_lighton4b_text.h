#pragma once
#include "qwen35_decoder.h"
#include "core/env_gate.h"
#include <set>
#include <string>
#include <utility>

namespace qwen35_ocr {
// A borrowed, exact text-tensor owner component for the immutable LightOn4B
// checkpoint schema. The caller retains tensor/backend lifetime. This component
// does not admit format 5, map vision, execute a decoder, or qualify a model.
// Pass only the text subset: exactly 426 stored tensors, with no lm_head alias.
struct lighton4b_text_owner {
    decoder_weights decoder;
    attention_config attention;
    int linear_key_heads = 0, linear_value_heads = 0, linear_head_dim = 0;
};

inline bool map_lighton4b_text(const core_gguf::tensor_map & text, lighton4b_text_owner & output) {
    output = {};
    // Experimental, default off; borrowed tensor metadata only, never inference admission.
    if (!core_env::on("CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER") || text.size() != 426) return false;
    lighton4b_text_owner candidate;
    std::set<std::string> consumed;
    std::set<ggml_tensor *> identities;
    auto take = [&](const std::string & name, int64_t a, int64_t b = 1, int64_t c = 1,
                    bool matrix = false) -> ggml_tensor * {
        auto found = text.find(name);
        if (found == text.end()) return nullptr;
        auto * tensor = found->second;
        if (!tensor || tensor->ne[0] != a || tensor->ne[1] != b || tensor->ne[2] != c || tensor->ne[3] != 1 ||
            !ggml_is_contiguous(tensor) ||
            (matrix ? tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 : tensor->type != GGML_TYPE_F32) ||
            !identities.insert(tensor).second || !consumed.insert(name).second)
            return nullptr;
        return tensor;
    };
    auto & decoder = candidate.decoder;
    decoder.embedding = take("l.embed_tokens.weight", 2560, 248320, 1, true);
    decoder.norm = take("l.norm.weight", 2560);
    if (!decoder.embedding || !decoder.norm) return false;
    decoder.head = decoder.embedding; // One stored tensor; official tied output projection.
    for (int i = 0; i < 32; ++i) {
        const auto prefix = "l.layers." + std::to_string(i) + ".";
        auto field = [&](const char * suffix, int64_t a, int64_t b = 1, int64_t c = 1, bool matrix = false) {
            return take(prefix + suffix, a, b, c, matrix);
        };
        decoder_layer_weights layer;
        layer.full_attention = i % 4 == 3;
        layer.input_norm = field("input_layernorm.weight", 2560);
        layer.post_norm = field("post_attention_layernorm.weight", 2560);
        layer.gate = field("mlp.gate_proj.weight", 2560, 9216, 1, true);
        layer.up = field("mlp.up_proj.weight", 2560, 9216, 1, true);
        layer.down = field("mlp.down_proj.weight", 9216, 2560, 1, true);
        if (!layer.input_norm || !layer.post_norm || !layer.gate || !layer.up || !layer.down) return false;
        if (layer.full_attention) {
            layer.attention = { field("self_attn.q_proj.weight", 2560, 8192, 1, true),
                                field("self_attn.k_proj.weight", 2560, 1024, 1, true),
                                field("self_attn.v_proj.weight", 2560, 1024, 1, true),
                                field("self_attn.o_proj.weight", 4096, 2560, 1, true),
                                field("self_attn.q_norm.weight", 256),
                                field("self_attn.k_norm.weight", 256) };
            const auto & a = layer.attention;
            if (!a.q || !a.k || !a.v || !a.out || !a.q_norm || !a.k_norm) return false;
        } else {
            layer.linear = { field("linear_attn.in_proj_qkv.weight", 2560, 8192, 1, true),
                             field("linear_attn.in_proj_z.weight", 2560, 4096, 1, true),
                             field("linear_attn.in_proj_a.weight", 2560, 32, 1, true),
                             field("linear_attn.in_proj_b.weight", 2560, 32, 1, true),
                             field("linear_attn.out_proj.weight", 4096, 2560, 1, true),
                             field("linear_attn.conv1d.weight", 4, 1, 8192),
                             field("linear_attn.A_log", 32),
                             field("linear_attn.dt_bias", 32),
                             field("linear_attn.norm.weight", 128) };
            const auto & l = layer.linear;
            if (!l.qkv || !l.z || !l.a || !l.b || !l.out || !l.conv || !l.a_log || !l.dt_bias || !l.norm) return false;
        }
        decoder.layers.push_back(layer);
    }
    if (consumed.size() != text.size()) return false;
    candidate.attention.query_heads = 16;
    candidate.attention.kv_heads = 4;
    candidate.attention.head_dim = 256;
    candidate.attention.rotary_dim = 64;
    candidate.linear_key_heads = 16;
    candidate.linear_value_heads = 32;
    candidate.linear_head_dim = 128;
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
