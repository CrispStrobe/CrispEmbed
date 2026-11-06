#pragma once
#include "core/env_gate.h"
#include "ggml.h"
#include <cstddef>
#include <cstring>
#include <initializer_list>

namespace qwen35_ocr {
// Oracle-matching weight dtype only. Actual quantized storage stays unchanged.
// Protected float tensors/vision and other architectures/types never enter.
inline bool lighton4b_bf16_dequant_weights_mode() {
    return core_env::on("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS") &&
           core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") && core_env::on("CRISPEMBED_LIGHTON4B_QUANT") &&
           core_env::on("CRISPEMBED_QWEN35_QUANT");
}
inline bool lighton4b_bf16_dequant_weights_enabled(const ggml_tensor * source) {
    if (!source || (source->type != GGML_TYPE_Q8_0 && source->type != GGML_TYPE_Q4_K) ||
        !lighton4b_bf16_dequant_weights_mode())
        return false;
    const char * name = ggml_get_name(source);
    if (!std::strcmp(name, "l.embed_tokens.weight")) return true;
    constexpr const char * prefix = "l.layers.";
    if (std::strncmp(name, prefix, std::strlen(prefix))) return false;
    name += std::strlen(prefix);
    int layer = 0, digits = 0;
    while (*name >= '0' && *name <= '9') {
        if (++digits > 2) return false;
        layer = layer * 10 + (*name++ - '0');
    }
    if (!digits || layer > 31) return false;
    for (const char * suffix :
         { ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight", ".linear_attn.in_proj_qkv.weight",
           ".linear_attn.in_proj_z.weight", ".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight",
           ".linear_attn.out_proj.weight", ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
           ".self_attn.v_proj.weight", ".self_attn.o_proj.weight" })
        if (!std::strcmp(name, suffix)) return true;
    return false;
}
inline void lighton4b_bf16_dequant_weights(const ggml_tensor * source, float * values, size_t count) {
    if (!values || !lighton4b_bf16_dequant_weights_enabled(source)) return;
    for (size_t i = 0; i < count; ++i) values[i] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[i]));
}
} // namespace qwen35_ocr
