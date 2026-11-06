#pragma once
#include "core/env_gate.h"
#include "gguf.h"
#include <cstring>
namespace qwen35_ocr {
// Custom Q4_K_M-style type layout; Q8 tied head is an explicit exception.
// Pinned upstream18ab62a8, llama-quant.cpp use_more_bits, natural layer order.
inline constexpr const char * lighton4b_q6_recipe = "lighton4b-q4km-q6-q8head-v1";
inline bool lighton4b_q6_enabled() {
#if defined(__EMSCRIPTEN__)
    return false; // Native candidate only; WASM remains unqualified.
#else
    return core_env::on("CRISPEMBED_LIGHTON4B_Q6_K");
#endif
}
inline constexpr bool lighton4b_q6_layer(int layer) {
    return layer >= 0 && layer < 32 && (layer < 4 || layer >= 28 || (layer - 4) % 3 == 2);
}
inline bool lighton4b_q6_metadata(gguf_context * metadata) {
    if (!metadata || !lighton4b_q6_enabled()) return false;
    const auto recipe = gguf_find_key(metadata, "qwen35_ocr.mixed_quant_recipe");
    const auto type = gguf_find_key(metadata, "general.file_type");
    return recipe >= 0 && type >= 0 && gguf_get_kv_type(metadata, recipe) == GGUF_TYPE_STRING &&
           !std::strcmp(gguf_get_val_str(metadata, recipe), lighton4b_q6_recipe) &&
           gguf_get_kv_type(metadata, type) == GGUF_TYPE_UINT32 && gguf_get_val_u32(metadata, type) == 15;
}
} // namespace qwen35_ocr
