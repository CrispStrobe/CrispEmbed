#pragma once
#include "core/env_gate.h"
#include "gguf.h"
#include <cstring>
namespace qwen35_ocr {
// Custom Q4_K_M-style type layout; Q8 tied head is an explicit exception.
// Pinned upstream18ab62a8, llama-quant.cpp use_more_bits, natural layer order.
inline constexpr const char * lighton4b_q6_recipe = "lighton4b-q4km-q6-q8head-v1";
// Native HF coordinate profile and upstream-equivalent type schedule only.
// Quant-block bytes are not publisher-equivalent: GDN V columns use128-wide tiles.
inline constexpr const char * lighton4b_imatrix_q4km_recipe = "lighton4b-weighted-q4km-style-q6head-v1";
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
inline bool lighton4b_imatrix_q4km_enabled() {
    return lighton4b_q6_enabled() && core_env::on("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD");
}
inline bool lighton4b_imatrix_q4km_metadata(gguf_context * metadata) {
    if (!metadata || !lighton4b_imatrix_q4km_enabled()) return false;
    auto string = [&](const char * key, const char * expected) {
        const auto index = gguf_find_key(metadata, key);
        return index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_STRING &&
               !std::strcmp(gguf_get_val_str(metadata, index), expected);
    };
    auto number = [&](const char * key, uint32_t expected) {
        const auto index = gguf_find_key(metadata, key);
        return index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(metadata, index) == expected;
    };
    return string("qwen35_ocr.mixed_quant_recipe", lighton4b_imatrix_q4km_recipe) &&
           string("qwen35_ocr.q6_names_sha256", "0ebc0b6104238b46de946f940f61cdc90e04a0eb187553026714fc1270cdb44c") &&
           string("qwen35_ocr.quant_source_f16_sha256",
                  "757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606") &&
           string("qwen35_ocr.quant_coordinate_profile", "hf-grouped-v1") &&
           string("crispembed.quantization.imatrix_source_sha256",
                  "10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7") &&
           string("crispembed.quantization.imatrix_profile", "lighton4b-text-inputs-hf-grouped-v1") &&
           number("crispembed.quantization.imatrix_expected_entries", 248) &&
           number("crispembed.quantization.imatrix_applied_entries", 248) && number("qwen35_ocr.q6_tensor_count", 33) &&
           number("general.file_type", 15);
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
