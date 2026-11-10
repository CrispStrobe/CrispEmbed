#pragma once
#include "lighton4b_q6_policy.h"
#include "core/gguf_loader.h"
namespace qwen35_ocr {
inline constexpr const char * lighton4b_gdn_out_q6_recipe = "lighton4b-q4k-gdn-out-q6-v1";
inline bool lighton4b_gdn_out_q6_enabled() {
    return lighton4b_q6_enabled() && core_env::on("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6");
}
inline bool lighton4b_gdn_out_q6_metadata_matches(gguf_context * metadata) {
    auto integer = [&](const char * key, uint32_t expected) {
        if (!metadata) return false;
        const int index = gguf_find_key(metadata, key);
        return index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(metadata, index) == expected;
    };
    return metadata && lighton4b_gdn_out_q6_enabled() && integer("general.file_type", 15) &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_recipe", "") == lighton4b_gdn_out_q6_recipe &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_base_sha256", "") ==
               "f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91" &&
           core_gguf::kv_str(metadata, "qwen35_ocr.q6_source_checkpoint_sha256", "") ==
               "6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67" &&
           core_gguf::kv_str(metadata, "qwen35_ocr.q6_names_sha256", "") ==
               "2b6d3e1109215160bbbd9760b1d11fc88bd6a34c70d9fceffdcbe93e68259845" &&
           integer("qwen35_ocr.mixed_quant_selected_count", 24) &&
           integer("qwen35_ocr.mixed_quant_payload_delta", 64880640);
}
} // namespace qwen35_ocr
