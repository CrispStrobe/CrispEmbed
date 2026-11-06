#pragma once
#include "core/gguf_loader.h"
namespace qwen35_ocr {
inline bool lighton4b_gdn_out_metadata_matches(gguf_context * metadata) {
    auto integer = [&](const char * key, uint32_t expected) {
        if (!metadata) return false;
        const int index = gguf_find_key(metadata, key);
        return index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(metadata, index) == expected;
    };
    return metadata &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_recipe", "") == "lighton4b-q4k-gdn-out-q8-v1" &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_base_sha256", "") ==
               "f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91" &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_donor_sha256", "") ==
               "5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1" &&
           core_gguf::kv_str(metadata, "qwen35_ocr.mixed_quant_manifest_sha256", "") ==
               "895d1f509006c3f8ed599555e85033b7cec1064ca33699ef4ca1ac151f60c663" &&
           integer("qwen35_ocr.mixed_quant_selected_count", 24) &&
           integer("qwen35_ocr.mixed_quant_payload_delta", 125829120);
}
} // namespace qwen35_ocr
