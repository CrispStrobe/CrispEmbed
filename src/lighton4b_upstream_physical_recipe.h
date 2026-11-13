#pragma once
#include "core/gguf_loader.h"
#include "lighton4b_upstream_physical_semantics.h"
namespace qwen35_ocr {
inline constexpr const char * lighton4b_upstream_physical_recipe = "lighton4b-upstream-q4km-physical-v1";
inline constexpr const char * lighton4b_upstream_physical_q6_recipe = "lighton4b-upstream-q6-physical-v1";
// A distinct format6 declaration. These are structural provenance checks; the
// hosted import controller independently authenticates complete source files and
// every persisted tensor. A metadata string alone is never quality proof.
inline bool lighton4b_upstream_physical_metadata_matches(gguf_context * metadata) {
    if (!metadata) return false;
    const bool full_q6 = lighton4b_upstream_physical_q6_enabled();
    auto string = [&](const char * key, const char * value) {
        const int at = gguf_find_key(metadata, key);
        return at >= 0 && gguf_get_kv_type(metadata, at) == GGUF_TYPE_STRING &&
               std::string(gguf_get_val_str(metadata, at)) == value;
    };
    auto integer = [&](const char * key, uint32_t value) {
        const int at = gguf_find_key(metadata, key);
        return at >= 0 && gguf_get_kv_type(metadata, at) == GGUF_TYPE_UINT32 && gguf_get_val_u32(metadata, at) == value;
    };
    auto boolean = [&](const char * key, bool value) {
        const int at = gguf_find_key(metadata, key);
        return at >= 0 && gguf_get_kv_type(metadata, at) == GGUF_TYPE_BOOL && gguf_get_val_bool(metadata, at) == value;
    };
    const int eog = gguf_find_key(metadata, "qwen35_ocr.upstream_physical.eog_ids");
    if (eog < 0 || gguf_get_kv_type(metadata, eog) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(metadata, eog) != GGUF_TYPE_INT32 || gguf_get_arr_n(metadata, eog) != 2)
        return false;
    const auto * ids = static_cast<const int32_t *>(gguf_get_arr_data(metadata, eog));
    if (!ids || ids[0] != 248044 || ids[1] != 248046) return false;
    return integer("qwen35_ocr.format_version", 6) && integer("general.file_type", full_q6 ? 18 : 15) &&
           boolean("qwen35_ocr.zero_centered_text_norm", false) && boolean("qwen35_ocr.tie_word_embeddings", true) &&
           string("qwen35_ocr.mixed_quant_recipe",
                  full_q6 ? lighton4b_upstream_physical_q6_recipe : lighton4b_upstream_physical_recipe) &&
           string("qwen35_ocr.upstream_physical.repository", "richardyoung/LightOnOCR-3-4B-GGUF") &&
           string("qwen35_ocr.upstream_physical.revision", "8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01") &&
           string("qwen35_ocr.upstream_physical.sha256",
                  full_q6 ? "290e2b46627c9d3f932c25ef6697fc71a37a9b226337d761a0190185eaa738b7"
                          : "b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26") &&
           string("qwen35_ocr.upstream_physical.protected_source_sha256",
                  "f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91") &&
           string("qwen35_ocr.upstream_physical.mapping_sha256",
                  full_q6 ? "f12db3045f003dd2bcde85ba4159beda86cbaec5617ebe7d9f40388336796390"
                          : "cf3af1c7f7893a06cf1660358e3ec7084dd6352730e8bcfd384e749e4f11edb1") &&
           string("qwen35_ocr.upstream_physical.source_code", "6c01da6278ab6c69ed67d9124806b50d2b299ba7") &&
           string("qwen35_ocr.upstream_physical.semantics",
                  "gamma-direct,a-direct,hf-state,tiled-out-input,upstream-control-types") &&
           integer("qwen35_ocr.upstream_physical.version", 1) &&
           integer("qwen35_ocr.upstream_physical.row_permuted_tensors", 96) &&
           integer("qwen35_ocr.upstream_physical.retained_out_tensors", 24) &&
           integer("qwen35_ocr.upstream_physical.protected_vision_tensors", 297);
}
} // namespace qwen35_ocr
