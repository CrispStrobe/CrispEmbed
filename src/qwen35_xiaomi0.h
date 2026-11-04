#pragma once
#include "core/gguf_loader.h"
#include <string>
#include <vector>

namespace qwen35_ocr {
namespace xiaomi0_profile {
inline constexpr const char * model = "xiaomi-research/Xiaomi-OCR-0";
inline constexpr const char * revision = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30";
inline constexpr const char * document =
    "Extract all information from the main body of the document image and represent it "
    "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
    "format, formulas in the document should be represented using LATEX format, and the "
    "parsing should be organized according to the reading order.";
inline constexpr const char * text = "Extract the text in the image.";
inline constexpr const char * table = "Parse the table in the image into OTSL.";
inline constexpr const char * formula = "Identify the formula in the image and represent it using LATEX format.";

inline bool metadata(gguf_context * meta) {
    return core_gguf::kv_u32(meta, "qwen35_ocr.format_version", 0) == 4 &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile", "") == "xiaomi-ocr-0" &&
           core_gguf::kv_str(meta, "qwen35_ocr.source_audit_sha256", "") ==
               "10956c0dfc2244b6a082ee0acaeb40de7b885d46ddfa67a5bca973556d32855f" &&
           core_gguf::kv_str(meta, "qwen35_ocr.checkpoint_sha256", "") ==
               "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348" &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.prompt_order", "") == "image-first" &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.default_task", "") == "document" &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.prompt.document", "") == document &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.prompt.text", "") == text &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.prompt.table", "") == table &&
           core_gguf::kv_str(meta, "qwen35_ocr.profile.prompt.formula", "") == formula &&
           core_gguf::kv_str(meta, "qwen35_ocr.mtp_policy", "") == "exact15-excluded-nonspeculative" &&
           core_gguf::kv_str(meta, "qwen35_ocr.layout_dependency", "") == "PP-DocLayoutV3" &&
           core_gguf::kv_u32(meta, "qwen35_ocr.primary_eos_id", 0) == 248046 &&
           core_gguf::kv_u32(meta, "tokenizer.ggml.eos_token_id", 0) == 248046 &&
           core_gguf::kv_u32(meta, "tokenizer.ggml.padding_token_id", 0) == 248044 &&
           core_gguf::kv_i32_array(meta, "qwen35_ocr.stop_token_ids") == std::vector<int>{ 248046 } &&
           core_gguf::kv_u32(meta, "qwen35_ocr.image_min_pixels", 0) == 65536 &&
           core_gguf::kv_u32(meta, "qwen35_ocr.image_max_pixels", 0) == 16777216 &&
           core_gguf::kv_u32(meta, "qwen35_ocr.native_patch_limit", 0) == 1024 &&
           core_gguf::kv_str_array(meta, "qwen35_ocr.missing_source_files") ==
               std::vector<std::string>{ "generation_config.json", "preprocessor_config.json" } &&
           !core_gguf::kv_bool(meta, "qwen35_ocr.native_alias_qualified", true) &&
           !core_gguf::kv_bool(meta, "qwen35_ocr.image_max_qualified", true) &&
           !core_gguf::kv_bool(meta, "qwen35_ocr.full_document_qualified", true);
}
inline bool task_prompt(const std::string & input, std::string & output) {
    if (input.empty())
        output = document;
    else if (input == document || input == text || input == table || input == formula)
        output = input;
    else
        return false;
    return true;
}
} // namespace xiaomi0_profile
} // namespace qwen35_ocr
