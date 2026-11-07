#pragma once
#include "gguf_loader.h"
namespace core_lighton4b_release_fence {
// Public OCR does not admit raw upstream qwen35 or either experimental 4B
// profile. Metadata-only rejection must happen before choosing a backend or
// loading payload. Prototype gates never override this shipping boundary.
inline bool blocked(const gguf_context * metadata) {
    if (!metadata) return false;
    auto * meta = const_cast<gguf_context *>(metadata);
    const auto arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (arch == "qwen35") return true;
    if (arch != "qwen35_ocr") return false;
    const auto version = core_gguf::kv_u32(meta, "qwen35_ocr.format_version", 0);
    return version == 5 || version == 6 || core_gguf::kv_str(meta, "qwen35_ocr.profile", "") == "lightonocr3-4b" ||
           core_gguf::kv_str(meta, "qwen35_ocr.mixed_quant_recipe", "") == "lighton4b-upstream-q4km-physical-v1";
}
} // namespace core_lighton4b_release_fence
