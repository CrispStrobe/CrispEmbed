#pragma once
#include "core/env_gate.h"
namespace qwen35_ocr {
// Xiaomi unmerged patches for the three retained example pages: 1,984 (qianziwen),
// 5,916 (receipt) and 9,184 (dense equations). Only the size that was actually
// captured is admitted; no page is downscaled to fit.
inline constexpr int kOrdinaryPatchLimit = 1024;
inline constexpr int kXiaomiPatchLimit = 9184;
// Bounded experiment only. The ordinary 1024-patch contract remains unchanged.
// No image is silently downscaled to fit either bound.
inline int ovisocr2_patch_limit() {
    return core_env::on("CRISPEMBED_QWEN35_OCR") && core_env::on("CRISPEMBED_OVISOCR2_FRONTEND") &&
                   core_env::on("CRISPEMBED_OVISOCR2_INFERENCE") && core_env::on("CRISPEMBED_OVISOCR2_LARGE_IMAGES")
               ? 2048
               : kOrdinaryPatchLimit;
}
// Requires both Xiaomi gates. A profile that is not Xiaomi never calls this, so a
// stray environment variable cannot widen an ordinary caller's bound.
inline bool xiaomi_patch_limit_enabled() {
    return core_env::on("CRISPEMBED_XIAOMI_OCR_FRONTEND") && core_env::on("CRISPEMBED_XIAOMI_OCR_INFERENCE");
}
inline int xiaomi_patch_limit() {
    return xiaomi_patch_limit_enabled() ? kXiaomiPatchLimit : kOrdinaryPatchLimit;
}
} // namespace qwen35_ocr
