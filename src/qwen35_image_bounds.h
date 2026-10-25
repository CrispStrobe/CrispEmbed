#pragma once
#include "core/env_gate.h"
namespace qwen35_ocr {
// Bounded experiment only. The ordinary 1024-patch contract remains unchanged.
// No image is silently downscaled to fit either bound.
inline int ovisocr2_patch_limit() {
    return core_env::on("CRISPEMBED_QWEN35_OCR") && core_env::on("CRISPEMBED_OVISOCR2_FRONTEND") &&
                   core_env::on("CRISPEMBED_OVISOCR2_INFERENCE") && core_env::on("CRISPEMBED_OVISOCR2_LARGE_IMAGES")
               ? 2048
               : 1024;
}
} // namespace qwen35_ocr
