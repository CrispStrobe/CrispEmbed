#pragma once
#include "qwen35_frontend.h"
#include "qwen35_vision.h"
#include "lighton4b_physical_f16_attention.h"
namespace qwen35_ocr {
inline bool lighton4b_physical_f16_vision_mode(bool physical) {
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION");
}
bool lighton4b_physical_f16_vision_available();
inline bool lighton4b_physical_f16_vision_valid(bool physical) {
    if (!lighton4b_physical_f16_vision_mode(physical)) return true;
    return core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL") && core_env::on("CRISPEMBED_QWEN35_OCR") &&
           core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE") &&
           core_env::on("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE") && core_env::on("CRISPEMBED_LIGHTON4B_VISION_OWNER") &&
           lighton4b_physical_f16_vision_available();
}
// Source-compatible still-image profile: F16 matrices, F32 parameters/states,
// P16/T2/M2, at most1024 patches. Unsupported requested profiles fail closed.
bool lighton4b_physical_f16_vision_cpu(const vision_weights &, const vision_config &, const image_inputs &, int threads,
                                       bool physical_owner, vision_outputs &);
} // namespace qwen35_ocr
