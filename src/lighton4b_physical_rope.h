#pragma once
#include "lighton4b_physical_f16_attention.h"
#include "qwen35_attention.h"
namespace qwen35_ocr {
inline bool lighton4b_physical_rope_mode(bool physical) {
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE");
}
inline bool lighton4b_physical_rope_valid(bool physical) {
    return !lighton4b_physical_rope_mode(physical) ||
           (core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL") && core_env::on("CRISPEMBED_QWEN35_OCR") &&
            !core_env::on("CRISPEMBED_LIGHTON4B_BF16_LINEAR") &&
            !core_env::on("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS") &&
            !core_env::on("CRISPEMBED_LIGHTON4B_HEAD_BF16") && lighton4b_physical_f16_attention_available());
}
// Source IMROPE, partial64/256 with sections11/11/10. Positions are explicit
// axis-major [3][tokens]; the reserved fourth source axis is zero. No weights.
// Outputs clear on entry and commit only after a complete finite CPU result.
bool lighton4b_physical_rope_cpu(const std::vector<float> & normalized, const attention_config & config,
                                  const float * positions, int tokens, int heads, int threads,
                                  std::vector<float> & rotated);
} // namespace qwen35_ocr
