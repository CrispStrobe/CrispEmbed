#pragma once
#include "core/env_gate.h"
#include <vector>
namespace qwen35_ocr {
struct attention_cache;
// Experimental, exact pinned Linux CPU ABI only; all unsupported builds return
// false. The public loader never admits this physical model as a release.
bool lighton4b_physical_f16_attention_available();
inline bool lighton4b_physical_f16_attention_mode(bool physical) {
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION");
}
inline bool lighton4b_physical_f16_attention_valid(bool physical) {
    if (!lighton4b_physical_f16_attention_mode(physical)) return true;
    return core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL") && core_env::on("CRISPEMBED_QWEN35_OCR") &&
           !core_env::on("CRISPEMBED_LIGHTON4B_BF16_LINEAR") &&
           !core_env::on("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS") &&
           !core_env::on("CRISPEMBED_LIGHTON4B_HEAD_BF16") && lighton4b_physical_f16_attention_available();
}
// Actual post-RoPE queries and fused raw Q/gate projections. Only new KV entries
// are rounded; prior entries must already be canonical F16 represented in F32.
// No model weights or full-model F32 cache are retained. Outputs commit on success.
bool lighton4b_physical_f16_attention_cpu(const std::vector<float> & queries, const std::vector<float> & fused_q_gate,
                                          const attention_cache & cache, int past, int tokens, int threads,
                                          std::vector<float> & pooled, attention_cache & canonical_cache);
} // namespace qwen35_ocr
