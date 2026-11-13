#pragma once
#include "core/env_gate.h"
#include <vector>
namespace qwen35_ocr {
inline bool lighton4b_physical_swiglu_mode(bool physical) {
#if defined(__EMSCRIPTEN__)
    (void)physical;
    return false;
#else
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU");
#endif
}
// Exact physical FFN geometry; GGML fused gate/up arithmetic only. Weights,
// projections and all legacy scalar/BF16 alternatives remain with callers.
bool lighton4b_physical_swiglu_cpu(const std::vector<float> & gate, const std::vector<float> & up, int tokens,
                                   int threads, std::vector<float> & output);
} // namespace qwen35_ocr
