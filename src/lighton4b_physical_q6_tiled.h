#pragma once
#include "core/env_gate.h"
#include <cstddef>
#include <vector>
namespace qwen35_ocr {
inline bool lighton4b_physical_q6_tiled_mode(bool physical) {
#if defined(__EMSCRIPTEN__)
    (void)physical;
    return false;
#else
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED");
#endif
}
bool lighton4b_physical_q6_tiled_available();
// Source6c01 AVX2 tiled Q6 block reduction, without persistent unpacked weights.
// Caller owns physical-profile/gate admission. Only batches >=8 are eligible;
// smaller batches retain the original native GGML route.
bool lighton4b_physical_q6_tiled_cpu(const void * packed, size_t bytes, const float * input,
                                    int width, int rows, int tokens, int threads, std::vector<float> & output);
} // namespace qwen35_ocr
