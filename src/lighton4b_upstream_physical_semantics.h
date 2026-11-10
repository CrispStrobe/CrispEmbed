#pragma once
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>
namespace qwen35_ocr {
// Only the independently authenticated physical upstream profile may set the
// per-owner flag. This never infers a layout from quantization types alone.
inline bool lighton4b_upstream_physical_enabled() {
#if defined(__EMSCRIPTEN__)
    return false; // native experiment; WASM remains explicitly unsupported
#else
    return core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL");
#endif
}
inline bool lighton4b_upstream_physical_mode_valid(bool physical) {
    return !physical || (lighton4b_upstream_physical_enabled() && !core_env::on("CRISPEMBED_LIGHTON4B_BF16_LINEAR") &&
                         !core_env::on("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS") &&
                         !core_env::on("CRISPEMBED_LIGHTON4B_HEAD_BF16"));
}
inline float lighton4b_upstream_gamma(float value, bool physical) {
    return physical ? value : (1.0f + value);
}
// Physical out weights retain 256-value blocks; their columns use tiled V
// heads. Native recurrent state stays HF grouped. Supports every admitted token
// count without a table or image-dependent constant, including aliased buffers.
inline bool lighton4b_upstream_v_to_tiled(const std::vector<float> & input, int tokens, std::vector<float> & output) {
    if (!lighton4b_upstream_physical_enabled() || tokens < 1 || tokens > 16 || input.size() != size_t(tokens) * 4096 ||
        !std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
        return false;
    std::vector<float> candidate(input.size());
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 32; ++h)
            for (int c = 0; c < 128; ++c)
                candidate[size_t(t) * 4096 + size_t(16 * (h % 2) + h / 2) * 128 + c] =
                    input[size_t(t) * 4096 + size_t(h) * 128 + c];
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
