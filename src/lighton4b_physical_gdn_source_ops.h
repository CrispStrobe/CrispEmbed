#pragma once
#include "core/env_gate.h"
namespace qwen35_ocr {
struct linear_inputs;
struct linear_outputs;
inline bool lighton4b_physical_gdn_source_ops_mode(bool physical) {
#if defined(__EMSCRIPTEN__)
    (void)physical;
    return false;
#else
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS");
#endif
}
// Physical LightOn4B only: source GGML normalization/nonlinear graph while
// retaining native scalar log1p softplus, whose bits match upstream 6c01.
// Projections, quantized weights, output projection and HF cache formats stay
// with their existing callers. Explicit mode fails closed on other geometry.
bool lighton4b_physical_gdn_source_ops_cpu(const linear_inputs & in, int key_heads, linear_outputs & out);
} // namespace qwen35_ocr
