#pragma once
namespace qwen35_ocr {
// Computed four-lane cascade statistics; no reference tensors or model state.
bool cascade_layer_norm(const float * x, int tokens, int width, float eps, const float * gamma, const float * beta,
                        float * output);
} // namespace qwen35_ocr
