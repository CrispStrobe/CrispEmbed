#pragma once
#include "qwen35_frontend.h"
namespace qwen35_ocr {
// Reference-compatible Pillow uint8 AA bicubic: fixed22-bit coefficients,
// horizontal pass first. No shared Torch resize arithmetic is changed.
bool lighton4b_pillow_resize_rgb(const uint8_t *, int height, int width, int out_height, int out_width,
                                 std::vector<uint8_t> &);
bool lighton4b_pillow_preprocess_rgb(const uint8_t *, int height, int width, bool upstream_physical_owner,
                                     image_inputs &);
} // namespace qwen35_ocr
