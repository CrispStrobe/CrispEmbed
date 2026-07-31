// Shared line-region extraction for detector-backed OCR engines.
#pragma once

#include <cstdint>
#include <vector>

namespace ocr_crop {

// Extract a clamped interleaved crop with symmetric padding where possible.
// The returned buffer is tightly packed and keeps the requested channel count.
std::vector<uint8_t> extract(const uint8_t * pixels, int width, int height, int channels, int x, int y, int crop_w,
                             int crop_h, int padding, int * out_width, int * out_height);

} // namespace ocr_crop
