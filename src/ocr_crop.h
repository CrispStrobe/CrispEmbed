// Shared line-region extraction for detector-backed OCR engines.
#pragma once

#include <cstdint>
#include <vector>

namespace ocr_crop {

// Extract a clamped interleaved crop with symmetric padding where possible.
// The returned buffer is tightly packed and keeps the requested channel count.
std::vector<uint8_t> extract(const uint8_t * pixels, int width, int height, int channels, int x, int y, int crop_w,
                             int crop_h, int padding, int * out_width, int * out_height);

// Apply the existing classical 0°/180° text-line check in place. Returns true
// when a 180° rotation was applied.
bool orient_180_rgb(std::vector<uint8_t> & pixels, int width, int height);
bool orient_180_gray(std::vector<uint8_t> & pixels, int width, int height);

} // namespace ocr_crop
