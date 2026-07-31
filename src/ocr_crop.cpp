#include "ocr_crop.h"

#include <algorithm>
#include <cstring>

namespace ocr_crop {

std::vector<uint8_t> extract(const uint8_t * pixels, int width, int height, int channels, int x, int y, int crop_w,
                             int crop_h, int padding, int * out_width, int * out_height) {
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!pixels || width <= 0 || height <= 0 || channels <= 0 || crop_w <= 0 || crop_h <= 0) return {};

    const int x0 = std::max(0, x - std::max(0, padding));
    const int y0 = std::max(0, y - std::max(0, padding));
    const int x1 = std::min(width, x + crop_w + std::max(0, padding));
    const int y1 = std::min(height, y + crop_h + std::max(0, padding));
    const int w = x1 - x0;
    const int h = y1 - y0;
    if (w <= 0 || h <= 0) return {};

    std::vector<uint8_t> result((size_t)w * h * channels);
    for (int row = 0; row < h; row++) {
        const uint8_t * source = pixels + ((size_t)(y0 + row) * width + x0) * channels;
        std::memcpy(result.data() + (size_t)row * w * channels, source, (size_t)w * channels);
    }
    if (out_width) *out_width = w;
    if (out_height) *out_height = h;
    return result;
}

} // namespace ocr_crop
