#include "ocr_crop.h"
#include "classical_preproc.h"

#include <algorithm>
#include <cstring>

namespace ocr_crop {

bool orient_180_rgb(std::vector<uint8_t> & pixels, int width, int height) {
    if (width < 8 || height < 8 || pixels.size() != (size_t)width * height * 3) return false;
    std::vector<uint8_t> gray((size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const uint8_t * p = pixels.data() + ((size_t)y * width + x) * 3;
            gray[(size_t)y * width + x] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
        }
    }
    float confidence = 0.0f;
    if (detect_text_angle(gray.data(), width, height, &confidence) != 180 || confidence < 0.75f) return false;
    for (int y = 0; y < (height + 1) / 2; y++) {
        for (int x = 0; x < width; x++) {
            const size_t a = ((size_t)y * width + x) * 3;
            const size_t b = ((size_t)(height - 1 - y) * width + (width - 1 - x)) * 3;
            if (a >= b) continue;
            for (int c = 0; c < 3; c++) std::swap(pixels[a + c], pixels[b + c]);
        }
    }
    return true;
}

bool orient_180_gray(std::vector<uint8_t> & pixels, int width, int height) {
    if (width < 8 || height < 8 || pixels.size() != (size_t)width * height) return false;
    float confidence = 0.0f;
    if (detect_text_angle(pixels.data(), width, height, &confidence) != 180 || confidence < 0.75f) return false;
    for (int y = 0; y < (height + 1) / 2; y++) {
        for (int x = 0; x < width; x++) {
            const size_t a = (size_t)y * width + x;
            const size_t b = (size_t)(height - 1 - y) * width + (width - 1 - x);
            if (a < b) std::swap(pixels[a], pixels[b]);
        }
    }
    return true;
}

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
