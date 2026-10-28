#include "qwen35_frontend.h"
#include "qwen35_image_bounds.h"
#include "core/env_gate.h"
#include "core/torch_bicubic_u8.h"
#include <algorithm>
#include <cmath>
#include <utility>
namespace qwen35_ocr {
namespace {
int round_even_factor(int value) {
    const int base = value / 32, rem = value % 32;
    return 32 * (base + (rem > 16 || (rem == 16 && base % 2)));
}
} // namespace
bool preprocess_rgb(const uint8_t * rgb, int height, int width, const image_config & config, image_inputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !rgb || height < 1 || width < 1 || height > 32768 || width > 32768 ||
        int64_t(height) * width > 16777216 || double(std::max(height, width)) / std::min(height, width) > 200 ||
        config.min_pixels < 1024 || config.max_pixels < config.min_pixels || config.max_pixels > 16777216)
        return false;
    int rh = round_even_factor(height), rw = round_even_factor(width);
    if (int64_t(rh) * rw > config.max_pixels) {
        const double beta = std::sqrt(double(height) * width / config.max_pixels);
        rh = std::max(32, int(std::floor(height / beta / 32)) * 32);
        rw = std::max(32, int(std::floor(width / beta / 32)) * 32);
    } else if (int64_t(rh) * rw < config.min_pixels) {
        const double beta = std::sqrt(double(config.min_pixels) / (double(height) * width));
        rh = int(std::ceil(height * beta / 32)) * 32;
        rw = int(std::ceil(width * beta / 32)) * 32;
    }
    const int bound =
        config.min_pixels == 448 * 448 && config.max_pixels == 2880 * 2880 ? ovisocr2_patch_limit() : 1024;
    if (rh < 32 || rw < 32 || int64_t(rh) * rw / 256 > bound) return false;
    image_inputs result;
    // Shared torch CPU uint8 AA bicubic (core/torch_bicubic_u8.h, moved from here).
    core_torch_resize::resize_rgb(rgb, height, width, rh, rw, result.resized_rgb);
    result.height = rh;
    result.width = rw;
    result.grid = { 1, rh / 16, rw / 16 };
    result.patches.reserve(size_t(rh) * rw * 6);
    // Original view/permute: merge-block row/col, intra-block row/col, C,T,P,P.
    for (int by = 0; by < rh / 32; ++by)
        for (int bx = 0; bx < rw / 32; ++bx)
            for (int iy = 0; iy < 2; ++iy)
                for (int ix = 0; ix < 2; ++ix)
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int py = 0; py < 16; ++py)
                                for (int px = 0; px < 16; ++px) {
                                    const int y = by * 32 + iy * 16 + py, x = bx * 32 + ix * 16 + px;
                                    result.patches.push_back(
                                        (float(result.resized_rgb[(size_t(y) * rw + x) * 3 + c]) - 127.5f) / 127.5f);
                                }
    output = std::move(result);
    return true;
}
} // namespace qwen35_ocr
