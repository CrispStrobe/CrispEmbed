#include "qwen35_frontend.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <utility>
namespace qwen35_ocr {
namespace {
struct axis_weights {
    std::vector<int> starts;
    std::vector<std::vector<int16_t>> values;
    unsigned precision = 0;
};
// Torch 2.10 CPU UpSampleKernel: double antialias weights, global-axis int16
// precision, signed half-away rounding; each uint8 pass rounds and clamps.
double cubic(double x) {
    x = std::abs(x);
    if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
axis_weights coefficients(int input, int output) {
    axis_weights result;
    result.starts.resize(output);
    result.values.resize(output);
    std::vector<std::vector<double>> weights(output);
    const double scale = double(input) / output, support = 2 * std::max(1.0, scale);
    double maximum = 0;
    for (int i = 0; i < output; ++i) {
        const double center = scale * (i + .5);
        const int start = std::max(0, int(center - support + .5));
        const int end = std::min(input, int(center + support + .5));
        result.starts[i] = start;
        double total = 0;
        for (int j = start; j < end; ++j) {
            const double value = cubic((j - center + .5) * (scale >= 1.0 ? 1.0 / scale : 1.0));
            weights[i].push_back(value);
            total += value;
        }
        for (auto & value : weights[i]) {
            value /= total;
            maximum = std::max(maximum, value);
        }
    }
    for (; result.precision < 22; ++result.precision)
        if (int(.5 + maximum * (1 << (result.precision + 1))) >= (1 << 15)) break;
    for (int i = 0; i < output; ++i)
        for (double value : weights[i]) {
            value *= 1 << result.precision;
            result.values[i].push_back(int16_t(int(value < 0 ? value - .5 : value + .5)));
        }
    return result;
}
int round_even_factor(int value) {
    const int base = value / 32, rem = value % 32;
    return 32 * (base + (rem > 16 || (rem == 16 && base % 2)));
}
uint8_t finish(int64_t value, unsigned precision) {
    // Explicit floor division matches arithmetic right shift even on negative sums.
    const int64_t divisor = int64_t(1) << precision;
    const int64_t rounded = value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
    return uint8_t(std::clamp<int64_t>(rounded, 0, 255));
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
    if (rh < 32 || rw < 32 || int64_t(rh) * rw / 256 > 1024) return false;
    std::vector<uint8_t> horizontal;
    const uint8_t * intermediate = rgb;
    if (rw != width) {
        const auto weights = coefficients(width, rw);
        horizontal.resize(size_t(height) * rw * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < rw; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (weights.precision - 1);
                    for (size_t k = 0; k < weights.values[x].size(); ++k)
                        sum += int64_t(rgb[(size_t(y) * width + weights.starts[x] + k) * 3 + c]) * weights.values[x][k];
                    horizontal[(size_t(y) * rw + x) * 3 + c] = finish(sum, weights.precision);
                }
        intermediate = horizontal.data();
    }
    image_inputs result;
    result.resized_rgb.resize(size_t(rh) * rw * 3);
    if (rh != height) {
        const auto weights = coefficients(height, rh);
        for (int y = 0; y < rh; ++y)
            for (int x = 0; x < rw; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (weights.precision - 1);
                    for (size_t k = 0; k < weights.values[y].size(); ++k)
                        sum += int64_t(intermediate[(size_t(weights.starts[y] + k) * rw + x) * 3 + c]) *
                               weights.values[y][k];
                    result.resized_rgb[(size_t(y) * rw + x) * 3 + c] = finish(sum, weights.precision);
                }
    } else
        std::copy_n(intermediate, result.resized_rgb.size(), result.resized_rgb.data());
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
