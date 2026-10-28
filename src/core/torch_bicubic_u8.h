// src/core/torch_bicubic_u8.h - torch CPU uint8 antialiased bicubic resize.
//
// Header-only. Moved verbatim from src/qwen35_image.cpp (Qwen2VL/LightOnOCR/
// OvisOCR preprocessing) so other torchvision-backed processors can share it;
// qwen35_image.cpp now calls this and its arithmetic is unchanged.
//
// This is what `torchvision.transforms.v2.functional.resize(uint8 CHW tensor,
// BICUBIC, antialias=True)` runs on CPU: torchvision keeps uint8 for bicubic
// (_geometry.py _do_native_uint8_resize_on_cpu, v0.28.0) and torch dispatches
// to the separable uint8 AA kernel (aten/src/ATen/native/cpu/UpSampleKernel.cpp,
// v2.13.0: HelperInterpCubic::aa_filter a = -0.5; double weights normalized per
// output pixel; _compute_index_ranges_int16_weights picks the largest precision
// < 22 whose rounded max weight stays below 2^15 and rounds weights half away
// from zero; each pass starts from 1 << (precision - 1), arithmetic-shifts and
// clamps to [0, 255]). The horizontal pass runs first; a pass whose size does
// not change is skipped (an identity pass is exact with these weights).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace core_torch_resize {

struct axis_weights {
    std::vector<int> starts;
    std::vector<std::vector<int16_t>> values;
    unsigned precision = 0;
};

static inline double cubic(double x) {
    x = std::abs(x);
    if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}

static inline axis_weights coefficients(int input, int output) {
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

static inline uint8_t finish(int64_t value, unsigned precision) {
    // Explicit floor division matches arithmetic right shift even on negative sums.
    const int64_t divisor = int64_t(1) << precision;
    const int64_t rounded = value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
    return uint8_t(std::clamp<int64_t>(rounded, 0, 255));
}

// HWC RGB uint8 (height x width) -> HWC RGB uint8 (out_height x out_width).
static inline void resize_rgb(const uint8_t * rgb, int height, int width, int out_height, int out_width,
                              std::vector<uint8_t> & out) {
    std::vector<uint8_t> horizontal;
    const uint8_t * intermediate = rgb;
    if (out_width != width) {
        const auto weights = coefficients(width, out_width);
        horizontal.resize(size_t(height) * out_width * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < out_width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (weights.precision - 1);
                    for (size_t k = 0; k < weights.values[x].size(); ++k)
                        sum += int64_t(rgb[(size_t(y) * width + weights.starts[x] + k) * 3 + c]) * weights.values[x][k];
                    horizontal[(size_t(y) * out_width + x) * 3 + c] = finish(sum, weights.precision);
                }
        intermediate = horizontal.data();
    }
    out.resize(size_t(out_height) * out_width * 3);
    if (out_height != height) {
        const auto weights = coefficients(height, out_height);
        for (int y = 0; y < out_height; ++y)
            for (int x = 0; x < out_width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (weights.precision - 1);
                    for (size_t k = 0; k < weights.values[y].size(); ++k)
                        sum += int64_t(intermediate[(size_t(weights.starts[y] + k) * out_width + x) * 3 + c]) *
                               weights.values[y][k];
                    out[(size_t(y) * out_width + x) * 3 + c] = finish(sum, weights.precision);
                }
    } else
        std::copy_n(intermediate, out.size(), out.data());
}

} // namespace core_torch_resize
