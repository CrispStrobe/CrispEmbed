/*
Source-compatible arithmetic/graph adapted from llama.cpp6c01 (MIT).
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "lighton4b_pillow_resize.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include "core/env_gate.h"
namespace qwen35_ocr {
namespace {
constexpr unsigned precision = 22;
struct weights {
    std::vector<int> start;
    std::vector<std::vector<int32_t>> values;
};
// Same operation order as pinned llama.cpp6c01 mtmd-image.cpp::resize_pillow.
double cubic(double x) {
    if (x < 0) x = -x;
    constexpr double a = -0.5;
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
}
weights coefficients(int input, int output) {
    weights out;
    out.start.resize(output);
    out.values.resize(output);
    const double scale = double(input) / output, filter_scale = std::max(1.0, scale), support = 2 * filter_scale,
                 ss = 1.0 / filter_scale;
    for (int i = 0; i < output; ++i) {
        const double center = (i + 0.5) * scale;
        const int begin = std::max(0, int(center - support + 0.5)), end = std::min(input, int(center + support + 0.5));
        out.start[i] = begin;
        std::vector<double> raw;
        double sum = 0;
        for (int j = begin; j < end; ++j) {
            const double value = cubic((j - center + 0.5) * ss);
            raw.push_back(value);
            sum += value;
        }
        for (double value : raw) {
            if (sum != 0) value /= sum;
            const double rounded = value * double(uint32_t(1) << precision) + (value < 0 ? -0.5 : 0.5);
            out.values[i].push_back(int32_t(rounded));
        }
    }
    return out;
}
uint8_t finish(int64_t sum) {
    const int64_t divisor = int64_t(1) << precision;
    const int64_t value = sum >= 0 ? sum / divisor : -((-sum + divisor - 1) / divisor);
    return uint8_t(std::clamp<int64_t>(value, 0, 255));
}
bool dimensions(int h, int w) {
    return h > 0 && w > 0 && h <= 32768 && w <= 32768 && int64_t(h) * w <= 16777216;
}
} // namespace
bool lighton4b_pillow_resize_rgb(const uint8_t * rgb, int h, int w, int oh, int ow, std::vector<uint8_t> & out) {
    out.clear();
    if (!rgb || !dimensions(h, w) || !dimensions(oh, ow) || !dimensions(h, ow) || !std::numeric_limits<double>::is_iec559) return false;
    const uint8_t * intermediate = rgb;
    std::vector<uint8_t> horizontal;
    if (ow != w) {
        const auto k = coefficients(w, ow);
        horizontal.resize(size_t(h) * ow * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < ow; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (precision - 1);
                    for (size_t j = 0; j < k.values[x].size(); ++j)
                        sum += int64_t(rgb[(size_t(y) * w + k.start[x] + j) * 3 + c]) * k.values[x][j];
                    horizontal[(size_t(y) * ow + x) * 3 + c] = finish(sum);
                }
        intermediate = horizontal.data();
    }
    out.resize(size_t(oh) * ow * 3);
    if (oh != h) {
        const auto k = coefficients(h, oh);
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x)
                for (int c = 0; c < 3; ++c) {
                    int64_t sum = int64_t(1) << (precision - 1);
                    for (size_t j = 0; j < k.values[y].size(); ++j)
                        sum += int64_t(intermediate[(size_t(k.start[y] + j) * ow + x) * 3 + c]) * k.values[y][j];
                    out[(size_t(y) * ow + x) * 3 + c] = finish(sum);
                }
    } else
        std::copy_n(intermediate, out.size(), out.data());
    return true;
}
bool lighton4b_pillow_preprocess_rgb(const uint8_t * rgb, int h, int w, bool upstream_physical_owner,
                                     image_inputs & out) {
    out = {};
    if (!upstream_physical_owner || !core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE") ||
        !core_env::on("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE"))
        return false;
    image_inputs value;
    // Keep the existing geometry and admission domain. Only requested physical
    // image arithmetic changes; the shared Torch/BF16 frontend stays intact.
    if (!preprocess_rgb(rgb, h, w, image_config{}, value) ||
        !lighton4b_pillow_resize_rgb(rgb, h, w, value.height, value.width, value.resized_rgb))
        return false;
    value.patches.clear();
    value.patches.reserve(size_t(value.height) * value.width * 6);
    for (int by = 0; by < value.height / 32; ++by)
        for (int bx = 0; bx < value.width / 32; ++bx)
            for (int iy = 0; iy < 2; ++iy)
                for (int ix = 0; ix < 2; ++ix)
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int py = 0; py < 16; ++py)
                                for (int px = 0; px < 16; ++px) {
                                    const int y = by * 32 + iy * 16 + py, x = bx * 32 + ix * 16 + px;
                                    const float scaled =
                                        float(value.resized_rgb[(size_t(y) * value.width + x) * 3 + c]) / 255.0f;
                                    value.patches.push_back((scaled - 0.5f) / 0.5f);
                                }
    out = std::move(value);
    return true;
}
} // namespace qwen35_ocr
