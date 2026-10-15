// Scalar four-lane adaptation of PyTorch RowwiseMoments (BSD-3-Clause).
// Pinned source: https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/moments_utils.h
// Full upstream notice: core/third_party/pytorch/LICENSE.
#include "qwen35_norm.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
namespace qwen35_ocr {
namespace {
struct moments {
    int count = 0;
    float mean = 0, square = 0;
};
static void merge(moments & a, const moments & b, bool vector_lane) {
    const int total = a.count + b.count;
    const float ratio = total ? float(b.count) / float(total) : 0.0f;
    const float delta = b.mean - a.mean;
    if (vector_lane) {
        const float correction = ratio * delta, accumulated_delta = delta * float(a.count);
        a.mean = a.mean + correction;
        a.square = std::fma(accumulated_delta, correction, a.square + b.square);
    } else {
        a.mean += ratio * delta;
        a.square += b.square + delta * delta * ratio * float(a.count);
    }
    a.count = total;
}
static std::pair<float, float> row(const float * x, int width, float eps) {
    float mean = 0, variance = 0;
    const int lanes = 4, vectors = width / lanes, chunks = (vectors + 15) / 16;
    int depth = 0;
    while ((1 << depth) < chunks) ++depth;
    std::array<std::array<moments, 16>, 16> stack{};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int n = std::min(16, vectors - chunk * 16);
        for (int lane = 0; lane < lanes; ++lane) {
            moments value;
            for (int j = 0; j < n; ++j) {
                const float sample = x[(chunk * 16 + j) * lanes + lane];
                const float delta = sample - value.mean;
                value.mean = std::fma(1.0f / float(j + 1), delta, value.mean);
                value.square = std::fma(delta, sample - value.mean, value.square);
                ++value.count;
            }
            merge(stack[0][lane], value, true);
        }
        int mask = chunk + 1;
        for (int d = 1; d < depth && !(mask & 1); ++d, mask >>= 1)
            for (int lane = 0; lane < lanes; ++lane) {
                merge(stack[d][lane], stack[d - 1][lane], true);
                stack[d - 1][lane] = {};
            }
    }
    for (int d = 1; d < depth; ++d)
        for (int lane = 0; lane < lanes; ++lane) merge(stack[0][lane], stack[d][lane], true);
    moments total;
    for (int i = vectors * lanes; i < width; ++i) {
        const float delta = x[i] - total.mean;
        ++total.count;
        total.mean += delta / float(total.count);
        total.square += delta * (x[i] - total.mean);
    }
    for (int lane = 0; lane < lanes; ++lane) merge(total, stack[0][lane], false);
    mean = total.mean;
    variance = total.square / float(width);
    return { mean, 1.0f / std::sqrt(variance + eps) };
}
} // namespace
bool cascade_layer_norm(const float * x, int tokens, int width, float eps, const float * gamma, const float * beta,
                        float * output) {
    // The measured checkpoint uses width768. Keep other shapes on their existing path.
    if (!x || !gamma || !beta || !output || tokens < 1 || width != 768 || eps != 1e-6f) return false;
    for (int t = 0; t < tokens; ++t) {
        const auto stats = row(x + size_t(t) * width, width, eps);
        for (int i = 0; i < width; ++i) {
            const size_t j = size_t(t) * width + i;
            const float normalized = (x[j] - stats.first) * stats.second;
            output[j] = std::fma(normalized, gamma[i], beta[i]);
            if (!std::isfinite(output[j])) return false;
        }
    }
    return true;
}
} // namespace qwen35_ocr
