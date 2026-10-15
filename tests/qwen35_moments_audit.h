#pragma once
// Validation-only scalar-lane adaptation of PyTorch RowwiseMoments:
// https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/moments_utils.h
// PyTorch contributors, BSD-3-Clause. Full notice: licenses/pytorch-LICENSE.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <utility>
namespace norm_audit {
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
static std::pair<float, float> row(const float * x, int width, int mode, bool fused_variance) {
    float mean = 0, variance = 0;
    if (mode < 2) {
        for (int i = 0; i < width; ++i) mean += x[i];
        mean /= width;
        for (int i = 0; i < width; ++i) {
            const float centered = x[i] - mean;
            if (fused_variance)
                variance = std::fma(centered, centered, variance);
            else {
                volatile float square = centered * centered;
                variance += square;
            }
        }
        variance /= width;
        if (mode == 1) {
            double sum = 0, square = 0;
            for (int i = 0; i < width; ++i) sum += double(x[i]);
            const double center = sum / width;
            for (int i = 0; i < width; ++i) {
                const double delta = double(x[i]) - center;
                square += delta * delta;
            }
            mean = float(center);
            variance = float(square / width);
        }
    } else {
        const int lanes = 1 << mode, vectors = width / lanes, chunks = (vectors + 15) / 16;
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
    }
    return { mean, 1.0f / std::sqrt(variance + 1e-6f) };
}
static void normalize(const std::vector<float> & x, const std::vector<float> & gamma, const std::vector<float> & beta,
                      int mode, std::vector<float> & output, std::vector<float> & means, std::vector<float> & rstds,
                      bool fused_variance = false) {
    const int width = int(gamma.size());
    output.resize(x.size());
    means.resize(x.size() / width);
    rstds.resize(means.size());
    for (size_t base = 0; base < x.size(); base += width) {
        const auto stats = row(x.data() + base, width, mode, fused_variance);
        means[base / width] = stats.first;
        rstds[base / width] = stats.second;
        for (int i = 0; i < width; ++i)
            output[base + i] = ((x[base + i] - stats.first) * stats.second) * gamma[i] + beta[i];
    }
}
// Conditional affine controls. Original moments never enter production inference.
static void affine(const std::vector<float> & x, const std::vector<float> & gamma, const std::vector<float> & beta,
                   const std::vector<float> & means, const std::vector<float> & rstds, bool fused,
                   std::vector<float> & output) {
    const size_t width = gamma.size();
    for (size_t j = 0; j < x.size(); ++j) {
        const float normalized = (x[j] - means[j / width]) * rstds[j / width];
        if (fused)
            output[j] = std::fma(normalized, gamma[j % width], beta[j % width]);
        else {
            volatile float weighted = normalized * gamma[j % width];
            output[j] = weighted + beta[j % width];
        }
    }
}
} // namespace norm_audit
