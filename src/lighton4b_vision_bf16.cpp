#include "qwen35_image_bounds.h"
#include "lighton4b_vision_bf16.h"
#include "qwen35_math.h"
#include "qwen35_norm.h"
#include "core/env_gate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <string>
#include <thread>

namespace qwen35_ocr {
// Scalar RowwiseMoments adaptation from qwen35_norm.cpp; PyTorch BSD-3-Clause
// notice remains in core/third_party/pytorch/LICENSE. The shared768-only
// measured helper is unchanged; this isolated candidate owns its1024 shape.
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
// PyTorch v2.13.0 reduced-type RowwiseMoments: AVX2 BF16 vectors
// convert to two eight-lane float vectors, each merged successively.
static std::pair<float, float> row_bf16_avx2(const float * x, int width, float eps, bool scalar_fma = false) {
    constexpr int lanes = 8, vector_width = 2 * lanes;
    const int vectors = width / vector_width, chunks = (vectors + 15) / 16;
    int depth = 0;
    while ((1 << depth) < chunks) ++depth;
    std::array<std::array<moments, lanes>, 16> stack{};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int n = std::min(16, vectors - chunk * 16);
        for (int half = 0; half < 2; ++half)
            for (int lane = 0; lane < lanes; ++lane) {
                moments value;
                for (int j = 0; j < n; ++j) {
                    const float sample = x[(chunk * 16 + j) * vector_width + half * lanes + lane];
                    const float delta = sample - value.mean;
                    value.mean = std::fma(delta, 1.0f / float(j + 1), value.mean);
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
    for (int i = vectors * vector_width; i < width; ++i) {
        const float delta = x[i] - total.mean;
        ++total.count;
        total.mean += delta / float(total.count);
        total.square += delta * (x[i] - total.mean);
    }
    for (int lane = 0; lane < lanes; ++lane) {
        const auto & value = stack[0][lane];
        if (!scalar_fma) {
            merge(total, value, false);
        } else {
            const int count = total.count + value.count;
            const float ratio = count ? float(value.count) / float(count) : 0.0f;
            const float delta = value.mean - total.mean;
            total.mean = std::fma(ratio, delta, total.mean);
            const float correction = (delta * delta) * ratio;
            total.square += std::fma(correction, float(total.count), value.square);
            total.count = count;
        }
    }
    return { total.mean, 1.0f / std::sqrt(total.square / float(width) + eps) };
}
// ATen vectorized mixed-type second pass uses separate multiply/add operators.
// Force those boundaries even under a compiler that permits contraction.
static float multiply_separately(float a, float b) {
    volatile float product = a * b;
    return product;
}
} // namespace

static void round_bf16(std::vector<float> & values) {
    for (float & value : values) value = lighton4b_bf16(value);
}
static bool shape(ggml_tensor * t, int columns, int rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool finite(const std::vector<float> & v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}
static bool bias(std::vector<float> & x, ggml_tensor * b, int width) {
    if (!shape(b, width, 1)) return false;
    std::vector<float> values(width);
    if (!tensor_floats(b, values.data())) return false;
    for (size_t i = 0; i < x.size(); ++i) x[i] += values[i % width];
    return finite(x);
}
// Scalar adaptation of PyTorch v2.13 BF16 dot fallback. Eight float
// registers of eight lanes consume64 operands, then reduce in the source tree.
// PyTorch BSD notice: core/third_party/pytorch/LICENSE; ggml MIT: ggml/LICENSE.
template <bool safe_products> static float source_bf16_dot(const float * x, const float * w, int width) {
    std::array<std::array<float, 8>, 8> sums{};
    for (int block = 0; block < width; block += 64)
        for (int reg = 0; reg < 8; ++reg)
            for (int lane = 0; lane < 8; ++lane) {
                const int at = block + reg * 8 + lane;
                if constexpr (safe_products)
                    sums[reg][lane] = sums[reg][lane] + x[at] * w[at];
                else
                    sums[reg][lane] = std::fma(x[at], w[at], sums[reg][lane]);
            }
    for (int offset = 4; offset; offset /= 2)
        for (int reg = 0; reg < offset; ++reg)
            for (int lane = 0; lane < 8; ++lane) sums[reg][lane] += sums[reg + offset][lane];
    const auto & z = sums[0];
    const float dot = ((z[0] + z[4]) + (z[2] + z[6])) + ((z[1] + z[5]) + (z[3] + z[7]));
    return dot + 0.0f; // Source adds the empty vector-tail reduction too.
}
// Original reduced-type non-transposed GEMM used by the Conv3D patch: four
// independent accumulators, then the source's left-associated scalar reduction.
template <bool safe_products> static float source_bf16_conv_dot(const float * x, const float * w, int width) {
    std::array<float, 4> sums{};
    for (int i = 0; i < width; i += 4)
        for (int lane = 0; lane < 4; ++lane) {
            if constexpr (safe_products)
                sums[lane] = sums[lane] + x[i + lane] * w[i + lane];
            else
                sums[lane] = std::fma(x[i + lane], w[i + lane], sums[lane]);
        }
    return ((sums[0] + sums[1]) + sums[2]) + sums[3];
}
static bool source_bf16_linear(ggml_tensor * w, ggml_tensor * b, const std::vector<float> & x, int width, int rows,
                               int threads, std::vector<float> & y, bool convolution = false) {
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !w || width < 1 || width > 4096 || width % 64 || rows < 1 ||
        rows > 4096 || threads < 1 || threads > 256 || w->ne[0] != width || w->ne[1] < 1 || w->ne[1] > 8192 ||
        w->ne[2] != 1 || w->ne[3] != 1 || !shape(b, int(w->ne[1]), 1) || x.size() != size_t(rows) * width)
        return false;
    const int output_width = int(w->ne[1]);
    if (convolution && (width != 1536 || output_width != 1024)) return false;
    std::vector<float> weights(size_t(width) * output_width), biases(output_width);
    if (!tensor_floats(w, weights.data()) || !tensor_floats(b, biases.data())) return false;
    bool safe_products = true;
    auto operand = [&](float value) {
        if (!std::isfinite(value) || lighton4b_bf16(value) != value) return false;
        // BF16 products have at most16 significant bits. In this normal range
        // the float product is exact, so separate multiply/add equals FMA.
        const float magnitude = std::abs(value);
        if (magnitude && (magnitude < 0x1p-63f || magnitude > 0x1p63f)) safe_products = false;
        return true;
    };
    if (!std::all_of(x.begin(), x.end(), operand) || !std::all_of(weights.begin(), weights.end(), operand) ||
        !std::all_of(biases.begin(), biases.end(), [](float v) { return std::isfinite(v) && lighton4b_bf16(v) == v; }))
        return false;
    y.resize(size_t(rows) * output_width);
    const int workers = std::min({ threads, 8, int(y.size()) });
    auto work = [&](int worker) {
        const size_t first = y.size() * size_t(worker) / size_t(workers);
        const size_t end = y.size() * size_t(worker + 1) / size_t(workers);
        for (size_t at = first; at < end; ++at) {
            const int token = int(at / output_width), output = int(at % output_width);
            const float * input = x.data() + size_t(token) * width;
            const float * weight = weights.data() + size_t(output) * width;
            const float dot = convolution ? (safe_products ? source_bf16_conv_dot<true>(input, weight, width)
                                                           : source_bf16_conv_dot<false>(input, weight, width))
                                          : (safe_products ? source_bf16_dot<true>(input, weight, width)
                                                           : source_bf16_dot<false>(input, weight, width));
            y[at] = lighton4b_bf16(dot + biases[output]);
        }
    };
    std::vector<std::thread> jobs;
    try {
        for (int worker = 1; worker < workers; ++worker) jobs.emplace_back(work, worker);
    } catch (...) {
        for (auto & job : jobs) job.join();
        return false;
    }
    work(0);
    for (auto & job : jobs) job.join();
    return finite(y);
}
static bool linear(ggml_tensor * w, ggml_tensor * b, const std::vector<float> & x, int width, int rows, int threads,
                   std::vector<float> & y, bool is_linear = true) {
    if (!is_linear && core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_CONV"))
        return source_bf16_linear(w, b, x, width, rows, threads, y, true);
    const bool source_order = is_linear ? core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR")
                                        : core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_PATCH_LINEAR");
    if (source_order) return source_bf16_linear(w, b, x, width, rows, threads, y);
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ w }, x.data(), width, rows, threads, projections)) return false;
    y = std::move(projections[0]);
    if (!bias(y, b, int(w->ne[1]))) return false;
    round_bf16(y);
    return finite(y);
}
static bool layer_norm(const std::vector<float> & x, ggml_tensor * w, ggml_tensor * b, int width, float eps,
                       std::vector<float> & y) {
    if (!shape(w, width, 1) || !shape(b, width, 1)) return false;
    std::vector<float> weights(width), biases(width);
    if (!tensor_floats(w, weights.data()) || !tensor_floats(b, biases.data())) return false;
    y.resize(x.size());
    const bool contracted_bf16 = core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM");
    const bool source_bf16 = contracted_bf16 || core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_NORM");
    for (int token = 0; token < int(x.size() / width); ++token) {
        const auto stats = source_bf16 ? row_bf16_avx2(x.data() + size_t(token) * width, width, eps, contracted_bf16)
                                       : row(x.data() + size_t(token) * width, width, eps);
        for (int i = 0; i < width; ++i) {
            const size_t at = size_t(token) * width + i;
            const float normalized = (x[at] - stats.first) * stats.second;
            y[at] = std::fma(normalized, weights[i], biases[i]);
            if (source_bf16) {
                const float bias = -stats.second * stats.first;
                const float shifted = multiply_separately(x[at], stats.second) + bias;
                y[at] = multiply_separately(shifted, weights[i]) + biases[i];
                if (contracted_bf16) {
                    const float normalized_fma = std::fma(x[at], stats.second, bias);
                    y[at] = std::fma(normalized_fma, weights[i], biases[i]);
                }
            }
        }
    }
    round_bf16(y);
    return finite(y);
}
static float linspace(int i, int count, int end) {
    if (count == 1) return 0;
    const float step = float(end) / float(count - 1);
    return i < count / 2 ? step * i : float(end) - step * (count - i - 1);
}
// The pinned Transformers helper forms F32 index*(side-1)/(size-1).
// Keep the existing linspace path unchanged behind the default-off alternative.
static float position_coordinate(int index, int size, int side, bool source_math) {
    if (!source_math) return linspace(index, size, side - 1);
    return (float(index) * float(side - 1)) / float(std::max(size - 1, 1));
}
struct position_taps {
    int index[4];
    float weight[4];
};
static position_taps interpolation_taps(int row, int column, int height, int width, int side, bool source_math) {
    const float r = position_coordinate(row, height, side, source_math),
                col = position_coordinate(column, width, side, source_math);
    const int r0 = int(r), c0 = int(col), r1 = std::min(r0 + 1, side - 1), c1 = std::min(c0 + 1, side - 1);
    const float dr = r - r0, dc = col - c0;
    position_taps taps = { { r0 * side + c0, r0 * side + c1, r1 * side + c0, r1 * side + c1 },
                           { (1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc } };
    if (source_math) {
        // Original linear hat kernel, including the F32 operation order.
        const float rw[2] = { std::max(1.0f - std::abs(dr), 0.0f), std::max(1.0f - std::abs(dr - 1.0f), 0.0f) };
        const float cw[2] = { std::max(1.0f - std::abs(dc), 0.0f), std::max(1.0f - std::abs(dc - 1.0f), 0.0f) };
        for (int i = 0; i < 4; ++i) taps.weight[i] = rw[i / 2] * cw[i % 2];
    }
    return taps;
}
static bool interpolate_position(const std::vector<float> & table, const vision_config & c,
                                 const std::vector<int> & rows, const std::vector<int> & columns,
                                 const std::vector<int> & heights, const std::vector<int> & widths,
                                 std::vector<float> & output) {
    if (rows.empty() || rows.size() != columns.size() || rows.size() != heights.size() ||
        rows.size() != widths.size() || table.size() != size_t(c.position_side) * c.position_side * c.width)
        return false;
    const bool source_math = false; // Pinned original uses torch.linspace, not the later interpolation alternative.
    output.resize(rows.size() * c.width);
    for (size_t t = 0; t < rows.size(); ++t) {
        const auto taps = interpolation_taps(rows[t], columns[t], heights[t], widths[t], c.position_side, source_math);
        for (int i = 0; i < c.width; ++i) {
            const float a = lighton4b_bf16(table[size_t(taps.index[0]) * c.width + i] * lighton4b_bf16(taps.weight[0])),
                        b = lighton4b_bf16(table[size_t(taps.index[1]) * c.width + i] * lighton4b_bf16(taps.weight[1]));
            const float e = lighton4b_bf16(table[size_t(taps.index[2]) * c.width + i] * lighton4b_bf16(taps.weight[2])),
                        f = lighton4b_bf16(table[size_t(taps.index[3]) * c.width + i] * lighton4b_bf16(taps.weight[3]));
            output[t * c.width + i] = lighton4b_bf16(lighton4b_bf16(lighton4b_bf16(a + b) + e) + f);
        }
    }
    return finite(output);
}
static bool geometry(const vision_config & c, int tokens, const std::vector<int> & grid, std::vector<int> & rows,
                     std::vector<int> & columns, std::vector<int> & offsets, std::vector<int> & heights,
                     std::vector<int> & widths) {
    if (grid.empty() || grid.size() % 3 || grid.size() > 3 * 16) return false;
    int64_t total = 0;
    offsets.push_back(0);
    for (size_t i = 0; i < grid.size(); i += 3) {
        const int t = grid[i], h = grid[i + 1], w = grid[i + 2];
        if (t < 1 || t > 16 || h < 1 || h > 1024 || w < 1 || w > 1024 || h % c.merge || w % c.merge ||
            int64_t(h) * w > ovisocr2_patch_limit())
            return false;
        total += int64_t(t) * h * w;
        if (total > tokens) return false;
        for (int frame = 0; frame < t; ++frame) {
            for (int br = 0; br < h / c.merge; ++br)
                for (int bc = 0; bc < w / c.merge; ++bc)
                    for (int r = 0; r < c.merge; ++r)
                        for (int col = 0; col < c.merge; ++col) {
                            rows.push_back(br * c.merge + r);
                            columns.push_back(bc * c.merge + col);
                            heights.push_back(h);
                            widths.push_back(w);
                        }
            offsets.push_back(int(rows.size()));
        }
    }
    return total == tokens;
}
// Adapted from SLEEF xexpf, revision 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors 2010 - 2024.
// Boost Software License 1.0: core/third_party/sleef/LICENSE.txt.
// Explicit scalar FMAs reproduce its AVX2 vector arithmetic; no Torch dependency.
static float source_avx2_exp(float x) {
    if (x < -104.0f) return 0.0f;
    const float scaled = x * 1.442695040888963407359924681001892137426645954152985934135449406931f;
    const int q = int(std::nearbyint(scaled));
    float s = std::fma(float(q), -0.693145751953125f, x);
    s = std::fma(float(q), -1.428606765330187045e-06f, s);
    float u = 0.000198527617612853646278381f;
    u = std::fma(u, s, 0.00139304355252534151077271f);
    u = std::fma(u, s, 0.00833336077630519866943359f);
    u = std::fma(u, s, 0.0416664853692054748535156f);
    u = std::fma(u, s, 0.166666671633720397949219f);
    u = std::fma(u, s, 0.5f);
    const float square = s * s;
    u = 1.0f + std::fma(square, u, s);
    // Arithmetic right shift in the source is floor(q / 2), including negatives.
    const int half = q >= 0 ? q / 2 : -((-q + 1) / 2);
    u = u * std::ldexp(1.0f, half);
    return u * std::ldexp(1.0f, q - half);
}
static bool source_avx2_softmax_supported() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && std::fegetround() == FE_TONEAREST;
#else
    return false;
#endif
}
// Supported source shape only. An unsupported request keeps the previous path.
static bool source_avx2_softmax(float * scores, int count, float & inverse) {
    if (count < 8 || count % 8 || !source_avx2_softmax_supported()) return false;
    float maximum = -INFINITY;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(scores[i])) return false;
        maximum = std::max(maximum, scores[i]);
    }
    std::array<float, 8> sum{};
    for (int i = 0; i < count; ++i) {
        scores[i] = source_avx2_exp(scores[i] - maximum);
        sum[i % 8] = sum[i % 8] + scores[i];
    }
    const float total = ((sum[0] + sum[4]) + (sum[2] + sum[6])) + ((sum[1] + sum[5]) + (sum[3] + sum[7]));
    inverse = 1.0f / total;
    return true;
}
static std::vector<float> attention(const std::vector<float> & qkv, const vision_config & c, int tokens,
                                    const std::vector<int> & offsets, const std::vector<float> & cosine,
                                    const std::vector<float> & sine) {
    const int width = c.width, d = width / c.heads, half = d / 2;
    std::vector<float> q(size_t(tokens) * width), k(q.size()), values(q.size()), pooled(q.size());
    for (int t = 0; t < tokens; ++t)
        for (int head = 0; head < c.heads; ++head) {
            const size_t base = size_t(t) * width + head * d;
            const size_t src = size_t(t) * 3 * width + head * d;
            std::copy_n(qkv.data() + src, d, q.data() + base);
            std::copy_n(qkv.data() + src + width, d, k.data() + base);
            std::copy_n(qkv.data() + src + 2 * width, d, values.data() + base);
            for (int i = 0; i < half; ++i) {
                const float co = cosine[size_t(t) * d + i], si = sine[size_t(t) * d + i];
                for (auto * v : { &q, &k }) {
                    const float left = (*v)[base + i], right = (*v)[base + i + half];
                    (*v)[base + i] = lighton4b_bf16(left * co - right * si);
                    (*v)[base + i + half] = lighton4b_bf16(right * co + left * si);
                }
            }
        }
    const float scale = 1.0f / std::sqrt(float(d));
    const bool precise = true;
    const bool source_order = core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION");
    const auto normal_operand = [](float x) {
        const float a = std::abs(x);
        return a == 0 || (a >= 0x1p-63f && a <= 0x1p63f);
    };
    const bool safe_qk =
        std::all_of(q.begin(), q.end(), normal_operand) && std::all_of(k.begin(), k.end(), normal_operand);
    if (source_order && d % 64) return {};
    for (size_t frame = 1; frame < offsets.size(); ++frame) {
        const int first = offsets[frame - 1], end = offsets[frame];
        if (source_order && (end - first) % 4) return {};
        std::vector<float> scores(size_t(end - first));
        for (int t = first; t < end; ++t)
            for (int head = 0; head < c.heads; ++head) {
                const size_t base = size_t(t) * width + head * d;
                float maximum = -INFINITY;
                for (int s = first; s < end; ++s) {
                    float dot = 0;
                    for (int i = 0; i < d; ++i) dot += q[base + i] * k[size_t(s) * width + head * d + i];
                    if (source_order) {
                        const float * key = k.data() + size_t(s) * width + head * d;
                        dot = safe_qk ? source_bf16_dot<true>(q.data() + base, key, d)
                                      : source_bf16_dot<false>(q.data() + base, key, d);
                    } else if (precise) {
                        double exact = 0;
                        for (int i = 0; i < d; ++i)
                            exact += double(q[base + i]) * double(k[size_t(s) * width + head * d + i]);
                        dot = float(exact);
                    }
                    scores[s - first] = lighton4b_bf16(lighton4b_bf16(dot) * scale);
                    maximum = std::max(maximum, scores[s - first]);
                }
                float inverse = 0, sum = 0;
                const bool exact_softmax = source_order &&
                                           core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2") &&
                                           source_avx2_softmax(scores.data(), int(scores.size()), inverse);
                if (!exact_softmax) {
                    for (float & score : scores) {
                        score = std::exp(score - maximum);
                        sum += score;
                    }
                    double exact_sum = 0;
                    if (precise)
                        for (float score : scores) exact_sum += double(score);
                    inverse = precise ? 1.0f / float(exact_sum) : 0.0f;
                }
                for (int i = 0; i < d; ++i) {
                    float result = 0;
                    for (int s = first; s < end; ++s)
                        result += (scores[s - first] / sum) * values[size_t(s) * width + head * d + i];
                    if (source_order) {
                        std::array<float, 4> partial{};
                        for (int s = first; s < end; s += 4)
                            for (int lane = 0; lane < 4; ++lane) {
                                const float p = lighton4b_bf16(scores[s + lane - first] * inverse);
                                partial[lane] =
                                    std::fma(p, values[size_t(s + lane) * width + head * d + i], partial[lane]);
                            }
                        result = ((partial[0] + partial[1]) + partial[2]) + partial[3];
                    } else if (precise) {
                        double exact = 0;
                        for (int s = first; s < end; ++s) {
                            const float probability = lighton4b_bf16(scores[s - first] * inverse);
                            exact += double(probability) * double(values[size_t(s) * width + head * d + i]);
                        }
                        result = float(exact);
                    }
                    pooled[base + i] = lighton4b_bf16(result);
                }
            }
    }
    return pooled;
}
bool lighton4b_vision_bf16_cpu(const vision_weights & w, const vision_config & c, const float * patches, int tokens,
                               const std::vector<int> & grid, int threads, vision_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16") || c.width != 1024 || c.heads != 16 || c.output != 2560 ||
        c.intermediate != 4096 || c.patch_width != 1536 || c.merge != 2 || c.position_side != 48 ||
        w.layers.size() != 24 || c.eps != 1e-6f || c.theta != 10000.0f || !core_env::on("CRISPEMBED_QWEN35_OCR") ||
        !patches || tokens < 1 || tokens > 4096 || threads < 1 || threads > 256 || c.width < 1 || c.width > 2048 ||
        c.heads < 1 || c.heads > 32 || c.width % c.heads || (c.width / c.heads) % 4 || c.merge < 1 || c.merge > 4 ||
        c.patch_width < 1 || c.patch_width > 4096 || c.position_side < 1 || c.position_side > 128 || c.output < 1 ||
        c.output > 8192 || c.intermediate < 1 || c.intermediate > 8192 || w.layers.empty() || w.layers.size() > 32 ||
        !std::isfinite(c.eps) || c.eps <= 0 || !std::isfinite(c.theta) || c.theta <= 1)
        return false;
    const int width = c.width, d = width / c.heads, group = c.merge * c.merge;
    if (!shape(w.patch, c.patch_width, width) || !shape(w.position, width, c.position_side * c.position_side) ||
        !shape(w.merger_up, width * group, width * group) || !shape(w.merger_down, width * group, c.output))
        return false;
    std::vector<int> rows, columns, offsets, heights, widths;
    if (!geometry(c, tokens, grid, rows, columns, offsets, heights, widths)) return false;
    std::vector<float> input(patches, patches + size_t(tokens) * c.patch_width);
    round_bf16(input);
    if (!linear(w.patch, w.patch_bias, input, c.patch_width, tokens, threads, output.patch, false)) return false;
    std::vector<float> table(size_t(c.position_side) * c.position_side * width);
    if (!tensor_floats(w.position, table.data())) return false;
    if (!interpolate_position(table, c, rows, columns, heights, widths, output.position)) return false;
    output.rotary_cos.resize(size_t(tokens) * d);
    output.rotary_sin.resize(output.rotary_cos.size());
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < d / 2; ++i) {
            const int axis = i / (d / 4), frequency = i % (d / 4);
            const float inv = 1.0f / std::pow(c.theta, float(2 * frequency) / float(d / 2));
            const float angle = inv * (axis == 0 ? rows[t] : columns[t]);
            for (int half = 0; half < 2; ++half) {
                const size_t at = size_t(t) * d + i + half * (d / 2);
                output.rotary_cos[at] = std::cos(angle);
                output.rotary_sin[at] = std::sin(angle);
            }
        }
    }
    output.embedded = output.patch;
    for (size_t i = 0; i < output.embedded.size(); ++i)
        output.embedded[i] = lighton4b_bf16(output.embedded[i] + output.position[i]);
    auto hidden = output.embedded;
    output.probes.resize(w.layers.size());
    for (size_t layer_index = 0; layer_index < w.layers.size(); ++layer_index) {
        const auto & layer = w.layers[layer_index];
        const bool capture = layer_index == 5 || layer_index == 11;
        auto & probe = output.probes[layer_index];
        if (!shape(layer.qkv, width, 3 * width) || !shape(layer.out, width, width) ||
            !shape(layer.up, width, c.intermediate) || !shape(layer.down, c.intermediate, width))
            return false;
        std::vector<float> normalized, projected, mixed;
        if (!layer_norm(hidden, layer.norm1, layer.norm1_bias, width, c.eps, normalized) ||
            !linear(layer.qkv, layer.qkv_bias, normalized, width, tokens, threads, projected))
            return false;
        if (capture) {
            probe.norm1 = normalized;
            probe.qkv = projected;
        }
        const auto pooled = attention(projected, c, tokens, offsets, output.rotary_cos, output.rotary_sin);
        if (!linear(layer.out, layer.out_bias, pooled, width, tokens, threads, mixed)) return false;
        if (capture) {
            probe.pooled = pooled;
            probe.attention = mixed;
        }
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = lighton4b_bf16(hidden[i] + mixed[i]);
        if (!layer_norm(hidden, layer.norm2, layer.norm2_bias, width, c.eps, normalized) ||
            !linear(layer.up, layer.up_bias, normalized, width, tokens, threads, projected))
            return false;
        if (capture) {
            probe.norm2 = normalized;
            probe.up = projected;
        }
        for (float & x : projected)
            x = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        round_bf16(projected);
        if (capture) probe.activation = projected;
        if (!linear(layer.down, layer.down_bias, projected, c.intermediate, tokens, threads, mixed)) return false;
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = lighton4b_bf16(hidden[i] + mixed[i]);
        if (capture) probe.down = mixed;
        if (!finite(hidden)) return false;
        output.hidden.push_back(hidden);
    }
    if (!layer_norm(hidden, w.merger_norm, w.merger_norm_bias, width, c.eps, output.merger_norm) ||
        !linear(w.merger_up, w.merger_up_bias, output.merger_norm, width * group, tokens / group, threads,
                output.merger_up))
        return false;
    output.merger_activation = output.merger_up;
    for (float & x : output.merger_activation) x = 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
    round_bf16(output.merger_activation);
    return linear(w.merger_down, w.merger_down_bias, output.merger_activation, width * group, tokens / group, threads,
                  output.merged);
}
} // namespace qwen35_ocr
