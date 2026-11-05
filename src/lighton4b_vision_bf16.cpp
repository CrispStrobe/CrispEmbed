#include "qwen35_image_bounds.h"
#include "lighton4b_vision_bf16.h"
#include "qwen35_math.h"
#include "qwen35_norm.h"
#include "core/env_gate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>

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
static std::pair<float, float> row_bf16_avx2(const float * x, int width, float eps) {
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
    for (int lane = 0; lane < lanes; ++lane) merge(total, stack[0][lane], false);
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
static bool linear(ggml_tensor * w, ggml_tensor * b, const std::vector<float> & x, int width, int rows, int threads,
                   std::vector<float> & y) {
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
    const bool source_bf16 = core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_NORM");
    for (int token = 0; token < int(x.size() / width); ++token) {
        const auto stats = source_bf16 ? row_bf16_avx2(x.data() + size_t(token) * width, width, eps)
                                       : row(x.data() + size_t(token) * width, width, eps);
        for (int i = 0; i < width; ++i) {
            const size_t at = size_t(token) * width + i;
            const float normalized = (x[at] - stats.first) * stats.second;
            y[at] = std::fma(normalized, weights[i], biases[i]);
            if (source_bf16) {
                const float bias = -stats.second * stats.first;
                const float shifted = multiply_separately(x[at], stats.second) + bias;
                y[at] = multiply_separately(shifted, weights[i]) + biases[i];
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
    for (size_t frame = 1; frame < offsets.size(); ++frame) {
        const int first = offsets[frame - 1], end = offsets[frame];
        std::vector<float> scores(size_t(end - first));
        for (int t = first; t < end; ++t)
            for (int head = 0; head < c.heads; ++head) {
                const size_t base = size_t(t) * width + head * d;
                float maximum = -INFINITY;
                for (int s = first; s < end; ++s) {
                    float dot = 0;
                    for (int i = 0; i < d; ++i) dot += q[base + i] * k[size_t(s) * width + head * d + i];
                    if (precise) {
                        double exact = 0;
                        for (int i = 0; i < d; ++i)
                            exact += double(q[base + i]) * double(k[size_t(s) * width + head * d + i]);
                        dot = float(exact);
                    }
                    scores[s - first] = lighton4b_bf16(lighton4b_bf16(dot) * scale);
                    maximum = std::max(maximum, scores[s - first]);
                }
                float sum = 0;
                for (float & score : scores) {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                double exact_sum = 0;
                if (precise)
                    for (float score : scores) exact_sum += double(score);
                const float inverse = precise ? 1.0f / float(exact_sum) : 0.0f;
                for (int i = 0; i < d; ++i) {
                    float result = 0;
                    for (int s = first; s < end; ++s)
                        result += (scores[s - first] / sum) * values[size_t(s) * width + head * d + i];
                    if (precise) {
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
    if (!linear(w.patch, w.patch_bias, input, c.patch_width, tokens, threads, output.patch)) return false;
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
