#include "qwen35_image_bounds.h"
#include "qwen35_vision.h"
#include "qwen35_math.h"
#include "qwen35_norm.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace qwen35_ocr {
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
    return bias(y, b, int(w->ne[1]));
}
static bool layer_norm(const std::vector<float> & x, ggml_tensor * w, ggml_tensor * b, int width, float eps,
                       std::vector<float> & y) {
    if (!shape(w, width, 1) || !shape(b, width, 1)) return false;
    std::vector<float> weights(width), biases(width);
    if (!tensor_floats(w, weights.data()) || !tensor_floats(b, biases.data())) return false;
    y.resize(x.size());
    if (width == 768 && eps == 1e-6f && core_env::on("CRISPEMBED_QWEN35_VISION_CASCADE_NORM"))
        return cascade_layer_norm(x.data(), int(x.size() / width), width, eps, weights.data(), biases.data(), y.data());
    for (size_t base = 0; base < x.size(); base += width) {
        float mean = 0, variance = 0;
        for (int i = 0; i < width; ++i) mean += x[base + i];
        mean /= width;
        for (int i = 0; i < width; ++i) {
            const float centered = x[base + i] - mean;
            variance += centered * centered;
        }
        variance /= width;
        if (core_env::on("CRISPEMBED_QWEN35_VISION_PRECISE_NORM")) {
            double sum = 0, square = 0;
            for (int i = 0; i < width; ++i) sum += double(x[base + i]);
            const double center = sum / width;
            for (int i = 0; i < width; ++i) {
                const double delta = double(x[base + i]) - center;
                square += delta * delta;
            }
            mean = float(center);
            variance = float(square / width);
        }
        const float scale = 1.0f / std::sqrt(variance + eps);
        for (int i = 0; i < width; ++i) y[base + i] = ((x[base + i] - mean) * scale) * weights[i] + biases[i];
    }
    return finite(y);
}
bool map_small_vision(const core_gguf::tensor_map & tensors, vision_weights & weights) {
    weights = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR")) return false;
    auto get = [&](const std::string & name) { return core_gguf::try_get(tensors, name.c_str()); };
    weights.patch = get("v.patch_embed.proj.weight");
    weights.patch_bias = get("v.patch_embed.proj.bias");
    weights.position = get("v.pos_embed.weight");
    weights.merger_norm = get("v.merger.norm.weight");
    weights.merger_norm_bias = get("v.merger.norm.bias");
    weights.merger_up = get("v.merger.linear_fc1.weight");
    weights.merger_up_bias = get("v.merger.linear_fc1.bias");
    weights.merger_down = get("v.merger.linear_fc2.weight");
    weights.merger_down_bias = get("v.merger.linear_fc2.bias");
    for (int i = 0; i < 12; ++i) {
        const auto prefix = "v.blocks." + std::to_string(i) + ".";
        auto field = [&](const char * name) { return get(prefix + name); };
        weights.layers.push_back({ field("norm1.weight"), field("norm1.bias"), field("norm2.weight"),
                                   field("norm2.bias"), field("attn.qkv.weight"), field("attn.qkv.bias"),
                                   field("attn.proj.weight"), field("attn.proj.bias"), field("mlp.linear_fc1.weight"),
                                   field("mlp.linear_fc1.bias"), field("mlp.linear_fc2.weight"),
                                   field("mlp.linear_fc2.bias") });
    }
    return true;
}
static float linspace(int i, int count, int end) {
    if (count == 1) return 0;
    const float step = float(end) / float(count - 1);
    return i < count / 2 ? step * i : float(end) - step * (count - i - 1);
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
                    (*v)[base + i] = left * co - right * si;
                    (*v)[base + i + half] = right * co + left * si;
                }
            }
        }
    const float scale = 1.0f / std::sqrt(float(d));
    const bool precise = core_env::on("CRISPEMBED_QWEN35_VISION_PRECISE_ATTN");
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
                    scores[s - first] = dot * scale;
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
                            const float probability = scores[s - first] * inverse;
                            exact += double(probability) * double(values[size_t(s) * width + head * d + i]);
                        }
                        result = float(exact);
                    }
                    pooled[base + i] = result;
                }
            }
    }
    return pooled;
}
bool vision_cpu(const vision_weights & w, const vision_config & c, const float * patches, int tokens,
                const std::vector<int> & grid, int threads, vision_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !patches || tokens < 1 || tokens > 4096 || threads < 1 ||
        threads > 256 || c.width < 1 || c.width > 2048 || c.heads < 1 || c.heads > 32 || c.width % c.heads ||
        (c.width / c.heads) % 4 || c.merge < 1 || c.merge > 4 || c.patch_width < 1 || c.patch_width > 4096 ||
        c.position_side < 1 || c.position_side > 128 || c.output < 1 || c.output > 8192 || c.intermediate < 1 ||
        c.intermediate > 8192 || w.layers.empty() || w.layers.size() > 32 || !std::isfinite(c.eps) || c.eps <= 0 ||
        !std::isfinite(c.theta) || c.theta <= 1)
        return false;
    const int width = c.width, d = width / c.heads, group = c.merge * c.merge;
    if (!shape(w.patch, c.patch_width, width) || !shape(w.position, width, c.position_side * c.position_side) ||
        !shape(w.merger_up, width * group, width * group) || !shape(w.merger_down, width * group, c.output))
        return false;
    std::vector<int> rows, columns, offsets, heights, widths;
    if (!geometry(c, tokens, grid, rows, columns, offsets, heights, widths)) return false;
    std::vector<float> input(patches, patches + size_t(tokens) * c.patch_width);
    if (!linear(w.patch, w.patch_bias, input, c.patch_width, tokens, threads, output.patch)) return false;
    std::vector<float> table(size_t(c.position_side) * c.position_side * width);
    if (!tensor_floats(w.position, table.data())) return false;
    output.position.resize(size_t(tokens) * width);
    output.rotary_cos.resize(size_t(tokens) * d);
    output.rotary_sin.resize(output.rotary_cos.size());
    for (int t = 0; t < tokens; ++t) {
        const float r = linspace(rows[t], heights[t], c.position_side - 1),
                    col = linspace(columns[t], widths[t], c.position_side - 1);
        const int r0 = int(r), c0 = int(col), r1 = std::min(r0 + 1, c.position_side - 1),
                  c1 = std::min(c0 + 1, c.position_side - 1);
        const float dr = r - r0, dc = col - c0;
        const int indices[4] = { r0 * c.position_side + c0, r0 * c.position_side + c1, r1 * c.position_side + c0,
                                 r1 * c.position_side + c1 };
        const float weights[4] = { (1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc };
        for (int i = 0; i < width; ++i) {
            const float a = table[size_t(indices[0]) * width + i] * weights[0],
                        b = table[size_t(indices[1]) * width + i] * weights[1];
            const float e = table[size_t(indices[2]) * width + i] * weights[2],
                        f = table[size_t(indices[3]) * width + i] * weights[3];
            output.position[size_t(t) * width + i] = ((a + b) + e) + f;
        }
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
    for (size_t i = 0; i < output.embedded.size(); ++i) output.embedded[i] += output.position[i];
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
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += mixed[i];
        if (!layer_norm(hidden, layer.norm2, layer.norm2_bias, width, c.eps, normalized) ||
            !linear(layer.up, layer.up_bias, normalized, width, tokens, threads, projected))
            return false;
        if (capture) {
            probe.norm2 = normalized;
            probe.up = projected;
        }
        for (float & x : projected)
            x = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        if (capture) probe.activation = projected;
        if (!linear(layer.down, layer.down_bias, projected, c.intermediate, tokens, threads, mixed)) return false;
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += mixed[i];
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
    return linear(w.merger_down, w.merger_down_bias, output.merger_activation, width * group, tokens / group, threads,
                  output.merged);
}
} // namespace qwen35_ocr
