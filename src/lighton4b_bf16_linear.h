#pragma once
#include "qwen35_lighton4b_text.h"
#include "qwen35_math.h"
#include <cstdint>
#include <cstring>

namespace qwen35_ocr {
// Experimental activation semantics only. Quantized weight bytes and independent
// F32 dequantization stay unchanged; the legacy path never calls this helper.
inline float lighton4b_bf16_round(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        bits |= 0x00400000U;
    else
        bits += 0x7fffU + ((bits >> 16) & 1U);
    bits &= 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}
inline void lighton4b_bf16_round(std::vector<float> & values) {
    for (auto & value : values) value = lighton4b_bf16_round(value);
}
inline bool lighton4b_bf16_linear_enabled() {
    return core_env::on("CRISPEMBED_LIGHTON4B_BF16_LINEAR");
}
inline bool lighton4b_bf16_grouped_cpu(const linear_weights & w, const float * hidden, int tokens, int threads,
                                       const float * history, const float * state, projected_linear_outputs & out) {
    out = {};
    if (!lighton4b_bf16_linear_enabled() || !hidden || tokens < 1 || tokens > 16) return false;
    constexpr int keys = 16, heads = 32, width = 128, channels = 8192, group = 4096, kernel = 4;
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ w.qkv, w.z, w.a, w.b }, hidden, 2560, tokens, threads, projections)) return false;
    out.qkv = std::move(projections[0]);
    out.z = std::move(projections[1]);
    out.a = std::move(projections[2]);
    out.b = std::move(projections[3]);
    for (auto * values : { &out.qkv, &out.z, &out.a, &out.b }) lighton4b_bf16_round(*values);
    std::vector<float> conv(channels * kernel), a_log(heads), dt(heads), norm(width);
    if (!tensor_floats(w.conv, conv.data()) || !tensor_floats(w.a_log, a_log.data()) ||
        !tensor_floats(w.dt_bias, dt.data()) || !tensor_floats(w.norm, norm.data()))
        return false;
    auto sigmoid = [](float x) { return x >= 0 ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x)); };
    auto source = [&](int channel, int token) {
        return token >= 0 ? out.qkv[size_t(token) * channels + channel]
               : history  ? history[size_t(channel) * kernel + kernel + token]
                          : 0.0f;
    };
    std::vector<float> q(size_t(tokens) * group), k(q.size()), v(q.size()), g(size_t(tokens) * heads), beta(g.size()),
        delta;
    for (int token = 0; token < tokens; ++token) {
        for (int channel = 0; channel < channels; ++channel) {
            float sum = 0;
            for (int tap = 0; tap < kernel; ++tap)
                sum += source(channel, token - kernel + 1 + tap) * conv[size_t(channel) * kernel + tap];
            sum = lighton4b_bf16_round(sum);
            const float value = lighton4b_bf16_round(sum * sigmoid(sum));
            if (channel < 4096) {
                const int compact = channel % 2048, head = compact / width, lane = compact % width;
                auto & dst = channel < 2048 ? q : k;
                for (int repeat = 0; repeat < 2; ++repeat)
                    dst[size_t(token) * group + (head * 2 + repeat) * width + lane] = value;
            } else
                v[size_t(token) * group + channel - 4096] = value;
        }
        for (int head = 0; head < heads; ++head) {
            const size_t i = size_t(token) * heads + head;
            const float time = out.a[i] + dt[head];
            g[i] = -std::exp(a_log[head]) * (time > 20 ? time : std::log1p(std::exp(time)));
            beta[i] = lighton4b_bf16_round(sigmoid(out.b[i]));
        }
    }
    // HF l2norm executes in the Q/K input dtype *before* the F32 recurrence.
    for (auto * values : { &q, &k })
        for (int row = 0; row < tokens * heads; ++row) {
            double square = 0;
            for (int lane = 0; lane < width; ++lane) {
                const float x = (*values)[size_t(row) * width + lane];
                square += lighton4b_bf16_round(x * x);
            }
            const float total = lighton4b_bf16_round(lighton4b_bf16_round(float(square)) + 1e-6f);
            const float inverse = lighton4b_bf16_round(1.0f / std::sqrt(total));
            for (int lane = 0; lane < width; ++lane) {
                auto & x = (*values)[size_t(row) * width + lane];
                x = lighton4b_bf16_round(x * inverse);
            }
        }
    if (!delta_pre_normalized_cpu(q.data(), k.data(), v.data(), g.data(), beta.data(), state, 1, tokens, heads, width,
                                  threads, delta, out.core.state))
        return false;
    lighton4b_bf16_round(delta);
    out.core.values.resize(delta.size());
    for (int row = 0; row < tokens * heads; ++row) {
        const size_t base = size_t(row) * width;
        const float scale = rms_scale_cpu(delta.data() + base, width, 1e-6f);
        for (int lane = 0; lane < width; ++lane) {
            const size_t i = base + lane;
            const float normalized = lighton4b_bf16_round(delta[i] * scale);
            const float weighted = lighton4b_bf16_round(normalized * norm[lane]);
            out.core.values[i] = lighton4b_bf16_round(weighted * (out.z[i] * sigmoid(out.z[i])));
        }
    }
    out.core.conv_history.resize(channels * kernel);
    for (int channel = 0; channel < channels; ++channel)
        for (int tap = 0; tap < kernel; ++tap)
            out.core.conv_history[size_t(channel) * kernel + tap] = source(channel, tokens - kernel + tap);
    projections.clear();
    if (!project_cpu({ w.out }, out.core.values.data(), group, tokens, threads, projections)) return false;
    out.attention = std::move(projections[0]);
    lighton4b_bf16_round(out.attention);
    return true;
}
} // namespace qwen35_ocr
