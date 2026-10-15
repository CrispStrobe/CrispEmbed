#pragma once
// Validation-only scalar trace of the hash-pinned native attention arithmetic.
// Every use must first reproduce the linked production pooled output exactly.
#include "qwen35_vision.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <vector>
namespace qwen35_attention_audit {
struct trace_result {
    std::vector<float> query, key, value, scores, probabilities, pooled_sample, pooled;
};
static trace_result trace(const std::vector<float> & qkv, const qwen35_ocr::vision_config & c, int tokens,
                          const std::vector<float> & cosine, const std::vector<float> & sine,
                          const std::vector<int> & query_ids, const trace_result * original_rotated = nullptr,
                          bool pin_cosine_fma = true) {
    const std::vector<int> offsets{ 0, tokens };
    const int width = c.width, d = width / c.heads, half = d / 2;
    std::vector<float> q(size_t(tokens) * width), k(q.size()), values(q.size()), pooled(q.size());
    trace_result result;
    result.scores.resize(size_t(c.heads) * query_ids.size() * tokens);
    result.probabilities.resize(result.scores.size());
    std::vector<int> query_lookup(tokens, -1);
    for (size_t i = 0; i < query_ids.size(); ++i) query_lookup[query_ids[i]] = int(i);
    if (original_rotated) {
        q = original_rotated->query;
        k = original_rotated->key;
        values = original_rotated->value;
    } else
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
                        if (pin_cosine_fma) {
                            // Match the actual GCC13 AArch64 production object:
                            // sine products round first; cosine multiply/add is fused.
                            const float right_sine = right * si, left_sine = left * si;
                            (*v)[base + i] = std::fma(left, co, -right_sine);
                            (*v)[base + i + half] = std::fma(right, co, left_sine);
                        } else {
                            (*v)[base + i] = left * co - right * si;
                            (*v)[base + i + half] = right * co + left * si;
                        }
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
                    // The actual production object uses ordered scalar FMA.
                    // An unconstrained copy vectorizes rounded products instead.
                    for (int i = 0; i < d; ++i) dot = std::fma(q[base + i], k[size_t(s) * width + head * d + i], dot);
                    if (precise) {
                        double exact = 0;
                        for (int i = 0; i < d; ++i)
                            exact += double(q[base + i]) * double(k[size_t(s) * width + head * d + i]);
                        dot = float(exact);
                    }
                    scores[s - first] = dot * scale;
                    maximum = std::max(maximum, scores[s - first]);
                }
                const int selected = query_lookup[t];
                const size_t row = (size_t(head) * query_ids.size() + (selected < 0 ? 0 : selected)) * tokens;
                if (selected >= 0) std::copy(scores.begin(), scores.end(), result.scores.begin() + row);
                float sum = 0;
                for (float & score : scores) {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                double exact_sum = 0;
                if (precise)
                    for (float score : scores) exact_sum += double(score);
                const float inverse = precise ? 1.0f / float(exact_sum) : 0.0f;
                if (selected >= 0)
                    for (int s = first; s < end; ++s)
                        result.probabilities[row + s] = precise ? scores[s - first] * inverse : scores[s - first] / sum;
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
    result.query = std::move(q);
    result.key = std::move(k);
    result.value = std::move(values);
    result.pooled = std::move(pooled);
    result.pooled_sample.resize(query_ids.size() * width);
    for (size_t i = 0; i < query_ids.size(); ++i)
        std::copy_n(result.pooled.data() + size_t(query_ids[i]) * width, width,
                    result.pooled_sample.data() + i * width);
    return result;
}
static std::vector<float> original_probabilities_pool(const trace_result & original, int tokens,
                                                      const qwen35_ocr::vision_config & c,
                                                      const std::vector<int> & query_ids) {
    const int d = c.width / c.heads;
    const bool precise = core_env::on("CRISPEMBED_QWEN35_VISION_PRECISE_ATTN");
    std::vector<float> pooled(query_ids.size() * c.width);
    for (size_t t = 0; t < query_ids.size(); ++t)
        for (int head = 0; head < c.heads; ++head)
            for (int i = 0; i < d; ++i) {
                float value = 0;
                double exact = 0;
                for (int s = 0; s < tokens; ++s) {
                    const float probability =
                        original.probabilities[(size_t(head) * query_ids.size() + t) * tokens + s];
                    const float v = original.value[size_t(s) * c.width + head * d + i];
                    value += probability * v;
                    if (precise) exact += double(probability) * double(v);
                }
                pooled[t * c.width + head * d + i] = precise ? float(exact) : value;
            }
    return pooled;
}
} // namespace qwen35_attention_audit
