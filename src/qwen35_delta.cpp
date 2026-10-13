#include "qwen35_delta.h"
#include "core/env_gate.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace qwen35_ocr {
static float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    const float exp = std::exp(x);
    return exp / (1.0f + exp);
}

bool linear_core_cpu(const linear_inputs & in, linear_outputs & out) {
    out = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !in.qkv || !in.z || !in.a || !in.b || !in.conv_weights || !in.a_log ||
        !in.dt_bias || !in.norm_weights || in.batch < 1 || in.batch > 4 || in.tokens < 1 || in.tokens > 4096 ||
        in.heads < 1 || in.heads > 128 || in.width < 1 || in.width > 256 || in.conv_kernel < 1 || in.conv_kernel > 16 ||
        in.threads < 1 || in.threads > 256 || !std::isfinite(in.norm_eps) || in.norm_eps <= 0)
        return false;
    const size_t heads = size_t(in.heads), width = size_t(in.width);
    const size_t group = heads * width, channels = 3 * group;
    const size_t rows = size_t(in.batch) * in.tokens * heads;
    const size_t elements = rows * width;
    if (elements > 16 * 1024 * 1024 || size_t(in.batch) * heads * width * width > 16 * 1024 * 1024) return false;
    std::vector<float> q(elements), k(elements), v(elements), g(rows), beta(rows), delta;
    out.conv_history.resize(size_t(in.batch) * channels * in.conv_kernel);
    auto history = [&](int batch, size_t channel, int token) {
        if (token >= 0) return in.qkv[(size_t(batch) * in.tokens + token) * channels + channel];
        return in.conv_history
                   ? in.conv_history[(size_t(batch) * channels + channel) * in.conv_kernel + in.conv_kernel + token]
                   : 0.0f;
    };
    for (int batch = 0; batch < in.batch; ++batch) {
        for (int token = 0; token < in.tokens; ++token) {
            for (size_t channel = 0; channel < channels; ++channel) {
                float sum = 0;
                for (int tap = 0; tap < in.conv_kernel; ++tap) {
                    sum += history(batch, channel, token - in.conv_kernel + 1 + tap) *
                           in.conv_weights[channel * in.conv_kernel + tap];
                }
                const size_t index = (size_t(batch) * in.tokens + token) * group + channel % group;
                auto & destination = channel < group ? q : channel < 2 * group ? k : v;
                destination[index] = sum * sigmoid(sum);
            }
            for (size_t head = 0; head < heads; ++head) {
                const size_t index = (size_t(batch) * in.tokens + token) * heads + head;
                const float time = in.a[index] + in.dt_bias[head];
                const float softplus = time > 20.0f ? time : std::log1p(std::exp(time));
                g[index] = -std::exp(in.a_log[head]) * softplus;
                beta[index] = sigmoid(in.b[index]);
            }
        }
        for (size_t channel = 0; channel < channels; ++channel) {
            for (int tap = 0; tap < in.conv_kernel; ++tap) {
                out.conv_history[(size_t(batch) * channels + channel) * in.conv_kernel + tap] =
                    history(batch, channel, in.tokens - in.conv_kernel + tap);
            }
        }
    }
    if (!delta_cpu(q.data(), k.data(), v.data(), g.data(), beta.data(), in.state, in.batch, in.tokens, in.heads,
                   in.width, in.threads, delta, out.state)) {
        out = {};
        return false;
    }
    out.values.resize(elements);
    for (size_t row = 0; row < rows; ++row) {
        double square = 0;
        for (size_t column = 0; column < width; ++column) {
            const float value = delta[row * width + column];
            square += double(value) * value;
        }
        const float scale = 1.0f / std::sqrt(float(square / width) + in.norm_eps);
        for (size_t column = 0; column < width; ++column) {
            const size_t index = row * width + column;
            out.values[index] = delta[index] * scale * in.norm_weights[column] * in.z[index] * sigmoid(in.z[index]);
            if (!std::isfinite(out.values[index])) {
                out = {};
                return false;
            }
        }
    }
    return true;
}

bool delta_cpu(const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * initial_state, int batch, int tokens, int heads, int width, int threads,
               std::vector<float> & output, std::vector<float> & final_state) {
    output.clear();
    final_state.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !q || !k || !v || !g || !beta || batch < 1 || tokens < 1 ||
        heads < 1 || width < 1 || threads < 1 || threads > 256)
        return false;
    // Bounded experimental CPU boundary; fail before multiplication/allocation.
    if (batch > 4 || tokens > 4096 || heads > 128 || width > 256) return false;
    const size_t rows = size_t(batch) * tokens * heads;
    const size_t elements = rows * width;
    const size_t states = size_t(batch) * heads * width * width;
    if (elements > 16 * 1024 * 1024 || states > 16 * 1024 * 1024) return false;
    const size_t memory = 1024 * 1024 + sizeof(float) * (12 * elements + 3 * states + 16 * rows);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ memory, nullptr, false }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    auto * context = ctx.get();
    auto input = [&](const float * values) {
        auto * tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, width, heads, tokens, batch);
        std::memcpy(tensor->data, values, elements * sizeof(float));
        return tensor;
    };
    auto normalize = [&](ggml_tensor * tensor) {
        // Official l2norm: rsqrt(sum(x*x) + 1e-6), not mean-based RMSNorm.
        auto * epsilon = ggml_new_f32(context, 1e-6f);
        auto * denominator =
            ggml_sqrt(context, ggml_add1(context, ggml_sum_rows(context, ggml_sqr(context, tensor)), epsilon));
        return ggml_div(context, tensor, denominator);
    };
    auto * query = normalize(input(q));
    auto * key = normalize(input(k));
    auto * value = input(v);
    auto * decay = ggml_new_tensor_4d(context, GGML_TYPE_F32, 1, heads, tokens, batch);
    auto * mixing = ggml_new_tensor_4d(context, GGML_TYPE_F32, 1, heads, tokens, batch);
    std::memcpy(decay->data, g, rows * sizeof(float));
    std::memcpy(mixing->data, beta, rows * sizeof(float));
    auto * state = ggml_new_tensor_4d(context, GGML_TYPE_F32, width, width, heads, batch);
    auto * state_data = static_cast<float *>(state->data);
    for (size_t head = 0; head < size_t(batch) * heads; ++head) {
        for (int key_axis = 0; key_axis < width; ++key_axis) {
            for (int value_axis = 0; value_axis < width; ++value_axis) {
                state_data[(head * width + value_axis) * width + key_axis] =
                    initial_state ? initial_state[(head * width + key_axis) * width + value_axis] : 0.0f;
            }
        }
    }
    // ggml applies 1/sqrt(D) to the query dot internally. Do not scale twice.
    auto * result = ggml_gated_delta_net(context, query, key, value, decay, mixing, state, 1);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, result);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    auto * data = static_cast<float *>(result->data);
    if (!std::all_of(data, data + elements + states, [](float x) { return std::isfinite(x); })) return false;
    output.assign(data, data + elements);
    final_state.resize(states);
    const float * packed_state = data + elements;
    for (size_t head = 0; head < size_t(batch) * heads; ++head) {
        for (int key_axis = 0; key_axis < width; ++key_axis) {
            for (int value_axis = 0; value_axis < width; ++value_axis) {
                final_state[(head * width + key_axis) * width + value_axis] =
                    packed_state[(head * width + value_axis) * width + key_axis];
            }
        }
    }
    return true;
}
} // namespace qwen35_ocr
