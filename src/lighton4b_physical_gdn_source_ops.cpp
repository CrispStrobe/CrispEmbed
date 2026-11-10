#include "lighton4b_physical_gdn_source_ops.h"
#include "lighton4b_upstream_physical_semantics.h"
#include "qwen35_delta.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
namespace qwen35_ocr {
namespace {
size_t tiled_head(size_t h) {
    return 16 * (h % 2) + h / 2;
}
size_t tiled_lane(size_t i) {
    return tiled_head(i / 128) * 128 + i % 128;
}
bool finite(const float * data, size_t count) {
    return data && std::all_of(data, data + count, [](float x) { return std::isfinite(x); });
}
} // namespace
bool lighton4b_physical_gdn_source_ops_cpu(const linear_inputs & in, int key_heads, linear_outputs & out) {
    out = {};
    if (!lighton4b_physical_gdn_source_ops_mode(in.a_is_pretransformed) ||
        !lighton4b_upstream_physical_mode_valid(true) || key_heads != 16 || in.heads != 32 || in.width != 128 ||
        in.batch != 1 || in.tokens < 1 || in.tokens > 16 || in.conv_kernel != 4 || in.threads < 1 || in.threads > 256 ||
        in.norm_eps != 1e-6f)
        return false;
    constexpr size_t channels = 8192, group = 4096, heads = 32, width = 128, states = 524288;
    const size_t n = size_t(in.tokens);
    if (!finite(in.qkv, n * channels) || !finite(in.z, n * group) || !finite(in.a, n * heads) ||
        !finite(in.b, n * heads) || !finite(in.conv_weights, channels * 4) || !finite(in.a_log, heads) ||
        !finite(in.dt_bias, heads) || !finite(in.norm_weights, width) ||
        (in.conv_history && !finite(in.conv_history, channels * 4)) || (in.state && !finite(in.state, states)) ||
        !std::all_of(in.a_log, in.a_log + heads, [](float x) { return x < 0.0f; }))
        return false;
    try {
        std::vector<float> mixed(n * channels), z(n * group), conv(channels * 4), history(channels * 3, 0.0f),
            decay(n * heads), beta(n * heads), state(states, 0.0f);
        for (size_t ch = 0; ch < channels; ++ch) {
            const size_t target = ch < 4096 ? ch : 4096 + tiled_lane(ch - 4096);
            for (size_t tap = 0; tap < 4; ++tap) conv[target * 4 + tap] = in.conv_weights[ch * 4 + tap];
            if (in.conv_history)
                for (size_t tap = 0; tap < 3; ++tap) history[target * 3 + tap] = in.conv_history[ch * 4 + tap + 1];
            for (size_t t = 0; t < n; ++t) mixed[t * channels + target] = in.qkv[t * channels + ch];
        }
        for (size_t h = 0; h < heads; ++h) {
            const size_t target = tiled_head(h);
            for (size_t t = 0; t < n; ++t) {
                const float time = in.a[t * heads + h] + in.dt_bias[h];
                const float softplus = time > 20.0f ? time : std::log1p(std::exp(time));
                decay[t * heads + target] = in.a_log[h] * softplus;
                beta[t * heads + target] = in.b[t * heads + h];
                for (size_t d = 0; d < width; ++d) z[t * group + target * width + d] = in.z[t * group + h * width + d];
            }
            if (in.state)
                for (size_t key = 0; key < width; ++key)
                    for (size_t value = 0; value < width; ++value)
                        state[(target * width + value) * width + key] = in.state[(h * width + key) * width + value];
        }
        if (!finite(decay.data(), decay.size())) return false;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({ 48ull * 1024 * 1024, nullptr, false }),
                                                                    ggml_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!context || !backend) return false;
        ggml_backend_cpu_set_n_threads(backend.get(), in.threads);
        auto * ctx = context.get();
        auto input = [&](const float * values, int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
            auto * tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c, d);
            std::memcpy(tensor->data, values, size_t(a * b * c * d) * sizeof(float));
            return tensor;
        };
        auto * x = input(mixed.data(), channels, n);
        auto * fifo = input(history.data(), 3, channels);
        auto * conv_input = ggml_concat(ctx, fifo, ggml_transpose(ctx, x), 0);
        auto * convolved = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, input(conv.data(), 4, channels)));
        const size_t row_bytes = channels * sizeof(float);
        auto * q =
            ggml_view_4d(ctx, convolved, width, key_heads, n, 1, width * sizeof(float), row_bytes, row_bytes * n, 0);
        auto * k = ggml_view_4d(ctx, convolved, width, key_heads, n, 1, width * sizeof(float), row_bytes, row_bytes * n,
                                2048 * sizeof(float));
        auto * v = ggml_view_4d(ctx, convolved, width, heads, n, 1, width * sizeof(float), row_bytes, row_bytes * n,
                                4096 * sizeof(float));
        q = ggml_scale(ctx, ggml_rms_norm(ctx, q, in.norm_eps / width), 1.0f / std::sqrt(float(width)));
        k = ggml_scale(ctx, ggml_rms_norm(ctx, k, in.norm_eps / width), 1.0f / std::sqrt(float(width)));
        auto * g = input(decay.data(), 1, heads, n);
        auto * b = ggml_sigmoid(ctx, input(beta.data(), 1, heads, n));
        auto * initial = input(state.data(), width, width, heads);
        auto * result = ggml_gated_delta_net(ctx, q, k, v, g, b, initial, 1);
        const size_t elements = n * group;
        if (ggml_nelements(result) != int64_t(elements + states)) return false;
        auto * values = ggml_view_4d(ctx, result, width, heads, n, 1, width * sizeof(float), group * sizeof(float),
                                     elements * sizeof(float), 0);
        auto * final_state =
            ggml_view_4d(ctx, result, width, width, heads, 1, width * sizeof(float), width * width * sizeof(float),
                         states * sizeof(float), elements * sizeof(float));
        auto * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, values, in.norm_eps), input(in.norm_weights, width));
        auto * gated = ggml_mul(ctx, normalized, ggml_silu(ctx, input(z.data(), width, heads, n)));
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, gated);
        ggml_build_forward_expand(graph, final_state);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
        const auto * post = static_cast<const float *>(gated->data);
        const auto * saved = static_cast<const float *>(final_state->data);
        if (!finite(post, elements) || !finite(saved, states)) return false;
        linear_outputs candidate;
        candidate.values.resize(elements);
        candidate.state.resize(states);
        candidate.conv_history.resize(channels * 4);
        for (size_t h = 0; h < heads; ++h) {
            const size_t source = tiled_head(h);
            for (size_t t = 0; t < n; ++t)
                for (size_t d = 0; d < width; ++d)
                    candidate.values[t * group + h * width + d] = post[t * group + source * width + d];
            for (size_t key = 0; key < width; ++key)
                for (size_t value = 0; value < width; ++value)
                    candidate.state[(h * width + key) * width + value] = saved[(source * width + value) * width + key];
        }
        for (size_t ch = 0; ch < channels; ++ch)
            for (int tap = 0; tap < 4; ++tap) {
                const int t = in.tokens - 4 + tap;
                candidate.conv_history[ch * 4 + size_t(tap)] = t >= 0 ? in.qkv[size_t(t) * channels + ch]
                                                               : in.conv_history
                                                                   ? in.conv_history[ch * 4 + size_t(t + 4)]
                                                                   : 0.0f;
            }
        out = std::move(candidate);
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}
} // namespace qwen35_ocr
