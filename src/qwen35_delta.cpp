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
bool delta_cpu(const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * initial_state, int batch, int tokens, int heads, int width, int threads,
               std::vector<float> & output, std::vector<float> & final_state) {
    output.clear();
    final_state.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !q || !k || !v || !g || !beta || batch < 1 || tokens < 1 ||
        heads < 1 || width < 1 || threads < 1)
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
