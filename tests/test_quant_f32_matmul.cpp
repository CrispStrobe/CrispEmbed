#include "core/quant_f32_matmul.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool check(ggml_type type, int columns) {
    constexpr int rows = 35, tokens = 3;
    ggml_context * ctx = ggml_init({ 8 * 1024 * 1024, nullptr, false });
    auto * backend = ggml_backend_cpu_init();
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend, 4);
    auto * w = ggml_new_tensor_2d(ctx, type, columns, rows);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, tokens);
    std::vector<float> original(columns * rows), dequantized(columns * rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            original[r * columns + c] = r == 0 ? (c == 0 ? 0.0f : 0.5f) : std::sin(float(r * columns + c)) * 0.25f;
        }
    }
    ggml_quantize_init(type);
    ggml_quantize_chunk(type, original.data(), w->data, 0, rows, columns, nullptr);
    ggml_get_type_traits(type)->to_float(w->data, dequantized.data(), columns * rows);
    auto * input = static_cast<float *>(x->data);
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < columns; ++c) input[t * columns + c] = t == 0 ? 1.0f : std::cos(float(t + c));
    }
    input[0] = 10000.0f; // Q8 activation scaling erases the other unit entries.
    auto * precise = core_cpu::quant_f32_matmul(ctx, w, x);
    auto * ordinary = ggml_mul_mat(ctx, w, x);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, precise);
    ggml_build_forward_expand(graph, ordinary);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    bool ok = precise->ne[0] == rows && precise->ne[1] == tokens;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int c = 0; c < columns; ++c) expected += double(dequantized[r * columns + c]) * input[t * columns + c];
            const float actual = static_cast<float *>(precise->data)[t * rows + r];
            ok &= std::isfinite(actual) && std::fabs(actual - expected) <= 2e-5 * std::fmax(1.0, std::fabs(expected));
        }
    }
    const float accurate = static_cast<float *>(precise->data)[0];
    const float lossy = static_cast<float *>(ordinary->data)[0];
    ok &= std::fabs(accurate - lossy) > 10; // Ensure this fixture actually exposes activation loss.
    std::printf("%s %s: %d rows x %d tokens, outlier dot FP32=%.6f quantized=%.6f\n", ok ? "PASS" : "FAIL",
                ggml_type_name(type), rows, tokens, accurate, lossy);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

int main() {
    const bool q8 = check(GGML_TYPE_Q8_0, 32);
    const bool q4 = check(GGML_TYPE_Q4_K, 256);
    return q8 && q4 ? 0 : 1;
}
