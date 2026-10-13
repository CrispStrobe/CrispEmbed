#include "core/gelu_tanh_f32.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>

static int crispembed_test_main() {
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    const float inputs[] = { 1.00001f, 1.00002f, -1.00001f, -1.00002f, 0, 0.1234567f, 11, -11 };
    auto * values = static_cast<float *>(x->data);
    for (int i = 0; i < 8; ++i) values[i] = inputs[i];
    auto * accurate = core_cpu::gelu_tanh_f32(ctx, x);
    auto * ordinary = ggml_gelu(ctx, x);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, accurate);
    ggml_build_forward_expand(graph, ordinary);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 1;
    auto * a = static_cast<float *>(accurate->data);
    auto * b = static_cast<float *>(ordinary->data);
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        const double v = inputs[i];
        const double ref = 0.5 * v * (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) * (v + 0.044715 * v * v * v)));
        ok &= std::isfinite(a[i]) && std::fabs(a[i] - ref) < 2e-6;
    }
    ok &= a[0] != a[1] && b[0] == b[1];
    printf("%s GELU: close FP32 inputs %.8f / %.8f; ordinary %.8f / %.8f\n", ok ? "PASS" : "FAIL", a[0], a[1], b[0],
           b[1]);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok ? 0 : 1;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
