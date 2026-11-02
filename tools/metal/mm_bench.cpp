// Build: c++ -O2 -std=c++17 -Iggml/include tools/metal/mm_bench.cpp -Lbuild-metal/ggml/src -lggml -lggml-base -o mm_bench
// Run:   mm_bench K M N prec_f32(0/1) [iterations]
// Micro-benchmark: ggml MUL_MAT on the default GPU backend, half vs PREC_F32.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    int K = atoi(argv[1]), M = atoi(argv[2]), N = atoi(argv[3]), prec = atoi(argv[4]),
        it = argc > 5 ? atoi(argv[5]) : 20;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 16 + ggml_graph_overhead());
    ggml_context * ctx = ggml_init({ meta.size(), meta.data(), true });
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    if (prec) ggml_mul_mat_set_prec(c, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    std::vector<float> h((size_t)K * std::max(M, N), 0.01f);
    ggml_backend_tensor_set(a, h.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, h.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(be, gf);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; i++) ggml_backend_graph_compute(be, gf);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / it;
    printf("K=%d M=%d N=%d prec_f32=%d: %.3f ms, %.1f GFLOP/s\n", K, M, N, prec, ms, 2.0 * K * M * N / ms / 1e6);
}
