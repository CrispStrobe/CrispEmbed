// Kernel-level timing for the qwen2vl_ocr (TeleOCR/MinerU) shapes on the GPU
// backend, to attribute end-to-end Metal latency. Times one op per graph
// (repeated), for the matmul precision routes and flash attention.
//
//   test-bench-qwen2vl-metal-ops [iters=10]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int32_t kPrecF32Hp = 0x48500001; // GGML_METAL_MUL_MAT_PREC_F32_HP

struct bench {
    ggml_backend_t backend;
    int iters;

    double run(const char * label, const std::function<ggml_tensor *(ggml_context *)> & build, double flop) {
        ggml_init_params ip{ ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * g = ggml_init(ip);
        ggml_tensor * out = build(g);
        ggml_cgraph * gf = ggml_new_graph(g);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g, backend);
        if (!buf) {
            std::printf("%-58s alloc failed\n", label);
            ggml_free(g);
            return 0;
        }
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (ggml_tensor * t = ggml_get_first_tensor(g); t; t = ggml_get_next_tensor(g, t)) {
            if (t->op != GGML_OP_NONE) continue;
            std::vector<float> v((size_t)ggml_nelements(t));
            for (auto & x : v) x = nd(rng) * 0.05f;
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<uint8_t> q(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, v.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, q.data(), 0, q.size());
            }
        }
        if (!ggml_backend_supports_op(backend, out)) {
            std::printf("%-58s unsupported on this backend\n", label);
            ggml_backend_buffer_free(buf);
            ggml_free(g);
            return 0;
        }
        ggml_backend_graph_compute(backend, gf); // warm-up / pipeline compile
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::printf("%-58s %9.3f ms  %8.1f GFLOP/s\n", label, ms, flop / ms * 1e-6);
        ggml_backend_buffer_free(buf);
        ggml_free(g);
        return ms;
    }
};

ggml_tensor * matmul(ggml_context * g, ggml_type wt, int k, int n, int m, int mode) {
    ggml_tensor * w = ggml_new_tensor_2d(g, wt, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, k, m);
    ggml_tensor * y = ggml_mul_mat(g, w, x);
    if (mode >= 1) ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (mode == 2) y->op_params[1] = kPrecF32Hp;
    return y;
}
} // namespace

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 10;
    ggml_backend_t backend = ggml_backend_init_best();
    if (!backend) return 1;
    std::printf("backend: %s, iters=%d\n", ggml_backend_name(backend), iters);
    bench b{ backend, iters };
    const char * modes[] = { "default", "PREC_F32", "PREC_F32+HP" };
    struct shape {
        const char * name;
        int k, n, m;
    } shapes[] = {
        { "vision ffn up  1280x3420 x952", 1280, 3420, 952 },     { "vision ffn down 3420x1280 x952", 3420, 1280, 952 },
        { "llm ffn up 1024x3072 x prompt 600", 1024, 3072, 600 }, { "llm ffn up 1024x3072 x1 (decode)", 1024, 3072, 1 },
        { "lm_head 1024x151936 x1 (decode)", 1024, 151936, 1 },
    };
    for (ggml_type wt : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        for (const auto & s : shapes) {
            if (ggml_blck_size(wt) > 1 && s.k % 256 != 0) continue; // vision stays F16 in published recipes
            for (int mode = 0; mode < 3; ++mode) {
                std::string label = std::string(ggml_type_name(wt)) + " " + s.name + " " + modes[mode];
                b.run(
                    label.c_str(), [&](ggml_context * g) { return matmul(g, wt, s.k, s.n, s.m, mode); },
                    2.0 * s.k * s.n * s.m);
            }
        }
    }
    // Flash attention: vision full block (hd80, 16 heads, 952 tokens), decode (hd64, 16q/8kv, 1 x 1292).
    for (int prec = 0; prec < 2; ++prec) {
        for (ggml_type kv : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
            std::string label =
                std::string("FA vision hd80 h16 n952 kv=") + ggml_type_name(kv) + (prec ? " PREC_F32" : "");
            b.run(
                label.c_str(),
                [&](ggml_context * g) {
                    ggml_tensor * q = ggml_new_tensor_3d(g, GGML_TYPE_F32, 80, 952, 16);
                    ggml_tensor * k = ggml_new_tensor_3d(g, kv, 80, 952, 16);
                    ggml_tensor * v = ggml_new_tensor_3d(g, kv, 80, 952, 16);
                    ggml_tensor * a = ggml_flash_attn_ext(g, q, k, v, nullptr, 0.11f, 0.0f, 0.0f);
                    if (prec) ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
                    return a;
                },
                4.0 * 80 * 952 * 952 * 16);
            label = std::string("FA decode hd64 h16/8 1x1292 kv=") + ggml_type_name(kv) + (prec ? " PREC_F32" : "");
            b.run(
                label.c_str(),
                [&](ggml_context * g) {
                    ggml_tensor * q = ggml_new_tensor_3d(g, GGML_TYPE_F32, 64, 1, 16);
                    ggml_tensor * k = ggml_new_tensor_3d(g, kv, 64, 1292, 8);
                    ggml_tensor * v = ggml_new_tensor_3d(g, kv, 64, 1292, 8);
                    ggml_tensor * m = ggml_new_tensor_2d(g, GGML_TYPE_F16, 1292, 1);
                    ggml_tensor * a = ggml_flash_attn_ext(g, q, k, v, m, 0.125f, 0.0f, 0.0f);
                    if (prec) ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
                    return a;
                },
                4.0 * 64 * 1292 * 16);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
