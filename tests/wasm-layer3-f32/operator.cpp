// TEST ONLY. No production encoder, injected workspace or original rerun.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

namespace {
constexpr size_t W = 313344, X = 1536 * 12, Y = 384 * 12, D = 1536 * 384;
bool finite(const float * p, size_t n) {
    if (!p) return false;
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(p[i])) return false;
    return true;
}
struct owner {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~owner() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};
}

// mode0: original IQ4_XS weight x F32 input (actual Q8_K conversion).
// mode1: actual graph cast of unchanged IQ4_XS coefficients to F32, then
// F32 x F32 matmul. Optional coefficient output only observes that cast.
extern "C" int crispembed_layer3_f32(int mode, const uint8_t * weight, int weight_bytes,
                                     const float * input, int input_count, float * output,
                                     int capacity, float * coefficients, int coefficient_count, int threads) {
    if ((mode != 0 && mode != 1) || !weight || weight_bytes != int(W) || input_count != int(X) ||
        !finite(input, X) || !output || capacity != int(Y) || threads < 1 || threads > 2 ||
        (coefficients ? coefficient_count != int(D) || mode != 1 : coefficient_count != 0)) return -1;
    for (size_t k = 0; k < W / 136; ++k) {
        uint16_t scale;
        std::memcpy(&scale, weight + k * 136, 2);
        if ((scale & 0x7c00) == 0x7c00) return -2;
    }
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) return -3;
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        owner o;
        o.ctx = ggml_init({ 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true });
        if (!o.ctx) return -4;
        auto * w = ggml_new_tensor_2d(o.ctx, GGML_TYPE_IQ4_XS, 1536, 384);
        ggml_set_name(w, "blk.3.ffn_down.weight");
        ggml_set_input(w);
        auto * x = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, 1536, 12);
        ggml_set_input(x);
        auto * operand = mode ? ggml_cast(o.ctx, w, GGML_TYPE_F32) : w;
        if (mode) {
            // Retain the actual connected cast node, even in ordinary calls.
            // Observer neutrality therefore changes only host readback.
            ggml_set_output(operand);
            if (operand->type != GGML_TYPE_F32 || ggml_nbytes(operand) != D * 4 ||
                ggml_get_type_traits_cpu(operand->type)->vec_dot_type != GGML_TYPE_F32) return -5;
        }
        auto * y = ggml_mul_mat(o.ctx, operand, x);
        ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(o.ctx, 32, false);
        ggml_build_forward_expand(graph, y);
        o.buffer = ggml_backend_alloc_ctx_tensors(o.ctx, backend.get());
        if (!o.buffer) return -6;
        ggml_backend_tensor_set(w, weight, 0, W);
        ggml_backend_tensor_set(x, input, 0, X * 4);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return -7;
        std::unique_ptr<float[]> pending(new float[Y]);
        ggml_backend_tensor_get(y, pending.get(), 0, Y * 4);
        if (!finite(pending.get(), Y)) return -8;
        std::unique_ptr<float[]> derived;
        if (coefficients) {
            derived.reset(new float[D]);
            ggml_backend_tensor_get(operand, derived.get(), 0, D * 4);
            if (!finite(derived.get(), D)) return -9;
        }
        // Publish only after every graph/readback/validation succeeds.
        if (coefficients) std::copy_n(derived.get(), D, coefficients);
        std::copy_n(pending.get(), Y, output);
        return int(Y);
    } catch (...) { return -10; }
}
