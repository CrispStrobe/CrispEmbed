// Standalone test-only actual GGML F32 operators. Never linked into CrispEmbed.
// Each operation preserves the exact frozen580 encoder graph's layouts.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace {
struct graph_owner {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~graph_owner() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};
bool finite(const float * input, size_t count) {
    if (!input) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(input[i])) return false;
    return true;
}
} // namespace

// op0=norm,1=mul,2=add,3=GELU-erf,4=flash. b_count specifies either an
// exact-width broadcast vector or a complete same-shape residual matrix.
// Raw pointers never escape this synchronous call; output commits only on success.
extern "C" int crispembed_f32_operator(int op, int width, int rows, int b_count, float epsilon, const float * a,
                                       const float * b, const float * c, float * output, int capacity, int threads) {
    if (!output || (width != 384 && width != 1536) || rows != 12 || capacity != width * rows || threads < 1 ||
        threads > 2 || op < 0 || op > 4 || !std::isfinite(epsilon) || epsilon != 1e-12f ||
        !finite(a, size_t(width) * rows))
        return -1;
    if ((op == 1 || op == 2) && ((b_count != width && b_count != width * rows) || !finite(b, b_count))) return -2;
    if (op == 4 && (width != 384 || b_count != width * rows || !finite(b, b_count) || !finite(c, b_count))) return -3;
    if ((op == 0 || op == 3) && (b_count != 0 || b || c)) return -2;
    if ((op == 1 || op == 2) && c) return -2;
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) return -4;
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        graph_owner owner;
        owner.context =
            ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true });
        if (!owner.context) return -5;
        auto * g = owner.context;
        ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, width, rows);
        ggml_tensor *y = nullptr, *z = nullptr, *result = nullptr;
        ggml_set_input(x);
        if (op == 0)
            result = ggml_norm(g, x, epsilon);
        else if (op == 3)
            result = ggml_gelu_erf(g, x);
        else {
            y = ggml_new_tensor_2d(g, GGML_TYPE_F32, width, b_count / width);
            ggml_set_input(y);
            if (op == 1)
                result = ggml_mul(g, x, y);
            else if (op == 2)
                result = ggml_add(g, x, y);
            else {
                z = ggml_new_tensor_2d(g, GGML_TYPE_F32, width, rows);
                ggml_set_input(z);
                // Frozen encoder: reshape [384,12] -> [32,12heads,12tokens,1]
                // then permute(0,2,1,3). No RoPE, mask, bias or precision cast.
                ggml_tensor * q = ggml_permute(g, ggml_reshape_4d(g, x, 32, 12, rows, 1), 0, 2, 1, 3);
                ggml_tensor * k = ggml_permute(g, ggml_reshape_4d(g, y, 32, 12, rows, 1), 0, 2, 1, 3);
                ggml_tensor * v = ggml_permute(g, ggml_reshape_4d(g, z, 32, 12, rows, 1), 0, 2, 1, 3);
                result = ggml_reshape_2d(g, ggml_flash_attn_ext(g, q, k, v, nullptr, 1.0f / sqrtf(32.0f), 0.0f, 0.0f),
                                         width, rows);
            }
        }
        if (!result) return -6;
        ggml_set_output(result);
        ggml_cgraph * graph = ggml_new_graph_custom(g, 64, false);
        ggml_build_forward_expand(graph, result);
        owner.buffer = ggml_backend_alloc_ctx_tensors(g, backend.get());
        if (!owner.buffer) return -7;
        ggml_backend_tensor_set(x, a, 0, size_t(width) * rows * 4);
        if (y) ggml_backend_tensor_set(y, b, 0, size_t(b_count) * 4);
        if (z) ggml_backend_tensor_set(z, c, 0, size_t(width) * rows * 4);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return -8;
        std::unique_ptr<float[]> pending(new float[size_t(width) * rows]);
        ggml_backend_tensor_get(result, pending.get(), 0, size_t(width) * rows * 4);
        if (!finite(pending.get(), size_t(width) * rows)) return -9;
        std::copy_n(pending.get(), size_t(width) * rows, output);
        return width * rows;
    } catch (...) {
        return -10;
    }
}
