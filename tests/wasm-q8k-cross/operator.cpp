// TEST ONLY. Actual IQ4_XS x F32/Q8_K graph; never linked into CrispEmbed.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {
constexpr size_t W = 313344, X = 1536 * 12, Q = 21024, Y = 384 * 12;
bool finite(const float * x, size_t n) {
    if (!x) return false;
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}
bool workspace(const uint8_t * q) {
    if (!q) return false;
    // Exact a06 block_q8_K: d:F32, qs:256 signed bytes, bsums:16 I16.
    for (size_t k = 0; k < 72; ++k) {
        const auto * p = q + k * 292;
        float d;
        std::memcpy(&d, p, 4);
        if (!std::isfinite(d)) return false;
        for (int group = 0; group < 16; ++group) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) {
                int8_t v;
                std::memcpy(&v, p + 4 + group * 16 + i, 1);
                sum += v;
            }
            int16_t stored;
            std::memcpy(&stored, p + 260 + group * 2, 2);
            if (sum != stored) return false;
        }
    }
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
} // namespace

// Capture-directory setting only, not a production/default gate. Each hosted
// module runs synchronous calls sequentially. Unset before direct-Q8_K calls.
extern "C" int crispembed_q8k_setenv(const char * directory) {
    return directory && *directory ? setenv("CRISPEMBED_LAYER3_KERNEL", directory, 1)
                                   : unsetenv("CRISPEMBED_LAYER3_KERNEL");
}

// mode0: actual F32->Q8_K conversion in mul_mat; the test overlay captures it.
// mode1: exact saved Q8_K tensor is the matmul operand; original_input is checked
// and retained for attribution, but is NOT an arithmetic operand in that mode.
extern "C" int crispembed_q8k_cross(int mode, const uint8_t * weight, int weight_bytes, const float * original_input,
                                    int input_count, const uint8_t * quantized, int quantized_bytes, float * output,
                                    int capacity, int threads) {
    if ((mode != 0 && mode != 1) || !weight || weight_bytes != int(W) || input_count != int(X) ||
        !finite(original_input, X) || !output || capacity != int(Y) || threads < 1 || threads > 2)
        return -1;
    if ((mode == 0 && (quantized || quantized_bytes)) ||
        (mode == 1 && (quantized_bytes != int(Q) || !workspace(quantized))))
        return -2;
    if (mode == 1 && std::getenv("CRISPEMBED_LAYER3_KERNEL")) return -3;
    // IQ4_XS block256: F16 scale + 2-byte scales_h + 4-byte scales_l + 128 qs.
    for (size_t k = 0; k < W / 136; ++k) {
        uint16_t d;
        std::memcpy(&d, weight + k * 136, 2);
        if ((d & 0x7c00) == 0x7c00) return -4;
    }
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) return -5;
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        owner o;
        o.ctx = ggml_init({ 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true });
        if (!o.ctx) return -6;
        auto * w = ggml_new_tensor_2d(o.ctx, GGML_TYPE_IQ4_XS, 1536, 384);
        ggml_set_name(w, "blk.3.ffn_down.weight");
        ggml_set_input(w);
        auto * x = ggml_new_tensor_2d(o.ctx, mode ? GGML_TYPE_Q8_K : GGML_TYPE_F32, 1536, 12);
        ggml_set_input(x);
        auto * y = ggml_mul_mat(o.ctx, w, x);
        ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(o.ctx, 32, false);
        ggml_build_forward_expand(graph, y);
        if (ggml_nbytes(w) != W || ggml_nbytes(x) != (mode ? Q : X * 4)) return -7;
        o.buffer = ggml_backend_alloc_ctx_tensors(o.ctx, backend.get());
        if (!o.buffer) return -8;
        ggml_backend_tensor_set(w, weight, 0, W);
        ggml_backend_tensor_set(x, mode ? static_cast<const void *>(quantized) : original_input, 0, mode ? Q : X * 4);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return -9;
        std::unique_ptr<float[]> pending(new float[Y]);
        ggml_backend_tensor_get(y, pending.get(), 0, Y * 4);
        if (!finite(pending.get(), Y)) return -10;
        std::copy_n(pending.get(), Y, output);
        return int(Y);
    } catch (...) {
        return -11;
    }
}
