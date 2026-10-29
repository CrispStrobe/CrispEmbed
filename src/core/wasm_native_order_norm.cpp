#include "wasm_native_order_norm.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace wasm_native_order_norm {
namespace {
constexpr int width = 384;
float add(float a, float b) noexcept {
    volatile float v = a + b;
    return v;
}
float subtract(float a, float b) noexcept {
    volatile float v = a - b;
    return v;
}
float multiply(float a, float b) noexcept {
    volatile float v = a * b;
    return v;
}
float divide(float a, float b) noexcept {
    volatile float v = a / b;
    return v;
}
void callback(ggml_tensor * destination, const ggml_tensor * source, int ith, int nth, void * userdata) {
    const auto * config = static_cast<const parameters *>(userdata);
    const int64_t rows = source->ne[1] * source->ne[2] * source->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t i1 = row % source->ne[1], i2 = (row / source->ne[1]) % source->ne[2];
        const int64_t i3 = row / (source->ne[1] * source->ne[2]);
        const auto * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(source->data) +
                                                         i1 * source->nb[1] + i2 * source->nb[2] + i3 * source->nb[3]);
        auto * y = reinterpret_cast<float *>(reinterpret_cast<char *>(destination->data) + i1 * destination->nb[1] +
                                             i2 * destination->nb[2] + i3 * destination->nb[3]);
        if (!compute_row(x, y, width, config->epsilon)) {
            // Invalid arithmetic cannot become a plausible finite tensor. No exceptions
            // cross a GGML worker boundary; a nonfinite result stays visible to the caller.
            for (int i = 0; i < width; ++i) y[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
}
} // namespace
bool eligible(bool bert, int h, int layers, int heads, int intermediate, bool pre_ln, float epsilon) {
    return bert && h == width && layers == 6 && heads == 12 && intermediate == 1536 && !pre_ln && epsilon == 1e-12f;
}
bool compute_row(const float * source, float * destination, int h, float epsilon) noexcept {
    if (!source || !destination || h != width || epsilon != 1e-12f) return false;
    // Validate before writing so tiny invalid-input controls preserve caller output.
    double sum = 0;
    for (int i = 0; i < width; ++i) {
        if (!std::isfinite(source[i])) return false;
        sum += double(source[i]);
    }
    const float mean = divide(float(sum), float(width));
    if (!std::isfinite(mean)) return false;
    // Reproduce a06 AVX2/FMA vec_cvar's F32 rounding and eight-lane tree,
    // not the WASM scalar-F64 variance. The operation has no architecture intrinsics.
    float centered[width];
    double square_sum = 0;
    for (int i = 0; i < width; i += 8) {
        float squares[8];
        for (int j = 0; j < 8; ++j) {
            centered[i + j] = subtract(source[i + j], mean);
            squares[j] = multiply(centered[i + j], centered[i + j]);
        }
        const float a0 = add(squares[4], squares[0]), a1 = add(squares[5], squares[1]);
        const float a2 = add(squares[6], squares[2]), a3 = add(squares[7], squares[3]);
        square_sum += double(add(add(a0, a2), add(a1, a3)));
    }
    const float variance = float(square_sum / double(width));
    const float scale = divide(1.f, std::sqrt(add(variance, epsilon)));
    if (!std::isfinite(variance) || !std::isfinite(scale)) return false;
    for (int i = 0; i < width; ++i) destination[i] = multiply(centered[i], scale);
    return true;
}
ggml_tensor * make(ggml_context * context, ggml_tensor * source, const parameters * config) {
    if (!context || !source || !config || source->type != GGML_TYPE_F32 || source->ne[0] != width ||
        source->nb[0] != sizeof(float) || config->epsilon != 1e-12f || source->ne[1] <= 0 || source->ne[2] <= 0 ||
        source->ne[3] <= 0)
        return nullptr;
    if (source->ne[1] > std::numeric_limits<int64_t>::max() / source->ne[2] ||
        source->ne[1] * source->ne[2] > std::numeric_limits<int64_t>::max() / source->ne[3])
        return nullptr;
    return ggml_map_custom1(context, source, callback, GGML_N_TASKS_MAX, const_cast<parameters *>(config));
}
} // namespace wasm_native_order_norm
