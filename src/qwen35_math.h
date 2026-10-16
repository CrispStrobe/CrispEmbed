#pragma once
#include "core/env_gate.h"
#include <cmath>
#include <cstddef>
#include <vector>
struct ggml_tensor;
namespace qwen35_ocr {
// Keep the source's F32 squares and scale. Optional F64 summation isolates
// reduction error without changing weights, epsilon or normalization order.
inline float rms_scale_cpu(const float * x, int width, float eps) {
    if (core_env::on("CRISPEMBED_QWEN35_PRECISE_RMS")) {
        double square = 0;
        for (int i = 0; i < width; ++i) square += double(x[i] * x[i]);
        return 1.0f / std::sqrt(float(square / width) + eps);
    }
    float square = 0;
    for (int i = 0; i < width; ++i) square += x[i] * x[i];
    return 1.0f / std::sqrt(square / width + eps);
}
// Shared F32 reference arithmetic for native experimental decoder paths.
// Quantized matrix loading is separately default-off; vectors/convs stay float.
bool matrix_type_supported(const ggml_tensor * source);
// Internal synchronous calibration observer, local to the calling thread.
// Registration and invocation require CRISPEMBED_QWEN35_IMATRIX=1. Clear with null.
using projection_observer = bool (*)(const char *, const float *, int, int, void *);
bool set_projection_observer(projection_observer observer, void * user);
// Bounded scheduling policy for the default-off parallel F16 conversion path.
int f16_conversion_workers(std::size_t elements, int threads);
bool tensor_floats(ggml_tensor * source, float * target, int threads = 1);
bool project_cpu(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width, int tokens, int threads,
                 std::vector<std::vector<float>> & output);
} // namespace qwen35_ocr
