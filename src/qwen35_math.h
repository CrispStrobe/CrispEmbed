#pragma once
#include <vector>
struct ggml_tensor;
namespace qwen35_ocr {
// Shared F32 reference arithmetic for native experimental decoder paths.
bool tensor_floats(ggml_tensor * source, float * target);
bool project_cpu(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width, int tokens, int threads,
                 std::vector<std::vector<float>> & output);
} // namespace qwen35_ocr
