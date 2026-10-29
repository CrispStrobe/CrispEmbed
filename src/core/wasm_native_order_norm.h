#pragma once
#include "ggml.h"

// Default-off WASM MiniLM experiment. No saved output or workspace input.
namespace wasm_native_order_norm {
struct parameters {
    float epsilon = 1e-12f; // context-owned, stable for the graph lifetime
};
bool eligible(bool bert, int width, int layers, int heads, int intermediate, bool pre_ln, float epsilon);
// Transactional one-row primitive used by the connected graph callback and tiny controls.
bool compute_row(const float * source, float * destination, int width, float epsilon) noexcept;
ggml_tensor * make(ggml_context * context, ggml_tensor * source, const parameters * config);
} // namespace wasm_native_order_norm
