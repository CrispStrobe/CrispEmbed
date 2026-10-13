#pragma once

#include <vector>

// Experimental dense Qwen3.5 recurrence boundary. Inputs/output use official HF
// layouts: q/k/v [B,T,H,D], g/beta [B,T,H], state [B,H,K,V]. The ggml state is
// transposed internally. No model loading, tokenizer or OCR API is exposed yet.
namespace qwen35_ocr {
struct linear_inputs {
    int batch, tokens, heads, width, conv_kernel, threads;
    float norm_eps;
    const float * qkv;
    const float * z;
    const float * a;
    const float * b;
    const float * conv_weights;
    const float * a_log;
    const float * dt_bias;
    const float * norm_weights;
    const float * conv_history; // [B,3*H*D,K], nullptr for a fresh request
    const float * state;        // [B,H,D,D], nullptr for a fresh request
};
struct linear_outputs {
    std::vector<float> values;       // [B,T,H,D], after gated normalization
    std::vector<float> conv_history; // [B,3*H*D,K]
    std::vector<float> state;        // [B,H,D,D]
};
bool linear_core_cpu(const linear_inputs & input, linear_outputs & output);

bool delta_cpu(const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * initial_state, int batch, int tokens, int heads, int width, int threads,
               std::vector<float> & output, std::vector<float> & final_state);
} // namespace qwen35_ocr
