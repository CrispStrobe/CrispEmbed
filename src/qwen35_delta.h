#pragma once

#include <vector>

// Experimental dense Qwen3.5 recurrence boundary. Inputs/output use official HF
// layouts: q/k/v [B,T,H,D], g/beta [B,T,H], state [B,H,K,V]. The ggml state is
// transposed internally. No model loading, tokenizer or OCR API is exposed yet.
namespace qwen35_ocr {
bool delta_cpu(const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * initial_state, int batch, int tokens, int heads, int width, int threads,
               std::vector<float> & output, std::vector<float> & final_state);
}
