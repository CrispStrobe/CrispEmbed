// deepseek_ocr2.h — DeepSeek-OCR-2 (3B MoE) document OCR engine.
//
// Architecture:
//   SAM-ViT-B (12L, 768d, windowed+global attention)
//   → Qwen2 encoder (24L, 896d, bidirectional SwiGLU transformer)
//   → Linear projector (896→1280)
//   → DeepSeek-V2 MoE decoder (12L, 1280d, 10H GQA, layer 0 dense,
//     layers 1-11: 64 routed experts top-6 + 2 shared experts)
//   → lm_head → greedy decode
//
// Source: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 (Apache-2.0)

#ifndef DEEPSEEK_OCR2_H
#define DEEPSEEK_OCR2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct deepseek_ocr2_context deepseek_ocr2_context;

/// Load a DeepSeek-OCR-2 GGUF model.
deepseek_ocr2_context * deepseek_ocr2_init(const char * model_path, int n_threads);

/// Recognize text from raw RGB pixels.
/// Returns pointer to UTF-8 text (owned by ctx, valid until next call).
const char * deepseek_ocr2_recognize_raw(
    deepseek_ocr2_context * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    int * out_len);

/// Recognize from pre-normalized float pixels (grayscale, [0,1]).
const char * deepseek_ocr2_recognize(
    deepseek_ocr2_context * ctx,
    const float * pixels, int width, int height,
    int * out_len);

/// Get per-token confidence scores from the last recognition.
const float * deepseek_ocr2_confidences(const deepseek_ocr2_context * ctx, int * n_tokens);

/// Get mean confidence from the last recognition.
float deepseek_ocr2_mean_confidence(const deepseek_ocr2_context * ctx);

void deepseek_ocr2_free(deepseek_ocr2_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // DEEPSEEK_OCR2_H
