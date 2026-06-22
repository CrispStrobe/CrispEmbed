// unlimited_ocr.h — Unlimited-OCR engine.
//
// Architecture:
//   SAM-ViT-B (12L, 768d, windowed+global attention)
//   -> CLIP-L/14 (24L, 1024d, receives SAM features as patch embeddings)
//   -> Fusion: concat(CLIP[:,1:], SAM.flatten) -> Linear(2048, 1280)
//   -> DeepSeek-V2 MoE decoder (12L, 1280d, 10H GQA, layer 0 dense,
//      layers 1-11: 64 routed experts top-6 + 2 shared experts)
//   -> lm_head -> greedy decode
//
// Source: Baidu Unlimited-OCR (Apache-2.0)

#ifndef UNLIMITED_OCR_H
#define UNLIMITED_OCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct unlimited_ocr_context unlimited_ocr_context;

/// Load an Unlimited-OCR GGUF model.
unlimited_ocr_context * unlimited_ocr_init(const char * model_path, int n_threads);

/// Recognize text from raw RGB pixels.
/// Returns pointer to UTF-8 text (owned by ctx, valid until next call).
const char * unlimited_ocr_recognize_raw(
    unlimited_ocr_context * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    int * out_len);

/// Recognize from pre-normalized float pixels (grayscale, [0,1]).
const char * unlimited_ocr_recognize(
    unlimited_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len);

/// Get per-token confidence scores from the last recognition.
const float * unlimited_ocr_confidences(const unlimited_ocr_context * ctx, int * n_tokens);

/// Get mean confidence from the last recognition.
float unlimited_ocr_mean_confidence(const unlimited_ocr_context * ctx);

void unlimited_ocr_free(unlimited_ocr_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // UNLIMITED_OCR_H
