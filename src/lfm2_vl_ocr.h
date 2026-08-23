// lfm2_vl_ocr.h — LFM2.5-VL-3B vision-language OCR engine.
//
// Architecture (LiquidAI/LFM2.5-VL-3B):
//
// Vision encoder (SigLIP2 NaFlex, 400M):
//   patches → Linear(768, 1152) + bilinear-interp pos embed
//   27 × pre-LayerNorm ViT block:
//     LayerNorm → QKV → bidirectional attention → residual
//     LayerNorm → GELU MLP (1152→4304→1152) → residual
//
// Projector:
//   pixel_unshuffle (2×) → Linear(4608, 2048) → GELU → Linear(2048, 2048)
//
// LLM decoder (LFM2.5, 2.6B, hybrid conv+attention):
//   embed_tokens(128000, 2048) → splice image_embeds at image_token positions
//   30 × pre-RMSNorm hybrid block:
//     Conv layers (22/30): in_proj(2048, 6144) → B*x gate → depthwise conv1d(k=3)
//                          → C gate → out_proj(2048, 2048) → residual
//     Attn layers (8/30):  Q/K LN → RoPE → GQA(32h/8kv) → residual
//     All layers:          RMSNorm → SwiGLU FFN (2048→10752→2048) → residual
//   embedding_norm(RMSNorm) → lm_head (tied) → greedy decode
//
// License: LFM-1.0 (revenue-capped; requires CRISPEMBED_ACCEPT_LFM_LICENSE=1)
// Source: https://huggingface.co/LiquidAI/LFM2.5-VL-3B

#ifndef LFM2_VL_OCR_H
#define LFM2_VL_OCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lfm2_vl_ocr_context lfm2_vl_ocr_context;

/// Load LFM2.5-VL from a split model: LLM GGUF + mmproj GGUF.
/// Returns nullptr on failure (including license not accepted).
lfm2_vl_ocr_context * lfm2_vl_ocr_init_split(const char * model_path, const char * mmproj_path, int n_threads);

/// Load from a single stacked GGUF (LLM + vision in one file).
lfm2_vl_ocr_context * lfm2_vl_ocr_init(const char * model_path, int n_threads);

/// Recognize text from raw RGB pixels.
/// Returns pointer to UTF-8 text (owned by ctx, valid until next call).
const char * lfm2_vl_ocr_recognize_raw(lfm2_vl_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len);

/// Recognize from pre-normalized float pixels (grayscale, [0,1]).
const char * lfm2_vl_ocr_recognize(lfm2_vl_ocr_context * ctx, const float * pixels, int width, int height,
                                   int * out_len);

/// Get per-token confidence scores from the last recognition.
const float * lfm2_vl_ocr_confidences(const lfm2_vl_ocr_context * ctx, int * n_tokens);

/// Get mean confidence from the last recognition.
float lfm2_vl_ocr_mean_confidence(const lfm2_vl_ocr_context * ctx);

/// Set the user prompt for the next recognition.
void lfm2_vl_ocr_set_prompt(lfm2_vl_ocr_context * ctx, const char * prompt);

/// Set max tokens for generation.
void lfm2_vl_ocr_set_max_tokens(lfm2_vl_ocr_context * ctx, int max_tokens);

void lfm2_vl_ocr_free(lfm2_vl_ocr_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // LFM2_VL_OCR_H
