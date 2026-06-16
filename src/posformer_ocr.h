// posformer_ocr.h — PosFormer Handwritten Math OCR via ggml.
//
// Architecture: DenseNet encoder + Transformer decoder + ARM (coverage).
// Loads from GGUF produced by convert-posformer-to-gguf.py.
// Source: SJTU-DeepVisionLab/PosFormer (BSD-2 license), trained on CROHME 2014.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct posformer_ocr_context posformer_ocr_context;

typedef struct posformer_ocr_hparams {
    // Encoder (DenseNet)
    int32_t growth_rate;        // 24
    int32_t num_layers;         // 16 (per block)
    int32_t input_channels;     // 1

    // Decoder (Transformer)
    int32_t d_model;            // 256
    int32_t nhead;              // 8
    int32_t num_decoder_layers; // 3
    int32_t dim_feedforward;    // 1024
    int32_t vocab_size;         // 113
    int32_t max_len;            // 200
    int32_t pad_token;          // 0
    int32_t sos_token;          // 1
    int32_t eos_token;          // 2

    // ARM
    int32_t arm_dc;             // 32
} posformer_ocr_hparams;

posformer_ocr_context * posformer_ocr_init(const char * model_path, int n_threads);
void                    posformer_ocr_free(posformer_ocr_context * ctx);
const posformer_ocr_hparams * posformer_ocr_get_hparams(const posformer_ocr_context * ctx);

const char * posformer_ocr_recognize(
    posformer_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len);

const char * posformer_ocr_recognize_raw(
    posformer_ocr_context * ctx,
    const uint8_t * pixel_bytes, int width, int height, int channels,
    int * out_len);

/// Get per-character confidence scores from the last recognition.
/// Returns array of length *n_chars (one per output token).
/// Each value is the softmax probability of the winning token.
/// Valid until the next recognize call. Returns NULL if no recognition done.
const float * posformer_ocr_confidences(const posformer_ocr_context * ctx, int * n_chars);

/// Get mean confidence across all tokens from the last recognition.
float posformer_ocr_mean_confidence(const posformer_ocr_context * ctx);

#ifdef __cplusplus
}
#endif
