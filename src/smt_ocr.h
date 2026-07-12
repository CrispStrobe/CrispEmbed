// smt_ocr.h — Sheet Music Transformer (SMT / SMT++) optical music recognition.
//
// Staff-notation image → bekern token sequence, via ggml graph compute.
//   1. Preprocess (grayscale, invert, [0,1], resize ×reduce_ratio)
//   2. ConvNext encoder (3 stages [64,128,256], 16× reduction) → (H'W', 256)
//   3. Transformer decoder (8L, d=256, 4H, UNSCALED attn, cross-attn key≠value,
//      ReLU FFN) with sinusoidal PE → autoregressive bekern tokens
//   4. Detokenize via the embedded bekern vocab
//
// Loaded from a GGUF produced by models/convert-smt-to-gguf.py (arch "smt_ocr").
// Follows the math_ocr.cpp VisionEncoderDecoder pattern.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct smt_ocr_context smt_ocr_context;

typedef struct smt_ocr_hparams {
    // Encoder (ConvNext)
    int32_t enc_num_stages;   // 3
    int32_t enc_reduction;    // 16 (total H/W downsample)
    int32_t enc_num_channels; // 1 (grayscale)
    int32_t enc_stem_kernel;  // 4
    // Decoder
    int32_t dec_layers; // 8
    int32_t d_model;    // 256
    int32_t num_heads;  // 4
    int32_t dim_ff;     // 256
    int32_t vocab_size; // 20578
    int32_t maxlen;     // 1281
    int32_t maxh, maxw; // 2D PE table caps
    // Special tokens
    int32_t bos_token, eos_token, pad_token;
    int32_t scale_attention; // 0 for SMT (QK^T is UNSCALED)
} smt_ocr_hparams;

/// Load an SMT GGUF model. NULL on failure.
smt_ocr_context * smt_ocr_init(const char * model_path, int n_threads);
void smt_ocr_free(smt_ocr_context * ctx);
const smt_ocr_hparams * smt_ocr_get_hparams(const smt_ocr_context * ctx);

/// Recognize on a grayscale image in [0,1] (row-major, width×height).
/// Returns a bekern token string (space-joined), owned by ctx, or NULL.
const char * smt_ocr_recognize(smt_ocr_context * ctx, const float * pixels, int width, int height, int * out_len);

/// Recognize on a raw pixel buffer (RGB/RGBA/gray uint8); resize+invert+grayscale
/// preprocessing is applied internally. Returns a bekern token string, or NULL.
const char * smt_ocr_recognize_raw(smt_ocr_context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                   int * out_len);

/// Recognize on a raw image file (PNG/JPEG); handles preprocessing internally.
const char * smt_ocr_recognize_file(smt_ocr_context * ctx, const char * image_path, int * out_len);

/// Per-stage parity harness. Loads a reference GGUF (from tools/dump_smt_reference.py),
/// feeds its `input_tensor`, runs the encoder + teacher-forces its `token_ids`
/// through the decoder, and prints cos_min/max_abs for every stage vs the
/// Python reference. Returns 0 if all present stages pass (cos_min ≥ 0.999).
int smt_ocr_run_diff(smt_ocr_context * ctx, const char * ref_gguf_path);

#ifdef __cplusplus
}
#endif
