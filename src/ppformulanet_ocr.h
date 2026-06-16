// ppformulanet_ocr.h — PPFormulaNet-S math OCR via ggml.
//
// Architecture: HGNetv2 CNN encoder + MBart Transformer decoder.
// Loads from GGUF produced by convert-ppformulanet-to-gguf.py.
//
// Source: PaddlePaddle/PaddleOCR PP-FormulaNet-S (Apache-2.0 license).
// 57M parameters, 384×384 RGB input, outputs LaTeX tokens.
//
// Encoder: HGNetv2 — pure CNN (Conv-BN-ReLU), 4 stages of HG_Blocks.
//   StemBlock → Stage0 (128ch) → Stage1 (512ch) → Stage2 (1024ch) → Stage3 (2048ch)
//   Flatten to (N, 2048), project to (N, 384) for decoder.
//
// Decoder: MBart — 2 layers, 16 heads, d_model=384, FFN=1536.
//   Post-LN, scale_embedding, greedy/beam-search decoding.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppformulanet_ocr_context ppformulanet_ocr_context;

typedef struct ppformulanet_ocr_hparams {
    // Encoder (HGNetv2)
    int32_t image_size;       // 384
    int32_t enc_hidden;       // 2048

    // Decoder (MBart)
    int32_t dec_layers;       // 2
    int32_t dec_heads;        // 16
    int32_t dec_d_model;      // 384
    int32_t dec_ffn_dim;      // 1536
    int32_t vocab_size;       // 50000
    int32_t max_seq_len;      // 1027
    int32_t cross_attn_dim;   // enc_hidden projected to dec_d_model

    // Special tokens
    int32_t bos_token;
    int32_t eos_token;
    int32_t pad_token;
    int32_t decoder_start_token;
} ppformulanet_ocr_hparams;

/// Load a PPFormulaNet-S GGUF model. Returns NULL on failure.
ppformulanet_ocr_context * ppformulanet_ocr_init(const char * model_path, int n_threads);

/// Free the context and all associated memory.
void ppformulanet_ocr_free(ppformulanet_ocr_context * ctx);

/// Get the model hyperparameters.
const ppformulanet_ocr_hparams * ppformulanet_ocr_get_hparams(const ppformulanet_ocr_context * ctx);

/// Run OCR on a grayscale image.
/// [pixels] — row-major grayscale float array, values in [0, 1].
/// [width], [height] — image dimensions.
/// Returns a null-terminated LaTeX string owned by the context.
const char * ppformulanet_ocr_recognize(
    ppformulanet_ocr_context * ctx,
    const float * pixels,
    int width, int height,
    int * out_len
);

/// Run OCR on raw pixel bytes (RGB or RGBA).
const char * ppformulanet_ocr_recognize_raw(
    ppformulanet_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width, int height, int channels,
    int * out_len
);

/// After a successful recognize call, returns the encoder output.
/// Shape: (*out_n_tokens, *out_hidden). Valid until the next call.
const float * ppformulanet_ocr_get_encoder_output(
    const ppformulanet_ocr_context * ctx,
    int * out_n_tokens,
    int * out_hidden
);

/// Get per-character confidence scores from the last recognition.
/// Returns array of length *n_chars (one per output token).
/// Each value is the softmax probability of the winning token.
/// Valid until the next recognize call. Returns NULL if no recognition done.
const float * ppformulanet_ocr_confidences(const ppformulanet_ocr_context * ctx, int * n_chars);

/// Get mean confidence across all tokens from the last recognition.
float ppformulanet_ocr_mean_confidence(const ppformulanet_ocr_context * ctx);

#ifdef __cplusplus
}
#endif
