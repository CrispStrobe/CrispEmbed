// math_ocr.h — pix2tex math OCR via ggml.
//
// Encoder-decoder architecture for image → LaTeX conversion:
//   1. Image preprocessing (grayscale, resize, normalize)
//   2. Hybrid CNN (ResNet backbone) + ViT encoder → patch embeddings
//   3. Transformer decoder with cross-attention → LaTeX token sequence
//   4. Detokenize → LaTeX string
//
// The model is loaded from a GGUF file produced by
// models/convert-pix2tex-to-gguf.py.
//
// This follows the same pattern as CrispASR's encoder-decoder
// (crispasr.cpp) — ggml graph construction, KV-cache for the
// decoder, greedy/beam-search decoding.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque context for a loaded pix2tex model.
typedef struct math_ocr_context math_ocr_context;

/// Model hyperparameters (read-only after init).
/// Architecture: DeiT encoder + TrOCR decoder (VisionEncoderDecoderModel).
typedef struct math_ocr_hparams {
    // Encoder (DeiT)
    int32_t enc_layers;        // num_hidden_layers (12 for small, 24 for large)
    int32_t enc_heads;         // num_attention_heads (6 small, 16 large)
    int32_t enc_hidden;        // hidden_size (384 small, 1024 large)
    int32_t enc_intermediate;  // intermediate_size (1536 small, 4096 large)
    int32_t image_size;        // 384
    int32_t patch_size;        // 16

    // Decoder (TrOCR)
    int32_t dec_layers;        // decoder_layers (6 small, 12 large)
    int32_t dec_heads;         // decoder_attention_heads (8 small, 16 large)
    int32_t dec_d_model;       // d_model (256 small, 1024 large)
    int32_t dec_ffn_dim;       // decoder_ffn_dim (1024 small, 4096 large)
    int32_t vocab_size;        // 1200 (pix2text-mfr) or 50265/64044 (TrOCR)
    int32_t max_seq_len;       // max_position_embeddings (512)
    int32_t cross_attn_dim;    // cross_attention_hidden_size = enc_hidden

    // Special tokens
    int32_t bos_token;
    int32_t eos_token;
    int32_t pad_token;
    int32_t decoder_start_token;
} math_ocr_hparams;

/// Load a pix2tex GGUF model. Returns NULL on failure.
/// [model_path] — path to the .gguf file.
/// [n_threads]  — number of CPU threads for ggml.
math_ocr_context * math_ocr_init(const char * model_path, int n_threads);

/// Free the context and all associated memory.
void math_ocr_free(math_ocr_context * ctx);

/// Get the model hyperparameters.
const math_ocr_hparams * math_ocr_get_hparams(const math_ocr_context * ctx);

/// Run math OCR on a grayscale image.
///
/// [pixels] — row-major grayscale float array, values in [0, 1].
///            Size: width × height.
/// [width]  — image width in pixels.
/// [height] — image height in pixels.
/// [out_len] — on success, receives the length of the returned string.
///
/// Returns a null-terminated LaTeX string owned by the context (valid
/// until the next call to math_ocr_recognize or math_ocr_free).
/// Returns NULL on failure.
const char * math_ocr_recognize(
    math_ocr_context * ctx,
    const float * pixels,
    int width,
    int height,
    int * out_len
);

/// Run math OCR on a raw image file (JPEG or PNG).
/// Handles loading + grayscale conversion + preprocessing internally.
/// Returns a LaTeX string or NULL.
const char * math_ocr_recognize_file(
    math_ocr_context * ctx,
    const char * image_path,
    int * out_len
);

/// Run math OCR on raw pixel bytes (RGB or RGBA).
/// [pixel_bytes] — raw pixel data.
/// [width], [height] — dimensions.
/// [channels] — 1 (gray), 3 (RGB), or 4 (RGBA).
/// Returns a LaTeX string or NULL.
const char * math_ocr_recognize_raw(
    math_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width,
    int height,
    int channels,
    int * out_len
);

/// Beam search decoding (beam_width > 1) or greedy (beam_width <= 1).
const char * math_ocr_recognize_beam(
    math_ocr_context * ctx,
    const float * pixels,
    int width,
    int height,
    int beam_width,
    int * out_len
);

const char * math_ocr_recognize_raw_beam(
    math_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width,
    int height,
    int channels,
    int beam_width,
    int * out_len
);

/// After a successful recognize call, returns the encoder output.
/// Shape: (*out_n_tokens, *out_hidden). Valid until the next call.
const float * math_ocr_get_encoder_output(
    const math_ocr_context * ctx,
    int * out_n_tokens,
    int * out_hidden
);

/// Get per-character confidence scores from the last recognition.
/// Returns array of length *n_chars (one per output token).
/// Each value is the softmax probability of the winning token.
/// Valid until the next recognize call. Returns NULL if no recognition done.
const float * math_ocr_confidences(const math_ocr_context * ctx, int * n_chars);

/// Get mean confidence across all tokens from the last recognition.
float math_ocr_mean_confidence(const math_ocr_context * ctx);

/// Encode N crops in one batched encoder pass (single ggml graph call).
///
/// [crops]    — array of N raw image buffers (interleaved RGB, 3 channels each)
/// [widths]   — widths[i] is the width of crops[i]
/// [heights]  — heights[i] is the height of crops[i]
/// [n_crops]  — number of crops (batch size B)
///
/// On success, stores per-crop encoder outputs internally.  Subsequent calls
/// to math_ocr_decode_batch_crop() retrieve the text for each crop.
/// Returns true on success.
bool math_ocr_encode_batch_raw(
    math_ocr_context * ctx,
    const uint8_t * const * crops,
    const int * widths,
    const int * heights,
    int n_crops
);

/// Decode crop at index [crop_idx] using the encoder output stored by the
/// most recent math_ocr_encode_batch_raw() call.
///
/// [crop_idx] — index in [0, n_crops-1] from the encode call.
/// Returns a null-terminated string valid until the next decode or recognize
/// call.  Returns NULL on error or if encode has not been called.
const char * math_ocr_decode_batch_crop(
    math_ocr_context * ctx,
    int crop_idx,
    int * out_len
);

#ifdef __cplusplus
}
#endif
