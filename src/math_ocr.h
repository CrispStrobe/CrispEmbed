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
typedef struct math_ocr_hparams {
    int32_t encoder_layers;
    int32_t decoder_layers;
    int32_t dim;
    int32_t heads;
    int32_t vocab_size;
    int32_t max_seq_len;
    int32_t patch_size;
    int32_t max_height;
    int32_t max_width;
    int32_t channels;
    int32_t bos_token;
    int32_t eos_token;
    int32_t pad_token;
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

#ifdef __cplusplus
}
#endif
