// hmer_ocr.h — Handwritten Math Expression Recognition via ggml.
//
// Architecture: DenseNet-121 encoder + GRU attention decoder (with coverage).
// Loads from GGUF produced by convert-hmer-to-gguf.py.
//
// Source model: whywhs/Pytorch-HMER (MIT license), trained on CROHME 2016.
// 112 LaTeX tokens, ~6.8M parameters, ~4-5 MB Q4_K.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hmer_ocr_context hmer_ocr_context;

typedef struct hmer_ocr_hparams {
    // Encoder (DenseNet-121)
    int32_t num_init_features;  // 64
    int32_t growth_rate;        // 32
    int32_t block_config[3];    // {6, 12, 24}
    int32_t input_channels;     // 2 (grayscale + mask)
    int32_t output_channels;    // 1024

    // Decoder (GRU + attention)
    int32_t hidden_size;        // 256
    int32_t output_size;        // 112 (vocab)
    int32_t sos_token;          // 111
    int32_t eol_token;          // 0
    int32_t max_seq_len;        // 48
} hmer_ocr_hparams;

/// Load an HMER GGUF model. Returns NULL on failure.
hmer_ocr_context * hmer_ocr_init(const char * model_path, int n_threads);

/// Free the context and all associated memory.
void hmer_ocr_free(hmer_ocr_context * ctx);

/// Get model hyperparameters.
const hmer_ocr_hparams * hmer_ocr_get_hparams(const hmer_ocr_context * ctx);

/// Run handwritten math OCR on a grayscale image.
///
/// [pixels]  — row-major grayscale float array, values in [0, 1].
/// [width]   — image width in pixels.
/// [height]  — image height in pixels.
/// [out_len] — receives the length of the returned string.
///
/// Returns a null-terminated LaTeX string owned by the context.
/// Valid until the next call to hmer_ocr_recognize or hmer_ocr_free.
const char * hmer_ocr_recognize(
    hmer_ocr_context * ctx,
    const float * pixels,
    int width, int height,
    int * out_len
);

/// Run handwritten math OCR on raw pixel bytes.
/// [channels] — 1 (gray), 3 (RGB), or 4 (RGBA).
const char * hmer_ocr_recognize_raw(
    hmer_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width, int height, int channels,
    int * out_len
);

#ifdef __cplusplus
}
#endif
