// ppformulanet_l_ocr.h — PP-FormulaNet-L math OCR via ggml.
//
// Architecture: SAM-style ViT encoder + MBart Transformer decoder.
// Loads from GGUF produced by convert-ppformulanet-l-to-gguf.py.
//
// Source: PaddlePaddle/PP-FormulaNet-L_safetensors (Apache-2.0).
// 181M parameters, 768×768 RGB input, outputs LaTeX tokens.
//
// Encoder: SAM ViT — 12 layers (768d, 12 heads), windowed + global attention.
//   Patch embed (16×16 → 48×48 patches) + pos_embed
//   Neck: Conv1×1 + LayerNorm2d + Conv3×3 + LayerNorm2d (768 → 256)
//   Projector: Conv3×3(s2) + Conv3×3(s2) + Linear + Linear (256 → 512)
//   Output: (144, 512) for decoder.
//
// Decoder: MBart PRE-LN — 8 layers, 16 heads, d_model=512, FFN=2048.
//   scale_embedding, greedy decoding.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppformulanet_l_ocr_context ppformulanet_l_ocr_context;

typedef struct ppformulanet_l_ocr_hparams {
    // Encoder (SAM ViT)
    int32_t image_size;       // 768
    int32_t patch_size;       // 16
    int32_t enc_hidden;       // 768
    int32_t enc_layers;       // 12
    int32_t enc_heads;        // 12
    int32_t enc_mlp_dim;      // 3072
    int32_t window_size;      // 14
    int32_t n_patches;        // 48 (image_size / patch_size)
    int32_t output_channels;  // 256 (neck output)

    // Decoder (MBart)
    int32_t dec_layers;       // 8
    int32_t dec_heads;        // 16
    int32_t dec_d_model;      // 512
    int32_t dec_ffn_dim;      // 2048
    int32_t vocab_size;       // 50000
    int32_t max_seq_len;      // 1024

    // Special tokens
    int32_t bos_token;
    int32_t eos_token;
    int32_t pad_token;
    int32_t decoder_start_token;
} ppformulanet_l_ocr_hparams;

/// Load a PP-FormulaNet-L GGUF model. Returns NULL on failure.
ppformulanet_l_ocr_context * ppformulanet_l_ocr_init(const char * model_path, int n_threads);

/// Free the context and all associated memory.
void ppformulanet_l_ocr_free(ppformulanet_l_ocr_context * ctx);

/// Get the model hyperparameters.
const ppformulanet_l_ocr_hparams * ppformulanet_l_ocr_get_hparams(const ppformulanet_l_ocr_context * ctx);

/// Run OCR on a grayscale image.
/// [pixels] — row-major grayscale float array, values in [0, 1].
const char * ppformulanet_l_ocr_recognize(
    ppformulanet_l_ocr_context * ctx,
    const float * pixels,
    int width, int height,
    int * out_len
);

/// Run OCR on raw pixel bytes (RGB or RGBA).
const char * ppformulanet_l_ocr_recognize_raw(
    ppformulanet_l_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width, int height, int channels,
    int * out_len
);

/// Run OCR on pre-processed CHW float tensor.
const char * ppformulanet_l_ocr_recognize_chw(
    ppformulanet_l_ocr_context * ctx,
    const float * chw_data,
    int * out_len
);

/// After a successful recognize call, returns the projected encoder output.
/// Shape: (*out_n_tokens, *out_hidden). Valid until the next call.
const float * ppformulanet_l_ocr_get_encoder_output(
    const ppformulanet_l_ocr_context * ctx,
    int * out_n_tokens,
    int * out_hidden
);

#ifdef __cplusplus
}
#endif
