// parseq_ocr.h — PARSeq scene text recognition via ggml.
//
// Architecture: ViT encoder + 1-layer two-stream Transformer decoder.
//   Encoder: 12-layer pre-LN ViT, GELU FFN, learned pos embed, patch [4,8]
//   Decoder: position queries + context self-attn → cross-attn → FFN
//   Head: Linear(embed_dim, 95) → 94 printable ASCII chars + EOS
//
// Source: baudm/parseq (Apache-2.0), ECCV 2022
// Loads GGUF from convert-parseq-to-gguf.py.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct parseq_ocr_context parseq_ocr_context;

typedef struct parseq_ocr_hparams {
    // Encoder (ViT)
    int32_t embed_dim;      // 384 (base) or 192 (tiny)
    int32_t enc_layers;     // 12
    int32_t enc_heads;      // 6 (base) or 3 (tiny)
    int32_t ffn_dim;        // 1536 (base) or 768 (tiny)
    int32_t patch_h;        // 4
    int32_t patch_w;        // 8
    int32_t img_h;          // 32
    int32_t img_w;          // 128
    int32_t n_patches;      // 128

    // Decoder
    int32_t dec_heads;      // 12 (base) or 6 (tiny), head_dim=32
    int32_t dec_ffn;        // 1536 (base) or 768 (tiny)
    int32_t max_label_len;  // 26 (T+1 = 25 chars + 1)

    // Tokens
    int32_t vocab_size;     // 95 (head output: 94 chars + EOS)
    int32_t n_tokens;       // 97 (embedding: 94 chars + BOS + EOS + PAD)
    int32_t bos_token;      // 0
    int32_t eos_token;      // 95 (in head output space: index 94)
    int32_t pad_token;      // 96
} parseq_ocr_hparams;

parseq_ocr_context * parseq_ocr_init(const char * model_path, int n_threads);
void                 parseq_ocr_free(parseq_ocr_context * ctx);
const parseq_ocr_hparams * parseq_ocr_get_hparams(const parseq_ocr_context * ctx);

/// Recognize text from a grayscale float image (values in [0,1]).
/// Returns null-terminated string owned by ctx.
const char * parseq_ocr_recognize(
    parseq_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len);

/// Recognize text from raw pixel bytes (RGB/RGBA/gray).
/// Handles preprocessing internally.
const char * parseq_ocr_recognize_raw(
    parseq_ocr_context * ctx,
    const uint8_t * pixel_bytes, int width, int height, int channels,
    int * out_len);

/// Get per-character confidence scores from the last recognition.
/// Returns array of length *n_chars (one per output character).
/// Each value is the softmax probability of the winning token.
/// Valid until the next recognize call. Returns NULL if no recognition done.
const float * parseq_ocr_confidences(
    const parseq_ocr_context * ctx,
    int * n_chars);

/// Get mean confidence across all characters from the last recognition.
float parseq_ocr_mean_confidence(const parseq_ocr_context * ctx);

#ifdef __cplusplus
}
#endif
