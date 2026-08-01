// tromr_ocr.h — Polyphonic-TrOMR optical music recognition via ggml.
//
// Staff-notation image → three parallel token streams (rhythm / pitch / lift),
// mergeable into semantic music notation (e.g. clef-G2+note-C4_quarter...).
//
// Architecture (github.com/NetEase/Polyphonic-TrOMR, Apache-2.0):
//   Encoder : timm hybrid ViT — ResNetV2 backbone (StdConv2dSame + GroupNorm,
//             layers [2,3,7], /16) → HybridEmbed 1x1 proj (1024→256) → ViT
//             (depth 4, 8 heads, dim 256, cls token, custom 2D pos-index).
//   Decoder : x_transformers Decoder (depth 4, 8 heads, dim 256; per depth:
//             self-attn → cross-attn → GLU-FF, attn-on-attn gating). Input =
//             rhythm+pitch+lift+abs-pos embeddings. 4 heads (rhythm/pitch/lift/
//             note). Autoregressive: predicts all 3 streams per step.
//
// Loaded from a GGUF produced by models/convert-tromr-to-gguf.py (arch "tromr_ocr").

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tromr_ocr_context tromr_ocr_context;

typedef struct tromr_ocr_hparams {
    int32_t channels;          // 1
    int32_t patch_size;        // 16
    int32_t max_height;        // 128
    int32_t max_width;         // 1280
    int32_t max_seq_len;       // 256
    int32_t encoder_dim;       // 256
    int32_t encoder_depth;     // 4 (ViT blocks)
    int32_t encoder_heads;     // 8
    int32_t decoder_dim;       // 256
    int32_t decoder_depth;     // 4 (x_transformers depths, ×3 sublayers)
    int32_t decoder_heads;     // 8
    int32_t num_rhythm_tokens; // 260
    int32_t num_pitch_tokens;  // 71
    int32_t num_lift_tokens;   // 7
    int32_t num_note_tokens;   // 2
    int32_t bos_token, eos_token, pad_token, nonote_token;
    float norm_mean, norm_std; // 0.7931 / 0.1738
} tromr_ocr_hparams;

tromr_ocr_context * tromr_ocr_init(const char * model_path, int n_threads);
void tromr_ocr_free(tromr_ocr_context * ctx);
const tromr_ocr_hparams * tromr_ocr_get_hparams(const tromr_ocr_context * ctx);

/// Recognize on a raw pixel buffer (RGB/RGBA/gray). Returns a space-joined
/// merged token string ("rhythm|pitch" per event), owned by ctx, or NULL.
const char * tromr_ocr_recognize_raw(tromr_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                     int channels, int * out_len);

/// Recognize on a raw image file (PNG/JPEG); handles preprocessing internally.
const char * tromr_ocr_recognize_file(tromr_ocr_context * ctx, const char * image_path, int * out_len);

/// Per-stage parity harness vs a reference GGUF (tools/dump_tromr_reference.py).
int tromr_ocr_run_diff(tromr_ocr_context * ctx, const char * ref_gguf_path);

#ifdef __cplusplus
}
#endif
