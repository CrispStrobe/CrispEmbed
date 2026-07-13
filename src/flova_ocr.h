// flova_ocr.h — Flova/omr_transformer optical music recognition via ggml.
//
// Handwritten / whiteboard staff image → LilyPond "simple notes" string
// (e.g. "c'2 a''8 c''8 r4 …"). The only permissive (Apache-2.0) handwritten-music
// OMR model. Donut VisionEncoderDecoder:
//   Encoder : DonutSwin (Swin-Base scale) — patch 4, window 10, embed_dim 128,
//             depths [2,2,14,2], heads [4,8,16,32], hidden 1024, image 583×409.
//   Decoder : mBART 4-layer (pre-norm) — d_model 1024, 16 heads, ffn 4096,
//             vocab 75, learned positions (offset 2), scale_embedding, GELU.
//             decoder_start/bos 56, eos 54 (</s>), pad 55.
//
// Loaded from a GGUF produced by models/convert-flova-to-gguf.py (arch "flova_ocr").

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct flova_ocr_context flova_ocr_context;

flova_ocr_context * flova_ocr_init(const char * model_path, int n_threads);
void flova_ocr_free(flova_ocr_context * ctx);

/// Recognize on a raw pixel buffer (RGB/RGBA/gray). Returns a LilyPond string
/// owned by ctx, or NULL.
const char * flova_ocr_recognize_raw(flova_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                     int channels, int * out_len);

/// Recognize on a raw image file (PNG/JPEG); handles preprocessing internally.
const char * flova_ocr_recognize_file(flova_ocr_context * ctx, const char * image_path, int * out_len);

/// Per-stage parity harness vs a reference GGUF (tools/dump_flova_reference.py).
int flova_ocr_run_diff(flova_ocr_context * ctx, const char * ref_gguf_path);

#ifdef __cplusplus
}
#endif
