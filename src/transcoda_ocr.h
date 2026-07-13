// transcoda_ocr.h — Transcoda-59M zero-shot Optical Music Recognition via ggml.
//
// Full-page score image → Humdrum **kern token sequence (compact 3000-token BPE).
//
// Architecture (clean-room from the paper arXiv 2605.10835, the HF config/data
// files, and an oracle activation dump — the AGPL reference *code* is never read):
//   Encoder : ConvNeXt-V2-Tiny (facebook/convnextv2-tiny-22k-224), run fully-
//             convolutionally on a 1485x1050 RGB page. dims [96,192,384,768],
//             depths [3,3,9,3], /32 → a 46x32x768 grid.  V2 block = dwconv7x7 →
//             LN → pwconv1(×4) → GELU → GRN → pwconv2, residual, NO LayerScale.
//   Bridge  : 2-layer projector 768→2048→512 + 2D sinusoidal PE over the grid.
//             cross-attn KEY = memory+PE, VALUE = memory (raw) — dual memory.
//   Decoder : 8-layer pre-LN cross-attn Transformer. d_model 512, 8 heads (hd 64),
//             ffn 1024 (GELU-erf). Self-attn RoPE (torchtune, θ=1e4, interleaved
//             pairs), causal, scale 1/√64. Untied LM head (vocab_projection).
//
// Weights are cc-by-4.0 (btrkeks/transcoda-59M-zeroshot-v1); attribution required.
// Loaded from a GGUF produced by models/convert-transcoda-to-gguf.py.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct transcoda_ocr_context transcoda_ocr_context;

typedef struct transcoda_ocr_hparams {
    int32_t d_model;       // 512
    int32_t n_layers;      // 8
    int32_t n_heads;       // 8  (head_dim 64)
    int32_t dim_ff;        // 1024
    int32_t vocab_size;    // 3000
    int32_t max_seq_len;   // 2048
    float rope_theta;      // 10000
    // encoder
    int32_t enc_num_stages;   // 4
    int32_t enc_num_channels; // 3
    int32_t enc_stem_kernel;  // 4
    int32_t enc_reduction;    // 32
    int32_t fixed_h, fixed_w; // 1485 x 1050
    float mean0, mean1, mean2; // 0.5 / 0.5 / 0.5
    float std0, std1, std2;    // 0.5 / 0.5 / 0.5
    int32_t bos_token, eos_token, pad_token;
} transcoda_ocr_hparams;

transcoda_ocr_context * transcoda_ocr_init(const char * model_path, int n_threads);
void transcoda_ocr_free(transcoda_ocr_context * ctx);
const transcoda_ocr_hparams * transcoda_ocr_get_hparams(const transcoda_ocr_context * ctx);

/// Recognize on a raw pixel buffer (RGB/RGBA/gray). Returns a '/'-joined **kern
/// token string, owned by ctx, or NULL.
const char * transcoda_ocr_recognize_raw(transcoda_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                         int channels, int * out_len);

/// Recognize on a raw image file (PNG/JPEG); handles preprocessing internally.
const char * transcoda_ocr_recognize_file(transcoda_ocr_context * ctx, const char * image_path, int * out_len);

/// Per-stage parity harness vs a reference GGUF (tools/dump_transcoda_reference.py).
int transcoda_ocr_run_diff(transcoda_ocr_context * ctx, const char * ref_gguf_path);

#ifdef __cplusplus
}
#endif
