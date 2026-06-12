// mixtex_ocr.h — MixTex Chinese+English LaTeX OCR
//
// Architecture: Swin-Tiny encoder + 4-layer RoBERTa decoder
//   86M params, input 500×400 RGB, output LaTeX tokens (BPE, 25681 vocab)
//
// Port of: MixTex/ZhEn-Latex-OCR (Apache-2.0)

#ifndef MIXTEX_OCR_H
#define MIXTEX_OCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mixtex_ocr_context mixtex_ocr_context;

typedef struct {
    // Encoder (Swin-Tiny)
    int patch_size;         // 4
    int window_size;        // 7
    int embed_dim;          // 96
    int enc_depths[4];      // {2, 2, 6, 2}
    int enc_heads[4];       // {3, 6, 12, 24}
    int enc_hidden;         // 768
    int image_h, image_w;   // 400, 500

    // Decoder (RoBERTa)
    int dec_hidden;         // 768
    int dec_layers;         // 4
    int dec_heads;          // 12
    int dec_ffn;            // 3072
    int vocab_size;         // 25681
    int max_position;       // 300
    int sos_token;          // 0
    int eos_token;          // 25678
} mixtex_ocr_hparams;

// Initialize from GGUF
mixtex_ocr_context * mixtex_ocr_init(const char * model_path, int n_threads);
void mixtex_ocr_free(mixtex_ocr_context * ctx);
const mixtex_ocr_hparams * mixtex_ocr_get_hparams(const mixtex_ocr_context * ctx);

// Recognize LaTeX from raw pixel bytes (RGB/RGBA/grayscale)
const char * mixtex_ocr_recognize(mixtex_ocr_context * ctx,
                                   const uint8_t * pixels,
                                   int width, int height, int channels,
                                   int * out_len);

// Recognize from float grayscale [0..1]
const char * mixtex_ocr_recognize_gray(mixtex_ocr_context * ctx,
                                        const float * pixels,
                                        int width, int height,
                                        int * out_len);

#ifdef __cplusplus
}
#endif

#endif // MIXTEX_OCR_H
