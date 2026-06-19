// smoldocling_ocr.h — SmolDocling OCR (SigLIP ViT + SmolLM2-135M decoder).
//
// Vision: SigLIP ViT (12L, 768d, 512px, patch=16, 1024 patches)
// Connector: pixel shuffle (scale=4) + linear projection (12288 -> 576)
// LLM: SmolLM2-135M (30L, 576d, GQA 9/3 heads, SwiGLU)
//
// Generates DocTags (structured document markup) from page images.
// Apache-2.0 (DS4SD/SmolDocling-256M-preview).

#ifndef SMOLDOCLING_OCR_H
#define SMOLDOCLING_OCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct smoldocling_context smoldocling_context;

smoldocling_context * smoldocling_init(const char * model_path, int n_threads);
void smoldocling_free(smoldocling_context * ctx);

// Recognize text from an image file (JPEG/PNG). Returns UTF-8 DocTags
// (owned by ctx, valid until next call or free).
const char * smoldocling_recognize(smoldocling_context * ctx,
                                   const char * image_path, int * out_len);

// Recognize from raw pixel bytes (RGB/RGBA/gray).
const char * smoldocling_recognize_raw(smoldocling_context * ctx,
                                       const uint8_t * pixels,
                                       int w, int h, int ch,
                                       int * out_len);

// Debug: run vision encoder only, return [n_tokens * vis_dim] floats.
// Caller must free() the returned pointer.
float * smoldocling_debug_vision(smoldocling_context * ctx,
                                 const uint8_t * pixels,
                                 int w, int h, int ch,
                                 int * out_n_tokens, int * out_dim);

// Debug: run vision + connector, return [n_tokens * llm_dim] floats.
float * smoldocling_debug_connector(smoldocling_context * ctx,
                                     const uint8_t * pixels,
                                     int w, int h, int ch,
                                     int * out_n_tokens, int * out_dim);

#ifdef __cplusplus
}
#endif

#endif // SMOLDOCLING_OCR_H
