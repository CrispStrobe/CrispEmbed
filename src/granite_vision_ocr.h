// granite_vision_ocr.h — Granite Vision 3.3-2B OCR (LLaVA-Next architecture).
//
// Vision: SigLIP ViT-SO400M (27L, 1152d, 384px, patch=14)
// Projector: MLP (4×1152→2048→2048)
// LLM: Granite-3.1-2B (40L, 2048d, GQA 32/8 heads)
// Multi-layer vision features from layers [-24, -20, -12, -1].
// LLaVA-Next dynamic resolution tiling.
//
// OCRBench 852 — highest in class for ≤3B models.
// Apache-2.0 (ibm-granite).

#ifndef GRANITE_VISION_OCR_H
#define GRANITE_VISION_OCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct granite_vision_context granite_vision_context;

granite_vision_context * granite_vision_init(const char * model_path, int n_threads);
void granite_vision_free(granite_vision_context * ctx);

// Recognize text from an image. Returns UTF-8 text (owned by ctx, valid
// until next call or free). prompt can be NULL for default OCR prompt.
const char * granite_vision_recognize(granite_vision_context * ctx,
                                      const uint8_t * pixels, int width, int height, int channels,
                                      const char * prompt, int * out_len);

// Set max generation tokens (default: 2048).
void granite_vision_set_max_tokens(granite_vision_context * ctx, int max_tokens);

// Run vision encoder + projector only (for crispembed-diff parity).
// image_f32: [3, img_h, img_w] float [0,1]. Callback receives named stages.
typedef void (*gv_dump_cb)(const char * name, const float * data, int n, void * ud);
void granite_vision_dump_vision(granite_vision_context * ctx,
                                 const float * image_f32, int img_h, int img_w,
                                 gv_dump_cb cb, void * ud);

// Per-token confidence from the last recognition.
const float * granite_vision_confidences(const granite_vision_context * ctx, int * n_tokens);
float granite_vision_mean_confidence(const granite_vision_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // GRANITE_VISION_OCR_H
