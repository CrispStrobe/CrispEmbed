// hat_sr.h — HAT (Hybrid Attention Transformer) super-resolution.
//
// CVPR 2023, MIT license. Swin-style window attention + overlapping
// cross-attention (OCAB) + channel attention blocks (CAB).
// SOTA single-image SR quality. ~21M params, ~40 MB F16 GGUF.
//
// Processes in overlapping tiles with Hann blending for large images.

#ifndef HAT_SR_H
#define HAT_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hat_sr_context hat_sr_context;

hat_sr_context * hat_sr_init(const char * model_path, int n_threads);
void hat_sr_free(hat_sr_context * ctx);

int hat_sr_scale(const hat_sr_context * ctx);

// Upscale an RGB image. Output allocated by this function.
// tile_size: 0 = auto (64). tile_overlap: 0 = auto (8).
// Caller frees with hat_sr_free_image().
int hat_sr_process(hat_sr_context * ctx,
                   const uint8_t * input, int width, int height,
                   int tile_size, int tile_overlap,
                   uint8_t ** output, int * out_width, int * out_height);

void hat_sr_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // HAT_SR_H
