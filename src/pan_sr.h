// pan_sr.h — PAN (Pixel Attention Network) whole-image super-resolution.
//
// Upscales any RGB image by 4× (or 2×). PaddleGAN PAN, Apache-2.0.
// Architecture: SCPA blocks (self-calibrated convolution + pixel attention)
// with nearest-neighbor upsample + pixel attention refinement.
// ~272K params, ~0.5 MB F16 GGUF. Processes in overlapping tiles for large images.

#ifndef PAN_SR_H
#define PAN_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pan_sr_context pan_sr_context;

// Initialize from GGUF model (general.architecture = "pan").
pan_sr_context * pan_sr_init(const char * model_path, int n_threads);
void pan_sr_free(pan_sr_context * ctx);

// Query the upscale factor (2 or 4).
int pan_sr_scale(const pan_sr_context * ctx);

// Upscale an RGB image.
// Input:  uint8 RGB [height, width, 3].
// Output: uint8 RGB [height*scale, width*scale, 3], allocated by this function.
// tile_size: 0 = auto (128). tile_overlap: 0 = auto (16).
// Caller frees with pan_sr_free_image().
// Returns 0 on success, -1 on error.
int pan_sr_process(pan_sr_context * ctx,
                   const uint8_t * input, int width, int height,
                   int tile_size, int tile_overlap,
                   uint8_t ** output, int * out_width, int * out_height);

void pan_sr_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // PAN_SR_H
