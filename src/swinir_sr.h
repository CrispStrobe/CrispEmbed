// swinir_sr.h — SwinIR-light image super-resolution (Apache-2.0).
//
// Swin Transformer for Image Restoration (ICCVW 2021, JingyunLiang).
// Lightweight variant: 4 RSTB × 6 Swin blocks, embed_dim=60, ~0.9M-4.2M params.
// Supports 2×, 3×, 4× upscale via PixelShuffle ending.
// Processes in overlapping tiles for large images.

#ifndef SWINIR_SR_H
#define SWINIR_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct swinir_sr_context swinir_sr_context;

// Initialize from GGUF model (general.architecture = "swinir").
swinir_sr_context * swinir_sr_init(const char * model_path, int n_threads);
void swinir_sr_free(swinir_sr_context * ctx);

// Query the upscale factor (2, 3, or 4).
int swinir_sr_scale(const swinir_sr_context * ctx);

// Upscale an RGB image.
// Input:  uint8 RGB [height, width, 3].
// Output: uint8 RGB [height*scale, width*scale, 3], allocated by this function.
// tile_size: 0 = auto (64). tile_overlap: 0 = auto (8).
// Caller frees with swinir_sr_free_image().
// Returns 0 on success, -1 on error.
int swinir_sr_process(swinir_sr_context * ctx,
                      const uint8_t * input, int width, int height,
                      int tile_size, int tile_overlap,
                      uint8_t ** output, int * out_width, int * out_height);

void swinir_sr_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // SWINIR_SR_H
