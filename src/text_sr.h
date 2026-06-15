// text_sr.h — Text super-resolution via NAFNet U-Net + PixelShuffle.
//
// Upscales low-resolution text images (72-150 DPI) to improve OCR accuracy.
// Architecture: NAFNet U-Net (same blocks as nafnet_denoise) with a
// PixelShuffle ending that outputs r*r*3 channels, rearranged to (3, H*r, W*r).
// A bicubic-upscaled input is added as a global residual.
//
// The model processes at the LOW resolution (fast) and only upscales at the end.
// Large images are tiled with overlap and blended to avoid seam artifacts.

#ifndef TEXT_SR_H
#define TEXT_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct text_sr_context text_sr_context;

// Initialize from a GGUF model (NAFNet-SR variant).
// The GGUF must contain "text_sr.upscale_factor" metadata (2 or 4).
text_sr_context * text_sr_init(const char * model_path, int n_threads);
void text_sr_free(text_sr_context * ctx);

// Query the upscale factor (2 or 4) from the loaded model.
int text_sr_upscale_factor(const text_sr_context * ctx);

// Upscale an RGB image.
// Input:  uint8 RGB pixels [height, width, 3].
// Output: uint8 RGB pixels [height*r, width*r, 3], allocated by this function.
//         Caller frees with text_sr_free_image().
// tile_size: processing tile size (0 = auto, typically 256).
// tile_overlap: overlap between tiles in pixels (0 = auto, typically 32).
// Returns 0 on success, -1 on error.
int text_sr_process(text_sr_context * ctx,
                    const uint8_t * input, int width, int height,
                    int tile_size, int tile_overlap,
                    uint8_t ** output, int * out_width, int * out_height);

void text_sr_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // TEXT_SR_H
