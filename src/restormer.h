// restormer.h — Restormer image restoration (denoise/deblur/SR).
//
// U-Net with Multi-DConv Head Transposed Attention (MDTA) + Gated-DConv
// Feed-Forward (GDFN). CVPR 2022, Apache-2.0. ~26M params, ~50 MB F16.
//
// Transposed attention: operates over channels (C×C), not spatial (HW×HW),
// making it efficient for high-resolution images without windowing.
//
// Processes in overlapping tiles with Hann blending for large images.
// Direct NAFNet replacement with better quality on real-world degradation.

#ifndef RESTORMER_H
#define RESTORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct restormer_context restormer_context;

// Initialize from GGUF model (general.architecture = "restormer").
restormer_context * restormer_init(const char * model_path, int n_threads);
void restormer_free(restormer_context * ctx);

// Process an RGB image (denoise/restore).
// Input:  uint8 RGB [height, width, 3].
// Output: uint8 RGB [height, width, 3], allocated by this function.
// tile_size: 0 = auto (128). tile_overlap: 0 = auto (16).
// Caller frees with restormer_free_image().
// Returns 0 on success, -1 on error.
int restormer_process(restormer_context * ctx,
                      const uint8_t * input, int width, int height,
                      int tile_size, int tile_overlap,
                      uint8_t ** output);

void restormer_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // RESTORMER_H
