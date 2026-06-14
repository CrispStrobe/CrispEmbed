// nafnet_denoise.h — NAFNet denoising CNN (megvii-research/NAFNet, MIT)
//
// U-Net with NAFBlocks (Non-linear Activation Free Network).
// Pre-trained on SIDD for image denoising.
// All operations are standard: Conv2d, LayerNorm2d, SimpleGate,
// depthwise conv, channel attention, PixelShuffle.
//
// Architecture (width32 SIDD):
//   Intro(3→32) → Enc[2,2,4,8] → Middle(12) → Dec[2,2,2,2] → Ending(32→3) + residual
//   Channels: 32 → 64 → 128 → 256 → 512 → 256 → 128 → 64 → 32

#ifndef NAFNET_DENOISE_H
#define NAFNET_DENOISE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nafnet_context nafnet_context;

// Initialize from GGUF model.
nafnet_context * nafnet_init(const char * model_path, int n_threads);
void nafnet_free(nafnet_context * ctx);

// Denoise an RGB image.
// Input: uint8 RGB pixels [h, w, 3].
// Output: uint8 RGB pixels [h, w, 3], allocated by caller (same size as input).
// Returns 0 on success, -1 on error.
int nafnet_process(nafnet_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output);

#ifdef __cplusplus
}
#endif

#endif // NAFNET_DENOISE_H
