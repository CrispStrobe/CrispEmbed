// safmn_sr.h — SAFMN super-resolution (Spatially-Adaptive Feature Modulation).
//
// Lightweight SR model (~228K params) from ICCV 2023. Upscales images 2×/4×.
// Architecture: Conv3x3 → 8 AttBlocks (SAFM + CCM) → Conv3x3 + PixelShuffle.
// Source: sunny2109/SAFMN (Apache-2.0).

#ifndef SAFMN_SR_H
#define SAFMN_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct safmn_context safmn_context;

/// Load SAFMN GGUF model.
safmn_context * safmn_init(const char * model_path, int n_threads);
void safmn_free(safmn_context * ctx);

/// Get the upscaling factor (2 or 4).
int safmn_get_scale(const safmn_context * ctx);

/// Super-resolve an RGB image.
/// Input: uint8 RGB [h, w, 3]. Output: uint8 RGB [h*scale, w*scale, 3].
/// Caller allocates output (w*scale * h*scale * 3 bytes).
/// Returns 0 on success, -1 on error.
int safmn_process(safmn_context * ctx,
                  const uint8_t * input, int width, int height,
                  uint8_t * output);

/// Super-resolve and return float output [3, H*scale, W*scale] (CHW, [0,1]).
/// Caller allocates (3 * w*scale * h*scale) floats.
/// For parity testing — avoids uint8 clamping.
int safmn_process_float(safmn_context * ctx,
                        const float * input_chw, int width, int height,
                        float * output_chw);

#ifdef __cplusplus
}
#endif

#endif // SAFMN_SR_H
