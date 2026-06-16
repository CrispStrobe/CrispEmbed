// dat_sr.h — DAT (Dual Aggregation Transformer) super-resolution.
//
// Architecture (DAT-light x2):
//   Conv3x3(3, 60) → LN → 18× DATB (spatial+channel dual attention) → LN
//   → 3conv residual → PixelShuffleDirect(2x)
//
// Each DATB block:
//   LN → Adaptive Spatial Attention (window-based, shifted, split channels)
//   → LN → Adaptive Channel Attention (transposed attention + AIM)
//   → LN → FFN (GELU, expansion=2)
//
// Source: https://github.com/zhengchen1999/DAT (Apache-2.0, ICCV 2023)
// ~830K params for DAT-light, ~1.6 MB F16 GGUF.

#ifndef DAT_SR_H
#define DAT_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dat_sr_context dat_sr_context;

/// Load a DAT SR GGUF model (general.architecture = "dat").
dat_sr_context * dat_sr_init(const char * model_path, int n_threads);

/// Upscale an RGB image. Returns newly allocated RGB buffer.
/// [pixels]: RGB uint8 row-major, [w]×[h].
/// [out_w], [out_h]: receive output dimensions.
/// Returns NULL on failure. Caller frees with dat_sr_free_image().
int dat_sr_process(dat_sr_context * ctx,
                   const uint8_t * pixels, int w, int h,
                   int tile_w, int tile_h,
                   uint8_t ** out, int * out_w, int * out_h);

void dat_sr_free_image(uint8_t * pixels);
void dat_sr_free(dat_sr_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // DAT_SR_H
