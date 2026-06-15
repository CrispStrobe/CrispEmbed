// esrgan_sr.h — Real-ESRGAN SRVGGNetCompact super-resolution.
//
// Lightweight 4× SR (~620K params) from xinntao/Real-ESRGAN (BSD-3-Clause).
// Architecture: 17× Conv3x3+PReLU + PixelShuffle(4) + global skip.

#ifndef ESRGAN_SR_H
#define ESRGAN_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct esrgan_context esrgan_context;

esrgan_context * esrgan_init(const char * model_path, int n_threads);
void esrgan_free(esrgan_context * ctx);
int esrgan_get_scale(const esrgan_context * ctx);

/// Super-resolve: uint8 RGB HWC in → uint8 RGB HWC out (caller allocates).
int esrgan_process(esrgan_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output);

/// Float CHW [0,1] in/out (for parity testing).
int esrgan_process_float(esrgan_context * ctx,
                         const float * input_chw, int width, int height,
                         float * output_chw);

#ifdef __cplusplus
}
#endif

#endif
