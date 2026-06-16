// adair.h — AdaIR all-in-one image restoration (ICLR 2025, MIT).
//
// Restormer backbone + Adaptive Frequency Learning Blocks (AFLB) with
// FFT-based spectral decomposition and cross-attention guidance.
// 28.8M params. 5 tasks: denoise, derain, dehaze, deblur, low-light.
// Source: c-yn/AdaIR.

#ifndef ADAIR_H
#define ADAIR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct adair_context adair_context;

adair_context * adair_init(const char * model_path, int n_threads);
void adair_free(adair_context * ctx);

/// Restore RGB image (same resolution, denoising-style).
/// Input/output: uint8 RGB [h, w, 3].
int adair_process(adair_context * ctx,
                  const uint8_t * input, int width, int height,
                  uint8_t * output);

/// Float CHW [0,1] in/out (for parity testing).
int adair_process_float(adair_context * ctx,
                        const float * input_chw, int width, int height,
                        float * output_chw);

#ifdef __cplusplus
}
#endif

#endif
