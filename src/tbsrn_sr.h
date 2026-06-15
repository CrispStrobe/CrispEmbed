// tbsrn_sr.h — TBSRN text-line super-resolution (PaddleOCR Telescope).
//
// Upscales a single text-line crop from 16×64 → 32×128 (2× factor).
// Architecture: Conv9+PReLU → 5× RecurrentResidualBlock(FeatureEnhancer)
//   → Conv3+BN → UpsampleBlock(PixelShuffle) → Conv9 → tanh.
//
// Designed to run between text detection and recognition: the detector
// crops text lines, each is resized to 16×64, upscaled by TBSRN to
// 32×128, then fed to the recognizer.
//
// Model: PaddleOCR sr_telescope (Apache-2.0), ~1.1M inference params, ~2MB F16.

#ifndef TBSRN_SR_H
#define TBSRN_SR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tbsrn_sr_context tbsrn_sr_context;

// Initialize from GGUF model (general.architecture = "tbsrn").
tbsrn_sr_context * tbsrn_sr_init(const char * model_path, int n_threads);
void tbsrn_sr_free(tbsrn_sr_context * ctx);

// Upscale a text-line crop.
// Input:  uint8 RGB pixels [h_in, w_in, 3]. Will be resized to 16×64 internally.
// Output: uint8 RGB pixels [32, 128, 3], allocated by this function.
//         Caller frees with tbsrn_sr_free_image().
// Returns 0 on success, -1 on error.
int tbsrn_sr_process(tbsrn_sr_context * ctx,
                     const uint8_t * input, int width, int height,
                     uint8_t ** output, int * out_width, int * out_height);

void tbsrn_sr_free_image(uint8_t * pixels);

#ifdef __cplusplus
}
#endif

#endif // TBSRN_SR_H
