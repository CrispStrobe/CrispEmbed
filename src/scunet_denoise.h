// scunet_denoise.h — SCUNet image denoising (Swin-Conv-UNet, Apache-2.0).
//
// Hybrid U-Net combining Swin Transformer blocks (shifted window attention)
// with residual Conv blocks. ~18M params, 70 MB F16 GGUF.
// Source: cszn/SCUNet (CVPR 2022, Apache-2.0).

#ifndef SCUNET_DENOISE_H
#define SCUNET_DENOISE_H

#include <stdint.h>

#include <functional>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scunet_context scunet_context;

scunet_context * scunet_init(const char * model_path, int n_threads);
void scunet_free(scunet_context * ctx);

/// Denoise RGB image. Input/output: uint8 RGB [h, w, 3].
/// Caller allocates output (same size as input).
int scunet_process(scunet_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output);

/// Float CHW [0,1] in/out (for parity testing).
int scunet_process_float(scunet_context * ctx,
                         const float * input_chw, int width, int height,
                         float * output_chw);

#ifdef __cplusplus
}

/// Debug variant with per-stage callback for parity testing.
using scunet_stage_cb = std::function<void(const char * name, const float * data, int n)>;
int scunet_process_float_debug(scunet_context * ctx,
                                const float * input_chw, int width, int height,
                                float * output_chw,
                                scunet_stage_cb cb);
#endif

#endif
