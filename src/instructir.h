// instructir.h — InstructIR all-in-one image restoration (ECCV 2024, MIT).
//
// Text-guided NAFNet U-Net: handles denoise, deblur, dehaze, derain,
// super-resolution, low-light enhancement, and general enhancement.
// Task selected by integer ID (0-6). Pre-computed prompt embeddings
// baked into GGUF — no text encoder needed at runtime.
// ~16M params. Source: mv-lab/InstructIR.

#ifndef INSTRUCTIR_H
#define INSTRUCTIR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct instructir_context instructir_context;

/// Task IDs for InstructIR.
enum instructir_task {
    INSTRUCTIR_DENOISE = 0,
    INSTRUCTIR_DEBLUR = 1,
    INSTRUCTIR_DEHAZE = 2,
    INSTRUCTIR_DERAIN = 3,
    INSTRUCTIR_SUPER_RES = 4,
    INSTRUCTIR_LOW_LIGHT = 5,
    INSTRUCTIR_ENHANCE = 6,
};

instructir_context * instructir_init(const char * model_path, int n_threads);
void instructir_free(instructir_context * ctx);
int instructir_get_n_tasks(const instructir_context * ctx);

/// Restore an RGB image with the specified task.
/// Input/output: uint8 RGB [h, w, 3] (same dimensions, denoising task).
int instructir_process(instructir_context * ctx, int task,
                       const uint8_t * input, int width, int height,
                       uint8_t * output);

/// Float CHW [0,1] in/out (for parity testing).
int instructir_process_float(instructir_context * ctx, int task,
                             const float * input_chw, int width, int height,
                             float * output_chw);

#ifdef __cplusplus
}
#endif

#endif
