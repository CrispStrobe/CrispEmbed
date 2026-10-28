#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Experimental MiniCPM-OCR-1.0 (openbmb/MiniCPM-OCR-1.0 @6796197b) CPU runtime.
// Default-off: requires CRISPEMBED_MINICPM_OCR=1. The text decoder reuses the
// qwen35 kernels, which carry their own gate, so CRISPEMBED_QWEN35_OCR=1 is
// needed as well. NOT parity-qualified; see docs/minicpm-ocr-inference-spec.md.
//
// Debug: CRISPEMBED_MINICPM_OCR_DUMP=<dir> writes .npy captures named like the
// reference (tools/minicpm-ocr/reference.py) for stage bisection.
typedef struct minicpm_ocr_context minicpm_ocr_context;
minicpm_ocr_context * minicpm_ocr_init(const char * model_path, int n_threads);
void minicpm_ocr_free(minicpm_ocr_context * ctx);
int minicpm_ocr_set_max_tokens(minicpm_ocr_context * ctx, int max_tokens);
int minicpm_ocr_get_max_tokens(const minicpm_ocr_context * ctx);
// Task prompt, e.g. "Text Recognition:" (default), "Table Recognition:",
// "Formula Recognition:", "Seal Recognition:", "Chart to Table:". Null/empty resets.
int minicpm_ocr_set_prompt(minicpm_ocr_context * ctx, const char * prompt);
// RGB, grayscale or RGBA contiguous uint8. Returned text lives until the next call/free.
const char * minicpm_ocr_recognize_raw(minicpm_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len);
const int * minicpm_ocr_token_ids(const minicpm_ocr_context * ctx, int * count);
const float * minicpm_ocr_confidences(const minicpm_ocr_context * ctx, int * count);
const char * minicpm_ocr_last_error(const minicpm_ocr_context * ctx);
#ifdef __cplusplus
}
#endif
