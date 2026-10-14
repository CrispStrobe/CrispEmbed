#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Experimental owning CPU context. Requires CRISPEMBED_QWEN35_OCR=1.
// Pinned LightOnOCR-3-0.8B format only. Q8_0/Q4_K/Q6_K matrices additionally
// require CRISPEMBED_QWEN35_QUANT=1; small vectors/convolutions remain F32/F16.
// One context is not reentrant; independent contexts own independent state.
typedef struct qwen35_ocr_context qwen35_ocr_context;
typedef enum qwen35_ocr_stop {
    QWEN35_OCR_NONE = 0,
    QWEN35_OCR_EOS = 1,
    QWEN35_OCR_MAX_TOKENS = 2,
    QWEN35_OCR_CONTEXT_LIMIT = 3,
    QWEN35_OCR_ERROR = 4
} qwen35_ocr_stop;
qwen35_ocr_context * qwen35_ocr_init(const char * model_path, int n_threads);
void qwen35_ocr_free(qwen35_ocr_context * ctx);
// Invalid settings reject without changing the current setting.
int qwen35_ocr_set_max_tokens(qwen35_ocr_context * ctx, int max_tokens);
int qwen35_ocr_get_max_tokens(const qwen35_ocr_context * ctx);
// RGB, grayscale or RGBA, contiguous uint8. Alpha is discarded (PIL RGB policy).
// Null prompt means empty. Returned buffers live until next recognition/free.
// Failure returns null, clears prior outputs and sets out_len=0. EOS-only is "".
const char * qwen35_ocr_recognize_raw(qwen35_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                      int channels, const char * prompt, int * out_len);
const char * qwen35_ocr_recognize_file(qwen35_ocr_context * ctx, const char * image_path, const char * prompt,
                                       int * out_len);
// Greedy selected-token probabilities, including EOS; IDs include EOS too.
const int * qwen35_ocr_token_ids(const qwen35_ocr_context * ctx, int * count);
const float * qwen35_ocr_confidences(const qwen35_ocr_context * ctx, int * count);
float qwen35_ocr_mean_confidence(const qwen35_ocr_context * ctx);
qwen35_ocr_stop qwen35_ocr_last_stop(const qwen35_ocr_context * ctx);
const char * qwen35_ocr_last_error(const qwen35_ocr_context * ctx);
#ifdef __cplusplus
}
#endif
