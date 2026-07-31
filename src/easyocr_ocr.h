#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct easyocr_ocr_context easyocr_ocr_context;

easyocr_ocr_context * easyocr_ocr_init(const char * model_path, int n_threads);
void easyocr_ocr_free(easyocr_ocr_context * ctx);
const char * easyocr_ocr_recognize(easyocr_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                   int channels, int * out_len);
int easyocr_ocr_diff(easyocr_ocr_context * ctx, const char * ref_path);

#ifdef __cplusplus
}
#endif
