#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct easyocr_ocr_context easyocr_ocr_context;

typedef struct easyocr_ocr_timing {
    double preprocess_ms;
    double graph_ms;
    double decode_ms;
    double total_ms;
} easyocr_ocr_timing;

easyocr_ocr_context * easyocr_ocr_init(const char * model_path, int n_threads);
void easyocr_ocr_free(easyocr_ocr_context * ctx);
bool easyocr_ocr_set_width(easyocr_ocr_context * ctx, int width);
float easyocr_ocr_last_confidence(const easyocr_ocr_context * ctx);
bool easyocr_ocr_last_timing(const easyocr_ocr_context * ctx, easyocr_ocr_timing * timing);
const char * easyocr_ocr_recognize(easyocr_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                   int channels, int * out_len);
int easyocr_ocr_diff(easyocr_ocr_context * ctx, const char * ref_path);

#ifdef __cplusplus
}
#endif
