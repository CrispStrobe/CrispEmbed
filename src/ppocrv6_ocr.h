#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocrv6_ocr_context ppocrv6_ocr_context;

ppocrv6_ocr_context * ppocrv6_ocr_init(const char * model_path, int n_threads);
void ppocrv6_ocr_free(ppocrv6_ocr_context * ctx);
const char * ppocrv6_ocr_recognize_raw(ppocrv6_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len);
const char * ppocrv6_ocr_recognize(ppocrv6_ocr_context * ctx, const float * pixels, int width, int height,
                                   int * out_len);

#ifdef __cplusplus
}
#endif
