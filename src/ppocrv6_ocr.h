#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocrv6_ocr_context ppocrv6_ocr_context;

ppocrv6_ocr_context * ppocrv6_ocr_init(const char * model_path, int n_threads);
void ppocrv6_ocr_free(ppocrv6_ocr_context * ctx);
// Override the opt-in graph acceptance for a routed page without mutating the
// process environment. Use -1 to restore environment-controlled behavior.
void ppocrv6_ocr_set_graph_accept(ppocrv6_ocr_context * ctx, int accept);
const char * ppocrv6_ocr_recognize_raw(ppocrv6_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len);
// Recognize a group of line crops while retaining caller order. Crops are
// internally visited by dynamic model width so width-keyed graph state can be
// reused; output strings are copied into caller-owned buffers.
int ppocrv6_ocr_recognize_raw_batch(ppocrv6_ocr_context * ctx, const uint8_t * const * pixels, const int * widths,
                                    const int * heights, const int * channels, int count, char * const * outputs,
                                    const int * capacities, int * lengths);
const char * ppocrv6_ocr_recognize(ppocrv6_ocr_context * ctx, const float * pixels, int width, int height,
                                   int * out_len);

#ifdef __cplusplus
}
#endif
