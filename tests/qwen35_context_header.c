#include "qwen35_ocr.h"
// Compile the engine header as C and exercise null-context ABI contracts.
int qwen35_context_c_header_smoke(void) {
    int n = 99;
    const int * ids = qwen35_ocr_token_ids(0, &n);
    return !ids && n == 0 && qwen35_ocr_get_max_tokens(0) == 0 && qwen35_ocr_last_stop(0) == QWEN35_OCR_ERROR &&
           qwen35_ocr_mean_confidence(0) == 0;
}
