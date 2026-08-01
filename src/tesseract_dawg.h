#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Validate one Tesseract SquishedDawg payload serialized as base64 metadata.
// This checks the wire header, bounds, and edge-run termination only; it does
// not apply dictionary scores to OCR hypotheses.
int tesseract_dawg_validate_base64(const char * payload, char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
