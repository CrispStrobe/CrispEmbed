#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Validate one Tesseract SquishedDawg payload serialized as base64 metadata.
// This checks the wire header, bounds, and edge-run termination only; it does
// not apply dictionary scores to OCR hypotheses.
int tesseract_dawg_validate_base64(const char * payload, char * error, size_t error_size);

// Diagnostic exact-word lookup over unichar IDs. Returns 1 only when the
// serialized DAWG contains the complete sequence; no score is produced.
int tesseract_dawg_contains_base64(const char * payload, const int * unichar_ids, size_t count);

// Diagnostic prefix legality lookup. Returns 1 when the sequence reaches a
// DAWG node, regardless of whether that node is a complete word.
int tesseract_dawg_has_prefix_base64(const char * payload, const int * unichar_ids, size_t count);

#ifdef __cplusplus
}
#endif
