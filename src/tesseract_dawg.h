#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tesseract_dawg_context tesseract_dawg_context;

// Parse and retain one validated DAWG payload for repeated diagnostic lookups.
tesseract_dawg_context * tesseract_dawg_init_base64(const char * payload, char * error, size_t error_size);
void tesseract_dawg_free(tesseract_dawg_context * ctx);
int tesseract_dawg_context_contains(const tesseract_dawg_context * ctx, const int * unichar_ids, size_t count);
int tesseract_dawg_context_has_prefix(const tesseract_dawg_context * ctx, const int * unichar_ids, size_t count);

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
