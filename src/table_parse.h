// table_parse.h — Rule-based table structure recognition.
//
// Extracts HTML <table> from a cropped table image using morphological
// line detection + grid intersection analysis + per-cell OCR.
//
// Algorithm:
//   1. Binarize (Otsu)
//   2. Detect horizontal/vertical ruling lines (morph open with long SEs)
//   3. Build row/column grid from line projections
//   4. For borderless tables, fall back to projection-based splitting
//   5. OCR each cell (via tesseract_lstm or callback)
//   6. Output HTML <table>

#ifndef TABLE_PARSE_H
#define TABLE_PARSE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct table_parse_context table_parse_context;

// Cell recognizer callback: called for each cell crop.
// (gray, w, h) is the grayscale cell image.
// Must return a null-terminated UTF-8 string (caller does NOT free it;
// the string must remain valid until the next call).
typedef const char * (*table_cell_ocr_fn)(
    void * user_data,
    const uint8_t * gray, int w, int h);

// Initialize table parser.
// ocr_model_path: Tesseract LSTM GGUF for built-in cell OCR (NULL = use callback).
// n_threads: CPU threads for OCR.
table_parse_context * table_parse_init(const char * ocr_model_path, int n_threads);
void table_parse_free(table_parse_context * ctx);

// Set an external OCR callback instead of the built-in Tesseract.
void table_parse_set_ocr(table_parse_context * ctx,
                         table_cell_ocr_fn fn, void * user_data);

// Parse a table image into HTML.
// Input: grayscale uint8 [height, width].
// Returns: allocated HTML string (caller frees with table_parse_free_string).
// Returns NULL on failure.
char * table_parse_to_html(table_parse_context * ctx,
                           const uint8_t * gray, int width, int height);

void table_parse_free_string(char * str);

// Intermediate: detect grid structure without OCR.
// Returns number of rows and columns found.
int table_parse_detect_grid(const uint8_t * gray, int width, int height,
                            int * out_n_rows, int * out_n_cols);

#ifdef __cplusplus
}
#endif

#endif // TABLE_PARSE_H
