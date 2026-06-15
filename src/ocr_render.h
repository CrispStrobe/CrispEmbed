// ocr_render.h — OCR result renderers (plain text, hOCR, ALTO, searchable PDF).
//
// Accumulates OCR results page-by-page, then renders to the chosen format.
// Inspired by Tesseract's result renderer pattern: begin → add_page → end.
//
// Plain text: concatenates pages with configurable separator (default \f).
// hOCR: XHTML with ocr_page/ocr_line/ocrx_word elements + bounding boxes.
// ALTO: XML following the ALTO 3.1 schema (text blocks, lines, strings).
// Searchable PDF: original image with invisible text layer overlay.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// OCR word/region result for rendering.
typedef struct ocr_render_word {
    const char * text;       // UTF-8 text
    int x, y, w, h;         // bounding box in page coordinates
    float confidence;        // [0, 1]
} ocr_render_word;

/// OCR line: a group of words on the same baseline.
typedef struct ocr_render_line {
    const ocr_render_word * words;
    int n_words;
    int x, y, w, h;         // line bounding box
} ocr_render_line;

/// OCR page result.
typedef struct ocr_render_page {
    const ocr_render_line * lines;
    int n_lines;
    int page_width, page_height;  // image dimensions
    const char * image_path;       // original image (for PDF embedding)
} ocr_render_page;

/// Output format.
typedef enum ocr_render_format {
    OCR_RENDER_TEXT = 0,    // plain text with page separators
    OCR_RENDER_HOCR = 1,   // hOCR XHTML
    OCR_RENDER_ALTO = 2,   // ALTO 3.1 XML
    OCR_RENDER_PDF  = 3,   // searchable PDF (image + invisible text)
} ocr_render_format;

typedef struct ocr_renderer ocr_renderer;

/// Create a renderer for the given format.
ocr_renderer * ocr_render_create(ocr_render_format format);

/// Set the page separator for plain text output (default: "\f").
void ocr_render_set_separator(ocr_renderer * r, const char * sep);

/// Begin a document.
void ocr_render_begin(ocr_renderer * r);

/// Add a page of OCR results.
void ocr_render_add_page(ocr_renderer * r, const ocr_render_page * page);

/// End the document and finalize output.
void ocr_render_end(ocr_renderer * r);

/// Get the rendered output as a string. Valid until free.
/// For PDF format, returns the raw PDF bytes (use ocr_render_output_size
/// for the byte count since PDFs contain binary data).
const char * ocr_render_output(const ocr_renderer * r);

/// Get the output size in bytes.
int ocr_render_output_size(const ocr_renderer * r);

/// Free the renderer and all associated memory.
void ocr_render_free(ocr_renderer * r);

#ifdef __cplusplus
}
#endif
