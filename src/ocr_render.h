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

#include <stddef.h>
#include <stdint.h>

// DLL export/import (Windows) — without this the ocr_render_* symbols are not
// exported from crispembed.dll, so consumers (e.g. CrispSorter) fail to link on
// Windows with LNK2019 even though macOS/Linux resolve them via default symbol
// visibility. Guarded so it composes with crispembed.h's identical definition.
#ifndef CRISPEMBED_API
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(CRISPEMBED_BUILD)
#define CRISPEMBED_API __declspec(dllexport)
#elif defined(CRISPEMBED_SHARED)
#define CRISPEMBED_API __declspec(dllimport)
#else
#define CRISPEMBED_API
#endif
#else
#if defined(CRISPEMBED_BUILD)
#define CRISPEMBED_API __attribute__((visibility("default")))
#else
#define CRISPEMBED_API
#endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// OCR word/region result for rendering.
typedef struct ocr_render_word {
    const char * text; // UTF-8 text
    int x, y, w, h;    // bounding box in page coordinates
    float confidence;  // [0, 1]; negative means unknown/omit optional XML score
} ocr_render_word;

/// OCR line: a group of words on the same baseline.
typedef struct ocr_render_line {
    const ocr_render_word * words;
    int n_words;
    int x, y, w, h; // line bounding box
} ocr_render_line;

/// OCR page result.
typedef struct ocr_render_page {
    const ocr_render_line * lines;
    int n_lines;
    int page_width, page_height; // image dimensions
    const char * image_path;     // original image (for PDF embedding)
} ocr_render_page;

/// Output format.
typedef enum ocr_render_format {
    OCR_RENDER_TEXT = 0, // plain text with page separators
    OCR_RENDER_HOCR = 1, // hOCR XHTML
    OCR_RENDER_ALTO = 2, // ALTO 3.1 XML
    OCR_RENDER_PDF = 3,  // searchable PDF (image + invisible text)
} ocr_render_format;

typedef struct ocr_renderer ocr_renderer;

/// Feature bits returned by crispembed_ocr_render_capabilities().
typedef enum ocr_render_capability {
    OCR_RENDER_CAP_PDF = 1,
    OCR_RENDER_CAP_UNICODE = 2,
    OCR_RENDER_CAP_RASTER_IMAGES = 4,
    OCR_RENDER_CAP_ARCHIVAL = 8,
    OCR_RENDER_CAP_BIDI = 16,
    OCR_RENDER_CAP_STREAM = 32,
} ocr_render_capability;

/// No model load required. ARCHIVAL declares PDF/A-2b. BIDI is withheld while
/// the explicit Unicode 17 layout option has unresolved reader failures.
CRISPEMBED_API uint32_t crispembed_ocr_render_capabilities(void);

/// Create a renderer for the given format.
CRISPEMBED_API ocr_renderer * ocr_render_create(ocr_render_format format);

/// Set the page separator for plain text output (default: "\f").
CRISPEMBED_API void ocr_render_set_separator(ocr_renderer * r, const char * sep);

/// Request PDF/A-2b: embedded sRGB output intent, fonts and XMP identification.
/// Use the binary-safe output-size API for archival documents, including those
/// without page images. This does not claim PDF/UA or recovered typography.
/// Must be called before ocr_render_begin(). Only affects PDF format.
CRISPEMBED_API void ocr_render_set_pdfa(ocr_renderer * r, int enabled);

/// Experimental Unicode 17 layout for the invisible PDF text layer. It remains
/// available for qualification, but BIDI is not advertised to applications yet.
/// Text supplied by callers must be in logical reading order. Visual positions
/// are estimated inside each OCR region; page images are never repainted.
/// The legacy default rejects RTL scripts/directional controls. Set before begin.
CRISPEMBED_API void ocr_render_set_pdf_bidi(ocr_renderer * r, int enabled);

/// Begin a document.
CRISPEMBED_API void ocr_render_begin(ocr_renderer * r);

/// Lower the per-document PDF ceilings after begin(), before the first page.
/// Defaults/hard maxima: 5000 pages and 128 MiB output. Additional fixed limits
/// are 16 MiB input text/geometry, 1 MiB per region and 100000 bidi fragments.
/// Values must be 1..5000 pages and 4096..134217728 bytes. begin() resets limits.
/// This bounds an in-memory export; large source images still require separate
/// decode/compression working memory. Any breach returns an error, never a
/// truncated document. Export a smaller page selection or downsample images.
CRISPEMBED_API void ocr_render_set_pdf_limits(ocr_renderer * r, int max_pages, int max_bytes);

/// Synchronous PDF byte sink. Return zero after consuming the complete slice,
/// nonzero on failure. The data is valid only during this call; no exception
/// may cross the C boundary. Do not re-enter this renderer from the callback.
typedef int (*ocr_render_write_callback)(void * context, const unsigned char * data, size_t size);

/// Stream the PDF to a caller-owned sink instead of accumulating output bytes.
/// Set after begin(), before any page. The callback runs only during end(), in
/// slices of at most 64 KiB, and is never called again after an error. The
/// context must remain valid until end() returns. begin() clears the sink.
/// max_output_bytes must be 4096..2147483648; callers may choose 256 MiB for a
/// mobile destination. Page/text/image/layout limits remain in force.
/// Any failure can leave a partial sink: write to a temporary destination and
/// publish only after end() succeeds. output()/output_size() stay empty in
/// this mode. The existing byte-returning API retains its 128 MiB ceiling.
CRISPEMBED_API void ocr_render_set_pdf_writer(ocr_renderer * r, ocr_render_write_callback callback, void * context,
                                              uint64_t max_output_bytes);

/// Physical PDF dimensions for the NEXT added page (72 points = 1 inch).
/// Set after begin(), before add_page(). Image and text coordinates scale
/// together from pixels into this size. The setting resets after add_page().
/// Without this call the legacy convention is one pixel per point.
CRISPEMBED_API void ocr_render_set_page_size_points(ocr_renderer * r, double width, double height);

/// Add a page of OCR results.
CRISPEMBED_API void ocr_render_add_page(ocr_renderer * r, const ocr_render_page * page);

/// End the document and finalize output.
CRISPEMBED_API void ocr_render_end(ocr_renderer * r);

/// Null on success; otherwise an owned error message, valid until begin/free.
/// Any failure clears output. No partial PDF or XML should be published.
CRISPEMBED_API const char * ocr_render_error(const ocr_renderer * r);

/// Get the rendered output as a string. Valid until free.
/// For PDF format, returns the raw PDF bytes (use ocr_render_output_size
/// for the byte count since PDFs contain binary data).
CRISPEMBED_API const char * ocr_render_output(const ocr_renderer * r);

/// Get the output size in bytes.
CRISPEMBED_API int ocr_render_output_size(const ocr_renderer * r);

/// Free the renderer and all associated memory.
CRISPEMBED_API void ocr_render_free(ocr_renderer * r);

#ifdef __cplusplus
}
#endif
