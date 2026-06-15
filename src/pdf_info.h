// pdf_info.h — PDF page DPI profiling.
//
// Minimal PDF parser that extracts image metadata from embedded images
// to compute effective page DPI. Used to auto-select OCR resolution:
// downsample high-DPI scans (saves compute) or upscale low-DPI images
// (improves recognition quality).
//
// Parses just enough PDF structure to find:
//   - Page /MediaBox (page dimensions in points)
//   - Image XObject /Width, /Height (pixel dimensions)
//   - Content stream CTM (cm operator → image display size)
//
// Zero external dependencies — self-contained PDF binary parsing.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Per-image metadata extracted from a PDF page.
typedef struct pdf_image_info {
    int pixel_w;        // image width in pixels
    int pixel_h;        // image height in pixels
    float display_w;    // display width in points (from CTM)
    float display_h;    // display height in points (from CTM)
    float dpi_x;        // effective DPI horizontal
    float dpi_y;        // effective DPI vertical
    int bits_per_comp;  // /BitsPerComponent (1, 2, 4, 8)
    int obj_id;         // PDF object ID of this image
} pdf_image_info;

/// Per-page DPI result.
typedef struct pdf_page_dpi_result {
    float dpi;              // weighted harmonic mean DPI across all images
    float dpi_min;          // minimum image DPI on this page
    float dpi_max;          // maximum image DPI on this page
    int n_images;           // number of images found
    float page_width_pt;    // page width in points
    float page_height_pt;   // page height in points
} pdf_page_dpi_result;

/// Compute effective DPI for a page in a PDF file.
///
/// [pdf_path] — path to the PDF file.
/// [page]     — 0-based page index.
/// [result]   — receives the DPI information.
///
/// Returns 0 on success, 1 on failure (bad PDF, page out of range,
/// no images found).
int pdf_page_dpi(const char * pdf_path, int page,
                 pdf_page_dpi_result * result);

/// Compute effective DPI for all pages. Returns array of results.
/// [n_pages] receives the page count.
/// Caller must free the returned array with pdf_dpi_free().
pdf_page_dpi_result * pdf_all_pages_dpi(const char * pdf_path,
                                         int * n_pages);

/// Free array returned by pdf_all_pages_dpi().
void pdf_dpi_free(pdf_page_dpi_result * results);

#ifdef __cplusplus
}
#endif
