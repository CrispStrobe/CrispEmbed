// cc_detect.h — Classical text line detection via connected components.
//
// Model-free, GPU-free text line detector. Cherry-picked from Leptonica's
// page segmentation algorithms (BSD-2). Zero model downloads required.
//
// Pipeline:
//   1. Binarize (Otsu or supplied 1-bit image)
//   2. Horizontal morphological close → merge chars into lines
//   3. Vertical whitespace subtraction → split columns
//   4. Small morphological open → remove noise
//   5. Connected components → bounding boxes
//
// Use as the "light tier" detector when DBNet/Surya are unavailable.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// A detected text region (bounding box).
typedef struct cc_text_region {
    int x, y, w, h;
} cc_text_region;

/// Detect text line regions in a grayscale image using connected components.
///
/// [gray]   — row-major uint8 grayscale pixels.
/// [width]  — image width.
/// [height] — image height.
/// [out_n]  — receives the number of detected regions.
///
/// Returns an array of cc_text_region, caller must free with cc_detect_free().
/// Regions are sorted top-to-bottom, left-to-right.
cc_text_region * cc_detect_lines(
    const uint8_t * gray, int width, int height,
    int * out_n);

/// Parameters for tuning the detection.
typedef struct cc_detect_params {
    int close_hsize;     // horizontal close SE width (default 30)
    int close_vsize;     // horizontal close SE height (default 1)
    int open_size;       // noise removal open kernel (default 3)
    int min_width;       // minimum region width (default 10)
    int min_height;      // minimum region height (default 5)
    uint8_t binarize_threshold; // 0 = auto (Otsu) (default 0)
} cc_detect_params;

/// Get default parameters.
cc_detect_params cc_detect_defaults(void);

/// Detect with custom parameters.
cc_text_region * cc_detect_lines_params(
    const uint8_t * gray, int width, int height,
    cc_detect_params params, int * out_n);

/// Free a region array returned by cc_detect_lines.
void cc_detect_free(cc_text_region * regions);

#ifdef __cplusplus
}
#endif
