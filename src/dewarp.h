// dewarp.h — Page dewarping for camera-captured and book-scanned documents.
//
// Cherry-picked from Leptonica's dewarp module (BSD-2, Dan Bloomberg).
// Reimplemented as self-contained C++ with no Leptonica dependency.
//
// Pipeline:
//   1. Find textline baselines (morph close → CC → midpoint trace)
//   2. Fit cubic splines to baselines → vertical disparity model
//   3. Build 2D disparity map (interpolate between baselines)
//   4. Apply bilinear warp to straighten the page
//
// Handles vertical curvature (book spine warping, phone captures).
// Does NOT handle horizontal perspective (use affine transform for that).

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Dewarp a grayscale page image.
///
/// [gray]     — row-major uint8 grayscale pixels.
/// [w, h]     — image dimensions.
/// [out]      — output: dewarped uint8 grayscale. Caller allocates w*h.
/// [out_w]    — receives output width (may differ from input).
/// [out_h]    — receives output height.
///
/// Returns 0 on success, 1 if dewarping could not be applied (too few
/// textlines found, image too small, etc.). When it fails, out is
/// a copy of the input.
int dewarp_page(const uint8_t * gray, int w, int h,
                uint8_t * out, int * out_w, int * out_h);

/// Dewarp parameters for tuning.
typedef struct dewarp_params {
    int min_lines;       // minimum textlines to build model (default 4)
    int sampling;        // horizontal sampling step for baselines (default 8)
    float max_curve;     // maximum allowed curvature in pixels (default 100)
} dewarp_params;

dewarp_params dewarp_defaults(void);

int dewarp_page_params(const uint8_t * gray, int w, int h,
                       dewarp_params params,
                       uint8_t * out, int * out_w, int * out_h);

#ifdef __cplusplus
}
#endif
