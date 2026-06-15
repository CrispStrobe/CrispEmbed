// classical_preproc.h — Classical document preprocessing algorithms.
//
// Cherry-picked from Leptonica (BSD-2), reimplemented as self-contained
// C++ with no dependencies beyond morph_fast.h. Provides the CPU-only,
// model-free, fast tier for every scan_cleanup step.
//
// Algorithms:
//   1. Adaptive Otsu binarization (tiled + smoothed)
//   2. Differential-square-sum deskew (pixFindSkew equivalent)
//   3. CC-based despeckle (remove small noise components)
//   4. Background normalization (gradient/shadow-robust)

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 1. Adaptive Otsu binarization
// ---------------------------------------------------------------------------

/// Binarize with per-tile Otsu thresholds, smoothed to avoid tile
/// boundary artifacts. Handles uneven lighting (gradients, shadows).
///
/// [gray]    — row-major uint8 grayscale.
/// [tile_w]  — tile width for local Otsu (0 = auto, typically 32-64).
/// [tile_h]  — tile height for local Otsu (0 = auto).
/// [smooth]  — smoothing kernel for tile thresholds (0 = auto).
/// [out]     — output: binarized uint8 (0 or 255). Caller allocates w*h.
void adaptive_otsu(const uint8_t * gray, int w, int h,
                   int tile_w, int tile_h, int smooth,
                   uint8_t * out);

// ---------------------------------------------------------------------------
// 2. Differential-square-sum deskew
// ---------------------------------------------------------------------------

/// Find the skew angle of a document image using Leptonica's differential
/// square-sum scoring. More robust than Hough-on-Sobel for sparse text.
///
/// [gray]      — row-major uint8 grayscale.
/// [out_angle] — receives the deskew angle in degrees (rotate by this to fix).
/// [out_conf]  — receives confidence (ratio of best to worst score; >3 = good).
/// Returns 0 on success, 1 if angle could not be determined.
int find_skew_angle(const uint8_t * gray, int w, int h,
                    float * out_angle, float * out_conf);

// ---------------------------------------------------------------------------
// 3. CC-based despeckle
// ---------------------------------------------------------------------------

/// Remove small connected components (noise speckles) from a binary image.
/// Components with both width < max_w AND height < max_h are removed.
///
/// [bits]   — 1-bit packed image (from morph_fast.h).
/// [w, h]   — image dimensions.
/// [wpl]    — words per line.
/// [max_w]  — maximum speckle width (default 5).
/// [max_h]  — maximum speckle height (default 5).
/// Returns new allocated 1-bit image with speckles removed.
uint32_t * despeckle_cc(const uint32_t * bits, int w, int h, int wpl,
                         int max_w, int max_h);

/// Convenience: despeckle a uint8 grayscale image.
/// Binarizes internally, despeckles, returns cleaned uint8 (0 or 255).
void despeckle_gray(const uint8_t * gray, int w, int h,
                    int max_w, int max_h, uint8_t * out);

// ---------------------------------------------------------------------------
// 4. Background normalization
// ---------------------------------------------------------------------------

/// Estimate and normalize the background of a grayscale document image.
/// Handles gradients and shadows better than simple morphological open.
/// Uses tile-based sampling of background intensity + smooth interpolation.
///
/// [gray]   — row-major uint8 grayscale.
/// [tile_w] — sampling tile width (0 = auto, ~32-64 px).
/// [tile_h] — sampling tile height (0 = auto).
/// [out]    — output: normalized uint8 grayscale. Caller allocates w*h.
void background_norm(const uint8_t * gray, int w, int h,
                     int tile_w, int tile_h, uint8_t * out);

// ---------------------------------------------------------------------------
// 5. Image downsampling calculator
// ---------------------------------------------------------------------------

/// Compute the optimal scale factor for downsampling before OCR.
/// Returns a factor in (0, 1] — multiply image dimensions by this factor.
/// target_dpi: desired OCR resolution (300 is typical for printed text).
/// current_dpi: source image DPI (0 = estimate from dimensions).
/// max_pixels: maximum total pixels after downsampling (0 = no limit).
float compute_downsample_factor(int w, int h, int current_dpi,
                                 int target_dpi, int max_pixels);

// ---------------------------------------------------------------------------
// 6. OCR quality scoring
// ---------------------------------------------------------------------------

/// Score OCR output quality by matching words against a dictionary.
/// Returns fraction of words found in the dictionary [0, 1].
/// [text] — OCR output text (UTF-8, space-separated words).
/// [dict] — array of dictionary words (must be sorted).
/// [n_dict] — number of words in dictionary.
float ocr_quality_score(const char * text,
                         const char ** dict, int n_dict);

// ---------------------------------------------------------------------------
// 7. Text angle classification (0° vs 180°)
// ---------------------------------------------------------------------------

/// Detect if text in an image is upside-down (rotated 180°).
/// Uses ascender/descender asymmetry + differential scoring at 0° and 180°.
///
/// Returns: 0 if correctly oriented, 180 if upside-down.
/// [confidence] receives a score in [0,1] — higher = more certain.
int detect_text_angle(const uint8_t * gray, int w, int h,
                       float * confidence);

// ---------------------------------------------------------------------------
// 8. TPS spatial transformer (learned dewarping)
// ---------------------------------------------------------------------------

/// Dewarp a grayscale page image using Thin-Plate Spline control points.
///
/// Given N source/target control point pairs, solves the TPS interpolation
/// system and applies the inverse warp with bilinear sampling.
///
/// [gray]    — row-major uint8 grayscale input (w * h).
/// [src_x/y] — source control point coordinates (length n).
/// [dst_x/y] — target (straightened) control point coordinates (length n).
/// [n]       — number of control points (>= 3).
/// [out]     — output: dewarped uint8 grayscale. Caller allocates w * h.
///
/// Returns 0 on success, 1 on failure (too few points, singular system).
int tps_dewarp(const uint8_t * gray, int w, int h,
               const float * src_x, const float * src_y,
               const float * dst_x, const float * dst_y, int n,
               uint8_t * out);

#ifdef __cplusplus
}
#endif
