// tps_warp.h — Thin-Plate Spline spatial transformer for learned dewarping.
//
// Two-stage API:
//   1. tps_solve()  — given source + target control points, compute TPS
//                     warp coefficients (affine + radial basis weights).
//   2. tps_warp()   — apply a solved TPS transform to a grayscale image
//                     via bilinear sampling.
//
// The TPS math is model-free (pure linear algebra). A localization network
// that predicts control point offsets can be layered on top later.
//
// Reference: F. Bookstein, "Principal Warps: Thin-Plate Splines and the
// Decomposition of Deformations", IEEE TPAMI 11(6), 1989.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque TPS coefficients (affine + radial basis weights for x and y).
/// Allocated by tps_solve(), freed by tps_free().
typedef struct tps_model tps_model;

/// Solve the TPS system for a set of corresponding control points.
///
/// [src_x, src_y] — source control point coordinates (length n).
/// [dst_x, dst_y] — target control point coordinates (length n).
/// [n]            — number of control points (>= 3).
///
/// Returns a tps_model on success, NULL on failure (singular system,
/// fewer than 3 points, etc.).
tps_model * tps_solve(const float * src_x, const float * src_y,
                      const float * dst_x, const float * dst_y,
                      int n);

/// Map a single point through the TPS transform.
///
/// Given a point (x, y) in the source coordinate system, compute the
/// corresponding point (out_x, out_y) in the target coordinate system.
void tps_map_point(const tps_model * model,
                   float x, float y,
                   float * out_x, float * out_y);

/// Warp a grayscale image using a solved TPS model.
///
/// The transform maps from OUTPUT coordinates back to INPUT coordinates
/// (inverse warp), so each output pixel samples the input at the TPS-
/// mapped location via bilinear interpolation.
///
/// [src]       — row-major uint8 grayscale input, src_w * src_h pixels.
/// [src_w/h]   — input image dimensions.
/// [model]     — TPS coefficients from tps_solve().
/// [dst]       — output: warped uint8 grayscale. Caller allocates dst_w * dst_h.
/// [dst_w/h]   — output image dimensions (may differ from input).
/// [bg]        — background value for out-of-bounds samples (typically 255).
void tps_warp(const uint8_t * src, int src_w, int src_h,
              const tps_model * model,
              uint8_t * dst, int dst_w, int dst_h,
              uint8_t bg);

/// Warp a grayscale image given raw control point arrays (convenience).
/// Internally calls tps_solve() + tps_warp() + tps_free().
/// Returns 0 on success, 1 on failure.
int tps_warp_points(const uint8_t * src, int src_w, int src_h,
                    const float * src_x, const float * src_y,
                    const float * dst_x, const float * dst_y,
                    int n_points,
                    uint8_t * dst, int dst_w, int dst_h,
                    uint8_t bg);

/// Free a TPS model allocated by tps_solve().
void tps_free(tps_model * model);

// =========================================================================
// Learned TPS localization network
// =========================================================================
// A small CNN that predicts control point displacements from an input image.
// Loaded from a GGUF model file (see models/convert-tps-loc-to-gguf.py).
//
// Architecture (PaddleOCR "small" variant, ~400K params):
//   Conv0: 3→16, 3x3, pad1, BN-folded + ReLU + MaxPool2x2
//   Conv1: 16→32, 3x3, pad1, BN-folded + ReLU + MaxPool2x2
//   Conv2: 32→64, 3x3, pad1, BN-folded + ReLU + MaxPool2x2
//   Conv3: 64→128, 3x3, pad1, BN-folded + ReLU + AdaptiveAvgPool(1)
//   FC1: 128→64 + ReLU
//   FC2: 64→num_fiducial*2
//
// Output: N control point coordinates in [-1, 1] normalized space.

/// Opaque localization network context.
typedef struct tps_locnet tps_locnet;

/// Load a TPS localization network from a GGUF file.
/// Returns NULL on failure.
tps_locnet * tps_locnet_load(const char * gguf_path);

/// Predict control point coordinates from a grayscale image.
///
/// [gray]   — row-major uint8 grayscale (w * h).
/// [w, h]   — image dimensions.
/// [out_x]  — receives predicted x coordinates (caller allocates num_fiducial).
/// [out_y]  — receives predicted y coordinates (caller allocates num_fiducial).
///
/// Coordinates are in image pixel space (not normalized).
/// Returns the number of fiducial points, or 0 on failure.
int tps_locnet_predict(tps_locnet * net,
                       const uint8_t * gray, int w, int h,
                       float * out_x, float * out_y);

/// Get the number of fiducial points this model predicts.
int tps_locnet_num_fiducial(const tps_locnet * net);

/// Free a localization network.
void tps_locnet_free(tps_locnet * net);

/// Full pipeline: load localization model, predict control points, TPS warp.
///
/// Predicts source (warped) control points from the input image and maps
/// them to a regular grid of target (straight) control points, then applies
/// the TPS warp.
///
/// [gray]       — input grayscale image (w * h).
/// [gguf_path]  — path to localization model GGUF.
/// [out]        — output: dewarped uint8 grayscale. Caller allocates w * h.
///
/// Returns 0 on success, 1 on failure.
int tps_auto_dewarp(const uint8_t * gray, int w, int h,
                    const char * gguf_path,
                    uint8_t * out);

#ifdef __cplusplus
}
#endif
