// morph_fast.h — Fast morphological operations on 1-bit packed images.
//
// Cherry-picked algorithms from Leptonica (BSD-2), reimplemented as
// self-contained C++ with no external dependencies. Operates on packed
// uint32 word arrays (32 pixels per word) for 5-6x speedup over float
// separable morphology, with 8x less memory.
//
// Supported operations:
//   morph_erode_brick()   — binary erosion with rectangular SE
//   morph_dilate_brick()  — binary dilation with rectangular SE
//   morph_open_brick()    — erosion then dilation (background estimate)
//   morph_close_brick()   — dilation then erosion
//
// The 1-bit representation:
//   - Row-major, each row padded to a multiple of 32 bits.
//   - Bit 0 of word 0 = leftmost pixel. MSB-first within each word.
//   - wpl = (width + 31) / 32  (words per line).
//   - Total buffer size = wpl * height * 4 bytes.

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Convert a float grayscale image [0,1] to 1-bit packed representation.
/// Pixels < threshold → 1 (foreground), >= threshold → 0 (background).
/// Caller must free the returned buffer with morph_free().
uint32_t * morph_float_to_1bit(const float * gray, int w, int h,
                                float threshold, int * out_wpl);

/// Convert 1-bit packed image back to float [0,1].
/// Foreground (1) → 0.0, background (0) → 1.0.
void morph_1bit_to_float(const uint32_t * bits, int w, int h, int wpl,
                          float * out_gray);

/// Convert uint8 grayscale to 1-bit. Pixels < threshold → 1.
uint32_t * morph_u8_to_1bit(const uint8_t * gray, int w, int h,
                             uint8_t threshold, int * out_wpl);

/// Binary erosion with brick (rectangular) structuring element.
/// hsize, vsize: SE dimensions (must be odd). Separable: horiz then vert.
/// Returns new allocated image. Caller frees with morph_free().
uint32_t * morph_erode_brick(const uint32_t * src, int w, int h, int wpl,
                              int hsize, int vsize);

/// Binary dilation with brick SE. Returns new allocated image.
uint32_t * morph_dilate_brick(const uint32_t * src, int w, int h, int wpl,
                               int hsize, int vsize);

/// Morphological opening (erode then dilate). Returns new allocated image.
uint32_t * morph_open_brick(const uint32_t * src, int w, int h, int wpl,
                             int hsize, int vsize);

/// Morphological closing (dilate then erode). Returns new allocated image.
uint32_t * morph_close_brick(const uint32_t * src, int w, int h, int wpl,
                              int hsize, int vsize);

/// Free a 1-bit image buffer returned by the above functions.
void morph_free(uint32_t * bits);

/// High-level: background whitening on float [0,1] grayscale.
/// Uses fast 1-bit morphological open internally. Equivalent to
/// scan_cleanup_whiten() but 5-6x faster for large kernels.
void morph_whiten_fast(const float * gray, int w, int h,
                        int kernel_size, float * dst);

#ifdef __cplusplus
}
#endif
