// scan_cleanup.h — document scan preprocessing (deskew, binarize, denoise, crop)
//
// Two tiers:
//   Tier 1 (classical, no model): deskew, Otsu/Sauvola binarization,
//     border crop, background whitening via morphological open.
//   Tier 2 (learned, GGUF model): small denoising CNN for complex
//     degradation — not yet implemented, pass model_path=NULL.
//
// Usage:
//   scan_cleanup_ctx * ctx = scan_cleanup_init(NULL, 4);  // tier 1 only
//   scan_cleanup_params p = scan_cleanup_defaults();
//   uint8_t * out = NULL;
//   int ow, oh;
//   scan_cleanup_process(ctx, pixels, w, h, channels, p, &out, &ow, &oh);
//   // out is RGB uint8, caller must free with scan_cleanup_free_image(out)
//   scan_cleanup_free(ctx);

#ifndef SCAN_CLEANUP_H
#define SCAN_CLEANUP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scan_cleanup_ctx scan_cleanup_ctx;

typedef struct {
    int   deskew;              // 1 = detect and correct skew (default: 1)
    int   crop_borders;        // 1 = remove dark scanner borders (default: 1)
    int   whiten_background;   // 1 = flatten uneven lighting (default: 1)
    int   binarize;            // 1 = adaptive binarization (default: 0)
    int   binarize_method;     // 0 = Otsu (global), 1 = Sauvola (adaptive)
    float sauvola_k;           // Sauvola sensitivity, default 0.2
    int   sauvola_window;      // Sauvola window size (odd), default 25
    int   morph_kernel;        // background whitening kernel size, default 51
    float border_threshold;    // border darkness threshold 0..1, default 0.15
    float deskew_max_angle;    // max correction angle in degrees, default 15.0
} scan_cleanup_params;

// Returns default params (deskew + crop + whiten enabled, binarize disabled)
scan_cleanup_params scan_cleanup_defaults(void);

// Initialize. model_path=NULL for tier-1-only (no learned model).
scan_cleanup_ctx * scan_cleanup_init(const char * model_path, int n_threads);
void scan_cleanup_free(scan_cleanup_ctx * ctx);

// Process image. Input: uint8 RGB/grayscale (channels=1 or 3).
// Output: allocated uint8 RGB buffer (*out_pixels), caller frees with
// scan_cleanup_free_image(). Returns 0 on success, -1 on error.
// Output dimensions may differ from input (after crop/deskew).
int scan_cleanup_process(scan_cleanup_ctx * ctx,
                         const uint8_t * pixels, int width, int height, int channels,
                         scan_cleanup_params params,
                         uint8_t ** out_pixels, int * out_width, int * out_height);

// Free an image buffer returned by scan_cleanup_process.
void scan_cleanup_free_image(uint8_t * pixels);

// Individual operations (for fine-grained control).
// All operate on grayscale float [0,1] images unless noted.

// Detect skew angle in degrees (positive = clockwise).
// Returns angle; 0.0 if no strong lines detected.
float scan_cleanup_detect_angle(const float * gray, int w, int h,
                                float max_angle_deg);

// Rotate image by angle (degrees, positive = counter-clockwise correction).
// Allocates *out (w_out * h_out floats). Caller frees with free().
void scan_cleanup_rotate(const float * gray, int w, int h, float angle_deg,
                         float ** out, int * w_out, int * h_out);

// Otsu global threshold. Returns threshold in [0,1].
float scan_cleanup_otsu(const float * gray, int w, int h);

// Sauvola adaptive binarization. Writes binary {0.0, 1.0} into dst (same size).
void scan_cleanup_sauvola(const float * gray, int w, int h,
                          int window, float k, float * dst);

// Detect content rectangle (crop dark borders).
// Returns crop rect (x0, y0, x1, y1) in pixel coordinates.
void scan_cleanup_find_content_rect(const float * gray, int w, int h,
                                    float border_threshold,
                                    int * x0, int * y0, int * x1, int * y1);

// Background whitening via morphological open.
// dst must be pre-allocated (w * h floats).
void scan_cleanup_whiten(const float * gray, int w, int h,
                         int kernel_size, float * dst);

#ifdef __cplusplus
}
#endif

#endif // SCAN_CLEANUP_H
