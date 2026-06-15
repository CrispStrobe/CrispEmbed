// dewarp.cpp — Page dewarping for camera-captured and book-scanned documents.
//
// Algorithm cherry-picked from Leptonica's dewarp2/dewarp3 (BSD-2, Bloomberg).
// Self-contained C++, uses morph_fast.h and cc_detect.h.
//
// Approach:
//   1. Binarize and find textline regions (via CC detection)
//   2. For each textline, trace the vertical center at regular x intervals
//   3. Fit a polynomial (cubic) to each baseline
//   4. Build a 2D vertical disparity map by interpolating between baselines
//   5. Apply the disparity map via bilinear interpolation

#include "dewarp.h"
#include "morph_fast.h"
#include "cc_detect.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Cubic polynomial fitting (least squares)
// ---------------------------------------------------------------------------
// Fit y = a0 + a1*x + a2*x^2 + a3*x^3 to a set of (x,y) points.
// Uses normal equations with Gaussian elimination.

struct cubic_poly {
    double a[4]; // a0, a1, a2, a3
    double eval(double x) const {
        return a[0] + x * (a[1] + x * (a[2] + x * a[3]));
    }
};

static bool fit_cubic(const float * xs, const float * ys, int n, cubic_poly & out) {
    if (n < 4) return false;

    // Build normal equations: A^T A c = A^T y where A = [1, x, x^2, x^3]
    double M[4][5] = {}; // augmented matrix [4x5]
    for (int i = 0; i < n; i++) {
        double x = xs[i], y = ys[i];
        double xp[7]; xp[0] = 1;
        for (int k = 1; k < 7; k++) xp[k] = xp[k-1] * x;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++)
                M[r][c] += xp[r + c];
            M[r][4] += xp[r] * y;
        }
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(M[r][col]) > fabs(M[pivot][col])) pivot = r;
        if (fabs(M[pivot][col]) < 1e-12) return false;
        if (pivot != col) std::swap(M[col], M[pivot]);
        double div = M[col][col];
        for (int c = col; c < 5; c++) M[col][c] /= div;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double factor = M[r][col];
            for (int c = col; c < 5; c++) M[r][c] -= factor * M[col][c];
        }
    }
    for (int i = 0; i < 4; i++) out.a[i] = M[i][4];
    return true;
}

// ---------------------------------------------------------------------------
// Baseline extraction
// ---------------------------------------------------------------------------
// For each detected textline region, sample the vertical midpoint at
// regular x intervals to trace the baseline.

struct baseline {
    std::vector<float> xs, ys; // sampled points
    int y_min, y_max;          // vertical extent
    float y_mean;              // mean y position
    cubic_poly poly;           // fitted polynomial
    bool valid;
};

static std::vector<baseline> extract_baselines(
    const uint8_t * gray, int w, int h,
    const cc_text_region * regions, int n_regions, int sampling)
{
    // Binarize for scanning
    // Simple: use Otsu threshold
    int hist[256] = {};
    for (int i = 0; i < w*h; i++) hist[gray[i]]++;
    double sum = 0; for (int i = 0; i < 256; i++) sum += (double)i*hist[i];
    double sumB = 0; int wB = 0; double maxv = 0; int best = 128;
    for (int t = 0; t < 256; t++) {
        wB += hist[t]; if (!wB) continue; int wF = w*h - wB; if (!wF) break;
        sumB += (double)t*hist[t]; double d = sumB/wB - (sum-sumB)/wF;
        double v = (double)wB*wF*d*d; if (v > maxv) { maxv = v; best = t; }
    }
    uint8_t thresh = (uint8_t)(best < 255 ? best + 1 : best);

    std::vector<baseline> baselines;
    for (int r = 0; r < n_regions; r++) {
        const cc_text_region & reg = regions[r];
        if (reg.w < 20 || reg.h < 3) continue;

        baseline bl;
        bl.y_min = reg.y;
        bl.y_max = reg.y + reg.h;
        bl.valid = false;

        // Sample midpoints at each x position (stepped by sampling)
        for (int x = reg.x; x < reg.x + reg.w; x += sampling) {
            // Find vertical extent of foreground at this column within the region
            int top = -1, bot = -1;
            for (int y = reg.y; y < reg.y + reg.h && y < h; y++) {
                if (gray[y * w + x] < thresh) {
                    if (top < 0) top = y;
                    bot = y;
                }
            }
            if (top >= 0 && bot >= 0) {
                bl.xs.push_back((float)x);
                bl.ys.push_back((float)(top + bot) / 2.0f); // midpoint
            }
        }

        if ((int)bl.xs.size() >= 4) {
            // Fit cubic polynomial
            bl.valid = fit_cubic(bl.xs.data(), bl.ys.data(), (int)bl.xs.size(), bl.poly);
            if (bl.valid) {
                bl.y_mean = 0;
                for (float y : bl.ys) bl.y_mean += y;
                bl.y_mean /= bl.ys.size();
            }
        }
        if (bl.valid) baselines.push_back(bl);
    }

    // Sort by mean y position (top to bottom)
    std::sort(baselines.begin(), baselines.end(),
              [](const baseline & a, const baseline & b) { return a.y_mean < b.y_mean; });
    return baselines;
}

// ---------------------------------------------------------------------------
// Build vertical disparity map
// ---------------------------------------------------------------------------
// For each pixel (x, y), compute the vertical shift needed to straighten
// the text. The shift is interpolated between the nearest baselines above
// and below.

static void build_disparity_map(
    const std::vector<baseline> & baselines,
    int w, int h, float * disparity)
{
    if (baselines.empty()) {
        memset(disparity, 0, w * h * sizeof(float));
        return;
    }

    // For each baseline, the disparity at x is:
    //   desired_y(x) = baseline.y_mean  (straightened to horizontal)
    //   actual_y(x)  = baseline.poly.eval(x)
    //   disparity     = desired_y - actual_y (positive = shift down)

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Find the two nearest baselines (above and below this y)
            int above = -1, below = -1;
            for (int i = 0; i < (int)baselines.size(); i++) {
                if (baselines[i].y_mean <= y) above = i;
                if (baselines[i].y_mean >= y && below < 0) below = i;
            }

            float disp = 0;
            if (above >= 0 && below >= 0 && above != below) {
                // Interpolate between the two baselines
                float ya = baselines[above].y_mean;
                float yb = baselines[below].y_mean;
                float t = (y - ya) / (yb - ya);

                float disp_a = baselines[above].y_mean - baselines[above].poly.eval(x);
                float disp_b = baselines[below].y_mean - baselines[below].poly.eval(x);
                disp = disp_a * (1 - t) + disp_b * t;
            } else if (above >= 0) {
                disp = baselines[above].y_mean - baselines[above].poly.eval(x);
            } else if (below >= 0) {
                disp = baselines[below].y_mean - baselines[below].poly.eval(x);
            }

            disparity[y * w + x] = disp;
        }
    }
}

// ---------------------------------------------------------------------------
// Apply disparity map via bilinear interpolation
// ---------------------------------------------------------------------------

static void apply_warp(const uint8_t * src, int w, int h,
                        const float * disparity,
                        uint8_t * dst) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dy = disparity[y * w + x];
            float src_y = y - dy; // source y position

            // Bilinear interpolation
            int y0 = (int)floorf(src_y);
            int y1 = y0 + 1;
            float fy = src_y - y0;

            if (y0 < 0 || y1 >= h) {
                dst[y * w + x] = 255; // white padding
                continue;
            }

            float v = src[y0 * w + x] * (1 - fy) + src[y1 * w + x] * fy;
            dst[y * w + x] = (uint8_t)std::max(0.0f, std::min(255.0f, v));
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

dewarp_params dewarp_defaults(void) {
    dewarp_params p;
    p.min_lines = 4;
    p.sampling = 8;
    p.max_curve = 100.0f;
    return p;
}

int dewarp_page_params(const uint8_t * gray, int w, int h,
                       dewarp_params params,
                       uint8_t * out, int * out_w, int * out_h) {
    if (!gray || w < 100 || h < 100 || !out) {
        if (out && gray) memcpy(out, gray, w * h);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 1;
    }

    // 1. Find textline regions using CC detection
    int n_regions = 0;
    cc_text_region * regions = cc_detect_lines(gray, w, h, &n_regions);
    if (!regions || n_regions < params.min_lines) {
        // Not enough textlines to build a model
        memcpy(out, gray, w * h);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        cc_detect_free(regions);
        return 1;
    }

    // 2. Extract baselines
    auto baselines = extract_baselines(gray, w, h, regions, n_regions, params.sampling);
    cc_detect_free(regions);

    if ((int)baselines.size() < params.min_lines) {
        memcpy(out, gray, w * h);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 1;
    }

    // Check if curvature exceeds threshold
    float max_disp = 0;
    for (auto & bl : baselines) {
        for (float x : bl.xs) {
            float d = fabsf(bl.y_mean - bl.poly.eval(x));
            if (d > max_disp) max_disp = d;
        }
    }
    if (max_disp < 2.0f) {
        // Image is already straight — no dewarping needed
        memcpy(out, gray, w * h);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 0; // success but no change needed
    }
    if (max_disp > params.max_curve) {
        // Curvature too extreme — model unreliable
        memcpy(out, gray, w * h);
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        return 1;
    }

    // 3. Build disparity map
    std::vector<float> disparity(w * h);
    build_disparity_map(baselines, w, h, disparity.data());

    // 4. Apply warp
    apply_warp(gray, w, h, disparity.data(), out);

    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return 0;
}

int dewarp_page(const uint8_t * gray, int w, int h,
                uint8_t * out, int * out_w, int * out_h) {
    return dewarp_page_params(gray, w, h, dewarp_defaults(), out, out_w, out_h);
}
