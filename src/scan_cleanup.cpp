// scan_cleanup.cpp — document scan preprocessing (tier 1: classical)
//
// Implements deskew, binarization (Otsu + Sauvola), border crop,
// and background whitening via morphological open. All pure C++,
// no external dependencies beyond stdlib + math.

#include "scan_cleanup.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Context ─────────────────────────────────────────────────────────

struct scan_cleanup_ctx {
    int n_threads;
    // Tier 2: model context would go here
};

scan_cleanup_params scan_cleanup_defaults(void) {
    scan_cleanup_params p;
    p.deskew            = 1;
    p.crop_borders      = 1;
    p.whiten_background = 1;
    p.binarize          = 0;
    p.binarize_method   = 0;    // Otsu
    p.sauvola_k         = 0.2f;
    p.sauvola_window    = 25;
    p.morph_kernel      = 51;
    p.border_threshold  = 0.15f;
    p.deskew_max_angle  = 15.0f;
    return p;
}

scan_cleanup_ctx * scan_cleanup_init(const char * /*model_path*/, int n_threads) {
    auto * ctx = new scan_cleanup_ctx;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    return ctx;
}

void scan_cleanup_free(scan_cleanup_ctx * ctx) {
    delete ctx;
}

void scan_cleanup_free_image(uint8_t * pixels) {
    free(pixels);
}

// ── Helpers ─────────────────────────────────────────────────────────

// Convert RGB uint8 to grayscale float [0,1]
static std::vector<float> to_gray_f32(const uint8_t * px, int w, int h, int ch) {
    std::vector<float> gray(w * h);
    if (ch == 1) {
        for (int i = 0; i < w * h; i++) {
            gray[i] = px[i] / 255.0f;
        }
    } else {
        for (int i = 0; i < w * h; i++) {
            float r = px[i * ch + 0];
            float g = px[i * ch + 1];
            float b = px[i * ch + 2];
            gray[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
        }
    }
    return gray;
}

// Convert grayscale float [0,1] to RGB uint8
static uint8_t * gray_to_rgb_u8(const float * gray, int w, int h) {
    uint8_t * out = (uint8_t *)malloc(w * h * 3);
    if (!out) return nullptr;
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::max(0.0f, std::min(255.0f, gray[i] * 255.0f + 0.5f));
        out[i * 3 + 0] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    return out;
}

// ── 1. Deskew ───────────────────────────────────────────────────────

// Sobel edge magnitude (horizontal + vertical)
static std::vector<float> sobel_edges(const float * gray, int w, int h) {
    std::vector<float> edges(w * h, 0.0f);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            // Sobel X
            float gx = -gray[(y-1)*w + (x-1)] + gray[(y-1)*w + (x+1)]
                       -2*gray[y*w + (x-1)]   + 2*gray[y*w + (x+1)]
                       -gray[(y+1)*w + (x-1)] + gray[(y+1)*w + (x+1)];
            // Sobel Y
            float gy = -gray[(y-1)*w + (x-1)] - 2*gray[(y-1)*w + x] - gray[(y-1)*w + (x+1)]
                       +gray[(y+1)*w + (x-1)] + 2*gray[(y+1)*w + x] + gray[(y+1)*w + (x+1)];
            edges[y * w + x] = sqrtf(gx * gx + gy * gy);
        }
    }
    return edges;
}

float scan_cleanup_detect_angle(const float * gray, int w, int h,
                                float max_angle_deg) {
    // Compute Sobel edges directly on grayscale (not binarized — preserves
    // gradient information for thin text lines and anti-aliased edges)
    auto edges = sobel_edges(gray, w, h);

    // Edge threshold: top 5% of edge magnitudes (generous to catch skewed lines)
    std::vector<float> sorted_edges(edges);
    std::sort(sorted_edges.begin(), sorted_edges.end());
    float edge_thresh = sorted_edges[(int)(sorted_edges.size() * 0.95f)];
    if (edge_thresh < 0.01f) edge_thresh = 0.01f;

    // 3. Hough transform for lines near horizontal
    // Only scan angles in [-max_angle, +max_angle] with 0.1 degree steps
    const float angle_step = 0.1f;
    int n_angles = (int)(2.0f * max_angle_deg / angle_step) + 1;
    int diag = (int)sqrtf((float)(w * w + h * h)) + 1;
    int n_rho = 2 * diag;

    std::vector<int> accum(n_angles * n_rho, 0);

    // Precompute sin/cos
    std::vector<float> cos_t(n_angles), sin_t(n_angles);
    for (int ai = 0; ai < n_angles; ai++) {
        float angle = (-max_angle_deg + ai * angle_step) * (float)M_PI / 180.0f;
        // Offset by 90 degrees so we detect near-horizontal lines
        cos_t[ai] = cosf(angle + (float)M_PI / 2.0f);
        sin_t[ai] = sinf(angle + (float)M_PI / 2.0f);
    }

    // Vote
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (edges[y * w + x] < edge_thresh) continue;
            for (int ai = 0; ai < n_angles; ai++) {
                int rho = (int)(x * cos_t[ai] + y * sin_t[ai]) + diag;
                if (rho >= 0 && rho < n_rho) {
                    accum[ai * n_rho + rho]++;
                }
            }
        }
    }

    // 4. Find peak angle
    int best_ai = 0;
    int best_votes = 0;
    for (int ai = 0; ai < n_angles; ai++) {
        for (int ri = 0; ri < n_rho; ri++) {
            if (accum[ai * n_rho + ri] > best_votes) {
                best_votes = accum[ai * n_rho + ri];
                best_ai = ai;
            }
        }
    }

    float best_angle = -max_angle_deg + best_ai * angle_step;

    // Require at least some votes to be meaningful
    if (best_votes < (int)(0.01f * w)) {
        return 0.0f;
    }

    return best_angle;
}

void scan_cleanup_rotate(const float * gray, int w, int h, float angle_deg,
                         float ** out, int * w_out, int * h_out) {
    float rad = angle_deg * (float)M_PI / 180.0f;
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    // Compute output dimensions to fit the entire rotated image
    float corners[4][2] = {
        {0, 0}, {(float)w, 0}, {0, (float)h}, {(float)w, (float)h}
    };
    float cx = w / 2.0f, cy = h / 2.0f;

    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (auto & c : corners) {
        float dx = c[0] - cx, dy = c[1] - cy;
        float rx = cos_a * dx - sin_a * dy + cx;
        float ry = sin_a * dx + cos_a * dy + cy;
        min_x = std::min(min_x, rx);
        max_x = std::max(max_x, rx);
        min_y = std::min(min_y, ry);
        max_y = std::max(max_y, ry);
    }

    int ow = (int)ceilf(max_x - min_x);
    int oh = (int)ceilf(max_y - min_y);
    float ox = min_x, oy = min_y;

    float * dst = (float *)calloc(ow * oh, sizeof(float));
    if (!dst) { *out = nullptr; *w_out = *h_out = 0; return; }

    // Fill with white (1.0) background
    for (int i = 0; i < ow * oh; i++) dst[i] = 1.0f;

    // Inverse mapping with bilinear interpolation
    float inv_cos = cos_a;   // cos(-a) = cos(a)
    float inv_sin = -sin_a;  // sin(-a) = -sin(a)

    for (int dy = 0; dy < oh; dy++) {
        for (int dx = 0; dx < ow; dx++) {
            // Map output pixel back to input coordinates
            float px = (dx + ox) - cx;
            float py = (dy + oy) - cy;
            float sx = inv_cos * px - inv_sin * py + cx;
            float sy = inv_sin * px + inv_cos * py + cy;

            if (sx < 0 || sx >= w - 1 || sy < 0 || sy >= h - 1) continue;

            // Bilinear interpolation
            int ix = (int)sx, iy = (int)sy;
            float fx = sx - ix, fy = sy - iy;

            float v00 = gray[iy * w + ix];
            float v10 = gray[iy * w + ix + 1];
            float v01 = gray[(iy + 1) * w + ix];
            float v11 = gray[(iy + 1) * w + ix + 1];

            dst[dy * ow + dx] = v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy)
                               + v01 * (1-fx) * fy     + v11 * fx * fy;
        }
    }

    *out = dst;
    *w_out = ow;
    *h_out = oh;
}

// ── 2. Binarization ─────────────────────────────────────────────────

float scan_cleanup_otsu(const float * gray, int w, int h) {
    const int BINS = 256;
    int hist[BINS] = {};
    int n = w * h;

    for (int i = 0; i < n; i++) {
        int bin = std::max(0, std::min(BINS - 1, (int)(gray[i] * (BINS - 1))));
        hist[bin]++;
    }

    // Between-class variance maximization
    float sum = 0;
    for (int i = 0; i < BINS; i++) sum += i * hist[i];

    float sum_b = 0;
    int w_b = 0;
    float max_var = 0;
    int best_t = 0;

    for (int t = 0; t < BINS; t++) {
        w_b += hist[t];
        if (w_b == 0) continue;
        int w_f = n - w_b;
        if (w_f == 0) break;

        sum_b += t * hist[t];
        float mean_b = sum_b / w_b;
        float mean_f = (sum - sum_b) / w_f;
        float var = (float)w_b * w_f * (mean_b - mean_f) * (mean_b - mean_f);

        if (var > max_var) {
            max_var = var;
            best_t = t;
        }
    }

    return (float)best_t / (BINS - 1);
}

void scan_cleanup_sauvola(const float * gray, int w, int h,
                          int window, float k, float * dst) {
    if (window % 2 == 0) window++;
    int half = window / 2;

    // Build integral images for sum and sum-of-squares
    // Use 1-indexed to simplify boundary handling
    int stride = w + 1;
    std::vector<double> integral(stride * (h + 1), 0.0);
    std::vector<double> integral_sq(stride * (h + 1), 0.0);

    for (int y = 0; y < h; y++) {
        double row_sum = 0, row_sq = 0;
        for (int x = 0; x < w; x++) {
            float v = gray[y * w + x];
            row_sum += v;
            row_sq  += v * v;
            integral[(y + 1) * stride + (x + 1)] = row_sum + integral[y * stride + (x + 1)];
            integral_sq[(y + 1) * stride + (x + 1)] = row_sq + integral_sq[y * stride + (x + 1)];
        }
    }

    const float R = 0.5f;  // dynamic range of normalized [0,1] image

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int x0 = std::max(0, x - half);
            int y0 = std::max(0, y - half);
            int x1 = std::min(w - 1, x + half);
            int y1 = std::min(h - 1, y + half);
            int area = (x1 - x0 + 1) * (y1 - y0 + 1);

            // Sum from integral image
            double s = integral[(y1+1)*stride + (x1+1)]
                     - integral[y0*stride + (x1+1)]
                     - integral[(y1+1)*stride + x0]
                     + integral[y0*stride + x0];
            double sq = integral_sq[(y1+1)*stride + (x1+1)]
                      - integral_sq[y0*stride + (x1+1)]
                      - integral_sq[(y1+1)*stride + x0]
                      + integral_sq[y0*stride + x0];

            double mean = s / area;
            double variance = sq / area - mean * mean;
            if (variance < 0) variance = 0;
            double stddev = sqrt(variance);

            float threshold = (float)(mean * (1.0 + k * (stddev / R - 1.0)));
            dst[y * w + x] = gray[y * w + x] > threshold ? 1.0f : 0.0f;
        }
    }
}

// ── 3. Border crop ──────────────────────────────────────────────────

void scan_cleanup_find_content_rect(const float * gray, int w, int h,
                                    float border_threshold,
                                    int * x0, int * y0, int * x1, int * y1) {
    // Project mean intensity per row and column
    std::vector<float> row_mean(h, 0.0f);
    std::vector<float> col_mean(w, 0.0f);

    for (int y = 0; y < h; y++) {
        float sum = 0;
        for (int x = 0; x < w; x++) sum += gray[y * w + x];
        row_mean[y] = sum / w;
    }
    for (int x = 0; x < w; x++) {
        float sum = 0;
        for (int y = 0; y < h; y++) sum += gray[y * w + x];
        col_mean[x] = sum / h;
    }

    // Find content bounds: rows/cols where mean > threshold
    int r0 = 0, r1 = h - 1;
    int c0 = 0, c1 = w - 1;

    while (r0 < h && row_mean[r0] < border_threshold) r0++;
    while (r1 > r0 && row_mean[r1] < border_threshold) r1--;
    while (c0 < w && col_mean[c0] < border_threshold) c0++;
    while (c1 > c0 && col_mean[c1] < border_threshold) c1--;

    // Sanity: ensure minimum 10% of image
    if (r1 - r0 < h / 10) { r0 = 0; r1 = h - 1; }
    if (c1 - c0 < w / 10) { c0 = 0; c1 = w - 1; }

    *x0 = c0;
    *y0 = r0;
    *x1 = c1;
    *y1 = r1;
}

// ── 4. Background whitening ─────────────────────────────────────────

// Min-pool (erode): sliding window minimum
static void min_pool_2d(const float * src, int w, int h, int k, float * dst) {
    int half = k / 2;
    std::vector<float> tmp(w * h);

    // Horizontal pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mn = 1.0f;
            int x0 = std::max(0, x - half);
            int x1 = std::min(w - 1, x + half);
            for (int xx = x0; xx <= x1; xx++) {
                mn = std::min(mn, src[y * w + xx]);
            }
            tmp[y * w + x] = mn;
        }
    }
    // Vertical pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mn = 1.0f;
            int y0 = std::max(0, y - half);
            int y1 = std::min(h - 1, y + half);
            for (int yy = y0; yy <= y1; yy++) {
                mn = std::min(mn, tmp[yy * w + x]);
            }
            dst[y * w + x] = mn;
        }
    }
}

// Max-pool (dilate): sliding window maximum
static void max_pool_2d(const float * src, int w, int h, int k, float * dst) {
    int half = k / 2;
    std::vector<float> tmp(w * h);

    // Horizontal pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mx = 0.0f;
            int x0 = std::max(0, x - half);
            int x1 = std::min(w - 1, x + half);
            for (int xx = x0; xx <= x1; xx++) {
                mx = std::max(mx, src[y * w + xx]);
            }
            tmp[y * w + x] = mx;
        }
    }
    // Vertical pass
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mx = 0.0f;
            int y0 = std::max(0, y - half);
            int y1 = std::min(h - 1, y + half);
            for (int yy = y0; yy <= y1; yy++) {
                mx = std::max(mx, tmp[yy * w + x]);
            }
            dst[y * w + x] = mx;
        }
    }
}

void scan_cleanup_whiten(const float * gray, int w, int h,
                         int kernel_size, float * dst) {
    if (kernel_size % 2 == 0) kernel_size++;

    int n = w * h;
    std::vector<float> eroded(n);
    std::vector<float> background(n);

    // Morphological open = erode then dilate
    min_pool_2d(gray, w, h, kernel_size, eroded.data());
    max_pool_2d(eroded.data(), w, h, kernel_size, background.data());

    // Divide source by background estimate, scale to [0, 1]
    for (int i = 0; i < n; i++) {
        float bg = background[i];
        if (bg < 0.01f) bg = 0.01f;  // avoid division by zero
        float v = gray[i] / bg;
        dst[i] = std::max(0.0f, std::min(1.0f, v));
    }
}

// ── Pipeline ────────────────────────────────────────────────────────

int scan_cleanup_process(scan_cleanup_ctx * ctx,
                         const uint8_t * pixels, int width, int height, int channels,
                         scan_cleanup_params params,
                         uint8_t ** out_pixels, int * out_width, int * out_height) {
    (void)ctx;

    if (!pixels || width <= 0 || height <= 0 || !out_pixels || !out_width || !out_height) {
        return -1;
    }

    // Convert to grayscale float [0,1]
    std::vector<float> gray = to_gray_f32(pixels, width, height, channels);
    int w = width, h = height;

    // 1. Deskew
    if (params.deskew) {
        float angle = scan_cleanup_detect_angle(gray.data(), w, h,
                                                params.deskew_max_angle);
        if (fabsf(angle) > 0.1f) {
            float * rotated = nullptr;
            int rw = 0, rh = 0;
            scan_cleanup_rotate(gray.data(), w, h, -angle, &rotated, &rw, &rh);
            if (rotated) {
                gray.assign(rotated, rotated + rw * rh);
                w = rw;
                h = rh;
                free(rotated);
            }
        }
    }

    // 2. Border crop
    if (params.crop_borders) {
        int x0, y0, x1, y1;
        scan_cleanup_find_content_rect(gray.data(), w, h,
                                       params.border_threshold,
                                       &x0, &y0, &x1, &y1);
        if (x0 > 0 || y0 > 0 || x1 < w - 1 || y1 < h - 1) {
            int cw = x1 - x0 + 1;
            int ch = y1 - y0 + 1;
            std::vector<float> cropped(cw * ch);
            for (int y = 0; y < ch; y++) {
                memcpy(&cropped[y * cw], &gray[(y + y0) * w + x0],
                       cw * sizeof(float));
            }
            gray = std::move(cropped);
            w = cw;
            h = ch;
        }
    }

    // 3. Background whitening
    if (params.whiten_background) {
        std::vector<float> whitened(w * h);
        scan_cleanup_whiten(gray.data(), w, h, params.morph_kernel, whitened.data());
        gray = std::move(whitened);
    }

    // 4. Binarization (optional, last step)
    if (params.binarize) {
        if (params.binarize_method == 1) {
            // Sauvola adaptive
            std::vector<float> bin(w * h);
            scan_cleanup_sauvola(gray.data(), w, h,
                                 params.sauvola_window, params.sauvola_k,
                                 bin.data());
            gray = std::move(bin);
        } else {
            // Otsu global
            float t = scan_cleanup_otsu(gray.data(), w, h);
            for (int i = 0; i < w * h; i++) {
                gray[i] = gray[i] > t ? 1.0f : 0.0f;
            }
        }
    }

    // Convert back to RGB uint8
    *out_pixels = gray_to_rgb_u8(gray.data(), w, h);
    *out_width = w;
    *out_height = h;

    return *out_pixels ? 0 : -1;
}
