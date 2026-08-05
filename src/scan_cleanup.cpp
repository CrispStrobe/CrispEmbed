// scan_cleanup.cpp — document scan preprocessing (tier 1: classical)
//
// Implements deskew, binarization (Otsu + Sauvola), border crop,
// and background whitening via morphological closing. All pure C++,
// no external dependencies beyond stdlib + math.

#include "scan_cleanup.h"
#include "classical_preproc.h"
#include "nafnet_denoise.h"
#include "core/env_gate.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <deque>
#include <vector>
#include <cstdio>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Context ─────────────────────────────────────────────────────────

struct scan_cleanup_ctx {
    int n_threads;
    nafnet_context * nafnet = nullptr; // Tier 2: learned denoising (optional)
    bool bench = false;
};

scan_cleanup_params scan_cleanup_defaults(void) {
    scan_cleanup_params p;
    p.deskew = 1;
    p.crop_borders = 1;
    p.whiten_background = 1;
    p.binarize = 0;
    p.binarize_method = 0; // Otsu
    p.sauvola_k = 0.2f;
    p.sauvola_window = 25;
    p.morph_kernel = 51;
    p.border_threshold = 0.15f;
    p.deskew_max_angle = 15.0f;
    p.despeckle = 1;
    p.despeckle_thresh = 0.25f;
    p.blackfilter = 1;
    p.blackfilter_thresh = 0.20f;
    p.deskew_consensus = 1;
    return p;
}

scan_cleanup_ctx * scan_cleanup_init(const char * model_path, int n_threads) {
    auto * ctx = new scan_cleanup_ctx;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->bench = core_env::on("CRISPEMBED_SCAN_CLEANUP_BENCH");
    if (model_path) {
        ctx->nafnet = nafnet_init(model_path, n_threads);
        if (!ctx->nafnet) {
            fprintf(stderr, "scan_cleanup: warning: failed to load denoising model, tier 2 disabled\n");
        }
    }
    return ctx;
}

void scan_cleanup_free(scan_cleanup_ctx * ctx) {
    if (ctx) {
        if (ctx->nafnet) nafnet_free(ctx->nafnet);
        delete ctx;
    }
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
            float gx = -gray[(y - 1) * w + (x - 1)] + gray[(y - 1) * w + (x + 1)] - 2 * gray[y * w + (x - 1)] +
                       2 * gray[y * w + (x + 1)] - gray[(y + 1) * w + (x - 1)] + gray[(y + 1) * w + (x + 1)];
            // Sobel Y
            float gy = -gray[(y - 1) * w + (x - 1)] - 2 * gray[(y - 1) * w + x] - gray[(y - 1) * w + (x + 1)] +
                       gray[(y + 1) * w + (x - 1)] + 2 * gray[(y + 1) * w + x] + gray[(y + 1) * w + (x + 1)];
            edges[y * w + x] = sqrtf(gx * gx + gy * gy);
        }
    }
    return edges;
}

float scan_cleanup_detect_angle(const float * gray, int w, int h, float max_angle_deg) {
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

    // 4. Find the skew angle by rho-projection *concentration* (energy), not the
    //    single highest (angle,rho) bin. The single-max-bin peak is noise-fragile
    //    and false-positives on sparse / already-aligned text (e.g. it reported
    //    ~4.2 deg on perfectly horizontal rendered text, rotating a clean page and
    //    distorting downstream OCR). At the true orientation, edges collapse into a
    //    few strong rho bins (high sum-of-squares); a wrong tilt smears them across
    //    many bins (low sum-of-squares).
    int best_ai = 0;
    double best_energy = -1.0;
    int best_peak = 0;
    for (int ai = 0; ai < n_angles; ai++) {
        const int * row = &accum[(size_t)ai * n_rho];
        double energy = 0.0;
        int peak = 0;
        for (int ri = 0; ri < n_rho; ri++) {
            energy += (double)row[ri] * (double)row[ri];
            if (row[ri] > peak) peak = row[ri];
        }
        if (energy > best_energy) {
            best_energy = energy;
            best_ai = ai;
            best_peak = peak;
        }
    }

    float best_angle = -max_angle_deg + best_ai * angle_step;

    // Require a meaningful line peak to trust the estimate at all.
    if (best_peak < (int)(0.01f * w)) {
        return 0.0f;
    }

    // Confidence gate: only deskew if the best angle's concentration clearly
    // beats the 0-degree (horizontal) baseline, otherwise assume the page is
    // already aligned and leave it untouched (no spurious rotation).
    int zero_ai = (int)(max_angle_deg / angle_step + 0.5f);
    if (zero_ai >= 0 && zero_ai < n_angles) {
        const int * zrow = &accum[(size_t)zero_ai * n_rho];
        double zero_energy = 0.0;
        for (int ri = 0; ri < n_rho; ri++) zero_energy += (double)zrow[ri] * (double)zrow[ri];
        if (best_energy < zero_energy * 1.10) return 0.0f;
    }
    if (fabsf(best_angle) < 0.3f) return 0.0f; // negligible skew

    return best_angle;
}

float scan_cleanup_detect_angle_consensus(const float * gray, int w, int h, float max_angle_deg) {
    float a_hough = scan_cleanup_detect_angle(gray, w, h, max_angle_deg);

    // Second opinion: differential-square-sum detector (classical_preproc.h).
    // Empirically its angle is OPPOSITE-signed vs the Hough detector (verified
    // on synthetic rotations: rotate(+3°) → hough=+3.0, dss=-3.5) and it only
    // sweeps ±7°, so cross-checking is limited to that range.
    std::vector<uint8_t> u8((size_t)w * h);
    for (int i = 0; i < w * h; i++) {
        float v = gray[i] * 255.0f + 0.5f;
        u8[i] = (uint8_t)std::max(0.0f, std::min(255.0f, v));
    }
    float a_dss = 0.0f, conf = 0.0f;
    int rc = find_skew_angle(u8.data(), w, h, &a_dss, &conf);
    bool dss_ok = (rc == 0 && conf >= 3.0f); // >3 = good per classical_preproc.h
    float a_dss_mapped = -a_dss;             // map into the Hough sign convention

    const float dss_sweep = 6.0f; // find_skew_angle sweeps ±7°; keep margin
    // DSS overestimates the magnitude by a resolution-dependent bias (its 4×
    // reduction: ~0.5° on 800px pages, ~1.2° on 400px) but the SIGN is always
    // reliable, so gate on sign agreement plus a generous magnitude band.
    const float agree_tol = 1.5f;

    if (a_hough != 0.0f) {
        if (fabsf(a_hough) > dss_sweep) return a_hough; // beyond DSS range, can't cross-check
        if (!dss_ok) return a_hough;                    // no usable second opinion
        bool sign_conflict = a_hough * a_dss_mapped < 0.0f && fabsf(a_dss_mapped) >= 0.5f;
        if (sign_conflict) return 0.0f;
        return fabsf(a_dss_mapped - a_hough) <= agree_tol ? a_hough : 0.0f;
    }

    // Hough gated out (below its confidence/peak thresholds). Fall back to the
    // DSS estimate only on a strong, clearly nonzero signal within range.
    if (dss_ok && conf >= 4.0f && fabsf(a_dss_mapped) >= 0.3f && fabsf(a_dss_mapped) <= dss_sweep &&
        fabsf(a_dss_mapped) <= max_angle_deg) {
        return a_dss_mapped;
    }
    return 0.0f;
}

void scan_cleanup_rotate(const float * gray, int w, int h, float angle_deg, float ** out, int * w_out, int * h_out) {
    float rad = angle_deg * (float)M_PI / 180.0f;
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    // Compute output dimensions to fit the entire rotated image
    float corners[4][2] = { { 0, 0 }, { (float)w, 0 }, { 0, (float)h }, { (float)w, (float)h } };
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
    if (!dst) {
        *out = nullptr;
        *w_out = *h_out = 0;
        return;
    }

    // Port 3 (verified): fill rotation corners with pure white (1.0). This is
    // deliberately NOT the detected paper gray — after background-whitening the
    // white corners stay white (1.0 = max, can't brighten), giving clean uniform
    // corners; filling with paper-gray instead makes the whitening's local
    // normalisation leave VISIBLE gray wedges at the corner boundaries (confirmed
    // by A/B image inspection — CER unchanged but the page looked worse). The
    // gray-wedge symptom was already resolved by the whitening closing fix
    // (6fdd1b5); no further change is needed here.
    for (int i = 0; i < ow * oh; i++) dst[i] = 1.0f;

    // Inverse mapping with bilinear interpolation
    float inv_cos = cos_a;  // cos(-a) = cos(a)
    float inv_sin = -sin_a; // sin(-a) = -sin(a)

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

            dst[dy * ow + dx] = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
        }
    }

    *out = dst;
    *w_out = ow;
    *h_out = oh;
}

int scan_cleanup_deskew_rgb(const uint8_t * pixels, int w, int h, int channels, float max_angle_deg,
                            uint8_t ** out_pixels, int * out_w, int * out_h) {
    if (!pixels || w <= 0 || h <= 0 || (channels != 1 && channels != 3 && channels != 4) || !out_pixels || !out_w ||
        !out_h) {
        return -1;
    }
    *out_pixels = nullptr;
    *out_w = w;
    *out_h = h;

    std::vector<float> gray = to_gray_f32(pixels, w, h, channels);
    float angle = scan_cleanup_detect_angle_consensus(gray.data(), w, h, max_angle_deg);
    if (fabsf(angle) <= 0.1f) return 0; // straight enough — no output buffer

    // Correction rotation (same -angle convention as the grayscale pipeline),
    // preserving all channels. White fill so document margins stay paper-like.
    float rad = -angle * (float)M_PI / 180.0f;
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    float cx = w / 2.0f, cy = h / 2.0f;
    float corners[4][2] = { { 0, 0 }, { (float)w, 0 }, { 0, (float)h }, { (float)w, (float)h } };
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

    uint8_t * dst = (uint8_t *)malloc((size_t)ow * oh * channels);
    if (!dst) return -1;
    memset(dst, 0xFF, (size_t)ow * oh * channels); // white (and opaque alpha)

    float inv_cos = cos_a;
    float inv_sin = -sin_a;
    for (int dy = 0; dy < oh; dy++) {
        for (int dx = 0; dx < ow; dx++) {
            float px = (dx + min_x) - cx;
            float py = (dy + min_y) - cy;
            float sx = inv_cos * px - inv_sin * py + cx;
            float sy = inv_sin * px + inv_cos * py + cy;
            if (sx < 0 || sx >= w - 1 || sy < 0 || sy >= h - 1) continue;
            int ix = (int)sx, iy = (int)sy;
            float fx = sx - ix, fy = sy - iy;
            const uint8_t * p00 = pixels + ((size_t)iy * w + ix) * channels;
            const uint8_t * p10 = p00 + channels;
            const uint8_t * p01 = p00 + (size_t)w * channels;
            const uint8_t * p11 = p01 + channels;
            uint8_t * q = dst + ((size_t)dy * ow + dx) * channels;
            for (int c = 0; c < channels; c++) {
                float v =
                    p00[c] * (1 - fx) * (1 - fy) + p10[c] * fx * (1 - fy) + p01[c] * (1 - fx) * fy + p11[c] * fx * fy;
                q[c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }
    }

    *out_pixels = dst;
    *out_w = ow;
    *out_h = oh;
    return 0;
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

void scan_cleanup_sauvola(const float * gray, int w, int h, int window, float k, float * dst) {
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
            row_sq += v * v;
            integral[(y + 1) * stride + (x + 1)] = row_sum + integral[y * stride + (x + 1)];
            integral_sq[(y + 1) * stride + (x + 1)] = row_sq + integral_sq[y * stride + (x + 1)];
        }
    }

    const float R = 0.5f; // dynamic range of normalized [0,1] image

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int x0 = std::max(0, x - half);
            int y0 = std::max(0, y - half);
            int x1 = std::min(w - 1, x + half);
            int y1 = std::min(h - 1, y + half);
            int area = (x1 - x0 + 1) * (y1 - y0 + 1);

            // Sum from integral image
            double s = integral[(y1 + 1) * stride + (x1 + 1)] - integral[y0 * stride + (x1 + 1)] -
                       integral[(y1 + 1) * stride + x0] + integral[y0 * stride + x0];
            double sq = integral_sq[(y1 + 1) * stride + (x1 + 1)] - integral_sq[y0 * stride + (x1 + 1)] -
                        integral_sq[(y1 + 1) * stride + x0] + integral_sq[y0 * stride + x0];

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

void scan_cleanup_find_content_rect(const float * gray, int w, int h, float border_threshold, int * x0, int * y0,
                                    int * x1, int * y1) {
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
    if (r1 - r0 < h / 10) {
        r0 = 0;
        r1 = h - 1;
    }
    if (c1 - c0 < w / 10) {
        c0 = 0;
        c1 = w - 1;
    }

    *x0 = c0;
    *y0 = r0;
    *x1 = c1;
    *y1 = r1;
}

// ── 4. Background whitening ─────────────────────────────────────────

// Monotonic deque 1D sliding-window extremum — O(n) total instead of O(n*k).
// is_min=true for min-pool (erode), false for max-pool (dilate).
static void slide_1d(const float * in, float * out, int len, int k, bool is_min) {
    int half = k / 2;
    std::deque<int> dq; // indices of candidates
    for (int i = 0; i < len; i++) {
        // Remove elements outside the window
        while (!dq.empty() && dq.front() < i - half) dq.pop_front();
        // Maintain monotonicity
        if (is_min) {
            while (!dq.empty() && in[dq.back()] >= in[i]) dq.pop_back();
        } else {
            while (!dq.empty() && in[dq.back()] <= in[i]) dq.pop_back();
        }
        dq.push_back(i);
        out[i] = in[dq.front()];
    }
}

// Min-pool (erode): separable 2-pass with monotonic deque — O(w*h) total
static void min_pool_2d(const float * src, int w, int h, int k, float * dst) {
    std::vector<float> tmp(w * h);
    // Horizontal pass
    for (int y = 0; y < h; y++) slide_1d(src + y * w, tmp.data() + y * w, w, k, true);
    // Vertical pass (column-wise via transposed access)
    std::vector<float> col(h), col_out(h);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) col[y] = tmp[y * w + x];
        slide_1d(col.data(), col_out.data(), h, k, true);
        for (int y = 0; y < h; y++) dst[y * w + x] = col_out[y];
    }
}

// Max-pool (dilate): separable 2-pass with monotonic deque — O(w*h) total
static void max_pool_2d(const float * src, int w, int h, int k, float * dst) {
    std::vector<float> tmp(w * h);
    // Horizontal pass
    for (int y = 0; y < h; y++) slide_1d(src + y * w, tmp.data() + y * w, w, k, false);
    // Vertical pass
    std::vector<float> col(h), col_out(h);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) col[y] = tmp[y * w + x];
        slide_1d(col.data(), col_out.data(), h, k, false);
        for (int y = 0; y < h; y++) dst[y * w + x] = col_out[y];
    }
}

void scan_cleanup_whiten(const float * gray, int w, int h, int kernel_size, float * dst) {
    if (kernel_size % 2 == 0) kernel_size++;

    int n = w * h;
    std::vector<float> dilated(n);
    std::vector<float> background(n);

    // Estimate the paper illumination with a morphological CLOSING (dilate then
    // erode). A document is DARK text on a BRIGHT background, so the background is
    // recovered by removing the dark text: dilation (max-pool) floods paper over
    // text strokes narrower than the kernel, then erosion (min-pool) restores the
    // paper geometry. Dividing by this closing flattens uneven lighting while
    // KEEPING the text (text/paper ≈ 0.1 stays dark).
    //
    // The previous OPENING (erode→dilate) estimated the wrong thing: on dense text
    // it has no kernel-sized text-free region, so the "background" collapsed to the
    // dark text level and gray/bg saturated to white — erasing the text.
    max_pool_2d(gray, w, h, kernel_size, dilated.data());
    min_pool_2d(dilated.data(), w, h, kernel_size, background.data());

    // Divide source by background estimate, scale to [0, 1]
    for (int i = 0; i < n; i++) {
        float bg = background[i];
        if (bg < 0.01f) bg = 0.01f; // avoid division by zero
        float v = gray[i] / bg;
        dst[i] = std::max(0.0f, std::min(1.0f, v));
    }
}

// ── Page split (Port 5 of the unpaper feature set) ──────────────────
// Detect a two-up (double-page) book spread and return the gutter column to split
// at, or -1 for a single page. Clean-room, projection-profile based: a spread is
// wide, and its column dark-pixel density has two content humps separated by a
// near-empty vertical gutter near the centre. Find the emptiest central column;
// accept it only if it is a real gutter (nearly no text) with substantial text on
// BOTH sides — so single pages, blank pages, and centred figures never false-split.
int scan_cleanup_detect_page_split(const uint8_t * pixels, int w, int h, int channels) {
    if (w <= 8 || h <= 8) return -1;
    if ((float)w / (float)h < 1.15f) return -1; // spreads are landscape; portrait = single

    std::vector<float> gray = to_gray_f32(pixels, w, h, channels);
    const float text_thresh = 0.5f;
    std::vector<int> content(w, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (gray[(size_t)y * w + x] < text_thresh) content[x]++;

    std::vector<int> srt(content);
    std::sort(srt.begin(), srt.end());
    int med = srt[w / 2];    // typical column's text density
    if (med <= 0) return -1; // blank

    int lo = (int)(0.30f * w), hi = (int)(0.70f * w);
    int gx = lo, gmin = content[lo];
    for (int x = lo; x <= hi; x++)
        if (content[x] < gmin) {
            gmin = content[x];
            gx = x;
        }
    if (gmin > (int)(0.15f * med)) return -1; // no clear (near-empty) gutter

    long lsum = 0, rsum = 0;
    for (int x = 0; x < gx; x++) lsum += content[x];
    for (int x = gx + 1; x < w; x++) rsum += content[x];
    float lmean = gx > 0 ? (float)lsum / gx : 0.0f;
    float rmean = (w - gx - 1) > 0 ? (float)rsum / (w - gx - 1) : 0.0f;
    if (lmean < 0.4f * med || rmean < 0.4f * med) return -1; // one side is empty → not a spread
    return gx;
}

// ── Content bbox (Port 6 of the unpaper feature set) ────────────────
// Tight bounding box of the printed content (text/ink), trimming blank margins of
// any colour — for centering / border alignment / normalized output geometry.
// Clean-room: per-row and per-column dark-pixel projection profiles; a row/column
// counts as "content" once its dark-pixel count exceeds a small floor, and the box
// is the first/last such row/column (with a tiny pad). OCR-neutral (tesseract
// ignores margins); this is a geometry helper for the caller.
int scan_cleanup_content_bbox(const uint8_t * pixels, int w, int h, int channels, int * x0, int * y0, int * x1,
                              int * y1) {
    if (w <= 0 || h <= 0) return -1;
    std::vector<float> gray = to_gray_f32(pixels, w, h, channels);
    const float text_thresh = 0.5f;
    std::vector<int> rowc(h, 0), colc(w, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (gray[(size_t)y * w + x] < text_thresh) {
                rowc[y]++;
                colc[x]++;
            }
    const int row_floor = std::max(1, (int)(0.003f * w)); // > ~0.3% of the row is ink
    const int col_floor = std::max(1, (int)(0.003f * h));
    int ry0 = 0, ry1 = h - 1, cx0 = 0, cx1 = w - 1;
    while (ry0 < h && rowc[ry0] < row_floor) ry0++;
    while (ry1 > ry0 && rowc[ry1] < row_floor) ry1--;
    while (cx0 < w && colc[cx0] < col_floor) cx0++;
    while (cx1 > cx0 && colc[cx1] < col_floor) cx1--;
    if (ry0 > ry1 || cx0 > cx1) return -1; // blank
    const int pad = 2;
    *x0 = std::max(0, cx0 - pad);
    *y0 = std::max(0, ry0 - pad);
    *x1 = std::min(w, cx1 + 1 + pad);
    *y1 = std::min(h, ry1 + 1 + pad);
    return 0;
}

// ── Despeckle (Port 1 of the unpaper feature set) ───────────────────
// Remove isolated dark specks (scanner dust, salt-and-pepper) with a
// decision-based 3x3 median: a pixel is replaced by its local median ONLY when it
// differs from that median by more than `thresh`. An isolated speck sits on light
// paper so its neighbourhood median is light → it is lifted; a text-stroke pixel
// has dark neighbours so its median ≈ itself → it is preserved. This keeps text
// intact while clearing impulse noise (which unpaper's cluster noisefilter, tuned
// for larger blobs, does not remove). Symmetric, so it also drops bright pinholes
// inside dark regions. Grayscale [0,1], in place.
static void scan_cleanup_despeckle(std::vector<float> & gray, int w, int h, float thresh) {
    if (w < 3 || h < 3) return;
    std::vector<float> out(gray);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float win[9];
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) win[k++] = gray[(y + dy) * w + (x + dx)];
            std::nth_element(win, win + 4, win + 9);
            float med = win[4];
            if (fabsf(gray[y * w + x] - med) > thresh) out[y * w + x] = med;
        }
    }
    gray.swap(out);
}

// ── Blackfilter (Port 2 of the unpaper feature set) ─────────────────
// Clear large SOLID dark regions that are not text — scanner-bed / lifted-page
// shadows, dark photocopy edges, big blobs/pinholes — which the rectangular
// border-crop can't reach. Clean-room: label 8-connected components of "very
// dark" pixels, then whiten a component only when it is both LARGE (bigger than
// any glyph) and SOLID (bounding-box fill ratio high). Text glyphs/lines are
// small or low-fill (thin strokes in their bbox), so they are kept.
//
// Hard guard against unpaper's failure mode (it blanked whole pages): never clear
// more than 40% of the page — if that much is "dark solid", it is probably a dark
// scan or an inverted image, not a shadow, so leave it alone.
static void scan_cleanup_blackfilter(std::vector<float> & gray, int w, int h, float thresh) {
    const int n = w * h;
    if (n < 64) return;
    std::vector<uint8_t> dark(n);
    for (int i = 0; i < n; i++) dark[i] = gray[i] < thresh ? 1 : 0;

    const int64_t min_area = std::max<int64_t>(64, (int64_t)(0.0008 * n)); // > a glyph
    const float min_fill = 0.50f;                                          // solid, not strokes
    const int64_t max_clear = (int64_t)(0.40 * n);                         // page-blank guard

    std::vector<int> label(n, 0);
    std::vector<int> stack;
    std::vector<int> to_clear; // pixel indices queued for whitening
    int64_t clear_total = 0;
    int cur = 0;
    const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };

    for (int s = 0; s < n; s++) {
        if (!dark[s] || label[s]) continue;
        cur++;
        stack.clear();
        stack.push_back(s);
        label[s] = cur;
        std::vector<int> comp;
        int minx = w, maxx = 0, miny = h, maxy = 0;
        double ring_sum = 0.0; // gray of the just-outside boundary ring
        int64_t ring_cnt = 0;
        while (!stack.empty()) {
            int p = stack.back();
            stack.pop_back();
            comp.push_back(p);
            int px = p % w, py = p / w;
            if (px < minx) minx = px;
            if (px > maxx) maxx = px;
            if (py < miny) miny = py;
            if (py > maxy) maxy = py;
            for (int k = 0; k < 8; k++) {
                int nx = px + dx[k], ny = py + dy[k];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                int q = ny * w + nx;
                if (dark[q]) {
                    if (!label[q]) {
                        label[q] = cur;
                        stack.push_back(q);
                    }
                } else {
                    ring_sum += gray[q]; // a non-dark neighbour → the outside ring
                    ring_cnt++;
                }
            }
        }
        int64_t area = (int64_t)comp.size();
        if (area < min_area) continue;
        int64_t bbox = (int64_t)(maxx - minx + 1) * (maxy - miny + 1);
        float fill = bbox > 0 ? (float)area / (float)bbox : 0.0f;
        if (fill < min_fill) continue; // strokes/text, keep
        // Sharpness gate: clear only if the region is bordered by BRIGHT paper (a
        // sharp-edged shadow/blob). A soft dark gradient (vignette/stain) is
        // bordered by more dark gradient — leave it for the whitening step, so we
        // never carve a black-edged hole out of a readable illumination gradient.
        float ring = ring_cnt > 0 ? (float)(ring_sum / ring_cnt) : 1.0f;
        if (ring < 0.55f) continue;
        clear_total += area;
        for (int p : comp) to_clear.push_back(p);
    }

    if (clear_total == 0 || clear_total > max_clear) return; // nothing, or guard tripped
    for (int p : to_clear) gray[p] = 1.0f;                   // whiten the shadow/blob
}

// ── Pipeline ────────────────────────────────────────────────────────

int scan_cleanup_process(scan_cleanup_ctx * ctx, const uint8_t * pixels, int width, int height, int channels,
                         scan_cleanup_params params, uint8_t ** out_pixels, int * out_width, int * out_height) {
    if (!pixels || width <= 0 || height <= 0 || !out_pixels || !out_width || !out_height) {
        return -1;
    }

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Convert to grayscale float [0,1]
    std::vector<float> gray = to_gray_f32(pixels, width, height, channels);
    int w = width, h = height;

    // 0. Despeckle (before deskew/whiten so specks don't skew angle detection
    //    or corrupt the morphological background estimate).
    if (params.despeckle) {
        auto t0 = std::chrono::steady_clock::now();
        scan_cleanup_despeckle(gray, w, h, params.despeckle_thresh);
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] despeckle: %.1f ms\n", ms);
        }
    }

    // 0b. Blackfilter: clear large solid dark shadows/blobs before deskew (a dark
    //     edge otherwise biases the Hough angle) and before whiten. It only clears
    //     SHARP-edged solid objects (real shadows/blobs), leaving SOFT dark
    //     gradients (vignettes/stains) for the whitening step — otherwise clearing
    //     a vignette's core would leave a gradient edge that whitening smears to
    //     black, destroying readable text.
    if (params.blackfilter) {
        auto t0 = std::chrono::steady_clock::now();
        scan_cleanup_blackfilter(gray, w, h, params.blackfilter_thresh);
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] blackfilter: %.1f ms\n", ms);
        }
    }

    // 1. Deskew
    if (params.deskew) {
        auto t0 = std::chrono::steady_clock::now();
        float angle = params.deskew_consensus
                          ? scan_cleanup_detect_angle_consensus(gray.data(), w, h, params.deskew_max_angle)
                          : scan_cleanup_detect_angle(gray.data(), w, h, params.deskew_max_angle);
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
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] deskew: %.1f ms\n", ms);
        }
    }

    // 2. Border crop
    if (params.crop_borders) {
        auto t0 = std::chrono::steady_clock::now();
        int x0, y0, x1, y1;
        scan_cleanup_find_content_rect(gray.data(), w, h, params.border_threshold, &x0, &y0, &x1, &y1);
        if (x0 > 0 || y0 > 0 || x1 < w - 1 || y1 < h - 1) {
            int cw = x1 - x0 + 1;
            int ch = y1 - y0 + 1;
            std::vector<float> cropped(cw * ch);
            for (int y = 0; y < ch; y++) {
                memcpy(&cropped[y * cw], &gray[(y + y0) * w + x0], cw * sizeof(float));
            }
            gray = std::move(cropped);
            w = cw;
            h = ch;
        }
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] crop: %.1f ms\n", ms);
        }
    }

    // 3. Background whitening (illumination correction: flattens uneven lighting
    //    and soft dark vignettes/stains that blackfilter deliberately left alone).
    if (params.whiten_background) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> whitened(w * h);
        scan_cleanup_whiten(gray.data(), w, h, params.morph_kernel, whitened.data());
        gray = std::move(whitened);
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] whiten: %.1f ms\n", ms);
        }
    }

    // 4. Learned denoising (tier 2, if model loaded)
    if (ctx->nafnet) {
        auto t0 = std::chrono::steady_clock::now();
        // Convert gray to RGB uint8 for NAFNet
        uint8_t * rgb_in = gray_to_rgb_u8(gray.data(), w, h);
        if (rgb_in) {
            std::vector<uint8_t> rgb_out(w * h * 3);
            if (nafnet_process(ctx->nafnet, rgb_in, w, h, rgb_out.data()) == 0) {
                // Convert back to grayscale float
                for (int i = 0; i < w * h; i++) {
                    float r = rgb_out[i * 3 + 0];
                    float g = rgb_out[i * 3 + 1];
                    float b = rgb_out[i * 3 + 2];
                    gray[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                }
            }
            free(rgb_in);
        }
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] denoise (NAFNet): %.1f ms\n", ms);
        }
    }

    // 5. Binarization (optional, last step)
    if (params.binarize) {
        auto t0 = std::chrono::steady_clock::now();
        if (params.binarize_method == 1) {
            // Sauvola adaptive
            std::vector<float> bin(w * h);
            scan_cleanup_sauvola(gray.data(), w, h, params.sauvola_window, params.sauvola_k, bin.data());
            gray = std::move(bin);
        } else {
            // Otsu global
            float t = scan_cleanup_otsu(gray.data(), w, h);
            for (int i = 0; i < w * h; i++) {
                gray[i] = gray[i] > t ? 1.0f : 0.0f;
            }
        }
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[scan_cleanup-bench] binarize: %.1f ms\n", ms);
        }
    }

    // Convert back to RGB uint8
    *out_pixels = gray_to_rgb_u8(gray.data(), w, h);
    *out_width = w;
    *out_height = h;

    if (bench) {
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[scan_cleanup-bench] total: %.1f ms\n", total_ms);
    }

    return *out_pixels ? 0 : -1;
}
