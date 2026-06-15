// classical_preproc.cpp — Classical document preprocessing algorithms.
//
// Cherry-picked from Leptonica (BSD-2-Clause, Dan Bloomberg).
// Self-contained C++, no Leptonica types. Uses morph_fast.h for 1-bit ops.

#include "classical_preproc.h"
#include "morph_fast.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// =========================================================================
// 1. Adaptive Otsu binarization (tiled + smoothed)
// =========================================================================
// Algorithm from Leptonica's pixOtsuAdaptiveThreshold:
//   - Divide image into tiles
//   - Compute Otsu threshold per tile
//   - Smooth the threshold map with box filter
//   - Interpolate per-pixel thresholds from smoothed map

static uint8_t otsu_single(const uint8_t * data, int n) {
    if (n <= 0) return 128;
    int hist[256] = {};
    for (int i = 0; i < n; i++) hist[data[i]]++;

    double sum = 0;
    for (int i = 0; i < 256; i++) sum += (double)i * hist[i];

    double sumB = 0;
    int wB = 0;
    double max_var = 0;
    int best = 128;

    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = n - wB;
        if (wF == 0) break;
        sumB += (double)t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double var = (double)wB * wF * (mB - mF) * (mB - mF);
        if (var > max_var) { max_var = var; best = t; }
    }
    return (uint8_t)best;
}

void adaptive_otsu(const uint8_t * gray, int w, int h,
                   int tile_w, int tile_h, int smooth,
                   uint8_t * out) {
    if (tile_w <= 0) tile_w = std::max(32, w / 10);
    if (tile_h <= 0) tile_h = std::max(32, h / 10);
    if (smooth <= 0) smooth = 3;

    int tw = (w + tile_w - 1) / tile_w;  // tiles across
    int th = (h + tile_h - 1) / tile_h;  // tiles down

    // Compute per-tile Otsu thresholds
    std::vector<float> tmap(tw * th);
    std::vector<uint8_t> tile_buf;
    tile_buf.reserve(tile_w * tile_h);

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            int x0 = tx * tile_w, y0 = ty * tile_h;
            int x1 = std::min(x0 + tile_w, w);
            int y1 = std::min(y0 + tile_h, h);
            tile_buf.clear();
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    tile_buf.push_back(gray[y * w + x]);
            tmap[ty * tw + tx] = (float)otsu_single(tile_buf.data(), (int)tile_buf.size());
        }
    }

    // Smooth the threshold map (box filter)
    if (smooth > 1) {
        int half = smooth / 2;
        std::vector<float> smoothed(tw * th);
        for (int ty = 0; ty < th; ty++) {
            for (int tx = 0; tx < tw; tx++) {
                float sum = 0; int cnt = 0;
                for (int dy = -half; dy <= half; dy++) {
                    for (int dx = -half; dx <= half; dx++) {
                        int sy = ty + dy, sx = tx + dx;
                        if (sy >= 0 && sy < th && sx >= 0 && sx < tw) {
                            sum += tmap[sy * tw + sx]; cnt++;
                        }
                    }
                }
                smoothed[ty * tw + tx] = sum / cnt;
            }
        }
        tmap = smoothed;
    }

    // Interpolate per-pixel thresholds and binarize
    for (int y = 0; y < h; y++) {
        // Find tile row and fractional position
        float fy = (float)y / tile_h - 0.5f;
        int ty0 = std::max(0, (int)fy);
        int ty1 = std::min(th - 1, ty0 + 1);
        float wy = fy - ty0;
        wy = std::max(0.0f, std::min(1.0f, wy));

        for (int x = 0; x < w; x++) {
            float fx = (float)x / tile_w - 0.5f;
            int tx0 = std::max(0, (int)fx);
            int tx1 = std::min(tw - 1, tx0 + 1);
            float wx = fx - tx0;
            wx = std::max(0.0f, std::min(1.0f, wx));

            // Bilinear interpolation of threshold
            float t = tmap[ty0 * tw + tx0] * (1 - wx) * (1 - wy)
                    + tmap[ty0 * tw + tx1] * wx * (1 - wy)
                    + tmap[ty1 * tw + tx0] * (1 - wx) * wy
                    + tmap[ty1 * tw + tx1] * wx * wy;

            out[y * w + x] = gray[y * w + x] < (uint8_t)(t + 0.5f) ? 0 : 255;
        }
    }
}

// =========================================================================
// 2. Differential-square-sum deskew (pixFindSkew)
// =========================================================================
// Algorithm:
//   1. Binarize + reduce image (4x)
//   2. For each candidate angle: vertical shear, compute row sums,
//      score = sum of (row_sum[i] - row_sum[i-1])^2
//   3. Sweep coarsely, then binary search around the peak.
//   4. Confidence = max_score / min_score.

// Vertical shear: for each column x, shift vertically by tan(angle)*(x - w/2).
// This moves pixels between rows, changing row sums when text is misaligned.
// At the correct deskew angle, row sums have the sharpest transitions.
static void vshear_1bit(const uint32_t * src, uint32_t * dst,
                         int w, int h, int wpl, float angle_rad) {
    memset(dst, 0, (size_t)wpl * h * sizeof(uint32_t));
    float tan_a = tanf(angle_rad);
    for (int x = 0; x < w; x++) {
        int shift = (int)(tan_a * (x - w / 2) + 0.5f);
        for (int y = 0; y < h; y++) {
            int sy = y - shift;
            if (sy < 0 || sy >= h) continue;
            // Copy bit at (x, sy) in src to (x, y) in dst
            if ((src[sy * wpl + (x >> 5)] >> (31 - (x & 31))) & 1)
                dst[y * wpl + (x >> 5)] |= (1u << (31 - (x & 31)));
        }
    }
}

// Count set bits per row
static void row_sums(const uint32_t * bits, int w, int h, int wpl,
                      int * sums) {
    for (int y = 0; y < h; y++) {
        const uint32_t * line = bits + y * wpl;
        int count = 0;
        for (int i = 0; i < wpl; i++) {
#ifdef __GNUC__
            count += __builtin_popcount(line[i]);
#else
            uint32_t v = line[i];
            v = v - ((v >> 1) & 0x55555555u);
            v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
            count += (((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
        }
        sums[y] = count;
    }
}

// Score: sum of squared differences between adjacent row sums
static double diff_square_sum(const int * sums, int h) {
    int skip = std::max(1, h / 20);  // skip 5% at top/bottom
    double score = 0;
    for (int i = skip; i < h - skip; i++) {
        double d = (double)(sums[i] - sums[i - 1]);
        score += d * d;
    }
    return score;
}

// Reduce 1-bit image by 2x (every 2x2 block → 1 if any set)
static uint32_t * reduce_2x(const uint32_t * src, int w, int h, int wpl,
                              int * out_w, int * out_h, int * out_wpl) {
    int rw = w / 2, rh = h / 2;
    int rwpl = (rw + 31) / 32;
    *out_w = rw; *out_h = rh; *out_wpl = rwpl;
    uint32_t * dst = (uint32_t *)calloc(rwpl * rh, sizeof(uint32_t));
    if (!dst) return nullptr;
    for (int y = 0; y < rh; y++) {
        const uint32_t * r0 = src + (y * 2) * wpl;
        const uint32_t * r1 = src + (y * 2 + 1) * wpl;
        uint32_t * dline = dst + y * rwpl;
        for (int x = 0; x < rw; x++) {
            int sx = x * 2;
            // Check any of the 4 pixels in the 2x2 block
            int bit = 0;
            auto get = [](const uint32_t *l, int px) {
                return (l[px >> 5] >> (31 - (px & 31))) & 1;
            };
            if (get(r0, sx) || get(r0, sx+1) || get(r1, sx) || get(r1, sx+1))
                dline[x >> 5] |= (1u << (31 - (x & 31)));
        }
    }
    return dst;
}

int find_skew_angle(const uint8_t * gray, int w, int h,
                    float * out_angle, float * out_conf) {
    if (out_angle) *out_angle = 0;
    if (out_conf) *out_conf = 0;
    if (!gray || w < 100 || h < 50) return 1;

    // Binarize
    int wpl = 0;
    uint32_t * bits = morph_u8_to_1bit(gray, w, h, 0, &wpl);
    if (!bits) return 1;
    // Auto-threshold: use otsu
    uint8_t thresh = 128; // simple default
    {
        int hist[256] = {};
        for (int i = 0; i < w*h; i++) hist[gray[i]]++;
        double sum = 0;
        for (int i = 0; i < 256; i++) sum += (double)i * hist[i];
        double sumB = 0; int wB = 0; double maxv = 0; int best = 128;
        for (int t = 0; t < 256; t++) {
            wB += hist[t]; if (!wB) continue;
            int wF = w*h - wB; if (!wF) break;
            sumB += (double)t*hist[t];
            double d = sumB/wB - (sum-sumB)/wF;
            double v = (double)wB*wF*d*d;
            if (v > maxv) { maxv = v; best = t; }
        }
        thresh = (uint8_t)best;
    }
    if (thresh < 255) thresh++;  // include pixels AT the threshold
    morph_free(bits);
    bits = morph_u8_to_1bit(gray, w, h, thresh, &wpl);
    if (!bits) return 1;

    // Reduce 4x for speed
    int rw, rh, rwpl;
    uint32_t * r1 = reduce_2x(bits, w, h, wpl, &rw, &rh, &rwpl);
    morph_free(bits);
    if (!r1) return 1;
    int rw2, rh2, rwpl2;
    uint32_t * reduced = reduce_2x(r1, rw, rh, rwpl, &rw2, &rh2, &rwpl2);
    free(r1);
    if (!reduced) return 1;

    // Sweep: -7° to +7° in 1° steps
    const float sweep_range = 7.0f;
    const float sweep_delta = 1.0f;
    const float deg2rad = 3.14159265f / 180.0f;
    int n_angles = (int)(2 * sweep_range / sweep_delta) + 1;

    std::vector<float> angles(n_angles);
    std::vector<double> scores(n_angles);
    std::vector<int> sums(rh2);
    uint32_t * sheared = (uint32_t *)calloc(rwpl2 * rh2, sizeof(uint32_t));
    if (!sheared) { free(reduced); return 1; }

    double max_score = 0, min_score = 1e30;
    int max_idx = 0;

    for (int i = 0; i < n_angles; i++) {
        angles[i] = -sweep_range + i * sweep_delta;
        vshear_1bit(reduced, sheared, rw2, rh2, rwpl2, angles[i] * deg2rad);
        row_sums(sheared, rw2, rh2, rwpl2, sums.data());
        scores[i] = diff_square_sum(sums.data(), rh2);
        if (scores[i] > max_score) { max_score = scores[i]; max_idx = i; }
        if (scores[i] < min_score) min_score = scores[i];
    }

    // Binary search refinement around the peak
    float lo = max_idx > 0 ? angles[max_idx - 1] : angles[0];
    float hi = max_idx < n_angles - 1 ? angles[max_idx + 1] : angles[n_angles - 1];
    float best_angle = angles[max_idx];
    double best_score = max_score;

    for (int iter = 0; iter < 12; iter++) {
        float mid_lo = (lo + best_angle) / 2;
        float mid_hi = (best_angle + hi) / 2;

        vshear_1bit(reduced, sheared, rw2, rh2, rwpl2, mid_lo * deg2rad);
        row_sums(sheared, rw2, rh2, rwpl2, sums.data());
        double score_lo = diff_square_sum(sums.data(), rh2);

        vshear_1bit(reduced, sheared, rw2, rh2, rwpl2, mid_hi * deg2rad);
        row_sums(sheared, rw2, rh2, rwpl2, sums.data());
        double score_hi = diff_square_sum(sums.data(), rh2);

        if (score_lo > best_score) {
            hi = best_angle; best_angle = mid_lo; best_score = score_lo;
        } else if (score_hi > best_score) {
            lo = best_angle; best_angle = mid_hi; best_score = score_hi;
        } else {
            lo = mid_lo; hi = mid_hi;
        }
    }

    free(sheared);
    free(reduced);

    if (out_angle) *out_angle = best_angle;
    if (out_conf) *out_conf = (min_score > 0) ? (float)(best_score / min_score) : 0;
    return 0;
}

// =========================================================================
// 3. CC-based despeckle
// =========================================================================
// Remove connected components smaller than max_w × max_h.
// Uses the CC labeling from cc_detect.cpp pattern but simpler:
// flood-fill each component, check size, zero out if small.

static inline int get_bit(const uint32_t * line, int x) {
    return (line[x >> 5] >> (31 - (x & 31))) & 1;
}
static inline void clear_bit(uint32_t * line, int x) {
    line[x >> 5] &= ~(1u << (31 - (x & 31)));
}
static inline void set_bit(uint32_t * line, int x) {
    line[x >> 5] |= (1u << (31 - (x & 31)));
}

uint32_t * despeckle_cc(const uint32_t * bits, int w, int h, int wpl,
                         int max_w, int max_h) {
    int n = wpl * h;
    uint32_t * out = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!out) return nullptr;
    memcpy(out, bits, n * sizeof(uint32_t));

    // Visited mask
    std::vector<uint8_t> visited(w * h, 0);

    // Stack for flood fill
    struct Pt { int x, y; };
    std::vector<Pt> stack;
    std::vector<Pt> component;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (visited[y * w + x]) continue;
            if (!get_bit(out + y * wpl, x)) continue;

            // Flood fill this component
            stack.clear();
            component.clear();
            stack.push_back({x, y});
            visited[y * w + x] = 1;
            int min_x = x, max_x = x, min_y = y, max_y = y;

            while (!stack.empty()) {
                Pt p = stack.back(); stack.pop_back();
                component.push_back(p);
                if (p.x < min_x) min_x = p.x;
                if (p.x > max_x) max_x = p.x;
                if (p.y < min_y) min_y = p.y;
                if (p.y > max_y) max_y = p.y;

                // 4-connected neighbors
                const int dx[] = {-1, 1, 0, 0};
                const int dy[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; d++) {
                    int nx = p.x + dx[d], ny = p.y + dy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    if (visited[ny * w + nx]) continue;
                    if (!get_bit(out + ny * wpl, nx)) continue;
                    visited[ny * w + nx] = 1;
                    stack.push_back({nx, ny});
                }
            }

            // Check if component is a speckle
            int cw = max_x - min_x + 1;
            int ch = max_y - min_y + 1;
            if (cw < max_w && ch < max_h) {
                // Remove it
                for (auto & p : component)
                    clear_bit(out + p.y * wpl, p.x);
            }
        }
    }

    return out;
}

void despeckle_gray(const uint8_t * gray, int w, int h,
                    int max_w, int max_h, uint8_t * out) {
    // Binarize: Otsu returns the optimal threshold; use thresh+1 so that
    // pixels AT the threshold value are treated as foreground (dark).
    uint8_t thresh = otsu_single(gray, w * h);
    if (thresh < 255) thresh++;
    int wpl = 0;
    uint32_t * bits = morph_u8_to_1bit(gray, w, h, thresh, &wpl);
    if (!bits) { memcpy(out, gray, w * h); return; }

    // Despeckle
    uint32_t * cleaned = despeckle_cc(bits, w, h, wpl, max_w, max_h);
    morph_free(bits);
    if (!cleaned) { memcpy(out, gray, w * h); return; }

    // Convert back: foreground (1) → 0, background (0) → 255
    for (int y = 0; y < h; y++) {
        const uint32_t * line = cleaned + y * wpl;
        for (int x = 0; x < w; x++)
            out[y * w + x] = get_bit(line, x) ? 0 : 255;
    }
    morph_free(cleaned);
}

// =========================================================================
// 4. Background normalization
// =========================================================================
// Algorithm from Leptonica's pixBackgroundNormSimple:
//   - Sample image at tile centers
//   - For each tile, estimate background as the high percentile of pixel values
//   - Smooth the background map
//   - Normalize: out = gray * target / background

void background_norm(const uint8_t * gray, int w, int h,
                     int tile_w, int tile_h, uint8_t * out) {
    if (tile_w <= 0) tile_w = std::max(16, w / 20);
    if (tile_h <= 0) tile_h = std::max(16, h / 20);

    int tw = (w + tile_w - 1) / tile_w;
    int th = (h + tile_h - 1) / tile_h;
    const uint8_t target = 200;  // target background level

    // Estimate background per tile (90th percentile)
    std::vector<float> bg_map(tw * th);
    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            int x0 = tx * tile_w, y0 = ty * tile_h;
            int x1 = std::min(x0 + tile_w, w);
            int y1 = std::min(y0 + tile_h, h);

            // Histogram for this tile
            int hist[256] = {};
            int cnt = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) {
                    hist[gray[y * w + x]]++;
                    cnt++;
                }

            // 90th percentile
            int acc = 0;
            int p90 = 255;
            int thresh90 = (int)(cnt * 0.9f);
            for (int v = 0; v < 256; v++) {
                acc += hist[v];
                if (acc >= thresh90) { p90 = v; break; }
            }
            bg_map[ty * tw + tx] = (float)std::max(1, p90);
        }
    }

    // Smooth background map (3x3 box)
    std::vector<float> bg_smooth(tw * th);
    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {
            float sum = 0; int cnt = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int sy = ty + dy, sx = tx + dx;
                    if (sy >= 0 && sy < th && sx >= 0 && sx < tw) {
                        sum += bg_map[sy * tw + sx]; cnt++;
                    }
                }
            }
            bg_smooth[ty * tw + tx] = sum / cnt;
        }
    }

    // Interpolate and normalize
    for (int y = 0; y < h; y++) {
        float fy = (float)y / tile_h - 0.5f;
        int ty0 = std::max(0, (int)fy);
        int ty1 = std::min(th - 1, ty0 + 1);
        float wy = std::max(0.0f, std::min(1.0f, fy - ty0));

        for (int x = 0; x < w; x++) {
            float fx = (float)x / tile_w - 0.5f;
            int tx0 = std::max(0, (int)fx);
            int tx1 = std::min(tw - 1, tx0 + 1);
            float wx = std::max(0.0f, std::min(1.0f, fx - tx0));

            float bg = bg_smooth[ty0*tw+tx0] * (1-wx)*(1-wy)
                      + bg_smooth[ty0*tw+tx1] * wx*(1-wy)
                      + bg_smooth[ty1*tw+tx0] * (1-wx)*wy
                      + bg_smooth[ty1*tw+tx1] * wx*wy;

            float v = (float)gray[y * w + x] * target / bg;
            out[y * w + x] = (uint8_t)std::max(0.0f, std::min(255.0f, v));
        }
    }
}

// =========================================================================
// 5. Image downsampling calculator
// =========================================================================

float compute_downsample_factor(int w, int h, int current_dpi,
                                 int target_dpi, int max_pixels) {
    float factor = 1.0f;

    // DPI-based downsampling
    if (current_dpi > 0 && target_dpi > 0 && current_dpi > target_dpi) {
        factor = (float)target_dpi / current_dpi;
    }

    // Pixel-count limit
    if (max_pixels > 0) {
        int pixels = w * h;
        if (pixels > max_pixels) {
            float px_factor = sqrtf((float)max_pixels / pixels);
            if (px_factor < factor) factor = px_factor;
        }
    }

    // Clamp to (0, 1]
    if (factor > 1.0f) factor = 1.0f;
    if (factor < 0.01f) factor = 0.01f;
    return factor;
}

// =========================================================================
// 6. OCR quality scoring
// =========================================================================

float ocr_quality_score(const char * text,
                         const char ** dict, int n_dict) {
    if (!text || !dict || n_dict <= 0) return 0.0f;

    int total_words = 0;
    int matched = 0;

    // Split text into words (space/newline/tab separated)
    const char * p = text;
    while (*p) {
        // Skip whitespace
        while (*p && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
        if (!*p) break;

        // Extract word
        const char * start = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        int len = (int)(p - start);
        if (len < 2) continue; // skip single chars

        total_words++;

        // Binary search in sorted dictionary
        std::string word(start, len);
        // Lowercase for comparison
        for (auto & c : word) c = (char)tolower((unsigned char)c);

        int lo = 0, hi = n_dict - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            int cmp = word.compare(dict[mid]);
            if (cmp == 0) { matched++; break; }
            if (cmp < 0) hi = mid - 1;
            else lo = mid + 1;
        }
    }

    return total_words > 0 ? (float)matched / total_words : 0.0f;
}

// =========================================================================
// 7. Text angle classification (0° vs 180°)
// =========================================================================
// Two heuristics combined:
//
// 1. Ascender/descender asymmetry: in correctly-oriented Latin text,
//    text lines have more ink mass in the upper half (ascenders: b,d,f,h,k,l,t)
//    than the lower half (descenders: g,j,p,q,y). Flip 180° and this reverses.
//
// 2. Top-heavy vs bottom-heavy: correctly-oriented text on a page typically
//    has more content near the top (headers, titles) than the bottom (margins).
//    This is weaker but helps for non-Latin scripts.

int detect_text_angle(const uint8_t * gray, int w, int h,
                       float * confidence) {
    if (!gray || w < 20 || h < 20) {
        if (confidence) *confidence = 0;
        return 0;
    }

    // Binarize (Otsu)
    int hist[256] = {};
    for (int i = 0; i < w*h; i++) hist[gray[i]]++;
    double sum = 0;
    for (int i = 0; i < 256; i++) sum += (double)i * hist[i];
    double sumB = 0; int wB = 0; double maxv = 0; int best = 128;
    for (int t = 0; t < 256; t++) {
        wB += hist[t]; if (!wB) continue; int wF = w*h - wB; if (!wF) break;
        sumB += (double)t*hist[t]; double d = sumB/wB - (sum-sumB)/wF;
        double v = (double)wB*wF*d*d; if (v > maxv) { maxv = v; best = t; }
    }
    uint8_t thresh = (uint8_t)(best < 255 ? best + 1 : best);

    // Count dark pixels per row
    std::vector<int> row_dark(h, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (gray[y * w + x] < thresh) row_dark[y]++;

    // Find text line regions (rows with significant dark pixel count)
    int min_dark = w / 20; // at least 5% of row width
    std::vector<std::pair<int,int>> text_bands; // (start_y, end_y)
    int band_start = -1;
    for (int y = 0; y < h; y++) {
        if (row_dark[y] >= min_dark) {
            if (band_start < 0) band_start = y;
        } else {
            if (band_start >= 0) {
                text_bands.push_back({band_start, y - 1});
                band_start = -1;
            }
        }
    }
    if (band_start >= 0) text_bands.push_back({band_start, h - 1});

    if (text_bands.empty()) {
        if (confidence) *confidence = 0;
        return 0;
    }

    // For each text band, compute upper/lower dark pixel ratio
    // Upper half should have more ink (ascenders) in correctly-oriented text
    double score_normal = 0, score_flipped = 0;
    for (auto & [y0, y1] : text_bands) {
        int band_h = y1 - y0 + 1;
        if (band_h < 4) continue;
        int mid = y0 + band_h / 2;
        int upper = 0, lower = 0;
        for (int y = y0; y < mid; y++) upper += row_dark[y];
        for (int y = mid; y <= y1; y++) lower += row_dark[y];
        // Correctly oriented: upper > lower (ascenders)
        score_normal += (double)upper;
        score_flipped += (double)lower;
    }

    // Also consider page-level asymmetry: more content near top = normal
    int top_third = 0, bot_third = 0;
    int third = h / 3;
    for (int y = 0; y < third; y++) top_third += row_dark[y];
    for (int y = h - third; y < h; y++) bot_third += row_dark[y];
    // Weight page-level asymmetry less than per-line asymmetry
    score_normal += top_third * 0.3;
    score_flipped += bot_third * 0.3;

    double total = score_normal + score_flipped;
    if (total < 1) {
        if (confidence) *confidence = 0;
        return 0;
    }

    float conf = (float)std::abs(score_normal - score_flipped) / (float)total;
    if (confidence) *confidence = conf;

    return score_normal >= score_flipped ? 0 : 180;
}
