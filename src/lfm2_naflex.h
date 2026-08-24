// lfm2_naflex.h — LFM2.5-VL NaFlex tiling math, weight-free.
//
// Extracted out of src/lfm2_vl_ocr.cpp so tests/test_lfm2_naflex.cpp can pin it
// against tools/lfm2_vl_tiling_oracle.py, which is HF's own
// image_processing_lfm2_vl.py functions extracted verbatim. Nothing here loads
// a model or touches ggml — the whole point is that the guard runs in
// milliseconds and can be written before the code it guards (dev guide HARD
// RULE 2c).
//
// Blueprint: transformers/models/lfm2_vl/image_processing_lfm2_vl.py
//   _is_image_too_large / _get_grid_layout / find_closest_aspect_ratio /
//   smart_resize.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lfm2_naflex {

// Processor defaults (processor_config.json / config.json for LFM2.5-VL-3B).
struct params {
    int encoder_patch_size = 16;
    int downsample_factor = 2;
    int tile_size = 512;
    int min_tiles = 1;
    int max_tiles = 10;
    int min_image_tokens = 64;
    int max_image_tokens = 256;
    float max_pixels_tolerance = 2.0f;
};

// Python's round() is banker's rounding (half to EVEN); std::round is not.
// round_by_factor runs on raw pixel counts, so a .5 lands often enough to
// matter — this is the single most likely silent divergence in the tiling math.
// std::nearbyint under the default FE_TONEAREST is half-to-even.
inline int round_by_factor(int number, int factor) {
    return (int)std::nearbyint((double)number / (double)factor) * factor;
}

// _is_image_too_large: does this page get split at all?
// NOTE the asymmetry with smart_resize, which is in the blueprint and is not a
// typo: the floor here is `encoder_patch_size`, not `encoder_patch_size *
// downsample_factor`.
inline bool is_image_too_large(int height, int width, const params & p) {
    const int P = p.encoder_patch_size, ds = p.downsample_factor;
    const int tf = P * ds;
    const double h_bar = std::max(P, round_by_factor(height, tf));
    const double w_bar = std::max(P, round_by_factor(width, tf));
    return h_bar * w_bar > (double)p.max_image_tokens * P * P * ds * ds * p.max_pixels_tolerance;
}

// target_ratios(min_tiles, max_tiles), ordered by tile count the way the
// blueprint's `sorted(set(...), key=area)` orders it.
inline std::vector<std::pair<int, int>> target_ratios(const params & p) {
    std::vector<std::pair<int, int>> r;
    for (int n = p.min_tiles; n <= p.max_tiles; n++)
        for (int w = 1; w <= n; w++)
            for (int h = 1; h <= n; h++)
                if (w * h >= p.min_tiles && w * h <= p.max_tiles) r.push_back({ w, h });
    std::sort(r.begin(), r.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) {
        if (a.first * a.second != b.first * b.second) return a.first * a.second < b.first * b.second;
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });
    r.erase(std::unique(r.begin(), r.end()), r.end());
    return r;
}

// find_closest_aspect_ratio, verbatim: strict `<` keeps the FIRST best in area
// order, and an exact tie only moves when the image covers more than half the
// candidate grid's area. That tie rule is what sends a 2048x2048 page to a 3x3
// grid and a 1000x1000 page to 2x2 — dropping it silently halves the tokens on
// large square scans.
inline void grid_layout(int height, int width, const params & p, int * out_grid_w, int * out_grid_h) {
    const double ar = (double)width / (double)height;
    const double area = (double)width * (double)height;
    double best_diff = std::numeric_limits<double>::infinity();
    int bw = 1, bh = 1;
    for (const auto & r : target_ratios(p)) {
        const double tar = (double)r.first / (double)r.second;
        const double diff = std::fabs(ar - tar);
        if (diff < best_diff) {
            best_diff = diff;
            bw = r.first;
            bh = r.second;
        } else if (diff == best_diff) {
            const double target_area = (double)p.tile_size * p.tile_size * r.first * r.second;
            if (area > 0.5 * target_area) {
                bw = r.first;
                bh = r.second;
            }
        }
    }
    *out_grid_w = bw;
    *out_grid_h = bh;
}

// smart_resize: round both sides to encoder_patch_size * downsample_factor and
// pull the pixel count into the [min_image_tokens, max_image_tokens] band.
// Shares its shape with image_preproc::smart_resize but is written out here so
// the guard covers the exact parameters this backend feeds it.
inline void smart_resize(int height, int width, const params & p, int * out_w, int * out_h) {
    const int P = p.encoder_patch_size, ds = p.downsample_factor;
    const int tf = P * ds;
    const double min_px = (double)p.min_image_tokens * P * P * ds * ds;
    const double max_px = (double)p.max_image_tokens * P * P * ds * ds;
    double h_bar = std::max(tf, round_by_factor(height, tf));
    double w_bar = std::max(tf, round_by_factor(width, tf));
    if (h_bar * w_bar > max_px) {
        const double beta = std::sqrt(((double)height * width) / max_px);
        h_bar = std::max((double)tf, std::floor(height / beta / tf) * tf);
        w_bar = std::max((double)tf, std::floor(width / beta / tf) * tf);
    } else if (h_bar * w_bar < min_px) {
        const double beta = std::sqrt(min_px / ((double)height * width));
        h_bar = std::ceil(height * beta / tf) * tf;
        w_bar = std::ceil(width * beta / tf) * tf;
    }
    *out_w = (int)w_bar;
    *out_h = (int)h_bar;
}

// Lfm2VlProcessor._compute_tokens_per_tile / _compute_tokens_for_image.
inline int tokens_per_tile(const params & p) {
    const int n = p.tile_size / p.encoder_patch_size;
    const int d = (n + p.downsample_factor - 1) / p.downsample_factor;
    return d * d;
}

inline int tokens_for_image(int resized_h, int resized_w, const params & p) {
    const int ds = p.downsample_factor;
    const int ph = ((resized_h / p.encoder_patch_size) + ds - 1) / ds;
    const int pw = ((resized_w / p.encoder_patch_size) + ds - 1) / ds;
    return ph * pw;
}

} // namespace lfm2_naflex
