// lfm2_vl_tiling.h — LFM2.5-VL multi-tile NaFlex layout, weight-free.
//
// The whole of the image-splitting decision: whether a page is tiled, into what
// grid, at what size, how big the thumbnail is, how many image tokens each
// piece contributes, and the exact special-token markup that wraps them.
//
// It lives in its own header, with no ggml and no model, for the reason
// dev-guide HARD RULE 3b gives: everything here is downstream of the tensors
// and therefore INVISIBLE to the per-stage diff harness. A wrong grid or a
// wrong token id produces perfectly healthy-looking activations from the wrong
// prompt. Extracting it makes it hermetically testable — see
// `tests/test_lfm2_tiling.cpp`, whose golden vectors come from transformers'
// own functions via `tools/lfm2_vl_tiling_oracle.py`.
//
// Blueprint: transformers 4.57.6
//   models/lfm2_vl/image_processing_lfm2_vl_fast.py
//     ::round_by_factor, ::find_closest_aspect_ratio,
//     Lfm2VlImageProcessorFast::{_target_ratios, _get_grid_layout,
//                                smart_resize, _is_image_too_large,
//                                crop_image_to_patches, resize_and_split}
//   models/lfm2_vl/processing_lfm2_vl.py
//     Lfm2VlProcessor::{_get_image_num_tokens, expand_text_with_placeholders}
//
// Defaults are the shipped `processor_config.json` of LiquidAI/LFM2.5-VL-3B,
// which is NOT the same as the Lfm2VlImageProcessorFast class defaults: the
// class says min_tiles=2 and BILINEAR, the config says min_tiles=1 and
// resample=3 (bicubic). The config wins at inference.

#pragma once

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lfm2_vl_tiling {

// ── processor_config.json ──────────────────────────────────────────────────
struct config {
    int   patch_size           = 16;    // encoder_patch_size
    int   downsample_factor    = 2;     // projector pixel_unshuffle factor
    int   tile_size            = 512;
    int   min_tiles            = 1;
    int   max_tiles            = 10;
    int   min_image_tokens     = 64;
    int   max_image_tokens     = 256;
    float max_pixels_tolerance = 2.0f;
    bool  use_thumbnail        = true;
    bool  do_image_splitting   = true;

    // OFF = reproduce upstream exactly, including its row/col swap (below).
    // ON = label tiles by their actual geometry. Opt-in, unvalidated; see
    // LFM2_VL_TILE_LABELS_GEOMETRIC in docs/lfm2_vl/PLAN.md.
    bool geometric_labels = false;
};

// ── Python's round(), which is BANKER'S rounding (half to EVEN) ────────────
//
// C++ std::round is half-AWAY-FROM-ZERO. The two disagree whenever the value
// lands exactly on k + 0.5 for even k, and the consequences are not subtle: a
// 144x4000 page is one tile and 252 image tokens under Python's rule, and a
// 1x10 split and 2812 image tokens under std::round.
//
// Implemented explicitly rather than via std::nearbyint so it does not depend
// on the process's floating-point rounding mode.
inline int round_half_to_even(double v) {
    const double f  = std::floor(v);
    const double d  = v - f;
    const long long lo = (long long) f;
    if (d > 0.5) return (int) (lo + 1);
    if (d < 0.5) return (int) lo;
    return (int) ((lo % 2 == 0) ? lo : lo + 1);  // exact .5 → the even neighbour
}

// round_by_factor(number, factor) — in exact integer arithmetic, so no
// floating-point representation question can arise. Non-negative values only,
// which is all an image dimension can be.
inline int round_by_factor(int value, int factor) {
    if (factor <= 0) return value;
    const int q = value / factor;
    const int r = value - q * factor;
    if (2 * r > factor) return (q + 1) * factor;
    if (2 * r < factor) return q * factor;
    return ((q % 2 == 0) ? q : q + 1) * factor;  // exact half → even quotient
}

// ── _target_ratios(min_tiles, max_tiles) ───────────────────────────────────
//
// Upstream builds this as `sorted(set(ratios), key=lambda x: x[0] * x[1])`.
// Python's sort is stable, so within one area the order is whatever the SET
// happened to iterate in — an artifact of CPython's tuple hashing, not a rule
// derivable from the comprehension. It is nonetheless deterministic (integer
// hashes are not randomised by PYTHONHASHSEED), and it MATTERS, because
// find_closest_aspect_ratio's tie-break replaces the incumbent, so the last
// admissible candidate in this order wins a tie.
//
// Hence: transcribed from the oracle, not recomputed. The table is pinned in
// tests/test_lfm2_tiling.cpp against the golden header.
struct ratio { int w, h; };

inline const ratio * default_target_ratios(int * n_out) {
    // _target_ratios(min_tiles=1, max_tiles=10)
    static const ratio kRatios[] = {
        {1, 1},  {1, 2}, {2, 1}, {3, 1}, {1, 3}, {2, 2}, {4, 1}, {1, 4}, {5, 1},
        {1, 5},  {1, 6}, {6, 1}, {3, 2}, {2, 3}, {7, 1}, {1, 7}, {4, 2}, {2, 4},
        {1, 8},  {8, 1}, {1, 9}, {3, 3}, {9, 1}, {2, 5}, {5, 2}, {10, 1}, {1, 10},
    };
    if (n_out) *n_out = (int) (sizeof(kRatios) / sizeof(kRatios[0]));
    return kRatios;
}

// ── find_closest_aspect_ratio ──────────────────────────────────────────────
inline ratio find_closest_aspect_ratio(double aspect_ratio, const ratio * ratios, int n_ratios, int width, int height,
                                       int image_size) {
    double best_diff = 1e300;
    ratio  best      = {1, 1};
    const double area = (double) width * (double) height;

    for (int i = 0; i < n_ratios; i++) {
        const double target = (double) ratios[i].w / (double) ratios[i].h;
        const double diff   = std::fabs(aspect_ratio - target);
        if (diff < best_diff) {
            best_diff = diff;
            best      = ratios[i];
        } else if (diff == best_diff) {
            // Equally close: prefer the ratio that better matches the area.
            // Note this REPLACES the incumbent, so later-in-order wins.
            const double target_area = (double) image_size * (double) image_size * ratios[i].w * ratios[i].h;
            if (area > 0.5 * target_area) best = ratios[i];
        }
    }
    return best;
}

// ── _get_grid_layout ───────────────────────────────────────────────────────
inline void grid_layout(int height, int width, const config & cfg, int * grid_w, int * grid_h, int * target_w,
                        int * target_h) {
    int n = 0;
    const ratio * ratios = default_target_ratios(&n);
    const ratio  g = find_closest_aspect_ratio((double) width / (double) height, ratios, n, width, height, cfg.tile_size);
    if (grid_w) *grid_w = g.w;
    if (grid_h) *grid_h = g.h;
    if (target_w) *target_w = cfg.tile_size * g.w;
    if (target_h) *target_h = cfg.tile_size * g.h;
}

// ── smart_resize ───────────────────────────────────────────────────────────
//
// Same algorithm as image_preproc::smart_resize, but with Python's rounding
// and this model's parameters. The shared helper uses std::round and is left
// alone: other engines' parity is measured against their own references.
inline void smart_resize(int height, int width, const config & cfg, int * out_w, int * out_h) {
    const int total_factor = cfg.patch_size * cfg.downsample_factor;
    const double min_px = (double) cfg.min_image_tokens * cfg.patch_size * cfg.patch_size * cfg.downsample_factor *
                          cfg.downsample_factor;
    const double max_px = (double) cfg.max_image_tokens * cfg.patch_size * cfg.patch_size * cfg.downsample_factor *
                          cfg.downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    if ((double) h_bar * (double) w_bar > max_px) {
        const double beta = std::sqrt(((double) height * (double) width) / max_px);
        h_bar = std::max(total_factor, (int) std::floor((double) height / beta / total_factor) * total_factor);
        w_bar = std::max(total_factor, (int) std::floor((double) width / beta / total_factor) * total_factor);
    } else if ((double) h_bar * (double) w_bar < min_px) {
        const double beta = std::sqrt(min_px / ((double) height * (double) width));
        h_bar = (int) std::ceil((double) height * beta / total_factor) * total_factor;
        w_bar = (int) std::ceil((double) width * beta / total_factor) * total_factor;
    }

    if (out_h) *out_h = h_bar;
    if (out_w) *out_w = w_bar;
}

// ── _is_image_too_large ────────────────────────────────────────────────────
//
// ⚠ The floor here is `patch_size` (16), NOT `patch_size * downsample_factor`
// (32) as in smart_resize. Upstream really does use two different floors; do
// not "harmonise" them.
inline bool is_image_too_large(int height, int width, const config & cfg) {
    const int total_factor = cfg.patch_size * cfg.downsample_factor;
    const int h_bar = std::max(cfg.patch_size, round_by_factor(height, total_factor));
    const int w_bar = std::max(cfg.patch_size, round_by_factor(width, total_factor));
    const double bound = (double) cfg.max_image_tokens * cfg.patch_size * cfg.patch_size * cfg.downsample_factor *
                         cfg.downsample_factor * (double) cfg.max_pixels_tolerance;
    return (double) h_bar * (double) w_bar > bound;
}

// ── _get_image_num_tokens ──────────────────────────────────────────────────
//
// ⚠ ceil, not floor. The processor rounds the patch grid UP here while the
// projector's pixel_unshuffle integer-divides it DOWN. With smart_resize's
// `patch_size * downsample_factor` factor the grid is always even and the two
// agree; the test pins that they still do.
inline int tokens_for_image(int image_h, int image_w, const config & cfg) {
    const int ph = image_h / cfg.patch_size;
    const int pw = image_w / cfg.patch_size;
    const int dh = (ph + cfg.downsample_factor - 1) / cfg.downsample_factor;
    const int dw = (pw + cfg.downsample_factor - 1) / cfg.downsample_factor;
    return dh * dw;
}

inline int tokens_per_tile(const config & cfg) {
    const int pt = cfg.tile_size / cfg.patch_size;
    const int dt = (pt + cfg.downsample_factor - 1) / cfg.downsample_factor;
    return dt * dt;
}

// ── the composite ──────────────────────────────────────────────────────────
struct layout {
    bool split      = false;
    int  grid_w     = 1;  // geometric: tiles across
    int  grid_h     = 1;  // geometric: tiles down
    int  rows       = 1;  // LABEL space — see the swap note below
    int  cols       = 1;
    int  n_tiles    = 1;
    int  n_images   = 1;  // tiles + thumbnail; = number of vision-encoder runs
    bool has_thumb  = false;
    int  target_w   = 0;  // whole image is resized to this before being cut up
    int  target_h   = 0;
    int  resized_w  = 0;  // smart_resize output = the thumbnail's size
    int  resized_h  = 0;
    int  tile_tokens  = 0;
    int  thumb_tokens = 0;
    int  total_tokens = 0;
};

// Mirrors resize_and_split() followed by expand_text_with_placeholders().
//
// ⚠ THE ROW/COL SWAP. resize_and_split unpacks
//
//     images, num_rows, num_cols = self.crop_image_to_patches(...)
//
// and crop_image_to_patches returns `(processed_images, grid_width,
// grid_height)`. So num_rows is the grid WIDTH and num_cols is the grid
// HEIGHT, and the <|img_row_R_col_C|> labels the processor emits are
// transposed relative to the pixel tiles on any non-square grid. A portrait A4
// is cut into 3 rows of 2 tiles and labelled as 2 rows of 3.
//
// That is what the deployed model is prompted with, so it is the parity target
// and the default here. `geometric_labels` restores the intuitive mapping and
// is opt-in and unvalidated.
inline layout compute_layout(int width, int height, const config & cfg) {
    layout L;

    // Upstream recomputes this locally, shadowing the argument.
    const bool do_split = !(cfg.min_tiles == 1 && cfg.max_tiles == 1);

    smart_resize(height, width, cfg, &L.resized_w, &L.resized_h);

    L.split = do_split && cfg.do_image_splitting && is_image_too_large(height, width, cfg);

    if (L.split) {
        grid_layout(height, width, cfg, &L.grid_w, &L.grid_h, &L.target_w, &L.target_h);
        L.n_tiles = L.grid_w * L.grid_h;
        if (cfg.geometric_labels) {
            L.rows = L.grid_h;
            L.cols = L.grid_w;
        } else {
            L.rows = L.grid_w;  // the swap
            L.cols = L.grid_h;
        }
    } else {
        L.grid_w = L.grid_h = 1;
        L.rows = L.cols = 1;
        L.n_tiles  = 1;
        L.target_w = L.resized_w;
        L.target_h = L.resized_h;
    }

    L.thumb_tokens = cfg.use_thumbnail ? tokens_for_image(L.resized_h, L.resized_w, cfg) : 0;
    L.tile_tokens  = tokens_per_tile(cfg);

    if (L.rows > 1 || L.cols > 1) {
        L.has_thumb    = L.thumb_tokens > 0;
        L.n_images     = L.n_tiles + (L.has_thumb ? 1 : 0);
        L.total_tokens = L.n_tiles * L.tile_tokens + L.thumb_tokens;
    } else {
        // Single tile: a bare run of <image>, length = the THUMBNAIL count.
        // (Upstream really does use the thumbnail count here; with
        // use_thumbnail=false it would emit zero image tokens. The shipped
        // config sets it true, so that branch is unreachable in practice.)
        L.has_thumb    = false;
        L.n_images     = 1;
        L.total_tokens = L.thumb_tokens;
    }
    return L;
}

// ── token markup ───────────────────────────────────────────────────────────
//
// Verified against the shipped GGUF vocab of LiquidAI/LFM2.5-VL-3B: all 100
// <|img_row_R_col_C|> ids are contiguous from 124908 with 0 mismatches, and
// <|img_thumbnail|> is the next id after them. No converter work is needed.
struct token_ids {
    int32_t image        = 124907;  // <image>
    int32_t row_col_base = 124908;  // <|img_row_1_col_1|>
    int32_t thumbnail    = 125008;  // <|img_thumbnail|>
    int32_t image_start  = 125009;  // <|image_start|>
    int32_t image_end    = 125010;  // <|image_end|>

    int32_t row_col(int row, int col) const {  // 1-based, as the token names are
        return row_col_base + (int32_t) ((row - 1) * 10 + (col - 1));
    }
};

// Emit <|image_start|> … <|image_end|> inclusive.
//
// The i-th <image> placeholder is consumed by the i-th row of the concatenated
// projector output, so this order and the order tiles are encoded in must be
// the same: tiles in pixel reading order, thumbnail last.
inline void build_image_markup(const layout & L, const token_ids & tok, std::vector<int32_t> & out) {
    out.clear();
    out.reserve((size_t) L.total_tokens + (size_t) L.n_tiles + 4);
    out.push_back(tok.image_start);

    if (L.rows > 1 || L.cols > 1) {
        for (int row = 1; row <= L.rows; row++) {
            for (int col = 1; col <= L.cols; col++) {
                out.push_back(tok.row_col(row, col));
                for (int t = 0; t < L.tile_tokens; t++) out.push_back(tok.image);
            }
        }
        if (L.thumb_tokens > 0) {
            out.push_back(tok.thumbnail);
            for (int t = 0; t < L.thumb_tokens; t++) out.push_back(tok.image);
        }
    } else {
        for (int t = 0; t < L.total_tokens; t++) out.push_back(tok.image);
    }

    out.push_back(tok.image_end);
}

}  // namespace lfm2_vl_tiling
