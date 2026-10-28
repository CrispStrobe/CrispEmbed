// minicpm_ocr_tiles.h - header-only slicing/grid/token arithmetic for
// openbmb/MiniCPM-OCR-1.0 (MiniCPMOcrImageProcessor, preprocessor_config.json:
// scale_resolution=448, max_slice_nums=9, patch_size=14, downsample_mode="4x",
// small_image_upsize="always"). See docs/minicpm-ocr-inference-spec.md,
// "Preprocessing" steps 1-5 and "Token layout".
//
// Pure integer/float math, no ggml: guarded by tests/test_minicpm_ocr_tiles.cpp.
// Items marked TODO(verify) are on the spec's "verify" list and must be
// confirmed against the hosted reference capture (input_ids, pixel_values,
// target sizes) before the native path is called parity.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace minicpm_ocr_tiles {

struct config {
    int scale_resolution = 448;
    int max_slice_nums = 9;
    int patch_size = 14;
    int merge = 2;          // downsample_mode "4x": 2x2 patch merge
    bool slice_mode = true; // false = overview only
    int pos_grid = 70;      // learned 70x70 position table (980/14)
};

struct size2 {
    int w = 0, h = 0;
};

// Python round(): half-to-even. std::nearbyint honours the default FE_TONEAREST
// mode, which is half-to-even as well.
inline int py_round(double x) {
    return int(std::nearbyint(x));
}

// MiniCPM-V `ensure_divide(length, patch_size)`:
// max(round(length / patch_size) * patch_size, patch_size).
inline int ensure_divide(double length, int unit) {
    const int r = py_round(length / unit) * unit;
    return r < unit ? unit : r;
}

// Sizes snap to patch_size * merge (28) so every image splits into whole 2x2
// patch blocks. TODO(verify): spec says "multiple of 28"; the MiniCPM-V 2.6
// processor used patch_size (14). Confirm with reference target sizes.
inline int snap_unit(const config & c) {
    return c.patch_size * c.merge;
}

// `find_best_resize`: area-preserving rescale to scale_resolution^2 (only when
// larger, unless allow_upscale), then snap each side.
inline size2 find_best_resize(double w, double h, const config & c, bool allow_upscale) {
    const double s = c.scale_resolution;
    if (w * h > s * s || allow_upscale) {
        const double r = w / h;
        h = double(int64_t(s / std::sqrt(r))); // python int(): truncation
        w = double(int64_t(h * r));
    }
    return { ensure_divide(w, snap_unit(c)), ensure_divide(h, snap_unit(c)) };
}

// `get_sliced_grid`: returns {cols, rows} or {0, 0} when the image is not sliced.
// Candidate counts multiple-1, multiple, multiple+1 (skipping 1 and > max);
// factorizations enumerated with ascending cols; strict '<' keeps the first
// minimum, so ties go to fewer columns (more rows). TODO(verify): tie order.
inline size2 sliced_grid(int w, int h, const config & c) {
    if (!c.slice_mode || w < 1 || h < 1) return {};
    const double s = c.scale_resolution;
    const double ratio = double(w) * h / (s * s);
    int multiple = int(std::ceil(ratio));
    if (multiple > c.max_slice_nums) multiple = c.max_slice_nums;
    if (multiple <= 1) return {};
    const double log_ratio = std::log(double(w) / h);
    size2 best{ 1, 1 };
    double best_error = INFINITY;
    for (int n = multiple - 1; n <= multiple + 1; ++n) {
        if (n == 1 || n > c.max_slice_nums || n < 1) continue;
        for (int m = 1; m <= n; ++m) {
            if (n % m) continue;
            const double error = std::fabs(log_ratio - std::log(double(m) / (n / m)));
            if (error < best_error) {
                best_error = error;
                best = { m, n / m };
            }
        }
    }
    return best;
}

// `get_refine_size`: per-cell best resize (upscale allowed) times the grid.
inline size2 refine_size(int w, int h, size2 grid, const config & c) {
    const int rw = ensure_divide(w, grid.w), rh = ensure_divide(h, grid.h);
    const size2 cell = find_best_resize(double(rw) / grid.w, double(rh) / grid.h, c, true);
    return { cell.w * grid.w, cell.h * grid.h };
}

// Visual tokens one image contributes after the 2x2 merge.
inline int tokens_for(size2 s, const config & c) {
    const int ph = s.h / c.patch_size, pw = s.w / c.patch_size;
    return (ph / c.merge) * (pw / c.merge);
}

struct image_part {
    bool overview = false;
    int row = 0, col = 0; // slice position (row-major)
    size2 size;           // resized size of this part
    int x0 = 0, y0 = 0;   // crop origin inside the refine image (slices only)
    int tokens = 0;
};

struct plan {
    size2 source;                  // original image
    size2 overview;                // overview resize target
    size2 grid;                    // {cols, rows}; {0,0} = no slices
    size2 refine;                  // whole-image resize target before cutting tiles
    std::vector<image_part> parts; // overview first, then slices row-major
    int total_tokens = 0;
};

inline bool make_plan(int w, int h, const config & c, plan & out) {
    out = {};
    if (w < 1 || h < 1 || c.patch_size < 1 || c.merge < 1 || c.scale_resolution < 1) return false;
    out.source = { w, h };
    out.grid = sliced_grid(w, h, c);
    const bool sliced = out.grid.w > 0;
    // Unsliced images are upscaled ("always"); the overview of a sliced image is not.
    out.overview = find_best_resize(w, h, c, !sliced);
    image_part ov;
    ov.overview = true;
    ov.size = out.overview;
    ov.tokens = tokens_for(ov.size, c);
    out.parts.push_back(ov);
    if (sliced) {
        out.refine = refine_size(w, h, out.grid, c);
        const int tw = out.refine.w / out.grid.w, th = out.refine.h / out.grid.h;
        for (int r = 0; r < out.grid.h; ++r)
            for (int col = 0; col < out.grid.w; ++col) {
                image_part p;
                p.row = r;
                p.col = col;
                p.size = { tw, th };
                p.x0 = col * tw;
                p.y0 = r * th;
                p.tokens = tokens_for(p.size, c);
                out.parts.push_back(p);
            }
    }
    for (const auto & p : out.parts) {
        if (p.size.w % snap_unit(c) || p.size.h % snap_unit(c) || p.tokens < 1) return false;
        out.total_tokens += p.tokens;
    }
    return true;
}

// MiniCPM-V NaViT position ids: torch.bucketize(frac, boundaries, right=True)
// with boundaries = torch.arange(1/G, 1.0, 1/G) (python-double start/step) and
// frac = torch.arange(0, 1 - 1e-6, 1/n) where 1/n is a float32 tensor value.
// torch's CPU arange evaluates start + step*i in double and casts to float32;
// bucketize(right=True) = number of boundaries <= x.
// TODO(verify): boundary rule and arange rounding against the reference vision
// embeddings (exact-boundary cases such as n=10, i=7 flip with float32 math).
inline std::vector<int> bucket_coords(int n, int grid) {
    std::vector<float> bounds;
    const double bstep = 1.0 / double(grid);
    for (int i = 0;; ++i) {
        const double v = bstep + bstep * i;
        if (!(v < 1.0)) break;
        bounds.push_back(float(v));
    }
    std::vector<int> out(n);
    const double step = double(1.0f / float(n));
    for (int i = 0; i < n; ++i) {
        const float x = float(step * i);
        int k = 0;
        while (k < int(bounds.size()) && bounds[k] <= x) ++k;
        out[i] = k;
    }
    return out;
}

// Row-major position-table ids for an (h_patches x w_patches) image.
inline std::vector<int> position_ids(int h_patches, int w_patches, int grid) {
    const auto bh = bucket_coords(h_patches, grid), bw = bucket_coords(w_patches, grid);
    std::vector<int> ids;
    ids.reserve(size_t(h_patches) * w_patches);
    for (int y = 0; y < h_patches; ++y)
        for (int x = 0; x < w_patches; ++x) ids.push_back(bh[y] * grid + bw[x]);
    return ids;
}

} // namespace minicpm_ocr_tiles
