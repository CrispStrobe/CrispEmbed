// minicpm_ocr_tiles.h - header-only slicing/grid/token arithmetic for
// openbmb/MiniCPM-OCR-1.0 (MiniCPMOcrImageProcessor, preprocessor_config.json:
// scale_resolution=448, max_slice_nums=9, patch_size=14, downsample_mode="4x",
// small_image_upsize="always"). See docs/minicpm-ocr-inference-spec.md,
// "Preprocessing" steps 1-5 and "Token layout".
//
// Pure integer/float math, no ggml: guarded by tests/test_minicpm_ocr_tiles.cpp.
// Every rule below is transcribed from transformers v5.21.0
// models/minicpm_ocr/image_processing_minicpm_ocr.py (cited per function) and
// vision_utils.py; see the decision table in docs/minicpm-ocr-inference-spec.md.
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

// Sizes snap to patch_size * merge_factor (28 in "4x" mode): find_best_resize,
// image_processing_minicpm_ocr.py:179-181.
inline int snap_unit(const config & c) {
    return c.patch_size * c.merge;
}

// `find_best_resize` (image_processing_minicpm_ocr.py:165-182): area-preserving rescale to scale_resolution^2 (only when
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

// `get_sliced_grid` (image_processing_minicpm_ocr.py:207-234): returns
// {cols, rows} or {0, 0} when the image is not sliced. Candidate counts
// multiple-1, multiple, multiple+1 (skipping 1 and > max); rows ascend; a
// strictly smaller error wins, and an exactly equal error is taken when it has
// more rows than the current best (so ties go to the most rows, across counts).
inline size2 sliced_grid(int w, int h, const config & c) {
    if (!c.slice_mode || w < 1 || h < 1) return {};
    const double s = c.scale_resolution;
    const double log_ratio = std::log(double(w) / double(h));
    const double ratio = double(w) * double(h) / (s * s);
    int multiple = int(std::ceil(ratio));
    if (multiple > c.max_slice_nums) multiple = c.max_slice_nums;
    if (multiple <= 1) return {};
    int best_rows = 1, best_cols = 1;
    double min_error = INFINITY;
    for (int n = multiple - 1; n <= multiple + 1; ++n) {
        if (n == 1 || n > c.max_slice_nums) continue;
        for (int rows = 1; rows <= n; ++rows) {
            if (n % rows) continue;
            const int cols = n / rows;
            const double error = std::fabs(log_ratio - std::log(double(cols) / double(rows)));
            if (error < min_error) {
                best_rows = rows;
                best_cols = cols;
                min_error = error;
            } else if (error == min_error && rows > best_rows) {
                best_rows = rows;
                best_cols = cols;
            }
        }
    }
    return { best_cols, best_rows };
}

// `get_refine_size` (image_processing_minicpm_ocr.py:184-205): per-cell best resize (upscale allowed) times the grid.
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
    // Unsliced images are upscaled ("always"); the overview of a sliced image is
    // not (image_processing_minicpm_ocr.py:289). Tiles are cut row-major from
    // the refine image (divide_to_patches, image_transforms.py:894-897).
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

// NaViT position ids, transformers v5.21.0 vision_utils.py:346-356
// (get_vision_nearest_position_ids):
//   boundaries = torch.arange(1/G, 1.0, 1/G)            # float32 (default dtype)
//   coords     = torch.arange(n) / n                     # int64 / int -> float32
//   bucket     = torch.bucketize(coords, boundaries, right=True)
// torch's CPU float arange has length ceil((end - start) / step) computed in
// double and values float(start + i * step) accumulated in double;
// true_divide promotes both operands to float32 and divides in float32;
// bucketize(right=True) counts the boundaries <= x.
inline std::vector<float> bucket_boundaries(int grid) {
    const double step = 1.0 / double(grid);
    const int64_t count = int64_t(std::ceil((1.0 - step) / step));
    std::vector<float> bounds;
    for (int64_t i = 0; i < count; ++i) bounds.push_back(float(step + step * double(i)));
    return bounds;
}

inline std::vector<int> bucket_coords(int n, int grid) {
    const auto bounds = bucket_boundaries(grid);
    std::vector<int> out(n);
    for (int i = 0; i < n; ++i) {
        const float x = float(i) / float(n);
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
