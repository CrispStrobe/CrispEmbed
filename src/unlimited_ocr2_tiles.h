// Unlimited-OCR-2 gundam-tiling token arithmetic (reference: modeling_unlimitedocr.py,
// nq = ceil((image_size/patch)/4), nqb = ceil((base_size/patch)/4); IMG span =
// nqb*(nqb+1) + 1 + (nq*w_t + 1) * (nq*h_t) when tiled).
#pragma once

namespace uocr2_tiles {

// Queries per side after SAM patchify (patch) and the two stride-2 neck convs (x4).
inline constexpr int queries_per_side(int img, int patch = 16) {
    return ((img / patch) + 3) / 4;
}

// Total IMG-span vision tokens: local grid (+ newline per row), global (+ newline per
// row), view separator. Untiled pages: wt = ht = 1, tiled = false.
inline constexpr int n_vis(int wt, int ht, bool tiled, int tile = 640, int base = 1024, int patch = 16) {
    const int nq = queries_per_side(tile, patch), nqb = queries_per_side(base, patch);
    return nqb * (nqb + 1) + 1 + (tiled ? (nq * wt + 1) * (nq * ht) : 0);
}

// The legacy "SAM reduced resolution" optimisation bilinearly upsamples a smaller SAM
// grid back to the base 16x16 grid. It must NOT fire for 640 gundam tiles, whose
// reference grid is 10x10 (100 tokens per tile, not 256).
inline constexpr bool sam_upsample_to_base(int n_vis_tok, int target_nvis, int nP, int nP_orig, bool native_tile_grid) {
    return !native_tile_grid && n_vis_tok < target_nvis && nP != nP_orig;
}

} // namespace uocr2_tiles
