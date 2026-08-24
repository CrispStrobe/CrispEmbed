// image_preprocess.cpp — in-process port of HF Qwen2VLImageProcessorFast.
// See image_preprocess.h for the parity caveat (bilinear vs torchvision bicubic).

#include "image_preprocess.h"
#include "scan_cleanup.h"

#define STB_IMAGE_IMPLEMENTATION
// NOT static — export symbols for use by ocr_detect, ocr_pipeline, math_ocr
#include "../ggml/examples/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace image_preproc {

bool smart_resize(int height, int width, int factor, int min_pixels, int max_pixels, int * out_h, int * out_w) {
    if (height <= 0 || width <= 0 || factor <= 0) return false;

    const int hi = std::max(height, width);
    const int lo = std::min(height, width);
    if ((float)hi / (float)lo > 200.0f) {
        std::fprintf(stderr, "image_preproc: aspect ratio > 200:1 (%dx%d) — refusing\n", height, width);
        return false;
    }

    auto round_to_factor = [&](float v) { return std::max(factor, (int)std::round(v / factor) * factor); };
    auto floor_to_factor = [&](float v) { return std::max(factor, (int)std::floor(v / factor) * factor); };
    auto ceil_to_factor = [&](float v) { return std::max(factor, (int)std::ceil(v / factor) * factor); };

    int h_bar = round_to_factor((float)height);
    int w_bar = round_to_factor((float)width);

    if ((double)h_bar * (double)w_bar > (double)max_pixels) {
        const float beta = std::sqrt(((float)height * (float)width) / (float)max_pixels);
        h_bar = floor_to_factor((float)height / beta);
        w_bar = floor_to_factor((float)width / beta);
    } else if ((double)h_bar * (double)w_bar < (double)min_pixels) {
        const float beta = std::sqrt((float)min_pixels / ((float)height * (float)width));
        h_bar = ceil_to_factor((float)height * beta);
        w_bar = ceil_to_factor((float)width * beta);
    }

    if (out_h) *out_h = h_bar;
    if (out_w) *out_w = w_bar;
    return true;
}

namespace {

// Catmull-Rom cubic kernel with a = -0.5. This matches PIL (Pillow) and
// torchvision.transforms with the PIL backend — the resize path HF vision
// processors (Qwen2-VL etc.) actually use, so a = -0.5 is the correct choice
// for HF parity (measured residual cos 0.999984). NOTE: OpenCV INTER_CUBIC and
// torch.nn.functional.interpolate(mode='bicubic') use a = -0.75 instead; the
// difference vs a = -0.5 is only cos ~0.99999 (max ~0.13/255), so it is not
// worth a runtime toggle — see PLAN "C5 remnants (a)".
//   |x| <  1: (a+2)|x|^3 - (a+3)|x|^2 + 1
//   |x| <  2: a|x|^3 - 5a|x|^2 + 8a|x| - 4a
//   else    : 0
inline float cubic_kernel(float x) {
    constexpr float a = -0.5f;
    x = std::abs(x);
    if (x < 1.0f) {
        return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    }
    return 0.0f;
}

// Per-output 1D resample weights for separable bicubic with antialias.
// Mirrors torchvision's `interpolate(antialias=True)` for downscale: when the
// scale `s = src/dst > 1`, we widen the kernel support by `s` and renormalize.
// For upscale (s < 1) we use unit support — same kernel torchvision uses.
struct resample1d {
    int out_size = 0;
    int in_size = 0;
    int support = 0;            // taps per output sample
    std::vector<int> indices;   // (out_size, support) clamped src indices
    std::vector<float> weights; // (out_size, support)
};

void build_cubic_weights(int in_size, int out_size, bool antialias, resample1d & r) {
    r.in_size = in_size;
    r.out_size = out_size;
    // Use double for index/weight math — torchvision's aa kernel uses float64
    // for the index computation even when sampling float32, and the small
    // sub-pixel differences propagate into the embedding cosine.
    const double scale = (double)in_size / (double)out_size; // > 1 for downscale
    const double filter_scale = (antialias && scale > 1.0) ? scale : 1.0;
    const double inv_filter = 1.0 / filter_scale;
    const double radius = 2.0 * filter_scale;
    // torchvision: support_size = ceil(2 * support).
    r.support = (int)std::ceil(2.0 * radius);
    if (r.support < 4) r.support = 4;
    r.indices.assign((size_t)out_size * r.support, 0);
    r.weights.assign((size_t)out_size * r.support, 0.0f);
    for (int i = 0; i < out_size; i++) {
        const double center = ((double)i + 0.5) * scale - 0.5;
        // torchvision: xmin = (int64_t)(center - support + 0.5)
        // For positive `center - support + 0.5` this is floor (truncation).
        const double left_d = center - radius + 0.5;
        const int left = (int)std::floor(left_d);
        double sum = 0.0;
        for (int k = 0; k < r.support; k++) {
            const int src_idx = left + k;
            const double w = cubic_kernel((float)((src_idx - center) * inv_filter));
            r.weights[(size_t)i * r.support + k] = (float)w;
            r.indices[(size_t)i * r.support + k] = std::min(std::max(src_idx, 0), in_size - 1);
            sum += w;
        }
        // Renormalize so the kernel always sums to 1 (matches torchvision).
        if (sum != 0.0) {
            const double inv = 1.0 / sum;
            for (int k = 0; k < r.support; k++) {
                r.weights[(size_t)i * r.support + k] *= (float)inv;
            }
        }
    }
}

// Separable bicubic resize with antialiasing on downscale. Operates in float32:
//   uint8 src (src_h, src_w, C) → float intermediate (src_h, dst_w, C) → float dst (dst_h, dst_w, C)
// Output is in [0, 255] before rescale/normalize. Values are clamped to
// [0, 255] to suppress cubic ringing at edges (torchvision does this for
// uint8 round-trip; for float intermediate we still clamp to keep rescale
// in [0, 1]).
void bicubic_resize_u8_to_f32(const uint8_t * src, int src_h, int src_w, float * dst, int dst_h, int dst_w,
                              int channels) {
    if (dst_h <= 0 || dst_w <= 0) return;
    resample1d wx, wy;
    build_cubic_weights(src_w, dst_w, /*antialias=*/true, wx);
    build_cubic_weights(src_h, dst_h, /*antialias=*/true, wy);

    // Pass 1: horizontal resample → (src_h, dst_w, C) float32.
    std::vector<float> mid((size_t)src_h * dst_w * channels, 0.0f);
#pragma omp parallel for schedule(static) if (src_h > 32)
    for (int y = 0; y < src_h; y++) {
        for (int xo = 0; xo < dst_w; xo++) {
            const int * xidx = wx.indices.data() + (size_t)xo * wx.support;
            const float * xw = wx.weights.data() + (size_t)xo * wx.support;
            float * out = mid.data() + ((size_t)y * dst_w + xo) * channels;
            for (int c = 0; c < channels; c++) out[c] = 0.0f;
            for (int k = 0; k < wx.support; k++) {
                const uint8_t * px = src + ((size_t)y * src_w + xidx[k]) * channels;
                const float w = xw[k];
                for (int c = 0; c < channels; c++) {
                    out[c] += w * (float)px[c];
                }
            }
        }
    }

// Pass 2: vertical resample → (dst_h, dst_w, C) float32 in [0, 255].
// We round to integer to mimic torchvision's uint8 resize (tvF.resize on
// a uint8 tensor casts to uint8 with round+clamp at the end of the AA
// bicubic). Skipping the round leaves sub-pixel precision but produces
// pixel values that diverge from HF's preprocessor by up to ~1/std.
#pragma omp parallel for schedule(static) if (dst_h > 32)
    for (int yo = 0; yo < dst_h; yo++) {
        const int * yidx = wy.indices.data() + (size_t)yo * wy.support;
        const float * yw = wy.weights.data() + (size_t)yo * wy.support;
        for (int x = 0; x < dst_w; x++) {
            float * out = dst + ((size_t)yo * dst_w + x) * channels;
            for (int c = 0; c < channels; c++) out[c] = 0.0f;
            for (int k = 0; k < wy.support; k++) {
                const float * px = mid.data() + ((size_t)yidx[k] * dst_w + x) * channels;
                const float w = yw[k];
                for (int c = 0; c < channels; c++) {
                    out[c] += w * px[c];
                }
            }
            for (int c = 0; c < channels; c++) {
                float v = std::min(std::max(out[c], 0.0f), 255.0f);
                out[c] = std::roundf(v);
            }
        }
    }
}

// Patchify a (T_patch, C, H, W) row-major float32 plane stack (post-normalize)
// into the BidirLM/Qwen2VL flat (n_patches, C*T_patch*P*P) row layout.
//
// Mirrors HF's permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9). Because we have a single
// image (batch=1, grid_t=1) this collapses to a 7-D permute over
// (T_patch, C, h_g, mg, P_h, w_g, mg, P_w) → (h_g, w_g, mg_h, mg_w, C, T_patch, P_h, P_w).
//
// The output is row-major (n_patches, row_dim) where:
//   n_patches = grid_t * grid_h * grid_w   (here grid_t=1)
//   row_dim   = C * T_patch * P * P
// and tokens are in merge-permuted order (matches the vision-tower's host_prep).
void patchify_qwen_layout(const float * frames, // (T_patch, C, H, W)
                          int channels, int T_patch, int H, int W, int patch_size, int merge_size,
                          float * out_patches, // (n_patches, row_dim)
                          int * out_grid_h, int * out_grid_w) {
    const int grid_h = H / patch_size;
    const int grid_w = W / patch_size;
    const int mg = merge_size;
    const int P = patch_size;
    const int row_dim = channels * T_patch * P * P;
    const size_t plane = (size_t)channels * H * W;
    auto frame_at = [&](int t) { return frames + (size_t)t * plane; };

    // Iterate in merge-permuted order (matches host_prep in bidirlm_vision.cpp).
    for (int hg = 0; hg < grid_h / mg; hg++) {
        for (int wg = 0; wg < grid_w / mg; wg++) {
            for (int mh = 0; mh < mg; mh++) {
                for (int mw = 0; mw < mg; mw++) {
                    const int row_idx = ((hg * (grid_w / mg) + wg) * mg + mh) * mg + mw;
                    float * dst = out_patches + (size_t)row_idx * row_dim;
                    int k = 0;
                    for (int c = 0; c < channels; c++) {
                        for (int t = 0; t < T_patch; t++) {
                            const float * src = frame_at(t) + (size_t)c * H * W;
                            const int row_start = (hg * mg + mh) * P;
                            const int col_start = (wg * mg + mw) * P;
                            for (int py = 0; py < P; py++) {
                                const float * row = src + (size_t)(row_start + py) * W + col_start;
                                for (int px = 0; px < P; px++) {
                                    dst[k++] = row[px];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (out_grid_h) *out_grid_h = grid_h;
    if (out_grid_w) *out_grid_w = grid_w;
}

// Convert the resized RGB float32 (H, W, C) to a temporal-padded (T_patch, C, H, W)
// stack with rescale to [0,1] and per-channel mean/std normalization.
void normalize_and_temporal_pad(const float * rgb_hwc, int H, int W, int channels, int T_patch, const float mean[3],
                                const float std_[3], std::vector<float> & out_tchw) {
    const size_t plane = (size_t)channels * H * W;
    out_tchw.assign(plane * T_patch, 0.0f);

    // Build the first frame: rescale (/255) then normalize per channel.
    // Re-layout from HWC → CHW.
    const float scale = 1.0f / 255.0f;
#pragma omp parallel for schedule(static) if (H > 32)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const float * px = rgb_hwc + (size_t)(y * W + x) * channels;
            for (int c = 0; c < channels; c++) {
                const float v = (px[c] * scale - mean[c]) / std_[c];
                out_tchw[(size_t)c * H * W + (size_t)y * W + x] = v;
            }
        }
    }
    // Repeat the first frame along the temporal axis (HF pads with the last
    // frame; for a single image this is identical).
    for (int t = 1; t < T_patch; t++) {
        std::memcpy(out_tchw.data() + (size_t)t * plane, out_tchw.data(), plane * sizeof(float));
    }
}

} // namespace

bool preprocess_rgb(const uint8_t * rgb, int height, int width, int channels, const config & cfg, result & out) {
    if (!rgb || height <= 0 || width <= 0) return false;
    if (channels < 3) {
        std::fprintf(stderr, "image_preproc: expected RGB(A) input, got %d channels\n", channels);
        return false;
    }
    if (cfg.deskew) {
        uint8_t * rot = nullptr;
        int rw2 = 0, rh2 = 0;
        if (scan_cleanup_deskew_rgb(rgb, width, height, channels, cfg.deskew_max_angle, &rot, &rw2, &rh2) == 0 && rot) {
            config c2 = cfg;
            c2.deskew = 0;
            bool ok = preprocess_rgb(rot, rh2, rw2, channels, c2, out);
            scan_cleanup_free_image(rot);
            return ok;
        }
        // detector reported no skew (or bad input) — continue unrotated
    }
    const int factor = cfg.patch_size * cfg.merge_size;
    // target_longest emulates a fixed-longest-side page render: smart_resize
    // sees the render's dims, while the single bicubic resample below still
    // goes straight from the original pixels to the final grid.
    int nom_h = height, nom_w = width;
    if (cfg.target_longest > 0) {
        const double s = (double)cfg.target_longest / (double)std::max(height, width);
        nom_h = std::max(1, (int)std::lround((double)height * s));
        nom_w = std::max(1, (int)std::lround((double)width * s));
    }
    int rh = 0, rw = 0;
    if (!smart_resize(nom_h, nom_w, factor, cfg.min_pixels, cfg.max_pixels, &rh, &rw)) {
        return false;
    }

    // Drop alpha if present.
    const int kC = 3;
    std::vector<uint8_t> rgb_pure;
    const uint8_t * rgb_use = rgb;
    if (channels != kC) {
        rgb_pure.resize((size_t)height * width * kC);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const uint8_t * src = rgb + (size_t)(y * width + x) * channels;
                uint8_t * dst = rgb_pure.data() + (size_t)(y * width + x) * kC;
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
        rgb_use = rgb_pure.data();
    }

    // Resize → float32 HWC in [0, 255]. Bicubic + antialias matches the
    // torchvision v2 default that HF Qwen2VLImageProcessorFast uses.
    std::vector<float> resized((size_t)rh * rw * kC, 0.0f);
    bicubic_resize_u8_to_f32(rgb_use, height, width, resized.data(), rh, rw, kC);

    // Normalize + temporal pad → (T_patch, C, H, W).
    std::vector<float> tchw;
    normalize_and_temporal_pad(resized.data(), rh, rw, kC, cfg.temporal_patch_size, cfg.mean, cfg.std, tchw);

    // Patchify.
    const int row_dim = kC * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    const int grid_h = rh / cfg.patch_size;
    const int grid_w = rw / cfg.patch_size;
    if (grid_h <= 0 || grid_w <= 0) {
        std::fprintf(stderr, "image_preproc: smart_resize produced 0-grid (rh=%d, rw=%d)\n", rh, rw);
        return false;
    }
    if (grid_h % cfg.merge_size != 0 || grid_w % cfg.merge_size != 0) {
        std::fprintf(stderr,
                     "image_preproc: grid (%dx%d) not divisible by merge_size %d — "
                     "smart_resize bug?\n",
                     grid_h, grid_w, cfg.merge_size);
        return false;
    }
    const int n_patches = grid_h * grid_w;
    out.patches.assign((size_t)n_patches * row_dim, 0.0f);
    int gh = 0, gw = 0;
    patchify_qwen_layout(tchw.data(), kC, cfg.temporal_patch_size, rh, rw, cfg.patch_size, cfg.merge_size,
                         out.patches.data(), &gh, &gw);
    out.n_patches = n_patches;
    out.row_dim = row_dim;
    out.grid_thw[0] = 1;
    out.grid_thw[1] = gh;
    out.grid_thw[2] = gw;
    out.resized_h = rh;
    out.resized_w = rw;
    return true;
}

bool preprocess_file(const char * path, const config & cfg, result & out) {
    if (!path) return false;
    int W = 0, H = 0, C = 0;
    // Force RGB: stbi loads JPEG/PNG/BMP/etc. into 8-bit interleaved.
    uint8_t * rgb = stbi_load(path, &W, &H, &C, 3);
    if (!rgb) {
        std::fprintf(stderr, "image_preproc: stbi_load failed for '%s': %s\n", path, stbi_failure_reason());
        return false;
    }
    bool ok = preprocess_rgb(rgb, H, W, /*channels=*/3, cfg, out);
    stbi_image_free(rgb);
    return ok;
}

// ── InternVL2 dynamic tiling ────────────────────────────────────────

// Find the closest aspect ratio from a set of possible (rows, cols) grids.
// Returns (best_rows, best_cols).
static void find_closest_aspect_ratio(int img_h, int img_w, int min_tiles, int max_tiles, int tile_size, int & out_rows,
                                      int & out_cols, int prior_rows = 0, int prior_cols = 0) {
    float target_aspect = (float)img_w / (float)img_h;
    float best_diff = 1e9f;
    out_rows = 0;
    out_cols = 0;

    const bool exclude = (prior_rows > 0 && prior_cols > 0);

    for (int n = min_tiles; n <= max_tiles; n++) {
        // Try all factorizations of n
        for (int r = 1; r <= n; r++) {
            if (n % r != 0) continue;
            int c = n / r;
            // MSAC pass 2: keep only grids that divide neither axis of the
            // coarse grid, so the second scale is genuinely different rather
            // than a refinement of the first. Upstream indexes its ratio
            // tuples (cols, rows), hence the pairing below.
            if (exclude && !(prior_cols % c != 0 && prior_rows % r != 0)) continue;
            float aspect = (float)c / (float)r;
            float diff = std::abs(aspect - target_aspect);
            if (diff < best_diff) {
                best_diff = diff;
                out_rows = r;
                out_cols = c;
            } else if (diff == best_diff) {
                // Upstream InternVL breaks aspect ties toward MORE tiles when
                // the source has the pixels to fill them:
                //   elif ratio_diff == best and area > 0.5*size*size*i[0]*i[1]
                // We previously preferred fewer tiles, which picks a different
                // grid for near-square inputs (e.g. 800x800 chose 1x1 where
                // upstream chooses 2x2). That matters on its own, and MSAC's
                // second pass is derived from this grid, so it has to match.
                const double area = (double)img_w * (double)img_h;
                if (area > 0.5 * (double)tile_size * (double)tile_size * (double)r * (double)c) {
                    out_rows = r;
                    out_cols = c;
                }
            }
        }
    }
}

// Bilinear resize of uint8 RGB into a float tile, with normalization.
// dst: (3, tile_h, tile_w) planar float, normalized.
static void resize_and_normalize_tile(const uint8_t * src, int src_w, int src_h, int src_stride, int channels,
                                      float * dst, int tile_w, int tile_h, const float mean[3], const float std_v[3]) {
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < tile_h; y++) {
            float sy = (float)y * src_h / tile_h;
            int iy0 = (int)sy;
            int iy1 = std::min(iy0 + 1, src_h - 1);
            float fy = sy - iy0;
            for (int x = 0; x < tile_w; x++) {
                float sx = (float)x * src_w / tile_w;
                int ix0 = (int)sx;
                int ix1 = std::min(ix0 + 1, src_w - 1);
                float fx = sx - ix0;

                int ch = std::min(c, channels - 1);
                float v00 = (float)src[iy0 * src_stride + ix0 * channels + ch];
                float v01 = (float)src[iy0 * src_stride + ix1 * channels + ch];
                float v10 = (float)src[iy1 * src_stride + ix0 * channels + ch];
                float v11 = (float)src[iy1 * src_stride + ix1 * channels + ch];

                float val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11);
                val /= 255.0f;
                dst[c * tile_h * tile_w + y * tile_w + x] = (val - mean[c]) / std_v[c];
            }
        }
    }
}

bool preprocess_internvl_rgb(const uint8_t * rgb, int height, int width, int channels, const internvl_config & cfg,
                             internvl_result & out) {
    const int S = cfg.image_size; // 448

    // Find best tiling grid
    int grid_r, grid_c;
    find_closest_aspect_ratio(height, width, cfg.min_dynamic_patch, cfg.max_dynamic_patch, S, grid_r, grid_c,
                              cfg.exclude_prior_rows, cfg.exclude_prior_cols);
    if (grid_r <= 0 || grid_c <= 0) {
        // Only reachable on an MSAC second pass whose exclusion admitted
        // nothing. Without exclusion min_dynamic_patch >= 1 always yields 1x1,
        // so the single-scale path keeps its previous behaviour.
        if (cfg.exclude_prior_rows > 0 && cfg.exclude_prior_cols > 0) return false;
        grid_r = 1;
        grid_c = 1;
    }
    int n_tiles = grid_r * grid_c;
    if (cfg.use_thumbnail) n_tiles += 1;

    out.n_tiles = n_tiles;
    out.tile_size = S;
    out.grid_rows = grid_r;
    out.grid_cols = grid_c;
    out.tiles.resize((size_t)n_tiles * 3 * S * S);

    // Resize image to fit the grid: (grid_r * S) x (grid_c * S)
    int target_h = grid_r * S;
    int target_w = grid_c * S;

    // Allocate resized image as uint8
    std::vector<uint8_t> resized(target_h * target_w * 3);
    int dst_stride = target_w * 3;
    for (int y = 0; y < target_h; y++) {
        float sy = (float)y * height / target_h;
        int iy0 = (int)sy;
        int iy1 = std::min(iy0 + 1, height - 1);
        float fy = sy - iy0;
        for (int x = 0; x < target_w; x++) {
            float sx = (float)x * width / target_w;
            int ix0 = (int)sx;
            int ix1 = std::min(ix0 + 1, width - 1);
            float fx = sx - ix0;
            for (int c = 0; c < 3; c++) {
                int ch = std::min(c, channels - 1);
                float v00 = (float)rgb[iy0 * width * channels + ix0 * channels + ch];
                float v01 = (float)rgb[iy0 * width * channels + ix1 * channels + ch];
                float v10 = (float)rgb[iy1 * width * channels + ix0 * channels + ch];
                float v11 = (float)rgb[iy1 * width * channels + ix1 * channels + ch];
                float val = (1 - fy) * ((1 - fx) * v00 + fx * v01) + fy * ((1 - fx) * v10 + fx * v11);
                resized[y * dst_stride + x * 3 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, val));
            }
        }
    }

    // Split into tiles and normalize
    int tile_idx = 0;
    for (int tr = 0; tr < grid_r; tr++) {
        for (int tc = 0; tc < grid_c; tc++) {
            // Extract tile region from resized image
            int y0 = tr * S;
            int x0 = tc * S;
            // Point to the start of this tile in the resized buffer
            // We need to resize_and_normalize from the sub-region
            // Since resized is already the right size, just copy the tile
            float * dst = out.tiles.data() + (size_t)tile_idx * 3 * S * S;
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < S; y++) {
                    for (int x = 0; x < S; x++) {
                        float val = (float)resized[(y0 + y) * dst_stride + (x0 + x) * 3 + c] / 255.0f;
                        dst[c * S * S + y * S + x] = (val - cfg.mean[c]) / cfg.std[c];
                    }
                }
            }
            tile_idx++;
        }
    }

    // Thumbnail: resize full image to S×S
    if (cfg.use_thumbnail) {
        float * dst = out.tiles.data() + (size_t)tile_idx * 3 * S * S;
        resize_and_normalize_tile(rgb, width, height, width * channels, channels, dst, S, S, cfg.mean, cfg.std);
        tile_idx++;
    }

    return true;
}

bool preprocess_internvl_msac_rgb(const uint8_t * rgb, int height, int width, int channels, const internvl_config & cfg,
                                  internvl_result & out) {
    const int S = cfg.image_size;

    // Pass 1: the ordinary grid.
    internvl_config c1 = cfg;
    c1.exclude_prior_rows = 0;
    c1.exclude_prior_cols = 0;
    c1.use_thumbnail = true;
    internvl_result coarse;
    if (!preprocess_internvl_rgb(rgb, height, width, channels, c1, coarse)) return false;

    // Pass 2: a finer grid (min 3 tiles) that divides neither axis of pass 1.
    internvl_config c2 = cfg;
    c2.min_dynamic_patch = 3;
    c2.exclude_prior_rows = coarse.grid_rows;
    c2.exclude_prior_cols = coarse.grid_cols;
    c2.use_thumbnail = true;
    internvl_result fine;
    if (!preprocess_internvl_rgb(rgb, height, width, channels, c2, fine)) return false;

    // Upstream appends a thumbnail only when a pass produced more than one
    // block, so `[:-1]` on a 1x1 pass drops its only tile and contributes
    // nothing. Reproduce that exactly rather than assuming a thumbnail is
    // always present.
    const int coarse_blocks = coarse.grid_rows * coarse.grid_cols;
    const int fine_blocks = fine.grid_rows * fine.grid_cols;
    const int coarse_body = (coarse_blocks == 1) ? 0 : coarse_blocks;
    const int fine_body = (fine_blocks == 1) ? 0 : fine_blocks;
    if (fine_body < 1) return false; // the fine thumbnail comes from this pass

    // fine[:-1] + coarse[:-1] + fine[-1:]
    const size_t tile_floats = (size_t)3 * S * S;
    out.n_tiles = fine_body + coarse_body + 1;
    out.tile_size = S;
    out.grid_rows = coarse.grid_rows; // the coarse grid is the page layout
    out.grid_cols = coarse.grid_cols;
    out.tiles.resize((size_t)out.n_tiles * tile_floats);

    float * dst = out.tiles.data();
    std::memcpy(dst, fine.tiles.data(), (size_t)fine_body * tile_floats * sizeof(float));
    dst += (size_t)fine_body * tile_floats;
    std::memcpy(dst, coarse.tiles.data(), (size_t)coarse_body * tile_floats * sizeof(float));
    dst += (size_t)coarse_body * tile_floats;
    std::memcpy(dst, fine.tiles.data() + (size_t)fine_body * tile_floats, tile_floats * sizeof(float));

    return true;
}

bool preprocess_internvl_file(const char * path, const internvl_config & cfg, internvl_result & out) {
    int W, H, C;
    uint8_t * rgb = stbi_load(path, &W, &H, &C, 3);
    if (!rgb) {
        std::fprintf(stderr, "image_preproc: stbi_load failed for '%s': %s\n", path, stbi_failure_reason());
        return false;
    }
    bool ok = preprocess_internvl_rgb(rgb, H, W, 3, cfg, out);
    stbi_image_free(rgb);
    return ok;
}

// Public wrapper exposing the file-local separable bicubic resampler (see the
// anonymous-namespace bicubic_resize_u8_to_f32 above) for reuse by other
// fixed-size preprocessors (e.g. mixtex_ocr's 400x500 ViTImageProcessor path).
// ── PIL-exact separable bicubic ──────────────────────────────────────────────
//
// The function above is TORCHVISION's `interpolate(antialias=True)`. HF's SLOW
// image processors (`resample: 3`) go through **PIL**, and the two are not the
// same resampler. Measured against `Image.resize(..., BICUBIC)` on the
// LFM2.5-VL fixtures, our torchvision port differs by up to 18/255 on thousands
// of pixels; the port below is bit-exact on one fixture and off by at most
// 1/255 on ~100 pixels of 750k on the rest (that residue is PIL's fixed-point
// integer coefficients against our double).
//
// Three things make the difference, and all three matter:
//   1. PIL clip8()s the intermediate back to **uint8 between the horizontal and
//      the vertical pass**. Carrying float through both passes is what produced
//      most of the 18/255.
//   2. PIL's tap count VARIES per output pixel — `xmax = (int)(center + support
//      + 0.5) - xmin` — where the torchvision form uses one fixed
//      `ceil(2*support)` for the whole axis. An extra near-zero tap changes the
//      whole kernel after renormalisation.
//   3. At the borders PIL CLAMPS the tap range to [0, in_size) and renormalises
//      over what is left; the torchvision form clamps the INDEX instead, i.e.
//      replicates the edge pixel. On a UI screenshot with content at the edge
//      that is a visible difference.
//
// Verbatim source: Pillow `src/libImaging/Resample.c`, `precompute_coeffs` +
// `ImagingResampleHorizontal_8bpc` / `ImagingResampleVertical_8bpc`.
// Pillow's 8-bit path is FIXED POINT: it quantises each normalised kernel to
// 22-bit integers and accumulates in ints. Reproducing that (rather than the
// same kernel in double) is what makes this bit-exact rather than off-by-one —
// verified 0 differing pixels against Image.resize(BICUBIC) on 430k-750k pixel
// page resizes and on the synthetic fixtures in tests/test_pil_resize.cpp.
static constexpr int kPilPrecisionBits = 32 - 8 - 2; // Pillow PRECISION_BITS

// Pillow evaluates its filter in DOUBLE. cubic_kernel() above is float32, and
// routing this path through it loses enough precision before the fixed-point
// quantisation to move ~25% of a small downscale by 1/255. Same a = -0.5.
static inline double pil_cubic_filter(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct pil_coeffs {
    int out_size = 0;
    int kmax = 0;
    std::vector<int> bmin;   // first source index per output sample
    std::vector<int> bcount; // tap count per output sample (VARIES)
    std::vector<int64_t> k;  // (out_size, kmax) fixed-point, row-major
};

static void pil_precompute_coeffs(int in_size, int out_size, pil_coeffs & c) {
    const double scale = (double)in_size / (double)out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 2.0 * filterscale; // bicubic support is 2.0
    const double ss = 1.0 / filterscale;
    const double one = (double)(1 << kPilPrecisionBits);

    c.out_size = out_size;
    c.kmax = (int)std::ceil(support) * 2 + 1;
    c.bmin.assign(out_size, 0);
    c.bcount.assign(out_size, 0);
    c.k.assign((size_t)out_size * c.kmax, 0);

    std::vector<double> kd(c.kmax, 0.0);
    for (int xx = 0; xx < out_size; xx++) {
        const double center = ((double)xx + 0.5) * scale;
        // Pillow clamps the tap RANGE to [0, in_size) and renormalises over
        // what survives; it does not clamp the index and replicate the edge.
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        if (xmax < 0) xmax = 0;
        if (xmax > c.kmax) xmax = c.kmax;

        double ww = 0.0;
        for (int x = 0; x < xmax; x++) {
            const double w = pil_cubic_filter(((double)(x + xmin) - center + 0.5) * ss);
            kd[x] = w;
            ww += w;
        }
        int64_t * k = c.k.data() + (size_t)xx * c.kmax;
        for (int x = 0; x < xmax; x++) {
            const double v = (ww != 0.0) ? kd[x] / ww : kd[x];
            // Pillow normalize_coeffs_8bpc: round away from zero.
            k[x] = (int64_t)(v < 0.0 ? -0.5 + v * one : 0.5 + v * one);
        }
        c.bmin[xx] = xmin;
        c.bcount[xx] = xmax;
    }
}

static inline int pil_clip8(int64_t acc) {
    const int64_t v = acc >> kPilPrecisionBits;
    return v <= 0 ? 0 : (v >= 256 ? 255 : (int)v);
}

void resize_bicubic_pil_u8_hwc(const uint8_t * src, int src_h, int src_w, float * dst, int dst_h, int dst_w,
                               int channels) {
    if (dst_h <= 0 || dst_w <= 0 || src_h <= 0 || src_w <= 0) return;
    pil_coeffs cx, cy;
    pil_precompute_coeffs(src_w, dst_w, cx);
    pil_precompute_coeffs(src_h, dst_h, cy);

    const int64_t half = (int64_t)1 << (kPilPrecisionBits - 1);

    // Pass 1: horizontal, uint8 -> uint8. The intermediate really is 8-bit:
    // carrying float through both passes is most of the 18/255 the
    // torchvision-shaped resampler differs by.
    std::vector<uint8_t> mid((size_t)src_h * dst_w * channels, 0);
#pragma omp parallel for schedule(static) if (src_h > 32)
    for (int y = 0; y < src_h; y++) {
        for (int xo = 0; xo < dst_w; xo++) {
            const int64_t * k = cx.k.data() + (size_t)xo * cx.kmax;
            const int xmin = cx.bmin[xo], n = cx.bcount[xo];
            uint8_t * out = mid.data() + ((size_t)y * dst_w + xo) * channels;
            for (int c = 0; c < channels; c++) {
                int64_t acc = half;
                const uint8_t * row = src + (size_t)y * src_w * channels + c;
                for (int x = 0; x < n; x++) acc += k[x] * (int64_t)row[(size_t)(xmin + x) * channels];
                out[c] = (uint8_t)pil_clip8(acc);
            }
        }
    }

    // Pass 2: vertical, uint8 -> the caller's float (integral values).
#pragma omp parallel for schedule(static) if (dst_h > 32)
    for (int yo = 0; yo < dst_h; yo++) {
        const int64_t * k = cy.k.data() + (size_t)yo * cy.kmax;
        const int ymin = cy.bmin[yo], n = cy.bcount[yo];
        for (int x = 0; x < dst_w; x++) {
            float * out = dst + ((size_t)yo * dst_w + x) * channels;
            for (int c = 0; c < channels; c++) {
                int64_t acc = half;
                for (int y = 0; y < n; y++) acc += k[y] * (int64_t)mid[((size_t)(ymin + y) * dst_w + x) * channels + c];
                out[c] = (float)pil_clip8(acc);
            }
        }
    }
}

void resize_bicubic_u8_hwc(const uint8_t * src, int src_h, int src_w, float * dst, int dst_h, int dst_w, int channels) {
    bicubic_resize_u8_to_f32(src, src_h, src_w, dst, dst_h, dst_w, channels);
}

} // namespace image_preproc
