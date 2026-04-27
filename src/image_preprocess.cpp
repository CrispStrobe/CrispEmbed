// image_preprocess.cpp — in-process port of HF Qwen2VLImageProcessorFast.
// See image_preprocess.h for the parity caveat (bilinear vs torchvision bicubic).

#include "image_preprocess.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "../ggml/examples/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace image_preproc {

bool smart_resize(int height, int width,
                  int factor, int min_pixels, int max_pixels,
                  int * out_h, int * out_w) {
    if (height <= 0 || width <= 0 || factor <= 0) return false;

    const int hi = std::max(height, width);
    const int lo = std::min(height, width);
    if ((float)hi / (float)lo > 200.0f) {
        std::fprintf(stderr, "image_preproc: aspect ratio > 200:1 (%dx%d) — refusing\n",
                     height, width);
        return false;
    }

    auto round_to_factor = [&](float v) {
        return std::max(factor, (int)std::round(v / factor) * factor);
    };
    auto floor_to_factor = [&](float v) {
        return std::max(factor, (int)std::floor(v / factor) * factor);
    };
    auto ceil_to_factor = [&](float v) {
        return std::max(factor, (int)std::ceil(v / factor) * factor);
    };

    int h_bar = round_to_factor((float)height);
    int w_bar = round_to_factor((float)width);

    if ((double)h_bar * (double)w_bar > (double)max_pixels) {
        const float beta = std::sqrt(((float)height * (float)width) / (float)max_pixels);
        h_bar = floor_to_factor((float)height / beta);
        w_bar = floor_to_factor((float)width  / beta);
    } else if ((double)h_bar * (double)w_bar < (double)min_pixels) {
        const float beta = std::sqrt((float)min_pixels / ((float)height * (float)width));
        h_bar = ceil_to_factor((float)height * beta);
        w_bar = ceil_to_factor((float)width  * beta);
    }

    if (out_h) *out_h = h_bar;
    if (out_w) *out_w = w_bar;
    return true;
}

namespace {

// Catmull-Rom cubic kernel with a = -0.5 (matches torchvision / OpenCV / PIL).
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
    int      out_size  = 0;
    int      in_size   = 0;
    int      support   = 0;      // taps per output sample
    std::vector<int>   indices;  // (out_size, support) clamped src indices
    std::vector<float> weights;  // (out_size, support)
};

void build_cubic_weights(int in_size, int out_size, bool antialias,
                          resample1d & r) {
    r.in_size = in_size;
    r.out_size = out_size;
    // Use double for index/weight math — torchvision's aa kernel uses float64
    // for the index computation even when sampling float32, and the small
    // sub-pixel differences propagate into the embedding cosine.
    const double scale = (double)in_size / (double)out_size;  // > 1 for downscale
    const double filter_scale = (antialias && scale > 1.0) ? scale : 1.0;
    const double inv_filter   = 1.0 / filter_scale;
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
        const int    left   = (int)std::floor(left_d);
        double sum = 0.0;
        for (int k = 0; k < r.support; k++) {
            const int    src_idx = left + k;
            const double w       = cubic_kernel((float)((src_idx - center) * inv_filter));
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
void bicubic_resize_u8_to_f32(const uint8_t * src, int src_h, int src_w,
                              float * dst, int dst_h, int dst_w,
                              int channels) {
    if (dst_h <= 0 || dst_w <= 0) return;
    resample1d wx, wy;
    build_cubic_weights(src_w, dst_w, /*antialias=*/true, wx);
    build_cubic_weights(src_h, dst_h, /*antialias=*/true, wy);

    // Pass 1: horizontal resample → (src_h, dst_w, C) float32.
    std::vector<float> mid((size_t)src_h * dst_w * channels, 0.0f);
    for (int y = 0; y < src_h; y++) {
        for (int xo = 0; xo < dst_w; xo++) {
            const int   * xidx = wx.indices.data() + (size_t)xo * wx.support;
            const float * xw   = wx.weights.data() + (size_t)xo * wx.support;
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

    // Pass 2: vertical resample → (dst_h, dst_w, C) float32. Clamp to [0, 255].
    for (int yo = 0; yo < dst_h; yo++) {
        const int   * yidx = wy.indices.data() + (size_t)yo * wy.support;
        const float * yw   = wy.weights.data() + (size_t)yo * wy.support;
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
                out[c] = std::min(std::max(out[c], 0.0f), 255.0f);
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
void patchify_qwen_layout(const float * frames,           // (T_patch, C, H, W)
                          int channels, int T_patch,
                          int H, int W,
                          int patch_size, int merge_size,
                          float * out_patches,            // (n_patches, row_dim)
                          int * out_grid_h, int * out_grid_w) {
    const int grid_h = H / patch_size;
    const int grid_w = W / patch_size;
    const int mg     = merge_size;
    const int P      = patch_size;
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
                    int   k = 0;
                    for (int c = 0; c < channels; c++) {
                        for (int t = 0; t < T_patch; t++) {
                            const float * src = frame_at(t)
                                              + (size_t)c * H * W;
                            const int row_start = (hg * mg + mh) * P;
                            const int col_start = (wg * mg + mw) * P;
                            for (int py = 0; py < P; py++) {
                                const float * row = src + (size_t)(row_start + py) * W
                                                  + col_start;
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
void normalize_and_temporal_pad(const float * rgb_hwc, int H, int W, int channels,
                                int T_patch,
                                const float mean[3], const float std_[3],
                                std::vector<float> & out_tchw) {
    const size_t plane = (size_t)channels * H * W;
    out_tchw.assign(plane * T_patch, 0.0f);

    // Build the first frame: rescale (/255) then normalize per channel.
    // Re-layout from HWC → CHW.
    const float scale = 1.0f / 255.0f;
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
        std::memcpy(out_tchw.data() + (size_t)t * plane,
                    out_tchw.data(),
                    plane * sizeof(float));
    }
}

}  // namespace

bool preprocess_rgb(const uint8_t * rgb,
                    int height, int width, int channels,
                    const config & cfg,
                    result & out) {
    if (!rgb || height <= 0 || width <= 0) return false;
    if (channels < 3) {
        std::fprintf(stderr, "image_preproc: expected RGB(A) input, got %d channels\n", channels);
        return false;
    }
    const int factor = cfg.patch_size * cfg.merge_size;
    int rh = 0, rw = 0;
    if (!smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels, &rh, &rw)) {
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
                uint8_t * dst       = rgb_pure.data() + (size_t)(y * width + x) * kC;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
        rgb_use = rgb_pure.data();
    }

    // Resize → float32 HWC in [0, 255]. Bicubic + antialias matches the
    // torchvision v2 default that HF Qwen2VLImageProcessorFast uses.
    std::vector<float> resized((size_t)rh * rw * kC, 0.0f);
    bicubic_resize_u8_to_f32(rgb_use, height, width,
                              resized.data(), rh, rw, kC);

    // Normalize + temporal pad → (T_patch, C, H, W).
    std::vector<float> tchw;
    normalize_and_temporal_pad(resized.data(), rh, rw, kC,
                               cfg.temporal_patch_size,
                               cfg.mean, cfg.std, tchw);

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
            "smart_resize bug?\n", grid_h, grid_w, cfg.merge_size);
        return false;
    }
    const int n_patches = grid_h * grid_w;
    out.patches.assign((size_t)n_patches * row_dim, 0.0f);
    int gh = 0, gw = 0;
    patchify_qwen_layout(tchw.data(), kC, cfg.temporal_patch_size,
                         rh, rw, cfg.patch_size, cfg.merge_size,
                         out.patches.data(), &gh, &gw);
    out.n_patches = n_patches;
    out.row_dim   = row_dim;
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
        std::fprintf(stderr, "image_preproc: stbi_load failed for '%s': %s\n",
                     path, stbi_failure_reason());
        return false;
    }
    bool ok = preprocess_rgb(rgb, H, W, /*channels=*/3, cfg, out);
    stbi_image_free(rgb);
    return ok;
}

}  // namespace image_preproc
