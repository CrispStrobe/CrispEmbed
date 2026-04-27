// image_preprocess.h — in-process image preprocessing for BidirLM-Omni /
// Qwen2VL-style vision towers. Removes the runtime dependency on the
// Python `transformers` AutoImageProcessor for the CLI / mobile builds.
//
// Pipeline (mirrors HF Qwen2VLImageProcessorFast):
//   1. Load image (stb_image, JPG/PNG/etc).
//   2. RGB-convert if needed.
//   3. smart_resize: round H and W to multiples of `factor = patch_size *
//      merge_size`, clamping the total pixel count to [min_pixels, max_pixels]
//      while keeping the aspect ratio as close as possible.
//   4. Resize the image to the new dimensions (Catmull-Rom bicubic with
//      antialiasing for downscale — same kernel/sampling torchvision v2
//      uses by default, so output should match HF byte-tight on the resize
//      step alone).
//   5. Rescale to [0, 1] and normalize per-channel with OpenAI CLIP mean/std.
//   6. Pad on the temporal axis to a multiple of `temporal_patch_size` by
//      repeating the last frame.
//   7. Patchify into (n_patches, in_C * T_patch * patch_size * patch_size)
//      flattened rows + (1, 3) grid_thw triplet (t=1, h, w).
//
// PARITY: bicubic + antialias matches torchvision v2's default for uint8
// tensor inputs (with rounded uint8 round-trip after resize). With the
// correct mean/std/min_pixels/max_pixels for the model, cosine vs HF's
// `Qwen2VLImageProcessorFast` lands at ~0.99998 on typical photographs;
// the residual is sub-pixel bicubic kernel weight quantization (PyTorch
// uses int16-quantized weights for uint8 input; we use float weights).

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <string>
#include <vector>

namespace image_preproc {

// Default OpenAI CLIP normalization (per-channel RGB) — used by some VL models.
constexpr float kCLIPMean[3] = { 0.48145466f, 0.4578275f,  0.40821073f };
constexpr float kCLIPStd[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };

// Defaults for BidirLM-Omni / Qwen2-VL: mean=std=0.5 (maps [0,1] → [-1,1]),
// min_pixels=256² (`shortest_edge`), max_pixels=1024² (`longest_edge`). Match
// the values in the model's `preprocessor_config.json`. Other VL models
// (e.g. CLIP-based) override mean/std via the `config` struct.
struct config {
    int   patch_size          = 16;
    int   temporal_patch_size = 2;
    int   merge_size          = 2;
    int   min_pixels          = 256 * 256;     // 65536
    int   max_pixels          = 1024 * 1024;   // 1048576
    float mean[3]             = { 0.5f, 0.5f, 0.5f };
    float std[3]              = { 0.5f, 0.5f, 0.5f };
};

struct result {
    // (n_patches, in_C * T_patch * patch_size * patch_size) row-major float32.
    // For the BidirLM-Omni defaults (in_C=3, T=2, P=16) the row width is 1536.
    std::vector<float> patches;
    int32_t grid_thw[3] = { 1, 0, 0 };  // (t, h_in_patches, w_in_patches)
    int     n_patches = 0;
    int     row_dim   = 0;
    // Resized H, W in pixels — useful for diagnostics / parity tests.
    int     resized_h = 0;
    int     resized_w = 0;
};

// HF parity smart_resize: round H and W to the nearest multiple of `factor`,
// clamp to [min_pixels, max_pixels], preserve aspect ratio.
// Returns (h_bar, w_bar) via out params. Aborts (returns false) if the
// aspect ratio exceeds 200:1.
bool smart_resize(int height, int width,
                  int factor, int min_pixels, int max_pixels,
                  int * out_h, int * out_w);

// Run the full preprocessing pipeline on a JPG/PNG/BMP/etc. file.
// Returns false on disk read / decode failure or absurd aspect ratio.
bool preprocess_file(const char * path,
                     const config & cfg,
                     result & out);

// Same, but takes pre-decoded RGB (or RGBA — alpha is dropped) interleaved
// uint8 pixels. Useful when the caller already has the bytes (e.g. from a
// Python wrapper).
bool preprocess_rgb(const uint8_t * rgb,
                    int height, int width, int channels,
                    const config & cfg,
                    result & out);

}  // namespace image_preproc
