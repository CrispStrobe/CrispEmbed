// minicpm_ocr_vision.h - MiniCPM-OCR-1.0 SigLIP-so400m-style vision tower +
// 4x (2x2 merge) projector, and image preprocessing (overview + slices).
// Experimental, default-off: only reachable through minicpm_ocr.cpp behind
// CRISPEMBED_MINICPM_OCR. See docs/minicpm-ocr-inference-spec.md.
#pragma once

#include "core/gguf_loader.h"
#include "minicpm_ocr_tiles.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minicpm_ocr {

struct vision_layer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *q_w = nullptr, *q_b = nullptr, *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr, *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr, *fc2_b = nullptr;
};

struct vision_weights {
    int width = 1152, heads = 16, patch = 14, pos_grid = 70, merge = 2, out_width = 1024;
    float eps = 1e-6f, merger_eps = 1e-6f;
    ggml_tensor *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr;
    std::vector<vision_layer> layers;
    ggml_tensor *post_w = nullptr, *post_b = nullptr;
    ggml_tensor *mln_w = nullptr, *mln_b = nullptr; // merger LayerNorm(4608)
    ggml_tensor *m1_w = nullptr, *m1_b = nullptr;   // Linear 4608 -> 4608
    ggml_tensor *m2_w = nullptr, *m2_b = nullptr;   // Linear 4608 -> 1024
};

// Binds v.* / m.* tensors and validates every shape. On failure `error` names
// the first missing or mis-shaped tensor.
bool map_vision(const core_gguf::tensor_map & tensors, vision_weights & w, std::string & error);

// One preprocessed image (overview or slice): normalized CHW float in [-1, 1].
struct image_part {
    int width = 0, height = 0, tokens = 0;
    bool overview = false;
    int row = 0, col = 0;
    std::vector<float> chw;
};

// Resize/slice/normalize per minicpm_ocr_tiles::plan. `rgb` is HWC uint8.
// Resample: torch CPU uint8 antialiased bicubic (core/torch_bicubic_u8.h), as
// tvF.resize runs for the TorchvisionBackend processor in transformers 5.21.0.
bool preprocess(const uint8_t * rgb, int width, int height, const minicpm_ocr_tiles::config & cfg,
                minicpm_ocr_tiles::plan & plan, std::vector<image_part> & parts);

// Optional per-image intermediate captures (rows = patches, row-major).
struct vision_debug {
    bool enabled = false;
    std::vector<float> embeddings, layer0, layer6, layer26, post_ln, merged;
};

// Optional wall-clock split of one encode() call (CRISPEMBED_MINICPM_OCR_TIMING):
// host im2col + graph build/alloc/upload, the SigLIP encoder (through post_layernorm)
// and the 2x2 merger/projector. With timing the encoder and merger run as two
// graphs (same operators on the same F32 values); without it, one graph as before.
struct vision_timing {
    double setup_ms = 0, encoder_ms = 0, projector_ms = 0, readback_ms = 0;
};

// Runs one image as an independent sequence (no cross-image attention) and
// appends its projector output ([tokens, out_width] row-major) to `merged`.
bool encode(const vision_weights & w, ggml_backend_t backend, const image_part & image, int threads,
            std::vector<float> & merged, vision_debug * debug, vision_timing * timing = nullptr);

} // namespace minicpm_ocr
