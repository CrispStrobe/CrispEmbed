// bidirlm_vision.h — internal vision-tower runtime for BidirLM-Omni.
//
// Loads visual.* tensors from the GGUF and produces:
//   - image_embeds:        (n_merged_tokens, output_dim=2048)
//   - deepstack_features:  3 × (n_merged_tokens, output_dim=2048)
//
// `n_merged_tokens` = (H_patches * W_patches) / spatial_merge_size² for a single image.
// For variable-resolution support the input is a flat tensor of patches plus a
// grid_thw triplet that tells us where the image boundaries are.
//
// The vision tower lives entirely in this header/source pair (no shared lib —
// crisp_audio's pattern would buy nothing here since BidirLM is the only
// consumer of this architecture today).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace bidirlm_vision {

struct hparams {
    uint32_t depth = 24;
    uint32_t hidden_size = 1024;
    uint32_t intermediate_size = 4096;
    uint32_t num_heads = 16;
    uint32_t in_channels = 3;
    uint32_t patch_size = 16;
    uint32_t spatial_merge_size = 2;
    uint32_t temporal_patch_size = 2;
    uint32_t out_hidden_size = 2048;
    uint32_t num_position_embeddings = 2304;  // 48² grid
    std::vector<int> deepstack_indexes = {8, 16, 24};
};

struct block {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct merger_weights {
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct model {
    hparams hp;

    ggml_tensor *patch_embed_w = nullptr, *patch_embed_b = nullptr;
    ggml_tensor *pos_embed_w = nullptr;

    std::vector<block> blocks;

    merger_weights merger;
    std::vector<merger_weights> deepstack;
};

struct context {
    model m;
    ggml_context * model_ctx = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
    int verbosity = 1;
};

// Load the vision tower from the same GGUF file the rest of the BidirLM
// model lives in. Returns false if the file does not contain a vision tower
// (e.g. text-only GGUFs). Existing backends from the parent context can be
// shared in via `shared_backend` to avoid double-allocating on the GPU.
bool load(context& ctx, const char* gguf_path, ggml_backend_t shared_backend,
          int n_threads, int verbosity);

void free_(context& ctx);

// Run the vision tower forward on a flat (n_patches, in_channels,
// temporal_patch_size, patch_size, patch_size) tensor — caller pre-flattens
// from PIL via the Python preprocessor.
//
// Returns:
//   image_embeds — malloc'd (n_merged_tokens, output_dim) row-major float
//   deepstack    — malloc'd (3, n_merged_tokens, output_dim) row-major float
//                  (one slab per deepstack hook)
//   *out_n_merged, *out_dim set on return
//
// On failure returns nullptr and frees nothing it didn't allocate.
struct encode_result {
    float* image_embeds = nullptr;     // (n_merged, output_dim)
    float* deepstack    = nullptr;     // (n_deepstack, n_merged, output_dim)
    int n_merged = 0;
    int output_dim = 0;
    int n_deepstack = 0;
};

bool encode(context& ctx,
            const float* pixel_patches, int n_patches,
            const int32_t* grid_thw, int n_images,
            encode_result& out,
            bool include_deepstack = true);

void encode_result_free(encode_result& r);

}  // namespace bidirlm_vision
