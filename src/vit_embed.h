// vit_embed.h — Standalone ViT image encoder for SigLIP / CLIP.
//
// Loads a GGUF produced by convert-siglip-to-gguf.py and runs the
// vision transformer forward pass: patch_embed → N × (LN→Attn→LN→MLP)
// → post_ln → attention_pool → embedding vector.
//
// Usage:
//   vit_embed::context ctx;
//   if (!vit_embed::load(&ctx, "siglip-base.gguf")) return 1;
//
//   // pixels: float[3 * H * W], RGB, normalized to [0,1] then
//   // (x - mean) / std per channel (constants in GGUF metadata).
//   std::vector<float> emb = vit_embed::encode(&ctx, pixels.data(), H, W);
//   // emb.size() == hidden_size (768 for SigLIP-base)
//
//   vit_embed::free(&ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vit_embed {

struct context;

// Load ViT GGUF. Returns true on success.
bool load(context** ctx, const char* path, int n_threads = 1);

// Encode an image. pixels is [3, H, W] in CHW order, already normalized
// (subtract image_mean, divide by image_std). H and W must match the
// model's image_size. Returns embedding vector.
std::vector<float> encode(context* ctx, const float* pixels, int H, int W);

// Get embedding dimension.
int dim(const context* ctx);

// Get expected image size.
int image_size(const context* ctx);

// Encode from an image file (JPG/PNG/BMP). Handles resize + normalize.
// Returns empty vector on failure.
std::vector<float> encode_file(context* ctx, const char* image_path);

// Free resources.
void free(context* ctx);

} // namespace vit_embed
