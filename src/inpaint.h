// Internal, opt-in MI-GAN image wrapper. This is not a stable public C ABI.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace inpaint {

constexpr int tile_size = 256;
constexpr int tile_stride = 192;
constexpr int max_dimension = 32768;
constexpr size_t max_pixels = size_t(64) * 1024 * 1024;
constexpr const char * checkpoint_sha256 = "8b82b2e82fc8e5a2e1f06827594aac1a5d66a9ca41e24199ddd969847788097f";
constexpr const char * model_sha256 = "b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b";
constexpr size_t model_bytes = 25185504;

struct context;

// Serial reuse keeps the weights loaded across frames. Both init/process need
// CRISPEMBED_MIGAN=1. Only the reviewed fixed256 CPU/F32 GGUF is accepted.
context * init(const char * path, int n_threads, std::string & error);
void free(context * ctx);

// Contiguous RGB8 HWC and a same-size binary mask: 0=keep, 255=fill. Output
// dimensions never change. Known pixels are byte-exact. Native256 tiles use
// 64-pixel overlap; small dimensions edge-replicate RGB/mask, without resizing.
// Output is unchanged on failure and may alias the input. Empty masks perform
// no model calls. One context must not be used concurrently.
bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error);

namespace detail {

// Pure image plumbing shared by production and model-free contract tests. The
// callback consumes exact256 RGB/mask tiles and supplies composited RGB bytes.
// It is not a second inference backend or a public model injection interface.
using tile_processor = bool (*)(void * opaque, const uint8_t * rgb, const uint8_t * fill_mask, uint8_t * output_rgb,
                                std::string & error);
bool process_tiles_rgb8(tile_processor process_tile, void * opaque, const uint8_t * rgb, size_t rgb_bytes,
                        const uint8_t * fill_mask, size_t mask_bytes, int width, int height, uint8_t * output_rgb,
                        size_t output_bytes, std::string & error);

} // namespace detail
} // namespace inpaint
