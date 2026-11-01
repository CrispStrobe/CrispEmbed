// Experimental native Big-LaMa, CPU/F32, with exact known-RGB preservation.
// The original architecture is fixed by the source/checkpoint contract.
#pragma once

#include "latent_inpaint_ops.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace lama {
constexpr int min_dimension = 9;
constexpr int max_dimension = 1024;
constexpr size_t max_pixels = size_t(max_dimension) * max_dimension;

struct context;
// Requires CRISPEMBED_LAMA=1 and the full GGUF SHA-256 in the converter receipt.
// The checkpoint/source/schema pins are independently checked from metadata.
context * init(const std::string & path, const std::string & expected_gguf_sha256, int threads, std::string & error);
void free(context * model);
bool process_rgb8(context * model, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error);

namespace detail {
using latent_inpaint::Tensor;
// Actual production helpers; the input is symmetrically padded to a multiple
// of eight, while convolution padding follows PyTorch's reflect convention.
Tensor reflect_pad(const Tensor & input, int padding);
Tensor preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width,
                       int height);
void composite_rgb8(const Tensor & prediction, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask,
                    size_t mask_bytes, int width, int height, uint8_t * output, size_t output_bytes);
} // namespace detail

#ifdef CRISPEMBED_LAMA_TESTING
using Observer = void (*)(void *, const std::string &, const latent_inpaint::Tensor &);
void set_observer(context * model, Observer callback, void * opaque);
// Runs the same generator as process_rgb8, starting from its four-channel
// masked RGB+white-fill-mask input. Intended for original-reference replay.
bool run_reference(context * model, const latent_inpaint::Tensor & input, latent_inpaint::Tensor & prediction,
                   std::string & error);
#endif
} // namespace lama
