// Experimental native Moebius / PixelHacker image-and-mask inference.
#pragma once

#include "latent_inpaint_ops.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace latent_inpaint {
struct Options {
    std::string model; // "moebius" or "pixelhacker"
    std::string denoiser_path, vae_path;
    std::string denoiser_sha256, vae_sha256; // converter receipt digests
    int threads = 1;
    int steps = 20;
    float guidance = 2.0f;
    float noise_offset = 0.0357f;
    uint64_t seed = 0;
    // Explicit approximation for larger/smaller images: reuse native512
    // overlapping tiles. False requires exactly512 and matches reference size.
    bool tiled = false;
};
struct context;
context * init(const Options & options, std::string & error);
void free(context * ctx);
bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error);

namespace detail {
// These are the real sampler's pure primitives, also used by hosted controls.
std::vector<int> ddim_timesteps(int requested_steps, const std::string & model);
std::vector<float> ddim_alphas();
Tensor ddim_step(const Tensor & latent, const Tensor & noise, int timestep, int previous_timestep,
                 const std::vector<float> & alphas);
Tensor preprocess_rgb8(const uint8_t * rgb, size_t bytes);
void composite_rgb8(const Tensor & decoded, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask,
                    size_t mask_bytes, uint8_t * output, size_t output_bytes);
} // namespace detail

#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
struct ReferenceNoise {
    Tensor posterior_image, posterior_masked, latent_noise, offset_noise;
};
using Observer = void (*)(void *, const std::string &, const Tensor &);
void set_observer(context * ctx, Observer callback, void * opaque);
// RGB is already in[-1,1], mask is binary0/1, both512. This entry executes
// the SAME pipeline as process_rgb8 with recorded original noise tensors.
bool run_reference(context * ctx, const Tensor & image, const Tensor & mask, const ReferenceNoise & noise,
                   Tensor & decoded, std::string & error);
#endif
} // namespace latent_inpaint
