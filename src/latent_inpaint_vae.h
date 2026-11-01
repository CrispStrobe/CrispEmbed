#pragma once
#include "latent_inpaint_ops.h"

namespace latent_inpaint {
constexpr float vae_scaling_factor = 0.13025f;
// SDXL AutoencoderKL at the pinned PixelHacker VAE checkpoint. Both models
// share it. Encoder returns [mean,logvar] in channel order, before sampling.
Tensor vae_encode_moments(Engine & engine, const Tensor & rgb);
Tensor vae_sample(const Tensor & moments, const Tensor & posterior_noise);
// Input is unscaled latent; output is RGB in the training [-1,1] domain.
Tensor vae_decode(Engine & engine, const Tensor & latent);
} // namespace latent_inpaint
