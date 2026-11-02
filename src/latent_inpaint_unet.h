// Native inference blueprints for the fixed-512 author Moebius/PixelHacker
// checkpoints. CPU/F32 is experimental; source preparation is not model parity.
#pragma once

#include "latent_inpaint_ops.h"

namespace latent_inpaint {

enum class Variant { moebius, pixelhacker };

// The nine CHW channels are [noisy latent(4), white=fill mask(1), masked
// image latent(4)]. conditional=true selects embedding rows 0..9; false
// selects rows 10..19. Each call has a fresh GLA recurrent state. The caller
// combines the two outputs using the reference classifier-free guidance rule.
// Requires the corresponding value-parsed, default-off environment gate.
// Throws on a malformed tensor/checkpoint or an unsupported geometry.
Tensor denoise(Engine & engine, Variant variant, const Tensor & input9, int timestep, bool conditional);

namespace detail {

// Parameter-free inference kernels are shared by the real network and the
// hosted component tests. CHW storage represents raster-ordered tokens.
Tensor timestep_embedding(int timestep, int channels = 320);
Tensor gla_recurrent(const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & gate_logits, int heads,
                     float gate_normalizer = 16.0f);
Tensor lambda_content(const Tensor & q, const Tensor & k, const Tensor & v, int heads);
Tensor lambda_relative(const Tensor & q, const Tensor & v, const Weight & relative, int heads);

} // namespace detail
} // namespace latent_inpaint
