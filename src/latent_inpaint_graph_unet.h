// Persistent, backend-resident graph blueprints for the pinned author UNets.
// All activations use ggml WHCN storage: [width, height, channels, batch].
#pragma once

#include "inpaint_graph.h"
#include "latent_inpaint_unet.h"

namespace latent_inpaint {

// Build one reusable denoising evaluation. The caller supplies the nine latent
// channels, the original 320-channel sinusoidal time embedding, and ten learned
// 3072-channel category tokens per sample. Batch may be one or two; batch two
// evaluates unconditional/conditional CFG branches without host activation
// transfers. Time and token tensors must have the same batch as the latent.
// The caller owns graph allocation, input updates, execution, and model gates.
ggml_tensor * graph_denoise(inpaint_graph::Builder & builder, Variant variant, ggml_tensor * input9,
                            ggml_tensor * time_embedding_320, ggml_tensor * conditioning_tokens_3072x10);

namespace graph_detail {

// These parameter-free graph kernels are shared with model-free analytical
// tests. Queries have heads*dk channels; a single dk-channel key and dv-channel
// value sequence is shared across all lambda heads, as in the author modules.
ggml_tensor * lambda_content(inpaint_graph::Builder & builder, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                             int heads);

// positional is the batched 15x15 convolution of V, with shape
// [width,height,dk,dv*batch]. Each V plane is a separate convolution sample.
ggml_tensor * lambda_position(inpaint_graph::Builder & builder, ggml_tensor * q, ggml_tensor * positional, int heads,
                              int value_channels);

// relative retains the checkpoint's PyTorch [N,M,dk,1] ordering, represented
// in ggml as [1,dk,M,N]. It is shared by every sample and every query head.
ggml_tensor * lambda_relative(inpaint_graph::Builder & builder, ggml_tensor * q, ggml_tensor * v,
                              ggml_tensor * relative, int heads);

// FLA's fused output operation: headwise RMSNorm(eps=1e-6), shared learned
// head-channel scale, and SiLU(gate), with no normalization across heads.
ggml_tensor * gated_rms_norm(inpaint_graph::Builder & builder, ggml_tensor * x, ggml_tensor * gate,
                             ggml_tensor * weight, int heads);

} // namespace graph_detail
} // namespace latent_inpaint
