// Portable device-resident spectral/sequence graph decompositions.
// No activation is read by the host and no CPU callback is part of these graphs.
#pragma once

#include "inpaint_graph.h"

namespace inpaint_graph {

// F32 WHCN real input -> [W/2+1,H,2*C,B], with channels [Re0,Im0,Re1,Im1,...].
// Each spatial axis is in [1,128]. Non-power-of-two axes use Bluestein graphs.
ggml_tensor * rfft2_ortho(Builder & builder, ggml_tensor * real);

// Inverse with the original real width supplied explicitly (odd widths are
// otherwise ambiguous). Applies the real projection used by irfftn even when
// learned spectral layers do not preserve Hermitian DC/Nyquist columns.
ggml_tensor * irfft2_ortho(Builder & builder, ggml_tensor * packed, int original_width);

// PixelHacker's causal raster-ordered recurrence, before the headwise RMSNorm,
// SiLU value gate and output projection. q/k/gate: [W,H,heads*dk,B]; v and output:
// [W,H,heads*dv,B]. dk and dv may differ; batches have independent zero states.
// Uses bounded chunks and per-key log-gate differences, never the square-head
// GGML_OP_GATED_LINEAR_ATTN operation or a host recurrent loop.
ggml_tensor * gla(Builder & builder, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * gate_logits,
                  int heads, float normalizer = 16.0f);

} // namespace inpaint_graph
