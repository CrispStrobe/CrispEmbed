// Portable device-resident spectral/sequence graph decompositions, plus
// separately gated native CPU candidates. GPU execution never uses callbacks
// or downloads an activation to execute one of these operators on the host.
#pragma once

#include "inpaint_graph.h"

namespace inpaint_graph {

// F32 WHCN real input -> [W/2+1,H,2*C,B], with channels [Re0,Im0,Re1,Im1,...].
// Each spatial axis is in [1,128]. Non-power-of-two axes use Bluestein graphs.
// Default-off CRISPEMBED_INPAINT_CPU_FFT=1 substitutes a threaded native CPU
// callback with cached radix-2/Bluestein geometry plans. It accesses resident
// CPU buffers directly; CUDA/Metal always retain the portable decomposition.
ggml_tensor * rfft2_ortho(Builder & builder, ggml_tensor * real);

// Inverse with the original real width supplied explicitly (odd widths are
// otherwise ambiguous). Applies the real projection used by irfftn even when
// learned spectral layers do not preserve Hermitian DC/Nyquist columns.
ggml_tensor * irfft2_ortho(Builder & builder, ggml_tensor * packed, int original_width);

// PixelHacker's causal raster-ordered recurrence, before the headwise RMSNorm,
// SiLU value gate and output projection. q/k/gate: [W,H,heads*dk,B]; v and output:
// [W,H,heads*dv,B]. dk and dv may differ; batches have independent zero states.
// By default, uses bounded chunks and per-key log-gate differences. A default-
// off CPU-only CRISPEMBED_INPAINT_CPU_GLA=1 candidate lowers rectangular heads
// into independent dk-wide value slices for ggml's existing recurrent kernel.
// CUDA/Metal always retain the portable chunk graph. Neither path downloads
// activations or executes a custom host callback.
// The production baseline is 64 tokens. Explicit 8/16/32-token chunks permit
// controlled operator comparisons without changing the default model path.
ggml_tensor * gla(Builder & builder, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * gate_logits,
                  int heads, float normalizer = 16.0f, int chunk_size = 64);

} // namespace inpaint_graph
