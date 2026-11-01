// CPU real FFT primitives for the gated Big-LaMa FourierUnit implementation.
#pragma once

#include "latent_inpaint_ops.h"

namespace lama {
namespace detail {
using latent_inpaint::Tensor;

// Exactly the FourierUnit channel packing: [C,H,W] real ->
// [2*C,H,W/2+1], channels real0,imag0,real1,imag1,...; norm="ortho".
// Radix-2 FFT and Bluestein convolution cover both power-of-two and arbitrary
// spatial lengths. There is no quadratic production DFT fallback.
Tensor rfft2_ortho(const Tensor & real);
Tensor irfft2_ortho(const Tensor & packed, int real_width);

} // namespace detail
} // namespace lama
