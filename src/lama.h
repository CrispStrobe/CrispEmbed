// Experimental Big-LaMa CPU/F32 backend. No public C ABI or qualified model alias.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef CRISPEMBED_LAMA_TESTING
#include <array>
#include <functional>
#endif

namespace lama {

constexpr int min_dimension = 9;
constexpr int max_dimension = 2048;
constexpr size_t max_padded_pixels = 2 * 1024 * 1024;
constexpr size_t max_graph_bytes = size_t(3) * 1024 * 1024 * 1024;

struct context;

// Weight residency after init (see load_tensors in lama.cpp). Storage-only
// quantization dequantizes to F32 at load; the default-off gates
// CRISPEMBED_LAMA_QUANT_MATMUL and CRISPEMBED_LAMA_F16_CONV keep 1x1 kernels in
// their stored type (ggml_mul_mat) and k>1 conv kernels in F16 respectively.
struct load_info {
    int resident_tensors = 0;  // loaded in their stored type and shape
    int converted_tensors = 0; // dequantized/reshaped at load
    int matmul_tensors = 0;    // 1x1 kernels consumed by ggml_mul_mat
    int f16_conv_tensors = 0;  // k>1 kernels held as F16 for the direct conv
    size_t weight_bytes = 0;   // backend weight buffers
};

// Both initialization and inference require core_env::on("CRISPEMBED_LAMA").
// The caller supplies a locally converted big-lama-18-075-v1 F32 model, or a
// crispembed-quantize "lama-v1" F16/Q8_0/Q4_K derivative of it. The
// converter preserves checkpoint provenance; that is not a weights license grant.
context * init(const char * path, int n_threads, std::string & error);
void free(context * ctx);
bool describe(const context * ctx, load_info & info);

// Binary mask convention: 255 fills, 0 preserves the original RGB bytes.
// The whole frame is padded symmetrically on the right and bottom to multiples
// of eight. No image resize, tiling, or approximate global Fourier operation.
// A context retains weights and the most recent shape's graph; calls are serial.
// On failure output is unchanged; output may alias rgb.
bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error);

// Model-free boundary helpers. NCHW F32 input is [known-R,known-G,known-B,mask]
// with RGB in [0,1], not MI-GAN's signed normalization/channel order.
bool padded_dimensions(int width, int height, int & padded_width, int & padded_height, std::string & error);
bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error);
bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, int width, int height, uint8_t * output_rgb,
                    size_t output_bytes, std::string & error);

// Raw graph entry used by the reference driver. Width/height must already be
// multiples of eight, at least sixteen; the output is padded NCHW [1,3,H,W].
bool run(context * ctx, const float * input_nchw, size_t input_count, int width, int height, float * raw_nchw,
         size_t output_count, std::string & error);

#ifdef CRISPEMBED_LAMA_TESTING
namespace test {
// Shape is ggml [W,H,C,N]; payload is contiguous NCHW F32. The observer build
// retains every allocation owner and is excluded from the production archive.
using observer = std::function<bool(const std::string &, const float *, size_t, const std::array<int64_t, 4> &)>;
bool run_observed(context * ctx, const float * input_nchw, size_t input_count, int width, int height, float * raw_nchw,
                  size_t output_count, const observer & observe, size_t & graph_buffer_bytes, std::string & error);
} // namespace test
#endif

} // namespace lama
