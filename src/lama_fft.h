#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

// CPU-only, orthonormal real 2D transforms used by LaMa's FourierUnit.
// ggml axes are [width,height,channels,batch]. The packed spectrum has axes
// [width/2+1,height,2*channels,batch], with real/imaginary channel planes adjacent.
namespace lama_fft {

// One status may be shared by all Fourier nodes in a graph. Its lifetime must
// extend through every graph computation; reset it before starting a new run.
// No callback throws through the C ggml API. A failed plane contains NaNs and
// the first error remains available here, including allocation failures.
struct execution_status {
    std::atomic<int> code{ 0 };

    void reset() noexcept { code.store(0, std::memory_order_relaxed); }
    bool ok() const noexcept { return code.load(std::memory_order_relaxed) == 0; }
    const char * message() const noexcept;
};

// Return nullptr for an unsupported shape or input type. These nodes support
// strided F32 input, produce contiguous F32 output, and require the CPU backend.
// Each spatial axis is limited to 8192, each plane to 1M values and the entire
// real tensor to 64M values. Concurrent scratch is bounded independently.
ggml_tensor * rfft2(ggml_context * ctx, ggml_tensor * input, execution_status * status = nullptr);
ggml_tensor * irfft2(ggml_context * ctx, ggml_tensor * input, int64_t original_width,
                     execution_status * status = nullptr);

// Contiguous equivalents for numerical contracts and small operator probes.
// Counts are in floats, not bytes. Input and output must not overlap. Functions
// clear error on success and reject nonfinite input/output. Output contents are
// unspecified on failure; callers must not publish a failed result.
bool rfft2_f32(const float * input, size_t input_count, float * output, size_t output_count, int64_t width,
               int64_t height, int64_t channels, int64_t batch, std::string & error);
bool irfft2_f32(const float * input, size_t input_count, float * output, size_t output_count, int64_t width,
                int64_t height, int64_t channels, int64_t batch, std::string & error);

} // namespace lama_fft
