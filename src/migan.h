// Experimental fixed-256 MI-GAN prototype. No public C ABI/registry.
// The reviewed CPU/F32 graph is the reference; GPU execution and non-F32
// pointwise weights are separately gated additions (docs/migan-metal.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace migan {

constexpr int image_size = 256;
constexpr size_t rgb_elements = size_t(image_size) * image_size * 3;
constexpr size_t mask_elements = size_t(image_size) * image_size;
constexpr size_t input_elements = size_t(image_size) * image_size * 4;

// Quantized/F16 containers carry this policy name and the reviewed F32 GGUF
// SHA they were derived from. Only *.conv2.weight pointwise matrices change type.
constexpr const char * quant_policy = "migan-pointwise-v1";

struct context;

enum class device_choice {
    automatic, // CRISPEMBED_MIGAN_GPU gate decides (see gpu_default()).
    cpu,       // The reviewed CPU backend, never a GPU device.
    gpu,       // crispasr_init_gpu_backend(); CPU fallback only if no GPU or an op is unsupported.
};

// True when automatic device selection picks a GPU. GPU is the default unless
// CRISPEMBED_MIGAN_GPU=0 (explicitly_off); see docs/migan-metal.md for the A/B.
bool gpu_default();

// Both init and run require core_env::on("CRISPEMBED_MIGAN"). The expected
// checkpoint digest must come from a reviewed original capture, not the file.
// Non-F32 (F16/Q8_0/Q4_K pointwise) files additionally need
// core_env::on("CRISPEMBED_MIGAN_QUANT") and the reviewed F32 GGUF SHA.
context * init(const char * path, const char * expected_checkpoint_sha256, int n_threads, std::string & error);
context * init_ex(const char * path, const char * expected_checkpoint_sha256, int n_threads, device_choice device,
                  const char * expected_source_gguf_sha256, std::string & error);
void free(context * ctx);
// Name of the backend actually executing the graph ("CPU", "MTL0", ...).
const char * backend_name(const context * ctx);
bool uses_gpu(const context * ctx);
// Weight precision label: "f32" for the reviewed file, else the dominant pointwise type.
const char * weight_precision(const context * ctx);

bool run(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
         std::string & error);
// Batch of `batch` contiguous [1,4,256,256] inputs → [1,3,256,256] outputs in
// one graph execution (one upload, one compute, one download).
bool run_batch(context * ctx, const float * input_nchw, size_t batch, float * raw_nchw, std::string & error);
// Largest batch run_batch accepts for this context (CRISPEMBED_MIGAN_BATCH, default per device).
int max_batch(const context * ctx);

// Fixed-size binary white=fill boundary. These pure contract helpers do not
// load or execute a model. No resize, mask threshold or implicit channel cast.
bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error);
bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb, size_t output_bytes,
                    std::string & error);

} // namespace migan
