// Experimental fixed-256/512 MI-GAN prototype. No public C ABI/registry.
// The reviewed CPU/F32 graph is the reference; GPU execution, non-F32
// pointwise weights (docs/migan-metal.md), the persistent graph and inpaint-v1
// precision variants (docs/inpainting-precision.md) are separately gated additions.
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

// A context snapshots CRISPEMBED_INPAINT_GRAPH at init. Unset/empty/0 keeps
// the original CPU implementation. The opt-in graph uses the process backend
// preference, uploads weights once, and retains its graph and allocator. CUDA
// and Metal requests fail if unavailable or any graph operation is unsupported.
// F32 accumulation is explicit on every matmul, including on Metal, unless
// CRISPEMBED_INPAINT_FAST=1 also opts into backend-default matrix precision
// (Metal may reduce operands internally to F16; this needs separate parity).
// CRISPEMBED_MIGAN_CPU_KERNELS=1 independently enables F32 CPU depthwise and
// fused activation kernels on the persistent CPU graph. It never changes
// weights or selects a CPU fallback for a requested GPU backend.
// Independent contexts may run concurrently; overlapping calls on one context
// fail without writing output. The caller must not free a context in use.
struct execution_stats {
    std::string backend;
    bool persistent = false;
    bool fast = false;
    bool cpu_kernels = false;
    bool cpu_plan_cached = false;
    bool cpu_threadpool_reused = false;
    int resolution = 0;
    size_t graph_nodes = 0;
    size_t graph_buffer_bytes = 0;
    size_t weight_buffer_bytes = 0;
    uint64_t graph_builds = 0;
    uint64_t graph_allocations = 0;
    uint64_t runs = 0;
    uint64_t reuses = 0;
    uint64_t weight_uploads = 0;
    uint64_t weight_upload_bytes = 0;
    uint64_t input_uploads = 0;
    uint64_t input_upload_bytes = 0;
    uint64_t output_downloads = 0;
    uint64_t output_download_bytes = 0;
    double last_build_ms = 0;
    double last_compute_ms = 0;
    double total_compute_ms = 0;
    double last_run_ms = 0;
};
bool get_execution_stats(context * ctx, execution_stats & stats, std::string & error);

// Both init and run require core_env::on("CRISPEMBED_MIGAN"). The expected
// checkpoint digest must come from a reviewed original capture, not the file.
// Non-F32 (F16/Q8_0/Q4_K pointwise) files additionally need
// core_env::on("CRISPEMBED_MIGAN_QUANT") and the reviewed F32 GGUF SHA.
context * init(const char * path, const char * expected_checkpoint_sha256, int n_threads, std::string & error);
// The existing init() remains the reviewed fixed256 profile. The new fixed512
// profile additionally requires CRISPEMBED_MIGAN512=1; it has no parity receipt
// until the original512/native acceptance panel is run. Digests must be supplied
// from the independent original capture, never copied from untrusted metadata.
context * init_variant(const char * path, const char * expected_checkpoint_sha256, int resolution, int n_threads,
                       std::string & error);
// Fixed256 init with an explicit device choice and, for migan-pointwise-v1
// derived files, the reviewed F32 GGUF digest they were derived from.
context * init_ex(const char * path, const char * expected_checkpoint_sha256, int n_threads, device_choice device,
                  const char * expected_source_gguf_sha256, std::string & error);
// Both of the above, for either resolution.
context * init_variant_ex(const char * path, const char * expected_checkpoint_sha256, int resolution, int n_threads,
                          device_choice device, const char * expected_source_gguf_sha256, std::string & error);
int image_resolution(const context * ctx);
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
// Fixed-size variants accept only256 or512. The original helpers above always
// remain256 so callers cannot silently switch profile by changing buffer sizes.
bool preprocess_rgb8_variant(int resolution, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask,
                             size_t mask_bytes, int width, int height, float * input_nchw, size_t input_count,
                             std::string & error);
bool composite_rgb8_variant(int resolution, const float * raw_nchw, size_t raw_count, const uint8_t * rgb,
                            size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb,
                            size_t output_bytes, std::string & error);

} // namespace migan
