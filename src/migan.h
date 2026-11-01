// Experimental fixed-256/512 MI-GAN F32 backend. No public C ABI/registry.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace migan {

constexpr int image_size = 256;
constexpr size_t rgb_elements = size_t(image_size) * image_size * 3;
constexpr size_t mask_elements = size_t(image_size) * image_size;
constexpr size_t input_elements = size_t(image_size) * image_size * 4;

struct context;

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
context * init(const char * path, const char * expected_checkpoint_sha256, int n_threads, std::string & error);
// The existing init() remains the reviewed fixed256 profile. The new fixed512
// profile additionally requires CRISPEMBED_MIGAN512=1; it has no parity receipt
// until the original512/native acceptance panel is run. Digests must be supplied
// from the independent original capture, never copied from untrusted metadata.
context * init_variant(const char * path, const char * expected_checkpoint_sha256, int resolution, int n_threads,
                       std::string & error);
int image_resolution(const context * ctx);
void free(context * ctx);
bool run(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
         std::string & error);

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
