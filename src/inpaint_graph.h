// Device-resident, reusable ggml graphs for the opt-in inpainting execution path.
#pragma once

#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace inpaint_graph {

// Caller owns the result. Explicit CUDA/Metal requests never fall back to CPU.
ggml_backend_t select_backend(int threads, std::string & name, std::string & error);
void validate_graph_backend(ggml_backend_t backend, ggml_cgraph * graph);

// Resolve the pinned CPU registry's pool API, including dynamic backend builds.
// Older modules without all three functions retain ggml's ordinary threading.
class CpuPool {
public:
    CpuPool(ggml_backend_t backend, int threads);
    ~CpuPool();
    CpuPool(const CpuPool &) = delete;
    CpuPool & operator=(const CpuPool &) = delete;
    bool active() const { return pool_ != nullptr; }

private:
    ggml_threadpool_t pool_ = nullptr;
    void (*free_)(ggml_threadpool_t) = nullptr;
};

struct Device {
    ggml_backend_t backend = nullptr;
    std::string name;
    bool gpu = false;
    bool fast = false;
    explicit Device(int threads);
    ~Device();
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;

private:
    std::unique_ptr<CpuPool> cpu_pool_;
};

using ShapeMap = std::map<std::string, std::vector<int64_t>>;
struct Weights {
    Device & device;
    core_gguf::WeightLoad loaded;
    Weights(Device & device, const char * path, const ShapeMap & schema);
    ~Weights();
    Weights(const Weights &) = delete;
    Weights & operator=(const Weights &) = delete;
};

// Explicitly allocated state shared by encoder, recurrent denoiser and decoder.
// Graph outputs copy directly into these tensors on the selected backend.
class State {
public:
    explicit State(Device & device, size_t capacity = 32);
    ~State();
    ggml_tensor * tensor(const std::string & name, const std::vector<int64_t> & shape);
    void allocate();
    void upload(ggml_tensor * tensor, const std::vector<float> & data);
    Device & device;

private:
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class Builder {
public:
    explicit Builder(Device & device, Weights * weights = nullptr, size_t capacity = 65536);
    ~Builder();
    Builder(const Builder &) = delete;
    Builder & operator=(const Builder &) = delete;
    ggml_context * ctx = nullptr;
    Device & device;
    bool fast() const { return device.fast; }
    ggml_tensor * weight(const std::string & name) const;
    bool has(const std::string & name) const;
    ggml_tensor * parameter(const std::string & name, const std::vector<int64_t> & shape);
    ggml_tensor * constant(const std::string & name, const std::vector<int64_t> & shape,
                           const std::vector<float> & data);
    ggml_tensor * constant_i32(const std::string & name, const std::vector<int64_t> & shape,
                               const std::vector<int32_t> & data);
    ggml_tensor * zeros(const std::string & name, const std::vector<int64_t> & shape);
    // All image/sequence operators use contiguous WHCN tensors.
    ggml_tensor * conv(const std::string & name, ggml_tensor * x, int stride = 1, int padding = 0, int groups = 1,
                       int dilation = 1);
    // Original IOHW weights are prepared once for bounded im2col/GEMM. The
    // supported profile has square kernels <=15, stride1..4, padding<kernel,
    // output_padding<stride and one to eight independent samples.
    ggml_tensor * conv_transpose(const std::string & name, ggml_tensor * x, int stride = 2, int padding = 1,
                                 int output_padding = 1);
    ggml_tensor * linear(const std::string & name, ggml_tensor * x, bool bias = true);
    ggml_tensor * group_norm(const std::string & name, ggml_tensor * x, int groups = 32, float eps = 1e-5f);
    ggml_tensor * layer_norm(const std::string & name, ggml_tensor * x, float eps = 1e-5f);
    ggml_tensor * batch_norm(const std::string & name, ggml_tensor * x, float eps = 1e-5f);
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int heads = 1);
    ggml_tensor * channel_slice(ggml_tensor * x, int start, int count);
    ggml_tensor * pad(ggml_tensor * x, int left, int right, int top, int bottom);
    ggml_tensor * reflect_pad(ggml_tensor * x, int padding);
    ggml_tensor * resize(ggml_tensor * x, int height, int width);
    void mark(const std::string & name, ggml_tensor * x);
    void finish(const std::vector<ggml_tensor *> & outputs, const std::string & label);
    void upload(ggml_tensor * tensor, const std::vector<float> & data);
    void compute();
    std::vector<float> download(ggml_tensor * tensor);
    size_t allocation_bytes() const;
    size_t nodes() const;
    uint64_t runs() const { return runs_; }
    uint64_t upload_bytes() const { return upload_bytes_; }
    uint64_t download_bytes() const { return download_bytes_; }
    double compute_ms() const { return compute_ms_; }
    // Number of optimized CPU depthwise nodes included in the finalized graph.
    // Each compute executes this many logical custom nodes; a node invokes
    // its callback on multiple workers. A gate alone does not prove selection.
    size_t cpu_depthwise_nodes() const { return cpu_depthwise_nodes_; }
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
    using Observer = void (*)(void *, const std::string &, ggml_tensor *, const std::vector<float> &);
    void set_observer(Observer callback, void * opaque);
#endif
private:
    ggml_tensor * conv_kernel(const std::string & name, ggml_tensor * kernel, ggml_tensor * bias, ggml_tensor * x,
                              int stride, int padding, int groups, int dilation);
    struct Initializer {
        ggml_tensor * tensor;
        std::vector<uint8_t> bytes;
        bool zero = false;
    };
    Weights * weights_;
    size_t capacity_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t cpu_plan_ = nullptr;
    std::string label_;
    bool ready_ = false;
    std::vector<Initializer> initializers_;
    std::map<std::string, ggml_tensor *> constants_;
    uint64_t runs_ = 0, upload_bytes_ = 0, download_bytes_ = 0;
    double compute_ms_ = 0;
    std::vector<ggml_tensor *> cpu_depthwise_ops_;
    size_t cpu_depthwise_nodes_ = 0;
    // ABI-identical object size in ordinary/test archives. Only the test
    // translation unit defines or allocates the pointed-to observer state.
    void * testing_state_ = nullptr;
};
} // namespace inpaint_graph
