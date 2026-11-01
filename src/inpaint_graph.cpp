#include "inpaint_graph.h"
#include "inpaint_weight_names.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gpu_backend_pref.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace inpaint_graph {
namespace {
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
struct TestState {
    Builder::Observer callback = nullptr;
    void * opaque = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> stages;
};
#endif
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("inpainting graph: " + message);
}
std::string lower(std::string value) {
    for (char & c : value) c = char(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
ggml_tensor * make(ggml_context * ctx, ggml_type type, const std::vector<int64_t> & shape) {
    require(!shape.empty() && shape.size() <= 4, "tensor rank must be in [1,4]");
    size_t elements = 1;
    for (int64_t n : shape) {
        require(n > 0 && uint64_t(n) <= UINT64_C(1) << 30 && elements <= (size_t(1) << 31) / size_t(n),
                "tensor shape exceeds the graph bound");
        elements *= size_t(n);
    }
    return ggml_new_tensor(ctx, type, int(shape.size()), shape.data());
}
ggml_tensor * channels_first(ggml_context * ctx, ggml_tensor * x) {
    // WHCN -> CWHN -> C,(W*H),1,N.
    return ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)), x->ne[2], x->ne[0] * x->ne[1], 1,
                           x->ne[3]);
}
ggml_tensor * image_layout(ggml_context * ctx, ggml_tensor * x, int64_t w, int64_t h) {
    x = ggml_reshape_4d(ctx, x, x->ne[0], w, h, x->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
}
ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, bool fast = false) {
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, fast ? GGML_PREC_DEFAULT : GGML_PREC_F32);
    return out;
}
ggml_tensor * concat_balanced(ggml_context * ctx, std::vector<ggml_tensor *> parts, int axis) {
    require(!parts.empty(), "cannot concatenate an empty tile sequence");
    // Repeatedly appending to a growing tensor copies quadratic total bytes.
    // Pair adjacent tiles at each level, retaining their exact spatial order.
    while (parts.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((parts.size() + 1) / 2);
        for (size_t i = 0; i < parts.size(); i += 2)
            next.push_back(i + 1 < parts.size() ? ggml_concat(ctx, parts[i], parts[i + 1], axis) : parts[i]);
        parts = std::move(next);
    }
    return parts.front();
}
ggml_tensor * image_matmul(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * columns, int64_t width,
                           int64_t height, bool fast, bool spatial_output) {
    if (!spatial_output || columns->ne[3] > 2)
        return image_layout(ctx, matmul(ctx, weights, columns, fast), width, height);
    // With activations on the left, GEMM writes spatial-first CHW directly.
    // Process each sample separately so immutable weights stay contiguous and
    // ggml's CPU SGEMM remains eligible; a zero-stride weight broadcast would
    // silently bypass that optimized kernel in the pinned backend.
    std::vector<ggml_tensor *> samples;
    for (int64_t sample = 0; sample < columns->ne[3]; ++sample) {
        ggml_tensor * input =
            ggml_view_2d(ctx, columns, columns->ne[0], columns->ne[1], columns->nb[1], size_t(sample) * columns->nb[3]);
        ggml_tensor * product = matmul(ctx, input, weights);
        samples.push_back(ggml_reshape_4d(ctx, product, width, height, weights->ne[1], 1));
    }
    return concat_balanced(ctx, std::move(samples), 3);
}
std::vector<float> read_small(ggml_tensor * tensor) {
    require(tensor->type == GGML_TYPE_F32 && ggml_nelements(tensor) <= 65536, "invalid small F32 affine tensor");
    std::vector<float> values(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}

// Restricted CPU specialization of the same zero-padded depthwise operation.
// Reuse the MI-GAN tap-major interior loop: independent output lanes retain
// the original ky/kx accumulation order, while the compiler can vectorize
// adjacent pixels. Border handling remains scalar. The callback owns no
// state, allocates no heap storage and never transfers activations.
void cpu_depthwise_3x3(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * kernel = dst->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && kernel->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(kernel) && ggml_is_contiguous(dst));
    GGML_ASSERT(kernel->ne[0] == 3 && kernel->ne[1] == 3 && kernel->ne[2] == 1 && kernel->ne[3] == input->ne[2]);
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    for (int axis = 0; axis < 4; ++axis) GGML_ASSERT(dst->ne[axis] == input->ne[axis]);
    const int64_t width = input->ne[0], height = input->ne[1], channels = input->ne[2];
    GGML_ASSERT(width >= 3 && height >= 3 && channels > 0 && input->ne[3] > 0);
    const int64_t rows = height * channels * input->ne[3];
    const int64_t chunk = (rows + nth - 1) / nth;
    const int64_t begin = chunk * ith, end = std::min(rows, begin + chunk);
    const float * source = static_cast<const float *>(input->data);
    const float * weights = static_cast<const float *>(kernel->data);
    float * output = static_cast<float *>(dst->data);
    constexpr int64_t block_size = 128;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t plane = row / height, y = row % height;
        const float * plane_source = source + plane * width * height;
        const float * filter = weights + (plane % channels) * 9;
        float * out = output + row * width;
        auto edge = [&](int64_t x) {
            float sum = 0.0f;
            for (int64_t ky = 0; ky < 3; ++ky) {
                const int64_t sy = y + ky - 1;
                if (sy < 0 || sy >= height) continue;
                for (int64_t kx = 0; kx < 3; ++kx) {
                    const int64_t sx = x + kx - 1;
                    if (sx >= 0 && sx < width) sum += filter[ky * 3 + kx] * plane_source[sy * width + sx];
                }
            }
            out[x] = sum;
        };
        edge(0);
        for (int64_t x = 1; x < width - 1; x += block_size) {
            const int64_t count = std::min(block_size, width - 1 - x);
            alignas(64) float accum[block_size] = {};
            for (int64_t ky = 0; ky < 3; ++ky) {
                const int64_t sy = y + ky - 1;
                if (sy < 0 || sy >= height) continue;
                for (int64_t kx = 0; kx < 3; ++kx) {
                    const float coefficient = filter[ky * 3 + kx];
                    const float * pixels = plane_source + sy * width + x + kx - 1;
                    for (int64_t i = 0; i < count; ++i) accum[i] += coefficient * pixels[i];
                }
            }
            std::copy_n(accum, count, out + x);
        }
        edge(width - 1);
    }
}
} // namespace

ggml_backend_t select_backend(int threads, std::string & name, std::string & error) {
    error.clear();
    name.clear();
    if (threads < 1 || threads > 64) {
        error = "inpainting graph threads must be in [1,64]";
        return nullptr;
    }
    std::string pref = lower(crispasr_get_gpu_backend_pref());
    if (pref.empty()) {
        const char * configured = std::getenv("CRISPEMBED_INPAINT_BACKEND");
        if (configured) pref = lower(configured);
    }
    if (pref.empty()) pref = "auto";
    if (pref == "mps" || pref == "mtl") pref = "metal";
    if (pref != "auto" && pref != "cpu" && pref != "cuda" && pref != "metal") {
        error = "inpainting graph backend must be auto, cpu, cuda or metal (mps aliases metal)";
        return nullptr;
    }
    if (pref != "cpu") {
        core_metal_cache::apply();
        ggml_backend_load_all();
        for (const char * wanted : { "cuda", "metal" }) {
            if (pref != "auto" && pref != wanted) continue;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                const auto type = ggml_backend_dev_type(dev);
                if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
                const std::string reg = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
                if ((std::string(wanted) == "cuda" && reg != "cuda") ||
                    (std::string(wanted) == "metal" && reg != "metal" && reg != "mtl"))
                    continue;
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) {
                    name = ggml_backend_name(backend);
                    return backend;
                }
            }
        }
        if (pref != "auto") {
            error = "requested inpainting backend '" + pref + "' is unavailable; no CPU fallback was selected";
            return nullptr;
        }
    }
    ggml_backend_t backend = core_cpu_backend::init();
    if (!backend) {
        error = "inpainting CPU backend is unavailable";
        return nullptr;
    }
    core_cpu_backend::set_n_threads(backend, threads);
    name = ggml_backend_name(backend);
    return backend;
}

void validate_graph_backend(ggml_backend_t backend, ggml_cgraph * graph) {
    require(backend && graph, "null backend or graph");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        require(ggml_backend_supports_op(backend, node),
                std::string(ggml_backend_name(backend)) + " does not support node " + std::to_string(i) + " " +
                    ggml_op_name(node->op) + " '" + ggml_get_name(node) + "'; graph was not executed");
    }
}

CpuPool::CpuPool(ggml_backend_t backend, int threads) {
    if (!core_cpu_backend::is_cpu(backend)) return;
    const auto registry = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Make = ggml_threadpool_t (*)(ggml_threadpool_params *);
    using Free = void (*)(ggml_threadpool_t);
    using Attach = void (*)(ggml_backend_t, ggml_threadpool_t);
    const auto make_pool = reinterpret_cast<Make>(ggml_backend_reg_get_proc_address(registry, "ggml_threadpool_new"));
    const auto free_pool = reinterpret_cast<Free>(ggml_backend_reg_get_proc_address(registry, "ggml_threadpool_free"));
    const auto attach =
        reinterpret_cast<Attach>(ggml_backend_reg_get_proc_address(registry, "ggml_backend_cpu_set_threadpool"));
    if (make_pool && free_pool && attach) {
        auto parameters = ggml_threadpool_params_default(threads);
        pool_ = make_pool(&parameters);
        if (pool_) {
            free_ = free_pool;
            attach(backend, pool_);
        }
    }
}
CpuPool::~CpuPool() {
    if (pool_) free_(pool_);
}

Device::Device(int threads) {
    std::string error;
    backend = select_backend(threads, name, error);
    require(backend != nullptr, error);
    gpu = !core_cpu_backend::is_cpu(backend);
    fast = gpu && core_env::on("CRISPEMBED_INPAINT_FAST");
    try {
        if (!gpu) cpu_pool_ = std::make_unique<CpuPool>(backend, threads);
    } catch (...) {
        ggml_backend_free(backend);
        backend = nullptr;
        throw;
    }
}
Device::~Device() {
    if (backend) ggml_backend_free(backend);
}

Weights::Weights(Device & selected, const char * path, const ShapeMap & schema) : device(selected) {
    try {
        require(path && *path && !schema.empty(), "model path and weight schema are required");
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path),
                                                                                core_gguf::free_metadata);
        require(bool(meta) && size_t(gguf_get_n_tensors(meta.get())) == schema.size(), "weight tensor census differs");
        const auto names = inpaint_weight_names::map(meta.get(), schema);
        uint64_t total = 0;
        for (const auto & entry : schema) {
            const int64_t id = gguf_find_tensor(meta.get(), names.at(entry.first).c_str());
            require(id >= 0 && gguf_get_tensor_type(meta.get(), id) == GGML_TYPE_F32,
                    "missing original F32 weight " + entry.first);
            std::array<int64_t, 4> expected{ 1, 1, 1, 1 };
            require(!entry.second.empty() && entry.second.size() <= 4, "invalid weight schema rank");
            uint64_t count = 1;
            for (size_t j = 0; j < entry.second.size(); ++j) {
                require(entry.second[j] > 0 && count <= (UINT64_C(1) << 30) / uint64_t(entry.second[j]),
                        "weight dimensions overflow");
                expected[entry.second.size() - j - 1] = entry.second[j];
                count *= uint64_t(entry.second[j]);
            }
            for (int axis = 0; axis < 4; ++axis)
                require(gguf_get_tensor_ne(meta.get(), id)[axis] == expected[axis],
                        "weight shape differs: " + entry.first);
            total += count;
        }
        require(total <= UINT64_C(1) << 30, "F32 weights exceed 4 GiB");
        meta.reset();
        if (!device.fast) {
            require(core_gguf::load_weights(path, device.backend, "inpainting-graph", loaded, true),
                    "cannot load resident weights");
            inpaint_weight_names::restore(loaded, names);
            return;
        }
        // Source bytes and their digest stay unchanged. This explicitly gated
        // GPU profile converts only matrix/dense-convolution weights at load.
        // Depthwise kernels and normalization parameters retain F32.
        ggml_backend_t cpu = core_cpu_backend::init();
        require(cpu != nullptr, "cannot open temporary CPU weight mapping");
        core_gguf::WeightLoad source;
        try {
            require(core_gguf::load_weights(path, cpu, "inpainting-fast-source", source, true),
                    "cannot map F32 source");
            inpaint_weight_names::restore(source, names);
            loaded.ctx = ggml_init({ (schema.size() + 16) * ggml_tensor_overhead(), nullptr, true });
            require(loaded.ctx != nullptr, "weight metadata allocation failed");
            for (const auto & entry : schema) {
                ggml_tensor * original = source.tensors.at(entry.first);
                const auto & shape = entry.second;
                const bool matrix = shape.size() == 2;
                // Moebius' positional bank has IC=1 but is a dense 15x15
                // convolution over independent value-plane samples. Keep its
                // matrix eligible for the same half/GEMM profile; true
                // depthwise kernels still retain their original F32 bytes.
                const std::string positional_suffix = ".pos_conv.weight";
                const bool positional = entry.first.size() >= positional_suffix.size() &&
                                        entry.first.compare(entry.first.size() - positional_suffix.size(),
                                                            positional_suffix.size(), positional_suffix) == 0;
                const bool dense = shape.size() == 4 && (shape[1] > 1 || positional);
                ggml_tensor * target =
                    ggml_new_tensor(loaded.ctx, matrix || dense ? GGML_TYPE_F16 : GGML_TYPE_F32, 4, original->ne);
                ggml_set_name(target, names.at(entry.first).c_str());
                loaded.tensors.emplace(entry.first, target);
            }
            loaded.buf = ggml_backend_alloc_ctx_tensors(loaded.ctx, device.backend);
            require(loaded.buf != nullptr, "device weight allocation failed");
            ggml_backend_buffer_set_usage(loaded.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            constexpr size_t chunk = size_t(1) << 20;
            std::vector<float> f32(chunk);
            std::vector<ggml_fp16_t> f16(chunk);
            for (const auto & entry : loaded.tensors) {
                ggml_tensor * original = source.tensors.at(entry.first);
                const size_t count = size_t(ggml_nelements(original));
                for (size_t offset = 0; offset < count; offset += chunk) {
                    const size_t n = std::min(chunk, count - offset);
                    ggml_backend_tensor_get(original, f32.data(), offset * sizeof(float), n * sizeof(float));
                    if (entry.second->type == GGML_TYPE_F16) {
                        ggml_fp32_to_fp16_row(f32.data(), f16.data(), int64_t(n));
                        ggml_backend_tensor_set(entry.second, f16.data(), offset * sizeof(ggml_fp16_t),
                                                n * sizeof(ggml_fp16_t));
                    } else
                        ggml_backend_tensor_set(entry.second, f32.data(), offset * sizeof(float), n * sizeof(float));
                }
            }
        } catch (...) {
            core_gguf::free_weights(source);
            ggml_backend_free(cpu);
            throw;
        }
        core_gguf::free_weights(source);
        ggml_backend_free(cpu);
    } catch (...) {
        core_gguf::free_weights(loaded);
        throw;
    }
}
Weights::~Weights() {
    core_gguf::free_weights(loaded);
}

State::State(Device & selected, size_t capacity) : device(selected) {
    ctx_ = ggml_init({ capacity * ggml_tensor_overhead(), nullptr, true });
    require(ctx_ != nullptr, "resident state metadata allocation failed");
}
State::~State() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
}
ggml_tensor * State::tensor(const std::string & name, const std::vector<int64_t> & shape) {
    require(!buffer_, "cannot add resident state after allocation");
    ggml_tensor * out = make(ctx_, GGML_TYPE_F32, shape);
    ggml_set_name(out, name.c_str());
    ggml_set_input(out);
    return out;
}
void State::allocate() {
    require(!buffer_, "resident state is already allocated");
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, device.backend);
    require(buffer_ != nullptr, "resident state allocation failed");
    ggml_backend_buffer_clear(buffer_, 0);
}
void State::upload(ggml_tensor * tensor, const std::vector<float> & data) {
    require(buffer_ && tensor && tensor->type == GGML_TYPE_F32 && size_t(ggml_nelements(tensor)) == data.size(),
            "resident input dimensions differ");
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

Builder::Builder(Device & selected, Weights * weights, size_t capacity)
    : device(selected), weights_(weights), capacity_(capacity) {
    require(capacity >= 64 && capacity <= 524288, "graph node capacity is out of bounds");
    ctx = ggml_init({ capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false), nullptr, true });
    require(ctx != nullptr, "graph metadata allocation failed");
}
Builder::~Builder() {
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
    delete static_cast<TestState *>(testing_state_);
#endif
    if (core_env::on("CRISPEMBED_INPAINT_PROFILE") && graph_)
        std::fprintf(stderr,
                     "inpaint_graph label=%s backend=%s precision=%s nodes=%zu allocation_bytes=%zu runs=%llu "
                     "upload_bytes=%llu download_bytes=%llu compute_ms=%.3f cpu_depthwise_nodes=%zu\n",
                     label_.c_str(), device.name.c_str(), fast() ? "mixed-f16" : "f32", nodes(), allocation_bytes(),
                     (unsigned long long)runs_, (unsigned long long)upload_bytes_, (unsigned long long)download_bytes_,
                     compute_ms_, cpu_depthwise_nodes_);
    if (cpu_plan_) ggml_backend_graph_plan_free(device.backend, cpu_plan_);
    if (allocator_) ggml_gallocr_free(allocator_);
    if (ctx) ggml_free(ctx);
}
ggml_tensor * Builder::weight(const std::string & name) const {
    require(weights_ && weights_->loaded.tensors.count(name), "missing weight " + name);
    return weights_->loaded.tensors.at(name);
}
bool Builder::has(const std::string & name) const {
    return weights_ && weights_->loaded.tensors.count(name);
}
ggml_tensor * Builder::parameter(const std::string & name, const std::vector<int64_t> & shape) {
    require(!graph_, "cannot modify a finalized graph");
    ggml_tensor * out = make(ctx, GGML_TYPE_F32, shape);
    ggml_set_name(out, name.c_str());
    ggml_set_input(out);
    // INPUT alone does not retain storage in the pinned gallocr. A persistent
    // graph must keep both uploaded-once inputs and immutable initializers
    // alive across every replay, including after their last use in one run.
    ggml_set_output(out);
    return out;
}
ggml_tensor * Builder::constant(const std::string & name, const std::vector<int64_t> & shape,
                                const std::vector<float> & data) {
    if (constants_.count(name)) return constants_.at(name);
    ggml_tensor * out = parameter(name, shape);
    require(size_t(ggml_nelements(out)) == data.size(), "constant shape differs: " + name);
    Initializer init{ out, std::vector<uint8_t>(data.size() * sizeof(float)), false };
    std::memcpy(init.bytes.data(), data.data(), init.bytes.size());
    initializers_.push_back(std::move(init));
    constants_[name] = out;
    return out;
}
ggml_tensor * Builder::constant_i32(const std::string & name, const std::vector<int64_t> & shape,
                                    const std::vector<int32_t> & data) {
    if (constants_.count(name)) return constants_.at(name);
    require(!graph_, "cannot modify a finalized graph");
    ggml_tensor * out = make(ctx, GGML_TYPE_I32, shape);
    require(size_t(ggml_nelements(out)) == data.size(), "index constant shape differs");
    ggml_set_name(out, name.c_str());
    ggml_set_input(out);
    ggml_set_output(out);
    Initializer init{ out, std::vector<uint8_t>(data.size() * sizeof(int32_t)), false };
    std::memcpy(init.bytes.data(), data.data(), init.bytes.size());
    initializers_.push_back(std::move(init));
    constants_[name] = out;
    return out;
}
ggml_tensor * Builder::zeros(const std::string & name, const std::vector<int64_t> & shape) {
    if (constants_.count(name)) return constants_.at(name);
    ggml_tensor * out = parameter(name, shape);
    initializers_.push_back({ out, {}, true });
    constants_[name] = out;
    return out;
}

ggml_tensor * Builder::linear(const std::string & name, ggml_tensor * x, bool bias) {
    ggml_tensor * w = weight(name + ".weight");
    require(w->ne[0] == x->ne[2] && w->ne[2] == 1 && w->ne[3] == 1, "linear shape differs: " + name);
    const bool spatial_output = !device.gpu && core_env::on("CRISPEMBED_INPAINT_GEMM_LAYOUT");
    ggml_tensor * out = image_matmul(ctx, w, channels_first(ctx, x), x->ne[0], x->ne[1], fast(), spatial_output);
    if (bias && has(name + ".bias"))
        out = ggml_add(ctx, out, ggml_reshape_4d(ctx, weight(name + ".bias"), 1, 1, out->ne[2], 1));
    return out;
}
ggml_tensor * Builder::conv(const std::string & name, ggml_tensor * x, int stride, int padding, int groups,
                            int dilation) {
    return conv_kernel(name, weight(name + ".weight"), has(name + ".bias") ? weight(name + ".bias") : nullptr, x,
                       stride, padding, groups, dilation);
}
ggml_tensor * Builder::conv_kernel(const std::string & name, ggml_tensor * w, ggml_tensor * bias, ggml_tensor * x,
                                   int stride, int padding, int groups, int dilation) {
    require(stride > 0 && padding >= 0 && groups > 0 && dilation > 0 && w->ne[2] * groups == x->ne[2],
            "convolution shape differs: " + name);
    ggml_tensor * out = nullptr;
    if (groups == x->ne[2] && w->ne[3] == x->ne[2]) {
        const bool specialized = !device.gpu && core_env::on("CRISPEMBED_INPAINT_CPU_DEPTHWISE") &&
                                 x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
                                 ggml_is_contiguous(w) && w->ne[0] == 3 && w->ne[1] == 3 && w->ne[2] == 1 &&
                                 stride == 1 && padding == 1 && dilation == 1 && x->ne[0] >= 3 && x->ne[1] >= 3;
        if (specialized) {
            ggml_tensor * sources[] = { x, w };
            out = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], x->ne[2], x->ne[3], sources, 2,
                                 cpu_depthwise_3x3, GGML_N_TASKS_MAX, nullptr);
            cpu_depthwise_ops_.push_back(out);
        } else {
            // The pinned CUDA depthwise kernel requires F32 weights.
            if (w->type != GGML_TYPE_F32) w = ggml_cast(ctx, w, GGML_TYPE_F32);
            out = ggml_conv_2d_dw_direct(ctx, w, x, stride, stride, padding, padding, dilation, dilation);
        }
    } else {
        require(groups == 1, "only dense and depthwise groups are supported");
        if (w->ne[0] == 1 && w->ne[1] == 1 && stride == 1 && padding == 0) {
            // Dense 1x1 is a GEMM: no im2col allocation or copy.
            out = image_matmul(ctx, ggml_reshape_2d(ctx, w, w->ne[2], w->ne[3]), channels_first(ctx, x), x->ne[0],
                               x->ne[1], fast(), !device.gpu && core_env::on("CRISPEMBED_INPAINT_GEMM_LAYOUT"));
        } else {
            const ggml_type patch_type = fast() && w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            const int64_t ow = (x->ne[0] + 2 * padding - dilation * (w->ne[0] - 1) - 1) / stride + 1;
            const int64_t oh = (x->ne[1] + 2 * padding - dilation * (w->ne[1] - 1) - 1) / stride + 1;
            require(ow > 0 && oh > 0, "convolution kernel exceeds image");
            const int64_t kernel_size = w->ne[0] * w->ne[1] * w->ne[2];
            // Bounded row strips prevent 512px VAE im2col temporaries from
            // occupying gigabytes. CUDA/Metal use larger strips to amortize
            // launches; CPU uses a smaller working set. Halo rows are copied
            // on-device, and all strips remain in one persistent graph.
            const size_t budget = (device.gpu ? size_t(128) : size_t(32)) << 20;
            const size_t bytes_per_row = size_t(kernel_size * ow * x->ne[3]) * ggml_type_size(patch_type);
            const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(oh, int64_t(budget / bytes_per_row)));
            std::vector<ggml_tensor *> parts;
            for (int64_t start = 0; start < oh; start += rows) {
                const int64_t count = std::min(rows, oh - start);
                ggml_tensor * strip = x;
                int pad_y = padding;
                if (rows < oh) {
                    const int64_t y0 = start * stride - padding;
                    const int64_t y1 = (start + count - 1) * stride - padding + dilation * (w->ne[1] - 1) + 1;
                    const int64_t first = std::max<int64_t>(0, y0), last = std::min<int64_t>(x->ne[1], y1);
                    strip = ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], last - first, x->ne[2], x->ne[3], x->nb[1],
                                                        x->nb[2], x->nb[3], size_t(first) * x->nb[1]));
                    if (y0 < 0 || y1 > x->ne[1]) strip = pad(strip, 0, 0, int(first - y0), int(y1 - last));
                    pad_y = 0;
                }
                ggml_tensor * patches =
                    ggml_im2col(ctx, w, strip, stride, stride, padding, pad_y, dilation, dilation, true, patch_type);
                ggml_tensor * columns =
                    ggml_reshape_4d(ctx, patches, kernel_size, patches->ne[1] * patches->ne[2], 1, x->ne[3]);
                ggml_tensor * part =
                    image_matmul(ctx, ggml_reshape_2d(ctx, w, kernel_size, w->ne[3]), columns, patches->ne[1],
                                 patches->ne[2], fast(), !device.gpu && core_env::on("CRISPEMBED_INPAINT_GEMM_LAYOUT"));
                parts.push_back(part);
            }
            out = concat_balanced(ctx, std::move(parts), 1);
        }
    }
    if (bias) out = ggml_add(ctx, out, ggml_reshape_4d(ctx, bias, 1, 1, out->ne[2], 1));
    return out;
}
ggml_tensor * Builder::conv_transpose(const std::string & name, ggml_tensor * x, int stride, int padding,
                                      int output_padding) {
    require(!graph_, "cannot add a transpose convolution to a finalized graph");
    ggml_tensor * w = weight(name + ".weight");
    require(x && x->type == GGML_TYPE_F32 && x->ne[0] > 0 && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] > 0 &&
                x->ne[3] <= 8 && stride >= 1 && stride <= 4 && w->ne[0] >= 1 && w->ne[0] <= 15 &&
                w->ne[0] == w->ne[1] && w->ne[3] == x->ne[2] && w->ne[2] > 0 && padding >= 0 && padding < w->ne[0] &&
                output_padding >= 0 && output_padding < stride &&
                (w->type == GGML_TYPE_F32 || (fast() && w->type == GGML_TYPE_F16)),
            "transpose convolution shape differs: " + name);
    require(x->ne[0] <= std::numeric_limits<int>::max() / stride &&
                x->ne[1] <= std::numeric_limits<int>::max() / stride &&
                uint64_t(ggml_nelements(x)) <= (UINT64_C(1) << 31) / uint64_t(stride * stride),
            "transpose convolution expansion exceeds the graph bound");
    const int64_t kernel = w->ne[0], input_channels = w->ne[3], output_channels = w->ne[2];
    const int64_t expanded_w = x->ne[0] * stride, expanded_h = x->ne[1] * stride;
    const int64_t ow = (x->ne[0] - 1) * stride - 2 * padding + kernel + output_padding;
    const int64_t oh = (x->ne[1] - 1) * stride - 2 * padding + kernel + output_padding;
    require(ow > 0 && oh > 0, "transpose convolution output is empty");

    // Prepare immutable IOHW -> flipped OIHW weights once per graph. Reading
    // checkpoint tensors here is load-time preparation, never an activation
    // transfer. Keep the transformed tensor resident across every replay.
    const std::string prepared_name = "graph.transpose." + name + ".weight";
    ggml_tensor * prepared = nullptr;
    if (constants_.count(prepared_name)) {
        prepared = constants_.at(prepared_name);
    } else {
        const size_t count = size_t(ggml_nelements(w));
        std::vector<float> source(count), transformed(count);
        if (w->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(w, source.data(), 0, count * sizeof(float));
        } else {
            std::vector<ggml_fp16_t> packed(count);
            ggml_backend_tensor_get(w, packed.data(), 0, count * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(packed.data(), source.data(), int64_t(count));
        }
        for (int64_t out = 0; out < output_channels; ++out)
            for (int64_t in = 0; in < input_channels; ++in)
                for (int64_t y = 0; y < kernel; ++y)
                    for (int64_t xx = 0; xx < kernel; ++xx) {
                        const size_t from =
                            size_t(((in * output_channels + out) * kernel + kernel - 1 - y) * kernel + kernel - 1 - xx);
                        const size_t to = size_t(((out * input_channels + in) * kernel + y) * kernel + xx);
                        transformed[to] = source[from];
                    }
        const ggml_type prepared_type = fast() ? GGML_TYPE_F16 : GGML_TYPE_F32;
        prepared = make(ctx, prepared_type, { kernel, kernel, input_channels, output_channels });
        ggml_set_name(prepared, prepared_name.c_str());
        ggml_set_input(prepared);
        ggml_set_output(prepared);
        Initializer init{ prepared, std::vector<uint8_t>(ggml_nbytes(prepared)), false };
        if (fast()) {
            std::vector<ggml_fp16_t> packed(count);
            ggml_fp32_to_fp16_row(transformed.data(), packed.data(), int64_t(count));
            std::memcpy(init.bytes.data(), packed.data(), init.bytes.size());
        } else {
            std::memcpy(init.bytes.data(), transformed.data(), init.bytes.size());
        }
        initializers_.push_back(std::move(init));
        constants_.emplace(prepared_name, prepared);
    }

    ggml_tensor * inserted = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    if (stride > 1) {
        // Integer nearest resize repeats each value; this shared geometry mask
        // retains only the (y % stride == 0, x % stride == 0) samples. Thus the
        // operation is exact zero insertion on CPU/CUDA/Metal, with no host
        // activation loop and no per-inference mask upload.
        const std::string mask_name = "graph.transpose.zeros." + std::to_string(expanded_w) + "." +
                                      std::to_string(expanded_h) + "." + std::to_string(stride);
        ggml_tensor * mask = nullptr;
        if (constants_.count(mask_name)) {
            mask = constants_.at(mask_name);
        } else {
            std::vector<float> pattern(size_t(expanded_w * expanded_h), 0);
            for (int64_t y = 0; y < expanded_h; y += stride)
                for (int64_t xx = 0; xx < expanded_w; xx += stride) pattern[size_t(y * expanded_w + xx)] = 1;
            mask = constant(mask_name, { expanded_w, expanded_h, 1, 1 }, pattern);
        }
        inserted = ggml_mul(ctx, resize(inserted, int(expanded_h), int(expanded_w)), mask);
    }
    // Dense convolution now uses the same bounded im2col/GEMM implementation
    // as every ordinary layer (including F16 matrix operands in fast mode).
    // Zero insertion retained stride-1 trailing zeros; output_padding keeps
    // exactly the requested subset of those final rows and columns.
    ggml_tensor * out = conv_kernel(name + ".lowered", prepared, nullptr, inserted, 1, int(kernel - 1 - padding), 1, 1);
    const int trailing = stride - 1 - output_padding;
    require(out->ne[0] - trailing == ow && out->ne[1] - trailing == oh,
            "lowered transpose convolution crop geometry differs");
    if (trailing)
        out = ggml_cont(ctx,
                        ggml_view_4d(ctx, out, ow, oh, out->ne[2], out->ne[3], out->nb[1], out->nb[2], out->nb[3], 0));
    if (has(name + ".bias"))
        out = ggml_add(ctx, out, ggml_reshape_4d(ctx, weight(name + ".bias"), 1, 1, out->ne[2], 1));
    return out;
}
ggml_tensor * Builder::group_norm(const std::string & name, ggml_tensor * x, int groups, float eps) {
    require(groups > 0 && x->ne[2] % groups == 0, "group norm channels differ");
    // Pinned Metal GROUP_NORM processes ne[2] and ignores ne[3]. Merge the
    // sample axis into channels and multiply groups so CFG samples remain
    // independent and every backend computes all samples.
    ggml_tensor * grouped = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3], 1);
    ggml_tensor * out = ggml_group_norm(ctx, grouped, int(groups * x->ne[3]), eps);
    out = ggml_reshape_4d(ctx, out, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    out = ggml_mul(ctx, out, ggml_reshape_4d(ctx, weight(name + ".weight"), 1, 1, x->ne[2], 1));
    return ggml_add(ctx, out, ggml_reshape_4d(ctx, weight(name + ".bias"), 1, 1, x->ne[2], 1));
}
ggml_tensor * Builder::layer_norm(const std::string & name, ggml_tensor * x, float eps) {
    ggml_tensor * out = ggml_norm(ctx, channels_first(ctx, x), eps);
    out = ggml_mul(ctx, out, weight(name + ".weight"));
    out = ggml_add(ctx, out, weight(name + ".bias"));
    return image_layout(ctx, out, x->ne[0], x->ne[1]);
}
ggml_tensor * Builder::batch_norm(const std::string & name, ggml_tensor * x, float eps) {
    const std::string key = name + ".graph_affine." + std::to_string(eps);
    if (!constants_.count(key + ".scale")) {
        const auto gamma = read_small(weight(name + ".weight"));
        const auto beta = read_small(weight(name + ".bias"));
        const auto mean = read_small(weight(name + ".running_mean"));
        const auto var = read_small(weight(name + ".running_var"));
        require(gamma.size() == size_t(x->ne[2]) && beta.size() == gamma.size() && mean.size() == gamma.size() &&
                    var.size() == gamma.size(),
                "batch norm shape differs");
        std::vector<float> scale(gamma.size()), bias(gamma.size());
        for (size_t i = 0; i < gamma.size(); ++i) {
            scale[i] = gamma[i] / std::sqrt(var[i] + eps);
            bias[i] = beta[i] - mean[i] * scale[i];
        }
        constant(key + ".scale", { 1, 1, x->ne[2], 1 }, scale);
        constant(key + ".bias", { 1, 1, x->ne[2], 1 }, bias);
    }
    return ggml_add(ctx, ggml_mul(ctx, x, constants_.at(key + ".scale")), constants_.at(key + ".bias"));
}

ggml_tensor * Builder::attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int heads) {
    require(heads > 0 && q->ne[2] % heads == 0 && q->ne[2] == k->ne[2] && v->ne[2] % heads == 0 &&
                k->ne[0] * k->ne[1] == v->ne[0] * v->ne[1] && q->ne[3] == k->ne[3] && k->ne[3] == v->ne[3],
            "attention shape differs");
    const int64_t width = q->ne[0], height = q->ne[1], n = width * height, nk = k->ne[0] * k->ne[1];
    const int64_t dk = q->ne[2] / heads, dv = v->ne[2] / heads, batch = q->ne[3];
    auto layout = [&](ggml_tensor * x, int64_t d, int64_t tokens) {
        ggml_tensor * a = ggml_reshape_4d(ctx, x, tokens, d, heads, batch);
        return ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3)); // D,T,H,B
    };
    q = layout(q, dk, n);
    k = layout(k, dk, nk);
    v = layout(v, dv, nk);
    ggml_tensor * out = nullptr;
    if ((!device.gpu && !core_env::on("CRISPEMBED_INPAINT_CPU_ATTENTION_TILES")) || fast()) {
        if (fast()) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }
        out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt(float(dk)), 0, 0);
        if (!fast()) ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        // Flash returns D,H,T,B; normalize to D,T,H,B.
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    } else {
        // Metal's pinned flash kernel uses half tiles. The strict GPU profile
        // uses bounded F32 tiles. The explicit CPU gate permits the same
        // evaluation order for original-model numerical attribution without
        // replacing the default CPU online attention implementation.
        ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
        constexpr int64_t tile = 128;
        std::vector<ggml_tensor *> chunks;
        for (int64_t start = 0; start < n; start += tile) {
            const int64_t count = std::min(tile, n - start);
            ggml_tensor * qt =
                ggml_view_4d(ctx, q, dk, count, heads, batch, q->nb[1], q->nb[2], q->nb[3], size_t(start) * q->nb[1]);
            ggml_tensor * scores = ggml_scale(ctx, matmul(ctx, k, qt), 1.0f / std::sqrt(float(dk)));
            ggml_tensor * chunk = matmul(ctx, vt, ggml_soft_max(ctx, scores));
            chunks.push_back(chunk);
        }
        out = concat_balanced(ctx, std::move(chunks), 1);
    }
    out = ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));
    return ggml_reshape_4d(ctx, out, width, height, dv * heads, batch);
}
ggml_tensor * Builder::channel_slice(ggml_tensor * x, int start, int count) {
    require(start >= 0 && count > 0 && int64_t(start) + count <= x->ne[2], "channel slice exceeds tensor");
    return ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], count, x->ne[3], x->nb[1], x->nb[2], x->nb[3],
                                       size_t(start) * x->nb[2]));
}
ggml_tensor * Builder::pad(ggml_tensor * x, int left, int right, int top, int bottom) {
    require(left >= 0 && right >= 0 && top >= 0 && bottom >= 0, "negative zero padding");
    // ggml Metal PAD supports only trailing padding. Pad on the trailing side
    // then roll; newly introduced cells are zero, so this is exact on all backends.
    ggml_tensor * out = ggml_pad(ctx, x, left + right, top + bottom, 0, 0);
    if (left || top) out = ggml_roll(ctx, out, left, top, 0, 0);
    return out;
}
ggml_tensor * Builder::reflect_pad(ggml_tensor * x, int padding) {
    require(padding >= 0 && padding < x->ne[0] && padding < x->ne[1], "reflection padding exceeds input");
    if (!padding) return x;
    if (core_env::on("CRISPEMBED_INPAINT_REFLECT_GATHER")) {
        // Reflect whole contiguous rows, then pad the width. This avoids the
        // two full activation transposes in the portable baseline. Geometry
        // indices are immutable and shared by equal-shaped layers.
        const int64_t height = x->ne[1], padded = height + 2 * padding;
        const int64_t planes = x->ne[2] * x->ne[3];
        require(padded <= std::numeric_limits<int32_t>::max() / planes, "reflection row indices exceed I32");
        const std::string key =
            "reflect.rows." + std::to_string(height) + "." + std::to_string(planes) + "." + std::to_string(padding);
        std::vector<int32_t> indices(size_t(padded * planes));
        for (int64_t plane = 0; plane < planes; ++plane)
            for (int64_t y = 0; y < padded; ++y) {
                int64_t source = y - padding;
                if (source < 0) source = -source;
                if (source >= height) source = 2 * height - 2 - source;
                indices[size_t(plane * padded + y)] = int32_t(plane * height + source);
            }
        ggml_tensor * rows =
            ggml_reshape_2d(ctx, ggml_is_contiguous(x) ? x : ggml_cont(ctx, x), x->ne[0], height * planes);
        rows = ggml_get_rows(ctx, rows, constant_i32(key, { padded * planes }, indices));
        rows = ggml_reshape_4d(ctx, rows, x->ne[0], padded, x->ne[2], x->ne[3]);
        return ggml_pad_reflect_1d(ctx, rows, padding, padding);
    }
    x = ggml_pad_reflect_1d(ctx, x, padding, padding);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_pad_reflect_1d(ctx, x, padding, padding);
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}
ggml_tensor * Builder::resize(ggml_tensor * x, int height, int width) {
    require(height > 0 && width > 0, "invalid resize dimensions");
    return ggml_interpolate(ctx, x, width, height, x->ne[2], x->ne[3], GGML_SCALE_MODE_NEAREST);
}
void Builder::mark(const std::string & name, ggml_tensor * x) {
    ggml_set_name(x, name.c_str());
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
    auto * state = static_cast<TestState *>(testing_state_);
    if (state && state->callback) {
        for (ggml_tensor * storage = x; storage; storage = storage->view_src) ggml_set_output(storage);
        state->stages.emplace_back(name, x);
    }
#endif
}
void Builder::finish(const std::vector<ggml_tensor *> & outputs, const std::string & label) {
    require(!graph_ && !outputs.empty(), "graph is already finalized or has no outputs");
    label_ = label;
    graph_ = ggml_new_graph_custom(ctx, capacity_, false);
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
    if (testing_state_)
        for (const auto & stage : static_cast<TestState *>(testing_state_)->stages)
            ggml_build_forward_expand(graph_, stage.second);
#endif
    for (ggml_tensor * output : outputs) {
        for (ggml_tensor * storage = output; storage; storage = storage->view_src) ggml_set_output(storage);
        ggml_build_forward_expand(graph_, output);
    }
    // Count the custom depthwise nodes in the executable graph. Constructed but
    // unused branches must not make a profile claim that a helper ran.
    for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i)
        if (std::find(cpu_depthwise_ops_.begin(), cpu_depthwise_ops_.end(), ggml_graph_node(graph_, i)) !=
            cpu_depthwise_ops_.end())
            ++cpu_depthwise_nodes_;
    cpu_depthwise_ops_.clear();
    cpu_depthwise_ops_.shrink_to_fit();
    validate_graph_backend(device.backend, graph_);
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(device.backend));
    require(allocator_ && ggml_gallocr_alloc_graph(allocator_, graph_), "persistent graph allocation failed: " + label);
    for (const auto & init : initializers_) {
        // Unused branches may leave an initializer outside the compiled graph.
        if (!init.tensor->buffer) continue;
        if (init.zero)
            ggml_backend_tensor_memset(init.tensor, 0, 0, ggml_nbytes(init.tensor));
        else
            ggml_backend_tensor_set(init.tensor, init.bytes.data(), 0, init.bytes.size());
    }
    initializers_.clear();
    initializers_.shrink_to_fit();
    if (!device.gpu) {
        cpu_plan_ = ggml_backend_graph_plan_create(device.backend, graph_);
        require(cpu_plan_ != nullptr, "persistent CPU graph plan allocation failed");
    }
    ready_ = true;
}
void Builder::upload(ggml_tensor * tensor, const std::vector<float> & data) {
    require(ready_ && tensor && tensor->buffer && ggml_is_contiguous(tensor) && tensor->type == GGML_TYPE_F32 &&
                size_t(ggml_nelements(tensor)) == data.size(),
            "graph input shape/storage differs");
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    upload_bytes_ += data.size() * sizeof(float);
}
void Builder::compute() {
    require(ready_, "graph has not been successfully finalized");
    const auto start = std::chrono::steady_clock::now();
    const auto status = cpu_plan_ ? ggml_backend_graph_plan_compute(device.backend, cpu_plan_)
                                  : ggml_backend_graph_compute(device.backend, graph_);
    require(status == GGML_STATUS_SUCCESS, "backend graph execution failed: " + label_);
    compute_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ++runs_;
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
    auto * state = static_cast<TestState *>(testing_state_);
    if (state && state->callback)
        for (const auto & stage : state->stages)
            state->callback(state->opaque, stage.first, stage.second, download(stage.second));
#endif
}
std::vector<float> Builder::download(ggml_tensor * tensor) {
    require(ready_ && tensor && tensor->buffer && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor),
            "output must have allocated contiguous F32 storage");
    std::vector<float> result(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    download_bytes_ += ggml_nbytes(tensor);
    return result;
}
size_t Builder::allocation_bytes() const {
    return allocator_ ? ggml_gallocr_get_buffer_size(allocator_, 0) : 0;
}
size_t Builder::nodes() const {
    return graph_ ? size_t(ggml_graph_n_nodes(graph_)) : 0;
}
#ifdef CRISPEMBED_INPAINT_GRAPH_TESTING
void Builder::set_observer(Observer callback, void * opaque) {
    if (!testing_state_) testing_state_ = new TestState;
    auto & state = *static_cast<TestState *>(testing_state_);
    state.callback = callback;
    state.opaque = opaque;
}
#endif
} // namespace inpaint_graph
