#include "lama.h"
#include "lama_graph.h"
#include "lama_fft.h"
#include "lama_schema.h"

#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lama {
using latent_inpaint::Engine;
using latent_inpaint::Tensor;
namespace {
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("Big-LaMa: " + message);
}
void enabled() {
    require(core_env::on("CRISPEMBED_LAMA"), "experimental CPU/F32 inference requires CRISPEMBED_LAMA=1");
}
bool digest_valid(const std::string & digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(),
                                              [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void dimensions(int width, int height) {
    require(width >= min_dimension && height >= min_dimension && width <= max_dimension && height <= max_dimension &&
                size_t(width) <= max_pixels / size_t(height),
            "image dimensions outside the supported9..1024 profile");
}
void buffers(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width, int height) {
    dimensions(width, height);
    const size_t pixels = size_t(width) * height;
    require(rgb && mask && rgb_bytes == pixels * 3 && mask_bytes == pixels, "RGB/mask buffer size differs");
    require(std::all_of(mask, mask + mask_bytes, [](uint8_t value) { return value == 0 || value == 255; }),
            "mask must be binary0/255, white=fill");
}
int rounded_eight(int value) {
    return (value + 7) / 8 * 8;
}
int symmetric_index(int index, int size) {
    const int phase = index % (2 * size);
    return phase < size ? phase : 2 * size - 1 - phase;
}
int reflect_index(int index, int size) {
    const int period = 2 * (size - 1);
    int phase = index % period;
    if (phase < 0) phase += period;
    return phase < size ? phase : period - phase;
}
Tensor relu(Tensor value) {
    for (float & scalar : value.v)
        if (scalar < 0) scalar = 0;
    return value;
}

void validate_model(const std::string & path, const std::string & expected_sha256) {
    require(digest_valid(expected_sha256),
            "a lowercase GGUF SHA-256 from the converter (or quantizer census) receipt is required");
    require(std::filesystem::is_regular_file(path), "model must be a regular GGUF file");
    const uintmax_t bytes = std::filesystem::file_size(path);
    require(bytes >= 32 && bytes <= UINT64_C(512) * 1024 * 1024, "model exceeds the bounded512MiB weight profile");
    const auto observed = core_sha256::file(path);
    require(!observed.empty() && observed == expected_sha256, "GGUF digest differs from supplied converter receipt");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path.c_str()),
                                                                            &core_gguf::free_metadata);
    require(bool(meta), "cannot read GGUF metadata");
    auto integer = [&](const char * key, uint32_t expected) {
        const int64_t id = gguf_find_key(meta.get(), key);
        return id >= 0 && gguf_get_kv_type(meta.get(), id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(meta.get(), id) == expected;
    };
    auto string = [&](const char * key, const char * expected) {
        return core_gguf::kv_str(meta.get(), key, "") == expected;
    };
    const int64_t lfu = gguf_find_key(meta.get(), "lama.lfu");
    // general.file_type 0 is the converter's exact F32 file. Any other value is
    // accepted only from crispembed-quantize's "lama-v2" policy (tools/quantize.cpp);
    // its per-tensor storage is checked again by the stored-weight loader.
    const int64_t file_type = gguf_find_key(meta.get(), "general.file_type");
    const bool file_type_valid =
        file_type >= 0 && gguf_get_kv_type(meta.get(), file_type) == GGUF_TYPE_UINT32 &&
        (gguf_get_val_u32(meta.get(), file_type) == 0 || string("lama.quantization.policy", "lama-v2"));
    require(string("general.architecture", "big_lama") && file_type_valid && string("general.license", "Apache-2.0") &&
                integer("lama.schema", 1) && string("lama.source.commit", detail::source_commit()) &&
                string("lama.contract.sha256", detail::source_contract_sha256()) &&
                string("lama.checkpoint.sha256", detail::checkpoint_sha256()) &&
                string("lama.checkpoint.revision", detail::checkpoint_revision()) &&
                string("lama.checkpoint.container", detail::checkpoint_container()) &&
                string("lama.checkpoint.member", detail::checkpoint_member()) &&
                string("lama.input.contract", detail::input_contract()) &&
                integer("lama.image.min_dimension", min_dimension) &&
                integer("lama.image.max_dimension", max_dimension) &&
                string("lama.weights.license.status", detail::weight_license_status()) &&
                string("lama.tensor.layout", detail::tensor_layout()) && integer("lama.generator.blocks", 18) &&
                string("lama.fft.norm", "ortho") && lfu >= 0 && gguf_get_kv_type(meta.get(), lfu) == GGUF_TYPE_BOOL &&
                !gguf_get_val_bool(meta.get(), lfu),
            "source, checkpoint, licence or generator contract differs");
}

// Weight-dependent operators of the generator. The converter's F32 file runs
// through the shared latent_inpaint::Engine (unchanged default). A lama-v2
// F16/quantized file, or an F32 file under CRISPEMBED_LAMA_STORED_OPS=1 (an
// A/B control), runs through StoredOps below, which repeats Engine's exact
// im2col + ggml_mul_mat algorithm for F32 kernels.
struct Ops {
    virtual ~Ops() = default;
    virtual bool has(const std::string & name) const = 0;
    virtual Tensor conv(const std::string & name, const Tensor & x, int stride) = 0;
    virtual Tensor conv_transpose(const std::string & name, const Tensor & x) = 0;
    virtual Tensor batch_norm(const std::string & name, const Tensor & x) const = 0;
};

struct EngineOps final : Ops {
    Engine engine;
    EngineOps(const std::string & path, int threads) : engine(path.c_str(), threads, detail::schema()) {}
    bool has(const std::string & name) const override { return engine.has(name); }
    Tensor conv(const std::string & name, const Tensor & x, int stride) override {
        return engine.conv(name, x, stride, 0);
    }
    Tensor conv_transpose(const std::string & name, const Tensor & x) override {
        return engine.conv_transpose(name, x, 2, 1, 1);
    }
    Tensor batch_norm(const std::string & name, const Tensor & x) const override { return engine.batch_norm(name, x); }
};

constexpr size_t patch_budget_elements = size_t(8) * 1024 * 1024; // as latent_inpaint_ops.cpp

struct OpGraph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    OpGraph() {
        constexpr size_t nodes = 64;
        ctx = ggml_init({ nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true });
        require(ctx != nullptr, "operator graph metadata allocation failed");
        graph = ggml_new_graph_custom(ctx, nodes, false);
    }
    ~OpGraph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
    OpGraph(const OpGraph &) = delete;
    OpGraph & operator=(const OpGraph &) = delete;
    void allocate(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        alloc = ggml_gallocr_new(core_cpu_backend::buffer_type());
        require(alloc && ggml_gallocr_alloc_graph(alloc, graph), "bounded operator allocation failed");
    }
};

bool stored_type_allowed(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
        return true;
    default:
        return false;
    }
}

bool include_listed(const char * name, void * user) {
    return static_cast<const std::set<std::string> *>(user)->count(name) != 0;
}

bool seek(FILE * file, size_t offset) {
#ifdef _WIN32
    return _fseeki64(file, int64_t(offset), SEEK_SET) == 0;
#else
    return fseeko(file, off_t(offset), SEEK_SET) == 0;
#endif
}

struct StoredOps final : Ops {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad resident; // tensors used in their stored type and layout
    ggml_context * converted_ctx = nullptr;
    ggml_backend_buffer_t converted_buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::map<std::string, std::vector<int64_t>> shapes; // original PyTorch order
    size_t quantized_kernels = 0, dequantized_kernels = 0, weight_bytes = 0;

    ~StoredOps() override {
        if (converted_buf) ggml_backend_buffer_free(converted_buf);
        if (converted_ctx) ggml_free(converted_ctx);
        core_gguf::free_weights(resident);
        if (backend) ggml_backend_free(backend);
    }

    StoredOps(const std::string & path, int threads) {
        const bool matmul = core_env::on("CRISPEMBED_LAMA_QUANT_MATMUL");
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path.c_str()),
                                                                                &core_gguf::free_metadata);
        require(bool(meta), "cannot read GGUF metadata");
        const auto schema = detail::schema();
        require(size_t(gguf_get_n_tensors(meta.get())) == schema.size(), "checkpoint tensor count differs");
        const bool quantized = core_gguf::kv_str(meta.get(), "lama.quantization.policy", "") == "lama-v2";
        struct Pending {
            std::string name;
            int64_t id;
        };
        std::set<std::string> keep;
        std::vector<Pending> convert;
        for (const auto & item : schema) {
            const int64_t id = gguf_find_tensor(meta.get(), item.first.c_str());
            require(id >= 0, "missing tensor: " + item.first);
            const ggml_type type = gguf_get_tensor_type(meta.get(), id);
            const auto & shape = item.second;
            std::array<int64_t, 4> expected{ 1, 1, 1, 1 };
            int64_t count = 1;
            for (size_t i = 0; i < shape.size(); ++i) {
                expected[shape.size() - i - 1] = shape[i];
                count *= shape[i];
            }
            const int64_t * ne = gguf_get_tensor_ne(meta.get(), id);
            bool exact = true;
            for (int axis = 0; axis < 4; ++axis) exact &= ne[axis] == expected[axis];
            const bool kernel = shape.size() == 4;
            const bool pinned = !kernel || item.first == "generator.model.1.ffc.convl2l.weight" ||
                                item.first == "generator.model.34.weight";
            if (pinned || !quantized) {
                require(type == GGML_TYPE_F32 && exact, "tensor must be F32 in its original shape: " + item.first);
            } else {
                require(stored_type_allowed(type), "unsupported stored tensor type: " + item.first);
                const bool flattened = ne[2] == 1 && ne[3] == 1 && ne[0] * ne[1] == count;
                require((type == GGML_TYPE_F32 || type == GGML_TYPE_F16) ? (exact || flattened) : flattened,
                        "tensor storage shape differs: " + item.first);
            }
            shapes[item.first] = shape;
            const bool transpose = item.first == "generator.model.24.weight" ||
                                   item.first == "generator.model.27.weight" ||
                                   item.first == "generator.model.30.weight";
            const bool patch_rows = exact || ne[0] == expected[0] * expected[1] * expected[2];
            const bool direct = type == GGML_TYPE_F32 ? exact : (matmul && kernel && !transpose && patch_rows);
            if (direct)
                keep.insert(item.first);
            else
                convert.push_back({ item.first, id });
            if (kernel && type != GGML_TYPE_F32) ++(direct ? quantized_kernels : dequantized_kernels);
        }
        backend = core_cpu_backend::init();
        require(backend != nullptr, "CPU backend unavailable");
        core_cpu_backend::set_n_threads(backend, threads);
        require(core_gguf::load_weights_filtered(path.c_str(), backend, include_listed, &keep, "lama", resident) &&
                    resident.tensors.size() == keep.size(),
                "cannot load resident weights");
        weight_bytes = resident.buf ? ggml_backend_buffer_get_size(resident.buf) : 0;
        for (const auto & entry : resident.tensors) {
            require(entry.second->data != nullptr, "CPU weight storage unavailable");
            tensors[entry.first] = entry.second;
        }
        if (!convert.empty()) {
            converted_ctx = ggml_init({ convert.size() * ggml_tensor_overhead() + 1024, nullptr, true });
            require(converted_ctx != nullptr, "dequantized weight metadata allocation failed");
            std::vector<ggml_tensor *> made;
            for (const auto & item : convert) {
                const auto & shape = shapes.at(item.name);
                std::array<int64_t, 4> ne{ 1, 1, 1, 1 };
                for (size_t i = 0; i < shape.size(); ++i) ne[shape.size() - i - 1] = shape[i];
                ggml_tensor * t = ggml_new_tensor_4d(converted_ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
                ggml_set_name(t, item.name.c_str());
                made.push_back(t);
            }
            converted_buf = ggml_backend_alloc_ctx_tensors(converted_ctx, backend);
            require(converted_buf != nullptr, "dequantized weight allocation failed");
            weight_bytes += ggml_backend_buffer_get_size(converted_buf);
            std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
            require(bool(file), "cannot reopen GGUF for dequantization");
            const size_t data_offset = gguf_get_data_offset(meta.get());
            std::vector<uint8_t> raw;
            std::vector<float> values;
            for (size_t i = 0; i < convert.size(); ++i) {
                const ggml_type type = gguf_get_tensor_type(meta.get(), convert[i].id);
                const int64_t count = ggml_nelements(made[i]);
                raw.resize(gguf_get_tensor_size(meta.get(), convert[i].id));
                values.resize(size_t(count));
                require(raw.size() == ggml_row_size(type, count) &&
                            seek(file.get(), data_offset + gguf_get_tensor_offset(meta.get(), convert[i].id)) &&
                            std::fread(raw.data(), 1, raw.size(), file.get()) == raw.size(),
                        "cannot read stored tensor: " + convert[i].name);
                if (type == GGML_TYPE_F32) {
                    std::memcpy(values.data(), raw.data(), raw.size());
                } else {
                    const ggml_type_traits * traits = ggml_get_type_traits(type);
                    require(traits && traits->to_float, "no dequantizer for tensor: " + convert[i].name);
                    traits->to_float(raw.data(), values.data(), count);
                }
                require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }),
                        "nonfinite stored tensor: " + convert[i].name);
                ggml_backend_tensor_set(made[i], values.data(), 0, values.size() * sizeof(float));
                tensors[convert[i].name] = made[i];
            }
        }
        std::fprintf(stderr,
                     "Big-LaMa stored-weight ops: %s, %zu kernels in stored type (mul_mat), %zu dequantized to F32, "
                     "%zu weight bytes\n",
                     quantized ? "lama-v2" : "F32", quantized_kernels, dequantized_kernels, weight_bytes);
    }

    ggml_tensor * tensor(const std::string & name) const {
        const auto found = tensors.find(name);
        require(found != tensors.end(), "missing weight: " + name);
        return found->second;
    }
    const float * vector(const std::string & name, size_t count) const {
        ggml_tensor * t = tensor(name);
        require(t->type == GGML_TYPE_F32 && size_t(ggml_nelements(t)) == count, "vector weight shape differs: " + name);
        return static_cast<const float *>(t->data);
    }
    bool has(const std::string & name) const override { return tensors.count(name) != 0; }

    Tensor conv(const std::string & name, const Tensor & x, int stride) override {
        require(latent_inpaint::finite(x) && stride >= 1 && stride <= 4, "invalid convolution input: " + name);
        const auto & shape = shapes.at(name + ".weight");
        const int oc = int(shape[0]), ic = int(shape[1]), kh = int(shape[2]), kw = int(shape[3]);
        require(ic == x.c && x.h >= kh && x.w >= kw, "convolution channels differ: " + name);
        const int oh = (x.h - kh) / stride + 1, ow = (x.w - kw) / stride + 1;
        Tensor out(oc, oh, ow);
        const float * bias = has(name + ".bias") ? vector(name + ".bias", size_t(oc)) : nullptr;
        ggml_tensor * kernel = tensor(name + ".weight");
        const size_t inner = size_t(ic) * kh * kw;
        require(inner <= patch_budget_elements, "convolution patch exceeds workspace budget");
        const size_t tile = std::min({ out.spatial(), size_t(4096), patch_budget_elements / inner });
        std::vector<float> patches(inner * tile), product(tile * size_t(oc));
        OpGraph graph;
        ggml_tensor * input = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, int64_t(inner), int64_t(tile));
        ggml_set_input(input);
        ggml_tensor * flat = ggml_n_dims(kernel) == 2 && kernel->ne[0] == int64_t(inner)
                                 ? kernel
                                 : ggml_reshape_2d(graph.ctx, kernel, int64_t(inner), oc);
        // F32: exactly Engine's [K,pixels] x [K,output] -> [pixels,output] (CHW).
        // Stored F16/quantized kernels must be src0: [K,output] x [K,pixels] ->
        // [output,pixels] (HWC), transposed while copying out.
        const bool f32 = kernel->type == GGML_TYPE_F32;
        ggml_tensor * y = f32 ? ggml_mul_mat(graph.ctx, input, flat) : ggml_mul_mat(graph.ctx, flat, input);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        graph.allocate(y);
        for (size_t start = 0; start < out.spatial(); start += tile) {
            const size_t used = std::min(tile, out.spatial() - start);
            std::fill(patches.begin(), patches.end(), 0.0f);
            for (size_t p = 0; p < used; ++p) {
                const int oy = int((start + p) / ow), ox = int((start + p) % ow);
                float * patch = patches.data() + p * inner;
                for (int ch = 0; ch < ic; ++ch)
                    for (int ky = 0; ky < kh; ++ky)
                        for (int kx = 0; kx < kw; ++kx)
                            patch[(size_t(ch) * kh + ky) * kw + kx] = x.at(ch, oy * stride + ky, ox * stride + kx);
            }
            ggml_backend_tensor_set(input, patches.data(), 0, patches.size() * sizeof(float));
            require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "CPU operator failed");
            ggml_backend_tensor_get(y, product.data(), 0, product.size() * sizeof(float));
            for (int ch = 0; ch < oc; ++ch) {
                float * dst = out.v.data() + size_t(ch) * out.spatial() + start;
                if (f32)
                    std::copy_n(product.data() + size_t(ch) * tile, used, dst);
                else
                    for (size_t p = 0; p < used; ++p) dst[p] = product[p * size_t(oc) + ch];
            }
        }
        if (bias)
            for (int ch = 0; ch < oc; ++ch)
                for (size_t p = 0; p < out.spatial(); ++p) out.v[size_t(ch) * out.spatial() + p] += bias[ch];
        return out;
    }

    Tensor conv_transpose(const std::string & name, const Tensor & x) override {
        // Engine::conv_transpose(stride 2, padding 1, output_padding 1) on the F32 kernel.
        constexpr int stride = 2, padding = 1, output_padding = 1;
        require(latent_inpaint::finite(x), "invalid transposed convolution input: " + name);
        const auto & shape = shapes.at(name + ".weight");
        require(shape[0] == x.c, "transposed convolution channels differ: " + name);
        const int oc = int(shape[1]), kh = int(shape[2]), kw = int(shape[3]);
        const int raw_h = (x.h - 1) * stride + kh, raw_w = (x.w - 1) * stride + kw;
        Tensor out(oc, raw_h - 2 * padding + output_padding, raw_w - 2 * padding + output_padding);
        const float * bias = has(name + ".bias") ? vector(name + ".bias", size_t(oc)) : nullptr;
        ggml_tensor * kernel = tensor(name + ".weight");
        require(kernel->type == GGML_TYPE_F32 && ggml_n_dims(kernel) == 4, "transposed kernel must be F32 4D");
        OpGraph graph;
        ggml_tensor * input = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, x.w, x.h, x.c, 1);
        ggml_set_input(input);
        ggml_tensor * y = ggml_conv_transpose_2d_p0(graph.ctx, kernel, input, stride);
        require(y->ne[0] == raw_w && y->ne[1] == raw_h && y->ne[2] == oc, "transposed convolution output differs");
        graph.allocate(y);
        ggml_backend_tensor_set(input, x.v.data(), 0, x.v.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "CPU operator failed");
        std::vector<float> row(size_t(out.w), 0.0f);
        for (int ch = 0; ch < oc; ++ch)
            for (int oy = 0; oy < out.h; ++oy) {
                const size_t offset = (size_t(ch) * raw_h + oy + padding) * raw_w + padding;
                ggml_backend_tensor_get(y, row.data(), offset * sizeof(float), row.size() * sizeof(float));
                for (int ox = 0; ox < out.w; ++ox) out.at(ch, oy, ox) = row[size_t(ox)] + (bias ? bias[ch] : 0.0f);
            }
        return out;
    }

    Tensor batch_norm(const std::string & name, const Tensor & x) const override {
        // Engine::batch_norm, eps 1e-5, identical expression order.
        constexpr float eps = 1e-5f;
        const size_t c = size_t(x.c);
        const float *scale = vector(name + ".weight", c), *bias = vector(name + ".bias", c),
                    *mean = vector(name + ".running_mean", c), *variance = vector(name + ".running_var", c);
        Tensor out(x.c, x.h, x.w);
        for (int ch = 0; ch < x.c; ++ch) {
            require(variance[ch] >= 0, "negative BatchNorm variance");
            const float factor = scale[ch] / std::sqrt(variance[ch] + eps);
            for (size_t p = 0; p < x.spatial(); ++p)
                out.v[size_t(ch) * x.spatial() + p] =
                    (x.v[size_t(ch) * x.spatial() + p] - mean[ch]) * factor + bias[ch];
        }
        return out;
    }
};

bool quantized_file(const std::string & path) {
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path.c_str()),
                                                                            &core_gguf::free_metadata);
    require(bool(meta), "cannot read GGUF metadata");
    return core_gguf::kv_str(meta.get(), "lama.quantization.policy", "") == "lama-v2";
}
} // namespace

struct context {
    std::unique_ptr<Ops> engine;
    std::unique_ptr<GraphModel> graph;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
#ifdef CRISPEMBED_LAMA_TESTING
    Observer observer = nullptr;
    void * observer_opaque = nullptr;
#endif
    void mark(const std::string & name, const Tensor & value) const {
#ifdef CRISPEMBED_LAMA_TESTING
        if (observer) observer(observer_opaque, name, value);
#else
        (void)name;
        (void)value;
#endif
    }
};

namespace detail {
Tensor reflect_pad(const Tensor & input, int padding) {
    require(latent_inpaint::finite(input) && padding >= 0 && padding <= 3 && input.h > padding && input.w > padding,
            "invalid reflection-padding input or extent");
    if (padding == 0) return input;
    Tensor result(input.c, input.h + 2 * padding, input.w + 2 * padding);
    for (int ch = 0; ch < input.c; ++ch)
        for (int y = 0; y < result.h; ++y)
            for (int x = 0; x < result.w; ++x)
                result.at(ch, y, x) =
                    input.at(ch, reflect_index(y - padding, input.h), reflect_index(x - padding, input.w));
    return result;
}

Tensor preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width,
                       int height) {
    buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
    Tensor result(4, rounded_eight(height), rounded_eight(width));
    for (int y = 0; y < result.h; ++y)
        for (int x = 0; x < result.w; ++x) {
            const size_t from = size_t(symmetric_index(y, height)) * width + symmetric_index(x, width);
            const float fill = mask[from] == 255 ? 1.0f : 0.0f;
            for (int ch = 0; ch < 3; ++ch) result.at(ch, y, x) = (float(rgb[from * 3 + ch]) / 255.0f) * (1.0f - fill);
            result.at(3, y, x) = fill;
        }
    return result;
}

void composite_rgb8(const Tensor & prediction, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask,
                    size_t mask_bytes, int width, int height, uint8_t * output, size_t output_bytes) {
    buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
    require(output && output_bytes == rgb_bytes && prediction.c == 3 && prediction.h == rounded_eight(height) &&
                prediction.w == rounded_eight(width) && latent_inpaint::finite(prediction),
            "invalid predicted RGB tensor or output buffer");
    std::vector<uint8_t> result(rgb, rgb + rgb_bytes);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t pixel = size_t(y) * width + x;
            if (mask[pixel])
                for (int ch = 0; ch < 3; ++ch)
                    result[pixel * 3 + ch] = uint8_t(std::clamp(prediction.at(ch, y, x), 0.0f, 1.0f) * 255.0f);
        }
    std::memcpy(output, result.data(), result.size());
}
} // namespace detail

namespace {
struct Features {
    Tensor local, global;
};

Tensor reflected_conv(Ops & engine, const std::string & name, const Tensor & input, int stride, int padding) {
    return engine.conv(name, padding ? detail::reflect_pad(input, padding) : input, stride);
}

Tensor spectral(context & model, const std::string & name, const Tensor & input) {
    Ops & engine = *model.engine;
    Tensor x = relu(engine.batch_norm(name + ".conv1.1", engine.conv(name + ".conv1.0", input, 1)));
    Tensor packed = detail::rfft2_ortho(x);
    model.mark(name + ".fu.rfft", packed);
    packed = relu(engine.batch_norm(name + ".fu.bn", engine.conv(name + ".fu.conv_layer", packed, 1)));
    model.mark(name + ".fu.spectral", packed);
    Tensor frequency = detail::irfft2_ortho(packed, x.w);
    model.mark(name + ".fu.irfft", frequency);
    // The pinned Big-LaMa config disables LFU and all spatial rescaling.
    return engine.conv(name + ".conv2", latent_inpaint::add(x, frequency), 1);
}

Features ffc(context & model, const std::string & name, const Features & input, int stride, int padding) {
    Ops & engine = *model.engine;
    const std::string prefix = name + ".ffc";
    Features result;
    result.local = reflected_conv(engine, prefix + ".convl2l", input.local, stride, padding);
    if (input.global.c) {
        result.local = latent_inpaint::add(result.local,
                                           reflected_conv(engine, prefix + ".convg2l", input.global, stride, padding));
    }
    if (engine.has(prefix + ".convl2g.weight")) {
        result.global = reflected_conv(engine, prefix + ".convl2g", input.local, stride, padding);
        if (input.global.c) {
            // Only residual blocks contain global inputs in this fixed graph;
            // their stride is one, so SpectralTransform has no AvgPool branch.
            require(stride == 1, "unsupported global downsampling outside the pinned architecture");
            result.global = latent_inpaint::add(result.global, spectral(model, prefix + ".convg2g", input.global));
        }
    }
    result.local = relu(engine.batch_norm(name + ".bn_l", result.local));
    if (result.global.c) result.global = relu(engine.batch_norm(name + ".bn_g", result.global));
    model.mark(name + ".local", result.local);
    if (result.global.c) model.mark(name + ".global", result.global);
    return result;
}

Tensor forward(context & model, const Tensor & input) {
    enabled();
    require(input.c == 4 && input.h >= 16 && input.w >= 16 && input.h <= max_dimension && input.w <= max_dimension &&
                input.h % 8 == 0 && input.w % 8 == 0 && latent_inpaint::finite(input),
            "invalid padded generator input");
    for (size_t p = 0; p < input.spatial(); ++p) {
        const float mask = input.v[3 * input.spatial() + p];
        require(mask == 0 || mask == 1, "generator mask must be binary0/1");
        for (int ch = 0; ch < 3; ++ch) {
            const float value = input.v[size_t(ch) * input.spatial() + p];
            require(value >= 0 && value <= 1 && (!mask || value == 0), "generator requires masked RGB in[0,1]");
        }
    }
    if (model.graph) {
#ifdef CRISPEMBED_LAMA_TESTING
        model.graph->set_observer(model.observer, model.observer_opaque);
#endif
        return model.graph->run(input);
    }
    model.mark("input", input);
    Features features = ffc(model, "generator.model.1", { detail::reflect_pad(input, 3), {} }, 1, 0);
    for (int block = 2; block <= 4; ++block)
        features = ffc(model, "generator.model." + std::to_string(block), features, 2, 1);
    for (int block = 5; block <= 22; ++block) {
        const std::string name = "generator.model." + std::to_string(block);
        Features residual = ffc(model, name + ".conv1", features, 1, 1);
        residual = ffc(model, name + ".conv2", residual, 1, 1);
        features.local = latent_inpaint::add(features.local, residual.local);
        features.global = latent_inpaint::add(features.global, residual.global);
        model.mark(name + ".local", features.local);
        model.mark(name + ".global", features.global);
    }
    Tensor x = latent_inpaint::concat(features.local, features.global);
    model.mark("generator.model.23", x);
    for (int block : { 24, 27, 30 }) {
        x = model.engine->conv_transpose("generator.model." + std::to_string(block), x);
        x = relu(model.engine->batch_norm("generator.model." + std::to_string(block + 1), x));
        model.mark("generator.model." + std::to_string(block + 2), x);
    }
    x = model.engine->conv("generator.model.34", detail::reflect_pad(x, 3), 1);
    for (float & value : x.v) value = 1.0f / (1.0f + std::exp(-value));
    require(latent_inpaint::finite(x), "nonfinite generated RGB");
    model.mark("prediction", x);
    return x;
}

struct Busy {
    context & model;
    explicit Busy(context & value) : model(value) {
        require(!model.busy.test_and_set(), "one context cannot process concurrent requests");
    }
    ~Busy() { model.busy.clear(); }
};
} // namespace

context * init(const std::string & path, const std::string & expected_gguf_sha256, int threads, std::string & error) {
    error.clear();
    try {
        enabled();
        require(threads >= 1 && threads <= 64, "thread count must be in[1,64]");
        validate_model(path, expected_gguf_sha256);
        auto model = std::make_unique<context>();
        // Persistent ggml graph (delivery 4, CRISPEMBED_INPAINT_GRAPH=1), else the
        // stored-weight ops for quantized/F16 files (research lama-v2), else the F32 engine.
        if (core_env::on("CRISPEMBED_INPAINT_GRAPH"))
            model->graph = std::make_unique<GraphModel>(path.c_str(), threads);
        else if (quantized_file(path) || core_env::on("CRISPEMBED_LAMA_STORED_OPS"))
            model->engine = std::make_unique<StoredOps>(path, threads);
        else
            model->engine = std::make_unique<EngineOps>(path, threads);
        return model.release();
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
}
void free(context * model) {
    delete model;
}

bool process_rgb8(context * model, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    error.clear();
    try {
        require(model != nullptr, "null context");
        enabled();
        Busy lock(*model);
        buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
        require(output && output_bytes == rgb_bytes, "output byte count differs");
        if (std::none_of(mask, mask + mask_bytes, [](uint8_t value) { return value != 0; })) {
            std::memmove(output, rgb, rgb_bytes);
            return true;
        }
        Tensor input = detail::preprocess_rgb8(rgb, rgb_bytes, mask, mask_bytes, width, height);
        Tensor prediction = forward(*model, input);
        detail::composite_rgb8(prediction, rgb, rgb_bytes, mask, mask_bytes, width, height, output, output_bytes);
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}

#ifdef CRISPEMBED_LAMA_TESTING
void set_observer(context * model, Observer callback, void * opaque) {
    require(model != nullptr, "null reference context");
    model->observer = callback;
    model->observer_opaque = opaque;
}
bool run_reference(context * model, const Tensor & input, Tensor & prediction, std::string & error) {
    error.clear();
    try {
        require(model != nullptr, "null reference context");
        Busy lock(*model);
        Tensor result = forward(*model, input);
        prediction = std::move(result);
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}
#endif
} // namespace lama
