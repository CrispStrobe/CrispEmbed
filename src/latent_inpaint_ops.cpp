#include "latent_inpaint_ops.h"

#include "core/cpu_ops.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "inpaint_weights.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace latent_inpaint {
namespace {
constexpr size_t max_tensor_elements = size_t(128) * 1024 * 1024;
constexpr size_t patch_budget_elements = size_t(8) * 1024 * 1024;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("latent inpaint: " + message);
}

void valid(const Tensor & x) {
    require(x.c > 0 && x.c <= 16384 && x.h > 0 && x.h <= 2048 && x.w > 0 && x.w <= 2048 &&
                size_t(x.c) <= max_tensor_elements / size_t(x.h) / size_t(x.w) && x.v.size() == size_t(x.c) * x.h * x.w,
            "invalid CHW tensor");
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    Graph() {
        constexpr size_t nodes = 64;
        ctx = ggml_init({ nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true });
        require(ctx != nullptr, "operator graph metadata allocation failed");
        graph = ggml_new_graph_custom(ctx, nodes, false);
    }
    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
    void allocate(ggml_tensor * output) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        alloc = ggml_gallocr_new(core_cpu_backend::buffer_type());
        require(alloc && ggml_gallocr_alloc_graph(alloc, graph), "bounded operator allocation failed");
    }
};
} // namespace

Tensor::Tensor(int channels, int height, int width) : c(channels), h(height), w(width) {
    require(c > 0 && h > 0 && w > 0 && c <= 16384 && h <= 2048 && w <= 2048,
            "tensor dimensions exceed the bounded CPU profile");
    require(size_t(c) <= max_tensor_elements / size_t(h) / size_t(w), "tensor exceeds 512 MiB activation limit");
    v.resize(size_t(c) * h * w);
}

struct Engine::Impl {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad loaded;
    // inpaint-v1 precision variants only: dequantized copies (default) or
    // stored F16/block matrices consumed by mul_mat (QUANT_MATMUL gate).
    std::unique_ptr<inpaint_weights::Converted> converted;
    std::unordered_map<std::string, Weight> views;
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    Observer observer = nullptr;
    void * opaque = nullptr;
#endif
    ~Impl() {
        converted.reset();
        core_gguf::free_weights(loaded);
        if (backend) ggml_backend_free(backend);
    }
    void compute(Graph & graph) {
        require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "CPU operator failed");
    }
};

Engine::Engine(const char * path, int threads, const ShapeMap & schema) : impl_(new Impl) {
    require(path && *path && threads >= 1 && threads <= 64, "invalid model path or thread count");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path),
                                                                            &core_gguf::free_metadata);
    require(bool(meta), "cannot read GGUF metadata");
    require(size_t(gguf_get_n_tensors(meta.get())) == schema.size(), "checkpoint tensor count differs");
    if (inpaint_weights::is_policy_file(meta.get())) {
        load_precision_variant(path, threads, schema, meta.get());
        return;
    }
    size_t total_elements = 0;
    for (const auto & item : schema) {
        const int64_t id = gguf_find_tensor(meta.get(), item.first.c_str());
        require(id >= 0 && gguf_get_tensor_type(meta.get(), id) == GGML_TYPE_F32,
                "missing or non-F32 tensor: " + item.first);
        const auto & shape = item.second;
        require(!shape.empty() && shape.size() <= 4, "invalid source schema rank");
        std::array<int64_t, 4> expected{ 1, 1, 1, 1 };
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            require(shape[i] > 0 && size_t(shape[i]) <= size_t(1) << 24, "invalid source schema dimension");
            require(count <= size_t(1024) * 1024 * 1024 / size_t(shape[i]), "weight size overflow");
            count *= size_t(shape[i]);
            expected[shape.size() - i - 1] = shape[i];
        }
        const int64_t * actual = gguf_get_tensor_ne(meta.get(), id);
        for (int axis = 0; axis < 4; ++axis)
            require(actual[axis] == expected[axis], "tensor shape differs: " + item.first);
        total_elements += count;
    }
    require(total_elements <= size_t(1024) * 1024 * 1024, "model exceeds 4 GiB F32 weight limit");
    meta.reset();
    impl_->backend = core_cpu_backend::init();
    require(impl_->backend != nullptr, "CPU backend unavailable");
    core_cpu_backend::set_n_threads(impl_->backend, threads);
    // Shared loader owns and releases the mapping. Do not duplicate 3.45 GB
    // denoiser weights in a second float cache. The fallback is the loader's
    // ordinary CPU allocation when zero-copy mmap is unavailable.
    require(core_gguf::load_weights(path, impl_->backend, "latent-inpaint", impl_->loaded, true),
            "cannot load CPU weights");
    for (const auto & item : schema) {
        ggml_tensor * tensor = impl_->loaded.tensors.at(item.first);
        require(tensor->data != nullptr, "CPU weight storage unavailable");
        impl_->views.emplace(item.first, Weight{ item.second, static_cast<const float *>(tensor->data),
                                                 size_t(ggml_nelements(tensor)) });
    }
}

void Engine::load_precision_variant(const char * path, int threads, const ShapeMap & schema, gguf_context * meta) {
    inpaint_weights::ShapeMap expected;
    for (const auto & item : schema) {
        const auto & shape = item.second;
        require(!shape.empty() && shape.size() <= 4, "invalid source schema rank");
        inpaint_weights::Shape ne{ 1, 1, 1, 1 };
        for (size_t i = 0; i < shape.size(); ++i) ne[shape.size() - i - 1] = shape[i];
        expected.emplace(item.first, ne);
    }
    std::string error;
    require(inpaint_weights::validate(meta, expected, "latent inpaint", error), error);
    impl_->backend = core_cpu_backend::init();
    require(impl_->backend != nullptr, "CPU backend unavailable");
    core_cpu_backend::set_n_threads(impl_->backend, threads);
    impl_->converted.reset(new inpaint_weights::Converted);
    inpaint_weights::Stats stats;
    require(inpaint_weights::load(path, meta, impl_->backend, expected, inpaint_weights::matmul_gate(),
                                  "latent inpaint", impl_->loaded, *impl_->converted, stats, error),
            error);
    for (const auto & item : schema) {
        ggml_tensor * tensor = impl_->loaded.tensors.at(item.first);
        require(tensor->data != nullptr, "CPU weight storage unavailable");
        const bool stored = tensor->type != GGML_TYPE_F32;
        // Direct float readers (embeddings, relative positions, depthwise and
        // normalization vectors) must be F32; only conv(groups=1)/linear
        // matrices may stay stored for the gated mul_mat path.
        require(!stored || ((item.second.size() == 2 || (item.second.size() == 4 && item.second[1] > 1)) &&
                            item.first.find("rel_pos_emb") == std::string::npos &&
                            item.first.find("embedding_layer") == std::string::npos),
                "stored precision tensor has a direct float consumer: " + item.first);
        impl_->views.emplace(item.first,
                             Weight{ item.second, stored ? nullptr : static_cast<const float *>(tensor->data),
                                     size_t(ggml_nelements(tensor)) });
    }
}

Engine::~Engine() = default;
const Weight & Engine::weight(const std::string & name) const {
    const auto found = impl_->views.find(name);
    require(found != impl_->views.end(), "missing weight: " + name);
    return found->second;
}
bool Engine::has(const std::string & name) const {
    return impl_->views.count(name) != 0;
}

Tensor Engine::conv(const std::string & name, const Tensor & x, int stride, int padding, int groups, int dilation) {
    valid(x);
    const Weight & wt = weight(name + ".weight");
    require(wt.shape.size() == 4 && stride > 0 && stride <= 4 && padding >= 0 && padding <= 32 && dilation > 0 &&
                dilation <= 4 && groups > 0 && x.c % groups == 0,
            "invalid convolution: " + name);
    const int oc = int(wt.shape[0]), ic = int(wt.shape[1]), kh = int(wt.shape[2]), kw = int(wt.shape[3]);
    require(ic * groups == x.c && oc % groups == 0, "convolution channels differ: " + name);
    const int oh = (x.h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    const int ow = (x.w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    require(x.h + 2 * padding >= dilation * (kh - 1) + 1 && x.w + 2 * padding >= dilation * (kw - 1) + 1,
            "convolution kernel exceeds input");
    Tensor out(oc, oh, ow);
    const Weight * bias = has(name + ".bias") ? &weight(name + ".bias") : nullptr;
    require(!bias || bias->size == size_t(oc), "convolution bias shape differs");
    ggml_tensor * kernel = impl_->loaded.tensors.at(name + ".weight");
    const bool stored = kernel->type != GGML_TYPE_F32;
    require(!stored || groups == 1, "stored precision kernel requires groups=1: " + name);

    if (groups == x.c && oc == x.c) {
        require(!stored, "depthwise kernel must be F32: " + name);
        // Reuse the same direct F32 depthwise ggml operator as MI-GAN. Unlike
        // legacy depthwise im2col this does not round activations to F16.
        Graph graph;
        ggml_tensor * input = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, x.w, x.h, x.c, 1);
        ggml_set_input(input);
        ggml_tensor * y =
            ggml_conv_2d_dw_direct(graph.ctx, kernel, input, stride, stride, padding, padding, dilation, dilation);
        graph.allocate(y);
        ggml_backend_tensor_set(input, x.v.data(), 0, x.v.size() * sizeof(float));
        impl_->compute(graph);
        ggml_backend_tensor_get(y, out.v.data(), 0, out.v.size() * sizeof(float));
    } else {
        // Bound im2col to 32 MiB. A full 512px VAE im2col can otherwise consume
        // over a GiB at one layer. Graph allocation is reused across tiles.
        const size_t inner = size_t(ic) * kh * kw;
        require(inner <= patch_budget_elements, "convolution patch exceeds workspace budget");
        const size_t tile = std::min({ out.spatial(), size_t(4096), patch_budget_elements / inner });
        const int ocg = oc / groups;
        std::vector<float> patches(inner * tile), product(tile * ocg);
        for (int group = 0; group < groups; ++group) {
            Graph graph;
            ggml_tensor * input = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, int64_t(inner), int64_t(tile));
            ggml_set_input(input);
            ggml_tensor * flat = ggml_reshape_2d(graph.ctx, kernel, int64_t(inner), oc);
            ggml_tensor * slice = stored ? flat
                                         : ggml_view_2d(graph.ctx, flat, int64_t(inner), ocg, inner * sizeof(float),
                                                        size_t(group) * ocg * inner * sizeof(float));
            // F32: [K,pixels] x [K,output] -> [pixels,output], already CHW.
            // Stored (gated): [K,output] x [K,pixels] -> [output,pixels].
            ggml_tensor * y = stored ? ggml_mul_mat(graph.ctx, slice, input) : ggml_mul_mat(graph.ctx, input, slice);
            ggml_mul_mat_set_prec(y, GGML_PREC_F32);
            graph.allocate(y);
            for (size_t start = 0; start < out.spatial(); start += tile) {
                const size_t used = std::min(tile, out.spatial() - start);
                std::fill(patches.begin(), patches.end(), 0.0f);
                for (size_t p = 0; p < used; ++p) {
                    const int oy = int((start + p) / ow), ox = int((start + p) % ow);
                    float * patch = patches.data() + p * inner;
                    for (int ch = 0; ch < ic; ++ch)
                        for (int ky = 0; ky < kh; ++ky) {
                            const int iy = oy * stride + ky * dilation - padding;
                            for (int kx = 0; kx < kw; ++kx) {
                                const int ix = ox * stride + kx * dilation - padding;
                                if (iy >= 0 && iy < x.h && ix >= 0 && ix < x.w)
                                    patch[(size_t(ch) * kh + ky) * kw + kx] = x.at(group * ic + ch, iy, ix);
                            }
                        }
                }
                ggml_backend_tensor_set(input, patches.data(), 0, patches.size() * sizeof(float));
                impl_->compute(graph);
                ggml_backend_tensor_get(y, product.data(), 0, product.size() * sizeof(float));
                if (stored) {
                    for (size_t p = 0; p < used; ++p)
                        for (int ch = 0; ch < ocg; ++ch)
                            out.v[size_t(ch) * out.spatial() + start + p] = product[p * size_t(ocg) + ch];
                } else {
                    for (int ch = 0; ch < ocg; ++ch)
                        std::copy_n(product.data() + size_t(ch) * tile, used,
                                    out.v.data() + size_t(group * ocg + ch) * out.spatial() + start);
                }
            }
        }
    }
    if (bias)
        for (int ch = 0; ch < oc; ++ch)
            for (size_t p = 0; p < out.spatial(); ++p) out.v[size_t(ch) * out.spatial() + p] += bias->data[ch];
    return out;
}

Tensor Engine::conv_transpose(const std::string & name, const Tensor & x, int stride, int padding, int output_padding) {
    valid(x);
    const Weight & wt = weight(name + ".weight");
    require(wt.shape.size() == 4 && wt.shape[0] == x.c && stride > 0 && stride <= 4 && padding >= 0 && padding <= 16 &&
                output_padding >= 0 && output_padding < stride && output_padding <= padding,
            "invalid transposed convolution: " + name);
    const int oc = int(wt.shape[1]), kh = int(wt.shape[2]), kw = int(wt.shape[3]);
    const int raw_h = (x.h - 1) * stride + kh, raw_w = (x.w - 1) * stride + kw;
    Tensor out(oc, raw_h - 2 * padding + output_padding, raw_w - 2 * padding + output_padding);
    const Weight * bias = has(name + ".bias") ? &weight(name + ".bias") : nullptr;
    require(!bias || bias->size == size_t(oc), "transposed convolution bias differs");
    Graph graph;
    ggml_tensor * input = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, x.w, x.h, x.c, 1);
    ggml_set_input(input);
    // The GGUF bytes preserve PyTorch [IC,OC,KH,KW], which is the native
    // ggml transposed-convolution kernel layout [KW,KH,OC,IC].
    ggml_tensor * y = ggml_conv_transpose_2d_p0(graph.ctx, impl_->loaded.tensors.at(name + ".weight"), input, stride);
    require(y->ne[0] == raw_w && y->ne[1] == raw_h && y->ne[2] == oc, "transposed convolution output differs");
    graph.allocate(y);
    ggml_backend_tensor_set(input, x.v.data(), 0, x.v.size() * sizeof(float));
    impl_->compute(graph);
    std::vector<float> row(size_t(out.w), 0.0f);
    for (int ch = 0; ch < oc; ++ch)
        for (int oy = 0; oy < out.h; ++oy) {
            const size_t offset = (size_t(ch) * raw_h + oy + padding) * raw_w + padding;
            ggml_backend_tensor_get(y, row.data(), offset * sizeof(float), row.size() * sizeof(float));
            for (int ox = 0; ox < out.w; ++ox) out.at(ch, oy, ox) = row[size_t(ox)] + (bias ? bias->data[ch] : 0.0f);
        }
    return out;
}

Tensor Engine::linear(const std::string & name, const Tensor & x, bool use_bias) {
    valid(x);
    const Weight & wt = weight(name + ".weight");
    require(wt.shape.size() == 2 && wt.shape[1] == x.c, "linear shape differs: " + name);
    const int oc = int(wt.shape[0]);
    Tensor out(oc, x.h, x.w);
    const Weight * bias = use_bias && has(name + ".bias") ? &weight(name + ".bias") : nullptr;
    require(!bias || bias->size == size_t(oc), "linear bias shape differs");
    ggml_tensor * matrix = impl_->loaded.tensors.at(name + ".weight");
    const bool stored = matrix->type != GGML_TYPE_F32;
    if (x.spatial() == 1 && !stored) {
        core_cpu::linear_cpu(x.v.data(), out.v.data(), x.c, oc, wt.data, bias ? bias->data : nullptr);
    } else {
        const size_t tile = std::min({ x.spatial(), size_t(4096), patch_budget_elements / size_t(x.c) });
        std::vector<float> rows(tile * x.c), result(tile * oc);
        Graph graph;
        ggml_tensor * input = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, x.c, int64_t(tile));
        ggml_set_input(input);
        ggml_tensor * y = stored ? ggml_mul_mat(graph.ctx, matrix, input) : ggml_mul_mat(graph.ctx, input, matrix);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        graph.allocate(y);
        for (size_t start = 0; start < x.spatial(); start += tile) {
            const size_t used = std::min(tile, x.spatial() - start);
            std::fill(rows.begin(), rows.end(), 0.0f);
            for (size_t p = 0; p < used; ++p)
                for (int ch = 0; ch < x.c; ++ch) rows[p * x.c + ch] = x.v[size_t(ch) * x.spatial() + start + p];
            ggml_backend_tensor_set(input, rows.data(), 0, rows.size() * sizeof(float));
            impl_->compute(graph);
            ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
            for (int ch = 0; ch < oc; ++ch)
                for (size_t p = 0; p < used; ++p)
                    out.v[size_t(ch) * out.spatial() + start + p] =
                        (stored ? result[p * size_t(oc) + ch] : result[size_t(ch) * tile + p]) +
                        (bias ? bias->data[ch] : 0.0f);
        }
    }
    return out;
}

Tensor Engine::group_norm(const std::string & name, const Tensor & x, int groups, float eps) const {
    valid(x);
    require(groups > 0 && x.c % groups == 0 && eps > 0, "invalid GroupNorm");
    const Weight &scale = weight(name + ".weight"), &bias = weight(name + ".bias");
    require(scale.size == size_t(x.c) && bias.size == size_t(x.c), "GroupNorm affine shape differs");
    Tensor out(x.c, x.h, x.w);
    const size_t group_size = x.v.size() / groups;
    for (int group = 0; group < groups; ++group) {
        const size_t begin = size_t(group) * group_size;
        double sum = 0;
        for (size_t i = 0; i < group_size; ++i) sum += x.v[begin + i];
        const float mean = float(sum / double(group_size));
        double variance = 0;
        for (size_t i = 0; i < group_size; ++i) {
            const double d = double(x.v[begin + i]) - mean;
            variance += d * d;
        }
        const float inv = 1.0f / std::sqrt(float(variance / double(group_size)) + eps);
        for (size_t i = 0; i < group_size; ++i) {
            const size_t index = begin + i, ch = index / x.spatial();
            out.v[index] = (x.v[index] - mean) * inv * scale.data[ch] + bias.data[ch];
        }
    }
    return out;
}

Tensor Engine::layer_norm(const std::string & name, const Tensor & x, float eps) const {
    valid(x);
    const Weight &scale = weight(name + ".weight"), &bias = weight(name + ".bias");
    require(scale.size == size_t(x.c) && bias.size == size_t(x.c) && eps > 0, "LayerNorm affine shape differs");
    Tensor out(x.c, x.h, x.w);
    std::vector<float> in(x.c), result(x.c);
    for (size_t p = 0; p < x.spatial(); ++p) {
        for (int ch = 0; ch < x.c; ++ch) in[ch] = x.v[size_t(ch) * x.spatial() + p];
        core_cpu::layernorm_cpu(in.data(), result.data(), x.c, scale.data, bias.data, eps);
        for (int ch = 0; ch < x.c; ++ch) out.v[size_t(ch) * x.spatial() + p] = result[ch];
    }
    return out;
}

Tensor Engine::batch_norm(const std::string & name, const Tensor & x, float eps) const {
    valid(x);
    const Weight &scale = weight(name + ".weight"), &bias = weight(name + ".bias"),
                 &mean = weight(name + ".running_mean"), &variance = weight(name + ".running_var");
    require(scale.size == size_t(x.c) && bias.size == size_t(x.c) && mean.size == size_t(x.c) &&
                variance.size == size_t(x.c) && eps > 0,
            "BatchNorm state shape differs");
    Tensor out(x.c, x.h, x.w);
    for (int ch = 0; ch < x.c; ++ch) {
        require(variance.data[ch] >= 0, "negative BatchNorm variance");
        const float factor = scale.data[ch] / std::sqrt(variance.data[ch] + eps);
        for (size_t p = 0; p < x.spatial(); ++p)
            out.v[size_t(ch) * x.spatial() + p] =
                (x.v[size_t(ch) * x.spatial() + p] - mean.data[ch]) * factor + bias.data[ch];
    }
    return out;
}

Tensor Engine::attention(const Tensor & q, const Tensor & k, const Tensor & v, int heads) {
    valid(q);
    valid(k);
    valid(v);
    require(heads > 0 && q.c == k.c && q.c % heads == 0 && v.c % heads == 0 && k.spatial() == v.spatial(),
            "attention dimensions differ");
    const int dk = q.c / heads, dv = v.c / heads;
    require(dk % 8 == 0 && dv % 8 == 0, "unsupported attention head dimension");
    Graph graph;
    ggml_tensor * qt = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, dk, int64_t(q.spatial()), heads, 1);
    ggml_tensor * kt = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, dk, int64_t(k.spatial()), heads, 1);
    ggml_tensor * vt = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, dv, int64_t(v.spatial()), heads, 1);
    ggml_set_input(qt);
    ggml_set_input(kt);
    ggml_set_input(vt);
    ggml_tensor * yt = ggml_flash_attn_ext(graph.ctx, qt, kt, vt, nullptr, 1.0f / std::sqrt(float(dk)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(yt, GGML_PREC_F32);
    graph.allocate(yt);
    auto upload = [&](ggml_tensor * dest, const Tensor & source, int dim) {
        std::vector<float> rows(source.v.size());
        for (int head = 0; head < heads; ++head)
            for (size_t p = 0; p < source.spatial(); ++p)
                for (int d = 0; d < dim; ++d)
                    rows[(size_t(head) * source.spatial() + p) * dim + d] =
                        source.v[size_t(head * dim + d) * source.spatial() + p];
        ggml_backend_tensor_set(dest, rows.data(), 0, rows.size() * sizeof(float));
    };
    upload(qt, q, dk);
    upload(kt, k, dk);
    upload(vt, v, dv);
    impl_->compute(graph);
    Tensor out(v.c, q.h, q.w);
    std::vector<float> rows(out.v.size());
    ggml_backend_tensor_get(yt, rows.data(), 0, rows.size() * sizeof(float));
    // FLASH_ATTN_EXT output [DV,heads,queries,batch].
    for (size_t p = 0; p < q.spatial(); ++p)
        for (int head = 0; head < heads; ++head)
            for (int d = 0; d < dv; ++d)
                out.v[size_t(head * dv + d) * q.spatial() + p] = rows[(p * heads + head) * dv + d];
    return out;
}

void Engine::mark(const std::string & name, const Tensor & x) const {
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    if (impl_->observer) impl_->observer(impl_->opaque, name, x);
#else
    (void)name;
    (void)x;
#endif
}
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
void Engine::set_observer(Observer callback, void * opaque) {
    impl_->observer = callback;
    impl_->opaque = opaque;
}
#endif

Tensor add(const Tensor & a, const Tensor & b) {
    valid(a);
    valid(b);
    require(a.c == b.c && a.h == b.h && a.w == b.w, "residual shape mismatch");
    Tensor result(a.c, a.h, a.w);
    for (size_t i = 0; i < a.v.size(); ++i) result.v[i] = a.v[i] + b.v[i];
    return result;
}
Tensor concat(const Tensor & a, const Tensor & b) {
    valid(a);
    valid(b);
    require(a.h == b.h && a.w == b.w, "concatenation spatial shape mismatch");
    Tensor result(a.c + b.c, a.h, a.w);
    std::copy(a.v.begin(), a.v.end(), result.v.begin());
    std::copy(b.v.begin(), b.v.end(), result.v.begin() + a.v.size());
    return result;
}
Tensor silu(Tensor x) {
    for (float & value : x.v) value /= 1.0f + std::exp(-value);
    return x;
}
Tensor gelu(Tensor x) {
    // Diffusers GEGLU defaults to the exact erf formulation, not the existing
    // core_cpu::gelu tanh approximation used by some language backends.
    for (float & value : x.v) value = 0.5f * value * (1.0f + std::erf(value * 0.7071067811865475244f));
    return x;
}
Tensor resize_nearest(const Tensor & x, int height, int width) {
    valid(x);
    Tensor out(x.c, height, width);
    for (int ch = 0; ch < x.c; ++ch)
        for (int y = 0; y < height; ++y)
            for (int z = 0; z < width; ++z)
                out.at(ch, y, z) = x.at(ch, int(int64_t(y) * x.h / height), int(int64_t(z) * x.w / width));
    return out;
}
Tensor pad(const Tensor & x, int left, int right, int top, int bottom) {
    valid(x);
    require(left >= 0 && right >= 0 && top >= 0 && bottom >= 0, "negative padding");
    Tensor out(x.c, x.h + top + bottom, x.w + left + right);
    for (int ch = 0; ch < x.c; ++ch)
        for (int y = 0; y < x.h; ++y)
            std::copy_n(x.v.data() + (size_t(ch) * x.h + y) * x.w, x.w,
                        out.v.data() + (size_t(ch) * out.h + y + top) * out.w + left);
    return out;
}
bool finite(const Tensor & x) {
    return x.c > 0 && x.c <= 16384 && x.h > 0 && x.h <= 2048 && x.w > 0 && x.w <= 2048 &&
           size_t(x.c) <= max_tensor_elements / size_t(x.h) / size_t(x.w) && x.v.size() == size_t(x.c) * x.h * x.w &&
           std::all_of(x.v.begin(), x.v.end(), [](float value) { return std::isfinite(value); });
}
} // namespace latent_inpaint
