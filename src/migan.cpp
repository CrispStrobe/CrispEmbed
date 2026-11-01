// Fixed-resolution graph follows original MI-GAN inference source at
// 2b793c5ece43f4253e32d4afc257120a5deed6f5 (MIT source and MIT weights).
// Graph decomposition also studied/adapted from Acly/vision.cpp at
// 26a752912d49f6c4ff4545b35a1bdf7400d349ed, Copyright (c) 2025 The vision.cpp
// authors, MIT; full notice: docs/third-party/vision-cpp-LICENSE.txt.
// No SH-GAN helpers or ONNX pipeline code is copied.

#include "migan.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"

#ifdef CRISPEMBED_MIGAN_TESTING
#include "../tests/migan_test_observer.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace migan {

namespace graph_impl {
struct graph_builder;
}

struct context {
    ggml_backend_t backend = nullptr;
    // GPU execution, the non-F32 pointwise formulation, the GPU pad rewrite and
    // batching are additions. The reviewed CPU/F32 batch1 path keeps its
    // original per-call graph (graph_builder with default options).
    bool gpu = false;
    bool f32_weights = true;
    std::string precision = "f32";
    std::string device_name = "CPU";
    int batch_limit = 1;
    core_gguf::WeightLoad weights;
    // Persistent graphs (allocated once, recomputed per call) keyed by batch.
    std::map<size_t, std::unique_ptr<graph_impl::graph_builder>> graphs;
    ~context();
};

namespace {

using shape = std::array<int64_t, 4>;
using schema = std::map<std::string, shape>;
constexpr size_t graph_capacity = 4096;
constexpr const char * source_commit = "2b793c5ece43f4253e32d4afc257120a5deed6f5";

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool gate(std::string & error) {
    return core_env::on("CRISPEMBED_MIGAN") || fail(error, "MI-GAN requires CRISPEMBED_MIGAN=1");
}

bool digest_valid(const char * value) {
    if (!value || std::strlen(value) != 64) return false;
    for (size_t i = 0; i < 64; ++i)
        if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f'))) return false;
    return true;
}

schema expected_schema() {
    schema shapes;
    auto conv = [&](const std::string & name, int ic, int oc, int kernel, bool dw, bool bias) {
        shapes[name + ".weight"] = { kernel, kernel, dw ? 1 : ic, oc };
        if (bias) shapes[name + ".bias"] = { oc, 1, 1, 1 };
    };
    auto up = [&](const std::string & name, int channels, int res) {
        shapes[name + ".filter_const"] = { res, res, 1, 1 };
        conv(name + ".filter", channels, channels, 4, true, false);
    };
    auto sep = [&](const std::string & name, int ic, int oc, int res, bool down, bool upwards, bool noise) {
        conv(name + ".conv1", ic, ic, 3, true, true);
        conv(name + ".conv2", ic, oc, 1, false, false);
        if (down) conv(name + ".downsample.filter", ic, ic, 4, true, false);
        if (upwards) up(name + ".upsample", oc, res);
        if (noise) {
            shapes[name + ".noise_const"] = { res, res, 1, 1 };
            shapes[name + ".noise_strength"] = { 1, 1, 1, 1 };
        }
    };
    auto channels = [](int res) { return std::min(32768 / res, 512); };
    for (int res = 256; res >= 4; res /= 2) {
        const std::string name = "encoder.b" + std::to_string(res);
        int ic = channels(res), oc = channels(std::max(res / 2, 4));
        if (res == 256) conv(name + ".fromrgb", 4, ic, 1, false, true);
        sep(name + ".conv1", ic, ic, res, false, false, false);
        sep(name + ".conv2", ic, oc, res, res > 4, false, false);
    }
    for (int res = 4; res <= 256; res *= 2) {
        const std::string name = "synthesis.b" + std::to_string(res);
        int ic = channels(std::max(res / 2, 4)), oc = channels(res);
        sep(name + ".conv1", ic, oc, res, false, res > 4, res > 4);
        sep(name + ".conv2", oc, oc, res, false, false, res > 4);
        conv(name + ".torgb", oc, 3, 1, false, true);
        if (res > 4) up(name + ".upsample", 3, res);
    }
    return shapes;
}

bool pointwise_name(const std::string & name) {
    const std::string suffix = ".conv2.weight";
    return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The reviewed F32 container is unchanged. A derived container (gated) may
// store only *.conv2.weight pointwise kernels as F16/Q8_0/Q4_K, flattened to
// [IC,OC] so the quantized row runs along the contracted input channels.
bool metadata_valid(gguf_context * meta, const char * expected_digest, const char * expected_source_gguf,
                    bool & f32_weights, std::string & precision, std::string & error) {
    if (!digest_valid(expected_digest)) return fail(error, "a reviewed lowercase checkpoint SHA-256 is required");
    const uint32_t file_type = core_gguf::kv_u32(meta, "general.file_type", UINT32_MAX);
    f32_weights = file_type == 0;
    if (core_gguf::kv_str(meta, "general.architecture", "") != "migan" ||
        core_gguf::kv_str(meta, "general.license", "") != "MIT" || file_type == UINT32_MAX ||
        core_gguf::kv_u32(meta, "migan.image_size", 0) != 256 ||
        core_gguf::kv_str(meta, "migan.source.commit", "") != source_commit ||
        core_gguf::kv_str(meta, "migan.checkpoint.sha256", "") != expected_digest ||
        core_gguf::kv_str(meta, "migan.tensor.layout", "") !=
            "original contiguous OIHW; GGUF dimensions reversed; scalar=[1]" ||
        core_gguf::kv_str(meta, "migan.input.contract", "") != "RGB8;binary-mask8;255=fill;fixed256")
        return fail(error, "MI-GAN GGUF source/checkpoint/F32/fixed-256 metadata differs");
    if (!f32_weights) {
        if (!core_env::on("CRISPEMBED_MIGAN_QUANT"))
            return fail(error, "non-F32 MI-GAN GGUF requires CRISPEMBED_MIGAN_QUANT=1");
        if (!digest_valid(expected_source_gguf) ||
            core_gguf::kv_str(meta, "migan.quantization.policy", "") != quant_policy ||
            core_gguf::kv_str(meta, "migan.quantization.source_gguf_sha256", "") != expected_source_gguf)
            return fail(error, "MI-GAN derived GGUF policy/source F32 identity differs");
    }
    const schema expected = expected_schema();
    if (expected.size() != 154 || gguf_get_n_tensors(meta) != 154)
        return fail(error, "MI-GAN requires exactly 154 original state tensors");
    std::map<ggml_type, size_t> pointwise_types;
    for (const auto & entry : expected) {
        const int64_t id = gguf_find_tensor(meta, entry.first.c_str());
        if (id < 0) return fail(error, "MI-GAN missing state: " + entry.first);
        const ggml_type type = gguf_get_tensor_type(meta, id);
        const int64_t * ne = gguf_get_tensor_ne(meta, id);
        if (type == GGML_TYPE_F32) {
            for (int axis = 0; axis < 4; ++axis)
                if (ne[axis] != entry.second[axis]) return fail(error, "MI-GAN state shape differs: " + entry.first);
            if (pointwise_name(entry.first)) ++pointwise_types[type];
            continue;
        }
        const bool allowed = type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K;
        if (f32_weights || !allowed || !pointwise_name(entry.first) || entry.second[0] != 1 || entry.second[1] != 1)
            return fail(error, "MI-GAN missing or non-F32 state: " + entry.first);
        if (ne[0] != entry.second[2] || ne[1] != entry.second[3] || ne[2] != 1 || ne[3] != 1 ||
            ne[0] % ggml_blck_size(type) != 0)
            return fail(error, "MI-GAN derived pointwise shape differs: " + entry.first);
        ++pointwise_types[type];
    }
    precision = "f32";
    if (!f32_weights) {
        size_t best = 0;
        for (const auto & entry : pointwise_types)
            if (entry.first != GGML_TYPE_F32 && entry.second > best) {
                best = entry.second;
                precision = ggml_type_name(entry.first);
            }
        if (best == 0) return fail(error, "MI-GAN derived GGUF changes no pointwise tensor");
    }
    return true;
}

} // namespace

namespace graph_impl {


struct options {
    size_t batch = 1;
    bool native_pad = true;       // ggml_pad_ext(2,1,2,1); GPU backends may lack left padding.
    bool im2col_pointwise = true; // Reviewed F32 im2col+mul_mat; non-F32 weights use the swapped form.
    bool f32_matmul = true;       // GGML_PREC_F32 hint on every pointwise mul_mat.
};

struct graph_builder {
    context & model;
    options opt;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
#ifdef CRISPEMBED_MIGAN_TESTING
    bool retain = false;
    std::vector<std::pair<std::string, ggml_tensor *>> stages;
#endif

    explicit graph_builder(context & model, options opt = {}) : model(model), opt(opt) {
        const size_t overhead =
            graph_capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false);
        ctx = ggml_init({ overhead, nullptr, true });
        if (!ctx) throw std::runtime_error("MI-GAN graph metadata allocation failed");
    }
    ~graph_builder() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }

    ggml_tensor * weight(const std::string & name) { return model.weights.tensors.at(name); }

    void mark(const std::string & name, ggml_tensor * tensor) {
        if (!tensor) return;
#ifdef CRISPEMBED_MIGAN_TESTING
        if (retain) {
            // Clamp/reshape stages are views. OUTPUT on the view alone does
            // not protect its allocation owner when gallocr expires the last
            // view. Retain every owner before allocation, not just the alias.
            for (ggml_tensor * storage = tensor; storage; storage = storage->view_src) ggml_set_output(storage);
            stages.emplace_back(name, tensor);
        }
#else
        (void)name;
#endif
    }

    ggml_tensor * activate(ggml_tensor * x) {
        // Do not mutate a prior Conv2d output retained by the test observer.
        x = ggml_leaky_relu(ctx, x, 0.2f, false);
        x = ggml_scale(ctx, x, 1.4142135623730950488f);
        return ggml_clamp(ctx, x, -256.0f, 256.0f);
    }

    // 1x1 convolution with weights as the quantizable src0: [C,W*H*N] channel-
    // innermost activations, one mul_mat, then back to WHCN. Required for
    // F16/Q8_0/Q4_K kernels stored as [IC,OC]; optional (A/B) for F32.
    ggml_tensor * pointwise_swapped(ggml_tensor * kernel, ggml_tensor * x) {
        const int64_t channels = x->ne[2], out = kernel->ne[ggml_n_dims(kernel) == 2 ? 1 : 3];
        ggml_tensor * w = ggml_reshape_2d(ctx, kernel, channels, out);
        ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C,W,H,N]
        xt = ggml_reshape_2d(ctx, xt, channels, x->ne[0] * x->ne[1] * x->ne[3]);
        ggml_tensor * product = ggml_mul_mat(ctx, w, xt); // [OC, W*H*N]
        if (opt.f32_matmul) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
        product = ggml_reshape_4d(ctx, product, out, x->ne[0], x->ne[1], x->ne[3]);
        return ggml_cont(ctx, ggml_permute(ctx, product, 2, 0, 1, 3)); // [W,H,OC,N]
    }

    ggml_tensor * convolution(const std::string & name, ggml_tensor * x, int stride, int pad, bool depthwise, bool bias,
                              bool record = true) {
        ggml_tensor * kernel = weight(name + ".weight");
        ggml_tensor * y;
        if (depthwise) {
            // Pinned CPU direct depthwise uses F32 source, F32 weight and F32
            // accumulation. WHCN stays contiguous; no F16 im2col temporary.
            y = ggml_conv_2d_dw_direct(ctx, kernel, x, stride, stride, pad, pad, 1, 1);
        } else if (kernel->type != GGML_TYPE_F32 || !opt.im2col_pointwise) {
            y = pointwise_swapped(kernel, x);
        } else {
            // All non-depthwise MI-GAN kernels are 1x1. Declare activation
            // type explicitly even though this pinned fork also chooses F32.
            ggml_tensor * patches = ggml_im2col(ctx, kernel, x, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F32);
            if (opt.batch == 1) {
                ggml_tensor * lhs = ggml_reshape_2d(ctx, patches, patches->ne[0], patches->ne[1] * patches->ne[2]);
                ggml_tensor * rhs =
                    ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]);
                ggml_tensor * product = ggml_mul_mat(ctx, lhs, rhs);
                if (opt.f32_matmul) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
                y = ggml_reshape_4d(ctx, product, patches->ne[1], patches->ne[2], kernel->ne[3], 1);
            } else {
                ggml_tensor * lhs =
                    ggml_reshape_2d(ctx, patches, patches->ne[0], patches->ne[1] * patches->ne[2] * patches->ne[3]);
                ggml_tensor * rhs =
                    ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]);
                ggml_tensor * product = ggml_mul_mat(ctx, lhs, rhs);
                if (opt.f32_matmul) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
                y = ggml_reshape_4d(ctx, product, patches->ne[1], patches->ne[2], patches->ne[3], kernel->ne[3]);
                y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 1, 3, 2));
            }
        }
        if (bias) {
            ggml_tensor * b = weight(name + ".bias");
            y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        }
        if (record) mark(name + ".output", y);
        return y;
    }

    ggml_tensor * downsample(const std::string & name, ggml_tensor * x) {
        mark(name + ".input.0", x);
        x = convolution(name + ".filter", x, 2, 1, true, false);
        mark(name + ".output", x);
        return x;
    }

    ggml_tensor * upsample(const std::string & name, ggml_tensor * x) {
        mark(name + ".input.0", x);
        x = ggml_interpolate(ctx, x, x->ne[0] * 2, x->ne[1] * 2, x->ne[2], x->ne[3], GGML_SCALE_MODE_NEAREST);
        x = ggml_mul(ctx, x, weight(name + ".filter_const"));
        if (opt.native_pad) {
            x = ggml_pad_ext(ctx, x, 2, 1, 2, 1, 0, 0, 0, 0);
            x = convolution(name + ".filter", x, 1, 0, true, false);
        } else {
            // Same zero-padded window: symmetric pad2 yields one extra trailing
            // row/column (2N+1); output index o still starts at input o-2, as
            // with pad_ext(2,1,2,1). Crop it. Used where PAD lacks left padding.
            const int64_t w = x->ne[0], h = x->ne[1];
            ggml_tensor * y = convolution(name + ".filter", x, 1, 2, true, false, false);
            y = ggml_view_4d(ctx, y, w, h, y->ne[2], y->ne[3], y->nb[1], y->nb[2], y->nb[3], 0);
            x = ggml_cont(ctx, y);
            mark(name + ".filter.output", x);
        }
        mark(name + ".output", x);
        return x;
    }

    ggml_tensor * separable(const std::string & name, ggml_tensor * x, bool down, bool up, bool noise) {
        mark(name + ".input.0", x);
        x = convolution(name + ".conv1", x, 1, 1, true, true);
        x = activate(x);
        if (down) x = downsample(name + ".downsample", x);
        x = convolution(name + ".conv2", x, 1, 0, false, false);
        if (up) x = upsample(name + ".upsample", x);
        if (noise) {
            ggml_tensor * n = ggml_mul(ctx, weight(name + ".noise_const"), weight(name + ".noise_strength"));
            x = ggml_add(ctx, x, n);
        }
        x = activate(x);
        mark(name + ".output", x);
        return x;
    }

    void build() {
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 256, 4, int64_t(opt.batch));
        ggml_set_input(input);
        mark("generator.input", input);
        std::map<int, ggml_tensor *> features;
        ggml_tensor * x = nullptr;
        for (int res = 256; res >= 4; res /= 2) {
            const std::string name = "encoder.b" + std::to_string(res);
            mark(name + ".input.0", x); // First original encoder argument is None.
            mark(name + ".input.1", input);
            if (res == 256) x = activate(convolution(name + ".fromrgb", input, 1, 0, false, true));
            ggml_tensor * feature = separable(name + ".conv1", x, false, false, false);
            x = separable(name + ".conv2", feature, res > 4, false, false);
            features[res] = feature;
            mark(name + ".output.0", x);
            mark(name + ".output.1", feature);
        }
        ggml_tensor * image = nullptr;
        for (int res = 4; res <= 256; res *= 2) {
            const std::string name = "synthesis.b" + std::to_string(res);
            mark(name + ".input.0", x);
            mark(name + ".input.1", features.at(res));
            mark(name + ".input.2", image);
            x = separable(name + ".conv1", x, false, res > 4, res > 4);
            x = ggml_add(ctx, x, features.at(res));
            x = separable(name + ".conv2", x, false, false, res > 4);
            if (image) image = upsample(name + ".upsample", image);
            ggml_tensor * rgb = convolution(name + ".torgb", x, 1, 0, false, true);
            image = image ? ggml_add(ctx, image, rgb) : rgb;
            mark(name + ".output.0", x);
            mark(name + ".output.1", image);
        }
        output = image;
        ggml_set_output(output);
        mark("generator.output", output);
        graph = ggml_new_graph_custom(ctx, graph_capacity, false);
        ggml_build_forward_expand(graph, output);
        // One backend executes the whole graph: a hard bound, not a scheduler
        // preference with silent per-op CPU fallback and host round-trips.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->type != GGML_TYPE_F32) throw std::runtime_error("MI-GAN graph contains a non-F32 activation");
            if (!ggml_backend_supports_op(model.backend, node))
                throw std::runtime_error(std::string("MI-GAN ") + (model.gpu ? "GPU" : "CPU") +
                                         " operation unsupported: " + ggml_op_desc(node));
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph))
            throw std::runtime_error("MI-GAN graph allocation failed");
    }

    bool compute(const float * data, float * raw, std::string & error) {
        ggml_backend_tensor_set(input, data, 0, opt.batch * input_elements * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS)
            return fail(error, std::string("MI-GAN ") + (model.gpu ? "GPU" : "CPU") + " graph computation failed");
        ggml_backend_tensor_get(output, raw, 0, opt.batch * rgb_elements * sizeof(float));
        for (size_t i = 0; i < opt.batch * rgb_elements; ++i)
            if (!std::isfinite(raw[i])) return fail(error, "MI-GAN nonfinite generator output");
        return true;
    }
};

} // namespace graph_impl

context::~context() {
    graphs.clear();
    core_gguf::free_weights(weights);
    if (backend) {
        if (gpu)
            crispasr_free_gpu_backend(backend);
        else
            ggml_backend_free(backend);
    }
}

namespace {

using graph_impl::graph_builder;

// Options for every graph of this context except the reviewed CPU/F32 batch1
// graph, which is always constructed with default options.
graph_impl::options context_options(const context & model, size_t batch) {
    graph_impl::options opt;
    opt.batch = batch;
    if (model.gpu) {
        opt.native_pad = !core_env::on("CRISPEMBED_MIGAN_GPU_PAD_REWRITE") && [&] {
            // Probe the actual op: ggml-metal rejects nonzero left padding.
            ggml_init_params params{ 4 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * probe = ggml_init(params);
            ggml_tensor * t = ggml_new_tensor_4d(probe, GGML_TYPE_F32, 8, 8, 2, 1);
            const bool ok = ggml_backend_supports_op(model.backend, ggml_pad_ext(probe, t, 2, 1, 2, 1, 0, 0, 0, 0));
            ggml_free(probe);
            return ok;
        }();
        opt.im2col_pointwise = !core_env::on("CRISPEMBED_MIGAN_GPU_SWAP_POINTWISE");
    }
    // Default keeps the F32 precision hint (Metal: F32 mul_mv_ext/_hp kernels).
    opt.f32_matmul = !core_env::on("CRISPEMBED_MIGAN_FAST_MATMUL");
    return opt;
}

bool reviewed_cpu_graph(const context & model, size_t batch) {
    return !model.gpu && model.f32_weights && batch == 1 && !core_env::on("CRISPEMBED_MIGAN_FAST_MATMUL");
}

bool input_valid(context * ctx, const float * input, size_t input_count, float * raw, size_t output_count,
                 std::string & error) {
    if (!gate(error)) return false;
    if (!ctx || !input || !raw || input_count != input_elements || output_count != rgb_elements)
        return fail(error, "MI-GAN requires fixed [1,4,256,256] input and [1,3,256,256] output");
    for (size_t i = 0; i < input_count; ++i)
        if (!std::isfinite(input[i])) return fail(error, "MI-GAN nonfinite input");
    return true;
}

bool rgb_mask_valid(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                    std::string & error) {
    if (!rgb || !mask || rgb_bytes != rgb_elements || mask_bytes != mask_elements)
        return fail(error, "MI-GAN requires RGB8 and same-size single-channel mask");
    for (size_t i = 0; i < mask_bytes; ++i)
        if (mask[i] != 0 && mask[i] != 255) return fail(error, "MI-GAN mask must be binary; 255=fill, 0=known");
    return true;
}

bool weights_finite(context & model, std::string & error) {
    std::vector<float> values;
    for (const auto & entry : model.weights.tensors) {
        ggml_tensor * t = entry.second;
        const int64_t n = ggml_nelements(t);
        values.resize(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
        } else {
            std::vector<uint8_t> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            const ggml_type_traits * traits = ggml_get_type_traits(t->type);
            if (!traits || !traits->to_float) return fail(error, "MI-GAN cannot dequantize " + entry.first);
            traits->to_float(raw.data(), values.data(), n);
        }
        for (float value : values)
            if (!std::isfinite(value)) return fail(error, "MI-GAN nonfinite checkpoint state: " + entry.first);
    }
    return true;
}

int batch_default(const context & model) {
    // Measured on M1 (docs/migan-metal.md): GPU batches amortize dispatch;
    // the CPU gains nothing from batching, so it keeps one tile per graph.
    int limit = model.gpu ? 4 : 1;
    if (const char * value = std::getenv("CRISPEMBED_MIGAN_BATCH")) {
        const int parsed = std::atoi(value);
        if (parsed >= 1 && parsed <= 16) limit = parsed;
    }
    return limit;
}

} // namespace

bool gpu_default() {
    return core_env::on("CRISPEMBED_MIGAN_GPU");
}

context * init(const char * path, const char * expected_checkpoint_sha256, int n_threads, std::string & error) {
    return init_ex(path, expected_checkpoint_sha256, n_threads, device_choice::automatic, nullptr, error);
}

context * init_ex(const char * path, const char * expected_checkpoint_sha256, int n_threads, device_choice device,
                  const char * expected_source_gguf_sha256, std::string & error) {
    error.clear();
    if (!gate(error)) return nullptr;
    if (!path || n_threads < 1 || n_threads > 64) {
        fail(error, "MI-GAN requires a model path and 1..64 CPU threads");
        return nullptr;
    }
    gguf_context * meta = core_gguf::open_metadata(path);
    if (!meta) {
        fail(error, "MI-GAN cannot open GGUF metadata");
        return nullptr;
    }
    bool f32_weights = true;
    std::string precision;
    const bool valid =
        metadata_valid(meta, expected_checkpoint_sha256, expected_source_gguf_sha256, f32_weights, precision, error);
    core_gguf::free_metadata(meta);
    if (!valid) return nullptr;
    try {
        std::unique_ptr<context> ctx(new context);
        ctx->f32_weights = f32_weights;
        ctx->precision = precision;
        const bool want_gpu = device == device_choice::gpu || (device == device_choice::automatic && gpu_default());
        if (want_gpu) {
            ggml_backend_t gpu = crispasr_init_gpu_backend();
            if (gpu && !core_cpu_backend::is_cpu(gpu)) {
                ctx->backend = gpu;
                ctx->gpu = true;
            } else if (gpu) {
                ggml_backend_free(gpu);
                std::fprintf(stderr, "MI-GAN: no GPU backend available; using the CPU backend\n");
            }
        }
        if (!ctx->backend) {
            ctx->backend = core_cpu_backend::init();
            if (!ctx->backend || !core_cpu_backend::is_cpu(ctx->backend)) {
                fail(error, "MI-GAN CPU backend unavailable");
                return nullptr;
            }
            core_cpu_backend::set_n_threads(ctx->backend, n_threads);
        }
        if (ctx->gpu) {
            ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend);
            ctx->device_name = dev ? ggml_backend_dev_name(dev) : ggml_backend_name(ctx->backend);
        }
        if (!core_gguf::load_weights(path, ctx->backend, "migan", ctx->weights)) {
            fail(error, "MI-GAN weight load failed");
            return nullptr;
        }
        if (!weights_finite(*ctx, error)) return nullptr;
        ctx->batch_limit = batch_default(*ctx);
        if (ctx->gpu) {
            // Build the batch1 graph now: an unsupported GPU op is reported at
            // init, and the context falls back to the reviewed CPU backend.
            try {
                std::unique_ptr<graph_builder> probe(new graph_builder(*ctx, context_options(*ctx, 1)));
                probe->build();
                ctx->graphs[1] = std::move(probe);
            } catch (const std::exception & exception) {
                std::fprintf(stderr, "MI-GAN: %s; reloading on the CPU backend\n", exception.what());
                return init_ex(path, expected_checkpoint_sha256, n_threads, device_choice::cpu,
                               expected_source_gguf_sha256, error);
            }
        }
        return ctx.release();
    } catch (const std::exception & exception) {
        fail(error, exception.what());
        return nullptr;
    }
}

void free(context * ctx) {
    delete ctx;
}

const char * backend_name(const context * ctx) {
    return ctx ? ctx->device_name.c_str() : "";
}

bool uses_gpu(const context * ctx) {
    return ctx && ctx->gpu;
}

const char * weight_precision(const context * ctx) {
    return ctx ? ctx->precision.c_str() : "";
}

int max_batch(const context * ctx) {
    return ctx ? ctx->batch_limit : 0;
}

namespace {
bool execute(context * ctx, const float * input_nchw, size_t batch, float * raw_nchw, std::string & error) {
    try {
        if (reviewed_cpu_graph(*ctx, batch)) {
            // Reviewed CPU/F32 path: unchanged per-call graph construction.
            graph_builder graph(*ctx);
            graph.build();
            return graph.compute(input_nchw, raw_nchw, error);
        }
        std::unique_ptr<graph_builder> & graph = ctx->graphs[batch];
        if (!graph) {
            std::unique_ptr<graph_builder> built(new graph_builder(*ctx, context_options(*ctx, batch)));
            built->build();
            graph = std::move(built);
        }
        return graph->compute(input_nchw, raw_nchw, error);
    } catch (const std::exception & exception) {
        ctx->graphs.erase(batch);
        return fail(error, exception.what());
    }
}
} // namespace

bool run(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
         std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, raw_nchw, output_count, error)) return false;
    return execute(ctx, input_nchw, 1, raw_nchw, error);
}

bool run_batch(context * ctx, const float * input_nchw, size_t batch, float * raw_nchw, std::string & error) {
    error.clear();
    if (!gate(error)) return false;
    if (!ctx || !input_nchw || !raw_nchw || batch < 1 || batch > size_t(ctx->batch_limit))
        return fail(error, "MI-GAN batch requires 1..max_batch contiguous fixed256 inputs");
    for (size_t i = 0; i < batch * input_elements; ++i)
        if (!std::isfinite(input_nchw[i])) return fail(error, "MI-GAN nonfinite input");
    return execute(ctx, input_nchw, batch, raw_nchw, error);
}

bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error) {
    error.clear();
    if (width != image_size || height != image_size || !input_nchw || input_count != input_elements)
        return fail(error, "MI-GAN preprocessing requires exactly 256x256 NCHW F32 output");
    if (!rgb_mask_valid(rgb, rgb_bytes, fill_mask, mask_bytes, error)) return false;
    for (size_t i = 0; i < mask_elements; ++i) {
        const float known = fill_mask[i] == 0 ? 1.0f : 0.0f;
        input_nchw[i] = known - 0.5f;
        for (size_t channel = 0; channel < 3; ++channel)
            input_nchw[(channel + 1) * mask_elements + i] =
                (float(rgb[3 * i + channel]) * 2.0f / 255.0f - 1.0f) * known;
    }
    return true;
}

bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb, size_t output_bytes,
                    std::string & error) {
    error.clear();
    if (!raw_nchw || raw_count != rgb_elements || !output_rgb || output_bytes != rgb_elements)
        return fail(error, "MI-GAN composition requires fixed RGB sizes");
    if (!rgb_mask_valid(rgb, rgb_bytes, fill_mask, mask_bytes, error)) return false;
    for (size_t i = 0; i < raw_count; ++i)
        if (!std::isfinite(raw_nchw[i])) return fail(error, "MI-GAN composition rejects nonfinite raw output");
    for (size_t i = 0; i < mask_elements; ++i)
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t hwc = 3 * i + channel;
            const float normalized = raw_nchw[channel * mask_elements + i] * 0.5f + 0.5f;
            const uint8_t generated = uint8_t(std::max(0.0f, std::min(1.0f, normalized)) * 255.0f);
            output_rgb[hwc] = fill_mask[i] ? generated : rgb[hwc];
        }
    return true;
}

#ifdef CRISPEMBED_MIGAN_TESTING
namespace test {
bool observer_retention_regression(std::string & error) {
    error.clear();
    try {
        auto execute = [&](bool use_actual_mark) {
            context model;
            model.backend = core_cpu_backend::init();
            if (!model.backend || !core_cpu_backend::is_cpu(model.backend))
                throw std::runtime_error("MI-GAN retention regression CPU unavailable");
            core_cpu_backend::set_n_threads(model.backend, 1);
            graph_builder graph(model);
            graph.retain = true;
            ggml_tensor * input = ggml_new_tensor_1d(graph.ctx, GGML_TYPE_F32, 4);
            ggml_set_input(input);
            ggml_set_output(input);
            ggml_tensor * owner = ggml_scale(graph.ctx, input, 2.0f);
            ggml_tensor * anchor = ggml_clamp(graph.ctx, ggml_reshape_2d(graph.ctx, owner, 2, 2), -256, 256);
            if (use_actual_mark)
                graph.mark("allocation-regression.anchor", anchor);
            else
                ggml_set_output(anchor); // Retained negative: the original view-only defect.
            ggml_tensor * consumer = ggml_scale(graph.ctx, anchor, 3.0f);
            // CONT cannot reuse its parent in-place. The old view owner is
            // expired here, making its same-sized allocation a competing slot.
            ggml_tensor * competing = ggml_cont(graph.ctx, consumer);
            ggml_tensor * output = ggml_scale(graph.ctx, competing, 5.0f);
            ggml_set_output(output);
            graph.graph = ggml_new_graph_custom(graph.ctx, graph_capacity, false);
            ggml_build_forward_expand(graph.graph, output);
            graph.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            if (!graph.allocator || !ggml_gallocr_alloc_graph(graph.allocator, graph.graph))
                throw std::runtime_error("MI-GAN retention regression allocation failed");
            const std::array<float, 4> values = { 1, -2, 3, -4 };
            const std::array<float, 4> expected_anchor = { 2, -4, 6, -8 };
            const std::array<float, 4> expected_output = { 30, -60, 90, -120 };
            std::array<float, 4> observed_anchor, observed_output;
            ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
            if (ggml_backend_graph_compute(model.backend, graph.graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("MI-GAN retention regression execution failed");
            ggml_backend_tensor_get(anchor, observed_anchor.data(), 0, sizeof(observed_anchor));
            ggml_backend_tensor_get(output, observed_output.data(), 0, sizeof(observed_output));
            if (observed_output != expected_output)
                throw std::runtime_error("MI-GAN retention regression arithmetic control failed");
            return std::make_pair(observed_anchor == expected_anchor, observed_anchor == expected_output);
        };
        const auto old_view_only = execute(false);
        const auto actual_mark = execute(true);
        if (old_view_only.first || !old_view_only.second || !actual_mark.first || actual_mark.second)
            return fail(error, "MI-GAN retention regression did not reproduce/correct view-owner reuse");
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

bool run_observed(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
                  const observer & observe, size_t & graph_buffer_bytes, std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, raw_nchw, output_count, error)) return false;
    try {
        graph_builder graph(*ctx, reviewed_cpu_graph(*ctx, 1) ? graph_impl::options{} : context_options(*ctx, 1));
        graph.retain = true;
        graph.build();
        graph_buffer_bytes = ggml_gallocr_get_buffer_size(graph.allocator, 0);
        if (!graph.compute(input_nchw, raw_nchw, error)) return false;
        std::vector<float> values;
        for (const auto & stage : graph.stages) {
            ggml_tensor * tensor = stage.second;
            if (!ggml_is_contiguous(tensor) || tensor->type != GGML_TYPE_F32)
                return fail(error, "MI-GAN observer requires contiguous F32 stage: " + stage.first);
            values.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            if (!observe(stage.first, values.data(), values.size(),
                         { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] }))
                return fail(error, "MI-GAN observer rejected stage: " + stage.first);
        }
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}
} // namespace test
#endif

} // namespace migan
