// Big-LaMa architecture follows Samsung LaMa source at
// 786f5936b27fb3dacd2b1ad799e4de968ea697e7, Apache-2.0.
// Source license: docs/third-party/lama-LICENSE.txt. Checkpoint rights are
// independent; no checkpoint or unverified weights license is distributed here.
#include "lama.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include "lama_fft.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lama {

struct graph_builder;
struct context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    // Non-resident stored tensors (F16/quantized, or row-flattened storage)
    // are dequantized at load into this separate buffer with their original
    // 4D shape; the stored bytes are never loaded for them.
    ggml_context * converted_ctx = nullptr;
    ggml_backend_buffer_t converted_buf = nullptr;
    // 1x1 kernels that run as ggml_mul_mat on their stored type
    // (CRISPEMBED_LAMA_QUANT_MATMUL=1 only).
    std::set<std::string> matmul;
    load_info info;
    std::unique_ptr<graph_builder> graph;
    ~context();
};

namespace {
using shape = std::array<int64_t, 4>;
using schema = std::map<std::string, shape>;
constexpr size_t graph_capacity = 8192;
constexpr const char * source_commit = "786f5936b27fb3dacd2b1ad799e4de968ea697e7";

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool gate(std::string & error) {
    return core_env::on("CRISPEMBED_LAMA") || fail(error, "Big-LaMa requires CRISPEMBED_LAMA=1");
}

bool digest_valid(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
                                             [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

schema expected_schema() {
    schema shapes;
    auto conv = [&](const std::string & name, int ic, int oc, int kernel, bool bias, bool transpose = false) {
        shapes[name + ".weight"] = { kernel, kernel, transpose ? oc : ic, transpose ? ic : oc };
        if (bias) shapes[name + ".bias"] = { oc, 1, 1, 1 };
    };
    auto bn = [&](const std::string & name, int channels) {
        for (const char * field : { "weight", "bias", "running_mean", "running_var" })
            shapes[name + "." + field] = { channels, 1, 1, 1 };
    };
    for (int i = 1; i <= 3; ++i) {
        const std::string name = "model." + std::to_string(i);
        const int ic = i == 1 ? 4 : (64 << (i - 2));
        const int oc = 64 << (i - 1);
        conv(name + ".ffc.convl2l", ic, oc, i == 1 ? 7 : 3, false);
        bn(name + ".bn_l", oc);
    }
    conv("model.4.ffc.convl2l", 256, 128, 3, false);
    conv("model.4.ffc.convl2g", 256, 384, 3, false);
    bn("model.4.bn_l", 128);
    bn("model.4.bn_g", 384);
    for (int i = 5; i <= 22; ++i) {
        for (const char * layer : { "conv1", "conv2" }) {
            const std::string name = "model." + std::to_string(i) + "." + layer;
            conv(name + ".ffc.convl2l", 128, 128, 3, false);
            conv(name + ".ffc.convl2g", 128, 384, 3, false);
            conv(name + ".ffc.convg2l", 384, 128, 3, false);
            const std::string spectral = name + ".ffc.convg2g";
            conv(spectral + ".conv1.0", 384, 192, 1, false);
            bn(spectral + ".conv1.1", 192);
            conv(spectral + ".fu.conv_layer", 384, 384, 1, false);
            bn(spectral + ".fu.bn", 384);
            conv(spectral + ".conv2", 192, 384, 1, false);
            bn(name + ".bn_l", 128);
            bn(name + ".bn_g", 384);
        }
    }
    for (int i = 0; i < 3; ++i) {
        const int index = 24 + 3 * i;
        const int ic = 512 >> i, oc = ic / 2;
        conv("model." + std::to_string(index), ic, oc, 3, true, true);
        bn("model." + std::to_string(index + 1), oc);
    }
    conv("model.34", 64, 3, 7, true);
    return shapes;
}

bool exact_u32(gguf_context * meta, const char * key, uint32_t expected) {
    const int64_t index = gguf_find_key(meta, key);
    return index >= 0 && gguf_get_kv_type(meta, index) == GGUF_TYPE_UINT32 && gguf_get_val_u32(meta, index) == expected;
}

// Storage roles. The original F32 profile (general.file_type 0) stays
// exactly as strict as before: every tensor F32 with its original 4D shape.
// A file written by crispembed-quantize's "lama-v1" policy may store conv
// kernels as F16 (original shape) or as F16/legacy/K quants with a 2D
// row-flattened storage shape of the same element count; the memory order is
// unchanged, so the loader restores the original 4D shape after dequantizing.
enum class role { vector, critical, transpose, pointwise, conv };

role tensor_role(const std::string & name, const shape & expected) {
    if (expected[1] == 1 && expected[2] == 1 && expected[3] == 1) return role::vector;
    if (name == "model.1.ffc.convl2l.weight" || name == "model.34.weight") return role::critical;
    if (name == "model.24.weight" || name == "model.27.weight" || name == "model.30.weight") return role::transpose;
    if (expected[0] == 1 && expected[1] == 1) return role::pointwise;
    return role::conv;
}

bool storage_type_allowed(ggml_type type) {
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

bool header_valid(gguf_context * meta, bool & quantized, std::string & error) {
    const int64_t file_type = gguf_find_key(meta, "general.file_type");
    const std::string policy = core_gguf::kv_str(meta, "lama.quantization.policy", "");
    quantized = !exact_u32(meta, "general.file_type", 0);
    const bool file_type_valid =
        file_type >= 0 && gguf_get_kv_type(meta, file_type) == GGUF_TYPE_UINT32 && (!quantized || policy == "lama-v1");
    if (core_gguf::kv_str(meta, "general.architecture", "") != "lama" || !file_type_valid ||
        !exact_u32(meta, "lama.schema.version", 1) || core_gguf::kv_str(meta, "lama.variant", "") != "big-lama" ||
        core_gguf::kv_str(meta, "lama.profile", "") != "big-lama-18-075-v1" ||
        core_gguf::kv_str(meta, "lama.source.commit", "") != source_commit ||
        core_gguf::kv_str(meta, "lama.source.license", "") != "Apache-2.0" ||
        core_gguf::kv_str(meta, "lama.fft.norm", "") != "ortho" ||
        core_gguf::kv_str(meta, "lama.input.contract", "") != "RGB8;binary-mask8;255=fill;symmetric-pad8" ||
        core_gguf::kv_str(meta, "lama.tensor.layout", "") !=
            "original contiguous OIHW/IOHW; GGUF dimensions reversed" ||
        !digest_valid(core_gguf::kv_str(meta, "lama.checkpoint.sha256", "")) ||
        !digest_valid(core_gguf::kv_str(meta, "lama.config.sha256", "")))
        return fail(error, "Big-LaMa GGUF architecture/source/profile/F32/input metadata differs");
    return true;
}

bool metadata_valid(gguf_context * meta, std::string & error) {
    bool quantized = false;
    if (!header_valid(meta, quantized, error)) return false;
    const schema expected = expected_schema();
    if (expected.size() != 837 || gguf_get_n_tensors(meta) != int64_t(expected.size()))
        return fail(error, "Big-LaMa requires exactly 837 original inference tensors (no training counters)");
    for (const auto & entry : expected) {
        const int64_t index = gguf_find_tensor(meta, entry.first.c_str());
        if (index < 0) return fail(error, "Big-LaMa missing tensor: " + entry.first);
        const ggml_type type = gguf_get_tensor_type(meta, index);
        const int64_t * ne = gguf_get_tensor_ne(meta, index);
        bool exact = true;
        for (int axis = 0; axis < 4; ++axis) exact &= ne[axis] == entry.second[axis];
        if (!quantized) {
            if (type != GGML_TYPE_F32) return fail(error, "Big-LaMa missing or non-F32 tensor: " + entry.first);
            if (!exact) return fail(error, "Big-LaMa tensor shape differs: " + entry.first);
            continue;
        }
        const role kind = tensor_role(entry.first, entry.second);
        if (kind == role::vector || kind == role::critical) {
            if (type != GGML_TYPE_F32 || !exact)
                return fail(error, "Big-LaMa quantized profile requires F32 original-shape tensor: " + entry.first);
            continue;
        }
        if (!storage_type_allowed(type))
            return fail(error, "Big-LaMa unsupported storage type for tensor: " + entry.first);
        const int64_t count = entry.second[0] * entry.second[1] * entry.second[2] * entry.second[3];
        const bool flattened = ne[2] == 1 && ne[3] == 1 && ne[0] > 0 && ne[1] > 0 && ne[0] * ne[1] == count;
        const bool shape_ok =
            (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) ? (exact || flattened) : (flattened && !exact);
        if (!shape_ok || ne[0] % ggml_blck_size(type) != 0)
            return fail(error, "Big-LaMa tensor storage shape differs: " + entry.first);
        if (kind == role::pointwise && flattened && ne[0] != entry.second[2])
            return fail(error, "Big-LaMa 1x1 kernel storage must keep input-channel rows: " + entry.first);
    }
    return true;
}

bool rgb_mask_valid(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width,
                    int height, std::string & error) {
    int pw, ph;
    if (!padded_dimensions(width, height, pw, ph, error)) return false;
    const size_t pixels = size_t(width) * height;
    if (!rgb || !mask || rgb_bytes != 3 * pixels || mask_bytes != pixels)
        return fail(error, "Big-LaMa requires RGB8 and a same-size single-channel mask");
    for (size_t i = 0; i < mask_bytes; ++i)
        if (mask[i] != 0 && mask[i] != 255) return fail(error, "Big-LaMa mask must be binary; 255=fill, 0=known");
    return true;
}

bool input_valid(context * ctx, const float * input, size_t input_count, int width, int height, float * output,
                 size_t output_count, std::string & error) {
    if (!gate(error)) return false;
    int pw, ph;
    if (!padded_dimensions(width, height, pw, ph, error)) return false;
    const size_t pixels = size_t(width) * height;
    if (!ctx || !input || !output || pw != width || ph != height || input_count != 4 * pixels ||
        output_count != 3 * pixels)
        return fail(error, "Big-LaMa raw graph requires padded NCHW [1,4,H,W] input and [1,3,H,W] output");
    for (size_t i = 0; i < input_count; ++i)
        if (!std::isfinite(input[i])) return fail(error, "Big-LaMa input contains nonfinite values");
    return true;
}
} // namespace

struct graph_builder {
    context & model;
    int width, height;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    lama_fft::execution_status fft_status;
#ifdef CRISPEMBED_LAMA_TESTING
    bool retain = false;
    std::vector<std::pair<std::string, ggml_tensor *>> stages;
#endif

    graph_builder(context & model, int width, int height) : model(model), width(width), height(height) {
        const size_t overhead =
            graph_capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false);
        ctx = ggml_init({ overhead, nullptr, true });
        if (!ctx) throw std::runtime_error("Big-LaMa graph metadata allocation failed");
    }
    ~graph_builder() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }

    ggml_tensor * weight(const std::string & name) { return model.weights.tensors.at(name); }

    void mark(const std::string & name, ggml_tensor * value) {
#ifdef CRISPEMBED_LAMA_TESTING
        if (retain && value) {
            for (ggml_tensor * owner = value; owner; owner = owner->view_src) ggml_set_output(owner);
            stages.emplace_back(name, value);
        }
#else
        (void)name;
        (void)value;
#endif
    }

    ggml_tensor * reflect(ggml_tensor * x, int pad) {
        if (pad == 0) return x;
        if (x->ne[0] <= pad || x->ne[1] <= pad)
            throw std::runtime_error("Big-LaMa reflection padding exceeds a feature dimension");
        // ggml's reflect excludes the edge, matching Conv2d padding_mode=reflect.
        // This is deliberately different from the input's symmetric padding.
        x = ggml_pad_reflect_1d(ctx, ggml_cont(ctx, x), pad, pad);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);
        return ggml_cont(ctx, ggml_transpose(ctx, x));
    }

    ggml_tensor * pointwise_matmul(const std::string & name, ggml_tensor * x) {
        // CRISPEMBED_LAMA_QUANT_MATMUL: a 1x1 stride-1 convolution is a matrix
        // product over channels, so the stored (F16/quantized) kernel can be
        // consumed directly. ggml [W,H,C] -> [C,W*H] rows, product, and back.
        ggml_tensor * w = weight(name + ".weight");
        if (ggml_n_dims(w) != 2) w = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]);
        const int64_t width = x->ne[0], height = x->ne[1], channels = x->ne[2];
        if (w->ne[0] != channels || x->ne[3] != 1)
            throw std::runtime_error("Big-LaMa 1x1 matmul kernel/input channel mismatch: " + name);
        ggml_tensor * rows =
            ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, x), width * height, channels)));
        ggml_tensor * y = ggml_mul_mat(ctx, w, rows);
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        return ggml_reshape_4d(ctx, y, width, height, w->ne[1], 1);
    }

    ggml_tensor * conv(const std::string & name, ggml_tensor * x, int stride = 1, int pad = 0, bool bias = false) {
        if (model.matmul.count(name + ".weight") && stride == 1 && pad == 0 && !bias) {
            ggml_tensor * y = pointwise_matmul(name, x);
            mark(name + ".output", y);
            return y;
        }
        x = reflect(x, pad);
        // The pinned CPU direct kernel batches its im2col patches in bounded
        // scratch. F32 weights use F32 patches and F32 dot products; a full
        // 7x7 im2col tensor would require gigabytes even for a modest frame.
        ggml_tensor * y = ggml_conv_2d_direct(ctx, weight(name + ".weight"), x, stride, stride, 0, 0, 1, 1);
        if (bias) {
            ggml_tensor * b = weight(name + ".bias");
            y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        }
        mark(name + ".output", y);
        return y;
    }

    ggml_tensor * batchnorm(const std::string & name, ggml_tensor * x) {
        // Original eval BatchNorm2d epsilon=1e-5, no running-stat updates.
        ggml_tensor * variance = ggml_scale_bias(ctx, weight(name + ".running_var"), 1.0f, 1e-5f);
        ggml_tensor * scale = ggml_div(ctx, weight(name + ".weight"), ggml_sqrt(ctx, variance));
        ggml_tensor * offset =
            ggml_sub(ctx, weight(name + ".bias"), ggml_mul(ctx, weight(name + ".running_mean"), scale));
        const int64_t channels = x->ne[2];
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_4d(ctx, scale, 1, 1, channels, 1)),
                     ggml_reshape_4d(ctx, offset, 1, 1, channels, 1));
        mark(name + ".output", x);
        return x;
    }

    ggml_tensor * fourier(const std::string & name, ggml_tensor * x) {
        mark(name + ".input", x);
        ggml_tensor * packed = lama_fft::rfft2(ctx, x, &fft_status);
        if (!packed) throw std::runtime_error("Big-LaMa Fourier input shape unsupported");
        mark(name + ".rfft", packed);
        packed = conv(name + ".conv_layer", packed);
        packed = batchnorm(name + ".bn", packed);
        packed = ggml_relu(ctx, packed);
        mark(name + ".relu.output", packed);
        ggml_tensor * result = lama_fft::irfft2(ctx, packed, x->ne[0], &fft_status);
        if (!result) throw std::runtime_error("Big-LaMa inverse Fourier input shape unsupported");
        mark(name + ".output", result);
        return result;
    }

    ggml_tensor * spectral(const std::string & name, ggml_tensor * x) {
        // Big-LaMa uses stride=1 and enable_lfu=False in every global branch.
        x = ggml_relu(ctx, batchnorm(name + ".conv1.1", conv(name + ".conv1.0", x)));
        mark(name + ".conv1.2.output", x);
        ggml_tensor * y = fourier(name + ".fu", x);
        y = conv(name + ".conv2", ggml_add(ctx, x, y));
        mark(name + ".output", y);
        return y;
    }

    struct pair {
        ggml_tensor * local;
        ggml_tensor * global;
    };

    pair ffc(const std::string & name, pair x, int stride, int pad, bool has_global_output) {
        ggml_tensor * local = conv(name + ".ffc.convl2l", x.local, stride, pad);
        if (x.global) local = ggml_add(ctx, local, conv(name + ".ffc.convg2l", x.global, stride, pad));
        local = ggml_relu(ctx, batchnorm(name + ".bn_l", local));
        ggml_tensor * global = nullptr;
        if (has_global_output) {
            global = conv(name + ".ffc.convl2g", x.local, stride, pad);
            if (x.global) global = ggml_add(ctx, global, spectral(name + ".ffc.convg2g", x.global));
            global = ggml_relu(ctx, batchnorm(name + ".bn_g", global));
        }
        mark(name + ".output.local", local);
        mark(name + ".output.global", global);
        return { local, global };
    }

    ggml_tensor * transpose_conv(const std::string & name, ggml_tensor * x) {
        // PyTorch ConvTranspose2d(k=3,s=2,p=1,output_padding=1): the p0
        // operation is 2*input+1; discard its top row and left column only.
        ggml_tensor * y = ggml_conv_transpose_2d_p0(ctx, weight(name + ".weight"), x, 2);
        y = ggml_cont(ctx, ggml_view_4d(ctx, y, 2 * x->ne[0], 2 * x->ne[1], y->ne[2], 1, y->nb[1], y->nb[2], y->nb[3],
                                        y->nb[0] + y->nb[1]));
        ggml_tensor * b = weight(name + ".bias");
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        mark(name + ".output", y);
        return y;
    }

    void build() {
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, 4, 1);
        ggml_set_input(input);
        mark("generator.input", input);
        pair x = ffc("model.1", { reflect(input, 3), nullptr }, 1, 0, false);
        x = ffc("model.2", x, 2, 1, false);
        x = ffc("model.3", x, 2, 1, false);
        x = ffc("model.4", x, 2, 1, true);
        for (int index = 5; index <= 22; ++index) {
            const std::string name = "model." + std::to_string(index);
            const pair residual = x;
            x = ffc(name + ".conv1", x, 1, 1, true);
            x = ffc(name + ".conv2", x, 1, 1, true);
            x.local = ggml_add(ctx, residual.local, x.local);
            x.global = ggml_add(ctx, residual.global, x.global);
            mark(name + ".output.local", x.local);
            mark(name + ".output.global", x.global);
        }
        ggml_tensor * y = ggml_concat(ctx, x.local, x.global, 2);
        mark("model.23.output", y);
        for (int index = 24; index <= 30; index += 3) {
            y = transpose_conv("model." + std::to_string(index), y);
            y = batchnorm("model." + std::to_string(index + 1), y);
            y = ggml_relu(ctx, y);
            mark("model." + std::to_string(index + 2) + ".output", y);
        }
        y = conv("model.34", y, 1, 3, true);
        output = ggml_sigmoid(ctx, y);
        mark("model.35.output", output);
        mark("generator.output", output);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, graph_capacity, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->type != GGML_TYPE_F32) throw std::runtime_error("Big-LaMa non-F32 graph activation");
            if (!ggml_backend_supports_op(model.backend, node))
                throw std::runtime_error("Big-LaMa CPU operation unsupported: " + std::string(ggml_op_name(node->op)));
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocator) throw std::runtime_error("Big-LaMa graph allocator unavailable");
        size_t required_bytes = 0;
        ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &required_bytes);
        if (required_bytes > max_graph_bytes)
            throw std::runtime_error("Big-LaMa graph exceeds the 3 GiB activation-buffer limit; use a smaller image");
        // reserve_n_size measures and records assignments but deliberately
        // leaves the backing buffer null. Explicit reserve is required before
        // alloc_graph on this pinned ggml; its topology check alone will not
        // notice that measurement did not allocate storage.
        if (!ggml_gallocr_reserve(allocator, graph) || !ggml_gallocr_alloc_graph(allocator, graph))
            throw std::runtime_error("Big-LaMa graph allocation failed; use a smaller image");
    }

    bool compute(const float * data, float * raw, std::string & error) {
        fft_status.reset();
        ggml_backend_tensor_set(input, data, 0, ggml_nbytes(input));
        if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS)
            return fail(error, "Big-LaMa CPU graph computation failed");
        if (!fft_status.ok())
            return fail(error, "Big-LaMa Fourier operation failed: " + std::string(fft_status.message()));
        std::vector<float> result(size_t(width) * height * 3);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        for (float value : result)
            if (!std::isfinite(value)) return fail(error, "Big-LaMa generator output is nonfinite");
        std::copy(result.begin(), result.end(), raw);
        return true;
    }
};

context::~context() {
    graph.reset();
    if (converted_buf) ggml_backend_buffer_free(converted_buf);
    if (converted_ctx) ggml_free(converted_ctx);
    core_gguf::free_weights(weights);
    if (backend) ggml_backend_free(backend);
}

namespace {
// Host F32 view of any loaded weight, whatever its stored type.
bool host_values(ggml_tensor * tensor, std::vector<float> & values) {
    values.resize(size_t(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) return false;
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    traits->to_float(raw.data(), values.data(), int64_t(values.size()));
    return true;
}

bool seek(FILE * file, size_t offset) {
#ifdef _WIN32
    return _fseeki64(file, int64_t(offset), SEEK_SET) == 0;
#else
    return fseeko(file, off_t(offset), SEEK_SET) == 0;
#endif
}

bool include_resident(const char * name, void * user) {
    return static_cast<const std::set<std::string> *>(user)->count(name) != 0;
}

struct conversion {
    std::string name;
    ggml_type target;
    ggml_tensor * tensor;
};

// Residency plan. Default: every tensor that is not F32 in its original
// shape is dequantized to F32 at load (storage-only quantization), so the
// graph is the unchanged F32 graph. Two default-off gates add real memory/
// speed paths: CRISPEMBED_LAMA_QUANT_MATMUL keeps 1x1 kernels in their stored
// type and runs them as ggml_mul_mat; CRISPEMBED_LAMA_F16_CONV holds the
// k>1 encoder/trunk conv kernels as F16 for the direct conv kernel (F16
// weights, F32 patches). Transposed convs and the first/last conv stay F32.
bool load_tensors(context & model, const char * path, gguf_context * meta, std::string & error) {
    const bool matmul_gate = core_env::on("CRISPEMBED_LAMA_QUANT_MATMUL");
    const bool f16_gate = core_env::on("CRISPEMBED_LAMA_F16_CONV");
    const schema expected = expected_schema();
    std::set<std::string> resident;
    std::vector<conversion> converted;
    for (const auto & entry : expected) {
        const int64_t index = gguf_find_tensor(meta, entry.first.c_str());
        const ggml_type type = gguf_get_tensor_type(meta, index);
        const int64_t * ne = gguf_get_tensor_ne(meta, index);
        bool exact = true;
        for (int axis = 0; axis < 4; ++axis) exact &= ne[axis] == entry.second[axis];
        const role kind = tensor_role(entry.first, entry.second);
        if (matmul_gate && kind == role::pointwise) {
            resident.insert(entry.first);
            model.matmul.insert(entry.first);
            continue;
        }
        const ggml_type target = f16_gate && kind == role::conv ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (type == target && exact)
            resident.insert(entry.first);
        else
            converted.push_back({ entry.first, target, nullptr });
    }
    if (!core_gguf::load_weights_filtered(path, model.backend, include_resident, &resident, "lama", model.weights))
        return fail(error, "Big-LaMa weight load failed");
    if (model.weights.tensors.size() != resident.size()) return fail(error, "Big-LaMa resident weight load incomplete");
    model.info.resident_tensors = int(resident.size());
    model.info.converted_tensors = int(converted.size());
    model.info.matmul_tensors = int(model.matmul.size());
    model.info.weight_bytes = model.weights.buf ? ggml_backend_buffer_get_size(model.weights.buf) : 0;
    if (converted.empty()) return true;
    model.converted_ctx = ggml_init({ converted.size() * ggml_tensor_overhead() + 1024, nullptr, true });
    if (!model.converted_ctx) return fail(error, "Big-LaMa conversion metadata allocation failed");
    for (auto & item : converted) {
        const shape & dims = expected.at(item.name);
        item.tensor = ggml_new_tensor_4d(model.converted_ctx, item.target, dims[0], dims[1], dims[2], dims[3]);
        ggml_set_name(item.tensor, item.name.c_str());
        if (item.target == GGML_TYPE_F16) ++model.info.f16_conv_tensors;
    }
    model.converted_buf = ggml_backend_alloc_ctx_tensors(model.converted_ctx, model.backend);
    if (!model.converted_buf) return fail(error, "Big-LaMa dequantized weight allocation failed");
    model.info.weight_bytes += ggml_backend_buffer_get_size(model.converted_buf);
    std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(path, "rb"), std::fclose);
    if (!file) return fail(error, "Big-LaMa cannot reopen GGUF for dequantization");
    const size_t data_offset = gguf_get_data_offset(meta);
    std::vector<uint8_t> raw;
    std::vector<float> values;
    std::vector<ggml_fp16_t> half;
    for (const auto & item : converted) {
        const int64_t index = gguf_find_tensor(meta, item.name.c_str());
        const ggml_type type = gguf_get_tensor_type(meta, index);
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        const int64_t count = ggml_nelements(item.tensor);
        raw.resize(gguf_get_tensor_size(meta, index));
        values.resize(size_t(count));
        if (raw.size() != ggml_row_size(type, count) ||
            !seek(file.get(), data_offset + gguf_get_tensor_offset(meta, index)) ||
            std::fread(raw.data(), 1, raw.size(), file.get()) != raw.size())
            return fail(error, "Big-LaMa cannot read stored tensor: " + item.name);
        if (type == GGML_TYPE_F32)
            std::memcpy(values.data(), raw.data(), raw.size());
        else if (traits && traits->to_float)
            traits->to_float(raw.data(), values.data(), count);
        else
            return fail(error, "Big-LaMa has no dequantizer for tensor: " + item.name);
        if (item.target == GGML_TYPE_F16) {
            half.resize(values.size());
            ggml_fp32_to_fp16_row(values.data(), half.data(), count);
            ggml_backend_tensor_set(item.tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(item.tensor, values.data(), 0, values.size() * sizeof(float));
        }
        model.weights.tensors[item.name] = item.tensor;
    }
    return true;
}
} // namespace

bool describe(const context * ctx, load_info & info) {
    if (!ctx) return false;
    info = ctx->info;
    return true;
}

context * init(const char * path, int n_threads, std::string & error) {
    error.clear();
    if (!gate(error)) return nullptr;
    if (!path || !*path || n_threads < 1 || n_threads > 64) {
        fail(error, "Big-LaMa requires a model path and 1..64 CPU threads");
        return nullptr;
    }
    try {
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path),
                                                                                core_gguf::free_metadata);
        if (!meta) {
            fail(error, "Big-LaMa cannot open GGUF metadata");
            return nullptr;
        }
        if (!metadata_valid(meta.get(), error)) return nullptr;
        std::unique_ptr<context> result(new context);
        result->backend = core_cpu_backend::init();
        if (!result->backend || !core_cpu_backend::is_cpu(result->backend)) {
            fail(error, "Big-LaMa CPU backend unavailable");
            return nullptr;
        }
        core_cpu_backend::set_n_threads(result->backend, n_threads);
        if (!load_tensors(*result, path, meta.get(), error)) return nullptr;
        meta.reset();
        std::vector<float> values;
        for (const auto & entry : result->weights.tensors) {
            if (!host_values(entry.second, values)) {
                fail(error, "Big-LaMa cannot read back tensor: " + entry.first);
                return nullptr;
            }
            const bool variance =
                entry.first.size() >= 11 && entry.first.compare(entry.first.size() - 11, 11, "running_var") == 0;
            for (float value : values)
                if (!std::isfinite(value) || (variance && value < 0)) {
                    fail(error, "Big-LaMa nonfinite state or negative running variance: " + entry.first);
                    return nullptr;
                }
        }
        return result.release();
    } catch (const std::exception & exception) {
        fail(error, exception.what());
        return nullptr;
    }
}

void free(context * ctx) {
    delete ctx;
}

bool padded_dimensions(int width, int height, int & padded_width, int & padded_height, std::string & error) {
    error.clear();
    if (width < min_dimension || height < min_dimension || width > max_dimension || height > max_dimension)
        return fail(error, "Big-LaMa dimensions must each be 9..2048 pixels (bottleneck reflection needs at least 2)");
    const int pw = (width + 7) / 8 * 8, ph = (height + 7) / 8 * 8;
    if (size_t(pw) * ph > max_padded_pixels) return fail(error, "Big-LaMa padded image exceeds 2,097,152 pixels");
    padded_width = pw;
    padded_height = ph;
    return true;
}

bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error) {
    error.clear();
    int pw, ph;
    if (!padded_dimensions(width, height, pw, ph, error) ||
        !rgb_mask_valid(rgb, rgb_bytes, fill_mask, mask_bytes, width, height, error))
        return false;
    const size_t padded_pixels = size_t(pw) * ph;
    if (!input_nchw || input_count != 4 * padded_pixels)
        return fail(error, "Big-LaMa preprocessing output must be padded NCHW [1,4,H,W]");
    for (int y = 0; y < ph; ++y) {
        // At most seven added pixels, less than the supported minimum size;
        // symmetric padding includes the edge and needs only one reflection.
        const int sy = y < height ? y : 2 * height - 1 - y;
        for (int x = 0; x < pw; ++x) {
            const int sx = x < width ? x : 2 * width - 1 - x;
            const size_t source = size_t(sy) * width + sx, dest = size_t(y) * pw + x;
            const float fill = fill_mask[source] == 255 ? 1.0f : 0.0f;
            for (size_t channel = 0; channel < 3; ++channel)
                input_nchw[channel * padded_pixels + dest] =
                    (float(rgb[3 * source + channel]) / 255.0f) * (1.0f - fill);
            input_nchw[3 * padded_pixels + dest] = fill;
        }
    }
    return true;
}

bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, int width, int height, uint8_t * output_rgb,
                    size_t output_bytes, std::string & error) {
    error.clear();
    int pw, ph;
    if (!padded_dimensions(width, height, pw, ph, error) ||
        !rgb_mask_valid(rgb, rgb_bytes, fill_mask, mask_bytes, width, height, error))
        return false;
    const size_t padded_pixels = size_t(pw) * ph;
    if (!raw_nchw || raw_count != 3 * padded_pixels || !output_rgb || output_bytes != rgb_bytes)
        return fail(error, "Big-LaMa composition requires padded NCHW RGB and original-size RGB8 output");
    for (size_t i = 0; i < raw_count; ++i)
        if (!std::isfinite(raw_nchw[i])) return fail(error, "Big-LaMa composition rejects nonfinite raw output");
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t source = size_t(y) * pw + x, dest = size_t(y) * width + x;
            for (size_t channel = 0; channel < 3; ++channel) {
                const float normalized = std::max(0.0f, std::min(1.0f, raw_nchw[channel * padded_pixels + source]));
                output_rgb[3 * dest + channel] =
                    fill_mask[dest] ? uint8_t(normalized * 255.0f) : rgb[3 * dest + channel];
            }
        }
    return true;
}

bool run(context * ctx, const float * input_nchw, size_t input_count, int width, int height, float * raw_nchw,
         size_t output_count, std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, width, height, raw_nchw, output_count, error)) return false;
    try {
        if (!ctx->graph || ctx->graph->width != width || ctx->graph->height != height) {
            ctx->graph.reset(); // Do not keep two large shape-specific buffers alive.
            std::unique_ptr<graph_builder> next(new graph_builder(*ctx, width, height));
            next->build();
            ctx->graph = std::move(next);
        }
        return ctx->graph->compute(input_nchw, raw_nchw, error);
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error) {
    error.clear();
    if (!gate(error)) return false;
    if (!ctx || !output_rgb || output_bytes != rgb_bytes)
        return fail(error, "Big-LaMa requires an initialized context and same-size output buffer");
    if (!rgb_mask_valid(rgb, rgb_bytes, fill_mask, mask_bytes, width, height, error)) return false;
    if (std::none_of(fill_mask, fill_mask + mask_bytes, [](uint8_t value) { return value != 0; })) {
        std::memmove(output_rgb, rgb, rgb_bytes);
        return true;
    }
    try {
        int pw, ph;
        if (!padded_dimensions(width, height, pw, ph, error)) return false;
        const size_t padded_pixels = size_t(pw) * ph;
        std::vector<float> input(4 * padded_pixels), raw(3 * padded_pixels);
        if (!preprocess_rgb8(rgb, rgb_bytes, fill_mask, mask_bytes, width, height, input.data(), input.size(), error) ||
            !run(ctx, input.data(), input.size(), pw, ph, raw.data(), raw.size(), error))
            return false;
        std::vector<uint8_t> result(rgb_bytes);
        if (!composite_rgb8(raw.data(), raw.size(), rgb, rgb_bytes, fill_mask, mask_bytes, width, height, result.data(),
                            result.size(), error))
            return false;
        std::memmove(output_rgb, result.data(), result.size());
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

#ifdef CRISPEMBED_LAMA_TESTING
namespace test {
bool run_observed(context * ctx, const float * input_nchw, size_t input_count, int width, int height, float * raw_nchw,
                  size_t output_count, const observer & observe, size_t & graph_buffer_bytes, std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, width, height, raw_nchw, output_count, error)) return false;
    try {
        ctx->graph.reset();
        graph_builder graph(*ctx, width, height);
        graph.retain = true;
        graph.build();
        graph_buffer_bytes = ggml_gallocr_get_buffer_size(graph.allocator, 0);
        if (!graph.compute(input_nchw, raw_nchw, error)) return false;
        std::vector<float> values;
        for (const auto & stage : graph.stages) {
            ggml_tensor * tensor = stage.second;
            if (!ggml_is_contiguous(tensor) || tensor->type != GGML_TYPE_F32)
                return fail(error, "Big-LaMa observer requires contiguous F32 stage: " + stage.first);
            values.resize(size_t(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            for (float value : values)
                if (!std::isfinite(value)) return fail(error, "Big-LaMa observed stage is nonfinite: " + stage.first);
            if (!observe(stage.first, values.data(), values.size(),
                         { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] }))
                return fail(error, "Big-LaMa observer rejected stage: " + stage.first);
        }
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}
} // namespace test
#endif

} // namespace lama
