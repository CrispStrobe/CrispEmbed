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
#include "inpaint_weights.h"
#include "ggml-alloc.h"
#include "inpaint_graph.h"

#ifdef CRISPEMBED_MIGAN_TESTING
#include "../tests/migan_test_observer.h"
#include <chrono>
#include <thread>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace migan {

namespace graph_impl {
struct graph_builder;
}

struct context {
    // Fixed256 (reviewed) or fixed512 (CRISPEMBED_MIGAN512).
    int resolution = image_size;
    bool persistent = false;
    bool fast = false;
    bool cpu_kernels = false;
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
    // GPU only: F32 pointwise kernels as [IC+64,OC] with zero rows, so a
    // contraction of K%128==0 is routed to ggml-metal's F32-tile mul_mm (_hp)
    // instead of the small-batch mul_mv_ext kernel. Zero padding is exact.
    std::map<const ggml_tensor *, ggml_tensor *> padded_k;
    ggml_context * padded_ctx = nullptr;
    ggml_backend_buffer_t padded_buf = nullptr;
    // CRISPEMBED_MIGAN_DEQUANT_WEIGHTS: F32 copies of non-F32 pointwise kernels
    // (weight-only quantization error; activations stay F32 in the matmul).
    std::map<const ggml_tensor *, ggml_tensor *> dequant;
    ggml_context * dequant_ctx = nullptr;
    ggml_backend_buffer_t dequant_buf = nullptr;
    // Issue71 persistent graph (CRISPEMBED_INPAINT_GRAPH, snapshotted at init)
    // and inpaint-v1 precision variants; independent of the gated GPU path above.
    std::unique_ptr<inpaint_graph::CpuPool> threadpool;
    // inpaint-v1 precision variants only (CRISPEMBED_INPAINT_QUANT=1): the
    // dequantized F32 copies, and whether 1x1 kernels stay stored for matmul.
    std::unique_ptr<inpaint_weights::Converted> converted;
    inpaint_weights::Stats variant_stats;
    bool quant_matmul = false;
    std::mutex mutex;
    execution_stats stats;
    std::unique_ptr<graph_impl::graph_builder> cached_graph;
#ifdef CRISPEMBED_MIGAN_TESTING
    execution_stats observed_stats;
    std::unique_ptr<graph_impl::graph_builder> observed_graph;
#endif
    ~context();
};

namespace {

using shape = std::array<int64_t, 4>;
using schema = std::map<std::string, shape>;
constexpr size_t graph_capacity = 4096;
constexpr const char * source_commit = "2b793c5ece43f4253e32d4afc257120a5deed6f5";
using clock_type = std::chrono::steady_clock;

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void profile(const execution_stats & stats, const char * kind) {
    if (!core_env::on("CRISPEMBED_INPAINT_PROFILE")) return;
    std::fprintf(stderr,
                 "migan graph %s: backend=%s resolution=%d persistent=%d precision=%s cpu_kernels=%d cpu_plan=%d "
                 "cpu_pool=%d nodes=%zu "
                 "graph_bytes=%zu weight_bytes=%zu builds=%" PRIu64 " allocations=%" PRIu64 " runs=%" PRIu64
                 " reuses=%" PRIu64 " weight_uploads=%" PRIu64 " weight_upload_bytes=%" PRIu64 " input_uploads=%" PRIu64
                 " input_upload_bytes=%" PRIu64 " output_downloads=%" PRIu64 " output_download_bytes=%" PRIu64
                 " build_ms=%.6f compute_ms=%.6f total_compute_ms=%.6f run_ms=%.6f\n",
                 kind, stats.backend.c_str(), stats.resolution, int(stats.persistent),
                 stats.fast ? "backend-default-may-reduce-f32" : "f32", int(stats.cpu_kernels),
                 int(stats.cpu_plan_cached), int(stats.cpu_threadpool_reused), stats.graph_nodes,
                 stats.graph_buffer_bytes, stats.weight_buffer_bytes, stats.graph_builds, stats.graph_allocations,
                 stats.runs, stats.reuses, stats.weight_uploads, stats.weight_upload_bytes, stats.input_uploads,
                 stats.input_upload_bytes, stats.output_downloads, stats.output_download_bytes, stats.last_build_ms,
                 stats.last_compute_ms, stats.total_compute_ms, stats.last_run_ms);
}

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool resolution_valid(int resolution, std::string & error) {
    return resolution == 256 || resolution == 512 || fail(error, "MI-GAN supports only fixed256 or fixed512");
}

bool gate(int resolution, std::string & error) {
    if (!core_env::on("CRISPEMBED_MIGAN")) return fail(error, "MI-GAN requires CRISPEMBED_MIGAN=1");
    return resolution != 512 || core_env::on("CRISPEMBED_MIGAN512") ||
           fail(error, "MI-GAN512 additionally requires CRISPEMBED_MIGAN512=1");
}

bool digest_valid(const char * value) {
    if (!value || std::strlen(value) != 64) return false;
    for (size_t i = 0; i < 64; ++i)
        if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f'))) return false;
    return true;
}

// These CPU callbacks are explicit ggml graph nodes, selected only by the
// default-off CPU kernel gate. They preserve F32 weights/accumulation and the
// original kernel traversal order. No callback is added to a GPU graph.
struct depthwise_spec {
    int stride;
    int padding;
};

void cpu_activation(ggml_tensor * dst, const ggml_tensor * input, int ith, int nth, void *) {
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(dst));
    const int64_t count = ggml_nelements(input);
    const int64_t chunk = (count + nth - 1) / nth;
    const int64_t begin = chunk * ith, end = std::min(count, begin + chunk);
    const float * source = static_cast<const float *>(input->data);
    float * output = static_cast<float *>(dst->data);
    for (int64_t i = begin; i < end; ++i) {
        const float value = source[i];
        // Match ggml's two signed branches, including signed-zero behavior.
        const float activated = (value > 0.0f ? value : 0.0f) + 0.2f * (value < 0.0f ? value : 0.0f);
        const float scaled = activated * 1.4142135623730950488f;
        output[i] = std::min(std::max(scaled, -256.0f), 256.0f);
    }
}

void cpu_depthwise(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & spec = *static_cast<const depthwise_spec *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * kernel = dst->src[1];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && kernel->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(input) && ggml_is_contiguous(kernel) && ggml_is_contiguous(dst));
    const int64_t width = input->ne[0], height = input->ne[1], channels = input->ne[2];
    const int64_t out_width = dst->ne[0], out_height = dst->ne[1];
    const int64_t kw = kernel->ne[0], kh = kernel->ne[1];
    const int64_t rows = out_height * channels * input->ne[3];
    const int64_t chunk = (rows + nth - 1) / nth;
    const int64_t begin = chunk * ith, end = std::min(rows, begin + chunk);
    const float * source = static_cast<const float *>(input->data);
    const float * weights = static_cast<const float *>(kernel->data);
    float * output = static_cast<float *>(dst->data);
    constexpr int64_t block_size = 128;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t plane = row / out_height, y = row % out_height;
        const float * plane_source = source + plane * width * height;
        const float * filter = weights + (plane % channels) * kw * kh;
        float * out = output + row * out_width;
        const int64_t first = std::min(out_width, int64_t((spec.padding + spec.stride - 1) / spec.stride));
        const int64_t valid_end = width + spec.padding >= kw ? (width + spec.padding - kw) / spec.stride + 1 : 0;
        const int64_t last = std::max(first, std::min(out_width, valid_end));
        auto edge = [&](int64_t x) {
            float sum = 0.0f;
            for (int64_t ky = 0; ky < kh; ++ky) {
                const int64_t sy = y * spec.stride + ky - spec.padding;
                if (sy < 0 || sy >= height) continue;
                for (int64_t kx = 0; kx < kw; ++kx) {
                    const int64_t sx = x * spec.stride + kx - spec.padding;
                    if (sx >= 0 && sx < width) sum += filter[ky * kw + kx] * plane_source[sy * width + sx];
                }
            }
            out[x] = sum;
        };
        for (int64_t x = 0; x < first; ++x) edge(x);
        // Move boundary branches outside the pixel loop. Each SIMD lane still
        // accumulates ky/kx in the same order as the generic F32 depthwise op.
        for (int64_t x = first; x < last; x += block_size) {
            const int64_t count = std::min(block_size, last - x);
            alignas(64) float accum[block_size] = {};
            for (int64_t ky = 0; ky < kh; ++ky) {
                const int64_t sy = y * spec.stride + ky - spec.padding;
                if (sy < 0 || sy >= height) continue;
                for (int64_t kx = 0; kx < kw; ++kx) {
                    const float coefficient = filter[ky * kw + kx];
                    const float * pixels = plane_source + sy * width + x * spec.stride + kx - spec.padding;
                    if (spec.stride == 1) {
                        for (int64_t i = 0; i < count; ++i) accum[i] += coefficient * pixels[i];
                    } else {
                        for (int64_t i = 0; i < count; ++i) accum[i] += coefficient * pixels[2 * i];
                    }
                }
            }
            std::copy_n(accum, count, out + x);
        }
        for (int64_t x = last; x < out_width; ++x) edge(x);
    }
}

schema expected_schema(int resolution) {
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
    for (int res = resolution; res >= 4; res /= 2) {
        const std::string name = "encoder.b" + std::to_string(res);
        int ic = channels(res), oc = channels(std::max(res / 2, 4));
        if (res == resolution) conv(name + ".fromrgb", 4, ic, 1, false, true);
        sep(name + ".conv1", ic, ic, res, false, false, false);
        sep(name + ".conv2", ic, oc, res, res > 4, false, false);
    }
    for (int res = 4; res <= resolution; res *= 2) {
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
// inpaint-v1 policy files (CRISPEMBED_INPAINT_QUANT) are validated separately.
bool metadata_valid(gguf_context * meta, const char * expected_digest, int resolution,
                    const char * expected_source_gguf, bool & f32_weights, std::string & precision,
                    std::string & error) {
    if (!digest_valid(expected_digest)) return fail(error, "a reviewed lowercase checkpoint SHA-256 is required");
    const uint32_t file_type = core_gguf::kv_u32(meta, "general.file_type", UINT32_MAX);
    f32_weights = file_type == 0;
    if (core_gguf::kv_str(meta, "general.architecture", "") != "migan" ||
        core_gguf::kv_str(meta, "general.license", "") != "MIT" || file_type == UINT32_MAX ||
        core_gguf::kv_u32(meta, "migan.image_size", 0) != uint32_t(resolution) ||
        core_gguf::kv_str(meta, "migan.source.commit", "") != source_commit ||
        core_gguf::kv_str(meta, "migan.checkpoint.sha256", "") != expected_digest ||
        core_gguf::kv_str(meta, "migan.tensor.layout", "") !=
            "original contiguous OIHW; GGUF dimensions reversed; scalar=[1]" ||
        core_gguf::kv_str(meta, "migan.input.contract", "") !=
            "RGB8;binary-mask8;255=fill;fixed" + std::to_string(resolution))
        return fail(error,
                    "MI-GAN GGUF source/checkpoint/F32/fixed-" + std::to_string(resolution) + " metadata differs");
    const schema expected = expected_schema(resolution);
    const size_t count = resolution == 256 ? 154 : 177;
    if (expected.size() != count || gguf_get_n_tensors(meta) != int64_t(count))
        return fail(error, "MI-GAN requires exactly " + std::to_string(count) + " original state tensors");
    // Precision variants: every tensor is F32 original-shape or a natural-row
    // stored matrix (gated); the F32 profile keeps its exact original check.
    if (inpaint_weights::is_policy_file(meta)) {
        f32_weights = false;
        precision = inpaint_weights::policy_name;
        return inpaint_weights::validate(meta, expected, "MI-GAN", error);
    }
    if (!f32_weights) {
        if (!core_env::on("CRISPEMBED_MIGAN_QUANT"))
            return fail(error, "non-F32 MI-GAN GGUF requires CRISPEMBED_MIGAN_QUANT=1");
        if (!digest_valid(expected_source_gguf) ||
            core_gguf::kv_str(meta, "migan.quantization.policy", "") != quant_policy ||
            core_gguf::kv_str(meta, "migan.quantization.source_gguf_sha256", "") != expected_source_gguf)
            return fail(error, "MI-GAN derived GGUF policy/source F32 identity differs");
    }
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
    bool pad_k = false;           // GPU: padded-contraction F32 pointwise (context::padded_k).
    // A/B re-implementation of feat/issue71-five-engines-graphs 06d75fb8's
    // persistent graph forms (batch1): transpose-cont pointwise and pad+roll.
    bool transpose_pointwise = false;
    bool roll_pad = false;
};

struct graph_builder {
    context & model;
    options opt;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_graph_plan_t cpu_plan = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<float> result;
    std::vector<std::unique_ptr<depthwise_spec>> depthwise_specs;
    const clock_type::time_point build_started = clock_type::now();
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
        if (cpu_plan) ggml_backend_graph_plan_free(model.backend, cpu_plan);
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }

    ggml_tensor * weight(const std::string & name) { return model.weights.tensors.at(name); }

    execution_stats & statistics() {
#ifdef CRISPEMBED_MIGAN_TESTING
        if (retain) return model.observed_stats;
#endif
        return model.stats;
    }

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
        if (model.cpu_kernels) {
            if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x))
                throw std::runtime_error("MI-GAN CPU activation requires contiguous F32 input");
            return ggml_map_custom1(ctx, x, cpu_activation, GGML_N_TASKS_MAX, nullptr);
        }
        x = ggml_leaky_relu(ctx, x, 0.2f, false);
        x = ggml_scale(ctx, x, 1.4142135623730950488f);
        return ggml_clamp(ctx, x, -256.0f, 256.0f);
    }

    // 1x1 convolution with weights as the quantizable src0: [C,W*H*N] channel-
    // innermost activations, one mul_mat, then back to WHCN. Required for
    // F16/Q8_0/Q4_K kernels stored as [IC,OC]; optional (A/B) for F32.
    ggml_tensor * pointwise_swapped(ggml_tensor * kernel, ggml_tensor * x) {
        const int64_t channels = x->ne[2], out = kernel->ne[ggml_n_dims(kernel) == 2 ? 1 : 3];
        ggml_tensor * w = ggml_n_dims(kernel) == 2 ? kernel : ggml_reshape_2d(ctx, kernel, channels, out);
        ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C,W,H,N]
        xt = ggml_reshape_2d(ctx, xt, channels, x->ne[0] * x->ne[1] * x->ne[3]);
        ggml_tensor * product = ggml_mul_mat(ctx, w, xt); // [OC, W*H*N]
        if (opt.f32_matmul) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
        product = ggml_reshape_4d(ctx, product, out, x->ne[0], x->ne[1], x->ne[3]);
        return ggml_cont(ctx, ggml_permute(ctx, product, 2, 0, 1, 3)); // [W,H,OC,N]
    }

    // Padded-contraction F32 pointwise: [C+64,P] activations x [C+64,OC]
    // weights; the 64 appended input channels are zero on both sides.
    ggml_tensor * pointwise_padded(ggml_tensor * padded, ggml_tensor * x) {
        const int64_t channels = x->ne[2], out = padded->ne[1];
        ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // [C,W,H,N]
        xt = ggml_reshape_2d(ctx, xt, channels, x->ne[0] * x->ne[1] * x->ne[3]);
        xt = ggml_pad(ctx, xt, int(padded->ne[0] - channels), 0, 0, 0);
        ggml_tensor * product = ggml_mul_mat(ctx, padded, xt); // [OC, W*H*N]
        if (opt.f32_matmul) ggml_mul_mat_set_prec(product, GGML_PREC_F32);
        product = ggml_reshape_4d(ctx, product, out, x->ne[0], x->ne[1], x->ne[3]);
        return ggml_cont(ctx, ggml_permute(ctx, product, 2, 0, 1, 3)); // [W,H,OC,N]
    }

    ggml_tensor * convolution(const std::string & name, ggml_tensor * x, int stride, int pad, bool depthwise, bool bias,
                              bool record = true) {
        ggml_tensor * kernel = weight(name + ".weight");
        ggml_tensor * y;
        if (kernel->type != GGML_TYPE_F32 && model.converted) {
            // CRISPEMBED_INPAINT_QUANT_MATMUL: a stored [IC,OC] 1x1 kernel is
            // consumed directly. [W,H,C] -> [C,W*H] rows, product, and back.
            if (depthwise || stride != 1 || pad != 0 || ggml_n_dims(kernel) != 2 || kernel->ne[0] != x->ne[2])
                throw std::runtime_error("MI-GAN stored kernel is not a pointwise matmul: " + name);
            const int64_t width = x->ne[0], height = x->ne[1], channels = x->ne[2];
            ggml_tensor * rows =
                ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, x), width * height, channels)));
            ggml_tensor * product = ggml_mul_mat(ctx, kernel, rows);
            ggml_mul_mat_set_prec(product, GGML_PREC_F32);
            y = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_transpose(ctx, product)), width, height, kernel->ne[1], 1);
        } else if (depthwise) {
            // Pinned CPU direct depthwise uses F32 source, F32 weight and F32
            // accumulation. WHCN stays contiguous; no F16 im2col temporary.
            if (model.cpu_kernels) {
                if (x->type != GGML_TYPE_F32 || kernel->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
                    !ggml_is_contiguous(kernel) || kernel->ne[0] != kernel->ne[1] ||
                    (kernel->ne[0] != 3 && kernel->ne[0] != 4) || (stride != 1 && stride != 2) || pad < 0 || pad > 1 ||
                    kernel->ne[2] != 1 || kernel->ne[3] != x->ne[2])
                    throw std::runtime_error("MI-GAN CPU depthwise shape/type contract differs");
                const int64_t span_width = x->ne[0] + 2 * pad - kernel->ne[0];
                const int64_t span_height = x->ne[1] + 2 * pad - kernel->ne[1];
                if (span_width < 0 || span_height < 0) throw std::runtime_error("MI-GAN CPU depthwise output is empty");
                const int64_t width = span_width / stride + 1, height = span_height / stride + 1;
                depthwise_specs.emplace_back(new depthwise_spec{ stride, pad });
                ggml_tensor * arguments[] = { x, kernel };
                y = ggml_custom_4d(ctx, GGML_TYPE_F32, width, height, x->ne[2], x->ne[3], arguments, 2, cpu_depthwise,
                                   GGML_N_TASKS_MAX, depthwise_specs.back().get());
            } else {
                y = ggml_conv_2d_dw_direct(ctx, kernel, x, stride, stride, pad, pad, 1, 1);
            }
        } else if (kernel->type == GGML_TYPE_F32 &&
                   (model.persistent || (opt.transpose_pointwise && x->ne[3] == 1 && stride == 1 && pad == 0))) {
            // OIHW 1x1 weights are [IC,OC]. WHCN activations are [W*H,IC].
            // Transpose channels once; activation-first matmul then directly
            // produces contiguous [W*H,OC]. Weight-first matmul would require
            // another full output transpose. Both CUDA and Metal require a
            // contiguous contraction axis, so a strided transpose view alone
            // is insufficient. No patch extraction or F16 temporary is used.
            // CRISPEMBED_INPAINT_GRAPH always uses this form; the gated GPU path
            // selects it with CRISPEMBED_MIGAN_GPU_TRANSPOSE_POINTWISE.
            if (kernel->ne[0] != 1 || kernel->ne[1] != 1 || stride != 1 || pad != 0 || x->ne[3] != 1)
                throw std::runtime_error("MI-GAN pointwise graph shape differs");
            ggml_tensor * pixels = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
            ggml_tensor * lhs = ggml_cont(ctx, ggml_transpose(ctx, pixels));
            ggml_tensor * rhs = ggml_reshape_2d(ctx, kernel, kernel->ne[2], kernel->ne[3]);
            ggml_tensor * product = ggml_mul_mat(ctx, lhs, rhs);
            // Pinned Metal honors this hint through its full-F32 dot path;
            // omitting it can silently use lower-precision simdgroup products.
            ggml_mul_mat_set_prec(product, (model.fast || !opt.f32_matmul) ? GGML_PREC_DEFAULT : GGML_PREC_F32);
            y = ggml_reshape_4d(ctx, product, x->ne[0], x->ne[1], kernel->ne[3], 1);
        } else if (model.dequant.count(kernel)) {
            y = pointwise_swapped(model.dequant.at(kernel), x);
        } else if (opt.pad_k && kernel->type == GGML_TYPE_F32 && model.padded_k.count(kernel)) {
            y = pointwise_padded(model.padded_k.at(kernel), x);
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
        if (model.persistent || opt.roll_pad) {
            // Metal's PAD supports only right/bottom zero padding. Appending
            // three zeros then rolling by +2 on each spatial axis produces
            // precisely left/top=2, right/bottom=1, without host operations.
            x = ggml_roll(ctx, ggml_pad(ctx, x, 3, 3, 0, 0), 2, 2, 0, 0);
            x = convolution(name + ".filter", x, 1, 0, true, false);
        } else if (opt.native_pad) {
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
        const int resolution = model.resolution;
        input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, resolution, resolution, 4, int64_t(opt.batch));
        ggml_set_input(input);
        mark("generator.input", input);
        std::map<int, ggml_tensor *> features;
        ggml_tensor * x = nullptr;
        for (int res = resolution; res >= 4; res /= 2) {
            const std::string name = "encoder.b" + std::to_string(res);
            mark(name + ".input.0", x); // First original encoder argument is None.
            mark(name + ".input.1", input);
            if (res == resolution) x = activate(convolution(name + ".fromrgb", input, 1, 0, false, true));
            ggml_tensor * feature = separable(name + ".conv1", x, false, false, false);
            x = separable(name + ".conv2", feature, res > 4, false, false);
            features[res] = feature;
            mark(name + ".output.0", x);
            mark(name + ".output.1", feature);
        }
        ggml_tensor * image = nullptr;
        for (int res = 4; res <= resolution; res *= 2) {
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
        finish(image);
    }

    void finish(ggml_tensor * final) {
        output = final;
        ggml_set_output(output);
        mark("generator.output", output);
        graph = ggml_new_graph_custom(ctx, graph_capacity, false);
        ggml_build_forward_expand(graph, output);
        // One backend owns every leaf, node and allocation: a hard bound, not a
        // scheduler preference with silent per-op CPU fallback and host round-trips.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            // Views/reshapes of F16/Q8_0/Q4_K weights carry the weight type;
            // every computed activation must still be F32.
            const bool view = node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE ||
                              node->op == GGML_OP_TRANSPOSE;
            if (node->type != GGML_TYPE_F32 && !(view && !model.f32_weights))
                throw std::runtime_error("MI-GAN graph contains a non-F32 activation");
            if (!ggml_backend_supports_op(model.backend, node))
                throw std::runtime_error("MI-GAN " + model.device_name +
                                         " operation unsupported: " + ggml_op_desc(node));
        }
        inpaint_graph::validate_graph_backend(model.backend, graph);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph))
            throw std::runtime_error("MI-GAN graph allocation failed");
        if (model.persistent) result.resize(size_t(ggml_nelements(output)));
        execution_stats & stats = statistics();
        if (model.persistent && core_cpu_backend::is_cpu(model.backend)) {
            cpu_plan = ggml_backend_graph_plan_create(model.backend, graph);
            if (!cpu_plan) throw std::runtime_error("MI-GAN persistent CPU graph plan allocation failed");
            stats.cpu_plan_cached = true;
            stats.cpu_threadpool_reused = model.threadpool && model.threadpool->active();
        }
        stats.graph_nodes = size_t(ggml_graph_n_nodes(graph));
        stats.graph_buffer_bytes = ggml_gallocr_get_buffer_size(allocator, 0);
        ++stats.graph_builds;
        ++stats.graph_allocations;
        stats.last_build_ms = elapsed_ms(build_started);
    }

    bool compute(const float * data, float * raw, std::string & error) {
        const auto started = clock_type::now();
        execution_stats & stats = statistics();
        ggml_backend_tensor_set(input, data, 0, ggml_nbytes(input));
        ++stats.input_uploads;
        stats.input_upload_bytes += ggml_nbytes(input);
        const auto compute_started = clock_type::now();
        const auto status = cpu_plan ? ggml_backend_graph_plan_compute(model.backend, cpu_plan)
                                     : ggml_backend_graph_compute(model.backend, graph);
        if (status != GGML_STATUS_SUCCESS) return fail(error, "MI-GAN graph computation failed on " + stats.backend);
        stats.last_compute_ms = elapsed_ms(compute_started);
        stats.total_compute_ms += stats.last_compute_ms;
        float * destination = model.persistent ? result.data() : raw;
        ggml_backend_tensor_get(output, destination, 0, ggml_nbytes(output));
        ++stats.output_downloads;
        stats.output_download_bytes += ggml_nbytes(output);
        for (size_t i = 0; i < size_t(ggml_nelements(output)); ++i)
            if (!std::isfinite(destination[i])) return fail(error, "MI-GAN nonfinite generator output");
        if (model.persistent) {
            std::memcpy(raw, result.data(), ggml_nbytes(output));
            if (stats.runs) ++stats.reuses;
        }
        ++stats.runs;
        stats.last_run_ms = elapsed_ms(started);
        profile(stats,
#ifdef CRISPEMBED_MIGAN_TESTING
                retain ? "observed" :
#endif
                       "production");
        return true;
    }
};

} // namespace graph_impl

context::~context() {
    // Graphs contain views of immutable weights, so release them before the
    // weight buffer, and release all buffers before their owning backend.
#ifdef CRISPEMBED_MIGAN_TESTING
    observed_graph.reset();
#endif
    cached_graph.reset();
    graphs.clear();
    if (padded_buf) ggml_backend_buffer_free(padded_buf);
    if (dequant_buf) ggml_backend_buffer_free(dequant_buf);
    if (dequant_ctx) ggml_free(dequant_ctx);
    if (padded_ctx) ggml_free(padded_ctx);
    converted.reset();
    core_gguf::free_weights(weights);
    if (backend) {
        if (gpu)
            crispasr_free_gpu_backend(backend);
        else
            ggml_backend_free(backend);
    }
    threadpool.reset();
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
        opt.pad_k = core_env::on("CRISPEMBED_MIGAN_GPU_PAD_K") && !model.padded_k.empty();
        opt.transpose_pointwise = core_env::on("CRISPEMBED_MIGAN_GPU_TRANSPOSE_POINTWISE");
        opt.roll_pad = core_env::on("CRISPEMBED_MIGAN_GPU_ROLL_PAD");
    }
    // Default keeps the F32 precision hint (Metal: F32 mul_mv_ext/_hp kernels).
    opt.f32_matmul = !core_env::on("CRISPEMBED_MIGAN_FAST_MATMUL");
    return opt;
}

// inpaint-v1 precision variants keep their per-call eager graph as well.
bool reviewed_cpu_graph(const context & model, size_t batch) {
    return !model.gpu && (model.f32_weights || model.converted) && batch == 1 &&
           !core_env::on("CRISPEMBED_MIGAN_FAST_MATMUL");
}

bool input_valid(context * ctx, const float * input, size_t input_count, float * raw, size_t output_count,
                 std::string & error) {
    if (!gate(ctx ? ctx->resolution : 256, error)) return false;
    if (ctx && ctx->persistent && !core_env::on("CRISPEMBED_INPAINT_GRAPH"))
        return fail(error, "MI-GAN persistent graph requires CRISPEMBED_INPAINT_GRAPH=1");
    const size_t pixels = ctx ? size_t(ctx->resolution) * ctx->resolution : 0;
    if (!ctx || !input || !raw || input_count != 4 * pixels || output_count != 3 * pixels)
        return fail(error, "MI-GAN requires fixed [1,4,H,W] input and [1,3,H,W] output at the loaded resolution");
    for (size_t i = 0; i < input_count; ++i)
        if (!std::isfinite(input[i])) return fail(error, "MI-GAN nonfinite input");
    return true;
}

bool rgb_mask_valid(int resolution, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                    std::string & error) {
    const size_t pixels = size_t(resolution) * resolution;
    if (!rgb || !mask || rgb_bytes != 3 * pixels || mask_bytes != pixels)
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

// GPU: padded-contraction copies of every F32 1x1 kernel with IC%128==0.
bool make_padded_k(context & model, std::string & error) {
    std::vector<std::pair<ggml_tensor *, int64_t>> chosen;
    for (const auto & entry : model.weights.tensors) {
        ggml_tensor * t = entry.second;
        const std::string & name = entry.first;
        const bool weight = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
        if (!weight || t->type != GGML_TYPE_F32 || t->ne[0] != 1 || t->ne[1] != 1 || t->ne[2] % 128 != 0) continue;
        if (name.find(".filter.") != std::string::npos) continue; // depthwise up/down filters
        chosen.push_back({ t, t->ne[2] });
    }
    if (chosen.empty()) return true;
    model.padded_ctx = ggml_init({ chosen.size() * ggml_tensor_overhead(), nullptr, true });
    if (!model.padded_ctx) return fail(error, "MI-GAN padded weight context failed");
    for (auto & c : chosen)
        model.padded_k[c.first] = ggml_new_tensor_2d(model.padded_ctx, GGML_TYPE_F32, c.second + 64, c.first->ne[3]);
    model.padded_buf = ggml_backend_alloc_ctx_tensors(model.padded_ctx, model.backend);
    if (!model.padded_buf) return fail(error, "MI-GAN padded weight allocation failed");
    std::vector<float> src, dst;
    for (auto & c : chosen) {
        ggml_tensor * t = c.first;
        const int64_t ic = t->ne[2], oc = t->ne[3];
        src.resize(ic * oc);
        dst.assign((ic + 64) * oc, 0.0f);
        ggml_backend_tensor_get(t, src.data(), 0, src.size() * sizeof(float));
        for (int64_t o = 0; o < oc; ++o) std::memcpy(&dst[o * (ic + 64)], &src[o * ic], ic * sizeof(float));
        ggml_backend_tensor_set(model.padded_k.at(t), dst.data(), 0, dst.size() * sizeof(float));
    }
    return true;
}

bool make_dequant(context & model, std::string & error) {
    std::vector<ggml_tensor *> chosen;
    for (const auto & entry : model.weights.tensors)
        if (entry.second->type != GGML_TYPE_F32 && pointwise_name(entry.first)) chosen.push_back(entry.second);
    if (chosen.empty()) return true;
    model.dequant_ctx = ggml_init({ chosen.size() * ggml_tensor_overhead(), nullptr, true });
    if (!model.dequant_ctx) return fail(error, "MI-GAN dequant context failed");
    for (ggml_tensor * t : chosen)
        model.dequant[t] = ggml_new_tensor_2d(model.dequant_ctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
    model.dequant_buf = ggml_backend_alloc_ctx_tensors(model.dequant_ctx, model.backend);
    if (!model.dequant_buf) return fail(error, "MI-GAN dequant allocation failed");
    std::vector<uint8_t> raw;
    std::vector<float> values;
    for (ggml_tensor * t : chosen) {
        raw.resize(ggml_nbytes(t));
        values.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), values.data(), int64_t(values.size()));
        ggml_backend_tensor_set(model.dequant.at(t), values.data(), 0, values.size() * sizeof(float));
    }
    return true;
}

int batch_default(const context & model) {
    // Measured on M1 (docs/migan-metal.md): a 4-tile Metal batch was not
    // faster than one tile per graph on 1080p frames, so batching is opt-in.
    (void)model;
    int limit = 1;
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
    return init_variant_ex(path, expected_checkpoint_sha256, image_size, n_threads, device_choice::automatic, nullptr,
                           error);
}

context * init_ex(const char * path, const char * expected_checkpoint_sha256, int n_threads, device_choice device,
                  const char * expected_source_gguf_sha256, std::string & error) {
    return init_variant_ex(path, expected_checkpoint_sha256, image_size, n_threads, device, expected_source_gguf_sha256,
                           error);
}

context * init_variant(const char * path, const char * expected_checkpoint_sha256, int resolution, int n_threads,
                       std::string & error) {
    return init_variant_ex(path, expected_checkpoint_sha256, resolution, n_threads, device_choice::automatic, nullptr,
                           error);
}

context * init_variant_ex(const char * path, const char * expected_checkpoint_sha256, int resolution, int n_threads,
                          device_choice device, const char * expected_source_gguf_sha256, std::string & error) {
    error.clear();
    if (!resolution_valid(resolution, error) || !gate(resolution, error)) return nullptr;
    if (!path || n_threads < 1 || n_threads > 64) {
        fail(error, "MI-GAN requires a model path and 1..64 CPU threads");
        return nullptr;
    }
    gguf_context * meta = core_gguf::open_metadata(path);
    if (!meta) {
        fail(error, "MI-GAN cannot open GGUF metadata");
        return nullptr;
    }
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta_owner(meta, &core_gguf::free_metadata);
    bool f32_weights = true;
    std::string precision;
    if (!metadata_valid(meta, expected_checkpoint_sha256, resolution, expected_source_gguf_sha256, f32_weights,
                        precision, error))
        return nullptr;
    const bool precision_variant = inpaint_weights::is_policy_file(meta);
    try {
        std::unique_ptr<context> ctx(new context);
        ctx->f32_weights = f32_weights;
        ctx->precision = precision;
        ctx->resolution = resolution;
        ctx->persistent = core_env::on("CRISPEMBED_INPAINT_GRAPH");
        ctx->stats.persistent = ctx->persistent;
        ctx->stats.resolution = resolution;
        if (ctx->persistent) {
            // The persistent graph selects its backend from the process
            // preference; an unavailable explicit GPU request is an init error.
            ctx->backend = inpaint_graph::select_backend(n_threads, ctx->stats.backend, error);
            if (!ctx->backend) return nullptr;
            if (core_cpu_backend::is_cpu(ctx->backend)) {
                ctx->threadpool = std::make_unique<inpaint_graph::CpuPool>(ctx->backend, n_threads);
                ctx->cpu_kernels = core_env::on("CRISPEMBED_MIGAN_CPU_KERNELS");
            } else {
                if (core_env::on("CRISPEMBED_MIGAN_CPU_KERNELS"))
                    throw std::runtime_error("MI-GAN CPU kernels require a selected CPU backend");
                ctx->fast = core_env::on("CRISPEMBED_INPAINT_FAST");
            }
            ctx->stats.fast = ctx->fast;
            ctx->stats.cpu_kernels = ctx->cpu_kernels;
            ctx->device_name = ctx->stats.backend;
        } else {
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
        }
        if (!ctx->backend) {
            ctx->backend = core_cpu_backend::init();
            if (!ctx->backend || !core_cpu_backend::is_cpu(ctx->backend)) {
                fail(error, "MI-GAN CPU backend unavailable");
                return nullptr;
            }
            core_cpu_backend::set_n_threads(ctx->backend, n_threads);
            ctx->stats.backend = ggml_backend_name(ctx->backend);
        }
        if (ctx->gpu) {
            ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend);
            ctx->device_name = dev ? ggml_backend_dev_name(dev) : ggml_backend_name(ctx->backend);
            ctx->stats.backend = ctx->device_name;
        }
        if (precision_variant) {
            ctx->converted.reset(new inpaint_weights::Converted);
            ctx->quant_matmul = inpaint_weights::matmul_gate();
            if (!inpaint_weights::load(path, meta, ctx->backend, expected_schema(resolution), ctx->quant_matmul,
                                       "MI-GAN", ctx->weights, *ctx->converted, ctx->variant_stats, error))
                return nullptr;
        } else if (!core_gguf::load_weights(path, ctx->backend, "migan", ctx->weights)) {
            fail(error, "MI-GAN weight load failed");
            return nullptr;
        }
        meta_owner.reset();
        if (!weights_finite(*ctx, error)) return nullptr;
        for (const auto & entry : ctx->weights.tensors) {
            ++ctx->stats.weight_uploads;
            ctx->stats.weight_upload_bytes += ggml_nbytes(entry.second);
        }
        // The persistent graph is one fixed batch1 graph.
        ctx->batch_limit = ctx->persistent ? 1 : batch_default(*ctx);
        if (ctx->gpu && !make_padded_k(*ctx, error)) return nullptr;
        if (!ctx->f32_weights && !precision_variant && core_env::on("CRISPEMBED_MIGAN_DEQUANT_WEIGHTS") &&
            !make_dequant(*ctx, error))
            return nullptr;
        if (ctx->gpu) {
            // Build the batch1 graph now: an unsupported GPU op is reported at
            // init, and the context falls back to the reviewed CPU backend.
            try {
                std::unique_ptr<graph_builder> probe(new graph_builder(*ctx, context_options(*ctx, 1)));
                probe->build();
                ctx->graphs[1] = std::move(probe);
            } catch (const std::exception & exception) {
                std::fprintf(stderr, "MI-GAN: %s; reloading on the CPU backend\n", exception.what());
                return init_variant_ex(path, expected_checkpoint_sha256, resolution, n_threads, device_choice::cpu,
                                       expected_source_gguf_sha256, error);
            }
        }
        ctx->stats.weight_buffer_bytes = ggml_backend_buffer_get_size(ctx->weights.buf);
        if (ctx->persistent) {
            ggml_backend_buffer_set_usage(ctx->weights.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
#ifdef CRISPEMBED_MIGAN_TESTING
            ctx->observed_stats = ctx->stats;
#endif
            // Build/support-check/allocate before publishing the context, so
            // explicit device incompatibility is an init error. Warm runs only
            // update the input, compute, and download the final output.
            ctx->cached_graph.reset(new graph_builder(*ctx));
            ctx->cached_graph->build();
            profile(ctx->stats, "ready");
        }
#ifdef CRISPEMBED_MIGAN_TESTING
        else {
            ctx->observed_stats = ctx->stats;
        }
#endif
        return ctx.release();
    } catch (const std::exception & exception) {
        fail(error, exception.what());
        return nullptr;
    }
}

int image_resolution(const context * ctx) {
    return ctx ? ctx->resolution : 0;
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

bool get_execution_stats(context * ctx, execution_stats & stats, std::string & error) {
    error.clear();
    if (!ctx) return fail(error, "MI-GAN statistics require a context");
    std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return fail(error, "MI-GAN context is already in use");
    stats = ctx->stats;
    return true;
}

bool run(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
         std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, raw_nchw, output_count, error)) return false;
    std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return fail(error, "MI-GAN context is already in use");
    if (ctx->persistent) {
        try {
            return ctx->cached_graph->compute(input_nchw, raw_nchw, error);
        } catch (const std::exception & exception) {
            return fail(error, exception.what());
        }
    }
    return execute(ctx, input_nchw, 1, raw_nchw, error);
}

bool run_batch(context * ctx, const float * input_nchw, size_t batch, float * raw_nchw, std::string & error) {
    error.clear();
    if (!gate(ctx ? ctx->resolution : image_size, error)) return false;
    if (!ctx || !input_nchw || !raw_nchw || batch < 1 || batch > size_t(ctx->batch_limit))
        return fail(error, "MI-GAN batch requires 1..max_batch contiguous inputs at the loaded resolution");
    if (ctx->persistent && !core_env::on("CRISPEMBED_INPAINT_GRAPH"))
        return fail(error, "MI-GAN persistent graph requires CRISPEMBED_INPAINT_GRAPH=1");
    const size_t elements = batch * 4 * size_t(ctx->resolution) * ctx->resolution;
    for (size_t i = 0; i < elements; ++i)
        if (!std::isfinite(input_nchw[i])) return fail(error, "MI-GAN nonfinite input");
    std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return fail(error, "MI-GAN context is already in use");
    if (ctx->persistent) {
        try {
            return ctx->cached_graph->compute(input_nchw, raw_nchw, error);
        } catch (const std::exception & exception) {
            return fail(error, exception.what());
        }
    }
    return execute(ctx, input_nchw, batch, raw_nchw, error);
}

bool preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width,
                     int height, float * input_nchw, size_t input_count, std::string & error) {
    return preprocess_rgb8_variant(image_size, rgb, rgb_bytes, fill_mask, mask_bytes, width, height, input_nchw,
                                   input_count, error);
}

bool preprocess_rgb8_variant(int resolution, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask,
                             size_t mask_bytes, int width, int height, float * input_nchw, size_t input_count,
                             std::string & error) {
    error.clear();
    if (!resolution_valid(resolution, error)) return false;
    const size_t pixels = size_t(resolution) * resolution;
    if (width != resolution || height != resolution || !input_nchw || input_count != 4 * pixels)
        return fail(error, "MI-GAN preprocessing requires the selected square NCHW F32 output size");
    if (!rgb_mask_valid(resolution, rgb, rgb_bytes, fill_mask, mask_bytes, error)) return false;
    for (size_t i = 0; i < pixels; ++i) {
        const float known = fill_mask[i] == 0 ? 1.0f : 0.0f;
        input_nchw[i] = known - 0.5f;
        for (size_t channel = 0; channel < 3; ++channel)
            input_nchw[(channel + 1) * pixels + i] = (float(rgb[3 * i + channel]) * 2.0f / 255.0f - 1.0f) * known;
    }
    return true;
}

bool composite_rgb8(const float * raw_nchw, size_t raw_count, const uint8_t * rgb, size_t rgb_bytes,
                    const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb, size_t output_bytes,
                    std::string & error) {
    return composite_rgb8_variant(image_size, raw_nchw, raw_count, rgb, rgb_bytes, fill_mask, mask_bytes, output_rgb,
                                  output_bytes, error);
}

bool composite_rgb8_variant(int resolution, const float * raw_nchw, size_t raw_count, const uint8_t * rgb,
                            size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, uint8_t * output_rgb,
                            size_t output_bytes, std::string & error) {
    error.clear();
    if (!resolution_valid(resolution, error)) return false;
    const size_t pixels = size_t(resolution) * resolution;
    if (!raw_nchw || raw_count != 3 * pixels || !output_rgb || output_bytes != 3 * pixels)
        return fail(error, "MI-GAN composition requires fixed RGB sizes");
    if (!rgb_mask_valid(resolution, rgb, rgb_bytes, fill_mask, mask_bytes, error)) return false;
    for (size_t i = 0; i < raw_count; ++i)
        if (!std::isfinite(raw_nchw[i])) return fail(error, "MI-GAN composition rejects nonfinite raw output");
    for (size_t i = 0; i < pixels; ++i)
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t hwc = 3 * i + channel;
            const float normalized = raw_nchw[channel * pixels + i] * 0.5f + 0.5f;
            const uint8_t generated = uint8_t(std::max(0.0f, std::min(1.0f, normalized)) * 255.0f);
            output_rgb[hwc] = fill_mask[i] ? generated : rgb[hwc];
        }
    return true;
}

#ifdef CRISPEMBED_MIGAN_TESTING
namespace test {
#include "../tests/migan_cpu_kernel_controls.h"

bool graph_backend_controls(int n_threads, int runs, execution_stats & stats, std::string & error) {
    error.clear();
    if (n_threads < 1 || n_threads > 64 || runs < 2 || runs > 1000)
        return fail(error, "MI-GAN graph controls require 1..64 threads and 2..1000 runs");
    try {
        context model;
        model.persistent = true;
        model.stats.persistent = true;
        model.backend = inpaint_graph::select_backend(n_threads, model.stats.backend, error);
        if (!model.backend) return false;
        if (core_cpu_backend::is_cpu(model.backend)) {
            model.threadpool = std::make_unique<inpaint_graph::CpuPool>(model.backend, n_threads);
            model.cpu_kernels = core_env::on("CRISPEMBED_MIGAN_CPU_KERNELS");
        } else {
            if (core_env::on("CRISPEMBED_MIGAN_CPU_KERNELS"))
                throw std::runtime_error("MI-GAN CPU kernels require a selected CPU backend");
            model.fast = core_env::on("CRISPEMBED_INPAINT_FAST");
        }
        model.stats.fast = model.fast;
        model.stats.cpu_kernels = model.cpu_kernels;
        constexpr int width = 5, height = 3, ic = 4, oc = 3;
        constexpr size_t pixels = width * height;
        model.weights.ctx = ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true });
        if (!model.weights.ctx) throw std::runtime_error("MI-GAN graph control weight context failed");
        std::map<std::string, std::vector<float>> weights;
        auto add_weight = [&](const std::string & name, shape dimensions, std::vector<float> values) {
            ggml_tensor * tensor = ggml_new_tensor_4d(model.weights.ctx, GGML_TYPE_F32, dimensions[0], dimensions[1],
                                                      dimensions[2], dimensions[3]);
            ggml_set_name(tensor, name.c_str());
            if (size_t(ggml_nelements(tensor)) != values.size())
                throw std::runtime_error("MI-GAN graph control weight shape mismatch");
            model.weights.tensors.emplace(name, tensor);
            weights.emplace(name, std::move(values));
        };
        add_weight("pointwise.weight", { 1, 1, ic, oc },
                   { 0.25f, -0.5f, 0.75f, -0.125f, -0.3f, 0.2f, -0.1f, 0.4f, 1.0f, 0.0f, -1.0f, 0.5f });
        add_weight("pointwise.bias", { oc, 1, 1, 1 }, { 0.0625f, -0.1875f, 0.25f });
        std::vector<float> insertion(4 * pixels, 0), filter(16 * oc);
        for (int y = 0; y < height * 2; y += 2)
            for (int x = 0; x < width * 2; x += 2) insertion[size_t(y) * width * 2 + x] = 1;
        for (size_t i = 0; i < filter.size(); ++i) filter[i] = float(int((i * 5 + i / 16 * 3) % 11) - 5) / 64;
        add_weight("upsample.filter_const", { width * 2, height * 2, 1, 1 }, insertion);
        add_weight("upsample.filter.weight", { 4, 4, 1, oc }, filter);
        model.weights.buf = ggml_backend_alloc_ctx_tensors(model.weights.ctx, model.backend);
        if (!model.weights.buf) throw std::runtime_error("MI-GAN graph control weight allocation failed");
        ggml_backend_buffer_set_usage(model.weights.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & entry : weights) {
            ggml_backend_tensor_set(model.weights.tensors.at(entry.first), entry.second.data(), 0,
                                    entry.second.size() * sizeof(float));
            ++model.stats.weight_uploads;
            model.stats.weight_upload_bytes += entry.second.size() * sizeof(float);
        }
        model.stats.weight_buffer_bytes = ggml_backend_buffer_get_size(model.weights.buf);
        graph_builder graph(model);
        graph.input = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, width, height, ic, 1);
        ggml_set_input(graph.input);
        ggml_tensor * mixed = graph.activate(graph.convolution("pointwise", graph.input, 1, 0, false, true));
        graph.finish(graph.upsample("upsample", mixed));
        size_t pad_nodes = 0, roll_nodes = 0, matmuls = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph.graph, i);
            if (node->op == GGML_OP_IM2COL) throw std::runtime_error("MI-GAN pointwise graph retained im2col");
            if (node->op == GGML_OP_PAD) {
                ++pad_nodes;
                for (int axis : { 0, 2, 4, 6 })
                    if (node->op_params[axis] != 0)
                        throw std::runtime_error("MI-GAN graph retained unsupported left/top padding");
            }
            if (node->op == GGML_OP_ROLL) ++roll_nodes;
            if (node->op == GGML_OP_MUL_MAT) {
                ++matmuls;
                if (node->op_params[0] != (model.fast ? GGML_PREC_DEFAULT : GGML_PREC_F32))
                    throw std::runtime_error("MI-GAN graph matmul lost F32 precision");
            }
        }
        if (pad_nodes != 1 || roll_nodes != 1 || matmuls != 1)
            throw std::runtime_error("MI-GAN graph control operation census differs");
        void * const input_address = graph.input->data;
        void * const output_address = graph.output->data;
        const ggml_cgraph * const graph_address = graph.graph;
        const ggml_gallocr_t allocator_address = graph.allocator;
        const ggml_backend_graph_plan_t cpu_plan_address = graph.cpu_plan;
        if (core_cpu_backend::is_cpu(model.backend) && !cpu_plan_address)
            throw std::runtime_error("MI-GAN CPU control did not cache its compute plan");
        const auto weights_before = model.stats.weight_uploads;
        const auto bytes_before = model.stats.graph_buffer_bytes;
        std::vector<float> input(ic * pixels), output(oc * 4 * pixels), activation(oc * pixels);
        double maximum_error = 0;
        for (int iteration = 0; iteration < runs; ++iteration) {
            for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 7 + (iteration % 2) * 3) % 19) - 9) / 8;
            input[0] = iteration % 2 ? -4096.0f : 4096.0f; // Exercise both clamp boundaries.
            for (int c = 0; c < oc; ++c)
                for (size_t p = 0; p < pixels; ++p) {
                    double value = weights.at("pointwise.bias")[c];
                    for (int k = 0; k < ic; ++k)
                        value += double(input[size_t(k) * pixels + p]) * weights.at("pointwise.weight")[c * ic + k];
                    float v = float(value);
                    if (v < 0) v *= 0.2f;
                    activation[size_t(c) * pixels + p] =
                        std::max(-256.0f, std::min(256.0f, v * 1.4142135623730950488f));
                }
            if (!graph.compute(input.data(), output.data(), error)) return false;
            for (int c = 0; c < oc; ++c)
                for (int y = 0; y < height * 2; ++y)
                    for (int x = 0; x < width * 2; ++x) {
                        double expected = 0;
                        for (int ky = 0; ky < 4; ++ky)
                            for (int kx = 0; kx < 4; ++kx) {
                                const int iy = y + ky - 2, ix = x + kx - 2;
                                if (iy < 0 || iy >= height * 2 || ix < 0 || ix >= width * 2 || iy % 2 || ix % 2)
                                    continue;
                                expected += double(activation[size_t(c) * pixels + size_t(iy / 2) * width + ix / 2]) *
                                            filter[c * 16 + ky * 4 + kx];
                            }
                        const double actual = output[size_t(c) * 4 * pixels + size_t(y) * width * 2 + x];
                        const double delta = std::abs(actual - expected);
                        maximum_error = std::max(maximum_error, delta);
                        if (!std::isfinite(actual) || delta > 8e-5 + 2e-6 * std::abs(expected))
                            throw std::runtime_error("MI-GAN graph pointwise/padding/depthwise scalar oracle differs");
                    }
            if (graph.input->data != input_address || graph.output->data != output_address ||
                graph.graph != graph_address || graph.allocator != allocator_address ||
                graph.cpu_plan != cpu_plan_address || model.stats.graph_buffer_bytes != bytes_before ||
                model.stats.weight_uploads != weights_before || model.stats.graph_builds != 1 ||
                model.stats.graph_allocations != 1 || model.stats.runs != uint64_t(iteration + 1) ||
                model.stats.reuses != uint64_t(iteration))
                throw std::runtime_error("MI-GAN warm graph reallocated, rebuilt, or reuploaded weights");
        }
        for (const auto & entry : weights) {
            std::vector<float> actual(entry.second.size());
            ggml_backend_tensor_get(model.weights.tensors.at(entry.first), actual.data(), 0,
                                    actual.size() * sizeof(float));
            if (std::memcmp(actual.data(), entry.second.data(), actual.size() * sizeof(float)) != 0)
                throw std::runtime_error("MI-GAN graph mutated immutable weights");
        }
        // An actual competing thread exercises the public context lock. Its
        // buffers satisfy the public full-tile contract; no inference may run.
        std::vector<float> blocked_input(input_elements, 0), blocked_output(rgb_elements, 13);
        bool rejected = false;
        {
            std::unique_lock<std::mutex> lock(model.mutex);
            std::thread competitor([&] {
                std::string busy_error;
                rejected = !run(&model, blocked_input.data(), blocked_input.size(), blocked_output.data(),
                                blocked_output.size(), busy_error) &&
                           busy_error == "MI-GAN context is already in use";
            });
            competitor.join();
        }
        if (!rejected || !std::all_of(blocked_output.begin(), blocked_output.end(), [](float v) { return v == 13; }))
            throw std::runtime_error("MI-GAN same-context concurrency did not reject without output mutation");
        if (model.cpu_kernels) cpu_kernel_regression(n_threads);
        stats = model.stats;
        std::printf("MIGAN_GRAPH_OPS {\"max_abs\":%.9g,\"scalar_oracle_passed\":true,\"weight_bytes_unchanged\":true,"
                    "\"allocation_identity_unchanged\":true,\"overlap_rejected\":true}\n",
                    maximum_error);
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

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

bool profile_ops(context * ctx, const float * input_nchw, size_t batch, const op_timer & timer, std::string & error) {
    error.clear();
    try {
        graph_builder graph(*ctx,
                            reviewed_cpu_graph(*ctx, batch) ? graph_impl::options{} : context_options(*ctx, batch));
        graph.build();
        std::vector<float> raw(batch * rgb_elements);
        if (!graph.compute(input_nchw, raw.data(), error)) return false;
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph.graph, i);
            ggml_context * holder = ggml_init({ ggml_graph_overhead_custom(4, false), nullptr, true });
            ggml_cgraph * single = ggml_new_graph_custom(holder, 4, false);
            ggml_graph_add_node(single, node);
            ggml_backend_graph_compute(ctx->backend, single); // warm the pipeline
            const auto start = std::chrono::steady_clock::now();
            for (int r = 0; r < 3; ++r) ggml_backend_graph_compute(ctx->backend, single);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 3;
            ggml_free(holder);
            ggml_tensor * src0 = node->src[0];
            timer(ggml_op_desc(node), node->ne, src0 ? src0->ne : node->ne, src0 ? ggml_type_name(src0->type) : "", ms);
        }
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

bool run_observed(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
                  const observer & observe, size_t & graph_buffer_bytes, std::string & error) {
    error.clear();
    if (!input_valid(ctx, input_nchw, input_count, raw_nchw, output_count, error)) return false;
    if (!observe) return fail(error, "MI-GAN observer callback is required");
    std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return fail(error, "MI-GAN context is already in use");
    try {
        std::unique_ptr<graph_builder> transient;
        graph_builder * graph;
        if (ctx->persistent) {
            if (!ctx->observed_graph) {
                std::unique_ptr<graph_builder> candidate(new graph_builder(*ctx));
                candidate->retain = true;
                candidate->build();
                ctx->observed_graph = std::move(candidate);
            }
            graph = ctx->observed_graph.get();
        } else {
            transient.reset(new graph_builder(*ctx, reviewed_cpu_graph(*ctx, 1) ? graph_impl::options{}
                                                                                : context_options(*ctx, 1)));
            transient->retain = true;
            transient->build();
            graph = transient.get();
        }
        graph_buffer_bytes = ggml_gallocr_get_buffer_size(graph->allocator, 0);
        if (!graph->compute(input_nchw, raw_nchw, error)) return false;
        std::vector<float> values;
        for (const auto & stage : graph->stages) {
            ggml_tensor * tensor = stage.second;
            if (!ggml_is_contiguous(tensor) || tensor->type != GGML_TYPE_F32)
                return fail(error, "MI-GAN observer requires contiguous F32 stage: " + stage.first);
            values.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
            ++ctx->observed_stats.output_downloads;
            ctx->observed_stats.output_download_bytes += values.size() * sizeof(float);
            if (!observe(stage.first, values.data(), values.size(),
                         { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] }))
                return fail(error, "MI-GAN observer rejected stage: " + stage.first);
        }
        profile(ctx->observed_stats, "observed-with-stages");
        return true;
    } catch (const std::exception & exception) {
        return fail(error, exception.what());
    }
}

bool get_observed_execution_stats(context * ctx, execution_stats & stats, std::string & error) {
    error.clear();
    if (!ctx) return fail(error, "MI-GAN observed statistics require a context");
    std::unique_lock<std::mutex> lock(ctx->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return fail(error, "MI-GAN context is already in use");
    stats = ctx->observed_stats;
    return true;
}
} // namespace test
#endif

} // namespace migan
