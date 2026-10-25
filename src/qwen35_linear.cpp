#include "qwen35_linear.h"
#include "qwen35_math.h"
#include "qwen35_profile.h"
#include "qwen35_weight_cache.h"
#include "core/env_gate.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <thread>

namespace qwen35_ocr {
namespace {
thread_local projection_observer observer = nullptr;
thread_local void * observer_user = nullptr;
} // namespace
bool set_projection_observer(projection_observer fn, void * user) {
    if (fn && !core_env::on("CRISPEMBED_QWEN35_IMATRIX")) return false;
    observer = fn;
    observer_user = user;
    return true;
}
static bool shape(ggml_tensor * tensor, int64_t columns, int64_t rows) {
    return tensor && tensor->ne[0] == columns && tensor->ne[1] == rows && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool matrix_type_supported(const ggml_tensor * source) {
    if (!source || !ggml_is_contiguous(source)) return false;
    if (source->type == GGML_TYPE_F32 || source->type == GGML_TYPE_F16) return true;
    if (!core_env::on("CRISPEMBED_QWEN35_QUANT") || source->ne[0] < 1 || source->ne[1] < 2 || source->ne[2] != 1 ||
        source->ne[3] != 1)
        return false;
    switch (source->type) {
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q6_K:
        return source->ne[0] % ggml_blck_size(source->type) == 0 && ggml_get_type_traits(source->type)->to_float;
    default:
        return false;
    }
}
int f16_conversion_workers(size_t elements, int threads) {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    (void)elements;
    (void)threads;
    return 1;
#else
    constexpr size_t grain = 65536;
    if (elements < 4 * grain || threads < 2 || !core_env::on("CRISPEMBED_QWEN35_PARALLEL_F16")) return 1;
    return int(std::min({ size_t(threads), size_t(8), elements / grain }));
#endif
}
bool tensor_floats(ggml_tensor * source, float * target, int threads) {
    if (!target || !matrix_type_supported(source)) return false;
    profile_timer stage(profile_event::source_buffer);
    std::vector<uint8_t> raw(ggml_nbytes(source));
    stage.next(profile_event::source_read);
    ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
    stage.next(profile_event::source_convert);
    const size_t count = size_t(ggml_nelements(source));
    if (source->type == GGML_TYPE_F32)
        std::memcpy(target, raw.data(), count * sizeof(float));
    else {
        const auto convert = ggml_get_type_traits(source->type)->to_float;
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
        const int workers = source->type == GGML_TYPE_F16 ? f16_conversion_workers(count, threads) : 1;
        if (workers > 1) {
            std::array<std::thread, 7> jobs;
            const auto chunk = [&](int index) {
                const size_t begin = count * size_t(index) / size_t(workers);
                const size_t end = count * size_t(index + 1) / size_t(workers);
                convert(raw.data() + begin * sizeof(ggml_fp16_t), target + begin, int64_t(end - begin));
            };
            bool fallback = false;
            try {
                for (int i = 1; i < workers; ++i) jobs[size_t(i - 1)] = std::thread(chunk, i);
            } catch (const std::exception &) {
                // Join every launched writer before safely overwriting serially.
                fallback = true;
            }
            if (!fallback) chunk(0);
            for (auto & job : jobs)
                if (job.joinable()) job.join();
            if (fallback) convert(raw.data(), target, int64_t(count));
        } else
#endif
            convert(raw.data(), target, int64_t(count));
    }
    stage.next(profile_event::source_validate);
    return std::all_of(target, target + count, [](float x) { return std::isfinite(x); });
}

struct converted_weight_cache::entry {
    ggml_backend_buffer_t buffer = nullptr;
    ~entry() {
        if (buffer) ggml_backend_buffer_free(buffer);
    }
};
std::size_t converted_weight_cache::environment_budget() {
    const char * value = std::getenv("CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES");
    if (!value) return default_budget;
    if (!*value) return 0;
    std::size_t result = 0;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > (maximum_budget - std::size_t(*p - '0')) / 10) return 0;
        result = result * 10 + std::size_t(*p - '0');
    }
    return result;
}
converted_weight_cache::converted_weight_cache(std::size_t budget) : budget_(budget <= maximum_budget ? budget : 0) {}
converted_weight_cache::~converted_weight_cache() = default;
void converted_weight_cache::clear() {
    entries_.clear();
    bytes_ = hits_ = 0;
}
const float * converted_weight_cache::get(ggml_tensor * source, int threads) {
    if (!core_env::on("CRISPEMBED_QWEN35_WEIGHT_CACHE") || !budget_ || !matrix_type_supported(source)) return nullptr;
    const auto found = entries_.find(source);
    if (found != entries_.end()) {
        ++hits_;
        return static_cast<const float *>(ggml_backend_buffer_get_base(found->second->buffer));
    }
    const auto count = std::size_t(ggml_nelements(source));
    if (!count || count > (budget_ - bytes_) / sizeof(float)) return nullptr;
    try {
        auto value = std::make_unique<entry>();
        value->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), count * sizeof(float));
        if (!value->buffer || ggml_backend_buffer_get_size(value->buffer) > budget_ - bytes_) return nullptr;
        auto * data = static_cast<float *>(ggml_backend_buffer_get_base(value->buffer));
        {
            profile_matrix_scope materialize(source);
            if (!tensor_floats(source, data, threads)) return nullptr;
        }
        const auto bytes = ggml_backend_buffer_get_size(value->buffer);
        entries_.emplace(source, std::move(value));
        bytes_ += bytes;
        return data;
    } catch (const std::bad_alloc &) {
        // No admitted state/counters change on allocation failure. The caller
        // retains the original serial/parallel materialization fallback.
        return nullptr;
    }
}

bool project_cpu(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width, int tokens, int threads,
                 std::vector<std::vector<float>> & output, converted_weight_cache * weights_cache) {
    profile_timer total(profile_event::project_total);
    output.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !hidden || matrices.empty() || matrices.size() > 4 || width < 1 ||
        width > 32768 || tokens < 1 || tokens > 16384 || threads < 1 || threads > 256)
        return false;
    constexpr size_t limit = 512ull * 1024 * 1024;
    size_t elements = size_t(width) * tokens;
    for (auto * matrix : matrices) {
        if (!matrix || matrix->ne[0] != width || matrix->ne[1] < 1 || matrix->ne[1] > 300000 || matrix->ne[2] != 1 ||
            matrix->ne[3] != 1 || !matrix_type_supported(matrix))
            return false;
        elements += size_t(ggml_nelements(matrix)) + size_t(matrix->ne[1]) * tokens;
        if (elements > limit) return false;
    }
    if (!std::all_of(hidden, hidden + size_t(width) * tokens, [](float x) { return std::isfinite(x); })) return false;
    if (observer && core_env::on("CRISPEMBED_QWEN35_IMATRIX"))
        for (auto * matrix : matrices)
            if (!observer(ggml_get_name(matrix), hidden, width, tokens, observer_user)) return false;
    profile_timer stage(profile_event::project_setup);
    std::array<const float *, 4> cached{};
    size_t scratch_elements = elements;
    if (weights_cache) {
        stage.next(profile_event::project_materialize);
        for (size_t i = 0; i < matrices.size(); ++i) {
            cached[i] = weights_cache->get(matrices[i], threads);
            if (cached[i]) scratch_elements -= size_t(ggml_nelements(matrices[i]));
        }
        stage.next(profile_event::project_setup);
    }
    const size_t memory = 1024 * 1024 + scratch_elements * sizeof(float);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ memory, nullptr, false }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return false;
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    stage.next(profile_event::project_input);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    std::memcpy(input->data, hidden, size_t(width) * tokens * sizeof(float));
    auto * graph = ggml_new_graph(ctx.get());
    std::vector<ggml_tensor *> results;
    for (size_t i = 0; i < matrices.size(); ++i) {
        auto * matrix = matrices[i];
        stage.next(profile_event::project_materialize);
        ggml_set_no_alloc(ctx.get(), cached[i] != nullptr);
        auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, matrix->ne[0], matrix->ne[1]);
        ggml_set_no_alloc(ctx.get(), false);
        if (cached[i]) {
            weight->data = const_cast<float *>(cached[i]);
        } else {
            profile_matrix_scope materialize(matrix);
            if (!tensor_floats(matrix, static_cast<float *>(weight->data), threads)) return false;
        }
        stage.next(profile_event::project_graph);
        auto * result = ggml_mul_mat(ctx.get(), weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        ggml_build_forward_expand(graph, result);
        results.push_back(result);
    }
    stage.next(profile_event::project_compute);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return false;
    stage.next(profile_event::project_output);
    for (auto * result : results) {
        auto * values = static_cast<float *>(result->data);
        const size_t count = size_t(ggml_nelements(result));
        if (!std::all_of(values, values + count, [](float x) { return std::isfinite(x); })) return false;
        output.emplace_back(values, values + count);
    }
    if (profile_active()) {
        stage.next(profile_event::project_cleanup);
        results.clear();
        backend.reset();
        ctx.reset();
    }
    return true;
}

bool linear_cpu(const linear_weights & weights, const float * hidden, int batch, int tokens, int threads,
                const float * conv_history, const float * state, projected_linear_outputs & output,
                converted_weight_cache * weights_cache) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !hidden || batch < 1 || batch > 4 || tokens < 1 || tokens > 4096 ||
        threads < 1 || threads > 256 || !weights.a_log || !weights.norm || !weights.qkv || !weights.conv)
        return false;
    const int64_t heads = weights.a_log->ne[0], width = weights.norm->ne[0], hidden_width = weights.qkv->ne[0];
    const int64_t kernel = weights.conv->ne[0];
    if (heads < 1 || heads > 128 || width < 1 || width > 256 || hidden_width < 1 || hidden_width > 8192 || kernel < 1 ||
        kernel > 16)
        return false;
    const int64_t group = heads * width;
    if (!shape(weights.a_log, heads, 1) || !shape(weights.dt_bias, heads, 1) || !shape(weights.norm, width, 1) ||
        !shape(weights.qkv, hidden_width, 3 * group) || !shape(weights.z, hidden_width, group) ||
        !shape(weights.a, hidden_width, heads) || !shape(weights.b, hidden_width, heads) ||
        !shape(weights.out, group, hidden_width) || weights.conv->ne[1] != 1 || weights.conv->ne[2] != 3 * group ||
        weights.conv->ne[3] != 1)
        return false;
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ weights.qkv, weights.z, weights.a, weights.b }, hidden, int(hidden_width), batch * tokens,
                     threads, projections, weights_cache))
        return false;
    output.qkv = std::move(projections[0]);
    output.z = std::move(projections[1]);
    output.a = std::move(projections[2]);
    output.b = std::move(projections[3]);
    std::vector<float> conv(size_t(3 * group * kernel)), a_log(size_t(heads), 0.0f), dt_bias(size_t(heads), 0.0f),
        norm(size_t(width), 0.0f);
    if (!tensor_floats(weights.conv, conv.data()) || !tensor_floats(weights.a_log, a_log.data()) ||
        !tensor_floats(weights.dt_bias, dt_bias.data()) || !tensor_floats(weights.norm, norm.data()))
        return false;
    const linear_inputs input{
        batch,           tokens,      int(heads),        int(width),      int(kernel),
        threads,         1e-6f,       output.qkv.data(), output.z.data(), output.a.data(),
        output.b.data(), conv.data(), a_log.data(),      dt_bias.data(),  norm.data(),
        conv_history,    state,
    };
    if (!linear_core_cpu(input, output.core)) return false;
    projections.clear();
    if (!project_cpu({ weights.out }, output.core.values.data(), int(group), batch * tokens, threads, projections,
                     weights_cache))
        return false;
    output.attention = std::move(projections[0]);
    return true;
}
} // namespace qwen35_ocr
