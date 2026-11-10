#include "lighton4b_hf_tiled_out.h"
#include "lighton4b_physical_native_ops.h"
#include "lighton4b_upstream_physical_semantics.h"
#include "qwen35_linear.h"
#include "lighton4b_parallel_dequant.h"
#include "lighton4b_bf16_dequant_weights.h"
#include "qwen35_math.h"
#include "qwen35_profile.h"
#include "qwen35_weight_cache.h"
#include "core/env_gate.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <new>
#include <atomic>
#include <mutex>
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
    else if (lighton4b_parallel_dequant(source->type, raw.data(), count, threads, target)) {
        // Same converted F32 values; existing validation and matmul are unchanged.
    } else {
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
    lighton4b_bf16_dequant_weights(source, target, count);
    stage.next(profile_event::source_validate);
    return std::all_of(target, target + count, [](float x) { return std::isfinite(x); });
}

// Independent quantized storage; never populates the legacy F32 entries.
struct converted_weight_cache::physical_storage {
    struct accounting {
        std::atomic<std::size_t> bytes{ 0 }, live{ 0 };
    };
    struct value {
        ggml_context * context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor * tensor = nullptr;
        const void * source_data = nullptr;
        ggml_backend_buffer_t source_buffer = nullptr;
        int64_t width = 0, rows = 0;
        std::shared_ptr<accounting> account;
        std::size_t charge = 0;
        ~value() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (context) ggml_free(context);
            if (account && charge) {
                account->bytes.fetch_sub(charge, std::memory_order_relaxed);
                account->live.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    };
    static constexpr std::size_t metadata_bytes = 65536;
    static constexpr std::size_t metadata_charge = metadata_bytes + 4096;
    const std::size_t budget;
    std::shared_ptr<accounting> account = std::make_shared<accounting>();
    mutable std::mutex mutex;
    std::map<const ggml_tensor *, std::shared_ptr<value>> entries;
    std::size_t hits = 0;
    explicit physical_storage(std::size_t limit) : budget(limit) {}
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        entries.clear();
        hits = 0; // outstanding aliases keep their allocation charged
    }
    std::shared_ptr<ggml_tensor> get(ggml_tensor * source, ggml_backend_dev_t device) {
        if (!source || source->type != GGML_TYPE_Q4_K || !source->data || !source->buffer ||
            !ggml_backend_buffer_is_host(source->buffer) || !ggml_is_contiguous(source) || source->ne[0] < 256 ||
            source->ne[0] > 32768 || source->ne[0] % 256 || source->ne[1] < 8 || source->ne[1] > 300000 ||
            source->ne[1] % 8 || source->ne[2] != 1 || source->ne[3] != 1 || !device ||
            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU)
            return {};
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = entries.find(source);
        if (found != entries.end()) {
            const auto & v = found->second;
            if (v->source_data != source->data || v->source_buffer != source->buffer || v->width != source->ne[0] ||
                v->rows != source->ne[1])
                return {};
            ++hits;
            return std::shared_ptr<ggml_tensor>(v, v->tensor);
        }
        const auto used = account->bytes.load(std::memory_order_relaxed);
        const auto packed = ggml_nbytes(source);
        if (account->live.load(std::memory_order_relaxed) >= 256 || used > budget || packed > budget - used ||
            physical_repack_entry_overhead > budget - used - packed)
            return {};
        auto * registry = ggml_backend_dev_backend_reg(device);
        if (!registry) return {};
        const auto function = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            registry, "ggml_backend_dev_get_extra_bufts");
        if (!function) return {};
        auto ** types = function(device);
        ggml_backend_buffer_type_t selected = nullptr;
        for (std::size_t i = 0; types && types[i] && i < 32; ++i) {
            if (std::strcmp(ggml_backend_buft_name(types[i]), "CPU_REPACK") == 0) {
                if (selected) return {};
                selected = types[i];
            }
        }
        if (!selected || ggml_backend_buft_get_alignment(selected) > 4096) return {};
        try {
            auto v = std::make_shared<value>();
            v->context = ggml_init({ metadata_bytes, nullptr, true });
            if (!v->context) return {};
            v->tensor = ggml_new_tensor_2d(v->context, source->type, source->ne[0], source->ne[1]);
            ggml_set_name(v->tensor, ggml_get_name(source));
            v->buffer = ggml_backend_alloc_ctx_tensors_from_buft(v->context, selected);
            if (!v->buffer || !v->tensor->data || !v->tensor->extra) return {};
            const auto allocated = ggml_backend_buffer_get_size(v->buffer);
            if (allocated < packed || allocated > packed + 8192 || metadata_charge > budget - used ||
                allocated > budget - used - metadata_charge)
                return {};
            ggml_backend_buffer_set_usage(v->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(v->tensor, source->data, 0, packed);
            v->source_data = source->data;
            v->source_buffer = source->buffer;
            v->width = source->ne[0];
            v->rows = source->ne[1];
            v->account = account;
            v->charge = allocated + metadata_charge;
            account->bytes.fetch_add(v->charge, std::memory_order_relaxed);
            account->live.fetch_add(1, std::memory_order_relaxed);
            entries.emplace(source, v);
            return std::shared_ptr<ggml_tensor>(v, v->tensor);
        } catch (const std::bad_alloc &) {
            return {}; // no arithmetic fallback or uncharged retained allocation
        }
    }
};
bool converted_weight_cache::enable_physical_repack(bool physical_owner, std::size_t budget) {
    if (!physical_owner || !lighton4b_physical_q4_repack_mode(true) || !lighton4b_physical_q4_repack_valid(true) ||
        !budget || budget > maximum_physical_repack_budget)
        return false;
    if (physical_) return physical_->budget == budget;
    try {
        physical_ = std::make_unique<physical_storage>(budget);
    } catch (const std::bad_alloc &) {
        return false;
    }
    return true;
}
bool converted_weight_cache::physical_repack_enabled() const {
    return bool(physical_);
}
std::shared_ptr<ggml_tensor> converted_weight_cache::get_physical_repacked(ggml_tensor * source,
                                                                           ggml_backend_device * device) {
    if (!physical_ || !lighton4b_physical_q4_repack_mode(true) || !lighton4b_physical_q4_repack_valid(true)) return {};
    return physical_->get(source, device);
}
std::size_t converted_weight_cache::physical_repack_bytes() const {
    return physical_ ? physical_->account->bytes.load(std::memory_order_relaxed) : 0;
}
std::size_t converted_weight_cache::physical_repack_entries() const {
    if (!physical_) return 0;
    std::lock_guard<std::mutex> lock(physical_->mutex);
    return physical_->entries.size();
}
std::size_t converted_weight_cache::physical_repack_hits() const {
    if (!physical_) return 0;
    std::lock_guard<std::mutex> lock(physical_->mutex);
    return physical_->hits;
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
    if (physical_) physical_->clear();
}
const float * converted_weight_cache::get(ggml_tensor * source, int threads) {
    if (!core_env::on("CRISPEMBED_QWEN35_WEIGHT_CACHE") || !budget_ || !source ||
        (source->type != GGML_TYPE_F32 && source->type != GGML_TYPE_F16) || !matrix_type_supported(source))
        return nullptr;
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
                 std::vector<std::vector<float>> & output, converted_weight_cache * weights_cache,
                 bool upstream_physical) {
    profile_timer total(profile_event::project_total);
    output.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !hidden || matrices.empty() || matrices.size() > 4 || width < 1 ||
        width > 32768 || tokens < 1 || tokens > 16384 || threads < 1 || threads > 256)
        return false;
    if (!lighton4b_physical_q6_tiled_valid(upstream_physical) ||
        !lighton4b_physical_native_ops_valid(upstream_physical) ||
        !lighton4b_physical_q4_repack_valid(upstream_physical) ||
        (upstream_physical &&
         bool(weights_cache && weights_cache->physical_repack_enabled()) != lighton4b_physical_q4_repack_mode(true)))
        return false;
    const bool native_ops = lighton4b_physical_native_component(upstream_physical, 1);
    constexpr size_t limit = 512ull * 1024 * 1024;
    size_t elements = size_t(width) * tokens;
    for (auto * matrix : matrices) {
        if (!matrix || matrix->ne[0] != width || matrix->ne[1] < 1 || matrix->ne[1] > 300000 || matrix->ne[2] != 1 ||
            matrix->ne[3] != 1 || !matrix_type_supported(matrix))
            return false;
        elements += (native_ops ? 0 : size_t(ggml_nelements(matrix))) + size_t(matrix->ne[1]) * tokens;
        if (elements > limit) return false;
    }
    if (!std::all_of(hidden, hidden + size_t(width) * tokens, [](float x) { return std::isfinite(x); })) return false;
    if (observer && core_env::on("CRISPEMBED_QWEN35_IMATRIX"))
        for (auto * matrix : matrices)
            if (!observer(ggml_get_name(matrix), hidden, width, tokens, observer_user)) return false;
    if (native_ops)
        return lighton4b_physical_matmul(matrices, hidden, width, tokens, threads, output, false, weights_cache);
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

// Experimental row streaming keeps the original projection path and its limit
// intact. Each tile uses the same F32 matmul, with contiguous source rows read
// through the backend API (including non-host-backed weights).
bool project_rows_cpu(ggml_tensor * matrix, const float * hidden, int width, int tokens, int threads, int tile_rows,
                      std::vector<float> & output, bool upstream_physical) {
    output.clear();
    const bool q6 = matrix && matrix->type == GGML_TYPE_Q6_K && lighton4b_q6_enabled() &&
                    core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") && core_env::on("CRISPEMBED_LIGHTON4B_QUANT");
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_QWEN35_STREAM_PROJECTION") || !matrix ||
        !hidden || width < 1 || width > 32768 || tokens < 1 || tokens > 16 || threads < 1 || threads > 256 ||
        tile_rows < 1 || tile_rows > 1024 || size_t(width) * tile_rows > 1048576 || matrix->ne[0] != width ||
        matrix->ne[1] < 1 || matrix->ne[1] > 300000 || matrix->ne[2] != 1 || matrix->ne[3] != 1 || !matrix->data ||
        !(matrix->view_src ? matrix->view_src->buffer : matrix->buffer) || !ggml_is_contiguous(matrix) ||
        (matrix->type != GGML_TYPE_F32 && matrix->type != GGML_TYPE_F16 && matrix->type != GGML_TYPE_Q8_0 &&
         matrix->type != GGML_TYPE_Q4_K && !q6))
        return false;
    if ((matrix->type == GGML_TYPE_Q8_0 || matrix->type == GGML_TYPE_Q4_K || q6) &&
        (!core_env::on("CRISPEMBED_QWEN35_QUANT") || width % ggml_blck_size(matrix->type) != 0 ||
         !ggml_get_type_traits(matrix->type)->to_float))
        return false;
    if (!std::all_of(hidden, hidden + size_t(width) * tokens, [](float x) { return std::isfinite(x); })) return false;
    if (!lighton4b_physical_q6_tiled_valid(upstream_physical) ||
        !lighton4b_physical_native_ops_valid(upstream_physical))
        return false;
    if (lighton4b_physical_native_component(upstream_physical, 4)) {
        try {
            if (observer && core_env::on("CRISPEMBED_QWEN35_IMATRIX") &&
                !observer(ggml_get_name(matrix), hidden, width, tokens, observer_user))
                return false;
            std::vector<std::vector<float>> result;
            if (!lighton4b_physical_matmul({ matrix }, hidden, width, tokens, threads, result, true)) return false;
            output = std::move(result[0]);
            return true;
        } catch (...) {
            return false;
        }
    }
    try {
        // Preserve calibration identity: observe the original named matrix once,
        // then suppress tile callbacks while retaining all old-path behavior.
        if (observer && core_env::on("CRISPEMBED_QWEN35_IMATRIX") &&
            !observer(ggml_get_name(matrix), hidden, width, tokens, observer_user))
            return false;
        struct observer_restore {
            projection_observer saved = observer;
            observer_restore() { observer = nullptr; }
            ~observer_restore() { observer = saved; }
        } restore;
        const size_t rows = size_t(matrix->ne[1]);
        const size_t row_bytes = ggml_row_size(matrix->type, width);
        if (rows > std::numeric_limits<size_t>::max() / row_bytes) return false;
        std::vector<float> complete(rows * tokens);
        std::vector<uint8_t> raw(row_bytes * size_t(tile_rows));
        // Only one bounded tile lives in this context. project_cpu owns a
        // separate bounded graph and never sees the full vocabulary matrix.
        const size_t memory = 1024 * 1024;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({ memory, nullptr, true }), ggml_free);
        if (!context) return false;
        auto * tile = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, tile_rows);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) return false;
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) return false;
        std::vector<std::vector<float>> projected;
        for (size_t begin = 0; begin < rows; begin += size_t(tile_rows)) {
            const size_t count = std::min(size_t(tile_rows), rows - begin);
            ggml_backend_tensor_get(matrix, raw.data(), begin * row_bytes, count * row_bytes);
            auto * values = static_cast<float *>(tile->data);
            if (matrix->type == GGML_TYPE_F32)
                std::memcpy(values, raw.data(), count * size_t(width) * sizeof(float));
            else
                ggml_get_type_traits(matrix->type)->to_float(raw.data(), values, int64_t(count * width));
            lighton4b_bf16_dequant_weights(matrix, values, count * size_t(width));
            tile->ne[1] = int64_t(count);
            tile->nb[2] = tile->nb[1] * count;
            tile->nb[3] = tile->nb[2];
            if (!project_cpu({ tile }, hidden, width, tokens, threads, projected)) return false;
            for (int token = 0; token < tokens; ++token)
                std::copy_n(projected[0].data() + size_t(token) * count, count,
                            complete.data() + size_t(token) * rows + begin);
        }
        output = std::move(complete);
        return true;
    } catch (...) {
        return false;
    }
}

bool linear_cpu(const linear_weights & weights, const float * hidden, int batch, int tokens, int threads,
                const float * conv_history, const float * state, projected_linear_outputs & output,
                converted_weight_cache * weights_cache) {
    output = {};
    if (weights.upstream_physical || weights.hf_tiled_out) return false;
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
                     threads, projections, weights_cache, weights.upstream_physical))
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
                     weights_cache, weights.upstream_physical))
        return false;
    output.attention = std::move(projections[0]);
    return true;
}
bool linear_grouped_cpu(const linear_weights & weights, int key_heads, const float * hidden, int batch, int tokens,
                        int threads, const float * conv_history, const float * state, projected_linear_outputs & output,
                        converted_weight_cache * weights_cache) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_QWEN35_GDN_GROUPED_HEADS") || !hidden ||
        key_heads < 1 || key_heads > 128 || batch < 1 || batch > 4 || tokens < 1 || tokens > 4096 || threads < 1 ||
        threads > 256)
        return false;
    for (auto * tensor : { weights.qkv, weights.z, weights.a, weights.b, weights.out, weights.conv, weights.a_log,
                           weights.dt_bias, weights.norm }) {
        const bool projection = tensor == weights.qkv || tensor == weights.z || tensor == weights.a ||
                                tensor == weights.b || tensor == weights.out;
        const bool quant = projection && core_env::on("CRISPEMBED_LIGHTON4B_QUANT") && tensor &&
                           (tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q4_K ||
                            (tensor->type == GGML_TYPE_Q6_K && lighton4b_q6_enabled() &&
                             core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT"))) &&
                           matrix_type_supported(tensor);
        if (!tensor || (!quant && tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) ||
            !ggml_is_contiguous(tensor))
            return false;
    }
    const int64_t heads = weights.a_log->ne[0], width = weights.norm->ne[0], hidden_width = weights.qkv->ne[0];
    const int64_t kernel = weights.conv->ne[0];
    if (heads < key_heads || heads > 128 || heads % key_heads || width < 1 || width > 256 || hidden_width < 1 ||
        hidden_width > 8192 || kernel < 1 || kernel > 16)
        return false;
    if (heads == key_heads) {
        if (linear_cpu(weights, hidden, batch, tokens, threads, conv_history, state, output, weights_cache))
            return true;
        output = {};
        return false;
    }
    const bool repack = lighton4b_physical_q4_repack_mode(weights.upstream_physical);
    if (!lighton4b_physical_q4_repack_valid(weights.upstream_physical) || weights.physical_q4_repack != repack ||
        (weights.upstream_physical && bool(weights_cache && weights_cache->physical_repack_enabled()) != repack))
        return false;
    if (!lighton4b_hf_tiled_out_mode_valid(weights.hf_tiled_out, weights.upstream_physical)) return false;
    if (weights.hf_tiled_out &&
        (key_heads != 16 || heads != 32 || width != 128 || hidden_width != 2560 || batch != 1 || tokens > 16))
        return false;
    if (weights.upstream_physical && (!lighton4b_upstream_physical_mode_valid(true) || key_heads != 16 || heads != 32 ||
                                      width != 128 || hidden_width != 2560 || batch != 1 || tokens > 16))
        return false;
    const int64_t group = heads * width, channels = (2 * int64_t(key_heads) + heads) * width;
    if (!shape(weights.a_log, heads, 1) || !shape(weights.dt_bias, heads, 1) || !shape(weights.norm, width, 1) ||
        !shape(weights.qkv, hidden_width, channels) || !shape(weights.z, hidden_width, group) ||
        !shape(weights.a, hidden_width, heads) || !shape(weights.b, hidden_width, heads) ||
        !shape(weights.out, group, hidden_width) || weights.conv->ne[1] != 1 || weights.conv->ne[2] != channels ||
        weights.conv->ne[3] != 1)
        return false;
    const auto fail = [&]() {
        output = {};
        return false;
    };
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ weights.qkv, weights.z, weights.a, weights.b }, hidden, int(hidden_width), batch * tokens,
                     threads, projections, weights_cache, weights.upstream_physical))
        return fail();
    output.qkv = std::move(projections[0]);
    output.z = std::move(projections[1]);
    output.a = std::move(projections[2]);
    output.b = std::move(projections[3]);
    std::vector<float> conv(size_t(channels * kernel)), a_log(size_t(heads), 0.0f), dt_bias(size_t(heads), 0.0f),
        norm(size_t(width), 0.0f);
    if (!tensor_floats(weights.conv, conv.data()) || !tensor_floats(weights.a_log, a_log.data()) ||
        !tensor_floats(weights.dt_bias, dt_bias.data()) || !tensor_floats(weights.norm, norm.data()))
        return fail();
    linear_inputs input{
        batch,           tokens,      int(heads),        int(width),      int(kernel),
        threads,         1e-6f,       output.qkv.data(), output.z.data(), output.a.data(),
        output.b.data(), conv.data(), a_log.data(),      dt_bias.data(),  norm.data(),
        conv_history,    state,
    };
    input.a_is_pretransformed = weights.upstream_physical;
    if (!linear_grouped_core_cpu(input, key_heads, output.core)) return fail();
    projections.clear();
    std::vector<float> tiled;
    const float * out_input = output.core.values.data();
    if (weights.upstream_physical) {
        if (!lighton4b_upstream_v_to_tiled(output.core.values, tokens, tiled)) return fail();
        out_input = tiled.data();
    } else if (weights.hf_tiled_out) {
        if (!lighton4b_hf_v_to_tiled(output.core.values, tokens, tiled)) return fail();
        out_input = tiled.data();
    }
    if (!project_cpu({ weights.out }, out_input, int(group), batch * tokens, threads, projections, weights_cache,
                     weights.upstream_physical))
        return fail();
    output.attention = std::move(projections[0]);
    return true;
}
} // namespace qwen35_ocr
