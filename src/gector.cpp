// Exact prepared-input BERT/GECToR graph for one pinned, unpadded source profile.
// Kept separate from the monolithic embedding/OCR library so its dependency
// boundary consists of this file, the shared GGUF loader and the ggml CPU backend.
#include "gector.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int H = CRISPEMBED_GECTOR_HIDDEN_SIZE;
constexpr int FF = 3072;
constexpr int HEADS = 12;
constexpr int HEAD_DIM = 64;
constexpr int LAYERS = 12;
constexpr float LN_EPSILON = 1e-12f;
constexpr float QK_SCALE = 0.3535533845424652f;
constexpr const char * SOURCE_REVISION = "8e3b50beef7d775cedb7b876009e88bcf2e8d6af";
constexpr const char * SOURCE_SHA256 = "6d7b43b4bd95c7a00c02c8babc7569d28558df894afe4874d96c202e6ab9a464";

void message(char * error, size_t capacity, const char * text) noexcept {
    if (error && capacity) std::snprintf(error, capacity, "%s", text ? text : "");
}
int failure(int status, char * error, size_t capacity, const char * text) noexcept {
    message(error, capacity, text);
    return status;
}
void require(bool condition, const std::string & text) {
    if (!condition) throw std::runtime_error(text);
}
bool enabled() {
    // Default-off prepared-input German grammar experiment; never a model alias.
    return core_env::on("CRISPEMBED_GECTOR");
}

struct Metadata {
    gguf_context * file = nullptr;
    ggml_context * tensors = nullptr;
    ~Metadata() {
        if (file) gguf_free(file);
        if (tensors) ggml_free(tensors);
    }
};

void exact_u32(gguf_context * g, const char * key, uint32_t wanted) {
    const int64_t i = gguf_find_key(g, key);
    require(i >= 0, std::string("Missing metadata: ") + key);
    // The converter writes uint32. Do not admit lossy u64-to-u32 casts.
    require(gguf_get_kv_type(g, i) == GGUF_TYPE_UINT32 && gguf_get_val_u32(g, i) == wanted,
            std::string("Unsupported metadata value/type: ") + key);
}
void exact_f32(gguf_context * g, const char * key, float wanted) {
    const int64_t i = gguf_find_key(g, key);
    require(i >= 0, std::string("Missing metadata: ") + key);
    require(gguf_get_kv_type(g, i) == GGUF_TYPE_FLOAT32 && gguf_get_val_f32(g, i) == wanted,
            std::string("Unsupported metadata value/type: ") + key);
}
void exact_string(gguf_context * g, const char * key, const char * wanted) {
    const int64_t i = gguf_find_key(g, key);
    require(i >= 0, std::string("Missing metadata: ") + key);
    require(gguf_get_kv_type(g, i) == GGUF_TYPE_STRING && std::strcmp(gguf_get_val_str(g, i), wanted) == 0,
            std::string("Unsupported metadata value/type: ") + key);
}

enum class TensorRole { vector, encoder, token_embedding, protected_matrix };
void validate_model(const char * path) {
    Metadata m;
    m.file = gguf_init_from_file(path, { true, &m.tensors });
    require(m.file && m.tensors, "Unable to read GGUF tensor metadata");
    exact_string(m.file, "general.architecture", "gector");
    exact_u32(m.file, "gector.format_version", 1);
    exact_u32(m.file, "gector.block_count", LAYERS);
    exact_u32(m.file, "gector.embedding_length", H);
    exact_u32(m.file, "gector.feed_forward_length", FF);
    exact_u32(m.file, "gector.attention.head_count", HEADS);
    exact_u32(m.file, "gector.context_length", CRISPEMBED_GECTOR_MAX_TOKENS);
    exact_u32(m.file, "gector.vocab_size", CRISPEMBED_GECTOR_VOCAB_SIZE);
    exact_u32(m.file, "gector.type_vocab_size", 2);
    exact_u32(m.file, "gector.label_count", CRISPEMBED_GECTOR_LABEL_COUNT);
    exact_u32(m.file, "gector.detect_count", CRISPEMBED_GECTOR_DETECT_COUNT);
    exact_f32(m.file, "gector.layer_norm_epsilon", LN_EPSILON);
    exact_f32(m.file, "gector.attention.qk_scale", QK_SCALE);
    exact_string(m.file, "gector.activation", "gelu_erf");
    exact_string(m.file, "gector.mask_policy", "unpadded_all_ones");
    exact_string(m.file, "gector.source.repository", "proofbird/gector-bert-de");
    exact_string(m.file, "gector.source.revision", SOURCE_REVISION);
    exact_string(m.file, "gector.source.sha256", SOURCE_SHA256);
    require(gguf_get_n_tensors(m.file) == 201, "Expected exactly 201 trained GECToR tensors");

    FILE * raw = ggml_fopen(path, "rb");
    require(raw != nullptr, "Unable to reopen GGUF for extent validation");
#if defined(_WIN32)
    const bool seek_ok = _fseeki64(raw, 0, SEEK_END) == 0;
    const int64_t file_size = seek_ok ? _ftelli64(raw) : -1;
#else
    const bool seek_ok = fseeko(raw, 0, SEEK_END) == 0;
    const int64_t file_size = seek_ok ? ftello(raw) : -1;
#endif
    std::fclose(raw);
    require(file_size > 0, "Unable to read GGUF file size");
    const uint64_t file_bytes = static_cast<uint64_t>(file_size);
    const uint64_t data_offset = gguf_get_data_offset(m.file);
    require(data_offset <= file_bytes, "Truncated GGUF header");
    std::set<std::string> names;
    auto tensor = [&](const std::string & name, int64_t columns, int64_t rows, TensorRole role) {
        const int64_t index = gguf_find_tensor(m.file, name.c_str());
        auto * t = ggml_get_tensor(m.tensors, name.c_str());
        require(index >= 0 && t, "Missing tensor: " + name);
        require(t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1,
                "Wrong tensor shape: " + name);
        bool allowed = t->type == GGML_TYPE_F32;
        if (role != TensorRole::vector) allowed = allowed || t->type == GGML_TYPE_F16;
        if (role == TensorRole::encoder || role == TensorRole::token_embedding)
            allowed = allowed || t->type == GGML_TYPE_Q8_0;
        if (role == TensorRole::encoder) allowed = allowed || t->type == GGML_TYPE_Q4_K;
        require(allowed, "Unsupported tensor precision: " + name);
        const uint64_t offset = gguf_get_tensor_offset(m.file, index);
        const uint64_t bytes = ggml_nbytes(t);
        require(offset <= file_bytes - data_offset && bytes <= file_bytes - data_offset - offset,
                "Truncated tensor payload: " + name);
        require(names.insert(name).second, "Duplicate tensor: " + name);
    };
    tensor("token_embd.weight", H, CRISPEMBED_GECTOR_VOCAB_SIZE, TensorRole::token_embedding);
    tensor("position_embd.weight", H, CRISPEMBED_GECTOR_MAX_TOKENS, TensorRole::protected_matrix);
    tensor("token_type_embd.weight", H, 2, TensorRole::vector);
    tensor("embd_ln.weight", H, 1, TensorRole::vector);
    tensor("embd_ln.bias", H, 1, TensorRole::vector);
    for (int i = 0; i < LAYERS; ++i) {
        const auto prefix = "enc." + std::to_string(i) + ".";
        for (const char * part : { "attn.q", "attn.k", "attn.v", "attn.o" }) {
            tensor(prefix + part + ".weight", H, H, TensorRole::encoder);
            tensor(prefix + part + ".bias", H, 1, TensorRole::vector);
        }
        tensor(prefix + "ffn.fc1.weight", H, FF, TensorRole::encoder);
        tensor(prefix + "ffn.fc1.bias", FF, 1, TensorRole::vector);
        tensor(prefix + "ffn.fc2.weight", FF, H, TensorRole::encoder);
        tensor(prefix + "ffn.fc2.bias", H, 1, TensorRole::vector);
        for (const char * part : { "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias" })
            tensor(prefix + part, H, 1, TensorRole::vector);
    }
    tensor("gector.labels.weight", H, CRISPEMBED_GECTOR_LABEL_COUNT, TensorRole::protected_matrix);
    tensor("gector.labels.bias", CRISPEMBED_GECTOR_LABEL_COUNT, 1, TensorRole::vector);
    tensor("gector.detect.weight", H, CRISPEMBED_GECTOR_DETECT_COUNT, TensorRole::protected_matrix);
    tensor("gector.detect.bias", CRISPEMBED_GECTOR_DETECT_COUNT, 1, TensorRole::vector);
}

bool overlaps(const float * a, size_t na, const float * b, size_t nb) {
    if (!a || !b || na == 0 || nb == 0) return false;
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a), pb = reinterpret_cast<uintptr_t>(b);
    // Counts below are bounded to fixed-profile outputs (at most 512*5000).
    return pa <= pb ? pb - pa < na * sizeof(float) : pa - pb < nb * sizeof(float);
}

// The frozen ONNX export replaces post-softmax NaNs with zero. With admitted
// unpadded finite inputs this is normally an identity, but retain its semantics.
void zero_nan(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void *) {
    const int64_t count = ggml_nelements(src);
    const auto * in = static_cast<const float *>(src->data);
    auto * out = static_cast<float *>(dst->data);
    for (int64_t i = count * ith / nth; i < count * (ith + 1) / nth; ++i)
        out[i] = std::isnan(in[i]) ? 0.0f : in[i];
}
} // namespace

struct crispembed_gector_context {
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_context * graph_context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * types = nullptr;
    ggml_tensor * labels = nullptr;
    ggml_tensor * detect = nullptr;
    ggml_tensor * hidden = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> stages;
    size_t graph_tokens = 0;
    bool graph_trace = false;
    bool graph_hidden = false;
    int threads = 1;
    std::mutex mutex;

    void clear_graph() {
        if (allocator) ggml_gallocr_free(allocator);
        if (graph_context) ggml_free(graph_context);
        allocator = nullptr;
        graph_context = nullptr;
        graph = nullptr;
        graph_tokens = 0;
        stages.clear();
    }
    ~crispembed_gector_context() {
        clear_graph();
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};

namespace {
void build_graph(crispembed_gector_context & c, size_t tokens, bool trace, bool keep_hidden) {
    c.clear_graph();
    constexpr size_t graph_nodes = 4096;
    const size_t metadata_bytes = graph_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false);
    c.graph_context = ggml_init({ metadata_bytes, nullptr, true });
    require(c.graph_context, "Unable to allocate graph metadata");
    auto * g = c.graph_context;
    c.graph = ggml_new_graph_custom(g, graph_nodes, false);
    auto weight = [&](const std::string & name) { return c.weights.tensors.at(name); };
    auto mm = [&](ggml_tensor * w, ggml_tensor * x) {
        auto * y = ggml_mul_mat(g, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return y;
    };
    auto linear = [&](const std::string & name, ggml_tensor * x) {
        return ggml_add(g, mm(weight(name + ".weight"), x), weight(name + ".bias"));
    };
    auto norm = [&](const std::string & name, ggml_tensor * x) {
        return ggml_add(g, ggml_mul(g, ggml_norm(g, x, LN_EPSILON), weight(name + ".weight")),
                        weight(name + ".bias"));
    };
    auto stage = [&](const std::string & name, ggml_tensor * t) {
        if (trace) {
            ggml_set_output(t);
            c.stages.emplace_back(name, t);
        }
    };
    const int64_t T = static_cast<int64_t>(tokens);
    c.ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    c.positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    c.types = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(c.ids, "input_ids");
    ggml_set_name(c.positions, "position_ids");
    ggml_set_name(c.types, "token_type_ids");
    for (auto * t : { c.ids, c.positions, c.types }) ggml_set_input(t);

    // Frozen HF BERT order: word + token type, then absolute positions, then LN.
    auto * cur = ggml_add(g, ggml_get_rows(g, weight("token_embd.weight"), c.ids),
                         ggml_get_rows(g, weight("token_type_embd.weight"), c.types));
    cur = ggml_add(g, cur, ggml_get_rows(g, weight("position_embd.weight"), c.positions));
    cur = norm("embd_ln", cur);
    stage("embedding", cur);
    for (int i = 0; i < LAYERS; ++i) {
        const std::string p = "enc." + std::to_string(i) + ".";
        auto * residual = cur;
        auto split_heads = [&](ggml_tensor * x) {
            return ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, x, HEAD_DIM, HEADS, T), 0, 2, 1, 3));
        };
        // Preserve the source's two separately rounded scaling operations.
        auto * q = ggml_scale(g, split_heads(linear(p + "attn.q", cur)), QK_SCALE);
        auto * k = ggml_scale(g, split_heads(linear(p + "attn.k", cur)), QK_SCALE);
        auto * v = split_heads(linear(p + "attn.v", cur));
        auto * scores = mm(k, q); // [T_key,T_query,heads]
        auto * probabilities = ggml_soft_max(g, scores);
        probabilities = ggml_map_custom1(g, probabilities, zero_nan, GGML_N_TASKS_MAX, nullptr);
        auto * vt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
        auto * attention = mm(vt, probabilities); // [head_dim,T_query,heads]
        attention = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, attention, 0, 2, 1, 3)), H, T);
        cur = norm(p + "ln1", ggml_add(g, residual, linear(p + "attn.o", attention)));
        residual = cur;
        auto * ff = ggml_gelu_erf(g, linear(p + "ffn.fc1", cur));
        cur = norm(p + "ln2", ggml_add(g, residual, linear(p + "ffn.fc2", ff)));
        stage("layer." + std::to_string(i), cur);
    }
    c.hidden = cur;
    if (keep_hidden) ggml_set_output(c.hidden);
    c.labels = linear("gector.labels", cur);
    c.detect = linear("gector.detect", cur);
    ggml_set_output(c.labels);
    ggml_set_output(c.detect);
    ggml_set_name(c.labels, "logits_labels");
    ggml_set_name(c.detect, "logits_d");
    stage("labels", c.labels);
    stage("detect", c.detect);
    ggml_build_forward_expand(c.graph, c.labels);
    ggml_build_forward_expand(c.graph, c.detect);
    c.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend));
    require(c.allocator && ggml_gallocr_alloc_graph(c.allocator, c.graph), "Unable to allocate CPU graph workspace");
    c.graph_tokens = tokens;
    c.graph_trace = trace;
    c.graph_hidden = keep_hidden;
}

std::vector<float> read_finite(ggml_tensor * t) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
            "Non-finite native output; no logits returned");
    return values;
}
} // namespace

extern "C" int crispembed_gector_validate_inputs(
    const int64_t * ids, size_t count, const int64_t * mask, size_t mask_count, size_t batch,
    char * error, size_t error_capacity) {
    message(error, error_capacity, "");
    if (batch != 1)
        return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity, "Only batch size one is supported");
    if (!ids || !mask || count == 0 || count > CRISPEMBED_GECTOR_MAX_TOKENS || mask_count != count)
        return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity,
                       "IDs/mask must have matching nonempty lengths from 1 through 512");
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= CRISPEMBED_GECTOR_VOCAB_SIZE)
            return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity, "Token ID outside [0,31102)");
        if (mask[i] != 1)
            return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity,
                           "Only an unpadded all-ones attention mask is supported");
    }
    return CRISPEMBED_GECTOR_OK;
}

extern "C" crispembed_gector_context * crispembed_gector_create(
    const char * path, int threads, char * error, size_t error_capacity) {
    message(error, error_capacity, "");
    if (!enabled()) {
        message(error, error_capacity, "German GECToR is disabled; opt in with CRISPEMBED_GECTOR=1");
        return nullptr;
    }
    if (!path || !*path || threads < 1 || threads > 64) {
        message(error, error_capacity, "A model path and a thread count from 1 through 64 are required");
        return nullptr;
    }
    try {
        validate_model(path); // Reject unsupported shapes/precisions before weight allocation.
        auto c = std::make_unique<crispembed_gector_context>();
        c->threads = threads;
        c->backend = ggml_backend_cpu_init();
        require(c->backend, "Unable to initialize ggml CPU backend");
        ggml_backend_cpu_set_n_threads(c->backend, threads);
        require(core_gguf::load_weights(path, c->backend, "gector", c->weights, true), "Unable to load GGUF weights");
        // The closed target uses host-resident CPU buffers. Reject invalid
        // F32/F16 values and non-finite quantization scales before operators
        // (notably softmax) can encounter malformed model data.
        for (const auto & entry : c->weights.tensors) {
            const auto * t = entry.second;
            require(t->data && ggml_validate_row_data(t->type, t->data, ggml_nbytes(t)),
                    "Invalid tensor data: " + entry.first);
        }
        return c.release();
    } catch (const std::exception & ex) {
        message(error, error_capacity, ex.what());
    } catch (...) {
        message(error, error_capacity, "Unknown model initialization failure");
    }
    return nullptr;
}

extern "C" void crispembed_gector_free(crispembed_gector_context * c) { delete c; }
extern "C" int crispembed_gector_threads(const crispembed_gector_context * c) { return c ? c->threads : 0; }

extern "C" int crispembed_gector_forward_trace(
    crispembed_gector_context * c,
    const int64_t * ids, size_t count, const int64_t * mask, size_t mask_count, size_t batch,
    float * labels, size_t label_capacity, float * detect, size_t detect_capacity,
    float * hidden, size_t hidden_capacity,
    crispembed_gector_trace_callback trace, void * trace_user, char * error, size_t error_capacity) {
    const int valid = crispembed_gector_validate_inputs(ids, count, mask, mask_count, batch, error, error_capacity);
    if (valid != CRISPEMBED_GECTOR_OK) return valid;
    if (!c) return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity, "Context is null");
    if (!enabled())
        return failure(CRISPEMBED_GECTOR_DISABLED, error, error_capacity, "German GECToR gate is disabled");
    const size_t nl = count * CRISPEMBED_GECTOR_LABEL_COUNT, nd = count * CRISPEMBED_GECTOR_DETECT_COUNT;
    const size_t nh = count * H;
    if (!labels || !detect || label_capacity < nl || detect_capacity < nd ||
        (hidden ? hidden_capacity < nh : hidden_capacity != 0) || overlaps(labels, nl, detect, nd) ||
        overlaps(labels, nl, hidden, nh) || overlaps(detect, nd, hidden, nh))
        return failure(CRISPEMBED_GECTOR_INVALID_ARGUMENT, error, error_capacity,
                       "Output buffers must have sufficient non-overlapping float element capacity");
    try {
        std::lock_guard<std::mutex> lock(c->mutex);
        const bool has_trace = trace != nullptr;
        if (c->graph_tokens != count || c->graph_trace != has_trace || c->graph_hidden != (hidden != nullptr))
            build_graph(*c, count, has_trace, hidden != nullptr);
        std::vector<int32_t> token_ids(count), positions(count), types(count, 0);
        for (size_t i = 0; i < count; ++i) {
            token_ids[i] = static_cast<int32_t>(ids[i]);
            positions[i] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(c->ids, token_ids.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set(c->positions, positions.data(), 0, count * sizeof(int32_t));
        ggml_backend_tensor_set(c->types, types.data(), 0, count * sizeof(int32_t));
        require(ggml_backend_graph_compute(c->backend, c->graph) == GGML_STATUS_SUCCESS, "ggml CPU compute failed");
        const auto label_values = read_finite(c->labels);
        const auto detect_values = read_finite(c->detect);
        const auto hidden_values = hidden ? read_finite(c->hidden) : std::vector<float>();
        if (trace) {
            for (const auto & entry : c->stages) {
                const auto values = read_finite(entry.second);
                if (trace(trace_user, entry.first.c_str(), values.data(), count, static_cast<size_t>(entry.second->ne[0])))
                    return failure(CRISPEMBED_GECTOR_TRACE_ERROR, error, error_capacity, "Diagnostic callback aborted delivery");
            }
        }
        std::copy(label_values.begin(), label_values.end(), labels);
        std::copy(detect_values.begin(), detect_values.end(), detect);
        if (hidden) std::copy(hidden_values.begin(), hidden_values.end(), hidden);
        return CRISPEMBED_GECTOR_OK;
    } catch (const std::bad_alloc &) {
        return failure(CRISPEMBED_GECTOR_ALLOCATION_ERROR, error, error_capacity, "Native allocation failed");
    } catch (const std::exception & ex) {
        return failure(CRISPEMBED_GECTOR_RUNTIME_ERROR, error, error_capacity, ex.what());
    } catch (...) {
        return failure(CRISPEMBED_GECTOR_RUNTIME_ERROR, error, error_capacity, "Unknown native inference failure");
    }
}

extern "C" int crispembed_gector_forward(
    crispembed_gector_context * c,
    const int64_t * ids, size_t count, const int64_t * mask, size_t mask_count, size_t batch,
    float * labels, size_t label_capacity, float * detect, size_t detect_capacity,
    float * hidden, size_t hidden_capacity, char * error, size_t error_capacity) {
    return crispembed_gector_forward_trace(c, ids, count, mask, mask_count, batch, labels, label_capacity,
                                           detect, detect_capacity, hidden, hidden_capacity, nullptr, nullptr,
                                           error, error_capacity);
}
