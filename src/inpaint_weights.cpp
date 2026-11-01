#include "inpaint_weights.h"

#include "core/env_gate.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

namespace inpaint_weights {
namespace {
bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool supported_stored(ggml_type type) {
    switch (type) {
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

// Natural matmul row: one output channel's (IC, KH, KW) or IN weights.
int64_t natural_row(const Shape & shape) {
    return shape[0] * shape[1] * shape[2];
}
int64_t elements(const Shape & shape) {
    return shape[0] * shape[1] * shape[2] * shape[3];
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
} // namespace

Converted::~Converted() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
}

bool quant_gate() {
    return core_env::on("CRISPEMBED_INPAINT_QUANT");
}
bool matmul_gate() {
    return core_env::on("CRISPEMBED_INPAINT_QUANT_MATMUL");
}

bool is_policy_file(gguf_context * meta) {
    const int64_t key = gguf_find_key(meta, policy_key);
    return key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_STRING;
}

bool validate(gguf_context * meta, const ShapeMap & expected, const char * tag, std::string & error) {
    const std::string name(tag);
    if (!quant_gate())
        return fail(error, name + " precision variant requires CRISPEMBED_INPAINT_QUANT=1 (F32 files need no gate)");
    const int64_t key = gguf_find_key(meta, policy_key);
    if (key < 0 || std::string(gguf_get_val_str(meta, key)) != policy_name)
        return fail(error, name + " precision variant must declare the inpaint-v1 quantization policy");
    if (gguf_get_n_tensors(meta) != int64_t(expected.size()))
        return fail(error, name + " precision variant tensor count differs");
    for (const auto & entry : expected) {
        const int64_t id = gguf_find_tensor(meta, entry.first.c_str());
        if (id < 0) return fail(error, name + " precision variant is missing tensor: " + entry.first);
        const ggml_type type = gguf_get_tensor_type(meta, id);
        const int64_t * ne = gguf_get_tensor_ne(meta, id);
        bool exact = true;
        for (int axis = 0; axis < 4; ++axis) exact &= ne[axis] == entry.second[axis];
        if (type == GGML_TYPE_F32) {
            if (!exact) return fail(error, name + " F32 tensor must keep its original shape: " + entry.first);
            continue;
        }
        const int64_t rows = natural_row(entry.second);
        if (!supported_stored(type) || ne[0] != rows || ne[1] * rows != elements(entry.second) || ne[2] != 1 ||
            ne[3] != 1 || rows % ggml_blck_size(type) != 0)
            return fail(error, name + " stored tensor must be a supported natural-row 2D matrix: " + entry.first);
    }
    return true;
}

bool load(const char * path, gguf_context * meta, ggml_backend_t backend, const ShapeMap & expected, bool keep_stored,
          const char * tag, core_gguf::WeightLoad & weights, Converted & converted, Stats & stats,
          std::string & error) {
    const std::string label(tag);
    std::set<std::string> resident;
    std::vector<std::pair<std::string, ggml_tensor *>> pending;
    for (const auto & entry : expected) {
        const int64_t id = gguf_find_tensor(meta, entry.first.c_str());
        const ggml_type type = gguf_get_tensor_type(meta, id);
        ++stats.stored_types[ggml_type_name(type)];
        if (type == GGML_TYPE_F32 || keep_stored) {
            resident.insert(entry.first);
            if (type != GGML_TYPE_F32) ++stats.stored;
        } else {
            pending.emplace_back(entry.first, nullptr);
        }
    }
    if (!core_gguf::load_weights_filtered(path, backend, include_resident, &resident, tag, weights))
        return fail(error, label + " precision-variant weight load failed");
    if (weights.tensors.size() != resident.size()) return fail(error, label + " resident weight load incomplete");
    stats.resident = int(resident.size());
    stats.weight_bytes = weights.buf ? ggml_backend_buffer_get_size(weights.buf) : 0;
    if (pending.empty()) return true;
    converted.ctx = ggml_init({ pending.size() * ggml_tensor_overhead() + 1024, nullptr, true });
    if (!converted.ctx) return fail(error, label + " dequantization metadata allocation failed");
    for (auto & item : pending) {
        const Shape & dims = expected.at(item.first);
        item.second = ggml_new_tensor_4d(converted.ctx, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
        ggml_set_name(item.second, item.first.c_str());
    }
    converted.buf = ggml_backend_alloc_ctx_tensors(converted.ctx, backend);
    if (!converted.buf) return fail(error, label + " dequantized weight allocation failed");
    stats.weight_bytes += ggml_backend_buffer_get_size(converted.buf);
    std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(path, "rb"), std::fclose);
    if (!file) return fail(error, label + " cannot reopen GGUF for dequantization");
    const size_t data_offset = gguf_get_data_offset(meta);
    std::vector<uint8_t> raw;
    std::vector<float> values;
    for (const auto & item : pending) {
        const int64_t id = gguf_find_tensor(meta, item.first.c_str());
        const ggml_type type = gguf_get_tensor_type(meta, id);
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        const int64_t count = ggml_nelements(item.second);
        raw.resize(gguf_get_tensor_size(meta, id));
        values.resize(size_t(count));
        if (raw.size() != ggml_row_size(type, count) ||
            !seek(file.get(), data_offset + gguf_get_tensor_offset(meta, id)) ||
            std::fread(raw.data(), 1, raw.size(), file.get()) != raw.size())
            return fail(error, label + " cannot read stored tensor: " + item.first);
        if (!traits || !traits->to_float) return fail(error, label + " has no dequantizer for: " + item.first);
        // Natural-row storage is a pure reshape of the original contiguous
        // layout, so dequantized values are already in original order.
        traits->to_float(raw.data(), values.data(), count);
        for (float value : values)
            if (!std::isfinite(value)) return fail(error, label + " nonfinite dequantized weight: " + item.first);
        ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float));
        weights.tensors[item.first] = item.second;
        ++stats.dequantized;
    }
    return true;
}

} // namespace inpaint_weights
