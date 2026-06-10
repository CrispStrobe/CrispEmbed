// crispembed_diff.h — ground-truth diff harness for CrispEmbed.
//
// Companion to tools/dump_reference.py. Loads a GGUF reference archive
// produced by the Python dumper, then compares any C++ tensor against
// the corresponding named reference tensor. Reports cosine similarity,
// max-abs diff, and element-wise metrics.
//
// Usage:
//
//   crispembed_diff::Ref ref;
//   if (!ref.load("/tmp/minilm-ref.gguf")) return 1;
//
//   auto r = ref.compare("enc_layer_0_3", cpp_layer3_data, n_elem);
//   printf("layer 3: cos_min=%.6f max_abs=%.2e  %s\n",
//          r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
//
// Per-layer diff recipe (how to localise encoder bugs):
//
//   1. PYTHON: dump per-layer reference activations.
//      python tools/dump_reference.py --model MODEL --output ref.gguf
//
//   2. C++: run the encoder with dump mode (CRISPEMBED_DUMP_LAYERS=1)
//      to capture per-layer intermediates.
//
//   3. Compare each stage. The first layer where cos_min drops below
//      0.999 is where the bug lives.
//
// Adapted from CrispASR's crispasr_diff.h with the same API contract.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace crispembed_diff {

struct Report {
    bool found = false;
    size_t n_elem = 0;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float rms = 0.0f;
    float cos_min = 1.0f;
    float cos_mean = 1.0f;
    std::vector<int64_t> shape;

    bool is_pass(float cos_threshold = 0.999f) const {
        return found && cos_min >= cos_threshold;
    }
};

class Ref {
public:
    Ref() = default;
    ~Ref() { clear(); }

    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;

    bool load(const std::string& path);

    bool has(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    std::pair<const float*, size_t> get_f32(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return {nullptr, 0};
        return {it->second.data.data(), it->second.data.size()};
    }

    std::vector<int64_t> shape(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return {};
        return it->second.shape;
    }

    Report compare(const std::string& name, const float* data, size_t n_elem,
                   int row_dim = -1) const;

    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (auto& kv : tensors_) names.push_back(kv.first);
        return names;
    }

    std::string meta(const std::string& key) const {
        auto it = metadata_.find(key);
        return it != metadata_.end() ? it->second : "";
    }

private:
    struct TensorData {
        std::vector<float> data;
        std::vector<int64_t> shape;
    };

    std::unordered_map<std::string, TensorData> tensors_;
    std::unordered_map<std::string, std::string> metadata_;

    void clear() { tensors_.clear(); metadata_.clear(); }
};

// ── Implementation (header-only for easy integration) ────────────────

inline bool Ref::load(const std::string& path) {
    // Minimal GGUF reader — just enough to extract F32 tensors.
    // Uses the gguf C API from ggml.
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "crispembed_diff: cannot open %s\n", path.c_str());
        return false;
    }

    // Read GGUF magic + version
    uint32_t magic = 0;
    fread(&magic, 4, 1, f);
    if (magic != 0x46554747) { // "GGUF" little-endian
        fprintf(stderr, "crispembed_diff: not a GGUF file: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    uint32_t version = 0;
    fread(&version, 4, 1, f);
    if (version < 2 || version > 3) {
        fprintf(stderr, "crispembed_diff: unsupported GGUF version %u\n", version);
        fclose(f);
        return false;
    }

    uint64_t n_tensors = 0, n_kv = 0;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    // Helper: read a GGUF string
    auto read_string = [&]() -> std::string {
        uint64_t len = 0;
        fread(&len, 8, 1, f);
        std::string s(len, '\0');
        fread(&s[0], 1, len, f);
        return s;
    };

    // Skip KV pairs (store string values for metadata)
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string();
        uint32_t vtype = 0;
        fread(&vtype, 4, 1, f);
        switch (vtype) {
            case 0: { uint8_t v; fread(&v, 1, 1, f); break; }   // UINT8
            case 1: { int8_t v; fread(&v, 1, 1, f); break; }    // INT8
            case 2: { uint16_t v; fread(&v, 2, 1, f); break; }  // UINT16
            case 3: { int16_t v; fread(&v, 2, 1, f); break; }   // INT16
            case 4: { uint32_t v; fread(&v, 4, 1, f); break; }  // UINT32
            case 5: { int32_t v; fread(&v, 4, 1, f); break; }   // INT32
            case 6: { float v; fread(&v, 4, 1, f); break; }     // FLOAT32
            case 7: { uint8_t v; fread(&v, 1, 1, f); break; }   // BOOL
            case 8: {                                             // STRING
                std::string v = read_string();
                metadata_[key] = v;
                break;
            }
            case 9: {                                             // ARRAY
                uint32_t arr_type; fread(&arr_type, 4, 1, f);
                uint64_t arr_n; fread(&arr_n, 8, 1, f);
                // Skip array elements
                for (uint64_t j = 0; j < arr_n; j++) {
                    switch (arr_type) {
                        case 0: case 1: case 7: { uint8_t v; fread(&v, 1, 1, f); break; }
                        case 2: case 3: { uint16_t v; fread(&v, 2, 1, f); break; }
                        case 4: case 5: { uint32_t v; fread(&v, 4, 1, f); break; }
                        case 6: { float v; fread(&v, 4, 1, f); break; }
                        case 8: read_string(); break;
                        case 10: { uint64_t v; fread(&v, 8, 1, f); break; }  // UINT64
                        case 11: { int64_t v; fread(&v, 8, 1, f); break; }   // INT64
                        case 12: { double v; fread(&v, 8, 1, f); break; }    // FLOAT64
                        default: break;
                    }
                }
                break;
            }
            case 10: { uint64_t v; fread(&v, 8, 1, f); break; }  // UINT64
            case 11: { int64_t v; fread(&v, 8, 1, f); break; }   // INT64
            case 12: { double v; fread(&v, 8, 1, f); break; }    // FLOAT64
            default: break;
        }
    }

    // Read tensor info
    struct TensorInfo {
        std::string name;
        uint32_t n_dims;
        std::vector<int64_t> dims;
        uint32_t type;
        uint64_t offset;
    };
    std::vector<TensorInfo> infos;
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo ti;
        ti.name = read_string();
        fread(&ti.n_dims, 4, 1, f);
        ti.dims.resize(ti.n_dims);
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            fread(&ti.dims[d], 8, 1, f);
        }
        fread(&ti.type, 4, 1, f);
        fread(&ti.offset, 8, 1, f);
        infos.push_back(std::move(ti));
    }

    // Align to 32 bytes for tensor data
    long pos = ftell(f);
    long aligned = (pos + 31) & ~31L;
    fseek(f, aligned, SEEK_SET);
    long data_start = aligned;

    // Read tensor data
    for (auto& ti : infos) {
        size_t n_elem = 1;
        for (auto d : ti.dims) n_elem *= (size_t)d;

        fseek(f, data_start + (long)ti.offset, SEEK_SET);

        TensorData td;
        td.shape = ti.dims;

        if (ti.type == 0) {
            // F32
            td.data.resize(n_elem);
            fread(td.data.data(), sizeof(float), n_elem, f);
        } else if (ti.type == 5) {
            // I32 (token IDs) — convert to float for storage
            std::vector<int32_t> ibuf(n_elem);
            fread(ibuf.data(), sizeof(int32_t), n_elem, f);
            td.data.resize(n_elem);
            for (size_t j = 0; j < n_elem; j++) td.data[j] = (float)ibuf[j];
        } else {
            fprintf(stderr, "crispembed_diff: skipping tensor '%s' (type %u, not F32/I32)\n",
                    ti.name.c_str(), ti.type);
            continue;
        }

        tensors_[ti.name] = std::move(td);
    }

    fclose(f);
    fprintf(stderr, "crispembed_diff: loaded %zu tensors from %s\n",
            tensors_.size(), path.c_str());
    return true;
}

inline Report Ref::compare(const std::string& name, const float* data,
                           size_t n_elem, int row_dim) const {
    Report r;
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        fprintf(stderr, "crispembed_diff: tensor '%s' not found in archive\n", name.c_str());
        return r;
    }
    r.found = true;
    r.shape = it->second.shape;

    const float* ref = it->second.data.data();
    size_t ref_n = it->second.data.size();
    size_t cmp_n = std::min(n_elem, ref_n);
    r.n_elem = cmp_n;

    if (cmp_n == 0) return r;

    // Element-wise metrics
    double sum_abs = 0, sum_sq = 0;
    float max_a = 0;
    for (size_t i = 0; i < cmp_n; i++) {
        float d = std::fabs(data[i] - ref[i]);
        if (d > max_a) max_a = d;
        sum_abs += d;
        sum_sq += (double)d * d;
    }
    r.max_abs = max_a;
    r.mean_abs = (float)(sum_abs / cmp_n);
    r.rms = (float)std::sqrt(sum_sq / cmp_n);

    // Per-row cosine similarity
    // Determine row size from shape or row_dim
    size_t D = 1;
    if (row_dim >= 0 && row_dim < (int)r.shape.size()) {
        D = (size_t)r.shape[row_dim];
    } else if (!r.shape.empty()) {
        D = (size_t)r.shape.back();  // last dim
    }
    if (D == 0) D = 1;

    size_t n_rows = cmp_n / D;
    if (n_rows == 0) {
        n_rows = 1;
        D = cmp_n;
    }

    double cos_sum = 0;
    float cos_worst = 2.0f;
    for (size_t row = 0; row < n_rows; row++) {
        const float* a = data + row * D;
        const float* b = ref + row * D;
        double dot = 0, na = 0, nb = 0;
        for (size_t j = 0; j < D; j++) {
            dot += (double)a[j] * b[j];
            na += (double)a[j] * a[j];
            nb += (double)b[j] * b[j];
        }
        float cos = (na > 1e-18 && nb > 1e-18)
            ? (float)(dot / (std::sqrt(na) * std::sqrt(nb)))
            : 0.0f;
        cos_sum += cos;
        if (cos < cos_worst) cos_worst = cos;
    }
    r.cos_min = cos_worst;
    r.cos_mean = (float)(cos_sum / n_rows);

    return r;
}

} // namespace crispembed_diff
