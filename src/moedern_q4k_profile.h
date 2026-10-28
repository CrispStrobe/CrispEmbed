#pragma once
// Separate opt-in padded Q4_K profile. Old F32 raw bounds/loaders stay strict.
#include "gguf.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace moedern_q4k {
inline int64_t storage_columns(int64_t semantic) {
    return ((semantic + 255) / 256) * 256;
}
inline bool compresses(int64_t semantic) {
    return semantic > 0 && storage_columns(semantic) / 256 * 144 < semantic * 4;
}
// Shared by the production matmul/id wrappers and hosted authored controls.
inline ggml_tensor * pad_input(ggml_context * g, ggml_tensor * input, int64_t logical, int64_t stored) {
    if (!input || input->type != GGML_TYPE_F32 || logical <= 0 || logical > 4096 || input->ne[0] != logical ||
        stored != storage_columns(logical))
        throw std::runtime_error("Q4_K logical activation/padding differs");
    return stored == logical ? input : ggml_pad(g, input, int(stored - logical), 0, 0, 0);
}
struct pin {
    const char *variant, *revision, *source_sha256, *source_format, *model_sha256, *original_report_sha256;
    uint64_t tensors, parameters;
};
inline const std::vector<pin> & pins() {
    static const std::vector<pin> p = {
        { "1.5M", "9d360b8b88ae11bea5784bf9012ff509fdd5c2da",
          "0e2b8773abd8c78a5825280f9a73cb808b88d56f0675a8e4bb7f0509f5b6cc51", "crispembed-moedern-text-f32-v1",
          "9f9792f0ae76430496ac93951cc09a0804468ba3f22b4314b303aea85dfbe8fa",
          "b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f", 32, 1484448 },
        { "27M", "d2b7b1feaaac1bf87510ca9525208c3daeabcc5a",
          "e836acbb507ebf9d26ad6a3f5e5712487f1b90f1430b1174acb07de07e85ce18", "crispembed-moedern-family-f32-v1",
          "ffbf5eb4ce1192686f3782bbb1dcfc3dfc7a7a4651660337beecd7b6208555e9",
          "00472cb369e935dd37cccf89b7027b12489de1586b916963d0ae560059de30d3", 32, 27124000 },
        { "115M", "66380d2ade3b5027bba83bdcdba052d0bb86dcdb",
          "4150412f66caaa058c90ad0ed6cc4e318a0ea146f4ce9e59de62f83975903766", "crispembed-moedern-family-f32-v1",
          "2d87f0fe73eaf01eb4271f3d3d0f49434b31c73308dfc22b6c307ce59d4c512e",
          "01c8a0e4886eae107103845847cb5d4675752b91a987e23ad26e7a1e98e44af6", 94, 114696576 }
    };
    return p;
}
inline bool text(const gguf_context * g, const char * key, const char * expected) {
    const auto k = gguf_find_key(g, key);
    return k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING && std::strcmp(gguf_get_val_str(g, k), expected) == 0;
}
inline const pin * metadata(const gguf_context * g, std::string & error) {
    for (const auto & p : pins())
        if (g && gguf_get_n_tensors(g) == int64_t(p.tensors) &&
            text(g, "general.architecture", "modernbert_hash_moe_v10") && text(g, "general.license", "apache-2.0") &&
            text(g, "modernbert_hash_moe_v10.source_revision", p.revision) &&
            text(g, "moedern.packed.format", "crispembed-moedern-q4k-padded-v1") &&
            text(g, "moedern.q4k.format", "q4_k-padded-columns-v1") &&
            text(g, "modernbert_hash_moe_v10.quant_recipe", "q4_k_padded_f32_critical") &&
            text(g, "moedern.q4k.variant", p.variant) && text(g, "moedern.q4k.source_sha256", p.source_sha256) &&
            text(g, "moedern.q4k.source_format", p.source_format) &&
            text(g, "moedern.packed.model_sha256", p.model_sha256) &&
            text(g, "moedern.packed.original_sha256", p.original_report_sha256) &&
            text(g, "moedern.packed.tokenizer_architecture", "moedern_unigram_tokenizer") &&
            text(g, "moedern.packed.tokenizer_sha256",
                 "3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1"))
            return &p;
    error = "MoEdern padded Q4_K source/profile differs";
    return nullptr;
}
struct columns {
    int64_t logical, stored;
};
inline std::map<std::string, columns> declarations(const gguf_context * g) {
    auto fail = [](bool ok) {
        if (!ok) throw std::runtime_error("Q4_K column declarations differ");
    };
    const auto names = gguf_find_key(g, "moedern.q4k.names"), logical = gguf_find_key(g, "moedern.q4k.logical_columns"),
               stored = gguf_find_key(g, "moedern.q4k.storage_columns");
    for (auto k : { names, logical, stored }) fail(k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_ARRAY);
    fail(gguf_get_arr_type(g, names) == GGUF_TYPE_STRING && gguf_get_arr_type(g, logical) == GGUF_TYPE_INT32 &&
         gguf_get_arr_type(g, stored) == GGUF_TYPE_INT32);
    const auto n = gguf_get_arr_n(g, names);
    fail(n > 0 && n <= 94 && gguf_get_arr_n(g, logical) == n && gguf_get_arr_n(g, stored) == n);
    const auto * a = static_cast<const int32_t *>(gguf_get_arr_data(g, logical));
    const auto * b = static_cast<const int32_t *>(gguf_get_arr_data(g, stored));
    std::map<std::string, columns> result;
    for (uint64_t i = 0; i < n; ++i) {
        fail(a[i] > 0 && a[i] <= 4096 && b[i] == storage_columns(a[i]) && compresses(a[i]));
        fail(result.emplace(gguf_get_arr_str(g, names, i), columns{ a[i], b[i] }).second);
    }
    return result;
}
inline bool file_bounds(const char * path, std::string & error) {
    try {
        if (ggml_blck_size(GGML_TYPE_Q4_K) != 256 || ggml_type_size(GGML_TYPE_Q4_K) != 144)
            throw std::runtime_error("Q4_K storage ABI differs");
        if (!path) throw std::runtime_error("null packed path");
        struct reader {
            std::ifstream file;
            uint64_t size = 0, at = 0;
            std::vector<unsigned char> data;
            explicit reader(const char * path) : file(path, std::ios::binary | std::ios::ate) {
                if (!file || file.tellg() <= 0) throw std::runtime_error("packed file unavailable");
                size = uint64_t(file.tellg());
                if (size > uint64_t(512) * 1024 * 1024) throw std::runtime_error("Q4_K packed file exceeds512MiB");
                file.seekg(0);
            }
            void ensure(uint64_t n) {
                const uint64_t cap = std::min(size, uint64_t(4) * 1024 * 1024);
                if (at > cap || n > cap - at) throw std::runtime_error("packed header exceeds4MiB/file");
                if (data.size() < at + n) {
                    const size_t end = size_t(std::min(cap, ((at + n + 65535) / 65536) * 65536));
                    const size_t before = data.size();
                    data.resize(end);
                    file.read(reinterpret_cast<char *>(data.data() + before), end - before);
                    if (size_t(file.gcount()) != end - before) throw std::runtime_error("truncated packed header");
                }
            }
            uint64_t integer(unsigned bytes) {
                ensure(bytes);
                uint64_t value = 0;
                for (unsigned i = 0; i < bytes; ++i) value |= uint64_t(data[size_t(at++)]) << (8 * i);
                return value;
            }
            void skip(uint64_t n) {
                ensure(n);
                at += n;
            }
            std::string string(bool keep = true) {
                const auto n = integer(8);
                if (n > 1024 * 1024) throw std::runtime_error("packed string exceeds1MiB");
                ensure(n);
                std::string value;
                if (keep) value.assign(reinterpret_cast<const char *>(data.data() + at), size_t(n));
                at += n;
                return value;
            }
            void value(uint32_t kind) {
                static constexpr unsigned sizes[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
                if (kind > 12) throw std::runtime_error("unknown packed metadata type");
                if (kind == 8) {
                    string(false);
                    return;
                }
                if (kind != 9) {
                    skip(sizes[kind]);
                    return;
                }
                const uint32_t element = uint32_t(integer(4));
                const uint64_t count = integer(8);
                if (element > 12 || element == 9 || count > 65536)
                    throw std::runtime_error("packed array type/count exceeds bounds");
                if (element == 8) {
                    for (uint64_t i = 0; i < count; ++i) string(false);
                } else
                    skip(count * sizes[element]);
            }
        } r(path);
        if (r.integer(4) != 0x46554747 || r.integer(4) != 3) throw std::runtime_error("Q4_K GGUFv3 required");
        const uint64_t n_tensors = r.integer(8);
        if (n_tensors != 32 && n_tensors != 94) throw std::runtime_error("Q4_K tensor count differs");
        const uint64_t count = r.integer(8);
        if (count > 128) throw std::runtime_error("packed metadata count exceeds128");
        std::set<std::string> keys;
        for (uint64_t i = 0; i < count; ++i) {
            const auto name = r.string();
            if (name.empty() || name.size() > 256 || name.find('\0') != std::string::npos || !keys.insert(name).second)
                throw std::runtime_error("invalid/duplicate packed metadata key");
            const uint32_t kind = uint32_t(r.integer(4));
            if (name == "general.alignment") {
                if (kind != 4 || r.integer(4) != 32) throw std::runtime_error("packed alignment must be32");
            } else
                r.value(kind);
        }
        std::set<std::string> names;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        uint64_t q4_count = 0;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            const auto name = r.string();
            if (name.empty() || name.size() > 256 || name.find('\0') != std::string::npos || !names.insert(name).second)
                throw std::runtime_error("invalid/duplicate packed tensor name");
            const uint64_t rank = r.integer(4);
            if (rank < 1 || rank > 4) throw std::runtime_error("packed tensor rank must be1..4");
            uint64_t elements = 1, columns = 0;
            for (uint64_t axis = 0; axis < rank; ++axis) {
                const auto n = r.integer(8);
                if (!n || n > uint64_t(1) << 31 || elements > uint64_t(2) * 1024 * 1024 * 1024 / n)
                    throw std::runtime_error("packed tensor dimensions exceed bounds");
                elements *= n;
                if (axis == 0) columns = n;
            }
            const auto type = r.integer(4);
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_Q4_K) throw std::runtime_error("Q4_K/F32 only");
            if (type == GGML_TYPE_Q4_K && columns % 256) throw std::runtime_error("Q4_K rows require256 columns");
            if (type == GGML_TYPE_Q4_K) ++q4_count;
            const auto offset = r.integer(8), bytes = type == GGML_TYPE_F32 ? elements * 4 : (elements / 256) * 144;
            if (offset % 32 || offset > r.size || bytes > r.size - offset)
                throw std::runtime_error("packed tensor offset/extent invalid");
            ranges.emplace_back(offset, offset + bytes);
        }
        if (!q4_count) throw std::runtime_error("actual Q4_K tensors required");
        const auto start = (r.at + 31) / 32 * 32;
        const auto padding = start - r.at;
        r.ensure(padding);
        for (uint64_t i = 0; i < padding; ++i)
            if (r.data[size_t(r.at + i)] != 0) throw std::runtime_error("nonzero packed header padding");
        std::sort(ranges.begin(), ranges.end());
        uint64_t end = 0;
        for (const auto & span : ranges) {
            if (span.first != (end + 31) / 32 * 32 || start > r.size || span.second > r.size - start)
                throw std::runtime_error("overlapping/gapped/truncated packed tensors");
            const auto gap = span.first - end;
            if (gap) {
                unsigned char bytes[31];
                r.file.clear();
                r.file.seekg(std::streamoff(start + end));
                r.file.read(reinterpret_cast<char *>(bytes), std::streamsize(gap));
                if (uint64_t(r.file.gcount()) != gap) throw std::runtime_error("truncated packed tensor padding");
                for (uint64_t i = 0; i < gap; ++i)
                    if (bytes[i]) throw std::runtime_error("nonzero packed tensor padding");
            }
            end = span.second;
        }
        const auto tail = r.size - start - end;
        if (tail > 31) throw std::runtime_error("unexpected packed trailing bytes");
        if (tail) {
            unsigned char bytes[31];
            r.file.clear();
            r.file.seekg(std::streamoff(start + end));
            r.file.read(reinterpret_cast<char *>(bytes), std::streamsize(tail));
            if (uint64_t(r.file.gcount()) != tail) throw std::runtime_error("truncated packed tail");
            for (uint64_t i = 0; i < tail; ++i)
                if (bytes[i]) throw std::runtime_error("nonzero packed tail padding");
        }
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
} // namespace moedern_q4k
