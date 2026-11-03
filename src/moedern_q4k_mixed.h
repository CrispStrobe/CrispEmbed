#pragma once
// Opt-in MIXED padded-Q4_K / Q8_0 / F32 MoEdern profile (portfolio Q4_K lane,
// docs/moedern-q4k-mixed.md). Selected only when CRISPEMBED_MOEDERN_Q4_K_MIXED
// is ON together with the padded Q4_K gates; the plain padded Q4_K profile in
// moedern_q4k_profile.h stays byte-identical and strict. A mixed file must:
//   - carry the explicit recipe `q4_k_padded_mixed_v1` and a named policy;
//   - declare every Q4_K tensor (zero-padded storage columns, as plain padded Q4_K);
//   - store every other matmul weight as Q8_0 (32-aligned rows) or F32;
//   - keep embeddings, router, norms, conv kernels and dense head F32.
#include "moedern_q4k_profile.h"
namespace moedern_q4k_mixed {
constexpr const char * RECIPE = "q4_k_padded_mixed_v1";
inline const moedern_q4k::pin * metadata(const gguf_context * g, std::string & error) {
    using moedern_q4k::text;
    const auto policy = gguf_find_key(g, "moedern.q4k.policy");
    const bool named = policy >= 0 && gguf_get_kv_type(g, policy) == GGUF_TYPE_STRING &&
                       std::strlen(gguf_get_val_str(g, policy)) > 0 && std::strlen(gguf_get_val_str(g, policy)) <= 64;
    for (const auto & p : moedern_q4k::pins())
        if (named && moedern_q4k::storage_metadata(g) && gguf_get_n_tensors(g) == int64_t(p.tensors) &&
            text(g, "general.architecture", "modernbert_hash_moe_v10") && text(g, "general.license", "apache-2.0") &&
            text(g, "modernbert_hash_moe_v10.source_revision", p.revision) &&
            text(g, "moedern.packed.format", "crispembed-moedern-q4k-padded-v1") &&
            text(g, "moedern.q4k.format", "q4_k-padded-columns-v1") &&
            text(g, "modernbert_hash_moe_v10.quant_recipe", RECIPE) && text(g, "moedern.q4k.variant", p.variant) &&
            text(g, "moedern.q4k.source_sha256", p.source_sha256) &&
            text(g, "moedern.q4k.source_format", p.source_format) &&
            text(g, "moedern.packed.model_sha256", p.model_sha256) &&
            text(g, "moedern.packed.original_sha256", p.original_report_sha256) &&
            text(g, "moedern.packed.tokenizer_architecture", "moedern_unigram_tokenizer") &&
            text(g, "moedern.packed.tokenizer_sha256",
                 "3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1"))
            return &p;
    error = "MoEdern mixed Q4_K source/profile/policy differs";
    return nullptr;
}
inline bool file_bounds(const char * path, std::string & error) {
    try {
        if (ggml_blck_size(GGML_TYPE_Q4_K) != 256 || ggml_type_size(GGML_TYPE_Q4_K) != 144 ||
            ggml_blck_size(GGML_TYPE_Q8_0) != 32 || ggml_type_size(GGML_TYPE_Q8_0) != 34)
            throw std::runtime_error("Q4_K/Q8_0 storage ABI differs");
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
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q8_0)
                throw std::runtime_error("mixed Q4_K/Q8_0/F32 only");
            if (type == GGML_TYPE_Q4_K && columns % 256) throw std::runtime_error("Q4_K rows require256 columns");
            if (type == GGML_TYPE_Q8_0 && columns % 32) throw std::runtime_error("Q8_0 rows require32 columns");
            if (type == GGML_TYPE_Q4_K) ++q4_count;
            const auto offset = r.integer(8), bytes = type == GGML_TYPE_F32    ? elements * 4
                                                      : type == GGML_TYPE_Q4_K ? (elements / 256) * 144
                                                                               : (elements / 32) * 34;
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
} // namespace moedern_q4k_mixed
