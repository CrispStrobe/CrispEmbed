#pragma once
// New, bounded MoEdern1.5M self-contained F32 profile. All historical loaders
// retain their original architecture/count rules. No inference math here.
#include "gguf.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace moedern_packed {
inline bool file_bounds(const char * path, std::string & error) {
    try {
        if (!path) throw std::runtime_error("null packed path");
        struct reader {
            std::ifstream file;
            uint64_t size = 0, at = 0;
            std::vector<unsigned char> data;
            explicit reader(const char * path) : file(path, std::ios::binary | std::ios::ate) {
                if (!file || file.tellg() <= 0) throw std::runtime_error("packed file unavailable");
                size = uint64_t(file.tellg());
                if (size > uint64_t(16) * 1024 * 1024) throw std::runtime_error("packed file exceeds16MiB");
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
        if (r.integer(4) != 0x46554747 || r.integer(4) != 3 || r.integer(8) != 32)
            throw std::runtime_error("packed GGUFv3/32 tensors required");
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
        uint64_t total = 0;
        for (int i = 0; i < 32; ++i) {
            const auto name = r.string();
            if (name.empty() || name.size() > 256 || name.find('\0') != std::string::npos || !names.insert(name).second)
                throw std::runtime_error("invalid/duplicate packed tensor name");
            const uint64_t rank = r.integer(4);
            if (rank < 1 || rank > 4) throw std::runtime_error("packed tensor rank must be1..4");
            uint64_t elements = 1;
            for (uint64_t axis = 0; axis < rank; ++axis) {
                const auto n = r.integer(8);
                if (!n || n > uint64_t(1) << 31 || elements > uint64_t(2) * 1024 * 1024 * 1024 / n)
                    throw std::runtime_error("packed tensor dimensions exceed bounds");
                elements *= n;
            }
            if (r.integer(4) != 0) throw std::runtime_error("packed model requires F32 tensors");
            const auto offset = r.integer(8), bytes = elements * 4;
            if (offset % 32 || offset > r.size || bytes > r.size - offset)
                throw std::runtime_error("packed tensor offset/extent invalid");
            ranges.emplace_back(offset, offset + bytes);
            total += bytes;
        }
        if (total != uint64_t(1484448) * 4) throw std::runtime_error("packed parameter inventory differs");
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
inline bool metadata_valid(const gguf_context * g, std::string & error) {
    auto text = [&](const char * key, const char * expected) {
        const auto at = gguf_find_key(g, key);
        return at >= 0 && gguf_get_kv_type(g, at) == GGUF_TYPE_STRING &&
               std::strcmp(gguf_get_val_str(g, at), expected) == 0;
    };
    if (!g || gguf_get_n_tensors(g) != 32 || !text("general.architecture", "modernbert_hash_moe_v10") ||
        !text("general.license", "apache-2.0") || !text("moedern.packed.format", "crispembed-moedern-text-f32-v1") ||
        !text("moedern.packed.tokenizer_architecture", "moedern_unigram_tokenizer") ||
        !text("moedern.packed.model_sha256", "9f9792f0ae76430496ac93951cc09a0804468ba3f22b4314b303aea85dfbe8fa") ||
        !text("moedern.packed.tokenizer_sha256", "3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1") ||
        !text("moedern.packed.original_sha256", "b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f") ||
        !text("moedern.packed.composed_source", "1a053116908224cf1c9358d7d24c85ba36e82eff") ||
        !text("moedern.packed.composed_report_sha256",
              "98af43c6780bbd11b1ec9ec4f26c7b2e3a2348507072408408bf726b2e89482e")) {
        error = "MoEdern packed source/profile differs";
        return false;
    }
    return true;
}
} // namespace moedern_packed
