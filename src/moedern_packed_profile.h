#pragma once
// Bounded self-contained F32 packed profiles: MoEdern1.5M (default) and the
// gated 27M/115M family profiles. All historical loaders
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
// Per-profile structural bounds. The default is the unchanged 1.5M profile; the
// family profiles (27M/115M) are only reachable behind CRISPEMBED_MOEDERN_FAMILY.
struct limits {
    uint64_t max_file_bytes, tensors, parameters;
    const char * file_message;
};
inline const limits & one_five_limits() {
    static const limits l = { uint64_t(16) * 1024 * 1024, 32, 1484448, "packed file exceeds16MiB" };
    return l;
}
struct family_pin {
    const char * variant, * revision, * model_sha256, * original_report_sha256;
    limits bounds;
};
// Model pins are the frozen-original F32 conversions (tests/results/moedern-colbert).
inline const std::vector<family_pin> & family_pins() {
    static const std::vector<family_pin> pins = {
        { "27M", "d2b7b1feaaac1bf87510ca9525208c3daeabcc5a",
          "ffbf5eb4ce1192686f3782bbb1dcfc3dfc7a7a4651660337beecd7b6208555e9",
          "00472cb369e935dd37cccf89b7027b12489de1586b916963d0ae560059de30d3",
          { uint64_t(128) * 1024 * 1024, 32, 27124000, "packed file exceeds128MiB" } },
        { "115M", "66380d2ade3b5027bba83bdcdba052d0bb86dcdb",
          "2d87f0fe73eaf01eb4271f3d3d0f49434b31c73308dfc22b6c307ce59d4c512e",
          "01c8a0e4886eae107103845847cb5d4675752b91a987e23ad26e7a1e98e44af6",
          { uint64_t(512) * 1024 * 1024, 94, 114696576, "packed file exceeds512MiB" } },
    };
    return pins;
}
inline bool file_bounds(const char * path, std::string & error, const limits & bound = one_five_limits()) {
    try {
        if (!path) throw std::runtime_error("null packed path");
        struct reader {
            std::ifstream file;
            uint64_t size = 0, at = 0;
            std::vector<unsigned char> data;
            explicit reader(const char * path, const limits & bound) : file(path, std::ios::binary | std::ios::ate) {
                if (!file || file.tellg() <= 0) throw std::runtime_error("packed file unavailable");
                size = uint64_t(file.tellg());
                if (size > bound.max_file_bytes) throw std::runtime_error(bound.file_message);
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
        } r(path, bound);
        if (r.integer(4) != 0x46554747 || r.integer(4) != 3 || r.integer(8) != bound.tensors)
            throw std::runtime_error(bound.tensors == 32 ? "packed GGUFv3/32 tensors required"
                                                         : "packed GGUFv3/profile tensor count required");
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
        for (uint64_t i = 0; i < bound.tensors; ++i) {
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
        if (total != bound.parameters * 4) throw std::runtime_error("packed parameter inventory differs");
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
// Family (27M/115M) packed profile; never accepts the 1.5M revision, which keeps
// metadata_valid above.
inline bool family_metadata_valid(const gguf_context * g, const family_pin & pin, std::string & error) {
    auto text = [&](const char * key, const char * expected) {
        const auto at = gguf_find_key(g, key);
        return at >= 0 && gguf_get_kv_type(g, at) == GGUF_TYPE_STRING &&
               std::strcmp(gguf_get_val_str(g, at), expected) == 0;
    };
    if (!g || gguf_get_n_tensors(g) != int64_t(pin.bounds.tensors) ||
        !text("general.architecture", "modernbert_hash_moe_v10") || !text("general.license", "apache-2.0") ||
        !text("modernbert_hash_moe_v10.source_revision", pin.revision) ||
        !text("moedern.packed.format", "crispembed-moedern-family-f32-v1") ||
        !text("moedern.packed.variant", pin.variant) ||
        !text("moedern.packed.tokenizer_architecture", "moedern_unigram_tokenizer") ||
        !text("moedern.packed.model_sha256", pin.model_sha256) ||
        !text("moedern.packed.tokenizer_sha256", "3ce7c3fa8b10249c5b704026566d183adb3de7b1879e7174e4654cdfd54ac4b1") ||
        !text("moedern.packed.original_sha256", pin.original_report_sha256)) {
        error = "MoEdern family packed source/profile differs";
        return false;
    }
    return true;
}
} // namespace moedern_packed
