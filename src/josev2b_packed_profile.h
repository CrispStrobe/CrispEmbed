#pragma once
#include "core/gguf_file_type.h"
// Strict new packed profile. Historical component loaders keep their own
// architecture/tensor-count guards; this is never an alternative for them.
#include "gguf.h"
#include <cstring>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace josev2b_packed {
constexpr const char * architecture = "josev2b_systemone";
inline bool text(const gguf_context * g, const char * name, const char * expected) {
    const auto id = gguf_find_key(g, name);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
           std::strcmp(gguf_get_val_str(g, id), expected) == 0;
}
inline bool number(const gguf_context * g, const char * name, uint32_t expected) {
    const auto id = gguf_find_key(g, name);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32 && gguf_get_val_u32(g, id) == expected;
}
// Inspect bounded raw header before GGUF can allocate metadata arrays. Prefetch
// can include up to64KiB after the header; only up to31 tail bytes are checked.
inline bool projection(const std::string & name) {
    if (name.rfind("l.layers.", 0) != 0) return false;
    for (const char * suffix :
         { ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight", ".self_attn.q_proj.weight",
           ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
           ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight" }) {
        const size_t n = std::strlen(suffix);
        if (name.size() > n && name.compare(name.size() - n, n, suffix) == 0) return true;
    }
    return false;
}
inline bool quantized(const gguf_context * g) {
    return g && (text(g, "josev2b.packed.format", "crispembed-josev-packed-q4k-v1") ||
                 text(g, "josev2b.packed.format", "crispembed-josev-packed-mixed-v2"));
}
inline bool file_bounds(const char * path, std::string & error, bool allow_quantized = false) {
    try {
        if (!path) throw std::runtime_error("null packed path");
        struct reader {
            std::ifstream file;
            uint64_t size = 0, at = 0;
            std::vector<unsigned char> data;
            explicit reader(const char * path) : file(path, std::ios::binary | std::ios::ate) {
                if (!file || file.tellg() <= 0) throw std::runtime_error("packed file unavailable");
                size = uint64_t(file.tellg());
                if (size > uint64_t(8) * 1024 * 1024 * 1024) throw std::runtime_error("packed file exceeds8GiB");
                file.seekg(0);
            }
            void ensure(uint64_t n) {
                const uint64_t cap = std::min(size, uint64_t(64) * 1024 * 1024);
                if (at > cap || n > cap - at) throw std::runtime_error("packed header exceeds64MiB/file");
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
                if (element > 12 || element == 9 || count > 1048576)
                    throw std::runtime_error("packed array type/count exceeds bounds");
                if (element == 8) {
                    for (uint64_t i = 0; i < count; ++i) string(false);
                } else
                    skip(count * sizes[element]);
            }
        } r(path);
        if (r.integer(4) != 0x46554747 || r.integer(4) != 3 || r.integer(8) != 442)
            throw std::runtime_error("packed GGUFv3/442 tensors required");
        const uint64_t count = r.integer(8);
        if (count > 4096) throw std::runtime_error("packed metadata count exceeds4096");
        bool mixed = false, mixed_v2 = false;
        uint32_t file_type = UINT32_MAX;
        std::set<std::string> keys;
        for (uint64_t i = 0; i < count; ++i) {
            const auto name = r.string();
            if (name.empty() || name.size() > 256 || name.find('\0') != std::string::npos || !keys.insert(name).second)
                throw std::runtime_error("invalid/duplicate packed metadata key");
            const uint32_t kind = uint32_t(r.integer(4));
            if (name == "general.alignment") {
                if (kind != 4 || r.integer(4) != 32) throw std::runtime_error("packed alignment must be32");
            } else if (name == "josev2b.packed.format") {
                if (kind != 8) throw std::runtime_error("packed format must be string");
                const auto format = r.string();
                mixed_v2 = format == "crispembed-josev-packed-mixed-v2";
                mixed = mixed_v2 || format == "crispembed-josev-packed-q4k-v1";
            } else if (name == "general.file_type") {
                if (kind != 4) throw std::runtime_error("packed file type must be uint32");
                file_type = uint32_t(r.integer(4));
            } else
                r.value(kind);
        }
        if (mixed && (!allow_quantized || !(core_gguf_file_type::known_q4_k(file_type) ||
                                            (mixed_v2 && file_type == core_gguf_file_type::q8_0))))
            throw std::runtime_error("Josev mixed quantized profile gate/type rejected");
        uint64_t actual_q4 = 0, actual_q8 = 0;
        std::set<std::string> names;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        uint64_t total = 0;
        for (int i = 0; i < 442; ++i) {
            const auto name = r.string();
            if (name.empty() || name.size() > 256 || name.find('\0') != std::string::npos || !names.insert(name).second)
                throw std::runtime_error("invalid/duplicate packed tensor name");
            const uint64_t rank = r.integer(4);
            if (rank < 1 || rank > 4) throw std::runtime_error("packed tensor rank must be1..4");
            uint64_t elements = 1, width = 0;
            for (uint64_t axis = 0; axis < rank; ++axis) {
                const auto n = r.integer(8);
                if (!n || n > uint64_t(1) << 31 || elements > uint64_t(2) * 1024 * 1024 * 1024 / n)
                    throw std::runtime_error("packed tensor dimensions exceed bounds");
                if (axis == 0) width = n;
                elements *= n;
            }
            const auto type = r.integer(4);
            uint64_t bytes = elements * 4;
            if (type != GGML_TYPE_F32) {
                if (!mixed || !projection(name) || rank != 2)
                    throw std::runtime_error("packed critical/legacy tensor requires F32");
                if (type == GGML_TYPE_F16)
                    bytes = elements * 2;
                else if (type == GGML_TYPE_Q8_0 && width % 32 == 0) {
                    bytes = elements / 32 * 34;
                    ++actual_q8;
                } else if (type == GGML_TYPE_Q4_K && width % 256 == 0) {
                    bytes = elements / 256 * 144;
                    ++actual_q4;
                } else
                    throw std::runtime_error("unsupported packed projection storage");
            }
            const auto offset = r.integer(8);
            if (offset % 32 || offset > r.size || bytes > r.size - offset)
                throw std::runtime_error("packed tensor offset/extent invalid");
            ranges.emplace_back(offset, offset + bytes);
            total += elements;
        }
        if (mixed && !core_gguf_file_type::matches_quantized_payload(file_type, actual_q8, actual_q4, mixed_v2))
            throw std::runtime_error("Quantized header requires matching actual Q8_0/Q4_K storage");
        if (total != uint64_t(1881825088 + 30472708)) throw std::runtime_error("packed parameter inventory differs");
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
inline bool metadata_valid(const gguf_context * g, std::string & error, bool allow_quantized = false) {
    if (!g || gguf_get_n_tensors(g) != 442 || !text(g, "general.architecture", architecture) ||
        !text(g, "general.license", "apache-2.0") ||
        !(text(g, "josev2b.packed.format", "crispembed-josev-packed-f32-v1") ||
          (allow_quantized && quantized(g) &&
           (number(g, "general.file_type", core_gguf_file_type::q4_k_s) ||
            number(g, "general.file_type", core_gguf_file_type::legacy_q4_k) ||
            (text(g, "josev2b.packed.format", "crispembed-josev-packed-mixed-v2") &&
             number(g, "general.file_type", core_gguf_file_type::q8_0))))) ||
        !text(g, "josev2b.packed.profile", "official-pytorch-text-batch1-512-no-extra-temperature") ||
        !text(g, "josev2b.packed.backbone_gguf_sha256",
              "8dc144a89e4b2bbf253fe395eeb9bc4d1eb7b3a27474442847535e55c79b361c") ||
        !text(g, "josev2b.packed.head_gguf_sha256",
              "5b0a4bfe769526569102e3cde115887593875124c94bb723fb9087bc40c29507") ||
        !text(g, "josev2b.packed.tokenizer_gguf_sha256",
              "6fe488e538b172d2f3f831e7422714b7158602d9d9f2d34f336db389363c9562") ||
        !text(g, "josev2b_backbone_reference.original_receipt_sha256",
              "51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090") ||
        !text(g, "josev2b_backbone_reference.original_worker_source", "b2edf8c83c9aca1f8a9210198d47798ac13828b8") ||
        !number(g, "josev2b_backbone_reference.hidden_size", 2048) ||
        !number(g, "josev2b_backbone_reference.intermediate_size", 6144) ||
        !number(g, "josev2b_backbone_reference.layers", 24) ||
        !number(g, "josev2b_backbone_reference.max_validated_tokens", 512) ||
        !text(g, "josev2b_head_reference.source_revision", "6f0a31c8ed683aba79097eb3f48df2c9ab560afd") ||
        !text(g, "josev2b_head_reference.checkpoint_sha256",
              "8816d51356876ef9b055e51291a00dfc4a8c741189ca088b59604d039d7852c5") ||
        !text(g, "josev2b_head_reference.original_receipt_sha256",
              "51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090") ||
        !text(g, "josev2b_head_reference.original_worker_source", "b2edf8c83c9aca1f8a9210198d47798ac13828b8") ||
        !text(g, "josev2b_head_reference.calibration_profile", "official-pytorch-no-extra-temperature") ||
        !number(g, "josev2b_head_reference.hidden_size", 2048) || !number(g, "josev2b_head_reference.width", 512) ||
        !number(g, "josev2b_head_reference.routing_layers", 2) || !number(g, "josev2b_head_reference.layers", 4) ||
        !number(g, "josev2b_head_reference.heads", 8) || !number(g, "josev2b_head_reference.feedforward", 2048)) {
        error = "Josev packed source/profile/configuration differs";
        return false;
    }
    return true;
}
} // namespace josev2b_packed
