// Preserve original checkpoint names while respecting ggml's 64-byte names.
#pragma once

#include "core/gguf_loader.h"
#include "core/sha256.h"
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace inpaint_weight_names {
inline constexpr const char * key = "crispembed.tensor_name_encoding";
inline constexpr const char * encoding = "sha256-56-v1";

inline bool encoded(const gguf_context * metadata) {
    const int64_t id = gguf_find_key(metadata, key);
    if (id < 0) return false;
    if (gguf_get_kv_type(metadata, id) != GGUF_TYPE_STRING || std::string(gguf_get_val_str(metadata, id)) != encoding)
        throw std::runtime_error("inpainting: unsupported tensor name encoding");
    return true;
}

inline std::string physical(const std::string & original, bool use_encoding = true) {
    // The format has a fixed UTF-8 byte boundary, independent of a future
    // backend's compiled GGML_MAX_NAME. Full source names stay in the schema.
    if (!use_encoding || original.size() < 64) return original;
    return "t." + core_sha256::bytes(original.data(), original.size()).substr(0, 56);
}

using Names = std::map<std::string, std::string>;
template <class Schema> Names map(const gguf_context * metadata, const Schema & schema) {
    const bool use_encoding = encoded(metadata);
    Names result;
    std::set<std::string> unique;
    for (const auto & item : schema) {
        const std::string name = physical(item.first, use_encoding);
        if (!unique.insert(name).second) throw std::runtime_error("inpainting: tensor name alias collision");
        result.emplace(item.first, name);
    }
    return result;
}

inline void restore(core_gguf::WeightLoad & loaded, const Names & names) {
    if (loaded.tensors.size() != names.size()) throw std::runtime_error("inpainting: tensor name census differs");
    decltype(loaded.tensors) canonical;
    for (const auto & item : names) {
        const auto found = loaded.tensors.find(item.second);
        if (found == loaded.tensors.end()) throw std::runtime_error("inpainting: missing encoded tensor " + item.first);
        canonical.emplace(item.first, found->second);
    }
    loaded.tensors.swap(canonical);
}
} // namespace inpaint_weight_names
