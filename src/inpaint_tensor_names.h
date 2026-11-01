// GGUF tensor names are limited to GGML_MAX_NAME-1 (63) bytes. Moebius and
// PixelHacker state names reach 89 bytes, so their converter stores every name
// of 64+ bytes under one deterministic abbreviation (applied in this order;
// tools/latent_inpaint_contract.py gguf_tensor_name is the Python twin).
// Shorter names, and every MI-GAN / VAE / LaMa name, are stored unchanged.
// Loaders look tensors up by gguf_name() and re-key them to original names.
#pragma once

#include <cstring>
#include <string>
#include <utility>

namespace inpaint_names {

inline std::string gguf_name(std::string name) {
    if (name.size() < 64) return name;
    static const std::pair<const char *, const char *> rules[] = {
        { "diff_model.", "dm." },   { ".transformer_blocks.", ".tb." },
        { ".attentions.", ".at." }, { "down_blocks.", "db." },
        { "up_blocks.", "ub." },    { "mid_block.", "mb." },
    };
    for (const auto & rule : rules)
        for (size_t at = name.find(rule.first); at != std::string::npos; at = name.find(rule.first, at))
            name.replace(at, std::strlen(rule.first), rule.second), at += std::strlen(rule.second);
    return name;
}

// Re-key a loaded tensor map entry from its stored alias to the original name.
template <typename Map> void restore_original_name(Map & tensors, const std::string & original) {
    const std::string alias = gguf_name(original);
    if (alias == original) return;
    const auto found = tensors.find(alias);
    if (found == tensors.end()) return;
    auto tensor = found->second;
    tensors.erase(found);
    tensors.emplace(original, tensor);
}

} // namespace inpaint_names
