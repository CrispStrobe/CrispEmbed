#pragma once
// Tool-only closed native-HF recipe. Never an upstream byte-layout/import alias.
#include "lighton4b_imatrix.h"
#include "sha256.h"
#include <cstdio>
#include <set>
namespace lighton4b_imatrix_q4km {
inline constexpr const char * recipe = "lighton4b-weighted-q4km-style-q6head-v1";
inline constexpr const char * source_sha = "757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606";
inline constexpr const char * names_sha = "0ebc0b6104238b46de946f940f61cdc90e04a0eb187553026714fc1270cdb44c";
inline bool ends(const std::string & name, const char * suffix) {
    const std::string value(suffix);
    return name.size() >= value.size() && name.compare(name.size()-value.size(),value.size(),value)==0;
}
inline std::set<std::string> q6_names() {
    std::set<std::string> names{"l.embed_tokens.weight"};
    for (const auto & row : lighton4b_imatrix::expected()) {
        int layer = -1;
        if (std::sscanf(row.first.c_str(), "l.layers.%d.", &layer) != 1 || layer < 0 || layer >= 32) continue;
        const bool selected = layer < 4 || layer >= 28 || (layer-4)%3 == 2;
        if (selected && (ends(row.first,".linear_attn.in_proj_qkv.weight") ||
                         ends(row.first,".self_attn.v_proj.weight") || ends(row.first,".mlp.down_proj.weight")))
            names.insert(row.first);
    }
    return names;
}
inline bool manifest_valid() {
    const auto names=q6_names(); std::string raw;
    for (const auto & name:names) raw+=name+"\n";
    return names.size()==33 && core_sha256::bytes(raw.data(),raw.size())==names_sha;
}
inline ggml_type type(const std::string & name) {
    const auto names=q6_names();
    if (names.count(name)) return GGML_TYPE_Q6_K;
    return lighton4b_imatrix::expected().count(name) ? GGML_TYPE_Q4_K : GGML_TYPE_F32;
}
} // namespace lighton4b_imatrix_q4km
