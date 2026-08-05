// imatrix_alias.h — the naming contract between the runtime's pre-merged
// BERT QKV tensor and the quantizer's imatrix lookup (F7, T19-E3 defect 2).
//
// BERT-family runtimes merge the per-layer attn q/k/v weights into one F32
// tensor at load time, so the imatrix collector only ever sees that merged
// matmul. Its input IS the shared QKV input (width n_embd), which makes the
// collected vector the correct importance for all three per-weight tensors —
// PROVIDED the two sides agree on the merged tensor's name. Both sides
// (src/crispembed.cpp names the tensor; tools/quantize.cpp resolves the
// alias) take the name from here, and tests/test_imatrix_alias.cpp guards
// the mapping hermetically: a silent mismatch reverts to the shipped defect
// where every attn.{q,k,v}.weight quantizes with NO importance.
#pragma once

#include <cstdio>
#include <cstring>
#include <string>

namespace core_imatrix {

// Canonical name of the runtime's merged QKV tensor for encoder layer `i`.
// Must never collide with a real GGUF tensor name — loaders read
// "attn.qkv.weight" / "attn_qkv.weight", never "qkv_merged".
inline std::string qkv_merged_name(int layer) {
    char buf[64];
    snprintf(buf, sizeof(buf), "enc.%d.attn.qkv_merged.weight", layer);
    return buf;
}

// Map a per-layer attention q/k/v weight name (either GGUF naming scheme:
// CrispEmbed "enc.<N>.attn.q.weight" or community "blk.<N>.attn_q.weight")
// to the merged collector name. Returns "" when `sname` is not a per-layer
// attention q/k/v weight (decoder "dec.*" tensors included: the decoder path
// never pre-merges, so an alias there would apply a wrong-provenance vector).
inline std::string qkv_merged_alias(const std::string & sname) {
    static const char * tails[] = {
        ".attn.q.weight", ".attn.k.weight", ".attn.v.weight", // CrispEmbed: enc.<N>.*
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", // community:  blk.<N>.*
    };
    for (const char * tail : tails) {
        const size_t tl = strlen(tail);
        if (sname.size() <= tl || sname.compare(sname.size() - tl, tl, tail) != 0) continue;
        const std::string head = sname.substr(0, sname.size() - tl); // "enc.<N>" / "blk.<N>"
        const size_t dot = head.find('.');
        if (dot == std::string::npos) return "";
        const std::string base = head.substr(0, dot);
        if (base != "enc" && base != "blk") return "";
        const std::string idx = head.substr(dot + 1);
        if (idx.empty() || idx.find_first_not_of("0123456789") != std::string::npos) return "";
        return "enc." + idx + ".attn.qkv_merged.weight";
    }
    return "";
}

} // namespace core_imatrix
