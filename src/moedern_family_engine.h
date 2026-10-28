#pragma once
// Size-parameterised MoEdern ColBERT engine (1.5M, 27M, 115M); gated by
// CRISPEMBED_MOEDERN_FAMILY and CRISPEMBED_MOEDERN_COLBERT. Internal C++ API.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace moedern_family {
struct profile {
    const char * name = "";
    int hidden = 0, heads = 0, kv_heads = 0, hash = 0, experts = 0, top_k = 0, expert_dim = 0, layers = 2;
    std::vector<int> conv_dilation; // per layer: 0 = attention+MoE, >0 = gated depthwise conv
};
struct context;
context * init(const char * path, int threads, std::string & failure);
context * init_q4k(const char * path, int threads, std::string & failure);
void free(context *);
const char * error(const context *);
const profile * get_profile(const context *);
// Frozen token IDs and supplied mask; returns L2-normalised [count,128] vectors.
bool encode_tokens(context *, const int32_t * ids, const uint8_t * supplied, size_t count, bool query,
                   std::vector<float> & vectors);
} // namespace moedern_family
