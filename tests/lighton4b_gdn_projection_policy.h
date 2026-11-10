#pragma once
#include "qwen35_decoder.h"
#include <string>
#include <vector>
namespace sensitivity {
enum class group { q4, qkv, z, ab, out, linear, mixers_down, q8 };
struct slot {
    std::string name;
    ggml_tensor ** target;
    int role;
};
inline std::vector<slot> slots(qwen35_ocr::decoder_weights & w) {
    std::vector<slot> result;
    if (w.layers.size() != 32) return result;
    for (int i = 0; i < 32; ++i) {
        auto & l = w.layers[i];
        const auto prefix = "l.layers." + std::to_string(i) + ".";
        auto add = [&](const char * name, ggml_tensor *& p, int role) {
            result.push_back({ prefix + name, &p, role });
        };
        add("mlp.gate_proj.weight", l.gate, 1);
        add("mlp.up_proj.weight", l.up, 1);
        add("mlp.down_proj.weight", l.down, 2);
        if (l.full_attention) {
            add("self_attn.q_proj.weight", l.attention.q, 3);
            add("self_attn.k_proj.weight", l.attention.k, 3);
            add("self_attn.v_proj.weight", l.attention.v, 3);
            add("self_attn.o_proj.weight", l.attention.out, 3);
        } else {
            add("linear_attn.in_proj_qkv.weight", l.linear.qkv, 0);
            add("linear_attn.in_proj_z.weight", l.linear.z, 4);
            add("linear_attn.in_proj_a.weight", l.linear.a, 5);
            add("linear_attn.in_proj_b.weight", l.linear.b, 5);
            add("linear_attn.out_proj.weight", l.linear.out, 6);
        }
    }
    return result;
}
inline bool selected(group g, int role) {
    const bool linear = role == 0 || role == 4 || role == 5 || role == 6;
    const bool mixer = linear || role == 3;
    return g == group::q8 || (g == group::qkv && role == 0) || (g == group::z && role == 4) ||
           (g == group::ab && role == 5) || (g == group::out && role == 6) || (g == group::linear && linear) ||
           (g == group::mixers_down && (mixer || role == 2));
}
inline const char * name(group g) {
    switch (g) {
    case group::q4:
        return "q4";
    case group::qkv:
        return "qkv";
    case group::z:
        return "z";
    case group::ab:
        return "ab";
    case group::out:
        return "out";
    case group::linear:
        return "linear";
    case group::mixers_down:
        return "mixers_down";
    case group::q8:
        return "q8";
    }
    return "invalid";
}
} // namespace sensitivity
