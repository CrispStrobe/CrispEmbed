#pragma once
#include "qwen35_decoder.h"
#include <string>
#include <vector>
namespace sensitivity {
enum class group { q4, mixers, down, ffn, q8 };
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
            add("self_attn.q_proj.weight", l.attention.q, 0);
            add("self_attn.k_proj.weight", l.attention.k, 0);
            add("self_attn.v_proj.weight", l.attention.v, 0);
            add("self_attn.o_proj.weight", l.attention.out, 0);
        } else {
            add("linear_attn.in_proj_qkv.weight", l.linear.qkv, 0);
            add("linear_attn.in_proj_z.weight", l.linear.z, 0);
            add("linear_attn.in_proj_a.weight", l.linear.a, 0);
            add("linear_attn.in_proj_b.weight", l.linear.b, 0);
            add("linear_attn.out_proj.weight", l.linear.out, 0);
        }
    }
    return result;
}
inline bool selected(group g, int role) {
    return g == group::q8 || (g == group::mixers && role == 0) || (g == group::down && role == 2) ||
           (g == group::ffn && role != 0);
}
inline const char * name(group g) {
    switch (g) {
    case group::q4:
        return "q4";
    case group::mixers:
        return "mixers";
    case group::down:
        return "down";
    case group::ffn:
        return "ffn";
    case group::q8:
        return "q8";
    }
    return "invalid";
}
} // namespace sensitivity
