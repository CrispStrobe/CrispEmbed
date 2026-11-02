#include "josev2b_backbone.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <string>

namespace josev2b_backbone {
namespace {
bool enabled() {
    return core_env::on("CRISPEMBED_QWEN35_OCR") && core_env::on("CRISPEMBED_JOSEV_BACKBONE");
}
bool shape(ggml_tensor * t, int x, int y = 1, int z = 1, bool mixed = false) {
    const bool type_ok =
        t && (t->type == GGML_TYPE_F32 ||
              (mixed && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_K)));
    return t && type_ok && ggml_is_contiguous(t) && t->ne[0] == x && t->ne[1] == y && t->ne[2] == z && t->ne[3] == 1;
}
bool valid(const qwen35_ocr::decoder_weights & w, bool mixed = false) {
    if (w.head || w.layers.size() != 24 || !shape(w.embedding, 2048, 248320) || !shape(w.norm, 2048)) return false;
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const auto & l = w.layers[i];
        if (l.full_attention != (i % 4 == 3) || !shape(l.input_norm, 2048) || !shape(l.post_norm, 2048) ||
            !shape(l.gate, 2048, 6144, 1, mixed) || !shape(l.up, 2048, 6144, 1, mixed) ||
            !shape(l.down, 6144, 2048, 1, mixed))
            return false;
        if (l.full_attention) {
            const auto & a = l.attention;
            if (!shape(a.q, 2048, 4096, 1, mixed) || !shape(a.k, 2048, 512, 1, mixed) ||
                !shape(a.v, 2048, 512, 1, mixed) || !shape(a.out, 2048, 2048, 1, mixed) || !shape(a.q_norm, 256) ||
                !shape(a.k_norm, 256))
                return false;
        } else {
            const auto & a = l.linear;
            if (!shape(a.qkv, 2048, 6144, 1, mixed) || !shape(a.z, 2048, 2048, 1, mixed) || !shape(a.a, 2048, 16) ||
                !shape(a.b, 2048, 16) || !shape(a.out, 2048, 2048, 1, mixed) || !shape(a.conv, 4, 1, 6144) ||
                !shape(a.a_log, 16) || !shape(a.dt_bias, 16) || !shape(a.norm, 128))
                return false;
        }
    }
    return true;
}
} // namespace
static bool map_impl(const core_gguf::tensor_map & tensors, qwen35_ocr::decoder_weights & weights, bool mixed) {
    weights = {};
    if (!enabled() || tensors.size() != 320) return false;
    std::set<std::string> consumed;
    auto get = [&](const std::string & name) {
        consumed.insert(name);
        return core_gguf::try_get(tensors, name.c_str());
    };
    qwen35_ocr::decoder_weights w;
    w.embedding = get("l.embed_tokens.weight");
    w.norm = get("l.norm.weight");
    for (int i = 0; i < 24; ++i) {
        const auto prefix = "l.layers." + std::to_string(i) + ".";
        auto field = [&](const char * name) { return get(prefix + name); };
        qwen35_ocr::decoder_layer_weights l;
        l.full_attention = i % 4 == 3;
        l.input_norm = field("input_layernorm.weight");
        l.post_norm = field("post_attention_layernorm.weight");
        l.gate = field("mlp.gate_proj.weight");
        l.up = field("mlp.up_proj.weight");
        l.down = field("mlp.down_proj.weight");
        if (l.full_attention) {
            l.attention = { field("self_attn.q_proj.weight"), field("self_attn.k_proj.weight"),
                            field("self_attn.v_proj.weight"), field("self_attn.o_proj.weight"),
                            field("self_attn.q_norm.weight"), field("self_attn.k_norm.weight") };
        } else {
            l.linear = { field("linear_attn.in_proj_qkv.weight"),
                         field("linear_attn.in_proj_z.weight"),
                         field("linear_attn.in_proj_a.weight"),
                         field("linear_attn.in_proj_b.weight"),
                         field("linear_attn.out_proj.weight"),
                         field("linear_attn.conv1d.weight"),
                         field("linear_attn.A_log"),
                         field("linear_attn.dt_bias"),
                         field("linear_attn.norm.weight") };
        }
        w.layers.push_back(l);
    }
    if (consumed.size() != tensors.size() || !valid(w, mixed)) return false;
    weights = std::move(w);
    return true;
}
static bool run_impl(const qwen35_ocr::decoder_weights & weights, const std::vector<int> & ids, int threads,
                     qwen35_ocr::decoder_outputs & output, bool mixed) {
    output = {};
    if (!enabled() || !valid(weights, mixed) || ids.empty() || ids.size() > 512 || threads < 1 || threads > 256)
        return false;
    std::vector<float> hidden, positions(3 * ids.size());
    if (!qwen35_ocr::embed_cpu(weights.embedding, ids, hidden)) return false;
    for (size_t axis = 0; axis < 3; ++axis)
        for (size_t i = 0; i < ids.size(); ++i) positions[axis * ids.size() + i] = float(i);
    const qwen35_ocr::attention_config config{};
    return qwen35_ocr::decoder_hidden_cpu(weights, config, hidden.data(), positions.data(), int(ids.size()), threads,
                                          output);
}
bool map(const core_gguf::tensor_map & tensors, qwen35_ocr::decoder_weights & weights) {
    return map_impl(tensors, weights, false);
}
bool map_quantized(const core_gguf::tensor_map & tensors, qwen35_ocr::decoder_weights & weights) {
    weights = {};
    return core_env::on("CRISPEMBED_JOSEV_QUANT") && core_env::on("CRISPEMBED_QWEN35_QUANT") &&
           map_impl(tensors, weights, true);
}
bool run(const qwen35_ocr::decoder_weights & weights, const std::vector<int> & ids, int threads,
         qwen35_ocr::decoder_outputs & output) {
    return run_impl(weights, ids, threads, output, false);
}
bool run_quantized(const qwen35_ocr::decoder_weights & weights, const std::vector<int> & ids, int threads,
                   qwen35_ocr::decoder_outputs & output) {
    output = {};
    return core_env::on("CRISPEMBED_JOSEV_QUANT") && core_env::on("CRISPEMBED_QWEN35_QUANT") &&
           run_impl(weights, ids, threads, output, true);
}
} // namespace josev2b_backbone
