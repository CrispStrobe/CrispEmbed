#include "lighton4b_physical_native_ops.h"
#include "lighton4b_physical_f16_attention.h"
#include "lighton4b_upstream_physical_semantics.h"
#pragma once
#include "qwen35_lighton4b_text.h"
#include "qwen35_math.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace qwen35_ocr {
struct lighton4b_attention_block_outputs {
    std::vector<float> input_norm, q, k, v, q_norm, k_norm, pooled, attention, attention_residual, post_norm, gate, up,
        activation, down, hidden;
    attention_cache cache;
};
// One full-width full-attention decoder block, batch1, at most16 new tokens. Explicit
// synthetic/component entry only: no full decoder, vocabulary, model admission
// or recognition. Layer weights are borrowed and caches are immutable inputs.
inline bool lighton4b_attention_block_cpu(const decoder_layer_weights & w, const float * input, const float * positions,
                                          int tokens, int threads, const attention_cache * previous,
                                          lighton4b_attention_block_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK") || !core_env::on("CRISPEMBED_QWEN35_OCR") || !input ||
        !positions || !w.full_attention || tokens < 1 || tokens > 16 || threads < 1 || threads > 256)
        return false;
    auto finite = [](const std::vector<float> & v) {
        return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    };
    const int past = previous ? previous->tokens : 0;
    if (past < 0 || past > 4096 - tokens ||
        (previous &&
         (past == 0 || previous->batch != 1 || previous->heads != 4 || previous->width != 256 ||
          previous->keys.size() != size_t(4) * past * 256 || previous->values.size() != previous->keys.size() ||
          !finite(previous->keys) || !finite(previous->values))))
        return false;
    std::set<ggml_tensor *> identities;
    auto shape = [&](ggml_tensor * t, int a, int b = 1, int c = 1, bool matrix = false) {
        return t && t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == 1 && ggml_is_contiguous(t) &&
               t->data && identities.insert(t).second && (matrix ? lighton4b_matrix_type(t) : t->type == GGML_TYPE_F32);
    };
    if (!shape(w.input_norm, 2560) || !shape(w.post_norm, 2560) || !shape(w.gate, 2560, 9216, 1, true) ||
        !shape(w.up, 2560, 9216, 1, true) || !shape(w.down, 9216, 2560, 1, true) ||
        !shape(w.attention.q, 2560, 8192, 1, true) || !shape(w.attention.k, 2560, 1024, 1, true) ||
        !shape(w.attention.v, 2560, 1024, 1, true) || !shape(w.attention.out, 4096, 2560, 1, true) ||
        !shape(w.attention.q_norm, 256) || !shape(w.attention.k_norm, 256))
        return false;
    if (!lighton4b_physical_f16_attention_valid(w.upstream_physical) ||
        (previous && previous->physical_f16_attention != lighton4b_physical_f16_attention_mode(w.upstream_physical)) ||
        !lighton4b_upstream_physical_mode_valid(w.upstream_physical) || !lighton4b_physical_native_ops_valid(w.upstream_physical) ||
        w.attention.upstream_physical != w.upstream_physical) return false;
    lighton4b_attention_block_outputs candidate;
    candidate.hidden.assign(input, input + size_t(tokens) * 2560);
    if (!finite(candidate.hidden)) return false;
    auto rms = [&](const std::vector<float> & x, ggml_tensor * norm, std::vector<float> & y) {
        std::vector<float> weights(2560);
        if (!tensor_floats(norm, weights.data()) || !finite(weights)) return false;
        if (lighton4b_physical_native_component(w.upstream_physical,2))
            return lighton4b_physical_rms(x.data(), weights.data(), 2560, tokens, threads, y);
        y.resize(x.size());
        for (int token = 0; token < tokens; ++token) {
            const size_t base = size_t(token) * 2560;
            const float scale = rms_scale_cpu(x.data() + base, 2560, 1e-6f);
            for (int i = 0; i < 2560; ++i) y[base + i] = (x[base + i] * scale) * lighton4b_upstream_gamma(weights[i], w.upstream_physical);
        }
        return finite(y);
    };
    if (!rms(candidate.hidden, w.input_norm, candidate.input_norm)) return false;
    attention_config config;
    config.query_heads = 16;
    config.kv_heads = 4;
    config.head_dim = 256;
    config.rotary_dim = 64;
    attention_outputs attention;
    if (!attention_cpu(w.attention, config, candidate.input_norm.data(), positions, 1, tokens, threads, previous,
                       attention))
        return false;
    candidate.q = std::move(attention.q);
    candidate.k = std::move(attention.k);
    candidate.v = std::move(attention.v);
    candidate.q_norm = std::move(attention.q_norm);
    candidate.k_norm = std::move(attention.k_norm);
    candidate.pooled = std::move(attention.pooled);
    candidate.attention = std::move(attention.attention);
    candidate.cache = std::move(attention.cache);
    if (candidate.attention.size() != candidate.hidden.size() || !finite(candidate.attention)) return false;
    for (size_t i = 0; i < candidate.hidden.size(); ++i) candidate.hidden[i] += candidate.attention[i];
    candidate.attention_residual = candidate.hidden;
    if (!rms(candidate.hidden, w.post_norm, candidate.post_norm)) return false;
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ w.gate, w.up }, candidate.post_norm.data(), 2560, tokens, threads, projections, nullptr, w.upstream_physical)) return false;
    candidate.gate = std::move(projections[0]);
    candidate.up = std::move(projections[1]);
    candidate.activation.resize(candidate.gate.size());
    for (size_t i = 0; i < candidate.gate.size(); ++i) {
        const float gate = candidate.gate[i];
        candidate.activation[i] = (gate / (1.0f + std::exp(-gate))) * candidate.up[i];
    }
    projections.clear();
    if (!project_cpu({ w.down }, candidate.activation.data(), 9216, tokens, threads, projections, nullptr, w.upstream_physical)) return false;
    candidate.down = std::move(projections[0]);
    for (size_t i = 0; i < candidate.hidden.size(); ++i) candidate.hidden[i] += candidate.down[i];
    if (!finite(candidate.hidden)) return false;
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
