#pragma once
#include "lighton4b_hf_tiled_out.h"
#include "lighton4b_physical_native_ops.h"
#include "qwen35_lighton4b_text.h"
#include "qwen35_math.h"
#include "lighton4b_bf16_linear.h"
#include "lighton4b_upstream_physical_semantics.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace qwen35_ocr {
struct lighton4b_linear_cache {
    int tokens = 0;
    bool bf16_activations = false;
    std::vector<float> conv, recurrent;
    bool upstream_physical = false;
    uint8_t physical_native_ops = 0;
    bool hf_tiled_out = false;
    bool physical_q4_repack = false;
};
struct lighton4b_linear_block_outputs {
    std::vector<float> input_norm, attention, attention_residual, post_norm, gate, up, activation, down, hidden;
    lighton4b_linear_cache cache;
};
// One full-width linear decoder block, batch1, at most16 new tokens. Explicit
// synthetic/component entry only: no full decoder, vocabulary, model admission
// or recognition. Layer weights are borrowed and caches are immutable inputs.
inline bool lighton4b_linear_block_cpu(const decoder_layer_weights & w, const float * input, int tokens, int threads,
                                       const lighton4b_linear_cache * previous,
                                       lighton4b_linear_block_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_LINEAR_BLOCK") || !core_env::on("CRISPEMBED_QWEN35_OCR") ||
        !core_env::on("CRISPEMBED_QWEN35_GDN_GROUPED_HEADS") || !input || w.full_attention || tokens < 1 ||
        tokens > 16 || threads < 1 || threads > 256)
        return false;
    auto finite = [](const std::vector<float> & v) {
        return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    };
    const bool bf16 = lighton4b_bf16_linear_enabled();
    const bool repack = lighton4b_physical_q4_repack_mode(w.upstream_physical);
    if (!lighton4b_physical_q4_repack_valid(w.upstream_physical) || w.physical_q4_repack != repack ||
        w.linear.physical_q4_repack != repack ||
        bool(w.physical_weights_cache && w.physical_weights_cache->physical_repack_enabled()) != repack ||
        (previous && previous->physical_q4_repack != repack)) return false;
    if (!lighton4b_hf_tiled_out_mode_valid(w.linear.hf_tiled_out,w.upstream_physical) ||
        (previous && previous->hf_tiled_out != w.linear.hf_tiled_out)) return false;
    if (!lighton4b_upstream_physical_mode_valid(w.upstream_physical) || !lighton4b_physical_native_ops_valid(w.upstream_physical) ||
        w.linear.upstream_physical != w.upstream_physical ||
        (previous && (previous->upstream_physical != w.upstream_physical || previous->physical_native_ops != lighton4b_physical_native_ops_mode(w.upstream_physical)))) return false;
    const int past = previous ? previous->tokens : 0;
    if (past < 0 || past > 4096 - tokens ||
        (previous && (previous->bf16_activations != bf16 || past == 0 || previous->conv.size() != 32768 ||
                      previous->recurrent.size() != 524288 || !finite(previous->conv) || !finite(previous->recurrent))))
        return false;
    std::set<ggml_tensor *> identities;
    auto shape = [&](ggml_tensor * t, int a, int b = 1, int c = 1, bool matrix = false) {
        return t && t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == 1 && ggml_is_contiguous(t) &&
               t->data && identities.insert(t).second && (matrix ? lighton4b_matrix_type(t) : t->type == GGML_TYPE_F32);
    };
    if (!shape(w.input_norm, 2560) || !shape(w.post_norm, 2560) || !shape(w.gate, 2560, 9216, 1, true) ||
        !shape(w.up, 2560, 9216, 1, true) || !shape(w.down, 9216, 2560, 1, true) ||
        !shape(w.linear.qkv, 2560, 8192, 1, true) || !shape(w.linear.z, 2560, 4096, 1, true) ||
        !shape(w.linear.a, 2560, 32, 1, true) || !shape(w.linear.b, 2560, 32, 1, true) ||
        !shape(w.linear.out, 4096, 2560, 1, true) || !shape(w.linear.conv, 4, 1, 8192) || !shape(w.linear.a_log, 32) ||
        !shape(w.linear.dt_bias, 32) || !shape(w.linear.norm, 128))
        return false;
    lighton4b_linear_block_outputs candidate;
    candidate.hidden.assign(input, input + size_t(tokens) * 2560);
    if (!finite(candidate.hidden)) return false;
    if (bf16) lighton4b_bf16_round(candidate.hidden);
    auto rms = [&](const std::vector<float> & x, ggml_tensor * norm, std::vector<float> & y) {
        std::vector<float> weights(2560);
        if (!tensor_floats(norm, weights.data()) || !finite(weights)) return false;
        if (lighton4b_physical_native_component(w.upstream_physical,2))
            return lighton4b_physical_rms(x.data(), weights.data(), 2560, tokens, threads, y);
        y.resize(x.size());
        for (int token = 0; token < tokens; ++token) {
            const size_t base = size_t(token) * 2560;
            const float scale = bf16 ? lighton4b_bf16_rms_scale(x.data() + base, 2560, 1e-6f)
                                     : rms_scale_cpu(x.data() + base, 2560, 1e-6f);
            for (int i = 0; i < 2560; ++i) y[base + i] = (x[base + i] * scale) * lighton4b_upstream_gamma(weights[i], w.upstream_physical);
        }
        if (bf16) lighton4b_bf16_round(y);
        return finite(y);
    };
    if (!rms(candidate.hidden, w.input_norm, candidate.input_norm)) return false;
    projected_linear_outputs linear;
    const bool linear_ok = bf16 ? lighton4b_bf16_grouped_cpu(w.linear, candidate.input_norm.data(), tokens, threads,
                                                             previous ? previous->conv.data() : nullptr,
                                                             previous ? previous->recurrent.data() : nullptr, linear)
                                : linear_grouped_cpu(w.linear, 16, candidate.input_norm.data(), 1, tokens, threads,
                                                     previous ? previous->conv.data() : nullptr,
                                                     previous ? previous->recurrent.data() : nullptr, linear, w.physical_weights_cache);
    if (!linear_ok) return false;
    candidate.attention = std::move(linear.attention);
    if (candidate.attention.size() != candidate.hidden.size() || !finite(candidate.attention)) return false;
    for (size_t i = 0; i < candidate.hidden.size(); ++i) candidate.hidden[i] += candidate.attention[i];
    if (bf16) lighton4b_bf16_round(candidate.hidden);
    candidate.attention_residual = candidate.hidden;
    if (!rms(candidate.hidden, w.post_norm, candidate.post_norm)) return false;
    std::vector<std::vector<float>> projections;
    if (!project_cpu({ w.gate, w.up }, candidate.post_norm.data(), 2560, tokens, threads, projections, w.physical_weights_cache, w.upstream_physical)) return false;
    candidate.gate = std::move(projections[0]);
    candidate.up = std::move(projections[1]);
    if (bf16) {
        lighton4b_bf16_round(candidate.gate);
        lighton4b_bf16_round(candidate.up);
    }
    candidate.activation.resize(candidate.gate.size());
    for (size_t i = 0; i < candidate.gate.size(); ++i) {
        const float gate = candidate.gate[i];
        const float activated = gate / (1.0f + std::exp(-gate));
        candidate.activation[i] = bf16 ? lighton4b_bf16_round(lighton4b_bf16_round(activated) * candidate.up[i])
                                       : activated * candidate.up[i];
    }
    projections.clear();
    if (!project_cpu({ w.down }, candidate.activation.data(), 9216, tokens, threads, projections, w.physical_weights_cache, w.upstream_physical)) return false;
    candidate.down = std::move(projections[0]);
    if (bf16) lighton4b_bf16_round(candidate.down);
    for (size_t i = 0; i < candidate.hidden.size(); ++i) candidate.hidden[i] += candidate.down[i];
    if (bf16) lighton4b_bf16_round(candidate.hidden);
    if (!finite(candidate.hidden)) return false;
    candidate.cache.tokens = past + tokens;
    candidate.cache.hf_tiled_out = w.linear.hf_tiled_out;
    candidate.cache.bf16_activations = bf16;
    candidate.cache.upstream_physical = w.upstream_physical;
    candidate.cache.physical_native_ops = lighton4b_physical_native_ops_mode(w.upstream_physical);
    candidate.cache.physical_q4_repack = repack;
    candidate.cache.conv = std::move(linear.core.conv_history);
    candidate.cache.recurrent = std::move(linear.core.state);
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
