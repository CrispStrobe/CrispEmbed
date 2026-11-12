#pragma once
#include "lighton4b_linear_block.h"
#include "lighton4b_bf16_dequant_weights.h"
#include "lighton4b_attention_block.h"

namespace qwen35_ocr {
struct lighton4b_layer_cache {
    bool full_attention = false;
    lighton4b_linear_cache linear;
    attention_cache attention;
};
struct lighton4b_decoder_cache {
    int tokens = 0;
    bool bf16_quant_weights = false;
    std::vector<lighton4b_layer_cache> layers;
    bool upstream_physical = false;
    uint8_t physical_native_ops = 0;
    bool physical_f16_attention = false;
    bool hf_tiled_out = false;
    bool physical_q4_repack = false;
    bool physical_gdn_source_ops = false;
    bool physical_q6_tiled = false;
    bool physical_source_prefill_chunks = false;
};
struct lighton4b_decoder_outputs {
    std::vector<std::vector<float>> hidden;
    std::vector<float> norm;
    lighton4b_decoder_cache cache;
};
// Hidden-state component only, batch1, exact32-layer LLLF schedule. Weight
// lifetime stays with caller. No embedding reads, logits, GGUF/profile admission
// or recognition. Each previous cache is immutable and belongs to one layer.
inline bool lighton4b_decoder_hidden_cpu(const decoder_weights & weights, const float * input, const float * positions,
                                         int tokens, int threads, const lighton4b_decoder_cache * previous,
                                         lighton4b_decoder_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_DECODER_COMPOSE") || !core_env::on("CRISPEMBED_LIGHTON4B_LINEAR_BLOCK") ||
        !core_env::on("CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK") || !core_env::on("CRISPEMBED_QWEN35_GDN_GROUPED_HEADS") ||
        !core_env::on("CRISPEMBED_QWEN35_OCR") || !input || !positions || tokens < 1 || tokens > 16 || threads < 1 ||
        threads > 256 || weights.layers.size() != 32)
        return false;
    const auto * norm = weights.norm;
    if (!norm || norm->type != GGML_TYPE_F32 || !norm->data || !ggml_is_contiguous(norm) || norm->ne[0] != 2560 ||
        norm->ne[1] != 1 || norm->ne[2] != 1 || norm->ne[3] != 1)
        return false;
    const bool bf16_quant_weights = lighton4b_bf16_dequant_weights_mode();
    if (!lighton4b_hf_tiled_out_mode_valid(weights.hf_tiled_out, weights.upstream_physical) ||
        (previous && previous->hf_tiled_out != weights.hf_tiled_out))
        return false;
    const bool physical_f16 = lighton4b_physical_f16_attention_mode(weights.upstream_physical);
    const bool repack = lighton4b_physical_q4_repack_mode(weights.upstream_physical);
    const bool source_chunks = lighton4b_source_prefill_chunks_mode(weights.upstream_physical);
    if (!lighton4b_source_prefill_chunks_valid(weights.upstream_physical) ||
        (previous && previous->physical_source_prefill_chunks != source_chunks))
        return false;
    const bool q6_tiled = lighton4b_physical_q6_tiled_mode(weights.upstream_physical);
    if (!lighton4b_physical_q6_tiled_valid(weights.upstream_physical) ||
        (previous && previous->physical_q6_tiled != q6_tiled))
        return false;
    const bool source_gdn = lighton4b_physical_gdn_source_ops_mode(weights.upstream_physical);
    if (previous && previous->physical_gdn_source_ops != source_gdn) return false;
    if (!lighton4b_physical_q4_repack_valid(weights.upstream_physical) || weights.physical_q4_repack != repack ||
        bool(weights.physical_weights_cache && weights.physical_weights_cache->physical_repack_enabled()) != repack ||
        (previous && previous->physical_q4_repack != repack))
        return false;
    for (const auto & layer : weights.layers)
        if (layer.physical_q4_repack != repack ||
            layer.physical_weights_cache != weights.physical_weights_cache.get() ||
            (layer.full_attention ? layer.attention.physical_q4_repack : layer.linear.physical_q4_repack) != repack)
            return false;
    if (!lighton4b_physical_f16_attention_valid(weights.upstream_physical) ||
        (previous && previous->physical_f16_attention != physical_f16))
        return false;
    if (!lighton4b_upstream_physical_mode_valid(weights.upstream_physical) ||
        !lighton4b_physical_native_ops_valid(weights.upstream_physical) ||
        (previous && (previous->upstream_physical != weights.upstream_physical ||
                      previous->physical_native_ops != lighton4b_physical_native_ops_mode(weights.upstream_physical))))
        return false;
    if (previous && previous->bf16_quant_weights != bf16_quant_weights) return false;
    const int past = previous ? previous->tokens : 0;
    if (physical_f16 && (threads != 4 || past > 2048 - tokens)) return false;
    if (past < 0 || past > 4096 - tokens || (previous && (past == 0 || previous->layers.size() != 32))) return false;
    auto empty_attention = [](const attention_cache & c) {
        return !c.batch && !c.heads && !c.tokens && !c.width && !c.upstream_physical && !c.physical_native_ops &&
               !c.physical_f16_attention && !c.physical_q4_repack && !c.physical_q6_tiled &&
               !c.physical_source_prefill_chunks && c.keys.empty() && c.values.empty();
    };
    auto finite = [](const std::vector<float> & v) {
        return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
    };
    for (int i = 0; i < 3 * tokens; ++i)
        if (!std::isfinite(positions[i])) return false;
    std::vector<float> norm_weights(2560);
    if (!tensor_floats(weights.norm, norm_weights.data()) || !finite(norm_weights)) return false;
    for (size_t i = 0; i < 32; ++i) {
        const bool full = i % 4 == 3;
        if (weights.layers[i].full_attention != full ||
            weights.layers[i].upstream_physical != weights.upstream_physical ||
            weights.layers[i].linear.hf_tiled_out != (weights.hf_tiled_out && !full))
            return false;
        if (previous) {
            const auto & c = previous->layers[i];
            if (c.full_attention != full) return false;
            if (full) {
                if (c.attention.physical_source_prefill_chunks != source_chunks ||
                    c.attention.physical_q6_tiled != q6_tiled || c.attention.physical_q4_repack != repack ||
                    c.attention.physical_f16_attention != physical_f16 || c.attention.tokens != past ||
                    c.attention.batch != 1 || c.attention.heads != 4 || c.attention.width != 256 ||
                    c.attention.keys.size() != size_t(past) * 1024 ||
                    c.attention.values.size() != c.attention.keys.size() || !finite(c.attention.keys) ||
                    !finite(c.attention.values) || c.linear.tokens || c.linear.bf16_activations ||
                    c.linear.upstream_physical || c.linear.physical_native_ops || c.linear.hf_tiled_out ||
                    c.linear.physical_q4_repack || c.linear.physical_gdn_source_ops || c.linear.physical_q6_tiled ||
                    c.linear.physical_source_prefill_chunks || !c.linear.conv.empty() || !c.linear.recurrent.empty())
                    return false;
            } else if (c.linear.physical_source_prefill_chunks != source_chunks ||
                       c.linear.physical_q6_tiled != q6_tiled || c.linear.physical_gdn_source_ops != source_gdn ||
                       c.linear.physical_q4_repack != repack || c.linear.tokens != past ||
                       c.linear.conv.size() != 32768 || c.linear.recurrent.size() != 524288 || !finite(c.linear.conv) ||
                       !finite(c.linear.recurrent) || !empty_attention(c.attention))
                return false;
        }
    }
    lighton4b_decoder_outputs candidate;
    candidate.cache.tokens = past + tokens;
    candidate.cache.hf_tiled_out = weights.hf_tiled_out;
    candidate.cache.bf16_quant_weights = bf16_quant_weights;
    candidate.cache.upstream_physical = weights.upstream_physical;
    candidate.cache.physical_native_ops = lighton4b_physical_native_ops_mode(weights.upstream_physical);
    candidate.cache.physical_f16_attention = physical_f16;
    candidate.cache.physical_q4_repack = repack;
    candidate.cache.physical_gdn_source_ops = source_gdn;
    candidate.cache.physical_q6_tiled = q6_tiled;
    candidate.cache.physical_source_prefill_chunks = source_chunks;
    std::vector<float> hidden(input, input + size_t(tokens) * 2560);
    for (size_t i = 0; i < 32; ++i) {
        const auto & w = weights.layers[i];
        lighton4b_layer_cache next;
        next.full_attention = w.full_attention;
        if (w.full_attention) {
            lighton4b_attention_block_outputs block;
            if (!lighton4b_attention_block_cpu(w, hidden.data(), positions, tokens, threads,
                                               previous ? &previous->layers[i].attention : nullptr, block))
                return false;
            hidden = std::move(block.hidden);
            next.attention = std::move(block.cache);
        } else {
            lighton4b_linear_block_outputs block;
            if (!lighton4b_linear_block_cpu(w, hidden.data(), tokens, threads,
                                            previous ? &previous->layers[i].linear : nullptr, block))
                return false;
            hidden = std::move(block.hidden);
            next.linear = std::move(block.cache);
        }
        candidate.hidden.push_back(hidden);
        candidate.cache.layers.push_back(std::move(next));
    }
    candidate.norm.resize(hidden.size());
    if (lighton4b_physical_native_component(weights.upstream_physical, 2)) {
        if (!lighton4b_physical_rms(hidden.data(), norm_weights.data(), 2560, tokens, threads, candidate.norm))
            return false;
    } else
        for (int token = 0; token < tokens; ++token) {
            const size_t base = size_t(token) * 2560;
            const float scale = rms_scale_cpu(hidden.data() + base, 2560, 1e-6f);
            for (int i = 0; i < 2560; ++i)
                candidate.norm[base + i] =
                    (hidden[base + i] * scale) * lighton4b_upstream_gamma(norm_weights[i], weights.upstream_physical);
        }
    if (!std::all_of(candidate.norm.begin(), candidate.norm.end(), [](float x) { return std::isfinite(x); }))
        return false;
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
