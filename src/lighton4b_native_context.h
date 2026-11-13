#pragma once
#include "lighton4b_gguf_inspection.h"
#include "lighton4b_decoder_compose.h"
#include "ggml-cpu.h"
#include "lighton4b_q4km_types.h"
#include "lighton4b_physical_f16_vision.h"
#include <set>

namespace qwen35_ocr {
enum class lighton4b_quant_recipe {
    q8_0,
    q4_k,
    attention_q8_ffn_q4,
    q4km_q6_q8head,
    gdn_out_q8,
    gdn_out_q6,
    imatrix_q4km_q6head,
    upstream_physical_q4km,
    upstream_physical_q6,
    hf_tiled_out_q4km
};
// Exact storage policy, separate from shape/source ownership and payload admission.
inline bool lighton4b_quant_recipe_matches(const decoder_weights & decoder, lighton4b_quant_recipe recipe) {
    if (recipe != lighton4b_quant_recipe::q8_0 && recipe != lighton4b_quant_recipe::q4_k &&
        recipe != lighton4b_quant_recipe::attention_q8_ffn_q4 && recipe != lighton4b_quant_recipe::q4km_q6_q8head &&
        recipe != lighton4b_quant_recipe::gdn_out_q8 && recipe != lighton4b_quant_recipe::gdn_out_q6 &&
        recipe != lighton4b_quant_recipe::imatrix_q4km_q6head &&
        recipe != lighton4b_quant_recipe::upstream_physical_q4km &&
        recipe != lighton4b_quant_recipe::upstream_physical_q6 && recipe != lighton4b_quant_recipe::hf_tiled_out_q4km)
        return false;
    auto type = [](const ggml_tensor * t, ggml_type expected) { return t && t->type == expected; };
    if (recipe == lighton4b_quant_recipe::upstream_physical_q6)
        return lighton4b_upstream_physical_q6_enabled() && !lighton4b_physical_q4_repack_mode(true) &&
               lighton4b_source_prefill_chunks_valid(true) && lighton4b_physical_q6_tiled_valid(true) &&
               lighton4b_q6_enabled() && lighton4b_upstream_physical_mode_valid(true) &&
               lighton4b_physical_f16_attention_valid(true) && lighton4b_physical_f16_vision_valid(true) &&
               lighton4b_full_q6_types_match(decoder);
    if (recipe == lighton4b_quant_recipe::upstream_physical_q4km)
        return !lighton4b_upstream_physical_q6_enabled() && lighton4b_source_prefill_chunks_valid(true) &&
               lighton4b_physical_q6_tiled_valid(true) && lighton4b_q6_enabled() &&
               lighton4b_upstream_physical_mode_valid(true) && lighton4b_physical_f16_attention_valid(true) &&
               lighton4b_physical_f16_vision_valid(true) && lighton4b_physical_q4_repack_valid(true) &&
               lighton4b_q4km_types_match(decoder, GGML_TYPE_Q6_K);
    const bool tiled = recipe == lighton4b_quant_recipe::hf_tiled_out_q4km;
    if (tiled && !lighton4b_hf_tiled_out_mode_valid(true, false)) return false;
    const bool q6head = recipe == lighton4b_quant_recipe::imatrix_q4km_q6head || tiled;
    if (decoder.layers.size() != 32 || decoder.head != decoder.embedding ||
        !type(decoder.embedding, q6head ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0))
        return false;
    if (recipe == lighton4b_quant_recipe::q4km_q6_q8head || q6head) {
        if (!lighton4b_q6_enabled() || (q6head && !tiled && !lighton4b_imatrix_q4km_enabled())) return false;
        return lighton4b_q4km_types_match(decoder, q6head ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0);
    }
    if (recipe == lighton4b_quant_recipe::gdn_out_q6 && !lighton4b_gdn_out_q6_enabled()) return false;
    const bool gdn_out = recipe == lighton4b_quant_recipe::gdn_out_q8 || recipe == lighton4b_quant_recipe::gdn_out_q6;
    const auto out_type = recipe == lighton4b_quant_recipe::gdn_out_q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
    const auto ffn = recipe == lighton4b_quant_recipe::q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K;
    const auto attention = (recipe == lighton4b_quant_recipe::q4_k || gdn_out) ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0;
    for (size_t i = 0; i < decoder.layers.size(); ++i) {
        const auto & layer = decoder.layers[i];
        if (layer.full_attention != (i % 4 == 3)) return false;
        for (auto * t : { layer.gate, layer.up, layer.down })
            if (!type(t, ffn)) return false;
        if (layer.full_attention) {
            for (auto * t : { layer.attention.q, layer.attention.k, layer.attention.v, layer.attention.out })
                if (!type(t, attention)) return false;
        } else {
            for (auto * t : { layer.linear.qkv, layer.linear.z, layer.linear.a, layer.linear.b })
                if (!type(t, attention)) return false;
            if (!type(layer.linear.out, gdn_out ? out_type : attention)) return false;
        }
    }
    return true;
}

// Exact profile selection is separate from typed metadata verification.
inline bool lighton4b_select_recipe(const decoder_weights & decoder, const std::string & tag,
                                    lighton4b_quant_recipe & recipe) {
    const bool physical = lighton4b_upstream_physical_enabled();
    if (core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL_Q6") && !physical) return false;
    const bool tiled = lighton4b_hf_tiled_out_enabled();
    const bool mixed = core_env::on("CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4");
    const bool out8 = core_env::on("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8");
    const bool out6 = core_env::on("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6");
    const bool q6head = core_env::on("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD");
    if (int(mixed) + int(out8) + int(out6) + int(q6head) + int(physical) + int(tiled) > 1) return false;
    if (tiled) {
        if (tag != lighton4b_hf_tiled_out_recipe || !lighton4b_hf_tiled_out_mode_valid(true, false)) return false;
        recipe = lighton4b_quant_recipe::hf_tiled_out_q4km;
    } else if (physical) {
        const bool full_q6 = lighton4b_upstream_physical_q6_enabled();
        if (tag != (full_q6 ? lighton4b_upstream_physical_q6_recipe : lighton4b_upstream_physical_recipe) ||
            !lighton4b_physical_native_ops_valid(true) || !lighton4b_physical_f16_attention_valid(true) ||
            !lighton4b_physical_f16_vision_valid(true))
            return false;
        recipe =
            full_q6 ? lighton4b_quant_recipe::upstream_physical_q6 : lighton4b_quant_recipe::upstream_physical_q4km;
    } else if (tag == lighton4b_imatrix_q4km_recipe) {
        if (!lighton4b_imatrix_q4km_enabled() || mixed || out8 || out6) return false;
        recipe = lighton4b_quant_recipe::imatrix_q4km_q6head;
    } else if (q6head) {
        return false; // Explicit profile gate must never choose an older/default recipe.
    } else if (tag == lighton4b_q6_recipe) {
        if (!lighton4b_q6_enabled() || mixed || out8 || out6) return false;
        recipe = lighton4b_quant_recipe::q4km_q6_q8head;
    } else if (out6) {
        if (!lighton4b_gdn_out_q6_enabled() || tag != lighton4b_gdn_out_q6_recipe) return false;
        recipe = lighton4b_quant_recipe::gdn_out_q6;
    } else if (out8) {
        if (tag != "lighton4b-q4k-gdn-out-q8-v1") return false;
        recipe = lighton4b_quant_recipe::gdn_out_q8;
    } else if (mixed) {
        if (tag != "lighton4b-q4k-mixers-q8-v1") return false;
        recipe = lighton4b_quant_recipe::attention_q8_ffn_q4;
    } else {
        if (!tag.empty() || decoder.layers.empty() || !decoder.layers.front().gate) return false;
        recipe = decoder.layers.front().gate->type == GGML_TYPE_Q8_0 ? lighton4b_quant_recipe::q8_0
                                                                     : lighton4b_quant_recipe::q4_k;
    }
    return lighton4b_quant_recipe_matches(decoder, recipe);
}

// Experimental payload owner. Admission is structural, never quality qualification.
// Legacy format5 stays immutable. Explicit physical format6 has its own closed
// provenance/semantics gate and retains the same protected-vision geometry.
class lighton4b_native_context {
    ggml_backend_t backend_ = nullptr;
    core_gguf::WeightLoad loaded_;
    lighton4b_model_owner model_;
    lighton4b_native_context() = default;

public:
    lighton4b_native_context(const lighton4b_native_context &) = delete;
    lighton4b_native_context & operator=(const lighton4b_native_context &) = delete;
    ~lighton4b_native_context() {
        model_.text.decoder.physical_weights_cache.reset();
        core_gguf::free_weights(loaded_);
        if (backend_) ggml_backend_free(backend_);
    }
    const lighton4b_model_owner & model() const { return model_; }
    static std::unique_ptr<lighton4b_native_context> open(const char * path) {
        if (!core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") || !core_env::on("CRISPEMBED_LIGHTON4B_QUANT") ||
            !core_env::on("CRISPEMBED_QWEN35_QUANT"))
            return nullptr;
        // This checks exact source, all geometry/EOS declarations and tensor schemas
        // before allocating or reading any weight payload.
        auto header = lighton4b_gguf_inspection::open(path);
        if (!header) return nullptr;
        const auto & decoder = header->model().text.decoder;
        lighton4b_quant_recipe recipe;
        if (!lighton4b_select_recipe(decoder, header->mixed_quant_recipe(), recipe)) return nullptr;
        if (recipe == lighton4b_quant_recipe::q4km_q6_q8head && !header->q6_profile_header()) return nullptr;
        if (recipe == lighton4b_quant_recipe::imatrix_q4km_q6head && !header->imatrix_q4km_profile_header())
            return nullptr;
        if ((recipe == lighton4b_quant_recipe::upstream_physical_q4km ||
             recipe == lighton4b_quant_recipe::upstream_physical_q6) &&
            !header->upstream_physical_profile_header())
            return nullptr;
        if (recipe == lighton4b_quant_recipe::hf_tiled_out_q4km && !header->hf_tiled_out_profile_header())
            return nullptr;
        if (recipe == lighton4b_quant_recipe::gdn_out_q8 && !header->gdn_out_recipe_metadata_matches()) return nullptr;
        if (recipe == lighton4b_quant_recipe::gdn_out_q6 && !header->gdn_out_q6_metadata_matches()) return nullptr;
        auto candidate = std::unique_ptr<lighton4b_native_context>(new lighton4b_native_context);
        candidate->backend_ = ggml_backend_cpu_init();
        if (!candidate->backend_ ||
            !core_gguf::load_weights(path, candidate->backend_, "lighton4b-native", candidate->loaded_, true) ||
            !map_lighton4b_model(candidate->loaded_.tensors, candidate->model_))
            return nullptr;
        if (recipe == lighton4b_quant_recipe::hf_tiled_out_q4km) {
            auto & weights = candidate->model_.text.decoder;
            weights.hf_tiled_out = true;
            for (auto & layer : weights.layers)
                if (!layer.full_attention) layer.linear.hf_tiled_out = true;
        }
        if (recipe == lighton4b_quant_recipe::upstream_physical_q4km ||
            recipe == lighton4b_quant_recipe::upstream_physical_q6) {
            auto & weights = candidate->model_.text.decoder;
            weights.upstream_physical = true;
            weights.physical_full_q6 = recipe == lighton4b_quant_recipe::upstream_physical_q6;
            weights.physical_q4_repack = lighton4b_physical_q4_repack_mode(true);
            if (weights.physical_q4_repack) {
                if (!lighton4b_physical_q4_repack_valid(true)) return nullptr;
                std::size_t reserve = 0, count = 0;
                std::set<const ggml_tensor *> seen;
                for (const auto & item : candidate->loaded_.tensors) {
                    const auto * tensor = item.second;
                    if (tensor->type != GGML_TYPE_Q4_K || !seen.insert(tensor).second) continue;
                    const auto amount = ggml_nbytes(tensor) + converted_weight_cache::physical_repack_entry_overhead;
                    if (amount > converted_weight_cache::maximum_physical_repack_budget - reserve) return nullptr;
                    reserve += amount;
                    ++count;
                }
                if (count != 216) return nullptr;
                auto cache = std::make_shared<converted_weight_cache>(0); // no converted F32 matrices
                if (!cache->enable_physical_repack(true)) return nullptr;
                weights.physical_weights_cache = std::move(cache);
            }
            for (auto & layer : weights.layers) {
                layer.upstream_physical = true;
                layer.physical_q4_repack = weights.physical_q4_repack;
                layer.physical_weights_cache = weights.physical_weights_cache.get();
                if (layer.full_attention) {
                    layer.attention.upstream_physical = true;
                    layer.attention.physical_q4_repack = weights.physical_q4_repack;
                } else {
                    layer.linear.upstream_physical = true;
                    layer.linear.physical_q4_repack = weights.physical_q4_repack;
                }
            }
        }
        return candidate;
    }
};

// Chunked prefill retains the existing <=16-token component contract. Caches
// remain immutable inputs and independent of the caller's previous cache.
// The opt-in source schedule requires explicit 0=text/1=image modalities.
inline bool lighton4b_prefill_cpu(const decoder_weights & weights, const float * hidden, const float * positions,
                                  int tokens, int threads, const lighton4b_decoder_cache * previous,
                                  lighton4b_decoder_outputs & output, const std::vector<int> * modalities = nullptr) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") || !hidden || !positions || tokens < 1 || tokens > 4096 ||
        (previous && (previous->tokens < 1 || previous->tokens > 4096 - tokens)))
        return false;
    const bool source_chunks = lighton4b_source_prefill_chunks_mode(weights.upstream_physical);
    std::vector<lighton4b_prefill_chunk> plan;
    if (!lighton4b_source_prefill_chunks_valid(weights.upstream_physical) ||
        (source_chunks && (!lighton4b_source_prefill_budget(tokens, previous ? previous->tokens : 0) || !modalities ||
                           modalities->size() != size_t(tokens) ||
                           !lighton4b_source_prefill_plan(
                               *modalities, plan, lighton4b_source_prefill_aligned_mode(weights.upstream_physical)))))
        return false;
    lighton4b_decoder_outputs candidate;
    lighton4b_decoder_cache cache;
    for (int offset = 0, part = 0; offset < tokens; ++part) {
        const int count = source_chunks ? plan[size_t(part)].tokens : std::min(16, tokens - offset);
        std::vector<float> chunk_positions(size_t(3 * count));
        if (source_chunks) {
            if (plan[size_t(part)].offset != offset ||
                !lighton4b_prefill_chunk_positions(positions, tokens, plan[size_t(part)], chunk_positions))
                return false;
        } else
            for (int axis = 0; axis < 3; ++axis)
                std::copy_n(positions + size_t(axis) * tokens + offset, count, chunk_positions.data() + axis * count);
        lighton4b_decoder_outputs chunk;
        if (!lighton4b_decoder_hidden_cpu(weights, hidden + size_t(offset) * 2560, chunk_positions.data(), count,
                                          threads, offset ? &cache : previous, chunk))
            return false;
        if (candidate.hidden.empty()) candidate.hidden.resize(32);
        for (size_t i = 0; i < 32; ++i)
            candidate.hidden[i].insert(candidate.hidden[i].end(), chunk.hidden[i].begin(), chunk.hidden[i].end());
        candidate.norm.insert(candidate.norm.end(), chunk.norm.begin(), chunk.norm.end());
        cache = std::move(chunk.cache);
        offset += count;
    }
    candidate.cache = std::move(cache);
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
