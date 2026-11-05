#pragma once
#include "lighton4b_gguf_inspection.h"
#include "lighton4b_decoder_compose.h"
#include "ggml-cpu.h"

namespace qwen35_ocr {
// Experimental payload owner. Admission is structural, never quality qualification.
// Uses the immutable format-5 source contract and frozen protected-vision recipe.
class lighton4b_native_context {
    ggml_backend_t backend_ = nullptr;
    core_gguf::WeightLoad loaded_;
    lighton4b_model_owner model_;
    lighton4b_native_context() = default;
public:
    lighton4b_native_context(const lighton4b_native_context &) = delete;
    lighton4b_native_context & operator=(const lighton4b_native_context &) = delete;
    ~lighton4b_native_context() {
        core_gguf::free_weights(loaded_);
        if (backend_) ggml_backend_free(backend_);
    }
    const lighton4b_model_owner & model() const { return model_; }
    static std::unique_ptr<lighton4b_native_context> open(const char * path) {
        if (!core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") ||
            !core_env::on("CRISPEMBED_LIGHTON4B_QUANT") || !core_env::on("CRISPEMBED_QWEN35_QUANT"))
            return nullptr;
        // This checks exact source, all geometry/EOS declarations and tensor schemas
        // before allocating or reading any weight payload.
        auto header = lighton4b_gguf_inspection::open(path);
        if (!header) return nullptr;
        const auto & decoder = header->model().text.decoder;
        const auto recipe = decoder.embedding->type;
        if (recipe != GGML_TYPE_Q8_0) return nullptr;
        const auto matrix_type = decoder.layers.front().gate->type;
        if (matrix_type != GGML_TYPE_Q8_0 && matrix_type != GGML_TYPE_Q4_K) return nullptr;
        for (const auto & layer : decoder.layers) {
            for (auto * t : { layer.gate, layer.up, layer.down })
                if (t->type != matrix_type) return nullptr;
            if (layer.full_attention) {
                for (auto * t : { layer.attention.q, layer.attention.k, layer.attention.v, layer.attention.out })
                    if (t->type != matrix_type) return nullptr;
            } else {
                for (auto * t : { layer.linear.qkv, layer.linear.z, layer.linear.a, layer.linear.b, layer.linear.out })
                    if (t->type != matrix_type) return nullptr;
            }
        }
        auto candidate = std::unique_ptr<lighton4b_native_context>(new lighton4b_native_context);
        candidate->backend_ = ggml_backend_cpu_init();
        if (!candidate->backend_ ||
            !core_gguf::load_weights(path, candidate->backend_, "lighton4b-native", candidate->loaded_, true) ||
            !map_lighton4b_model(candidate->loaded_.tensors, candidate->model_)) return nullptr;
        return candidate;
    }
};

// Chunked prefill retains the existing <=16-token component contract. Caches
// remain immutable inputs and independent of the caller's previous cache.
inline bool lighton4b_prefill_cpu(const decoder_weights & weights, const float * hidden, const float * positions,
                                  int tokens, int threads, const lighton4b_decoder_cache * previous,
                                  lighton4b_decoder_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT") || !hidden || !positions || tokens < 1 || tokens > 4096 ||
        (previous && (previous->tokens < 1 || previous->tokens > 4096 - tokens))) return false;
    lighton4b_decoder_outputs candidate;
    lighton4b_decoder_cache cache;
    for (int offset = 0; offset < tokens; offset += 16) {
        const int count = std::min(16, tokens - offset);
        std::vector<float> chunk_positions(size_t(3 * count));
        for (int axis = 0; axis < 3; ++axis)
            std::copy_n(positions + size_t(axis) * tokens + offset, count, chunk_positions.data() + axis * count);
        lighton4b_decoder_outputs chunk;
        if (!lighton4b_decoder_hidden_cpu(weights, hidden + size_t(offset) * 2560, chunk_positions.data(), count,
                                         threads, offset ? &cache : previous, chunk)) return false;
        if (candidate.hidden.empty()) candidate.hidden.resize(32);
        for (size_t i = 0; i < 32; ++i)
            candidate.hidden[i].insert(candidate.hidden[i].end(), chunk.hidden[i].begin(), chunk.hidden[i].end());
        candidate.norm.insert(candidate.norm.end(), chunk.norm.begin(), chunk.norm.end());
        cache = std::move(chunk.cache);
    }
    candidate.cache = std::move(cache);
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
