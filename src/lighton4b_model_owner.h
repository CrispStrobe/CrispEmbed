#pragma once
#include "qwen35_lighton4b_text.h"
#include "qwen35_vision.h"

namespace qwen35_ocr {
// Atomic borrowed ownership only. No payload access, GGUF/profile admission,
// frontend, decoder execution or quality qualification. Caller retains lifetime.
struct lighton4b_model_owner {
    lighton4b_text_owner text;
    vision_weights vision;
    vision_config vision_parameters;
};
inline bool map_lighton4b_model(const core_gguf::tensor_map & tensors, lighton4b_model_owner & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER") || tensors.size() != 723) return false;
    core_gguf::tensor_map text, vision;
    std::set<ggml_tensor *> identities;
    for (const auto & item : tensors) {
        auto * t = item.second;
        if (!t || !ggml_is_contiguous(t) || !identities.insert(t).second) return false;
        if (item.first.rfind("l.", 0) == 0) {
            text.emplace(item);
        } else if (item.first.rfind("v.", 0) == 0) {
            // Converter stores all norms/biases in F32 in both F16 recipes.
            // Frozen Q8/Q4 candidates protect every vision tensor as F32.
            if (core_env::on("CRISPEMBED_LIGHTON4B_QUANT") && t->type != GGML_TYPE_F32) return false;
            const bool matrix = t->ne[0] > 1 && t->ne[1] > 1;
            if (matrix ? t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 : t->type != GGML_TYPE_F32) return false;
            vision.emplace(item);
        } else
            return false;
    }
    lighton4b_model_owner candidate;
    if (!map_lighton4b_text(text, candidate.text) ||
        !map_lighton4b_vision(vision, candidate.vision, candidate.vision_parameters))
        return false;
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
