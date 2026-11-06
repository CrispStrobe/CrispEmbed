#pragma once
#include "lighton4b_model_owner.h"
#include "lighton4b_gdn_out_recipe.h"
#include "lighton4b_gdn_out_q6_recipe.h"
#include <memory>

namespace qwen35_ocr {
// Owns real GGUF tensor headers, never model payloads. This is a structural
// inspection context, deliberately separate from qwen35_ocr_context. No backend,
// tokenizer, inference method or payload-backed qualification is exposed.
class lighton4b_gguf_inspection {
    gguf_context * metadata_ = nullptr;
    ggml_context * headers_ = nullptr;
    lighton4b_model_owner model_;
    lighton4b_gguf_inspection() = default;

public:
    lighton4b_gguf_inspection(const lighton4b_gguf_inspection &) = delete;
    lighton4b_gguf_inspection & operator=(const lighton4b_gguf_inspection &) = delete;
    ~lighton4b_gguf_inspection() {
        if (headers_) ggml_free(headers_);
        if (metadata_) gguf_free(metadata_);
    }
    const lighton4b_model_owner & model() const { return model_; }
    std::string mixed_quant_recipe() const { return core_gguf::kv_str(metadata_, "qwen35_ocr.mixed_quant_recipe", ""); }
    bool gdn_out_recipe_metadata_matches() const { return lighton4b_gdn_out_metadata_matches(metadata_); }
    bool gdn_out_q6_metadata_matches() const { return lighton4b_gdn_out_q6_metadata_matches(metadata_); }
    bool q6_profile_header() const { return lighton4b_q6_metadata(metadata_); }
    static std::unique_ptr<lighton4b_gguf_inspection> open(const char * path) {
        if (!core_env::on("CRISPEMBED_LIGHTON4B_GGUF_INSPECTION") ||
            !core_env::on("CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER") ||
            !core_env::on("CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER") ||
            !core_env::on("CRISPEMBED_LIGHTON4B_VISION_OWNER") || !core_env::on("CRISPEMBED_QWEN35_OCR") || !path ||
            !*path)
            return nullptr;
        auto candidate = std::unique_ptr<lighton4b_gguf_inspection>(new lighton4b_gguf_inspection);
        gguf_init_params params{ true, &candidate->headers_ };
        candidate->metadata_ = gguf_init_from_file(path, params);
        auto * g = candidate->metadata_;
        if (!g || !candidate->headers_ || gguf_get_n_tensors(g) != 723) return nullptr;
        auto string = [&](const char * key, const char * expected) {
            const auto index = gguf_find_key(g, key);
            return index >= 0 && gguf_get_kv_type(g, index) == GGUF_TYPE_STRING &&
                   std::string(gguf_get_val_str(g, index)) == expected;
        };
        auto integer = [&](const char * key, uint32_t expected) {
            const auto index = gguf_find_key(g, key);
            return index >= 0 && gguf_get_kv_type(g, index) == GGUF_TYPE_UINT32 &&
                   gguf_get_val_u32(g, index) == expected;
        };
        auto boolean = [&](const char * key) {
            const auto index = gguf_find_key(g, key);
            return index >= 0 && gguf_get_kv_type(g, index) == GGUF_TYPE_BOOL && gguf_get_val_bool(g, index);
        };
        if (!string("general.architecture", "qwen35_ocr") ||
            !string("general.source.huggingface.repository", "lightonai/LightOnOCR-3-4B") ||
            !string("general.source.huggingface.revision", "b71010f095dc3735de7ee6b9969bff8f8f50b39a") ||
            !string("qwen35_ocr.profile", "lightonocr3-4b") || !integer("qwen35_ocr.format_version", 5) ||
            !boolean("qwen35_ocr.tie_word_embeddings") || !boolean("qwen35_ocr.zero_centered_text_norm"))
            return nullptr;
        // Validate geometry and distinct EOS declarations before exposing headers.
        const std::pair<const char *, uint32_t> geometry[]{ { "embedding_length", 2560 },
                                                            { "feed_forward_length", 9216 },
                                                            { "block_count", 32 },
                                                            { "vocab_size", 248320 },
                                                            { "attention.head_count", 16 },
                                                            { "attention.head_count_kv", 4 },
                                                            { "attention.key_length", 256 },
                                                            { "attention.value_length", 256 },
                                                            { "attention.full_attention_interval", 4 },
                                                            { "rope.dimension_count", 64 },
                                                            { "linear_attention.key_head_count", 16 },
                                                            { "linear_attention.value_head_count", 32 },
                                                            { "linear_attention.key_length", 128 },
                                                            { "linear_attention.value_length", 128 },
                                                            { "linear_attention.conv_kernel", 4 },
                                                            { "vision.block_count", 24 },
                                                            { "vision.embedding_length", 1024 },
                                                            { "vision.feed_forward_length", 4096 },
                                                            { "vision.attention.head_count", 16 },
                                                            { "vision.patch_size", 16 },
                                                            { "vision.temporal_patch_size", 2 },
                                                            { "vision.spatial_merge_size", 2 },
                                                            { "vision.position_embedding_count", 2304 },
                                                            { "vision.projection_dim", 2560 },
                                                            { "primary_eos_id", 248046 },
                                                            { "source_text_eos_id", 248044 },
                                                            { "source_top_level_eos_id", 248046 } };
        for (const auto & item : geometry)
            if (!integer((std::string("qwen35_ocr.") + item.first).c_str(), item.second)) return nullptr;
        core_gguf::tensor_map tensors;
        for (auto * t = ggml_get_first_tensor(candidate->headers_); t;
             t = ggml_get_next_tensor(candidate->headers_, t)) {
            if (t->data || !tensors.emplace(ggml_get_name(t), t).second) return nullptr;
        }
        if (!map_lighton4b_model(tensors, candidate->model_)) return nullptr;
        return candidate;
    }
};
} // namespace qwen35_ocr
