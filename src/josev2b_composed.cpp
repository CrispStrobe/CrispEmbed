// Apache-2.0 publisher blueprint: Josev-2B@6f0a31c8, joint_schema_model.py
// SHA256 0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3.
#include "josev2b_composed.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "josev2b_answer_serializer.h"
#include "josev2b_backbone.h"
#include "josev2b_head.h"
#include "josev2b_packed_profile.h"
#include "josev2b_text_frontend.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#ifdef CRISPEMBED_JOSEV_COMPOSED_TESTING
#include "../tests/josev2b_composed_observer.h"
#endif
namespace josev2b {
namespace {
void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
bool enabled() {
    return core_env::on("CRISPEMBED_JOSEV_COMPOSED") && core_env::on("CRISPEMBED_JOSEV_BACKBONE") &&
           core_env::on("CRISPEMBED_QWEN35_OCR") && core_env::on("CRISPEMBED_JOSEV_HEAD");
}
bool profile(const char * path) {
    auto * meta = core_gguf::open_metadata(path);
    if (!meta) return false;
    const bool ok = core_gguf::kv_str(meta, "general.architecture", "") == "josev2b_backbone_reference" &&
                    core_gguf::kv_str(meta, "josev2b_backbone_reference.original_receipt_sha256", "") ==
                        josev2b_head::original_receipt_sha256 &&
                    core_gguf::kv_str(meta, "josev2b_backbone_reference.original_worker_source", "") ==
                        josev2b_head::original_worker_source &&
                    core_gguf::kv_u32(meta, "josev2b_backbone_reference.hidden_size", 0) == 2048 &&
                    core_gguf::kv_u32(meta, "josev2b_backbone_reference.intermediate_size", 0) == 6144 &&
                    core_gguf::kv_u32(meta, "josev2b_backbone_reference.layers", 0) == 24 &&
                    core_gguf::kv_u32(meta, "josev2b_backbone_reference.max_validated_tokens", 0) == 512;
    core_gguf::free_metadata(meta);
    return ok;
}
} // namespace
struct composed_context::implementation {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad owner;
    qwen35_ocr::decoder_weights weights;
    text_frontend frontend;
    answer_serializer serializer;
    josev2b_head::context * head = nullptr;
    int threads = 0;
    bool packed = false;
    ~implementation() {
        josev2b_head::free(head);
        core_gguf::free_weights(owner);
        if (backend) ggml_backend_free(backend);
    }
};
composed_context::composed_context() : impl_(new implementation) {}
composed_context::~composed_context() = default;
std::unique_ptr<composed_context> composed_context::load(const char * backbone, const char * head,
                                                         const char * tokenizer, int threads, std::string & error) {
    error.clear();
    try {
        require(enabled(), "Josev composed component gates are off");
        require(backbone && head && tokenizer && threads >= 1 && threads <= 256,
                "Josev composed paths/thread count invalid");
        auto result = std::unique_ptr<composed_context>(new composed_context);
        auto & p = *result->impl_;
        require(p.serializer.enabled(), "Josev answer serializer gate is off");
        // Reject frontend/schema/profile problems before the multi-gigabyte load.
        require(p.frontend.load(tokenizer, error), "Josev tokenizer component rejected");
        require(profile(backbone), "Josev backbone provenance/profile rejected");
        p.head = josev2b_head::init(head, josev2b_head::original_receipt_sha256, josev2b_head::original_worker_source,
                                    "cpu", threads, error);
        require(p.head != nullptr, "Josev head component rejected");
        p.backend = ggml_backend_cpu_init();
        require(p.backend != nullptr, "Josev explicit CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(p.backend, threads);
        require(core_gguf::load_weights(backbone, p.backend, "josev-composed", p.owner, true),
                "Josev backbone load failed");
        require(josev2b_backbone::map(p.owner.tensors, p.weights), "Josev strict320F32/no-head map rejected");
        p.threads = threads;
        return result;
    } catch (const std::exception & e) {
        if (error.empty()) error = e.what();
        return nullptr;
    }
}
std::unique_ptr<composed_context> composed_context::load_packed(const char * path, int threads, std::string & error) {
    error.clear();
    try {
        require(enabled() && core_env::on("CRISPEMBED_JOSEV_PACKED"), "Josev packed/component gates are off");
        require(path && threads >= 1 && threads <= 64, "Josev packed path/1..64 threads required");
        auto result = std::unique_ptr<composed_context>(new composed_context);
        auto & p = *result->impl_;
        require(p.serializer.enabled(), "Josev answer serializer gate is off");
        // Bounds and embedded tokenizer provenance are checked before any weight
        // allocation. Old metadata-only frontend load() remains strict zero-tensor.
        require(p.frontend.load_packed(path, error), "Josev embedded tokenizer rejected");
        auto partition = [](const core_gguf::tensor_map & all, core_gguf::tensor_map & backbone,
                            core_gguf::tensor_map & head) {
            require(all.size() == 442, "Packed owner requires exactly442 tensors");
            for (const auto & item : all) {
                if (item.first.rfind("l.", 0) == 0)
                    backbone.emplace(item);
                else if (item.first.rfind("head.", 0) == 0)
                    head.emplace(item);
                else
                    require(false, "Unknown packed tensor namespace");
            }
            require(backbone.size() == 320 && head.size() == 122, "Packed320/122 partition differs");
        };
        {
            ggml_context * raw_tensors = nullptr;
            const gguf_init_params params{ true, &raw_tensors };
            auto * raw_metadata = gguf_init_from_file(path, params);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(raw_tensors, ggml_free);
            std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(raw_metadata, gguf_free);
            require(metadata && tensors, "Cannot read packed tensor metadata");
            require(josev2b_packed::metadata_valid(metadata.get(), error), "Packed source/profile rejected");
            core_gguf::tensor_map all, backbone, head;
            for (auto * t = ggml_get_first_tensor(tensors.get()); t; t = ggml_get_next_tensor(tensors.get(), t))
                require(all.emplace(ggml_get_name(t), t).second, "Duplicate packed GGML tensor name");
            partition(all, backbone, head);
            qwen35_ocr::decoder_weights validated;
            require(josev2b_backbone::map(backbone, validated), "Packed backbone metadata schema rejected");
            require(josev2b_head::validate_packed_shapes(head, error), "Packed head metadata schema rejected");
        }
        p.backend = ggml_backend_cpu_init();
        require(p.backend, "Josev packed CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(p.backend, threads);
        require(core_gguf::load_weights(path, p.backend, "josev-packed", p.owner, true),
                "Packed F32 weight load failed");
        core_gguf::tensor_map backbone, head;
        partition(p.owner.tensors, backbone, head);
        require(josev2b_backbone::map(backbone, p.weights), "Packed backbone tensor map rejected");
        p.head = josev2b_head::init_borrowed_cpu(head, threads, error);
        require(p.head != nullptr, "Packed borrowed head rejected");
        p.threads = threads;
        p.packed = true;
        return result;
    } catch (const std::exception & e) {
        if (error.empty()) error = e.what();
        return nullptr;
    }
}
bool composed_context::decide(const std::string & request, std::string & response, std::string & error) {
    response.clear();
    composed_result result;
    if (!decide(request, result, error)) return false;
    response = std::move(result.response_json);
    return true;
}
bool composed_context::decide(const std::string & request, composed_result & result, std::string & error) {
    result = {};
    error.clear();
    try {
        require(enabled(), "Josev composed component gates are off");
        auto & p = *impl_;
        require(!p.packed || core_env::on("CRISPEMBED_JOSEV_PACKED"), "Josev packed gate is off");
        encoded_text encoded;
        require(p.frontend.encode_record(request, encoded, error), "Josev request rejected");
        const std::vector<int> ids(encoded.input_ids.begin(), encoded.input_ids.end());
        qwen35_ocr::decoder_outputs backbone;
        require(josev2b_backbone::run(p.weights, ids, p.threads, backbone), "Josev native backbone failed");
        require(backbone.norm.size() == ids.size() * 2048 && backbone.logits.empty(),
                "Josev native hidden-only output shape differs");
        josev2b_head::input input;
        input.tokens = int(ids.size());
        input.hidden = backbone.norm;
        for (const auto & q : encoded.questions) {
            josev2b_head::field f;
            f.id = q.question_id;
            f.type = q.question_type;
            f.question = { q.question_span.start, q.question_span.end };
            require(q.option_spans.size() == q.lexical_ids.size(), "Josev lexical option count differs");
            for (size_t option = 0; option < q.option_spans.size(); ++option) {
                const auto span = q.option_spans[option];
                require(span.start >= 0 && span.end > span.start && span.end <= int(ids.size()),
                        "Josev lexical span bounds differ");
                const std::vector<int> lexical(q.lexical_ids[option].begin(), q.lexical_ids[option].end());
                require(lexical == std::vector<int>(ids.begin() + span.start, ids.begin() + span.end),
                        "Josev lexical IDs differ from actual input span");
                std::vector<float> rows;
                require(qwen35_ocr::embed_cpu(p.weights.embedding, lexical, rows), "Josev lexical embedding failed");
                f.options.push_back({ span.start, span.end });
                for (size_t column = 0; column < 2048; ++column) {
                    float sum = 0;
                    for (size_t token = 0; token < lexical.size(); ++token) sum += rows[token * 2048 + column];
                    const float mean = sum / float(lexical.size());
                    require(std::isfinite(mean), "Josev lexical mean nonfinite");
                    f.lexical_means.push_back(mean);
                }
            }
            input.fields.push_back(std::move(f));
        }
        josev2b_head::output output;
#ifdef CRISPEMBED_JOSEV_COMPOSED_TESTING
        require(josev2b_head::test::run_observed(p.head, input, output, composed_test::observe_head, error),
                "Josev observed native head failed");
#else
        require(josev2b_head::run(p.head, input, output, error), "Josev native head failed");
#endif
        require(output.fields.size() == encoded.questions.size(), "Josev native output field count differs");
        std::vector<std::vector<float>> probabilities;
        for (size_t i = 0; i < output.fields.size(); ++i) {
            require(output.fields[i].id == encoded.questions[i].question_id, "Josev output field order differs");
            probabilities.push_back(output.fields[i].probabilities);
        }
        std::string candidate;
        require(p.serializer.serialize(request, encoded, probabilities, candidate, error),
                "Josev answer serialization failed");
#ifdef CRISPEMBED_JOSEV_COMPOSED_TESTING
        std::vector<float> embedding;
        require(qwen35_ocr::embed_cpu(p.weights.embedding, ids, embedding), "Observed embedding failed");
        composed_test::observe(encoded, embedding, backbone, input, output, candidate);
#endif
        result.response_json = std::move(candidate);
        result.native_fields = std::move(output);
        return true;
    } catch (const std::exception & e) {
        if (error.empty()) error = e.what();
        return false;
    }
}
} // namespace josev2b
