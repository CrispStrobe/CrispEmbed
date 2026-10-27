// Test-only full-prefill observations. No new arithmetic/production observers.
#include "josev2b_backbone.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/json.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
uint32_t integer(std::istream & input) {
    unsigned char b[4];
    input.read(reinterpret_cast<char *>(b), 4);
    require(bool(input), "Truncated packet integer");
    return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
}
std::string bytes(std::istream & input, size_t count) {
    std::string result(count, '\0');
    input.read(result.data(), count);
    require(bool(input), "Truncated packet text");
    return result;
}
struct input_case {
    std::string id;
    std::vector<int> ids;
};
std::vector<input_case> packet(const char * path) {
    std::ifstream input(path, std::ios::binary);
    require(bytes(input, 8) == std::string("JOSBKV1\0", 8), "Wrong input packet magic");
    require(bytes(input, 64) == "51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090",
            "Wrong original receipt");
    require(bytes(input, 40) == "b2edf8c83c9aca1f8a9210198d47798ac13828b8", "Wrong original worker");
    require(bytes(input, 64) == "9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7",
            "Wrong input freeze");
    require(integer(input) == 4, "All four cases required");
    const std::vector<std::string> names = { "mixed-de", "null-and-options", "json-unicode", "state-only-clip" };
    const std::vector<int> lengths = { 319, 197, 165, 512 };
    std::vector<input_case> result;
    for (size_t i = 0; i < 4; ++i) {
        const uint32_t n = integer(input), tokens = integer(input);
        require(n == names[i].size() && tokens == uint32_t(lengths[i]), "Frozen case order/length differs");
        input_case row{ bytes(input, n), {} };
        require(row.id == names[i], "Frozen case identity differs");
        for (uint32_t j = 0; j < tokens; ++j) {
            const auto id = integer(input);
            require(id < 248320, "Token ID outside exact vocabulary");
            row.ids.push_back(int(id));
        }
        require(bytes(input, tokens) == std::string(tokens, '\1'), "Only all-attended unpadded input is qualified");
        result.push_back(std::move(row));
    }
    require(input.peek() == std::char_traits<char>::eof(), "Extra packet bytes");
    return result;
}
bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
void finite(const std::vector<float> & data, size_t count) {
    require(data.size() == count && std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); }),
            "Incomplete/nonfinite native F32 stage");
}
std::string quoted(const std::string & text) {
    return "\"" + core_json::json_escape_strict(text) + "\"";
}
struct weights_owner {
    core_gguf::WeightLoad weights;
    ~weights_owner() { core_gguf::free_weights(weights); }
};
int body(int argc, char ** argv) {
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_JOSEV_BACKBONE")) return 2;
    const bool contracts = argc == 3 && std::string(argv[1]) == "--contracts";
    if (!contracts && argc != 4) return 2;
    ggml_backend_t backend = nullptr;
    try {
        const auto inputs = packet(argv[2]); // Validate exact frozen adapter before model access.
        if (contracts) {
            std::puts("PASS exact four-case token/mask packet contracts; no model access");
            return 0;
        }
        const std::filesystem::path output(argv[3]);
        require(!std::filesystem::exists(output), "Fresh task-owned probe output required");
        std::filesystem::create_directories(output);
        auto * metadata = core_gguf::open_metadata(argv[1]);
        require(metadata, "Backbone GGUF metadata missing");
        const bool profile = core_gguf::kv_str(metadata, "general.architecture", "") == "josev2b_backbone_reference" &&
                             core_gguf::kv_str(metadata, "josev2b_backbone_reference.original_receipt_sha256", "") ==
                                 "51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090" &&
                             core_gguf::kv_str(metadata, "josev2b_backbone_reference.original_worker_source", "") ==
                                 "b2edf8c83c9aca1f8a9210198d47798ac13828b8" &&
                             core_gguf::kv_u32(metadata, "josev2b_backbone_reference.hidden_size", 0) == 2048 &&
                             core_gguf::kv_u32(metadata, "josev2b_backbone_reference.intermediate_size", 0) == 6144 &&
                             core_gguf::kv_u32(metadata, "josev2b_backbone_reference.layers", 0) == 24 &&
                             core_gguf::kv_u32(metadata, "josev2b_backbone_reference.max_validated_tokens", 0) == 512;
        core_gguf::free_metadata(metadata);
        require(profile, "Exact original-only GGUF profile required");
        backend = ggml_backend_cpu_init();
        require(backend, "Explicit CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, 2);
        bool failed = false;
        {
            weights_owner owner;
            require(core_gguf::load_weights(argv[1], backend, "josev-backbone-probe", owner.weights, true),
                    "Backbone load failed");
            qwen35_ocr::decoder_weights weights;
            require(josev2b_backbone::map(owner.weights.tensors, weights) && !weights.head,
                    "Strict320F32/no-head map rejected");
            std::ofstream report(output / "native.json");
            require(bool(report), "Native report open failed");
            report << "{\"schema\":\"crispembed-josev-backbone-native-v1\",\"backend\":\"CPU\",\"cases\":[";
            for (size_t c = 0; c < inputs.size(); ++c) {
                if (c) report << ',';
                const auto & input = inputs[c];
                report << "{\"id\":" << quoted(input.id) << ",\"tokens\":" << input.ids.size();
                try {
                    const size_t count = input.ids.size() * 2048;
                    std::vector<float> embedding, repeated_embedding;
                    require(qwen35_ocr::embed_cpu(weights.embedding, input.ids, embedding), "Embedding failed");
                    finite(embedding, count);
                    qwen35_ocr::decoder_outputs ordinary, repeated;
                    require(josev2b_backbone::run(weights, input.ids, 2, ordinary), "Ordinary hidden prefill failed");
                    require(qwen35_ocr::embed_cpu(weights.embedding, input.ids, repeated_embedding),
                            "Repeated embedding failed");
                    require(same(embedding, repeated_embedding), "Embedding repeat changed bits");
                    require(josev2b_backbone::run(weights, input.ids, 2, repeated), "Repeated hidden prefill failed");
                    require(ordinary.logits.empty() && repeated.logits.empty() && ordinary.hidden.size() == 24 &&
                                repeated.hidden.size() == 24,
                            "Unexpected head output or missing layer rows");
                    std::vector<const std::vector<float> *> values{ &embedding };
                    for (size_t i = 0; i < 24; ++i) {
                        finite(ordinary.hidden[i], count);
                        finite(repeated.hidden[i], count);
                        require(same(ordinary.hidden[i], repeated.hidden[i]), "Ordinary/repeat layer bits differ");
                        values.push_back(&ordinary.hidden[i]);
                    }
                    finite(ordinary.norm, count);
                    finite(repeated.norm, count);
                    require(same(ordinary.norm, repeated.norm), "Ordinary/repeat final norm bits differ");
                    values.push_back(&ordinary.norm);
                    char folder[16];
                    std::snprintf(folder, sizeof(folder), "%02zu", c);
                    std::filesystem::create_directory(output / folder);
                    std::string stages;
                    for (size_t i = 0; i < values.size(); ++i) {
                        const std::string label = i == 0    ? "backbone/embedding"
                                                  : i == 25 ? "backbone/final_norm"
                                                            : "backbone/layer/" + std::to_string(i - 1);
                        char filename[32];
                        std::snprintf(filename, sizeof(filename), "%02zu/%02zu.f32", c, i);
                        std::ofstream raw(output / filename, std::ios::binary);
                        raw.write(reinterpret_cast<const char *>(values[i]->data()), count * sizeof(float));
                        require(bool(raw), "Native raw stage write failed");
                        if (i) stages += ',';
                        stages += "{\"stage\":" + quoted(label) + ",\"occurrence\":0,\"shape\":[1," +
                                  std::to_string(input.ids.size()) + ",2048],\"file\":" + quoted(filename) + "}";
                    }
                    report
                        << ",\"completed\":true,\"ordinary_repeat_bit_exact\":true,\"logits_empty\":true,\"stages\":["
                        << stages << ']';
                } catch (const std::exception & e) {
                    failed = true;
                    report << ",\"completed\":false,\"error\":" << quoted(e.what());
                }
                report << '}';
                report.flush();
            }
            report << "]}\n";
            require(bool(report), "Native report write failed");
            // Weights and every mapped view die before their explicit backend.
            ggml_backend_synchronize(backend);
            if (!failed)
                std::puts("PASS four-case104stage ordinary/repeat bit identity, finite F32, no vocabulary projection");
        }
        ggml_backend_free(backend);
        backend = nullptr;
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        if (backend) ggml_backend_free(backend);
        std::fprintf(stderr, "Backbone probe structural rejection: %s\n", e.what());
        return 2;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
