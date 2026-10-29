#pragma once
#include "core/env_gate.h"
#include "core/gguf_file_type.h"
#include "core/gguf_loader.h"
#include "tokenizer.h"
#include "ultrabert_tokenizer_mode.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ultrabert_profile {
constexpr int max_tokens = 512;
constexpr size_t max_text_bytes = 65536;
inline const char * query_prefix() {
    return "Instruct: Given a question, retrieve passages that answer the question\nQuery: ";
}
inline void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
// The pinned checkpoint has 218 F32 weights. Validate before backend allocation.
inline std::map<std::string, std::pair<int64_t, int64_t>> schema() {
    std::map<std::string, std::pair<int64_t, int64_t>> result = { { "token_embd.weight", { 768, 49152 } },
                                                                  { "output_norm.weight", { 768, 1 } } };
    for (int i = 0; i < 24; ++i) {
        const auto p = "blk." + std::to_string(i) + ".";
        for (const auto * name : { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight" })
            result[p + name] = { 768, 768 };
        for (const auto * name : { "attn_norm.weight", "ffn_norm.weight" }) result[p + name] = { 768, 1 };
        for (const auto * name : { "ffn_gate.weight", "ffn_up.weight" }) result[p + name] = { 768, 1024 };
        result[p + "ffn_down.weight"] = { 1024, 768 };
    }
    return result;
}
inline void validate(gguf_context * g, bool quantized = false) {
    require(g != nullptr, "missing metadata");
    auto str = [&](const char * key) { return core_gguf::kv_str(g, key, ""); };
    auto u = [&](const char * key, uint32_t value) {
        require(core_gguf::kv_u32(g, key, UINT32_MAX) == value, std::string("unsupported ") + key);
    };
    require(str("general.architecture") == "ultrabert" &&
                str("ultrabert.source.revision") == "367d5343b5a0164d57b810cedf5de4b8983d5984" &&
                str("ultrabert.source.weight_sha256") ==
                    "da8be29564b69e4f90fc8f2683e1f369c3d3575e75f7fe798643e28de02e2b9c",
            "unsupported source profile");
    const auto file_type = core_gguf::kv_u32(g, "general.file_type", UINT32_MAX);
    if (quantized) {
        const auto key = gguf_find_key(g, "general.file_type");
        require(key >= 0 && gguf_get_kv_type(g, key) == GGUF_TYPE_UINT32, "quantized general.file_type must be UINT32");
        require(core_gguf_file_type::known_q4_k(file_type) || file_type == core_gguf_file_type::q8_0,
                "unsupported quantized general.file_type");
    } else
        u("general.file_type", GGML_FTYPE_ALL_F32);
    u("ultrabert.context_length", 32768);
    u("ultrabert.embedding_length", 768);
    u("ultrabert.block_count", 24);
    u("ultrabert.feed_forward_length", 1024);
    u("ultrabert.attention.head_count", 6);
    u("ultrabert.attention.head_count_kv", 6);
    u("ultrabert.attention.key_length", 128);
    u("ultrabert.attention.value_length", 128);
    require(!core_gguf::kv_bool(g, "ultrabert.attention.causal", true) &&
                core_gguf::kv_f32(g, "ultrabert.attention.layer_norm_rms_epsilon", 0) == 1e-6f &&
                str("ultrabert.rope.layout") == "split-half" && str("ultrabert.pooling") == "cls" &&
                core_gguf::kv_bool(g, "ultrabert.pooling.include_prompt", false) &&
                !core_gguf::kv_bool(g, "ultrabert.pooling.normalize_default", true) &&
                str("ultrabert.prompt.query") == query_prefix() && str("ultrabert.prompt.default").empty() &&
                str("ultrabert.prompt.document").empty(),
            "unsupported attention/pooling/prompt profile");
    const auto windows = core_gguf::kv_i32_array(g, "ultrabert.attention.window_radius");
    const auto thetas = core_gguf::kv_f32_array(g, "ultrabert.rope.freq_base_per_layer");
    require(windows.size() == 24 && thetas.size() == 24, "missing layer schedule");
    for (int i = 0; i < 24; ++i)
        require(windows[i] == (i % 3 == 2 ? -1 : 128) && thetas[i] == (i % 3 == 2 ? 2560000.f : 10000.f),
                "unsupported layer schedule");
    require(str("tokenizer.ggml.model") == "gpt2" && str("tokenizer.ggml.pre") == "gpt2" &&
                str("ultrabert.tokenizer.single_template") == "<|endoftext|> $A" &&
                str("ultrabert.tokenizer.pair_template") == "<|endoftext|> $A <|sep|> $B" &&
                core_gguf::kv_bool(g, "tokenizer.ggml.add_bos_token", false) &&
                !core_gguf::kv_bool(g, "tokenizer.ggml.add_eos_token", true),
            "unsupported tokenizer template");
    for (auto * key :
         { "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id" })
        u(key, 0);
    u("tokenizer.ggml.separator_token_id", 15);
    u("tokenizer.ggml.mask_token_id", 16);
    const auto expected = schema();
    require(gguf_get_n_tensors(g) == (int64_t)expected.size(), "unexpected tensor count");
    for (const auto & item : expected) {
        const auto index = gguf_find_tensor(g, item.first.c_str());
        require(index >= 0, "missing tensor: " + item.first);
        const auto * ne = gguf_get_tensor_ne(g, index);
        const auto type = gguf_get_tensor_type(g, index);
        const bool critical = item.first == "token_embd.weight" || item.first.find("norm.weight") != std::string::npos;
        const bool allowed =
            type == GGML_TYPE_F32 ||
            (quantized && !critical && (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K));
        require(allowed && ne[0] == item.second.first && ne[1] == item.second.second && ne[2] == 1 && ne[3] == 1,
                "unsupported tensor shape/precision: " + item.first);
    }
    if (quantized) {
        uint64_t actual_q8 = 0, actual_q4 = 0;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            actual_q8 += gguf_get_tensor_type(g, i) == GGML_TYPE_Q8_0;
            actual_q4 += gguf_get_tensor_type(g, i) == GGML_TYPE_Q4_K;
        }
        require(core_gguf_file_type::matches_quantized_payload(file_type, actual_q8, actual_q4, true),
                "Quantized header requires matching actual Q8_0/Q4_K tensors");
    }
}
inline bool load_tokenizer(gguf_context * g, BPETokenizer & raw, std::vector<std::string> & vocab) {
    vocab = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    const auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    if (vocab.size() != 49152 || merges.size() != 48900 || vocab[0] != "<|endoftext|>" || vocab[15] != "<|sep|>" ||
        vocab[16] != "<|mask|>")
        return false;
    if (!raw.load(vocab, merges, -1, 0, -1, -1, false, 1048576)) return false;
    configure_tokenizer_mode(raw);
    return true;
}
inline bool tokenize(const BPETokenizer & raw, const std::string & text, std::vector<int32_t> & ids) {
    ids.clear();
    // Literal special-token matching is not covered by the initial BPE profile.
    // Fail explicitly rather than encoding a reserved surface as ordinary pieces.
    if (text.size() > max_text_bytes || text.find("<|") != std::string::npos) return false;
    const auto encoded = raw.encode(text);
    // Explicit, UltraBERT-only single-text right truncation. Keep BOS attended
    // and count it inside the 512-token bound, matching the original wrapper.
    if (encoded.ids.size() >= max_tokens && !core_env::on("CRISPEMBED_ULTRABERT_TRUNCATE")) return false;
    ids.push_back(0);
    const size_t count = std::min(encoded.ids.size(), size_t(max_tokens - 1));
    ids.insert(ids.end(), encoded.ids.begin(), encoded.ids.begin() + count);
    return true;
}
} // namespace ultrabert_profile
