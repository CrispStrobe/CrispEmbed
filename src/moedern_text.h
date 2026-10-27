#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moedern_text {
enum class role { query, document };
struct result {
    role task = role::query;
    std::vector<int32_t> ids;
    std::vector<uint8_t> supplied_mask, effective_mask, scoring_mask, expansion_positions;
    std::vector<float> all_vectors, selected_vectors; // Contiguous [S,128]/[selected,128].
};
struct context;
// Internal raw-text API only. Requires default-off COMPOSED and existing native/
// frontend gates. CPU/F32, 1.5M only; the tokenizer is a separate metadata GGUF.
context * init(const char * model_gguf, const char * tokenizer_gguf, int threads, std::string & error);
context * init_packed(const char * packed_gguf, int threads, std::string & error);
void free(context * ctx);
const char * error(const context * ctx);
bool encode(context * ctx, const std::string & text, role task, result & output);
// Both model inputs originate from raw text here; no vector/ID injection API.
bool score(context * ctx, const std::string & query, const std::string & document, float & value);
} // namespace moedern_text
