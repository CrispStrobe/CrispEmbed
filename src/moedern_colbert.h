#pragma once
// Experimental token-ID API only. Not registered in the public CrispEmbed ABI.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace moedern_colbert {
enum class role { query, document };
enum class scalar { f32, i64 };
struct tensor_view {
    const void * data = nullptr;
    scalar type = scalar::f32;
    std::vector<int64_t> shape; // C/NumPy order, including batch=1 where applicable.
    size_t bytes = 0;
};
// Synchronous, read-only borrowed snapshot. Copy before callback return.
// Returning false rejects this encode; no observer is stored in the context.
using observer = std::function<bool(const std::string &, const tensor_view &)>;
struct result {
    std::vector<float> vectors; // All S*128 normalized rows, including masked positions.
    std::vector<uint8_t> supplied_mask, effective_mask;
};
struct context;
// CRISPEMBED_MOEDERN_COLBERT must be value-parsed ON at init and encode.
// Only the separately declared 1.5M strict-F32 GGUF is accepted.
context * init(const char * path, int threads, std::string & error);
void free(context * ctx);
const char * error(const context * ctx);
bool encode_tokens(context * ctx, const int32_t * ids, const uint8_t * supplied_mask, size_t count, role task,
                   result & out, const observer & dump = {});
// Explicit scoring policy, separate from backbone attention. skip_ids must come
// from the pinned original tokenizer/config resolver, never fitted from outputs.
bool scoring_mask(const std::vector<int32_t> & ids, const std::vector<uint8_t> & supplied, role task,
                  const std::vector<int32_t> & skip_ids, std::vector<uint8_t> & scoring, std::string & error);
bool select_rows(const std::vector<float> & all, const std::vector<uint8_t> & mask, std::vector<float> & selected);
bool maxsim(const std::vector<float> & query_rows, const std::vector<float> & document_rows, float & score);
} // namespace moedern_colbert
