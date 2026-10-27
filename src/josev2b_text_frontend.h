#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace josev2b {
struct text_span {
    int start = 0, end = 0;
};
struct text_field {
    std::string question_id;
    int question_type = 0; // Publisher: noul=0, choice=1, score=2.
    text_span question_span;
    std::vector<text_span> option_spans;
    std::vector<std::string> option_ids;
    std::vector<std::vector<int32_t>> lexical_ids;
};
struct encoded_text {
    std::vector<int32_t> input_ids, attention_mask;
    std::vector<text_field> questions;
    std::string record_id;
};
// Dedicated text-only, batch-one frontend. No model weights/backend/public ABI.
// The default-off gate is captured at load; changing the environment does not
// change an existing context. Metadata component must contain zero tensors.
class text_frontend {
public:
    bool load(const char * metadata_gguf, std::string & error);
    // New exact442-tensor profile only; does not relax metadata-only load().
    bool load_packed(const char * packed_gguf, std::string & error);
    bool encode_record(const std::string & request_json, encoded_text & output, std::string & error,
                       int max_length = 512) const;
    bool enabled() const { return enabled_; }

private:
    bool load_file(const char * path, bool packed, std::string & error);
    bool enabled_ = false;
    std::vector<std::string> vocab_;
    std::vector<int32_t> added_;
    std::unordered_map<std::string, int32_t> tokens_, merges_;
    std::vector<int32_t> tokens(const std::string & text) const;
};
} // namespace josev2b
