#pragma once
#include "moedern_colbert.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace moedern_colbert {
struct text_tokens {
    std::vector<int32_t> ids;
    std::vector<uint8_t> supplied_mask, effective_mask, scoring_mask, expansion_positions;
};
// Dedicated 1.5M text/role policy, no model/backend/public ABI dependency.
// Snapshots default-off CRISPEMBED_MOEDERN_TEXT_FRONTEND at load.
class text_frontend {
public:
    bool load(const char * tokenizer_component, std::string & error);
    bool load_packed(const char * packed_gguf, std::string & error);
    bool load_packed_q4k(const char * packed_gguf, std::string & error);
    bool encode(const std::string & text, role task, text_tokens & output, std::string & error) const;
    bool enabled() const { return enabled_; }
    // Source revision of a packed family (27M/115M) model; empty for 1.5M and
    // for every load made while CRISPEMBED_MOEDERN_FAMILY is off.
    const std::string & family_revision() const { return family_revision_; }
#ifdef CRISPEMBED_MOEDERN_TEXT_TESTING
    std::string test_normalize(const std::string & text) const;
    std::vector<std::string> test_pretokens(const std::string & normalized) const;
    std::vector<int32_t> test_unigram(const std::string & segment) const;
#endif
private:
    bool load_file(const char *, bool packed, std::string & error, bool padded_q4k = false);
    struct node {
        std::map<unsigned char, int> next;
        int id = -1;
    };
    bool enabled_ = false;
    std::string family_revision_;
    std::vector<std::string> vocab_, added_content_, normalized_content_;
    std::vector<double> scores_;
    std::vector<int32_t> added_ids_, added_flags_, normalized_ids_, skip_;
    std::unordered_map<uint32_t, std::string> normalization_;
    std::vector<node> trie_;
    double unknown_score_ = 0;
    std::string normalize(const std::string & text) const;
    std::vector<std::string> pretokens(const std::string & normalized) const;
    void unigram(const std::string & segment, std::vector<int32_t> & ids) const;
    void normalized(const std::string & text, std::vector<int32_t> & ids) const;
    void raw(const std::string & text, std::vector<int32_t> & ids) const;
};
} // namespace moedern_colbert
