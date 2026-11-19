#ifndef CRISPEMBED_SPACY_LEXICAL_H
#define CRISPEMBED_SPACY_LEXICAL_H

#include "spacy_numeric.h"

#include <memory>
#include <string_view>

namespace crispembed {
namespace spacy {

struct LexicalToken {
    std::string text, norm, prefix, suffix, shape;
    std::size_t byte_start = 0, byte_end = 0;
    std::size_t scalar_start = 0, scalar_end = 0;
    std::size_t utf16_start = 0, utf16_end = 0;
    bool space_after = false;
    bool is_space = false;
    std::uint64_t orth_id = 0;
    FeatureRow features{};
};

// Strict Unicode scalar boundaries, including the terminal byte offset.
// Rejects overlong encodings, surrogate values, incomplete sequences and >U+10FFFF.
std::vector<std::size_t> utf8_boundaries(std::string_view text);

// spaCy 3.8.18 StringStore semantics: reserved symbols first, otherwise
// MurmurHash64A over the original UTF-8 bytes with seed 1. Empty string is 0.
std::uint64_t spacy_string_id(std::string_view text);

// Python `re.search` with spaCy-exported patterns, compiled through the same
// pinned Python-to-PCRE2 translation as the tokenizer (attribute-ruler REGEX).
class PythonRegex {
public:
    explicit PythonRegex(const std::string & pattern);
    ~PythonRegex();
    PythonRegex(PythonRegex &&) noexcept;
    PythonRegex & operator=(PythonRegex &&) noexcept;
    bool search(std::string_view text) const;
    // Python re.findall spans (byte offsets), non-overlapping, left to right.
    std::vector<std::pair<std::size_t, std::size_t>> find_all(std::string_view text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class LexicalModel {
public:
    // The caller must authenticate these bytes against the model resource hash.
    // This constructor also rejects unsupported schema/Unicode/tokenizer policies.
    explicit LexicalModel(const std::string & authenticated_resource_json);
    ~LexicalModel();
    LexicalModel(LexicalModel &&) noexcept;
    LexicalModel & operator=(LexicalModel &&) noexcept;
    LexicalModel(const LexicalModel &) = delete;
    LexicalModel & operator=(const LexicalModel &) = delete;

    // Preserve original source spans and a single ASCII space in space_after.
    // Maximum 1 MiB UTF-8 and 2,048 output tokens; never truncates. No model I/O,
    // network, locale-dependent classification, normalization or Python runtime.
    // Calls on an admitted immutable model are safe from independent threads.
    std::vector<LexicalToken> tokenize(std::string_view text, const CancelCheck & cancelled = {}) const;

    // CPython str.lower() (pinned Unicode tables, contextual final sigma) and
    // str.isalpha() (nonempty, every scalar alphabetic) for arbitrary valid
    // UTF-8, as used by attribute-ruler LOWER patterns and rule lemmatizers.
    std::string python_lower(std::string_view text) const;
    bool python_isalpha(std::string_view text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spacy
} // namespace crispembed

#endif
