#ifndef CRISPEMBED_SPACY_NUMERIC_H
#define CRISPEMBED_SPACY_NUMERIC_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Internal inference-only contract for de_core_news_sm 3.8.0. This module
// consumes exact lexical feature IDs; it does not tokenize or claim complete
// linguistic analysis. The public model loader owns admission and provenance.
namespace crispembed {
namespace spacy {

struct TensorView {
    std::vector<std::size_t> shape;
    const float * data = nullptr;
    std::size_t size = 0;
};

using TensorMap = std::unordered_map<std::string, TensorView>;
using FeatureRow = std::array<std::uint64_t, 6>;

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> data;

    TensorView view() const { return { { rows, cols }, data.data(), data.size() }; }
};

// Views supplied to the sink are borrowed only for the duration of the call.
// A sink that retains a trace must copy it. No tracing occurs when it is empty.
using TraceSink = std::function<void(const std::string &, const TensorView &)>;
using CancelCheck = std::function<bool()>;

class NumericCancelled : public std::runtime_error {
public:
    NumericCancelled() : std::runtime_error("spaCy numeric inference cancelled") {}
};

struct NumericOutput {
    Matrix tok2vec;
    Matrix tagger;
    Matrix morphologizer;
    Matrix lemmatizer;
    // Row 0 is learned parser padding. Row i+1 belongs to input token i.
    // Columns flatten [feature=8, hidden=64, maxout_piece=2], in that order.
    Matrix parser_cache;
};

// Thinc 8.3.14 uint64 specialization. All arithmetic wraps at 64 bits and
// returns the four ordered 32-bit keys before modulo by embedding table rows.
std::array<std::uint32_t, 4> hash_keys(std::uint64_t value, std::uint32_t seed);

class NumericModel {
public:
    // Weights remain borrowed and immutable for this object's entire lifetime,
    // including every active run. Validates exact supported tensor shapes,
    // counts and finite values; throws std::invalid_argument on a mismatch.
    explicit NumericModel(TensorMap tensors);

    // Computes one document. Padding belongs to this document and evolves
    // across all four residual layers. Empty features produce empty outputs.
    // max_tokens is an admission bound, not a measured latency guarantee.
    NumericOutput run(const std::vector<FeatureRow> & features, std::size_t max_tokens, const TraceSink & trace = {},
                      const CancelCheck & cancelled = {}) const;

    // Scores one parser state using the cache from the same document/model.
    // Feature IDs must be -1 (learned padding) or valid zero-based token IDs,
    // ordered B0,B1,S0,S1,S2,L(B0,1),L(S0,1),R(S0,1). This method does not
    // decide action legality or implement dependency transitions.
    std::vector<float> parser_scores(const Matrix & cache, const std::array<std::int32_t, 8> & feature_ids,
                                     const TraceSink & trace = {}, const CancelCheck & cancelled = {}) const;

private:
    TensorMap tensors_;
    const TensorView & tensor(const std::string & name) const;
};

} // namespace spacy
} // namespace crispembed

#endif // CRISPEMBED_SPACY_NUMERIC_H
