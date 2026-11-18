#ifndef CRISPEMBED_SPACY_NER_H
#define CRISPEMBED_SPACY_NER_H

#include "spacy_numeric.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_gallocr;
typedef struct ggml_backend * ggml_backend_t;

// Opt-in named-entity recognizer for de_core_news_sm 3.8.0 (CRISPEMBED_SPACY_NER,
// separate ner.gguf). Inference semantics: spaCy release-v3.8.18
// spacy/pipeline/_parser_internals/{ner.pyx,_state.pxd} and
// spacy/ml/parser_model.pyx (MIT). The component owns an independent tok2vec
// (NORM, PREFIX, SUFFIX, SHAPE; width 96; depth 4; maxout 3) computed as one
// ggml graph, a precomputed lower layer (nF=3: B0, E0, B0-1) and a scalar
// greedy BILUO decoder: first maximum among valid classes, entities never
// cross a parser sentence start or begin on whitespace. No preset entities.
namespace crispembed {
namespace spacy {

struct Entity {
    std::uint32_t start = 0;  // first token
    std::uint32_t end = 0;    // one past the last token
    std::uint32_t label = 0;  // index into NerModel::labels()
};

class NerModel {
public:
    // Weights (F32, "ner." names) stay owned by the caller and resident in a
    // buffer usable by the CPU backend for this object's lifetime.
    NerModel(std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend);
    ~NerModel();
    NerModel(const NerModel &) = delete;
    NerModel & operator=(const NerModel &) = delete;

    static const std::vector<std::string> & labels();  // LOC, PER, ORG, MISC

    // features use columns 0-3 (NORM..SHAPE) and 5 (IS_SPACE); sentence_starts
    // are the parser's final boundaries (token 0 is always a start).
    // Not reentrant; callers serialize. Empty input returns no entities.
    std::vector<Entity> run(const std::vector<FeatureRow> & features, const std::vector<std::uint8_t> & sentence_starts,
                            const TraceSink & trace = {}, const CancelCheck & cancelled = {}) const;

private:
    ggml_tensor * weight(const std::string & name) const;
    std::map<std::string, ggml_tensor *> weights_;
    ggml_backend_t backend_;
    mutable ggml_gallocr * allocator_ = nullptr;
    std::vector<float> lower_bias_, lower_pad_, upper_w_, upper_b_;
};

} // namespace spacy
} // namespace crispembed

#endif // CRISPEMBED_SPACY_NER_H
