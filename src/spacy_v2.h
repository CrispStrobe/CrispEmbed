#ifndef CRISPEMBED_SPACY_V2_H
#define CRISPEMBED_SPACY_V2_H

#include "spacy_decode.h"
#include "spacy_lexical.h"
#include "spacy_ner.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

// Version-2 spaCy pipelines (models/convert-spacy-pipeline-to-gguf.py): the
// official spaCy 3.8 `*_core_*_sm` CNN family beyond the frozen German v1
// profile. Components run in the source pipeline order:
//   tok2vec -> tagger? -> morphologizer? -> parser -> attribute_ruler ->
//   lemmatizer (rule: English/French) -> ner?
// Inference semantics: spaCy release-v3.8.18 pipeline/{tagger,morphologizer,
// attributeruler,lemmatizer}.py, lang/{en,fr}/lemmatizer.py and
// matcher/matcher.pyx (MIT). The German v1 path (spacy_decode.h) is unchanged.
namespace crispembed {
namespace spacy {

class PipelineV2 {
public:
    // `resources_json` must already be authenticated against the model's
    // pinned resource hash. Weights (F32) stay owned by the caller and resident
    // for this object's lifetime, as does `lexical` (Python lower/isalpha).
    PipelineV2(const std::string & resources_json, std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend,
               const LexicalModel & lexical);
    ~PipelineV2();
    PipelineV2(const PipelineV2 &) = delete;
    PipelineV2 & operator=(const PipelineV2 &) = delete;

    const std::string & language() const;
    bool has_ner() const;

    struct Output {
        DecodeOutput annotations;
        std::vector<Entity> entities;
        std::vector<std::string> entity_labels;
    };

    // Not reentrant; callers serialize (the pipeline holds its context mutex).
    // Empty input returns an empty output. Bounds as in decode_annotations.
    Output run(const std::vector<LexicalToken> & tokens, const TraceSink & trace = {},
               const CancelCheck & cancelled = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spacy
} // namespace crispembed

#endif // CRISPEMBED_SPACY_V2_H
