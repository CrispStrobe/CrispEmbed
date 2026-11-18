#ifndef CRISPEMBED_SPACY_GGML_H
#define CRISPEMBED_SPACY_GGML_H

#include "spacy_numeric.h"

#include <map>
#include <string>

struct ggml_tensor;
struct ggml_gallocr;
typedef struct ggml_backend * ggml_backend_t;

// Opt-in ggml graph engine for the de_core_news_sm 3.8.0 document pass
// (CRISPEMBED_SPACY_GGML). It computes the same NumericOutput and emits the
// same trace stage names/shapes as NumericModel::run, but with ggml_get_rows
// hash embedding, ggml_mul_mat affines, ggml_pool_1d maxout and ggml_norm.
// Matrix weights may be F32, F16, BF16, Q8_0, Q5_0 or Q4_0; ggml_mul_mat then
// rounds activations to the weight's dot type, which is part of what the
// precision A/B measures. Reduction order differs from the scalar reference,
// so results are not bit-identical even at F32. Parser state scoring stays
// with NumericModel::parser_scores (F32 parser tensors).
namespace crispembed {
namespace spacy {

class GgmlNumeric {
public:
    // Weights stay owned by the caller and must remain resident in a buffer
    // usable by `backend` (a CPU backend) for this object's lifetime.
    GgmlNumeric(std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend);
    ~GgmlNumeric();
    GgmlNumeric(const GgmlNumeric &) = delete;
    GgmlNumeric & operator=(const GgmlNumeric &) = delete;

    // Not reentrant: callers serialize runs (the pipeline holds its context mutex).
    // Empty input is rejected; the scalar model owns the empty-document contract.
    NumericOutput run(const std::vector<FeatureRow> & features, std::size_t max_tokens, const TraceSink & trace = {},
                      const CancelCheck & cancelled = {}) const;

private:
    ggml_tensor * weight(const std::string & name) const;
    std::map<std::string, ggml_tensor *> weights_;
    ggml_backend_t backend_;
    mutable ggml_gallocr * allocator_ = nullptr;
};

} // namespace spacy
} // namespace crispembed

#endif // CRISPEMBED_SPACY_GGML_H
