// Optional CPU/F32 spaCy numeric development ABI. This version consumes
// prepared lexical features; it does not accept raw text or return a parse.
#ifndef CRISPEMBED_SPACY_PIPELINE_H
#define CRISPEMBED_SPACY_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CRISPEMBED_SPACY_SHARED)
#if defined(CRISPEMBED_SPACY_BUILD)
#define CRISPEMBED_SPACY_API __declspec(dllexport)
#else
#define CRISPEMBED_SPACY_API __declspec(dllimport)
#endif
#else
#define CRISPEMBED_SPACY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CRISPEMBED_SPACY_ABI_VERSION 1
#define CRISPEMBED_SPACY_MAX_TOKENS 2048
#define CRISPEMBED_SPACY_FEATURE_COUNT 6

typedef struct crispembed_spacy_context crispembed_spacy_context;
typedef struct crispembed_spacy_result crispembed_spacy_result;

enum crispembed_spacy_status {
    CRISPEMBED_SPACY_OK = 0,
    CRISPEMBED_SPACY_DISABLED = 1,
    CRISPEMBED_SPACY_INVALID_ARGUMENT = 2,
    CRISPEMBED_SPACY_MODEL_ERROR = 3,
    CRISPEMBED_SPACY_ALLOCATION_ERROR = 4,
    CRISPEMBED_SPACY_RUNTIME_ERROR = 5,
    CRISPEMBED_SPACY_CANCELLED = 6,
    CRISPEMBED_SPACY_TRACE_ERROR = 7
};

enum crispembed_spacy_capability {
    CRISPEMBED_SPACY_PREPARED_FEATURE_SCORES = 1,
    CRISPEMBED_SPACY_PARSER_STATE_SCORES = 2
};

// Borrowed callback views expire when the callback returns. Nonzero requests
// abortion; no successful partial result is returned. Callbacks must not
// reenter or free this context. Cancellation may read an external atomic flag.
typedef int (*crispembed_spacy_trace_callback)(void * user, const char * stage, const float * values,
                                               const size_t * shape, size_t rank);
typedef int (*crispembed_spacy_cancel_callback)(void * user);

CRISPEMBED_SPACY_API uint32_t crispembed_spacy_abi_version(void);
CRISPEMBED_SPACY_API uint32_t crispembed_spacy_capabilities(void);

// Requires CRISPEMBED_SPACY=1 and the externally trusted lowercase SHA-256 of
// model.gguf. Never downloads. The host must keep the admitted file immutable
// while opening it. Only the exact 39-tensor source profile and CPU/F32 are
// supported. Errors are NUL-terminated when error_capacity is nonzero.
CRISPEMBED_SPACY_API crispembed_spacy_context * crispembed_spacy_create(const char * model_path,
                                                                        const char * expected_sha256, char * error,
                                                                        size_t error_capacity);

// Calls on a context are serialized internally. The caller owns its lifetime:
// free only after all active calls/callbacks finish. Results own their arrays
// independently and may outlive the context. No document text is retained.
CRISPEMBED_SPACY_API void crispembed_spacy_free(crispembed_spacy_context * context);

// Row-major uint64 IDs in NORM,PREFIX,SUFFIX,SHAPE,SPACY,IS_SPACE order.
// feature_count must equal token_count*6; last two columns must be 0 or 1.
// Zero tokens with zero features is an explicit empty result. On every error
// *result is null. No truncation or hidden segmentation occurs.
CRISPEMBED_SPACY_API int crispembed_spacy_forward_features(crispembed_spacy_context * context,
                                                           const uint64_t * features, size_t token_count,
                                                           size_t feature_count, crispembed_spacy_result ** result,
                                                           crispembed_spacy_trace_callback trace, void * trace_user,
                                                           crispembed_spacy_cancel_callback cancel, void * cancel_user,
                                                           char * error, size_t error_capacity);

// Matrices named tok2vec/tagger/morphologizer/lemmatizer are immutable borrowed
// views until result_free. Scores are unnormalized, not probabilities. This
// API exposes no token spans, lemmas, morphology decisions or dependencies.
CRISPEMBED_SPACY_API int crispembed_spacy_result_matrix(const crispembed_spacy_result * result, const char * name,
                                                        const float ** values, size_t * rows, size_t * columns);
CRISPEMBED_SPACY_API void crispembed_spacy_result_free(crispembed_spacy_result * result);

// Eight IDs in B0,B1,S0,S1,S2,L(B0,1),L(S0,1),R(S0,1) order. -1 means learned
// padding. This scores supplied states only: no action mask or transition is
// inferred. output_capacity is a count of floats and must be at least 137.
// Result and context must use the same admitted model. Caller owns output.
CRISPEMBED_SPACY_API int crispembed_spacy_parser_scores(crispembed_spacy_context * context,
                                                        const crispembed_spacy_result * result,
                                                        const int32_t * state_features, size_t state_feature_count,
                                                        float * output, size_t output_capacity,
                                                        crispembed_spacy_trace_callback trace, void * trace_user,
                                                        crispembed_spacy_cancel_callback cancel, void * cancel_user,
                                                        char * error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
