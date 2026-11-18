// Optional CPU/F32 German spaCy ABI. Prepared-feature scoring remains available
// independently; text services require separately authenticated resources.
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
#define CRISPEMBED_SPACY_TEXT_ABI_VERSION 1
#define CRISPEMBED_SPACY_MAX_TEXT_BYTES (1024u * 1024u)

typedef struct crispembed_spacy_context crispembed_spacy_context;
typedef struct crispembed_spacy_result crispembed_spacy_result;
typedef struct crispembed_spacy_document crispembed_spacy_document;

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
    CRISPEMBED_SPACY_PARSER_STATE_SCORES = 2,
    CRISPEMBED_SPACY_TOKENIZE_UTF8 = 4,
    CRISPEMBED_SPACY_ANALYZE_UTF8 = 8,
    // Context-only: an attached NER component adds entities to analyses.
    CRISPEMBED_SPACY_ENTITIES_UTF8 = 16
};

// Borrowed callback views expire when the callback returns. Nonzero requests
// abortion; no successful partial result is returned. Callbacks must not
// reenter or free this context. Cancellation may read an external atomic flag.
typedef int (*crispembed_spacy_trace_callback)(void * user, const char * stage, const float * values,
                                               const size_t * shape, size_t rank);
typedef int (*crispembed_spacy_cancel_callback)(void * user);

CRISPEMBED_SPACY_API uint32_t crispembed_spacy_abi_version(void);
// Legacy numeric-service mask, retained for existing prepared-feature clients.
CRISPEMBED_SPACY_API uint32_t crispembed_spacy_capabilities(void);
CRISPEMBED_SPACY_API uint32_t crispembed_spacy_text_abi_version(void);
// Returns zero for null. Text capabilities require create_with_resources.
CRISPEMBED_SPACY_API uint32_t crispembed_spacy_context_capabilities(const crispembed_spacy_context * context);

// Requires CRISPEMBED_SPACY=1 and the externally trusted lowercase SHA-256 of
// model.gguf. Never downloads. The host must keep the admitted file immutable
// while opening it. Only the exact 39-tensor source profile and CPU/F32 are
// supported. Errors are NUL-terminated when error_capacity is nonzero.
CRISPEMBED_SPACY_API crispembed_spacy_context * crispembed_spacy_create(const char * model_path,
                                                                        const char * expected_sha256, char * error,
                                                                        size_t error_capacity);

// Extends model admission with resources.json. Both supplied lowercase hashes
// are required; the resource hash must also match the model's closed profile.
// Bytes are bounded, authenticated before parsing, and owned by the context.
// No file is subsequently reread during text processing.
CRISPEMBED_SPACY_API crispembed_spacy_context * crispembed_spacy_create_with_resources(
    const char * model_path, const char * expected_model_sha256, const char * resources_path,
    const char * expected_resources_sha256, char * error, size_t error_capacity);

// Calls on a context are serialized internally. The caller owns its lifetime:
// free only after all active calls/callbacks finish. Results own their arrays
// independently and may outlive the context. The context retains no document
// text after a call; document results own their token strings.
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

// All text fields are counted UTF-8; embedded NUL is valid and length must be
// honored. Views borrow immutable document-owned storage, valid until free.
typedef struct crispembed_spacy_text_view {
    const char * data;
    size_t length;
} crispembed_spacy_text_view;

enum crispembed_spacy_token_flag {
    CRISPEMBED_SPACY_SPACE_AFTER = 1,
    CRISPEMBED_SPACY_IS_SPACE = 2,
    CRISPEMBED_SPACY_SENTENCE_START = 4,
    CRISPEMBED_SPACY_SENTENCE_END = 8
};

typedef struct crispembed_spacy_token_view {
    crispembed_spacy_text_view text;
    crispembed_spacy_text_view norm;
    crispembed_spacy_text_view prefix;
    crispembed_spacy_text_view suffix;
    crispembed_spacy_text_view shape;
    crispembed_spacy_text_view lemma;
    crispembed_spacy_text_view tag;
    crispembed_spacy_text_view pos;
    crispembed_spacy_text_view morph;
    crispembed_spacy_text_view dependency;
    size_t byte_start;
    size_t byte_end;
    size_t scalar_start;
    size_t scalar_end;
    size_t utf16_start;
    size_t utf16_end;
    uint64_t orth_id;
    uint64_t features[CRISPEMBED_SPACY_FEATURE_COUNT];
    // Absolute zero-based head index; roots refer to themselves. Tokenization
    // alone reports -1 and empty lemma/tag/pos/morph/dependency fields.
    int32_t head;
    uint32_t flags;
} crispembed_spacy_token_view;

// Counts refer to the original, unnormalized source. Whitespace after tokens
// follows spaCy's single-ASCII-space convention; other whitespace is retained
// as tokens. Malformed UTF-8, >1 MiB text and >2,048 tokens fail explicitly.
// Empty input succeeds with zero tokens. Failed calls set *document=nullptr;
// cancellation never returns a successful partial document.
CRISPEMBED_SPACY_API int crispembed_spacy_tokenize_utf8(crispembed_spacy_context * context, const char * text,
                                                        size_t text_bytes, crispembed_spacy_document ** document,
                                                        crispembed_spacy_cancel_callback cancel, void * cancel_user,
                                                        char * error, size_t error_capacity);

CRISPEMBED_SPACY_API int crispembed_spacy_analyze_utf8(crispembed_spacy_context * context, const char * text,
                                                       size_t text_bytes, crispembed_spacy_document ** document,
                                                       crispembed_spacy_trace_callback trace, void * trace_user,
                                                       crispembed_spacy_cancel_callback cancel, void * cancel_user,
                                                       char * error, size_t error_capacity);

CRISPEMBED_SPACY_API size_t crispembed_spacy_document_token_count(const crispembed_spacy_document * document);
CRISPEMBED_SPACY_API int crispembed_spacy_document_is_analyzed(const crispembed_spacy_document * document);
// Invalid arguments clear *token when nonnull and return INVALID_ARGUMENT.
CRISPEMBED_SPACY_API int crispembed_spacy_document_token(const crispembed_spacy_document * document, size_t index,
                                                         crispembed_spacy_token_view * token);
// Opt-in named entities (CRISPEMBED_SPACY_NER=1). Attaches the separate
// de_core_news_sm 3.8.0 NER component (ner.gguf, exact closed F32 profile,
// externally trusted lowercase SHA-256) to a context created with resources.
// Subsequent analyze calls also decode entities; tokenize/forward calls and
// existing token views are unchanged. Attaching twice is rejected.
CRISPEMBED_SPACY_API int crispembed_spacy_attach_ner(crispembed_spacy_context * context, const char * ner_path,
                                                     const char * expected_sha256, char * error,
                                                     size_t error_capacity);

// Entity spans use the same offsets as token views; token_end and the
// *_end offsets are exclusive. label is one of LOC, PER, ORG, MISC.
typedef struct crispembed_spacy_entity_view {
    crispembed_spacy_text_view label;
    size_t token_start;
    size_t token_end;
    size_t byte_start;
    size_t byte_end;
    size_t scalar_start;
    size_t scalar_end;
    size_t utf16_start;
    size_t utf16_end;
} crispembed_spacy_entity_view;

// Zero unless the document was analyzed by a context with attached NER.
CRISPEMBED_SPACY_API size_t crispembed_spacy_document_entity_count(const crispembed_spacy_document * document);
// Invalid arguments clear *entity when nonnull and return INVALID_ARGUMENT.
CRISPEMBED_SPACY_API int crispembed_spacy_document_entity(const crispembed_spacy_document * document, size_t index,
                                                          crispembed_spacy_entity_view * entity);
// Documents may outlive their creating context. Free only after view readers
// finish; concurrent readers of an immutable document are allowed.
CRISPEMBED_SPACY_API void crispembed_spacy_document_free(crispembed_spacy_document * document);

#ifdef __cplusplus
}
#endif
#endif
