// crispembed.h — C API for text embedding inference via ggml.
//
// Usage:
//   crispembed_context * ctx = crispembed_init("model.gguf", 4);
//   float * vec = crispembed_encode(ctx, "Hello world", &n_dim);
//   // vec is [n_dim] L2-normalized embedding
//   crispembed_free(ctx);

#pragma once

#include <stdint.h>

// DLL export/import on Windows.
// - CRISPEMBED_BUILD:  defined when building the shared library (exports)
// - CRISPEMBED_SHARED: defined when consuming the shared library (imports)
// - neither:           static library use (empty, no attribute)
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(CRISPEMBED_BUILD)
#    define CRISPEMBED_API __declspec(dllexport)
#  elif defined(CRISPEMBED_SHARED)
#    define CRISPEMBED_API __declspec(dllimport)
#  else
#    define CRISPEMBED_API
#  endif
#else
#  if defined(CRISPEMBED_BUILD)
#    define CRISPEMBED_API __attribute__((visibility("default")))
#  else
#    define CRISPEMBED_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct crispembed_context;

// Model hyperparameters (read-only after init)
struct crispembed_hparams {
    int32_t n_vocab;
    int32_t n_max_tokens;    // max sequence length
    int32_t n_embd;          // embedding dimension (hidden size)
    int32_t n_head;          // attention heads
    int32_t n_layer;         // transformer layers
    int32_t n_intermediate;  // FFN intermediate size
    int32_t n_output;        // output embedding dimension (may differ from n_embd)
    float   layer_norm_eps;
};

// Initialize: load GGUF model, allocate ggml backends.
// n_threads: CPU threads for matmul (0 = auto).
// Returns NULL on failure.
CRISPEMBED_API crispembed_context * crispembed_init(const char * model_path, int n_threads);

// Set Matryoshka output dimension. 0 = use model default.
// Must be <= model's native dimension. The embedding is truncated
// and re-normalized to the specified dimension.
CRISPEMBED_API void crispembed_set_dim(crispembed_context * ctx, int dim);

// Set a text prefix prepended to all inputs before tokenization.
// Pass NULL or "" to clear. Typical values:
//   "query: "                                   (E5, Jina)
//   "search_query: " / "search_document: "      (Nomic)
//   "Represent this sentence for searching relevant passages: "  (BGE)
CRISPEMBED_API void crispembed_set_prefix(crispembed_context * ctx, const char * prefix);

// Get the current prefix (empty string if none set).
CRISPEMBED_API const char * crispembed_get_prefix(const crispembed_context * ctx);

// Get model hyperparameters.
CRISPEMBED_API const crispembed_hparams * crispembed_get_hparams(const crispembed_context * ctx);

// Model registry / auto-download helpers shared by the CLI and wrappers.
CRISPEMBED_API const char * crispembed_cache_dir(void);
CRISPEMBED_API const char * crispembed_resolve_model(const char * arg, int auto_download);
CRISPEMBED_API int crispembed_n_models(void);
CRISPEMBED_API const char * crispembed_model_name(int index);
CRISPEMBED_API const char * crispembed_model_desc(int index);
CRISPEMBED_API const char * crispembed_model_filename(int index);
CRISPEMBED_API const char * crispembed_model_size(int index);

// Encode a single text string. Returns a pointer to a float array of
// length *out_n_dim (the model's output embedding dimension). The
// returned pointer is valid until the next encode() call or free().
// The embedding is L2-normalized.
CRISPEMBED_API const float * crispembed_encode(crispembed_context * ctx,
                                                const char * text,
                                                int * out_n_dim);

// Encode a batch of texts. Returns embeddings as a flat array
// [n_texts * dim]. Pointer valid until next call or free().
CRISPEMBED_API const float * crispembed_encode_batch(crispembed_context * ctx,
                                                      const char ** texts,
                                                      int n_texts,
                                                      int * out_n_dim);

// ---------------------------------------------------------------------------
// Sparse retrieval (BGE-M3 sparse head, SPLADE-style)
// ---------------------------------------------------------------------------

// Returns 1 if this model has a sparse projection head.
CRISPEMBED_API int crispembed_has_sparse(const crispembed_context * ctx);

// Encode text to a sparse term-weight vector over the input vocabulary.
// On success: *out_indices[i] = vocab token id, *out_values[i] = weight (> 0).
// Buffers are owned by ctx and valid until the next call on this ctx.
// Returns the number of non-zero entries (0 on failure or no non-zeros).
CRISPEMBED_API int crispembed_encode_sparse(crispembed_context * ctx,
                                             const char        * text,
                                             const int32_t    ** out_indices,
                                             const float      ** out_values);

// ---------------------------------------------------------------------------
// Multi-vector retrieval (ColBERT-style)
// ---------------------------------------------------------------------------

// Returns 1 if this model has a ColBERT projection head.
CRISPEMBED_API int crispembed_has_colbert(const crispembed_context * ctx);

// Encode text to per-token L2-normalized embeddings.
// Returns flat [*out_n_tokens * *out_dim] array. Valid until next call or free().
CRISPEMBED_API const float * crispembed_encode_multivec(crispembed_context * ctx,
                                                         const char         * text,
                                                         int                * out_n_tokens,
                                                         int                * out_dim);

// ---------------------------------------------------------------------------
// Reranker / cross-encoder
// ---------------------------------------------------------------------------

// Returns 1 if this model has a classifier head (reranker).
CRISPEMBED_API int crispembed_is_reranker(const crispembed_context * ctx);

// Score a (query, document) pair. Returns raw logit (higher = more relevant).
// The model must be a cross-encoder (crispembed_is_reranker() == 1).
CRISPEMBED_API float crispembed_rerank(crispembed_context * ctx,
                                        const char         * query,
                                        const char         * document);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Free all resources.
CRISPEMBED_API void crispembed_free(crispembed_context * ctx);

#ifdef __cplusplus
}
#endif
