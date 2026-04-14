// crispembed.h — C API for text embedding inference via ggml.
//
// Usage:
//   crispembed_context * ctx = crispembed_init("model.gguf", 4);
//   float * vec = crispembed_encode(ctx, "Hello world", &n_dim);
//   // vec is [n_dim] L2-normalized embedding
//   crispembed_free(ctx);

#pragma once

#include <stdint.h>

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
crispembed_context * crispembed_init(const char * model_path, int n_threads);

// Get model hyperparameters.
const crispembed_hparams * crispembed_get_hparams(const crispembed_context * ctx);

// Encode a single text string. Returns a pointer to a float array of
// length *out_n_dim (the model's output embedding dimension). The
// returned pointer is valid until the next encode() call or free().
// The embedding is L2-normalized.
const float * crispembed_encode(crispembed_context * ctx,
                                 const char * text,
                                 int * out_n_dim);

// Encode a batch of texts. Returns embeddings as a flat array
// [n_texts * dim]. Pointer valid until next call or free().
const float * crispembed_encode_batch(crispembed_context * ctx,
                                       const char ** texts,
                                       int n_texts,
                                       int * out_n_dim);

// Free all resources.
void crispembed_free(crispembed_context * ctx);

#ifdef __cplusplus
}
#endif
