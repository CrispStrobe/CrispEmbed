// wasm/embed_wrapper.c — Emscripten entry point for CrispEmbed text embeddings.
//
// Thin JS-friendly API exposed via EXPORTED_FUNCTIONS for ccall/cwrap.
//
// Model loading flow:
//   1. JS fetches the GGUF file via fetch() and writes it to Emscripten MEMFS
//      using FS.writeFile('/models/model.gguf', data)
//   2. JS calls wasm_embed_init('/models/model.gguf', n_threads) which delegates
//      to crispembed_init — the C++ code opens the MEMFS file via fopen/fread
//      (the mmap path is disabled under __EMSCRIPTEN__).

#include "crispembed.h"
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WASM_EXPORT
#endif

// Version string for the JS loading banner.
WASM_EXPORT
const char * wasm_embed_version(void) {
    return "crispembed-embed-wasm-0.1.0";
}

// Initialize an embedding context from a GGUF file already in MEMFS.
// Returns an opaque pointer (passed back to encode/free), or NULL on failure.
WASM_EXPORT
void * wasm_embed_init(const char * model_path, int n_threads) {
    return crispembed_init(model_path, n_threads);
}

// Return the output embedding dimension for the loaded model.
WASM_EXPORT
int wasm_embed_dim(void * ctx) {
    if (!ctx) return 0;
    const crispembed_hparams * hp = crispembed_get_hparams(ctx);
    if (!hp) return 0;
    return hp->n_output > 0 ? hp->n_output : hp->n_embd;
}

// Set prefix (e.g. "query: " or "search_document: ").
WASM_EXPORT
void wasm_embed_set_prefix(void * ctx, const char * prefix) {
    if (ctx) crispembed_set_prefix(ctx, prefix);
}

// Encode a single text string. Returns a malloc'd copy of the embedding
// that the caller owns — the JS/Dart side must call free() after copying
// to a typed array. *out_n_dim receives the dimension count.
// Returns NULL on failure.
WASM_EXPORT
float * wasm_embed_encode_copy(void * ctx, const char * text, int * out_n_dim) {
    if (!ctx || !text) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    int dim = 0;
    const float * vec = crispembed_encode(ctx, text, &dim);
    if (!vec || dim <= 0) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    if (out_n_dim) *out_n_dim = dim;
    // Copy to a caller-owned buffer so the pointer survives past the
    // next encode call (crispembed_encode returns a context-internal ptr).
    float * copy = (float *)malloc(dim * sizeof(float));
    if (!copy) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    memcpy(copy, vec, dim * sizeof(float));
    return copy;
}

// Encode a batch of texts. Returns a flat malloc'd array [n_texts * dim].
// Caller must free(). *out_n_dim receives the per-text dimension.
WASM_EXPORT
float * wasm_embed_encode_batch_copy(void * ctx, const char ** texts,
                                      int n_texts, int * out_n_dim) {
    if (!ctx || !texts || n_texts <= 0) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    int dim = 0;
    const float * vecs = crispembed_encode_batch(ctx, texts, n_texts, &dim);
    if (!vecs || dim <= 0) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    if (out_n_dim) *out_n_dim = dim;
    size_t total = (size_t)n_texts * dim * sizeof(float);
    float * copy = (float *)malloc(total);
    if (!copy) {
        if (out_n_dim) *out_n_dim = 0;
        return NULL;
    }
    memcpy(copy, vecs, total);
    return copy;
}

// Free the embedding context.
WASM_EXPORT
void wasm_embed_free(void * ctx) {
    if (ctx) crispembed_free(ctx);
}

// Emscripten requires a main() for executables.
int main(void) {
    return 0;
}
