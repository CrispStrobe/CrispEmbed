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
// Audio encoding (omnimodal embedding models)
// ---------------------------------------------------------------------------

// Returns 1 if this context can encode raw audio into the model's shared
// embedding space (e.g. BidirLM-Omni). Audio support is provided by the
// crisp_audio sibling library — built only if CRISP_AUDIO_DIR was found
// at configure time.
CRISPEMBED_API int crispembed_has_audio(const crispembed_context * ctx);

// Encode raw 16 kHz mono float32 PCM into a single L2-normalized vector
// in the model's shared embedding space (same dim as crispembed_encode()
// text output, suitable for cross-modal cosine similarity).
//
// On success returns a buffer of *out_dim floats, owned by ctx and valid
// until the next call on this ctx (or crispembed_free). Returns NULL on
// failure (no audio tower, malformed input, etc.).
CRISPEMBED_API const float * crispembed_encode_audio(crispembed_context * ctx,
                                                      const float        * pcm_samples,
                                                      int                  n_samples,
                                                      int                * out_dim);

// ---------------------------------------------------------------------------
// Image encoding (omnimodal embedding models)
// ---------------------------------------------------------------------------

// Returns 1 if this context has a vision tower (visual.* tensors in the
// GGUF). Determined lazily — the first encode_image call confirms.
CRISPEMBED_API int crispembed_has_vision(const crispembed_context * ctx);

// Encode a flat (n_patches × in_C·T_patch·P²) pixel-patch buffer into a
// single L2-normalized vector in the model's shared embedding space (mean
// pooled over merged tokens). Suitable for cross-modal cosine similarity.
//
// pixel_patches: float32, shape (n_patches, 1536) row-major — produced
//   by the Python preprocessor (Qwen2VLImageProcessorFast or equivalent).
// grid_thw: int32, shape (n_images, 3) — (t, h_patches, w_patches) per image.
//
// Returns a buffer of *out_dim floats, owned by ctx and valid until the next
// call. Returns NULL on failure (no vision tower, malformed input).
CRISPEMBED_API const float * crispembed_encode_image(crispembed_context * ctx,
                                                      const float        * pixel_patches,
                                                      int                  n_patches,
                                                      const int32_t      * grid_thw,
                                                      int                  n_images,
                                                      int                * out_dim);

// Raw (un-pooled, un-normalized) vision tower output. Returns a single
// concatenated buffer of layout:
//   [image_embeds (n_merged × dim), deepstack_0, deepstack_1, ..., deepstack_{k-1}]
// each slab being (n_merged × dim) row-major.
//
// out_n_merged, out_dim, out_n_deepstack get the per-slab shape and the
// number of deepstack slabs. Total floats = (1 + n_deepstack) * n_merged * dim.
//
// Used by the parity test in tests/test_bidirlm_vision.py to compare against
// HF's BidirLMOmniVisionModel(...) tuple output.
//
// Buffer owned by ctx and valid until the next call.
CRISPEMBED_API const float * crispembed_encode_image_raw(crispembed_context * ctx,
                                                          const float        * pixel_patches,
                                                          int                  n_patches,
                                                          const int32_t      * grid_thw,
                                                          int                  n_images,
                                                          int                * out_n_merged,
                                                          int                * out_dim,
                                                          int                * out_n_deepstack);

// Text-conditioned multimodal embedding. The decoder runs over `text` with
// the vision tower's `image_embeds` rows spliced into `inputs_embeds` at
// every `image_token_id` placeholder, and `deepstack_features[k]` added at
// the same positions after the k-th transformer layer. 3D MRoPE position
// ids are derived from `grid_thw`. The result is a single L2-normalized
// vector in the model's shared embedding space.
//
// `text` is expected to already contain the right number of `image_token_id`
// placeholders (`(t * h * w) / spatial_merge_size²` per image). Build it
// with the HF chat template / Qwen2VL processor, or via `python/crispembed/image.py`
// helpers — see examples/demo/bidirlm_image_text.py.
//
// Returns NULL if the model has no vision tower, no MRoPE metadata, or if
// the placeholder count does not match the merged-token count.
CRISPEMBED_API const float * crispembed_encode_text_with_image(
        crispembed_context * ctx,
        const char         * text,
        const float        * pixel_patches,
        int                  n_patches,
        const int32_t      * grid_thw,
        int                  n_images,
        int                * out_dim);

// Lower-level variant that takes pre-tokenized int32 token ids instead of a
// text string. Use this when you need byte-identical parity with the HF
// reference path (it shares its tokenizer state) — it skips CrispEmbed's
// internal BPE tokenizer entirely. `n_tokens` is the length of `token_ids`.
//
// Same multimodal semantics as `crispembed_encode_text_with_image` (image
// splice + DeepStack injection + 3D MRoPE). Returns L2-normalized vector,
// owned by ctx, valid until the next call.
CRISPEMBED_API const float * crispembed_encode_with_image_ids(
        crispembed_context * ctx,
        const int32_t      * token_ids,
        int                  n_tokens,
        const float        * pixel_patches,
        int                  n_patches,
        const int32_t      * grid_thw,
        int                  n_images,
        int                * out_dim);

// In-process file-based image embedding. Loads the image (JPG/PNG/BMP/…)
// from disk via stb_image, runs CrispEmbed's C++ preprocessor (smart_resize
// + bilinear + OpenAI CLIP normalize + Qwen2VL patchify), then runs the
// vision tower. Returns a single L2-normalized cross-modal vector, mean
// pooled over merged tokens.
//
// PARITY CAVEAT: the C++ preprocessor uses bilinear resize (vs torchvision
// bicubic + antialias used by HF). Expect cosine ≈ 0.95–0.98 on real
// photographs. Use the Python wrapper (`crispembed.image.preprocess_image`)
// for byte-tight parity testing against HF. The native path is for runtime
// use without the `transformers` dependency (CLI, mobile, server hot path).
//
// Returns NULL on disk read / decode failure or if the model has no vision
// tower.
CRISPEMBED_API const float * crispembed_encode_image_file(
        crispembed_context * ctx,
        const char         * image_path,
        int                * out_dim);

// In-process file-based image-conditioned text embedding. Same parity
// caveat as crispembed_encode_image_file. `text` must contain the right
// number of `image_token_id` placeholders for the (resized) grid that
// smart_resize will produce — easiest is to call crispembed_preprocess_image
// first to learn `grid_thw`, then build the text template.
CRISPEMBED_API const float * crispembed_encode_text_with_image_file(
        crispembed_context * ctx,
        const char         * text,
        const char         * image_path,
        int                * out_dim);

// Standalone preprocessor (no inference): runs the C++ image preprocessor
// on `image_path` and returns the flat (n_patches, 1536) float buffer plus
// the (1, 3) grid_thw triplet. Buffers are owned by ctx and valid until the
// next preprocessor call. Returns NULL on disk read / decode failure.
//
// Useful for building text templates before calling
// crispembed_encode_with_image_ids: count = (h * w) / spatial_merge_size².
CRISPEMBED_API const float * crispembed_preprocess_image(
        crispembed_context * ctx,
        const char         * image_path,
        int                * out_n_patches,
        int                * out_row_dim,
        int32_t              out_grid_thw[3]);

// Preprocess an already-decoded interleaved uint8 RGB(A) buffer. Skips the
// stb_image JPEG/PNG decode step — useful when the caller has already
// decoded with PIL/libjpeg-turbo (matching HF Qwen2VLImageProcessorFast)
// and wants byte-tight pixel-value parity through the resize/normalize/
// patchify pipeline.
//
// `rgb` must be (height, width, channels) row-major uint8. `channels`
// must be 3 or 4 (alpha is dropped).
//
// Same output contract as crispembed_preprocess_image. Buffers owned by
// ctx, valid until the next preprocessor call.
CRISPEMBED_API const float * crispembed_preprocess_image_rgb(
        crispembed_context * ctx,
        const uint8_t      * rgb,
        int                  height,
        int                  width,
        int                  channels,
        int                * out_n_patches,
        int                * out_row_dim,
        int32_t              out_grid_thw[3]);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Free all resources.
CRISPEMBED_API void crispembed_free(crispembed_context * ctx);

#ifdef __cplusplus
}
#endif
