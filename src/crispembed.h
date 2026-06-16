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

// In C++, `struct crispembed_context;` alone makes `crispembed_context` a
// usable type. In C, callers would have to write `struct crispembed_context *`
// everywhere — so we provide a typedef so plain C consumers can use the bare
// `crispembed_context *` form. (Discovered by the install verification test
// at find_package() time — a C-only consumer of the installed header.)
typedef struct crispembed_context crispembed_context;

// Model hyperparameters (read-only after init)
typedef struct crispembed_hparams {
    int32_t n_vocab;
    int32_t n_max_tokens;    // max sequence length
    int32_t n_embd;          // embedding dimension (hidden size)
    int32_t n_head;          // attention heads
    int32_t n_layer;         // transformer layers
    int32_t n_intermediate;  // FFN intermediate size
    int32_t n_output;        // output embedding dimension (may differ from n_embd)
    float   layer_norm_eps;
    int32_t n_experts;           // MoE: total number of experts (0 = dense FFN)
    int32_t n_experts_per_tok;   // MoE: top-K routing (e.g. 2)
} crispembed_hparams;

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

// LoRA adapter hot-swap (decoder models with per-task LoRA, e.g. Jina v5).
// Returns 1 on success, 0 on failure (no such adapter, not a decoder model,
// model has no LoRA). Pass NULL or "" to deactivate all adapters (base only).
CRISPEMBED_API int crispembed_set_lora(crispembed_context * ctx, const char * adapter_name);

// Get the currently active LoRA adapter name. Returns "" if none active.
CRISPEMBED_API const char * crispembed_get_lora(const crispembed_context * ctx);

// List available LoRA adapters. Sets *out_names to a null-terminated array
// of strings and *out_count to the number of adapters. The pointers are
// owned by the context and valid until crispembed_free().
// Returns 0 if no LoRA adapters are available.
CRISPEMBED_API int crispembed_list_lora(const crispembed_context * ctx,
                                         const char *** out_names,
                                         int * out_count);

// Get model hyperparameters.
CRISPEMBED_API const crispembed_hparams * crispembed_get_hparams(const crispembed_context * ctx);

// Model registry / auto-download helpers shared by the CLI and wrappers.
CRISPEMBED_API const char * crispembed_cache_dir(void);
CRISPEMBED_API const char * crispembed_resolve_model(const char * arg, int auto_download);
// Get recommended prefix for a model. Returns NULL if not needed.
CRISPEMBED_API const char * crispembed_query_prefix(const char * model_name);
CRISPEMBED_API const char * crispembed_passage_prefix(const char * model_name);

// Context-based lookup — prefers colbert.query_prefix / colbert.document_prefix
// from GGUF metadata, falls back to the name table.
CRISPEMBED_API const char * crispembed_ctx_query_prefix(const crispembed_context * ctx);
CRISPEMBED_API const char * crispembed_ctx_passage_prefix(const crispembed_context * ctx);

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

// ColBERT MaxSim scoring: score = sum_i(max_j(dot(Q[i], D[j])))
// Q and D must be L2-normalized per-token embeddings.
// Returns the late interaction score (higher = more relevant).
CRISPEMBED_API float crispembed_colbert_score(
    const float * query_vecs,  int n_query,
    const float * doc_vecs,    int n_doc,
    int dim);

// Batch ColBERT scoring: score K documents against one query.
// doc_vecs_list[k] points to [doc_n_tokens[k] * dim] float arrays.
// Writes K scores to out_scores. Returns 0 on success.
CRISPEMBED_API int crispembed_colbert_score_batch(
    const float * query_vecs,  int n_query,
    const float ** doc_vecs_list, const int * doc_n_tokens,
    int n_docs, int dim,
    float * out_scores);

// ---------------------------------------------------------------------------
// Per-token contextual embeddings (any encoder model)
// ---------------------------------------------------------------------------

// Encode text to per-token L2-normalised final hidden states. Unlike
// encode_multivec, this works on any encoder model — it skips the ColBERT
// projection and returns the encoder's raw output directly. Designed for
// SimAlign-style word aligners that take cosine similarity of contextual
// token embeddings across two languages.
//
// Returns a flat [*out_n_tokens × *out_dim] array valid until the next
// encode_* call. NULL on decoder models or tokenisation failure.
CRISPEMBED_API const float * crispembed_encode_tokens(crispembed_context * ctx,
                                                       const char         * text,
                                                       int                * out_n_tokens,
                                                       int                * out_dim);

// Same as encode_tokens but returns RAW (unnormalized) hidden states.
// Required for token classification (NER) where the classifier head was
// trained on unnormalized encoder output.
CRISPEMBED_API const float * crispembed_encode_tokens_raw(crispembed_context * ctx,
                                                           const char         * text,
                                                           int                * out_n_tokens,
                                                           int                * out_dim);

// After encode_tokens, returns a pointer to the [n_tokens] vocab IDs of the
// returned embeddings. Valid until the next encode_* call.
CRISPEMBED_API const int32_t * crispembed_last_token_ids(const crispembed_context * ctx);

// Look up the surface form of a vocab token by id. Returns "" (empty) for
// out-of-range ids. Pointer is owned by the context.
CRISPEMBED_API const char * crispembed_token_str(const crispembed_context * ctx,
                                                  int32_t id);

// Tokenizer family. Callers use this to interpret subword markers:
//   1 = WordPiece (## prefix continues a word)
//   2 = SentencePiece (U+2581 ▁ prefix starts a new word)
//   3 = BPE (no SimAlign support — for decoder models)
//   0 = unknown
CRISPEMBED_API int crispembed_tokenizer_kind(const crispembed_context * ctx);

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
// Standalone ViT image embedding (SigLIP, CLIP)
// ---------------------------------------------------------------------------

typedef struct crispembed_vit_context crispembed_vit_context;

// Load a ViT model from GGUF. Returns NULL on failure.
CRISPEMBED_API crispembed_vit_context * crispembed_vit_init(
        const char * model_path, int n_threads);

// Get embedding dimension.
CRISPEMBED_API int crispembed_vit_dim(const crispembed_vit_context * ctx);

// Encode an image file. Returns pointer to *out_dim floats,
// owned by ctx, valid until next call. Returns NULL on failure.
CRISPEMBED_API const float * crispembed_vit_encode_file(
        crispembed_vit_context * ctx,
        const char * image_path,
        int * out_dim);

// Free ViT context.
CRISPEMBED_API void crispembed_vit_free(crispembed_vit_context * ctx);

// ---------------------------------------------------------------------------
// Standalone CLIP text encoding
// ---------------------------------------------------------------------------

typedef struct crispembed_clip_text_context crispembed_clip_text_context;

// Load a CLIP text encoder from GGUF. Returns NULL on failure.
CRISPEMBED_API crispembed_clip_text_context * crispembed_clip_text_init(
        const char * model_path, int n_threads);

// Get embedding dimension.
CRISPEMBED_API int crispembed_clip_text_dim(const crispembed_clip_text_context * ctx);

// Encode text. Returns pointer to *out_dim floats,
// owned by ctx, valid until next call. Returns NULL on failure.
CRISPEMBED_API const float * crispembed_clip_text_encode(
        crispembed_clip_text_context * ctx,
        const char * text,
        int * out_dim);

// Free CLIP text context.
CRISPEMBED_API void crispembed_clip_text_free(crispembed_clip_text_context * ctx);

// ---------------------------------------------------------------------------
// Face detection & recognition (CNN models)
// ---------------------------------------------------------------------------

// Opaque context for CNN face models (SCRFD, ArcFace, SFace, AuraFace).
typedef struct crispembed_face_context crispembed_face_context;

// Detected face bounding box + landmarks.
typedef struct crispembed_face_detection {
    float x, y, w, h;        // bounding box in original image coordinates
    float confidence;
    float landmarks[10];     // 5 points × (x, y)
} crispembed_face_detection;

// Face with embedding.
typedef struct crispembed_face_result {
    crispembed_face_detection det;
    const float * embedding;  // L2-normalized, dim = crispembed_face_dim()
    int embedding_dim;
} crispembed_face_result;

// Load a CNN face model (detection or recognition). Returns NULL on failure.
CRISPEMBED_API crispembed_face_context * crispembed_face_init(
        const char * model_path, int n_threads);

// Get embedding dimension (recognition models).
CRISPEMBED_API int crispembed_face_dim(const crispembed_face_context * ctx);

// Get model type: "detection" or "recognition".
CRISPEMBED_API const char * crispembed_face_type(const crispembed_face_context * ctx);

// Detect faces from an image file. Uses letterbox resize + coordinate
// scaling to return bounding boxes in original image coordinates.
// *out_n_faces is set to the number of detected faces.
// Returns array of detections (owned by ctx, valid until next call).
// det_size: detection input resolution (0 = default 640).
CRISPEMBED_API const crispembed_face_detection * crispembed_detect_faces(
        crispembed_face_context * ctx,
        const char * image_path,
        float conf_threshold,
        int det_size,
        int * out_n_faces);

// Encode a face from image + 5-point landmarks (from detection).
// Performs similarity-transform alignment to 112×112 and encodes.
// Returns L2-normalized embedding (owned by ctx, valid until next call).
CRISPEMBED_API const float * crispembed_encode_face(
        crispembed_face_context * ctx,
        const char * image_path,
        const float * landmarks_10,
        int * out_dim);

// Full pipeline: detect → align → encode. Takes separate detector and
// recognizer contexts. Returns array of face results with embeddings.
// *out_n_faces is set to the number of detected faces.
// Returns array (owned by det_ctx, valid until next call).
// det_size: detection input resolution (0 = default 640).
CRISPEMBED_API const crispembed_face_result * crispembed_face_pipeline(
        crispembed_face_context * det_ctx,
        crispembed_face_context * rec_ctx,
        const char * image_path,
        float conf_threshold,
        int det_size,
        int * out_n_faces);

// Free face context.
CRISPEMBED_API void crispembed_face_free(crispembed_face_context * ctx);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Free all resources.
CRISPEMBED_API void crispembed_free(crispembed_context * ctx);

// ---------------------------------------------------------------------------
// Unified Math OCR — auto-detects model architecture from GGUF metadata.
// Supports pix2tex (printed), PP-FormulaNet (printed/small), PP-FormulaNet-L
// (printed/best), Texo-Distill (printed/tiny), HMER (handwritten), BTTR
// (handwritten), PosFormer (handwritten), MixTex (Chinese+English LaTeX),
// PARSeq (scene text, ViT+Transformer, 94-char ASCII, Apache-2.0),
// Qwen2.5-VL (VLM, German docs), InternVL2.5/2-1B (VLM, EN+DE),
// GLM-OCR (VLM, 8 langs, OmniDocBench #1).
// ---------------------------------------------------------------------------

// Initialize. Reads "general.architecture" from the GGUF to dispatch
// to the correct inference backend (pix2tex, hmer, bttr, ppformulanet,
// mixtex, parseq, qwen2vl, internvl2, glm-ocr, got-ocr, tesseract_lstm).
CRISPEMBED_API void * crispembed_math_ocr_init(const char * model_path, int n_threads);
CRISPEMBED_API void crispembed_math_ocr_free(void * ctx);

// Recognize from raw pixel bytes (RGB/RGBA/grayscale).
CRISPEMBED_API const char * crispembed_math_ocr_recognize(
    void * ctx, const uint8_t * pixel_bytes,
    int width, int height, int channels, int * out_len);

// Recognize from grayscale float pixels [0..1].
CRISPEMBED_API const char * crispembed_math_ocr_recognize_gray(
    void * ctx, const float * pixels,
    int width, int height, int * out_len);

// ---------------------------------------------------------------------------
// Handwritten Math OCR — HMER (DenseNet-121 + GRU attention decoder)
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_hmer_ocr_init(const char * model_path, int n_threads);
CRISPEMBED_API void crispembed_hmer_ocr_free(void * ctx);
CRISPEMBED_API const char * crispembed_hmer_ocr_recognize(
    void * ctx, const uint8_t * pixel_bytes, int width, int height, int channels, int * out_len);
CRISPEMBED_API const char * crispembed_hmer_ocr_recognize_gray(
    void * ctx, const float * pixels, int width, int height, int * out_len);

// ---------------------------------------------------------------------------
// Handwritten Math OCR — BTTR (DenseNet + Transformer decoder)
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_bttr_ocr_init(const char * model_path, int n_threads);
CRISPEMBED_API void crispembed_bttr_ocr_free(void * ctx);
CRISPEMBED_API const char * crispembed_bttr_ocr_recognize(
    void * ctx, const uint8_t * pixel_bytes, int width, int height, int channels, int * out_len);
CRISPEMBED_API const char * crispembed_bttr_ocr_recognize_gray(
    void * ctx, const float * pixels, int width, int height, int * out_len);

// ---------------------------------------------------------------------------
// General OCR Pipeline — text detection (DBNet) + recognition (TrOCR)
// ---------------------------------------------------------------------------

/// OCR result for a single detected text region.
typedef struct crispembed_ocr_result {
    float x, y, w, h;       // bounding box in original image coordinates
    float confidence;        // detection confidence
    const char * text;       // recognized text (owned by pipeline ctx)
    int text_len;
} crispembed_ocr_result;

/// Load OCR pipeline (detection + recognition models).
/// Returns opaque context, or NULL on failure.
CRISPEMBED_API void * crispembed_ocr_init(
    const char * det_model_path,
    const char * rec_model_path,
    int n_threads);

/// Free OCR pipeline context.
CRISPEMBED_API void crispembed_ocr_free(void * ctx);

/// Run full OCR on an image file. Returns array of results sorted in
/// reading order (top-to-bottom, left-to-right). Array is owned by ctx
/// and valid until the next call to crispembed_ocr or crispembed_ocr_free.
CRISPEMBED_API const crispembed_ocr_result * crispembed_ocr(
    void * ctx,
    const char * image_path,
    int * out_n_results);

/// Run text recognition only on a single image crop (no detection).
/// Returns recognized text string (owned by ctx, valid until next call).
CRISPEMBED_API const char * crispembed_ocr_recognize(
    void * ctx,
    const char * image_path,
    int * out_len);

// ---------------------------------------------------------------------------
// OCR Pipeline (orchestrator) — source-type routing + per-stage image cleanup
// (classical + NAFNet) + engine + text-yield/confidence accept-gate escalation.
// Composes scan_cleanup + the ggml OCR engines; see ocr_orchestrator.h.
// ---------------------------------------------------------------------------

/// Flat configuration for the OCR pipeline (slice A: single DBNet+TrOCR engine).
typedef struct crispembed_ocr_pipeline_params {
    int   router;            // 1 = classify source-type (screenshot/scan/photo) and route
    int   cleanup_enabled;   // 1 = run per-stage scan cleanup before OCR
    int   min_chars;         // accept-gate: minimum recognized characters
    float min_confidence;    // accept-gate: minimum mean region confidence (0 = ignore)
    const char * det_model;     // DBNet detection GGUF
    const char * rec_model;     // TrOCR recognition GGUF
    const char * nafnet_model;  // NAFNet denoise GGUF for tier-2 (NULL/"" = classical only)
    const char * sr_model;      // text SR GGUF for low-DPI upscaling (NULL/"" = off)
    const char * vlm_model;     // optional single-shot VLM escalation GGUF (NULL/"" = none)
    int          vlm_engine;    // VLM engine when vlm_model set: 0=GOT 1=GLM 2=Qwen2-VL 3=InternVL2
    const char * punct_model;   // optional post-OCR punctuation/spacing restorer (FireRedPunc/PCS); NULL/"" = off
} crispembed_ocr_pipeline_params;

CRISPEMBED_API crispembed_ocr_pipeline_params crispembed_ocr_pipeline_defaults(void);

/// Build a pipeline context from `params`. Returns opaque context or NULL.
CRISPEMBED_API void * crispembed_ocr_pipeline_init(
    const crispembed_ocr_pipeline_params * params, int n_threads);

/// Run the full pipeline on an image. Returns the per-region results array
/// (owned by ctx, valid until the next run/free) and, via out-params, the
/// joined reading-order `full_text` (owned by ctx) and mean confidence.
CRISPEMBED_API const crispembed_ocr_result * crispembed_ocr_pipeline_run(
    void * ctx, const char * image_path, int * out_n_results,
    const char ** out_full_text, float * out_mean_confidence);

CRISPEMBED_API void crispembed_ocr_pipeline_free(void * ctx);
// (Full per-stage builder API declared after the scan-cleanup section below,
//  since crispembed_ocr_stage embeds crispembed_scan_cleanup_params.)

// ---------------------------------------------------------------------------
// Layout Detection — RT-DETRv2 document layout analysis (17 classes).
// Detects: text, title, table, figure, formula, caption, section_header,
// list_item, footnote, page_header, page_footer, code, document_index,
// checkbox_selected, checkbox_unselected, form, key_value_region.
// ---------------------------------------------------------------------------

typedef struct crispembed_layout_region {
    float x1, y1, x2, y2;   // bbox in original image coordinates
    float score;             // detection confidence
    int label;               // class index (0..16)
    const char * label_name; // static string, do not free
} crispembed_layout_region;

CRISPEMBED_API void * crispembed_layout_init(const char * model_path, int n_threads);
CRISPEMBED_API void crispembed_layout_free(void * ctx);

/// Detect layout regions in an image file. Returns array of regions
/// (owned by ctx, valid until next call). Sets *out_n to count.
CRISPEMBED_API const crispembed_layout_region * crispembed_layout_detect(
    void * ctx, const char * image_path, float score_threshold, int * out_n);

// ---------------------------------------------------------------------------
// Surya Text Detection — EfficientViT segformer (91 languages).
// Segmentation-based text line detection. Returns bounding boxes with
// confidence scores.
// ---------------------------------------------------------------------------

typedef struct crispembed_text_det_result {
    float x0, y0, x1, y1;   // bbox in original image coordinates
    float confidence;
} crispembed_text_det_result;

CRISPEMBED_API void * crispembed_text_det_init(const char * model_path, int n_threads);
CRISPEMBED_API void crispembed_text_det_free(void * ctx);

/// Detect text lines in an image (from raw pixel bytes).
/// Returns array of text regions (owned by ctx, valid until next call).
CRISPEMBED_API const crispembed_text_det_result * crispembed_text_det(
    void * ctx, const uint8_t * pixels, int width, int height, int channels,
    float text_threshold, float low_threshold, int * out_n);

/// Get raw heatmap from last detection. Returns [2, out_h, out_w] float array.
CRISPEMBED_API const float * crispembed_text_det_heatmap(
    void * ctx, int * out_h, int * out_w);

// ---------------------------------------------------------------------------
// Named Entity Recognition — GLiNER zero-shot NER.
// Detects arbitrary entity types specified at inference time.
// Currently supports GLiNER-LFM (LFM2.5 bidirectional backbone).
// ---------------------------------------------------------------------------

typedef struct crispembed_ner_entity {
    int start_char;      // character offset in input text
    int end_char;        // character offset (exclusive)
    const char * text;   // extracted span (owned by ctx, valid until next call)
    const char * label;  // entity label (owned by ctx, valid until next call)
    float score;         // confidence [0, 1]
} crispembed_ner_entity;

/// Load NER model from GGUF. Auto-detects architecture.
CRISPEMBED_API void * crispembed_ner_init(const char * model_path, int n_threads);

/// Free NER context.
CRISPEMBED_API void crispembed_ner_free(void * ctx);

/// Extract named entities with zero-shot labels.
/// labels: array of entity type strings (e.g. "person", "organization")
/// n_labels: number of entity types
/// threshold: confidence threshold (0.0-1.0, recommended 0.5)
/// Returns entity count; sets *out_entities to result array (owned by ctx).
CRISPEMBED_API int crispembed_ner_extract(
    void * ctx, const char * text,
    const char ** labels, int n_labels,
    float threshold,
    crispembed_ner_entity ** out_entities);

// ---------------------------------------------------------------------------
// LiLT — Language-independent Layout Transformer for document understanding.
// Dual-stream encoder (RoBERTa + layout transformer with BiACM) for token
// classification. Used for form understanding (FUNSD: question/answer/header).
// ---------------------------------------------------------------------------

typedef struct crispembed_lilt_token {
    int token_id;
    int label_id;
    const char * label;   // label string — owned by ctx
    float score;          // softmax confidence
} crispembed_lilt_token;

/// Load LiLT model from GGUF. Returns opaque context or NULL.
CRISPEMBED_API void * crispembed_lilt_init(const char * model_path, int n_threads);

/// Free LiLT context.
CRISPEMBED_API void crispembed_lilt_free(void * ctx);

/// Run token classification.
/// input_ids: [n_tokens] token ids (including BOS/EOS).
/// bbox: [n_tokens * 4] bounding boxes (x0, y0, x1, y1) in [0, 1000].
/// Returns array of token results (owned by ctx, valid until next call).
CRISPEMBED_API const crispembed_lilt_token * crispembed_lilt_classify(
    void * ctx, const int32_t * input_ids, const int32_t * bbox,
    int n_tokens, int * out_n);

/// Get number of labels.
CRISPEMBED_API int crispembed_lilt_num_labels(void * ctx);

/// Get label name by id.
CRISPEMBED_API const char * crispembed_lilt_label_name(void * ctx, int label_id);

// ---------------------------------------------------------------------------
// Key Information Extraction (KIE) — OCR + NER pipeline.
// Chains OCR (text detection + recognition) with GLiNER zero-shot NER to
// extract structured key-value fields from document images (receipts,
// invoices, forms, business cards).
// ---------------------------------------------------------------------------

typedef struct crispembed_kie_field {
    const char * label;   // field name (e.g. "total", "date") — owned by ctx
    const char * value;   // extracted text — owned by ctx
    float score;          // NER confidence [0, 1]
    float x, y, w, h;    // bounding box in original image coordinates
} crispembed_kie_field;

typedef struct crispembed_kie_result {
    const crispembed_kie_field * fields; // array of extracted fields — owned by ctx
    int n_fields;                        // number of fields
    const char * ocr_text;               // raw OCR text — owned by ctx
    float ocr_confidence;                // mean OCR confidence
    int n_ocr_regions;                   // number of OCR text regions
} crispembed_kie_result;

/// Initialize KIE pipeline. Requires OCR models (det + rec) and a NER model.
/// Returns opaque context, or NULL on failure.
CRISPEMBED_API void * crispembed_kie_init(
    const char * ocr_det_model,   // text detection GGUF (DBNet)
    const char * ocr_rec_model,   // text recognition GGUF (TrOCR/VLM)
    const char * ner_model,       // GLiNER NER GGUF
    int n_threads);

/// Like crispembed_kie_init but also wires a LiLT GGUF for layout-aware token
/// classification (Phase 2). ner_model / lilt_model may be "" / NULL.
CRISPEMBED_API void * crispembed_kie_init_lilt(
    const char * ocr_det_model,
    const char * ocr_rec_model,
    const char * ner_model,
    const char * lilt_model,
    int n_threads);

/// Extract structured fields from a document image.
/// labels: array of field names to extract (e.g. "total", "date", "vendor")
/// n_labels: number of labels
/// threshold: NER confidence threshold (0.0-1.0, recommended 0.5)
/// Returns result struct (owned by ctx, valid until next call or free).
CRISPEMBED_API crispembed_kie_result crispembed_kie_extract(
    void * ctx, const char * image_path,
    const char ** labels, int n_labels,
    float threshold);

/// Free KIE pipeline context.
CRISPEMBED_API void crispembed_kie_free(void * ctx);

// ---------------------------------------------------------------------------
// Scan cleanup — document scan preprocessing (deskew, denoise, crop, whiten).
// Tier 1: classical image processing, no model needed (model_path=NULL).
// Tier 2: learned denoising CNN via GGUF model (not yet implemented).
// ---------------------------------------------------------------------------

typedef struct {
    int   deskew;
    int   crop_borders;
    int   whiten_background;
    int   binarize;
    int   binarize_method;     // 0 = Otsu, 1 = Sauvola
    float sauvola_k;
    int   sauvola_window;
    int   morph_kernel;
    float border_threshold;
    float deskew_max_angle;
} crispembed_scan_cleanup_params;

CRISPEMBED_API crispembed_scan_cleanup_params crispembed_scan_cleanup_defaults(void);

CRISPEMBED_API void * crispembed_scan_cleanup_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_scan_cleanup_free(void * ctx);

/// Process a scan image. Input: uint8 RGB or grayscale.
/// Allocates output buffer (*out_pixels, RGB uint8). Caller frees via
/// crispembed_scan_cleanup_free_image().
/// Returns 0 on success, -1 on error.
CRISPEMBED_API int crispembed_scan_cleanup_process(
    void * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    crispembed_scan_cleanup_params params,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_scan_cleanup_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// Text Super-Resolution — upscale low-DPI text images before OCR.
// NAFNet U-Net + PixelShuffle ending, processes in overlapping tiles.
// ---------------------------------------------------------------------------

/// Initialize text SR from a GGUF model. Returns opaque context (NULL on error).
CRISPEMBED_API void * crispembed_text_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_text_sr_free(void * ctx);

/// Query upscale factor (2 or 4) from the loaded model.
CRISPEMBED_API int crispembed_text_sr_upscale_factor(const void * ctx);

/// Upscale an RGB image. Allocates output buffer (*out_pixels, RGB uint8).
/// tile_size/tile_overlap: 0 = auto. Caller frees with crispembed_text_sr_free_image().
/// Returns 0 on success, -1 on error.
CRISPEMBED_API int crispembed_text_sr_process(
    void * ctx,
    const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_text_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// TBSRN Text-Line Super-Resolution — PaddleOCR Telescope (Apache-2.0).
// Upscales text-line crops (any size → 32×128) for improved recognition.
// Designed to run between detection and recognition in the OCR pipeline.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_tbsrn_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_tbsrn_sr_free(void * ctx);

/// Upscale a text-line crop. Input resized to 16×64, output is 32×128.
/// Caller frees with crispembed_tbsrn_sr_free_image().
CRISPEMBED_API int crispembed_tbsrn_sr_process(
    void * ctx,
    const uint8_t * pixels, int width, int height,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_tbsrn_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// PAN Whole-Image Super-Resolution — PaddleGAN PAN (Apache-2.0).
// 4× (or 2×) upscale with SCPA blocks + pixel attention. ~272K params.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_pan_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_pan_sr_free(void * ctx);
CRISPEMBED_API int    crispembed_pan_sr_scale(const void * ctx);

CRISPEMBED_API int crispembed_pan_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_pan_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// DAT (Dual Aggregation Transformer) Super-Resolution — ICCV 2023 (Apache-2.0).
// DAT-light x2: ~830K params, dual spatial+channel attention with AIM.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_dat_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_dat_sr_free(void * ctx);

CRISPEMBED_API int crispembed_dat_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_w, int tile_h,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_dat_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// SAFMN Whole-Image Super-Resolution — sunny2109/SAFMN (Apache-2.0).
// 4× (or 2×) upscale with SAFM+CCM AttBlocks + PixelShuffle. ~228K params.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_safmn_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_safmn_sr_free(void * ctx);
CRISPEMBED_API int    crispembed_safmn_sr_scale(const void * ctx);

CRISPEMBED_API int crispembed_safmn_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_safmn_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// SwinIR-light Whole-Image Super-Resolution — JingyunLiang/SwinIR (Apache-2.0).
// Swin Transformer for Image Restoration (ICCVW 2021).
// Lightweight: 4 RSTB × 6 Swin blocks, embed_dim=60, ~0.9M–4.2M params.
// Supports 2×, 3×, 4× upscale via PixelShuffle ending.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_swinir_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_swinir_sr_free(void * ctx);
CRISPEMBED_API int    crispembed_swinir_sr_scale(const void * ctx);

CRISPEMBED_API int crispembed_swinir_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_swinir_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// Real-ESRGAN Whole-Image Super-Resolution — xinntao/Real-ESRGAN (BSD-3).
// 4× upscale with SRVGGNetCompact (17 Conv+PReLU + PixelShuffle). ~620K params.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_esrgan_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_esrgan_sr_free(void * ctx);
CRISPEMBED_API int    crispembed_esrgan_sr_scale(const void * ctx);

CRISPEMBED_API int crispembed_esrgan_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_esrgan_sr_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// Restormer — Multi-task image restoration (denoise/deblur/SR). CVPR 2022.
// U-Net with MDTA (transposed attention) + GDFN. ~26M params, Apache-2.0.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_restormer_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_restormer_free(void * ctx);

CRISPEMBED_API int crispembed_restormer_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels);

CRISPEMBED_API void crispembed_restormer_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// SCUNet — Image denoising (Swin-Conv-UNet hybrid blocks). CVPR 2022.
// U-Net combining Swin Transformer + residual Conv blocks. ~18M params, Apache-2.0.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_scunet_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_scunet_free(void * ctx);

CRISPEMBED_API int crispembed_scunet_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    uint8_t ** out_pixels);

CRISPEMBED_API void crispembed_scunet_free_image(uint8_t * pixels);

// ---------------------------------------------------------------------------
// HAT — Hybrid Attention Transformer SR. CVPR 2023, MIT license.
// SOTA single-image SR. Window attention + OCAB + CAB. ~21M params.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_hat_sr_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_hat_sr_free(void * ctx);
CRISPEMBED_API int    crispembed_hat_sr_scale(const void * ctx);

CRISPEMBED_API int crispembed_hat_sr_process(
    void * ctx, const uint8_t * pixels, int width, int height,
    int tile_size, int tile_overlap,
    uint8_t ** out_pixels, int * out_width, int * out_height);

CRISPEMBED_API void crispembed_hat_sr_free_image(uint8_t * pixels);
// InstructIR — All-in-one image restoration (NAFNet+ICB, ECCV 2024).
// 7 tasks: denoise, deblur, dehaze, derain, super_resolution, low_light, enhance.
// ~16M params, MIT license. Source: mv-lab/InstructIR.
// ---------------------------------------------------------------------------

CRISPEMBED_API void * crispembed_instructir_init(const char * model_path, int n_threads);
CRISPEMBED_API void   crispembed_instructir_free(void * ctx);
CRISPEMBED_API int    crispembed_instructir_n_tasks(const void * ctx);

CRISPEMBED_API int crispembed_instructir_process(
    void * ctx, int task,
    const uint8_t * pixels, int width, int height,
    uint8_t ** out_pixels);

CRISPEMBED_API void crispembed_instructir_free_image(uint8_t * pixels);

/// Variant with individual params (for FFI bindings that can't pass structs by value).
CRISPEMBED_API int crispembed_scan_cleanup_process_simple(
    void * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    int deskew, int crop_borders, int whiten_background, int binarize,
    uint8_t ** out_pixels, int * out_width, int * out_height);

// ── OCR pipeline: full per-stage builder ────────────────────────────────────
// For complete user control: a flat array of stages (grouped into per-source-
// type chains in reading order). The caller (Rust/Dart) owns the rich config
// and marshals it into this array — no JSON dependency in C++. Declared here
// (not by the other crispembed_ocr_pipeline_* decls) because crispembed_ocr_stage
// embeds crispembed_scan_cleanup_params, defined just above.

/// One pipeline stage: engine + models + cleanup + engine params + accept-gate.
typedef struct crispembed_ocr_stage {
    int   source_type;   // 0=auto 1=screenshot 2=scanned_doc 3=photo
    int   engine;        // 0=dbnet_trocr 1=surya 2=got 3=glm 4=qwen2vl 5=internvl2 6=tesseract
    const char * model_a; // det / single-model GGUF
    const char * model_b; // rec GGUF (dbnet_trocr / surya)
    int   cleanup_enabled;
    int   denoise;        // NAFNet tier-2 for this stage
    crispembed_scan_cleanup_params cleanup;  // the 10 classical knobs
    // Engine params (only the relevant ones are used per engine):
    float det_prob_threshold;
    float det_box_threshold;
    int   det_target_short;
    int   vlm_max_tokens;     // 0 = engine default
    const char * vlm_prompt;  // NULL/"" = engine default
    // Accept-gate:
    int   min_chars;
    float min_confidence;
} crispembed_ocr_stage;

/// Build a pipeline from an explicit ordered stage array (full tweakability).
/// Stages with the same `source_type` form that type's chain, in array order.
CRISPEMBED_API void * crispembed_ocr_pipeline_init_stages(
    int router,
    const char * nafnet_model,
    const char * sr_model,      // optional text SR GGUF for low-DPI upscaling (NULL/"" = off)
    const char * punct_model,   // optional post-OCR punctuation/spacing restorer (NULL/"" = off)
    const crispembed_ocr_stage * stages,
    int n_stages,
    int n_threads);

// ---------------------------------------------------------------------------
// Punctuation Restoration — FireRedPunc / PCS
// ---------------------------------------------------------------------------
// Post-processing for CTC output (Tesseract LSTM, ASR, etc.): inserts
// spaces, punctuation, and capitalization into unpunctuated text.
// Two backends: FireRedPunc (BERT Chinese+multilingual, 5 classes) and
// PCS (XLM-R, punctuation + capitalization + segmentation).
// Copied from CrispASR; will be refactored into shared crisp_punc/ library.

/// Load a punctuation model (auto-detects FireRedPunc vs PCS from GGUF arch).
CRISPEMBED_API void * crispembed_punct_init(const char * model_path, int n_threads);
/// Free punct context.
CRISPEMBED_API void   crispembed_punct_free(void * ctx);
/// Add punctuation/spaces to text. Returns string owned by context,
/// valid until next call or free.
CRISPEMBED_API const char * crispembed_punct_process(void * ctx, const char * text);

// ---------------------------------------------------------------------------
// OCR Result Renderers — text, hOCR, ALTO, PDF
// ---------------------------------------------------------------------------
// Re-exported from ocr_render.h for convenience. See that header for the
// full struct definitions (ocr_render_word, ocr_render_line, ocr_render_page,
// ocr_renderer). The C API here provides a simpler one-shot interface.

/// Render OCR results (from crispembed_ocr_pipeline_run) to a format string.
/// format: "text" | "hocr" | "alto" | "pdf"
/// Returns newly allocated string (caller frees with free()).
/// For multi-page: call repeatedly and concatenate, or use ocr_render.h directly.
CRISPEMBED_API char * crispembed_ocr_render(
    const crispembed_ocr_result * results, int n_results,
    int page_width, int page_height,
    const char * format);

// ---------------------------------------------------------------------------
// Classical Preprocessing — dewarp, CC detect, deskew, adaptive binarize
// ---------------------------------------------------------------------------
// CPU-only, model-free, fast tier. No GGUF models needed.

/// PDF DPI profiling — analyse embedded images in a PDF page.
/// Returns 0 on success, -1 on error. Sets *out_dpi to the mean DPI
/// across all raster images on the page, and *out_n_images to the count.
CRISPEMBED_API int crispembed_pdf_page_dpi(
    const char * pdf_path, int page,
    float * out_dpi, int * out_n_images);

/// Dewarp a grayscale page (straighten curved text lines).
/// [out] must be pre-allocated (w*h bytes). Returns 0 on success.
CRISPEMBED_API int crispembed_dewarp(
    const uint8_t * gray, int w, int h,
    uint8_t * out, int * out_w, int * out_h);

/// TPS-based dewarp with explicit control points.
/// [out] must be pre-allocated (w*h bytes). Returns 0 on success.
CRISPEMBED_API int crispembed_tps_dewarp(
    const uint8_t * gray, int w, int h,
    const float * src_x, const float * src_y,
    const float * dst_x, const float * dst_y, int n,
    uint8_t * out);

/// TPS auto-dewarp using a learned localizer model (GGUF).
/// [out] must be pre-allocated (w*h bytes). Returns 0 on success.
CRISPEMBED_API int crispembed_tps_auto_dewarp(
    const uint8_t * gray, int w, int h,
    const char * model_path,
    uint8_t * out);

/// Detect text line regions using connected components (model-free).
/// Returns array of {x,y,w,h} regions, caller frees with free().
CRISPEMBED_API crispembed_ocr_result * crispembed_cc_detect(
    const uint8_t * gray, int w, int h, int * out_n);

/// Find skew angle (degrees) of a document image.
/// Returns 0 on success; *angle is the rotation needed to deskew.
CRISPEMBED_API int crispembed_find_skew(
    const uint8_t * gray, int w, int h,
    float * angle, float * confidence);

/// Adaptive Otsu binarization (handles uneven lighting).
/// [out] must be pre-allocated (w*h bytes), receives 0/255 values.
CRISPEMBED_API void crispembed_adaptive_binarize(
    const uint8_t * gray, int w, int h, uint8_t * out);

/// Background normalization (handles gradients/shadows).
/// [out] must be pre-allocated (w*h bytes).
CRISPEMBED_API void crispembed_background_norm(
    const uint8_t * gray, int w, int h, uint8_t * out);

/// Despeckle: remove small noise components from a binary image.
/// [out] must be pre-allocated (w*h bytes), receives 0/255 values.
CRISPEMBED_API void crispembed_despeckle(
    const uint8_t * gray, int w, int h,
    int max_speckle_w, int max_speckle_h, uint8_t * out);

// ---------------------------------------------------------------------------
// Table Structure Recognition — rule-based grid detection + per-cell OCR.
// Extracts an HTML <table> from a cropped grayscale table image using
// morphological line detection and grid intersection analysis.
// ---------------------------------------------------------------------------

/// Initialize table parser.
/// ocr_model_path: Tesseract LSTM GGUF for built-in cell OCR (NULL = no OCR,
///                 cells will be empty in the returned HTML).
/// n_threads: CPU threads for OCR. Returns NULL on failure.
CRISPEMBED_API void * crispembed_table_parse_init(
    const char * ocr_model_path, int n_threads);

/// Free table parser context. Safe to call with NULL.
CRISPEMBED_API void crispembed_table_parse_free(void * ctx);

/// Parse a grayscale table image into an HTML string.
/// gray: uint8 grayscale pixels, row-major [height × width].
/// Returns a newly allocated HTML string — caller must free with
/// crispembed_table_parse_free_string(). Returns NULL on failure.
CRISPEMBED_API char * crispembed_table_parse_to_html(
    void * ctx, const uint8_t * gray, int width, int height);

/// Free a string returned by crispembed_table_parse_to_html().
CRISPEMBED_API void crispembed_table_parse_free_string(char * str);

/// Detect the grid structure of a table image without running OCR.
/// Sets *out_n_rows and *out_n_cols to the number of rows and columns found.
/// Returns the total number of cells (rows * cols), or 0 on failure.
CRISPEMBED_API int crispembed_table_parse_detect_grid(
    const uint8_t * gray, int width, int height,
    int * out_n_rows, int * out_n_cols);

#ifdef __cplusplus
}
#endif
