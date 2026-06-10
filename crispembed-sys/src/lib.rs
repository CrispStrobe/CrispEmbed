//! Raw FFI bindings to libcrispembed.
//! Mirrors the public C API in src/crispembed.h exactly.

use std::ffi::{c_char, c_float, c_int};
use std::os::raw::c_void;

/// Opaque handle to a loaded crispembed model.
#[repr(C)]
pub struct CrispembedContext(c_void);

/// Read-only model hyperparameters returned by `crispembed_get_hparams`.
#[repr(C)]
pub struct CrispembedHparams {
    pub n_vocab: i32,
    pub n_max_tokens: i32,
    pub n_embd: i32,
    pub n_head: i32,
    pub n_layer: i32,
    pub n_intermediate: i32,
    pub n_output: i32,
    pub layer_norm_eps: f32,
}

// ------------------------------------------------------------------
// Face detection & recognition types
// ------------------------------------------------------------------

/// Opaque handle to a loaded face model (detector or recogniser).
pub type CrispembedFaceContext = std::ffi::c_void;

/// Opaque handle to a pix2tex math OCR context (encoder-decoder model).
#[repr(C)]
pub struct MathOcrContext(c_void);

/// Opaque handle to a standalone ViT image embedding context (SigLIP, CLIP).
#[repr(C)]
pub struct VitContext(c_void);

/// Bounding box, confidence score and 5-point facial landmarks returned by
/// the face detector.
#[repr(C)]
pub struct CrispembedFaceDetection {
    /// Left edge of the bounding box (pixels).
    pub x: f32,
    /// Top edge of the bounding box (pixels).
    pub y: f32,
    /// Width of the bounding box (pixels).
    pub w: f32,
    /// Height of the bounding box (pixels).
    pub h: f32,
    /// Detection confidence in `[0, 1]`.
    pub confidence: f32,
    /// Five facial landmarks as `[x0, y0, x1, y1, …, x4, y4]` (pixels).
    pub landmarks: [f32; 10],
}

/// One entry returned by the full face pipeline: detection metadata paired
/// with the recognition embedding for that face.
#[repr(C)]
pub struct CrispembedFaceResult {
    /// Detection bounding box and landmarks.
    pub det: CrispembedFaceDetection,
    /// Pointer to the embedding floats (owned by the context).
    pub embedding: *const f32,
    /// Length of the `embedding` buffer.
    pub embedding_dim: i32,
}

extern "C" {
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Load a GGUF model file and initialise backends.
    /// `n_threads` = 0 for auto-detect. Returns NULL on failure.
    pub fn crispembed_init(model_path: *const c_char, n_threads: c_int) -> *mut CrispembedContext;

    /// Free all resources. Safe to call with NULL.
    pub fn crispembed_free(ctx: *mut CrispembedContext);

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Get model hyperparameters (valid for the lifetime of `ctx`).
    pub fn crispembed_get_hparams(ctx: *const CrispembedContext) -> *const CrispembedHparams;

    /// Truncate output to `dim` dimensions (Matryoshka). 0 = model default.
    pub fn crispembed_set_dim(ctx: *mut CrispembedContext, dim: c_int);

    /// Set a text prefix prepended to all inputs before tokenization.
    /// Pass NULL or empty string to clear.
    pub fn crispembed_set_prefix(ctx: *mut CrispembedContext, prefix: *const c_char);

    /// Get the current prefix (empty string if none set).
    pub fn crispembed_get_prefix(ctx: *const CrispembedContext) -> *const c_char;

    /// Returns the shared model cache directory.
    pub fn crispembed_cache_dir() -> *const c_char;

    /// Resolve a model argument to a local GGUF path.
    pub fn crispembed_resolve_model(arg: *const c_char, auto_download: c_int) -> *const c_char;

    /// Registry accessors for wrapper-side model listing.
    pub fn crispembed_n_models() -> c_int;
    pub fn crispembed_model_name(index: c_int) -> *const c_char;
    pub fn crispembed_model_desc(index: c_int) -> *const c_char;
    pub fn crispembed_model_filename(index: c_int) -> *const c_char;
    pub fn crispembed_model_size(index: c_int) -> *const c_char;

    // ------------------------------------------------------------------
    // Dense embedding
    // ------------------------------------------------------------------

    /// Encode a single text. Returns pointer to `*out_n_dim` floats.
    /// Buffer is owned by `ctx` and valid until the next encode call.
    pub fn crispembed_encode(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        out_n_dim: *mut c_int,
    ) -> *const c_float;

    /// Encode a batch of `n_texts` strings in one graph pass.
    /// Returns flat `[n_texts * dim]` array. Buffer valid until next call.
    pub fn crispembed_encode_batch(
        ctx: *mut CrispembedContext,
        texts: *const *const c_char,
        n_texts: c_int,
        out_n_dim: *mut c_int,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Capability queries
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Prefix lookup (name-table and GGUF metadata)
    // ------------------------------------------------------------------

    /// Query prefix for the named model from the built-in name table, or NULL.
    pub fn crispembed_query_prefix(model_name: *const c_char) -> *const c_char;

    /// Passage prefix for the named model from the built-in name table, or NULL.
    pub fn crispembed_passage_prefix(model_name: *const c_char) -> *const c_char;

    /// Query prefix from GGUF metadata (`colbert.query_prefix`), or NULL.
    pub fn crispembed_ctx_query_prefix(ctx: *const CrispembedContext) -> *const c_char;

    /// Passage/document prefix from GGUF metadata, or NULL.
    pub fn crispembed_ctx_passage_prefix(ctx: *const CrispembedContext) -> *const c_char;

    /// Returns 1 if the model has a sparse retrieval head (BGE-M3 sparse).
    pub fn crispembed_has_sparse(ctx: *const CrispembedContext) -> c_int;

    /// Returns 1 if the model has a ColBERT multi-vector head.
    pub fn crispembed_has_colbert(ctx: *const CrispembedContext) -> c_int;

    /// Returns 1 if the model is a reranker (cross-encoder with classifier).
    pub fn crispembed_is_reranker(ctx: *const CrispembedContext) -> c_int;

    // ------------------------------------------------------------------
    // Sparse encode (BGE-M3 SPLADE-style)
    // ------------------------------------------------------------------

    /// Encode `text` to a sparse term-weight vector.
    /// On success, `*out_indices[i]` = vocab token id, `*out_values[i]` = weight.
    /// Both buffers are owned by `ctx` and valid until the next call.
    /// Returns the number of non-zero entries, or 0 on failure.
    pub fn crispembed_encode_sparse(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        out_indices: *mut *const i32,
        out_values: *mut *const c_float,
    ) -> c_int;

    // ------------------------------------------------------------------
    // Multi-vector encode (ColBERT)
    // ------------------------------------------------------------------

    /// Encode `text` to per-token L2-normalised embeddings.
    /// Returns a flat `[*out_n_tokens × *out_dim]` array valid until next call.
    /// Returns NULL on failure or if the model has no ColBERT head.
    pub fn crispembed_encode_multivec(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        out_n_tokens: *mut c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Per-token contextual embeddings (any encoder model)
    // ------------------------------------------------------------------

    /// Encode `text` to per-token L2-normalised final hidden states from
    /// any encoder model (BERT / XLM-R / etc.). Unlike `encode_multivec`,
    /// this is NOT gated on the ColBERT head — it returns the raw
    /// contextual hidden states. Designed for SimAlign-style word aligners.
    pub fn crispembed_encode_tokens(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        out_n_tokens: *mut c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// Pointer to the token IDs from the most recent `encode_tokens` call.
    /// `NULL` if `encode_tokens` has not been called or failed.
    pub fn crispembed_last_token_ids(ctx: *const CrispembedContext) -> *const i32;

    /// Look up the surface form of a vocab token by id. Returns a pointer
    /// to an empty string on out-of-range ids; the pointer is owned by the
    /// context and stable for its lifetime.
    pub fn crispembed_token_str(ctx: *const CrispembedContext, id: i32) -> *const c_char;

    /// Tokenizer family: 1=WordPiece, 2=SentencePiece, 3=BPE, 0=unknown.
    /// Callers use this to interpret subword markers (`##` vs `▁`).
    pub fn crispembed_tokenizer_kind(ctx: *const CrispembedContext) -> c_int;

    // ------------------------------------------------------------------
    // Reranker
    // ------------------------------------------------------------------

    /// Score a (query, document) pair. Returns raw logit (higher = more relevant).
    /// The model must be a reranker (`crispembed_is_reranker` == 1).
    pub fn crispembed_rerank(
        ctx: *mut CrispembedContext,
        query: *const c_char,
        document: *const c_char,
    ) -> c_float;

    // ------------------------------------------------------------------
    // Audio encoding (BidirLM-Omni and similar omnimodal models)
    // ------------------------------------------------------------------

    /// Returns 1 if this build of CrispEmbed has audio support compiled in.
    /// Whether a particular *model* has an audio tower is determined at the
    /// first `crispembed_encode_audio` call.
    pub fn crispembed_has_audio(ctx: *const CrispembedContext) -> c_int;

    /// Encode raw 16 kHz mono float32 PCM into the model's shared embedding
    /// space. Returns NULL on failure (no audio tower, malformed input).
    /// On success returns a buffer of `*out_dim` floats owned by `ctx`,
    /// valid until the next call on this context.
    pub fn crispembed_encode_audio(
        ctx: *mut CrispembedContext,
        pcm_samples: *const c_float,
        n_samples: c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Image encoding (BidirLM-Omni vision tower)
    // ------------------------------------------------------------------

    /// Returns 1 if vision support is built in. Whether a *specific* GGUF
    /// has a vision tower is determined at first `encode_image` call.
    pub fn crispembed_has_vision(ctx: *const CrispembedContext) -> c_int;

    /// Encode a flat (n_patches × 1536) pixel-patch buffer + grid_thw into
    /// a single L2-normalized vector. Buffer owned by ctx, valid until the
    /// next call.
    pub fn crispembed_encode_image(
        ctx: *mut CrispembedContext,
        pixel_patches: *const c_float,
        n_patches: c_int,
        grid_thw: *const i32,
        n_images: c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// Raw vision tower output (un-pooled, un-normalized). Layout of the
    /// returned buffer:
    ///   `[image_embeds, deepstack_0, deepstack_1, …]`
    /// each slab being `(n_merged × dim)` row-major. Total floats =
    /// `(1 + n_deepstack) * n_merged * dim`.
    pub fn crispembed_encode_image_raw(
        ctx: *mut CrispembedContext,
        pixel_patches: *const c_float,
        n_patches: c_int,
        grid_thw: *const i32,
        n_images: c_int,
        out_n_merged: *mut c_int,
        out_dim: *mut c_int,
        out_n_deepstack: *mut c_int,
    ) -> *const c_float;

    /// Text-conditioned multimodal embedding. Splices vision tower image
    /// embeds into the decoder at `image_token_id` placeholders in `text`.
    /// Returns a single L2-normalized vector owned by `ctx`, valid until
    /// the next call. Returns NULL if the model has no vision tower or the
    /// placeholder count does not match the merged-token count.
    pub fn crispembed_encode_text_with_image(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        pixel_patches: *const c_float,
        n_patches: c_int,
        grid_thw: *const i32,
        n_images: c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// Pre-tokenized variant of `crispembed_encode_text_with_image`. Takes
    /// int32 token ids instead of a text string. Skips CrispEmbed's internal
    /// BPE tokenizer for byte-identical parity with the HF reference path.
    /// Returns L2-normalized vector owned by `ctx`, valid until next call.
    pub fn crispembed_encode_with_image_ids(
        ctx: *mut CrispembedContext,
        token_ids: *const i32,
        n_tokens: c_int,
        pixel_patches: *const c_float,
        n_patches: c_int,
        grid_thw: *const i32,
        n_images: c_int,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// In-process file-based image embedding. Loads the image from disk via
    /// stb_image, runs the C++ preprocessor, then runs the vision tower.
    /// Returns a single L2-normalized vector owned by `ctx`, valid until
    /// the next call. Returns NULL on disk/decode failure or missing tower.
    pub fn crispembed_encode_image_file(
        ctx: *mut CrispembedContext,
        image_path: *const c_char,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// In-process file-based image-conditioned text embedding. `text` must
    /// contain the right number of `image_token_id` placeholders for the
    /// grid produced by `smart_resize` on `image_path`.
    /// Same parity caveat as `crispembed_encode_image_file`.
    pub fn crispembed_encode_text_with_image_file(
        ctx: *mut CrispembedContext,
        text: *const c_char,
        image_path: *const c_char,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// Standalone preprocessor (no inference). Runs the C++ image preprocessor
    /// on `image_path` and returns the flat `(n_patches, 1536)` float buffer.
    /// `out_grid_thw` receives the `(1, 3)` grid triplet `[t, h, w]`.
    /// Buffers are owned by `ctx` and valid until the next preprocessor call.
    /// Returns NULL on disk read / decode failure.
    pub fn crispembed_preprocess_image(
        ctx: *mut CrispembedContext,
        image_path: *const c_char,
        out_n_patches: *mut c_int,
        out_row_dim: *mut c_int,
        out_grid_thw: *mut i32,
    ) -> *const c_float;

    /// Preprocess an already-decoded interleaved uint8 RGB(A) buffer. Skips
    /// the stb_image decode step. `channels` must be 3 or 4 (alpha dropped).
    /// Same output contract as `crispembed_preprocess_image`. Buffers owned
    /// by `ctx`, valid until the next preprocessor call.
    pub fn crispembed_preprocess_image_rgb(
        ctx: *mut CrispembedContext,
        rgb: *const u8,
        height: c_int,
        width: c_int,
        channels: c_int,
        out_n_patches: *mut c_int,
        out_row_dim: *mut c_int,
        out_grid_thw: *mut i32,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Face detection & recognition
    // ------------------------------------------------------------------

    /// Opaque handle to a loaded face model (detector or recogniser).
    /// The type alias is defined outside the extern block; the raw pointer
    /// is used throughout this API.

    /// Load a face model from `model_path` and return an opaque context.
    /// `n_threads` = 0 for auto-detect. Returns NULL on failure.
    pub fn crispembed_face_init(
        model_path: *const c_char,
        n_threads: c_int,
    ) -> *mut CrispembedFaceContext;

    /// Returns the embedding dimension produced by a recognition model,
    /// or 0 for a pure detection model.
    pub fn crispembed_face_dim(ctx: *const CrispembedFaceContext) -> c_int;

    /// Returns a NUL-terminated string identifying the model type
    /// (e.g. `"scrfd"`, `"sface"`). Pointer valid for the lifetime of `ctx`.
    pub fn crispembed_face_type(ctx: *const CrispembedFaceContext) -> *const c_char;

    /// Detect faces in `image_path`.
    /// Returns a pointer to `*out_n_faces` `CrispembedFaceDetection` structs
    /// owned by `ctx`, valid until the next call on this context.
    /// Returns NULL on failure.
    pub fn crispembed_detect_faces(
        ctx: *mut CrispembedFaceContext,
        image_path: *const c_char,
        conf_threshold: f32,
        det_size: c_int,
        out_n_faces: *mut c_int,
    ) -> *const CrispembedFaceDetection;

    /// Encode the face described by `landmarks` (10 floats: 5 × [x,y]) from
    /// `image_path` into a face embedding.
    /// Returns a pointer to `*out_dim` floats owned by `ctx`, valid until
    /// the next call on this context. Returns NULL on failure.
    pub fn crispembed_encode_face(
        ctx: *mut CrispembedFaceContext,
        image_path: *const c_char,
        landmarks: *const f32,
        out_dim: *mut c_int,
    ) -> *const f32;

    /// Run the full detect-then-recognise pipeline.
    /// `det_ctx` must be a detection model; `rec_ctx` must be a recognition model.
    /// Returns a pointer to `*out_n_faces` `CrispembedFaceResult` structs
    /// owned by `det_ctx`, valid until the next call on either context.
    /// Returns NULL on failure.
    pub fn crispembed_face_pipeline(
        det_ctx: *mut CrispembedFaceContext,
        rec_ctx: *mut CrispembedFaceContext,
        image_path: *const c_char,
        conf_threshold: f32,
        det_size: c_int,
        out_n_faces: *mut c_int,
    ) -> *const CrispembedFaceResult;

    /// Free all resources held by a face context. Safe to call with NULL.
    pub fn crispembed_face_free(ctx: *mut CrispembedFaceContext);

    // ------------------------------------------------------------------
    // Math OCR (pix2tex) — image → LaTeX via ViT encoder + transformer decoder.
    // ------------------------------------------------------------------

    /// Initialize a math OCR context from a pix2tex GGUF model.
    /// Separate from the main embedding context (encoder-decoder architecture).
    /// `n_threads` = 0 for auto-detect. Returns NULL on failure.
    pub fn crispembed_math_ocr_init(
        model_path: *const c_char,
        n_threads: c_int,
    ) -> *mut MathOcrContext;

    /// Recognize math in raw pixel bytes (RGB or RGBA).
    /// `pixel_bytes` is `(height, width, channels)` row-major uint8.
    /// `channels` must be 3 or 4.
    /// Returns a NUL-terminated LaTeX string owned by the context, valid
    /// until the next call. `out_len` receives the byte length (may be NULL).
    /// Returns NULL on failure.
    pub fn crispembed_math_ocr_recognize(
        ctx: *mut MathOcrContext,
        pixel_bytes: *const u8,
        width: c_int,
        height: c_int,
        channels: c_int,
        out_len: *mut c_int,
    ) -> *const c_char;

    /// Free all resources held by a math OCR context. Safe to call with NULL.
    pub fn crispembed_math_ocr_free(ctx: *mut MathOcrContext);

    // ------------------------------------------------------------------
    // Standalone ViT image embedding (SigLIP, CLIP)
    // ------------------------------------------------------------------

    /// Initialize a standalone ViT context from a SigLIP/CLIP GGUF model.
    /// `n_threads` = 0 for auto-detect. Returns NULL on failure.
    pub fn crispembed_vit_init(
        model_path: *const c_char,
        n_threads: c_int,
    ) -> *mut VitContext;

    /// Returns the embedding dimension produced by the ViT model.
    pub fn crispembed_vit_dim(ctx: *const VitContext) -> c_int;

    /// Encode an image file to a dense embedding via the ViT model.
    /// Returns a pointer to `*out_dim` floats owned by `ctx`, valid until
    /// the next call. Returns NULL on failure.
    pub fn crispembed_vit_encode_file(
        ctx: *mut VitContext,
        image_path: *const c_char,
        out_dim: *mut c_int,
    ) -> *const c_float;

    /// Free all resources held by a ViT context. Safe to call with NULL.
    pub fn crispembed_vit_free(ctx: *mut VitContext);

    // ── Layout Detection (RT-DETRv2) ──

    pub fn crispembed_layout_init(model_path: *const c_char, n_threads: c_int) -> *mut c_void;
    pub fn crispembed_layout_free(ctx: *mut c_void);
    pub fn crispembed_layout_detect(
        ctx: *mut c_void,
        image_path: *const c_char,
        score_threshold: c_float,
        out_n: *mut c_int,
    ) -> *const LayoutRegion;
}

/// Layout detection result (matches `crispembed_layout_region` in C).
#[repr(C)]
pub struct LayoutRegion {
    pub x1: c_float,
    pub y1: c_float,
    pub x2: c_float,
    pub y2: c_float,
    pub score: c_float,
    pub label: c_int,
    pub label_name: *const c_char,
}
