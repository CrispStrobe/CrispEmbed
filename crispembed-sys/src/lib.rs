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
    pub n_vocab:        i32,
    pub n_max_tokens:   i32,
    pub n_embd:         i32,
    pub n_head:         i32,
    pub n_layer:        i32,
    pub n_intermediate: i32,
    pub n_output:       i32,
    pub layer_norm_eps: f32,
}

extern "C" {
    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Load a GGUF model file and initialise backends.
    /// `n_threads` = 0 for auto-detect. Returns NULL on failure.
    pub fn crispembed_init(
        model_path: *const c_char,
        n_threads:  c_int,
    ) -> *mut CrispembedContext;

    /// Free all resources. Safe to call with NULL.
    pub fn crispembed_free(ctx: *mut CrispembedContext);

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Get model hyperparameters (valid for the lifetime of `ctx`).
    pub fn crispembed_get_hparams(
        ctx: *const CrispembedContext,
    ) -> *const CrispembedHparams;

    /// Truncate output to `dim` dimensions (Matryoshka). 0 = model default.
    pub fn crispembed_set_dim(ctx: *mut CrispembedContext, dim: c_int);

    // ------------------------------------------------------------------
    // Dense embedding
    // ------------------------------------------------------------------

    /// Encode a single text. Returns pointer to `*out_n_dim` floats.
    /// Buffer is owned by `ctx` and valid until the next encode call.
    pub fn crispembed_encode(
        ctx:       *mut CrispembedContext,
        text:      *const c_char,
        out_n_dim: *mut c_int,
    ) -> *const c_float;

    /// Encode a batch of `n_texts` strings in one graph pass.
    /// Returns flat `[n_texts * dim]` array. Buffer valid until next call.
    pub fn crispembed_encode_batch(
        ctx:       *mut CrispembedContext,
        texts:     *const *const c_char,
        n_texts:   c_int,
        out_n_dim: *mut c_int,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Capability queries
    // ------------------------------------------------------------------

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
        ctx:         *mut CrispembedContext,
        text:        *const c_char,
        out_indices: *mut *const i32,
        out_values:  *mut *const c_float,
    ) -> c_int;

    // ------------------------------------------------------------------
    // Multi-vector encode (ColBERT)
    // ------------------------------------------------------------------

    /// Encode `text` to per-token L2-normalised embeddings.
    /// Returns a flat `[*out_n_tokens × *out_dim]` array valid until next call.
    /// Returns NULL on failure or if the model has no ColBERT head.
    pub fn crispembed_encode_multivec(
        ctx:          *mut CrispembedContext,
        text:         *const c_char,
        out_n_tokens: *mut c_int,
        out_dim:      *mut c_int,
    ) -> *const c_float;

    // ------------------------------------------------------------------
    // Reranker
    // ------------------------------------------------------------------

    /// Score a (query, document) pair. Returns raw logit (higher = more relevant).
    /// The model must be a reranker (`crispembed_is_reranker` == 1).
    pub fn crispembed_rerank(
        ctx:      *mut CrispembedContext,
        query:    *const c_char,
        document: *const c_char,
    ) -> c_float;
}
