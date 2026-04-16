//! Safe Rust wrapper for crispembed text embedding inference.
//!
//! # Quick start
//!
//! ```no_run
//! use crispembed::CrispEmbed;
//!
//! let mut model = CrispEmbed::new("/path/to/model.gguf", 0).unwrap();
//! let vec = model.encode("Hello, world!");
//! println!("dim={}, first={:.4}", vec.len(), vec[0]);
//!
//! // Batch (single graph pass)
//! let vecs = model.encode_batch(&["foo", "bar", "baz"]);
//!
//! // Sparse (BGE-M3)
//! if model.has_sparse() {
//!     let sparse = model.encode_sparse("query text");
//!     for (vocab_id, weight) in &sparse {
//!         println!("  token {} → {:.4}", vocab_id, weight);
//!     }
//! }
//! ```

use std::ffi::{CStr, CString};

pub use crispembed_sys::CrispembedHparams;

/// A loaded crispembed model.
///
/// Not `Sync` — do not share between threads. Each thread should hold its
/// own `CrispEmbed` instance. `Send`-safe: you can move it across threads.
pub struct CrispEmbed {
    ctx: *mut crispembed_sys::CrispembedContext,
    dim: usize,
}

// Safety: the underlying C library serialises all mutable access through
// the opaque context pointer; we hold the only reference.
unsafe impl Send for CrispEmbed {}

impl CrispEmbed {
    /// Load a GGUF model file.
    ///
    /// - `model_path` — path to the `.gguf` file.
    /// - `n_threads`  — CPU thread count; pass `0` for automatic.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path)
            .map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe {
            crispembed_sys::crispembed_init(path.as_ptr(), n_threads)
        };
        if ctx.is_null() {
            return Err(format!("crispembed_init failed for '{model_path}'"));
        }
        let dim = unsafe {
            let hp = crispembed_sys::crispembed_get_hparams(ctx);
            if hp.is_null() { 0 } else { (*hp).n_output as usize }
        };
        Ok(Self { ctx, dim })
    }

    /// Output embedding dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// Set Matryoshka truncation dimension. Pass `0` to use the model default.
    pub fn set_dim(&mut self, dim: i32) {
        unsafe { crispembed_sys::crispembed_set_dim(self.ctx, dim) }
    }

    // ------------------------------------------------------------------
    // Capability queries
    // ------------------------------------------------------------------

    /// Returns `true` if the model has a sparse retrieval head (BGE-M3 sparse).
    pub fn has_sparse(&self) -> bool {
        unsafe { crispembed_sys::crispembed_has_sparse(self.ctx) != 0 }
    }

    /// Returns `true` if the model has a ColBERT multi-vector head.
    pub fn has_colbert(&self) -> bool {
        unsafe { crispembed_sys::crispembed_has_colbert(self.ctx) != 0 }
    }

    /// Returns `true` if the model is a cross-encoder reranker.
    pub fn is_reranker(&self) -> bool {
        unsafe { crispembed_sys::crispembed_is_reranker(self.ctx) != 0 }
    }

    // ------------------------------------------------------------------
    // Dense encode
    // ------------------------------------------------------------------

    /// Encode a single text to an L2-normalised embedding.
    pub fn encode(&mut self, text: &str) -> Vec<f32> {
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode(self.ctx, ctext.as_ptr(), &mut n_dim)
        };
        if ptr.is_null() || n_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, n_dim as usize) }.to_vec()
    }

    /// Encode multiple texts in a single GPU graph pass.
    ///
    /// Returns one embedding per input text in the same order.
    pub fn encode_batch(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return vec![];
        }
        let cstrings: Vec<CString> = texts
            .iter()
            .filter_map(|t| CString::new(*t).ok())
            .collect();
        if cstrings.len() != texts.len() {
            return vec![];
        }
        let ptrs: Vec<*const i8> = cstrings.iter().map(|s| s.as_ptr()).collect();

        let mut n_dim: i32 = 0;
        let flat = unsafe {
            crispembed_sys::crispembed_encode_batch(
                self.ctx,
                ptrs.as_ptr(),
                ptrs.len() as i32,
                &mut n_dim,
            )
        };
        if flat.is_null() || n_dim <= 0 {
            return vec![];
        }
        let dim = n_dim as usize;
        let raw = unsafe { std::slice::from_raw_parts(flat, dim * texts.len()) };
        raw.chunks(dim).map(|c| c.to_vec()).collect()
    }

    // ------------------------------------------------------------------
    // Sparse encode (BGE-M3 SPLADE-style)
    // ------------------------------------------------------------------

    /// Encode `text` to a sparse term-weight vector.
    ///
    /// Returns a list of `(vocab_token_id, weight)` pairs with `weight > 0`.
    /// Returns an empty vector if the model has no sparse head or encoding fails.
    pub fn encode_sparse(&mut self, text: &str) -> Vec<(i32, f32)> {
        if !self.has_sparse() { return vec![]; }
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut indices_ptr: *const i32  = std::ptr::null();
        let mut values_ptr:  *const f32  = std::ptr::null();
        let n = unsafe {
            crispembed_sys::crispembed_encode_sparse(
                self.ctx,
                ctext.as_ptr(),
                &mut indices_ptr,
                &mut values_ptr,
            )
        };
        if n <= 0 || indices_ptr.is_null() || values_ptr.is_null() {
            return vec![];
        }
        let indices = unsafe { std::slice::from_raw_parts(indices_ptr, n as usize) };
        let values  = unsafe { std::slice::from_raw_parts(values_ptr,  n as usize) };
        indices.iter().zip(values.iter()).map(|(&i, &v)| (i, v)).collect()
    }

    // ------------------------------------------------------------------
    // Multi-vector encode (ColBERT)
    // ------------------------------------------------------------------

    /// Encode `text` to per-token L2-normalised embeddings.
    ///
    /// Returns one `Vec<f32>` per (non-padding) token.
    /// Returns an empty vector if the model has no ColBERT head or encoding fails.
    pub fn encode_multivec(&mut self, text: &str) -> Vec<Vec<f32>> {
        if !self.has_colbert() { return vec![]; }
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_tokens: i32 = 0;
        let mut out_dim:  i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_multivec(
                self.ctx,
                ctext.as_ptr(),
                &mut n_tokens,
                &mut out_dim,
            )
        };
        if ptr.is_null() || n_tokens <= 0 || out_dim <= 0 {
            return vec![];
        }
        let dim  = out_dim as usize;
        let raw  = unsafe { std::slice::from_raw_parts(ptr, (n_tokens * out_dim) as usize) };
        raw.chunks(dim).map(|c| c.to_vec()).collect()
    }

    // ------------------------------------------------------------------
    // Reranker
    // ------------------------------------------------------------------

    /// Score a (query, document) pair. Returns a raw relevance logit.
    ///
    /// Higher is more relevant. Returns `f32::NAN` if the model is not a
    /// reranker or if encoding fails.
    pub fn rerank(&mut self, query: &str, document: &str) -> f32 {
        if !self.is_reranker() { return f32::NAN; }
        let cq = match CString::new(query)    { Ok(s) => s, Err(_) => return f32::NAN };
        let cd = match CString::new(document) { Ok(s) => s, Err(_) => return f32::NAN };
        unsafe {
            crispembed_sys::crispembed_rerank(self.ctx, cq.as_ptr(), cd.as_ptr())
        }
    }
}

impl Drop for CrispEmbed {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_free(self.ctx) }
    }
}
