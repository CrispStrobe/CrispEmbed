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
use std::path::Path;

pub use crispembed_sys::CrispembedHparams;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub desc: String,
    pub filename: String,
    pub size: String,
}

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
        let resolved = Self::resolve_model(model_path, None)?;
        Self::new_resolved(&resolved, n_threads)
    }

    fn new_resolved(model_path: &str, n_threads: i32) -> Result<Self, String> {
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

    pub fn cache_dir() -> String {
        let ptr = unsafe { crispembed_sys::crispembed_cache_dir() };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        }
    }

    pub fn resolve_model(model_path: &str, auto_download: Option<bool>) -> Result<String, String> {
        let should_download = auto_download.unwrap_or_else(|| {
            !model_path.contains(".gguf") && !model_path.contains('/') && !model_path.contains('\\')
        });
        if Path::new(model_path).is_file() {
            return Ok(model_path.to_string());
        }

        // Prefer an existing cache hit before asking the native resolver to
        // download. Mirror native selection semantics: exact match first,
        // then the first fuzzy substring match.
        let cache_dir = Self::cache_dir();
        if !cache_dir.is_empty() {
            let model_key = model_path.to_ascii_lowercase();
            let models = Self::list_models();

            if let Some(model) = models.iter().find(|model| {
                model.name.eq_ignore_ascii_case(model_path)
                    || model.filename.eq_ignore_ascii_case(model_path)
            }) {
                let cached = Path::new(&cache_dir).join(&model.filename);
                if cached.is_file() {
                    return Ok(cached.to_string_lossy().into_owned());
                }
            }

            if let Some(model) = models.iter().find(|model| {
                model.name.to_ascii_lowercase().contains(&model_key)
                    || model.filename.to_ascii_lowercase().contains(&model_key)
            }) {
                let cached = Path::new(&cache_dir).join(&model.filename);
                if cached.is_file() {
                    return Ok(cached.to_string_lossy().into_owned());
                }
            }
        }

        let arg = CString::new(model_path)
            .map_err(|e| format!("invalid model path: {e}"))?;
        let ptr = unsafe {
            crispembed_sys::crispembed_resolve_model(arg.as_ptr(), if should_download { 1 } else { 0 })
        };
        if ptr.is_null() {
            return Err(format!("could not resolve model '{model_path}'"));
        }
        let resolved = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        if resolved.is_empty() {
            Err(format!("could not resolve model '{model_path}'"))
        } else {
            Ok(resolved)
        }
    }

    pub fn list_models() -> Vec<ModelInfo> {
        let n = unsafe { crispembed_sys::crispembed_n_models() };
        let mut models = Vec::with_capacity(n.max(0) as usize);
        for i in 0..n {
            let read = |ptr: *const i8| {
                if ptr.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
                }
            };
            models.push(ModelInfo {
                name: read(unsafe { crispembed_sys::crispembed_model_name(i) }),
                desc: read(unsafe { crispembed_sys::crispembed_model_desc(i) }),
                filename: read(unsafe { crispembed_sys::crispembed_model_filename(i) }),
                size: read(unsafe { crispembed_sys::crispembed_model_size(i) }),
            });
        }
        models
    }

    /// Output embedding dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// Set Matryoshka truncation dimension. Pass `0` to use the model default.
    pub fn set_dim(&mut self, dim: i32) {
        unsafe { crispembed_sys::crispembed_set_dim(self.ctx, dim) }
    }

    /// Set a text prefix prepended to all inputs before tokenization.
    ///
    /// Typical values:
    /// - `"query: "` (E5, Jina v5)
    /// - `"search_query: "` / `"search_document: "` (Nomic)
    /// - `"Represent this sentence for searching relevant passages: "` (BGE)
    ///
    /// Pass an empty string to clear.
    pub fn set_prefix(&mut self, prefix: &str) {
        let cp = CString::new(prefix).unwrap_or_default();
        unsafe { crispembed_sys::crispembed_set_prefix(self.ctx, cp.as_ptr()) }
    }

    /// Get the current prefix (empty string if none set).
    pub fn prefix(&self) -> String {
        let ptr = unsafe { crispembed_sys::crispembed_get_prefix(self.ctx) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        }
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

    /// Encode multiple texts and return one embedding per input in the same order.
    ///
    /// The current native dense batch implementation runs items sequentially
    /// to preserve exact agreement with repeated single-text encodes.
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

    // ------------------------------------------------------------------
    // Bi-encoder reranking (cosine similarity of L2-normalised embeddings)
    // ------------------------------------------------------------------

    /// Rank documents by cosine similarity to the query embedding.
    ///
    /// Encodes query and all documents in a single batch, computes dot
    /// products of L2-normalised embeddings (= cosine similarity), and
    /// returns `(document_index, score)` pairs sorted by score descending.
    ///
    /// If `top_n` is `Some(k)`, only the top-k results are returned.
    pub fn rerank_biencoder(
        &mut self,
        query: &str,
        documents: &[&str],
        top_n: Option<usize>,
    ) -> Vec<(usize, f32)> {
        let mut all_texts: Vec<&str> = Vec::with_capacity(1 + documents.len());
        all_texts.push(query);
        all_texts.extend_from_slice(documents);

        let embeddings = self.encode_batch(&all_texts);
        if embeddings.is_empty() || embeddings.len() != all_texts.len() {
            return vec![];
        }

        let query_vec = &embeddings[0];
        let mut scored: Vec<(usize, f32)> = embeddings[1..]
            .iter()
            .enumerate()
            .map(|(i, doc_vec)| {
                let dot: f32 = query_vec.iter().zip(doc_vec.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(k) = top_n {
            scored.truncate(k);
        }
        scored
    }
}

impl Drop for CrispEmbed {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_free(self.ctx) }
    }
}
