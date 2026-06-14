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
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe { crispembed_sys::crispembed_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("crispembed_init failed for '{model_path}'"));
        }
        let dim = unsafe {
            let hp = crispembed_sys::crispembed_get_hparams(ctx);
            if hp.is_null() {
                0
            } else {
                (*hp).n_output as usize
            }
        };
        Ok(Self { ctx, dim })
    }

    pub fn cache_dir() -> String {
        let ptr = unsafe { crispembed_sys::crispembed_cache_dir() };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
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

        let arg = CString::new(model_path).map_err(|e| format!("invalid model path: {e}"))?;
        let ptr = unsafe {
            crispembed_sys::crispembed_resolve_model(
                arg.as_ptr(),
                if should_download { 1 } else { 0 },
            )
        };
        if ptr.is_null() {
            return Err(format!("could not resolve model '{model_path}'"));
        }
        let resolved = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
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
                    unsafe { CStr::from_ptr(ptr) }
                        .to_string_lossy()
                        .into_owned()
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
    pub fn dim(&self) -> usize {
        self.dim
    }

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
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Query prefix from GGUF metadata (`colbert.query_prefix`), or empty string.
    pub fn ctx_query_prefix(&self) -> String {
        let ptr = unsafe { crispembed_sys::crispembed_ctx_query_prefix(self.ctx) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        }
    }

    /// Passage/document prefix from GGUF metadata, or empty string.
    pub fn ctx_passage_prefix(&self) -> String {
        let ptr = unsafe { crispembed_sys::crispembed_ctx_passage_prefix(self.ctx) };
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
        let ptr =
            unsafe { crispembed_sys::crispembed_encode(self.ctx, ctext.as_ptr(), &mut n_dim) };
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
        let cstrings: Vec<CString> = texts.iter().filter_map(|t| CString::new(*t).ok()).collect();
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
        if !self.has_sparse() {
            return vec![];
        }
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut indices_ptr: *const i32 = std::ptr::null();
        let mut values_ptr: *const f32 = std::ptr::null();
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
        let values = unsafe { std::slice::from_raw_parts(values_ptr, n as usize) };
        indices
            .iter()
            .zip(values.iter())
            .map(|(&i, &v)| (i, v))
            .collect()
    }

    // ------------------------------------------------------------------
    // Multi-vector encode (ColBERT)
    // ------------------------------------------------------------------

    /// Encode `text` to per-token L2-normalised embeddings.
    ///
    /// Returns one `Vec<f32>` per (non-padding) token.
    /// Returns an empty vector if the model has no ColBERT head or encoding fails.
    pub fn encode_multivec(&mut self, text: &str) -> Vec<Vec<f32>> {
        if !self.has_colbert() {
            return vec![];
        }
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_tokens: i32 = 0;
        let mut out_dim: i32 = 0;
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
        let dim = out_dim as usize;
        let raw = unsafe { std::slice::from_raw_parts(ptr, (n_tokens * out_dim) as usize) };
        raw.chunks(dim).map(|c| c.to_vec()).collect()
    }

    // ------------------------------------------------------------------
    // Per-token contextual embeddings (any encoder model)
    // ------------------------------------------------------------------

    /// Tokenizer family: `1`=WordPiece (`##` continuation), `2`=SentencePiece
    /// (`▁` word-start marker), `3`=BPE, `0`=unknown. Aligner code uses this
    /// to interpret subword markers when grouping tokens back into words.
    pub fn tokenizer_kind(&self) -> i32 {
        unsafe { crispembed_sys::crispembed_tokenizer_kind(self.ctx) as i32 }
    }

    /// Encode `text` to per-token L2-normalised final hidden states. Works
    /// on any encoder model — not gated on the ColBERT head. Returns one
    /// `(token_string, embedding)` tuple per non-padding token, in order.
    ///
    /// Designed for SimAlign-style word aligners that take cosine
    /// similarity over contextual token embeddings between two languages.
    pub fn encode_tokens(&mut self, text: &str) -> Vec<(String, Vec<f32>)> {
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_tokens: i32 = 0;
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_tokens(
                self.ctx,
                ctext.as_ptr(),
                &mut n_tokens,
                &mut out_dim,
            )
        };
        if ptr.is_null() || n_tokens <= 0 || out_dim <= 0 {
            return vec![];
        }
        let dim = out_dim as usize;
        let n = n_tokens as usize;
        let raw = unsafe { std::slice::from_raw_parts(ptr, n * dim) };
        let ids_ptr = unsafe { crispembed_sys::crispembed_last_token_ids(self.ctx) };
        let ids: &[i32] = if ids_ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(ids_ptr, n) }
        };

        let mut out = Vec::with_capacity(n);
        for (t, vec) in raw.chunks(dim).enumerate() {
            let tok_id = ids.get(t).copied().unwrap_or(-1);
            let tok = if tok_id < 0 {
                String::new()
            } else {
                let cstr = unsafe { crispembed_sys::crispembed_token_str(self.ctx, tok_id) };
                if cstr.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(cstr) }
                        .to_string_lossy()
                        .into_owned()
                }
            };
            out.push((tok, vec.to_vec()));
        }
        out
    }

    // ------------------------------------------------------------------
    // Audio encoding (BidirLM-Omni and similar)
    // ------------------------------------------------------------------

    /// Whether this build of CrispEmbed has audio support compiled in.
    /// (Whether *this model* has an audio tower is only known after the
    /// first `encode_audio` call — failures return an empty vector.)
    pub fn has_audio(&self) -> bool {
        unsafe { crispembed_sys::crispembed_has_audio(self.ctx) != 0 }
    }

    /// Encode raw 16 kHz mono float32 PCM into the model's shared
    /// embedding space (same dim as `encode(text)` for omnimodal models,
    /// suitable for cross-modal cosine similarity).
    ///
    /// Returns an empty vector if the model lacks an audio tower or
    /// encoding fails.
    pub fn encode_audio(&mut self, pcm: &[f32]) -> Vec<f32> {
        if !self.has_audio() {
            return vec![];
        }
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_audio(
                self.ctx,
                pcm.as_ptr(),
                pcm.len() as i32,
                &mut out_dim,
            )
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }

    // ------------------------------------------------------------------
    // Image encoding (BidirLM-Omni vision tower)
    // ------------------------------------------------------------------

    /// Whether this build has vision support compiled in. Whether the
    /// loaded GGUF has a vision tower is only known after the first
    /// `encode_image` call.
    pub fn has_vision(&self) -> bool {
        unsafe { crispembed_sys::crispembed_has_vision(self.ctx) != 0 }
    }

    /// Encode pre-flattened pixel patches into the model's shared
    /// embedding space. Mean-pools across merged tokens and L2-normalizes.
    ///
    /// `pixel_patches` is row-major `(n_patches, 1536)` float32, produced
    /// by an HF `Qwen2VLImageProcessorFast`-equivalent pipeline.
    /// `grid_thw` is `(n_images, 3)` int32.
    pub fn encode_image(&mut self, pixel_patches: &[f32], grid_thw: &[i32]) -> Vec<f32> {
        if grid_thw.len() % 3 != 0 {
            return vec![];
        }
        let n_images = (grid_thw.len() / 3) as i32;
        // patch_flat_dim = 1536 for the standard 16×16 RGB / temporal=2 config —
        // could be queried from hparams, but matching the converter contract is
        // simpler. Caller is expected to have used a compatible preprocessor.
        let n_patches = if pixel_patches.is_empty() {
            0
        } else {
            // Recover from grid_thw: sum of t·h·w across images.
            grid_thw.chunks(3).map(|c| c[0] * c[1] * c[2]).sum::<i32>()
        };
        if n_patches <= 0 {
            return vec![];
        }

        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_image(
                self.ctx,
                pixel_patches.as_ptr(),
                n_patches,
                grid_thw.as_ptr(),
                n_images,
                &mut out_dim,
            )
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }

    /// Raw vision tower output (un-pooled, un-normalized).
    /// Returns `(image_embeds, deepstack_features)` where each entry is a
    /// flat row-major buffer of `n_merged * dim` floats.
    pub fn encode_image_raw(
        &mut self,
        pixel_patches: &[f32],
        grid_thw: &[i32],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let empty = (Vec::new(), Vec::new());
        if grid_thw.len() % 3 != 0 {
            return empty;
        }
        let n_images = (grid_thw.len() / 3) as i32;
        let n_patches = grid_thw.chunks(3).map(|c| c[0] * c[1] * c[2]).sum::<i32>();
        if n_patches <= 0 {
            return empty;
        }

        let mut n_merged: i32 = 0;
        let mut out_dim: i32 = 0;
        let mut n_deepstack: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_image_raw(
                self.ctx,
                pixel_patches.as_ptr(),
                n_patches,
                grid_thw.as_ptr(),
                n_images,
                &mut n_merged,
                &mut out_dim,
                &mut n_deepstack,
            )
        };
        if ptr.is_null() || n_merged <= 0 {
            return empty;
        }
        let per_slab = (n_merged * out_dim) as usize;
        let total = (1 + n_deepstack as usize) * per_slab;
        let flat = unsafe { std::slice::from_raw_parts(ptr, total) };
        let img = flat[..per_slab].to_vec();
        let mut deepstack = Vec::with_capacity(n_deepstack as usize);
        for k in 0..n_deepstack as usize {
            let beg = (1 + k) * per_slab;
            deepstack.push(flat[beg..beg + per_slab].to_vec());
        }
        (img, deepstack)
    }

    /// Load an image from disk, preprocess it, and encode it via the vision
    /// tower. Returns an L2-normalized cross-modal embedding (same dim as
    /// `encode(text)` for omnimodal models).
    ///
    /// The C++ preprocessor uses bilinear resize (vs torchvision bicubic +
    /// antialias used by HF) — expect cosine ≈ 0.95–0.98 on photographs.
    ///
    /// Returns an empty vector on failure (disk error, missing vision tower).
    pub fn encode_image_file(&mut self, path: &str) -> Vec<f32> {
        let cpath = match CString::new(path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_image_file(self.ctx, cpath.as_ptr(), &mut out_dim)
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }

    /// Load an image from disk, preprocess it, and produce a text-conditioned
    /// multimodal embedding. `text` must contain the correct number of
    /// `image_token_id` placeholders for the grid that `smart_resize` will
    /// produce — call `preprocess_image` first to learn `grid_thw`, then
    /// build the text template.
    ///
    /// Same bilinear-resize parity caveat as `encode_image_file`.
    ///
    /// Returns an empty vector on failure.
    pub fn encode_text_with_image_file(&mut self, text: &str, image_path: &str) -> Vec<f32> {
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let cpath = match CString::new(image_path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_text_with_image_file(
                self.ctx,
                ctext.as_ptr(),
                cpath.as_ptr(),
                &mut out_dim,
            )
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }

    // ------------------------------------------------------------------
    // Reranker
    // ------------------------------------------------------------------

    /// Score a (query, document) pair. Returns a raw relevance logit.
    ///
    /// Higher is more relevant. Returns `f32::NAN` if the model is not a
    /// reranker or if encoding fails.
    pub fn rerank(&mut self, query: &str, document: &str) -> f32 {
        if !self.is_reranker() {
            return f32::NAN;
        }
        let cq = match CString::new(query) {
            Ok(s) => s,
            Err(_) => return f32::NAN,
        };
        let cd = match CString::new(document) {
            Ok(s) => s,
            Err(_) => return f32::NAN,
        };
        unsafe { crispembed_sys::crispembed_rerank(self.ctx, cq.as_ptr(), cd.as_ptr()) }
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
                let dot: f32 = query_vec
                    .iter()
                    .zip(doc_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
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

// ======================================================================
// Face detection & recognition
// ======================================================================

pub use crispembed_sys::{CrispembedFaceDetection, CrispembedFaceResult};

/// A loaded face model — either a detector (e.g. SCRFD) or a recogniser
/// (e.g. SFace).
///
/// Not `Sync` — do not share between threads. Each thread should hold its
/// own `CrispFace` instance. `Send`-safe: you can move it across threads.
pub struct CrispFace {
    ctx: *mut crispembed_sys::CrispembedFaceContext,
}

// Safety: same guarantee as CrispEmbed — the C library serialises all
// mutable access through the opaque context pointer.
unsafe impl Send for CrispFace {}

impl CrispFace {
    /// Load a face model from `model_path`.
    ///
    /// - `model_path` — path to the model file.
    /// - `n_threads`  — CPU thread count; pass `0` for automatic.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe { crispembed_sys::crispembed_face_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("crispembed_face_init failed for '{model_path}'"));
        }
        Ok(Self { ctx })
    }

    /// Embedding dimension produced by a recognition model.
    /// Returns `0` for a pure detection model.
    pub fn dim(&self) -> usize {
        unsafe { crispembed_sys::crispembed_face_dim(self.ctx) }.max(0) as usize
    }

    /// Model type string (e.g. `"scrfd"`, `"sface"`).
    pub fn model_type(&self) -> String {
        let ptr = unsafe { crispembed_sys::crispembed_face_type(self.ctx) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Detect faces in the image at `image_path`.
    ///
    /// - `conf_threshold` — minimum detection confidence in `[0, 1]`.
    ///
    /// Returns a `Vec` of [`CrispembedFaceDetection`] structs (bounding box,
    /// confidence, 5-point landmarks). Returns an empty vector on failure.
    pub fn detect(
        &mut self,
        image_path: &str,
        conf_threshold: f32,
    ) -> Vec<CrispembedFaceDetection> {
        self.detect_with_size(image_path, conf_threshold, 0)
    }

    /// Like [`detect`] but with a configurable detection input resolution.
    /// `det_size = 0` uses the default (640).
    pub fn detect_with_size(
        &mut self,
        image_path: &str,
        conf_threshold: f32,
        det_size: i32,
    ) -> Vec<CrispembedFaceDetection> {
        let path = match CString::new(image_path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_faces: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_detect_faces(
                self.ctx,
                path.as_ptr(),
                conf_threshold,
                det_size,
                &mut n_faces,
            )
        };
        if ptr.is_null() || n_faces <= 0 {
            return vec![];
        }
        // Copy structs out of the context-owned buffer so the caller owns them.
        let slice = unsafe { std::slice::from_raw_parts(ptr, n_faces as usize) };
        slice
            .iter()
            .map(|d| CrispembedFaceDetection {
                x: d.x,
                y: d.y,
                w: d.w,
                h: d.h,
                confidence: d.confidence,
                landmarks: d.landmarks,
            })
            .collect()
    }

    /// Encode the face described by `landmarks` (10 floats: 5 × [x, y])
    /// cropped from `image_path` into a face embedding.
    ///
    /// Returns an empty vector on failure.
    pub fn encode_face(&mut self, image_path: &str, landmarks: &[f32; 10]) -> Vec<f32> {
        let path = match CString::new(image_path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_encode_face(
                self.ctx,
                path.as_ptr(),
                landmarks.as_ptr(),
                &mut out_dim,
            )
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }
}

impl Drop for CrispFace {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_face_free(self.ctx) }
    }
}

// ----------------------------------------------------------------------

/// Combined detector + recogniser pipeline.
///
/// Holds one detection context and one recognition context. Call [`run`]
/// to detect all faces in an image and return their embeddings in one shot.
///
/// Both contexts are freed independently when this struct is dropped.
///
/// # Example
///
/// ```no_run
/// use crispembed::CrispFacePipeline;
///
/// let mut pipeline = CrispFacePipeline::new(
///     "/path/to/scrfd.gguf",
///     "/path/to/sface.gguf",
///     0,
/// ).unwrap();
///
/// let faces = pipeline.run("/path/to/photo.jpg", 0.5);
/// for (det, emb) in &faces {
///     println!("face at ({}, {}) conf={:.2} emb_dim={}", det.x, det.y, det.confidence, emb.len());
/// }
/// ```
pub struct CrispFacePipeline {
    det_ctx: *mut crispembed_sys::CrispembedFaceContext,
    rec_ctx: *mut crispembed_sys::CrispembedFaceContext,
}

// Safety: same guarantee as CrispEmbed.
unsafe impl Send for CrispFacePipeline {}

impl CrispFacePipeline {
    /// Load a detection model and a recognition model.
    ///
    /// - `det_path`  — path to the face detector model file.
    /// - `rec_path`  — path to the face recogniser model file.
    /// - `n_threads` — CPU thread count; pass `0` for automatic.
    pub fn new(det_path: &str, rec_path: &str, n_threads: i32) -> Result<Self, String> {
        let det_cpath = CString::new(det_path).map_err(|e| format!("invalid det_path: {e}"))?;
        let rec_cpath = CString::new(rec_path).map_err(|e| format!("invalid rec_path: {e}"))?;

        let det_ctx =
            unsafe { crispembed_sys::crispembed_face_init(det_cpath.as_ptr(), n_threads) };
        if det_ctx.is_null() {
            return Err(format!(
                "crispembed_face_init failed for detector '{det_path}'"
            ));
        }

        let rec_ctx =
            unsafe { crispembed_sys::crispembed_face_init(rec_cpath.as_ptr(), n_threads) };
        if rec_ctx.is_null() {
            // Free the already-allocated detector context before returning.
            unsafe { crispembed_sys::crispembed_face_free(det_ctx) };
            return Err(format!(
                "crispembed_face_init failed for recogniser '{rec_path}'"
            ));
        }

        Ok(Self { det_ctx, rec_ctx })
    }

    /// Embedding dimension of the recogniser model.
    pub fn dim(&self) -> usize {
        unsafe { crispembed_sys::crispembed_face_dim(self.rec_ctx) }.max(0) as usize
    }

    /// Model type strings `(detector_type, recogniser_type)`.
    pub fn model_type(&self) -> (String, String) {
        let read = |ptr: *const std::ffi::c_char| {
            if ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned()
            }
        };
        (
            read(unsafe { crispembed_sys::crispembed_face_type(self.det_ctx) }),
            read(unsafe { crispembed_sys::crispembed_face_type(self.rec_ctx) }),
        )
    }

    /// Detect all faces in `image_path` and encode each one.
    ///
    /// - `conf_threshold` — minimum detection confidence in `[0, 1]`.
    ///
    /// Returns a `Vec` of `(detection, embedding)` pairs — one per detected
    /// face — sorted in the same order as the detector output.
    /// Returns an empty vector on failure or when no faces are found.
    pub fn run(
        &mut self,
        image_path: &str,
        conf_threshold: f32,
    ) -> Vec<(CrispembedFaceDetection, Vec<f32>)> {
        self.run_with_size(image_path, conf_threshold, 0)
    }

    /// Like [`run`] but with a configurable detection input resolution.
    /// `det_size = 0` uses the default (640).
    pub fn run_with_size(
        &mut self,
        image_path: &str,
        conf_threshold: f32,
        det_size: i32,
    ) -> Vec<(CrispembedFaceDetection, Vec<f32>)> {
        let path = match CString::new(image_path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut n_faces: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_face_pipeline(
                self.det_ctx,
                self.rec_ctx,
                path.as_ptr(),
                conf_threshold,
                det_size,
                &mut n_faces,
            )
        };
        if ptr.is_null() || n_faces <= 0 {
            return vec![];
        }
        let results = unsafe { std::slice::from_raw_parts(ptr, n_faces as usize) };
        results
            .iter()
            .map(|r| {
                let det = CrispembedFaceDetection {
                    x: r.det.x,
                    y: r.det.y,
                    w: r.det.w,
                    h: r.det.h,
                    confidence: r.det.confidence,
                    landmarks: r.det.landmarks,
                };
                let emb = if r.embedding.is_null() || r.embedding_dim <= 0 {
                    vec![]
                } else {
                    unsafe { std::slice::from_raw_parts(r.embedding, r.embedding_dim as usize) }
                        .to_vec()
                };
                (det, emb)
            })
            .collect()
    }
}

impl Drop for CrispFacePipeline {
    fn drop(&mut self) {
        unsafe {
            crispembed_sys::crispembed_face_free(self.det_ctx);
            crispembed_sys::crispembed_face_free(self.rec_ctx);
        }
    }
}

// ======================================================================
// Math OCR (pix2tex) — image → LaTeX via ViT encoder + transformer decoder.
// ======================================================================

/// A loaded pix2tex math OCR model (encoder-decoder).
///
/// Not `Sync` — do not share between threads. Each thread should hold its
/// own `MathOcr` instance. `Send`-safe: you can move it across threads.
///
/// # Example
///
/// ```no_run
/// use crispembed::MathOcr;
///
/// let mut ocr = MathOcr::new("/path/to/model.gguf", 0).unwrap();
///
/// // pixels: (height, width, 3) row-major uint8 RGB
/// let pixels: Vec<u8> = vec![255u8; 64 * 64 * 3];
/// if let Some(latex) = ocr.recognize(&pixels, 64, 64) {
///     println!("LaTeX: {latex}");
/// }
/// ```
pub struct MathOcr {
    ctx: *mut crispembed_sys::MathOcrContext,
}

// Safety: the underlying C library serialises all mutable access through
// the opaque context pointer; we hold the only reference.
unsafe impl Send for MathOcr {}

impl MathOcr {
    /// Load a math OCR GGUF model file (auto-detects architecture).
    ///
    /// - `model_path` — path to the `.gguf` file.
    /// - `n_threads`  — CPU thread count; pass `0` for automatic.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe { crispembed_sys::crispembed_math_ocr_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("crispembed_math_ocr_init failed for '{model_path}'"));
        }
        Ok(Self { ctx })
    }

    /// Recognize math in a raw grayscale or RGB(A) pixel buffer.
    ///
    /// - `pixel_bytes` — `(height, width, channels)` row-major uint8;
    ///   `channels` must be 3 (RGB) or 4 (RGBA, alpha is dropped by the C lib).
    ///   For convenience, a single-channel (grayscale) buffer of length
    ///   `width * height` may be passed as `channels = 1` — but the C API
    ///   documents 3 or 4, so prefer converting to RGB first.
    /// - `width`, `height` — image dimensions in pixels.
    ///
    /// Returns `Some(latex_string)` on success, `None` on failure.
    pub fn recognize(&mut self, pixel_bytes: &[u8], width: i32, height: i32) -> Option<String> {
        // Infer channel count from buffer length.
        let total = pixel_bytes.len();
        let expected_rgb = (width * height * 3) as usize;
        let expected_rgba = (width * height * 4) as usize;
        let channels = if total == expected_rgba {
            4
        } else if total == expected_rgb {
            3
        } else {
            // Fall back: pass whatever length we have; the C lib will validate.
            3
        };

        let mut out_len: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_math_ocr_recognize(
                self.ctx,
                pixel_bytes.as_ptr(),
                width,
                height,
                channels,
                &mut out_len,
            )
        };
        if ptr.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// Recognize math from grayscale float pixels [0..1].
    ///
    /// - `pixels` — `(height, width)` row-major float32, values in [0, 1].
    /// - `width`, `height` — image dimensions.
    ///
    /// Returns `Some(latex_string)` on success, `None` on failure.
    pub fn recognize_gray(&mut self, pixels: &[f32], width: i32, height: i32) -> Option<String> {
        let mut out_len: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_math_ocr_recognize_gray(
                self.ctx,
                pixels.as_ptr(),
                width,
                height,
                &mut out_len,
            )
        };
        if ptr.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

impl Drop for MathOcr {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_math_ocr_free(self.ctx) }
    }
}

// ---------------------------------------------------------------------------
// General OCR Pipeline — text detection (DBNet) + recognition (TrOCR)
// ---------------------------------------------------------------------------

/// General OCR pipeline — detects text regions (DBNet) then recognizes each
/// crop (TrOCR). Wraps `crispembed_ocr_init` / `crispembed_ocr`.
///
/// Not `Sync`. Each thread should hold its own instance.
pub struct OcrPipeline {
    ctx: *mut std::ffi::c_void,
}

unsafe impl Send for OcrPipeline {}

/// A single detected + recognized text region.
pub struct OcrResult {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub confidence: f32,
}

impl OcrPipeline {
    /// Load an OCR pipeline from detection + recognition GGUF models.
    pub fn new(det_model: &str, rec_model: &str, n_threads: i32) -> Result<Self, String> {
        let det = CString::new(det_model).map_err(|e| format!("invalid det path: {e}"))?;
        let rec = CString::new(rec_model).map_err(|e| format!("invalid rec path: {e}"))?;
        let ctx = unsafe {
            crispembed_sys::crispembed_ocr_init(det.as_ptr(), rec.as_ptr(), n_threads)
        };
        if ctx.is_null() {
            return Err(format!("crispembed_ocr_init failed"));
        }
        Ok(Self { ctx })
    }

    /// Detect and recognize text in an image file.
    pub fn run(&mut self, image_path: &str) -> Vec<OcrResult> {
        let path = match CString::new(image_path) {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        let mut n: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_ocr(self.ctx, path.as_ptr(), &mut n)
        };
        if ptr.is_null() || n <= 0 {
            return Vec::new();
        }
        let mut results = Vec::with_capacity(n as usize);
        for i in 0..n as isize {
            let r = unsafe { &*ptr.offset(i) };
            let text = if r.text.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(r.text) }.to_string_lossy().into_owned()
            };
            results.push(OcrResult {
                text,
                x: r.x,
                y: r.y,
                w: r.w,
                h: r.h,
                confidence: r.confidence,
            });
        }
        results
    }
}

impl Drop for OcrPipeline {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_ocr_free(self.ctx) }
    }
}

// ======================================================================
// Standalone ViT image embedding (SigLIP, CLIP)
// ======================================================================

/// A loaded standalone ViT model for image embedding (SigLIP, CLIP).
///
/// Not `Sync` — do not share between threads. Each thread should hold its
/// own `CrispVit` instance. `Send`-safe: you can move it across threads.
///
/// # Example
///
/// ```no_run
/// use crispembed::CrispVit;
///
/// let mut vit = CrispVit::new("/path/to/siglip.gguf", 0).unwrap();
/// println!("dim = {}", vit.dim());
/// let embedding = vit.encode_file("/path/to/image.jpg");
/// println!("embedding length = {}", embedding.len());
/// ```
pub struct CrispVit {
    ctx: *mut crispembed_sys::VitContext,
}

// Safety: the underlying C library serialises all mutable access through
// the opaque context pointer; we hold the only reference.
unsafe impl Send for CrispVit {}

impl CrispVit {
    /// Load a SigLIP/CLIP GGUF model file.
    ///
    /// - `model_path` — path to the `.gguf` file.
    /// - `n_threads`  — CPU thread count; pass `0` for automatic.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe { crispembed_sys::crispembed_vit_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("crispembed_vit_init failed for '{model_path}'"));
        }
        Ok(Self { ctx })
    }

    /// Embedding dimension produced by the ViT model.
    pub fn dim(&self) -> i32 {
        unsafe { crispembed_sys::crispembed_vit_dim(self.ctx) }
    }

    /// Encode an image file to a dense embedding via the ViT model.
    ///
    /// Returns an L2-normalized embedding vector. Returns an empty vector
    /// on failure (bad path, unsupported image format, etc.).
    pub fn encode_file(&mut self, image_path: &str) -> Vec<f32> {
        let cpath = match CString::new(image_path) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let mut out_dim: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_vit_encode_file(self.ctx, cpath.as_ptr(), &mut out_dim)
        };
        if ptr.is_null() || out_dim <= 0 {
            return vec![];
        }
        unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) }.to_vec()
    }
}

impl Drop for CrispVit {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_vit_free(self.ctx) }
    }
}

// ── Layout Detection (RT-DETRv2) ──

/// A detected layout region.
pub struct DetectedRegion {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub label: i32,
    pub label_name: String,
}

/// RT-DETRv2 document layout detection (17 region classes).
pub struct CrispLayout {
    ctx: *mut std::ffi::c_void,
}

unsafe impl Send for CrispLayout {}

impl CrispLayout {
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(model_path).map_err(|e| e.to_string())?;
        let ctx = unsafe { crispembed_sys::crispembed_layout_init(c_path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("Failed to load layout model: {}", model_path));
        }
        Ok(Self { ctx })
    }

    pub fn detect(&self, image_path: &str, threshold: f32) -> Vec<DetectedRegion> {
        let c_path = std::ffi::CString::new(image_path).unwrap();
        let mut n: i32 = 0;
        let ptr = unsafe {
            crispembed_sys::crispembed_layout_detect(
                self.ctx, c_path.as_ptr(), threshold, &mut n,
            )
        };
        let mut results = Vec::new();
        if !ptr.is_null() && n > 0 {
            for i in 0..n as usize {
                let r = unsafe { &*ptr.add(i) };
                let name = if r.label_name.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(r.label_name) }
                        .to_str().unwrap_or("").to_string()
                };
                results.push(DetectedRegion {
                    x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2,
                    score: r.score, label: r.label, label_name: name,
                });
            }
        }
        results
    }
}

impl Drop for CrispLayout {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { crispembed_sys::crispembed_layout_free(self.ctx) }
        }
    }
}

// ── Named Entity Recognition (GLiNER) ──

/// A single extracted named entity.
pub struct NerEntity {
    pub text: String,
    pub label: String,
    pub start: i32,
    pub end: i32,
    pub score: f32,
}

/// GLiNER zero-shot named entity recognition.
///
/// Not `Sync` -- do not share between threads. Each thread should hold its
/// own `CrispNER` instance. `Send`-safe: you can move it across threads.
///
/// # Example
///
/// ```no_run
/// use crispembed::CrispNER;
///
/// let mut ner = CrispNER::new("/path/to/gliner.gguf", 0).unwrap();
/// let entities = ner.extract(
///     "John works at Google in New York.",
///     &["person", "organization", "location"],
///     0.5,
/// );
/// for e in &entities {
///     println!("{} ({}) [{}, {}) score={:.3}", e.text, e.label, e.start, e.end, e.score);
/// }
/// ```
pub struct CrispNER {
    ctx: *mut crispembed_sys::NerContext,
}

unsafe impl Send for CrispNER {}

impl CrispNER {
    /// Load a NER GGUF model file (auto-detects architecture).
    ///
    /// - `model_path` -- path to the `.gguf` file.
    /// - `n_threads`  -- CPU thread count; pass `0` for automatic.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let ctx = unsafe { crispembed_sys::crispembed_ner_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(format!("crispembed_ner_init failed for '{model_path}'"));
        }
        Ok(Self { ctx })
    }

    /// Extract named entities from `text` using zero-shot `labels`.
    ///
    /// - `labels`    -- entity types to detect (e.g. `["person", "organization"]`).
    /// - `threshold` -- confidence threshold in `[0, 1]` (recommended 0.5).
    ///
    /// Returns a `Vec<NerEntity>` sorted by position in the input text.
    /// Returns an empty vector on failure.
    pub fn extract(&mut self, text: &str, labels: &[&str], threshold: f32) -> Vec<NerEntity> {
        let ctext = match CString::new(text) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let clabels: Vec<CString> = labels
            .iter()
            .filter_map(|l| CString::new(*l).ok())
            .collect();
        if clabels.len() != labels.len() {
            return vec![];
        }
        let label_ptrs: Vec<*const i8> = clabels.iter().map(|s| s.as_ptr()).collect();

        let mut out_entities: *mut crispembed_sys::NerEntity = std::ptr::null_mut();
        let n = unsafe {
            crispembed_sys::crispembed_ner_extract(
                self.ctx,
                ctext.as_ptr(),
                label_ptrs.as_ptr(),
                label_ptrs.len() as i32,
                threshold,
                &mut out_entities,
            )
        };
        if n <= 0 || out_entities.is_null() {
            return vec![];
        }
        let raw = unsafe { std::slice::from_raw_parts(out_entities, n as usize) };
        raw.iter()
            .map(|e| {
                let text = if e.text.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(e.text) }
                        .to_string_lossy()
                        .into_owned()
                };
                let label = if e.label.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(e.label) }
                        .to_string_lossy()
                        .into_owned()
                };
                NerEntity {
                    text,
                    label,
                    start: e.start_char,
                    end: e.end_char,
                    score: e.score,
                }
            })
            .collect()
    }
}

impl Drop for CrispNER {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_ner_free(self.ctx) }
    }
}
