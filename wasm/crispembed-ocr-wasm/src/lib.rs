//! # crispembed-ocr-wasm
//!
//! Rust wasm-bindgen wrapper for the CrispEmbed OCR WASM module.
//!
//! This crate provides a safe Rust API that calls into the pre-compiled
//! Emscripten WASM module (`crispembed_ocr.js` + `.wasm`). It must be
//! loaded alongside the Emscripten module in the browser.
//!
//! ## Usage from Rust (wasm-pack)
//!
//! ```rust,ignore
//! use crispembed_ocr_wasm::{OcrRecognizer, OcrPipelineResult};
//!
//! let ocr = OcrRecognizer::new("/model.gguf", 1).await?;
//! let result = ocr.recognize_rgba(&pixels, width, height)?;
//! println!("{} (confidence: {})", result.text, result.confidence);
//! ocr.free();
//! ```
//!
//! ## Usage from JS (via wasm-pack)
//!
//! ```js
//! import { OcrRecognizer } from 'crispembed-ocr-wasm';
//! const ocr = await OcrRecognizer.new('/model.gguf', 1);
//! const result = ocr.recognize_rgba(pixelData, 640, 480);
//! console.log(result.text, result.confidence);
//! ocr.free();
//! ```

use serde::Serialize;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// JS imports — bindings to the Emscripten module's exported C functions.
// These assume the CrispEmbedOCR module is already initialized and the
// functions are available as globals (via Module._wasm_ocr_*).
// ---------------------------------------------------------------------------

#[wasm_bindgen]
extern "C" {
    /// The Emscripten module object (set by the JS bootstrap).
    #[wasm_bindgen(js_namespace = ["globalThis"], js_name = "_crispembed_module")]
    static MODULE: JsValue;
}

/// Call a C function via Module.ccall.
fn ccall(name: &str, ret_type: &str, arg_types: &[&str], args: &[JsValue]) -> JsValue {
    let module: &JsValue = &MODULE;
    let ccall_fn = js_sys::Reflect::get(module, &"ccall".into()).unwrap();
    let ccall_fn: js_sys::Function = ccall_fn.into();

    let arg_types_arr = js_sys::Array::new();
    for t in arg_types {
        arg_types_arr.push(&JsValue::from_str(t));
    }
    let args_arr = js_sys::Array::new();
    for a in args {
        args_arr.push(a);
    }

    ccall_fn
        .call3(
            module,
            &JsValue::from_str(name),
            &JsValue::from_str(ret_type),
            &arg_types_arr.into(),
        )
        .unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result from single-model OCR recognition.
#[derive(Serialize)]
#[wasm_bindgen]
pub struct OcrResult {
    #[wasm_bindgen(getter_with_clone)]
    pub text: String,
    pub confidence: f64,
}

/// Result region from pipeline OCR.
#[derive(Serialize)]
#[wasm_bindgen]
pub struct OcrRegion {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub confidence: f64,
    #[wasm_bindgen(getter_with_clone)]
    pub text: String,
}

/// Pipeline result with full text and per-region details.
#[derive(Serialize)]
#[wasm_bindgen]
pub struct OcrPipelineResult {
    #[wasm_bindgen(getter_with_clone)]
    pub text: String,
    pub confidence: f64,
    pub n_regions: u32,
    #[wasm_bindgen(getter_with_clone)]
    pub regions_json: String,
}

/// Single-model OCR recognizer (TrOCR / pix2tex).
///
/// Wraps `wasm_ocr_init` / `wasm_ocr_recognize_copy` / `wasm_ocr_free`.
#[wasm_bindgen]
pub struct OcrRecognizer {
    ctx_ptr: u32,
}

#[wasm_bindgen]
impl OcrRecognizer {
    /// Initialize from a GGUF model already in MEMFS.
    #[wasm_bindgen(constructor)]
    pub fn new(model_path: &str, n_threads: u32) -> Result<OcrRecognizer, JsError> {
        let ptr = ccall(
            "wasm_ocr_init",
            "number",
            &["string", "number"],
            &[JsValue::from_str(model_path), JsValue::from(n_threads)],
        );
        let ctx_ptr = ptr.as_f64().unwrap_or(0.0) as u32;
        if ctx_ptr == 0 {
            return Err(JsError::new("wasm_ocr_init failed"));
        }
        Ok(OcrRecognizer { ctx_ptr })
    }

    /// Recognize text from RGBA pixel bytes.
    pub fn recognize_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<OcrResult, JsError> {
        // Allocate WASM memory and copy pixels
        let module: &JsValue = &MODULE;
        let malloc_fn: js_sys::Function =
            js_sys::Reflect::get(module, &"_malloc".into())?.into();
        let free_fn: js_sys::Function =
            js_sys::Reflect::get(module, &"_free".into())?.into();

        let n_bytes = pixels.len();
        let pixel_ptr = malloc_fn
            .call1(module, &JsValue::from(n_bytes))
            .map_err(|e| JsError::new(&format!("malloc failed: {:?}", e)))?
            .as_f64()
            .unwrap_or(0.0) as u32;

        let len_ptr = malloc_fn
            .call1(module, &JsValue::from(4u32))
            .map_err(|e| JsError::new(&format!("malloc failed: {:?}", e)))?
            .as_f64()
            .unwrap_or(0.0) as u32;

        // Copy pixels into WASM heap via HEAPU8.set
        let heapu8 = js_sys::Reflect::get(module, &"HEAPU8".into())?;
        let heapu8: js_sys::Uint8Array = heapu8.into();
        let js_pixels = js_sys::Uint8Array::from(pixels);
        heapu8.set(&js_pixels, pixel_ptr);

        // Call recognize
        let str_ptr = ccall(
            "wasm_ocr_recognize_copy",
            "number",
            &["number", "number", "number", "number", "number", "number"],
            &[
                JsValue::from(self.ctx_ptr),
                JsValue::from(pixel_ptr),
                JsValue::from(width),
                JsValue::from(height),
                JsValue::from(4u32),
                JsValue::from(len_ptr),
            ],
        )
        .as_f64()
        .unwrap_or(0.0) as u32;

        // Read result
        let text = if str_ptr != 0 {
            let t = ccall("UTF8ToString", "string", &["number"], &[JsValue::from(str_ptr)]);
            let _ = free_fn.call1(module, &JsValue::from(str_ptr));
            t.as_string().unwrap_or_default()
        } else {
            String::new()
        };

        let confidence = ccall(
            "wasm_ocr_mean_confidence",
            "number",
            &["number"],
            &[JsValue::from(self.ctx_ptr)],
        )
        .as_f64()
        .unwrap_or(0.0);

        // Cleanup
        let _ = free_fn.call1(module, &JsValue::from(pixel_ptr));
        let _ = free_fn.call1(module, &JsValue::from(len_ptr));

        Ok(OcrResult { text, confidence })
    }

    /// Set maximum decode tokens.
    pub fn set_max_tokens(&self, max_tokens: u32) {
        ccall(
            "wasm_ocr_set_max_tokens",
            "",
            &["number", "number"],
            &[JsValue::from(self.ctx_ptr), JsValue::from(max_tokens)],
        );
    }

    /// Get the WASM module version.
    pub fn version(&self) -> String {
        ccall("wasm_ocr_version", "string", &[], &[])
            .as_string()
            .unwrap_or_default()
    }

    /// Free the OCR context.
    pub fn free(self) {
        ccall(
            "wasm_ocr_free",
            "",
            &["number"],
            &[JsValue::from(self.ctx_ptr)],
        );
    }
}

/// Full OCR pipeline (detection + recognition).
///
/// Wraps `wasm_ocr_pipeline_init` / `wasm_ocr_pipeline_run` / `wasm_ocr_pipeline_free`.
#[wasm_bindgen]
pub struct OcrPipeline {
    ctx_ptr: u32,
}

#[wasm_bindgen]
impl OcrPipeline {
    /// Initialize pipeline with detection and recognition models in MEMFS.
    #[wasm_bindgen(constructor)]
    pub fn new(
        det_model_path: &str,
        rec_model_path: &str,
        n_threads: u32,
    ) -> Result<OcrPipeline, JsError> {
        let ptr = ccall(
            "wasm_ocr_pipeline_init",
            "number",
            &["string", "string", "number"],
            &[
                JsValue::from_str(det_model_path),
                JsValue::from_str(rec_model_path),
                JsValue::from(n_threads),
            ],
        );
        let ctx_ptr = ptr.as_f64().unwrap_or(0.0) as u32;
        if ctx_ptr == 0 {
            return Err(JsError::new("wasm_ocr_pipeline_init failed"));
        }
        Ok(OcrPipeline { ctx_ptr })
    }

    /// Run pipeline on an image in MEMFS. Returns JSON array of regions.
    pub fn run(&self, image_path: &str) -> Result<String, JsError> {
        let ptr = ccall(
            "wasm_ocr_pipeline_run",
            "number",
            &["number", "string"],
            &[JsValue::from(self.ctx_ptr), JsValue::from_str(image_path)],
        );
        let str_ptr = ptr.as_f64().unwrap_or(0.0) as u32;
        if str_ptr == 0 {
            return Ok("[]".to_string());
        }
        let json = ccall("UTF8ToString", "string", &["number"], &[JsValue::from(str_ptr)])
            .as_string()
            .unwrap_or_default();

        let module: &JsValue = &MODULE;
        let free_fn: js_sys::Function =
            js_sys::Reflect::get(module, &"_free".into())
                .unwrap()
                .into();
        let _ = free_fn.call1(module, &JsValue::from(str_ptr));

        Ok(json)
    }

    /// Free the pipeline context.
    pub fn free(self) {
        ccall(
            "wasm_ocr_pipeline_free",
            "",
            &["number"],
            &[JsValue::from(self.ctx_ptr)],
        );
    }
}

/// Scan cleanup (classical preprocessing — no model needed).
#[wasm_bindgen]
pub struct ScanCleanup {
    ctx_ptr: u32,
}

#[wasm_bindgen]
impl ScanCleanup {
    /// Initialize scan cleanup. Pass empty string for classical-only (no NAFNet).
    #[wasm_bindgen(constructor)]
    pub fn new(model_path: &str, n_threads: u32) -> Result<ScanCleanup, JsError> {
        let ptr = ccall(
            "wasm_scan_cleanup_init",
            "number",
            &["string", "number"],
            &[JsValue::from_str(model_path), JsValue::from(n_threads)],
        );
        let ctx_ptr = ptr.as_f64().unwrap_or(0.0) as u32;
        if ctx_ptr == 0 {
            return Err(JsError::new("wasm_scan_cleanup_init failed"));
        }
        Ok(ScanCleanup { ctx_ptr })
    }

    /// Free the cleanup context.
    pub fn free(self) {
        ccall(
            "wasm_scan_cleanup_free",
            "",
            &["number"],
            &[JsValue::from(self.ctx_ptr)],
        );
    }
}
