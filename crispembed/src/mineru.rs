//! MinerU2.5-Pro native GGML inference. Requires CRISPEMBED_MINERU25_PRO=1.
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use crispembed_sys as ffi;

/// Owns a resident model. Mutable inference methods serialize context use;
/// returned JSON strings are copied before the next native request.
pub struct MinerU {
    ctx: NonNull<ffi::MineruContext>,
}

impl MinerU {
    pub fn new(model: &str, threads: i32) -> Result<Self, String> {
        let path = CString::new(model).map_err(|e| e.to_string())?;
        let ctx = unsafe { ffi::crispembed_mineru_init(path.as_ptr(), threads) };
        NonNull::new(ctx).map(|ctx| Self { ctx })
            .ok_or_else(|| "MinerU model/profile/gate initialization failed".to_owned())
    }

    fn validate_rgb(pixels: &[u8], width: i32, height: i32, max_tokens: i32) -> Result<(), String> {
        if width <= 0 || height <= 0 || !(0..=8192).contains(&max_tokens) {
            return Err("invalid MinerU image dimensions or token cap".to_owned());
        }
        let count = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3));
        if count != Some(pixels.len()) || pixels.len() > 300_000_000 {
            return Err("RGB byte length does not match image dimensions".to_owned());
        }
        Ok(())
    }

    fn copy_result(&self, result: *const std::ffi::c_char) -> Result<String, String> {
        if result.is_null() {
            let error = unsafe { ffi::crispembed_mineru_last_error(self.ctx.as_ptr()) };
            return Err(unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned());
        }
        Ok(unsafe { CStr::from_ptr(result) }.to_string_lossy().into_owned())
    }

    /// Run one region prompt; JSON retains completion IDs, EOS/cap and timing.
    pub fn predict_rgb(&mut self, pixels: &[u8], width: i32, height: i32,
                       prompt: &str, max_tokens: i32) -> Result<String, String> {
        Self::validate_rgb(pixels, width, height, max_tokens)?;
        let prompt = CString::new(prompt).map_err(|e| e.to_string())?;
        let result = unsafe { ffi::crispembed_mineru_predict(self.ctx.as_ptr(), pixels.as_ptr(),
            width, height, 3, prompt.as_ptr(), max_tokens) };
        self.copy_result(result)
    }

    /// Text-only requests support the upstream cross-page table merge prompt.
    pub fn predict_text(&mut self, prompt: &str, max_tokens: i32) -> Result<String, String> {
        if !(0..=8192).contains(&max_tokens) { return Err("invalid token cap".to_owned()); }
        let prompt = CString::new(prompt).map_err(|e| e.to_string())?;
        let result = unsafe { ffi::crispembed_mineru_predict(self.ctx.as_ptr(), std::ptr::null(),
            0, 0, 0, prompt.as_ptr(), max_tokens) };
        self.copy_result(result)
    }

    /// Native basic two-step page JSON. Full MinerU postprocessing is exposed
    /// by the Python CrispMinerU adapter; see docs/mineru25-pro.md.
    pub fn extract_native_basic(&mut self, pixels: &[u8], width: i32, height: i32,
                                image_analysis: bool, max_tokens: i32) -> Result<String, String> {
        Self::validate_rgb(pixels, width, height, max_tokens)?;
        let result = unsafe { ffi::crispembed_mineru_extract(self.ctx.as_ptr(), pixels.as_ptr(),
            width, height, 3, i32::from(image_analysis), max_tokens) };
        self.copy_result(result)
    }
}

impl MinerU {
    /// Complete native document route (requires CRISPEMBED_MINERU25_PRO_FULL_DOC=1).
    /// `options_json` is a JSON object (see crispembed.h) or "" for defaults.
    /// The result's "blocks"/"markdown" are byte-identical to mineru-vl-utils.
    pub fn extract_document(&mut self, pixels: &[u8], width: i32, height: i32,
                            options_json: &str, max_tokens: i32) -> Result<String, String> {
        Self::validate_rgb(pixels, width, height, max_tokens)?;
        let options = CString::new(options_json).map_err(|e| e.to_string())?;
        let result = unsafe { ffi::crispembed_mineru_extract_document(self.ctx.as_ptr(), pixels.as_ptr(),
            width, height, 3, options.as_ptr(), max_tokens) };
        self.copy_result(result)
    }

    /// Render the last extract_document page: "mineru-json", "mineru-markdown",
    /// or the Marker renderers "markdown", "html", "json", "chunks".
    pub fn last_document(&mut self, format: &str) -> Result<String, String> {
        let format = CString::new(format).map_err(|e| e.to_string())?;
        let result = unsafe { ffi::crispembed_mineru_last_document(self.ctx.as_ptr(), format.as_ptr()) };
        self.copy_result(result)
    }
}

impl Drop for MinerU {
    fn drop(&mut self) { unsafe { ffi::crispembed_mineru_free(self.ctx.as_ptr()) } }
}
