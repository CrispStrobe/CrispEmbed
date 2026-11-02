//! Native OXR recognition and reading order. Requires CRISPEMBED_OXR=1.
//! Layout detection and full document postprocessing are separate operations.
use std::ffi::{c_char, CStr, CString};
use std::ptr::NonNull;
use crispembed_sys as ffi;

/// Owns one resident native model. Mutable methods serialize requests and copy
/// returned JSON before another request can invalidate native result storage.
pub struct Oxr {
    ctx: NonNull<ffi::OxrContext>,
}

impl Oxr {
    pub fn new(model: &str, threads: i32) -> Result<Self, String> {
        if threads < 0 { return Err("threads must be nonnegative".to_owned()); }
        let path = CString::new(model).map_err(|e| e.to_string())?;
        let ctx = unsafe { ffi::crispembed_oxr_init(path.as_ptr(), threads) };
        NonNull::new(ctx).map(|ctx| Self { ctx })
            .ok_or_else(|| "OXR model/profile/gate initialization failed".to_owned())
    }

    fn parameters(prompt: &str, max_tokens: i32, penalty: f32) -> Result<CString, String> {
        if !(0..=32768).contains(&max_tokens) || !penalty.is_finite() || penalty <= 0.0 {
            return Err("invalid OXR token cap or repetition penalty".to_owned());
        }
        if prompt.len() > 65536 { return Err("prompt exceeds 65536 bytes".to_owned()); }
        CString::new(prompt).map_err(|e| e.to_string())
    }

    fn copy_result(&self, result: *const c_char) -> Result<String, String> {
        if result.is_null() {
            let error = unsafe { ffi::crispembed_oxr_last_error(self.ctx.as_ptr()) };
            return Err(if error.is_null() { "OXR inference failed".to_owned() }
                else { unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned() });
        }
        Ok(unsafe { CStr::from_ptr(result) }.to_string_lossy().into_owned())
    }

    /// Recognize one region or run an exact reading-order prompt on a page.
    /// JSON includes completion IDs, EOS/cap and timing. A capped response is
    /// retained explicitly and must not be treated as complete document output.
    pub fn predict_rgb(&mut self, pixels: &[u8], width: i32, height: i32,
                       prompt: &str, max_tokens: i32, repetition_penalty: f32) -> Result<String, String> {
        if width <= 0 || height <= 0 {
            return Err("image dimensions must be positive".to_owned());
        }
        let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(3));
        if expected != Some(pixels.len()) || pixels.len() > 300_000_000 {
            return Err("RGB byte length does not match valid image dimensions".to_owned());
        }
        let prompt = Self::parameters(prompt, max_tokens, repetition_penalty)?;
        let result = unsafe { ffi::crispembed_oxr_predict(self.ctx.as_ptr(), pixels.as_ptr(),
            width, height, 3, prompt.as_ptr(), max_tokens, repetition_penalty) };
        self.copy_result(result)
    }

    pub fn predict_text(&mut self, prompt: &str, max_tokens: i32,
                        repetition_penalty: f32) -> Result<String, String> {
        let prompt = Self::parameters(prompt, max_tokens, repetition_penalty)?;
        let result = unsafe { ffi::crispembed_oxr_predict(self.ctx.as_ptr(), std::ptr::null(),
            0, 0, 0, prompt.as_ptr(), max_tokens, repetition_penalty) };
        self.copy_result(result)
    }
}

impl Oxr {
    /// Native C++ document route (requires CRISPEMBED_OXR_NATIVE_DOC=1): the
    /// translated pinned OXR document helper over supplied layout. `pages` are
    /// tightly packed RGB images with (width, height). `layout_json` is the
    /// official D-FINE batch response; `options_json` may be empty. Returns the
    /// upstream-shaped document JSON (outputs.json, outputs.markdown, metadata).
    pub fn process_pages_rgb(&mut self, pages: &[(&[u8], i32, i32)], layout_json: &str,
                             options_json: &str) -> Result<String, String> {
        self.process_pages_impl(pages, Some(layout_json), options_json)
    }

    fn process_pages_impl(&mut self, pages: &[(&[u8], i32, i32)], layout_json: Option<&str>,
                          options_json: &str) -> Result<String, String> {
        if pages.is_empty() { return Err("OXR requires at least one page".to_owned()); }
        let mut ptrs = Vec::with_capacity(pages.len());
        let (mut widths, mut heights, mut channels) = (Vec::new(), Vec::new(), Vec::new());
        for (pixels, width, height) in pages {
            let expected = (*width as usize).checked_mul(*height as usize).and_then(|n| n.checked_mul(3));
            if *width <= 0 || *height <= 0 || expected != Some(pixels.len()) {
                return Err("RGB byte length does not match valid page dimensions".to_owned());
            }
            ptrs.push(pixels.as_ptr());
            widths.push(*width);
            heights.push(*height);
            channels.push(3);
        }
        let layout = match layout_json { Some(j) => Some(CString::new(j).map_err(|e| e.to_string())?), None => None };
        let options = CString::new(options_json).map_err(|e| e.to_string())?;
        let layout_ptr = layout.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let result = unsafe { ffi::crispembed_oxr_process_pages(self.ctx.as_ptr(), ptrs.as_ptr(),
            widths.as_ptr(), heights.as_ptr(), channels.as_ptr(), pages.len() as i32,
            std::ptr::null(), std::ptr::null(), layout_ptr, options.as_ptr()) };
        self.copy_result(result)
    }

    /// Attach the native OXR D-FINE GGUF as the default layout provider; then
    /// `process_pages_native` (empty `layout_json`) detects layout in C++.
    pub fn set_document_layout_model(&mut self, model: &str, threads: i32) -> Result<(), String> {
        let path = CString::new(model).map_err(|e| e.to_string())?;
        if unsafe { ffi::crispembed_oxr_document_layout_model(self.ctx.as_ptr(), path.as_ptr(), threads) } == 1 {
            return Ok(());
        }
        Err(self.copy_result(std::ptr::null()).unwrap_err())
    }

    /// Native route with the attached D-FINE layout provider.
    pub fn process_pages_native(&mut self, pages: &[(&[u8], i32, i32)], options_json: &str) -> Result<String, String> {
        self.process_pages_impl(pages, None, options_json)
    }

    /// Every native request of the last document call, as a JSON array.
    pub fn last_trace(&self) -> String {
        let t = unsafe { ffi::crispembed_oxr_last_trace(self.ctx.as_ptr()) };
        if t.is_null() { return "[]".to_owned(); }
        unsafe { CStr::from_ptr(t) }.to_string_lossy().into_owned()
    }

    /// Render the last native document via the shared document model
    /// ("markdown", "html", "json" or "chunks").
    pub fn render_document(&mut self, format: &str) -> Result<String, String> {
        let f = CString::new(format).map_err(|e| e.to_string())?;
        let result = unsafe { ffi::crispembed_oxr_render_document(self.ctx.as_ptr(), f.as_ptr()) };
        self.copy_result(result)
    }
}

impl Drop for Oxr {
    fn drop(&mut self) { unsafe { ffi::crispembed_oxr_free(self.ctx.as_ptr()) } }
}
