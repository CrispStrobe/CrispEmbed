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

impl Drop for Oxr {
    fn drop(&mut self) { unsafe { ffi::crispembed_oxr_free(self.ctx.as_ptr()) } }
}
