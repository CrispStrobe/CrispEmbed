//! Native OXR-1.0 D-FINE layout detector. Requires CRISPEMBED_OXR=1 and
//! CRISPEMBED_ACCEPT_OXR_LICENSE=1 (OXR License, licenses/oxr-LICENSE).
//! Returns the OXR layout service's /inference JSON body for one page.
use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use crispembed_sys as ffi;

/// Owns one resident layout model; `&mut self` serializes requests.
pub struct OxrLayout {
    ctx: NonNull<ffi::OxrLayoutContext>,
}

impl OxrLayout {
    pub fn new(model: &str, threads: i32) -> Result<Self, String> {
        if threads < 0 { return Err("threads must be nonnegative".to_owned()); }
        let path = CString::new(model).map_err(|e| e.to_string())?;
        let ctx = unsafe { ffi::crispembed_oxr_layout_init(path.as_ptr(), threads) };
        NonNull::new(ctx).map(|ctx| Self { ctx })
            .ok_or_else(|| "OXR layout model/gate initialization failed".to_owned())
    }

    /// `pixels` are interleaved 8-bit gray, RGB or RGBA (`channels` 1/3/4).
    /// `confidence`: the service's optional form field (None = all candidates).
    pub fn detect(&mut self, pixels: &[u8], width: i32, height: i32, channels: i32,
                  confidence: Option<f32>) -> Result<String, String> {
        if width <= 0 || height <= 0 || !matches!(channels, 1 | 3 | 4) {
            return Err("invalid image dimensions or channel count".to_owned());
        }
        let expected = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(channels as usize));
        if expected != Some(pixels.len()) {
            return Err("pixel byte length does not match the image dimensions".to_owned());
        }
        let threshold = match confidence {
            Some(c) if c.is_finite() && (0.0..=1.0).contains(&c) => c,
            Some(_) => return Err("confidence must be a finite number in [0, 1]".to_owned()),
            None => f32::NAN,
        };
        let result = unsafe { ffi::crispembed_oxr_layout_detect(self.ctx.as_ptr(), pixels.as_ptr(),
            width, height, channels, threshold) };
        if result.is_null() {
            let error = unsafe { ffi::crispembed_oxr_layout_last_error(self.ctx.as_ptr()) };
            return Err(if error.is_null() { "OXR layout inference failed".to_owned() }
                else { unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned() });
        }
        Ok(unsafe { CStr::from_ptr(result) }.to_string_lossy().into_owned())
    }

    /// Timings JSON (ms) of the last `detect` call.
    pub fn timings(&mut self) -> String {
        let t = unsafe { ffi::crispembed_oxr_layout_timings(self.ctx.as_ptr()) };
        if t.is_null() { return "{}".to_owned(); }
        unsafe { CStr::from_ptr(t) }.to_string_lossy().into_owned()
    }
}

impl Drop for OxrLayout {
    fn drop(&mut self) { unsafe { ffi::crispembed_oxr_layout_free(self.ctx.as_ptr()) } }
}
