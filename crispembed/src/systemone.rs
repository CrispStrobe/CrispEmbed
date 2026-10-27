//! Full publisher SystemOne request/response JSON. Gemma APIs are unchanged.
use crate::CrispEmbed;
use crispembed_sys::CrispembedSystemoneContext;
use std::ffi::{CStr, CString};

pub struct SystemOne {
    ctx: *mut CrispembedSystemoneContext,
}
impl SystemOne {
    /// Load a packed CPU/F32 model. Native model gates must be explicitly enabled.
    pub fn new(model_path: &str, n_threads: i32) -> Result<Self, String> {
        Self::new_with_download(model_path, n_threads, None)
    }
    pub fn new_with_download(
        model_path: &str,
        n_threads: i32,
        auto_download: Option<bool>,
    ) -> Result<Self, String> {
        if !(0..=64).contains(&n_threads) {
            return Err("SystemOne threads must be0..64".into());
        }
        let resolved = CrispEmbed::resolve_model(model_path, auto_download)?;
        let path = CString::new(resolved).map_err(|_| "Model path contains NUL")?;
        let ctx = unsafe { crispembed_sys::crispembed_systemone_init(path.as_ptr(), n_threads) };
        if ctx.is_null() {
            return Err(Self::error(ctx));
        }
        Ok(Self { ctx })
    }
    fn error(ctx: *const CrispembedSystemoneContext) -> String {
        let value = unsafe { crispembed_sys::crispembed_systemone_error(ctx) };
        if value.is_null() {
            "SystemOne operation failed".into()
        } else {
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }
    }
    /// Raw JSON preserves question and criterion insertion order, including ties.
    pub fn run_json(&mut self, request: &str) -> Result<String, String> {
        if request.len() > 65536 {
            return Err("SystemOne request exceeds64KiB".into());
        }
        let request = CString::new(request).map_err(|_| "SystemOne request contains NUL")?;
        let value =
            unsafe { crispembed_sys::crispembed_systemone_run_json(self.ctx, request.as_ptr()) };
        if value.is_null() {
            return Err(Self::error(self.ctx));
        }
        Ok(unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned())
    }
    pub fn profile(&self) -> String {
        let value = unsafe { crispembed_sys::crispembed_systemone_profile(self.ctx) };
        if value.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }
    }
}
impl Drop for SystemOne {
    fn drop(&mut self) {
        unsafe { crispembed_sys::crispembed_systemone_free(self.ctx) };
        self.ctx = std::ptr::null_mut();
    }
}
