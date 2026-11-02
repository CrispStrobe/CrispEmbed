//! Native Marker-style PDF/image conversion. No Python or Surya model runtime.
use crispembed_sys as ffi;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};

#[derive(Clone, Copy, Debug)]
pub enum MarkerMode {
    Auto = 0,
    Text = 1,
    Ocr = 2,
}

#[derive(Clone, Debug)]
pub struct MarkerOptions {
    pub model_path: String,
    pub pdfium_path: String,
    pub password: String,
    pub mode: MarkerMode,
    pub n_threads: i32,
    pub first_page: i32,
    pub last_page: i32,
    pub dpi: i32,
    pub max_tokens: i32,
    pub max_layout_tokens: i32,
    pub max_blocks: i32,
    pub max_pages: i32,
    pub max_pixels: u64,
    pub max_asset_bytes: u64,
    pub include_page_headers: bool,
    pub extract_images: bool,
}
impl Default for MarkerOptions {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            pdfium_path: String::new(),
            password: String::new(),
            mode: MarkerMode::Auto,
            n_threads: 4,
            first_page: 0,
            last_page: -1,
            dpi: 144,
            max_tokens: 1024,
            max_layout_tokens: 2048,
            max_blocks: 512,
            max_pages: 1000,
            max_pixels: 16000000,
            max_asset_bytes: 128 * 1024 * 1024,
            include_page_headers: false,
            extract_images: true,
        }
    }
}

/// Owns a conversion context. Mutable methods keep borrowed C results from
/// escaping concurrent conversion; returned strings and assets are copied.
pub struct Marker {
    ctx: *mut ffi::CrispembedMarkerContext,
    max_pixels: u64,
}
impl Marker {
    pub fn new(options: MarkerOptions) -> Result<Self, String> {
        let model = CString::new(options.model_path).map_err(|_| "model path contains NUL")?;
        let pdfium = CString::new(options.pdfium_path).map_err(|_| "PDFium path contains NUL")?;
        let password = CString::new(options.password).map_err(|_| "password contains NUL")?;
        let mut c = unsafe { ffi::crispembed_marker_default_config() };
        c.model_path = model.as_ptr();
        c.pdfium_path = pdfium.as_ptr();
        c.password = password.as_ptr();
        c.mode = options.mode as i32;
        c.n_threads = options.n_threads;
        c.first_page = options.first_page;
        c.last_page = options.last_page;
        c.dpi = options.dpi;
        c.max_tokens = options.max_tokens;
        c.max_layout_tokens = options.max_layout_tokens;
        c.max_blocks = options.max_blocks;
        c.max_pages = options.max_pages;
        c.max_pixels = options.max_pixels;
        c.max_asset_bytes = options.max_asset_bytes;
        c.include_page_headers = options.include_page_headers as i32;
        c.extract_images = options.extract_images as i32;
        let ctx = unsafe { ffi::crispembed_marker_init(&c) };
        if ctx.is_null() {
            return Err(Self::error(ctx));
        }
        Ok(Self {
            ctx,
            max_pixels: options.max_pixels,
        })
    }
    fn error(ctx: *const ffi::CrispembedMarkerContext) -> String {
        let p = unsafe { ffi::crispembed_marker_error(ctx) };
        if p.is_null() {
            "Marker operation failed".into()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
    fn string(&self, p: *const std::ffi::c_char) -> Result<String, String> {
        if p.is_null() {
            Err(Self::error(self.ctx))
        } else {
            unsafe { CStr::from_ptr(p) }
                .to_str()
                .map(str::to_owned)
                .map_err(|e| e.to_string())
        }
    }
    pub fn convert(&mut self, file: &str, format: &str) -> Result<String, String> {
        let file = CString::new(file).map_err(|_| "input path contains NUL")?;
        if unsafe { ffi::crispembed_marker_convert(self.ctx, file.as_ptr()) } == 0 {
            return Err(Self::error(self.ctx));
        }
        self.render(format)
    }
    pub fn convert_image(
        &mut self,
        pixels: &[u8],
        width: i32,
        height: i32,
        channels: i32,
        format: &str,
    ) -> Result<String, String> {
        if width <= 0 || height <= 0 || ![1, 3, 4].contains(&channels) {
            return Err("invalid image dimensions/channels".into());
        }
        let n = (width as u64) * (height as u64);
        if n > self.max_pixels || n.checked_mul(channels as u64) != Some(pixels.len() as u64) {
            return Err("image size mismatch or pixel limit exceeded".into());
        }
        if unsafe {
            ffi::crispembed_marker_convert_raw(self.ctx, pixels.as_ptr(), width, height, channels)
        } == 0
        {
            return Err(Self::error(self.ctx));
        }
        self.render(format)
    }
    pub fn render(&mut self, format: &str) -> Result<String, String> {
        let format = CString::new(format).map_err(|_| "format contains NUL")?;
        self.string(unsafe { ffi::crispembed_marker_render(self.ctx, format.as_ptr()) })
    }
    pub fn metadata_json(&mut self) -> Result<String, String> {
        self.string(unsafe { ffi::crispembed_marker_metadata(self.ctx) })
    }
    pub fn assets(&self) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let count = unsafe { ffi::crispembed_marker_asset_count(self.ctx) };
        let mut out = BTreeMap::new();
        for i in 0..count {
            let name = self.string(unsafe { ffi::crispembed_marker_asset_name(self.ctx, i) })?;
            let mut size = 0;
            let data = unsafe { ffi::crispembed_marker_asset_data(self.ctx, i, &mut size) };
            if data.is_null() && size != 0 {
                return Err("invalid native asset".into());
            }
            out.insert(
                name,
                if size == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(data, size) }.to_vec()
                },
            );
        }
        Ok(out)
    }
}
impl Drop for Marker {
    fn drop(&mut self) {
        unsafe { ffi::crispembed_marker_free(self.ctx) };
    }
}
