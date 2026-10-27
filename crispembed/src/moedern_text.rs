//! Dedicated raw-text query/document vectors. Parent registers this module after audit.
use crate::CrispEmbed;
use std::ffi::{c_char, c_void, CStr, CString};
#[repr(C)]
struct NativeResult {
    abi_version: u32,
    struct_bytes: u32,
    role: i32,
    n_tokens: i32,
    n_selected: i32,
    dimension: i32,
    expansion_count: i32,
    ids: *const i32,
    supplied_mask: *const u8,
    effective_mask: *const u8,
    scoring_mask: *const u8,
    expansion_positions: *const u8,
    all_vectors: *const f32,
    selected_vectors: *const f32,
}
extern "C" {
    fn crispembed_moedern_init(path: *const c_char, threads: i32) -> *mut c_void;
    fn crispembed_moedern_free(ctx: *mut c_void);
    fn crispembed_moedern_encode(
        ctx: *mut c_void,
        text: *const c_char,
        role: i32,
    ) -> *const NativeResult;
    fn crispembed_moedern_score(
        ctx: *mut c_void,
        query: *const c_char,
        document: *const c_char,
        value: *mut f32,
    ) -> i32;
    fn crispembed_moedern_error(ctx: *const c_void) -> *const c_char;
}
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum Role {
    Query = 0,
    Document = 1,
}
pub struct Tokens {
    pub role: Role,
    pub n_tokens: usize,
    pub n_selected: usize,
    pub dimension: usize,
    pub ids: Vec<i32>,
    pub supplied_mask: Vec<u8>,
    pub effective_mask: Vec<u8>,
    pub scoring_mask: Vec<u8>,
    pub expansion_positions: Vec<u8>,
    pub all_vectors: Vec<f32>,
    pub selected_vectors: Vec<f32>,
}
pub struct MoEdern {
    ctx: *mut c_void,
}
fn input(text: &str) -> Result<CString, String> {
    if text.len() > 65536 {
        return Err("MoEdern text exceeds64KiB".into());
    }
    CString::new(text).map_err(|_| "MoEdern text contains NUL".into())
}
unsafe fn copied<T: Copy>(pointer: *const T, n: usize) -> Result<Vec<T>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if pointer.is_null() {
        return Err("MoEdern result pointer is null".into());
    }
    Ok(std::slice::from_raw_parts(pointer, n).to_vec())
}
impl MoEdern {
    pub fn new(path: &str, threads: i32, download: Option<bool>) -> Result<Self, String> {
        if !(0..=2).contains(&threads) {
            return Err("MoEdern threads must be0..2".into());
        }
        let path = CString::new(CrispEmbed::resolve_model(path, download)?)
            .map_err(|_| "Model path contains NUL")?;
        let ctx = unsafe { crispembed_moedern_init(path.as_ptr(), threads) };
        if ctx.is_null() {
            return Err(Self::error(ctx));
        }
        Ok(Self { ctx })
    }
    fn error(ctx: *const c_void) -> String {
        let value = unsafe { crispembed_moedern_error(ctx) };
        if value.is_null() {
            "MoEdern operation failed".into()
        } else {
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }
    }
    pub fn encode(&mut self, text: &str, role: Role) -> Result<Tokens, String> {
        let text = input(text)?;
        let raw = unsafe { crispembed_moedern_encode(self.ctx, text.as_ptr(), role as i32) };
        if raw.is_null() {
            return Err(Self::error(self.ctx));
        }
        let r = unsafe { &*raw };
        if r.abi_version != 1
            || r.struct_bytes as usize != std::mem::size_of::<NativeResult>()
            || r.role != role as i32
            || r.dimension != 128
            || r.n_selected < 1
            || r.n_selected > r.n_tokens
            || r.n_tokens > 512
            || r.expansion_count
                != if matches!(role, Role::Query) {
                    r.n_tokens
                } else {
                    0
                }
        {
            return Err("MoEdern result ABI/shape differs".into());
        }
        let n = r.n_tokens as usize;
        let selected = r.n_selected as usize;
        let result = Tokens {
            role,
            n_tokens: n,
            n_selected: selected,
            dimension: 128,
            ids: unsafe { copied(r.ids, n)? },
            supplied_mask: unsafe { copied(r.supplied_mask, n)? },
            effective_mask: unsafe { copied(r.effective_mask, n)? },
            scoring_mask: unsafe { copied(r.scoring_mask, n)? },
            expansion_positions: unsafe {
                copied(r.expansion_positions, r.expansion_count as usize)?
            },
            all_vectors: unsafe { copied(r.all_vectors, n * 128)? },
            selected_vectors: unsafe { copied(r.selected_vectors, selected * 128)? },
        };
        if !result
            .all_vectors
            .iter()
            .chain(result.selected_vectors.iter())
            .all(|v| v.is_finite())
        {
            return Err("Nonfinite MoEdern vectors".into());
        }
        Ok(result)
    }
    pub fn score(&mut self, query: &str, document: &str) -> Result<f32, String> {
        let q = input(query)?;
        let d = input(document)?;
        let mut value = 0.;
        if unsafe { crispembed_moedern_score(self.ctx, q.as_ptr(), d.as_ptr(), &mut value) } != 1 {
            return Err(Self::error(self.ctx));
        }
        if !value.is_finite() {
            return Err("Nonfinite MoEdern score".into());
        }
        Ok(value)
    }
}
impl Drop for MoEdern {
    fn drop(&mut self) {
        unsafe { crispembed_moedern_free(self.ctx) };
        self.ctx = std::ptr::null_mut();
    }
}
