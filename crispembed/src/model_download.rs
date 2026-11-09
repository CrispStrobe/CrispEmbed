//! Read-only model registry metadata for application-owned model installers.
//!
//! These calls never download, inspect a cache, accept a license or change the
//! process environment. Consumers must validate HTTPS URLs and safe filenames,
//! require/verify available SHA-256 pins, and honor the reported license before
//! installing a payload. A missing pin is explicitly `None`.

use std::ffi::{c_char, CStr, CString};

/// One exact registry payload, including its optional registered SHA-256 pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadFile {
    pub filename: String,
    pub url: String,
    pub sha256: Option<String>,
}

/// Exact-name model download metadata copied from immutable native storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadInfo {
    pub name: String,
    pub license: String,
    pub model_card_url: String,
    pub requires_license_acceptance: bool,
    pub primary: ModelDownloadFile,
    pub companions: Vec<ModelDownloadFile>,
}

fn copy_string(value: *const c_char) -> Result<String, String> {
    if value.is_null() {
        return Ok(String::new());
    }
    // Safety: the C API guarantees process-lifetime immutable registry strings.
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| "Native model registry contains invalid UTF-8".to_owned())
}

fn copy_file(value: crispembed_sys::CrispembedModelDownloadFileV1) -> Result<ModelDownloadFile, String> {
    Ok(ModelDownloadFile {
        filename: copy_string(value.filename)?,
        url: copy_string(value.url)?,
        sha256: if value.sha256.is_null() {
            None
        } else {
            Some(copy_string(value.sha256)?)
        },
    })
}

/// Look up one exact, case-sensitive registry name. Unknown names return None.
/// Filenames, substrings and paths are intentionally not resolved.
pub fn model_download_info(exact_name: &str) -> Result<Option<ModelDownloadInfo>, String> {
    let name = CString::new(exact_name)
        .map_err(|_| "Model name contains an embedded NUL byte".to_owned())?;
    let mut raw = std::mem::MaybeUninit::<crispembed_sys::CrispembedModelDownloadInfoV1>::uninit();
    // Safety: CString is NUL terminated; raw points to the exact public v1 layout.
    // On status 1 the native API initializes every field and borrows static data.
    let status = unsafe {
        crispembed_sys::crispembed_get_model_download_info_v1(name.as_ptr(), raw.as_mut_ptr())
    };
    match status {
        0 => Ok(None),
        1 => {
            let raw = unsafe { raw.assume_init() };
            let mut companions = Vec::new();
            if raw.has_companion != 0 {
                companions.push(copy_file(raw.companion)?);
            }
            Ok(Some(ModelDownloadInfo {
                name: copy_string(raw.name)?,
                license: copy_string(raw.license)?,
                model_card_url: copy_string(raw.model_card_url)?,
                requires_license_acceptance: raw.requires_license_acceptance != 0,
                primary: copy_file(raw.primary)?,
                companions,
            }))
        }
        _ => Err("Native model registry lookup rejected its arguments".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_native_pin_stays_absent() {
        let filename = CString::new("fixture.gguf").unwrap();
        let url = CString::new("https://example.invalid/fixture.gguf").unwrap();
        let raw = crispembed_sys::CrispembedModelDownloadFileV1 {
            filename: filename.as_ptr(),
            url: url.as_ptr(),
            sha256: std::ptr::null(),
        };
        let copied = copy_file(raw).unwrap();
        assert_eq!(copied.filename, "fixture.gguf");
        assert_eq!(copied.url, "https://example.invalid/fixture.gguf");
        assert_eq!(copied.sha256, None);
    }
}
