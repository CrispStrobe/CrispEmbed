// Actual FFI and safe wrapper linked to the production model manager.
// No mocked FFI, models or downloads.
extern crate self as crispembed_sys;
#[path = "../crispembed-sys/src/lib.rs"]
mod ffi;
pub use ffi::*;
#[path = "../crispembed/src/model_download.rs"]
mod model_download;

use model_download::model_download_info;

#[test]
fn exact_ocr_metadata_is_owned_and_pinned() {
    let metadata = model_download_info("ppocrv6-tiny-rec").unwrap().unwrap();
    assert_eq!(metadata.name, "ppocrv6-tiny-rec");
    assert_eq!(metadata.primary.filename, "PP-OCRv6_tiny_rec-q8-head.gguf");
    assert_eq!(metadata.primary.sha256.as_deref(),
        Some("4a1243c4fe48a24153163f30c087c74c8bccba8b77df2bbf2b8c0783e48c6377"));
    assert!(metadata.primary.url.starts_with("https://"));
    assert!(!metadata.requires_license_acceptance);
    assert!(metadata.companions.is_empty());
    let second = model_download_info("lfm2-vl").unwrap().unwrap();
    assert_eq!(metadata.name, "ppocrv6-tiny-rec");
    assert_eq!(second.license, "lfm1.0");
    assert!(second.requires_license_acceptance);
    assert_eq!(second.companions.len(), 1);
    assert_eq!(second.companions[0].filename, "mmproj-LFM2.5-VL-3B-F16.gguf");
    assert_eq!(second.companions[0].sha256.as_deref(),
        Some("6f912eb9fcef619b31c9afff7662a222da9ce6db89327eb465dbafd2fb6a2d7c"));
}

#[test]
fn unknown_fuzzy_filename_and_nul_requests_do_not_resolve() {
    for name in ["", "ppocrv6", "PPOCRV6-TINY-REC", "PP-OCRv6_tiny_rec-q8-head.gguf", "/tmp/model.gguf"] {
        assert_eq!(model_download_info(name).unwrap(), None);
    }
    assert!(model_download_info("ppocrv6-tiny-rec\0ignored").is_err());
}

#[test]
fn simultaneous_reads_preserve_separate_owned_results() {
    let handles: Vec<_> = (0..8).map(|i| std::thread::spawn(move || {
        let name = if i % 2 == 0 { "ppocrv6-tiny-det" } else { "lfm2-vl" };
        for _ in 0..20 {
            let info = model_download_info(name).unwrap().unwrap();
            assert_eq!(info.name, name);
            assert!(info.primary.sha256.is_some());
        }
    })).collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
