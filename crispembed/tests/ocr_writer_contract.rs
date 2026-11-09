// Linked to the real model-free native renderer and exact production Rust API
// by tools/check_ocr_writer_rust.py on a hosted runner. No wrapper is copied.
#[cfg(test)]
mod writer_contracts {
    use crispembed::*;
    use std::io::{self, Write};
    use std::path::PathBuf;

    fn artifacts() -> PathBuf {
        static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let root = std::env::var_os("OCR_WRITER_ARTIFACTS").map(PathBuf::from).unwrap_or_else(||
                std::env::temp_dir().join(format!("crisp-ocr-writer-test-{}", std::process::id())));
            std::fs::create_dir_all(&root).unwrap();
            if !root.join("rgb.png").exists() {
                // The production decoder recognizes PNM by its header. A
                // dependency-free test image keeps ordinary cargo tests usable.
                let mut image = b"P6\n320 240\n255\n".to_vec();
                image.extend(std::iter::repeat_n(240u8, 320 * 240 * 3));
                std::fs::write(root.join("rgb.png"), image).unwrap();
            }
            root
        }).clone()
    }

    fn region(text: &str) -> OcrRegion {
        OcrRegion {
            text: text.to_owned(),
            x: 20.0,
            y: 30.0,
            w: 280.0,
            h: 30.0,
            confidence: 0.9,
            orientation_corrected: false,
        }
    }

    fn render<W: Write>(writer: &mut W, limit: u64, text: &str) -> io::Result<u64> {
        let regions = [region(text)];
        let image = artifacts().join("rgb.png");
        let page = OcrRenderPageInput {
            regions: &regions,
            page_width: 320,
            page_height: 240,
            image_path: image.to_str(),
        };
        ocr_render_pdf_to_writer_checked(&[page], true, Some(&[(160.0, 120.0)]), false, limit, writer)
    }

    #[derive(Default)]
    struct ShortWriter {
        bytes: Vec<u8>,
        writes: usize,
        flushes: usize,
    }

    impl Write for ShortWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            let size = bytes.len().min(7);
            self.bytes.extend_from_slice(&bytes[..size]);
            Ok(size)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn short_writes_preserve_the_complete_pdf() {
        let mut writer = ShortWriter::default();
        let count = render(&mut writer, 256 * 1024 * 1024, "Rust short writes café 中文 😀").unwrap();
        assert_eq!(count, writer.bytes.len() as u64);
        assert!(writer.writes > 100);
        assert_eq!(writer.flushes, 1);
        assert!(writer.bytes.starts_with(b"%PDF-1.7"));
        assert!(writer.bytes.ends_with(b"%%EOF\n"));
        std::fs::write(artifacts().join("archival-rust-short.pdf"), writer.bytes).unwrap();
    }

    #[test]
    fn preserves_the_original_operating_system_error() {
        const DENIED: i32 = if cfg!(windows) { 5 } else { 13 };
        struct Failing {
            writes: usize,
            flushed: bool,
        }
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                Err(io::Error::from_raw_os_error(DENIED))
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushed = true;
                Ok(())
            }
        }
        let mut writer = Failing { writes: 0, flushed: false };
        let error = render(&mut writer, 256 * 1024 * 1024, "Original I/O error").unwrap_err();
        assert_eq!(error.raw_os_error(), Some(DENIED));
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(writer.writes, 1);
        assert!(!writer.flushed);
    }

    #[test]
    fn cancellation_preserves_interrupted_and_stops_callbacks() {
        struct Cancelled { calls: usize, bytes: usize }
        impl Write for Cancelled {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.calls >= 3 {
                    Err(io::Error::new(io::ErrorKind::Interrupted, "export cancelled"))
                } else {
                    self.bytes += bytes.len();
                    Ok(bytes.len())
                }
            }
            fn flush(&mut self) -> io::Result<()> { panic!("cancelled destination must not flush") }
        }
        let mut writer = Cancelled { calls: 0, bytes: 0 };
        let error = render(&mut writer, 256 * 1024 * 1024, "Cancelled export").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(error.to_string(), "export cancelled");
        assert_eq!(writer.calls, 3);
        assert!(writer.bytes > 0);
    }

    #[test]
    fn zero_progress_is_write_zero() {
        struct Zero;
        impl Write for Zero {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> { Ok(0) }
            fn flush(&mut self) -> io::Result<()> { panic!("must not flush failed writer") }
        }
        let error = render(&mut Zero, 256 * 1024 * 1024, "No progress").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
    }

    #[test]
    fn writer_panic_never_unwinds_across_c_and_next_job_recovers() {
        struct Panics;
        impl Write for Panics {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("injected writer panic") }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let error = render(&mut Panics, 256 * 1024 * 1024, "Panic containment").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(error.to_string(), "PDF writer panicked");
        let mut recovered = Vec::new();
        assert!(render(&mut recovered, 256 * 1024 * 1024, "Recovered job").unwrap() > 0);
        assert!(recovered.ends_with(b"%%EOF\n"));
    }

    #[test]
    fn panic_payload_destructor_cannot_unwind_across_c() {
        use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
        struct Payload(Arc<AtomicUsize>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
                panic!("injected panic-payload destructor");
            }
        }
        struct Panics(Arc<AtomicUsize>);
        impl Write for Panics {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                std::panic::panic_any(Payload(self.0.clone()))
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut writer = Panics(dropped.clone());
        let error = render(&mut writer, 256 * 1024 * 1024, "Panic payload containment").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(error.to_string(), "PDF writer panicked");
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn flush_failure_is_returned_after_complete_write() {
        struct FlushFailure(Vec<u8>);
        impl Write for FlushFailure {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.write(bytes) }
            fn flush(&mut self) -> io::Result<()> { Err(io::Error::from_raw_os_error(28)) }
        }
        let mut writer = FlushFailure(Vec::new());
        let error = render(&mut writer, 256 * 1024 * 1024, "Flush error").unwrap_err();
        assert_eq!(error.raw_os_error(), Some(28));
        assert!(writer.0.ends_with(b"%%EOF\n"));
    }

    #[test]
    fn low_limit_never_writes_past_the_requested_bound() {
        let text: String = (0x4e00..0x4f00).map(|cp| char::from_u32(cp).unwrap()).collect();
        let mut writer = Vec::new();
        let error = render(&mut writer, 4096, &text).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("byte limit"));
        assert!(!writer.is_empty() && writer.len() <= 4096);
        assert!(!writer.ends_with(b"%%EOF\n"));
        let mut recovered = Vec::new();
        assert!(render(&mut recovered, 256 * 1024 * 1024, &text).unwrap() > 4096);
    }

    #[test]
    fn invalid_configuration_does_not_touch_the_writer() {
        let mut writer = Vec::new();
        assert_eq!(render(&mut writer, 4095, "Low cap").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(writer.is_empty());
        assert_eq!(render(&mut writer, 2 * 1024 * 1024 * 1024 + 1, "High cap").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(writer.is_empty());
        assert_eq!(render(&mut writer, 256 * 1024 * 1024, "invalid\0text").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(writer.is_empty());
    }

    #[test]
    fn real_file_retains_two_pages_and_physical_sizes() {
        let first = [region("Rust file page one café 中文 😀")];
        let second = [region("Rust file page two complete")];
        let image = artifacts().join("rgb.png");
        let pages = [OcrRenderPageInput { regions: &first, page_width: 320, page_height: 240, image_path: image.to_str() },
                     OcrRenderPageInput { regions: &second, page_width: 320, page_height: 240, image_path: image.to_str() }];
        let path = artifacts().join("archival-rust-file.pdf");
        let mut file = std::fs::File::create(&path).unwrap();
        let count = ocr_render_pdf_to_writer_checked(&pages, true, Some(&[(160.0, 120.0), (180.0, 135.0)]),
            false, 256 * 1024 * 1024, &mut file).unwrap();
        file.sync_all().unwrap();
        assert_eq!(count, file.metadata().unwrap().len());
    }

    #[test]
    #[ignore = "explicit hosted large-file gate with retained PDF and resource receipt"]
    fn streams_above_the_legacy_buffer_limit() {
        let root = PathBuf::from(std::env::var_os("OCR_WRITER_LARGE").expect("large artifact path"));
        let image = root.join("source.png");
        let regions: Vec<_> = (1..=11).map(|index| vec![OcrRegion {
            text: format!("Large streamed page {index} of 11 complete"),
            x: 100.0, y: 100.0, w: 1800.0, h: 60.0, confidence: 1.0, orientation_corrected: false,
        }]).collect();
        let pages: Vec<_> = regions.iter().map(|regions| OcrRenderPageInput {
            regions, page_width: 2048, page_height: 2048, image_path: image.to_str(),
        }).collect();
        let sizes = vec![(512.0, 512.0); pages.len()];
        let path = root.join("archival-large-stream.pdf");
        let mut file = std::fs::File::create(&path).unwrap();
        let count = ocr_render_pdf_to_writer_checked(&pages, true, Some(&sizes), false,
            256 * 1024 * 1024, &mut file).unwrap();
        file.sync_all().unwrap();
        assert_eq!(count, file.metadata().unwrap().len());
        assert!(count > 128 * 1024 * 1024, "fixture must exceed the byte API ceiling");
        assert!(count < 256 * 1024 * 1024);
        println!("STREAMED_BYTES={count}");
    }
}
