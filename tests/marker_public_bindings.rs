use crispembed::{Marker, MarkerMode, MarkerOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "PDFium library and owned PDF are required");
    let opts = MarkerOptions {
        mode: MarkerMode::Text,
        pdfium_path: args[1].clone(),
        ..Default::default()
    };
    let mut marker = Marker::new(opts).expect("native Marker initialization");
    let mut checks = 0;
    for format in ["markdown", "html", "json", "chunks"] {
        let result = marker
            .convert(&args[2], format)
            .expect("actual PDF conversion");
        assert!(result.contains("Hello from CrispEmbed."));
        checks += 1;
    }
    assert!(marker
        .metadata_json()
        .unwrap()
        .contains("crispembed-marker-native"));
    checks += 1;
    assert!(marker.assets().unwrap().is_empty());
    checks += 1;
    assert!(marker.render("bad-format").is_err());
    checks += 1;
    assert!(marker.render("markdown").is_ok());
    checks += 1;
    assert!(marker
        .convert_image(&[255, 255], 2, 2, 3, "markdown")
        .is_err());
    checks += 1;
    assert!(marker.convert("bad\0path", "json").is_err());
    checks += 1;
    assert!(marker.convert("missing-marker-input.pdf", "json").is_err());
    checks += 1;
    assert!(marker.render("json").is_err());
    checks += 1;
    assert!(marker.convert(&args[2], "markdown").is_ok());
    checks += 1;
    println!("{{\"client\":\"Rust\",\"passed\":{checks},\"total\":{checks}}}");
}
