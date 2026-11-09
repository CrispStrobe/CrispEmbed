// Actual Rust/native calls. The lifecycle mode uses the owned PDFium ABI
// fixture, not a PDF parser or model, and needs test-only serde_json = "1".
use crispembed::{Marker, MarkerJobStatus, MarkerMode, MarkerOptions};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn legacy(args: &[String]) {
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

struct OwnedDirectory(PathBuf);

impl OwnedDirectory {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "marker-rust-lifecycle-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create exclusive owned fixture directory");
        Self(path)
    }

    fn file(&self, name: &str) -> String {
        self.0
            .join(name)
            .to_str()
            .expect("UTF-8 fixture path")
            .to_owned()
    }
}

impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct Checks(Vec<Value>);

impl Checks {
    fn check(&mut self, condition: bool, name: &str) {
        assert!(condition, "{name}");
        self.0.push(json!({"name": name, "passed": true}));
    }

    fn rejects<T>(&mut self, result: Result<T, String>, fragment: &str, name: &str) {
        match result {
            Err(error) => self.check(error.contains(fragment), &format!("{name}: {error}")),
            Ok(_) => panic!("{name}: operation unexpectedly succeeded"),
        }
    }

    fn progress(
        &mut self,
        marker: &mut Marker,
        state: &str,
        completed: usize,
        next: usize,
        name: &str,
    ) -> Value {
        let value: Value =
            serde_json::from_str(&marker.progress_json().expect("native progress JSON")).unwrap();
        self.check(
            value["schema"] == "crispembed.marker-progress/1"
                && value["state"] == state
                && value["completed_pages"] == completed
                && value["next_page"] == next
                && value["total_pages"] == if state == "idle" { 0 } else { 3 },
            name,
        );
        value
    }

    fn report(&self, scenario: &str) {
        println!(
            "{}",
            json!({
                "client": "Rust", "suite": "native-lifecycle", "scenario": scenario,
                "model_weights_used": false, "pdf_rendering_qualified": false,
                "checks": self.0, "passed": self.0.len(), "total": self.0.len()
            })
        );
    }
}

fn fixture_options(pdfium: &str) -> MarkerOptions {
    MarkerOptions {
        mode: MarkerMode::Text,
        pdfium_path: pdfium.to_owned(),
        extract_images: false,
        ..Default::default()
    }
}

fn render_all(marker: &mut Marker) -> Vec<String> {
    ["markdown", "html", "json", "chunks"]
        .into_iter()
        .map(|format| marker.render(format).expect("native lifecycle rendering"))
        .collect()
}

fn source_maps_match(chunks: &Value, pages: usize) -> bool {
    let sources = chunks["source_records"].as_array().expect("source records");
    let assembled = chunks["assembled_chunks"]
        .as_array()
        .expect("assembled chunks");
    if sources.len() != pages || assembled.len() != pages {
        return false;
    }
    for chunk in assembled {
        let text = chunk["text"].as_str().unwrap().as_bytes();
        let mut covered = 0;
        for range in chunk["source_ranges"].as_array().expect("source ranges") {
            let source = sources
                .iter()
                .find(|source| source["id"] == range["source_id"]);
            let Some(source) = source else { return false };
            let original = source["text"].as_str().unwrap().as_bytes();
            let start = range["text_start"].as_u64().unwrap() as usize;
            let end = range["text_end"].as_u64().unwrap() as usize;
            let source_start = range["source_start"].as_u64().unwrap() as usize;
            let source_end = range["source_end"].as_u64().unwrap() as usize;
            if range["transform"] != "copy"
                || start != covered
                || start > end
                || end > text.len()
                || source_start > source_end
                || source_end > original.len()
                || text[start..end] != original[source_start..source_end]
            {
                return false;
            }
            covered = end;
        }
        if covered != text.len() {
            return false;
        }
    }
    true
}

fn lifecycle(pdfium: &str, scenario: &str) {
    assert!(
        ["artifacts-off", "incremental-off", "enabled"].contains(&scenario),
        "unknown lifecycle scenario"
    );
    for (name, expected) in [
        ("CRISPEMBED_MARKER", "1"),
        (
            "CRISPEMBED_MARKER_ARTIFACTS",
            if scenario == "artifacts-off" {
                "0"
            } else {
                "1"
            },
        ),
        (
            "CRISPEMBED_MARKER_INCREMENTAL",
            if scenario == "incremental-off" {
                "0"
            } else {
                "1"
            },
        ),
        ("CRISPEMBED_MARKER_SOURCE_LINKS", "1"),
        ("CRISPEMBED_MARKER_REFERENCES", "1"),
        ("CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE", "0"),
        ("CRISPEMBED_MARKER_REPETITION_CHECK", "0"),
        ("CRISPEMBED_TELEOCR", "0"),
        ("CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE", "0"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "explicit probe gate {name}"
        );
    }
    let folder = OwnedDirectory::new();
    let source = folder.file("owned.pdf");
    let raw = folder.file("raw.cemk");
    let partial_view = folder.file("partial.cemk");
    let view = folder.file("complete.cemk");
    fs::write(
        &source,
        b"%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nowned Rust client input\n",
    )
    .unwrap();
    let options = fixture_options(pdfium);
    let mut marker = Marker::new(options.clone()).expect("model-free native Marker context");
    let mut checks = Checks::default();
    checks.progress(&mut marker, "idle", 0, 0, "new native context is idle");

    if scenario != "enabled" {
        let output = marker
            .convert(&source, "markdown")
            .expect("legacy native conversion with lifecycle gate off");
        checks.check(
            output.contains("Third owned page is complete."),
            "gate-off keeps synchronous conversion usable",
        );
        let gate = if scenario == "artifacts-off" {
            checks.rejects(
                marker.save_extraction(&view),
                "CRISPEMBED_MARKER_ARTIFACTS",
                "native artifact save gate",
            );
            checks.rejects(
                marker.open_extraction(&view),
                "CRISPEMBED_MARKER_ARTIFACTS",
                "native artifact open gate",
            );
            checks.check(
                !Path::new(&view).exists(),
                "closed artifact gate writes no file",
            );
            "CRISPEMBED_MARKER_ARTIFACTS"
        } else {
            marker
                .save_extraction(&view)
                .expect("artifact save independent of incremental gate");
            marker
                .open_extraction(&view)
                .expect("artifact open independent of incremental gate");
            checks.check(
                marker.render("markdown").unwrap() == output,
                "artifacts work with incremental gate off",
            );
            "CRISPEMBED_MARKER_INCREMENTAL"
        };
        checks.rejects(
            marker.start(&source, &raw, false),
            gate,
            "native start gate reaches Rust",
        );
        checks.rejects(marker.step(1), gate, "native step gate reaches Rust");
        checks.check(
            !Path::new(&raw).exists() && marker.render("markdown").unwrap() == output,
            "rejected lifecycle calls preserve view and create no checkpoint",
        );
        checks.report(scenario);
        return;
    }

    marker
        .start(&source, &raw, false)
        .expect("actual native job start");
    checks.progress(
        &mut marker,
        "running",
        0,
        0,
        "start creates an empty three-page job",
    );
    let initial = fs::read(&raw).unwrap();
    checks.check(
        initial.starts_with(b"CRISPMK\0"),
        "start writes a native extraction checkpoint",
    );
    for budget in [0, 65] {
        checks.rejects(
            marker.step(budget),
            "page budget",
            "Rust rejects invalid page budget",
        );
    }
    checks.check(
        fs::read(&raw).unwrap() == initial,
        "wrapper validation leaves checkpoint bytes unchanged",
    );
    checks.check(
        marker.step(1).unwrap() == MarkerJobStatus::Running,
        "one-page native step remains running",
    );
    checks.progress(
        &mut marker,
        "running",
        1,
        1,
        "one-page step advances only one cursor",
    );
    let partial = render_all(&mut marker);
    let committed = fs::read(&raw).unwrap();
    checks.check(
        committed != initial
            && partial[0].contains("First owned page")
            && !partial[0].contains("Second owned page"),
        "partial Rust view contains only the committed page",
    );
    marker
        .save_extraction(&partial_view)
        .expect("save partial processed view");
    let mut reader_options = fixture_options("");
    reader_options.dpi = 150;
    {
        let mut reader = Marker::new(reader_options.clone()).unwrap();
        reader
            .open_extraction(&partial_view)
            .expect("partial reopen without PDFium or model paths");
        checks.check(
            render_all(&mut reader) == partial,
            "partial processed artifact preserves all four renders",
        );
        let metadata: Value = serde_json::from_str(&reader.metadata_json().unwrap()).unwrap();
        checks.check(
            metadata["configuration"]["extraction_state"] == "partial",
            "partial reopen retains extraction state",
        );
    }
    checks.rejects(
        marker.save_extraction(&raw),
        "separately",
        "processed save cannot overwrite raw checkpoint",
    );
    marker.cancel();
    let progress = checks.progress(
        &mut marker,
        "running",
        1,
        1,
        "cancellation request preserves current cursor",
    );
    checks.check(
        progress["cancel_requested"] == true,
        "Rust cancellation reaches the native flag",
    );
    checks.check(
        marker.step(64).unwrap() == MarkerJobStatus::Cancelled,
        "next boundary observes cancellation",
    );
    checks.progress(
        &mut marker,
        "cancelled",
        1,
        1,
        "cancelled progress retains completed units",
    );
    checks.check(
        marker.step(1).unwrap() == MarkerJobStatus::Cancelled
            && fs::read(&raw).unwrap() == committed,
        "repeated cancelled steps do not rewrite the checkpoint",
    );
    checks.check(
        render_all(&mut marker) == partial,
        "cancelled view stays inspectable",
    );
    drop(marker);

    let mut resumed = Marker::new(options.clone()).unwrap();
    resumed
        .start(&source, &raw, true)
        .expect("fresh-context native resume");
    let progress = checks.progress(
        &mut resumed,
        "running",
        1,
        1,
        "fresh context resumes committed cursor",
    );
    checks.check(
        progress["cancel_requested"] == false,
        "resume clears cancellation",
    );
    checks.check(
        resumed.step(1).unwrap() == MarkerJobStatus::Running,
        "resumed page budget remains bounded",
    );
    checks.progress(
        &mut resumed,
        "running",
        2,
        2,
        "resumed step commits only the next page",
    );
    checks.check(
        resumed.step(64).unwrap() == MarkerJobStatus::Complete,
        "resumed job completes remaining page",
    );
    checks.progress(
        &mut resumed,
        "complete",
        3,
        3,
        "completed progress is exact",
    );
    let complete = render_all(&mut resumed);
    checks.check(
        ["First owned page", "Second owned page", "Third owned page"]
            .iter()
            .all(|text| complete[0].matches(*text).count() == 1),
        "resume neither loses nor duplicates pages",
    );
    let document: Value = serde_json::from_str(&complete[2]).unwrap();
    let pages = document["children"].as_array().unwrap();
    checks.check(
        pages.len() == 3
            && ["iv", "章-2", "📖-3"].iter().enumerate().all(|(i, label)| {
                pages[i]["printed_page_label"] == *label && pages[i]["physical_page_index"] == i
            }),
        "Rust preserves distinct physical indices and Unicode printed labels",
    );
    checks.check(
        source_maps_match(&serde_json::from_str(&complete[3]).unwrap(), 3),
        "Rust chunk source ranges match exact immutable UTF-8 bytes",
    );
    checks.check(
        resumed.assets().unwrap().is_empty(),
        "text-only lifecycle has no image assets",
    );
    let complete_raw = fs::read(&raw).unwrap();
    checks.check(
        resumed.step(1).unwrap() == MarkerJobStatus::Complete
            && fs::read(&raw).unwrap() == complete_raw,
        "completed step is idempotent",
    );
    resumed
        .save_extraction(&view)
        .expect("save completed processed artifact");
    drop(resumed);

    let mut reader = Marker::new(reader_options).unwrap();
    reader
        .open_extraction(&view)
        .expect("standalone processed reopen");
    checks.check(
        render_all(&mut reader) == complete,
        "completed reopen preserves all four renders and source maps",
    );
    checks.rejects(
        reader.open_extraction(&raw),
        "matching processing settings",
        "raw checkpoint rejects changed reader configuration",
    );
    checks.check(
        render_all(&mut reader) == complete,
        "failed raw open preserves prior processed view",
    );
    let mut corrupt = fs::read(&view).unwrap();
    corrupt[24] = if corrupt[24] == b'0' { b'1' } else { b'0' };
    let broken = folder.file("corrupted.cemk");
    fs::write(&broken, corrupt).unwrap();
    checks.rejects(
        reader.open_extraction(&broken),
        "integrity mismatch",
        "native corruption error reaches Rust",
    );
    checks.check(
        render_all(&mut reader) == complete,
        "corrupt open preserves the complete Rust view",
    );
    let mut finished = Marker::new(options.clone()).unwrap();
    finished
        .start(&source, &raw, true)
        .expect("resume completed checkpoint");
    checks.check(
        finished.step(1).unwrap() == MarkerJobStatus::Complete
            && fs::read(&raw).unwrap() == complete_raw,
        "fresh-context completed resume performs no extra work",
    );
    checks.check(
        finished.render("markdown").unwrap() == complete[0],
        "completed raw resume retains exact text",
    );

    let failing = folder.file("failing.pdf");
    let failed_raw = folder.file("failed.cemk");
    fs::write(
        &failing,
        b"%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nFAIL_SECOND_PAGE\n",
    )
    .unwrap();
    let mut failed = Marker::new(options.clone()).unwrap();
    failed.start(&failing, &failed_raw, false).unwrap();
    checks.rejects(failed.step(64), "page 1", "later page failure reaches Rust");
    checks.progress(
        &mut failed,
        "error",
        1,
        1,
        "native failure retains committed progress",
    );
    checks.check(
        failed.render("markdown").unwrap() == partial[0],
        "failed job retains its complete first page",
    );
    let retained = fs::read(&failed_raw).unwrap();
    drop(failed);
    let mut retry = Marker::new(options).unwrap();
    retry.start(&failing, &failed_raw, true).unwrap();
    checks.rejects(
        retry.step(1),
        "page 1",
        "failed checkpoint resumes at the uncompleted page",
    );
    checks.check(
        fs::read(&failed_raw).unwrap() == retained,
        "failed retry preserves durable checkpoint bytes",
    );
    checks.report(scenario);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--lifecycle") {
        assert_eq!(
            args.len(),
            4,
            "--lifecycle <owned-PDFium-library> <artifacts-off|incremental-off|enabled>"
        );
        lifecycle(&args[2], &args[3]);
    } else {
        legacy(&args);
    }
}
