# PDF Craft feature: local execution evidence

The owner authorized local builds and tests on 9 November 2026 while GitHub
Actions remained queued. This receipt records actual Linux x86_64 CPU execution
of native source `d4d508c460c6e172fd1d7d4afc6b9c78652ad607` on feature branch
`feature/pdf-craft-document-quality-20261109` ([draft PR 192](https://github.com/CrispStrobe/CrispEmbed/pull/192)).
That source includes main through `dc4f51c755bf78139f1f6ea87941748b05e75971`.

## Results

| Executed control | Result | Evidence |
| --- | --- | --- |
| Native document/source links/repetition/Unicode/references | 185/185 | [Native first attempt](native/native-suite-attempt-1.json) |
| Native artifact round trips, malformed data, limits and atomic failure | 87/87 | [Native corrected environment](native/native-suite-attempt-2.json) |
| Native bounded lifecycle, cancellation/resume, identity and rollback | 400/400 | [Native corrected environment](native/native-suite-attempt-2.json) |
| Native bundle save and alias protection | 38/38 | [Native first attempt](native/native-suite-attempt-1.json) |
| Real PDFium backend / C API | 39/39 and 86/86 | [Real PDFium](native/real-pdfium-attempt-1.json) |
| Real PDFium Python/CLI/HTTP on eight owned PDFs | 112/112 | [Public surfaces](real-pdf-clients/summary.json) |
| TeleOCR converter with actual synthetic GGUF readback | 19/19 | [Converter](python/test_teleocr_converter-attempt-1.json) |
| Python client controls | 17/17 | [Clients](python/test_marker_clients-attempt-1.json) |
| TeleOCR native metadata/admission | 57/57 | [TeleOCR loader](python/test_teleocr_loader-attempt-1.json) |
| Jina/CC-BY-NC native direct, local and cached admission | 106/106 | [Noncommercial admission](python/test_unlimited_ocr_license-attempt-1.json) |
| Actual native Python lifecycle | 33/33 | [Python lifecycle](python/test_marker_lifecycle-attempt-1.json) |
| Actual Rust/Dart clients | Six scenarios, 111 named assertions | [Client evidence](clients/README.md) |
| Python wheel/sdist/rebuilt wheel and notice negatives | 26/26 | [Package evidence](packages/README.md) |
| Flutter dry-run notice selection | 15/15 scoped checks | [Package evidence and limits](packages/README.md) |
| CI selection, environment inventory, clean exit, workflow pins and pin controls | Five successful commands | [Repository guards](guards/summary.json) |

The native API/backend first attempt also passes 74/74 and 18/18 checks without
real PDFium selected. The real-PDFium attempt extends those suites; these counts
overlap and should not be added as independent coverage.

## Source and build identity

[The root receipt](receipt.json) records compiler/build settings, binary hashes,
original and redacted log hashes, and initial failures. GCC 14.2.0, CMake 3.31.10,
Ninja 1.13.0 and pinned GGML `3a27b6c665b191cb230be367db5c52d81dd3e2ac` built
Release static/shared libraries and test binaries. CUDA/Vulkan/Metal and native
CPU probing were disabled; the portable CPU baseline check passed. No new
production dependency or model weights were used.

The tested native and Python sources match the stated commit. Rust and Dart
client fixtures were formatted with the provisioned compilers before execution;
their exact executed hashes and resolved dependency locks are retained in the
client evidence. Documentation/evidence commits do not change these native bytes.
Python uses 3.13.5, NumPy 2.3.5, gguf 0.19.0 and Pillow 12.0.0. The actual PDFium
library comes from existing test dependency pypdfium2 5.10.1 / PDFium 151.0.7891.0;
its hash is recorded independently from the owned ABI fixture.

## Initial attempts and limits

The first native artifact and lifecycle tests could not obtain the default
temporary directory and failed before any assertions. Both passed on the same
binary after `TMPDIR` was set to an existing task-owned writable directory.
Both attempts and logs are preserved. Initial isolated Python setup found no
system `ensurepip`; bootstrapping that isolated environment with the existing
pip succeeded. Neither required a source or expectation change.

The owned ABI fixture proves lifecycle and FFI behavior without OCR. Separate
real-PDFium tests parse and render owned digital PDFs; they do not qualify model
quality on scanned documents. Windows/macOS, Flutter app installation, a retained
Flutter archive readback and representative OCR/retrieval evaluation remain open.
The successful Flutter selection probe used `--dry-run --skip-validation`
because a Flutter SDK was unavailable; no package was uploaded. Its in-memory
archive stream was not retained, so byte readback is explicitly unqualified.

Build binaries and large setup artifacts are not stored here. Public evidence
uses logical placeholders for local paths; original log hashes are retained
where redaction changed text. No release, model publication or default promotion
follows from these results.

## Reproduce the core native controls

Use an existing writable temporary directory, a C++17 compiler and the pinned
submodule. Configure a Release build with shared libraries enabled and
`CRISPEMBED_NATIVE=OFF`, `GGML_NATIVE=OFF`, `GGML_BLAS=OFF`, `GGML_LLAMAFILE=ON`,
`GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, `GGML_METAL=OFF`. Build and run
`test-marker-document`, `test-marker-artifact`, `test-marker-lifecycle`,
`test-marker-api`, `test-marker-backend` and `test-marker-save`. The lifecycle
target builds its owned fixture library automatically. Run the Python admission
and lifecycle drivers with explicit `--lib` / `--pdfium` arguments, and use
`tests/run_marker_binding_lifecycle.py` for actual Rust/Dart execution. Pass an
explicit real PDFium library to `test-marker-backend` for its integration cases;
pass that library and `tests/fixtures/marker/simpledigital.pdf` to `test-marker-api`.
