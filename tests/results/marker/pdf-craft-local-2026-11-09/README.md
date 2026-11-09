# PDF Craft feature: local execution evidence

**Final corrected native source:**
[`891ea3d3632c6a3c3155e25b34705da6a02ec807`](https://github.com/CrispStrobe/CrispEmbed/commit/891ea3d3632c6a3c3155e25b34705da6a02ec807).
The [final receipt](final-receipt.json) binds successful replay to the rebuilt
library SHA-256 `decb81ce3f68f30db24cc6eb686f712750e86f68dbd5f9cc6930d65f87210328`.
The original execution and failures remain below and in separate attempt folders.

## Final corrected-build results

| Executed control | Result | Evidence |
| --- | --- | --- |
| Native document / artifact / lifecycle / save | 185 / 87 / 480 / 38 checks, all pass | [Native replay](final-native/receipt.json) |
| Actual PDFium backend / C API | 39 / 86 checks, all pass | [Native replay](final-native/receipt.json) |
| Actual Rust and Dart clients | Six scenarios, 111 assertions, all pass | [Full client results](final-clients/results.json) |
| Python/CLI/HTTP on eight owned PDFs | 112/112 | [Public surfaces](final-public-surfaces/summary.json) |
| Strict real-PDF lifecycle / cross-build artifacts | 48/48 and 14/14 | [Real-PDF summary](real-pdf-lifecycle/final-summary.json) |
| TeleOCR loader / Jina admission / actual Python lifecycle | 57 / 106 / 33 checks, all pass | [Native Python replay](real-pdf-lifecycle/final-summary.json) |
| Current Python/Dart wrappers with immutable older v0.17.15 | 68/68 | [Older-library compatibility](legacy-v0.17.15/README.md) |

The real-PDF comparison exposed four stable provenance fields missing from
incremental views. Both the [native regression](provenance/provenance-regression-before.json)
and [strict real-PDF attempt 4](real-pdf-lifecycle/attempt-4/receipt.json) fail on
the original build. The correction writes the existing provenance into the
initial raw checkpoint, so empty/partial/complete snapshots and their processed
views retain it. Processed reopening keeps the original artifact metadata.

The final real-PDF test excludes only `elapsed_seconds` between separate runs,
plus `completed_pages`, `next_page`, `extraction_phase` and `extraction_state`
between ordinary and incremental APIs. Every other JSON/chunks field remains
exact; Markdown/HTML match byte for byte. All four processed-artifact outputs
are byte-exact with no exclusions. Fourteen cross-build checks preserve earlier
processed bytes and metadata, while raw resume correctly rejects the changed
build identity without modifying old files.

Actual older-library probes verify that all six new symbols are absent in the
hash-verified v0.17.15 binary. Each new-method failure preserves all four earlier
renderings, and subsequent legacy conversion still works. No wrapper change was
needed. Initial probe-only PDFium ownership and renderer/timing expectations are
retained separately from the production provenance defect and final results.

The 19 converter, 17 Python client and package-attribution controls below retain
their initial source/binary identities. Their inputs were unchanged by the
provenance correction. Final Rust/Dart dependency locks match the retained
initial locks exactly. Windows/macOS execution, full Flutter/archive/app checks
and representative OCR/retrieval quality remain unqualified.

## Initial execution checkpoint

The owner authorized local builds and tests on 9 November 2026 while GitHub
Actions remained queued. This receipt records actual Linux x86_64 CPU execution
of native source `d4d508c460c6e172fd1d7d4afc6b9c78652ad607` on feature branch
`feature/pdf-craft-document-quality-20261109` ([draft PR 192](https://github.com/CrispStrobe/CrispEmbed/pull/192)).
That source includes main through `dc4f51c755bf78139f1f6ea87941748b05e75971`.

### Initial results

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
