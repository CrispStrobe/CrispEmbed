# Marker artifacts and incremental extraction

This opt-in implementation is tracked in [the PDF Craft task list](pdf-craft-tasks.md).
Validation is tracked by exact source and platform; local Linux execution is
owner-authorized while hosted checks remain queued. Model-free lifecycle controls
do not qualify OCR accuracy, cancellation latency or corpus-scale throughput.

## Enable before creating a context

```sh
export CRISPEMBED_MARKER=1
export CRISPEMBED_MARKER_ARTIFACTS=1
export CRISPEMBED_MARKER_INCREMENTAL=1
```

All gates use `core_env::on()`: unset, empty, `0`, `false`, `off` and `no` are off,
case-insensitively. Auto/OCR modes also require the existing TeleOCR feature and
Qwen Research acknowledgement. Text-mode contexts and processed artifact viewing
load no model weights. The existing `convert()` API remains all-or-nothing.

## Extract, checkpoint and reopen

```python
from crispembed import CrispMarker

with CrispMarker(mode="text", pdfium_path="/path/to/libpdfium.so") as marker:
    marker.start("book.pdf", "book.checkpoint.cemk", resume=False)
    while marker.progress["state"] == "running":
        progress = marker.step(page_budget=1)
        print(progress["completed_pages"], progress["total_pages"])
    marker.save_extraction("book.extraction.cemk")
    markdown = marker.render("markdown")

# A processed artifact needs neither the original PDF nor PDFium for viewing.
with CrispMarker(mode="text") as viewer:
    markdown = viewer.open_extraction("book.extraction.cemk")
    chunks = viewer.render("chunks")
```

The PDFium path above is an illustrative caller-selected local path. Create the
checkpoint's parent directory first. A new job refuses an existing checkpoint;
choose `resume=True` or a new filename. Checkpoint paths cannot alias the input,
model or PDFium library, including existing hard links. Save a processed view
separately from the raw resumable checkpoint.

Call `cancel()` to request cancellation at a page boundary. A later context
resumes with the same extraction options and
`start("book.pdf", "book.checkpoint.cemk", resume=True)`. A completed checkpoint
resumes directly to a complete view without extracting pages again.

## Public interfaces

| C suffix after `crispembed_marker_` | Python | Rust | Dart |
| --- | --- | --- | --- |
| `save_extraction` | `save_extraction(path)` | `save_extraction(path)` | `saveExtraction(path)` |
| `open_extraction` | `open_extraction(path, format)` | `open_extraction(path)` | `openExtraction(path)` |
| `start` | `start(input, checkpoint, resume=...)` | `start(input, checkpoint, resume)` | `start(input, checkpoint, resume: ...)` |
| `step` | `step(page_budget)` | `step(page_budget)` | `step(pageBudget)` |
| `cancel` | `cancel()` | `cancel()` | `cancel()` |
| `progress` | `progress` | `progress_json()` | `progress` |

The shipped C configuration struct layout is unchanged. New symbols are additive.
Python and Dart look them up only when called, preserving legacy conversion on
older native libraries. Rust consumers of new methods need a matching native
library. Returned strings/dictionaries and safe-wrapper assets are owned copies.

`start` returns success/failure. Native `step` returns `ERROR=0`, `RUNNING=1`,
`COMPLETE=2` or `CANCELLED=3` and accepts a page budget of 1 through 64. Python
raises on error and returns progress on success; Rust/Dart return a typed status.
An invalid step argument or disabled gate puts an active job in the error state;
correct the problem and explicitly resume from the checkpoint.

Progress schema `crispembed.marker-progress/1` contains `state`,
`completed_pages`, `total_pages`, `next_page` and `cancel_requested`. Physical
indices are zero-based, selected first/last bounds inclusive, and `next_page`
is the first uncompleted physical page. Query between bounded step calls.
Partial rendering explicitly records `extraction_state=partial`.

Cancellation is cooperative: it does not interrupt a PDFium call or token loop
mid-page. Python can request it from another thread; the native caller must keep
the context alive until the request returns. Rust/Dart can cancel between bounded
calls. No background thread or unbounded queue is created by these APIs.

## Durability and memory

Each completed page is atomically checkpointed before the next starts. The raw
page records are retained independently of the processed view: grouping,
normalization, source linking and references run on a copy. A page failure or
checkpoint write failure discards that page and its new assets while retaining
earlier completed units. The old checkpoint survives failed replacement.
The primary error remains separately inspectable from partial output.

Only one page raster is live at a time. Accumulated raw IR and a processed view
remain in memory, bounded by page/block/asset and artifact limits; this first
implementation is not a constant-memory document indexer. Completed raw
extraction can be retried via resume if later document processing fails.

Legacy conversion still clears results on failure. Artifact opening validates
into a temporary document before replacing the current view, so a failed open
preserves the previous usable result. See [the format](marker-artifact.md) for
encoding, resource limits, source-map validation and atomic-file behavior.

## Identity and reuse

Resume compares the full identity map and selected page bounds before reusing
pages. It records SHA-256 identities for source bytes, model weights when used,
explicit PDFium, processing settings, runtime settings and native source/build
configuration. The build fingerprint hashes packaged native/ggml sources and
compilation settings without requiring Git. Changes invalidate resume
conservatively; compiled/native source identity is separate from model quality.

Model identity/settings are recorded around initialization. Reconfiguration
requires a new context. Identities are checked again before each step, and
stateful forced-token/reference injection is rejected. Keep selected files
immutable during a step: hashing is not a filesystem snapshot or an author
signature. Passwords and absolute input/model paths are not serialized in the
identity map. Floating-point/model equivalence across hardware is not claimed.

Incremental PDF extraction requires explicit PDFium bytes for identity matching.
Opening a raw checkpoint requires matching processing settings. A processed
artifact reopens without processing or inference; both paths enforce integrity
and bounded parsing. Opening extracted content does not grant rights to use a
restricted model.

## Validation tasks

- [x] Artifact roundtrip, malformed-input and atomic-failure tests: 87 checks.
- [x] Native lifecycle controls with an owned PDFium ABI fixture: 400 checks;
  budgets, Unicode labels, cancellation, fresh-context resume, stale identity,
  corrupted checkpoint and later-page failure.
- [x] Actual-native Python lifecycle: 33 checks; the separate 17 Python client
  controls cover image alpha handling and bundle ownership/preflight.
- [ ] Execute legacy conversion against an older native library lacking the
  new lifecycle symbols; source review confirms lazy symbol lookup.
- [x] Actual Rust/Dart lifecycle execution in six separately gated
  scenarios, including partial/full four-render reopen and exact source maps.
  All 111 named assertions pass.
- [ ] Hosted Windows document/artifact/lifecycle controls, Xcode generation
  and CocoaPods syntax checks.
- [ ] Exact final-source ordinary CI and independent review recorded in the
  [task checklist](pdf-craft-tasks.md).

The owned ABI fixture tests lifecycle and marshaling; it is not a substitute for
real PDF rendering or representative model/document quality evaluation.

Checked items above were executed on Linux CPU at native source
`d4d508c460c6e172fd1d7d4afc6b9c78652ad607`. The
[retained evidence](../tests/results/marker/pdf-craft-local-2026-11-09/README.md)
identifies toolchains, actual binaries and client sources, initial environment
failures and remaining platform limits.

The Linux Build job runs the real public clients through
`tests/run_marker_binding_lifecycle.py`. It compiles isolated Rust and Dart
consumers against the already-built shared library and owned fixture, then runs
`artifacts-off`, `incremental-off` and `enabled` in each language. The supervisor
requires named nonempty results, successful process exits and all six scenarios;
timeouts, crashes and missing reports fail. Its JSON, command logs, native/source
hashes and resolved dependency locks are retained with the loader diagnostics.
The test-only Rust JSON parser is pinned to MIT/Apache-2.0 `serde_json` 1.0.145;
production crate dependencies are unchanged. Dart uses the existing `ffi` 2.2.0
test dependency. These controls can also run in the owner-authorized local
environment; record the actual compiler, platform, source and result.
