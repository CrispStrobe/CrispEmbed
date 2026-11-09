# Marker Rust and Dart lifecycle evidence — 9 November 2026

The first actual local run passed all **6 scenarios and 111 named assertions**.
All 13 supervised version, package, build, analysis and execution commands exited
successfully. The user explicitly authorized local validation for this task.

| Client | Artifacts off | Incremental off | Enabled lifecycle | Total |
|---|---:|---:|---:|---:|
| Rust | 8 | 6 | 41 | 55 |
| Dart | 8 | 6 | 42 | 56 |

The tests call the shipped wrappers and native library. Their owned three-page
PDFium ABI fixture supplies deterministic text, page labels and a later-page
failure. This validates native API/client behavior; it does not establish real
PDF parsing/rendering, model inference or production document accuracy.

The native source is `d4d508c460c6e172fd1d7d4afc6b9c78652ad607`.
Relative to that commit, only formatting in the two client fixture files changes
the executed client source. The exact fixture and native binary hashes are in
[summary.json](summary.json), with all selected source hashes in
[results.json](results.json). The fixture formatting is intended for the next
checkpoint; no runtime or wrapper repair was required by these checks.

The enabled scenarios cover bounded progress, partial artifact save/reopen,
cancellation, fresh-context resume, completed-step idempotence, physical versus
Unicode printed page labels, immutable source byte ranges, standalone processed
reopen, rejected corrupt or mismatched artifacts, and retention of completed
pages across a later-page failure. The disabled scenarios verify native refusal
without losing the existing synchronous view or creating a checkpoint.

Rust 1.95.0 and Dart 3.13.7 were initially unavailable. Their official pinned
archives were installed after checksum verification; [toolchains.json](toolchains.json)
records the release metadata and hashes. This setup prerequisite was resolved
before the first client attempt. There were no failed client test attempts.

[Cargo.lock](Cargo.lock) and [pubspec.lock](pubspec.lock) preserve the resolved
package versions and checksums. Test-only direct dependencies are
`serde_json = 1.0.145` and `ffi = 2.2.0`. Compact logs retain actual tool output;
private checkout and temporary-directory prefixes are replaced with placeholders.
No model, native executable, library or toolchain archive is included.
