# PDF Craft adoption: implementation tasks

Updated 9 November 2026. The user authorized the complete roadmap after the
initial [source review](pdf-craft-adoption.md). Work stays on upstream feature
branch `feature/pdf-craft-document-quality-20261109`, reviewed in
[draft PR 192](https://github.com/CrispStrobe/CrispEmbed/pull/192). Save coherent
checkpoints upstream throughout implementation. No release or default promotion
is part of this task.

`[x]` means the stated work was performed, not that a separate pending runtime
validation passed. Implementation and validation have separate checkboxes.

## 0. Source and dependency boundary

- [x] Pin the reviewed pdf-craft source to `cf003c711b6a63cd0a0a64bfc490bd319ab20d92`.
- [x] Retain the selected helper's MIT notice in distributed `NOTICE.marker`.
- [x] Exclude Poppler, Qt, EasyDict and the MinerU-referenced paragraph merger.
- [x] Implement default-off Unicode repetition diagnostics and model-free cases.
- [x] Correct new TeleOCR GGUF license metadata and add admission regressions.
- [ ] Record executed hosted results for the final integrated source, preserving
  initial failures and superseded/cancelled runs separately.

## 1. Native noncommercial model admission

Owner: model admission lane. Independent of document artifact work.

- [x] Identify Jina and other declared noncommercial Unlimited profiles from
  GGUF model identity/source/license metadata, not filenames.
- [x] Require explicit value-parsed acknowledgement before backend or weight
  allocation, including direct local paths and cached files.
- [x] Keep ordinary permissive Unlimited profiles unaffected; do not treat
  experimental feature enablement as license acknowledgement.
- [x] Add metadata-only public-loader controls for missing/false acknowledgement,
  identity variants, accepted admission and permissive controls.
- [x] Document exact terms, gate and limits in [Jina admission](jina-ocr-license-gate.md).
- [ ] Record executed native model-admission evidence.

## 2. Persist and reopen native extraction

Owner: artifact lane. Foundation for resume and reusable indexing.

- [x] Define a versioned, bounded artifact that round-trips the full native IR,
  raw bytes, source links, geometry, warnings and embedded assets.
- [x] Hash payloads and retain source/model/preprocessing identities; reject
  unsupported versions, corruption, excessive nesting/counts and unsafe names.
- [x] Save atomically, keeping an existing artifact intact after failure.
- [x] Add opt-in public save/open operations that reopen extracted content
  without loading an OCR model or re-running extraction.
- [x] Add Python, Rust and Dart accessors with existing ABI compatibility.
- [ ] Verify exact round trips, malformed artifacts, resource limits and
  render equivalence through hosted native and public-client tests.

## 3. Source-linked chunks and continuation

Owner: document structure lane. Artifact schema must preserve these records.

- [x] Retain immutable source text and explicit UTF-8 byte ranges for transformed
  or assembled text; never fabricate smaller source rectangles.
- [x] Represent continuation targets explicitly, including adjacent page body
  text separated by retained footnotes or marginalia.
- [x] Expose assembled retrieval chunks with each contributing page/block/range.
- [x] Preserve legacy rendering and chunk output with the feature disabled.
- [ ] Verify cross-page joins, interruption/ambiguity, Unicode boundaries,
  immutable source evidence and gate-off equivalence in hosted tests.

## 4. Bounded processing, cancellation and resume

Owner: public lifecycle lane. Depends on artifact contract.

- [x] Add an explicit incremental API with a caller-supplied page budget,
  progress/status and cancellation checked at safe page boundaries.
- [x] Persist completed raw extraction units before final document processing;
  avoid retaining page raster buffers across iterations.
- [x] Resume only when input/model/preprocessing identities and selection agree.
- [x] Expose partial results honestly and retain the last durable checkpoint
  after cancellation or a later page failure.
- [x] Keep the existing all-or-nothing conversion API and ABI intact.
- [x] Bind the lifecycle in C/Python/Rust/Dart and author interruption, identity,
  completed-resume and failure-recovery controls.
- [ ] Execute those controls on hosted owned fixtures at final source.

## 5. Multilingual whitespace and source offsets

Owner: document structure lane; independently default-off.

- [x] Normalize only explicitly supported prose padding and CJK boundaries.
- [x] Preserve join controls, variation selectors, intentional whitespace,
  source bytes and offsets; exclude code, formulas and structured assets.
- [ ] Verify non-Latin, emoji, combining characters, invalid UTF-8 and mixed
  inline boundaries without changing disabled-path output.

## 6. Footnotes, references and page labels

Owner: document structure lane; PDF provider support owned by lifecycle lane.

- [x] Keep physical page index separate from the printed PDF page label.
- [x] Read page labels through optional PDFium APIs with explicit fallback.
- [x] Assign stable note/reference identities and expose unresolved/ambiguous
  links without guessing a cross-page target.
- [x] Preserve references and labels through artifacts and source-linked chunks.
- [ ] Verify repeated labels, duplicate notes, missing targets and physical-page
  provenance with owned model-free fixtures.

## 7. Integration and evidence

- [x] Update `PLAN.md`, Marker guides, environment-variable inventory and this
  checklist at each coherent checkpoint.
- [x] Wire new sources/tests into native builds, Rust source packaging and CI
  dependency selection; retain bounded diagnostics as workflow artifacts.
- [x] Complete available local C++ formatting, Python/YAML/shell syntax,
  environment inventory and whitespace checks. Builds/suites and platform
  language validation remain hosted under repository policy.
- [x] Independently review artifact limits, source-range truthfulness, native
  model admission and all public bindings before marking implementation done.
- [x] Add actual Rust/Dart lifecycle probes and a bounded hosted supervisor,
  with six gate scenarios, named checks and retained dependency/native identities.
- [x] Select native client execution when Rust Marker wrappers or ABI build
  inputs change; preserve crate-only selection for isolated package changes.
- [x] Wire Windows document/artifact/lifecycle execution and macOS Xcode
  generation/CocoaPods syntax checks into the existing Build workflow.
- [ ] Read successful results for all six Rust/Dart scenarios and the added
  Windows/macOS checks; preparation and static review are not execution.
- [ ] Verify a produced Flutter package/archive includes all nine notice files
  (package root and both platform-local directories); app-install qualification
  remains separate from byte-copy and Ruby syntax checks.
- [ ] Record exact final-source runs and remaining limits. No OCR accuracy,
  representative retrieval quality or release claim follows from model-free
  contract tests.

## Implementation checkpoints

The artifact, source-link, lifecycle and public-binding implementations and owned
controls are now present. See [artifacts](marker-artifact.md),
[lifecycle/API examples](marker-lifecycle.md) and [source links](marker-source-links.md).
Attribution gaps in Rust, native archive and Flutter packaging are corrected;
exact notice copies were checked. Shipping remains a separate operation. All executed-test checkboxes
stay open until hosted results are read at the final source.

## Review corrections

- [x] Pin the actually loaded PDFium library digest; reject stale live handles
  after the selected library file changes. Add an owned replacement regression.
- [x] Include image-resize runtime settings, C/C++ compiler settings and selected
  build configuration in resume identity.
- [x] Keep raw extraction block limits separate from processed tree-node limits.
- [x] Preserve contiguous chunk mappings when dehyphenation follows removed
  trailing whitespace in a reopened artifact. Add an owned source-map regression.
- [x] Carry exact MIT/Apache notices through native archives, Cargo vendoring
  and Flutter application/package resources.
- [x] Move configuration-dependent Marker identity definitions from source
  properties to native targets for the Xcode generator.
- [x] Ship platform-local CocoaPods notice resources for `:path` installs,
  where `prepare_command` does not run; preserve archive notice replacement.

## Validation continuation

The 9 November continuation merged main's documentation-only update
`3646f8058103eb106f8b234ad0c9b1eb057fa9d2` into this feature branch as
[`a236e65b8eb8eed4e47dafd9dca9074ef575c2d4`](https://github.com/CrispStrobe/CrispEmbed/commit/a236e65b8eb8eed4e47dafd9dca9074ef575c2d4).
The existing build/controls jobs had no executed steps when rechecked. Source
review found the Xcode-definition and CocoaPods path-install issues above;
both are corrected, and the missing Rust/Dart execution coverage is authored.
The hosted supervisor records failures rather than treating unavailable
toolchains, crashes or absent probe results as skipped successes.

The Rust probe's pinned `serde_json` 1.0.145 dependency is test-only and
[MIT/Apache-2.0](https://docs.rs/crate/serde_json/1.0.145/source/Cargo.toml).
No production dependency or model was added. Keep all executed-validation
checkboxes open until final-source hosted results are available.

## Current evidence

Initial implementation checkpoint: `93aa882ee371c2dbdcaa8060ffe9ac7e74995e07`.
At continuation, its [Build](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42594063974)
was pending and [controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42594064524)
queued. Re-query exact runs before reporting their outcome. Formatting,
environment inventory, Python syntax and whitespace checks had passed locally;
no runtime pass is inferred from that evidence.

Full implementation checkpoint: [`0e543c81c3537eebda0ba756a94d42fb22c81490`](https://github.com/CrispStrobe/CrispEmbed/commit/0e543c81c3537eebda0ba756a94d42fb22c81490).
Its [Build](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42598879883)
was pending and [model-free controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42598880031)
queued at review. Follow-up source-map and notice corrections are saved in later
commits on the same branch. The PR check list is the live final-source gate;
no native or client runtime pass has yet been observed.
