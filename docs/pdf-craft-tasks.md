# PDF Craft adoption: implementation tasks

Updated 9 November 2026. The user authorized the complete roadmap after the
initial [source review](pdf-craft-adoption.md). Work stays on upstream feature
branch `feature/pdf-craft-document-quality-20261109`, reviewed in
[draft PR 192](https://github.com/CrispStrobe/CrispEmbed/pull/192). Save coherent
checkpoints upstream throughout implementation. No release or default promotion
is part of this task.

`[x]` means the stated work was performed, not that a separate pending runtime
validation passed. Implementation and validation have separate checkboxes.

On 9 November the owner explicitly authorized builds and tests in the current
environment instead of waiting for the GitHub queue. This task now uses bounded
local Linux CPU execution, preserving exact source/toolchain identities and
initial failures. Windows/macOS and hosted results remain distinct; a local
pass does not imply those platform checks passed. This authorization is scoped
to this feature work and does not change the repository-wide guidance.

## 0. Source and dependency boundary

- [x] Pin the reviewed pdf-craft source to `cf003c711b6a63cd0a0a64bfc490bd319ab20d92`.
- [x] Retain the selected helper's MIT notice in distributed `NOTICE.marker`.
- [x] Exclude Poppler, Qt, EasyDict and the MinerU-referenced paragraph merger.
- [x] Implement default-off Unicode repetition diagnostics and model-free cases.
- [x] Correct new TeleOCR GGUF license metadata and add admission regressions.
- [ ] Record executed hosted results for the final integrated source, preserving
  initial failures and superseded/cancelled runs separately.

## Gap closure continuation

The owner asked to close the remaining gaps on 9 November. Continue bounded
local work while reading already-completed platform jobs; an aggregate queued
workflow does not mean every matrix job is queued. Keep fixes and executed
evidence in separate, incrementally pushed checkpoints.

- [x] Freeze the [57-page OCR and 17-query retrieval protocol](../tests/regression/marker/pdf-craft-corpus-v1.json)
  before downloading model/image inputs or producing candidate outputs. It pins
  the existing licensed corpus, scorers, actual native engine, PDFium, Tesseract
  language data and permissive MiniLM model. Report structural validity,
  non-regression and absolute quality separately; retain every failed threshold.
- [x] Fetch and verify all 57 exact image byte pins and the eight required
  model/support assets without exclusions or input repinning; retain the
  [complete input receipt](../tests/results/marker/pdf-craft-corpus-2026-11-09/input-assets.json).
  Four accounting controls cover coordinate scaling, complete UTF-8 windows,
  truthful source ranges and dehyphenation before the full run.
- [ ] Execute the frozen protocol with fresh Tesseract OCR, real native Marker
  processing and real native MiniLM embeddings; do not imply fresh TeleOCR
  execution or use retained OCR output as fresh inference.
- [x] Preserve exact Windows/macOS/iOS job results and source identities in
  the [platform receipt](../tests/results/marker/pdf-craft-platform-2026-11-09/README.md):
  Windows document/artifact/lifecycle 185/82/471; macOS 185/87/480 plus C API,
  save/backend; iOS device-SDK compilation. These pass on the source-equivalent
  PR checkout. Keep the independently failed development-artifact notice
  readback and incomplete iOS archive separate from native execution success.
- [ ] Exercise an actual retained Flutter archive with all nine notice copies,
  full SDK validation and an installed Linux Flutter consumer.
- [ ] Verify CocoaPods path-install resources with actual macOS/iOS consumers.
- [ ] Close native dependency and license-notice omissions found by package
  consumers; verify development artifacts as well as release archive recipes.
- [x] Install a locally built Python wheel with the complete native dependency
  closure and load it outside the checkout without a build-tree library path:
  [40 package and 51 actual real-PDF checks pass](../tests/results/marker/pdf-craft-local-2026-11-09/installed-wheel/README.md).

The frozen OCR route deliberately uses an Apache-2.0 engine and model. Its
quality result is scoped to that provider and the named corpus, including the
existing partial CORD gold and derived layout limitations. Model-free contract
coverage and unchanged output alone cannot qualify OCR or retrieval quality.

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
- [x] Record executed native model-admission evidence: 106 Jina/Unlimited and
  57 TeleOCR loader controls pass on Linux CPU; 19 TeleOCR converter tests pass.

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
- [x] Verify exact round trips, malformed artifacts, resource limits and
  render equivalence through native and public-client tests.

## 3. Source-linked chunks and continuation

Owner: document structure lane. Artifact schema must preserve these records.

- [x] Retain immutable source text and explicit UTF-8 byte ranges for transformed
  or assembled text; never fabricate smaller source rectangles.
- [x] Represent continuation targets explicitly, including adjacent page body
  text separated by retained footnotes or marginalia.
- [x] Expose assembled retrieval chunks with each contributing page/block/range.
- [x] Preserve legacy rendering and chunk output with the feature disabled.
- [x] Verify cross-page joins, interruption/ambiguity, Unicode boundaries,
  immutable source evidence and gate-off equivalence in native tests.

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
- [x] Execute those controls on owned fixtures at the validated source: 480
  native and 33 actual Python lifecycle checks pass on Linux CPU.

## 5. Multilingual whitespace and source offsets

Owner: document structure lane; independently default-off.

- [x] Normalize only explicitly supported prose padding and CJK boundaries.
- [x] Preserve join controls, variation selectors, intentional whitespace,
  source bytes and offsets; exclude code, formulas and structured assets.
- [x] Verify non-Latin, emoji, combining characters, invalid UTF-8 and mixed
  inline boundaries without changing disabled-path output.

## 6. Footnotes, references and page labels

Owner: document structure lane; PDF provider support owned by lifecycle lane.

- [x] Keep physical page index separate from the printed PDF page label.
- [x] Read page labels through optional PDFium APIs with explicit fallback.
- [x] Assign stable note/reference identities and expose unresolved/ambiguous
  links without guessing a cross-page target.
- [x] Preserve references and labels through artifacts and source-linked chunks.
- [x] Verify repeated labels, duplicate notes, missing targets and physical-page
  provenance with owned model-free fixtures.

## 7. Integration and evidence

- [x] Update `PLAN.md`, Marker guides, environment-variable inventory and this
  checklist at each coherent checkpoint.
- [x] Wire new sources/tests into native builds, Rust source packaging and CI
  dependency selection; retain bounded diagnostics as workflow artifacts.
- [x] Complete available local C++ formatting, Python/YAML/shell syntax,
  environment inventory and whitespace checks. The subsequent owner instruction
  authorizes local builds/suites; platform results remain separately identified.
- [x] Independently review artifact limits, source-range truthfulness, native
  model admission and all public bindings before marking implementation done.
- [x] Add actual Rust/Dart lifecycle probes and a bounded hosted supervisor,
  with six gate scenarios, named checks and retained dependency/native identities.
- [x] Select native client execution when Rust Marker wrappers or ABI build
  inputs change; preserve crate-only selection for isolated package changes.
- [x] Wire Windows document/artifact/lifecycle execution and macOS Xcode
  generation/CocoaPods syntax checks into the existing Build workflow.
- [x] Read successful results for all six actual Rust/Dart scenarios: 111
  named assertions, all process exits successful.
- [x] Exercise current Python/Dart wrappers against hash-verified v0.17.15:
  68 checks cover legacy conversion, absent new symbols and recovery.
- [x] Execute the strict real-PDF lifecycle regression: 48 checks, including
  actual PDFium identity, source ranges, page labels and exact processed reopen.
- [x] Verify cross-build processed-artifact retention and raw identity refusal:
  14 checks preserve earlier outputs, metadata and artifact bytes.
- [x] Read the added Windows/macOS results and preserve exact source/toolchain
  identities in the [platform evidence](../tests/results/marker/pdf-craft-platform-2026-11-09/README.md).
  Both native suites pass; local Linux execution was not used as a substitute.
- [x] Verify actual Python wheel/sdist/rebuilt-wheel bytes and missing-notice
  rejection: 26 checks pass with the new native library.
- [x] Verify Dart's actual publish dry-run selects all nine exact notice files:
  15 scoped checks pass with `--skip-validation`, without uploading.
- [ ] Read back a retained Flutter package/archive containing all nine notice files
  (package root and both platform-local directories); app-install qualification
  remains separate from byte-copy and Ruby syntax checks.
- [x] Record exact local source/toolchain results, initial failures and remaining
  limits in the [execution evidence](../tests/results/marker/pdf-craft-local-2026-11-09/README.md).
- [ ] Record exact final-source hosted results. No OCR accuracy,
  representative retrieval quality or release claim follows from model-free
  contract tests.

## Implementation checkpoints

The artifact, source-link, lifecycle and public-binding implementations and owned
controls are now present. See [artifacts](marker-artifact.md),
[lifecycle/API examples](marker-lifecycle.md) and [source links](marker-source-links.md).
Attribution gaps in Rust, native archive and Flutter packaging are corrected;
exact notice copies were checked. Shipping remains a separate operation. Checked
execution boxes above refer to the observed Linux CPU results; platform and
model-quality gaps retain their own open boxes.

## Review corrections

- [x] Reproduce missing stable provenance in incremental views with the real
  PDF parser and a native regression, preserving the failing attempts.
- [x] Use one provenance helper for legacy completion and initial raw
  checkpoints, so partial/final/raw-reopened views inherit extraction-time
  metadata without overwriting stored processed-artifact provenance.
- [x] Add a supervised owned real-PDF lifecycle regression to the Linux native
  lane using the existing pinned PDFium test dependency.
- [x] Record rebuilt-source provenance and strict real-PDF lifecycle results:
  480 native lifecycle checks and 48 real-PDF checks pass at `891ea3d`.
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
No production dependency or model was added. The owner subsequently authorized
local execution while hosted runs remain queued. Keep each validation box open
until its actual result is read; preserve hosted and platform gaps separately.

## Executed local evidence

Initial native source `d4d508c460c6e172fd1d7d4afc6b9c78652ad607`, incorporating main
through `dc4f51c755bf78139f1f6ea87941748b05e75971`, builds successfully with GCC
14.2.0, CMake 3.31.10, portable CPU flags and pinned GGML. All 185 document,
87 artifact, 400 lifecycle and 38 save checks pass. Actual Python/Rust/Dart,
model admission and package results are linked above. Real PDFium 151.0.7891.0
passes 39 backend and 86 C API checks, plus 112 Python/CLI/HTTP checks on eight
owned PDFs. Five repository guards also pass.

The first native artifact/lifecycle attempts failed before any checks because
the default temporary directory was absent. Both passed unchanged after setting
a task-local `TMPDIR`. Those attempts and exact binary/source hashes remain in
the evidence. Rust/Dart fixture changes in this checkpoint are formatter output;
their executed source hashes are recorded separately from the native commit.

The final native source is `891ea3d3632c6a3c3155e25b34705da6a02ec807`.
Its [receipt](../tests/results/marker/pdf-craft-local-2026-11-09/final-receipt.json)
records the successful replay after the provenance correction: native lifecycle
coverage grows from 400 to 480 checks, and 48 real-PDF plus 14 cross-build checks
pass. Both new regressions fail against the original omission; the failed
native log and strict real-PDF attempt remain preserved. The final real-PDF
comparison excludes only five run/lifecycle metadata keys, and processed reopen
compares all four complete output byte strings without exclusions.

## Historical hosted snapshots

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
commits on the same branch. At that historical review no native or client runtime
pass had been observed. The local evidence above supersedes that execution gap;
the PR check list remains the live hosted gate.
