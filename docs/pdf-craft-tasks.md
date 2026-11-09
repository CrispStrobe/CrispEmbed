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

- [ ] Identify Jina and other declared noncommercial Unlimited profiles from
  GGUF model identity/source/license metadata, not filenames.
- [ ] Require explicit value-parsed acknowledgement before backend or weight
  allocation, including direct local paths and cached files.
- [ ] Keep ordinary permissive Unlimited profiles unaffected; do not treat
  experimental feature enablement as license acknowledgement.
- [ ] Add metadata-only public-loader controls for missing/false acknowledgement,
  identity variants, accepted admission and permissive controls.
- [ ] Document exact terms, gate, limits and executed hosted evidence.

## 2. Persist and reopen native extraction

Owner: artifact lane. Foundation for resume and reusable indexing.

- [ ] Define a versioned, bounded artifact that round-trips the full native IR,
  raw bytes, source links, geometry, warnings and embedded assets.
- [ ] Hash payloads and retain source/model/preprocessing identities; reject
  unsupported versions, corruption, excessive nesting/counts and unsafe names.
- [ ] Save atomically, keeping an existing artifact intact after failure.
- [ ] Add opt-in public save/open operations that reopen extracted content
  without loading an OCR model or re-running extraction.
- [ ] Add Python, Rust and Dart accessors with existing ABI compatibility.
- [ ] Verify exact round trips, malformed artifacts, resource limits and
  render equivalence through hosted native and public-client tests.

## 3. Source-linked chunks and continuation

Owner: document structure lane. Artifact schema must preserve these records.

- [ ] Retain immutable source text and explicit UTF-8 byte ranges for transformed
  or assembled text; never fabricate smaller source rectangles.
- [ ] Represent continuation targets explicitly, including adjacent page body
  text separated by retained footnotes or marginalia.
- [ ] Expose assembled retrieval chunks with each contributing page/block/range.
- [ ] Preserve legacy rendering and chunk output with the feature disabled.
- [ ] Verify cross-page joins, interruption/ambiguity, Unicode boundaries,
  immutable source evidence and gate-off equivalence in hosted tests.

## 4. Bounded processing, cancellation and resume

Owner: public lifecycle lane. Depends on artifact contract.

- [ ] Add an explicit incremental API with a caller-supplied page budget,
  progress/status and cancellation checked at safe page boundaries.
- [ ] Persist completed raw extraction units before final document processing;
  avoid retaining page raster buffers across iterations.
- [ ] Resume only when input/model/preprocessing identities and selection agree.
- [ ] Expose partial results honestly and retain the last durable checkpoint
  after cancellation or a later page failure.
- [ ] Keep the existing all-or-nothing conversion API and ABI intact.
- [ ] Bind the lifecycle in C/Python/Rust/Dart and test interruption, mismatched
  identities, completed resume and failure recovery on hosted owned fixtures.

## 5. Multilingual whitespace and source offsets

Owner: document structure lane; independently default-off.

- [ ] Normalize only explicitly supported prose padding and CJK boundaries.
- [ ] Preserve join controls, variation selectors, intentional whitespace,
  source bytes and offsets; exclude code, formulas and structured assets.
- [ ] Verify non-Latin, emoji, combining characters, invalid UTF-8 and mixed
  inline boundaries without changing disabled-path output.

## 6. Footnotes, references and page labels

Owner: document structure lane; PDF provider support owned by lifecycle lane.

- [ ] Keep physical page index separate from the printed PDF page label.
- [ ] Read page labels through optional PDFium APIs with explicit fallback.
- [ ] Assign stable note/reference identities and expose unresolved/ambiguous
  links without guessing a cross-page target.
- [ ] Preserve references and labels through artifacts and source-linked chunks.
- [ ] Verify repeated labels, duplicate notes, missing targets and physical-page
  provenance with owned model-free fixtures.

## 7. Integration and evidence

- [ ] Update `PLAN.md`, Marker guides, environment-variable inventory and this
  checklist at each coherent checkpoint.
- [ ] Wire new sources/tests into native builds, Rust source packaging and CI
  dependency selection; retain bounded diagnostics as workflow artifacts.
- [ ] Complete formatting, syntax and whitespace checks locally; builds and
  suites run on hosted CI under repository policy.
- [ ] Independently review artifact limits, source-range truthfulness, native
  model admission and all public bindings before marking implementation done.
- [ ] Record exact final-source runs and remaining limits. No OCR accuracy,
  representative retrieval quality or release claim follows from model-free
  contract tests.

## Current evidence

Initial implementation checkpoint: `93aa882ee371c2dbdcaa8060ffe9ac7e74995e07`.
At continuation, its [Build](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42594063974)
was pending and [controls](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42594064524)
queued. Re-query exact runs before reporting their outcome. Formatting,
environment inventory, Python syntax and whitespace checks had passed locally;
no runtime pass is inferred from that evidence.
