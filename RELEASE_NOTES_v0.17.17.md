# CrispEmbed v0.17.17

Patch release, 10 November 2026. Changes since [v0.17.16](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.16),
through `e7856b49` on 10 November 2026.

This release adds the native foundations for CrispSorter's desktop and mobile
OCR document workspace: phone-photo preparation, bounded and archival reviewed
PDF export, per-call licence acceptance, and resumable, source-linked document
extraction. All new paths are additive or opt-in. Existing entry points, model
defaults and OCR routes are unchanged for existing callers.

## Phone-photo preparation and native foundations

[PR #189](https://github.com/CrispStrobe/CrispEmbed/pull/189):

- True projective crops with validated, labelled corners; measured skew angle
  and full-resolution signed rotation; byte-exact identity transforms. Explicit
  length-checked C and Rust photo APIs with owned Rust image results.
- Conservative blank-image handling and finite/range/size checks, with
  exceptions contained at the C boundary.
- Exact registry model metadata: actual SHA-256, companion files and licence
  fields; explicit cross-target native linking.
- Windows: UTF-8 file names work in the real image loader and the writer.

## Reviewed PDF export: bounded writers and PDF/A-2b

[PR #189](https://github.com/CrispStrobe/CrispEmbed/pull/189) and
[PR #196](https://github.com/CrispStrobe/CrispEmbed/pull/196), integrated in
[PR #201](https://github.com/CrispStrobe/CrispEmbed/pull/201):

- Reviewed PDFs can be written through a bounded writer callback
  (Rust `ocr_render_pdf_to_writer_checked` over any `Write`) with byte
  limits, cooperative cancellation, and preserved I/O, flush and partial-write
  errors. Panics in Rust writers are contained.
- PDF/A-2b output with an embedded font, sRGB output intent, synchronized
  XMP metadata and document identifiers. Source page pixels and physical page
  sizes are preserved; Unicode mappings are scalar-safe. Checked against a
  pinned veraPDF 1.30.2, including deliberately damaged controls that must fail.
- Per-renderer options via Rust `ocr_render_pages_checked_with_options`.
- **Bidirectional text is not advertised.** `crispembed_ocr_render_capabilities()`
  reports `47` (PDF, Unicode, raster images, archival, streaming), without the
  BIDI bit: right-to-left text in a PDF text layer is still refused safely. The
  opt-in layout code and its strict PDFium/Poppler extraction diagnostics ship
  for evaluation only and run in a separate, non-blocking CI step.

## Per-call licence acceptance

[PR #191](https://github.com/CrispStrobe/CrispEmbed/pull/191):

- `crispembed_resolve_model_accepting(arg, auto_download, accepted_spdx)`
  accepts exactly one licence tag for one call. `all`, `*`, empty and NULL are
  refused; nothing is stored and no environment variable is set.
- `crispembed_model_license_info(...)` reports the licence facts of the entry a
  name resolves to (needs acceptance / permissive / unknown).
- Bindings: Rust `CrispEmbed::resolve_model_accepting` /
  `model_license_info`, Python `resolve_model_accepting` / `model_license_info`,
  Dart `resolveModelAccepting` / `modelLicenseInfo`.
- `crispembed_resolve_model` and `CRISPEMBED_ACCEPT_LICENSE` are unchanged.

## Reusable, resumable document extraction

[PR #192](https://github.com/CrispStrobe/CrispEmbed/pull/192):

- Versioned extraction artifacts (full IR, assets, source maps, SHA-256
  integrity, atomic replacement) that reopen without OCR or PDFium.
- Opt-in source-linked chunks, cross-page continuation, footnote/reference
  identities and printed page labels distinct from physical page indices.
- C/Python/Rust/Dart start, step, progress, cancellation, checkpoint/resume,
  save and open interfaces. Resume verifies input, model, PDFium, build and
  processing identity; cancellation happens at page boundaries.
- Default-off Unicode repetition diagnostics that never rewrite content.
- Native admission for CC-BY-NC models now also covers direct files and cached
  aliases, before any weights are allocated. TeleOCR GGUF licence metadata
  corrected.

## Inpainting

[PR #197](https://github.com/CrispStrobe/CrispEmbed/pull/197): the DBNet text
mask used by `--mask-from-text` now scores components over undilated pixels, so
thin first subtitle lines are covered. OCR detection is unchanged;
`CRISPEMBED_INPAINT_TEXT_MASK_SCORE_UNDILATED=0` restores the previous scoring.

## Packaging and CI

- Every package ships the Marker, TeleOCR, Apache-2.0 and SheenBidi notices and
  `ocr_render.h`; Flutter CocoaPods path installs carry platform-local notices.
- iOS Flutter pod: the `-force_load` archive path stays quoted through
  CocoaPods, so plugin paths containing spaces link.
- Windows CUDA release job tolerates a failed ccache setup (#195).

No default model, default OCR route or pub.dev publication changes.
