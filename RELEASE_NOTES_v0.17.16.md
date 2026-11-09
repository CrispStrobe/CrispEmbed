# CrispEmbed v0.17.16

Patch release, 9 November 2026. Changes since [v0.17.15](https://github.com/CrispStrobe/CrispEmbed/releases/tag/v0.17.15),
through `c99056a2` on 9 November 2026.

The main addition is an additive OCR review and export API used by CrispSorter's
OCR document workspace. This release also fixes PP-OCRv6 detection on empty and
changing-size video frames, and registers several experimental quantized OCR
aliases with their evaluation evidence. No default model or default OCR route
changes for existing callers, except the MinerU2.5-Pro full-document route noted
below.

## OCR review snapshots and checked document exports

[PR #179](https://github.com/CrispStrobe/CrispEmbed/pull/179) adds, without
changing existing entry points:

- **Versioned review snapshots** (`crispembed_ocr_pipeline_review_json`; Rust
  `CrispOcrPipeline::review_json`): per-region geometry, text, and confidence
  provenance that distinguishes detector scores from recognition evidence, with
  malformed text bytes preserved rather than replaced. See
  [docs/ocr-review-snapshot.md](https://github.com/CrispStrobe/CrispEmbed/blob/v0.17.16/docs/ocr-review-snapshot.md).
- **Checked OCR rendering** (`crispembed_ocr_render_capabilities`, Rust
  `ocr_render_pages_checked[_with_sizes]`): searchable PDF output with lossless
  raster embedding, Unicode text, document-sized page tables and per-page
  physical size; hOCR/ALTO interoperability checks; unsupported bidi text fails
  safely instead of producing a wrong text layer. Unknown export formats now
  return an error from the checked API instead of silently rendering text. See
  [docs/ocr-export.md](https://github.com/CrispStrobe/CrispEmbed/blob/v0.17.16/docs/ocr-export.md).
- Optional OCR language and text-stage availability is reported to callers.

## Fixes

- **PP-OCRv6** ([#165](https://github.com/CrispStrobe/CrispEmbed/pull/165), fixes #164):
  two default-off detector options accept a valid empty probability map and
  rebuild the retained graph when the input size changes, instead of rerunning
  scalar inference or reusing a graph built for the first frame size. The
  scalar fallback and comparison mode are preserved.
- Environment-gate "off" words are synchronized with CrispASR ([#163](https://github.com/CrispStrobe/CrispEmbed/pull/163)).

## Behaviour change: MinerU2.5-Pro full-document route

- `CRISPEMBED_MINERU25_PRO_FULL_DOC` is now **on by default for F16/F32**
  checkpoints and stays opt-in for quantized GGUFs, detected from tensor types at
  initialization (new C ABI `is_quantized` / `full_doc_enabled`). The default-gate
  rerun matched 8/8 cases ([#174](https://github.com/CrispStrobe/CrispEmbed/pull/174)).

## Experimental aliases (opt-in, SHA-pinned)

These are registered as **experimental**; defaults are unchanged.

- GOT-OCR2: `got-ocr2-q8_0-experimental` and `got-ocr2-q8-q4k-mix-experimental`
  ([#178](https://github.com/CrispStrobe/CrispEmbed/pull/178)); `got-ocr2-q4_k-experimental`
  is superseded but kept. F16-r2 remains the default. GOT-OCR2 is
  Tongyi Qianwen Research (non-commercial) and stays licence-gated.
- Unlimited-OCR-2: `unlimited-ocr2-q8_0-experimental` (q8sh16), qualified at
  ≥ 0.97 on CPU and CUDA ([#180](https://github.com/CrispStrobe/CrispEmbed/pull/180),
  [#181](https://github.com/CrispStrobe/CrispEmbed/pull/181)). A Q4_K_MIX re-score on the
  current runtime came out about 0.03 lower ([#184](https://github.com/CrispStrobe/CrispEmbed/pull/184));
  the bisect in [#190](https://github.com/CrispStrobe/CrispEmbed/pull/190) attributes this to
  the F32-vision default changing the F16 reference, not to the quantization:
  the published Q4_K_MIX bytes and decoded text are unchanged.
- OXR: default-off `CRISPEMBED_OXR_DECODE_F32_ACT` A/B gate; the German
  divergence is attributed to activation quantization and shared Q8_0 weights
  ([#183](https://github.com/CrispStrobe/CrispEmbed/pull/183)).
- Experimental aliases are exercised through real CLI and Python consumers with
  fresh-cache hash checks ([#173](https://github.com/CrispStrobe/CrispEmbed/pull/173),
  [#176](https://github.com/CrispStrobe/CrispEmbed/pull/176)).

## Evaluation records

- OCR gold corpus v1 (57 licensed pages with layout, order and table
  annotations) and a layout scorer ([#174](https://github.com/CrispStrobe/CrispEmbed/pull/174)).
- GOT-OCR2 gold attribution closed: pixels are exact on PIL input and remaining
  divergences are near-tie argmax flips ([#180](https://github.com/CrispStrobe/CrispEmbed/pull/180), [#182](https://github.com/CrispStrobe/CrispEmbed/pull/182)).
- LightOn4B, MiniCPM and publication-evidence records
  ([#168](https://github.com/CrispStrobe/CrispEmbed/pull/168)–[#172](https://github.com/CrispStrobe/CrispEmbed/pull/172), [#175](https://github.com/CrispStrobe/CrispEmbed/pull/175)).
  These are records, not new qualifications or default changes.

## Validation

- Full Build, Rust, WASM and inpainting checks pass on the release source, and a
  dry-run release build of the packaged artifacts precedes the tag.
- Native/package source versions are 0.17.16. pub.dev publication remains
  separately unqualified and is not part of this release.

[Full comparison since v0.17.15](https://github.com/CrispStrobe/CrispEmbed/compare/v0.17.15...v0.17.16).
