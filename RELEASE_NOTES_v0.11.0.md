# CrispEmbed v0.11.0

Document-OCR release: result renderers, a classical (model-free, GPU-free)
preprocessing + detection tier, page dewarping, Arabic OCR, and the Rust
bindings that let CrispSorter ship structured / searchable output.

## ✨ Highlights

### OCR result renderers (text / hOCR / ALTO / searchable PDF)
- New `src/ocr_render.{h,cpp}` — accumulate OCR results page-by-page
  (`create → begin → add_page* → end`) and render to **plain text**, **hOCR**
  (XHTML w/ `ocr_page`/`ocr_line`/`ocrx_word` + boxes), **ALTO 3.1** XML, or a
  **searchable PDF** (image + invisible positioned text layer). 36/36 unit tests.
- Exposed through the **CLI + C API + all language bindings** (Rust, Dart, Python).
- **Rust**: one-shot `crispembed::ocr_render(&[OcrRegion], w, h, fmt)` plus the
  lower-level `crispembed::ocr_render_pages(&[OcrRenderPageInput], fmt) -> Vec<u8>`
  — **multi-page** (one document across pages) and **binary-safe** (PDF via
  `output_size`, no NUL truncation).

### Classical preprocessing + detection tier (no model, no GPU)
- **CC-based text-line detector** (`cc_detect`) — connected-components line
  detection as a light tier when DBNet/Surya aren't available.
- **Classical preprocessing** (`classical_preproc`) — adaptive Otsu / Sauvola
  binarization, deskew, despeckle, background normalization.
- **Fast 1-bit morphology** (`morph_fast`) — ~21× faster, ~32× less memory than
  the float separable morph.
- **Page dewarping** (`dewarp`) — cubic baseline fitting + disparity warp for
  curved/skewed scans.

### Arabic OCR — Qari-OCR (2B, Apache-2.0)
- Port of NAMAA-Space Qari-OCR (Qwen2-VL-2B, full tashkeel/diacritics). No new
  C++ — the existing `qwen2vl_ocr` engine reads dims from GGUF metadata.
  Registry: `qari-ocr` (F16 / Q8_0 / Q4_K). Converter handles Qwen2-VL vs
  Qwen2.5-VL config field-name differences.

### Post-OCR punctuation
- FireRedPunc / PCS punctuation-restore models **registered** in the model
  manager, so `--punct-model fireredpunc` (etc.) auto-resolves.

## 🛠 Fixes
- **`crispembed` crate now compiles**: the renderer/preproc bindings referenced
  an undefined `OcrRegion` and used `libc::free` without depending on `libc` —
  added `pub type OcrRegion = OcrResult` + the `libc` dependency. (Unblocks all
  Rust consumers.)
- Qwen2-VL→GGUF converter: config field-name handling + a broken print stmt.
- Parity-test robustness (LLM layer-path discovery, deps).

## 📦 Consuming this release
- CrispSorter: bump `CRISPEMBED_REF` `v0.10.1` → `v0.11.0` in `release.yml` /
  `ci.yml` to pick up the renderer + binding lib bundles.
- New surface used downstream: `crispembed::ocr_render_pages` (CrispSorter's
  `ocr --render hocr|alto|pdf`).

**Full changelog**: `v0.10.1...v0.11.0` (auto-generated below).
