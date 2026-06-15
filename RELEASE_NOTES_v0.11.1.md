# CrispEmbed v0.11.1

Incremental release on top of v0.11.0 — PDF/A archival output, OCR-quality
refinements, and a CI fix for the Windows CUDA build.

## ✨ Added
- **PDF/A-2b archival output** — `ocr_render_set_pdfa()` adds XMP conformance
  metadata (`pdfaid:part=2`/`conformance=B`) + sRGB OutputIntent to the
  searchable-PDF renderer. Bound for Rust via `ocr_render_pages(.., pdfa)`
  (powers CrispSorter `ocr --render pdf --pdfa`).
- **Text angle classification** — detect 0° vs 180° page orientation
  (classical, model-free).
- **Refined DBNet postprocessing** — contour + rotated-rect + polygon scoring
  for tighter text-region boxes.

## 🛠 Fixes
- **Release CI: `windows-x86_64-cuda` "No CUDA toolset found"** — copy the CUDA
  MSBuild integration (`.props`/`.targets`) into Visual Studio's
  `BuildCustomizations` after the Jimver network install, so CMake's VS
  generator can `enable_language(CUDA)`. (The Windows-CUDA bundle should now
  build green.)
- **Qwen2-VL `generate()` mRoPE** — pass the actual `grid_thw` instead of a
  dummy `[1,1,1]`.

## 📦 Consuming this release
- CrispSorter: bump `CRISPEMBED_REF` `v0.11.0` → `v0.11.1` to pick up the
  `ocr_render_set_pdfa` binding (enables `ocr --render pdf --pdfa`).

**Full changelog**: `v0.11.0...v0.11.1` (auto-generated below).
