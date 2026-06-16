# CrispEmbed v0.11.4

Rust-binding completeness + CI repairs for the document-AI engines.

## ✨ Added (Rust bindings)
- **SAFMN + Real-ESRGAN** SR — `crispembed-sys` FFI was missing (the safe
  `CrispSafmnSr` / `CrispEsrganSr` wrappers wouldn't compile); now bound.
- **Table structure recognition** — `CrispTableParse` (new / `to_html` /
  `detect_grid`) over `crispembed_table_parse_*`.
- **LiLT** layout-aware token classification — `CrispLiLT` (new / `classify`)
  over `crispembed_lilt_*`.

These complete the Rust surface CrispSorter consumes: scan-restoration
(Restormer deblur, configurable SR engine, dewarp), `cc_detect`, table
extraction, and PDF-DPI / super-resolution / KIE.

## 🛠 CI fixes
- **windows-x86_64** + **-cuda**: `M_PI` guard for `text_sr`/`pan_sr`/
  `restormer` (MSVC); windows-cuda now builds via `ilammy/msvc-dev-cmd` + Ninja
  (the hardcoded vcvars path failed on the windows-2025 runner).
- **Build WASM**: untracked accidentally-committed `build-embed-wasm/` +
  `build-wasm/` trees (stale CMakeCache pointed at a VPS path → emcmake abort).
- **Python wheels**: repaired the `_binding.py` corruption from the
  feat/restormer merge (duplicate TBSRN / tangled SAFMN); exported
  `CrispSafmnSr` + `CrispEsrganSr`.

## 📦 Consuming
- CrispSorter: bump `CRISPEMBED_REF` → `v0.11.4` for the SAFMN/ESRGAN/table/LiLT
  bindings used by the scan-restoration + table features.
