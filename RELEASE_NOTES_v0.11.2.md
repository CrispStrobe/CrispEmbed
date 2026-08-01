# CrispEmbed v0.11.2

A big document-AI release: image **restoration + super-resolution** (incl.
deblur), **dewarping**, **KIE**, **table structure**, **GPU enablement across all
engines**, and the orchestrator/binding hooks that CrispSorter's new PDF-DPI,
super-resolution, and KIE features consume.

## ✨ Image restoration & super-resolution
- **Real-ESRGAN** (620K, SRVGGNetCompact) — real-world SR for **blur** / noise /
  compression. **Restormer** (26M, MDTA+GDFN U-Net) — multi-task
  **denoise + SR + deblur** in one model. **SAFMN** (228K, lightweight SR).
- Text/image SR engines: **NAFNet-SR**, **TBSRN** (text-line), **PAN** (4×
  whole-image, 272K). Orchestrator **auto-detects PAN vs NAFNet-SR**; SR is a
  pipeline pre-processor (`sr_model`) for low-DPI upscaling before OCR.

## ✨ Document geometry & structure
- **TPS dewarping** — model-free thin-plate-spline page straightening + a CNN
  control-point localization network.
- **Table structure recognition** — rule-based HTML extraction.
- **PDF DPI profiling** — estimate a page's mean raster DPI to drive auto-OCR
  resolution.

## ✨ KIE (key-information extraction)
- **LiLT** layout-aware KIE (dual-stream encoder + BiACM), **BERT** fixed-label
  NER (token classification, auto-detect dispatch), and the GLiNER-based
  `crispembed_kie_*` pipeline — extract structured fields from forms/invoices.

## ✨ Performance
- **GPU enablement across all 25 engines** (graph-based via
  `ggml_backend_init_best`; GPU-safe scalar weight reads).

## 🛠 Fixes
- **Qwen2-VL vision FFN** uses exact `gelu_erf` (not the tanh approx) — accuracy.
- **Release CI: windows-x86_64-cuda** now builds via the **Ninja + vcvars**
  recipe (mirrors CrispASR) instead of the VS generator's broken CUDA-toolset
  detection — the windows-cuda bundle should build green.
- LiLT / TBSRN / SAFMN / Real-ESRGAN per-layer parity verified.

## 📦 Consuming this release
- CrispSorter: bump `CRISPEMBED_REF` `v0.11.1` → `v0.11.2` for the orchestrator
  `sr_model` hook, `CrispPanSr` (low-res pre-OCR super-resolution), and
  `pdf_page_dpi` (auto PDF render DPI).

**Full changelog**: `v0.11.1...v0.11.2` (auto-generated below).
