# CrispEmbed — History

Completed milestones and work log. See PLAN.md for current roadmap.

---

## June 20, 2026 — Performance optimization sweep

Full line-by-line audit of all ~57K lines across 60+ runtimes, followed by
systematic implementation of the highest-impact items.

### Core infrastructure
- **SIMD `dot_product()`** in `cpu_ops.h` — AVX2+FMA (x86-64) + NEON (ARM),
  used by `linear_cpu`, `mha_1q_cpu`, and all callers. 710+ FMA instructions
  in libcrispembed.so. `-march=native` via `CRISPEMBED_NATIVE` cmake option.
- **`DequantCache`** — per-context init-time weight caching, eliminates
  thousands of redundant dequant+alloc per decode session. Deployed to 15+
  runtimes (smoldocling, granite, bttr, hmer, posformer, 7 SR runtimes, etc.).
- **`RoPEFreqTable`** — precomputed frequency table eliminates `powf` per
  element per step. Deployed to smoldocling, granite.
- **`otsu_threshold()`** — extracted from 4 duplicated implementations to
  shared `cpu_ops.h`.
- **`std::unordered_map`** for tensor lookup in `gguf_loader.h` (was `std::map`).

### Runtime migrations
- **bttr/hmer/posformer** — replaced ~900 lines of duplicated conv2d/relu/
  layernorm/linear with `core_cpu` shared versions (SIMD-accelerated).
- **tesseract_lstm + gliner_ner** — LSTM gate inner loops use `dot_product()`.
- **smoldocling/granite** — DequantCache, RoPEFreqTable, SIMD linear_cpu,
  SIMD LM head matmul. Removed unused local helpers.
- **scunet_denoise** — hoisted per-pixel heap allocations outside spatial loops
  (was 100K+ allocs per swin block).
- **math_ocr** — global dequant cache → per-context DequantCache.
- **pcs** — FC head weights cached at init (no per-call GPU→CPU transfer).
- **mel.cpp** — SIMD mel projection via `dot_product()` (~38M MACs accelerated).
- **Orchestrator** — pre-load image once, pass pixels to 9 VLM engines.

### SR/restoration tiling
Added Hann-window overlap tiling to all 6 runtimes that lacked it:
esrgan_sr, safmn_sr, nafnet_denoise, scunet_denoise, instructir, adair.
Configurable via env vars. Small images bypass tiling.

### Other
- Sliding-window min/max pool in scan_cleanup (O(1) amortized via monotonic
  deque, was O(K) per pixel — ~50x for K=51).
- pdf_info: mmap instead of fread for large PDFs.
- layout_detect: ~30 debug printfs gated behind LAYOUT_DEBUG env var.
- ppformulanet_l: removed 370 lines of dead scalar encoder code.
- restormer: removed dead rst_gdfn() stub, fixed double variance computation.
- BPE merge: priority queue O(N log N) in bpe.h + tokenizer_bpe.cpp.

### Pix2Struct full rewrite
- **ggml graph encoder**: 12-layer T5 encoder as single ggml graph with
  `ggml_flash_attn_ext` (scale=1.0 for T5), GeGLU FFN, `ggml_rms_norm`.
  Encoder time: ~930ms (was ~2-5s scalar).
- **KV-cached decoder**: incremental self-attn cache (O(T) not O(T²)),
  cross-attn K/V pre-computed once via ggml graph. Decoder step0 cos=1.0000.
- **Batched patch projection**: 128 sequential `linear_cpu` → single
  `ggml_mul_mat` in encoder graph.

### Decoder allocation hoisting (6 runtimes)
Pre-allocated `dec_scratch` struct on each context, reused across all steps:
- bttr_ocr (~30 allocs/step), posformer_ocr (~36), hmer_ocr (~15),
  math_ocr (scalar path), parseq_ocr (~18), pix2struct (72).

### Flash attention adoption
- **decoder_embed**: both single-text and batch paths → `ggml_flash_attn_ext`.
  F16 causal mask for non-bidirectional models.
- **bidirlm_vision**: F16 block-diagonal mask (halves mask memory).

### Batched linear for SR attention
- `linear_batch_cpu` primitive added to `core/cpu_ops.h`.
- dat_sr, swinir_sr, hat_sr, scunet_denoise, mixtex_ocr: per-token QKV/proj/FFN
  loops converted to single batched calls + SIMD dot_product in attention.

### Quantitative results
- 70+ optimization items completed (from 53 originally identified + extras).
- ~1500 lines of duplicated code removed across the codebase.
- SIMD active in 30+ runtimes via shared `dot_product()` / `linear_cpu`.
- All changes verified: 99/99 cpu_ops tests, 97/97 vlm_attention tests,
  live MiniLM-L6 embedding inference bit-identical to baseline.

---

## June 19, 2026 — core/cpu_ops.h refactoring (Phase 1)

Extracted ~100 lines of CPU-scalar helper functions duplicated across 6+ engine
files into a shared header-only `src/core/cpu_ops.h` (namespace `core_cpu`).

**Functions extracted:** to_f32 (GPU-safe dequant), layernorm_cpu (raw + tensor
overloads), layernorm2d_cpu, rmsnorm_cpu, linear_cpu (raw + tensor overloads),
conv2d_cpu (with groups), gelu (tanh approx), gelu_erf (exact), silu/silu_inplace,
softmax, hardswish_inplace, relu6_inplace, relu_inplace, mha_1q_cpu.

**Engines refactored:** surya_det, got_ocr, ppformulanet_l_ocr, ppformulanet_ocr,
deepseek_ocr2, mixtex_ocr. Net: −728 lines deleted, +74 lines added (using decls).

**Key design decisions:**
- No default `eps` parameter — every call site must be explicit to prevent silent
  behavior changes across engines that historically used 1e-5, 1e-6, or 1e-12.
- `gelu` vs `gelu_erf`: two variants because engines use different approximations
  (mixtex uses erf-exact matching `nn.GELU()`; ppformulanet_l uses tanh approx).
- `conv2d_cpu` has `groups=1` default so engines without grouped convolution
  don't need to change call sites.
- All `to_f32` upgraded to GPU-safe `ggml_backend_tensor_get` path (some engines
  previously used direct `t->data` access which fails on GPU backends).

**Testing:** 88 unit tests (test_core_cpu_ops.cpp), verified parity on surya-det
(q8_0, f16), glm-ocr-diff (cos=1.000000 all checkpoints), mixtex-diff (identical
to main branch output), ppformulanet (q4_0).

---

## June 19, 2026 — OCR Confidence, HF Uploads, LFM2.5

### Per-character/token confidence for all OCR engines

Added softmax-based confidence tracking to every OCR engine's greedy
decode loop. 15 engines now expose `<engine>_confidences()` +
`<engine>_mean_confidence()`. Wired through: C API
(`crispembed_math_ocr_confidences`), Rust FFI, Python
(`CrispMathOcr.confidences()`, `CrispOcrOrchestrator.region_rec_confidence()`),
and Server (JSON `"confidence"` + `"token_confidences"` fields).

Engines: parseq, tesseract_lstm, math_ocr, hmer, bttr, posformer, mixtex,
ppformulanet, ppformulanet_l, glm_ocr, got_ocr, qwen2vl_ocr, internvl2_ocr,
granite_vision, lightonocr. Test suite: 44/44 pass (26 unit + 18 live).

### dots.ocr — REMOVED from main (license issue)

dots.ocr (rednote-hilab) claims MIT on HuggingFace but has a supplemental
"dots.ocr LICENSE AGREEMENT" with PRC governing law (Hangzhou Arbitration),
unilateral license amendment (90-day forced migration), prohibited uses,
mandatory "Built with dots.mocr" attribution, and trademark restrictions.
Code moved to feat/dots-ocr branch only, with license warnings. HF repo
set to private.

### New model registry entries

- **FireRed-OCR** (Qwen3-VL 2B) — `cstr/firered-ocr-crispembed-GGUF`
- **H2OVL-Mississippi-0.8B** — smallest VLM OCR (OCRBench 751, 398MB Q4_K)
- **Nanonets-OCR2-1.5B** — Qwen2-VL pruned (16L), runs on qwen2vl_ocr
- **german-ocr-3.1** — Qwen2.5-VL fine-tune for German business docs
  (new `merge-llamacpp-qwen2vl-gguf.py` tool for split llama.cpp GGUFs)

### LFM2.5-Embedding + ColBERT (LiquidAI)

- LFM2.5-Embedding-350M: 1024d CLS hybrid embeddings, 11 languages
- LFM2.5-ColBERT-350M: per-token 128d multi-vector output
- Both: converter, parity test, registry, HF upload

### HuggingFace uploads

All OCR model repos now have F16 + Q8_0 + Q4_K GGUFs with READMEs:
granite-vision, lightonocr, dots-ocr, firered-ocr. DeepSeek-OCR-2
quantization running on Kaggle (6.4GB F16 too large for VPS).

### Bug fixes

- Layout Q8_0/F16 crash: `tensor_to_f32()` for all decoder weight reads
- MixTex decoder: parity VERIFIED (cos=1.0 — was reference GGUF inconsistency)
- LightOnOCR prompt: correct chat template token IDs for OCR output
- Qwen2-VL KV cache: cont V view fix for correct token-for-token decode

---

## June 16, 2026 — LightOnOCR-2-1B (OCR Arena #2)

End-to-end port of [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
(Apache-2.0, 1B params, OCR Arena #2 with ELO 1697).

- **Architecture**: Pixtral ViT (24L, 1024d, 2D RoPE, SiLU FFN) + spatial merge 2×2
  projection + Qwen3 decoder (28L, 1024d, GQA 16/8, QK norm, SwiGLU)
- **Converter**: `models/convert-lightonocr-to-gguf.py` — lazy safetensors loading
- **Engine**: `src/lightonocr.{h,cpp}` — vision encoder + projection + decoder
- **Key challenge**: Pixtral 2D RoPE (interleaved h/w frequencies, not mRoPE)
- **QK norm fix**: model produced EOS without chat template prompt framing;
  fixed by embedding prefix/suffix text tokens around image features
- **GGUF**: F16 (2.2GB), Q8_0 (1.0GB), Q4_K (622MB) — `cstr/lightonocr-GGUF`
- **Dispatch**: `--ocr` auto-detects from GGUF arch, `--ocr-engine lightonocr`
- **Orchestrator**: wired as single-shot VLM engine
- **Decode**: O(n²) full recompute per token (KV cache TODO)

---

## June 15-16, 2026 — KIE, LiLT, BERT NER, LID, Truecasing, Shared Libraries

### Key Information Extraction (KIE)

Two-phase pipeline for extracting structured fields from document images.

**Phase 1 — OCR + NER**: Chains OCR orchestrator (text detection + recognition)
with GLiNER zero-shot NER. Character offset tracking maps NER entities back to
source OCR regions with bounding boxes.
- Files: `src/kie_pipeline.{h,cpp}`, C API `crispembed_kie_*`
- CLI: `--kie FILE --kie-labels "total,date,vendor"`
- Server: `POST /kie/extract`
- Bindings: Python `CrispKIE`, Dart `CrispKIE`

**Phase 2 — LiLT Layout Transformer**: Dual-stream encoder (RoBERTa 768d +
layout transformer 192d) with BiACM (bidirectional attention complementation).
Token classification for form understanding (FUNSD: question/answer/header).
- Architecture: 130.7M params, 12 layers, 12 heads, MIT license
- Parity: 25/25 layers cos=1.000000 vs HuggingFace
- Files: `src/lilt_kie.{h,cpp}`, converter, ref dumper, diff test
- HF models: `cstr/lilt-funsd-GGUF`, `cstr/lilt-base-GGUF` (F32/Q8_0/Q4_K)

### BERT / XLM-R Fixed-Label NER

Fixed-label token classification NER using existing BERT/XLM-R encoders with
a Linear(hidden, num_labels) head. Auto-detected from GGUF (`ner.classifier.weight`).
Same `crispembed_ner_*` API — backend auto-dispatched (GLiNER vs BERT NER).

- `dslim/bert-base-NER`: 110M, CoNLL-03, 9 labels (PER/LOC/ORG/MISC), MIT
- `Davlan/xlm-roberta-base-ner-hrl`: 278M, 10 languages, 9 labels, MIT
- GELU fix: switched all BERT FFN to erf-exact (matching HF/PyTorch)
- Cased tokenizer fix: auto-detect `do_lower_case` from vocab content
- `crispembed_encode_tokens_raw()`: unnormalized hidden states for classification
- HF models: `cstr/bert-base-NER-GGUF`, `cstr/xlmr-ner-hrl-GGUF`

### Language Identification (LID)

Text-based LID integrated into OCR orchestrator for automatic Tesseract model
selection. ISO 639-1 → Tesseract 639-3 mapping (12 languages).

- Shared library: `CrispASR/crisp_lid/` (fastText + CLD3 + dispatch)
- Orchestrator: `config.lid_model`, runs LID after OCR, populates `result.detected_lang`
- Tesseract auto-select: `model_b = "auto"` → resolves `tesseract-{lang}-q8_0.gguf`
- Server: `POST /lid/detect`, `--lid MODEL` flag
- Bindings: Python `CrispTextLID`, Dart `CrispTextLID`
- C API: `crispembed_ocr_pipeline_detected_lang()`

### Truecasing

Post-OCR truecasing (German noun capitalization) via BiLSTM character-level model.

- Shared library: `CrispASR/crisp_truecase/` (stat + CRF + BiLSTM)
- Orchestrator: `config.truecase_model`, applied to `full_text` after OCR
- CLI: `--truecase-model MODEL`
- Bindings: Python `CrispTruecaser`, Dart `CrispTruecaser`

### Shared Libraries (cross-repo with CrispASR)

Extracted 3 new shared libraries to eliminate code drift between CrispASR and CrispEmbed:

| Library | Purpose | LOC |
|---------|---------|-----|
| `crisp_punc/` | Punctuation restoration (FireRedPunc + PCS) | 1666 |
| `crisp_lid/` | Text LID (fastText + CLD3 + dispatch) | 2098 |
| `crisp_truecase/` | Truecasing (stat + CRF + BiLSTM) | 1002 |

All follow the `crisp_audio/` pattern: self-contained CMakeLists, auto-detect
core target (`crispasr-core` or `crispembed-core`), conditional fallback to
local copies when sibling repo is absent.

### Table Structure Recognition

Rule-based table parser: morphological line detection → grid intersection →
per-cell OCR → HTML `<table>` output. No model needed.
- Files: `src/table_parse.{h,cpp}`, C API, CLI `--table`, server `POST /table/parse`
- Test: 14/14 pass (ruled + borderless grids)

### Orchestrator Tests

Comprehensive test suite: 56/56 PASS across 10 sections (classifier, accept-gate,
multi-stage escalation, chain selection, C API, edge cases, punctuation).

### Handover Prompts

All 18 handover prompts completed.

---

## June 2026 — Text Super-Resolution (PAN, TBSRN, NAFNet-SR)

Three engines for upscaling low-resolution text images before OCR, integrated
into the document preprocessing pipeline.

### PAN 4× whole-image super-resolution

Pixel Attention Network (PAN) for 4× upscaling of full document pages.

- **Architecture**: shallow feature extraction (Conv3×3) → 6 SC-PA blocks
  (depthwise-separable conv + pixel attention gates) → PixelShuffle(4) upsampler.
  272K parameters, C++ forward pass.
- **GGUF**: `pan-x4-f16.gguf` — 0.5 MB F16.
- **Converter**: `models/convert-pan-to-gguf.py`.
- **Parity**: cos=0.999654 vs PyTorch reference (F16, full-page input).
- **License**: Apache-2.0.

### TBSRN 2× per-line super-resolution

Text Before Super-Resolution Network (TBSRN) for 2× upscaling of individual
OCR text-line crops (telescope training scheme).

- **Architecture**: shallow feature extraction → 3 residual groups (6 TSA blocks
  each, transformer-style self-attention on spatial tokens) → PixelShuffle(2)
  upsampler. 1.1M parameters, C++ forward pass.
- **GGUF**: `tbsrn-telescope-f16.gguf` — 2 MB F16.
- **Converter**: `models/convert-tbsrn-to-gguf.py`.
- **Parity**: cos=0.999985 vs PyTorch reference (F16, 32×128 text-line crop).
- **License**: Apache-2.0.

### NAFNet-SR engine (no model yet)

Engine scaffolding for NAFNet-SR custom super-resolution models. Reuses the
existing `nafnet_denoise.cpp` architecture with a configurable upsampling tail.
No pre-trained GGUF included — supply a custom trained checkpoint via `--sr-model`.

### Integration matrix

| Surface | PAN (`--pan-sr`) | TBSRN (`--tbsrn-sr`) | NAFNet-SR (`--sr-model`) |
|---------|-----------------|----------------------|--------------------------|
| C API | `crispembed_pan_sr_*` | `crispembed_tbsrn_sr_*` | `crispembed_nafnet_sr_*` |
| CLI | `--pan-sr` | `--tbsrn-sr` | `--sr-model` |
| Server | `POST /pan/sr` | `POST /tbsrn/sr` | — |
| Python | `CrispPanSr` | `CrispTbsrnSr` | — |
| Rust | `CrispPanSr` | `CrispTbsrnSr` | — |

New files: `src/pan_sr.{h,cpp}`, `src/tbsrn_sr.{h,cpp}`,
`models/convert-pan-to-gguf.py`, `models/convert-tbsrn-to-gguf.py`,
`tools/dump_pan_reference.py`, `tools/dump_tbsrn_reference.py`,
`tests/test_pan_sr.cpp`, `tests/test_tbsrn_sr.cpp`.

### Auto-SR in orchestrator

The orchestrator's `--sr-model` now auto-detects PAN vs NAFNet-SR from
the GGUF architecture metadata. Tested on 75 DPI single-line text:
- 75 DPI raw → OCR: `C Melbe Wesld1` (garbage)
- 75 DPI + PAN 4x → OCR: `Hello Werdd 123` (1 char error, readable)
- 150 DPI raw → OCR: `Hello World 123` (perfect, no SR needed)

Finding: do NOT apply classical cleanup (binarize/deskew) to low-DPI
images — it destroys sub-10px text. PAN alone is sufficient.

---

## June 2026 — Tesseract LSTM OCR + classical preprocessing + renderers

### Tesseract LSTM line-recognition engine

Ported Tesseract's LSTM line-recognition engine to CrispEmbed via GGML.
126 languages from `.traineddata` files (435 KB–1.7 MB Q8_0 per language).

- Converter (`convert-tesseract-to-gguf.py`): recursive binary tree parser,
  int8 dequant, gate reorder, GGUF emit. Supports tessdata_best + tessdata_fast.
- C++ engine (`tesseract_lstm.{h,cpp}`): Conv stacking → FC+tanh → MaxPool →
  SummLSTM → LSTMs → Softmax → CTC decode. Pure CPU, no ggml graph.
- Python reference (`dump_tesseract_reference.py`): pure-numpy forward pass.
- Parity: 8/8 stages cos_min=1.000000. Spaces + punctuation emitted natively.
- 12 language GGUFs on HuggingFace (`cstr/tesseract-lstm-GGUF`).

### Classical preprocessing (from Leptonica, BSD-2)

CPU-only, model-free, fast tier. Self-contained C++, no Leptonica dependency.

- 1-bit DWA morphology (`morph_fast`): 21x speedup, 32x less memory.
- CC text line detection (`cc_detect`): model-free, 4.3ms/page, zero downloads.
- Adaptive Otsu (`classical_preproc`): per-tile + bilinear interpolation.
- Differential-square-sum deskew: 3ms/page, binary search on 4x-reduced image.
- CC despeckle: flood-fill + size filter.
- Background normalization: tile-based 90th-percentile + smoothing.
- Page dewarping (`dewarp`): cubic baseline fitting + disparity warp. 10ms.

### OCR result renderers (`ocr_render`)

Plain text (configurable separator), hOCR (XHTML), ALTO 3.1 (XML),
searchable PDF (invisible text layer, rendering mode 3). 36/36 tests.
Wired into CLI (`--output-format`), C API, Rust, Python.

### Punctuation restoration

FireRedPunc + PCS copied from CrispASR. Auto-detect from GGUF arch.
CLI `--punct-model`, C API, orchestrator integration. Registered in model_mgr.

### OCR pipeline orchestrator

Wired into HTTP server, Python, Dart, Rust. Full params in all layers.
CORS headers. VLM escalation in Rust. Verbose logging (`CRISPEMBED_VERBOSE_OCR`).
GOT-OCR2 GPU scheduler fix. CC detect as model-free detector option.

### Wiring

All new capabilities: C API + Rust FFI + safe Rust + Python bindings.
docs/contributing.md updated with utility library checklist + integration matrix.
### Additional improvements (June 15)

- **Searchable PDF with image**: JPEG XObject embedding + glyph-width-aware
  text positioning (Tm matrix, font scaled to match bbox width).
- **PDF/A-2b metadata**: XMP metadata stream + sRGB OutputIntent.
- **Refined DBNet postprocessing**: Moore contour tracing + convex hull +
  min-area rotated rectangle (rotating calipers) + polygon-interior scoring.
- **Text angle classification**: heuristic 0°/180° detection via
  ascender/descender asymmetry.
- **Image downsampling calculator**: DPI + max_pixels aware.
- **OCR quality scoring**: dictionary-based word matching.

63 new tests total, all passing.

---

## June 2026 — Qari-OCR (Arabic with diacritics, 2B, Apache-2.0)

Port of NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct — Arabic OCR with
full tashkeel (diacritics) support. Fine-tuned from Qwen2-VL-2B-Instruct
via LoRA (r=16, α=16, 324 adapter pairs) on 50K Arabic OCR samples.

**Architecture**: Same Qwen2-VL family as existing `qwen2vl_ocr.cpp`:
- Vision: 32L ViT (embed_dim=1280, hidden_size=1536, 16 heads)
- Spatial merger: 2×2, mlp 5120→1536
- LLM: 28L Qwen2 (1536d, GQA 12Q/2KV, FFN=8960)
- Total: ~2B params

**No new C++ code** — the existing qwen2vl_ocr engine reads all dimensions
from GGUF metadata and handles both Qwen2-VL-2B and Qwen2.5-VL-3B.

**Converter fix**: Qwen2-VL config uses `embed_dim`/`mlp_ratio`/`in_chans`
instead of Qwen2.5-VL's `intermediate_size`/`in_channels`/`out_hidden_size`.
Added `getattr` fallbacks in `convert-qwen2vl-to-gguf.py`. Key insight:
vision `hidden_size` (1536) ≠ ViT block dim (`embed_dim`=1280) — must
write `embed_dim` as the GGUF vision.hidden_size for correct block computation.

**Conversion**: Kaggle kernel (16 GB RAM needed) merges 324 LoRA pairs
tensor-by-tensor into fp16 base, then converts to GGUF + quantizes.
Took 4 kernel iterations to get right (config field name mismatches).

**GGUFs**: `cstr/qari-ocr-crispembed-GGUF` — F16 (4.7 GB), Q8_0 (2.3 GB),
Q4_K (1.6 GB). Registry entry: `qari-ocr`.

**Parity**: Not yet verified per-layer (needs Kaggle). The qwen2vl engine
has cos=1.000 parity on Qwen2.5-VL-3B; the 2B variant uses the same code
path with different dimensions. Test kernel prepared but not yet run.

**Performance** (published): WER=0.221, CER=0.059, BLEU=0.597.

---

## June 2026 — Scan cleanup (document preprocessing pipeline)

Two-tier document scan preprocessing module — pure C++, no external
tool dependencies.

### Tier 1 — Classical (no model needed)

Four operations, ~500 LOC in `src/scan_cleanup.{h,cpp}`:

1. **Deskew**: Sobel edge detection → Hough line accumulator → median angle
   → bilinear affine rotation. Detects 3-degree skew exactly on synthetic tests.
2. **Binarization**: Otsu global (histogram between-class variance) and
   Sauvola adaptive (integral image for O(1) per-pixel local mean/stddev).
3. **Border crop**: row/column intensity projection → content rectangle detection.
4. **Background whitening**: morphological open (min-pool → max-pool) estimates
   background surface, then divide to normalize. Reduces background variance
   to near zero.

### Tier 2 — Learned denoising (NAFNet, MIT license)

Port of megvii-research/NAFNet (ECCV 2022) for image restoration.
Non-linear Activation Free Network — uses SimpleGate (channel split ×
element-wise multiply) instead of ReLU/GELU.

**Architecture**: U-Net with NAFBlocks.
- Intro: Conv3x3 (3→32)
- Encoder: [2,2,4,8] NAFBlocks at [32,64,128,256] channels
- Downsampling: Conv2x2 stride 2
- Middle: 12 NAFBlocks at 512 channels
- Decoder: [2,2,2,2] NAFBlocks with PixelShuffle(2) upsampling + skip connections
- Ending: Conv3x3 (32→3) + input residual
- 29.2M params, pre-trained on SIDD (smartphone denoising)

**NAFBlock**: LayerNorm2d → Conv1x1(c→2c) → DepthwiseConv3x3(2c) →
SimpleGate(2c→c) → SCA(AvgPool→Conv1x1) → Conv1x1(c→c) → residual×beta
→ LayerNorm2d → Conv1x1(c→2c) → SimpleGate → Conv1x1(c→c) → residual×gamma

**Implementation**: CPU-scalar forward pass in `src/nafnet_denoise.{h,cpp}`.
All standard ops: conv2d (1x1, 3x3, depthwise), LayerNorm2d, element-wise
multiply, global average pool, PixelShuffle.

**Parity** (64x64, all vs PyTorch reference):
- F32:  cos=0.9980, max_diff=48 px
- F16:  cos=0.9980, max_diff=48 px
- Q8_0: cos=0.9980, max_diff=47 px
- Q4_K: cos=0.9977, max_diff=48 px

Residual gap from 1.0 is uint8 quantization at input/output boundaries
(PyTorch processes float32 end-to-end; C++ goes u8→f32→model→f32→u8).

**Bug found**: `to_f32()` dequant function returned zeros for Q8_0/Q4_K
types instead of using `ggml_get_type_traits()->to_float`. Fixed.

**Quantizer fix**: added `.beta`/`.gamma` to the `is_add_operand` guard
in `tools/quantize.cpp` so NAFNet's per-channel residual scale factors
are never quantized (they're tiny [1,C,1,1] tensors used in element-wise
multiply — quantizing them corrupts the residual connections).

**GGUFs**: `cstr/nafnet-sidd-GGUF` — F16 (56 MB), Q8_0 (30 MB), Q4_K (16 MB).
Registry entry: `nafnet-denoise`.

### Wiring

All surfaces wired:
- **C API**: `crispembed_scan_cleanup_{init,process,free,defaults}` +
  `crispembed_scan_cleanup_process_simple` (for FFI without struct-by-value)
- **CLI**: `--cleanup` (preprocess before OCR), `--cleanup-only FILE` (standalone)
- **Server**: `POST /scan/cleanup` (always available, no model needed)
- **Python**: `CrispScanCleanup` class with `.process()` (file/PIL/numpy)
- **Rust**: `CrispScanCleanup` safe wrapper
- **Dart/Flutter**: `CrispScanCleanup` via `process_simple` FFI

**New files**: `src/scan_cleanup.{h,cpp}`, `src/nafnet_denoise.{h,cpp}`,
`models/convert-nafnet-to-gguf.py`, `tools/dump_nafnet_reference.py`,
`tests/test_scan_cleanup.cpp`.

---

## June 2026 — Surya detector GPU backend (Metal on M1)

`surya_det.cpp` hardcoded `ggml_backend_cpu_init()`, so even after the CUDA
build was fixed on Kaggle (GGML_CUDA_NO_VMM=ON) the detector still ran CPU-only.
Switched to `ggml_backend_init_best()` so the stage 0-2 and stage-3-block0
graphs run on the best available backend — Metal on Apple Silicon, CUDA
elsewhere — with `SURYA_DET_FORCE_CPU=1` to pin CPU for parity debugging and a
CPU fallback if no GPU backend initialises.

One gotcha: the scalar LiteMLA and decode-head paths dequantised weights via
`to_f32()`, which read `t->data` directly. That is fine for a CPU buffer but
`t->data` is not a valid host pointer on a GPU buffer, so the reads were routed
through `ggml_backend_tensor_get()` instead.

Verified on an M1 (Apple7, MTL0): F16 and Q8_0 both run on Metal, heatmap
parity vs CPU to ~3 decimals (sub-pixel bounding-box drift from F16 matmul
accumulation order). Stage 0-2 graph ~4.4 s GPU vs ~5.9 s CPU, stage-3 block0
~0.75 s vs ~0.94 s; the speedup is modest because LiteMLA + decode head stay
CPU-scalar. CUDA build separately confirmed on Kaggle P100 (Q8_0+F16 → 17
regions). Surya GPU is now marked done in PLAN.md.

---

## June 2026 — Surya detector Q8_0/Q4_K crash fix

The surya text detector (`surya_det.cpp`) crashed on quantized models (Q8_0, Q4_K)
with a segfault in `ggml_compute_forward_dup`. Root cause: two issues in `g_conv()`:

1. **Reshape before dequant**: `ggml_reshape_4d` on Q8_0 tensors created `ne[0]=3`
   (for 3×3 conv kernels), violating Q8_0's block alignment requirement (32 elements
   per block). The subsequent cast operation read invalid block data.

2. **Q→F16 cast unsupported**: ggml only implements quantized→F32 dequantization,
   not quantized→F16. The direct `ggml_cast(Q8_0, F16)` hit `GGML_ABORT`.

**Fix**: Dequant Q→F32 first, then reshape to 4D, then cast F32→F16 for `ggml_conv_2d`.
All four variants (F32, F16, Q8_0, Q4_K) now detect identically on synthetic test images.

Kaggle P100 testing confirmed F16 works (195s, 17 regions detected). CUDA cmake
still fails due to upstream ggml `CUDA::cuda_driver` target issue on Kaggle.

---

## June 2026 — GOT-OCR2 engine (0.7B, SAM ViT-B + Qwen2-0.5B, Apache-2.0)

Port of stepfun-ai/GOT-OCR2_0 — end-to-end document OCR handling plain text,
LaTeX math, tables, and formatted output. Fourth VLM in CrispEmbed.

**Architecture**: SAM ViT-B (12L, 768d, 12 heads, LayerNorm+GELU, windowed
attention ws=14 with global at [2,5,8,11], decomposed relative position encoding)
→ Neck (Conv 768→256, 1×1 → LN2d → Conv 256→256, 3×3 → LN2d) → Downsample
(Conv 256→512→1024, stride 2, 4096→256 tokens) → Linear(1024,1024) projector
→ Qwen2-0.5B (24L, 1024d, MHA 16/16, SiLU SwiGLU, standard RoPE θ=1M)
→ autoregressive generation with KV cache.

**Key differences from GLM-OCR**: Vision uses LayerNorm+GELU (not RMSNorm+SiLU),
no Q/K norm, SAM-style windowed+global attention with decomposed RPE (not CogViT).
LLM is standard pre-norm Qwen2 (2 norms/layer, not post-norm 4 norms/layer),
MHA (not GQA), standard RoPE (not mRoPE), tied word embeddings.

**Parity**: All checkpoints cos ≥ 0.999 (vision layers, neck, downsample,
projector, LLM layers).

**GGUFs**: `cstr/got-ocr2-crispembed-GGUF` — F16 (1.34 GB), Q8_0 (569 MB),
Q4_K (422 MB).

**New files**: `src/got_ocr.{h,cpp}`, `models/convert-got-ocr-to-gguf.py`,
`tools/dump_got_ocr_reference.py`, `tests/test_got_ocr_diff.cpp`.

---

## June 2026 — GLM-OCR engine (0.9B, CogViT + GLM-0.5B, MIT)

Port of zai-org/GLM-OCR — #1 on OmniDocBench V1.5, 8 languages, MIT license.
Third VLM in CrispEmbed, with three architectural firsts:

**Architecture**: CogViT (24L, 1024d, RMSNorm+SwiGLU, Q/K RMSNorm, Conv3D
patches) → RMSNorm → Conv2D downsample (stride 2, 576→144 tokens) → Merger
(proj + SwiGLU + LayerNorm) → GLM-0.5B (16L, 1536d, GQA 16/8).

**Unique features**: post-norm (4 norms/layer), Q upscale (1536→2048),
learned Conv2D downsample, mRoPE sections [16,24,24].

**Full pipeline**: KV cache (F16, prefill+decode), vision-text splice
(144 image tokens), tokenizer decode, E2E image→text verified.

**Parity**: 11/11 cos=1.000000 (8 vision + 3 LLM).

**GGUFs**: `cstr/glm-ocr-crispembed-GGUF` — F16 (2.5 GB), Q8_0 (1.1 GB),
Q4_K (849 MB).

**New files**: `src/glm_ocr.{h,cpp}`, `models/convert-glm-ocr-to-gguf.py`,
`tools/dump_glm_ocr_reference.py`, `tests/test_glm_ocr_{diff,e2e,image}.cpp`.

---

## June 2026 — Layout detection fixes + BGE-M3 crash fix

**Layout detection (RT-DETRv2):** Three bugs fixed, score 0.047 → 0.114:
1. AIFI self-attention head interleaving — permute `[hd, N, nh] → [hd, nh, N]`
   before reshape. Encoder features now exact-match Python.
2. Initial reference points — RT-DETRv2 uses `sigmoid(gather(enc_bbox_head(ALL) +
   logit_anchors, top_k))`, not `enc_bbox_head(gathered_queries)`.
3. Identified decoder `cpu_linear` weight convention mismatch (remaining gap).

**BGE-M3 crash:** `clip_text::load()` accepted any model with a tokenizer, loading
BGE-M3 (250K vocab XLM-R) as a 49K-vocab CLIP model → crash. Fixed by checking for
`clip_text.hidden_size` metadata key. BGE-M3 now loads correctly with sparse + ColBERT heads.

**AuraFace Q4_K:** 124 MB → 35 MB (3.5x compression), cos=0.961 vs F16.

---

## June 2026 — GLiNER DeBERTa-v3 NER (Apache-2.0)

Added DeBERTa-v3-base backbone to GLiNER NER — `urchade/gliner_medium-v2.1`,
the most popular GLiNER model (25k+ downloads), fully Apache-2.0 licensed.

**Architecture:** DeBERTa-v3-base (12L, 768h, disentangled c2c+c2p+p2c attention
with log-bucketed relative positions) + 768→512 projection + BiLSTM (hidden=256)
+ GLiNER markerV0 head (start+end only, no first-token projection).

**Implementation:** Unified `src/gliner_ner.cpp` with dual-backbone support.
Backbone auto-detected from `gliner.backbone` GGUF metadata. SentencePiece
tokenizer (128K vocab) via existing `tokenizer_spm.cpp`.

**Quantization:** F32 (747 MB), Q8_0 (198 MB, identical output), Q4_K (152 MB,
minor span merging at edges).

**New files:** `models/convert-gliner-deberta-to-gguf.py`, HF repo at
`cstr/gliner-deberta-GGUF`.

---

## June 2026 — PARSeq scene text recognition (Apache-2.0)

Scene text recognition port: PARSeq (ECCV 2022, baudm/parseq, Apache-2.0).
First dedicated scene text (non-math, non-document) OCR model in CrispEmbed.
Two variants: base (24M params) and tiny (6M params).

**Architecture**: 12-layer pre-LN ViT encoder (patch [4,8], img 32×128,
128 tokens, GELU FFN, fused QKV) → 1-layer two-stream Transformer decoder
(XLNet-style: position queries attend to context via norm_q/norm_c, then
cross-attend to encoder memory) → Linear head (95 classes: 94 printable
ASCII chars + EOS).

**Key design**: Two-stream attention where context tokens combine position
queries + character embeddings. Token ordering: EOS=0, chars=1..94, BOS=95,
PAD=96 (not the typical BOS-first). Single query per AR step for efficiency.

**Variants**:
- Base: embed_dim=384, 6 enc heads, 12 dec heads (head_dim=32)
  F32=91MB, Q8_0=24MB, Q4_K=13MB
- Tiny: embed_dim=192, 3 enc heads, 6 dec heads
  F16=12MB, Q8_0=6MB

**Encoder**: runs as ggml graph (flash_attn_ext, BLAS-backed matmuls).
Patch embedding done CPU-side (non-square kernel [4,8] not supported by
ggml_conv_2d). **Decoder**: CPU-scalar (1 layer, graph overhead not worth it).

**Parity**: Verified identical output to PyTorch on multiple test images.
All quantization levels (F32/Q8_0/Q4_K) produce identical decoded text.

**New files**: `src/parseq_ocr.{h,cpp}`, `models/convert-parseq-to-gguf.py`,
`tools/dump_parseq_reference.py`, `tests/test_parseq.cpp`.

**Bugs found during port**:
1. Token ordering: PARSeq uses `[EOS, chars, BOS, PAD]` not `[BOS, chars, EOS, PAD]`
   — BOS=95, EOS=0 in both head output and embedding space.
2. Context construction: `ctx[0] = embed(BOS)`, `ctx[k] = pos_queries[k-1] + embed(pred)`
   — position queries are added to character embeddings, except BOS which has none.
3. norm_c: context K/V in self-attention must be LayerNorm'd via norm_c (not raw).
4. Head excludes BOS and PAD: 95 output classes = EOS(0) + 94 chars(1..94).

**License**: Apache-2.0 (baudm/parseq). Fully commercial.

---

## June 2026 — InternVL2.5-2B OCR engine (VLM, MIT)

Full vision-language model port: InternVL2.5-2B (2.1B params, MIT license)
for multilingual document OCR. Second VLM in CrispEmbed after Qwen2.5-VL,
with KV cache for efficient autoregressive generation.

**Architecture**: InternViT-300M (24L, 1024d, 16 heads, LayerNorm + GELU +
LayerScale, 448×448 per tile) → pixel unshuffle (4:1, 1024→4096 dim) →
MLP projector (LN-Linear-GELU-Linear, 4096→2048) → InternLM2.5-1.8B
decoder (24L, 2048d, GQA 16/8, SwiGLU, RMSNorm, RoPE θ=1M).

**Key features**:
- Dynamic tiling: 1-12 tiles of 448×448 + optional thumbnail
- KV cache: F16 persistent cache, prefill+decode verified identical
- Vision-text splice: mask-based embedding replacement at `<IMG_CONTEXT>`
- C++ tokenizer decode: SentencePiece BPE from GGUF vocab (▁→space, byte fallback)
- OCRBench ~830 (top tier for models under 3B)

**Parity (F32, all vs Python reference via diff harness):**
- Vision encoder: 4/4 layers cos=1.000000
- Pixel unshuffle + MLP projector: cos=1.000000
- LLM decoder: 2/2 layers cos=1.000000

**E2E verification**: German invoice (600×400, 7 tiles) correctly extracts
invoice number, date, recipient, address, all line items with prices, and
net total.

**New files**: `src/internvl2_ocr.{h,cpp}`, `models/convert-internvl2-to-gguf.py`,
`tools/dump_internvl2_reference.py`, `tests/test_internvl2_{diff,e2e,image}.cpp`,
`tests/test_internvl2_ocr.py`, `hf_readmes/internvl2.5-2b-crispembed-GGUF.md`.

**GGUFs**: `cstr/internvl2.5-2b-crispembed-GGUF` — F16 (4.9 GB), Q8_0 (2.2 GB),
Q4_K (1.4 GB). Vision weights kept at Q8_0 floor in quantizer.

**License**: MIT (OpenGVLab/InternVL2_5-2B).

**Sibling variants on the same engine** (no new code — just GGUF conversion +
registry entries, the InternViT vision tower and projector are shared):
- **InternVL2-1B** (0.9B, MIT) — InternViT-300M + Qwen2-0.5B decoder. Edge/WASM
  target, OCRBench 779. GGUFs: F16 (~1.8 GB), Q8_0 (~1.0 GB), Q4_K (~0.5 GB).
- **H2OVL-Mississippi-2B** (2.1B, Apache-2.0) — InternViT + H2O-Danube2-1.8B
  (Mistral arch). OCRBench 782. GGUFs: F16 (1.2 GB), Q4_K (457 MB).

---

## June 2026 — GLiNER zero-shot NER (LFM2.5 backbone)

Added zero-shot Named Entity Recognition via SauerkrautLM-LFM2.5-GLiNER.
First non-embedding, non-OCR NLP task in CrispEmbed.

**Architecture:** LFM2.5-350M bidirectional backbone (ported from CrispASR's
LFM2-Audio implementation) with:
- 16 layers (10 ShortConv + 6 GQA attention), SwiGLU FFN
- Bidirectional attention (no causal mask) + symmetric conv padding
- Layer fusion (squeeze-and-excitation with sigmoid gates)
- BiLSTM (1-layer bidirectional, word-level)
- GLiNER head: SpanMarkerV1 span representation + dot-product scorer

**Parity (all vs Python reference via diff harness):**
- All 16 backbone layers: cos=1.000000
- Layer fusion: cos=1.000000
- BiLSTM: cos=1.000000
- End-to-end: 17/17 entities match across 5 test texts, mean score Δ=0.030

**New files:** `src/gliner_ner.{h,cpp}` (C++ runtime), `models/convert-gliner-lfm-to-gguf.py`
(converter), `tools/dump_gliner_reference.py` (reference dumper), C API
(`crispembed_ner_*`), server `POST /ner/extract`, Python `CrispNER`, Rust `CrispNER`,
Dart `CrispNER`.

**License:** LFM Open License v1.0 (free under $10M revenue).

---

## June 2026 — Qwen2.5-VL OCR engine (German document OCR)

### Qwen2.5-VL-3B port (feat/keyven-german-ocr branch → merged to main)

Full vision-language model port: Qwen2.5-VL-3B-Instruct as the base
for Keyven/german-ocr-3 (German business document OCR fine-tune).
First VLM in CrispEmbed — all prior OCR models were encoder-decoder
without a language model backbone.

**Architecture**: 32-layer ViT (1280d, 16 heads, 14×14 patches, 2D RoPE,
windowed attention) → spatial merger (2×2 merge, RMSNorm, FC-GELU_erf-FC,
5120→2048d) → 36-layer Qwen2.5 LLM decoder (2048d, GQA 16Q/2KV heads,
SwiGLU FFN 11008d, mRoPE sections=[16,24,24], rope_theta=1M).

**Parity**: cos=1.000000 across all checkpoints:
- Vision encoder: 32/32 ViT layers + spatial merger
- LLM decoder: 2/2 tested layers with mRoPE
- Patch embedding, token embedding: exact match

**End-to-end generation**: Q4_K (2.6 GB) produces coherent German text
from test invoice image. Prompt: "Extrahiere die Rechnung im Bild als JSON"
→ Output: "Um die Rechnung im Bild als" (8 tokens, greedy).

**GGUFs uploaded** to `cstr/qwen2.5-vl-3b-crispembed-GGUF`:
- F16: 7.57 GiB (converted on Kaggle, 73s)
- Q8_0: 3.93 GiB (2x compression)
- Q4_K: 2.61 GiB (3x, vision weights kept at Q8_0 floor)

**Key technical challenges solved**:
1. **Memory-efficient reference dumper** — numpy-based layer-by-layer
   forward pass via safetensors (600 MB peak vs 7.5 GB for PyTorch load).
2. **ggml_set_output()** — without it, graph allocator reuses intermediate
   tensor memory; reading post-compute gives garbage. Gate behind diff mode.
3. **GQA interleave** — `ggml_repeat` tiles [0,1,0,1,...] but GQA needs
   [0,0,...,1,1,...]. Fix: reshape to 4D, repeat on inner dim, reshape back.
4. **mRoPE neghalf** — `GGML_ROPE_TYPE_MROPE` uses neghalf rotation with
   dim pairs (j, j+half), not adjacent (j, j+1). Position tensor layout:
   [t0..tn, h0..hn, w0..wn, 0..0] (4 × n_tokens).
5. **Vision-text splicing** — `x = embed * keep_mask + image_patches`
   (keep_mask=0 at image_pad positions).
6. **Quantizer vision floor** — Q4_K degrades OCR; vision encoder weights
   forced to Q8_0 minimum in `tools/quantize.cpp`.
7. **AutoConfig version hell** — Kaggle's older transformers nests LLM
   params in text_config differently. Fixed: read raw config.json directly.
8. **WASM build fix** — `-sENVIRONMENT=web,worker` required when `-pthread`
   is enabled (pre-existing CI failure, fixed as part of this work).

**Standalone CLI pipeline** (completed 2026-06-12):
- C++ image preprocessor wired into `recognize_raw()` — smart_resize,
  bicubic, normalize, patchify via `image_preprocess.cpp`
- BPE tokenizer loaded from GGUF at init — `set_prompt()` tokenizes
  any text, chat template built in C++ with proper token IDs
- GPT-2 byte decoder for UTF-8 output text
- KV cache: prefill extracts per-layer K/V, decode steps reuse cache
  (O(1) per token instead of O(n) full recompute)
- GGUFs v2 on HuggingFace: all three (F16, Q8_0, Q4_K) include BPE
  tokenizer data (vocab + merges)

**Files added**:
- `src/qwen2vl_ocr.{h,cpp}` — C++ engine + C ABI (~1500 lines)
- `models/convert-qwen2vl-to-gguf.py` — GGUF converter (lazy tensor, with tokenizer)
- `tools/dump_qwen2vl_reference.py` — parity reference dumper
- `tools/qwen2vl_tokenize.py` — chat template tokenizer helper
- `tools/kaggle/qwen2vl-convert/` — Kaggle conversion + quantization kernel
- `tests/test_qwen2vl.cpp` — unit + smoke tests (14/14 pass)
- `tests/test_qwen2vl_diff.cpp` — per-layer parity diff test
- `tests/test_qwen2vl_e2e.cpp` — end-to-end generation test

**Remaining** (see PLAN.md blueprint):
- Load Keyven/german-ocr-3 fine-tuned weights (same arch, different weights)
- Windowed ViT attention (correct but slower without it)
- Python bindings, CrispCalc Dart catalog

---

## June 2026 (late) — surya text detector + MixTex LaTeX OCR

### surya-ocr-2 text detector port

EfficientViT-Large segformer (38M params, 91 languages incl. German).
Segmentation-based text line detection. OpenRail-M license (free <$5M).

**Architecture**: Stem + 4 CNN stages (FusedMBConv, MBConv) + 6
EfficientVitBlock (LiteMLA linear attention) + SegFormer FPN decode head.
Input 1200×1200, output 300×300 heatmap → polygon bounding boxes.

**Parity**: Verified exact match vs Python reference (heatmap max=0.9649,
mean=0.0113, both exact). Per-stage activation means match to 4dp through
all 10 encoder stages + decode head.

**Performance**: ggml graph acceleration for stages 0-2 + block0
(17s graph vs ~10min scalar = 35x). Total: ~1 min (was ~13 min).

**Quantized**: F32=147MB, F16=74MB, Q8_0=41MB (3.6x), Q4_K=23MB (6.5x).
All uploaded to https://huggingface.co/cstr/surya-det-GGUF

**Fully wired**: C ABI (`crispembed_text_det_*`), HTTP server
(`POST /text/detect`), Python bindings (`CrispTextDetect`), model
registry with auto-download, test binaries.

**Bugs found and fixed**:
1. `H /= 2` gives wrong result for odd H (75→37 instead of 38)
2. Stage 2+3 MBConv used ReLU6 instead of Hardswish
3. F16 GGUF: bias tensors need F32 cast before ggml_add

### MixTex Chinese+English LaTeX OCR port

Swin-Tiny encoder + 4-layer RoBERTa decoder (86M params, Apache-2.0).
First Swin (shifted-window attention) encoder in CrispEmbed.

**Architecture**: Patch embed (Conv2d 4×4) → 4 Swin stages
(depths=[2,2,6,2], window_size=7, shifted windows, relative position
bias) → patch merging → final LayerNorm → 4-layer RoBERTa decoder
with cross-attention → BPE tokenizer (25681 tokens, LaTeX + CJK).

**Parity**: cos=1.000000 on all encoder blocks (non-shifted and shifted).
Per-block diff harness verified: enc_embed, s0_b0_out, s0_b1_ln1,
s0_b1_attn_out_windowed, s0_b1_attn_merged, s0_b1_attn_res, s0_b1_out
all pass with max_abs < 2e-5. Quantized (Q8_0) produces identical output.

**Bugs found and fixed** (6 total):
1. Swin PatchMerging must pad odd dims before halving (125→126→63 not 125→62)
2. Cyclic shift sign convention — `cyclic_shift(shift_h=s)` computes
   `out[y]=in[(y+s)%H]` but `torch.roll(shifts=s)` computes
   `out[y]=in[(y-s)%H]`. Signs were inverted for both forward and reverse.
3. Pad-then-shift order — HF Swin pads to window-size multiples FIRST,
   then applies torch.roll. C++ was shifting on the unpadded grid then
   padding. This changes where boundary tokens end up in windows.
4. GELU variant — C++ used tanh approximation, HF Swin uses `nn.GELU()`
   (exact erf). Changed to `0.5 * x * (1 + erff(x / sqrt(2)))`.
5. PatchMerging 2×2 concat order — HF concatenates [TL, BL, TR, BR] but
   C++ had [TL, TR, BL, BR]. All 4 encoder stages diverged.
6. RoBERTa position embedding offset — positions start at index 2
   (padding_idx=1), not 0. Using index 0 reads wrong embeddings.

**Decoder parity** (step 0, all vs HF reference):
All checkpoints cos=1.000000 — embedding, self-attention, cross-attention
Q/K/V, all 4 decoder layers, and step-0 logits. Real math formula
`x^2 + y^2 = r^2` produces correct LaTeX matching HF for ~15 tokens.

**Debugging methodology**: Systematic per-step diff comparison with
named Python reference tensors. The per-step approach was critical:
encoder blocks all passed but stage output failed → PatchMerging bug.
Decoder embedding + self-attention passed but cross-attention failed
→ pre-computed K/V from wrong encoder output → traced back to PatchMerging.

**GGUFs**: F32=329MB, F16=165MB, Q8_0, Q4_K.
Wired into unified math OCR dispatch (auto-detect from GGUF arch).

---

## June 2026 (late) — PosFormer handwritten math OCR

### PosFormer port (feat/posformer-port branch)

PosFormer = BTTR + Attention Refinement Module (ARM) for coverage-aware
decoding. Source: SJTU-DeepVisionLab/PosFormer (BSD-2, academic-only).
6.4M params, 113 LaTeX tokens, 24.9 MB F32 GGUF.

**Architecture**: DenseNet encoder (same as BTTR) + 3-layer Transformer
decoder (d=256, 8 heads, FFN=1024) + shared ARM module. ARM applies
coverage-based attention refinement between decoder layers 0→1 and 1→2.

**CROHME 2014 eval (986 images, greedy L2R)**:
- Raw match:    552/986 = **56.0%** (vs BTTR 49.2%, HMER 36.1%)
- Parsed match: 605/986 = **61.4%** (vs BTTR 49.8%, HMER 36.3%)
- Published 62.7% uses bi-directional beam search; ~6pp gap is expected.

**Quantized**: Q8_0 (12 MB), Q4_K (10 MB) — both lossless on test images.
Uploaded to HuggingFace: https://huggingface.co/cstr/posformer-hw-GGUF

**Port verified**: per-step logit cosine similarity = 1.000000 vs PyTorch
reference (max diff < 0.00001). Four encoder/decoder bugs found and fixed:
1. Spurious ReLU after feature projection Conv1x1
2. LayerNorm and 2D positional encoding order swapped
3. Sin/cos frequency indexing error (cos used wrong frequency in each pair)
4. Missing decoder input LayerNorm (decoder.norm after embed + pos_enc)

**Debugging methodology**: PyTorch-side layer dump scripts
(tests/parity/posformer_*.py) + C++ POSFORMER_DUMP env-gated intermediate
dumps. Compare cosine similarity per-layer, per-step. First divergence
at layer 0 self-attention output led to finding the missing LayerNorm.

**Files**: `posformer_ocr.{h,cpp}`, `convert-posformer-to-gguf.py`,
`test_posformer.cpp`, `test_posformer_batch.cpp`,
`tests/parity/posformer_*.py`.

**Training pipeline** (v25, 25 iterations to get right):
Kaggle kernel at https://www.kaggle.com/code/chr1str/posformer-train-on-mathwriting
W&B at https://wandb.ai/cze-github/posformer-hmer

Key issues solved during Kaggle kernel development:
- P100 GPU (sm_60): install torch+cu118 (supports sm_60), not CPU fallback
- Auth: clone CrispASR at runtime, import kaggle_harness (3-tier auth).
  Dataset mounts at `/kaggle/input/datasets/chr1str/crispasr-hf-token/`,
  NOT `/kaggle/input/crispasr-hf-token/`. Harness patched to scan both.
- **Vocab bug**: `build_vocab_from_zip` sorted by frequency, scrambling
  110/113 token indices. Model trained 25 epochs was unusable. Fixed:
  use canonical PosFormer dictionary.txt (alphabetical order).
- OOV: 14 CROHME captions have `'` not in dictionary. Filtered.
- Validation speed: override beam_size=10 bidir → beam_size=1 greedy.
  Full bidir takes 30-60 min per val epoch.
- Heartbeat: `kh.build_heartbeat("train")` for Kaggle keepalive.

**Training progress** (correct vocab, label smoothing 0.1):
- Epoch 8: 22.4% beam=1
- Epoch 64: 43.4% beam=1 (pre-LR-fix)
- Epoch 93: 57.0% beam=1 (LR=0.005, surpasses SJTU published 56.0%)
- Epoch 108: 59.3% beam=1 (CROHME-only ceiling)
- Epoch 125: 61.9% val_ExpRate (CROHME + 1000 MathWriting, LR=0.005)
- **Epoch 182: 60.5% beam=1 / 60.3% beam=10** (CROHME + 2000 MathWriting,
  LR=0.00125 after ReduceLROnPlateau drop). Best verified full eval.
- W&B peak: 62.03% val_ExpRate at step 304,204

Key findings:
- MathWriting augmentation (2000 samples) broke the 59.3% CROHME-only ceiling
- ReduceLROnPlateau drop (0.005→0.00125) triggered the 62% peak
- Beam=10 bi-directional does NOT help (60.3% < 60.5% beam=1)
- Model is better at greedy than bi-directional decoding
- deepcopy/MathWriting-human on HF has pre-rasterized images (no InkML parsing)

See PLAN.md for v2 expanded vocab design (183 tokens, 206K samples).

**License**: SJTU weights = academic-only. Retrained weights on CROHME
= CC BY-NC-SA 3.0 (NC). Fine for "buy me a coffee" app: app code is
commercial, weights downloaded separately with NC terms acceptance.
All handwritten math datasets are NC. The C++ inference is clean-room.

---

## June 2026 — OCR feature parity across all surfaces

### PosFormer port merged to main
- `posformer_ocr.cpp` (961 LOC): DenseNet encoder + Transformer decoder
  with Attention Refinement Module (ARM), ported from `feat/posformer-port`
- Wired into unified dispatcher (`MATH_OCR_POSFORMER` enum + all switch blocks)
- Converter: `models/convert-posformer-to-gguf.py`
- Registry: `posformer-crohme` at `cstr/posformer-crohme-GGUF` (CC BY-NC-SA 3.0)
- 57% exact match on CROHME 2014 (best handwritten model)

### General OCR pipeline (detect + recognize) wired everywhere
- **CLI**: `--ocr-det MODEL --ocr-rec MODEL --ocr IMAGE` (new flags)
- **Server**: `POST /ocr` endpoint (detect text regions → recognize each crop)
- **Python**: `CrispOcrPipeline(det_model, rec_model)` — `run()` + `recognize()`
- **Rust**: `OcrPipeline::new()` / `run()` + `MathOcr::recognize_gray()`
- **Flutter/Dart**: `CrispOcrPipeline` class + `OcrResult` + FFI typedefs

### Registry expanded
- Added: pix2tex-mfr, texo-distill, posformer-crohme, dbnet-det,
  trocr-printed, layout-heron (6 new entries, 8 OCR total)

### Stale worktrees cleaned
- Merged and removed: feat/posformer-port, feat/layout-detect-fix,
  feat/layout-parity, feat/ocr-detect
- CrispASR: removed worktree-feat+tts-watermark-metadata,
  worktree-fix-piper-roundtrip

---

## June 2026 — RT-DETRv2 Layout Detection

### Document layout analysis: ResNet-50 + HybridEncoder + deformable decoder
- Architecture: ResNet-50-D backbone + HybridEncoder (AIFI self-attention +
  FPN/PAN with CSP-RepVGG blocks) + 6-layer transformer decoder with
  deformable multi-scale cross-attention (300 queries, 17 classes)
- 14 parity bugs found and fixed via systematic layer-by-layer diff:
  AIFI pos/LN/residual, PAN lateral features, cpu_linear weight convention,
  converter weight transposition (Gemm/Split/Transpose patterns),
  decoder_input_proj Conv convention, valid_mask, query_pos_head architecture,
  bilinear resize, grid_sample alignment
- All encoder stages cos=1.0 with exact input (verified via crispembed-diff)
- Detection score 0.934 on test images (HF reference: 0.955)
- Performance: 21s with BLAS (was 178s without — 8.5x speedup)
- Quantized: F32 161 MB, Q8_0 43 MB (3.7x compression)
- Published: huggingface.co/cstr/layout-heron-gguf (F32 + Q8_0)
- Fully wired: C ABI, CLI (`--layout`), server (`POST /layout/detect`),
  Python (`CrispLayout`), Rust (`CrispLayout`), Dart/Flutter
- Source: docling-project/docling-layout-heron (Apache-2.0, 42M params)

---

## June 2026 — WASM build (math OCR in browser)

### CrispEmbed compiled to WebAssembly via Emscripten
- `build-wasm.sh`: emcmake cmake, CPU-only, SIMD128, MODULARIZE=1
- Output: `crispembed_ocr.js` (62K) + `crispembed_ocr.wasm` (999K)
- `wasm/ocr_wrapper.c`: thin C entry point exposing `wasm_ocr_init`,
  `wasm_ocr_recognize_gray`, `wasm_ocr_recognize`, `wasm_ocr_free`
- Emscripten guards: `model_mgr.cpp` (disable curl/wget),
  `gguf_loader.cpp` (skip mmap, use fread fallback)
- `cmake/FindThreads.cmake`: stub override creates no-op Threads::Threads
  target, avoiding -pthread and SharedArrayBuffer/COOP/COEP requirement
- Integrated into CrispCalc web/PWA: `dart:js_interop` bridge, IndexedDB
  model caching, conditional import selects WASM provider on web
- All existing OCR models work: pix2tex, HMER, BTTR, PosFormer, Texo,
  PP-FormulaNet-L (auto-detected from GGUF architecture tag)
- Tested end-to-end: model load (16.8 MB, 1.5s) + encoder (578 tokens)
  + decoder (201 tokens) → LaTeX output in Node.js

### HuggingFace Space
- `hf-space/`: Docker build (two-stage) + Gradio UI (3 tabs: text
  embeddings, math OCR, health)
- Pattern: C++ `crispembed-server` on :8090 + Gradio on :7860
- Default models: all-MiniLM-L6-v2 (text) + hmer-hw (OCR)
- Tested: cos=0.785 for similar texts, `x² + 1 = 0` → `x ^ { 2 } + 1 = 0`
- Live at https://huggingface.co/spaces/cstr/CrispEmbed

### CI
- `build-wasm.yml`: builds WASM on push/PR, uploads artifacts
- `deploy-hf-space.yml`: auto-deploys `hf-space/` to HuggingFace on push

---

## June 2026 — PP-FormulaNet-L OCR (181M params)

### Full in-graph ViT encoder with decomposed RPE
- **Full ggml graph encoder**: all 12 ViT layers run as single ggml graphs
  with attention + decomposed relative position bias entirely in-graph
- Window batching: all 16 windows × 12 heads processed as one batch dimension
  via reshape + permute, enabling efficient batched matmuls
- Decomposed RPE in-graph: two matmuls (rp_h@Q, rp_w@Q_permuted) with
  broadcast-add to attention scores (Granite NLE pattern)
- LN ordering fix: for windowed layers, LayerNorm1 applied on CPU before
  window partition to match HF's LN→pad→QKV ordering. Prevents LN(0)=bias
  corruption of padding tokens (cos jumped from 0.973 to 0.9999)
- Per-layer parity: cos ≥ 0.99997 on all 12 layers
- Performance: 53s encoder with BLAS+Q8_0 (60s F32, was 77s hybrid, ~500s scalar)

### Printed math OCR: SAM-ViT encoder + MBart decoder
- New architecture: SAM-style ViT encoder (12 layers, 768d, 12 heads)
  with windowed attention (ws=14) + global attention (layers 2,5,8,11)
  and decomposed relative position bias
- Neck: Conv1x1 + LayerNorm2d + Conv3x3 + LayerNorm2d (768 → 256)
- Multi-modal projector: 2× Conv3x3(stride=2) + 2× Linear (256 → 512)
  Output: (144, 512) sequence for decoder
- MBart PRE-LN decoder: 8 layers, 16 heads, d_model=512, FFN=2048
- 768x768 RGB input, UniMERNet preprocessing pipeline
- Encoder parity: cos=0.999962 vs HuggingFace reference (F32)
- Quantization: F32 (692 MB), F16 (347 MB), Q8_0 (241 MB, cos=0.999940),
  Q4_K (122 MB, cos=0.997595) — all produce identical decoded LaTeX
- Smart Q8_0: critical tensors (embeddings, LN, rel_pos, lm_head) in F16
- Auto-detected from GGUF metadata (`general.architecture = ppformulanet_l`)
- Wired into unified `--ocr` CLI, C ABI, model registry, CrispCalc Dart catalog
- Source: PaddlePaddle/PP-FormulaNet-L_safetensors (Apache-2.0)
- New GGUF loader helper: `kv_i32_array()` for int32 metadata arrays

### Full-stack wiring
- HTTP server: `POST /math/ocr` endpoint (`--ocr` flag, stb_image load, JSON response)
- Python bindings: `CrispMathOcr` class with `recognize()` and `recognize_gray()`
- Updated contributing.md with server + Python binding steps
- Updated public C header comments to list all supported architectures

## June 2026 — PPFormulaNet-S / Texo-Distill OCR

### Printed math OCR: HGNetv2 + MBart decoder (20M params)
- New architecture: HGNetv2 CNN encoder (StemBlock, 4 HG_Stages, LightConvBNAct)
  + MBart Transformer decoder (2 layers, 16 heads, 384 d_model)
- Conv-BN folding in GGUF converter: all BatchNorm absorbed into preceding Conv2d
- CPU-side CNN forward pass for encoder (all standard ops: conv2d, relu, maxpool, concat)
- MBart PRE-LN decoder: LayerNorm before attention/FFN, residual skips LN
- UniMERNet preprocessing: aspect-ratio-preserving resize + black pad + grayscale
  normalize (mean=0.7931, std=0.1738)
- ODR fix: renamed internal dec_layer → ppfn_dec_layer to avoid linker collision
  with decoder_embed_internal.h
- Added `--ocr` CLI flag for unified auto-detection (pix2tex/hmer/bttr/ppformulanet)
- Quantized: F16 (39 MB), Q8_0 (22 MB, identical quality), Q4_K (13 MB, degraded)
- GGUF models published: huggingface.co/cstr/texo-distill-gguf
- Diff regime: encoder cos=1.000000, decoder verified via layer-by-layer debug traces
- Source: Texo (AGPL-3.0) distilled from PP-FormulaNet-S (Apache-2.0)
  trained on UniMER-1M (CC-BY-4.0)

## June 2026 — Nomic v2 MoE Encoder

### Mixture-of-Experts encoder support
- First MoE embedding model: nomic-embed-text-v2-moe (8 experts, top-2, GELU)
- Fully in-graph MoE routing: ggml_top_k + ggml_get_rows + ggml_mul_mat_id
- Mixed architecture: odd layers = MoE FFN, even layers = dense GELU FFN
- Converter handles GPT2-style config (NomicBERT extends GPT2Config),
  per-layer MoE/dense auto-detection, expert weight stacking [n_exp, dim, dim]
- Fixed missing Wqkv + out_proj biases (present in v2-moe but not v1.5)
- Exact erf-GELU activation (NomicBERT uses nn.GELU(approximate='none'))
- Parity: cos=1.000000 vs HuggingFace on all test texts
- Quantized variants: F16 (1344 MB), Q8_0 (1122 MB, cos=0.9996), Q4_K (1095 MB, cos=0.966)
- GGUFs published to cstr/nomic-embed-text-v2-moe-GGUF on HuggingFace
- Extended parity_layers_bert.py harness with --arch nomic (QKV split, MoE expert tensor diff)
- Added CRISPEMBED_DUMP_LAYERS env var for per-layer intermediate tensor dumps

---

## June 2026 — LoRA Hot-Swap, Batched Decoder, Face Pipeline

### LoRA adapter hot-swap
- Runtime switching between Jina v5 per-task LoRA adapters (retrieval,
  classification, clustering, text-matching) without re-loading the model
- Pre-compute approach: `W' = W + (α/r)·B@A` on CPU at switch time (~10-50ms)
- Converter `--lora-mode=separate` stores base weights + per-adapter A/B
  tensors (F16) in a single GGUF with metadata
- Lazy base weight snapshot with dequant→merge→requant for quantized models
- C API: `crispembed_set_lora/get_lora/list_lora`
- CLI: `--lora NAME`, `--list-lora`
- Python: `set_lora()`, `lora` property, `list_lora()` on CrispEmbed

### Batched decoder graph
- Single ggml graph compute for N decoder texts (was: N sequential passes)
- Block-diagonal causal mask (text i cannot attend to text j), padding
  positions get self-attention to prevent softmax NaN
- Independent RoPE positions per text, pad to T_max
- Per-text last-token / mean pooling after graph compute
- **3.3x speedup** on batch of 4 (Jina v5 nano, CPU)
- Parity: cos >= 0.999 vs sequential encoding on all test texts

### Face pipeline Python completion
- `CrispFacePipeline` exported in `__init__.py`
- `from_registry()` class methods on `CrispFace` and `CrispFacePipeline`
  for auto-download by registry name
- Unit tests (`tests/test_face_python.py`): 12 tests covering detection,
  recognition, pipeline, match, edge cases
- Example script (`examples/face_search.py`): index faces from directory,
  query by image, top-K cosine matches

### BTTR beam search decoder
- Beam search with configurable width (default 5) for BTTR handwritten
  math OCR — improves exact-match accuracy over greedy decoding

### Windows CI fix
- `M_PI` undefined on MSVC: added `#ifndef M_PI` portable fallback in
  `bttr_ocr.cpp`

---

## June 2026 — CLIP/SigLIP Vision + Text, YuNet, HMER/BTTR OCR

### YuNet lightweight face detection
- 228 KB GGUF (vs SCRFD 16 MB), ShuffleNetV2 backbone, 640×640 input
- IoU 0.99 vs OpenCV FaceDetectorYN, score diff < 0.01, landmark diff < 2px
- Converter unchanged (existing `convert-face-to-gguf.py` handles YuNet's ops)
- Key gotcha: ggml Transpose op does real 2D transpose for n_dims==2 tensors,
  requiring different spatial indexing for 1-channel (cls/obj) vs multi-channel
  (bbox/kps) outputs
- Uploaded to `cstr/yunet-GGUF`, in auto-download registry

### CLIP text encoder (new module)
- `clip_text_embed.{h,cpp}`: pre-LN transformer with causal mask, EOS pooling,
  text_projection, BPE tokenizer embedded in GGUF
- `convert-clip-text-to-gguf.py`: extracts text tower + tokenizer from HF CLIP
- C API (`crispembed_clip_text_*`), Python wrapper (`CrispClipText`), server
  `/clip/text` endpoint
- Cross-modal text↔image search works end-to-end
- Uploaded: `cstr/clip-text-base-GGUF` (244 MB), `cstr/clip-text-large-GGUF` (474 MB)

### CLIP / SigLIP vision models
- Fixed `vit_embed.cpp`: CLS token prepend, CLS pooling for CLIP, quick_gelu
  via FP32 ggml primitives, attention pooling residual skip connection
- Converted and uploaded 6 vision GGUFs:
  - `cstr/clip-vit-base-patch16-GGUF` (329 MB, MIT)
  - `cstr/clip-vit-large-patch14-GGUF` (1.2 GB)
  - `cstr/clip-vit-large-patch14-336-GGUF` (1.2 GB)
  - `cstr/siglip-large-256-GGUF` (1.2 GB, Apache 2.0)
  - `cstr/siglip-so400m-patch14-384-GGUF` (1.6 GB)

### Handwritten math OCR (HMER + BTTR)
- HMER: DenseNet-121 encoder + GRU attention decoder (with coverage).
  Source: whywhs/Pytorch-HMER (code: MIT), trained on CROHME 2016
  (CC BY-NC-SA 3.0). Weights inherit NC.
  112 LaTeX tokens, ~6.8M params, ~4-5 MB Q4_K.
  `hmer_ocr.{h,cpp}`, `convert-hmer-to-gguf.py`. CLI: `--hmer FILE`.
  Auto-detect image polarity and invert if needed. Dequant support.

- BTTR: DenseNet encoder (growth=24, 3 blocks) + Transformer decoder
  (3 layers, 8 heads, d=256). Source: Green-Wood/BTTR (code: MIT),
  trained on CROHME 2014 (CC BY-NC-SA 3.0). Weights inherit NC.
  113 LaTeX tokens, 6.5M params. 49.2% raw / 49.8% parsed on CROHME.
  `bttr_ocr.{h,cpp}`, `convert-bttr-to-gguf.py`.
  BN folded into conv, fused QKV preserved.

### SFace quantization (conv2d quant support)
- Converter flattens 4D conv weights to 2D [OC, IC*KH*KW] for quantization
- Runtime: dequant Q8/Q4→F32, reshape back to 4D, cast to F16 for ggml_conv_2d
- SFace results: F32=37MB, Q8_0=10MB (cos=0.9999), Q4_K=6MB (cos=0.974)
- Uploaded to `cstr/sface-GGUF` (F32 + Q8_0 + Q4_K)
- Same pattern applies to AuraFace and SCRFD (reconverted with flat conv)
- AuraFace: 249 MB (Q8_0 only compresses FC → 212 MB; conv rows too small for Q8_0)
- SCRFD: 17 MB (minimal Q8_0 gain — detection heads are small)
- AuraFace F16: 249→125 MB (2.0x, lossless — conv2d casts to F16 anyway)
- SCRFD F16: 17→8 MB (2.0x, lossless)
- Added F16 support to quantizer (Q8_0/Q4_K need row÷32; F16 has no alignment limit)

### Face model quantized graph replay fixed
- YuNet F16/Q8_0 inference via graph replayer now works (was crashing)
- Three fixes: (1) parse Conv group attrs before 2D→4D reshape for
  correct depthwise IC detection, (2) handle ggml_n_dims returning 2
  for 4D weights with trailing 1s via element count validation,
  (3) dequant Q→F32 before F16 cast (ggml only supports Q→F32)
- Q8_0 detection matches F32 with minor quantization drift (conf 0.731 vs 0.749)
- Old-style 4D-weight GGUFs and new-style 2D-flattened GGUFs both work
- YuNet parity verified: sub-pixel match vs OpenCV FaceDetectorYN on both
  single-face and multi-face images (x/y/w/h diff < 0.5px, conf diff < 0.01)
- Raw tensor cos vs ONNX (0.35-0.85) is a false alarm — planar (ggml) vs
  interleaved (ONNX) layout of the same correct data; decoded coords match

### SigLIP text encoder verified
- cos=1.000000 vs HuggingFace on all test texts
- SentencePiece BPE vocab decoded correctly by Viterbi unigram algorithm
- Key finding: SP BPE training doesn't change inference algorithm — Viterbi works

### Model registry expansion
- 13 new auto-download entries: 2 face detection (yunet, scrfd-det-10g),
  2 face recognition (auraface-v1, sface), 8 vision/text (CLIP + SigLIP),
  1 SigLIP-base
- Total registry: ~58 models

### Vision parity fixed: cos 0.8 → 1.0
- Root cause: patch embedding `ggml_permute(2,1,0)` produced column-major
  spatial ordering (t = ow*OH + oh), but HuggingFace uses row-major
  (t = oh*OW + ow via flatten(2)). Every patch beyond (0,0) got the
  wrong position embedding.
- Fix: `ggml_permute(1,2,0,3)` produces [D, OW, OH] which flattens to
  row-major matching HF. Per-layer cos goes from ~0.3 to 1.000000.
- Final embedding cos = 0.9998 vs HuggingFace (SigLIP-base-384)
- CLIP ViT also verified: cos=1.000000, max_diff=0.000001 (clip-vit-base-patch16)
- Was NOT "FP32 non-associativity" as previously hypothesized — it was
  a simple permutation index bug that scrambled patch positions

---

## v0.7.0 — May 2026

### Registry status

45 models in registry, 151 GGUF variants published on HF:
25 encoder models + 11 decoder models + 12 rerankers + 1 SPLADE + 2 multimodal.
Typical per-model: F32 + Q8_0 + Q4_K; about a dozen also have Q5_K / Q6_K / F16.

Key parity results (cos vs HuggingFace reference):

| Model | Type | Dim | CosSim |
|-------|------|-----|--------|
| all-MiniLM-L6-v2 | BERT | 384 | 1.000000 |
| bge-small/base/large-en-v1.5 | BERT | 384/768/1024 | 1.000000 |
| gte-base/large-en-v1.5 | GTE | 768/1024 | 1.000000 |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 1.000000 |
| nomic-embed-text-v2-moe | NomicBERT MoE | 768 | 1.000000 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000000 |
| all-mpnet-base-v2 | MPNet | 768 | 1.000000 |
| multilingual-e5-small/base/large | XLM-R | 384/768/1024 | 1.000000 |
| snowflake-arctic-embed-m/l | BERT/XLM-R | 768/1024 | 1.000000 |
| bge-m3 (dense+sparse+ColBERT) | XLM-R | 1024 | 1.000000 |
| splade-pp-en-v1 | BERT SPLADE | 768 | 1.000000 |
| granite-embedding-278m/107m | XLM-R | 768/384 | 1.000000 |
| gte-modernbert-base | ModernBERT | 768 | 0.9999 |
| pixie-rune-v1 | XLM-R | 1024 | 0.999993 |
| octen-0.6b | Qwen3 | 1024 | 0.999891 |
| octen-8b | Qwen3 | 4096 | 0.965 (Q4_K vs bf16 HF) |
| qwen3-embed-4b | Qwen3 | 2560 | 0.974 (Q4_K vs bf16 HF) |
| harrier-0.6b / harrier-270m | Qwen3/Gemma3 | 1024/640 | 0.999959/948 |
| jina-v5-nano/small | Qwen3 | 1024 | 0.999941 |
| bge-reranker-v2-m3 | XLM-R reranker | - | verified |
| ms-marco-MiniLM-L-6/12-v2 | BERT reranker | - | verified |

### Optimizations completed

- ggml_backend_sched GPU dispatch (encoder + decoder full-graph)
- All 45 models quantized (Q8_0 + Q4_K) and uploaded to HuggingFace
- Graph/work buffer reuse: 27.8 texts/s server throughput (gte-small)
- Matryoshka dimension truncation via -d N flag
- BLAS/MKL/CUDA/Vulkan/Metal build support
- Windows build scripts
- C++ quantizer with K-quant fallback chain
- QKV weight fusion (1 matmul vs 3 per layer)
- Flash attention with optional position bias mask
- ggml graph decoder for math OCR (27x speedup over scalar)

### Bindings and platforms

| Binding | CrispEmbed | CrispASR |
|---|---|---|
| C API | Complete | Complete (whisper.h) |
| Python (ctypes) | Complete + tested | Complete + tested |
| Rust (crate) | Complete + tested | Complete + compiled |
| Dart/Flutter (FFI) | Complete | Created |
| iOS (Metal) | CI green | CI green |
| Android (NDK) | CI green (arm64/armv7/x86_64) | CI green |
| Windows | CI green | CI green |
| macOS (Metal) | CI green | CI green |
| Linux | CI green | CI green |

### CrispEmbed advantages over fastembed-rs

- **ColBERT multi-vector** retrieval (fastembed-rs doesn't have it)
- **Matryoshka dimension truncation** (fastembed-rs doesn't have it)
- **GGUF quantization** (Q8_0, Q4_K — smaller than ONNX INT8/INT4)
- **9.5x faster on MiniLM-L6** (most popular embedding model)
- **GPU dispatch** via ggml_backend_sched (CUDA/Metal/Vulkan)
- **Ollama-compatible** server with 4 API dialects
- **Flutter/Dart** wrapper for mobile apps
- **iOS/Android** build scripts with full CI
- **20MB binary** vs ~500MB Python+ONNX environment

### Commercially permissive stack (no NC restrictions)

The full pipeline uses only Apache 2.0 / MIT models:
- Text: any CrispEmbed encoder model (BERT/XLM-R/etc.)
- Image: SigLIP (Apache 2.0) or CLIP (MIT)
- Face detection: SCRFD (Apache 2.0) or YuNet (Apache 2.0)
- Face recognition: AuraFace-v1 512-D (Apache 2.0) or SFace 128-D (Apache 2.0)
- Face landmarks: MediaPipe FaceLandmarker (Apache 2.0)
- Audio: CrispASR (our own, Apache 2.0)

### Resolved known issues

1. **NomicBERT** — Root cause: gate/up weights (fc11/fc12) were swapped in old GGUF;
   also needed Ollama tensor name fallback. F32 cos=1.0, Q8_0 cos=0.998.

2. **EmbeddingGemma-300m** — cos=1.0000 F32, 0.9998 Q8_0, 0.9954 Q5_K.
   Root causes: missing `is_bidirectional=1`, wrong pooling, BPE merges not loading,
   Dense layers being quantized. All fixed.

3. **Jina v5 nano/small** — Models use task-specific LoRA adapters; converter now
   merges `retrieval` adapter. Nano F32 cos=1.0, Small F32 cos=0.9999.

4. **all-mpnet-base-v2** — Old GGUF was missing `relative_attention_bias.weight`.
   Reconverted with bias tensor. cos=0.987-0.999.

5. **gte-modernbert-base** — Validation wrongly required `ln1` for pre-LN models.
   Fixed validation. cos=0.9999.

6. **DeBERTa-v2 disentangled attention** — c2p/p2c relative position bias with
   log-bucket encoding now fully implemented. mxbai-rerank-xsmall-v1 and
   mxbai-rerank-base-v1 both working.

7. **Full regression sweep (2026-05-17)**: 34 models tested, all pass. 5 models
   fixed and re-uploaded to HF.

---

## May 2026 — Multimodal & Vision

### BidirLM-Omni (text + audio + image)

- [x] Text path through `decoder_embed.cpp` (cos >= 0.999 vs HF bf16)
- [x] Audio path through `bidirlm_audio.cpp` + crisp_audio (cos = 0.995 vs HF)
- [x] Vision tower in `bidirlm_vision.cpp` (cos >= 0.999 vs HF bf16)
- [x] DeepStack injection + 3D interleaved-MRoPE (cos = 0.998903 vs HF bf16)
- [x] `crispembed_encode_text_with_image` C ABI + Python wrapper
- [x] `crispembed_encode_with_image_ids` (pre-tokenized variant for parity tests)
- [x] CLI `--image FILE` + `--image-raw patches.f32 --grid-thw T,H,W`
- [x] Decoder `ggml_backend_sched` initialization
- [x] Memory-efficient lite parity test
- [x] In-process C++ image preprocessor (smart_resize + Catmull-Rom bicubic)
- [x] BPE special-token handling for Qwen-style tokens
- [x] Stale-GGUF fallbacks for missing metadata
- [x] Image batching in `encode_text_with_image`

### Phase 8: Vision — Image Embeddings, Face Detection & Recognition

#### 8A. SigLIP Image Embedding (DONE)

cos=0.996 vs HF. Uploaded to cstr/siglip-base-GGUF.
- GGUF converter, ViT forward path, image preprocessing
- CLI: `crispembed -m siglip-base.gguf --image photo.jpg`

#### 8B. Face Detection — SCRFD (DONE)

Scores match ONNX Runtime. Uploaded to cstr/scrfd-det-10g-GGUF.
- Generic ONNX graph replayer (Conv, ReLU, Add, Pool, Resize, Concat, Sigmoid)
- FPN + multi-scale detection heads + NMS
- Letterbox preprocessing + coordinate scaling
- C API, Python, Rust, Dart wrappers

#### 8C. Face Recognition — AuraFace + SFace (DONE)

cos=0.9999 vs ONNX for both models.
- BN folding/precomputation, 512-D/128-D embeddings
- Full detect-align-encode pipeline
- C API, Python, Rust, Dart wrappers
- Server API: `/detect`, `/face` endpoints

---

## April 2026 — RAG Feature Parity

- [x] Full Python/Rust/Dart wrapper: sparse, ColBERT, reranker, set_dim, set_prefix
- [x] Bi-encoder reranking API (Python + Rust + Dart): cosine similarity ranking
- [x] Prompt prefix system (C/Rust/Python/Dart): auto-prepend query/passage prefixes
- [x] 21 verified embedding models (cos >= 0.999 vs HuggingFace)
- [x] 5 reranker models (bge-reranker-base, ms-marco L6/L12, mxbai-rerank xsmall/base)
- [x] 27 HuggingFace repos with GGUF models + README cards
- [x] RAG retrieval quality benchmark (tests/bench_rag.py): MRR@10, NDCG@10, Recall@k
- [x] Reranking benchmark (tests/bench_rerank.py): cross-encoder vs bi-encoder
- [x] Head-to-head benchmark vs FastEmbed:
  - MiniLM-L6: CrispEmbed **9.5x faster** single, **10.8x faster** batch
  - BGE-small: FastEmbed 1.7x faster (ONNX graph JIT optimization)
  - Arctic-M: tied on batch (126 vs 127ms)
  - cos = 0.999999-1.000000 cross-engine on all models
- [x] Demo apps (Python + Rust) for both CrispEmbed and CrispASR

---

## May 12, 2026 — Face Pipeline Complete

Full detect -> align -> encode pipeline for face recognition.

### RAG parity: prompt prefixes + new models

- Added auto-prefix system: BGE, E5, Nomic, Jina models get query/passage
  prefixes auto-applied.
- Converted 3 new models: SPLADE-PP-en-v1, granite-embedding-278m/107m.
- Model registry: 47 models total.

### SCRFD preprocessing + anchor decode fixes (3 bugs)

1. RGB-BGR channel swap
2. Anchor center offset (integer grid, no 0.5 offset)
3. Top-left placement (not centered letterbox)

### SCRFD anchor decode fix (data layout mismatch)

Channel-last vs interleaved indexing. After fix: detection counts match
InsightFace exactly on all test images.

### Face alignment fix (4 sign errors in normal equations)

After fix: alignment matches InsightFace `norm_crop` with MAE=0.00.
Per-face embedding cos=0.994-0.999 vs InsightFace ArcFace.

### Pipeline implementation

- `cnn_embed::detect_file()` — letterbox resize, coordinate scaling
- `cnn_embed::encode_aligned()` — 5-point landmark similarity transform + encode
- `cnn_embed::face_pipeline()` — detect -> align -> encode in one call
- CLI, C API, Server API, Python/Rust/Dart wrappers all complete

### Models converted

- SCRFD-10GF (16.1 MB)
- w600k_r50 ArcFace (166 MB)
- AuraFace-v1 (248.6 MB)
- SFace (36.8 MB)

---

## May 11-12, 2026 — Vision Models & Parity Fixes

### SigLIP image embedding
- Converter: `models/convert-siglip-to-gguf.py`
- Forward path: `src/vit_embed.cpp` — cos=0.996 vs HF mean-pool
- Native `--image` flag with stb_image preprocessing
- Uploaded: cstr/siglip-base-GGUF

### Face detection (SCRFD)
- Generic ONNX graph replayer in `src/cnn_embed.cpp`
- FPN backbone + multi-scale detection heads
- Anchor decode + NMS at strides 8/16/32
- Semicolon delimiter for ONNX tensor names with commas

### Face recognition (SFace + AuraFace)
- SFace MobileFaceNet: cos=0.9999 vs ONNX, 128-D
- AuraFace ResNet-100: cos=0.9999 vs ONNX, 512-D
- BN folding/precomputation at converter time
- PReLU: relu(x) + slope * (x - relu(x))
- Conv F32->F16 auto-cast for ggml_conv_2d

### Text model parity fixes (35 models)
- GTE v1.5: post-LN + GeGLU half swap + NTK RoPE
- Jina reranker v2: post-LN + position offset
- NomicBERT: SwiGLU fc11/fc12 swap
- Ollama format: auto-strip prefix, dual metadata keys, pooling type mapping

---

## Apr 12, 2026 — v0.1.0 Release

30-commit session: FastConformer extraction, granite 3.x support,
NeMo FC-CTC, omniASR, Silero LID, CI, Windows, Vulkan, benchmarks.
Tagged v0.1.0 release with multi-platform binaries.
