# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## 🚧 Active work in flight (update + push to `main` at EVERY checkpoint)

Multiple sessions/worktrees run in parallel and push to `main` concurrently.
Before starting a task, add a row; at every checkpoint update it and push this
file to `main` so others see what's claimed (avoids duplicate work + CI-cancel
races). Remove the row when the branch lands.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-07-31 | `main` | External document-parser-informed OCR pipeline: structured routing, in-memory handoffs, service contracts, batching, and benchmark gates | **IN PROGRESS** |
| 2026-07-31 | `main` | Real-world public-domain OCR corpus and manifest-driven multi-engine live benchmarks | **IN PROGRESS** |

> **Board cleared 2026-07-20** — all 18 previously-listed in-flight items had
> landed; the index + preserved specifics are in `HISTORY.md` "July 20, 2026 —
> PLAN.md active-work board cleared". Add a row here when you START a task; remove
> it when the branch lands.
>
> Completed milestones live in `HISTORY.md`; technical deep-dives in
> `LEARNINGS.md`. This file tracks the current architecture and what is
> still **pending**.

## OCR pipeline workstream — actionable items

This workstream is informed by the external document-parser comparison, but keeps CrispEmbed's
ggml portability (CPU, CUDA, Metal, Vulkan, and WASM). Items are scoped so each
can land and be measured independently.

### O1 — Restore a trustworthy OCR baseline [COMPLETED]

- Fix duplicate region emission in the batched DBNet + TrOCR path.
- Add a regression test for one output region per detected region and no
  duplicated reading-order text.
- Record baseline latency and region/text counts in `PERFORMANCE.md`.

**Started:** DBNet postprocessing now handles degenerate one-point contours;
the local fox fixture improves from 0 to 10 detected regions. The remaining
baseline work is an automated model-backed assertion and sequential/batched
comparison.

**Done when:** batch and sequential recognition produce equivalent region counts
and no duplicate text on the OCR fixture set. The benchmark harness now accepts
`--expect-regions` and repeated `--expect-text` assertions for CI.

### O2 — Define a structured document result contract [COMPLETED]

- Add a C++ `ocr_document` result containing page dimensions, text regions,
  layout regions, tables, formulas, confidence, and engine provenance.
- Keep the existing orchestrator result and C API source-compatible; provide an
  adapter first, then migrate callers.
- Add serialization tests for empty, text-only, and mixed structured results.

**Started:** `ocr_orchestrator::result` now carries page dimensions and optional
layout regions. Layout inference is lazy and remains disabled unless
`config.layout_model` is set; existing callers and default latency are unchanged.

**Done when:** callers can consume one structured result without depending on a
specific OCR engine.

### O3 — Add CPU-only region routing after layout detection [COMPLETED]

- Introduce a pure routing module with `text`, `table`, `formula`, and
  `fallback` destinations.
- Route by layout label, confidence tier, containment/overlap, and explicit
  per-request feature policy; suppress duplicate text when a specialized
  recognizer owns a region.
- Unit-test every decision seam without model weights.

**Started:** `ocr_orchestrator::result` now carries the model-free routing plan;
table/formula/image policy is explicit in `config` and text-only by default.

**Done when:** a synthetic page produces a deterministic routing plan and the
existing specialized engines can be dispatched from it.

### O4 — Remove temporary image files from stage handoffs [COMPLETED]

- Add an in-memory RGB image/crop view shared by cleanup, detection, and
  recognizers; retain file APIs as load-and-forward wrappers.
- Make cleanup output ownership explicit and avoid unnecessary copies.

**Started:** `ocr_detect::detect_rgb` and `ocr_pipeline::run_raw` now accept
borrowed interleaved pixels; file APIs forward through them. The orchestrator
cleanup handoff still uses a temporary PNG and is the next O4 slice.

**Done when:** cleanup → detection/recognition runs without creating
`/tmp/crispembed_ocr_*.png`, with CPU/Metal output parity.

### O5 — Make capabilities and failures explicit [COMPLETED]

- Add an OCR capability query for loaded engines, languages, output types, and
  structure stages.
- Validate incompatible requests before inference; use stable errors instead of
  silent empty structure results.
- Add image dimension/pixel guards and per-item batch error isolation.

**Started:** enabling table/formula routing now fails at initialization unless
the required layout and specialized GGUF backends are configured.

**Done when:** every advertised feature is executable or rejected with a stable,
test-covered reason.

### O6 — Add reusable pipeline pooling and batch execution [COMPLETED]

- Define a bounded OCR pipeline pool for server use; retain the current path for
  single-threaded and WASM builds.
- Batch compatible crop recognition, cap batch size, and isolate bad inputs.
- Add queue/deadline metrics before changing defaults.

**Started:** DBNet+TrOCR inference contexts now serialize mutable decoder state
with an internal mutex, preventing concurrent callers from corrupting KV/cache
state. `ocr_pipeline_pool` now provides bounded isolated contexts with blocking
slot acquisition. The basic C OCR API selects the pool size from
`CRISPEMBED_OCR_POOL_SIZE` (default `1`); server-level queue/deadline metrics
remain a follow-up operational enhancement.

**Done when:** concurrent requests do not share mutable decoder state and batch
  throughput improves without changing decoded text.

### O7 — Establish unified accuracy/performance gates [COMPLETED]

- Add fixtures for receipt, form, dense page, screenshot, photo, table, and
  formula workloads.
- Measure CER/WER or exact-match, region recall, structure accuracy, p50/p95
  latency, memory, and batch throughput.
- Add regression thresholds and decoded-output checks for optimizations.

**Started:** `tests/ocr_benchmark.py` runs the real detector and pipeline test
binaries and reports region counts, decoded regions, and stage timings as text
or JSON. It uses local GGUFs and does not download models implicitly.

**Done when:** one reproducible command reports OCR quality and cost, suitable
for CI. **Complete:** `tests/ocr_benchmark.py` provides this command and JSON
output.

### O8 — Make corpus provenance and real-world coverage explicit [IN PROGRESS]

- Keep deterministic/reference fixtures for unit and per-stage parity checks,
  but do not use them as claims about real-world OCR quality.
- Add at least one public-domain/CC0 input for every production stage: text
  detection/recognition, layout, tables, cleanup, orientation, handwriting,
  multilingual routing, super-resolution, PDF routing, formulas, and OMR.
- Record source page, license, URL, SHA-256, and annotation status for every
  vendored asset. Add derived rotation/skew variants only from public-domain
  inputs.
- Acquire larger CC0 receipt and Arabic document sets separately, with a
  documented acceptance/download step instead of silently bundling them.

**Started:** `tests/regression/cc0_sources.json`, the fetched
`tests/regression/images/cc0/` seed set, and `corpus_manifest.json` now cover
receipts, forms, tables, Arabic printed/handwritten text, handwriting, cleanup,
orientation, layout, specialist lanes, and a dedicated German lane (modern
photo document, historical German print, and Kurrent handwriting). Gold
transcription review remains open for these robustness fixtures.

### O9 — Benchmark every available engine on shared inputs [IN PROGRESS]

- Use the checked-in regression manifest to enumerate every engine with a
  sample and local GGUF; report missing samples/models as explicit skips.
- Record cold load and warm inference time, return status, output excerpt, and
  CER/exact match when a gold transcription exists.
- Keep full-page VLM, ordinary OCR, math, and OMR scores separate; do not
  compare specialist outputs as plain-text OCR.
- Run the same engine sweep on the public-domain corpus after human gold
  annotations land, then add per-engine quality/latency thresholds.
- Maintain a complete matrix for model-backed engines even when a GGUF is not
  cached; the benchmark can fetch the manifest-pinned artifact with
  `--download-missing`.

**Started:** `tests/ocr_engine_benchmark.py` completed the local M1 Metal
sweep: 11 engines completed, 2 timed out/errored, and the remaining entries
were explicitly reported as missing samples, missing models, or model-needed
ports. SmolDocling is live-tested; Tesseract-LSTM is measured through DBNet
line crops; Unlimited-OCR is being fetched for its live run. Tesseract-LSTM
and PARSeq are present as recognizer-only rows; the DBNet+TrOCR document
baseline is measured separately by `tests/ocr_benchmark.py`. Results are
written as JSON with no silent omission. Unlimited-OCR subsequently completed
on M1 Metal from the system volume in 45,967 ms with correct two-region text
output; its GGUF was restored to the backup volume afterward.

The checked-in matrix now has a model-free CI coverage guard at
`tests/regression/test_engine_matrix.py`: all 23 portfolio engines must remain
present with a lane, runtime, fixture, and explicit availability status.

### O10 — Preprocessing inventory, parity, and live outcome gates [IN PROGRESS]

The OCR front-end needs its own measured regression track. Our restoration
inventory is broader than the lightweight OCR reference pipelines, but we are
missing several inexpensive geometry/orientation safeguards that often matter
more than another restoration model.

#### Existing CrispEmbed preprocessing

- Classical scan cleanup: dual-detector deskew consensus, border/content crop,
  background whitening, Otsu/Sauvola binarization, and fast binary morphology.
- Page analysis: PDF effective-DPI profiling, page split detection, content
  bounding box detection, source-type classification, and classical dewarp.
- Orientation: heuristic 0°/180° text-crop correction and rotated detection
  boxes; no learned page-orientation model yet.
- Learned restoration: NAFNet denoise, SCUNet, Restormer, InstructIR, and
  AdaIR.
- Super-resolution: PAN, TBSRN, HAT, DAT, ESRGAN, SwinIR, and SAFMN.
- Learned/classical dewarp: TPS dewarp and the classical baseline.
- VLM policy: full-page VLMs skip destructive scan cleanup and perform their
  own letterboxing/resizing; variable-resolution VLMs honor the max-pixels
  budget.

#### Reference capabilities to reproduce or explicitly reject

- Detector geometry: configurable minimum/maximum side limits, short-side
  target sizing, minimum-height padding, wide/short-image padding, and
  aspect-ratio-preserving letterbox policy.
- DB postprocessing: configurable segmentation threshold, box threshold,
  unclip ratio, optional dilation, candidate cap, and fast/accurate score mode.
- Line orientation: a dedicated 0°/180° classifier, confidence threshold, and
  an explicit all-lines mode for mixed-orientation documents.
- Page orientation: a learned 0°/90°/180°/270° classifier for PDF pages and
  photographed pages.
- Crop preparation: one shared policy for classifier geometry, recognizer
  geometry, aspect-preserving padding, and full-resolution recognition crops.
- PDF ingestion: native page rendering, page-image rotation, worker-pool
  accumulation, and the same preprocessing/OCR path as image inputs.
- Operational controls: per-stage enable/disable flags, hard errors for
  unavailable optional stages, request deadlines, and stage-level metrics.

#### Implementation slices

1. **O10.1 — Live preprocessor benchmark harness.** Add
   `tests/ocr_preprocessor_benchmark.py`. For every real CC0/German fixture,
   run raw input, classical cleanup variants, deskew, binarization, dewarp,
   denoise, and every locally available SR/restoration model. Record stage
   latency, output dimensions, pixel statistics, detector regions, OCR text,
   confidence, and CER/exact match when gold text exists. Also report text
   delta versus the raw-image baseline when no verified gold transcription is
   available. Synthetic degradations remain unit stress tests, not quality
   claims.

2. **O10.2 — Problematic-input corpus.** Extend the public-domain corpus with
   verified derived variants: ±4°/±8° skew, dark border, uneven illumination,
   haze, speckle, low-DPI downsample, JPEG damage, 90°/180°/270° rotation,
   perspective/curved-page distortion, and mixed upright/upside-down lines.
   Every derived file must retain its parent SHA-256 and transformation recipe.

3. **O10.3 — Detector geometry policy.** Add a shared configuration object and
   C API fields for `min_side_len`, `max_side_len`, `min_height`,
   `width_height_ratio`, padding mode, `unclip_ratio`, dilation, score mode,
   and candidate cap. Default to safe current behavior; expose compatibility
   presets for short text strips, wide receipts, dense scans, and photos.

4. **O10.4 — Learned line orientation.** Port a small permissively licensed
   0°/180° line-angle classifier to GGUF/ggml. Integrate it after detection
   and before every line recognizer, including Tesseract-LSTM crops. Retain
   the current heuristic as a no-model fallback. Add per-line angle,
   confidence, and whether a rotation was applied to structured results.

5. **O10.5 — Learned page orientation.** Port a small four-way page-orientation
   model. Apply it before PDF/image routing only when confidence clears a
   configurable threshold. Never rotate VLM inputs implicitly unless the
   caller enables the option, because VLM letterboxing is model-specific.

6. **O10.6 — Shared crop preprocessing.** Consolidate classifier and
   recognizer crop resizing/padding into one tested module. Support
   aspect-preserving and stretch modes, fixed height, maximum width, and
   grayscale/RGB contracts. Add parity fixtures for short, tall, wide,
   upside-down, and tightly clipped lines.

7. **O10.7 — PDF render/autorotate path.** Add native page rendering and
   page-level accumulation where the platform supports it. Reuse PDF DPI
   profiling to select render DPI, then apply page orientation and the normal
   document pipeline. Keep the existing parser-only path for minimal builds.

8. **O10.8 — Stage routing and safeguards.** Make preprocessing selection
   evidence-based: classical cleanup for scans, no destructive cleanup for
   VLMs/photos, denoise for noisy captures, SR only for low-DPI inputs, and
   orientation only above confidence thresholds. Add accept-gate comparisons
   so a preprocessor is rejected when it lowers confidence or worsens CER
   beyond the configured tolerance.

#### Required benchmark output

Each fixture/stage row must include:

- input and output dimensions, channels, and file checksum;
- cold load time, warm stage time, and peak/working-set estimate where
  available;
- detector box count, recognized region count, mean confidence, and text;
- gold CER/exact match when verified, otherwise raw-baseline text delta;
- `helped`, `neutral`, `harmed`, `unavailable`, or `error` classification;
- stderr tail and stable failure reason for model/backend failures.

#### Acceptance gates

- Every production preprocessor has at least one real CC0/German live fixture.
- Every problematic-input variant runs through raw plus all applicable stages.
- No default preprocessor may worsen verified CER beyond its configured gate.
- A stage that cannot run is reported explicitly; it is never silently skipped.
- Orientation, geometry, cleanup, and restoration effects are reported
  separately, so a strong recognizer cannot hide a harmful preprocessor.
- Results are reproducible from one command and committed as benchmark JSON;
  large GGUFs support the external-volume no-copy path via `UOCR_MMAP=1`.

### Validation follow-up — external document parser [COMPLETED]

- Unit gates passed: region router, pipeline pool, orchestrator (62/62), and
  render tests.
- Live M1 Metal gate passed: DBNet detected 10/10 fox fixture regions and
  TrOCR recognized 10/10; measured warm total was 5.0–5.3 s/image, with 8/10
  exact words and 6.1% CER.
- The comparison implementation's live execution is environment-blocked, not silently skipped: the
  CPU configure probe lacks OpenCV development files, while the production
  path requires CUDA/TensorRT and this host has no NVIDIA device/usable Docker
  daemon. The documented NVIDIA numbers are recorded in
  `PERFORMANCE.md` as reference claims only.
- Next actionable benchmark item: run both engines on a shared corpus on an
  NVIDIA host, then add detector/recognizer quality and throughput thresholds
  to `tests/ocr_benchmark.py`.
- Quantization A/B resolved the current fox errors: TrOCR-small-printed Q4_K
  produced 8/10 exact words, while the same ggml pipeline with the recommended
  Q8_0 model produced 10/10. Keep Q8_0 as the default quality model; do not
  treat Q4_K as a quality-preserving OCR quantization.
- Q8 is now the benchmark/WASM/example default. The pipeline rejects filenames
  identifying TrOCR Q4_K unless `CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1` is set.
  Text crops also receive a classical 0°/180° orientation check, and results
  now expose TrOCR mean/per-character confidence values.
- Added parity-facing structured output: deterministic reading-order indices
  and lightweight Markdown export are available from the orchestrator result
  and C API after each page run.
- Added modular server/API discovery: `/capabilities`, `/health/live`, and
  `/health/ready`; structured pipeline responses now include reading order and
  Markdown. Pipeline params and native server flags can independently enable
  layout, Tesseract-backed table cells, and PP-FormulaNet formulas.
- Added a `unified` pipeline stage backed by `crispembed_ocr_model_*`: any
  metadata-dispatched GGUF engine can now be selected as an escalation or
  specialist stage without adding another orchestrator-specific enum. This
  preserves the existing modular engine matrix, including Tesseract-LSTM,
  PARSeq, VLMs, math, and music engines where full-page/crop routing makes
  sense.

### Sequencing and boundaries

Land O1 first, then O2/O3 as the structured result and router foundation. O4 is
the first performance refactor; O5/O6 apply mainly to server builds. O7 starts
with CPU fixtures and expands to Metal/CUDA where hardware is available. Do not
replace ggml with TensorRT or make the core runtime NVIDIA-only.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.11)

```
Input text / image / audio
    │
    ├─► Text ──► Tokenizer (WordPiece / SentencePiece / BPE)
    │              │
    │              ├─► Encoder path (BERT, XLM-R, MPNet, NomicBERT,
    │              │     ModernBERT, GTE v1.5, DeBERTa-v2, SPLADE)
    │              │     Token + Pos [+ Type] embeddings
    │              │     N × Transformer layer (LN → MHA → FFN → residual)
    │              │     Pooling (mean / CLS) + optional heads
    │              │     → dense / sparse / ColBERT / reranker output
    │              │
    │              ├─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │              │     Token embeddings + RoPE
    │              │     N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │              │     Last-token / mean pooling + L2 normalize
    │              │
    │              └─► LFM2 path (LFM2.5, lfm2_embed.cpp)
    │                    RMSNorm + GQA, 350M, BOS-only tokenization
    │                    → dense / ColBERT multi-vector output
    │
    ├─► Image ──► ViT path (SigLIP/CLIP: vit_embed.cpp)
    │               Conv2D patch embed → transformer → mean pool → L2
    │
    ├─► Image ──► BidirLM-Omni vision (bidirlm_vision.cpp)
    │               Qwen2VL ViT + patch merger + DeepStack
    │               → image_embeds spliced into decoder
    │
    ├─► Image ──► CNN path (cnn_embed.cpp)
    │               SCRFD/YuNet face detection (FPN + anchor decode + NMS)
    │               ArcFace/SFace/AuraFace face recognition
    │
    ├─► Audio ──► BidirLM-Omni audio (bidirlm_audio.cpp)
    │               crisp_audio Whisper-shape encoder → mean pool → 2048-d
    │
    ├─► Math  ──► DeiT encoder + TrOCR decoder (math_ocr.cpp)
    │               Printed math → LaTeX via ggml graph compute
    │
    ├─► Math  ──► HMER: DenseNet-121 + GRU attention (hmer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2016)
    │
    ├─► Math  ──► BTTR: DenseNet + Transformer decoder (bttr_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2014, 53% exact match)
    │
    ├─► Math  ──► PosFormer: BTTR + ARM coverage (posformer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME, improved over BTTR)
    │
    ├─► Math  ──► MixTex: Swin-Tiny + RoBERTa (mixtex_ocr.cpp)
    │               Chinese+English LaTeX OCR (25681 BPE vocab)
    │
    ├─► Math  ──► PP-FormulaNet-S: HGNetv2 + MBart (ppformulanet_ocr.cpp)
    │               57M params, 384×384 input
    │
    ├─► Math  ──► PP-FormulaNet-L: SAM-ViT + MBart (ppformulanet_l_ocr.cpp)
    │               181M params, 768×768 input
    │
    ├─► OCR   ──► DBNet + TrOCR pipeline (ocr_pipeline.cpp)
    │               Text detection → recognition → reading-order sort
    │
    ├─► OCR   ──► Surya-OCR-2 detector (surya_det.cpp)
    │               EfficientViT + SegFormer, 38M, 91 languages
    │
    ├─► OCR   ──► Qwen2.5-VL / Qwen2-VL (qwen2vl_ocr.cpp)
    │               VLM doc OCR; german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets
    │
    ├─► Layout ─► RT-DETRv2 docling-heron (layout_detect.cpp)
    │               ResNet-50 + deformable xattn, 17 document classes
    │
    ├─► OCR   ──► PARSeq scene text recognition (parseq_ocr.cpp)
    │               ViT + Transformer, 24M, 94-char ASCII, Apache-2.0
    │
    ├─► OCR   ──► InternVL2 (internvl2_ocr.cpp)
    │               InternViT + InternLM2.5 VLM, 1B/2B, MIT (+ H2OVL)
    │
    ├─► OCR   ──► GLM-OCR (glm_ocr.cpp)
    │               CogVLM2 + GLM-4, 0.9B, 8 languages, MIT
    │
    ├─► OCR   ──► GOT-OCR2 (got_ocr.cpp)
    │               SAM ViT-B + Qwen2-0.5B, document+math+table, Apache-2.0
    │
    ├─► OCR   ──► LightOnOCR-2-1B (lightonocr.cpp)
    │               Pixtral ViT + Qwen3, 1B, OCR Arena #2, Apache-2.0
    │
    ├─► OCR   ──► DeepSeek-OCR-2 (deepseek_ocr2.cpp)
    │               SAM ViT + Qwen2 + MoE decoder, 3.4B, multilingual
    │
    ├─► OCR   ──► Granite Vision 3.3-2B (granite_vision_ocr.cpp)
    │               SigLIP2 + Granite-3.1-2B, OCRBench 852, Apache-2.0
    │
    ├─► OCR   ──► Tesseract LSTM (tesseract_lstm.cpp)
    │               DBNet detection + per-line LSTM, 126 languages
    │
    ├─► NER   ──► BERT/XLM-R token classification (bert_ner.cpp)
    │               Fixed-label NER: PER/LOC/ORG/MISC, auto-detected
    │
    ├─► NER   ──► GLiNER zero-shot (gliner_ner.cpp)
    │               LFM2.5/DeBERTa-v3 + BiLSTM + span matching
    │
    ├─► KIE   ──► OCR + NER pipeline (kie_pipeline.cpp)
    │               Phase 1: OCR→NER. Phase 2: LiLT layout-aware
    │
    ├─► KIE   ──► LiLT layout transformer (lilt_kie.cpp)
    │               Dual-stream RoBERTa + BiACM, 130M, FUNSD, MIT
    │
    ├─► LID   ──► Text language identification (crisp_lid)
    │               CLD3 / GlotLID, Tesseract auto-select
    │
    ├─► Table ──► Rule-based table structure (table_parse.cpp)
    │               Line detection + grid + cell OCR → HTML
    │
    ├─► OCR   ──► PaddleOCR-VL (qwen2vl_ocr.cpp) — DONE
    │               NaViT ViT + ERNIE-4.5-0.3B, 109 langs, Apache-2.0
    │               OmniDocBench SOTA 96.3% (1.6) / 0.9B variant
    │
    ├─► Math  ──► Uni-MuMER-Qwen3-VL-2B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 2.1B, Apache-2.0, 82% CROHME
    │
    ├─► Math  ──► Uni-MuMER-Qwen2.5-VL-3B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 3.4B, Apache-2.0, 82.25% CROHME
    │
    │   ── PLANNED ──
    │
    └─► OCR   ──► SmolDocling (256M, Apache-2.0) — DONE: SigLIP + SmolLM2, DocTags
                    Idefics3/SmolVLM, IBM Research, DocTags output (tiny, EN-only)
```

(Evaluated and **rejected** for licensing: dots.ocr — supplemental PRC
agreement (rednote/Xiaohongshu), not pure MIT; MinerU2.5-Pro — commercial
thresholds + gated HF; Hunyuan-OCR — custom Tencent license, excludes
EU/UK/South Korea. See the next-gen table below.)

## Supported architectures (v0.11)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| NomicBERT MoE encoder | SentencePiece | Post-LN, MoE 8-expert top-2, GELU, RoPE | nomic-embed-text-v2-moe |
| ModernBERT encoder | BPE | Pre-LN, GeGLU, RoPE, per-layer theta | gte-modernbert-base |
| GTE v1.5 encoder | WordPiece | Post-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | WordPiece | Post-LN, c2p/p2c disentangled attn | mxbai-rerank-xsmall/base-v1 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M, EmbeddingGemma-300m |
| LFM2 (bidirectional) | GPT-2 BPE | Pre-norm RMSNorm, GQA, RoPE, BOS-only | LFM2.5-Embedding-350M, LFM2.5-ColBERT |
| BidirLM-Omni | GPT-2 BPE | Bidirectional Qwen3, MRoPE, DeepStack | BidirLM-Omni-2.5B |
| ViT (SigLIP/CLIP) | — | Conv2D patch embed, CLS/mean/attn pool | siglip-base, clip-vit-base |
| CLIP text | CLIP BPE | Pre-LN, causal mask, EOS pool | clip-text-base/large |
| CNN (SCRFD/YuNet) | — | FPN, anchor decode, NMS | scrfd-det-10g, yunet |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |
| PosFormer | — | DenseNet + Transformer + ARM | posformer (handwritten math) |
| MixTex | BPE (25681) | Swin-Tiny + RoBERTa 4L decoder | mixtex (CN+EN LaTeX) |
| PP-FormulaNet-S | BPE (50000) | HGNetv2 CNN + MBart 2L decoder | ppformulanet (57M) |
| PP-FormulaNet-L | BPE (50000) | SAM-ViT + MBart 8L decoder | ppformulanet-l (181M) |
| DBNet | — | ResNet-18 + FPN + DB head | text detection (12M) |
| Surya-Det | — | EfficientViT + SegFormer | surya-ocr-2 detector (38M, 91 langs) |
| RT-DETRv2 | — | ResNet-50 + deformable xattn | layout-heron (17 classes) |
| Qwen2.5-VL / Qwen2-VL / Qwen3-VL | tiktoken | ViT-32L + spatial merger + Qwen LLM; runtime ne-fix for transposed-weight GGUFs | german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets, PaddleOCR-VL |
| InternVL2 | tiktoken | InternViT + InternLM2.5 LLM | internvl2-1b/2b, H2OVL |
| GLM-OCR | BPE | CogVLM2 + GLM-4 decoder | glm-edge-ocr (0.9B) |
| GOT-OCR2 | BPE | SAM ViT-B + Qwen2-0.5B | got-ocr2 (0.7B) |
| LightOnOCR | tiktoken | Pixtral ViT + Qwen3 decoder | lightonocr-2-1b (1B) |
| DeepSeek-OCR-2 | tiktoken | SAM ViT + Qwen2 + MoE decoder | deepseek-ocr2 (3.4B) |
| Granite Vision | tiktoken/BPE | SigLIP2 ViT + Granite-3.1 LLM | granite-vision-3.3-2b |
| PARSeq | — | ViT + AR/NAR Transformer | parseq (24M, 94-char) |
| Tesseract LSTM | — | DBNet det + LSTM line rec | 126 languages |
| LiLT | RoBERTa BPE | RoBERTa + layout transformer + BiACM | lilt-funsd (130M) |
| BERT NER | WordPiece/SP | BERT/XLM-R + Linear classifier | bert-ner, xlmr-ner-hrl |
| Table parser | — | Rule-based morphology + grid detection | table_parse (no model) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |
| crisp_punc | CrispASR/crisp_punc/ | shared library (FireRedPunc + PCS) |
| crisp_lid | CrispASR/crisp_lid/ | shared library (CLD3 + GlotLID) |
| crisp_truecase | CrispASR/crisp_truecase/ | shared library (stat + CRF + BiLSTM) |

## File layout (current)

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md                     architecture + roadmap (this file)
├── HISTORY.md                  completed milestones
├── LEARNINGS.md                technical notes
├── PERFORMANCE.md              benchmarks
├── ggml/                       (submodule)
├── src/
│   ├── crispembed.{h,cpp}      C API + encoder graph + OCR-model dispatch
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── lfm2_embed.cpp          LFM2.5 dense + ColBERT multi-vector
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr / bttr_ocr / posformer_ocr / mixtex_ocr / ppformulanet*  math OCR
│   ├── qwen2vl_ocr / internvl2_ocr / glm_ocr / got_ocr / lightonocr      VLM OCR
│   ├── deepseek_ocr2 / granite_vision_ocr / parseq_ocr / tesseract_lstm  OCR engines
│   ├── tokenizer*.{h,cpp}      WordPiece + SentencePiece + BPE
│   └── core/                   shared helpers (gguf_loader, bpe, mel, cpu_ops)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/                     GGUF conversion scripts
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending work

Only genuinely-open, in-progress, or reference material lives below. **Completed
milestones — the imatrix quant rollout (C1), batched-encoder throughput (C3),
prefix KV cache (C4), mtmd-preprocessing port (C5), flash-attn epilogue audit
(C6), mmproj interop, the June-2026 optimization-TODO sweep, per-backend perf
passes, the SR conv→ggml sweep, the regression-guardrail closure, the CUDA
device-pointer fixes, and the scan_cleanup / unpaper feature ports — have moved
to `HISTORY.md`** (deep technical notes in `LEARNINGS.md`). Before starting any
item: read LEARNINGS "measure the DOMINANT cost before fixing a flagged
micro-gap" and "the build dir was silently CPU-only"; verify
`GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`; check `git worktree list` +
`git log main..<branch>` for a concurrent session's finished work; all edits in
a worktree (ggml symlink dance, see CLAUDE.md).

### Shipped ecosystem-compat work (A1–A4, modern-bert, A3 parity harness) — see HISTORY.md

The A1–A4 JSON/hparam/CI hardening, the community `modern-bert` BPE-tokenizer
fix, the encoder ground-truth parity harness, and the e5-small/granite matrix
closure all shipped 2026-07-16 (verified in code 2026-07-20). Details + preserved
specifics: HISTORY.md "July 20, 2026" + "July 16, 2026"; deep-dives in LEARNINGS.md.

### Community-matrix coverage roadmap — candidate archs to add

Matrix entries (10): bge-small + all-MiniLM (`bert`, split-QKV/abs-pos/WordPiece),
nomic-v1.5 (`nomic-bert`), nomic-v2-moe (`nomic-bert-moe`), gte-modernbert
(`modern-bert`, gpt2-BPE), granite-107m (`bert` + `t5`/SPM, CLS), gte-base-en-v1.5
(`NewModel` tanh-GeGLU), Qwen3-Embedding-0.6B (`qwen3` decoder), LFM2.5-Embedding
(`lfm2`), embeddinggemma-300m-qat (`gemma-embedding`). Each remaining family below
exercises a DISTINCT loader/graph path not yet guarded against a
third-party GGUF. Ordered by coverage value; every one is a load + shape +
garbage-guard + HF per-stage entry (the granite recipe), each MUST be gated on the
per-stage structural cosine (a garbage-guard-only pass hides an e5-style shift).
Availability probed 2026-07-16 (repos listed are candidates, not yet validated):

| Candidate | arch / path it covers | Fits dense driver? | Candidate community GGUF | Watch-out |
|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | `qwen3` DECODER embed — last-token pool, **causal**, gpt2-BPE decoder path (distinct from modern-bert's ENCODER BPE) | ✅ (last-token) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (official) + many | **ADDED + validated (2026-07-16), CLEAN — no loader change.** decoder_embed.cpp already takes blk.N.* + the gpt2-BPE KV-merges path is handled. Final cosine vs HF: q8 mean 0.999727, **f16 mean 1.000000** (graph exact); garbage margin 0.58 |
| **EmbeddingGemma-300m** | `gemma-embedding` — mean pool, **Dense bottleneck + Matryoshka** projection | ✅ (mean) | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`, `unsloth/…`, `lmstudio-community/…` | **ADDED + SHIPPED (2026-07-17, `138ee0c`).** Real bug was the tokenizer (SPM loaded as char-level BPE), not Dense/norm; arch-gated routing to `decoder_embed.cpp` + SPM-BPE bigram-merge mode + Dense baked via `models/add-st-dense-to-gguf.py`. HF-full parity **0.985**; registry `embeddinggemma-300m-qat`; matrix entry (HF gate), 10/10 PASS; HF `cstr/embeddinggemma-300m-GGUF`. Full write-up: HISTORY.md 2026-07-17; deep-dive: LEARNINGS.md "Community `gemma-embedding`". |
| **LFM2.5-Embedding-350M** | `lfm2` bidirectional hybrid — ShortConv + attention, **BOS-only wrap** | ✅ (CLS, pooling_type=2) | `LiquidAI/LFM2.5-Embedding-350M-GGUF` (official) | **FIXED + SHIPPED (2026-07-16), added to matrix.** Was a loader gap — `lfm2_embed` requires our `lfm.*` tensor names + `lfm2.<our>` hparam keys + a `lfm2.layer_types` c/a string; the official llama.cpp export uses `blk.N.*` + canonical `lfm2.*` keys + no layer-types string. Same class as modern-bert (alias gap), bigger. **Complete fix recipe (exact tensor + hparam maps, layer-type-from-tensor-presence, per-stage gate) in the "FOUND (2026-07-16): official `lfm2`…" subsection just below.** GGUFs already downloaded. Needs a quiet box for the build + `test-lfm2-diff` per-stage validation |
| **GTE-v1.5 (gte-base-en-v1.5)** | `NewModel` NTK-RoPE + GeGLU **tanh** (the path the modern-bert `geglu_erf` gate was explicitly kept OFF for) | ✅ | `cstr/gte-base-en-v1.5-GGUF` (our own; llama.cpp ❌ so third-party rare) | **ADDED + validated per-stage (2026-07-16).** q8 vs HF fp32: emb_ln_out gate 0.999927, all layers PASS (encoder_out 0.9926). Guards the tanh-GeGLU branch stays correct next to modern-bert's erf branch. Arch coverage (own GGUF), not ecosystem-compat |
| **MPNet (all-mpnet-base-v2)** | MPNet two-stream / T5-style rel-attn bias — **we are unique** | ✅ | `cstr/all-mpnet-base-v2-GGUF` (our own; no third-party — llama.cpp ❌) | **ADDED (2026-07-20)** — matrix guard `all-mpnet-base-v2`; HF final-cos 0.997 realistic / 0.987 short (f32==q8_0 → structural residual, not quant); guards the unique rel-attn-bias graph. Arch coverage, not ecosystem-compat |
| **XLM-R-large / multilingual-e5-large** | `bert`+SPM XLM-R at 1024-dim | ✅ | `soichisumi/…-Q8_0-GGUF`, `phate334/…`, `walsons/…` | **EXPECT the e5-small position-offset FAILURE** (XLM-R needs offset 2; community `bert`-arch GGUFs omit `position_offset`). Add ONLY if a community GGUF declares the offset — else it documents the same known gap |
| **SPLADE-v3 (sparse)** | MLM/sparse head — `has_sparse` path, NOT dense | sparse metric (sparse-cos), not the garbage guard | `mradermacher/Splade-V3-GGUF` — **HEADLESS, unusable (2026-07-20)** | **Driver already does SPLADE** (CLI `--sparse`, `crispembed_encode_sparse`, `splade-pp-en-v1` ships at sparse-cos 0.996, `audit_gguf_heads` guards the head through quant). The COMMUNITY GGUF can't be supported: `mradermacher/Splade-V3-GGUF` (arch `bert`, 197 tensors, inspected) has **NO `cls.predictions.*`/MLM head** — llama.cpp drops it at convert, so it loads as a plain dense encoder (same class as e5-small/EmbeddingGemma "community export drops the head"; no loader alias recovers an absent tensor). Only OUR converter (`convert-bert --crisp`, reads checkpoint files) keeps the head. **`naver/splade-v3` ADDED + SHIPPED (2026-07-20):** `convert-bert --crisp` (MLM head detected+kept) → f16/q8_0/iq4_xs+imatrix, sparse-cos vs HF **1.0000 / 1.0000 / 0.9971**; HF `cstr/splade-v3-GGUF` (CC-BY-NC-SA-4.0 card + attribution); registry `splade-v3`/`splade-v3-q8` (NC → `--accept-license`), `upload_to_hf.py` + `audit_gguf_heads` entries. |
| **DeBERTa-v2** | disentangled c2p/p2c rel-attn (`rel_embd`, `position_buckets`) — **we are unique**, highest-complexity encoder path | ✅ | **none found** (llama.cpp ❌, no community GGUF exists) | Blocked on the absence of any third-party GGUF; only our own conversion exists |

Status: **Qwen3-Embedding, LFM2.5-Embedding, granite-107m, GTE-v1.5, EmbeddingGemma,
and MPNet all ADDED**; **e5-small CLOSED** (under-specified export). **Genuinely
remaining candidates:** **XLM-R-large** expected to reproduce the e5 offset gap
(add only as a documented negative, or if a community GGUF declares
`position_offset`); **DeBERTa-v2** blocked on GGUF availability (no third-party
export exists). Do each
on a quiet box (250K-vocab SPM reads + HF forwards are slow under contention) and
gate on the per-stage structural cosine. **SPLADE-v3 is NOT a remaining driver
gap** — sparse retrieval ships (`splade-pp-en-v1`, sparse-cos 0.996); the community
`Splade-V3-GGUF` is headless (documented above), so only an optional converter-add
of our own `naver/splade-v3` GGUF remains.

### Transcoda OMR decode enhancements (deferred, 2026-07-13)

The shipped `transcoda_ocr` engine uses greedy decode (byte-identical to the HF
reference; persistent device-KV, 2.4–4×). The paper's two higher-accuracy decode
modes are **deferred** — both are large, and neither is byte-exactly validatable,
so they were intentionally NOT shipped (byte-exact-or-bust discipline). Concrete
plans for a follow-up session:

- **Beam search (width 3)** — the paper's headline (OMR-NED 18.46% vs greedy
  ~higher on Verovio-synth). HF config: `num_beams=3, length_penalty=1.0,
  repetition_penalty=1.1, early_stopping=True`.
  - *Where*: a `decode_beam(ctx, n_beams)` in `src/transcoda_ocr.cpp`, gated
    `TRANSCODA_OCR_NUM_BEAMS=N` (opt-in; greedy stays the default). Per-beam
    next-token logits via either B independent persistent KV caches (extend
    `pk_*` to a `[..., B]` beam dim) or the full-recompute `run_decoder` per beam
    (simplest, O(B·L²) — fine for opt-in).
  - *Algorithm* (mirror HF `BeamSearchScorer`): keep B live beams (init scores
    `[0,-inf,-inf]`), each step apply per-unique-token rep-penalty + `log_softmax`,
    add to beam score, take top-`2B` over the flattened `B×vocab`, route eos
    candidates to a finished pool with score `/(len**length_penalty)`, keep the
    top-B non-eos as the next beams; early-stop when B finished hypotheses exist;
    return the best finished (or best live) hypothesis.
  - *Validation*: (1) on the confident synth page `sample_page.png`, HF beam-3 ==
    greedy, so mine must be **byte-exact == greedy** there (a real regression
    gate); (2) on a real Polish scan (`btrkeks/polish-scores`, license "other" —
    LOCAL validation only, do NOT commit the image), HF beam-3 diverges from
    greedy at accent/ornament tokens (`16b#JJ`→`16bJJ`) and spine markers
    (`*^`/`*v`) — target **CER-close** to the HF beam-3 dump (byte-exact over a
    512-token uncapped scan is not realistically achievable; cascading). HF
    references already captured: `scratch-transcoda/oracle_beam3.kern.txt`,
    `polish_beam3.kern.txt`.

- **Grammar-constrained decode** — guarantees structurally-valid `**kern`
  (paper's `grammars/kern.gbnf` via xgrammar logits processors). Large: needs a
  GBNF parser + a per-step token-mask constraint engine (llama.cpp's
  `llama-grammar` is the reference, ~1k LOC). *Where*: a `kern_grammar.{h,cpp}`
  constraint module + a mask hook in the decode loop, gated
  `TRANSCODA_OCR_GRAMMAR=1`. *Validation*: structural only (every output parses as
  valid kern); no byte-exact HF target (xgrammar's tie-breaking differs). Lowest
  priority — greedy already emits valid kern on clean inputs.

### Optical Music Recognition (OMR) — models to port (2026-07-12)

OMR is "OCR for staff notation": the winning modern approach is exactly the
TexTeller shape — vision encoder + autoregressive transformer decoder emitting
a linearized notation token sequence. This reuses the existing
VisionEncoderDecoder machinery (`math_ocr.cpp` path). Output format is
irrelevant to us (bekern / **kern / MusicXML / LilyPond are all parseable
downstream), so we optimize for arch fit + license, not output dialect.

**Two distinct problems:** printed staff notation (tractable, MIT weights
exist) and handwritten (hard; the real license risk is on the *training
data*, not the code — see landmine below).

**Licensing methodology — AGPL *code* is NOT a blocker (verified 2026-07-13).**
The gate is the **weights** license (we redistribute GGUFs) and the **engine
authorship**, which are independent of an upstream repo's *code* license:
- If the **weights** are permissive (MIT / Apache / **CC BY**), the GGUF is
  redistributable regardless of the training-code license. AGPL/GPL on the
  *code* does not attach to CC-BY *weights*. (Training-data license only matters
  if we redistribute the data or retrain — not for shipping pretrained weights.)
- The **engine** is written **clean-room**: run the upstream Python as an
  *oracle* (reference-activation dumps — no derivative) and implement from a
  **facts-only spec** (architecture, tensor shapes, op order, hparams, eps/scale
  — all uncopyrightable) + the paper + configs. Never transcribe AGPL source
  line-by-line. Two-team wall: the brief-writer may read the AGPL `.py`; the
  implementer sees only the facts brief. (Permissive blueprints don't need this.)
- Hard rejects shrink to: **gated / unlicensed / non-permissive weights**, or an
  **11B+ base under a restrictive model license**, or a **non-single-model
  pipeline** (poor ggml fit).

| # | Model | Params | License (code / weights) | Architecture | Output | Handles | Effort | Status |
|---|-------|--------|--------------------------|-------------|--------|---------|--------|--------|
| 1 | **Sheet Music Transformer (SMT)** | 21.4M | **MIT / MIT** | ConvNext encoder + Transformer decoder | bekern | Printed polyphonic | Low | **DONE** — `src/smt_ocr.cpp` shipped (per-stage cos 1.0, 96.3% GrandStaff) |
| 1b | **SMT++ full-page** | ~10.9M | **MIT / MIT** — `PRAIG/smt-fp-grandstaff` (public, **not gated**, verified HF card) | full-page extension of SMT (curriculum-trained) | bekern | **Full-page pianoform** (no separate layout stage) | **Low–Med** | **DOABLE — top permissive target.** Verify arch delta vs base SMT first: deep-research *refuted* (2-1) the "same-arch, curriculum-only" claim, so confirm the graph before assuming free reuse. If same graph → near-free extension of shipped SMT |
| 2 | **Transcoda-59M-zeroshot** | 58.8M | **AGPL code / CC BY 4.0 weights** (`btrkeks/transcoda-59M-zeroshot-v1`, verified HF card) | ConvNeXt-V2-Tiny enc + 8L Transformer dec (d512/8h, **RoPE**) | **kern | **Full-page + historical scans** (zero-shot); **current OMR-NED SOTA** (Polish 63.97%, Verovio 18.46% — beats SMT++ & Legato) | **Med** | **DOABLE — accuracy leader.** Weights CC BY 4.0 → GGUF redistribution clean (attribute). Engine **clean-room** (code is AGPL). Arch fully in-tree: ConvNeXt-V2 ≈ SMT's ConvNext, RoPE decoder ≈ Qwen3; add 3000-token kern BPE tok; optional GBNF grammar-constrained decode. Training data `polish-scores` = `license: other` (irrelevant to CC-BY weight redistribution) |
| 3 | Polyphonic-TrOMR (NetEase) | ~22M | **Apache-2.0 / Apache-2.0** | ViT + multi-head Transformer decoder (rhythm/pitch/lift/note) | symbolic text | Printed polyphonic photos | Medium | **DONE** — `src/tromr_ocr.cpp` (cos 1.0 / 100% argmax / byte-exact); `cstr/tromr-GGUF` |
| 4 | **Flova/omr_transformer** | 143M | Apache-2.0 / Apache-2.0 | Donut VED (DonutSwin + 4L mBART) | LilyPond | artificial + **handwritten** + whiteboard (monophonic) | Medium | **DONE** — `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact); `cstr/flova-omr-GGUF` (f32 + q8_0); CLI + registry wired |
| 5 | oemer | 2× U-Net | MIT / MIT | 2 segmentation U-Nets + numpy reconstruction | MusicXML | Printed, photos, skewed | High | Reference-only — multi-model + rule-based reconstruction, poor ggml fit |
| ~~6~~ | ~~Legato~~ | ~11B | MIT (trained delta) / **Llama-3.2 license + GATED** | frozen Llama-3.2-11B-Vision + trained decoder | ABC | full-page | — | **REJECTED** — 11B base under Meta's Llama license + contact-gated weights; MIT covers only the delta. Too big + non-permissive base |
| ~~7~~ | ~~starry / FindLab~~ | — | **no code license / gated, unlicensed weights** | 7-microservice pipeline (PyTorch+TF+ONNX) | LilyPond/kern | complex polyphonic | — | **REJECTED** — not a single model (poor ggml fit) *and* weights token-gated with no stated license |
| ~~8~~ | ~~Clarity-OMR~~ | — | (unverified) | PDF→MusicXML **pipeline** | MusicXML | printed | High | Reference-only — multi-stage pipeline, not a single VED model |
| ~~9~~ | ~~homr (liebharc)~~ | — | **AGPL-3.0** (code) | pipeline + TrOMR | MusicXML | printed/camera | — | **REJECTED** — pipeline (poor ggml fit); the underlying TrOMR is already shipped separately (Apache-2.0) |

**Recommended priority (updated 2026-07-20 — SMT/SMT++/TrOMR/Flova/Transcoda ALL shipped):**

1. ~~**SMT++ full-page**~~ **SHIPPED** — `smt_ocr.cpp` (arch reuse confirmed) +
   registry `smt-fp`; HF `cstr/smt-fp-grandstaff-GGUF`. (Was the "best next step";
   done.)
2. ~~**Transcoda-59M**~~ **SHIPPED** — clean-room `src/transcoda_ocr.cpp` +
   registry `transcoda`; HF `cstr/transcoda-omr-GGUF` (CC-BY-4.0). The only genuine
   remaining Transcoda work is the *optional* beam-3 + GBNF `**kern`
   grammar-constrained decode (still greedy-only; see "Transcoda OMR decode
   enhancements" — the sole open OMR-engine lever).

3. **Handwritten *polyphonic* — the real remaining gap.** No permissive model
   fills it: Flova (shipped) is monophonic-toy; the strong performers are all
   rejected (Legato = Llama-11B/gated, starry = gated/unlicensed pipeline, homr =
   AGPL pipeline). Reach it by *fine-tuning* a shipped graph (SMT or Transcoda)
   on synthetic + license-clean handwritten-style data — same engine, new weights.

3. **Polyphonic-TrOMR — DONE (2026-07-13).** Genuinely accurate model (reads
   clefs/keys/rhythms/pitches correctly on real photos). The ggml engine
   `src/tromr_ocr.cpp` (ResNetV2 SAME-pad backbone + hybrid ViT encoder →
   x-transformers 12-sublayer decoder with SIGLU attn-on-attn + GEGLU FF → 4
   parallel heads, autoregressive over rhythm/pitch/lift streams) is written,
   wired (dispatcher + CMake + `test-tromr-diff` + CLI `--ocr` auto-detect), and
   **validated CPU-only vs the reference model**: every diff-harness stage cos
   **1.0** (backbone, ViT context, all 12 decoder blocks, all 4 logit heads),
   **100% per-position argmax agreement** teacher-forced (66/66, 85/85), greedy
   decode **byte-exact** vs the authors' `examples/{1,2,3}.txt`, Metal == CPU.
   q8_0 also decodes byte-exact. ~~**Remaining:** HF upload `cstr/tromr-GGUF`
   (f32 + q8_0) + `model_mgr.cpp` registry entry.~~ **DONE** — `tromr` registry
   entry (model_mgr.cpp) points at `cstr/tromr-GGUF`.
   Corrections vs the (now-removed) handover brief found in validation: ViT scale
   is **32^-0.5** not 64^-0.5; the converter emitted tensor names >64 chars that
   the ggml loader rejects (`GGML_MAX_NAME`) → shortened the backbone prefix to
   `enc.bb`; the quantizer must keep `enc.bb`/`enc.proj` convs unquantized
   (flatten+quantize → reshape-to-4D abort). See LEARNINGS.md.
   Weights: `tromr/workspace/checkpoints/img2score_epoch47.pth` (86.3 MB)
   committed directly into the Apache-2.0 repo (not LFS → covered by the repo
   license), with a 4-file tokenizer set (`tokenizer_{lift,pitch,rhythm,note}.json`).
   Architectural wrinkle vs SMT: TrOMR is **not** a single autoregressive stream
   — it has *parallel classification heads* (rhythm / pitch / lift / note) per
   decoder timestep, so the port needs 4 output projections + a merge step, not
   one LM head. `homr` wraps this same model but is AGPL — weights taken from
   the NetEase repo, not homr.

**Reuse map (assessed 2026-07-12, feat/smt-omr worktree):** ~70% of the SMT
port reuses existing infra —
- **Decoder + decode loop + C ABI:** `src/math_ocr.cpp` is already SMT's exact
  shape ("Hybrid CNN + ViT encoder → cross-attention Transformer decoder → token
  sequence"): KV-cached decoder, greedy + beam decode, batched encode, per-token
  confidences. SMT's "classic Transformer decoder" == TrOCR == this; port by
  config, not new graph code.
- **Converter:** `models/convert-trocr-safetensors-to-gguf.py` already handles
  the decoder + top-level `decoder_start_token_id`. New `convert-smt-to-gguf.py`
  = that file + a ConvNext encoder tensor mapping.
- **ConvNext encoder (the one new piece):** CrispASR has ConvNeXt blocks in
  `f5_tts / vibevoice / qwen3_tts / kugelaudio / outetts_wavtok` (1-D/audio, but
  identical block: dwconv → LN → pwconv → GELU → pwconv → layer-scale → residual)
  + `core/activation.h`; CrispEmbed has mature `ggml_conv_2d` engines (`swinir`,
  `nafnet`, `cnn_embed`, `adair`, `tbsrn`) for the 2-D image side. Adapt, not
  invent.
- **Shared load/preproc/vocab:** math_ocr grayscale-resize-normalize;
  `core/{gguf_loader,cpu_ops,bpe}.h`; bekern = fixed lookup vocab (simpler than
  any in-tree BPE).
- New work = 2-D ConvNext encoder + bekern vocab + encoder-side converter.

**Confirmed SMT architecture (2026-07-12, from SMT++ source + safetensors header):**
Total **21.4M params, F32, 360 tensors, 85.5 MB** `model.safetensors`. Greedy
manual decode (no HF `.generate()`), seed `<bos>=4426`, stop `<eos>=8822`,
`pad=0`, up to `maxlen=1281` steps.
- **⚠ Convert against SMT++ tensor names, NOT SMT-main.** The shipped
  grandstaff/camera-grandstaff weights only match `SMT-plusplus/smt_model/
  modeling_smt.py` (`input_attention`/`cross_attention`/`ffNet`/`out_layer`); the
  SMT-main repo has a rewritten module whose names match no checkpoint.
  `smt-string-quartets` ships **no weights** (README only).
- **Encoder** = stock HF `ConvNextModel(num_channels=1, num_stages=3,
  hidden_sizes=[64,128,256], depths=[3,3,9])`. Plain ConvNeXt, no attention. Stem
  Conv2d(1→64,k4,s4)+LN; stage-1/2 downsample Conv2d(k2,s2); **16× H/W reduction**.
  Last stage already outputs 256 = `d_model`, so **no encoder→decoder projection**.
  `encoder.layernorm` (pooler LN) is in the ckpt but **dead** on the inference path
  (`last_hidden_state` is pre-pooler). Tensors:
  `encoder.embeddings.patch_embeddings.{weight[64,1,4,4],bias}`,
  `encoder.encoder.stages.{0,1,2}.layers.{i}.{dwconv,layernorm,pwconv1,pwconv2,layer_scale_parameter}`,
  `encoder.encoder.stages.{1,2}.downsampling_layer.{0=LN,1=Conv2d}`.
- **Decoder** = 8 layers, d_model=256, **4 heads** (hd=64), **FFN dim=256 (1×, not
  4×)**, activation **ReLU** (+ `end_relu` before the head). Post-norm:
  self-attn→norm1→cross-attn→norm2→FFN→norm3. Token emb `nn.Embedding[20578,256]`;
  **embeddings NOT tied** to head. LM head = `Conv1d(256→20578,k1)` →
  `decoder.out_layer.weight[20578,256,1]` (squeeze trailing 1 → Linear) + bias.
  Tensors: `decoder.embedding.weight`, `decoder.decoder.layers.{0..7}.
  {input_attention,cross_attention}.{lq,lk,lv,out_proj}.{weight,bias}`,
  `.ffNet.{0,3}.{weight,bias}`, `.{norm1,norm2,norm3}.{weight,bias}`.
- **Positional encodings are NOT in the checkpoint — bake as constants.** (a) 1-D
  sinusoidal added to decoder token embeddings; (b) 2-D sinusoidal
  (`dim=256`, first 128ch=row H, last 128ch=col W, `div=exp(-arange(0,dim//2,2)/dim·ln1e4)`).
- **⚠ Cross-attention key≠value:** encoder output flattened over H×W;
  the 2-D PE is added to the **KEYS only**; **VALUES are the raw** flattened
  features. Query = decoder states. Cross-attn has no mask; self-attn is causal.
- **Preprocessing:** grayscale, **always color-invert** (`RandomInvert(p=1.0)` —
  mandatory, not augmentation), `ToTensor` → **[0,1], NO mean/std normalize**.
  `cv2.resize` bilinear at `reduce_ratio=0.5`, height floored/capped ~256px
  (`maxh=256`, `maxw=3056`).
- **bekern vocab** = fixed word-level lookup (NOT BPE), `out_categories=20578`,
  identical across grandstaff/camera. `w2i`/`i2w` embedded in `config.json`
  (875 kB) and as `vocab/*.npy`. Split GT on whitespace/`·` delimiter; layout
  tokens `<b>` break / `<s>` space / `<t>` tab.
- **SMT vs SMT++:** identical neural graph; SMT++ gains are training-side
  (curriculum + synthetic full pages). Full-page = same graph, bigger images +
  longer decode + layout tokens, no extra module. **Target single-system
  grandstaff first** (the only checkpoints with published weights).

**Port progress (2026-07-12, feat/smt-omr worktree):**
- ✅ `models/convert-smt-to-gguf.py` — torch-free, verbatim SMT++ names, squeezes
  `out_layer` 1×1 conv→Linear, bakes 1-D decoder PE, records `smt.scale_attention=
  False`. Verified GGUF: arch `smt_ocr`, 361 tensors, 20578-tok vocab, 83 MB.
- ✅ `tools/dump_smt_reference.py` — loads REAL SMT++ model (hooks, not a
  re-forward), dumps 18 per-stage F32 tensors → `smt_ref.gguf`. Validated on a
  real GrandStaff test image: enc 336×128→`(256,8,21)` (16× reduction, 168 mem
  tokens), decode emits correct bekern (`**ekern_1.0 <t> … *clefG2 <b> …`).
  Test assets in scratchpad: `smt-grandstaff/`, `SMT-plusplus/` clone, `gs_test0.png`
  (+ `.gt.txt`), `smt_ref.gguf`. Note: cloned `SMTConfig` needs a
  `super().__init__(**kwargs)` patch to load under transformers 4.57.
- ✅ `src/smt_ocr.{h,cpp}` ggml engine (ConvNext encoder + cross-attn decoder +
  greedy decode) + `tests/test_smt_diff.cpp` + CMake wiring. **Full per-stage
  parity vs `smt_ref.gguf` (CPU):** enc_stage0/1/2 + enc_output + mem_key
  cos_min ≥ 0.999996; dec_tok_emb + dec_layer0–7 + logits cos_min = **1.000000**.
  Native greedy decode emits correct bekern (header/clefs/meter/barlines match
  GT exactly; `*k[]` vs GT `*k[b-]` is the model's own prediction — the Python
  ref emits `*k[]` too). Bugs found & fixed during bring-up: (a) off-trunk
  `enc_stageN` snapshots weren't in the graph (`to_tokens` forks off the trunk)
  → `ggml_build_forward_expand` each; (b) `crispembed_diff.h` GGUF reader only
  decodes F32 (its I32 branch checks a stale type id 5, but this ggml tags I32
  as 26) → dumper now stores `token_ids` as F32.
- ✅ Preprocessing parity: `recognize_raw` now does cv2-bilinear resize +
  RandomInvert + BGR-as-RGB grayscale → native decode is **token-identical to
  HF** on real GrandStaff scores (100% on 3/4; 4th matched to the ref cap), CPU
  and Metal.
- ✅ Wiring: `src/crispembed.cpp` dispatcher (`arch=="smt_ocr"` → all 4 switches),
  so `crispembed -m smt.gguf --ocr score.png` works end-to-end (verified 69/69 vs
  HF); `smt_ocr_recognize_raw` added; `examples/cli/model_mgr.cpp` registry entry
  (`smt-grandstaff`). Server/bindings inherit via the generic `crispembed_ocr_model_*`.
- ✅ Quantize: `tools/quantize.cpp` keeps SMT conv kernels (`dwconv`/`downsampling`)
  and the baked PE (`positional`) F32; engine reshapes the quantizer's flattened
  2-D conv headers back to 4-D. **q8_0 (24 MB) decodes identically to HF (100%);
  q4_k (17 MB) is too lossy for the AR decode (~32%) — ship f32 + q8_0 only.**
- ✅ KV-cache: incremental decode (cross K/V precomputed once, self K/V grown per
  step via concat). Token-identical to the full-recompute path (kept behind
  `SMT_OCR_FULL_DECODE=1` for A/B) and to HF, CPU + Metal. **5.4× faster** (0.37 s
  vs 1.98 s for ~100 tokens); the gain grows with sequence length.
- ✅ GGUF upload: `cstr/smt-grandstaff-GGUF` (f32 83 MB + q8_0 24 MB + MIT model
  card; card license verified `mit`). Registry auto-download works end-to-end.
- ✅ **Preprocessing fixed → SMT WORKS at 96.3%.** The engine had been *inverting*
  the image (SMT-plusplus's `convert_img_to_tensor` has `RandomInvert(p=1.0)`), but
  `smt-grandstaff` is an **SMT-main** model whose preprocessing is `Grayscale→
  ToTensor` with **NO invert**. Inverting → ~30%; correct (non-inverted) → **96.3%**
  on the clean `antoniorv6/grandstaff` test split (per-image 91.8/96.2/96.7/99.6%).
  Full pipeline: RGB (no cv2-BGR swap), `reduce_ratio=1.0`, `width=min(w,3056)`,
  `height=max(h,256)`, grayscale, no invert. Fixed in `recognize_raw` + the dumper.
- ✅ **Fully validated:** per-stage diff cos=1.0; C++ decode == Python blueprint
  (100% token agreement, 10 fresh images); **C++ engine vs ground truth = 96.3%.**
  SMT-plusplus's unscaled forward confirmed correct (SMT-main's forward → 0% garbage
  on this checkpoint). The port was exact all along — the invert was the only bug.
  Lesson: [[validate-intermediates-and-outputs]] — a "reads-structure-not-detail"
  pattern across models was a preprocessing/input bug, not model quality; derive
  preprocessing from the model's OWN repo (SMT-main, not the SMT-plusplus fork).

**Landmines:**
- **⚠ SMT attention is UNSCALED.** `MHA.forward` computes `bmm(q,k)` then softmax
  with **no** `1/sqrt(head_dim)` — `self.scale_factor` is defined but never
  applied (verified in source, not the abstract). The C++ must NOT scale QK^T
  (converter records `smt.scale_attention=False`). Also: token embeddings are
  **not** scaled by `sqrt(d_model)` (no `scale_embedding`).
- **Cross-attn key≠value:** memory_key = flattened encoder features **+ 2-D PE**;
  memory_value = **raw** flattened features. Easy to wire both to the same tensor.
- **Encoder `last_hidden_state` is pre-pooler-LN** → `encoder.layernorm` in the
  ckpt is dead weight; don't apply it. Feature map is `(256, H/16, W/16)`.
- **Handwritten training-data license trap:** the canonical handwritten OMR
  datasets — **MUSCIMA++ / CVC-MUSCIMA — are CC BY-NC-SA (non-commercial)**.
  Training weights on them contaminates the *weights* for commercial use (same
  pattern as the old PosFormer/BTTR/HMER math models). PrIMuS / Camera-PrIMuS /
  GrandStaff are printed/synthetic and license-clean. Keep handwritten training
  data NC-free from day one if shipped weights must be commercially usable.
- **VisionEncoderDecoder `decoder_start_token_id`** comes from the *top-level*
  config, not the nested decoder config (the TexTeller start-token bug that
  poisoned position-0 KV — see the TexTeller 3.0 entry above). SMT's converter
  must resolve the start token the same way.
- Watch F16 Metal matmul overflow on large activations (see
  [[metal-mul-mm-f16-overflow]]) as with all VED ports.

**Sources:** SMT [github.com/antoniorv6/SMT](https://github.com/antoniorv6/SMT) ·
[SMT++](https://github.com/antoniorv6/SMT-plusplus) ·
[HF smt-grandstaff (MIT)](https://huggingface.co/antoniorv6/smt-grandstaff) ·
[PRAIG collection](https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd).
oemer [github.com/BreezeWhite/oemer (MIT)](https://github.com/BreezeWhite/oemer).
TrOMR [github.com/NetEase/Polyphonic-TrOMR (Apache-2.0, weights `img2score_epoch47.pth` 86 MB in-repo)](https://github.com/NetEase/Polyphonic-TrOMR).
[Flova/omr_transformer (Apache-2.0)](https://huggingface.co/Flova/omr_transformer).
homr [github.com/liebharc/homr (AGPL-3.0)](https://github.com/liebharc/homr).

### OCR — next-gen models to port

| # | Model | Params | OmniDocBench | License | Architecture | Status |
|---|-------|--------|-------------|---------|-------------|--------|
| ~~1~~ | ~~dots.ocr~~ | ~~3B~~ | ~~88.4%~~ | ~~NOT pure MIT~~ | — | REJECTED: supplemental PRC license (rednote/Xiaohongshu) |
| 2 | **PaddleOCR-VL-0.9B** | 0.9B | — | Apache-2.0 | NaViT + ERNIE-4.5-0.3B | **DONE + verified E2E** (2026-07-02): reuses qwen2vl_ocr engine; fox.png → "The quick brown fox…" on CPU+Metal. Was SIGSEGV-ing (ERNIE head_dim=128≠D/heads) + empty output (SPM vocab loaded as GPT-2 BPE); both fixed. Q8_0/Q4_K on HF |
| 3 | **PaddleOCR-VL-1.6** | 0.9B | 96.3% SOTA | Apache-2.0 | NaViT + ERNIE-4.5-0.3B (same arch, improved training) | **DONE**: same engine/fixes as 0.9B; Q8_0/Q4_K on HF |
| ~~4~~ | ~~MinerU2.5-Pro~~ | ~~1.2B~~ | ~~90.7%~~ | ~~NOT pure Apache~~ | — | REJECTED: commercial thresholds, mandatory attribution, gated HF |
| 5 | **SmolDocling** | 256M | — | Apache-2.0 | Idefics3/SmolVLM, IBM Research | DONE: engine + parity cos=0.9999, HF `cstr/smoldocling-GGUF` |
| ~~6~~ | ~~Hunyuan-OCR~~ | ~~1B~~ | — | ~~Custom Tencent~~ | — | REJECTED: excludes EU/UK/South Korea |
| 7 | **Qari-OCR** | 4B | Apache-2.0 | Qwen2-VL fine-tune (Arabic only) | **DONE (shipped)** — registry `qari-ocr` → `qari-ocr-2b-q4_k.gguf`. Vision parity fixed; direct "output only text" prompt; filename-independent `general.name` detection. |

~~**Remaining**~~ **DONE (both shipped)**: FireRed-OCR (registry `firered-ocr` / `firered-ocr-q4k`, Qwen3-VL 2B) and german-ocr-3 (registry `german-ocr-3.1`, Qwen2.5-VL) both reuse the qwen2vl_ocr engine; runtime ne-fix handles GGUF converters that store weights in PyTorch (out, in) order. (NB: `src/fireredpunc.cpp` is a different model — BERT punctuation, not FireRed-OCR.)

#### OCRBench leaderboard reference (small VLMs, ≤3B)

| Rank | Model | LLM | Params | OCRBench | License | Status |
|------|-------|-----|--------|----------|---------|--------|
| 1 | Granite Vision 3.3-2B | Granite-3.1-2B | 3B | 852 | Apache-2.0 | **Ported** |
| 2 | InternVL2.5-2B* | InternLM2.5-1.8B | 2.1B | ~830 | MIT | **Ported** |
| 3 | MiniMonkey | InternLM2-1.8B | ~2B | 806 | — | Low priority |
| 4 | H2OVL-Mississippi-2B | H2O-Danube-1.8B | 2.1B | 782 | Apache-2.0 | **Ported** |
| 5 | InternVL2-1B | Qwen2-0.5B | 0.9B | 779 | MIT | **Ported** (edge) |
| 6 | InternVL2-4B | Phi-3-mini | ~4B | 776 | MIT | Low (too big) |
| 7 | H2OVL-Mississippi-0.8B | H2O-Danube3-0.5B | 0.8B | 751 | Apache-2.0 | Low (tiny) |

*InternVL2.5-2B not on the original leaderboard slice but scores higher than
InternVL2-2B (768).

### llama.cpp parity — support matrix (reference)

A living audit of which CrispEmbed architectures llama.cpp supports (upstream
`ggml-org/llama.cpp` @ ~`4fc4ec5`, July 2026), how it implements them, and where
we remain unique. Deep technical notes live in `LEARNINGS.md → "llama.cpp
implementation reference"`. **The convergence backlog derived from this audit
(C1 imatrix quant, C3 batched throughput, C4 prefix KV, C5 mtmd preprocessing,
C6 flash-attn epilogue, mmproj interop) all shipped — see HISTORY.md.** This
section is kept only as the capability reference. Any future borrow must still
land behind an A/B on BOTH speed and quality, on CPU and Metal.

#### Support matrix (CrispEmbed arch → llama.cpp)

Text-embedding encoders:

| CrispEmbed | in llama.cpp | llama.cpp arch id | note |
|---|---|---|---|
| BERT | ✅ | `bert` | one shared `bert.cpp` graph, config from GGUF |
| XLM-RoBERTa | ✅ | `bert` | RoBERTa/XLM-R fold into `bert`; pos-offset + SPM handled |
| NomicBERT | ✅ | `nomic-bert` | SwiGLU + RoPE |
| NomicBERT-MoE | ✅ | `nomic-bert-moe` | PR #12466; 8-expert top-2 |
| ModernBERT | ✅ | `modern-bert` | SWA global/local + per-layer RoPE θ |
| MPNet | ❌ | — | T5-style rel-attn bias unimplemented — **we are unique** |
| GTE-v1.5 (`NewModel`) | ❌ | — | NTK-RoPE `NewModel` unsupported (#6821) — **we are unique** |
| DeBERTa-v2 | ❌ | — | disentangled c2p/p2c has no ggml graph — **we are unique** |
| SPLADE (sparse) | ❌ | — | MLM head dropped at convert — **we are unique** |
| bge-m3 sparse+ColBERT | ❌ (dense only) | — | tri-head only in fork `iz0eyj/llama.cpp-mv` — **we are unique** |

Decoder / hybrid embedders:

| CrispEmbed | in llama.cpp | arch id | note |
|---|---|---|---|
| Qwen3-Embedding | ✅ | `qwen3` (embed mode) | last-token, **causal** (Qwen3-Emb is trained causal — correct); Instruct/Query prefix is caller-side |
| EmbeddingGemma | ✅ | `gemma-embedding` | Dense/Matryoshka projection supported via `--sentence-transformers-dense-modules`; mean, non-causal |
| LFM2 / LFM2.5 | ✅ | `lfm2` (+`lfm2moe`) | PR #14620; ShortConv via `ggml_ssm_conv`, conv tensors F32 |
| LFM2.5-Embedding | ✅ | `lfm2` embed | official LiquidAI GGUFs, bidirectional |
| LFM2.5-ColBERT | ⚠️ partial | `lfm2` + `--pooling none` | per-token out; MaxSim client-side |
| BidirLM-Omni | ❌ | — | not present — **we are unique** |

Reranking: `--pooling rank` (RANK=4), `/v1/rerank` (PR #9510). bge-reranker-v2-m3
/ base, jina-v2, ms-marco-MiniLM ✅. Qwen3-Reranker ✅ (needs `cls.output.weight`
+ template). mxbai-rerank (DeBERTa-v2) ❌.

Vision / VLM-OCR (via `libmtmd`, projector-id keyed):

| CrispEmbed | in llama.cpp | projector id | note |
|---|---|---|---|
| Qwen2/2.5-VL | ✅ | `qwen2vl_merger` / `qwen2.5vl_merger` | 2D RoPE `build_rope_2d()`, window-attn |
| Qwen3-VL (+MoE) | ✅ | `qwen3vl_merger` | **DeepStack + IMROPE** — same family as our BidirLM-Omni |
| InternVL2/2.5/3 | ✅* | `internvl` | OpenGVLab (non-HF) checkpoints only |
| GLM-4V / GLM-OCR | ✅ | `glm4v` | AIMv2 tower, **dynamic** resize — ours matches now (Glm46VImageProcessor Qwen2VL smart-resize, shipped `dfd5653`; verified OCR 2026-07-13) |
| Granite Vision 3.x | ✅ | `mlp` (LLaVA-Next) | multi-level feature concat + anyres |
| SmolVLM/SmolDocling/Idefics3 | ✅ | `idefics3` | SigLIP + pixel-shuffle |
| Pixtral / LightOnOCR-1B | ✅ | `pixtral` / `lightonocr` | LightOnOCR-2 declined (#18943) |
| DeepSeek-OCR / Unlimited-OCR | ✅ | `deepseekocr` / `deepseekocr2` | hybrid SAM+CLIP DeepEncoder |
| PaddleOCR-VL | ✅ | `paddleocr` | NaViT + M-RoPE (`ggml_rope_multi`) |
| GOT-OCR2 | ❌ | — | SAM path exists only inside DeepSeek-OCR — **we are unique** |
| CLIP/SigLIP standalone image **or** text embed | ❌ | — | mtmd is tower-only (per-patch, LLM-sized); no text tower — **we are unique** |
| Math OCR (pix2tex/TrOCR/HMER/BTTR/PosFormer/MixTex/PP-FormulaNet/PARSeq/Tesseract/Pix2Struct) | ❌ | — | enc-dec/CTC out of llama.cpp's class — **we are unique** |

**Reverse interop (import a stock llama.cpp mmproj INTO CrispEmbed):** shipped +
validated for the three rows where both a working CrispEmbed loader and a
downloadable mmproj exist — `qwen2vl_merger`, `idefics3`, `internvl` — via the
auto-detecting `models/merge-llamacpp-gguf.py` (see the status block below and
README "Importing a stock llama.cpp VL model"). Qwen2-VL is bidirectional
(export too). The rest need either a non-crashing dynamic-preproc loader
(`glm4v`) or an mmproj llama.cpp doesn't ship (`GOT-OCR2`).

Entirely outside the ggml ecosystem (CrispEmbed-only): **face** (YuNet/SCRFD/
AuraFace/SFace), **detection/layout** (DBNet/RT-DETRv2/Surya-Det), **NER/KIE**
(GLiNER/LiLT; BERT-NER only an *unmerged* PR #19725), **LID** (CLD3/GlotLID),
**punctuation** (FireRedPunc/Fullstop/PCS), and **image restoration/SR** (NAFNet/
SwinIR/HAT/Restormer/SCUNet/SAFMN/DAT/InstructIR/AdaIR — only ESRGAN/RRDBNet
exists, and in `stable-diffusion.cpp`, not llama.cpp).

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| ~~Qwen3-VL multimodal~~ **DONE** | — | — | Qwen3-VL OCR/VLM shipped: engine (`qwen2vl_ocr.cpp` DeepStack + interleaved-mRoPE + qk-norm) + registry `qwen3vl-2b`. (Only a Qwen3-VL *embedding* model — not the OCR path — would still be open, if ever wanted.) |

### DeepSeek-OCR-2 performance (remaining levers)

The pipeline is now mostly on Metal (encoder, MoE decode, SAM convs + patch
embed, LM head) — full OCR ~9 min (never completed) → ~12 s warm. Profiled
warm breakdown: load ~9 s cold / 0.8 s warm · SAM ~4.7 s · decode ~3.8 s ·
enc+proj ~1.1 s. Remaining levers, ranked by leverage:

- [x] **#1 Load-path prefetch — DONE, but not the bottleneck.** Added
  `madvise(MADV_SEQUENTIAL/WILLNEED)` to `core_gguf::load_weights` (correct
  practice, helps genuinely disk-bound cold loads on other systems). On *this*
  machine it didn't move the needle, and the diagnostic explains why: the disk
  reads 2.1 GB in **1.17 s** and a warm load is **0.8 s** — so the ~9–18 s cold
  loads are **memory-pressure / swap**, not readahead. During a run the process
  holds ~5 GB (2.1 model + 1.3 stacked experts + 0.65 embed-f32 + Metal) on a
  16 GB box, so file pages and new allocations contend and swap. → the real load
  lever is **reducing the footprint** (#3, #4), not prefetch.
- [~] **#2 Decode graph reuse — PARTIAL (KV persistent; graph NOT).** Corrected
  2026-07-20 (code-verified): the **KV cache is persistent** device tensors
  (`alloc_ds_kv_cache`, written in-graph via `ggml_cpy`/views) — but the decode
  **graph is still rebuilt + freed every layer, every token** (`build_llm_layer_attn`
  → `ggml_free(lag.gctx)` in the per-layer loop). No persistent/single multi-layer
  graph yet; the `g_ds_build_us`/`g_ds_compute_us` profiler exists precisely to
  decide if the persistent-single-graph port is worth it. So **the persistent
  single-step decode graph + F16 KV (cache is F32 today) are genuinely OPEN for
  deepseek specifically** — the one engine the GPU-decode "done" note above does
  NOT fully cover.
- [x] **#3 Per-row embedding dequant — DONE (core win).** The decode hot path
  `get_embedding` lambda is per-row (`ggml_backend_tensor_get` + `to_float`),
  replacing the 655 MB full-table copy held across decode. (Sub-detail corrected:
  the prompt-assembly `put_tok` still full-table-dequants once via `to_f32`, freed
  after — not per-row; line refs drifted to ~2424/~1897.)
- [x] **#4 Converter-emitted stacked experts (memory) — DONE
  (`feat/ds-ocr2-stacked-experts`).** Converter emits `ffn_{gate,up,down}_exps
  [in,out,n_exp]` (byte-identical to `stack_moe_experts`); loader loads them
  directly + per-expert views for the `DS_MOE_CPU` fallback + backward-compat.
  Kaggle-reconverted + byte-validated vs source; f16/q4_k on HF as `-stacked`
  (non-clobber). **M1 Metal q4_k A/B: peak footprint 5.27→3.97 GB (−1.30 GB),
  decoded output identical on all 3 loader paths.** Registry auto-download default
  **promoted to `deepseek-ocr2-q4_k-stacked.gguf`** (loader backward-compatible).
  Deep-dive in LEARNINGS.
- [ ] **#5 SAM flash-attention (marginal, skip unless needed).** The SAM
  attention uses a decomposed rel-pos bias (rel_h/rel_w added to scores), which
  blocks `ggml_flash_attn_ext` unless the bias is materialized as a [T,T] mask —
  fiddly, and the win is small (~3–4 s SAM is mostly the genuine 4096-token
  global attention compute).

All deepseek perf paths are env-gated with validated CPU fallbacks
(`DS_QWEN2_SCALAR`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU`, `DS_MMAP`,
`DS_REF` parity harness, `DS_DBG` timers).

### Open performance levers

Each needs a target GGUF (q8_0 preferred, to isolate from q4_k noise) and a
before/after parity + latency measurement — never land a "perf" change on a
compile-only check. A/B every change against ground truth and gate behind an env
var (see `../crispasr-crispembed-dev.md` "A/B every perf optimization").

- **ENCODER (embedding) path — the domain the 2026-07-16 community-GGUF work
  landed in, and NOT otherwise in this backlog (encoders are fast: 6–22 layers,
  batched).** One concrete micro-lever spotted:
  - **MoE FFN redundant `ggml_repeat` (nomic-bert-moe / nomic-embed-text-v2-moe).**
    The MoE FFN in `src/crispembed.cpp` explicitly expands the input
    `cur [H,TB] → [H,K,TB]` with `ggml_repeat` before `ggml_mul_mat_id`. llama.cpp's
    canonical MoE reshapes to `[H,1,TB]` and lets `mul_mat_id` BROADCAST the
    singleton expert-slot dim, so the repeat materializes K copies of the
    activations per MoE layer for nothing (6 MoE layers × K=2 on nomic-v2-moe).
    Gate landed on `main` (`5abc4de`), broadcast path behind
    `CRISPEMBED_MOE_NO_REPEAT=1` (default keeps the repeat).
    **Correctness VALIDATED (2026-07-16):** default vs `CRISPEMBED_MOE_NO_REPEAT=1`
    on `nomic-embed-text-v2-moe` is **BYTE-IDENTICAL (max_abs_diff=0.0, cos=1.0)**
    at BOTH f16 and q4_k (50-token input) — the broadcast is exactly the repeat, so
    HF cosine is unchanged by construction. **Latency INCONCLUSIVE / neutral:** a
    7-run bench A/B (graph-compute, T=50, Metal) gave repeat median ~188 ms vs
    norepeat ~195 ms but with ±100 ms run-to-run swings at load ~9 — the
    distributions fully overlap, so no reliable delta (matches the "may be
    perf-neutral" expectation; the repeat materializes only ~1.8 MB total). **Flip
    decision deferred:** per the dev-guide rule (flip only when it wins on speed AND
    quality), a clean flip needs a genuine quiet box (load <3) for a back-to-back
    median; until then keep opt-in — correctness is no longer the blocker, only a
    trustworthy latency number is.
#### HEADLINE remaining lever — GPU recognizer AR decode (scoped 2026-07-20)

PERFORMANCE.md calls the per-region CPU-bound token loop "the real speed path".
**This is NOT greenfield:** `internvl2_ocr` (maturity rank 1) already ships the
target pattern — `ggml_flash_attn_ext` LLM decode + **F16 KV in ggml tensors
(zero-copy view + `ggml_cpy` writes)** + prefill/decode separation + sched GPU
dispatch; `glm_ocr` (rank 2) and `got_ocr` (rank 3) confirm it. The project =
**propagate that proven pattern to the laggard engines**, ranked by leverage
(PERFORMANCE.md "Optimization maturity ranking" + "Opportunities"). Beyond the
KV swap, the top layer is a **persistent single-step decode graph** (build once,
`gallocr` once at max KV, dispatch sched-free per step, **re-set ALL inputs each
compute** — the moonshine/OMR pattern; already proved here: smt-fp 18×, transcoda
2.4–4×, byte-identical).

**✅ CODE-VERIFIED 2026-07-20 — ALREADY DONE; PERFORMANCE.md's maturity table is
STALE.** Auditing the actual LLM-decode path of every engine (not the table): they
**all default to a ggml F16-KV GPU decode**, with the `core_vlm` CPU-scalar path
kept only as a gated fallback. So this "headline project" is closed. Evidence:
- **`qwen2vl_ocr`** — F16 GPU KV (`GGML_TYPE_F16` on `ctx.backend`) + `ggml_flash_attn_ext`
  + `build_decode_step_graph` (0 `core_vlm`).
- **`smoldocling_ocr`** — `sd_run_llm_body` ggml graph handles decode T=1 with F16
  backend KV; `use_ggml = (llm_sched && sd_alloc_kv_cache())` is the DEFAULT,
  `sd_llm_decode_step` (`core_vlm`) is the fallback.
- **`granite_vision_ocr`** — `gv_run_llm_body` ggml + F16 backend KV is DEFAULT
  (`if (!getenv("CRISPEMBED_GRANITE_LLM_SCALAR")) use_graph = gv_alloc_kv_cache()`),
  diff-validated cos 0.9999 vs granite-llm-ref; `core_vlm` is the opt-out. The old
  "10–50× / entire LLM CPU-scalar" is stale.
- **`pix2struct`** — KV cache + DequantCache (Phase 2/3), not "no KV, O(T²)".
- **`deepseek_ocr2`** — ggml per-layer graphs + flash + `alloc_kv` (not `core_vlm`).
- **`internvl2`/`glm`/`got`/`lightonocr`** — the reference implementations.

**Only genuine sliver left (micro, not the headline):** `deepseek_ocr2` builds
per-layer graphs (≈12 builds/token) rather than one multi-layer graph — a graph-shape
tidy, F16 KV already present. And the *persistent single-step graph* (build once,
reuse) is only in qwen2vl/lightonocr/deepseek; the others rebuild the step graph each
token but already on-GPU. Both are marginal vs the closed headline. **PERFORMANCE.md's
"Optimization maturity ranking" + "Opportunities" tables need a refresh to match.**

**Tier 2 — polish:** `lightonocr` GPU dispatch (has persistent F16 KV, GPU=No);
`internvl2` native GQA in flash (skip `ggml_repeat`).

**Landmines (non-negotiable):**
- **CUDA contiguity (LEARNING 35):** `ggml_get_rows` needs a contiguous index
  (`ggml_cont` before it). "Correct on CPU AND Metal" is NOT sufficient — CUDA has
  stricter per-op asserts; the decoded-roundtrip MUST run on a real CUDA box
  (Kaggle P100) before flipping any GPU default. [[flashattn-ext-already-permutes]]
- **Metal `set_output` snapshots LIE** on the sched — bisect on the genuine
  truncated output (`..._MAX_LAYERS=N`), not per-intermediate snapshots.
  [[set-output-on-view-stale]]
- **Metal `mul_mm` F16 overflow** (large ×N activations) → scale 1/256 pre-matmul,
  ×256 post. [[metal-mul-mm-f16-overflow]]
- **CPU-pinned decode re-copies GPU weights every token** — `load_weights_split`
  (encoder→GPU, decoder→CPU) to kill cross-backend traffic; **per-step GPU dispatch
  is launch-bound for tiny models** — the persistent graph is the win, not
  sched-free per-step. Measure the CPU baseline on the right BLAS first (parakeet
  lesson: a "GPU idle" gap was half a CPU-BLAS artifact).
- ggml scheduler: run side graphs before alloc; never reset between alloc and
  compute on the same graph.

**Validation gates (per change; env-gate every path, NEVER delete the scalar one):**
1. Per-stage `crispembed_diff` structural parity (cos ≥ 0.999).
2. **Decoded-output roundtrip is the ONLY acceptance test** — OCR a real doc, read
   the text. Test BOTH f16 AND q4_k.
3. A/B back-to-back under IDENTICAL load on a quiet box (loaded timing lies ±20%);
   final GPU-default flip gated on a Kaggle CUDA decoded-roundtrip.
4. Add a regression entry with `expected_text`; keep `<ENGINE>_CPU_DECODE=1` fallback.

**Sequencing:** ~3–5 focused sessions, each needs a quiet box + one Kaggle run.
S1 `smoldocling_ocr` (core_vlm→ggml LLM decode) → S2 `granite_vision_ocr` (10–50×, instrument-first)
→ S3 `deepseek_ocr2` (single graph + F16 KV). qwen2vl/pix2struct already done; the
the pattern first.

- **SR/restoration — fused ggml graphs: COMPLETE (2026-07-13).** Every engine
  now runs a fused ggml graph, not per-conv mini-graphs. Ported this session:
  - **SAFMN** (`8594cee`): whole forward = ONE fused graph (erf-GELU) — **2.2×
    faster AND more accurate (cos 1.000000 vs 0.994)**. Tiny/overhead-bound, so
    fusion is a big win; Metal is a net loss here (default CPU, `SAFMN_SR_METAL`).
  - **NAFNet** (`14a8393`) + **InstructIR** (`e1eb1dc`): fused per-block graph,
    cos ≥ 0.999998, output identical to legacy. NAFNet-family = **compute-bound**,
    so fusion is perf-NEUTRAL (cleaner, not faster). NAFNet defaults to Metal
    (modest ~15%; `NAFNET_CPU`); InstructIR is CPU-only (GPU conv_2d hits a Metal
    f32×f16 mul_mv pipeline issue). Gates: `NAFNET_LEGACY` / `INSTRUCTIR_LEGACY`.
  - **Restormer**: was ALREADY fused — `rst_transformer_block_ggml` (MDTA + GDFN
    in one graph) is the default; `RESTORMER_SCALAR` is the fallback (cos 0.999997
    both). Only the stale "CPU-scalar" header was corrected.
  - **scunet, swinir, tbsrn, hat, adair, dat**: already build a single graph
    (`forward_expand=1`, no per-conv helpers) — verified sensible (swinir 0.9984,
    dat 0.99999, hat 0.89 q8_0). No work needed; the "CPU-scalar" labels were loose.
  **Key finding:** the fusion win depends on overhead-bound (tiny SAFMN → 2.2×)
  vs compute-bound (NAFNet/InstructIR → perf-neutral). Metal helps only where
  per-dispatch overhead is small relative to compute. Env gates per engine.
- **SR-on-GPU — conv weight residency (research, deferred).** The entire SR
  family computes convs on a CPU-only `enc_sched` with CPU-resident F32 kernels;
  there is no GPU sibling to match. Real SR-on-GPU needs Metal `ggml_conv_2d` for
  these shapes + a GPU-resident weight/graph path the family currently avoids —
  research, not a residency toggle. Reprioritized down.
- **Decode-step graph cache — remaining decoders.** Shipped (sched-free gallocr,
  reserved once at max KV, byte-identical, per-engine env gate) for got_ocr,
  internvl2, glm_ocr, lightonocr, math_ocr. **Still open, each needs the
  single-backend decode check first:** `smoldocling` (CPU LM head outside the
  graph), `granite` (shares the vision sched), `deepseek_ocr2` (per-layer-per-step
  → needs the persistent-graph variant). Modest win (~3% light decoders, ~0% heavy;
  real value is load-insensitivity). `qwen2vl` does NOT fit (multi-backend decode).
- **ggml-metal ICB replay / op-count reduction (the real Metal decode lever).**
  Warm Metal decode is ~82% GPU-execute (per-kernel launch across ~355 sequential
  ops), so ICB (which only collapses the ~18% host-encode) caps at ~18% and is
  NOT justified for CrispEmbed's light decoders. The tractable in-tree lever is
  **fewer, bigger ops per step** — fuse per-layer norm/scale/bias chains, QKV,
  the GLU elementwise chain, prefer `ggml_soft_max_ext`. Per-decoder graph surgery
  in each `build_decoder_step_graph`; verify output cos ≈ 1.0 + node-count +
  latency per model. Re-measure heavy decoders with `CRISPASR_METAL_PROFILE=1`
  before any ICB work. **Caveat (measured 2026-07-13):** the math_ocr ~30%
  cont-removal does NOT generalize to decoder-only VLM engines — got_ocr's cached
  decode already feeds K/V as cache views, so only Q's cont was removable
  (byte-identical, but latency within noise; `5011848`, `GOT_OCR_ATTN_CONT=1`).
  **Op-fusion measured marginal too (2026-07-13):** (a) Metal already auto-fuses
  (`use fusion=true`; `kernel_norm_mul_add`, `kernel_bin_fuse` kernels handle the
  norm/scale/bias + GLU elementwise chains at dispatch), so graph-level elementwise
  fusion is redundant there; (b) attention is already flash-fused; (c) these
  decode steps are compute-bound (got_ocr ~89% GPU-execute), capping any dispatch
  reduction at the ~11% host slice; (d) the trocr decoder is already lean (319
  nodes, 55 ms/16 tok — the ViT *encoder* at 212 ms is trocr's real cost, not the
  decoder). The only non-auto-fusable win is **QKV concat-matmul** (3→1), but a
  probe (`GOT_OCR_QKV_FUSE`, 2026-07-13) confirmed it's not worth it: `ggml_concat`
  **mishandles q4_k** (garbage output) and re-concatenating per step is 3× slower,
  so a correct fusion needs manual load-time q4_k row-block byte-stacking — and on
  a memory-bound T=1 decode that only saves ~2 matmul launches/layer (~4%).
  Deferred; see HISTORY.
  (DeepSeek-OCR-2's MoE-compute lever is detailed in its own subsection above.)
- **unlimited_ocr — remaining deferred items.** `UOCR_PD=1` persistent T=1 decode
  graph (blocked on a small flash-attn padded-vs-exact-KV numerical drift that
  changes argmax by ~step 3; ~14% decode win if solved); `UOCR_OPT_GGML_WINDOW=1`
  (SAM window partition in-graph, ~2–5%, deferred); SAM flash-attn (won't — the
  decomposed RPE bias defeats the O(T) benefit).
- **text_sr — blocked on a public checkpoint** (NAFNet text-SR; registry URL
  empty, no shipped GGUF). Conv paths are guarded transitively by the `nafnet`
  entry; PixelShuffle/bicubic tail unguarded. To train one on clean (Apache/MIT)
  data see `docs/text_sr_training_data.md`.
- **esrgan tile-loop parallelism (concurrency project, deferred).** Intra-op
  threading measured SLOWER (tiled convs don't thread-scale). The real lever is
  running whole 128px tiles concurrently → needs per-thread backend+sched
  replication (the tile loop shares one `ctx->enc_sched`). Verify on a quiet box.
- **TrOCR recognizer accuracy/speed.** eos/length-penalty parity is still TODO
  (the trigram-repeat bug is fixed). The bigger levers: swap DBNet-ic15
  (scene-text) for a document-text detector on dense pages; steer document OCR to
  the doc-VLMs (PaddleOCR-VL / SmolDocling); GPU (WebGPU/Metal) recognizer decode
  is the real speed path (the per-region AR token loop is CPU-bound).

### Open correctness / infrastructure

- **CUDA regression — the 4 FAILs are RESOLVED / explained (P100-verified 2026-07-13).**
  A diagnostic kernel (`tools/kaggle/crispembed-cuda-diag`, Tesla P100 / Pascal
  sm_60) diagnosed each under its env gates, then a 2nd run verified the fix:
  - **`layout-heron` — FIXED (`49cb38a`).** The flash→manual attention fallback
    removed the `fattn.cu:602` abort; P100 CUDA now runs `test-layout-diff` to
    **8/8 stages PASS, DIFF PASSED** (dec_0_cross_out 0.977). ✅
  - **`glm-ocr` + `internvl2` — FIXED (`7998f3c`): it was a stdout banner, NOT
    vision garbage.** Both engines printed their load banner (`glm_ocr: loading…
    Vision:… LLM:… KV cache… Ready`) via `printf` → **stdout**, and `run_one`'s
    `--ocr` text-match captures stdout — so `actual` = the banner (cer 4.3/5.4,
    mis-read as "Class-B CUDA vision garbage"). The P100 diagnostic proved both
    OCR the fox **correctly** on CUDA *and* CPU; only the harness saw the banner.
    Routed all banners to stderr to match the passing engines (qwen2vl_ocr, …). ✅
  - **`granite-vision` — text OCR PASSES**; the projector diff drift is
    cross-toolchain FP strictness (identical CUDA=CPU=scalar on P100), threshold
    already 0.95. ✅
  - **Bottom line: NONE of the 4 were real CUDA vision divergences.** It was one
    genuine CUDA bug (layout flash-abort on Pascal) + a stdout-banner harness bug
    (glm/internvl2) + cross-toolchain FP threshold strictness (granite). The
    diagnostic-first approach (test on the box via env gates) was essential — a
    blind "fix the Class-B vision divergence" would have chased a non-existent bug.
  - **RESULT: portfolio 14 → 0 FAIL** across the fix waves (harness `be6ec54`;
    parser `2af57b1`; layout flash→manual `49cb38a`; banner→stderr `7998f3c`;
    parser value-dump/nameless `c26abc4`; layout perm-tolerant `debug/layout-cross`).
    glm-ocr, internvl2, granite all PASS on P100 now. **All original FAILs fixed** —
    every "Class-B" one was a harness/output bug, not CUDA vision divergence.
  - **The last FAIL (`layout-heron` `dec_0_cross_out`) — ROOT-CAUSED + FIXED
    (`debug/layout-cross`).** NOT flaky and NOT an inference bug. The apparent
    "non-determinism" (0.977 v2 vs −0.034 v14 on P100; −0.08/−0.19 on Metal
    manual/flash) is a **query-permutation comparison artifact**. The 300 decoder
    queries are chosen by `partial_sort` over ~8400 near-tie encoder proposals
    (`layout_detect.cpp:1318`); a tiny backend FP delta in enc_output (Metal/CUDA
    vs the CPU/Python reference — max_abs 0.02, cos 0.99999) reshuffles near-tie
    ranks, so "query i" in our output is a *different physical proposal* than the
    reference's "query i". Instrumented proof: the initial queries themselves show
    per-query cos mean 0.78 / 111 below 0.9 (matching cross_out's mean 0.79), the
    top-5 ranks agree, and the cross_out **values are correct** (best-cosine
    matching each ref query → cos_mean 0.999, 299/300 unique = clean bijection).
    Final boxes are unaffected (score-sort + NMS). **Fix:** `test_layout_diff.cpp`
    compares this stage permutation-tolerantly (`perm_tolerant_cos`); now PASS on
    Metal (0.947/0.999), Metal+flash (0.947), CPU (0.967/0.999). Guardrail keeps
    full power — simulated scrambles (feature-shuffle/sign-flip/roll) collapse to
    ≤0.08 vs the 0.85 gate, and s3..enc_output still guard the encoder-scramble
    class strictly at 0.99. Manifest threshold 0.97→0.85 + comment corrected (the
    old "backend-independent" note was wrong).

  Original diagnostic detail (the run that overturned 3 of the 4 assumptions):
  - **`layout-heron` — REAL CUDA bug (fixable).** `test-layout-diff` aborts:
    `ggml/src/ggml-cuda/fattn.cu:602 fatal error` in `ggml_cuda_flash_attn_ext`
    → `GGML_ABORT` because Pascal (sm_60) has **no flash-attention kernel**
    (`get_best_fattn_kernel == BEST_FATTN_KERNEL_NONE`). With
    `LAYOUT_DETECT_FORCE_CPU=1` **all 8 stages PASS (cos 1.0)** — so the graph is
    correct; the engine just runs `flash_attn_ext` on a single CUDA backend that
    bypasses the scheduler's `supports_op` CPU-fallback. **Fix:** don't use the
    CUDA flash kernel where it's unsupported — either (a) route layout attention
    through a scheduler that honours `ggml_cuda_flash_attn_ext_supported` (returns
    false on sm_60 → runs on CPU), or (b) give `layout_detect` a manual masked
    attention fallback (`mul_mat`+`soft_max_ext`+`mul_mat`, mask=nullptr = full
    attn) selected when flash is unsupported. Verify: `test-layout-diff` PASSES on
    P100. NOTE T4 (Turing sm_75) HAS flash — this only bites Pascal.
  - **`granite-vision` — NOT a CUDA bug.** The projector stages fail **identically
    on CUDA, `GRANITE_VIS_SCALAR`, AND full-CPU (`GRANITE_CPU`)** on the P100 box
    (cos 0.952 / 0.958 / 0.955 — same to 2 dp across all three), while they PASS on
    the Mac. So it is a **cross-toolchain FP-strictness gap** (Kaggle gcc vs Mac
    clang on high-magnitude projector activations, max_abs ~2.7–4.3), NOT a CUDA
    divergence, and the **OCR text passes** (cer 0.163). **Fix:** relax the
    projector-stage diff thresholds (≈0.95, they gate a real crater by going
    negative) — a parity-harness strictness fix, not a model change.
  - **`glm-ocr` — NOT a CUDA bug.** `test-glm-ocr-diff` vis_layers 14–23 fail at
    cos 0.96–0.98 **identically on CUDA and CPU** on P100 (vis_layer_23: CUDA
    0.9630 vs CPU 0.9632; max_abs up to 217) — same cross-toolchain strictness as
    granite. And on a clean generated fox image glm reads it **correctly on CUDA**
    (`"The quick brown fox jumps over the lazy dog 12345"`). So glm's vision is not
    CUDA-garbage. Its portfolio FAIL is the **text-match on the repo `fox.png`
    (800×200)** specifically — untested CPU-vs-CUDA yet (see below).
  - **`internvl2` — reads a generated fox CORRECTLY on P100 CUDA** (identical to
    CPU). No ref uploaded, so no per-stage diff. Its portfolio FAIL is likewise the
    text-match on the repo `fox.png` (800×200), not universal vision garbage.
  - **Open sub-question (glm + internvl2 portfolio garbage):** the repo `fox.png`
    is 800×200 (the diagnostic used a 640×96 render). Next diagnostic run must OCR
    the **repo** `tests/regression/images/fox.png` under default vs `*_FORCE_CPU`
    for both engines — if CPU is also garbage there, it's a Kaggle-BUILD issue
    (like granite/glm diff), not CUDA; if only CUDA is garbage, it's a genuine
    larger-image CUDA vision divergence to localize. The vis-diff being CPU=CUDA
    identical strongly suggests the former.
  - **Full data:** the diagnostic log is on Kaggle
    (`chr1s4/crispembed-cuda-diagnostic-4-remaining-fails`, transcript in
    `/kaggle/working/diag.log`); see HISTORY.md.
- **DBNet detector — mostly resolved (2026-07-13).** The CPY abort was already
  fixed (`dequant_rows_f32` via get_rows); the real cost was the CPU postprocess
  (43 s → 1.5 s, scanline box scoring `74b8ac5`, see HISTORY). Detection graph
  compute is only ~3 s on CPU and Metal `conv_transpose_2d` is still ~13× slower,
  so **CPU stays the correct default** — a faster Metal `conv_transpose_2d` (or a
  1/4-res prob-map + cheap upscale) is the only remaining, low-value, upstream
  lever for GPU-default detection.
- **bidirlm-omni GGUF re-quant follow-up.** The text-tower converter bug is fixed
  and `bidirlm-omni-2.5b-q8_0.gguf` re-uploaded (text cos 1.0 f16 / 0.9992 q8_0),
  but the repo's f16 + imatrix q4_k/q5_k/q6_k and the whole `-textonly` repo are
  still the OLD (text-broken) conversion — regenerate them from the fresh f16
  (imatrix variants via the imatrix pipeline). Kaggle-only (16 GB Mac OOMs).
- **Regression-guardrail residuals.** `bert_ner` dumper written but its ref is
  download-blocked; face *recognition* (arcface/sface) unguarded (no local rec
  GGUF; detection is guarded). All SR/restoration (11) + esrgan/safmn + lilt +
  lfm2 + the closed engines are auto-guarded in `tests/regression/manifest.json`.
- **`core/vlm_decoder.h` — deferred.** A unified scalar decode loop; only 2 scalar
  engines remain, so abstracting is premature. Revisit if a 3rd appears.
