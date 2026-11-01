# GPU support matrix (every engine, Metal focus)

Branch `feat/metal-audit`, audit started 2026-11-01 on an Apple M1 (16 GiB,
8-core GPU). This is the inventory step: every cell below was read from `src/`
and from committed receipts **before** any new run. Rows that this audit later
re-measured carry a `2026-11-01` receipt under
[`tests/results/gpu-audit/metal-m1/`](../tests/results/gpu-audit/metal-m1/).
The older, OCR-only [`ocr_backend_matrix.md`](ocr_backend_matrix.md) stays as
the historical record; where the two disagree, this file is newer and says why.

## Reading rule

**Backend selection** (what the code does at load time on a Metal build):

| Code | Meaning |
|---|---|
| `GPU` | GPU backend by default (`crispasr_init_gpu_backend()` or equivalent), CPU fallback in the scheduler |
| `GPU*` | GPU by default but parts are pinned to CPU by design (named in the row) |
| `kind` | per-backend-kind default: GPU when a CUDA device is present, **CPU on Metal/CPU-only boxes** |
| `gated` | CPU by default; a GPU path exists behind an opt-in gate |
| `wload` | weights can be loaded on the GPU backend, but all compute runs on a CPU scheduler or scalar code |
| `CPU` | `ggml_backend_cpu_init()` unconditionally; ggml graphs exist but are CPU-scheduled |
| `none` | no ggml graph at all (hand-written host code / classical image processing) |

**Metal status**: `exact` (same output as CPU, receipt cited), `budget`
(differs from CPU inside the engine's own committed budget), `wrong`
(differs beyond budget or is known broken), `slower` (correct but measured
slower than CPU, so CPU stays default), `unknown` (no CPU-vs-Metal A/B on
record — an end-to-end Metal run alone is not parity). A date is attached to
every measured claim.

Gate spellings are exact. Legacy gates parsed with `getenv() != nullptr` or
`atoi()` are flagged `(presence)`/`(atoi)` because `X=0` behaves differently
from `core_env::on()` there.

## 1. Embedding, rerank, sparse, ColBERT, classification (`crispembed_init`)

| Engine / family | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| Encoder embedders (BERT, XLM-R, ModernBERT, NomicBERT, GTE, E5, BGE-M3 dense, MiniLM, arctic, granite-embedding) and cross-encoder rerankers | `crispembed.cpp` | `GPU` | **`exact`** on float weights (MiniLM F32 cos 0.9999996, ms-marco F32 rerank \|Δ\| 4.7e-4); **`budget`** on quantized weights (q8/q4 cos 0.9966-0.99989 Metal vs CPU — the CPU arm quantizes activations to q8 for its dot products, Metal keeps them F32; where a matching F32 reference exists Metal is the CLOSER arm: bge-small 0.99990 vs 0.99964, ms-marco \|Δ\| 0.024 vs 0.035), 2026-11-01. Ettin (ModernBERT) q8 rerank \|Δ\| 0.10, same order — see §Audit | `CRISPEMBED_FORCE_CPU` (value-parsed), `--gpu-backend`, `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB` | `tests/results/gpu-audit/metal-m1/embed/*.json` |
| Decoder embedders (Qwen3-Embedding, F2LLM, harrier, jina-v5, EmbeddingGemma) | `decoder_embed.cpp` via `crispembed.cpp` | `GPU` | **`exact`** on float weights (F2LLM-0.6B cos 0.9999997, jina-v5-nano 0.9999979); **`budget`** quantized (Qwen3-Emb Q8 0.99943, harrier q4_k 0.99929, EmbeddingGemma Q8 0.99991, jina q4_k 0.99310, F2LLM q4_k 0.97738 — same activation-quantization cause), 2026-11-01. Metal 1.7-3.3x faster steady-state except Q8_0 decoders (Qwen3-Emb 4.6 s vs 3.7 s: known Metal q8_0 mul_mv cost) | `CRISPEMBED_FORCE_CPU` | `tests/results/gpu-audit/metal-m1/embed/*.json` |
| LFM2.5 embedding / LFM2 encoder | `lfm2_embed.cpp` | `GPU` for the embedding graph; the **official masked-LM encoder path runs only on CPU** (`lfm2_encoder_enabled` requires a CPU backend unless forced) | `budget` for the embedding graph (LFM2.5-Embedding-350M Q8 cos 0.99983, Metal 2.0 s vs CPU 1.7 s — slower, Q8_0), 2026-11-01; encoder path not measured on Metal | `CRISPEMBED_LFM2_ENCODER` (tri-state, `=1` forces on GPU) | `docs/lfm2-encoder.md` (CPU parity only) |
| BiDirLM omni (text/vision/audio) | `crispembed.cpp`, `bidirlm_vision.cpp`, `bidirlm_audio.cpp` | `GPU` (shares the context backend); audio is host code | `unknown` | `CRISPEMBED_FORCE_CPU` | none |
| CLIP text tower | `clip_text_embed.cpp` | `GPU` | `unknown` | `CLIP_TEXT_FORCE_CPU` (atoi) | none |
| ViT image embedder | `vit_embed.cpp` | `GPU` | `unknown` | `VIT_EMBED_FORCE_CPU` (atoi) | none |
| CNN image/face embedder (`crispembed_face_init`) | `cnn_embed.cpp` | `GPU` | `unknown` | `CNN_EMBED_FORCE_CPU` (atoi) | none |
| UltraBERT | `ultrabert_embed.cpp` | `CPU` (no GPU code path) | n/a | `CRISPEMBED_ULTRABERT` (load gate) | `tests/results/ultrabert/` (CPU only) |
| MoEdern text / family / ColBERT | `moedern_text_engine.cpp`, `moedern_family_engine.cpp`, `moedern_colbert.cpp` | `CPU` | n/a | `CRISPEMBED_MOEDERN_COLBERT` (load gate) | `tests/results/moedern-colbert/` (CPU only) |
| EmbeddingGemma2 typed decision | `gemma_decision.cpp` | `CPU` | n/a | — | `tests/results/gemma-decision/` (CPU only) |
| Josev2B composed / SystemOne | `josev2b_composed.cpp`, `systemone_api.cpp` | `CPU` (head has its own `"cpu"` device argument) | n/a | Josev component gates | `docs/josev2b-head-gpu.md` (head GPU study, Python) |

## 2. Default OCR pipeline, detectors, recognizers

| Engine | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| DBNet detector (`ocr_detect`, default pipeline + tesseract/easyocr lanes) | `ocr_detect.cpp` | `kind` | `slower` (~139 s Metal vs ~10 s CPU, 1472x736 map, issue #25, 2026-08) | `OCR_DETECT_USE_GPU` (tri-state), `OCR_DETECT_FORCE_CPU` (atoi) | `ocr_backend_matrix.md` DBNet row |
| TrOCR line recognizer (default pipeline; `math_ocr` loader) | `math_ocr.cpp` | `GPU` | `unknown` vs CPU; 8/10 words exact end-to-end on Metal (2026-07-31) | `MATH_OCR_FORCE_CPU`, `MATH_OCR_DEC_CPU` (atoi) | PERFORMANCE "DBNet degenerate-component fallback (2026-07-31, Apple M1 Metal)" |
| PP-OCRv6 detector | `ppocrv6_det.cpp` | `kind` | `slower` (1693 ms Metal vs 187 ms CPU, 2026-08-04) | `CRISPEMBED_PPOCRV6_DET_GPU` (tri-state), `CRISPEMBED_PPOCRV6_DET_GPU_LOAD`, `CRISPEMBED_PPOCRV6_FORCE_CPU` (presence) | `ocr_backend_matrix.md` PP-OCRv6 row |
| PP-OCRv6 recognizer | `ppocrv6_ocr.cpp` | `GPU` | `exact` (26/26 fixtures byte-identical to scalar, 2026-08-02/04) | `CRISPEMBED_PPOCRV6_NO_GRAPH`, `..._BATCH_GRAPH`, `..._FORCE_CPU` (presence) | PERFORMANCE "Issue #45 ... (M1, 2026-08-09)", "PP-OCR rec Metal batch profile" |
| PP-LCNet orientation | `pplcnet_orientation.cpp` | `gated` | `wrong` on 1/10 fixtures (final-logit drift 1.07/3.22) and `slower` (1.15 s vs 0.36 s), 2026-08-01 | `PPLCNET_ORIENTATION_GRAPH`, `..._GRAPH_PIPELINE`, `..._GRAPH_ACCEPT`, `..._GRAPH_CPU` | `ocr_backend_matrix.md` PP-LCNet row |
| Tesseract-LSTM | `tesseract_lstm.cpp` | `none` (hand-written LSTM; GPU backend only ever used to read the GGUF) | n/a | `CRISPEMBED_TESSERACT_GPU_LOAD`, `CRISPEMBED_TESSERACT_FORCE_CPU` (presence) | PERFORMANCE "Cost of a GPU backend an engine never computes on" |
| EasyOCR CRNN recognizer | `easyocr_ocr.cpp` | `GPU` | `unknown` vs CPU (per-stage diff ≥0.99972 vs Python) | `EASYOCR_FORCE_CPU` (presence), `EASYOCR_WIDTH_SORT` | PERFORMANCE "EasyOCR GGML parity benchmarks — Apple M1, 2026-08-01" |
| EasyOCR CRAFT detector (not on the pipeline path) | `easyocr_craft.cpp` | `GPU` | `exact` (CPU-forced and Metal byte-identical) but `slower` than CPU (850 ms vs 396 ms Python CPU) 2026-08-01 | `EASYOCR_CRAFT_FORCE_CPU` (presence) | PERFORMANCE "EasyOCR GGML parity benchmarks" |
| Surya detector | `surya_det.cpp` | `GPU*` (stage-3 LiteMLA CPU-scalar) | `unknown` | `SURYA_DET_FORCE_CPU` (value-parsed), `SURYA_DET_SCALAR` | none Metal |
| PARSeq | `parseq_ocr.cpp` | `GPU` | `unknown` | `PARSEQ_OCR_FORCE_CPU` (atoi) | PERFORMANCE M1 sweep 2026-07-31 (smoke only) |
| TPS localisation net | `tps_locnet.cpp` | `wload` | n/a | `TPS_LOCNET_GPU_LOAD`, `TPS_LOCNET_FORCE_CPU` | — |
| Layout (D-FINE/heron) | `layout_detect.cpp` | excluded (other agent) | — | — | — |

## 3. VLM / document OCR engines

| Engine | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| GOT-OCR2 | `got_ocr.cpp` | `GPU*` (window partition CPU) | `unknown` vs CPU; identical OCR across quants on Metal | `GOT_OCR_FORCE_CPU` (atoi) | `docs/got-ocr2.md`, PERFORMANCE "macOS Metal" GOT table |
| GLM-OCR | `glm_ocr.cpp` | `GPU` | `wrong` in vision (amplifying drift; decoded text moves on the hardest fixture), 2026-08-07 | `GLM_OCR_FORCE_CPU` (faithful arm), `GLM_OCR_VISION_FLASH`, `..._F16MM`, `..._BAKE_F32` | PERFORMANCE "GLM-OCR vision on Metal ... (Apple M1, 2026-08-07)" |
| InternVL2 / H2OVL | `internvl2_ocr.cpp` | `GPU*` (host pixel-unshuffle) | `budget` (same text Metal vs CPU, diff-harness gap only) | `INTERNVL2_OCR_FORCE_CPU` (atoi) | PERFORMANCE InternVL2 Metal/CPU stage table |
| DeepSeek-OCR2 | `deepseek_ocr2.cpp` | `GPU*` (window partition CPU; opt-in CPU MoE/SAM conv/LM head) | `budget` (CPU-vs-Metal table-cell whitespace only; same CER both Metal arms), 2026-08-05 | `DS2_FORCE_CPU`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU` | `tests/results/ds-gates/SUMMARY.md`, `tests/results/g6/SUMMARY.md` |
| Unlimited-OCR | `unlimited_ocr.cpp` | `GPU*` | `unknown` vs CPU (Metal run CER 0.010, 2026-07-31) | `UOCR_MOE_CPU`, `UOCR_SAM_CONV_CPU`, `UOCR_LMHEAD_CPU` | `tests/results/uocr-gates/`, PERFORMANCE "Full local matrix" |
| Granite-Vision | `granite_vision_ocr.cpp` | `GPU` (direct `ggml_backend_dev_by_type`, bypasses `--gpu-backend`) | `unknown` | `CRISPEMBED_GRANITE_CPU` (presence) | PERFORMANCE O5 VLM decode graph |
| SmolDocling | `smoldocling_ocr.cpp` | `GPU*` (SigLIP vision on GPU; connector/LLM/KV CPU) — **the old matrix's "No" is stale since 2026-08-05** | `budget` (CPU arm byte-identical; Metal F16 rounding moves one trajectory), 2026-08-05 | `SMOLDOCLING_FORCE_CPU` (atoi) | `tests/results/g1/SUMMARY.md` |
| LFM2.5-VL | `lfm2_vl_ocr.cpp` | `GPU` | `unknown` vs CPU (Metal transcript byte-identical to llama.cpp on one receipt) | `LFM2_VL_FORCE_CPU`, many feature gates | `docs/lfm2_vl/PLAN.md` |
| LightOnOCR | `lightonocr.cpp` | `gated` | `unknown` | `CRISPEMBED_LIGHTONOCR_GPU`, `CRISPEMBED_LIGHTONOCR_FORCE_CPU` | — |
| Pix2Struct | `pix2struct.cpp` | `kind` | `exact` decode graph on Metal (byte-identical) but `slower` at nt1, 2026-08-07 | `CRISPEMBED_PIX2STRUCT_ENC_GPU`, `CRISPEMBED_PIX2STRUCT_GGML_DECODE` | PERFORMANCE "O7 ppformulanet-l ... pix2struct" |
| LightOnOCR-3 / OvisOCR2 / OvisOCR3 (Qwen3.5) | `qwen35_ocr.cpp` and `qwen35_*.cpp` | `CPU` | n/a | `CRISPEMBED_OVISOCR2_*`, `CRISPEMBED_OVISOCR3_*` | `tests/results/qwen35-ocr/` (CPU F16 qualified) |
| MiniCPM-OCR | `minicpm_ocr.cpp` | `CPU` | n/a | `CRISPEMBED_MINICPM_OCR` | `tests/results/minicpm-ocr/` |
| Qwen2-VL / Qwen3-VL / olmOCR / TeleOCR | `qwen2vl_ocr.cpp` | excluded (other agent) | — | — | — |
| MinerU2.5-Pro, Marker, OXR | `mineru_*.cpp`, `marker*.cpp`, `oxr_ocr.cpp` | excluded (other agents) | — | — | — |

## 4. Formula, music and specialist recognizers

| Engine | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| PP-FormulaNet (base) | `ppformulanet_ocr.cpp` | `CPU` | n/a | — | — |
| PP-FormulaNet-L | `ppformulanet_l_ocr.cpp` | `GPU` | `unknown` | `PPFNL_OCR_FORCE_CPU` (atoi) | — |
| MixTeX | `mixtex_ocr.cpp` | `CPU` | n/a | — | — |
| TexTeller / Pix2TeX / TrOCR-math | `math_ocr.cpp` | `GPU` | `unknown` | `MATH_OCR_FORCE_CPU`, `MATH_OCR_DEC_CPU` | M1 sweep (smoke) |
| HMER / BTTR / PosFormer | `hmer_ocr.cpp`, `bttr_ocr.cpp`, `posformer_ocr.cpp` | `CPU` | n/a | — | `tests/results/posformer/` |
| FLOVA | `flova_ocr.cpp` | `CPU` | n/a | — | — |
| SMT / SMT++ | `smt_ocr.cpp` | `GPU` | `unknown` | `SMT_OCR_FORCE_CPU` (atoi) | — |
| Polyphonic-TrOMR | `tromr_ocr.cpp` | `GPU` | `unknown` | `TROMR_OCR_FORCE_CPU` (atoi) | — |
| Transcoda | `transcoda_ocr.cpp` | `GPU` | `unknown` | `TRANSCODA_OCR_FORCE_CPU` (atoi) | — |

## 5. NER, KIE, punctuation, segmentation

| Engine | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| GLiNER | `gliner_ner.cpp` | `GPU` | `unknown` | `CRISPEMBED_FORCE_CPU` | — |
| BERT-NER | `bert_ner.cpp` | encoder through the `crispembed` context (`GPU`); classifier head `wload` (host argmax) | `unknown` | `BERT_NER_GPU_LOAD` (presence), `BERT_NER_FORCE_CPU` | — |
| LiLT KIE | `lilt_kie.cpp` | `GPU` | `unknown` | `LILT_KIE_FORCE_CPU` (atoi) | — |
| FireRedPunc | `fireredpunc.cpp` | `GPU` | `unknown` | `FIREREDPUNC_BACKEND=cpu|gpu` | — |
| PCS punctuation/case | `pcs.cpp` | `GPU` | `unknown` | `PCS_FORCE_CPU` (presence) | — |
| SAM 2.1 | `sam2.cpp` | `GPU` | `unknown` | `SAM2_FORCE_CPU` (value-parsed) | — |
| Text LID / truecaser | `crispembed.cpp` text-LID | `none` | n/a | — | — |

## 6. Super-resolution, restoration, cleanup

| Engine | Source | Backend | Metal status | Gates | Evidence |
|---|---|---|---|---|---|
| NAFNet denoise | `nafnet_denoise.cpp` | `GPU` (fused-block graph on `init_best`) | `unknown` | `NAFNET_CPU`, `NAFNET_FORCE_CPU` (presence/atoi) | — |
| SwinIR | `swinir_sr.cpp` | `wload` (convs on CPU sched, attention scalar) | n/a | `SWINIR_SR_FORCE_CPU`, `SWINIR_SR_SCALAR` | PERFORMANCE backlog P2 "SR-on-GPU" |
| HAT | `hat_sr.cpp` | `wload` | n/a | `HAT_SR_FORCE_CPU`, `HAT_SR_SCALAR` | same |
| DAT | `dat_sr.cpp` | `wload` (scalar default; `DAT_SR_GGML_CONV` CPU sched) | n/a | `DAT_SR_GGML_CONV` | same |
| PAN | `pan_sr.cpp` | `wload` | n/a | `PAN_SR_FORCE_CPU` | same |
| SCUNet | `scunet_denoise.cpp` | `wload` | n/a | `SCUNET_FORCE_CPU`, `SCUNET_SCALAR` | same |
| TBSRN | `tbsrn_sr.cpp` | `wload` | n/a | `TBSRN_SR_FORCE_CPU`, `TBSRN_SR_SCALAR` | same |
| Text-SR (NAFNet-SR) | `text_sr.cpp` | `wload` | n/a | `TEXT_SR_GPU_LOAD`, `TEXT_SR_FORCE_CPU` | — |
| AdaIR | `adair.cpp` | `wload` | n/a | `ADAIR_FORCE_CPU`, `ADAIR_SCALAR` | — |
| SAFMN | `safmn_sr.cpp` | `gated` (one fused graph) | `slower` (tiny model) | `SAFMN_SR_METAL` (presence) | `safmn_sr.cpp` header note |
| ESRGAN | `esrgan_sr.cpp` | `CPU` | n/a | — | `docs/esrgan_sr.md` |
| Restormer | `restormer.cpp` | `CPU` | n/a | — | — |
| InstructIR | `instructir.cpp` | `CPU` | n/a | — | `docs/instructir.md` |
| MI-GAN / inpaint | `migan.cpp`, `inpaint.cpp` | excluded (other agent) | — | — | — |
| Dewarp, scan cleanup, CC detect, morphology, face align, TPS warp, classical preproc | various | `none` | n/a | — | — |

## Summary of the inventory (before any run)

- **GPU by default:** 30 engines/families. Only five have a Metal-vs-CPU A/B on
  record (PP-OCRv6 rec, EasyOCR CRAFT, InternVL2, DeepSeek-OCR2, SmolDocling);
  one is known wrong (GLM vision).
- **CUDA-only GPU default (`kind`):** DBNet, PP-OCRv6 det, Pix2Struct — Metal
  measured slower.
- **CPU-forced with ggml graphs that ggml-Metal could run:** UltraBERT,
  MoEdern ×3, EmbeddingGemma2 decision, Josev2B, Qwen3.5 OCR family, MiniCPM,
  PP-FormulaNet, MixTeX, HMER/BTTR/PosFormer, FLOVA, ESRGAN, Restormer,
  InstructIR, and the `wload` SR family.
- **No GPU code at all:** Tesseract-LSTM, BERT-NER host compute, text LID,
  classical image processing.

The audit sections below are appended as each engine is measured.

## Acceptance bars (owner policy, 1 November 2026)

[`docs/precision-acceptance.md`](precision-acceptance.md) (feat/teleocr `b3c6440c`):
a path qualifies when its final-output cosine, minimum over cases, exceeds
**F32 > 0.99, F16 > 0.98, Q8_0 > 0.97, Q4_K > 0.94**. The Metal receipts here
compare the Metal arm against the CPU arm of the **same artifact** (and, where
a higher-precision GGUF of the same model was available, both arms against it);
the strict budgets stay diagnostics. "Qualifies" below means the Metal arm
clears the bar against the CPU arm of the same artifact.

## Audit 2026-11-01 (Apple M1, `GGML_METAL=ON` build of `feat/metal-audit`)

Method: [`ab_embed.py`](../tests/results/gpu-audit/metal-m1/ab_embed.py) and
[`ab_cmd.py`](../tests/results/gpu-audit/metal-m1/ab_cmd.py) run each arm as a
separate process of one binary, interleaved; the Metal arm must print `MTL0`
and the CPU arm (`--gpu-backend cpu`) must not. Timings are one-shot CLI wall
clock (load included), second run, on a shared, contended M1: indicative only.

### Embedding / rerank (`crispembed.cpp`, GPU default) — all qualify

| Model | Precision | cos min Metal vs CPU | Metal s | CPU s | Note |
|---|---|---:|---:|---:|---|
| all-MiniLM-L6-v2 | F32 | 0.9999996 | 0.16 | 0.20 | max abs 1.5e-4 |
| all-MiniLM-L6-v2 | Q8_0 | 0.99936 | 0.15 | 0.14 | vs F32 GGUF: Metal closer on 11/12 texts |
| bge-small-en-v1.5 | Q8_0 | 0.99980 | 0.17 | 0.28 | vs F16 GGUF: Metal 0.99990, CPU 0.99964 |
| multilingual-e5-small | Q8_0 | 0.99989 | 0.33 | 0.34 | |
| arctic-embed-m-v2 | Q4_K | 0.99656 | 0.55 | 1.36 | |
| bge-m3 (dense) | IQ4_XS | 0.99670 | 1.20 | 3.72 | iq4_xs artifact carries no sparse/ColBERT heads: those modes not reached |
| Qwen3-Embedding-0.6B | Q8_0 | 0.99943 | 4.60 | 3.71 | Metal slower (Q8_0 mul_mv, docs/metal-q8_0-mul_mv-slow-m1.md) |
| F2LLM-v2-0.6B | F16 | 0.9999997 | — | — | timing contended (41-77 s both arms) |
| F2LLM-v2-0.6B | Q4_K | 0.97738 | 1.28 | 4.23 | clears Q4_K bar; lowest of the set |
| jina-v5-nano | F16 | 0.9999979 | 0.63 | 1.31 | |
| jina-v5-nano | Q4_K | 0.99310 | 0.54 | 1.15 | |
| harrier-270m | Q4_K | 0.99929 | 0.72 | 1.22 | |
| EmbeddingGemma-300m | Q8_0 | 0.99991 | 0.51 | 0.92 | |
| LFM2.5-Embedding-350M | Q8_0 | 0.99983 | 1.99 | 1.72 | Metal slower (Q8_0) |
| ms-marco-MiniLM-L-6 (rerank) | F32 | score \|Δ\| 4.7e-4 | 0.15 | 0.40 | same order |
| ms-marco-MiniLM-L-6 (rerank) | Q8_0 | \|Δ\| 0.011 | 0.15 | 0.12 | vs F32: Metal 0.024, CPU 0.035 |
| mxbai-rerank-xsmall | Q8_0 | \|Δ\| 0.027 | 13.4 | 16.2 | |
| ettin-reranker-17m | F32 | \|Δ\| 0.0022 | 0.27 | 0.27 | |
| ettin-reranker-17m | Q8_0 | \|Δ\| 0.10 (logit ~10) | 0.13 | 0.09 | vs F32: Metal 0.072, CPU 0.165; same order |

Finding: on float weights Metal equals CPU to ~1e-6 cosine. On quantized
weights the gap is **not a Metal defect**: ggml-cpu quantizes the activation
side of every Q8_0/K-quant dot product to Q8, Metal keeps it F32, and every
comparison against a higher-precision reference put the Metal arm closer.
No default changes: GPU stays the embedding default. Reference artifacts that
did not match their quantized sibling (both arms equally far: MiniLM text 6
0.973, jina-v5-nano F16 vs Q4_K ~0.87-0.96, the local F2LLM "F16" file
~0.0) are recorded in the receipts as artifact mismatches, not backend results.

### CPU-only controls on hosted CI (run 41327259125, source `81eb3a68`)

`tools/ci-heavy/gpu_audit_cpu_controls.sh`: DBNet with
`CRISPEMBED_DBNET_DECONV_MATMUL=1` produces byte-identical OCR JSON (boxes,
text, confidences) to the default `conv_transpose_2d` on fox, scan_page_pd and
scan_strip; ESRGAN and InstructIR default output is byte-identical with the
new gates unset vs `=0`. Receipt: `tests/results/gpu-audit/hosted-cpu-41327259125/`.
