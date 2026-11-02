# GPU support matrix (every engine, Metal and CUDA)

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

| Engine / family | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| Encoder embedders (BERT, XLM-R, ModernBERT, NomicBERT, GTE, E5, BGE-M3 dense, MiniLM, arctic, granite-embedding) and cross-encoder rerankers | `crispembed.cpp` | `GPU` | **`exact`** on float weights (MiniLM F32 cos 0.9999996, ms-marco F32 rerank \|Δ\| 4.7e-4); **`budget`** on quantized weights (q8/q4 cos 0.9966-0.99989 Metal vs CPU — the CPU arm quantizes activations to q8 for its dot products, Metal keeps them F32; where a matching F32 reference exists Metal is the CLOSER arm: bge-small 0.99990 vs 0.99964, ms-marco \|Δ\| 0.024 vs 0.035), 2026-11-01. Ettin (ModernBERT) q8 rerank \|Δ\| 0.10, same order — see §Audit | **qualifies** (2026-11-01/02): F32/F16 cos ≥ 0.9999989 vs the CPU F32 reference; Q8_0 0.9729-0.99999; Q4_K/IQ4 0.9444-0.99994, CUDA within ±0.004 of the CPU arm against the same reference. BGE-M3 **sparse 1.00000 / ColBERT 0.99918** (F16 with heads). Below the Q4_K bar on BOTH arms (artifact): clip-text-base-q4_k 0.933/0.925; our bge-m3 q8_0 ColBERT 0.593/0.594. CUDA 2-10x faster from ~100M params; tiny models faster on CPU (load); receipts R1, R2, R3 `embed-*` | `CRISPEMBED_FORCE_CPU` (value-parsed), `--gpu-backend`, `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB` | `tests/results/gpu-audit/metal-m1/embed/*.json` |
| Decoder embedders (Qwen3-Embedding, F2LLM, harrier, jina-v5, EmbeddingGemma) | `decoder_embed.cpp` via `crispembed.cpp` | `GPU` | **`exact`** on float weights (F2LLM-0.6B cos 0.9999997, jina-v5-nano 0.9999979); **`budget`** quantized (Qwen3-Emb Q8 0.99943, harrier q4_k 0.99929, EmbeddingGemma Q8 0.99991, jina q4_k 0.99310, F2LLM q4_k 0.97738 — same activation-quantization cause), 2026-11-01. Metal 1.7-3.3x faster steady-state except Q8_0 decoders (Qwen3-Emb 4.6 s vs 3.7 s: known Metal q8_0 mul_mv cost) | **qualifies** except the f2llm-v2-0.6b-q4_k-imatrix artifact (0.782 CUDA / 0.789 CPU vs F32: artifact, both arms). Qwen3-Emb F32/Q8/Q4 1.0000/0.9989/0.9530, jina 1.0000/0.9997/0.9782, harrier 1.0000/0.9996/0.9685, EmbeddingGemma 1.0000/0.9995/0.9501; 2-10x faster (Qwen3 q8 1.4 s vs 10.9 s); stage diff Qwen3 q8 final 0.99923 CUDA / 0.99921 CPU; receipts R1 `embed-*`, R2 `diff-qwen3-embed-q8` | `CRISPEMBED_FORCE_CPU` | `tests/results/gpu-audit/metal-m1/embed/*.json` |
| LFM2.5 embedding / LFM2 encoder | `lfm2_embed.cpp` | `GPU` for the embedding graph; the **official masked-LM encoder path runs only on CPU** (`lfm2_encoder_enabled` requires a CPU backend unless forced) | `budget` for the embedding graph (LFM2.5-Embedding-350M Q8 cos 0.99983, Metal 2.0 s vs CPU 1.7 s — slower, Q8_0), 2026-11-01; encoder path not measured on Metal | embedding graph **qualifies**: F16 1.00000, Q8 0.99975, Q4_K 0.97772 (CPU arm 0.97838); 6-13x faster; stage diff q8 layer_15 0.99956 both arms. Encoder path not measured; receipts R1 `embed-lfm2-*`, R2 `diff-lfm2-embed-q8` | `CRISPEMBED_LFM2_ENCODER` (tri-state, `=1` forces on GPU) | `docs/lfm2-encoder.md` (CPU parity only) |
| BiDirLM omni (text/vision/audio) | `crispembed.cpp`, `bidirlm_vision.cpp`, `bidirlm_audio.cpp` | `GPU` (shares the context backend); audio is host code | `unknown` | `budget`: q8 text embeddings cos 0.99859 (flattened over 12 texts) vs CPU, image 0.99997; stage diff vs Python: text final 0.99909 (CPU 0.99911), vision worst deepstack.1 0.99243 (CPU 0.99207); 2.3 s vs 33 s. Audio not run; receipts R2 `bidirlm-*`, `diff-bidirlm-*` | `CRISPEMBED_FORCE_CPU` | none |
| CLIP text tower | `clip_text_embed.cpp` | `GPU` | `unknown` | **qualifies** F32 1.00000 / Q8 0.99834; Q4_K 0.933 (CPU 0.925, artifact below bar on both); stage diff final 1.000000; receipts R2 `embed-clip-text-*`, `diff-clip-text` | `CLIP_TEXT_FORCE_CPU` (atoi) | none |
| ViT image embedder | `vit_embed.cpp` | `GPU` | `unknown` | **`exact`**: siglip-base JSON cos 1.000000, clip-vit-base-patch16 1.000000; stage diff vs Python 0.99155 both arms; receipts R2 `siglip-base.image`, `clip-vit-b16.image`, `diff-vit-siglip` | `VIT_EMBED_FORCE_CPU` (atoi) | none |
| CNN image/face embedder (`crispembed_face_init`) | `cnn_embed.cpp` | `GPU` | `unknown` | detection **`exact`** (scrfd byte-identical, stage diff 1.0, 0.9 s vs 1.8 s); recognition not run (biometric gate); receipts R2 `scrfd.detect`, `diff-face-scrfd` | `CNN_EMBED_FORCE_CPU` (atoi) | none |
| UltraBERT | `ultrabert_embed.cpp` | `CPU` (no GPU code path) | n/a | n/a (CPU only; no port this audit) | `CRISPEMBED_ULTRABERT` (load gate) | `tests/results/ultrabert/` (CPU only) |
| MoEdern text / family / ColBERT | `moedern_text_engine.cpp`, `moedern_family_engine.cpp`, `moedern_colbert.cpp` | `CPU` | n/a | n/a (CPU only; no port this audit) | `CRISPEMBED_MOEDERN_COLBERT` (load gate) | `tests/results/moedern-colbert/` (CPU only) |
| EmbeddingGemma2 typed decision | `gemma_decision.cpp` | `CPU` | n/a | n/a (CPU only) | — | `tests/results/gemma-decision/` (CPU only) |
| Josev2B composed / SystemOne | `josev2b_composed.cpp`, `systemone_api.cpp` | `CPU` (head has its own `"cpu"` device argument) | n/a | n/a (CPU only) | Josev component gates | `docs/josev2b-head-gpu.md` (head GPU study, Python) |

## 2. Default OCR pipeline, detectors, recognizers

| Engine | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| DBNet detector (`ocr_detect`, default pipeline + tesseract/easyocr lanes) | `ocr_detect.cpp` | `kind` | **`budget`, `slower`** (2026-11-01): Metal det gives the same text, ≤1 px boxes, conf ±0.002; stock Metal `conv_transpose_2d` 10.3 s vs CPU 4.3 s (fox lane). New gate `CRISPEMBED_DBNET_DECONV_MATMUL=1` → Metal lane 2.36 s, output identical to stock Metal, byte-identical on CPU (3 fixtures, hosted); still not faster than the CPU detector on scan_page_pd (64.0 s vs 63.7 s, TrOCR dominates; 295/295 boxes within 1 px of CPU), so the Metal default stays CPU | **`budget`, default stays GPU**: boxes ≤ 1 px, 10/10, 95/95, 295/295; det-on-CPU arm (`OCR_DETECT_USE_GPU=0`) same wall time (TrOCR dominates). `CRISPEMBED_DBNET_DECONV_MATMUL=1` identical output to stock CUDA and not faster (113.7 s vs 114.8 s) → stays off; receipts R1 `dbnet-trocr.*` | `OCR_DETECT_USE_GPU` (tri-state), `OCR_DETECT_FORCE_CPU` (atoi), `CRISPEMBED_DBNET_DECONV_MATMUL` (new, default off) | `tests/results/gpu-audit/metal-m1/ocr/dbnet-*.json`, `tests/results/gpu-audit/hosted-cpu-41327259125/` |
| TrOCR line recognizer (default pipeline; `math_ocr` loader) | `math_ocr.cpp` | `GPU` | **`budget`** (2026-11-01): fox byte-identical; scan_page_pd 295/295 boxes, 290/295 words; scan_strip 92/95 (q8 activation rounding, differences go both ways); Metal 1.3-1.7x faster | **`budget`**: fox and scan_strip texts identical, scan_page_pd 290/295 words (q8 activation rounding, as on Metal); 2.0-2.1x faster (114.8 s vs 241 s page); receipts R1 `dbnet-trocr.*` | `MATH_OCR_FORCE_CPU`, `MATH_OCR_DEC_CPU` (atoi) | `tests/results/gpu-audit/metal-m1/ocr/dbnet-trocr.*.json` |
| PP-OCRv6 detector | `ppocrv6_det.cpp` | `kind` | **`slower`** re-confirmed 2026-11-01: det on Metal 7.37 s vs default 3.21 s (scan_page_pd), char ratio 0.9997 | **`exact`/faster, GPU default confirmed**: 38/38 texts, 2.22 s vs 4.53 s with det on CPU vs 17.2 s all-CPU; receipts R1 `ppocrv6-small.*` | `CRISPEMBED_PPOCRV6_DET_GPU` (tri-state), `CRISPEMBED_PPOCRV6_DET_GPU_LOAD`, `CRISPEMBED_PPOCRV6_FORCE_CPU` (presence) | `tests/results/gpu-audit/metal-m1/ocr/ppocrv6-small-detgpu.scan_page_pd.json` |
| PP-OCRv6 recognizer | `ppocrv6_ocr.cpp` | `GPU` | **`exact`**: byte-identical on fox and scan_page_pd, 3.21 s vs 5.38 s (2026-11-01) | **`exact`** (byte-identical fox, 38/38 scan_page_pd). Stage diff vs the committed fox ref fails on BOTH arms at head_input (0.187): ref fixture/artifact mismatch, not CUDA; receipts R1, R2 `diff-ppocrv6-small-rec` | `CRISPEMBED_PPOCRV6_NO_GRAPH`, `..._BATCH_GRAPH`, `..._FORCE_CPU` (presence) | `tests/results/gpu-audit/metal-m1/ocr/ppocrv6-small.*.json` |
| PP-LCNet orientation | `pplcnet_orientation.cpp` | `gated` | `wrong` on 1/10 fixtures (final-logit drift 1.07/3.22) and `slower` (1.15 s vs 0.36 s), 2026-08-01 | `unknown` (not run) | `PPLCNET_ORIENTATION_GRAPH`, `..._GRAPH_PIPELINE`, `..._GRAPH_ACCEPT`, `..._GRAPH_CPU` | `ocr_backend_matrix.md` PP-LCNet row |
| Tesseract-LSTM | `tesseract_lstm.cpp` | `none` (hand-written LSTM; GPU backend only ever used to read the GGUF) | n/a | n/a | `CRISPEMBED_TESSERACT_GPU_LOAD`, `CRISPEMBED_TESSERACT_FORCE_CPU` (presence) | PERFORMANCE "Cost of a GPU backend an engine never computes on" |
| EasyOCR CRNN recognizer | `easyocr_ocr.cpp` | `GPU` | **`exact`**: `--ocr-engine easyocr` fox byte-identical (2026-11-01) | **`exact` vs Python**: per-stage logits 0.99998-0.99999 on both arms at widths 200/540/610, page output equals Metal and the M1 CPU. The Kaggle **x86 CPU** arm drops lines (fox 1/2, scan_strip 7/10 regions): recognizer returns empty text after a width rebuild there — an x86 ggml-cpu defect, not CUDA (§CUDA audit); receipts R3 `dbnet-easyocr-iso.*`, R4 `diff-easyocr-*` | `EASYOCR_FORCE_CPU` (presence), `EASYOCR_WIDTH_SORT`, `CRISPEMBED_EASYOCR_DEBUG_REGIONS` (new, diagnostic) | `tests/results/gpu-audit/metal-m1/ocr/dbnet-easyocr.fox.json` |
| EasyOCR CRAFT detector (not on the pipeline path) | `easyocr_craft.cpp` | `GPU` | `exact` (CPU-forced and Metal byte-identical) but `slower` than CPU (850 ms vs 396 ms Python CPU) 2026-08-01 | `unknown` (not on the pipeline path, not run) | `EASYOCR_CRAFT_FORCE_CPU` (presence) | PERFORMANCE "EasyOCR GGML parity benchmarks" |
| Surya detector | `surya_det.cpp` | `GPU*` (stage-3 LiteMLA CPU-scalar) | `unknown` | **fixed, `exact`**: every CUDA run aborted (`GGML_ASSERT(ggml_is_contiguous(src0))`, relu on non-contiguous LiteMLA Q/K views); after the fix F16/Q8 byte-identical, 7.0 s vs 18.5 s; receipts R3 `surya-det*` (rc −6), R4 `surya-det*` | `SURYA_DET_FORCE_CPU` (value-parsed), `SURYA_DET_SCALAR` | none Metal |
| PARSeq | `parseq_ocr.cpp` | `GPU` | **`exact`**: parseq and parseq-tiny byte-identical (2026-11-01); one-line crops are faster on CPU (0.07 vs 0.11 s) | **`exact`** (parseq, parseq-tiny byte-identical); one line is faster on CPU (0.21-0.42 s vs 0.72-0.87 s, CUDA init); receipts R2 `parseq*.fox` | `PARSEQ_OCR_FORCE_CPU` (atoi) | `tests/results/gpu-audit/metal-m1/ocr/parseq*.json` |
| TPS localisation net | `tps_locnet.cpp` | `wload` | n/a | n/a | `TPS_LOCNET_GPU_LOAD`, `TPS_LOCNET_FORCE_CPU` | — |
| Layout (D-FINE/heron) | `layout_detect.cpp` | excluded (other agent) | — | excluded | — | — |

## 3. VLM / document OCR engines

| Engine | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| GOT-OCR2 | `got_ocr.cpp` | `GPU*` (window partition CPU) | **decoded `exact`** on fox, Q4_K, 128-token cap (15.4 s vs 14.7 s, tie), 2026-11-01; logits cosine not captured | decoded **`exact`**: q4_k/q8_0/F16 byte-identical CUDA vs CPU, 2.2 s vs 22-39 s; stage diff q4_k worst llm_layer_0 0.99508 (CPU 0.99516). The **F16 artifact decodes `colorcolor…` on BOTH arms** (artifact/engine defect, not CUDA); receipts R2 `got-ocr2-*`, `diff-got-ocr2-q4k`, R3 `got-ocr2-q8.fox` | `GOT_OCR_FORCE_CPU` (atoi); runs need `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1` | `tests/results/gpu-audit/metal-m1/vlm/got-ocr2-q4k.fox.json` |
| GLM-OCR | `glm_ocr.cpp` | `GPU` | `wrong` in vision (amplifying drift; decoded text moves on the hardest fixture), 2026-08-07 | decoded **`exact`** (q8 fox + receipt, F16 fox byte-identical; 2.3-5.3 s vs 12-30 s). Stage diff vs the Python ref fails on BOTH arms with the same drift (vis_post_norm 0.9648 CUDA / 0.9642 CPU, llm_layer_1 0.913 both): CUDA tracks CPU; the Metal amplification is Metal-specific; receipts R2 `glm-ocr-*`, `diff-glm-ocr-q8` | `GLM_OCR_FORCE_CPU` (faithful arm), `GLM_OCR_VISION_FLASH`, `..._F16MM`, `..._BAKE_F32` | PERFORMANCE "GLM-OCR vision on Metal ... (Apple M1, 2026-08-07)" |
| InternVL2 / H2OVL | `internvl2_ocr.cpp` | `GPU*` (host pixel-unshuffle) | **decoded `exact`**: H2OVL-800M F16 fox identical, 98.6 s vs 165.3 s (2026-11-01) | decoded **`exact`**: H2OVL-800M q8 byte-identical, 2.7 s vs 103 s; receipts R2 `h2ovl-800m-q8.fox` | `INTERNVL2_OCR_FORCE_CPU` (atoi) | `tests/results/gpu-audit/metal-m1/vlm/h2ovl-800m-f16.fox.json`, PERFORMANCE InternVL2 table |
| DeepSeek-OCR2 | `deepseek_ocr2.cpp` | `GPU*` (window partition CPU; opt-in CPU MoE/SAM conv/LM head) | `budget` (CPU-vs-Metal table-cell whitespace only; same CER both Metal arms), 2026-08-05 | `unknown` (not in this audit's scope list) | `DS2_FORCE_CPU`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU` | `tests/results/ds-gates/SUMMARY.md`, `tests/results/g6/SUMMARY.md` |
| Unlimited-OCR | `unlimited_ocr.cpp` | `GPU*` | `unknown` vs CPU (Metal run CER 0.010, 2026-07-31) | `unknown` (not run) | `UOCR_MOE_CPU`, `UOCR_SAM_CONV_CPU`, `UOCR_LMHEAD_CPU` | `tests/results/uocr-gates/`, PERFORMANCE "Full local matrix" |
| Granite-Vision | `granite_vision_ocr.cpp` | `GPU` (direct `ggml_backend_dev_by_type`, **ignores `--gpu-backend cpu`**: use `CRISPEMBED_GRANITE_CPU=1`) | **decoded `exact`**: q4_k fox identical, 112.8 s vs 132.4 s (2026-11-01) | decoded **`exact`**: q4_k/q8_0 byte-identical, 4.1-6.0 s vs 62-70 s. Stage diff q8: CUDA worst stage_7 0.913 vs CPU 0.950 (the last three stages drift more on CUDA; decoded text unaffected) — diagnostic; receipts R2 `granite-vision-*`, `diff-granite-vision-q8` | `CRISPEMBED_GRANITE_CPU` (presence) | `tests/results/gpu-audit/metal-m1/vlm/granite-vision-q4k.fox.json` |
| SmolDocling | `smoldocling_ocr.cpp` | `GPU*` (SigLIP vision on GPU; connector/LLM/KV CPU) — the old matrix's "No" is stale since 2026-08-05 | `budget` (2026-08-05). 2026-11-01 re-run not reached: the only local artifact is a pre-2026-08-04 GGUF without the `<row_r_col_c>` tokens and its Metal arm crashed (rc -11) before any CPU arm; replacing it was skipped (disk under 15 GB) | **`exact`** F16 (byte-identical, 6.2 s vs 30.6 s); q8 one location token differs (`loc_128` vs `loc_127`), text identical; receipts R2 `smoldocling-*` | `SMOLDOCLING_FORCE_CPU` (atoi) | `tests/results/g1/SUMMARY.md`, `tests/results/gpu-audit/metal-m1/vlm_batch.log` |
| LFM2.5-VL | `lfm2_vl_ocr.cpp` | `GPU` | `unknown` vs CPU (Metal transcript byte-identical to llama.cpp on one receipt) | `unknown` (not run) | `LFM2_VL_FORCE_CPU`, many feature gates | `docs/lfm2_vl/PLAN.md` |
| LightOnOCR | `lightonocr.cpp` | `kind` (GPU when a CUDA device is present, since 2026-11-02; CPU on Metal/CPU) | **decoded `exact`** but **`slower`**: 1B q4_k fox identical, 18.6 s Metal vs 6.4 s CPU (2026-11-01) — stays opt-in | **CUDA default since 2026-11-02**: 1B q4_k fox/receipt byte-identical, scan_strip char ratio 0.963 at 256 tokens (CPU and `=0` arms byte-identical); 1.5-3.4 s vs 17-41 s; receipts R2, R3 `lightonocr-*` | `CRISPEMBED_LIGHTONOCR_GPU` (tri-state since 2026-11-02: =1 any GPU, =0 CPU, unset = CUDA only), `CRISPEMBED_LIGHTONOCR_FORCE_CPU` | `tests/results/gpu-audit/metal-m1/vlm/lightonocr-1b-q4k-gpu.fox.json` |
| Pix2Struct | `pix2struct.cpp` | `kind` | `exact` decode graph on Metal (byte-identical) but `slower` at nt1, 2026-08-07 | `exact` (CUDA default since 2026-08-08, prior kernel; not re-run) | `CRISPEMBED_PIX2STRUCT_ENC_GPU`, `CRISPEMBED_PIX2STRUCT_GGML_DECODE` | PERFORMANCE "O7 ppformulanet-l ... pix2struct" |
| LightOnOCR-3 / OvisOCR2 / OvisOCR3 (Qwen3.5) | `qwen35_ocr.cpp` and `qwen35_*.cpp` | `CPU` | n/a | n/a (CPU only) | `CRISPEMBED_OVISOCR2_*`, `CRISPEMBED_OVISOCR3_*` | `tests/results/qwen35-ocr/` (CPU F16 qualified) |
| MiniCPM-OCR | `minicpm_ocr.cpp` | `CPU` | n/a | n/a (CPU only) | `CRISPEMBED_MINICPM_OCR` | `tests/results/minicpm-ocr/` |
| Qwen2-VL / Qwen3-VL / olmOCR / TeleOCR | `qwen2vl_ocr.cpp` | excluded (other agent) | — | excluded | — | — |
| MinerU2.5-Pro, Marker, OXR | `mineru_*.cpp`, `marker*.cpp`, `oxr_ocr.cpp` | excluded (other agents) | — | excluded | — | — |

## 4. Formula, music and specialist recognizers

| Engine | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| PP-FormulaNet (base) | `ppformulanet_ocr.cpp` | `CPU` | n/a | n/a (CPU only) | — | — |
| PP-FormulaNet-L | `ppformulanet_l_ocr.cpp` | `GPU` | `unknown` | decoded **`exact`**: q8/F16 byte-identical; only 1.1-1.3x (43.8 s vs 50 s); receipts R2 `ppformulanet-l-*` | `PPFNL_OCR_FORCE_CPU` (atoi) | — |
| MixTeX | `mixtex_ocr.cpp` | `CPU` | n/a | n/a (CPU only) | — | — |
| TexTeller / Pix2TeX / TrOCR-math | `math_ocr.cpp` | `GPU` | `unknown`: TexTeller-3 q8 decodes garbage on formula_quadratic in **both** arms (known unusable on this fixture), no verdict | `unknown` (not run) | `MATH_OCR_FORCE_CPU`, `MATH_OCR_DEC_CPU` | `tests/results/gpu-audit/metal-m1/ocr/texteller.formula.json` |
| HMER / BTTR / PosFormer | `hmer_ocr.cpp`, `bttr_ocr.cpp`, `posformer_ocr.cpp` | `CPU` | n/a | n/a (CPU only; no port this audit) | — | `tests/results/posformer/` |
| FLOVA | `flova_ocr.cpp` | `CPU` | n/a | n/a (CPU only) | — | — |
| SMT / SMT++ | `smt_ocr.cpp` | `GPU` | `unknown` | **`exact`**: F32/q8 byte-identical, 1.7 s vs 2.5-2.7 s; receipts R2 `smt-*` | `SMT_OCR_FORCE_CPU` (atoi) | — |
| Polyphonic-TrOMR | `tromr_ocr.cpp` | `GPU` | `unknown` | **`exact`**: F32/q8 byte-identical, 2.0 s vs 3.8 s; receipts R2 `tromr-*` | `TROMR_OCR_FORCE_CPU` (atoi) | — |
| Transcoda | `transcoda_ocr.cpp` | `GPU` | `unknown` | **`exact`**: q8 byte-identical, 2.2 s vs 24.2 s; receipts R2 `transcoda-q8.page` | `TRANSCODA_OCR_FORCE_CPU` (atoi) | — |

## 5. NER, KIE, punctuation, segmentation

| Engine | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| GLiNER | `gliner_ner.cpp` | `GPU` | `unknown` | `budget`: same entities, scores \|Δ\| ≤ 0.006 (DeBERTa F32 byte-identical); receipts R2 `gliner-*` | `CRISPEMBED_FORCE_CPU` | — |
| BERT-NER | `bert_ner.cpp` | encoder through the `crispembed` context (`GPU`); classifier head `wload` (host argmax) | `unknown` | `unknown` (not run) | `BERT_NER_GPU_LOAD` (presence), `BERT_NER_FORCE_CPU` | — |
| LiLT KIE | `lilt_kie.cpp` | `GPU` | `unknown` | **`exact`**: 24 stages 1.000000 vs Python on both arms; receipts R2 `diff-lilt` | `LILT_KIE_FORCE_CPU` (atoi) | — |
| FireRedPunc | `fireredpunc.cpp` | `GPU` | `unknown` | **`exact`** (byte-identical on the same OCR text, F32 and q8); receipts R3 `punct-firered-*` | `FIREREDPUNC_BACKEND=cpu\|gpu` | — |
| PCS punctuation/case | `pcs.cpp` | `GPU` | `unknown` | `budget`: q8 on identical input differs in 3 punctuation decisions (char ratio 0.9963); receipts R3 `punct-pcs-q8.scan_strip` | `PCS_FORCE_CPU` (presence) | — |
| SAM 2.1 | `sam2.cpp` | `GPU` | `unknown` | **`exact`**: F32/F16 masks byte-identical (IoU 1.0), 1.1 s vs 5.2-9.6 s; receipts R2 `sam2-*` | `SAM2_FORCE_CPU` (value-parsed) | — |
| Text LID / truecaser | `crispembed.cpp` text-LID | `none` | n/a | n/a | — | — |

## 6. Super-resolution, restoration, cleanup

| Engine | Source | Backend | Metal status | CUDA (Kaggle T4) | Gates | Evidence |
|---|---|---|---|---|---|---|
| NAFNet denoise | `nafnet_denoise.cpp` | `GPU` (fused-block graph on `init_best`) | `unknown` | **`exact`** F16 (byte-identical), q8 max 1/255; stage diff 0.999995 both; 3.3 s vs 24.6 s; receipts R2 `nafnet-*`, `diff-nafnet-f16` | `NAFNET_CPU`, `NAFNET_FORCE_CPU` (presence/atoi) | — |
| SwinIR | `swinir_sr.cpp` | `wload` (convs on CPU sched, attention scalar) | n/a | `wload` (compute CPU): stage diff 0.998403 both arms; receipts R2 `diff-swinir` | `SWINIR_SR_FORCE_CPU`, `SWINIR_SR_SCALAR` | PERFORMANCE backlog P2 "SR-on-GPU" |
| HAT | `hat_sr.cpp` | `wload` | n/a | `wload`; stage diff output 0.0 on BOTH arms (reference/harness fault, not CUDA); receipts R2 `diff-hat` | `HAT_SR_FORCE_CPU`, `HAT_SR_SCALAR` | same |
| DAT | `dat_sr.cpp` | `wload` (scalar default; `DAT_SR_GGML_CONV` CPU sched) | n/a | `wload`: stage diff 0.999995 both arms; receipts R2 `diff-dat` | `DAT_SR_GGML_CONV` | same |
| PAN | `pan_sr.cpp` | `kind` (whole-image graph on CUDA since 2026-11-02, gate new); `wload` elsewhere | n/a | **CUDA default since 2026-11-02** (new `CRISPEMBED_PAN_GPU`): max 1/255, PSNR 102 dB, 2.1 s vs 31.6 s. Stage diff vs the committed ref is 0.0 on BOTH arms (ref fixture); receipts R2, R3 `pan-*`, R2 `diff-pan-gpu` | `CRISPEMBED_PAN_GPU` (new, tri-state: =1 any GPU, =0 CPU graph, unset = CUDA only), `PAN_SR_FORCE_CPU` | same |
| SCUNet | `scunet_denoise.cpp` | `wload` | n/a | `wload`: stage diff 1.000000 both arms; receipts R2 `diff-scunet` | `SCUNET_FORCE_CPU`, `SCUNET_SCALAR` | same |
| TBSRN | `tbsrn_sr.cpp` | `wload` | n/a | `wload`: stage diff 0.999985 both arms; receipts R2 `diff-tbsrn` | `TBSRN_SR_FORCE_CPU`, `TBSRN_SR_SCALAR` | same |
| Text-SR (NAFNet-SR) | `text_sr.cpp` | `wload` | n/a | `unknown` (not run) | `TEXT_SR_GPU_LOAD`, `TEXT_SR_FORCE_CPU` | — |
| AdaIR | `adair.cpp` | `wload` | n/a | `wload`: stage diff 0.999846 both arms; receipts R2 `diff-adair` | `ADAIR_FORCE_CPU`, `ADAIR_SCALAR` | — |
| SAFMN | `safmn_sr.cpp` | `gated` (one fused graph) | `slower` (tiny model) | `unknown` (not run) | `SAFMN_SR_METAL` (presence) | `safmn_sr.cpp` header note |
| ESRGAN | `esrgan_sr.cpp` | **`GPU` on Metal (2026-11-01) and CUDA (2026-11-02)**, CPU elsewhere | **`budget`**: max \|Δ\| 1/255, PSNR 69.3-75.7 dB vs the CPU path; 1.9-2.0 s vs 8.4-8.9 s | **CUDA default since 2026-11-02**: max 1/255, PSNR 98-106 dB, 1.5-1.8 s vs 14-14.8 s; `=0` byte-identical to CPU; receipts R2 `esrgan.*`, R3 `esrgan-default.fox` | `CRISPEMBED_ESRGAN_GPU` (tri-state: =1 any GPU, =0 historical CPU, unset = Metal or CUDA) | `tests/results/gpu-audit/metal-m1/ocr/esrgan-*.json` |
| Restormer | `restormer.cpp` | `kind` (GPU when a CUDA device is present, since 2026-11-02; CPU elsewhere) | n/a | **CUDA default since 2026-11-02** (new `CRISPEMBED_RESTORMER_GPU`): max 1/255, PSNR 86-104 dB, 13.0-13.4 s vs 229-235 s; stage diff 0.999992 both arms; receipts R2, R3 `restormer-*`, R2 `diff-restormer-gpu` | `CRISPEMBED_RESTORMER_GPU` (new, tri-state: =1 any GPU, =0 CPU, unset = CUDA only) | — |
| InstructIR | `instructir.cpp` | `kind` (GPU when a CUDA device is present, since 2026-11-02; CPU on Metal/CPU) | `budget` but **`slower`**: max \|Δ\| 1/255 (PSNR 76.7 dB), 14.5 s vs 10.4 s — stays off | **CUDA default since 2026-11-02**: max 1/255, PSNR 104 dB, 3.7 s vs 30.3-32.6 s; receipts R2 `instructir.*`, R3 `instructir-default.*` | `CRISPEMBED_INSTRUCTIR_GPU` (tri-state: =1 any GPU, =0 CPU, unset = CUDA only) | `tests/results/gpu-audit/metal-m1/ocr/instructir-gpu.scan_strip.json` |
| MI-GAN / inpaint | `migan.cpp`, `inpaint.cpp` | excluded (other agent) | — | excluded | — | — |
| Dewarp, scan cleanup, CC detect, morphology, face align, TPS warp, classical preproc | various | `none` | n/a | n/a | — | — |

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

The audit sections below record what was measured on 2026-11-01; every row still marked `unknown` above was NOT reached.

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

## CUDA audit 2026-11-01/02 (Kaggle Tesla T4, sm_75, branch `feat/cuda-audit`)

The `CUDA (Kaggle T4)` column in every table above comes from this audit.
Harness: [`tools/kaggle/cuda-audit/`](../tools/kaggle/cuda-audit/) (kernel
`${KAGGLE_ACCOUNT}/crispembed-cuda-audit`, `stage.py` pins an exact pushed commit).
Each arm is a separate process of the same `GGML_CUDA=ON` build
(`CMAKE_CUDA_ARCHITECTURES=75`, native x86 CPU flags, `-t 4` on 4 Xeon vCPUs):

- **CUDA arm**: `--gpu-backend cuda` (test binaries: GPU visible, engine default);
- **CPU arm**: `--gpu-backend cpu` **and** `CUDA_VISIBLE_DEVICES=""`, so no GPU exists
  and per-kind defaults resolve exactly as on a CPU-only box;
- every arm records peak `nvidia-smi` memory above idle (`cuda_used`), so a
  silently-CPU "CUDA" arm shows up in the receipt instead of passing as parity.

Embedders compare against the CPU arm of the family's F32/F16 GGUF (the owner bar
applies to that reference, by the artifact's recipe label); generators compare
decoded text; images compare pixels (centered cosine, max |Δ|, PSNR); stage diffs
run the committed `test-*-diff` tools against the Python references. Timings are
best-of-two one-shot CLI wall clock, model load included.

Receipts (`tests/results/gpu-audit/cuda-kaggle/`, account name scrubbed):
R1 `run1-8f781770` (23 jobs, then the kernel died in an EasyOCR fetch),
R2 `run2-e0da5d08` (71 jobs), R3 `run3-c38304b4` (default flips verified, isolation
arms), R4 `run4-7d34bbbd` (Surya fix, EasyOCR vs Python), R5 `run5-ccb53204`
(x86 CPU EasyOCR diagnosis). `summarize.py RUN_DIR...` renders them as tables.

### Defaults changed (A/B cleared the bar and was faster; `=0` restores CPU via `core_env::explicitly_off`)

| Engine | Gate | CUDA s | CPU s | Output CUDA vs CPU |
|---|---|---:|---:|---|
| ESRGAN x4 F32 (fox, scan_strip) | `CRISPEMBED_ESRGAN_GPU` (Metal already on) | 1.5-1.8 | 14.0-14.8 | max \|Δ\| 1/255, PSNR 98-106 dB |
| InstructIR F16 (scan_strip) | `CRISPEMBED_INSTRUCTIR_GPU` (Metal stays off: slower) | 3.7 | 30.3-32.6 | max \|Δ\| 1/255, PSNR 104 dB |
| PAN x4 F16 (scan_strip) | `CRISPEMBED_PAN_GPU` (new) | 2.1 | 31.6-33.9 | max \|Δ\| 1/255, PSNR 102 dB |
| Restormer F16 (scan_strip, fox) | `CRISPEMBED_RESTORMER_GPU` (new) | 13.0-13.4 | 229-235 | max \|Δ\| 1/255, PSNR 86-104 dB; stage diff 0.999992 both |
| LightOnOCR 1B q4_k (fox, receipt, scan_strip) | `CRISPEMBED_LIGHTONOCR_GPU` (Metal stays off: slower) | 1.5-3.4 | 17-41 | 2/3 byte-identical, scan_strip char ratio 0.963 |

The per-kind test is `crispasr_gpu_device_kind_present("CUDA")`
(`src/core/gpu_backend_pref.h`); `--gpu-backend cpu` still wins. In each case the
true default arm (no env) ran on the GPU, and `=0` was byte-identical to the CPU arm
(R3).

### Fixed

- **Surya detector aborted on every CUDA run** (rc −6,
  `GGML_ASSERT(ggml_is_contiguous(src0))` in ggml-cuda `unary.cu`): `relu` was
  applied to non-contiguous LiteMLA Q/K views, which ggml-cpu accepts. The views
  are now copied first (identical values on CPU). After the fix, F16 and Q8 are
  byte-identical to CPU and take 7.0 s vs 18.5 s (R3 → R4).

### Findings that are not CUDA defects (same on both arms, or CPU-side)

- **EasyOCR, x86 CPU drops lines.** On the Kaggle x86 CPU arm the page lane
  returns 1/2 regions on fox and 7/10 on scan_strip. CUDA, Metal and an M1 CPU
  build all return the full text. The recognizer itself matches Python on both
  arms (logits 0.99998-0.99999 at widths 200/540/610, R4). The dropped line comes
  back as empty text after a width rebuild (canvas 704 → 576); see R5.
- **GOT-OCR2 F16 artifact** decodes `colorcolor…` on both arms; q4_k/q8_0 are correct.
- **Artifacts below their bar against F32 on both arms**: f2llm-v2-0.6b-q4_k-imatrix
  (0.78/0.79), clip-text-base-q4_k (0.933/0.925), and the bge-m3 q8_0 we quantized
  here (ColBERT 0.59 on both arms; dense/sparse fine).
- The published bge-m3 GGUFs carry **no sparse/ColBERT heads**: the converter's
  default Ollama format drops them, and `--crisp` keeps them.
- Stage diffs that fail identically on both arms (reference fixture or harness
  issue): PP-OCRv6 small rec (head_input 0.187), PAN (0.0), HAT (0.0), GLM-OCR q8
  vision (vis_post_norm 0.965; CUDA tracks CPU, so Metal's drift is Metal-specific).
- Granite-Vision q8 stage diff: CUDA's last three projector/LLM stages drift more
  (0.913 vs 0.950 CPU); decoded text is byte-identical.

### Not reached on CUDA

PP-LCNet, EasyOCR CRAFT, TexTeller/Pix2TeX, BERT-NER, LFM2.5-VL, LFM2 encoder
path, Unlimited-OCR/DeepSeek-OCR2, Text-SR, SAFMN, CNN face recognition
(biometric gate), BiDirLM audio. CPU-only engines with no port this audit:
UltraBERT, MoEdern, Josev2B, EmbeddingGemma2 decision, PP-FormulaNet base,
MixTeX, HMER/BTTR/PosFormer, FLOVA, Qwen3.5-OCR family, MiniCPM. The `wload` SR
family (SwinIR/HAT/DAT/SCUNet/TBSRN/AdaIR) still computes on CPU; their stage
diffs pass on the CUDA build (weights loaded on the GPU), and PAN/Restormer
show the GPU-port pattern (8-18x).
