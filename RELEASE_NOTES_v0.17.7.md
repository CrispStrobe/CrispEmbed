# CrispEmbed v0.17.7

An OCR correctness-and-speed release — 194 commits since v0.17.6, almost all of
it in the OCR lanes. Four engines were decoding **wrong or empty output** and
are fixed. Several defaults changed as a result of measured A/Bs, so read
*Changed defaults* first if you pin recognized text or benchmark against a
previous tag.

Every performance number below is a measurement recorded in `PERFORMANCE.md`
with its fixture, box and arm count. Where a lever was measured and **lost**,
it is listed under *Measured and rejected* rather than quietly dropped.

---

## Changed defaults

These change output or device placement without any env var being set.

| Default | Was | Now | Evidence |
|---|---|---|---|
| Tesseract segmentation router (`CRISPEMBED_TESSERACT_SEG_ROUTER`) | off (dbnet-first) | **on** | Fraktur CER 0.2360 → 0.1988 at ~2.3x the speed |
| Tesseract composed recoding (`CRISPEMBED_TESSERACT_RECODE_COMPOSE`) | off | **auto-on for multi-code (CJK) recoders** | single-code Latin models unchanged; `=0`/`=1` keep absolute precedence |
| Tesseract pageseg legacy rows (`..._LEGACY_BAND_ROWS`) | off | **on** | Fraktur CER 0.271 → 0.218 |
| PP-OCRv6 detector residency | CPU everywhere | **GPU graph on CUDA**, CPU on Metal | det stage 9516 → 595 ms (P100); page total 12.7 → 3.8 s |
| DBNet detector residency | CPU everywhere | **GPU graph on CUDA**, CPU on Metal | 3.12 → 0.53 s on `scan_page_pd`; 295 = 295 boxes, max coord Δ 1.0 px |
| pix2struct decode | scalar CPU loop | **ggml decode graph on CUDA**, scalar elsewhere | decoder ~9-12.8x on P100, decoded text byte-identical in every arm |

The three residency flips are per-backend-*kind*, not global: a Metal or CPU
build behaves exactly as it did in v0.17.6 (verified byte-identical to the
pre-flip binary on an M1 with no CUDA). Each has an explicit opt-out —
`OCR_DETECT_USE_GPU=0`, `CRISPEMBED_PPOCRV6_DET_GPU=0`,
`CRISPEMBED_PIX2STRUCT_GGML_DECODE=0`.

The router flip carries a second fix: it was previously **vetoing explicit
callers**, so `--tesseract-pageseg` did not reliably get the classical route.
It now routes without overriding an explicit request.

---

## Fixed: PP-OCRv6 quantized recognition returned empty text for every crop

Quantized graph-resident weights were uploaded as raw F32 bytes into tensors
declared as `q8_0` blocks. The recognizer then read noise and every crop
decoded to the empty string. This surfaced as a "CUDA gives 0 results" report
but was **never CUDA-specific** — it reproduced on Metal once the graph path
was exercised with a quantized model. Fixed in `c288b816`, proven byte-exact
against the scalar reference on P100 for both fixtures.

## Fixed: pix2struct produced repetition garbage, and then raw token ids

Three independent defects, all ours, not the model's:

- **T5 relative attention bias sign was inverted**, which collapsed every
  decode position to bucket 0. The "known" repetition degeneration was this.
- **Within-patch flatten order was CHW; HF trains HWC.** Encoder parity was
  off for every image. Captions are now HF-exact.
- **The tokenizer the GGUF always carried was never read**, so output came
  back as raw ids instead of text.

The `--fp16` path in the `instructir` / `safmn` / `tps-loc` converters was
also fixed (the `raw_dtype` label-but-don't-convert trap) as part of a
seven-converter audit; the other four were verified safe.

## Fixed: GLM-OCR thin-strip hallucination and runaway repetition

The LM applied mRoPE to NEOX-style split-half pairs on weights trained with
interleaved pairs. Q/K rows are now permuted at load. Both the thin-strip
hallucination and the `-ich` runaway were this one bug — port output is now
text-identical to HF on the strip fixture. Separately, `glm_ocr` was the one
VLM engine decoding with a bare argmax; it now uses the shared
no-repeat-ngram greedy guard like the others.

## Fixed: `UOCR_PD=1` crashed, and four engines ignored `n_threads`

- `UOCR_PD=1` no longer segfaults. The trigger was sched-graph replay; the
  page-detection path re-initialises per step by default now. This also
  required the ggml pin bump (`c76aaacb` → `890278a8`) that makes
  alloc-once/compute-many replay safe.
- `ppocrv6_det` and three more engines silently discarded the `n_threads`
  they were handed and ran single-threaded. Found by audit after the first
  instance, fixed across the bug class.

## Fixed: `CRISPEMBED_WARN_UNK=0` turned the warning *on*

A bare `getenv() != nullptr` gate — the defect class the repo already audited
once. Now uses `core_env::explicitly_off()`.

---

## New

**RAM preflight for large VLM engines.** The big VLMs refuse to load rather
than swap-thrash the host into unusability. Tunable with `CRISPEMBED_RAM_GUARD`
and `CRISPEMBED_RAM_GUARD_AVAILABLE_MB`.

**Page-level CJK for Tesseract.** The PP-OCRv6 detector can now host the
Tesseract recognition stage, giving a working full-page path for CJK
traineddata — there was previously no working page route (whole-page rec-only,
dbnet fragments, or a hardwired generic pipeline).

**Misroute guard.** The CLI warns when a line recognizer is handed a full page
— the single most common way to get plausible-looking nonsense out of a
correctly working model.

**`languages` in the model registry**, populated from dictionary scans and
surfaced in `--list-models`, plus `tools/scan_model_languages.py` which now
also handles embedding and reranker GGUFs.

**`docs/LANGUAGES.md`** — an evidence-based OCR/embedding language matrix,
built on a new committed Japanese fixture
(`tests/regression/images/japanese_print.png` + ground truth). Every
multilingual embedder and all three shipped rerankers are now verified on
Japanese against English-only controls, with reusable harnesses
(`tests/embed_language_eval.py`, `tests/reranker_language_eval.py`). Two
caveats are documented rather than papered over: the reranker table has no
negative control (no English-only reranker exists in the registry), and the
accented-Latin divergence below.

**A `[UNK]`-rate warning** at runtime when more than half of a sequence's
tokens tokenize to `[UNK]` — the signature of a wrong tokenizer or an
unsupported script.

**New opt-in gates** (all default-off, all measured):
`CRISPEMBED_TESSERACT_MIN_REC_CONFIDENCE` (region confidence floor),
`CRISPEMBED_CONV2D_MK` (register-blocked GEMM micro-kernel),
`CRISPEMBED_LAYOUT_VALPROJ_GPU`, `CRISPEMBED_PIX2STRUCT_ENC_GPU`,
`GLM_OCR_VISION_BAKE_F32`.

---

## Faster

All byte-identical output unless noted.

| Change | Effect | Where |
|---|---|---|
| PP-OCRv6 Metal flat-dispatch im2col (was 70% of the rec graph) | **2.3x** recognize, **1.6x** layout_detect | M1 |
| PP-OCRv6 detector auto-CUDA | det stage **16x** (9516 → 595 ms), page 12.7 → 3.8 s | P100 |
| DBNet detector auto-CUDA | **~6x** det-only (3.12 → 0.53 s) | P100 |
| pix2struct ggml decode graph, device-resident KV | decoder **10.6x** q8_0, **12.8x** f16 | P100 |
| `conv2d` micro-kernel on the PP-OCR det scalar path | **−26%** process CPU (M1: −34%) | M1 + x86 |
| `conv2d` micro-kernel on the HMER coverage-attention path | **−25.7%** process CPU | M1 |
| `conv2d` micro-kernel on the ppformulanet-l neck/proj block | **−31%** stage (464 → 309 ms) | M1 |
| layout Phase 2: level input projection (the real 64% hotspot, not the deform loop) → AXPY + threaded | **2.7x** Phase 2 | M1 |
| layout value projections batched into one GPU graph (opt-in) | **3.5x** stage, Phase 2 −35% | M1 |
| pix2struct threaded `lm_head` | **−11%** | M1 |
| Tesseract segmentation router default | **2.3x** on the Fraktur page | M1 |

### Fraktur page quality, cumulative

`german_official_print.jpg`, `frk`:

| Step | CER | WER |
|---|---|---|
| v0.17.6 shipped default (dbnet-first) | 0.2360 | 0.4043 |
| classical pageseg route, as it stood in v0.17.6 | 0.412 | — |
| + reject separator rows before recognition | 0.271 | — |
| + band-clustered legacy rows, rise-gated widening | 0.218 | — |
| **v0.17.7 shipped default** (router → classical + cleanup) | **0.1988** | 0.4113 |
| + `CRISPEMBED_TESSERACT_MIN_REC_CONFIDENCE=0.5` (opt-in) | **0.1489** | 0.3333 |

The two middle rows are the classical route improving; the router flip is what
makes that route the default. Note the router arm's WER is slightly *worse*
than the dbnet-first arm (0.4113 vs 0.4043) while CER is much better — it
recovers characters in more regions (22 vs 21, 1111 vs 1014 chars).

The confidence floor works by *not emitting* the regions the recognizer itself
scored as junk (22 regions → 18), which is why it stays opt-in.

---

## Measured and rejected

Recorded so nobody re-runs them:

- **GLM vision `FLASH` and F16 matmul** are 30-39% faster on the vision encoder
  and a **quality loss** (char-CER 0.0386 vs 0.0193 against the reference).
  Both stay gated off.
- **Wider recode beams and DAWG scoring make Fraktur worse**, monotonically,
  at 3-52x the cost (CER 0.1959 greedy → 0.2125 at beam 4 → 0.2194 at beam 8;
  `--dawg-score` costs 366 s a page for 0.2116).
- **`LAYOUT_CONV_F16`** shows no win on P100 and drifts regions; stays gated.
- **pix2struct ggml decode on CPU**: the x86 1.65x was threading wall-clock,
  not kernel work. No flip on M1.
- **posformer and pplcnet orientation** micro-kernel flips measured dead flat.
  Not shipped.
- **Vulkan on Kaggle: not possible** — the container ships no NVIDIA Vulkan
  userspace. (The first probe's "successes" were an artifact of laundering
  return codes through `| tail`; v2 is the honest run.)

## Known issues

- **Accented Latin text diverges from HF on uncased WordPiece embedders**
  (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`). HF's `BasicTokenizer` with
  `do_lower_case=True` strips accents before WordPiece lookup (`café` → `cafe`,
  `Müller` → `muller`); CrispEmbed's per-byte lowercase path does not, so
  ordinary German/French/Spanish/Portuguese words fall out of vocabulary and
  become `[UNK]`. This is newly *documented*, not newly introduced — see
  `docs/LANGUAGES.md`. The fix is deferred; the new `[UNK]`-rate warning at
  least makes it visible. Multilingual SentencePiece embedders are unaffected.
- **GLM-OCR vision on Metal diverges from the HF reference.** Bisected to an
  amplification cliff at layer 13, not a broken kernel — a layer taking an
  input matching to 0.19% emits one 18.8% larger. Decoded text is usable;
  strict parity is not yet reached.
- The full-page Fraktur result is still below official Tesseract: fewer
  regions and non-zero CER/WER on the `scan_strip` comparison.

## Assets

Same matrix as v0.17.6 — Linux x86_64 (CPU + CUDA + bundled-CUDA), Linux
arm64, macOS arm64, Windows x86_64 (CPU + CUDA + Vulkan), Android
arm64-v8a/armeabi-v7a, iOS arm64, and the OCR/embed WASM bundles. Every leg
still pins `GGML_NATIVE=OFF` and is checked by `scripts/check-cpu-baseline.py`.
