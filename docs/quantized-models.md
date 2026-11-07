# Quantized models: how far each one is from the original

Owner decision, 7 November 2026: the quantized files below are published, most of them as
**experimental**. This page says, model by model, how experimental each file is and how far its
quality is on par with the original or the F16 file. Every number comes from a receipt committed
under `tests/results/` (or the recipe document it cites). A cell that reads *not measured* has no
receipt behind it. The same tables appear on each Hugging Face model card.

Status labels:

- **recommended**: matches the original exactly on the frozen panel (or is the reference itself).
- **caveat**: usable, with a known limitation stated in the row.
- **experimental**: clears the owner's logit- or vector-cosine bar for its precision tier
  ([precision acceptance](precision-acceptance.md): Q8_0 > 0.97, Q4_K > 0.94), but its decoded
  output or vectors are measurably different from the original. Do not use it where results must
  match exactly.

Embedding models: **do not mix quantized vectors into an index built with F32 vectors** (or the
reverse). Rebuild the index with the precision you query with.

Registry aliases of experimental files end in `-experimental` (except the older OCR aliases noted
below, whose descriptions say so). All aliases are pinned to a Hub commit and SHA-256 in
`examples/cli/model_hashes.h`.

## Overview

| Model | File / alias | Precision | Status | One-line quality summary |
|---|---|---|---|---|
| UltraBERT-Embed-150M | `ultrabert-q4k-experimental` | Q4_K (last four FFN) | experimental | min embedding cosine 0.9779; SciFact nDCG@10 -0.0018 |
| MoEdernColBERT 27M | `moedern-27m-q4k-experimental` | Q4_K_MIX | experimental | min token-vector cosine 0.9685; retrieval not measured |
| MoEdernColBERT 115M | `moedern-115m-q4k-experimental` | Q4_K_MIX | experimental | min token-vector cosine 0.9793; retrieval not measured |
| Josev-2B SystemOne | `josev2b-systemone-q8_0-experimental` | Q8_0 | experimental | option argmax 4/4, typed JSON exact 1/4 |
| Josev-2B SystemOne | `josev2b-systemone-q4k-mix-experimental` | Q4_K_MIX | experimental | option argmax 4/4, typed JSON exact 0/4 |
| OXR-1.0 | `oxr-1.0-q8_0` | Q8_0-LARGE | experimental | 4/5 token-exact; German page diverges at one decode step |
| OXR-1.0 | `oxr-1.0-q4_k-head` | Q4_K-HEAD | experimental | 4/5 token-exact; German page diverges at one decode step |
| GOT-OCR2 | `got-ocr2-q4_k-experimental` | Q4_K | experimental | 1/3 token-exact; diverges at token 83 / 20 |
| MiniCPM-OCR-1.0 | `minicpm-ocr-q4_k-mix-experimental` | Q4_K_MIX | experimental | 4/5 token-exact; cosine 0.9671 |
| Unlimited-OCR-2 | `unlimited-ocr2-q4_k-mix-experimental` | Q4_K_MIX | experimental | 0/4 token-exact vs F16; text equal, boxes differ |
| TeleOCR | `teleocr-q4_k-mix` | Q4_K_MIX | experimental | 5/6 token-exact; three CJK characters change |
| MinerU2.5-Pro-2604 | `mineru25-pro-q4_k-mix` | Q4_K_MIX | experimental | 6/8 token-exact; layout differences |
| MI-GAN 512 | `migan512-f16` | F16 | recommended | cosine 0.999998 |
| MI-GAN 512 | `migan512-q4_k` | Q4_K | experimental | 0.9019 masked cosine on a real text-removal crop |

## UltraBERT-Embed-150M

Receipt `tests/results/ultrabert/remaining-q4-shards-v1-review.json` (run 40871150002, portable
hosted x86 CPU, SciFact 300 queries / 5,483 rows, against the original F32 model). Publication
`tests/results/ultrabert/publication.json`.

| Precision | Size | Min embedding cosine | Mean embedding cosine | nDCG@10 | MRR@10 | Recall@100 | Top-1 agreement | Strict vector / v1 task gates | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Original F32 | 581.2 MiB | 1 | 1 | 0.7667 | 0.7290 | 0.9750 | reference | reference | reference (not published as GGUF) |
| Q4_K `q4k-last-four-ffn` | 550.3 MiB | 0.9779 | not measured | 0.7649 (-0.0018) | 0.7279 (-0.0011) | 0.9767 (+0.0017) | 0.973 | fail (max abs normalized 0.0282; top-1 below 0.99) | experimental |

Not measured: GPU, ARM/Apple, other corpora or languages, long context, latency. The file carries the
quantizer header of runtime commit `018cf85b`, where its quality was measured; loading it with the
current runtime has not been re-run.

## MoEdernColBERT 27M / 115M

Receipts `tests/results/moedern-colbert/q4k-lane-27m.json` and `q4k-lane-115m.json` (ten frozen
cases against the original, portable CPU builds). Publications `publication-q4k-27m.json` and
`publication-q4k-115m.json`. The CPU-only engine has no GPU arm.

| Model | Precision | Size | Status | Token IDs/masks | Min token-vector cosine x86 / ARM | Min per-case cosine | MaxSim max abs err | SciFact nDCG@10 | Public clients | Known divergences |
|---|---|---:|---|---|---|---|---|---|---|---|
| 27M | F32 | 104.52 MiB | recommended | exact | 1.0000 / 1.0000 | 1.0000 | 1.9e-6 | not measured | C/Python/CLI | none |
| 27M | Q8_0 (not published) | 35.67 MiB | fails 0.97 bar | exact | 0.9315 / 0.9315 | 0.9959 | 0.070 | not measured | not measured | one-token expert flip (q-de) |
| 27M | Q4_K_MIX | 30.53 MiB | experimental | exact | 0.9685 / 0.9688 | 0.9953 | 0.183 | not measured | not measured | worst token q-empty; strict 0.002/0.99999 not met |
| 115M | F32 | 438.59 MiB | recommended | exact | 1.0000 / 1.0000 | 1.0000 | 5.7e-6 | not measured | C/Python/CLI | none |
| 115M | Q8_0 (not published) | 134.55 MiB | fails 0.97 bar | exact | 0.9670 / 0.9672 | 0.9949 | 0.539 | not measured | not measured | one-token expert flip (d-unicode) |
| 115M | Q4_K_MIX | 85.34 MiB | experimental | exact | 0.9793 / 0.9789 | 0.9933 | 0.520 | not measured | not measured | worst token d-truncate (x86); strict budgets not met |

The 27M file's internal policy string is the measured recipe `q8+exps_up+exps_down+attn_q`
(identical rules to `moedern27m_q4k_v1`). Both published files are byte-identical to the measured ones.

## Josev-2B SystemOne

Receipt `tests/results/josev2b/q4k-sweep-41615751516.json` (four frozen requests against the frozen
original, portable hosted CPU). Publication `tests/results/josev2b/publication-q4k.json`; both files
are byte-identical to the measured ones.

| Precision | File | Size | Head-logit cos (min) | Final-norm cos (min) | Option argmax | Typed JSON exact | Max prob. error | Strict 0.002/0.99999 stages | Task panel / GPU | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| F32 | `josev2b-systemone-f32.gguf` | 7.66 GB | reference | reference | 4/4 | 4/4 | 0 | pass | not measured / no GPU path | recommended |
| Q8_0 | `josev2b-systemone-q8_0.gguf` | 3.63 GB | 0.99995 | 0.99944 | 4/4 | 1/4 | 0.0059 | failed (200) | not measured | experimental |
| Q4_K_MIX (`josev_q4k_down8_v1`) | `josev2b-systemone-q4_k-down8.gguf` | 3.10 GB | 0.99739 | 0.94911 | 4/4 | 0/4 | 0.0358 | failed (203) | not measured | experimental |

## OXR-1.0

Receipts `tests/results/oxr/precision.json`, `precision-cosine.json`, `metal-m1/` and
`kaggle-cuda-t4-v2/report.json`.

| File | Status | Size | Token IDs vs original (5 cases) | Final-logit cosine (min) | Decoded differences |
|---|---|---:|---|---|---|
| `oxr-1.0-f16.gguf` | recommended | 4.44 GB | 5/5 exact | not measured (IDs exact) | none |
| `oxr-1.0-q8_0.gguf` (Q8_0-LARGE) | experimental | 2.03 GB | 4/5; `german` differs | 0.9977 (`table`); `german` 0.9996 | `german` CER 0.0111 |
| `oxr-1.0-q4_k-head.gguf` (Q4_K-HEAD) | experimental | 1.88 GB | 4/5; `german` differs | 0.9980 (`table`); `german` 0.9996 | `german` CER 0.0233 |

Both stay at cosine ≥ 0.998 against the F16 run on CPU, Metal (M1) and CUDA (T4). German page: both
quantized files choose a different token at a single decode step (first differing position: Metal
35 / 209, CUDA 55 / 209 for Q8_0-LARGE / Q4_K-HEAD; not recorded on CPU). The cause is decode-step
activation quantization (the integer mat-vec quantizes the single-row activation to Q8_0 blocks),
not the vision tower. The opt-in F32-activation decode A/B is still open.

## GOT-OCR2 (research licence)

Receipts under `tests/results/got-ocr2-f16/`. Licence: Tongyi Qianwen RESEARCH LICENSE —
research/evaluation only; `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1` is required. The Hub repo
ships the verbatim LICENSE and the section 2(c) NOTICE.

| Variant | Status | Token IDs vs original (3 cases) | First divergent token | Character ratio | Stage cosine (min) | Final-logit cosine |
|---|---|---|---|---|---|---|
| F16-r2 (`got-ocr2-f16`) | recommended | 3/3 exact | none | 1.0 / 1.0 / 1.0 | 1.000000 | not measured |
| Q4_K (`got-ocr2-q4_k-experimental`) | experimental | 1/3 (`fox`) | `receipt_example` 83, `simple_form` 20 | 1.0 / 0.8041 / 0.9679 | vision 0.997745, decoder 0.995157 | not measured |
| Q8_0 (file on Hub, no alias) | unqualified | not measured | not measured | not measured | not measured | not measured |

Since 7 November 2026 the default `got-ocr2` alias points to the token-exact F16-r2 (same file as `got-ocr2-f16`, pinned commit `c0f1aa7b`); the Q4_K bytes are only behind `got-ocr2-q4_k-experimental`. A mixed-precision Q4 search is open in PLAN.

## MiniCPM-OCR-1.0

Receipts `tests/results/minicpm-ocr/native-parity.json`, `bench.json`, `q4k-precision.json`;
publication `tests/results/minicpm-ocr/publication-v3.json`.

| Precision | Size | Token IDs vs original (5 cases) | Decoded / CER | Final-logit cosine (min) | Known divergences | Status |
|---|---:|---|---|---|---|---|
| F16 | 2.61 GB | 5/5 exact | identical (CER 0) | not measured | none | recommended |
| Q8_0 | 1.53 GB | 5/5 exact | identical (CER 0) | 0.9998 | none | recommended |
| Q4_K_MIX (`minicpm_q4k_v1`) | 1.02 GB | 4/5; `para-en` from token 54 | `para-en` umlaut line read as `Gr×e, Stra×e, ×pfel`; CER not measured | 0.9671 | `para-en`@54 in every 4-bit arm; real pages and GPU not measured | experimental |

## Unlimited-OCR-2

Receipts `tests/results/unlimited-ocr2/trial.json`, `verify-text-v2.json`,
`q4k-precision-q4kns.json`, `docs/unlimited-ocr2-q4k-recipes.md`; publication
`tests/results/unlimited-ocr2/publication-v2.json`.

| Precision | Size | Token IDs (4 tiled pages) | Decoded text | Final-logit cosine (min) | Known divergences | Status |
|---|---:|---|---|---|---|---|
| F16 | 6,383 MiB | 4/4 exact vs original | byte-exact 4/4 | reference | none | recommended |
| Q8_0 | 3,397 MiB | 2/4 exact; table@7, receipt@131/258 | text unchanged; one box coordinate off by 1-3 | 0.9739 CPU / 0.9555 CUDA | below the 0.97 Q8 bar on CUDA | caveat |
| Q4_K_MIX (`unlimited_ocr2_q4k_shared8_v1`) | 2,188.6 MiB | 0/4 exact vs F16 (first divergence 30 / 10 / 28 / 4) | text equal to F16 with boxes stripped 4/4; boxes differ by up to 2 px; CER not measured | 0.9634 CPU / 0.9540 CUDA | layout box coordinates; real scans and Metal not measured | experimental |

The 7 November republish embeds the 831 added tokenizer pieces in the GGUF metadata, so the F16/Q8_0
whole-file SHAs differ from the earlier Hub commit while the tensor data hashes are identical. The
older `unlimited-ocr2*` aliases stay pinned to the earlier commit.

## TeleOCR

Receipts `tests/results/teleocr/hosted-precision.json`, `summary.json`,
`tests/results/qwen2vl-q4k/teleocr-q4k-precision.json`. Licence: the vision tower is Qwen Research
(non-commercial).

| File | Status | Size | Token IDs vs original (6 cases) | Final-logit cosine (min) | Decoded differences |
|---|---|---:|---|---|---|
| `teleocr-f16.gguf` | recommended (experimental model) | 2.53 GB | 6/6 exact | not measured | none |
| `teleocr-q8_0.gguf` | caveat (experimental model) | 2.11 GB | 6/6 exact | not measured | none |
| `teleocr-q4_k-m.gguf` | caveat (experimental model) | 1.92 GB | 6/6 exact | not measured | none |
| `teleocr-q4_k-mix.gguf` | experimental | 1.20 GB | 5/6; `table` from token 35 | CPU 0.9677, Metal 0.9677, CUDA 0.9570 | three CJK characters change |

Strict per-stage numerics are false at every precision, F32 included (174 stage passes / 276
failures), although decoded IDs of F32/F16/Q8_0/Q4_K_M match exactly.

## MinerU2.5-Pro-2604

Receipts `tests/results/mineru25/precision.json`, `tests/results/qwen2vl-q4k/mineru-q4k-precision.json`.

| File | Status | Size | Token IDs vs original (8 cases) | Final-logit cosine (min) | Decoded differences |
|---|---|---:|---|---|---|
| `mineru25-pro-2604-f16.gguf` | recommended | 2.59 GB | 8/8 exact | not measured | none |
| `mineru25-pro-2604-q8_0.gguf` | recommended | 1.92 GB | 8/8 exact | not measured | none |
| `mineru25-pro-2604-q4_k-mix.gguf` | experimental | 0.77 GB | 6/8; `german` from 7, `layout-page` from 52 (CPU; 180 Metal, 48 CUDA) | CPU 0.9673, Metal 0.9602, CUDA 0.9611 | German masthead and "Infrafttreten"; one layout box coordinate `079`→`078` |

Use F16 or Q8_0 when layout boxes and text must match exactly.

## MI-GAN 512 (inpainting)

Receipts under `tests/results/migan512/` and `docs/inpainting-precision.md`.

| File | Status | Min cosine, synthetic panel | Min masked-region cosine, real photos |
|---|---|---|---|
| `migan512-f32` | reference | 1 | reference |
| `migan512-f16` | recommended | 0.999998 | not separately measured |
| `migan512-q8_0` | caveat | 0.9965 | not separately measured |
| `migan512-q4_k` | experimental | 0.9770 | **0.9019** on a street crop with a DBNet text mask (below 0.94); exact-mask photos ≥ 0.962 |

Representative subtitle/video quality is not measured.
