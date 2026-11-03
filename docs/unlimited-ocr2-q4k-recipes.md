# Unlimited-OCR-2 Q4_K recipes

Goal: the smallest Q4_K-class GGUF whose final-output cosine clears the owner's Q4_K
bar ([precision-acceptance.md](precision-acceptance.md), > 0.94), and Q8_0 > 0.97.

## Method

- **Cases**: the four frozen original captures in `tests/results/unlimited-ocr2/trial.json`
  (paragraphs, table, receipt, german; 146/138/306/164 original-Python IDs). The pages are
  re-rendered with the frozen renderer and must reproduce the recorded PNG SHA-256s
  (`unlimited_ocr2_q4k.py render` pins matplotlib's DejaVu fonts, as on the original Kaggle
  capture; all four match on hosted Linux).
- **Reference**: native F16 (the published `unlimited-ocr-2-f16.gguf`, revision
  `e3af0c0c`) on the CPU, tiled + fast decoder. In every sweep it reproduces the original
  IDs exactly on all four cases free-running, so it stands for the original model.
- **Metric**: teacher forcing on the original IDs (`CRISPEMBED_UNLIMITED_OCR2_FORCE_IDS`,
  default off) and the raw lm_head logits of every step (`CRISPEMBED_UNLIMITED_OCR2_LOGITS_OUT`):
  per-position cosine over the full 129,280-entry vocabulary, at every forced position plus
  the EOS position. "Final cos min" is the minimum over every position of every case.
  Free-running decoded IDs and the text with box coordinates stripped are compared too.
- **Build**: portable (`-DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF`, and the heavy-cpu
  `build_mode=portable` once merged), recorded in each run's `build-mode.txt`; hosted CPU
  AMD EPYC 9V45, 4 threads.

Tools: `tools/unlimited_ocr2_quant_recipes.json`, `tools/ci-heavy/unlimited_ocr2_q4k_sweep.sh`,
`tools/unlimited_ocr2_q4k.py`. Receipts: `tests/results/unlimited-ocr2/q4k-sweep/<run>/`.

## The `v.pos_embed` stop (v3) and its fix

The retained v3 Kaggle panel stopped before any Q4 forward with "Protected source
type/shape/payload differs: v.pos_embed". The quantizer did keep the tensor: F16, payload
byte-identical (SHA-256 `4ad5e4c3...`). The published F16 stores it as `[3145728, 1]`, and
ggml's GGUF writer drops trailing unit axes, so the output stores `[3145728]`; the v3
audit compared raw stored shapes. The protection is not dropped: the v3 contract
(`tools/unlimited-ocr2-q4k/contract.py`) and this sweep's census
(`unlimited_ocr2_q4k.py census`) compare type, payload SHA-256 and the semantic shape
(trailing unit axes canonicalized) of `v.pos_embed`, both patch embeddings, the
newline/separator/CLS vectors and the output norm. Every recipe below passes
(`protected_ok`). The 1-D `v.pos_embed` is never quantized (the quantizer only quantizes
tensors of rank >= 2); every q8_0-based recipe and the policy also keep the CLIP table
`c.pos_embed` at F16 (production q8_0/q4_k store it Q8_0).

## Results (hosted CPU, portable)

Size in MiB of the whole GGUF; Q8_0 = 3,397 MiB (`q8`), F16 = 6,383 MiB.

### Baselines and sensitivity (base Q8_0, one group lowered)

| Recipe | Contents | MiB | Final cos min (case) | Free-run IDs exact | Text = F16 (boxes stripped) |
|---|---|---|---|---|---|
| q8 | all Q8_0, c.pos_embed F16 | 3397 | 0.9739 (receipt) | 1/4 | 4/4 |
| emb | + embedding Q4_K | 3319 | 0.9741 (german) | - | - |
| egu | + expert gate/up Q4_K | 2627 | 0.9745 (receipt) | - | - |
| head | + lm_head Q4_K | 3319 | 0.9706 (receipt) | - | - |
| edn | + expert down IQ4_NL | 3012 | 0.9587 (german) | - | - |
| edq40 | + expert down Q4_0 | 3012 | 0.9565 (paragraphs) | - | - |
| clip | + CLIP blocks Q4_K | 3253 | 0.9468 (paragraphs) | - | - |
| att | + decoder attn q/k/v/o Q4_K | 3360 | 0.9426 (table) | - | - |
| shx | + shared experts Q4_K | 3361 | 0.9192 (paragraphs) | - | - |
| sam | + SAM blocks Q4_K | 3357 | **0.7989** (german) | - | - |
| exp4 | all routed experts 4-bit (gate/up Q4_K, down IQ4_NL) | 2242 | 0.9630 (table) | 0/4 | 4/4 |
| q4k | production `q4_k` (12 Q4_0 fallbacks) | 2152 | 0.9567 (german) | 0/4 | 4/4 |

Runs 41589463950 (q8/q4k/exp4 + free), 41589467110 (egu/edn/edq40/shx),
41589470353 (att/head/emb/clip/sam). Mean per-position cosine stays >= 0.994 for every
recipe except `sam` (0.989); the minimum sits on a few box-coordinate digits (positions
4-13 of table/german), where adjacent numbers are near-ties. Every free-running
divergence, Q8_0 included, is a box coordinate (max delta 3 px); the text with boxes
stripped equals F16 on all four cases. The MoE router makes single groups non-monotone
(e.g. `shx` below `exp4`): a perturbation that flips one expert choice at a coordinate
token costs more than the bytes suggest. The SAM encoder is the one hard no for 4-bit
(the production quantizer already keeps `v.*`/`c.*` at Q8_0).

### Round 2 (base production q4_k, Q4_0 fallbacks replaced)

ROUND2

## Winner

WINNER

## Device confirmation

CUDA
