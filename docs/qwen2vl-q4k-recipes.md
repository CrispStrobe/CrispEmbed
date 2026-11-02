# Q4_K recipes for TeleOCR and MinerU2.5-Pro

Goal: the smallest Q4_K-class GGUF whose final-output cosine clears the owner's
Q4_K bar (docs/precision-acceptance.md, > 0.94). The metric is the minimum
over every generated position of every frozen case of the cosine between the
candidate's logits and the native CPU reference, teacher-forced on the
committed original-Python IDs (`CRISPEMBED_QWEN2VL_FORCE_IDS`). References as
in [qwen2vl-metal.md](qwen2vl-metal.md): TeleOCR = CPU F16 (F32 compute on the
bf16-exact weights, 6 cases); MinerU = CPU F32 converted from the pinned
checkpoint (8 cases). Earlier Q4_K recipes all missed the bar on CPU, Metal
and CUDA with near-identical scores (TeleOCR Q4_K_M 0.905, MinerU "q4_k"
0.914), so the shortfall is weight precision, not runtime.

Tools: `tools/qwen2vl_quant_recipes.json` (named recipes),
`tools/ci-heavy/qwen2vl_q4k_sweep.sh` (hosted CPU: quantize, teacher-force,
score, census), `tools/qwen2vl_quant_census.py` (quantizer argv, stored-byte
census, sweep table). Receipts: `tests/results/qwen2vl-q4k/`.

## Quantizer changes

- `--tensor-type` also accepts `q4_k`, `q5_0`, `q4_0` and `iq4_nl`, so a recipe
  can start from a Q8_0 base and lower single tensor classes. Overrides still
  apply after the built-in policies (last match wins), and the K-quant
  256-column fallback chain still applies (Q4_K -> Q4_0 for unaligned rows).
- Opt-in importance matrix for this engine: `src/qwen2vl_ocr.cpp` installs the
  existing collector (`CRISPEMBED_IMATRIX_OUT=<file>`, no-op when unset) and
  flushes it when the context is freed. The collector now also sees TeleOCR's
  CPU matmuls (a graph-local F32 cast of an F16 weight, and the
  `quant_f32_matmul` custom op). The sweep calibrates on six regression images
  that are not frozen evaluation cases (`calibrate()` in the sweep script).

## Two structural findings

- **MinerU's 896-wide decoder cannot hold Q4_K.** Q4_K needs 256-column rows;
  every decoder matrix except `ffn_down` (4864 wide) fell back to legacy
  **Q4_0**, so the old "plain Q4_K" was mostly Q4_0. Same bytes, IQ4_NL
  (32-column blocks, non-linear codebook) instead of Q4_0 on those rows lifts
  the whole-decoder 4-bit cosine from 0.9098 to 0.9754 (`dec4q40` vs `dec4`).
- **The vision tower, not the decoder, is the bulk.** TeleOCR: vision 1,197 MiB
  F16 of a 2,015 MiB Q8_0 file (its 3420-wide `ffn_down` cannot take any block
  quant and stays F16); MinerU: 638 MiB Q8_0 of a 1,269 MiB all-Q8_0 file. The
  vision merger and patch embedding are always copied at source precision by
  the existing host-side guard, so they were never quantized.

## Sensitivity (hosted CPU, Q8_0 baseline with one group lowered)

Size delta is against the recipe's own Q8_0 baseline `q8`; cosine is the
final-output minimum over all cases (worst case in brackets).

### TeleOCR (baseline `q8`: decoder Q8_0, vision and tied embedding F16; 2,015 MiB, 0.9985)

| Lowered group | MiB | Δ MiB | Final cos min |
|---|---|---|---|
| embedding Q8_0 | 1876 | -139 | 0.9984 (layout) |
| embedding Q6_K | 1840 | -175 | 0.9979 (layout) |
| decoder attn_q Q4_K | 1987 | -28 | 0.9916 (formula) |
| vision Q6_K | 1464 | -551 | 0.9909 (table) |
| vision Q8_0 | 1577 | -438 | 0.9908 (table) |
| decoder attn_k Q4_K | 2001 | -14 | 0.9905 (layout) |
| decoder attn_o Q4_K | 1987 | -28 | 0.9865 (code) |
| embedding Q4_K | 1802 | -213 | 0.9864 (table) |
| decoder attn_v Q4_K | 2001 | -14 | 0.9844 (formula) |
| vision FFN gate/up Q4_K | 1631 | -384 | 0.9737 (table) |
| decoder layers 0-3 Q4_K | 1985 | -30 | 0.9688 (formula) |
| decoder ffn_gate Q4_K | 1973 | -42 | 0.9625 (layout) |
| decoder ffn_up Q4_K | 1973 | -42 | 0.9596 (formula) |
| decoder layers 24-27 Q4_K | 1985 | -30 | 0.9500 (formula) |
| vision Q5_K | 1402 | -613 | 0.9170 (table) |
| decoder ffn_down Q4_K | 1973 | -42 | **0.9150** (formula) |
| vision attention Q4_K | 1728 | -287 | **0.9113** (table) |
| decoder layers 4-23 Q4_K | 1865 | -150 | **0.9063** (formula) |
| vision Q4_K (all) | 1344 | -672 | **0.6943** (table) |

TeleOCR's Qwen3-0.6B-derived decoder is the most sensitive part per byte:
Q4_K on `ffn_down` alone (42 MiB saved) already fails, and every 4-layer group
costs 0.03-0.05. Vision attention (`attn_qkv`/`attn_proj`, which carry the
massive activations) cannot take Q4_K or Q5_K; the vision FFN can. The table
case is the vision-sensitive one, formula/layout the decoder-sensitive ones.

### MinerU2.5-Pro (baseline `q8`: everything Q8_0 incl. vision; 1,269 MiB, 0.9970)

| Lowered group (896-wide rows IQ4_NL, ffn_down Q4_K) | MiB | Δ MiB | Final cos min |
|---|---|---|---|
| vision F16 (for reference) | 1832 | +562 | 0.9993 (japanese) |
| decoder attn_o | 1260 | -9 | 0.9971 (german) |
| decoder layers 0-2 | 1248 | -21 | 0.9971 (japanese) |
| decoder attn_k | 1268 | -1 | 0.9970 (japanese) |
| decoder attn_q | 1260 | -9 | 0.9970 (german) |
| vision Q6_K | 1124 | -145 | 0.9954 (japanese) |
| decoder ffn_up | 1219 | -50 | 0.9944 (japanese) |
| decoder attn_v | 1268 | -1 | 0.9936 (japanese) |
| embedding IQ4_NL | 1204 | -65 | 0.9926 (japanese) |
| decoder layers 3-20 | 1141 | -128 | 0.9921 (japanese) |
| vision FFN Q4_K | 1069 | -200 | 0.9908 (japanese) |
| decoder ffn_gate | 1219 | -50 | 0.9907 (japanese) |
| decoder ffn_down Q4_K | 1219 | -50 | 0.9900 (german) |
| vision attention Q4_K | 1169 | -100 | 0.9885 (japanese) |
| vision Q5_K | 1044 | -225 | 0.9860 (japanese) |
| decoder layers 21-23 | 1248 | -21 | 0.9826 (german) |
| vision Q4_K (all) | 969 | -300 | 0.9804 (table) |
| decoder all 4-bit | 1099 | -171 | 0.9754 (japanese) |
| decoder all "q4_k" (Q4_0 fallback) | 1099 | -171 | **0.9098** (japanese) |

MinerU tolerates 4-bit almost everywhere; its sensitive spots are the last
three decoder layers, `ffn_gate`/`ffn_down` and the Q4_0 fallback itself.

## Combined recipes (hosted CPU)

| Model | Recipe | Contents (base Q8_0) | MiB | Final cos min |
|---|---|---|---|---|
| TeleOCR | r2 | vision attn Q8_0, vision FFN gate/up Q4_K, embedding Q6_K | 1269 | 0.9725 (table) |
| TeleOCR | r1 | r2 + decoder attn_q/attn_k Q4_K | 1227 | 0.9669 (layout) |
| TeleOCR | r7 | r1 with vision attn Q6_K | 1178 | 0.9658 (table) |
| TeleOCR | r8 | r7 + decoder attn_o Q4_K | 1150 | 0.9640 (table) |
| TeleOCR | **r9** | r7 with embedding Q4_K | **1140** | **0.9677** (table) |
| TeleOCR | r5 | r1 + vision attn_proj Q4_K | 1202 | 0.6260 (table) |
| MinerU | r1 | vision Q4_K, decoder IQ4_NL + ffn_down Q4_K, embedding Q8_0 | 799 | 0.9611 (japanese) |
| MinerU | **r2** | r1 with embedding IQ4_NL | **734** | **0.9521 / 0.9673** (japanese) |
| MinerU | r3 | r1 with decoder layers 21-23 Q8_0 | 820 | 0.9729 (table) |
| MinerU | r4 | r3 with embedding IQ4_NL | 755 | 0.9702 / 0.9732 (table) |
| MinerU | r5 | vision FFN Q4_K only, decoder 4-bit, embedding Q8_0 | 899 | 0.9595 (japanese) |

Same-byte reruns: TeleOCR r1 reproduced to 1e-6 (its CPU path is F32
activations against dequantized rows). MinerU r2/r4 (identical GGUF SHA-256)
scored 0.9521/0.9702 on one hosted runner and 0.9673/0.9732 on another: the
hosted builds use `-march=native`, and MinerU's CPU matmuls quantize the
activations in ISA-specific kernels, so its CPU score moves by up to 0.015 with
the runner. Both stay above the bar; the scripts now record `lscpu`.

### Importance matrix

Calibration: six regression images that are not frozen evaluation cases
(document, form, historical receipt, formula photo, transcoda page, formula),
256 new tokens each, on the unquantized source (`imatrix.gguf` kept with the
receipts). With it, Q4_K/IQ4_NL got **worse** wherever the recipe already
passed (TeleOCR r1 0.9669 -> 0.9395, vision FFN 0.9737 -> 0.9588; MinerU r1
0.9611 -> 0.9471, decoder 4-bit 0.9754 -> 0.9727) and better only where it
fails anyway (TeleOCR decoder Q4_K 0.815 -> 0.870, vision Q4_K 0.694 ->
0.709). Six calibration pages are a small, differently distributed set; the
path stays opt-in and no winner uses it.

## Winners

| Model | Policy (`CRISPEMBED_QUANTIZE_QWEN2VL_POLICY`) | MiB | vs Q8_0 | vs F16 |
|---|---|---|---|---|
| TeleOCR | `teleocr_q4k_v1` (= r9) | 1,140 | 2,015 (published Q8_0; Q4_K_M 1,834) | 2,409 |
| MinerU2.5-Pro | `mineru_q4k_v1` (= r2) | 734 | 1,832 (published Q8_0, vision F16); 1,269 all-Q8_0 | 2,471 |

`teleocr_q4k_v1` (base `q8_0`, from the F16 GGUF): vision `attn_qkv`/`attn_proj`
Q6_K, vision FFN gate/up Q4_K, vision FFN down F16 (3420-wide, no block quant
fits), tied embedding/LM head Q4_K, decoder `attn_q`/`attn_k` Q4_K, the rest of
the decoder Q8_0. Q4_K covers the largest share of the vision FFN and the
embedding; the decoder stays mostly Q8_0 because it is the sensitive part.

`mineru_q4k_v1` (base `q8_0`, from the F32 conversion): the whole vision tower
Q4_K, decoder 896-wide matrices and the tied embedding IQ4_NL, decoder
`ffn_down` Q4_K. Every quantized matrix is 4-bit.

The policies are rule tables in `tools/quantize.cpp` equal to the measured
recipes (`tests/test_qwen2vl_quant_policy.py`); the quantizer refuses them for
any other model or base type and stamps `qwen2vl.quantization.policy`. Locally,
`teleocr_q4k_v1` and the hand-written r9 rules give identical tensor data
(census `tensor_data_sha256`).

## Device confirmation

Final cos min = minimum over all positions of all cases; decoded = free-running
IDs against the original-Python IDs.

| Model | Recipe | Device | Final cos min | Decoded exact | Divergences |
|---|---|---|---|---|---|
| TeleOCR | r9 | CPU (hosted) | 0.9677 (table) | see CPU policy run | |
| TeleOCR | r9 | Metal (M1) | 0.9677 (table) | 5/6 | table@35 |
| MinerU | r2 | CPU (hosted, 2 runners) | 0.9521 / 0.9673 (japanese) | 6/8 | german@7, layout-page@52 |
| MinerU | r2 | Metal (M1) | 0.9602 (german) | 6/8 | german@7, layout-page@180 |
| MinerU | r4 | CPU (hosted) | 0.9732 (table) | 6/8 | german@7, layout-page@48 |
| MinerU | r4 | Metal (M1) | 0.9643 (table) | 6/8 | german@7, layout-page@48 |

<!-- CUDA and CPU policy rows: see below -->

Decoded changes, recorded rather than hidden: MinerU German page reads the
masthead as "Reichs-Gefehbl. 1900. Ausgegeben zu Berlin ..." instead of
"Reichs-Gefehblatt. № 1." and spells "Infrafttreten" for "Inkrafttreten";
layout-page moves one box coordinate by 1 (`079` -> `078`). TeleOCR's table
case changes three CJK characters (`指标`->`指`, `对位`->`对住`, `外观`->`外现`).
The all-Q8_0 MinerU recipe decoded 8/8; the Q4_K bar is a logit-cosine bar
and these recipes meet it, but they are not token-exact.

Worst intermediate stage, for the record: the MinerU German prompt's
final-norm row (an attention-sink token) drops to 0.08-0.11 with Q4_K vision
(0.29 with Q8_0 vision on CPU); TeleOCR's code prefill row 0.49. Neither moves
the logits below the bar.
