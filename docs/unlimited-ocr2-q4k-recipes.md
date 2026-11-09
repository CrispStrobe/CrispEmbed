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

Run 41616154269. `q4kn` = production `q4_k` with the 12 Q4_0 fallbacks (expert
`ffn_down`, 896 wide, and the 6848-wide dense layer-0 `ffn_down`) stored IQ4_NL and
`c.pos_embed` F16, so the file holds no legacy Q4_0. The repeated `q4k` arm reproduced run
41589463950 to every printed digit (portable build, different runner).

| Recipe | Contents | MiB | Final cos min (case) | Free-run IDs exact | Text = F16 (boxes stripped) |
|---|---|---|---|---|---|
| q4kne | q4kn + embedding Q4_K | 2074 | 0.9512 (german) | 0/4 | 3/4 |
| q4kn | q4_k, no Q4_0 | 2153 | 0.9523 (german) | 0/4 | 3/4 |
| q4kns | q4kn + shared experts Q8_0 | 2189 | 0.9634 (receipt) | 0/4 | 4/4 |
| q4kna | q4kn + decoder attention Q8_0 | 2190 | **0.9237** (paragraphs) | 0/4 | 4/4 |

The one text change in `q4kn`/`q4kne` is the receipt's first layout label
(`title<LEVEL1>` read as `text`). Raising attention to Q8_0 lowered the minimum
(`q4kna`): with the shared experts at 4 bits it moves a different router decision. The
shared experts are the sensitive part of the decoder (cf. `shx` alone).

## Winners

| Policy (`CRISPEMBED_QUANTIZE_UNLIMITED_OCR2_POLICY`) | Recipe | MiB | vs Q8_0 | CPU final cos min | CUDA T4 final cos min | Text = F16 (boxes stripped) |
|---|---|---|---|---|---|---|
| `unlimited_ocr2_q4k_v1` (smallest above the bar) | q4kne | 2,074 | 3,397 (-39%) | 0.9512 (german) | 0.9443 (receipt) | 3/4 |
| `unlimited_ocr2_q4k_shared8_v1` (recommended) | q4kns | 2,189 | 3,397 (-36%) | 0.9634 (receipt) | 0.9540 (paragraphs) | 4/4 |

Both: base `q4_k` from the published F16; routed expert gate/up, decoder attention and
the small DASH/index projections Q4_K; expert `ffn_down` and the dense layer-0 `ffn_down`
IQ4_NL (no Q4_0); the production guards keep SAM/CLIP vision, projector, embedding and
lm_head at Q8_0; norms, `v.pos_embed` and
`c.pos_embed` F16. `v1` adds `l.embed_tokens.weight=q4_k`; `shared8_v1` keeps the
shared experts Q8_0. The quantizer refuses either policy for another model or base type
and stamps `unlimited_ocr.quantization.policy`; `tests/test_unlimited_ocr2_quant_policy.py`
keeps the C++ tables equal to the measured recipes.

`v1` is the smallest recipe that clears 0.94 on both devices, by 0.004 on CUDA, and changes
one layout label (receipt `title` -> `text`). `shared8_v1` costs 115 MiB more, clears with
0.014 margin on CUDA and keeps the text; it is the `Q4_K_MIX` publication recipe
(`docs/unlimited-ocr2-publication-v1.json`, `precision_gate` cosine, receipt
`tests/results/unlimited-ocr2/q4k-precision-q4kns.json`). Not published. The publisher now
quantizes a recipe from its declared source (F16 here) and passes `quantize_env` as
environment; the publication run converts F16 afresh and must match the published F16
SHA-256 (`487b887e...`) the receipts were measured on, or be re-scored.

No recipe is token-exact against the original: every free-running divergence, Q8_0
included, is a box coordinate (max 2-3 px) or, for `q4kn`/`v1`, that one label; every case
reaches EOS before the 1,024-token cap.

## Device confirmation (Kaggle Tesla T4, portable host build, CUDA arch 75)

Reference: the same CPU F16 teacher-forced run on the Kaggle CPU (Intel Xeon 2.0 GHz; it
reproduces the original IDs on 4/4 there too). Kernel `crispembed-unlimited-ocr2-q4k-cuda`
v1 (commit `a65b3b3c`) and v2 (`a44590b7`); receipts `q4k-sweep/kaggle-cuda-v1`, `-v2`.

| Arm | MiB | CUDA final cos min (case) | Free-run IDs exact | Text = F16 (boxes stripped) |
|---|---|---|---|---|
| F16 on CUDA | 6383 | 0.9789 (german) | 4/4 | 4/4 |
| q8 (Q8_0) | 3397 | **0.9555** (paragraphs) | 1/4 | 4/4 |
| exp4 | 2242 | 0.9658 (receipt) | 0/4 | 4/4 |
| q4kns = `shared8_v1` | 2189 | 0.9540 (paragraphs) | 0/4 | 4/4 |
| q4kn | 2153 | 0.9497 (german) | 0/4 | 3/4 |
| q4kne = `v1` | 2074 | 0.9443 (receipt) | 0/4 | 3/4 |

The CUDA backend alone costs about 0.02 (F16 on CUDA 0.9789 against CPU F16). Q8_0 on CUDA
is **below the 0.97 Q8_0 bar** (0.9555; CPU 0.9739): TODO, a Q8_0 recipe for CUDA (e.g.
shared experts / attention F16) before the published Q8_0 is claimed qualified on CUDA.
(`q8` is quantized from the published F16 with `c.pos_embed` F16, so it is not
byte-identical to the published F32-sourced Q8_0; the published file was not re-scored.)

## Round 3: CUDA Q8_0 lane (Kaggle T4, 8 November 2026, source 15d8fefe)

[Summary](../tests/results/unlimited-ocr2/q4k-sweep/kaggle-cuda-v3/summary.md). Same kernel and T4 as v1/v2.
The current runtime has moved: the F16 CUDA control's paragraphs case is now 0.9989 (v2 0.9879) and plain
`q8` is **0.97000** (v2 0.9555), exactly on the bar. Keeping decoder groups at F16 adds margin:

| recipe | MiB | CUDA final cos min (case) |
|---|---|---|
| q8 | 3397 | 0.9700 (paragraphs) |
| q8at16 (attention F16) | 3468 | 0.9726 (receipt) |
| q8hd16 (head + embedding F16) | 3693 | 0.9737 (receipt) |
| q8sh16 (shared experts F16) | 3465 | 0.9762 (receipt) |
| q8sa16 (shared + attention F16) | 3535 | 0.9791 (german) — equal to the F16 CUDA control (0.9789) |

All recipes keep box-stripped text equal to F16 on 4/4. The CPU leg of the "≥ 0.97 on CPU and CUDA" bar has
not been re-measured for the new recipes or the current runtime.

CPU leg (Kaggle CPU kernel `crispembed-uocr2-q8-cpu`, same source; [summary](../tests/results/unlimited-ocr2/q4k-sweep/kaggle-cpu-v3/summary.md)):

| recipe | MiB | CPU min | CUDA min | min of both |
|---|---|---|---|---|
| q8 | 3397 | 0.9743 | 0.9700 | 0.9700 (on the bar) |
| q8sh16 | 3465 | 0.9728 | 0.9762 | **0.9728** |
| q8sa16 | 3535 | 0.9729 | 0.9791 | 0.9729 |

All three pass "≥ 0.97 on CPU and CUDA"; box-stripped text equals F16 4/4 on both. Recommended:
`q8sh16` (+68 MiB over q8, same CPU/CUDA margin as q8sa16 at half the extra size). Publication needs an
owner decision.

## Round 4: Q4_K_MIX re-score on the current runtime (Kaggle T4 + CPU, 8 November 2026)

[CUDA](../tests/results/unlimited-ocr2/q4k-sweep/kaggle-cuda-v4/summary.md),
[CPU](../tests/results/unlimited-ocr2/q4k-sweep/kaggle-cpu-v4/summary.md); source `3bb0e47f`.

| recipe | MiB | CPU min (3 Nov) | CPU min (now) | CUDA min (3 Nov) | CUDA min (now) |
|---|---|---|---|---|---|
| q4kns (= published Q4_K_MIX) | 2189 | 0.9634 | 0.9320 (german) | 0.9540 | 0.9291 (german) |
| q4kns16 (shared experts F16) | 2256 | - | 0.9283 (german) | - | 0.9326 (german) |

The same recipe and F16 reference now score ~0.03 lower on both backends, worst on the German case, while
Q8_0 moved the other way (round 3). Box-stripped text still equals F16 4/4. Keeping the shared experts F16
does not help, so the loss sits in the 4-bit routed experts/attention. **Open:** attribute the runtime change
between the 3 November sweep and now (candidates: the F32-vision default flip and any decoder/MoE kernel
changes); bisect with the round-1 commit vs current on the German case before any new Q4 search.

## Round 4b: bisect of the Q4_K_MIX drop (Kaggle CPU, 9 November 2026)

Same kernel, recipe `q4kns` (= published Q4_K_MIX), four frozen cases, each arm teacher-forced against its own
native F16 run. Receipts: [old](../tests/results/unlimited-ocr2/q4k-sweep/bisect-old/summary.md),
[now](../tests/results/unlimited-ocr2/q4k-sweep/bisect-now/summary.md),
[now, VISION_F32=0](../tests/results/unlimited-ocr2/q4k-sweep/bisect-nowvf16/summary.md).

| arm | source | final cos min (case) | per case |
|---|---|---|---|
| old | `1148513b` (3 Nov sweep) | 0.9634 (receipt) | paragraphs 0.9693, table 0.9674, receipt 0.9634, german 0.9679 |
| now | current main | 0.9320 (german) | paragraphs 0.9482, table 0.9734, receipt 0.9648, german 0.9320 |
| now, `CRISPEMBED_UNLIMITED_OCR2_VISION_F32=0` | current main | **0.9634 (receipt)** | identical to old, all four cases |

Attributed: the drop is entirely the 3 November F32-vision default flip. Nothing else in the runtime changed the
Q4_K_MIX numbers. With F32 vision the F16 reference itself runs F32 x F32 vision products (the variant that is
more often exact against the original), while the Q4_K vision weights stay Q4-limited, so the Q4-vs-reference
gap widens, mostly on the German page. This is a reference change, not a Q4 regression: the published file is
byte-identical and decodes the same box-stripped text (4/4). The card and alias numbers are now stated for the
default runtime (0.9320 CPU / 0.9291 CUDA), with the F16-vision path's 0.9634 / 0.9540 as the historical figure.
A better 4-bit candidate would keep the SAM/CLIP block matrices above Q4 (the `sam`/`clip` round-1 arms showed
SAM is the most Q4-sensitive group).
