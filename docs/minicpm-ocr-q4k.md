# MiniCPM-OCR-1.0 Q8_0 / Q4_K recipes

Goal: the smallest GGUF whose final-output cosine clears the owner's bars
([precision-acceptance.md](precision-acceptance.md): Q8_0 > 0.97, Q4_K and mixed
Q4_K > 0.94), measured against the **original model**.

## Metric and reference

- **Original logits.** The frozen original capture (Transformers 5.21.0, torch
  2.13.0+cpu, F32, greedy; report SHA256 `aeb1756d...`, five cases `line`,
  `para-en`, `table`, `tall`, `tiny`, 420 generated IDs) keeps only the first-step
  logits. The sweep therefore recomputes the original logits at **every generated
  position**: one Transformers 5.21.0 F32 forward per case over the frozen prompt plus
  the frozen generated IDs (teacher forcing). Before scoring it checks the recomputed
  inputs against the frozen capture: `input_ids` and `pixel_values` bit-equal, row 0
  against the frozen first-step `lm_head` (cosine >= 0.99999999999957, max abs
  4.0e-5 on every case), and argmax at every position equal to the frozen IDs (420/420).
- **Native candidate.** The CLI runs each case teacher-forced on the same frozen IDs
  with two new test-only gates, `CRISPEMBED_MINICPM_OCR_FORCE_IDS` (file of IDs that
  replace the greedy choice) and `CRISPEMBED_MINICPM_OCR_LOGITS_OUT` (raw F32 scores per
  step). Both are default off (unset, empty or `0`).
- **Final cos min** = minimum over all positions of all five cases of the cosine
  between candidate and original logits. **Decoded exact** comes from the forced
  argmax. Greedy decoding is a function of the prefix, so a case whose forced argmax
  matches at every step decodes the frozen IDs exactly. Cases that diverge (and,
  in the policy run, every case) are also decoded free-running.
- **Worst stage** = the lowest single-row cosine of the prefill decoder layers 0-23 and
  the final norm against the frozen original arrays.
- **Build:** hosted CPU, portable (`-DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF`). The
  `q4k` arm produced bit-identical scores on an EPYC 9V74 and an EPYC 9V45 runner.
  MiniCPM's decoder multiplies F32 activations by dequantized rows, and vision runs in
  F32 by default (`CRISPEMBED_MINICPM_OCR_VISION_F32`).
- **No CUDA arm.** The MiniCPM-OCR runtime is CPU-only (`ggml_backend_cpu_init`, see
  [gpu-support-matrix.md](gpu-support-matrix.md)), so there is no GPU path to confirm
  on Kaggle.

Tools: `tools/minicpm_quant_recipes.json` (named arms),
`tools/ci-heavy/minicpm_q4k_sweep.sh` (build, original capture, quantize, score) and
`tools/minicpm-ocr/q4k_sweep.py`. Receipts: `tests/results/minicpm-ocr/q4k-sweep/<run>/`.

## Structure

The checkpoint is 1.30 B parameters: a Qwen3.5-0.8B decoder (width 1024, 24 layers:
18 gated-delta and 6 full attention), a tied 248,100 x 1024 embedding, a 27-layer
SigLIP vision tower (width 1152, FFN 4304), a ViT merger (4608 -> 17216 -> 1152) and an
MLP merger (4608 -> 1024).

- Every decoder input width (1024/2048/3584) is 256-aligned, so decoder Q4_K is real
  Q4_K.
- Vision rows of 1152 are not 256-aligned. A Q4_K request on them would fall back to
  Q4_0, and the quantizer's generic vision floor (`v.*`) keeps them at Q8_0 under a
  `q4_k` base anyway. The 4-bit vision arms therefore use IQ4_NL (32-column blocks).
- `fc2` (input 4304) admits no 32-column block and stays F16 in every arm. So does the
  4-D patch embedding: quantizing from the F16 GGUF keeps its original four axes (the
  earlier Q8 load failure came from flattening a 4-D F32 tensor). The position
  embedding is copied as-is.

## Baselines (hosted CPU, portable)

| Arm | MiB | Final cos min (worst case) | Decoded exact | Run |
|---|---|---|---|---|
| F16 | 2,491.7 | 1.0000 (all cases >= 0.99999) | 5/5 | 41591527932 |
| Q8_0 (patch F16, fc2 F16) | 1,455.2 | **0.9998** (para-en) | 5/5 | 41591527932 |
| plain `q4_k` (decoder+merger Q4_K, embedding/vision Q8_0) | 1,205.6 | **0.9724** (para-en) | 4/5, para-en@54 | 41591527932, 41614620614 |
| seven-arm v3 Q4 (decoder Q4_K, rest F16) | 1,809.5 | 0.9690 (para-en) | 5/5 | 41591527932 |

The F32 arm stopped before inference with `graph decoder initialization failed` on
the F32 GGUF, most likely the decoder's F32 weight copies on a 16 GB runner. main has
since added a low-memory fallback for those copies. That receipt is kept (`41591527932/f32/initial-graph-init-failure.log`).
F16 serves as the unquantized control.

## Sensitivity (Q8_0 baseline with one group lowered)

| Lowered group | MiB | Δ MiB | Final cos min | Decoded |
|---|---|---|---|---|
| ViT merger 4-bit (`linear_1` Q4_K, rest IQ4_NL) | 1405.4 | -50 | 0.9998 | 5/5 |
| embedding Q6_K | 1396.6 | -59 | 0.9994 | 5/5 |
| MLP merger Q4_K | 1442.9 | -12 | 0.9980 | 4/5 (para-en@54) |
| decoder gated-delta projections Q4_K | 1365.0 | -90 | 0.9958 | 4/5 (para-en@54) |
| embedding Q4_K | 1334.1 | -121 | 0.9939 | 5/5 |
| vision FFN `fc1` IQ4_NL | 1391.4 | -64 | 0.9936 | 5/5 |
| decoder `ffn_down` Q4_K | 1413.2 | -42 | 0.9917 | 5/5 |
| vision attention IQ4_NL | 1386.9 | -68 | 0.9895 | 5/5 |
| decoder FFN Q4_K | 1329.2 | -126 | 0.9790 | 4/5 (para-en@60) |
| decoder full-attention projections Q4_K | 1434.2 | -21 | 0.9775 | 5/5 |
| vision blocks IQ4_NL | 1323.1 | -132 | 0.9749 | 5/5 |
| decoder all projections Q4_K | 1218.0 | -237 | 0.9683 | 5/5 |

Every case's minimum is on `para-en`. Per byte, the decoder's six full-attention
layers are the most sensitive group (-0.022 for 21 MiB). The ViT merger and a Q6_K
embedding are almost free.

## Combined recipes

| Recipe | Contents (base Q8_0 from F16) | MiB | Final cos min | Decoded | Run |
|---|---|---|---|---|---|
| **mix1** | decoder, merger, embedding Q4_K; vision blocks IQ4_NL; ViT merger 4-bit | **902.4** | **0.9489** | 4/5 (para-en@54) | 41614616185 |
| mix3 | mix1 with decoder full attention Q8_0 | 923.4 | 0.9075 | 4/5 | 41614616185 |
| **mix2** | mix1 with vision attention Q8_0 | **970.8** | **0.9671** | 4/5 (para-en@54) | 41614620614 |
| mix4 | mix2 with decoder full attention Q8_0 | 991.8 | 0.9357 | 4/5 | 41614620614 |
| mix5 | mix2 with embedding Q6_K | 1033.3 | 0.9706 | 4/5 | 41614616185 |

Raising the six full-attention layers back to Q8_0 *lowered* the score: 0.9489 to
0.9075 (mix1 to mix3) and 0.9671 to 0.9357 (mix2 to mix4). On `para-en` the errors
from the groups do not add up independently, so a small recipe change can move the
minimum by 0.03-0.04 in either direction. mix1 clears the bar by only 0.009, and
mix3, which uses more bits, fails it. mix2 clears it with 0.027 to spare.

## Decoded changes

Every arm that lowers precision decodes `para-en` differently from position 54
onward. The original reads the umlaut line (rendered with PIL's default font, which
draws the missing glyphs as boxes) as `Zeile mit Umlauten: Grxe, Straxe, pfel.`. The
quantized arms read `Gr×e, Stra×e, ×pfel.`: the same box glyphs, transcribed as `×`.
Both spellings are wrong; they are wrong in different ways, from glyphs the
model cannot read. The divergence also appears in arms with cosine 0.996-0.998
(gated-delta Q4_K, merger Q4_K), so it reflects a near-tie in the original
logits rather than heavy quantization damage. The other four cases are token-exact in
every arm.

Worst intermediate stage: decoder layer 23 on `para-en`, single-row cosine
0.02-0.10 for the 4-bit mixes (global layer cosine 0.85-0.89, max abs 28 against an
original magnitude of 212). Q8_0's worst row is the final norm at 0.498. These are
attention-sink rows, and they do not bring the logits below the bar.

## Winners and policies

| Model | Policy (`CRISPEMBED_QUANTIZE_MINICPM_POLICY`) | MiB | vs Q8_0 (1,455) | Final cos min |
|---|---|---|---|---|
| MiniCPM-OCR-1.0 | `minicpm_q4k_v1` (= mix2, recommended) | 970.8 | 66.7% | 0.9671 |
| MiniCPM-OCR-1.0 | `minicpm_q4k_min_v1` (= mix1, smallest measured above 0.94) | 902.4 | 62.0% | 0.9489 |

Policy confirmation run 41628919702 (portable, heavy-cpu `build_mode=portable`, EPYC 7763)
built both files through `CRISPEMBED_QUANTIZE_MINICPM_POLICY`. `minicpm_q4k_v1` has the
same tensor data as the hand-written mix2 (`tensor_data_sha256` `27e0e75a...`) and
scores 0.9671 again. The difference from the earlier mix2 run on an EPYC 9V45 is at most
3.4e-8 per case, all of it from the recomputed original logits. Decoded free-running on
every case: `line`, `table`, `tall` and `tiny` are token- and text-exact. `para-en`
diverges at 54 and still ends with EOS. Both policies hit EOS on all five cases, and
none reaches the token cap. Receipts: `tests/results/minicpm-ocr/q4k-precision.json`
(`minicpm_q4k_v1`, publication) and `q4k-min-precision.json` (`minicpm_q4k_min_v1`).

Both policies are rule tables in `tools/quantize.cpp`, kept equal to the measured
recipes by `tests/test_minicpm_quant_policy.py`. The quantizer refuses them unless
the source is a `minicpm_ocr` GGUF and the base type is `q8_0`, and it stamps
`minicpm_ocr.quantization.policy` into the output. Runtime: quantized decoder
tensors still need `CRISPEMBED_QWEN35_QUANT=1`; the IQ4_NL vision weights are
dequantized to F32 at load.

```sh
CRISPEMBED_QUANTIZE_MINICPM_POLICY=minicpm_q4k_v1 crispembed-quantize minicpm-ocr-1.0-f16.gguf minicpm-ocr-1.0-q4_k-mix.gguf q8_0
```

## Publication

`docs/minicpm-ocr-publication-v1.json` carries a `Q4_K_MIX` recipe: from `F16`,
`type` q8_0, `quantize_env` `CRISPEMBED_QUANTIZE_MINICPM_POLICY=minicpm_q4k_v1`,
`precision_gate` cosine and `precision_tier` q4_k. Its receipt is
`tests/results/minicpm-ocr/q4k-precision.json`, built from the policy confirmation
run 41628919702 (worst 0.9671; `minicpm_ocr_publish.py` accepts it). `tools/ci-heavy/minicpm_ocr_publish.py` now accepts a cosine-gated receipt:
every case must be above the tier's bar, the build must be portable and the receipt
must name the GGUF hash. Policy switches travel to the workflow as environment. The
recipe was published experimentally and re-verified in [run 42371025754](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42371025754); [the receipt](../tests/results/minicpm-ocr/publication-42371025754.json) records the immutable commit and hashes. The `minicpm-ocr-q4_k-mix-experimental` alias is registered; downstream qualification remains separate. Apache-2.0 licence handling is unchanged.

## Reproduce

```sh
gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/minicpm_q4k_sweep.sh \
  -f args="q8,q4k,mix2 mix2 all" -f isolated_lane=q4k-minicpm-ocr-a \
  -f pip="torch==2.13.0 torchvision transformers==5.21.0 pillow==11.3.0 numpy==2.2.6 gguf==0.17.1 kaggle==2.2.5 six"
```

Arguments: the arm list, the arms that get the stage comparison, and the arms (or
`all`) that are also decoded free-running. The run needs the Kaggle secrets: it
downloads only the frozen report, the PNGs and the small arrays from the original
reference kernel's output.
