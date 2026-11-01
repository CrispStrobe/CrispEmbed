# Inpainting precision variants (F16 / Q8_0 / Q4_K): inpaint-v1

Experimental, default-off precision variants for the issue-71 MI-GAN512,
Moebius and PixelHacker engines (and the shared SDXL VAE). The F32 profiles and
their frozen original/native budgets are unchanged; a variant is measured, not
qualified, against those F32 budgets. The owner's precision acceptance is
**output cosine > 0.94 against both native F32 and the original capture**.
Results and receipts: `tests/results/migan512/`, `tests/results/moebius/`,
`tests/results/pixelhacker/`.

## Qualification summary against the owner's bars

Owner acceptance (`docs/precision-acceptance.md` on feat/teleocr, b3c6440c):
final-output cosine against the original, minimum over cases, must exceed
**F32 > 0.99, F16 > 0.98, Q8_0 > 0.97, Q4_K > 0.94**. The strict per-stage
budgets stay recorded as diagnostics only. Minimum over the synthetic panel
and the real-photo panel (masked-region cosine vs F32 for photos):

| Engine | F32 | F16 | Q8_0 | Q4_K |
| --- | --- | --- | --- | --- |
| MI-GAN512 | 1.000000 — qualifies | 0.999998 — qualifies | 0.9965 — qualifies | 0.9770 synthetic, 0.9019 street/DBNet photo (vs F32) — see below |
| Moebius | 1.000000 (stage drift cos >= 0.9999995) — qualifies | 0.999935 (photos) — qualifies | 0.9795 (photos) — qualifies | **fails** on photos (0.859); mixed recipe pending |
| PixelHacker | 1.000000 — qualifies | 0.999999 (synthetic; no F16 photo run) — qualifies | 0.9859 (photos) — qualifies | 0.9425 (photos) — qualifies, borderline |

The owner's bar is against the original. On photos no original exists, so
the photo column compares with native F32, which is itself at cosine 1.0 against
the original on the synthetic panel. One caveat: MI-GAN512 Q4_K's street/DBNet
crop reaches only 0.9019 against F32 in the masked region, while every exact-mask
photo (>= 0.962), the full-photo street runs (0.947 / 0.965) and the synthetic
panel (>= 0.977) clear 0.94. The engine-level row above therefore records Q4_K
as qualified on the synthetic panel but below the bar for that DBNet crop.

## Gates

| Gate | Effect |
| --- | --- |
| `CRISPEMBED_INPAINT_QUANT=1` | Accept an `inpaint-v1` GGUF (F32 files need no gate). Default: every stored tensor is dequantized to F32 at load into its original shape, so the unchanged F32 graph runs. |
| `CRISPEMBED_INPAINT_QUANT_MATMUL=1` | Keep stored F16/block matrices and consume them directly with `ggml_mul_mat`: MI-GAN pointwise convs, latent `conv(groups=1)` (1x1 and k x k through im2col) and linear layers, on the eager CPU path and on the persistent graph path (`CRISPEMBED_INPAINT_GRAPH=1`). |

Both are value-parsed `core_env::on` gates. Precision variants run the strict
graph profile only (`CRISPEMBED_INPAINT_FAST` is rejected for them).

## Quantizer policy (`tools/quantize.cpp`, `inpaint-v1`)

`crispembed-quantize IN-F32.gguf OUT.gguf f16|q8_0|q4_k` on a GGUF whose
`general.architecture` is `migan`, `moebius`, `pixelhacker` or `sdxl_vae` uses a
self-contained family path that runs **before** the generic name guards (the
generic `downsample` guard would otherwise copy PixelHacker's `downsamplers.*`
3x3 kernels verbatim after the generic 4D pre-scan had already rewritten their
header shape; OXR had to skip the same pre-scan).

* F32-pinned: vectors (bias, norms, BN statistics, scalars); every kernel with
  one input channel per group (depthwise, Moebius `pos_conv`); MI-GAN fixed
  buffers (`noise_const`, `filter_const`, `noise_strength`) and `fromrgb` /
  `torgb`; `embedding_layer`; relative-position tables; the denoisers'
  `diff_model.conv_in*` / `conv_out*` and the VAE `conv_in`/`conv_out`/
  `quant_conv`/`post_quant_conv`.
* Everything else: the requested type, stored as a **natural-row** matrix
  (`[IC*KH*KW, OC]` for a conv, a pure reshape of the contiguous OIHW bytes;
  `[IN, OUT]` for a linear). K-quants fall back along Q4_K->Q4_0, Q5_K->Q5_0,
  Q6_K->Q8_0 when the row is not 256-aligned, then to F16.
* MI-GAN, measured: a sub-Q8 request keeps `synthesis.*` at Q8_0 (below).
* `--tensor-type GLOB=TYPE` overrides apply to every non-pinned tensor.

The file records `inpaint.quantization.policy=inpaint-v1` and the requested
type. Latent variants carry an adjacent `crispembed-inpaint-precision-variant-v1`
receipt naming their source F32 conversion; the latent comparator verifies that
source with the unchanged F32 receipt/readback checks.

## GGUF tensor names

GGUF names hold at most 63 bytes (`GGML_MAX_NAME`). 540 Moebius and 384
PixelHacker state names are longer (up to 89 bytes), so the first hosted F32
replay could not open any denoiser GGUF. Names of 64+ bytes are now stored under
one deterministic abbreviation (`src/inpaint_tensor_names.h`; Python twin
`gguf_tensor_name`): `diff_model.`->`dm.`, `.transformer_blocks.`->`.tb.`,
`.attentions.`->`.at.`, `down_blocks.`->`db.`, `up_blocks.`->`ub.`,
`mid_block.`->`mb.` (max 57 bytes, collision-free). All loaders map lookups
through it and re-key tensors to original names. MI-GAN, VAE and LaMa names
are unaffected.

## MI-GAN512 results (hosted CPU, ubuntu 4 vCPU)

Checkpoint `migan_512_places2.pt` (README-linked Drive file, 29,553,256 bytes)
first observed at SHA-256 `1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a`
(hosted run 41292563046, recorded before any torch load; not an author-declared
digest). F32 GGUF `4bec681e…1739`, 29,500,736 bytes, 177 tensors byte-exact.

Original/native, frozen gates (max abs 0.002, cosine 0.99999; RGB max 1, MAE
0.1, PSNR 50; known bytes exact), run 41296465856:

| Arm | Frozen panel | Full-stage cases | Max abs vs original | Fill RGB PSNR |
| --- | --- | --- | --- | --- |
| F32 eager | 8/8 pass | 273/273 stages, observer bit-exact | 1.8e-5 | 90.0–95.6 dB (max diff 1) |
| F32 graph (`CRISPEMBED_INPAINT_GRAPH=1`) | 8/8 pass | identical to eager | 1.8e-5 | same |

Precision arms (owner gate: generator-output cosine > 0.94 vs native F32 and vs
original; minimum over the eight fixtures). The frozen F32 gates are reported
unchanged and are expected to fail for quantized weights (F16 3/8, Q8_0/Q4_K
0/8):

| Variant | GGUF bytes | min cos vs original | min cos vs F32 | max abs vs original | Owner gate |
| --- | ---: | ---: | ---: | ---: | --- |
| F32 | 29,500,736 | 1.000000 | — | 1.8e-5 | pass |
| F16 (eager, matmul, graph, graph+matmul) | 17,982,208 | 0.999998 | 0.999998 | 0.0188 | pass |
| Q8_0 dequantized (eager = graph) | 12,583,168 | 0.998580 | 0.998580 | 1.04 | pass |
| Q8_0 stored matmul (eager = graph) | 12,583,168 | 0.996499 | 0.996499 | 1.30 | pass |
| Q4_K dequantized (eager = graph) | 11,143,424 | 0.977746 | 0.977746 | 2.16 | pass |
| Q4_K stored matmul (eager = graph) | 11,143,424 | 0.976972 | 0.976972 | 2.18 | pass |

Q4_K recipe sweep (run 41294631042, `--tensor-type` overrides): all-Q4_K
(9,703,680 bytes) fell to **0.790** on the full-mask fixture; Q8_0 for the 256/512
levels 0.845; Q8_0 for levels 4–32 0.923; Q8_0 encoder 0.789; **Q8_0 synthesis
0.978** (adopted into the policy). The worst case is always the full mask, where
the deep synthesis path generates every pixel.

Latency and memory: generator 1.97–2.38 s per 512 tile (median per arm); the
production CLI takes 1.9–2.4 s and 297–315 MB peak RSS (eager) or 364–376 MB
(graph). Weights are under 30 MB, so activations dominate memory and precision
does not change speed: the graph path is about 6% faster than eager; stored
matmul is 5–13% slower than dequantized at load (extra transposes).

## Live ground truth (subtitles burned into real photos)

`tools/inpaint_live_eval.py`: face.png plus two Wikimedia Commons photos, a
two-line subtitle, exact text mask and DBNet (`dbnet-ic15-q8_0`, dilate 6)
masks, outside-mask exactness, masked-region PSNR/MAE against the clean photo
versus a naive band-mean fill. MI-GAN512, run 41296465856 (masked PSNR, dB):

| Geometry / mask | face (naive) | landscape (naive) | street (naive) |
| --- | --- | --- | --- |
| 512 crop, exact | 24.02 (14.49) | 16.01 (14.64) | 13.21 (12.69) |
| 512 crop, DBNet | 14.70 | 13.20 | 10.52 |
| full photo (512 tiles), exact | 24.53 (14.36) | 20.68 (19.89) | 13.08 (13.91) |
| full photo, DBNet | 17.88 | 18.00 | 11.05 |

Outside-mask pixels are exact on every exact-mask run. The DBNet runs score
below the naive fill on two photos: the detector mask misses parts of the
outlined glyphs, so residual text inside the scored region dominates. This is a
mask-quality limit, not an inpainter regression. Precision arms against F32 on
the same photos (masked-region agreement, exact masks): F16 58–68 dB (cosine
>= 0.999995); Q8_0 34–50 dB (>= 0.9986); Q4_K 19.6–34 dB (cosine 0.962–0.999,
and 0.899 on the street crop with the DBNet mask). Graph arms equal eager to
the byte at F32 and match eager-matmul at Q8_0. CLI time: 2.0–2.2 s per crop,
4–12 s per full photo; peak RSS 307–325 MB (exact mask) and 518–1,166 MB with
DBNet.

Preservation: `cstr/crispembed-migan512-private` (private, verified) holds the
Q8_0 and Q4_K GGUFs. The account then hit HF's private storage limit, so F16/F32
and all latent GGUFs could not be stored. Every file is reproducible from the
recorded checkpoint digest with the converter and `crispembed-quantize`; the
receipts record each digest.

## Moebius and PixelHacker: original/native parity (hosted CPU, 4 vCPU)

Original references: unchanged upstream pipelines on Kaggle T4 CUDA (F32,
TF32 off, deterministic), four owned fixtures, 19 DDIM steps, all random
inputs captured and replayed. Moebius used the exact contract environment;
PixelHacker used the recorded torch 2.6.0 deviation (see the conversion guide).
The native comparator's budgets are unchanged: component max abs 0.002,
relative L2 0.003, cosine 0.99999; RGB max 2, MAE 0.1; known pixels exact.

| Engine / path | Final RGB vs original | Known pixels | Strict components | Worst failing component |
| --- | --- | --- | --- | --- |
| Moebius F32 eager (41295724435) | max 1, generated MAE <= 0.0008: **pass** (4/4) | exact | 85/608 exceed | max abs 0.0346 (`unet.step18`), cosine 0.99999955, rel L2 9.5e-4 |
| Moebius F32 graph, after pin fix (41302077937) | max 1: pass; 84–88 dB vs eager | exact | 89/608 exceed | max abs 0.0379, cosine 0.99999947 |
| PixelHacker F32 eager (41295993654) | max 1, generated MAE <= 0.0007: **pass** (4/4) | exact | 163/702 exceed | max abs 0.0379 (`up_blocks.3.attentions.0`), cosine 0.99999999, rel L2 1.3e-4 |
| PixelHacker F32 graph, after pin fix (41305472141) | max 1: pass; 84–88 dB vs eager | exact | 161/702 exceed | same pattern |

So the frozen synthetic panel is **not** passed by either latent engine, and it
is not relabelled. Every failing component exceeds only the absolute 0.002
budget, on large-magnitude late-step UNet and attention tensors, while
cosine >= 0.9999995 and relative L2 <= 1e-3 (inside its 0.003 budget). The
first exceeding stage in every case is the VAE posterior moments (max abs
0.004–0.005 on L2 ~2,900). This matches CPU-vs-CUDA F32 accumulation-order
drift rather than an operator mismatch, and the decoded images agree to within
one RGB level. An eps-relative budget would be a policy change for the owner.

Initial failures kept on record: the first F32 replay could not open the
denoiser GGUFs (63-byte name limit); before the gallocr pin fix, graph replays
were wrong after their first execution (decoded cosine 0.12–0.16, generated
MAE about 95 levels).

## Moebius and PixelHacker precision variants

Owner gate: decoded-image cosine vs the original capture, and generated-RGB
cosine vs native F32 eager, both > 0.94 (minimum over the four fixtures).
Denoiser and VAE are quantized to the same requested type.

| Engine / variant (arm) | Denoiser + VAE bytes | min decoded cos vs original | min RGB cos vs F32 | RGB PSNR vs F32 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Moebius F32 | 905.3 + 334.6 MB | 1.000000 | — | — | pass |
| Moebius F16 (eager) | 484.7 + 167.6 MB | 1.000000 | 1.000000 | 65–68 dB | pass |
| Moebius Q8_0 dequantized | 287.7 + 89.3 MB | 0.999354 | 0.999844 | 39.8–44.5 dB | pass |
| Moebius Q8_0 stored matmul | 287.7 + 89.3 MB | 0.999248 | 0.999820 | 39.2–41.4 dB | pass |
| Moebius Q4_K dequantized | 182.6 + 47.5 MB | 0.976839 | 0.995215 | 24.4–26.5 dB | pass |
| Moebius Q4_K stored matmul | 182.6 + 47.5 MB | 0.977105 | 0.995256 | 24.5–26.5 dB | pass |
| PixelHacker F32 | 3,449 + 334.6 MB | 1.000000 | — | — | pass |
| PixelHacker F16 (eager and graph) | 1,725.6 + 167.6 MB | 0.999999 | 1.000000 | 66.7–77.2 dB | pass |
| PixelHacker Q8_0 dequantized | 917.9 + 89.3 MB | 0.999631 | 0.999905 | 42.1–45.2 dB | pass |
| PixelHacker Q8_0 stored matmul | 917.9 + 89.3 MB | 0.999417 | 0.999853 | 40.2–41.3 dB | pass |
| PixelHacker Q4_K dequantized | 487.1 + 47.5 MB | 0.990843 | 0.997756 | 28.4–31.3 dB | pass |
| PixelHacker Q4_K stored matmul | 487.1 + 47.5 MB | 0.990318 | 0.997626 | 28.1–31.1 dB | pass |

Q4_K images visibly differ from F32 (about 25 dB) even though the cosine gate
passes; treat Q4_K as a size option with a quality cost. Stored matmul is where
the memory saving is: Moebius CLI peak RSS 2.0 GB (F32 or dequantized) falls to
1.17 GB (Q8_0) and 1.03 GB (Q4_K); PixelHacker falls from 4.5 GB to 1.8 GB
(Q8_0).

## Graph path versus eager (A/B, same captures)

Persistent graphs (`CRISPEMBED_INPAINT_GRAPH=1`, CPU backend, strict F32
accumulation), after the initializer/parameter pin fix:

| Arm | Moebius min decoded cos vs original | PixelHacker min decoded cos vs original | Notes |
| --- | ---: | ---: | --- |
| F32 graph | 1.000000 | 1.000000 | 84–88 dB vs eager F32; same strict-budget pattern |
| F16 graph | 1.000000 | 0.999999 | equals eager F16 |
| Q8_0 dequantized graph | 0.999354 | — | equals eager Q8_0 to 6 digits |
| Q8_0 stored-matmul graph | 0.999171 | 0.999278 | RGB cos vs F32 >= 0.9998 |
| Q4_K dequantized graph | 0.976839 | — | equals eager Q4_K |
| Q4_K stored-matmul graph | 0.976929 | 0.990206 | RGB cos vs F32 >= 0.9952 |

One 512 inpaint through the production CLI (19 steps x 2 CFG arms, 4 vCPU;
runner-to-runner variance about +-30%):

| Engine | F32 eager | F32 graph | Q8_0 matmul eager | Q8_0 matmul graph | Q4_K matmul graph |
| --- | --- | --- | --- | --- | --- |
| Moebius | 510 s / 2.0 GB | 384–422 s / 3.8 GB | 558 s / 1.17 GB | 181 s / 3.0 GB | 192 s / 2.9 GB |
| PixelHacker | 857 s / 4.5 GB | 714–722 s / 6.3 GB | 695 s / 1.8 GB | 482 s / 3.7 GB | 497 s / 3.2 GB |

The graph path is the faster one, and stored-type mul_mat on it is fastest
(Moebius Q8_0 is 2.8x faster than eager F32). Its resident graph and activation
buffers raise peak RSS above eager, though. The lowest memory is eager with
stored matmul. These are delivery-4 graphs plus this branch's fixes; delivery 5
(67406abe) further optimizes the CPU path, and its merged replays are recorded
separately.

## Live ground truth: Moebius and PixelHacker (512 crops, seed 0)

Masked-region PSNR against the clean photo (naive band-mean fill in brackets);
outside-mask pixels are exact on every exact-mask run.

| Engine / arm | face exact / DBNet | landscape exact / DBNet | street exact / DBNet | s per photo, RSS |
| --- | --- | --- | --- | --- |
| naive fill | 14.49 | 14.64 | 12.69 | — |
| MI-GAN512 F32 | 24.02 / 14.70 | 16.01 / 13.20 | 13.21 / 10.52 | 2.2 s, 0.31 GB |
| Moebius F32 graph | 22.27 / 18.54 | 16.81 / 14.67 | 12.20 / 9.97 | 250 s, 3.7 GB |
| Moebius Q8_0 matmul graph | 23.41 / 18.50 | 16.89 / 14.66 | 12.25 / 9.99 | 193 s, 3.0 GB |
| PixelHacker F32 graph | 23.84 / 19.20 | 16.93 / 14.86 | 12.64 / 11.11 | 721 s, 6.2 GB |
| PixelHacker Q8_0 matmul graph | 23.81 / 19.26 | 16.92 / 14.89 | 12.69 / 11.17 | 411 s, 3.6 GB |
| PixelHacker Q4_K matmul graph | 22.99 / 19.26 | 16.54 / 14.89 | 12.45 / 10.58 | 503 s, 3.2 GB |

Diffusion engines use a native seed stream, so a variant's image differs from
F32 by sampling as well as by weights. Masked-region agreement with the same
engine's F32 graph run: Moebius Q8_0 23–30 dB (RGB cosine 0.980–0.998);
PixelHacker Q8_0 23.5–34.2 dB (0.986–0.999) and Q4_K 17.2–27.5 dB (0.943–0.996).
All three engines beat the naive fill on the face and landscape photos. The
busy street scene stays near the naive score for every engine. With DBNet
masks, the latent models handle glyph outlines that the detector misses better
than MI-GAN does: face 18.5–19.3 dB against MI-GAN's 14.7.

### Q4_K on real photos: Moebius falls below the owner gate

The live run (41305480287) contradicts the synthetic Q4_K pass. Measured as
masked-region agreement with Moebius F32 graph on the same photos and seed:

| Moebius arm | face exact / DBNet | landscape exact / DBNet | street exact / DBNet |
| --- | --- | --- | --- |
| F16 graph | 58.9 dB, cos 0.999997 / 57.9 dB, 0.999996 | 56.8 / 53.9 dB, >= 0.999992 | 52.6 / 46.5 dB, >= 0.999935 |
| Q8_0 matmul graph | 28.3 dB, 0.9966 / 30.3 dB, 0.9979 | 26.0 / 24.9 dB, >= 0.9936 | 23.1 / 21.5 dB, 0.9856 / 0.9795 |
| Q4_K matmul graph | 22.5 dB, 0.9885 / 17.7 dB, 0.9646 | 16.8 / 17.7 dB, 0.958 / 0.967 | 15.7 dB, **0.918** / 13.3 dB, **0.859** |

Moebius Q4_K also lowers ground-truth quality: landscape crop 14.77 dB against
16.81 dB at F32. **Moebius Q4_K therefore does not meet the owner gate on real
photos** and must not be offered as qualified. PixelHacker Q4_K stays above the
gate on the same photos (minimum 0.9425, street/DBNet, borderline). Moebius F16
and Q8_0, and PixelHacker F16/Q8_0, pass on both panels. TODO: a Moebius Q4_K
recipe that keeps attention and VAE matrices at Q8_0, measured on this live
panel before any further claim.

## Delivery-5 optimized CPU path (merged 67406abe)

Arms suffixed `-o` run the persistent graph with each engine's measured
candidate gates (`CRISPEMBED_INPAINT_GEMM_LAYOUT=1` plus
`CRISPEMBED_MOEBIUS_RELATIVE_REASSOC=1`/`CRISPEMBED_MOEBIUS_LOCAL_REASSOC=1`
for Moebius, or `CRISPEMBED_INPAINT_CPU_GLA=1` for PixelHacker;
`CRISPEMBED_MIGAN_CPU_KERNELS=1` for MI-GAN).

- With the candidates **off**, the merged code is numerically identical to the
  pre-merge results: MI-GAN512 on all 14 arms (41309726181) and PixelHacker
  graph F32/Q8_0 (41309728413).
- **Moebius F32 `-o`** (41314009419): decoded cosine 1.000000 vs the CUDA
  original, 84–89 dB vs eager F32, the same strict-component pattern (5/83 on
  edge_touch), CLI 358 s / 3.7 GB. On the live photos it matches the F32 graph
  within one RGB level (82–90 dB), at 385 s per photo.
- **PixelHacker F32 `-o`** (41314011582): decoded cosine 1.000000 vs the CUDA/FLA
  original, 84–89 dB vs eager. CLI 386 s against 714–805 s for the plain graph
  and 857 s eager, i.e. 2.1–2.2x faster, at 6.3 GB.
- Initial failure, kept on record: with the GEMM-layout candidate, stored
  quantized weights became ggml `src1` ("CPU does not support node MUL_MAT").
  Fixed by keeping non-F32 weights on the weight-left layout; the quantized
  `-o` arms are re-measured in separate lanes.
- **MI-GAN512 `-o`** (`CRISPEMBED_MIGAN_CPU_KERNELS=1`, run 41322452391): F32
  passes all 8 frozen fixtures. Generator 1.43 s against 2.53 s eager and 2.42 s
  plain graph (1.77x on 4 vCPUs; delivery 5 measured 3.2x on 8 EPYC cores).
  F16/Q8_0/Q4_K `-o` arms, dequantized and stored-matmul, reproduce exactly the
  precision minima above (0.999998 / 0.9965 / 0.9770) at 1.42–1.95 s.
