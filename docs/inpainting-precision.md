# Inpainting precision variants (F16 / Q8_0 / Q4_K): inpaint-v1

Experimental, default-off precision variants for the issue-71 MI-GAN512,
Moebius and PixelHacker engines (and the shared SDXL VAE). The F32 profiles and
their frozen original/native budgets are unchanged; a variant is measured, not
qualified, against those F32 budgets. The owner's precision acceptance is
**output cosine > 0.94 against both native F32 and the original capture**.
Results and receipts: `tests/results/migan512/`, `tests/results/moebius/`,
`tests/results/pixelhacker/`.

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
