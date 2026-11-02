# MI-GAN on GPU (Metal) and F16/Q8_0/Q4_K containers

Branch `feat/metal-migan`. This adds gated GPU execution and derived weight
containers to the existing MI-GAN256 graph ([prototype](migan-native-prototype.md),
[CLI](inpainting-cli.md)). The reviewed CPU/F32 batch-1 graph is unchanged and
remains the ground truth: every GPU/quantized result below is measured against
it, stage by stage, with the frozen `0.002`/`0.99999` float gates and the
existing decoded-image gates (known RGB bytes exact; fill max 1, MAE 0.1,
PSNR 50 dB). Receipts: [tests/results/migan/metal-m1/](../tests/results/migan/metal-m1/).

## Runtime paths and gates

| Gate | Default | Effect |
| --- | --- | --- |
| `CRISPEMBED_MIGAN` | off | Required for any MI-GAN load/run (unchanged). |
| `CRISPEMBED_MIGAN_GPU` | see [decision](#default-decision) | GPU through `crispasr_init_gpu_backend()` (`--gpu-backend` honoured). No GPU device, or an op the device rejects, reloads on the reviewed CPU backend with a stderr note. `=0` forces CPU. |
| `CRISPEMBED_MIGAN_BATCH` | 1 | Tiles of one band row per graph execution (1..16); 4 measured slower on the M1. |
| `CRISPEMBED_MIGAN_QUANT` | off | Admits `migan-pointwise-v1` F16/Q8_0/Q4_K containers. |
| `CRISPEMBED_MIGAN_GPU_PAD_K` | see decision | F32 pointwise via zero-padded contraction (below). |
| `CRISPEMBED_MIGAN_FAST_MATMUL` | off | Drops the `GGML_PREC_F32` hint (Metal half-tile `mul_mm`). Fails strict F32 budgets; improves Q8_0 on Metal. |
| `CRISPEMBED_MIGAN_DEQUANT_WEIGHTS` | off | F32 copies of quantized pointwise kernels at load (weight-only error). |
| `CRISPEMBED_MIGAN_GPU_SWAP_POINTWISE` | off | Weight-first F32 pointwise (A/B only). |
| `CRISPEMBED_MIGAN_GPU_TRANSPOSE_POINTWISE`, `CRISPEMBED_MIGAN_GPU_ROLL_PAD` | off | Re-implementation of the `feat/issue71-five-engines-graphs` (06d75fb8) persistent-graph forms, for the A/B in this document. |
| `CRISPEMBED_MIGAN_GPU_PAD_REWRITE` | off | Force the pad rewrite even where `PAD` supports left padding. |

The whole graph runs on one backend; every node must be supported by it (no
scheduler, no per-op CPU fallback, no host round trip inside a tile). Graphs
are built and allocated once per context and batch size and recomputed per
call; weights stay resident. A batch is one upload, one compute, one download.

GPU-specific graph rewrites (all exact in real arithmetic, all verified by the
stage comparison):

- **Left padding.** ggml-metal's `PAD` rejects non-zero left padding, so
  `pad_ext(2,1,2,1)` + 4x4 depthwise (pad 0) becomes the same depthwise with
  symmetric pad 2 and a crop of the trailing row/column (2N+1 → 2N).
- **Pointwise precision.** With `GGML_PREC_F32`, ggml-metal routes an F32
  `mul_mat` whose contraction is a multiple of 128 to the small-batch
  `mul_mv_ext` kernel, which is precise but slow for 65,536 pixel rows; other
  contractions reach the F32-tile `mul_mm_hp` kernel. `PAD_K` appends 64 zero
  input channels to resident copies of the F32 1x1 kernels and to the
  transposed activations, so the same precise arithmetic runs in `mul_mm_hp`.
- **Batching.** `N>1` keeps the reviewed im2col form with a final permute.

## Derived containers (`migan-pointwise-v1`)

`crispembed-quantize F32.gguf OUT.gguf f16|q8_0|q4_k` (MI-GAN family policy in
`tools/quantize.cpp`). Only the 28 separable `*.conv2.weight` 1x1 kernels
(~97% of parameters) change type, stored flattened as `[IC,OC]`; fromRGB/toRGB,
every depthwise/up/down filter, biases, noise and insertion masks stay F32.
Q4_K needs `IC%256`; the three IC=128 layers fall back to Q8_0. The container
records the policy and the source F32 GGUF SHA; the loader requires
`CRISPEMBED_MIGAN_QUANT=1`, that SHA, and checks every tensor's type/shape.
Non-F32 pointwise kernels run weight-first (`mul_mat(W[IC,OC], x[IC,P])`).

| Container | Bytes | SHA-256 |
| --- | ---: | --- |
| F32 (reviewed) | 25,185,504 | `b8fcefc3…809c8b` |
| F16 | 13,716,288 | `379520248ca7c4be49a6d82ae48bbf49223bca75430027771f1913dd78a9857a` |
| Q8_0 | 8,340,288 | `56254f4e45d73d81c12adcfc8410543ea667814d8d88bd7759b9ac9328487607` |
| Q4_K (+3 Q8_0) | 5,505,856 | `7f9402c31be2c98ab1aff772e32c3f98260756f1a4aa88c63fe658917e68ca5c` |

Quantization is deterministic on this host; cross-platform byte identity is
untested, so the files are validated by policy and source identity rather than
byte-pinned.

## Measured on this M1 (1 November 2026)

Apple M1, 16 GiB, shared with other agents' jobs; load average 5–11, so all
timings are diagnostic and contended. CPU reference uses 4 threads.

### Metal F32, `subtitle_bar` frozen fixture

[Stage log](../tests/results/migan/metal-m1/subtitle_bar-f32-metal-vs-cpu-stages.log):
all **237/237** observed stages pass the unchanged strict gates against the
CPU/F32 graph; worst stage max abs `7.63e-6`, worst row cosine `0.999999986`.
Generator output max abs `1.49e-6`, global cosine `1.000000000`. Ordinary and
observed Metal outputs are bit identical. Decoded composite vs CPU/F32 and vs the
**original PyTorch composite**: exact fill (MSE 0, PSNR infinite), 0 known-pixel
errors. Full generated frame vs CPU: max 1 level on 3 of 196,608 channels
(MSE 1.5e-5, PSNR 96.3 dB).

[Matmul formulation A/B](../tests/results/migan/metal-m1/subtitle_bar-f32-metal-matmul-variants.txt)
(warm median per 256 tile, 5 iterations):

| Metal variant | Strict stage failures | Output max abs | Metal ms | CPU/F32 ms (same run) |
| --- | ---: | ---: | ---: | ---: |
| default (im2col + `PREC_F32`) | 0 | 1.49e-6 | 309.3 | 511.0 |
| `FAST_MATMUL` (half-tile `mul_mm`) | 187 (max 0.0154) | 1.93e-3 | 179.3 | 460.6 |
| `SWAP_POINTWISE` | 0 | 1.49e-6 | 315.6 | 462.5 |
| `SWAP_POINTWISE` + `FAST_MATMUL` | 198 | 2.55e-3 | 270.5 | 442.0 |

The fast profile is ~1.7x faster but fails the strict stage budget (output
cosine still 0.9999999); it stays a gated diagnostic, as the strict profile is the
ground-truth comparison. The [per-node profile](../tests/results/migan/metal-m1/f32-metal-per-node-profile.txt)
(each node as its own command buffer, so launch overhead is included) attributes
the strict profile mainly to the F32 `mul_mv_ext` pointwise matmuls (65,536 and
16,384 pixel rows) and the separate leaky-ReLU/scale/clamp/add passes. `PAD_K`
targets the former; its A/B is queued (below).

### Owner acceptance (docs/precision-acceptance.md on feat/teleocr)

Cosine of the decoded output vs the **original PyTorch** generated image and
masked region, minimum over the eight frozen fixtures (centered RGB8). Strict
stage budgets are diagnostics only. [Aggregate](../tests/results/migan/metal-m1/m1-acceptance-report.json),
per-fixture [summaries](../tests/results/migan/metal-m1/stage-logs/).

| Variant | Device | Gen. cos (min) | Masked cos (min) | Bar | Verdict | Min masked PSNR vs original | Known-pixel errors | Worst stage row cos | Strict stage fails | Warm ms/tile |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F32 | Metal | 0.999999991 | 0.999999991 | 0.99 | pass | 89.9 dB (5/8 exact) | 0 | 1.0000 | 0/1896 | 328 |
| F16 | Metal | 0.999995347 | 0.999995347 | 0.98 | pass | 61.8 dB | 0 | 0.9932 | 1270/1896 | 277 |
| F16 | CPU | 0.999995348 | 0.999995348 | 0.98 | pass | 61.8 dB | 0 | 0.9932 | 1270/1896 | 804 |
| Q8_0 | Metal | 0.997253133 | 0.997253133 | 0.97 | pass | 34.9 dB | 0 | -0.887 | 1808/1896 | 409 |
| Q8_0 `FAST_MATMUL` | Metal | 0.999213359 | 0.999213359 | 0.97 | pass | 40.3 dB | 0 | -0.890 | 1808/1896 | 255 |
| Q8_0 | CPU | 0.997272527 | 0.997272527 | 0.97 | pass | 35.0 dB | 0 | -0.887 | 1808/1896 | 584 |
| Q4_K | Metal | 0.939663278 | 0.809899884 | 0.94 | **fail** | 15.1 dB | 0 | -0.993 | 1808/1896 | 580 |
| Q4_K `FAST_MATMUL` | Metal | 0.944052855 | 0.810345668 | 0.94 | **fail** (masked) | 15.1 dB | 0 | -0.974 | 1808/1896 | 251 |
| Q4_K | CPU | 0.939494270 | 0.804902089 | 0.94 | **fail** | 15.0 dB | 0 | -0.996 | 1808/1896 | 652 |

The CPU column's F32 reference time in the same runs was 470–630 ms (contended).
Negative worst row cosines are near-zero rows of intermediate stages; the
global stage cosines stay >= 0.93. Q4_K's worst fixtures are `full` (generated
0.9395) and `center` (masked 0.805, the 80x80 hole the model actually invents).

Runtime as the limiting factor (owner hypothesis): with `GGML_PREC_F32`,
ggml-metal quantizes Q8_0/Q4_K **activations** to Q8 to mirror the CPU dot
products; the default half-tile `mul_mm` keeps F16 activations and is both more
accurate (Q8_0 generated cosine 0.99725 → 0.99921) and faster (409 → 255 ms).
CPU quantized matmul always quantizes activations (`vec_dot_type`).
`CRISPEMBED_MIGAN_DEQUANT_WEIGHTS=1` (F32 copies at load) isolates the
weight-only error, and `CRISPEMBED_QUANTIZE_MIGAN_Q8=prefixes` keeps sensitive
pointwise layers at Q8_0 in a Q4_K container; Q4_K variants keeping
`synthesis.`, `encoder.`, the 128/256-resolution blocks, or the <=64 blocks at
Q8_0 (5.7–8.1 MB) were queued at the end of this session (see PLAN). Q4_K
remains rejected until one passes both cosines; its strict and acceptance
failures stay on record.

Q4_K bytes differ across platforms: the hosted x86 lane produced
`d468f0fe…6551072` vs this M1's `7f9402c3…68ca5c` (SIMD quantization); F16 and
Q8_0 are byte-identical across both.

### Hosted CPU lane ([run 41327638027](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41327638027), x86_64, 4 vCPU)

[Receipts](../tests/results/migan/metal-m1/hosted-cpu-41327638027/). Against
CPU/F32 on the eight fixtures: F16 min output cosine 0.99999939, Q8_0
0.99728, Q4_K 0.93925 (same ordering as the M1). Known pixels exact in all.
1080p CPU frames (warm, 4 threads): `subtitle2` (14 tiles) 17.0 s, `bar`
(20 tiles) 24.2 s, i.e. ~0.83 tiles/s for F32 and Q8_0/Q4_K alike; F16 is
slower (0.71 tiles/s, F16 activation conversion). Delivery 5 `test-migan-graph`
on the same runner: eager 1,368 ms, persistent graph 1,085 ms, persistent graph
+ `CRISPEMBED_MIGAN_CPU_KERNELS` **630 ms**, all exact vs eager (max abs 0).

### Metal variants and the five-engines graph (M1, eight fixtures, 237 stages each)

| Metal F32 variant | Strict fails | Min masked PSNR vs original | Warm ms/tile (median) |
| --- | ---: | ---: | ---: |
| default (im2col, `PREC_F32`) | 0/1896 | 89.9 dB | 348 |
| `PAD_K` | 0/1896 | 85.6 dB | **284** |
| five-engines forms (transpose + pad/roll) | 0/1896 | 89.9 dB | 370 |
| `FAST_MATMUL` | 1539/1896 | 60.7 dB (2 fixtures fail image gate) | 458 (contended) |

Delivery 5's own `test-migan-graph --backend metal` (strict) on the same M1:
exact (max abs 1.85e-6, RGB exact) at **~815 ms** per ordinary image call,
`--fast` ~430 ms; its CPU eager arm in the same process ~1,000 ms. Its Metal
graph is therefore exact but slower than this branch's Metal path (284–348 ms
generator, a few ms of pre/post-processing excluded), and its pointwise form is
slower than the im2col form here when measured in this harness (370 vs 348 ms).

### 1080p synthetic subtitle frames on Metal (F32, warm median, contended)

| Batch / variant | `subtitle2` (14 tiles) | `bar` (20 tiles) | Pixel parity vs CPU/F32 |
| --- | ---: | ---: | --- |
| 1, default | 10.7 s (1.31 tiles/s) | 11.9 s (1.68 tiles/s) | max 1 level, PSNR 95.7 dB, known exact |
| 4, default | 12.5 s | 20.0 s | same |
| 1, `PAD_K` | 11.0 s | 12.9 s | max 1, PSNR 89.7/87.9 dB |
| 4, `PAD_K` | — | 14.0 s | same |

The same-process CPU/F32 frames took 17–60 s (machine load 6–27), so the frame
speedup is 1.6–4.7x depending on load. Batching four tiles did not help on the
M1 and is now opt-in (`CRISPEMBED_MIGAN_BATCH`, default 1). Frame tiles cost
~0.6–0.8 s each versus ~0.3 s for an isolated generator call under the same
conditions; that gap (load or per-tile overhead) is unattributed.

## Default decision

**GPU stays default-off** (`core_env::on("CRISPEMBED_MIGAN_GPU")`). Metal F32
passes the image gates and the owner F32 bar on all eight fixtures and is
1.8–2.3x faster per tile than this M1's CPU/F32 graph, but delivery 5's
optimized CPU graph (630 ms on 4 x86 vCPU, 274 ms on 8 EPYC threads) has not
been run on the M1's CPU, and the branches must be reconciled first. `PAD_K`
is the fastest exact Metal form (284 ms) and the recommended GPU profile when
the default flips. F16 and Q8_0 pass the owner bars on CPU and Metal; Q4_K
fails. CUDA and Vulkan are untested.
