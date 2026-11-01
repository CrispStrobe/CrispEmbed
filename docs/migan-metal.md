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
| `CRISPEMBED_MIGAN_BATCH` | GPU 4, CPU 1 | Tiles of one band row per graph execution (1..16). |
| `CRISPEMBED_MIGAN_QUANT` | off | Admits `migan-pointwise-v1` F16/Q8_0/Q4_K containers. |
| `CRISPEMBED_MIGAN_GPU_PAD_K` | see decision | F32 pointwise via zero-padded contraction (below). |
| `CRISPEMBED_MIGAN_FAST_MATMUL` | off | Drops the `GGML_PREC_F32` hint (Metal half-tile `mul_mm`). Diagnostic. |
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

### Pending measurements (queued behind the shared GPU lock)

The following were prepared but had not acquired `m1-gpu-lock.sh` after more
than seven hours of queueing behind other agents' jobs, so no result is claimed:
`PAD_K`, the five-engines graph forms (`TRANSPOSE_POINTWISE`+`ROLL_PAD`), F16/
Q8_0/Q4_K on CPU and Metal (strict gates, image gates, output cosine vs F32),
1920x1080 synthetic subtitle frames (warm frame time, tiles/s, batch 1 vs 4),
the remaining seven frozen fixtures, and delivery-5's `test-migan-graph`
(CPU with/without `CRISPEMBED_MIGAN_CPU_KERNELS`, Metal strict/fast) on this M1.

## Default decision

**GPU stays default-off** (`core_env::on("CRISPEMBED_MIGAN_GPU")`). One frozen
fixture is exact and ~1.65x faster than the 4-thread CPU graph, but the full
eight-fixture panel, 1080p frame throughput and the comparison with
delivery-5's optimized CPU graph (reported 273.6 ms/256 tile on 8 EPYC threads)
are not yet measured on this machine. Flip only via `core_env::explicitly_off`
after those pass. CUDA and Vulkan are untested; the code path is backend-generic
(`crispasr_init_gpu_backend()`, per-node support check, CPU reload fallback).
