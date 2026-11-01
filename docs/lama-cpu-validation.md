# LaMa CPU validation: executed results and retained failure

**The complete LaMa implementation has been executed locally, but it has not
passed its full numerical qualification.** Seven of eight frozen synthetic
fixtures passed all 227 stage checks in each of three implementations. The
full-mask fixture failed intermediate checks in all three, while its final
prediction and RGB checks passed. The failure remains recorded; neither the
thresholds nor the original reference outputs were changed to admit it.

The [public execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json)
contains the complete 24-arm synthetic matrix, every stage metric and native
payload digest, original capture identities, the additional FFT diagnosis,
ordinary-image timing samples and photographic RGB checks. This page explains
what those observations establish and where the qualification stops.

## Exact implementation and inputs

All native comparisons use the same pinned Big-LaMa F32 GGUF. Its digest is
`24736f267c68e013f0000ccb69560c4353b6a6d7983fc3cc2537d9f2e03f304c`;
its file size is 204,299,072 bytes, including 204,228,108 bytes of F32 tensor
payload. The converter verified exact readback of all 837 exported tensors.
The 152 omitted generator tensors are inference-inert batch-normalization
tracking counters, not learned parameters. See the conversion receipt in the
[execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json) and
[the conversion contract](lama-conversion.md).

The original reference runs execute the pinned upstream generator from commit
`786f5936b27fb3dacd2b1ad799e4de968ea697e7`, using PyTorch 2.8.0+cpu and NumPy
2.1.3. The official ZIP digest is
`f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`.
That identity describes the complete archive. The separately observed inner
`big-lama/models/best.ckpt` digest is retained in the receipt and is explicitly
not described as an independently published checkpoint pin. Original numerical
source files are checked against the [frozen source contract](lama-source-contract.json).

| Variant | Execution | Candidate settings |
| --- | --- | --- |
| Eager | Existing native CPU implementation in the current executable | Graph and all candidate gates off |
| Corrected portable graph | Persistent ggml graph with F32 operations | Graph on; candidate gates off |
| Optimized CPU graph | The same graph interface with three CPU candidates | CPU FFT, reflection gather and convolution GEMM layout on |

The portable baseline includes the persistent-input lifetime correction found
by actual replay tests. The original graph revision that overwrote constants
and subsequently asserted is not a valid timing baseline. The optimized
configuration retains F32 weights and graph tensors; its FFT callback uses F64
complex arithmetic internally. All three candidate gates remain off by default.
See [the component investigation](inpainting-cpu-optimization.md) for the
individual operators, fixed component budgets and original failure evidence.

## Complete synthetic matrix

Each original synthetic capture contains 227 independently captured stages.
Original ordinary execution and execution with capture hooks produced
bit-identical outputs. The native comparator also checks that observing stages
does not change its output. Every arm completed with the full expected stage
census; this is not a partial-success matrix.

| Fixture | Input width × height | Eager | Portable graph | Optimized CPU graph |
| --- | --- | --- | --- | --- |
| Subtitle bar | 256 × 256 | 227/227 pass | 227/227 pass | 227/227 pass |
| Edge touch | 257 × 249 | 227/227 pass | 227/227 pass | 227/227 pass |
| Center | 128 × 160 | 227/227 pass | 227/227 pass | 227/227 pass |
| Empty mask | 64 × 64 | 227/227 pass | 227/227 pass | 227/227 pass |
| Full mask | 128 × 128 | **37 stage failures** | **34 stage failures** | **34 stage failures** |
| One pixel | 63 × 65 | 227/227 pass | 227/227 pass | 227/227 pass |
| Wide | 320 × 96 | 227/227 pass | 227/227 pass | 227/227 pass |
| Thin strip | 256 × 256 | 227/227 pass | 227/227 pass | 227/227 pass |

This is 1,816 original stage captures and **5,448 native/original stage
comparisons**: 21 fully passing case/variant pairs and three failing pairs.
There are 105 failed stage comparisons across the three full-mask arms.
All 24 arms passed their final prediction, RGB and observer-neutrality checks.
The [execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json)
retains the initial stopped run and the explicitly continued runs that
completed the remaining cases; it does not rewrite the first failure as a pass.

The budgets were declared before these runs:

| Comparison | Fixed acceptance |
| --- | --- |
| Intermediate stage | Finite; maximum absolute error ≤ 0.002, relative L2 ≤ 0.0003, cosine ≥ 0.99999 |
| Final prediction | Maximum absolute error ≤ 0.0002, with the same relative L2 and cosine requirements |
| Filled RGB channels | Maximum difference ≤ 1, mean absolute difference ≤ 0.1, PSNR ≥ 50 dB |
| Known RGB channels | Exactly preserved |
| Original input tensor | Exact equality |

Small final-image differences cannot replace a failed intermediate check.
Conversely, this fixture failure does not establish that the optimization
introduced a regression: the existing eager implementation fails the same
fixture. The result supports retaining experimental gates and continuing
numerical attribution, not claiming full-model parity.

## What the FFT diagnosis established

The first failing stage in every full-mask arm is
`generator.model.5.conv2.ffc.convg2g.fu.rfft`. Its largest error is at the same
real DC coefficient: the original value is 21.875442504882812, while the
eager, portable and optimized outputs are approximately 21.8782406,
21.8780403 and 21.8780384. Their maximum absolute differences are
0.00279808, 0.00259781 and 0.00259590, above the fixed 0.002 stage limit.
These values and every subsequent stage metric are preserved in the
[public record](../tests/results/inpainting/lama-cpu-2026-11-01.json).

To distinguish FFT math from an already different FFT input, the diagnostic
re-executed the unchanged original model and captured the real tensor entering
that exact Fourier unit. Both the original packed Fourier output and final
prediction matched their frozen original captures bit for bit. The resulting
original real tensor was then fed to each **actual production FFT helper**,
linked from the already built native libraries.

| Production FFT receiving identical original input | Maximum absolute error versus original FFT | Fixed component limit |
| --- | ---: | ---: |
| Eager FFT | 0.00000190735 | 0.0001 |
| Portable ggml FFT | 0.00000005960 | 0.0001 |
| CPU FFT callback | 0.00000190735 | 0.0001 |

All three pass on the identical original input. Together with the measured
pre-FFT feature drift and the concentration of the error at DC, this points to
earlier numerical drift amplified by the DC reduction. That is an attribution
inference, not a completed root-cause fix. The experiment does not identify the
earlier operation responsible, and it does not waive any full-model failure.
It avoids changing the FFT implementation merely to compensate for an upstream
mismatch.
Exact original input/output hashes, producer hashes, the probe binary hash,
three command receipts and the unchanged component gate are in
`full_mask_fft_attribution` in the [execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json).

## Ordinary photographic inputs and timing scope

The real inputs are centered, integer crops of two existing repository
photographs with recorded public-domain provenance: a formula blackboard and
an archival document. Each is exercised at 256 and 512 pixels without resizing
or filtering. Four geometry/fixture cases therefore represent two source
photographs. The reference uses the original generator; generated or native
outputs are never substituted for original reference pixels.

The native benchmark calls the ordinary image API, including preprocessing and
final RGB composition, with observers disabled. Each arm loads one context,
records one cold call, then makes five timed warm calls using that context.
Loading and SHA validation are recorded separately. Repeated output must be
bit-identical and known pixels must be unchanged. The checker additionally
compares the saved native RGB against the independently captured original RGB
under the same fixed filled-pixel budgets.

All twelve photograph/variant pairs passed those checks. The largest filled
RGB difference was one integer level; the worst filled-channel mean absolute
difference was 0.000569662, well below the fixed 0.1 limit. Known pixels were
exactly preserved and every warm repeat was bit-identical. These photographic
runs compare final pixels; they do not add intermediate captures or replace the
separate full-mask failure. See `ordinary_photo_benchmarks` in the
[execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json).

The machine is the same eight-CPU-quota AMD EPYC 9V74 host with an 8 GiB memory
limit used for the [component measurements](inpainting-cpu-optimization.md).
No GPU was available. Arm order is sequential, not randomized; five samples
do not establish confidence intervals or a cross-machine throughput claim.
Original PyTorch capture times are diagnostic and are not a paired speed
baseline. Complete raw native samples and per-process memory observations are
retained in the [execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json).

A disclosed 1.4-second, model-free Python test overlapped one initial eager
512-pixel formula timing arm. That initial arm is retained as diagnostic
evidence and excluded from performance summaries; only that timing arm is
repeated. Its numerical verdict remains valid. The event interval and exact
replacement identity are recorded, rather than selecting samples based on
whether their timings look favorable.

The following medians use the same executable, eight threads and five warm
calls per arm. The eager 512-pixel formula entry uses the explicit quiet
confirmation. Load/validation and cold-call times remain separate in the raw
record. Ratios compare these exact configurations and include all three
candidate gates together.

| Photo crop | Eager warm ms | Portable graph warm ms | Optimized CPU warm ms | Portable / optimized |
| --- | ---: | ---: | ---: | ---: |
| Formula, 256 × 256 | 2,439.95 | 2,622.01 | 1,108.54 | 2.37× |
| Document, 256 × 256 | 2,501.57 | 2,715.44 | 1,138.63 | 2.38× |
| Formula, 512 × 512 | 12,079.97 | 10,011.42 | 4,116.47 | 2.43× |
| Document, 512 × 512 | 11,793.32 | 10,334.10 | 4,156.08 | 2.49× |

The optimized ordinary-image processes peaked between 329,816 and 540,484 KiB
RSS across these four cases, approximately 322–528 MiB. This is observed
process memory for these geometries and includes the model; it is neither a
1024-pixel memory estimate nor a graph-arena-only figure. The executable digest,
all raw samples, cold times and individual process peaks are recorded in the
[public evidence](../tests/results/inpainting/lama-cpu-2026-11-01.json).

## Production CLI execution

The actual `crispembed` executable was also run with the optimized CPU graph on
the 512-pixel formula fixture. The independent
[live CLI runner](../tests/test_inpaint_engine_live.py) checked five PNG outputs:
one ordinary invocation, one empty-mask invocation and a three-frame batch
containing a filled image, an empty image and the same filled image again.
All passed. Decoded filled-image bytes exactly matched the saved ordinary API
output; empty-mask outputs exactly matched the source image. The batch's actual
`lama-generator` profile reported two executions of one graph, providing reuse
evidence in addition to repeated-pixel equality.

The runner also verified input/model/fixture digests, PNG geometry and CRCs,
expected inpainting metadata, the selected CPU/F32 backend, clean batch stdout
and preservation of every pinned input. Its five output receipts and subprocess
logs are represented in `production_cli` in the
[execution record](../tests/results/inpainting/lama-cpu-2026-11-01.json).
This is production-path integration evidence. Agreement between two native API
entry points does not independently qualify the original model or resolve the
full-mask intermediate failure.

## Limits and redistribution

The tests characterize these finite CPU fixtures. They do not qualify CUDA,
Metal, mixed precision, 1024-pixel execution, population-level image quality or
temporal consistency. The optimized ordinary-image timing combines three gates;
its end-to-end difference does not isolate the contribution of a single gate.

The 204 MB GGUF is a storage measurement, not a process-memory guarantee.
Weights, graph arenas, original tensors, backend workspace, FFT stacks and
cached plans contribute separately to peak memory. Observed child-process RSS
is included in the record; it must not be replaced with the model file size or
graph-arena size.

License evidence remains as documented in [the conversion guide](lama-conversion.md):
the upstream maintainer's Apache-2.0 statement is preserved, while a dedicated
checkpoint license file was not found. That documentation gap is not resolved
by successfully converting or running the weights. This repository's evidence
contains identities and metrics, not redistributed model weights.
