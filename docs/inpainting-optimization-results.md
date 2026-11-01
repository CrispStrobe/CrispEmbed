# Inpainting optimization: executed CPU results

This change adds measured CPU optimizations to the five issue71 inpainting
profiles and exercises actual converted weights, ordinary image APIs and the
production CLI. It preserves the existing implementations for comparison.
The new candidates remain explicit, default-off environment gates.

**The strongest completed result is MI-GAN:** its optimized output agrees
with the prior native implementation at every captured stage, while the
recorded ordinary image calls are substantially faster. LaMa also produces
faster, passing final images on the tested photographs, but its full-mask
synthetic fixture retains intermediate numerical failures. The latter is a
recorded limitation, not a threshold adjustment.

The target machine was a Linux AMD EPYC 9V74 worker with an eight-CPU quota,
nine CPUs in affinity and an 8 GiB memory limit. The build enables native
CPU instructions and llamafile matrix multiplication. **CUDA and Metal were
not available for execution.** Device graphs and hosted test lanes are
included, but CPU measurements do not establish GPU performance or parity.
Hardware, build and exact operator evidence are in the
[component record](../tests/results/inpainting/cpu-components-2026-11-01.json).

## What changed

| Area | Implementation | Scope and tradeoff |
| --- | --- | --- |
| Persistent graphs | Retain model graphs, geometry constants and allocated workspaces across image calls | Avoids rebuilding the graph; retained inputs require real storage |
| MI-GAN CPU kernels | Fuse activation work and vectorize depthwise convolution | Keeps the measured F32 operation order and captured native bytes |
| Dense convolutions | Feed the CPU matrix multiplication with a layout suited to its optimized kernel | Same weights and F32 storage; actual supported batch geometry is bounded |
| Reflection padding | Gather height rows with cached integer indices, retaining the existing width reflection | Removes repeated padding work without changing reflected pixels |
| LaMa Fourier transforms | One CPU ggml custom node per FFT, using cached radix-2/Bluestein plans | F32 graph storage, F64 complex arithmetic internally; portable GPU graph retained |
| Moebius relative attention | Change contraction order to avoid the large positional intermediate | Algebraically equivalent; floating-point reduction order changes |
| Moebius local attention | Compute query/kernel coefficients and combine bounded strips of value neighborhoods | Limits patch/coefficient workspace; padding alternative stays separately gated |
| Moebius CPU depthwise convolution | Reuse the MI-GAN tap-major loop in a restricted 3x3 CPU graph node | No activation transfers; four actual model cases retain exactly the prior final RGB |
| PixelHacker gated linear attention | Adapt ggml's existing CPU recurrent kernel to rectangular heads | Reduces the portable operation count; some measured graph arenas are larger |
| Model loading | Bounded physical tensor names plus separately gated hardware SHA-256 | Every tensor payload and full-file digest remains unchanged; unsupported CPUs use the scalar hash |

The persistent-input correction is a correctness fix discovered by replay:
the pinned ggml allocator could reuse an immutable input's allocation after
its last use. Such leaves now retain output ownership as well. Baseline
timings use that corrected implementation. The original failing graph is
preserved as diagnostic evidence and is not a valid performance baseline.
See [the correction and operator report](inpainting-cpu-optimization.md).

These optimizations do not depend on reducing image resolution, changing
weights, quantizing the models, shortening the diffusion schedule or relaxing
comparison budgets. The measured latent profile uses the stock 20 requested
DDIM steps, which execute 19 timesteps under the original strength slicing.
Model loading and validation are timed separately from cold and warm image
calls. Output composition preserves every unmasked RGB byte.

## Complete-image measurements

Each MI-GAN and LaMa warm median below is from five ordinary image API calls in
one loaded context after a separate cold call. PixelHacker's completed pair
uses three warm calls per arm, as separately stated below. Timings include preprocessing,
inference and RGB composition, with stage observers disabled. All raw samples,
exact model/input hashes, executable identities and memory observations remain
in the linked records. These are measurements of this machine and these
fixtures, without a cross-machine throughput claim or confidence interval.

### MI-GAN

| Model / threads | Existing eager ms | Persistent graph ms | Optimized CPU ms | Eager / optimized |
| --- | ---: | ---: | ---: | ---: |
| 256 / 4 | 1,072.62 | 1,015.62 | 411.60 | 2.61× |
| 256 / 8 | 895.01 | 602.97 | 273.62 | 3.27× |
| 512 / 4 | 2,169.13 | 1,927.51 | 697.67 | 3.11× |
| 512 / 8, quiet confirmation | 1,604.95 | 1,259.96 | 495.24 | 3.24× |

These rows use the formula-photograph fixture at each model's native size.
The optimized processes peaked at approximately 196 MiB for 256 and 373 MiB
for 512 in these recorded arms. An initial 512/eight-thread timing group was
contended by unrelated CPU work and is excluded, with its raw samples retained.
The explicit quiet confirmation reuses the original measured executable;
it is a separately identified run, not a replacement of the failed admission.
See the [MI-GAN record](../tests/results/inpainting/migan-cpu-2026-11-01.json).

Both baseline and optimized implementations passed 20 original-model cases
across the two resolutions, with 1,070 stage comparisons per implementation.
All 1,070 corresponding native stage payload hashes were identical between
implementations. At each resolution, two synthetic cases contain full stage
captures, six additional synthetic cases and two photographs contain the
declared anchor stages. This is a finite regression panel, not a population
study of inpainting quality.

Against the original model, the worst intermediate maximum absolute error
was approximately 0.00001899, below the unchanged 0.002 stage budget. Filled
RGB channels differed by at most one integer level; the worst filled-channel
mean absolute difference was approximately 0.00008929. Known pixels remained
exact. The optimized path preserves the existing native result, rather than
trading the observed image quality for these speed gains.

### LaMa

| Photo crop / 8 threads | Existing eager ms | Portable graph ms | Optimized CPU ms | Portable / optimized |
| --- | ---: | ---: | ---: | ---: |
| Formula, 256 × 256 | 2,439.95 | 2,622.01 | 1,108.54 | 2.37× |
| Document, 256 × 256 | 2,501.57 | 2,715.44 | 1,138.63 | 2.38× |
| Formula, 512 × 512 | 12,079.97 | 10,011.42 | 4,116.47 | 2.43× |
| Document, 512 × 512 | 11,793.32 | 10,334.10 | 4,156.08 | 2.49× |

All twelve photograph/implementation pairs passed original final-RGB checks;
the largest filled-channel difference was one level and the worst filled
mean absolute difference was 0.000569662. Every repeated output was identical
and all known pixels were exact. Optimized processes peaked at approximately
322–528 MiB RSS. These gains combine CPU FFT, reflection gather and the GEMM
layout; they are not separate attribution measurements for each gate.
The eager formula/512 row uses an explicitly recorded quiet confirmation
after an overlapping test invalidated the initial timing arm.

The independent synthetic panel contains 5,448 original/native stage
comparisons: eight fixtures, three implementations, 227 stages each.
Seven fixtures pass every stage in all three implementations. The full-mask
fixture retains 37 eager, 34 portable and 34 optimized stage failures, despite
passing final prediction, RGB and observer-neutrality checks. Feeding the
same original input directly into all three production FFT helpers passes
the independent FFT budget. This supports an earlier-drift/DC-amplification
explanation, but the exact earlier cause remains unresolved. The full-model
strict verdict therefore remains negative.
See [the LaMa report](lama-cpu-validation.md) and its
[complete evidence](../tests/results/inpainting/lama-cpu-2026-11-01.json).
The subsequent [first-layer attribution](lama-first-layer-attribution.md)
locates a small original/native difference in the first convolution, before
the Fourier blocks. The first-block native outputs reproduce the earlier
retained payloads exactly. Alternate batch-normalization formulas did not
remove that drift, so no numerical formula or acceptance limit was changed.

### Moebius

The full four-case original/native panel has executed with relative and local
lambda reassociation plus the CPU GEMM layout, with local padding disabled.
Every case runs the original 19-step schedule twice: ordinary execution and
execution with intermediate capture. All four ordinary/observed decoded
outputs are bit-identical, and every final generated/composited RGB gate passes.

| Case | Strict intermediate failures | Generated RGB maximum / mean difference | Composited RGB maximum / mean difference |
| --- | ---: | ---: | ---: |
| Subtitle bar | 18 / 359 | 1 / 0.0002683004 | 1 / 0.00003560384 |
| Edge-touching | 13 / 83 | 1 / 0.0002250671 | 1 / 0.00003687541 |
| Center | 4 / 83 | 1 / 0.0002187093 | 1 / 0.00001398722 |
| Empty mask | 0 / 83 | 1 / 0.0001614889 | 0 / 0 |

The means in this table cover the complete corresponding RGB image, unlike
the masked-channel means used by the MI-GAN and LaMa photo checks. All known
pixels are exact. All 608 intermediate comparisons are finite and pass their
relative-L2 and cosine budgets, while 35 exceed the unchanged absolute-error
limit. **Strict intermediate parity therefore remains negative.** The first
subtitle exceedance is already present at the VAE moments; a bounded
attribution run found the same moments with the new GEMM layout enabled and
disabled. This does not establish the earlier cause or waive the failure.

The stage-capture process peaked at approximately 4.32 GiB RSS. That is a
correctness-run footprint with retained activations, not an ordinary-image
memory measurement. Its per-image times are diagnostic, not the repeated
warm benchmark. The original reference replays the original PyTorch noise
tensors; ordinary production calls use the native deterministic noise stream.
Matching seed numbers alone do not make those streams interchangeable.

The [latent graph guide](latent-inpaint-unets.md) and
[CPU evidence](../tests/results/inpainting/latent-cpu-2026-11-01.json) retain
every stage error, original/native identity, callback-neutrality check and
component timing. The selected local contraction improves its component
medians by 1.15–4.18× across the three spatial sizes; padding regresses the
two larger sizes and stays off. These component ratios are not complete-image
speedups.

The subsequent CPU depthwise integration was replayed on all four cases with
the same original inputs and fixed thresholds. All eight generated/composited
RGB outputs are byte-identical to the previous local-attention candidate;
all 608 recorded reference metric dictionaries and the same 35 failure names
also match. Actual finalized graphs report 47 specialized depthwise nodes per
denoising step, for 3,572 logical node executions across the four cases, and
zero in either VAE graph. See the
[depthwise model evidence](../tests/results/inpainting/moebius-depthwise-cpu-2026-11-01.json).
This observed replay establishes the stated output preservation. Its timings
overlapped other work and are not a complete-image speedup measurement.

The subsequent ordinary API pair also completed: one cold image and three
warm images per arm on the fixed 512-pixel formula photograph. All eight
images have the same RGB digest, including the saved cold checkpoints, and
every known pixel is exact. Both arms use the existing relative/local lambda
and GEMM choices; the candidate adds depthwise and SHA acceleration.

| Cost | Existing local/GEMM profile | Depthwise plus SHA profile |
| --- | ---: | ---: |
| Load and validation, one observation | 4.973 s | 1.277 s |
| Cold image | 166.949 s | 392.587 s |
| Three warm images | 142.796, 139.512, 266.263 s | 134.088, 181.397, 195.887 s |
| Warm median | 142.796 s | 181.397 s |
| Peak process RSS | 3,733,784 KiB | 3,727,376 KiB |

**This pair does not qualify a complete-image speedup.** The candidate's
warm median is slower, both series vary substantially, and resource snapshots
record competing work. All samples are retained. The loading observations
include verification and initialization after independent pre-run file hashing;
they do not measure cold storage. Depthwise remains a separate experimental
option rather than a recommended addition to the throughput preset. See the
[ordinary API evidence](../tests/results/inpainting/moebius-api-cpu-2026-11-01.json).

### PixelHacker

The current eager and optimized CPU implementations each completed one cold
and three warm ordinary image calls on the same 512 × 512 formula photograph.
Both used the same executable and F32 models, eight threads and the stock
20-requested-step sampler. The candidate selected persistent graphs, CPU GEMM
layout and gated linear attention, plus SHA-NI for complete model-file
verification. The depthwise candidate remains off for this profile.

| Measurement | Eager | Optimized CPU + SHA-NI |
| --- | ---: | ---: |
| Initialization and model verification | 16.535945 s | 3.195765 s |
| Cold image call | 314.108361 s | 149.552855 s |
| Warm call 1 | 310.958028 s | 144.243473 s |
| Warm call 2 | 311.826501 s | 141.174032 s |
| Warm call 3 | 306.760212 s | 145.701230 s |
| Warm median | 310.958028 s | 144.243473 s |
| Peak process RSS | 4,478,836 KiB | 6,275,732 KiB |

The observed warm-median ratio is **2.1558** and the loading/verification ratio
is **5.1743**. **All timing qualification flags remain false.** These are
diagnostic observations of one fixture, three warm calls and a fixed eager-then-
candidate run order on a shared host. The loading measurement includes native
initialization after independent pre-run hashing and does not measure cold
storage or isolate each optimization's contribution.

The independently recomputed final RGB comparison passes the unchanged native
limits: maximum filled-channel error **1**, MAE **0.0000949435764**, PSNR
**88.3561 dB**, and exact known pixels. Each implementation repeats its own
bytes across all four calls. Actual candidate CPU/F32 profiles account for
four encoder, 76 denoiser-step and four decoder runs, proving the retained
19-step schedule and context reuse. The eager arm has no equivalent graph
counter. The candidate's verified SHA profiles account for every complete
input block of both model files.

The [current API evidence](../tests/results/inpainting/pixelhacker-api-cpu-2026-11-01.json)
retains the full samples, output comparison and provenance; the
[PixelHacker guide](pixelhacker-cpu-validation.md) explains the configuration,
memory and acceptance scope. This finite native-to-native agreement does not
establish original PixelHacker parity. The original CUDA/FLA implementation
was not run, and perceptual, temporal and GPU qualification remain false.

The [earlier v2 record](../tests/results/inpainting/pixelhacker-cpu-2026-11-01.json)
remains unchanged, including its completed older candidate, user-cancelled
eager arm, unexecuted portable-graph arm, initial timing-print failure and
independent revalidation. Its samples are not pooled into this current pair.
The current plan did not require a separate portable-graph attribution arm
after native final-image agreement passed. Production CLI evidence is recorded
separately from this ordinary API comparison.

## Select the measured CPU paths

Build a Release CPU executable using the instructions in
[the graph guide](inpainting-ggml-graphs.md). Supply the exact reviewed GGUF
and checkpoint hashes required by [the engine guide](inpainting-engines.md).
Select the engine's admission gate and `CRISPEMBED_INPAINT_GRAPH=1`, then
enable only its applicable measured candidates:

| Engine | Additional candidate gates |
| --- | --- |
| MI-GAN256 / MI-GAN512 | `CRISPEMBED_MIGAN_CPU_KERNELS=1` |
| LaMa | `CRISPEMBED_INPAINT_CPU_FFT=1`, `CRISPEMBED_INPAINT_REFLECT_GATHER=1`, `CRISPEMBED_INPAINT_GEMM_LAYOUT=1` |
| Moebius | `CRISPEMBED_INPAINT_GEMM_LAYOUT=1`, `CRISPEMBED_MOEBIUS_RELATIVE_REASSOC=1`, `CRISPEMBED_MOEBIUS_LOCAL_REASSOC=1` |
| PixelHacker | `CRISPEMBED_INPAINT_GEMM_LAYOUT=1`, `CRISPEMBED_INPAINT_CPU_GLA=1` |

Keep `CRISPEMBED_INPAINT_FAST=0` for the measured F32 profile. The separately
gated Moebius local GEMM padding candidate remains off: its component timing
regressed at the two larger model geometries. The CPU attention-tile probe
also remains off; it did not remove the observed VAE discrepancy. Literal
`0` disables every new candidate. These settings are experimental choices,
not changes to production defaults.

`CRISPEMBED_INPAINT_CPU_DEPTHWISE=1` is available for an explicit Moebius
experiment. Its operator benchmark is faster and its tested final images
match exactly, but the noisy ordinary API pair above does not establish a
full-image gain. Keep it separate from the preset until it is measured on
the intended machine.

`CRISPEMBED_INPAINT_SHA256_ACCEL=1` separately enables hardware-assisted
model-file verification on supported x86-64 CPUs. It hashes every byte and
compares the same trusted complete-file digest. The scalar implementation is
retained on unsupported targets and when the flag is off. Profiling records
the verified model identity and actual complete/accelerated block counts;
requesting the flag alone is not evidence of hardware execution. See
[model hashing](inpaint-model-hashing.md) for the dispatch and controls.

The production CLI can reuse one loaded model context across a directory or
TSV batch. Prefer this when processing several frames: launching a new process
per image repeats model reading, digest verification and graph construction.
The live test runner exercises a filled/empty/filled batch and checks both
decoded bytes and the actual graph reuse profile. See
[the live runner](../tests/test_inpaint_engine_live.py).

## Reproduce an ordinary-image benchmark

The generic native benchmark target supports all five engines. It calls the
ordinary image API, verifies known pixels and repeated output, records loading
and one cold call separately, and prints every warm sample before its summary.
Its output file must be new. The photo preparer below uses Pillow to decode
two pinned repository images and does not import a model framework.
An optional `--checkpoint-output NEW_RAW` preserves the first completed cold
image before warm sampling starts. This write is outside the image timer.
An interrupted benchmark can therefore retain a completed output without
claiming that its timing series or quality validation finished. Both output
destinations must be new, distinct regular-file paths.

For the measured LaMa configuration, after obtaining the reviewed model and
its conversion receipt:

```sh
cmake --build build-inpaint --target test-inpaint-benchmark --parallel 2
python tools/dump_migan_photo_reference.py prepare \
  --size 512 --output bench-photo512

CRISPEMBED_LAMA=1 \
CRISPEMBED_INPAINT_GRAPH=1 \
CRISPEMBED_INPAINT_FAST=0 \
CRISPEMBED_INPAINT_PROFILE=1 \
CRISPEMBED_INPAINT_CPU_FFT=1 \
CRISPEMBED_INPAINT_REFLECT_GATHER=1 \
CRISPEMBED_INPAINT_GEMM_LAYOUT=1 \
./build-inpaint/test-inpaint-benchmark \
  --engine lama --model models/big-lama.gguf \
  --model-sha256 "$LAMA_GGUF_SHA256" \
  --rgb bench-photo512/formula_photo_bar/input.rgb8 \
  --mask bench-photo512/formula_photo_bar/fill-mask.u8 \
  --width 512 --height 512 --threads 8 --runs 5 --backend cpu \
  --output bench-photo512/lama-optimized.rgb8
```

Use the exact GGUF digest from the conversion receipt. Run each comparison
configuration in a fresh process with a distinct output filename and explicit
gate values: the eager baseline has graph/candidate gates off; the portable
graph has graph on and candidates off. For latent engines, add their VAE file
and digest, select the corresponding admission gate, and use `--steps 20` at
512 × 512. MI-GAN512 also requires its reviewed checkpoint digest. Raw RGB is
packed RGB8; masks contain exactly one byte per pixel, either zero or 255.

The benchmark itself does not supply an original-model quality verdict.
Original capture/comparison tools and the real PNG CLI runner provide those
separate checks. Stage profiling, benchmark raw samples and whole-process RSS
describe different costs and should remain separately labelled when comparing
results.

## Final local checks and live coverage

The resumed closeout executes **244 Python/CLI tests** and **20 native test
configurations**, all passing. The latter cover portable and selected CPU
operators, FFT/GLA, local and relative lambda attention, the separately gated
padding alternative, MI-GAN kernels, loader contracts, sampler/pipeline
contracts and image composition. The image wrapper contains 68 assertions.
Production executable and ordinary benchmark symbol audits exclude the
test-only observer APIs; an observer-enabled comparator provides the positive
control. The new SHA targets additionally pass 6,985 normal and 6,988
forced-fallback assertions, and the private ordinary API controller passes
21 mocked lifecycle/evidence tests. See the
[resumed control record](../tests/results/inpainting/validation-rest-2026-11-01.json).
The [earlier closeout](../tests/results/inpainting/validation-closeout-2026-11-01.json)
remains preserved separately.

The current production CLI binary completed **12 actual invocations and 16 PNG
outputs** across MI-GAN256, MI-GAN512, LaMa at 256 pixels, Moebius and PixelHacker.
Filled outputs match the pinned ordinary API bytes for each profile, every
known input channel is preserved, and empty-mask outputs equal their inputs.
Both latent profiles also complete filled/empty/filled batches with exact
fresh-context versus reused-context results. Actual profiles show 1/19/1
encoder/denoiser/decoder calls for a single filled image, zero graph calls for
the empty image, and 2/38/2 for the batch. Complete-model SHA-NI verification
is checked against the exact digests and executed input-block counts.
See the [final CLI evidence](../tests/results/inpainting/inpaint-cli-final-2026-11-01.json).

The earlier production CLI panel remains separate. It covers both MI-GAN
sizes and LaMa at 256/512, with its own executable, exact API comparisons and
batch reuse; MI-GAN256 also retains the boundary/tiling smoke. These historical
outputs are not added to the current binary's 16-PNG count. Functional CLI
checks do not qualify throughput, original intermediate parity or broader
image quality.

The final archive preserves initial failures, excluded timing arms, the
PixelHacker cancellation and the later timing-only revalidation. It contains
the exact committed source, pinned ggml source, complete and incremental
patches, selected raw execution evidence and actual output comparisons for
all five profiles, alongside the earlier comparison sheet. The integrated
depthwise path has separate executed evidence. The
larger convolution-workspace screen is retained as an experiment that did
not justify changing the production workspace policy.

## Limits that remain

- CUDA and Metal numerical behavior, peak device memory and latency require
  execution on those backends. The Apple implementation uses ggml's Metal
  backend; this package does not introduce a PyTorch MPS runtime.
- The original PixelHacker FLA implementation requires CUDA. CPU native-to-native
  agreement cannot establish parity with that original implementation.
- F32 weight files smaller than 4 GB do not imply a process below 4 GB.
  VAE activations, denoiser graphs and workspaces contribute additional memory.
- The photo fixtures are two source photographs at declared crop sizes.
  Representative subtitle removal, human perceptual quality, tile seams,
  difficult masks and video flicker require their own evaluation.
- The finite regression panels preserve measured outputs; they do not justify
  a default flip, unrestricted geometry, quantized weights or an application
  release. Every initial failure and excluded timing arm remains available.

The source package includes native tests, original capture tools, live CLI
checks and hosted CPU/CUDA/Metal workflows. It does not include model weights
or framework environments. Conversion receipts retain the model/source/base
licences separately; see the engine and conversion guides for acquisition and
verification instructions.
