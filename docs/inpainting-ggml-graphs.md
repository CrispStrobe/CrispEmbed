# Persistent inpainting graphs on CPU, CUDA and Metal

Issue71 now has an additional execution path that builds complete, reusable
ggml graphs for MI-GAN256, MI-GAN512, Big-LaMa, Moebius and PixelHacker. Select
it with `CRISPEMBED_INPAINT_GRAPH=1`, together with the model's existing gate.
The earlier CPU implementations remain available for controlled comparison.

**Evidence status, 2026-11-01:** native CPU builds, component tests and operator
benchmarks have now run on an explicitly admitted AMD EPYC machine with an
8-CPU quota and 8 GiB memory limit. The initial spectral replay failure was
diagnosed and corrected before its performance baseline was measured. The
[CPU optimization report](inpainting-cpu-optimization.md) and its
[raw component evidence](../tests/results/inpainting/cpu-components-2026-11-01.json)
record both the failure and subsequent results. MI-GAN also passed original
stage/final-image checks on 20 cases across its 256- and 512-pixel models;
see [its separate CPU evidence](../tests/results/inpainting/migan-cpu-2026-11-01.json).
LaMa has also completed its synthetic original-model matrix: seven of eight
fixtures pass all 227 stages in every CPU variant, while the full-mask fixture
retains strict intermediate failures despite passing final RGB checks. See
[the LaMa validation report](lama-cpu-validation.md); complete numerical
qualification is explicitly false. Latent-model qualification remains separate.
CUDA and Metal execution has not been measured on this CPU-only machine, and
component speedups are not end-to-end performance claims.

The Apple path is ggml's native **Metal** backend. `mps` is accepted as a CLI
alias for `metal`; it does not introduce PyTorch or an MPS runtime dependency.
The existing source/checkpoint identities and weight-license records remain
in [the engine guide](inpainting-engines.md). The downloadable source contains
no model weights.

## What the execution path changes

| Model | Reusable graph and device-residency changes |
| --- | --- |
| MI-GAN256 / MI-GAN512 | One complete production graph and allocation per context; weights uploaded once; 1x1 convolutions bypass im2col; Metal-compatible padding; distinct retained-reference graph only in validation builds. |
| Big-LaMa | Complete FFC generator graph; real/imaginary Fourier activations stay on the selected backend; geometry constants cached; one bounded graph cache for the current image geometry. |
| Moebius | Complete denoiser graph; lambda contractions use batched matrix operations; local positional convolutions batch the value planes; shared VAE, CFG and DDIM execute on the selected backend. |
| PixelHacker | Complete denoiser graph; rectangular per-key gated attention is decomposed into bounded device operations; shared VAE, CFG and DDIM execute on the selected backend. |

The runtime reuses `core_gguf` ownership/mmap support, the ggml backend registry,
the existing GPU preference and Metal cache policy, CPU backend facilities,
`core_env` gates and the existing input, mask, tiling, PNG and file-protection
code. The pinned ggml revision is unchanged. There is no new vendor FFT
dependency. An optional CPU FFT callback operates directly on CPU backend
buffers; GPU graphs continue to use portable device operations.

### Diffusion graphs and transfers

Moebius and PixelHacker use three cached graphs: VAE encoding, a reusable
denoising/CFG/DDIM step, and VAE decoding. Encoding batches the image and masked
image. Denoising batches the unconditional and conditional branches. The latent,
masked latent and downsampled mask live in explicitly allocated backend state.
The denoiser's final copy updates that state on the same device.

Each sampling iteration uploads a 320-float sinusoidal timestep embedding and
four DDIM coefficients: **1,296 input bytes**. Activations and predicted noise
are not downloaded between sampling iterations in the production path. Initial
images, masks and deterministic noise are uploaded once per tile, and decoded
RGB is downloaded after the decoder. Validation builds deliberately download
retained stages; their timing must not be used as production performance.

The sampler retains the original DDIM arithmetic order, independent posterior
noise streams, mask semantics and model-specific timestep schedule. Batch-two
group normalization merges the sample axis into channels while multiplying
the number of groups, because this pinned Metal kernel does not process the
fourth dimension. This preserves independent normalization of each sample.

### Fourier transforms and gated attention

By default, LaMa's orthonormal real transforms use a separable radix-2 butterfly graph.
Non-power-of-two axes use Bluestein convolution, with a fixed kernel computed
from geometry only. Supported Fourier axes are 1 through 128, covering the
existing generator's maximum 1024-pixel image dimension after its three
downsampling stages. Inverse transforms preserve the real projection required
when learned spectral layers produce non-Hermitian boundary frequencies.

`CRISPEMBED_INPAINT_CPU_FFT=1` selects a CPU-only alternative: one ggml CPU
custom node per transform, with cached radix-2/Bluestein geometry adapted from
the existing LaMa FFT implementation. Inputs and outputs remain F32; internal
FFT arithmetic uses F64 complex values. The callback accesses resident CPU
buffers without intermediate backend transfers. CUDA/Metal do not select it.
Its finite geometry cache and worker scratch are separate from graph allocator
bytes. This flag is default off; the [CPU report](inpainting-cpu-optimization.md)
documents measured component gains, unchanged error gates and memory costs.

PixelHacker needs unequal key/value widths and a distinct decay for each key
channel. The pinned `ggml_gated_linear_attn` operator requires square heads and
does not supply the required Metal path. The implementation therefore uses
64-token chunks by default, stable log-gate prefix differences, causal within-chunk
contractions and a device-resident rectangular carry state. Every invocation
starts with independent zero state. This is a portable graph decomposition;
it does not claim the throughput of the original specialized Triton kernels.

`CRISPEMBED_INPAINT_CPU_GLA=1` enables an additional CPU-only path through the
pinned F32 recurrent kernel. It splits rectangular value heads into square
groups, repeats Q/K/per-key decay, pads the last group when needed and retains
the original query scale. Independent batch/head state starts at zero on
every invocation. The flag is default off, and GPU graphs retain the portable
chunk path. The benchmark can sweep chunk sizes 8/16/32/64 without changing
the production default. Component tests cover unequal widths, tails, extreme
gates, causality and replay; complete PixelHacker acceptance is a separate gate.

### CPU execution and bounded workspaces

CPU graphs reuse both their execution plan and a worker pool. The pool is
resolved from the pinned CPU registry, including dynamic-backend builds;
older modules without all required pool functions keep ordinary ggml threading.
MI-GAN reports whether pool and plan reuse actually occurred. Persistent
parameters and immutable initializers are retained across replay: the pinned
allocator requires `OUTPUT` as well as `INPUT` for these allocated tensors.
Actual input/constant bytes are checked by the replay controls. Latent GGUF
conversion also uses an explicit, collision-checked compact encoding for
source tensor names that exceed ggml's 64-byte field, preserving the original
names in the schema and conversion receipt. See the
[correction evidence](inpainting-cpu-optimization.md#the-corrected-baseline).

Dense 1x1 operations avoid im2col. Other dense convolutions use row strips with
halos, targeting a 32 MiB patch workspace on CPU and 128 MiB on GPU. These caps
apply to the patch temporary, not total memory. Weights, skip activations,
multiple cached component graphs, FFT/GLA intermediates and backend workspace
also consume memory. The earlier “under 4 GB” model-size criterion is not a
guarantee that the entire application fits in 4 GB of RAM or VRAM.

`CRISPEMBED_INPAINT_CPU_DEPTHWISE=1` selects a restricted CPU specialization
for contiguous F32 depthwise convolutions with a 3x3 kernel, multiplier one,
stride one, padding one and dilation one. It adapts the existing MI-GAN
tap-major interior loop, with scalar borders and no intermediate activation
transfers. Other geometries retain the ordinary dispatch. Finalized graph
profiles include `cpu_depthwise_nodes`, so tests can distinguish actual custom
node selection from a requested flag. The four-case Moebius replay selects
47 such nodes per denoising step and preserves the prior final RGB exactly;
its original intermediate failures remain documented in the
[depthwise evidence](../tests/results/inpainting/moebius-depthwise-cpu-2026-11-01.json).

Model-file verification has a separate default-off
`CRISPEMBED_INPAINT_SHA256_ACCEL` gate. Supported x86-64 CPUs use SHA-NI for
complete input blocks; the original digest, padding and file-admission checks
are retained. This affects loading rather than graph arithmetic. See the
[model hashing guide](inpaint-model-hashing.md) for exact verification tests
and actual execution profiles.

LaMa's transposed convolutions use exact zero insertion followed by bounded
convolution/GEMM. Flipped/channel-transposed immutable kernels and insertion
masks are prepared once and retained; mixed GPU kernels remain F16. This
avoids the pinned direct transposed-convolution loop for dense layers. Row
strips and strict attention tiles use balanced concatenation to avoid copying
every earlier tile again at each append. The small-layer/CPU crossover versus
the retained direct implementation still requires timing on the target device.

## Backend and precision selection

| Setting | Meaning |
| --- | --- |
| `CRISPEMBED_INPAINT_GRAPH=1` | Enable the new persistent graph path. Default off. |
| `--gpu-backend cpu` | Execute the graph on the CPU backend, honoring `--threads`. |
| `--gpu-backend cuda` | Require a CUDA backend and its supported graph operations. |
| `--gpu-backend metal` | Require native Metal. `mps` is an alias. |
| `--gpu-backend auto` | Prefer CUDA, then Metal, then CPU. Operation validation still applies to the selected backend. |
| `CRISPEMBED_INPAINT_BACKEND=cpu\|cuda\|metal\|auto` | Backend selection for native callers and reference programs. An explicit process/CLI preference takes precedence. |
| `CRISPEMBED_INPAINT_FAST=1` | Explicit GPU mixed-precision diagnostic profile. Default off. |
| `CRISPEMBED_INPAINT_CPU_FFT=1` | CPU-only cached FFT callback; F32 storage and F64 FFT arithmetic. Default off; GPU uses the portable graph. |
| `CRISPEMBED_INPAINT_CPU_GLA=1` | CPU-only F32 recurrent adapter for rectangular GLA. Default off; GPU uses portable chunks. |
| `CRISPEMBED_INPAINT_PROFILE=1` | Emit actual backend, graph/allocation/reuse/transfer counts and compute timing. |

An explicitly requested CUDA or Metal backend does not silently fall back to
CPU. Before allocating/running a graph, the implementation checks every node
against the selected backend and reports an unsupported operation. There is no
CPU scheduler fallback for intermediate layers. Allocation or execution errors
are returned through the existing error path; output composition remains
transactional.

**Strict portable profile:** original F32 weights and activations, explicit
F32 matrix precision, and F32 spectral/GLA arithmetic. The opt-in CPU FFT
candidate keeps F32 storage and uses F64 internally; it is reported separately
as `f32-storage-f64-fft`. The CPU recurrent GLA candidate remains F32.
The pinned Metal flash-attention
kernel uses half matrix tiles and refuses the F32 precision request, so strict
GPU attention uses bounded F32 matrix/softmax tiles. CPU uses the existing
tiled F32 flash-attention implementation. Strict Metal can therefore be slower
than its mixed profile; it is the comparison baseline.

**Fast profile:** LaMa and latent dense/matrix weights are converted to F16 once
while loading the original, digest-verified F32 GGUF. Depthwise and normalization
parameters remain F32. Dense matrix operations permit backend-default precision;
attention uses the native flash kernel with F16 keys and values. MI-GAN retains
F32 weight storage and explicitly permits backend-default matrix precision.
FFT and GLA contractions retain their stricter numerical treatment. The file
format, source digests and model contract are unchanged. This profile must pass
the same full-model acceptance criteria before adoption; component diagnostic
tolerances do not qualify final images.

## Build and run

Build on an appropriate development/CI host. The source archive includes the
pinned `ggml/` contents; a Git checkout needs its submodule initialized. A
common configuration is:

```sh
cmake -S . -B build-inpaint -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DCRISPEMBED_NATIVE=ON \
  -DGGML_LLAMAFILE=ON -DGGML_BLAS=OFF -DGGML_VULKAN=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-inpaint --config Release --target crispembed-cli --parallel 2
```

For CUDA, configure with `-DGGML_CUDA=ON -DGGML_METAL=OFF` and
`-DGGML_CUDA_GRAPHS=ON`; use a CUDA architecture supported by the intended
deployment GPU. The hosted driver uses `-DCMAKE_CUDA_ARCHITECTURES=native` for
its admitted machine. Stable graph identities and allocations allow ggml's
CUDA graph reuse when the backend admits that graph; no unconditional CUDA
capture success is claimed.

For Apple Silicon, configure with `-DGGML_METAL=ON -DGGML_CUDA=OFF`,
`-DCMAKE_OSX_ARCHITECTURES=arm64` and `-DGGML_METAL_EMBED_LIBRARY=ON`.
For portable distribution, use the repository's portable CPU configuration
instead of `NATIVE=ON`. A machine-specific build is not a portable release.

Example using a converted, reviewed LaMa model:

```sh
CRISPEMBED_LAMA=1 \
CRISPEMBED_INPAINT_GRAPH=1 \
CRISPEMBED_INPAINT_PROFILE=1 \
./build-inpaint/crispembed \
  --inpaint-engine lama \
  --inpaint-model big-lama-f32.gguf \
  --inpaint-model-sha256 "$LAMA_GGUF_SHA256" \
  --gpu-backend cuda \
  --inpaint frame.png --mask mask.png \
  --inpaint-output repaired.png --threads 8
```

Use `--gpu-backend metal` on Apple hardware or `--gpu-backend cpu` for the CPU
path. Add `CRISPEMBED_INPAINT_FAST=1` only for an explicitly selected mixed GPU
trial. Use the existing engine guide for MI-GAN512's reviewed checkpoint hash
and the latent models' VAE/model receipt hashes. The same settings apply to
directory and TSV batches; those modes reuse contexts and graph allocations.

## Qualification and benchmarking

The manual workflow is
[inpaint-graphs.yml](../.github/workflows/inpaint-graphs.yml), driven by
[inpaint_graphs.py](../tools/ci-heavy/inpaint_graphs.py). CPU runs use hosted
Linux. Metal and CUDA require explicit admission of provisioned native runners.
A standard macOS VM is not assumed to expose an Apple GPU. The workflow
downloads no models and records source, ggml, executable and configuration
identities in its build receipt.

Four model-free native controls exercise the actual implementation:

- `test-inpaint-graph`: convolution, patch-strip seams, normalization including
  batch two, padding, attention tails and resident recurrent state.
- `test-inpaint-graph-spectral`: radix-2/Bluestein versus independent FFT/DFT
  references, arbitrary learned spectra, rectangular GLA, chunk tails,
  extreme gates, causality and replay.
- `test-latent-inpaint-graph-unet`: scalar lambda-content/local/relative and
  headwise gated-normalization comparisons, two batches and malformed geometry.
- `test-migan-graph --contracts`: the actual MI-GAN graph helpers versus scalar
  expectations, explicit backend selection and replay.

Each test accepts `--backend cpu|cuda|metal`. A passed component check is not
a passed full model. Existing original-stage and final-image comparators keep
their original thresholds. Once original captures and converted models exist,
[inpaint_graph_replay.py](../tools/ci-heavy/inpaint_graph_replay.py) runs LaMa or
either latent comparator on an explicit graph backend, optionally alongside
the eager CPU baseline. It verifies supplied hashes, records binary/build
identity and requires actual graph profiler evidence. It never recaptures or
rewrites the original tensors to make a comparison pass.

`test-migan-graph --model ...` provides same-process cold/warm measurement,
changed-input replay, graph/weight reuse, baseline comparison and observer
neutrality for a reviewed local MI-GAN model. The workflow's optional model
phase requires the model and checkpoint hashes. For all engines, record load
and first-graph costs separately from steady inference, include geometry,
mask, precision, thread count, hardware and backend versions, and retain
unaltered numerical failures alongside timings.

Measure CPU thread counts on the intended machine and retain the fastest
configuration that passes the same comparisons. Measure strict and mixed GPU
profiles separately. Observer runs retain activations and copy them to the
host; they are correctness runs, not timing runs. Graph allocator bytes are
not peak process RSS, driver allocation or peak VRAM. Representative image and
subtitle quality, seam behavior and temporal video consistency remain separate
acceptance tasks. The measured CPU candidates remain default off; promotion
requires the corresponding full-model and target-hardware evidence.

### Full-model CPU measurements

These are completed CPU measurements and explicit qualification limits. The
MI-GAN and LaMa timing rows use five warm ordinary API calls after one cold
call. Moebius's stage-capture run is not an ordinary performance benchmark.
PixelHacker now has a completed eager/candidate ordinary API pair, with three
warm calls per arm and passing native final-image agreement. Its timing
qualification flags remain false; the measurements are diagnostic observations
from one fixture and a fixed run order on this shared CPU host.

| Engine / configuration | Original-reference result | Ordinary-API warm median | Runtime peak memory | Evidence |
| --- | --- | --- | --- | --- |
| MI-GAN256 / 8 threads / optimized CPU | Original panel passes; captured native stages unchanged | 273.62 ms | About 196 MiB RSS | [MI-GAN record](../tests/results/inpainting/migan-cpu-2026-11-01.json) |
| MI-GAN512 / 8 threads / quiet confirmation | Original panel passes; captured native stages unchanged | 495.24 ms | About 373 MiB RSS | [MI-GAN record](../tests/results/inpainting/migan-cpu-2026-11-01.json) |
| LaMa / formula512 / 8 threads / optimized CPU | Photo RGB passes; synthetic full-mask strict stages fail | 4,116.47 ms | Photo arms about 322–528 MiB RSS | [LaMa record](../tests/results/inpainting/lama-cpu-2026-11-01.json) |
| Moebius / local and relative reassociation, GEMM layout | All four final RGB gates pass; 35/608 strict stage failures; ordinary paired outputs exact | 142.796 s baseline; 181.397 s with depthwise + SHA (+27.03%); timings unqualified | 3,733,784 / 3,727,376 KiB ordinary process RSS | [Ordinary API pair](../tests/results/inpainting/moebius-api-cpu-2026-11-01.json), [original replay](../tests/results/inpainting/latent-cpu-2026-11-01.json) |
| PixelHacker / 8 threads / eager versus optimized CPU + SHA | Original CUDA reference unavailable; native formula512 RGB passes max error 1 and exact known pixels | 310.958 s eager; 144.243 s candidate; diagnostic ratio 2.1558, timings unqualified | 4,478,836 / 6,275,732 KiB process RSS | [Current PixelHacker API pair](../tests/results/inpainting/pixelhacker-api-cpu-2026-11-01.json) |

The [complete results](inpainting-optimization-results.md) distinguish image
checks, intermediate limits, timing exclusions and incomplete comparisons.
No row establishes CUDA/Metal performance, broad perceptual quality or a
release/default promotion. A stored model smaller than 4 GB does not imply a
process below 4 GB.

The [current production CLI panel](../tests/results/inpainting/inpaint-cli-final-2026-11-01.json)
separately passes 12 invocations and 16 PNG outputs across all five profiles
on one executable, with exact API comparisons, empty masks, actual SHA-NI
verification and both latent filled/empty/filled batches. Functional CLI
success does not qualify device throughput or erase the original-stage limits.
