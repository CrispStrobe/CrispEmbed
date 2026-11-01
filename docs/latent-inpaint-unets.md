# Native Moebius and PixelHacker denoisers

The source in `src/latent_inpaint_unet.cpp` implements both complete author
denoisers at the published 512-pixel image geometry. It consumes nine
`64 x 64` latent channels: four noisy-image channels, one white-fill mask
channel, and four encoded masked-image channels. It produces four predicted
noise channels. This is an experimental CPU/F32 path; implementation and
component checks do not establish full model parity or usable CPU latency.

This document describes the retained eager correctness implementation. The
additional `src/latent_inpaint_graph_unet.cpp` builds both complete denoisers
as backend-resident ggml graphs, including batch-two guidance. See
[persistent CPU/CUDA/Metal execution](inpainting-ggml-graphs.md) for its
operator decompositions, mixed profile and outstanding hardware acceptance.

## Source contracts

| Component | Pinned original source |
| --- | --- |
| Moebius UNet, lambda attention, depthwise residual blocks and MixFFN | [hustvl/Moebius at b88d462bacb9af6e7128a3b4cc4a07418bedfd61](https://github.com/hustvl/Moebius/tree/b88d462bacb9af6e7128a3b4cc4a07418bedfd61) |
| PixelHacker UNet and GLA configuration | [hustvl/PixelHacker at f5567db2871598aa178fe7a34c520dd478a0b41b](https://github.com/hustvl/PixelHacker/tree/f5567db2871598aa178fe7a34c520dd478a0b41b) |
| Gated linear attention equations and layer defaults | [flash-linear-attention v0.3.2](https://github.com/fla-org/flash-linear-attention/blob/v0.3.2/fla/layers/gla.py) |
| PixelHacker standard transformer and timestep behavior | [Diffusers v0.30.2](https://github.com/huggingface/diffusers/tree/v0.30.2/src/diffusers/models) |

The loader restores original `diff_model.*` and `embedding_layer.weight`
names from the physical GGUF keys. Long names use the deterministic aliases
described in [bounded GGUF tensor names](latent-inpaint-conversion.md#bounded-gguf-tensor-names);
canonical schema and observer labels stay unchanged. Conversion collapses
only the singleton depth axis of Moebius'
`[K,1,1,15,15]` positional convolution to `[K,1,15,15]`; the storage order and
coefficients do not change. Batch normalization running means and variances
remain part of inference. Its training counters are not arithmetic inputs.

Both models use a learned `20 x 3072` category embedding and a `3072 -> 768`
projection. Conditional execution selects rows 0–9; the unconditional
classifier-free guidance execution selects rows 10–19. There is no text
tokenizer or text encoder. The timestep embedding is 320 channels, cosine
then sine, with frequency shift zero; its two-layer SiLU MLP has width 1280.

## Reused CrispEmbed components

The denoisers use `latent_inpaint::Engine`, whose GGUF ownership and CPU
execution are shared with the VAE. All ordinary and grouped convolutions,
channel projections, group/layer/batch normalization, nearest upsampling and
standard cross-attention go through that engine. It reuses the repository's
GGUF loader, ggml CPU execution and normalization facilities. Lambda content
aggregation and GLA head normalization reuse `core_cpu::dot_product`,
`core_cpu::softmax` and `core_cpu::rmsnorm_cpu`.

Qwen's gated delta attention is not mathematically interchangeable with GLA.
The new GLA kernel implements the actual key-gated recurrent equation, with
fresh state for every image, timestep and guidance branch. Its equivalence
to the upstream chunked GPU kernel is mathematical; its floating-point
summation order differs and must be measured during model qualification.

## Architecture details that affect output

### Moebius

There are three down blocks with two residual/attention pairs each, no middle
block, and three up blocks with three pairs each. All nine encoder skip
tensors are consumed. The feature widths are 320, 640 and 1280.

The depthwise-separable convolution is the original timm block, including
depthwise convolution, batch normalization, ReLU, pointwise convolution,
second batch normalization, and an internal residual when the channel counts
match. Each enclosing residual block also has its own outer skip. Removing
these internal operations would change the trained model.

Local lambda self-attention uses eight query heads, shared keys/values,
evaluation-mode batch normalization on query/value, and a softmax over key
positions. The positional contribution is the original 15-by-15 convolution.
Its singleton depth dimension lets native code stream independent value
planes through the shared convolution operator without constructing a large
five-dimensional activation.

The persistent graph also has a default-off local contraction candidate,
`CRISPEMBED_MOEBIUS_LOCAL_REASSOC=1`. It projects each query through the learned
spatial kernel and then combines those coefficients with value neighborhoods.
It retains the separate `Q` dot convolution-bias term. Row strips cap the
combined patch and coefficient tensors at 32 MiB on CPU and 128 MiB on GPU;
that cap excludes the rest of the graph, halos and retained partial outputs.

The separate CPU-only `CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM=1` candidate appends
zeros to align the reduction dimensions to multiples of 16. For a 15-by-15
kernel the neighborhood width becomes 240, and a 40-channel query becomes
48 channels for that matrix product. It accounts for both original and padded
patch copies in the strip cap and rejects a geometry whose single row exceeds
the budget. Padding is effective only with local reassociation enabled on CPU.
These operations preserve the mathematical learned function; their F32
reduction order changes. The original graph remains the default while the
independent scalar, replay, measured speed and image checks decide admission.
The measured CPU candidate uses local reassociation with padding disabled;
the component results below explain that selection. It is not a default
change or a claim that all original intermediate parity gates pass.

Interactive lambda cross-attention uses the ten category tokens and the
learned spatial `rel_pos_emb` tensor. No positional interpolation is performed;
the fixed geometry is validated. MixFFN uses pointwise expansion and SiLU,
depthwise convolution, a split SiLU gate, and pointwise contraction.

### PixelHacker

There are four down blocks with two residual pairs each, a residual/attention/
residual middle block, and four up blocks with three pairs each. Attention is
present in the first three down blocks, the middle block, and the last three
up blocks. All twelve encoder skip tensors are consumed.

GLA self-attention uses eight heads, key width `C/2`, value width `C`, a
16-dimensional key-gate projection, and
`exp(logsigmoid(gate_logits)/16)` key-state decay. The query scale is
`1/sqrt(head_key_width)`. A current token updates the state before producing
its output. Per-head RMS normalization uses epsilon `1e-6`, followed by the
learned output SiLU gate and output projection. This is causal over raster
tokens, matching the original FLA operator; there is no invented reverse pass.

Cross-attention uses ordinary eight-head scaled dot-product attention against
the ten category tokens. Its output projection retains its bias. GEGLU uses
the exact erf GELU, rather than the tanh approximation used by some other
CrispEmbed models. Spatial transformer group normalization uses `1e-6`;
residual group normalization and transformer layer normalization use `1e-5`.

## Validation and boundaries

`tests/test_latent_inpaint_attention.cpp` exercises the production timestep,
GLA, lambda-content and relative-lambda kernels with independent analytical
answers. It covers multiple heads, gate decay, causal ordering, state reset,
softmax axis, learned relative-position layout, nonfinite inputs and malformed
dimensions. This is a hosted component target, separate from model inference.

The complete graph exposes original module names through the test-only engine
observer. Freeze original image/mask/latent noise inputs, then compare the
first differing module, errors and magnitudes for both guidance branches and
the final composited RGB image. The first CPU execution results are recorded
below. Full intermediate parity, representative image quality and portable
performance remain unqualified.

The entry point requires the corresponding value-parsed default-off gate,
`CRISPEMBED_MOEBIUS=1` or `CRISPEMBED_PIXELHACKER=1`. Only the declared
fixed-512 model configuration is implemented. Other checkpoints, dynamic
geometry, quantization and GPU acceleration require their own qualification.

## Executed CPU evidence

The [latent CPU evidence record](../tests/results/inpainting/latent-cpu-2026-11-01.json)
preserves the actual runs on an eight-CPU-quota AMD EPYC 9V74 host with an
8 GiB memory limit. No CUDA or Metal hardware was available. It records
successful checks and failures under the original, unchanged acceptance
budgets; a passing final image is not treated as a passing intermediate replay.

The original Moebius CPU capture completed all four frozen synthetic cases:
`subtitle_bar`, `edge_touch`, `center` and `empty`. Each case used the full
20-requested/19-executed-step schedule and ran once normally and once with
noise replay and observation. Every pair was bit-identical. The capture
contains 608 stage records and reached a measured process peak of
3,121,936 KiB RSS. Its explicit CPU import profile is described below.

### Full synthetic panel

The selected `candidate-local` profile enables the relative-lambda, GEMM-layout
and unpadded local-lambda candidates. Its complete four-case replay used eight
CPU threads, strict F32, all 19 denoiser steps, and both ordinary and observed
native execution for every case. All generated/composited RGB gates passed;
all known pixels were exact; and each ordinary/observed decoded-float pair
was bit-identical. The fixed intermediate gate still fails in three cases:

| Case | Failed original stage gates | Generated RGB maximum / MAE | Composited RGB maximum / MAE |
| --- | ---: | ---: | ---: |
| `subtitle_bar` | 18 / 359 | 1 / 0.0002683004 | 1 / 0.00003560384 |
| `edge_touch` | 13 / 83 | 1 / 0.0002250671 | 1 / 0.00003687541 |
| `center` | 4 / 83 | 1 / 0.0002187093 | 1 / 0.00001398722 |
| `empty` | 0 / 83 | 1 / 0.0001614889 | 0 / 0 |

All 608 compared stages are finite and pass the frozen relative-L2 and cosine
limits. The 35 failures exceed the unchanged absolute-error limit of 0.002.
The first failing stages are `vae.image.moments` for `subtitle_bar`,
`unet.step15.unconditional` for `edge_touch`, and
`unet.step18.unconditional` for `center`. The empty-mask case still executes
the complete model and compares its generated RGB, so its result is more than
an empty-mask compositing check. **Full original intermediate parity remains
failed; the acceptance limits were not widened.**

The process completed in 1,121.714 seconds with a peak of 4,526,532 KiB RSS,
including four ordinary and four observed forwards. Individual ordinary
forwards took 129.988–137.854 seconds. These are diagnostic timings across
different cases, not a repeated same-input performance benchmark. The
preserved executable, unchanged before/after digest, source observations,
conversion receipts, every compared stage and all failures are recorded.

### Earlier diagnostic and attribution

The first native diagnostic selected `subtitle_bar` from that intact original
capture and enabled the relative-lambda and GEMM-layout candidates. It ran
the ordinary graph path and then its observed replay on the same inputs:

| Check | Observed result |
| --- | --- |
| Native ordinary/observed decoded floats | Bit-identical |
| Generated RGB against original | Maximum error 1; mean absolute error 0.000281016 |
| Composited RGB against original | Maximum error 1; mean absolute error 0.0000343323 |
| Known input pixels | Byte-identical |
| Intermediate stage gates | 18 of 359 fail the fixed maximum-absolute-error gate |
| Earliest divergence | `vae.image.moments`: maximum error 0.00206852 versus the 0.002 limit |
| Process peak RSS | 4,515,944 KiB |
| Complete intermediate parity | **Failed; no qualification claim** |

The fixed relative-L2 and cosine checks pass for those intermediate failures.
A bounded encoder-only attribution run found byte-identical outputs with the
new GEMM layout enabled and disabled. An alternative strict F32 attention
decomposition and the retained eager path also exceed at least one original
absolute-error gate. These observations exclude the GEMM layout as the source
of the earliest drift; they do not establish a unique numerical root cause.
All failed metrics, original identities and the first divergent native tensor
identity remain in the evidence record.

The first ordinary native forward took 160.888 seconds; its graph profile
attributed 135.500 seconds to 19 denoiser steps, 12.169 seconds to the encoder
and 13.174 seconds to the decoder. These are diagnostic observations, with
the process receipt explicitly marking timing unqualified. They motivate the
separately gated local-lambda contraction experiment, not a full-model speed
claim. The original capture also records diagnostic times rather than a
controlled original/native speed comparison.

A subsequent strict F32 component run compared the original local positional
contraction, bounded row-strip reassociation, and row strips with CPU GEMM
zero padding. All three passed the same independent scalar, strip-seam,
finite-value, immutable-input and repeated-output checks. The run used eight
CPU threads, one warmup and five timed executions per variant, rotating their
execution order. Timings include graph computation and exclude construction,
allocation and explicit activation transfers.

| Feature geometry, batch 2 | Original median | Row strips median | Padded row strips median | Row-strip ratio |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64, query/value width 40 | 276.67 ms | 239.72 ms | 325.27 ms | 1.15× |
| 32 × 32, query/value width 80 | 193.21 ms | 101.37 ms | 109.09 ms | 1.91× |
| 16 × 16, query/value width 160 | 161.48 ms | 38.60 ms | 33.61 ms | 4.18× |

The largest full-output difference between unpadded row strips and the
original component was 0.00000524521, below the unchanged component tolerance.
At the three geometries, its graph allocation fell from approximately
137.0/125.9/140.3 MiB to 59.2/42.4/33.6 MiB. These are component arenas,
not total model memory. The zero-padding variant was slower at the two
larger resolutions, so the selected next full-image profile enables
`CRISPEMBED_MOEBIUS_LOCAL_REASSOC=1` and explicitly sets
`CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM=0`.

The same preserved executable measured the relative-lambda reassociation at
4.73×, 7.65× and 12.42× the original component speed for those geometries.
Every timing sample, executable identity, observed source identity and
numerical result is retained in the evidence record. The short samples vary;
their ratios must not be presented as an end-to-end speedup. Complete-image
measurement and the original parity failures remain separate evidence.

The completed row-strip subtitle run took 137.854 seconds ordinarily, versus
160.888 seconds in the earlier native diagnostic. Its denoiser profile fell
from 135.500 to 112.469 seconds. This comparison remains diagnostic because
it uses distinct preserved binaries and separate executions. The two native
composited images differ in only three channel values, each by one RGB level;
the generated images differ in 20 channel values, also by one level. Both
runs retain their original strict-stage failures.

Both Moebius and PixelHacker full graphs were separately constructed and
allocated with the actual verified GGUF files, including observed graphs.
Those probes performed no model computation and are not inference memory or
quality results. The original PixelHacker FLA kernel requires CUDA; no CPU
substitute was presented as original PixelHacker evidence.

## Native reference replay

`test-latent-inpaint` is a separate executable linked to the observer-enabled
test archive. Production archives must contain neither the public reference
entry point nor engine observer setters. The hosted driver checks these symbols
as well as building the actual `crispembed` CLI.

The original capture protocol and immutable checkpoint identities are described
in [the conversion/reference guide](latent-inpaint-conversion.md). After actual
original capture and exact conversion, replay one model using its real converter
receipt hashes:

```sh
CRISPEMBED_MOEBIUS=1 build/test-latent-inpaint \
  --model moebius \
  --denoiser artifacts/moebius-places2-f32.gguf \
  --vae artifacts/sdxl-vae-f32.gguf \
  --model-sha256 ACTUAL_DENOISER_GGUF_SHA256 \
  --vae-sha256 ACTUAL_VAE_GGUF_SHA256 \
  --reference-dir artifacts/moebius-original \
  --out artifacts/moebius-native --threads 2
```

The PixelHacker invocation changes the model/path and sets
`CRISPEMBED_PIXELHACKER=1`. The placeholders are the SHA-256 values in the
adjacent `.gguf.json` files **after conversion**. They are deliberately not
fabricated hashes of uncreated files. All output directories must be new.

### Original Moebius execution on a CPU

The original package initializer imports an unused PixelHacker teacher, whose
FLA/Triton import requires an active GPU driver. The Moebius student itself
uses lambda attention and can execute on a CPU. The reference dumper therefore
offers the explicit `--import-profile moebius-student-only-cpu-v1` option with
`--model moebius --device cpu`. Ordinary upstream package imports remain the
default, `upstream-package-imports-v1`.

The CPU profile skips only `model_lib/__init__.py` and
`removal/v1_2/__init__.py`. It installs package search paths and imports the
actual pinned student, block, removal-model and pipeline modules under their
original names. The skipped initializers import the unused teacher and a
training dataset. No numerical source, layer, attention equation or FLA
operation is replaced. This profile does not make the original PixelHacker
GLA kernel CPU-compatible.

The adapter requires a fresh process. It rejects previously imported
`model_lib`, `removal` or `utils_infer` modules, verifies the pinned source
before importing, and then checks the paths and hashes of the numerical
modules actually loaded. The original dependency versions, checkpoint
identity and ordinary-versus-observed bit-exact replay checks still apply.
The top-level reference report records the profile, the exact skipped
initializer list, `numerical_sources_modified: false`, and the import
adapter's SHA-256. Native replay validates these paired metadata fields and
rejects unknown profiles or a student-only profile used for another model or
device. See [the adapter](../tools/latent_inpaint_original_import.py) and its
[small import controls](../tests/test_latent_inpaint_original_import.py).

Before loading a model, the runner validates the original run and each case's
complete manifest, exact normalized RGB/mask fixture, explicit random inputs,
all tensor shapes, raw byte counts, hashes and finite-value statistics. A
separate architecture-derived inventory requires every expected stage; removing
a failing stage from the manifest cannot turn a run into a pass. Unknown keys,
duplicate JSON names, nonfinite JSON numbers, path traversal and symlinked
reference files are rejected.

The conversion receipts must match the pinned source, original checkpoint,
source contract, tensor census, permitted layout transforms and actual GGUF
digest. The runner independently streams and hashes every GGUF tensor against
its receipt. Production loading then validates the same source/shape/F32
profile before inference.

The native reference entry executes the real image encoder, learned denoiser,
DDIM loop, decoder and final pixel composition, using the original's recorded
random draws. Internal UNet stages are compared in both classifier-free
guidance branches on the first step of `subtitle_bar`; all cases compare both
VAE encodes, all 19 step outputs, guided noise, decoder floats and generated/
composited RGB. Native-only GLA recurrent-state observations are checked for
finite values and explicitly counted as unreferenced diagnostics.

All three floating gates must pass: maximum absolute error ≤0.002, relative
L2 ≤0.003 and cosine ≥0.99999. Generated and composited RGB must have maximum
error ≤2 and mean absolute error ≤0.1. Every known input pixel must remain
byte-identical. These budgets are fixed in the source contract before model
execution and cannot be relaxed through runner arguments.

Each case writes `stages.jsonl` as observations arrive, then a complete report
with all failures in original stage order. The earliest numerical divergence
also retains its native F32 tensor and original file identity. Both final native
RGB arrays are saved. A crash or deadline therefore leaves partial evidence;
an incomplete run is never recorded as a pass. A one-case diagnostic is marked
separately from a successful complete four-case synthetic panel. Neither flag
claims representative image quality, speed, memory use or release qualification.

## Hosted workflow

[`.github/workflows/latent-inpaint.yml`](../.github/workflows/latent-inpaint.yml)
has a manual dispatch only. Its default `build-only` phase runs on Ubuntu,
builds the native components and actual CLI, and runs small analytical,
real ggml operator, sampler, parser, model-loading guard and driver controls. It downloads no
checkpoint and installs no original inference framework.

The optional `original-native` phase requires a worker deliberately assigned
the labels `self-hosted`, `linux`, `x64`, `latent-inpaint-gpu`. Repository
variables `MOEBIUS_REFERENCE_PYTHON` and `PIXELHACKER_REFERENCE_PYTHON` identify
the absolute interpreter paths of the separately provisioned original
environments. The driver verifies the pinned Diffusers/Torch/FLA versions,
Diffusers source files, original module imports and actual CUDA availability
before downloading full weights. It does not install or silently patch a
possibly incompatible framework environment. The runner must also provide
Python 3.9 or later for orchestration, CMake, a C++17 compiler, Git and `nm`.

The heavy lane checks resources before model work and requires at least 25GiB
free disk and 12GiB available host RAM. Those are admission floors, not measured
peak requirements or a guarantee that a particular CUDA device can run the
original. Actual memory/device details and software versions are retained.
If an original CUDA operator cannot execute, reference capture fails and no
substitute reference is used.

The worker downloads only the exact public source files and complete checkpoint
bytes frozen in the contract, runs the unaltered original first, converts the
verified weights, and executes native replay. A final real-CLI empty-mask
control checks image loading, model selection, GGUF loading, PNG output and
exact known-pixel preservation. This inexpensive CLI control does not count as
another diffusion parity result.

Choose `subtitle_bar` for one diagnostic with all first-step intermediates, or
`full-panel` for all four original fixtures. Each original fixture runs once
normally and once with hooks/noise replay; the native CPU path then executes
its own full forward. Runtime on this hosted workflow has not been measured,
so a complete panel may exceed the 165-minute driver deadline. The workflow reserves the remaining
15 minutes of its 180-minute job for evidence upload. Timeouts preserve failure
status, stage logs and any captured arrays; model files remain in worker scratch.

The orchestration driver intentionally requires a clean Git checkout at the
dispatched commit with the exact ggml submodule. Source-archive users can run
the CMake targets and reference/replay commands directly after establishing
their source provenance; an archive without Git metadata is not admitted as
an authenticated CI checkout.

The local CPU runs described above include an actual original Moebius capture,
native full forward, intermediate comparison, and component measurements.
They do not qualify the hosted CUDA/Metal paths or establish complete original
intermediate parity. The original PixelHacker CUDA computation has not run on
this CPU-only machine. Hosted execution results must be retained independently
from the local CPU evidence.
