# Native Moebius and PixelHacker denoisers

The source in `src/latent_inpaint_unet.cpp` implements both complete author
denoisers at the published 512-pixel image geometry. It consumes nine
`64 x 64` latent channels: four noisy-image channels, one white-fill mask
channel, and four encoded masked-image channels. It produces four predicted
noise channels. This is an experimental CPU/F32 path; implementation and
component checks do not establish full model parity or usable CPU latency.

## Source contracts

| Component | Pinned original source |
| --- | --- |
| Moebius UNet, lambda attention, depthwise residual blocks and MixFFN | [hustvl/Moebius at b88d462bacb9af6e7128a3b4cc4a07418bedfd61](https://github.com/hustvl/Moebius/tree/b88d462bacb9af6e7128a3b4cc4a07418bedfd61) |
| PixelHacker UNet and GLA configuration | [hustvl/PixelHacker at f5567db2871598aa178fe7a34c520dd478a0b41b](https://github.com/hustvl/PixelHacker/tree/f5567db2871598aa178fe7a34c520dd478a0b41b) |
| Gated linear attention equations and layer defaults | [flash-linear-attention v0.3.2](https://github.com/fla-org/flash-linear-attention/blob/v0.3.2/fla/layers/gla.py) |
| PixelHacker standard transformer and timestep behavior | [Diffusers v0.30.2](https://github.com/huggingface/diffusers/tree/v0.30.2/src/diffusers/models) |

The loader retains original `diff_model.*` and `embedding_layer.weight`
names. Conversion collapses only the singleton depth axis of Moebius'
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
the final composited RGB image. A full model pass, image quality, peak memory,
and latency remain pending until those original/native runs complete.

The entry point requires the corresponding value-parsed default-off gate,
`CRISPEMBED_MOEBIUS=1` or `CRISPEMBED_PIXELHACKER=1`. Only the declared
fixed-512 model configuration is implemented. Other checkpoints, dynamic
geometry, quantization and GPU acceleration require their own qualification.

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
its own full forward. Runtime has not been measured, so a complete panel may
exceed the 165-minute driver deadline. The workflow reserves the remaining
15 minutes of its 180-minute job for evidence upload. Timeouts preserve failure
status, stage logs and any captured arrays; model files remain in worker scratch.

The orchestration driver intentionally requires a clean Git checkout at the
dispatched commit with the exact ggml submodule. Source-archive users can run
the CMake targets and reference/replay commands directly after establishing
their source provenance; an archive without Git metadata is not admitted as
an authenticated CI checkout.

Current local evidence is limited to C++17 warning-enabled syntax checks and
small standard-library contract controls. No hosted build, original full-model
forward, native full-model parity, representative visual review or performance
result is asserted by this source change.
