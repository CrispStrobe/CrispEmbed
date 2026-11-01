# Big-LaMa CPU/F32 inpainting

> **Internal research branch only (`feat/issue71-lama-research`, owner decision 2026-11-01).**
> Big-LaMa checkpoint rights are unresolved (see `docs/inpainting-feasibility.md`): this branch is
> for internal research and testing, is never merged into a release branch, and no LaMa weights,
> GGUF, registry entry or publication may come from it. MI-GAN (MIT weights) lives on the main
> integration branch.


This is a complete experimental native graph and CLI integration for the
single-image LaMa portion of [issue 71](https://github.com/CrispStrobe/CrispEmbed/issues/71).
It adds the Fourier operations missing from the pinned ggml API, the full
Big-LaMa generator, strict local model conversion, whole-image preprocessing,
image composition and original/native comparison tools. It remains default-off
behind `CRISPEMBED_LAMA=1`.

**Status:** source implementation and model-free test preparation are complete.
The delivery report records the small checks actually executed. Native linking,
compiled operator tests, actual checkpoint conversion, original-model capture
and numerical parity have not been executed in this editing environment.
The repository reserves those workloads for hosted CI. No model qualification,
performance result, model alias or runtime release is implied by this document.

For build, model conversion, image/mask commands, text masks and batch options,
use the [inpainting CLI guide](inpainting-cli.md).

## Source and exact supported profile

The implementation follows the original
[LaMa source at 786f5936b27fb3dacd2b1ad799e4de968ea697e7](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7),
especially [`ffc.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py)
and the [Big-LaMa configuration](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/big-lama.yaml).
The [source receipt](lama-native-source-contract.json) pins the source hashes
and ggml revision. The upstream
[Apache-2.0 license](third-party/lama-LICENSE.txt), including Samsung Research's
copyright notice, is included. This C++/ggml port implements the selected
generator; it does not copy the training pipeline or distribute a checkpoint.

| Property | Supported value |
| --- | --- |
| GGUF architecture and profile | `lama`, `big-lama-18-075-v1`, schema 1 |
| Input/output | Four input channels; three output channels |
| Initial width and downsampling | 64 channels; three downsampling layers |
| Residual trunk | 18 blocks; 128 local and 384 global channels |
| Global ratio | 0.75 |
| Fourier units | 36, one in each residual convolution's global branch |
| Fourier normalization | `ortho` in both directions |
| Local Fourier units / gating / spatial transform | Disabled |
| Output activation | Sigmoid |
| BatchNorm evaluation | Frozen mean/variance, epsilon `1e-5` |
| Precision/backend | F32 weights and activations; CPU |
| State | 837 inference tensors, 204,228,108 F32 payload bytes |

The converter deliberately rejects alternate generators rather than interpreting
them as this profile. It verifies and omits 152 scalar BatchNorm training
counters. The native loader independently enforces the tensor inventory, exact
shapes and types, source/profile metadata, finite values and valid variances.
Tensor names retain the original generator's `model.*` names after removing a
checkpoint's optional `generator.` prefix. GGUF dimensions reverse the original
contiguous PyTorch dimensions; tensor bytes remain unchanged.

Checkpoint identity is supplied explicitly by the operator. Conversion reports
the checkpoint SHA-256, config SHA-256, each tensor SHA-256, output GGUF identity
and the converter identities. The report does not establish a checkpoint license.
No download alias or redistributed weights are added.

## Graph and Fourier implementation

`src/lama.cpp` builds the original encoder, split local/global residual trunk
and decoder. The local and global results are added in each residual block and
concatenated local-first before upsampling. All 18 residual blocks are present.

Each global spectral branch reduces 384 channels to 192, applies BatchNorm and
ReLU, runs its FourierUnit, adds the reduced spatial input, and projects back
to 384 channels. The FourierUnit operates as follows:

1. Transform both spatial axes using orthonormal real FFT.
2. Pack the complex values into adjacent real/imaginary channel planes:
   `[W/2+1, H, 2*C, N]` in ggml coordinates.
3. Apply the learned real 1×1 convolution, BatchNorm and ReLU.
4. Reinterpret the paired channels as complex values and apply orthonormal
   inverse real FFT using the explicit original width.

`src/lama_fft.cpp` supplies CPU `ggml_custom_4d` nodes. Power-of-two axes use
radix-2 FFT; other lengths use Bluestein convolution. Image dimensions need not
be powers of two. The internal FFT arithmetic uses double-precision complex
scratch and narrows results to F32; this is a numerical implementation choice,
and its difference from PyTorch complex64 must be measured by the reference
comparison. It is not assumed bit-exact.

The inverse reconstructs missing horizontal frequencies using conjugates at
the opposite vertical frequency. Learned spectra may no longer satisfy all
endpoint symmetries; the inverse takes the real projection, including the
DC/Nyquist behavior. Tests compare this with an independent half-spectrum
formula, rather than testing only round trips.

Callbacks do not throw across ggml's C boundary. They record the first failure
in a graph-owned atomic status and fill failed planes with NaNs. The caller
checks that status and output finiteness before returning image data. Concurrent
FFT scratch is bounded to 128 MiB by limiting the number of active workers.

Convolutions use the pinned CPU direct F32 kernel, which batches patch scratch
instead of materializing a full-frame 7×7 im2col tensor. The decoder's
`ConvTranspose2d(k=3, stride=2, padding=1, output_padding=1)` is represented by
the zero-padding transposed convolution followed by removal of the top row and
left column. Original IOHW kernel storage matches ggml's expected dimensions.

## Image contract, caching and limits

The public CLI thresholds mask grayscale at 128. Internal input is tightly
packed RGB8 and a binary mask with 255 meaning fill. RGB is normalized to
`[0,1]`, known pixels are retained in the first three channels, and the fill
mask is channel four. This differs from MI-GAN's channel order and signed RGB
normalization; the paths do not share those preprocessing routines.

Image and mask are symmetrically padded to the next multiple of eight on the
right and bottom. Symmetric padding includes the edge. The generator's inner
convolutions instead use reflection padding that excludes the edge. After
inference, generated RGB is clamped, scaled by 255 and truncated to uint8,
then cropped to the original dimensions. Every unmasked RGB byte comes directly
from the original decoded image. No neural output changes those pixels.

LaMa processes the entire padded frame. A crop or tile would change the global
Fourier context, so there is no hidden tile approximation. Supported dimensions
are 9–2048 per axis and at most 2,097,152 padded pixels. The minimum avoids
one-pixel bottlenecks where the original reflection operation is undefined.
The geometry admits 1920×1080; actual memory allocation remains a separate check.

The graph allocator is measured before backing allocation and capped at 3 GiB.
This is an activation-buffer limit, not a process memory limit. Weights, input
and output arrays, CPU kernel workspace and bounded FFT scratch are additional.
Oversized graphs fail with an actionable error before allocation. The
implementation does not claim that every admitted geometry fits every machine.

A context retains the model and its most recent padded-size graph. Same-size
batch frames reuse both. A size change frees the previous graph before creating
the next one. Empty masks skip inference. Calls are serial, output remains
unchanged on failure, and output may alias the input image. Directory and TSV
batch modes continue to process independent frames without temporal inference.

## Validation protocol

The ordinary [inpainting workflow](../.github/workflows/inpaint-cli.yml) builds
the CLI and model-free contracts on Linux, macOS and Windows. It includes
`test-lama-fft`, `test-lama` and the isolated CLI driver. The observer archive and
`test-lama-native` are separate validation targets, not production dependencies.

The manual [LaMa diagnostic workflow](../.github/workflows/lama-parity.yml)
requires explicit HTTPS checkpoint/config URLs and independently recorded
SHA-256 values. It builds and runs model-free contracts before model access.
The model workflow then follows this order:

1. Convert the local checkpoint through the restricted metadata reader.
2. Independently load the checkpoint using PyTorch's tensor loader and verify
   every converted F32 tensor byte against that load.
3. Import the unchanged, hash-verified original generator and capture all five
   authored fixtures: subtitle, edge, odd-size padded, empty and full masks.
4. Freeze the complete original report identity before native inference.
5. Run the native capture and compare 713 required stages per fixture, plus
   complete composed RGB and inference-control results.
6. Run the production CLI on those fixtures, including persistent batch,
   repeated frame and fresh-context comparisons.

The original importer bypasses unrelated package factory initializers. Two
unused training dependencies have fail-closed import adapters: spatial
rotation and Lightning seeding. Invoking either aborts. The original modules
that import them remain unchanged, their source hashes are checked, and the
report records zero adapter calls. Hooks copy data before in-place activations;
the reused decoder ReLU is labeled by invocation. An unobserved original
forward must equal the observed original forward exactly.

The native observer retains every recorded tensor's allocation owner, including
views, and must produce the same raw output as the ordinary build path. It also
checks byte-exact preprocessing, repeated cached-graph output, a fresh context,
raw versus image-wrapper composition, and exact known pixels. Intermediate
capture is separately limited to 65,536 padded pixels.

The comparison reports the earliest captured differing stage, output magnitude,
maximum and mean absolute error, RMSE, relative L2 error, cosine, and final RGB
differences. The fixed budgets are declared in the comparison code before any
model run:

| Quantity | Gate |
| --- | --- |
| Preprocessed generator input | Byte-exact |
| Each intermediate maximum absolute error | ≤0.002 |
| Each intermediate relative L2 error | ≤0.0002 |
| Each intermediate cosine | ≥0.99999 |
| Composed RGB maximum byte difference | ≤1 intensity level |
| Known RGB, repeat/fresh/observer controls | Exact |

These are diagnostic budgets for a newly declared protocol. They are not a
claim that the implementation already passes. A numerical failure is retained
with its first differing stage; a high cosine alone does not pass. The workflow
uploads reference/native arrays, small images, reports and notices, but neither
the checkpoint nor the GGUF model.

### Commands on a hosted build machine

After conversion and the CPU build described in the CLI guide:

```sh
cmake --build build-inpaint --config Release --parallel 2 \
  --target test-lama-fft test-lama test-lama-native
./build-inpaint/test-lama-fft
./build-inpaint/test-lama

python3 tools/dump_lama_reference.py \
  --source-dir /path/to/pinned-lama-source \
  --checkpoint /path/to/big-lama/models/best.ckpt \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --config /path/to/big-lama/config.yaml \
  --converted-model models/big-lama-f32.gguf \
  --output lama-original --threads 2

python3 tools/run_lama_native.py \
  --binary build-inpaint/test-lama-native \
  --model models/big-lama-f32.gguf \
  --original-dir lama-original --output-dir lama-native --threads 2

python3 tests/test_inpaint_model.py --binary build-inpaint/crispembed \
  --model models/big-lama-f32.gguf --backend lama \
  --reference lama-original --output-dir lama-cli
```

The reference step requires the pinned PyTorch, NumPy, Pillow and PyYAML versions
listed in the workflow. The converter, native runtime and comparison tools have
their own smaller dependency requirements. All output directories in these
commands must be new.

## Remaining qualification

Actual hosted execution is the next gate. Once numerical differences are
understood, a separately frozen, licensed set of real subtitle and scene images
is needed to judge inpainting quality, texture continuity, large masks and
runtime/memory behavior at useful resolutions. The small authored fixtures do
not measure perceptual quality or establish a quality ranking against MI-GAN.

F16, quantization, accelerator kernels, LFU variants, refinement and other LaMa
profiles are outside this implementation. STTN, ProPainter, flow estimation,
temporal propagation and video flicker qualification remain outside the user's
requested single-image scope. No placeholder flags advertise those paths.
