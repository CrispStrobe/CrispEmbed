# Experimental subtitle and image inpainting

This implements the image-and-mask CLI requested in
[issue 71](https://github.com/CrispStrobe/CrispEmbed/issues/71), with persistent
batch processing and optional DBNet text masks. The default engine is the
existing MI-GAN fixed256 CPU/F32 graph. Additional explicit profiles are
MI-GAN512, Big-LaMa, Moebius and PixelHacker. See the
[engine selection and preparation guide](inpainting-engines.md) for their
source contracts, model conversion and command examples. Every engine requires
its own opt-in gate; unset, empty and `0` keep it disabled.

The graph's earlier [original reference](../tests/results/migan/original-reference.json)
and [native comparison](../tests/results/migan/native-parity.json) are unchanged.
Those receipts cover the original fixed256 experiment. The new tiling, text-mask
and CLI paths require the new hosted checks described below; they do not inherit
a realistic subtitle-quality or temporal-consistency qualification from that
experiment. This is a source feature, not a new runtime release or model alias.

## Build

Use a source checkout with its pinned `ggml` submodule populated. A source
archive that includes `ggml/` already contains that dependency. On a build host:

```sh
cmake -S . -B build-inpaint -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF
cmake --build build-inpaint --config Release --target crispembed-cli --parallel 2
```

The executable is `build-inpaint/crispembed` with single-configuration
generators, or `build-inpaint/Release/crispembed.exe` with Visual Studio.
These inpainting paths use the CPU even when other CrispEmbed engines are
built with GPU backends. No Python runtime, PyTorch or CUDA is required to run
inpainting.

## Prepare the default MI-GAN256 model

Each engine accepts an explicit source/checkpoint contract. The default
MI-GAN256 preparation helper uses Python 3.10 or later and only its standard
library:

```sh
python3 tools/prepare_migan_inpaint.py --download \
  --output-dir models/migan-inpaint
```

The output directory must be new. The helper retrieves the official checkpoint
and the pinned upstream source/weight notices, checks their identities, extracts
the state through a restricted metadata reader, and reproduces the original
GGUF. It checks every one of the 154 tensors and the complete output digest.
It does not execute upstream model code or change the original converter and
reference protocol. The exact-checkpoint reproduction is an explicit optional
hosted validation step; authored metadata fixtures alone do not prove that the
live checkpoint has been successfully converted.

The resulting model is:

```text
models/migan-inpaint/migan-256-places2-f32-experimental.gguf
25,185,504 bytes
SHA-256 b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b
```

Keep `replication.json` and the `licenses/` directory beside it. The native
loader independently checks the model's byte count and full hash before loading
it, then uses the existing strict tensor/metadata/checkpoint checks.
The MI-GAN256 profile rejects F16, quantized and other-architecture files.
Select the other engines explicitly; their model receipts and gates differ.

For offline preparation, supply the official checkpoint and an upstream MI-GAN
source directory at commit `2b793c5ece43f4253e32d4afc257120a5deed6f5` containing
the matching `LICENSE` and `LICENSE-WEIGHTS`:

```sh
python3 tools/prepare_migan_inpaint.py \
  --checkpoint /path/to/migan_256_places2.pt \
  --source-dir /path/to/MI-GAN \
  --output-dir models/migan-inpaint
```

If the official download service returns a confirmation or error page, the
helper rejects it. It never accepts a different checkpoint or skips a checksum
to make a download succeed.

## One frame

```sh
export CRISPEMBED_MIGAN=1
INPAINT_MODEL=models/migan-inpaint/migan-256-places2-f32-experimental.gguf
./build-inpaint/crispembed --inpaint-model "$INPAINT_MODEL" \
  --inpaint frame.png --mask mask.png > cleaned.png
```

The mask must have the same width and height as the image. Its grayscale values
are thresholded at 128: white fills, black preserves. Antialiased masks therefore
have an explicit binary interpretation. The output is a same-size RGB PNG, and
every RGB byte outside the final mask is copied from the decoded input. Alpha
channels are discarded. An empty mask produces unchanged RGB pixels.

For an explicit output file, including in Windows PowerShell:

```powershell
$env:CRISPEMBED_MIGAN = "1"
.\build-inpaint\Release\crispembed.exe `
  --inpaint-model models\migan-inpaint\migan-256-places2-f32-experimental.gguf `
  --inpaint frame.png --mask mask.png --inpaint-output cleaned.png
```

The common PNG writer includes CrispEmbed's existing provenance metadata and
optional configured signing. Diagnostic and progress messages go to stderr.
The inpainting command rejects the legacy PPM output override, so stdout cannot
silently contain a different format. `--inpaint-help` needs no model or gate.

## Text masks and subtitle regions

Use a DBNet GGUF to detect text quadrilaterals, rasterize them, and expand the
mask. A detector remains loaded across a batch, just like the inpainting model.

```sh
CRISPEMBED_MIGAN=1 ./build-inpaint/crispembed \
  --inpaint-model "$INPAINT_MODEL" --inpaint frame.png \
  --mask-from-text --mask-text-model dbnet.gguf \
  --mask-region 0,800,1920,280 --mask-dilate 3 \
  --inpaint-output cleaned.png
```

That example assumes a 1920×1080 frame. A region is `x,y,width,height` in original
image pixels and must fit wholly inside every processed frame. Dilation happens
first, then the final mask is clipped to the region. This prevents changes above
or outside the selected subtitle band. The region limits the resulting mask;
the detector still sees the complete image.

`--mask-dilate N` accepts 0–256 pixels. The default is 3 for text masks and 0 for
provided mask files. Dilation uses a square neighborhood. Text-mask mode requires
`general.architecture=dbnet`; PP-OCR and Surya model files are not silently
treated as DBNet. No text is a successful empty mask; failed detection and
nonfinite detector outputs are errors. Extreme aspect ratios that collapse a
resized detector dimension are rejected before graph creation.

## A frame sequence without repeated model loading

Directory mode matches image and mask filenames, processes PNG files in lexical
order, and creates the destination directory if necessary:

```sh
CRISPEMBED_MIGAN=1 ./build-inpaint/crispembed \
  --inpaint-model "$INPAINT_MODEL" \
  --inpaint-dir frames --mask-dir masks --output-dir cleaned-frames \
  --threads 2
```

The scan is nonrecursive and accepts `.png` extensions without case sensitivity.
Use zero-padded frame numbers such as `000001.png` to make lexical order match
video order. Replace `--mask-dir masks` with `--mask-from-text
--mask-text-model dbnet.gguf` to generate masks for each frame.

For explicit frame selection, supply a TSV manifest with three columns:

| Image | Mask | Output |
| --- | --- | --- |
| `frames/000001.png` | `masks/000001.png` | `cleaned/000001.png` |
| `frames/000002.png` | `masks/000002.png` | `cleaned/000002.png` |

The actual file uses literal tab separators, one frame per line:

```sh
CRISPEMBED_MIGAN=1 ./build-inpaint/crispembed \
  --inpaint-model "$INPAINT_MODEL" --inpaint-list frames.tsv --threads 2
```

`--inpaint-list -` reads the same format from stdin. File-manifest relative paths
resolve beside the manifest; stdin-relative paths resolve from the working
directory. Blank lines, comment-only `#` lines and CRLF are accepted. Parent
directories for listed outputs must already exist. In text-mask mode the mask
column must be `-`; other columns must be file paths. Batches emit no images to
stdout, so PNG streams are never concatenated accidentally.

Frames run serially with one loaded inpainting model. The command validates
paths, image headers, mask dimensions, region bounds and output conflicts for
the entire batch before model loading or output creation. It then decodes one
frame at a time. A later decoding or inference failure stops at that frame and
retains earlier completed outputs. The error names the frame number and path.
This reduces repeated model loading; it does not provide temporal propagation,
motion estimation or blending across frames.

## Image geometry and file behavior

For the default MI-GAN256 profile, images larger than 256 pixels use overlapping native 256×256 tiles with a
192-pixel stride. The last tile on each axis aligns to the image boundary;
small dimensions use edge replication. The model is never resized or converted
to a different resolution. Tiles always read the original frame, empty tiles
are skipped, and integer weighted overlap averages affect masked pixels only.
The fixed256 single-tile case preserves the existing composed RGB result.

Dimensions are limited to 32,768 per side and 64×1024×1024 pixels overall.
The overlap accumulator uses a 256-row band instead of full-frame float planes.
Large masks and high-resolution frames can still require many CPU model calls.
Threads are explicitly limited to 1–64, default 1. There is no latency or
real-time video claim.

Named output files must be new unless `--inpaint-overwrite` is supplied.
Inputs, masks, model files and manifests are protected even with that option.
Replacements are staged beside the destination and renamed only after encoding,
writing, flushing and closing succeed. Failed preparation or inference leaves
existing outputs intact. Shell redirection is controlled by the shell; prefer
`--inpaint-output` when CrispEmbed should enforce overwrite checks.

Exit status is 0 for success, 2 for option/preflight/gate errors, and 1 for model,
decode, detector, inference or write errors. Contexts are serial and must not be
used concurrently by internal callers.

## Validation and remaining work

The [inpainting workflow](../.github/workflows/inpaint-cli.yml) prepares Linux,
macOS and Windows CPU builds. It runs:

- The real tiling/composition and SHA-256 C++ contracts without a model.
- The existing model-free MI-GAN contracts, gate and observer-retention controls.
- The actual CLI parser, PNG writer, masks, preflight and batches with a separate,
  test-only deterministic model/detector driver. That driver is not installed.
- Production CLI help/routing and rejection of a model with the wrong hash.
- Standard-library model-replication, PNG and existing reference-driver tests.

Manual workflow dispatch can also enable `model_smoke`. On Linux this reproduces
the exact checkpoint and runs the production CLI on authored empty/subtitle/
edge/multiple-tile fixtures, checking known-pixel identity and single/batch/repeat
agreement. It retains small PNG/log/JSON evidence, never the model. The smoke
has no perceptual ground truth and is not a realistic subtitle-quality benchmark.

The source delivery's validation report distinguishes checks actually executed
from prepared hosted checks. No hosted result is asserted merely because this
workflow exists. Native compilation, actual model replication and runtime tests
must pass before treating the new interface as validated on a platform.

The additional [engine profiles](inpainting-engines.md) add their own
source-derived native implementations and independent reference harnesses.
Their full-model parity and hosted runtime evidence are still pending.

Real subtitle/scene perceptual and seam quality, temporal consistency,
other precisions, GPU inference, C ABI/server/language bindings,
model publication and downstream release qualification remain separate work.
The issue's requested video propagation and motion compensation are not exposed
as nonfunctional flags.
