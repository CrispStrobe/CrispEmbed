# Inpainting engines and code reuse

Issue 71 has explicit source implementations for MI-GAN256,
MI-GAN512, Big-LaMa, Moebius and PixelHacker, with the earlier CPU/F32 path
and additional default-off [persistent CPU/CUDA/Metal graphs](inpainting-ggml-graphs.md).
The surrounding
[image/mask CLI](inpainting-cli.md) is shared: file and directory input, TSV
manifests, optional DBNet masks, mask dilation/region clipping, PNG output and
one persistent model context across a batch. White mask pixels are filled;
unmasked RGB8 bytes are preserved exactly.

The new engines are experimental and off by default. The source package
includes converters, original-reference capture tools, native comparison
programs and hosted workflows. Actual conversions, CPU execution and live
CLI checks are now recorded in the
[optimization results](inpainting-optimization-results.md). That report
separates complete-image timing, original-model comparison, native agreement
and retained numerical failures. A passing CLI check does not by itself
qualify the underlying model or a different backend.

## Choose a profile

| `--inpaint-engine` | Required environment gate | Geometry | Model files |
| --- | --- | --- | --- |
| `migan256` (default) | `CRISPEMBED_MIGAN=1` | Native 256 tiles; stride 192 | Existing exact-pinned F32 MI-GAN256 GGUF |
| `migan512` | `CRISPEMBED_MIGAN=1` and `CRISPEMBED_MIGAN512=1` | Native 512 tiles; stride 384 | F32 MI-GAN512 GGUF plus reviewed checkpoint/GGUF digests |
| `lama` | `CRISPEMBED_LAMA=1` | Each dimension 9–1024; symmetric padding to a multiple of 8, then crop | Pinned Big-LaMa generator F32 GGUF plus its conversion digest |
| `moebius` | `CRISPEMBED_MOEBIUS=1` | Exactly 512×512 by default; explicit independent 512 tiling optional | Moebius F32 GGUF and shared SDXL VAE F32 GGUF; both digests |
| `pixelhacker` | `CRISPEMBED_PIXELHACKER=1` | Same as Moebius | PixelHacker F32 GGUF and the same VAE; both digests |

These are stored-weight profiles below 4 GB, not a promise of peak memory
below 4 GB. From the original tensor census, Moebius plus the widened F32 VAE
contains about 1.24 GB of tensor data; PixelHacker plus VAE contains about 3.78 GB.
GGUF metadata adds a small amount. Measured CPU latency and process memory
for the executed geometries are in the optimization report; latent runtime
memory can exceed 4 GB. The explicit mixed GPU graph profile converts selected
weights at load; the source GGUF contract remains F32. The converter and native loader reject unrecognized
checkpoints, shapes, precision or metadata instead of inferring compatibility
from a filename. No model weights are included in the source package.

## MI-GAN512

The existing MI-GAN graph, checkpoint naming and equalized convolution
semantics are reused, with the original 512 encoder/synthesis level and its
channel schedule. The 256 profile retains its existing immutable model and
reference identities. See [the512 source/reference contract](migan512-native.md).

Obtain the official README-linked `migan_512_places2.pt`, record and review its
complete SHA-256, and capture the original reference before conversion. The
author does not publish a digest in the inspected download listing; a caller
supplied hash is therefore explicit provenance, not an invented author pin.

```sh
python tools/dump_migan512_reference.py \
  --source-dir upstream/MI-GAN --checkpoint checkpoints/migan_512_places2.pt \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" --output reference/migan512
python models/convert-migan-to-gguf.py --image-size 512 \
  --source-dir upstream/MI-GAN --checkpoint checkpoints/migan_512_places2.pt \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --original-report reference/migan512/report.json \
  --output models/migan512.gguf
CRISPEMBED_MIGAN=1 CRISPEMBED_MIGAN512=1 ./build-inpaint/crispembed \
  --inpaint-engine migan512 --inpaint-model models/migan512.gguf \
  --inpaint-model-sha256 "$GGUF_SHA256" \
  --inpaint-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --inpaint frame.png --mask mask.png --inpaint-output cleaned.png
```

Use the digest from the actual conversion receipt for `GGUF_SHA256`. An
original-reference report is model evidence; arbitrary edited JSON is not a
substitute for running the pinned original model.

## Big-LaMa

The native graph implements the original 18 Fast Fourier Convolution residual
blocks, including their local/global paths, evaluation BatchNorm, reflected
convolution padding and final transposed convolutions. `src/lama_fft.cpp`
provides orthonormal real 2D FFT/inverse FFT through radix 2/Bluestein transforms.
This supplies the missing Fourier operation without changing the vendored ggml
revision or introducing PyTorch as a runtime dependency. A tiny independent
DFT oracle is included in the model-free native test.

The model path accepts the pinned Big-LaMa checkpoint through
`models/convert-lama-to-gguf.py`. The exact source/checkpoint census and licence
provenance are retained with that converter. The
[conversion and licence guide](lama-conversion.md) and
[native reference guide](lama-native.md) describe the complete procedure.
Its input is the official README-linked ZIP, pinned to 381,428,720 bytes and SHA-256
`f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`.
The converter records the extracted checkpoint's hash separately; it does not
mislabel the archive digest as a published digest of the inner checkpoint.
The inference generator has 837 exported tensors and 204,228,108 bytes of F32
payload (about 204.2 MB), excluding the small GGUF header/alignment overhead.

```sh
python models/convert-lama-to-gguf.py \
  --checkpoint-archive checkpoints/big-lama.zip --source-dir upstream/lama \
  --output models/big-lama.gguf
```

After conversion, use the GGUF digest from its receipt:

```sh
CRISPEMBED_LAMA=1 ./build-inpaint/crispembed \
  --inpaint-engine lama --inpaint-model models/big-lama.gguf \
  --inpaint-model-sha256 "$LAMA_GGUF_SHA256" \
  --inpaint frame.png --mask mask.png --inpaint-output cleaned.png
```

The explicit 9–1024 limit bounds the first CPU implementation and preserves
the original bottleneck reflection semantics. This profile does not silently
resize or tile larger input. A larger supported geometry needs its own memory
and original-reference evidence.

## Moebius and PixelHacker

Both models reuse one native SDXL AutoencoderKL encoder/decoder and DDIM
sampler. No text encoder is required: the original checkpoints carry learned
conditional/unconditional embeddings. The denoisers implement their original
lambda-attention or gated-linear-attention graphs. The
[reuse audit](latent-inpaint-reuse.md) lists the existing CrispEmbed components,
new math, upstream licences and exact weight footprint. The
[conversion guide](latent-inpaint-conversion.md) contains frozen revisions and
download provenance.

Prepare exact source checkouts and checkpoints on a build/validation host,
then convert each requested component. The shared VAE uses the pinned
PixelHacker source checkout for its conversion contract:

```sh
python models/convert-latent-inpaint-to-gguf.py --model moebius \
  --source-dir upstream/Moebius --checkpoint checkpoints/moebius.pth \
  --output models/moebius.gguf
python models/convert-latent-inpaint-to-gguf.py --model pixelhacker \
  --source-dir upstream/PixelHacker --checkpoint checkpoints/pixelhacker.pth \
  --output models/pixelhacker.gguf
python models/convert-latent-inpaint-to-gguf.py --model sdxl_vae \
  --source-dir upstream/PixelHacker --checkpoint checkpoints/sdxl-vae.pth \
  --output models/sdxl-vae.gguf
```

Each conversion creates an adjacent `.gguf.json` receipt and `.gguf.licenses`
directory. Supply the complete GGUF file digests from those receipts:

```sh
CRISPEMBED_MOEBIUS=1 ./build-inpaint/crispembed \
  --inpaint-engine moebius --inpaint-model models/moebius.gguf \
  --inpaint-model-sha256 "$MOEBIUS_GGUF_SHA256" \
  --inpaint-vae models/sdxl-vae.gguf --inpaint-vae-sha256 "$VAE_GGUF_SHA256" \
  --inpaint frame-512.png --mask mask-512.png --inpaint-output cleaned.png \
  --inpaint-seed 0 --threads 2
```

For PixelHacker, set `CRISPEMBED_PIXELHACKER=1`, select
`--inpaint-engine pixelhacker` and supply its model/digest. The CLI defaults
are 20 requested DDIM steps, guidance 2.0 for Moebius or 4.5 for PixelHacker, and
noise offset 0.0357. Original strength slicing executes 19 timesteps for this
20-step setting. Supported options are `--inpaint-steps 2..100`,
`--inpaint-guidance 0..30`, `--inpaint-noise-offset 0..1` and an unsigned 64-bit
`--inpaint-seed`. All floating-point options must be finite.

The native generator specifies its own deterministic noise stream. A matching
numeric seed does not imply the same random numbers as PyTorch. The original
capture records both VAE posterior noises, diffusion noise and channel-offset
noise; the native parity program replays those exact tensors.

Use `--inpaint-tiles` explicitly for other dimensions. It reuses the same 512
overlap tiler as MI-GAN512, but each diffusion tile is independent. It changes
image-level context and can produce seams; it has no full-frame parity claim.
The strict 512 path remains the reference geometry. Internal C++ callers set
all model-specific sampling options explicitly, including PixelHacker's
guidance 4.5; `Options::guidance` itself defaults to 2.0.

## Validation and scope

The ordinary inpainting workflow builds the real CLI, exercises model-free
operator and parser controls on three desktop platforms, and retains the
existing optional MI-GAN256 smoke. Model-specific manual workflows prepare
original captures and native comparisons with fixed numerical budgets.
The latent reference uses the pinned original pipelines, checks that capture
hooks do not change output, and compares scheduler, VAE, denoiser and final
RGB stages. PixelHacker's original Flash Linear Attention requires a compatible
CUDA reference worker. Captured original outputs can then be replayed by the
native CPU, CUDA or Metal graph implementation without that framework runtime.

Use the [recorded CPU results](inpainting-optimization-results.md) for the
tested model identities and configurations, and run the same qualification
on the intended GPU before relying on that backend. Source and bounded
metadata checks alone establish neither model parity nor realistic removal quality.
Video propagation, optical flow, temporal consistency, STTN/ProPainter,
quantized GGUF profiles and shared C ABI/server/language bindings are outside
this image CLI addition. Native GPU graphs and runtime mixed precision are
included as separately gated source paths; their qualification remains pending.
Frame batches share
loaded weights and process each frame independently.
