# Experimental native Big-LaMa

Big-LaMa now has a separate native CPU path behind `CRISPEMBED_LAMA=1`. It runs the original generator architecture using F32 weights and activations, including all 18 residual FFC blocks and their 36 Fourier units. The existing MI-GAN profiles retain their own gates and contracts.

**Status:** implementation, safe conversion, model-free controls and an original-reference replay workflow are present. This change does not include an executed original/native numerical comparison or a quality benchmark. The gate remains off by default. A successful converter receipt does not qualify native inference.

## Architecture and reuse

The implementation follows the [pinned original FFC generator](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py) and [Big-LaMa configuration](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/big-lama.yaml).

| Component | Native implementation |
| --- | --- |
| Initial and downsampling convolutions | Shared `latent_inpaint::Engine`, with explicit reflection padding |
| Bottleneck | 128 local and 384 global channels; 18 residual blocks, two FFC layers each |
| Global-to-global branch | Original 1×1 projection, inference BatchNorm/ReLU, FourierUnit, residual spectral sum and output projection |
| FourierUnit | Real 2D FFT, interleaved real/imaginary channels, original learned 1×1 convolution and BatchNorm/ReLU, inverse real FFT |
| FFT sizes | Radix-2 transforms plus Bluestein convolution for other lengths; no quadratic production fallback |
| FFT normalization | `ortho`, with the original real-spectrum channel ordering and DC/Nyquist treatment |
| Local Fourier unit | Disabled, as specified by the pinned original configuration |
| Spatial rescaling and spectral position encoding | Absent in this fixed original profile |
| Upsampling | Three original 3×3 transposed convolutions, stride 2, padding 1, output padding 1; shared GGML F32 operator |
| Output | Original 7×7 convolution and sigmoid, followed by cropped RGB8 composition |
| Weight loading | Existing GGUF loader and owned mapping; no duplicate complete float weight cache |

The Fourier implementation uses double precision for its bounded complex working buffers and converts each operator boundary back to F32. This reduces transform accumulation error; it does not assert bit equality with a particular Torch FFT backend. The original-reference gate compares the Fourier inputs, learned spectra, inverse transforms, block outputs and final pixels separately.

The frozen census contains **837 exported generator tensors**, totaling **204,228,108 F32 payload bytes**. Conversion retains the original `generator.model.*` names. It omits 152 inference-inert BatchNorm counters and the checkpoint's evaluation-only networks, recording those omissions in the receipt. The complete original generator state, including counters, is restored when capturing the original reference.

## Image and mask contract

The native API accepts interleaved RGB8 and a same-size mask containing only 0 and 255. White means fill. Known RGB bytes are copied exactly from the input; generated values cannot modify them.

Supported dimensions are **9 through 1024 pixels on each axis**, with at most 1024² pixels. The image and mask are padded on the right and bottom to a multiple of eight using the original dataset's **symmetric** convention, which repeats the edge. Convolution padding uses **reflect**, which excludes the edge. These are distinct operations. No input resizing or independent tiling is introduced in this profile.

The lower bound ensures the padded bottleneck has at least two pixels per axis, as required by the original reflected convolutions. Output is cropped to the original dimensions. The original sigmoid output is clamped to `[0,1]`, multiplied by 255 and truncated to uint8. The final hard mask composition preserves all known bytes. Empty masks return the input without executing the generator.

Calls reject invalid dimensions, byte counts, nonbinary masks and nonfinite tensors. Rejections leave output buffers unchanged. Input/output aliasing is supported. A single context rejects concurrent inference calls; use separate contexts for independent requests.

## Checkpoint identity and licensing evidence

The accepted input is the archive linked by the [original repository README](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/README.md), pinned to its immutable host revision:

- Repository: `smartywu/big-lama`.
- Revision: `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`.
- Archive: `big-lama.zip`, 381,428,720 bytes.
- Archive SHA-256: `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`.
- Checkpoint member: `big-lama/models/best.ckpt`, 410,046,389 bytes.

The official archive identity is the independent pin. Its inner checkpoint hash is computed during validated extraction and recorded separately; it is not presented as an independently published hash. GGUF metadata explicitly records `official-zip`, the archive revision and the checkpoint member so these identities cannot be confused.

The original code carries [Apache-2.0](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/LICENSE). A collaborator's statement in [original issue 230](https://github.com/advimman/lama/issues/230) points to that licence for the pretrained weights. The supplied archive does not contain a dedicated weight-licence text. The conversion receipt and native metadata preserve this distinction using `Apache-2.0-maintainer-comment; dedicated-weight-licence-text-absent`; they do not invent a separate licence grant.

## Convert and use

Run model downloads, conversion and inference on an appropriate hosted worker. The converter uses a bounded inert checkpoint parser and streamed GGUF writer; it does not import Torch or execute checkpoint classes.

```sh
python models/convert-lama-to-gguf.py \
  --checkpoint-archive big-lama.zip \
  --source-dir lama-original \
  --output big-lama-f32.gguf
```

The source checkout must match the pinned files in `docs/lama-source-contract.json`. The converter produces the GGUF, an adjacent `.gguf.json` receipt and licence notices. Existing destinations are rejected.

Use the `gguf_sha256` value from that receipt:

```sh
CRISPEMBED_LAMA=1 crispembed \
  --inpaint-engine lama \
  --inpaint-model big-lama-f32.gguf \
  --inpaint-model-sha256 RECEIPT_GGUF_SHA256 \
  --inpaint input.png \
  --mask fill-mask.png \
  --inpaint-output repaired.png \
  --threads 2
```

The shared CLI also provides directory/list processing and text-mask preparation. See `docs/inpainting-cli.md`. The native loader verifies the full model-file digest, all semantic metadata fields and types, the exact tensor inventory and all F32 tensor shapes before inference. It rejects LFU, alternate checkpoints, altered source profiles and modified input semantics.

## Original-first numerical acceptance

`.github/workflows/lama-inpaint.yml` is manual and defaults to `build-only`. This builds the real CLI and native targets, runs model-free controls, and checks that production archives do not export test observer entry points. It downloads no model.

Choose `original-native` explicitly to run the pinned CPU reference. The workflow installs the declared Torch and supporting dependencies, downloads and validates the official archive, executes the original generator, and only then converts the model for native replay. The original generator and its numerical imports execute unchanged from verified source. Empty parent package shells avoid unrelated training factory imports; they do not replace numerical modules.

Each selected fixture records **227 required stages**. Capture hooks must leave the original output bit-identical. The native comparator likewise checks observer neutrality, complete stage coverage, exact conditioned-input values, the public image wrapper and runtime gate rejection. Reference files, RGB inputs, masks and expected composites are hash checked before native execution. RGB and mask hashes must also match the frozen owned fixtures, independently of any hashes declared by the reference bundle.

Eight deterministic fixtures cover a subtitle bar, an edge-touching hole, a center hole, an empty mask, a full mask, a one-pixel mask, a wide image and a thin strip. Several dimensions create non-power-of-two Fourier sizes after padding/downsampling. These fixtures test implementation behavior; they are not a representative image-quality benchmark.

The acceptance limits were frozen before any native checkpoint run:

| Check | Required result |
| --- | --- |
| Intermediate F32 maximum absolute difference | ≤ 0.002 |
| Intermediate relative L2 error | ≤ 0.0003 |
| Intermediate cosine similarity | ≥ 0.99999 |
| Final float prediction maximum absolute difference | ≤ 0.0002 |
| Masked RGB8 maximum absolute difference | ≤ 1 |
| Masked RGB8 mean absolute difference | ≤ 0.1 |
| Masked RGB8 PSNR | ≥ 50 dB |
| Known RGB errors | 0 |
| Original and native observer neutrality | Bit identical within each implementation |

These limits cannot be relaxed through an environment variable or reference metadata. Failures retain the first failing stage, all available logs and numerical evidence. The workflow produces raw stages, native/reference images and a difference image amplified 16×. It reports the selected-case result without automatically promoting the experimental profile.

For a provisioned hosted worker, the same orchestration can be invoked directly:

```sh
python tools/ci-heavy/lama_inpaint.py \
  --phase original-native --cases full-panel \
  --output NEW_EVIDENCE_DIRECTORY --scratch NEW_SCRATCH_DIRECTORY \
  --deadline-minutes 165
```

`test-lama-ops` independently checks the production FFT against a tiny direct DFT, odd/even inverse transforms, spectral real projection, both padding conventions, composition and loader guards. `test-lama-diff --contracts` validates replay coverage and negative controls without model weights. Neither model-free test establishes full generator parity.
