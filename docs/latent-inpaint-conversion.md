# Moebius and PixelHacker: conversion and original reference

This lane provides a strict source contract, a framework-free streaming GGUF
converter, and an original-module reference capture. They are preparation for
the default-off native CPU/F32 implementations. Source checks and conversion
checks do not establish native numerical parity or representative image quality.

## Reused CrispEmbed pieces

- `tools/prepare_migan_inpaint.py` establishes the inert PyTorch ZIP metadata
  reader pattern: allow-listed storage records and tensor constructors, with no
  imported class or executable checkpoint callable. The latent converter extends
  that pattern for FP16 VAE storage, int64 BatchNorm training counters, and the
  one named rank-five convolution layout.
- `tools/migan_reference_contract.py` establishes the minimal F32 GGUF encoding,
  reversed GGUF dimensions, SHA-256 identity, and repository-owned fixture
  patterns. The larger models use streaming tensor payloads and a streaming
  readback to avoid holding several GB of weights twice in Python memory.
- `tools/dump_migan_reference.py` establishes original-module hooks, independent
  intermediate arrays, exact known-pixel compositing and hook-neutrality checks.
  Diffusion additionally captures and replays every random input.

No generator, VAE or sampler reimplementation is used as the original reference.
The native graph remains independent from the original framework execution.

## Exact original sources and checkpoints

The machine-readable authority is
[`latent-inpaint-source-contract.json`](latent-inpaint-source-contract.json).
It includes the original file hashes, exact checkpoint metadata census,
Diffusers source hashes, author model cards, transformations, and acceptance
budgets frozen before any model forward.

| Component | Source / weight revision | Original tensors → exported tensors | Published checkpoint bytes |
|---|---|---:|---:|
| Moebius Places2 | Code `b88d462bacb9af6e7128a3b4cc4a07418bedfd61`; weights `cd01f47fb648219d3fa605806c5ce00b713faa5e` | 1,203 → 1,079 | 905,298,356 |
| PixelHacker Places2 | Code `f5567db2871598aa178fe7a34c520dd478a0b41b`; weights `012fd343158936a265b8a0ee38a791a7a2841f45` | 753 → 753 | 3,449,345,440 |
| Shared SDXL VAE | Author PixelHacker weights revision `012fd343158936a265b8a0ee38a791a7a2841f45` | 248 → 248 | 167,394,306 |

Original checkpoint SHA-256 values:

```text
Moebius:     6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a
PixelHacker: ef5e28199cd03344ac22a6e6ff456ba8b868e83fb90d8174cc2ad83d7ef4296c
SDXL VAE:    a59d7ea697f2942d22002dc3469e8c53db807a6b78f7f5ec03bd4c1f70f98efe
```

The original state census was read from bounded HTTP Range metadata requests;
no complete checkpoint or full-model forward was needed to derive its keys,
shapes, dtypes and strides. The converter nevertheless verifies the **entire**
checkpoint's frozen byte count and SHA-256 before interpreting that metadata.

Only one denoiser plus the shared VAE is needed for an inference profile. The
other domain-specific checkpoints are alternative models and are not accepted
by these fixed profiles. The F32 converter widens the published FP16 VAE exactly;
the resulting weights occupy approximately **1.24 GB for Moebius + VAE**, or
**3.78 GB for PixelHacker + VAE**, plus small GGUF headers/alignment. These are
weight sizes, not peak runtime RAM measurements.

### Explicit transformations

- All floating tensors become F32, preserving every original F32 byte and every
  FP16 value exactly.
- Moebius's 124 scalar `*.num_batches_tracked` training counters are checked and
  individually hashed in the receipt, then omitted. BatchNorm inference uses
  the retained `running_mean` and `running_var`.
- Exactly 15 Moebius `*.attn1.pos_conv.weight` tensors have original shape
  `[K,1,1,15,15]`. Their singleton depth axis is removed to store `[K,1,15,15]`
  in GGUF's four-dimensional tensor format. The bytes and effective 2D
  convolution are unchanged.
- Other original names and contiguous tensor order are preserved. Extra,
  missing, differently shaped, nonfinite, differently typed or aliased state
  tensors are rejected.

## Hosted conversion

Keep full builds, model downloads, conversion and inference on a hosted worker,
as required by the repository's contribution policy. The converter never
downloads a model automatically. Acquire the exact public checkpoint URLs in
the contract and check out the exact original revisions first.

```sh
python models/convert-latent-inpaint-to-gguf.py \
  --model moebius \
  --source-dir original/Moebius \
  --checkpoint checkpoints/moebius-places2.bin \
  --output artifacts/moebius-places2-f32.gguf

python models/convert-latent-inpaint-to-gguf.py \
  --model pixelhacker \
  --source-dir original/PixelHacker \
  --checkpoint checkpoints/pixelhacker-places2.bin \
  --output artifacts/pixelhacker-places2-f32.gguf

python models/convert-latent-inpaint-to-gguf.py \
  --model sdxl_vae \
  --source-dir original/PixelHacker \
  --checkpoint checkpoints/sdxl-vae.bin \
  --output artifacts/sdxl-vae-f32.gguf
```

The output parent must exist. Existing output files, receipts or licence
directories are rejected. Each conversion writes:

- The completed GGUF.
- `<output>.json`, including the converted file SHA-256 and byte count, exact
  per-tensor readback hashes, original checkpoint provenance, omitted-counter
  hashes and explicit `native_qualified: false`.
- `<output>.licenses/`, retaining the original source licence, author weight
  model card and source/provenance notice.

Use the **actual converted GGUF SHA-256 from this receipt** when opening native
models. An upstream checkpoint digest alone does not authenticate a converted
GGUF's payload. Do not invent or publish a converted-model digest before the
conversion has run.

## Independent original capture

Use a separate process and environment for each original. The source contract
requires Moebius's Diffusers **0.38.0** / Torch **2.7.1**, or PixelHacker's
Diffusers **0.30.2** / Torch **2.3.0**, and Flash Linear Attention **0.3.2**.
Other actual dependency versions are recorded. Eight inference-critical
Diffusers files per version, including VAE, normalization, attention and DDIM,
are compared byte-for-byte with the pinned official source before execution.
PixelHacker's original Flash Linear Attention path needs a compatible CUDA
worker; the capture does not silently replace that operator with a different
reference implementation.

```sh
python tools/dump_latent_inpaint_reference.py \
  --model moebius \
  --source-dir original/Moebius \
  --checkpoint checkpoints/moebius-places2.bin \
  --vae-checkpoint checkpoints/sdxl-vae.bin \
  --output artifacts/moebius-original \
  --device cuda:0 --threads 2
```

Run the equivalent command for `pixelhacker` in its own environment. The default
four owned fixtures are `subtitle_bar`, `edge_touch`, `center` and `empty`.
`--cases subtitle_bar` selects a bounded first diagnostic capture. Each selected
case executes exactly one ordinary original pipeline and one observed replay.
The aggregate report says which cases actually ran; a subset is not a complete
four-case qualification.

The ordinary execution supplies four captured random tensors: the full-image
VAE posterior draw, masked-image posterior draw, diffusion noise, and channel
noise offset. The observed execution replays those identical tensors at the
original library's random-function boundaries. The final float image must be
bit-identical between the ordinary and observed original executions. Native
parity uses these explicit inputs, so matching a framework RNG seed is not
mistaken for matching its generated samples.

Each case's `reference.json` binds raw little-endian F32/CHW files by shape,
byte count and SHA-256. It includes normalized input/mask, all random inputs,
the exact 1,000-element cumulative alpha schedule, both VAE moment tensors,
scaled latents, initial noisy latents, both CFG arms' outputs, guided noise and
the result of every scheduler step, final decoder output and exact-composited
RGB8. The first denoising iteration of `subtitle_bar` also captures actual
original UNet module boundaries to locate the earliest difference.

The original settings are 20 requested DDIM steps, strengths 0.99/0.999, and
noise offset 0.0357. Both strengths produce **19 executed timesteps**, starting
at 900 and ending at 0, with leading spacing. Guidance is 2.0 for the declared
Moebius profile and 4.5 for PixelHacker. VAE posterior sampling and scaling
factor 0.13025 are part of the contract. The original's optional Gaussian
paste/compensation is disabled; CrispEmbed's exact binary-mask composition
preserves known pixels byte-for-byte.

## Acceptance and current evidence

Every declared floating stage must meet all three initial F32 gates:
maximum absolute error ≤0.002, relative L2 ≤0.003, and cosine ≥0.99999.
Generated/composited RGB must meet maximum error ≤2 and mean absolute error
≤0.1; known pixels must be exact. Preserve the first failing stage, original
magnitudes and final images. Do not change the budget after observing a failure.

The small standard-library suite covers unsafe pickle globals, drifted source
state, dtype/stride/shape mismatches, hash-before-parse, FP16 widening, permitted
singleton transforms, omitted counters, ZIP member bounds, malformed storage,
GGUF streaming readback, and overwrite protection:

```sh
python -m unittest discover -s tests -p test_latent_inpaint_contract.py -v
```

The initial source preparation passed these 16 tests. Complete checkpoint
conversion, original-model capture, native build/link/model execution and
representative quality remain separate hosted acceptance gates. No runtime
speed or peak-memory result follows from the source preparation.

## Weight licence records

[Moebius's official README](https://github.com/hustvl/Moebius/tree/b88d462bacb9af6e7128a3b4cc4a07418bedfd61)
expressly grants Apache-2.0 for code and pretrained model weights and permits
commercial use. Its author-owned Hugging Face card currently labels the weights
MIT. Both evidence records are preserved; this is not represented as an
explicit dual-licence election.

[PixelHacker's author model card](https://huggingface.co/hustvl/PixelHacker/blob/012fd343158936a265b8a0ee38a791a7a2841f45/README.md)
declares MIT for its weights. Its source implementation is separately
Apache-2.0. The shared VAE is the pretrained
[standalone SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae), also published
under MIT. The reference paper's own publication licence is not used as a
model-weight licence.

## Hosted evidence and corrections (1 November 2026)

- **GGUF tensor names.** 540 Moebius and 384 PixelHacker state names exceed the
  63-byte GGUF name limit, so the first converted denoisers could not be opened
  by ggml. The converter now stores names of 64+ bytes under the abbreviation
  rule in `src/inpaint_tensor_names.h` (twin `gguf_tensor_name`); the receipt
  keeps original names, and every loader maps through the rule. The layout
  metadata string records it.
- **Original CUDA capture** runs on a Kaggle GPU worker
  (`tools/kaggle/latent-inpaint-capture`, stage.py then kpush.py): pinned
  sources/checkpoints are verified, a separate uv interpreter is provisioned,
  and the unchanged dumper captures the four-case panel. The output is written
  as verified gzip parts that `tools/ci-heavy/latent_native.sh` downloads for
  the hosted CPU replay. Moebius ran with the exact contract pins (T4, 88 s).
  PixelHacker's pins cannot be installed together: FLA ops live in `fla-core`,
  torch 2.3.0 with triton 3.2.0 fails at FLA import, and transformers 4.40.0
  lacks `utils.deprecation`. Its capture uses torch 2.6.0 as an explicit
  `--allow-dependency` deviation recorded in the report (the native comparator
  accepts only that recorded form) and transformers 4.48.3 (T4, 288 s).
- **Precision variants** are quantized from these F32 files; see
  [the precision guide](inpainting-precision.md).
