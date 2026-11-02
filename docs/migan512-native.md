# Experimental MI-GAN-512 CPU/F32 support

This extension adds the original MI-GAN `Generator(resolution=512)` to the existing
native engine. It preserves the reviewed fixed256 profile and its independent
reference artifacts. The512 path is implemented but **native numerical parity,
real-image quality, runtime and peak memory remain unmeasured** until the hosted
acceptance panel runs. Source checks are not a model qualification.

## Reuse and model differences

| Existing component | How512 reuses it |
| --- | --- |
| `src/migan.cpp` | Same graph builder, separable depthwise/pointwise convolutions, original filter constants, noise buffers and strengths, skip features, leaky ReLU/gain/clamp and toRGB accumulation. |
| `core_gguf` | Same metadata reader and CPU weight loader; strict names, shapes, F32 precision, source commit and checkpoint digest validation. |
| `core_cpu_backend` | Same explicitly selected CPU backend and thread control. No implicit GPU fallback or alternate inference runtime. |
| `core_env` | Existing `CRISPEMBED_MIGAN` plus a separate default-off `CRISPEMBED_MIGAN512` gate, rechecked on inference. Literal `0` disables either gate. |
| `src/inpaint.cpp` | Same transactional image wrapper, binary white=fill masks, edge padding, known-pixel preservation and overlap composition;512 tiles use a384-pixel stride. |
| Original reference utilities | Same lossless F32 GGUF writer/reader and inventory methodology, with a separate512 contract, fixtures, capture and native observer comparison. |

The original512 model adds one encoder level and one synthesis level. Its first
encoder has64 channels; the256 level retains128 channels but no longer owns the
fromRGB layer. The full original state has **177 tensors**, including every512
noise and fixed interpolation buffer. The256 state still has154 tensors.

The production graph arithmetic is shared, with the selected resolution controlling
loop bounds and tensor sizes. The old APIs remain fixed256 wrappers. The new APIs
accept exactly256 or512, so a caller cannot infer a model profile from arbitrary
buffer lengths. This is fixed-resolution inference; overlap tiling is a separate
CrispEmbed wrapper, not a reproduction of the author's ONNX crop/resize pipeline.

## Source and weights

The blueprint is the unchanged standalone source at
[`2b793c5ece43f4253e32d4afc257120a5deed6f5`](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/lib/model_zoo/migan_inference.py).
Its SHA-256 is
`1e35a181e516742ae32af935848e6a9a4382f528ea7096db6658f14cd8b9ace0`.
The independent capture executes that module at512; it does not use the native
implementation to generate expected outputs.

The author's [README](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/README.md)
links the original `migan_512_places2.pt` and explicitly applies
[`LICENSE-WEIGHTS`](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/LICENSE-WEIGHTS)
(MIT) to pretrained models.

Public original-folder metadata identifies the checkpoint as
[`1D_YCuCgo20S2256sqpedsmENNm2WMtVY`](https://drive.google.com/file/d/1D_YCuCgo20S2256sqpedsmENNm2WMtVY/view),
**29,553,256 bytes**. No author-issued SHA-256 was present in that metadata.
The new contract leaves its upstream and first-observed digests null instead of
inventing a pin. A hosted original capture must record the actual checkpoint
hash, then conversion records the complete GGUF hash. These hashes identify
artifacts; they do not by themselves establish native parity.

The runtime accepts a512 model only when the caller supplies both independently
reviewed hashes. It hashes the entire GGUF before loading, rejects files larger
than64 MiB, then verifies the source/precision/resolution/tensor schema. There is
no download alias and no self-trust of a digest stored inside the supplied file.
The256 wrapper retains its exact previously reviewed model bytes and digest.

## Internal API

```cpp
// Includes inpaint.h; both gates must be enabled for512.
inpaint::context * ctx = inpaint::init_variant(
    model_path, reviewed_checkpoint_sha256, reviewed_gguf_sha256,
    512, n_threads, error);
// The existing process_rgb8/free functions then work with this context.
```

`migan::init_variant` exposes the strict native model, while
`migan::preprocess_rgb8_variant` and `migan::composite_rgb8_variant` expose its
fixed-size RGB boundary. The original helpers always remain256.

`inpaint::detail::process_tiles_rgb8_variant` is pure image plumbing: it takes a
resolution, a tile callback and the same RGB/mask/output arguments as the existing
tiler. A512 tile callback can therefore be shared by other experimental model
families. Empty masks make no callback calls. Invalid masks and late callback
failures leave the caller's output untouched. Known RGB pixels remain byte-exact.

## Hosted reference and conversion

Run full model capture, builds and inference on hosted CI in accordance with the
repository's execution policy. The existing original-source directory must be
checked out at the pinned revision; the capture verifies all files listed in
`docs/migan512-reference-contract.json` before loading the checkpoint.

After obtaining and independently recording the original checkpoint SHA-256:

```sh
python tools/dump_migan512_reference.py \
  --source-dir original-migan \
  --checkpoint migan_512_places2.pt \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --output original512 --threads 2

python models/convert-migan-to-gguf.py \
  --image-size 512 \
  --source-dir original-migan \
  --checkpoint migan_512_places2.pt \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --original-report original512/report.json \
  --output migan-512-places2-f32-experimental.gguf
```

The original capture contains eight authored fixtures: subtitle bar, edge-touching
mask, empty/full masks, center region, one pixel, checkerboard and thin strip.
The subtitle and edge cases include intermediate convolution/block/filter
snapshots and a hook-neutrality comparison against ordinary original execution.
Each case records raw/composited output and preserves known pixels. Conversion
requires the complete capture and independently rereads every converted tensor
byte and shape; its receipt keeps `native_qualified` false.

The native test target `test-migan512-diff` uses the existing test-only observer
interface; it is not linked into the production library. For each captured case:

```sh
CRISPEMBED_MIGAN=1 CRISPEMBED_MIGAN512=1 \
  build/bin/test-migan512-diff \
  migan-512-places2-f32-experimental.gguf \
  original512/subtitle_bar/reference.gguf \
  "$CHECKPOINT_SHA256" \
  original512/subtitle_bar/input.rgb8 \
  original512/subtitle_bar/fill-mask.u8 \
  original512/subtitle_bar/composited.rgb8 \
  native512/subtitle_bar
```

Repeat for all eight cases without changing thresholds. The comparator reports
all captured stages, the earliest failing stage, magnitudes, maximum absolute
error, MAE, cosine, known-pixel errors and masked RGB8 quality. Full-stage cases
also compare ordinary and observed native execution byte-for-byte. The frozen
acceptance gates are maximum absolute float error0.002 and cosine0.99999;
filled RGB8 maximum difference1, MAE0.1 and PSNR50 dB; every known byte exact.
A successful synthetic panel still does not prove representative subtitle or
video quality, temporal consistency, performance, or a4 GiB memory ceiling.

## Verification status

Small local checks cover the177-tensor schema, rejection of256 state, immutable
fixture identities, mask dimensions/polarity, capture prerequisites and digest
syntax. Syntax-only C++ checks cover production/observed graph translation units,
image tiling and the new native comparator. Model-free C++ cases exercise512
padding, overlaps,384-stride origins, row-band recycling, malformed resolutions
and transactional failure; their compiled execution belongs to hosted CI.
No checkpoint download, original/native inference, link test or completed hosted
qualification is claimed by this source implementation.

## Hosted evidence (1 November 2026)

The hosted panel has now run (`tools/ci-heavy/migan512_qualify.sh`; receipts
`tests/results/migan512/`). Checkpoint first observed at
`1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a`
(29,553,256 bytes); F32 GGUF `4bec681ef10f8ca914c5331f007a0605773cd09106d4fd6d9fcc0992f0a61739`
(29,500,736 bytes). Eager and persistent-graph (`CRISPEMBED_INPAINT_GRAPH=1`)
native runs pass all eight fixtures under the unchanged gates: 273/273
full-stage comparisons with bit-exact observer neutrality, filled RGB max
difference 1, PSNR 90–96 dB, every known byte exact. Generator time about 2.0 s
per 512 tile on 4 hosted vCPUs; CLI peak RSS about 310 MB (eager) or 375 MB
(graph). F16/Q8_0/Q4_K precision variants and the real-photo evaluation are in
[the precision guide](inpainting-precision.md). Representative subtitle/video
quality, temporal consistency and publication remain open.
