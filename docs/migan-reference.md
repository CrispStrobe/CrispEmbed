# MI-GAN fixed-256 original reference protocol

Frozen 26 October 2026, before native candidate execution. This source/reference
preparation implements no public inpainting API or registry alias. The exact
source, checkpoint declaration, fixture hashes and numerical/output gates are in
[the machine-readable contract](migan-reference-contract.json). The earlier
[issue71 feasibility review](inpainting-feasibility.md) remains the source of the
broader scope and LaMa licensing/FFT limits.

The first experiment uses the official **MI-GAN256 Places2** checkpoint,
`migan_256_places2.pt`, declared25,231,129 bytes in the official
[README-linked model folder](https://drive.google.com/drive/folders/1xNtvN2lto0p5yFKOEEg9RioMjGrYM74w).
Its public file ID is `1rEJ8uJszUOkP95hcq4Bbr-4M-dlyDeMa`. The author has not
declared a payload SHA in the reviewed source. The first hosted capture must
record its **first observed** SHA and tensor inventory before deserialization;
review and freeze that digest before a native candidate or publication. A
transport digest is not an author signature. No checkpoint payload was accessed
on the shared development host during preparation.

## Original blueprint and source boundary

The runner imports the original, unchanged standalone
[`Generator(resolution=256)`](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/lib/model_zoo/migan_inference.py)
at commit`2b793c5ece43f4253e32d4afc257120a5deed6f5`. The hosted driver downloads
only pinned small files and checks every source byte count/SHA before importing
the module. That module depends only on NumPy and PyTorch. It does not import
SH-GAN, training helpers or the separate ONNX crop/smoothing pipeline.

Upstream MI-GAN has an [MIT source LICENSE](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/LICENSE)
and a separate [MIT LICENSE-WEIGHTS](https://github.com/Picsart-AI-Research/MI-GAN/blob/2b793c5ece43f4253e32d4afc257120a5deed6f5/LICENSE-WEIGHTS),
explicitly attached to pretrained models by the pinned README. Both exact notices
are preserved with the hosted artifacts and small review. No SH-GAN source or
ONNX Gaussian helper is copied into this implementation. The pinned SH-GAN
license-file gap remains a restriction on copying its source. Existing
[vision.cpp MI-GAN](https://github.com/Acly/vision.cpp/tree/26a752912d49f6c4ff4545b35a1bdf7400d349ed)
is a useful MIT implementation to study, but it is not the original reference;
this preparation copies none of its implementation.

The graph uses depthwise3×3 and pointwise1×1 convolutions, two leaky-ReLU/gain/
clamp applications per separable layer, fixed4×4 down/up filters, asymmetric
upsample padding, encoder skip features, additive RGB synthesis and checkpoint
noise. The independent schema enumerates154 state entries. Every parameter,
12 noise constants,12 noise strengths and12 upsample insertion masks must load
strictly and remain finite F32. Actual noise values are reported, never assumed
zero from initialization. The original module determines operation order.

## Fixed image and mask contract

Inputs are exactly256×256 RGB8 plus an equal-size single-channel binary mask.
White255 means *fill*, black0 means *known*. Convert once to
`known = 1 - fill/255`; then use the original demo's
`concat([known - 0.5, (RGB.float()*2/255 - 1)*known])` in NCHW F32. Reject other
dimensions, channel/byte counts and nonbinary mask values. This first protocol
performs no implicit resize, bbox crop, dilation or thresholding.

The original output is finite F32`[1,3,256,256]`. Decode exactly as the original
demo: `(raw*0.5+0.5).clamp(0,1)*255`, truncate to uint8, then restore every known
RGB pixel from the original bytes. Empty and full masks run the original model;
empty-mask composition must preserve the entire input exactly. Fixed-size
identity geometry requires none of the demo's arbitrary-size OpenCV resizing.

Eight repository-authored deterministic inputs cover a block-glyph subtitle bar,
an edge-touching hole, empty/full masks, a central square, one pixel, checkerboard
and a thin strip. They use integer-generated colors and geometry with no font,
dataset or RNG dependency. Their raw RGB/mask hashes are frozen in the contract.
These patterns establish a bounded numerical/contract experiment; they do not
establish representative scene completion, real subtitle removal or video quality.

## Capture, conversion and acceptance

[The hosted workflow](../.github/workflows/migan-reference.yml) runs the original
capture **before** importing/invoking any converter or native implementation.
It uses Python3.12, CPU PyTorch2.8.0, NumPy2.2.6 and Pillow11.3.0; the receipt
retains actual platform/backend/version/thread fingerprints and producer hashes.
Use its `checkpoint_sha256` input for an already-reviewed digest. An empty input
is permitted only for the first-observation protocol; it does not qualify weights
for distribution.

The subtitle and edge cases retain block/separable inputs, every convolution
output and block/separable/down/up outputs, cloned immediately before later
in-place operations. A separate unhooked forward must be bit exact to the
observed forward. The other six cases retain original preprocessed input, known
mask, generator raw F32 and decoded RGB. Raw arrays are `.npy`, and a separate
diagnostic`reference.gguf` exposes the same original-order names. Neither is a
weight file. Images, array hashes, L2 norms and min/max values remain in the
original report. Large arrays remain in hosted artifacts; review only the small
JSON/notices/images locally.

Only after a complete original report may
[`convert-migan-to-gguf.py`](../models/convert-migan-to-gguf.py) execute against
that exact checkpoint SHA, source/contract, per-tensor inventory and retained
capture bytes. It preserves every original state name and every F32 payload bit,
including constants/noise. Contiguous original-order OIHW/CHW bytes are unchanged;
GGUF dimension declarations reverse NumPy axes. Scalar noise strength becomes
one GGUF element. `migan.image_size=256` and source/checkpoint identities are
metadata. Independent readback checks metadata, dimensions and every tensor byte.
No F16/quantization recipe is part of this reference stage. F32 weight storage
does not guarantee F32 ggml activation arithmetic; a native prototype must audit
actual kernels and activation intermediates separately.

Frozen candidate gates require every original float stage and final raw output
to have cosine≥0.99999 and max absolute error≤0.002, with finite values and
reported absolute errors/magnitudes. Two all-zero vectors pass cosine; exactly
one zero vector fails. Report the earliest differing stage in execution order,
MAE, RMS/L2 and both magnitudes. Final filled RGB8 requires max error≤1,
MAE≤0.1 and PSNR≥50dB; known RGB pixels must be bit exact. Empty fill metrics
are null, with exact known pixels still required. Preserve generated/composited
images and all failed stages. A native run must also exercise negative dimensions,
mask polarity/nonbinary rejection and literal-zero/default gate controls. Passing
component cosine alone cannot waive decoded-image or contract failure.

The workflow retains `migan-original-raw-RUN_ID` (source, notices, checkpoint,
arrays, decoded images and optional experimental F32 model) and
`migan-original-review-RUN_ID` (small reports/notices/images), each90 days.
The first hosted checkpoint observation, source loading, actual tensor contents,
original completion and optional conversion remain **pending until a reviewed
workflow receipt exists**. Native/API/backend/quality/release qualification are
separate gates. Preparation changes no runtime version, app pin or default.

Small local validation is limited to
`python tests/test_migan_reference_contract.py -v`, covering frozen fixtures,
mask/shape negatives, missing/extra tensor rejection, constant/noise retention,
GGUF byte order, scalar mapping and independent dimension declarations without
NumPy, PyTorch, model access or inference.
