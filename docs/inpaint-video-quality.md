# Reproducible inpainting video panel

`tests/inpaint_video_quality.py` generates five owned synthetic clips and runs
each through the same CLI/model twice: independent frames (`--inpaint-temporal 0`)
and the temporal candidate (window 2 by default). No input images, clips, models
or dependencies are downloaded by this harness. Each clip contains six 96×64
RGB8 frames: 30 frames per arm, 60 total. CPU inference uses two threads by default.
This is a bounded regression panel, not a representative film-quality benchmark
or an original-model parity result.

## Fixed inputs

Every clean pixel and subtitle glyph is authored in the generator. The protocol
has no random-number dependency. `tests/test_inpaint_video_quality.py` freezes the
five input-record SHA-256 hashes; a deliberate fixture/protocol change must retain
the old evidence and use a separately reviewed successor protocol.

| Case | Background and mask | Purpose |
| --- | --- | --- |
| `static` | Flat tinted background; clean first/last frames | Exact background propagation and flat-color preservation |
| `fade` | Textured background brightens by two levels each frame; unchanged subtitle | Detect cumulative stale reuse even though each neighboring change is below the static threshold |
| `moving-fine-texture` | One-pixel moving checkerboard with nearly equal luma and strongly different color | Detect cell-mean motion cancellation and chromatic texture mistaken for flat color |
| `subtitle-change` | Fixed texture; authored glyph changes halfway; clean endpoints | Changed-mask bookkeeping and temporal transition behavior |
| `scene-cut` | Abrupt change of background/color at frame 3; unchanged subtitle | Avoid propagation or blending across a hard scene change |

The harness writes each clip's `input.rgbm`, `clean.rgb` and `mask.gray`.
Input records contain RGB8 followed by an 8-bit mask; only masked pixels are
replaced by white subtitle pixels. Clean frames are ground truth. Dimensions,
record counts and all input hashes are in `report.json`.

## Model-free checks

These commands check the metric oracle and transport/evidence handling. They do
not measure or qualify a real model:

```sh
python tests/test_inpaint_video_quality.py -v
python tests/inpaint_video_quality.py \
  --binary build-inpaint/test-inpaint-cli-driver --fake-engine \
  --source-sha "$(git rev-parse HEAD)" \
  --output-dir video-quality-fake
```

Use the corresponding `Release/*.exe` path on Windows. The explicit fake mode
creates a tiny placeholder model accepted only by the test driver and clears
inherited test-failure and declared model/load controls before enabling the two
required engine/stream gates. CPU thread variables remain explicit and recorded.
A production binary rejects that placeholder.
Every report carries `fake_engine` and `model_quality_qualified: false`.

## Hosted CPU measurement with the reviewed model

Build and inference belong on hosted CI. Reproduce the existing reviewed
MI-GAN256 F32 model separately; that tool validates the immutable checkpoint,
tensor bytes, model digest and license notices:

```sh
python tools/prepare_migan_inpaint.py --download --output-dir video-model
CRISPEMBED_MIGAN=1 CRISPEMBED_INPAINT_STREAM=1 \
  CRISPEMBED_INPAINT_GRAPH=0 CRISPEMBED_INPAINT_FAST=0 CRISPEMBED_MIGAN_GPU=0 \
  CRISPEMBED_MIGAN_CPU_KERNELS=0 CRISPEMBED_MIGAN_FAST_MATMUL=0 \
  CRISPEMBED_MIGAN_DEQUANT_WEIGHTS=0 CRISPEMBED_MIGAN_QUANT=0 \
  CRISPEMBED_INPAINT_SHA256_ACCEL=0 CRISPEMBED_MIGAN_BATCH=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tests/inpaint_video_quality.py \
  --binary build-inpaint/crispembed \
  --model video-model/migan-256-places2-f32-experimental.gguf \
  --expected-model-sha256 b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b \
  --source-sha "$(git rev-parse HEAD)" \
  --threads 2 --timeout 120 --output-dir video-quality-real
```

The real-model path requires the engine/stream gates and a full source commit.
It explicitly requests the CPU backend. The command fixes optional graph/fast/GPU,
CPU-kernel, fast-matmul, dequantization, quantized-model admission and accelerated
hashing gates at `0` for the reviewed F32 baseline. `CRISPEMBED_MIGAN_BATCH=1`
is an integer tile capacity; `0` is ignored by the runtime, so use explicit `1`.
Real-arm environment values are preserved exactly and recorded, including these
six MI-GAN/load controls. `runtime_environment` records only `OMP_NUM_THREADS`
and `OPENBLAS_NUM_THREADS`; unset values are null. The harness does not capture
the whole environment or credentials. The model is
approximately 25 MB, with no framework import. Model replication/download errors
must fail the hosted run and retain its logs; they are not a successful quality
measurement. Do not substitute another model merely to obtain a green result.

The timeout is per arm/clip and terminates a timed-out process while retaining
its partial output and stderr. Hosted workflow timeouts provide the overall
build/download/panel budget. Run this small panel after runtime changes when
quality evidence is needed; ordinary model-free checks do not download weights.

## What is measured

`report.json` records the source commit; binary, model and harness SHA-256;
actual model bytes; declared gates/load controls and CPU thread environment;
fixed inputs; output/log hashes; and these
measurements for each arm of each clip:

- **Masked MSE and PSNR:** channel error against clean ground truth only at
  pixels selected by the current frame's mask. PSNR is null for exact zero MSE,
  with `masked_exact: true`, rather than a nonstandard JSON infinity.
- **Temporal error MAE/MSE:** first difference of reconstruction error,
  `(output[t] - clean[t]) - (output[t-1] - clean[t-1])`, over the union of the
  adjacent masks. This discounts actual scene motion and includes subtitle
  appearance/disappearance. It is an error/flicker proxy, not a perceptual score.
- **Known pixels:** exact unmasked RGB preservation, mismatching channel count
  and maximum absolute difference. Any known-pixel change fails the panel.
- **Engine frames:** parsed from the CLI's actual stderr summary. Missing or
  inconsistent frame counters fail the panel.
- **Elapsed time and peak RSS:** per subprocess, including initialization and
  stream I/O. Linux/macOS use the specific child's `wait4.ru_maxrss`; platforms
  without that interface record null, never zero as a fabricated measurement.

The report always retains numeric changes and regression flags for masked MSE
and temporal-error MAE. Without declared numeric bounds, `passed` means the
process, exact record count, known-pixel and counter checks passed. It does not
mean every metric improved, nor does it qualify a model. Read each clip's two
arms and `comparison` before making a quality decision. Timing is diagnostic:
the independent arm precedes the temporal arm on one host, so it is not a
randomized performance benchmark.

For a separately declared regression protocol, supply
`--max-mse-increase` and/or `--max-temporal-mae-increase` **before** measuring.
These are absolute upper bounds on candidate-minus-independent error, not PSNR
or relative ratios. Violations fail the command and remain in the report.
Never relax a bound after observing a failing result or overwrite its evidence.

Output directories must be new. Reports are checkpointed after each arm, and
launch errors, timeouts, malformed output and failing metric verdicts retain
their bytes/logs. Keep the complete output directory with the hosted run's
artifact, plus the model replication receipt and license notices. A passing
synthetic panel does not authorize a default flip, claim real-film parity,
qualify other engines/precisions/backends, or close issue 71.

## Stream failure contract

The CLI rejects an incomplete record without sending it to the engine. It then
flushes all preceding complete records and returns nonzero with the failing
input frame number. If inference fails while flushing that valid prefix, its
earlier frame number/error takes precedence. An inference failure at EOF keeps
every successfully completed output before it; later frames are not emitted.
Output-write/flush errors are failures. Image-format settings do not alter or
block the raw RGB stream.
