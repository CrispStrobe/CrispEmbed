# MI-GAN fixed-256 CPU/F32 prototype

**1 November source follow-up:** the opt-in [inpainting CLI](inpainting-cli.md)
now wraps this unchanged graph with native256 tiling, PNG/mask handling and
persistent batches. The original boundary and reviewed results below describe
the fixed256 experiment. They do not qualify the new wrapper or extend its
numerical, backend or perceptual claims; the CLI's hosted checks remain separate.

This is a separately built experimental graph, following the unchanged original
module and [frozen reference protocol](migan-reference.md). Source preparation
and native execution are separate: no native result qualifies before the first
actual original checkpoint SHA, tensor inventory and eight-case capture have
been reviewed and frozen. Preparation alone establishes no numerical, image,
backend or representative inpainting quality result.

[`src/migan.cpp`](../src/migan.cpp) implements the original256→4 encoder and
4→256 synthesis graph on the CPU backend, retaining every learned filter,
upsample insertion mask, noise constant and noise strength. The loader requires
exactly154 original-name finite F32 tensors, their fixed shapes,
`migan.image_size=256`, original source commit and a separately supplied reviewed
checkpoint SHA matching GGUF metadata. The owning hosted driver must also check
the actual GGUF SHA against the reviewed conversion receipt.

Both model initialization and execution require the shared value-parsed
`core_env::on("CRISPEMBED_MIGAN")` gate. Unset, empty and literal0 reject before
loading/execution. The context uses the CPU backend directly and makes no GPU
or fallback claim. A context is used serially; it is not a concurrent inference
API. The first comparison uses two native CPU threads, while the original uses
four: those timings cannot establish a paired speedup.

Pointwise/fromRGB/toRGB convolution explicitly constructs F32 im2col followed by
F32 matmul and requests F32 precision. Direct depthwise convolution uses the
pinned ggml's contiguous WHCN F32 source/kernel/F32 accumulation path. Every
constructed graph node must be F32 and supported by the selected CPU backend.
This makes activation precision an executable check. The graph uses the loaded
4×4 filters, nearest2× upsample, loaded insertion masks and exact
`pad_ext(left=2,right=1,top=2,bottom=1)`. Noise is multiplied and added from the
loaded state at its original position. Leaky ReLU, sqrt2 gain and clamp256 occur
after depthwise and after pointwise/upsample/noise in each separable layer.
Additions and observed convolution outputs remain distinct; the original raw
convolution hooks snapshot values before later in-place Python operations.

The C++ header exposes an internal prepared-F32 graph plus fixed-size RGB8/
binary-mask preprocessing and original uint8 truncation/composition helpers.
Dimensions and byte counts are strict; nonbinary masks and nonfinite input/
output reject. White255 means fill. Known RGB bytes are copied from the original
image. The original prototype has no arbitrary-size resize/crop, ONNX smoothing
wrapper, public C ABI, orchestrator/server/binding/registry integration or
installed archive. The later CLI uses a separate tiling wrapper; this internal
graph's fixed256 contract remains unchanged.

## Hosted validation interface

Build only the explicit `crispembed-migan` and `test-migan-diff` targets in
hosted CI. The targets are `EXCLUDE_FROM_ALL`, and neither is added to the
production shared library or install lists. `test-migan-diff` links a separate
`crispembed-migan-test` archive compiled with a test-only observer; the ordinary
prototype archive contains no observer callback or original reference payload.

The test command is:

```sh
test-migan-diff MODEL.gguf REFERENCE.gguf CHECKPOINT_SHA256 \
  RGB.raw FILL_MASK.raw ORIGINAL_COMPOSITED.raw DUMP_DIR
```

All three `.raw` inputs are exact256×256 uint8 payloads (RGB HWC, mask HW, RGB
HWC). The hosted driver derives their bytes from the frozen original case and
checks their identities before execution. `REFERENCE.gguf` is that case's
separate original diagnostic artifact. The harness performs actual RGB8
preprocessing, compares its tensor to the original and executes that native
input. It never substitutes golden outputs or reconstructed operator outputs.

Subtitle-bar and edge cases execute a second observed graph. Its retained nodes
are marked `ggml_set_output` **before allocation**, preventing allocator reuse
from destroying intermediates. Retained views also mark every `view_src`
allocation owner: the pinned allocator can free an unmarked owner when its last
view expires even if the view itself is an output. Ordinary and observed final output bytes must
match exactly. Each original stage must be compared exactly once, including
block inputs and tuple outputs; missing or extra stage coverage rejects.
The other six cases execute only the ordinary graph and final comparison.
References and native arrays stay remote; only small reports/images need local
review. Allocator bytes and timings are diagnostics, not backend or speed proof.

`crispembed-diff` supplies stage maximum absolute error, MAE, RMS, per-row/global
cosine and both L2 magnitudes. The test enforces the frozen0.002/0.99999 gates
directly, without the harness's optional cosine environment override. It records
the first failing stage in execution order, checks finite values and dimensions,
and writes actual native `.f32` arrays for attribution. Decoded fill RGB8 must
pass max1/MAE0.1/PSNR50dB; known pixels must be bit exact. Empty fills have null
image metrics, and exact filled images have null PSNR plus zero MSE and
`exact_fill=true` (infinite PSNR). Both native generated/composited raw RGB8
images are preserved for the owning hosted driver to encode and review.

Machine-readable stdout records use `MIGAN_CONTRACT`, `MIGAN_GATES`,
`MIGAN_DIFF`, `MIGAN_IMAGE` and `MIGAN_SUMMARY` prefixes. Exit0 requires complete
coverage, every float/image gate and observer neutrality; exit1 retains numerical
or decoded-image failures, and exit2 indicates setup/contract/coverage failure.
`--contracts` exercises pure RGB/mask helpers, and `--gate-controls` exercises
unset/empty/literal0 without a model; `--reject MODEL SHA` checks a
hosted malformed-model control with the gate enabled. These are validation
interfaces, not product CLI options.

The first [native run40336246545](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40336246545)
is rejected and retained. It completed all494 float comparisons, all eight image
gates, ordinary/observed final neutrality, eight malformed-model controls and
source/member checks, but128 intermediate views per full case were stale.
Earliest `encoder.b256.conv1.input.0` max error14.8653 and subtitle L2
2149.386/1805.116 share the exact payload SHA of a later passing depthwise output,
evidence of allocator reuse. All eight final raw stages pass; subtitle/edge
maxima3.33786e-6/2.29478e-6 and every known RGB byte match. This does not waive
the256 failed intermediate comparisons or qualify the initial prototype receipt.

The correction changes only the test observer's retention flags. A hosted
`--observer-retention` regression uses the real private mark helper on a small
clamp/reshape view, followed by a consuming operation and competing allocation.
It requires the old view-only path to reproduce corruption, the owner-chain path
to preserve the anchor, and both final arithmetic outputs to match exactly.
Every rerun also compares all eight ordinary final payload SHAs to that reviewed
initial run. Original arrays, weights, numerical/image budgets and ordinary graph
arithmetic remain unchanged; the full494-field replay is still required.

No local builds, framework imports, model access or inference are part of this
preparation. Formatting and existing dependency-free contracts can be checked
locally; actual compilation, eight-case comparison, resource measurements and
negative model variants require hosted execution after the original freeze.
Any failed alternative, strict flag or initial run must remain retained.

## Reusable workflow and frozen replay inputs

[`migan-native.yml`](../.github/workflows/migan-native.yml) supports direct
dispatch and reusable calls. A build-only caller needs no checkpoint identities:

```yaml
jobs:
  migan-native-build:
    uses: ./.github/workflows/migan-native.yml
    with:
      phase: build-only
```

Build-only compiles the explicit targets, executes pure RGB/mask/gate controls
and checks actual archive symbols. It retains the executable, ordinary archive
and observer archive; no model download, framework import or inference occurs.
The original frozen reference may still be pending during this build check.

Replay requires all of these explicit, independently reviewed inputs:

| Input | Required identity |
| --- | --- |
| `phase` | `replay` |
| `reference_run` | Successful original producer run ID |
| `reference_head` | Its exact40-hex Git commit |
| `reference_workflow` | Its exact workflow path; the original workflow or existing metadata dispatcher |
| `reference_artifact_id` | Its `migan-original-raw-RUN` artifact ID |
| `reference_archive_sha256` | Artifact ZIP SHA matching service metadata and hosted download |
| `reference_report_sha256` | Original `original/report.json` byte SHA |
| `conversion_report_sha256` | Original F32 conversion receipt byte SHA |
| `checkpoint_sha256` | Reviewed first-observed actual official checkpoint SHA |
| `model_gguf_sha256` | Original lossless F32 GGUF SHA |

The hosted driver refuses replay when any identity is absent. It checks the
successful producer, source, artifact manifest and every member, checkpoint,
notices, original/converted tensor mapping, fixture pixel hashes, original
NPY/GGUF arrays and original RGB decoding/composition. All eight original cases
must verify before native model loading. It reuses frozen arrays and weights;
there is no original recapture or reconversion. Eight malformed-model controls
cover metadata, checkpoint, missing noise, extra probe, wrong shape, nonfinite
state and F16 rejection.

Native stdout is parsed strictly, including duplicate/missing records, original
stage execution order, per-stage verdicts, totals and every image gate. An
independent NumPy reader recomputes raw float errors, row/global cosines,
magnitudes and decoded RGB8 metrics. Log-rounding checks do not widen acceptance
thresholds. It verifies uint8 decoding/composition directly from native raw floats
and checks the retained ordinary output bytes against observed final bytes.
Numerical failures are retained across all eight cases; setup failures retain
partial reports/logs. The producer reports bounded candidate success separately
from independent review/qualification; representative quality remains false.

`migan-native-raw-RUN` retains native arrays, build archives, logs and PNGs;
`migan-native-review-RUN` retains small JSON/log/symbol/image evidence and copies
of original/conversion receipts plus notices. Both expire after90 days. The
workflow records exact source/build flags, binary/archive hashes, CPU fingerprint,
GNU-time peak RSS and wall time. Large raw arrays remain remote. Small stdlib
tests exercise parser contradictions, required identities, strict JSON and safe
ZIP extraction without NumPy, Torch or model access.

The graph decomposition also studies/adapts
[vision.cpp at26a752912d49f6c4ff4545b35a1bdf7400d349ed](https://github.com/Acly/vision.cpp/tree/26a752912d49f6c4ff4545b35a1bdf7400d349ed).
Its full [MIT notice](third-party/vision-cpp-LICENSE.txt), Copyright2025 The
vision.cpp authors, is retained. The original MI-GAN source/weight MIT notices
remain with original artifacts. No SH-GAN or ONNX smoothing source is copied.
This prototype changes no runtime release, application pin, model alias or
default. F16/quantization, other backends, real subtitle/scene quality and video
remain separate protocols.


## Reviewed fixed256 result

[Actual40344396533](../tests/results/migan/native-parity.json) at2526fcac passes
all494 unchanged float comparisons, eight image gates and all eight malformed
model controls. The tiny allocator regression passes, both full observers are
bit-neutral, and all eight ordinary final SHAs equal the retained first run.
Worst intermediate max absolute error is1.77096e-5; minimum row cosine is
0.99999998399. Worst final raw error remains1.40667e-5; known RGB errors are0,
fill differences at most1 and worst MAE7.12077e-5. Root review verifies the
actual46-member small ZIP,17 source hashes, command logs and decoded PNG metrics.
Raw arrays/models remain remotely retained until24 January2027; their actual
transport and member identities are in the receipt. This bounded CPU/F32
prototype is positive; representative quality and production qualification
flags remain false. No old rejection, default or numerical budget changes.
