# Qwen3.5 OCR: first checkpoint and native port blueprint

Started 2026-10-13 with **LightOnOCR-3-0.8B**, the smallest missing checkpoint
in the permissive OCR queue. This is an experimental native port, **not supported OCR
inference**. No registry alias, calibrated Q4 quality result
or production HF publication exists for this lane yet.

## Immutable sources

- [Checkpoint/config/processor/tokenizer](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/tree/36d8636925327c671bd8e479c517a004f7566bd9),
  revision `36d8636925327c671bd8e479c517a004f7566bd9`, Apache-2.0 model card.
- [Qwen3.5-0.8B base family](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17),
  revision `2fc06364715b967f1860aea9cf38778875588b17`, Apache-2.0.
- Official reference: Transformers **5.5.4**, commit
  `75d3bdcd4b3ba70cee21287218d9764f33da41f0`:
  [model implementation](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
  Do not substitute moving `main`: its vision/state code already differs.

The hosted script checks immutable metadata and the base Apache license text
before downloading the checkpoint. It retains cards/license hashes and processor
configs in the report. Re-audit license/base provenance for every additional model.

## Actual inference blueprint

The 0.8B checkpoint has 24 decoder blocks of width 1024: eighteen linear
attention blocks and six full attention blocks (every fourth block). Text FFN
width is 3584. Full attention has 8 query heads, 2 KV heads and **head dimension
256**, which differs from hidden-size/head-count. The Q projection includes an
additional output gate per head; preserve its per-head interleaving when mapping.

Linear attention projects Q/K/V, Z and the scalar A/B controls. It applies a
depthwise causal convolution of width 4 followed by SiLU, per-head Q/K L2
normalization and a gated delta recurrence. The decay is
`g = -exp(A_log) * softplus(a + dt_bias)` and beta is `sigmoid(b)`. Values and
keys have sixteen heads of width 128. Prefill writes both convolution history
and recurrent state; one-token continuation consumes and updates them. State
must reset between requests and remain isolated between contexts. Delta output
uses a gated RMSNorm with Z before its output projection.

Text RMSNorm is **zero-centered** (`1 + weight`); delta's gated RMSNorm is
ordinary multiplicative normalization. Keeping checkpoint norm weights unchanged
requires that distinction in the native graph. Decoder FFN is SiLU/SwiGLU.

Text MRoPE is interleaved and **partial**: 64 rotary dimensions of each 256-wide
head, sections `[11, 11, 10]`, theta 10,000,000. Preserve unrotated dimensions,
image temporal/height/width positions and continuation offsets. Do not reuse a
full-head Qwen3-VL rotation or assume ordinary sequential positions.

The vision tower is 12 layers, width 768, 12 heads, FFN width 3072; patch size
16, temporal patch size 2 and spatial merge size 2. It has learned interpolated
position embeddings, vision RoPE, LayerNorm, tanh-approximate GELU blocks and
a final patch merger to width 1024. No DeepStack. Official processing uses RGB,
bicubic resize, mean/std 0.5, min/max pixels 65,536/16,777,216. Still-image frames
are duplicated and patches reordered into spatial merge groups by the processor.
Capture official packed pixels/grid and position values instead of a manual
preprocessor that happens to match the C++ code.

Tokenizer vocabulary capacity is 248,320 with tied input/output weights.
Qwen3.5's regex splits digits individually; storing the BPE vocabulary alone
does not implement that pretokenizer. The checkpoint's top-level EOS is 248046,
while nested text EOS is 248044 and the tokenizer names `<|im_end|>` as EOS.
Use the effective official generation configuration and compare stopping IDs;
do not silently collapse these into one value. Capture the actual chat template
for empty transcription and `grounding` prompts, and test normalized boxes
separately from transcription.

## Delivered bootstrap and verification

- [Strict exporter](../models/convert-qwen35-ocr-to-gguf.py): dense architecture
  checks, full checkpoint names/shapes compared with the official meta model,
  no unknown/missing tensor drops, tied-head validation/alias, no remote code,
  immutable revision, lossless config/template metadata and tensor manifest.
  HF prefixes become `l.*` and `v.*`; patch-convolution weights flatten in
  C,T,H,W order because ggml cannot represent a five-axis tensor. Source norm
  values are preserved. Only F32/F16 exports are currently enabled.
- [Official reference capture](../tools/dump_qwen35_ocr_reference.py): actual
  processor, full vision and decoder layer hooks, positional inputs, logits,
  prefill/continuation cache states, full-prefix comparison, greedy generated
  IDs and decoded output. It checks required stage coverage and fails on missing
  stages/nonfinite values. Synthetic transcript accuracy is explicitly unscored.
- [Hosted bootstrap](../tools/ci-heavy/qwen35_ocr_bootstrap.py): source/license
  checks, complete GGUF tensor byte/shape roundtrip, malformed shape/unknown-tensor
  rejection, official cached-vs-full-prefix logits/argmax and reference artifacts.
  Model/export weights stay in remote scratch; only reports/reference probes are
  retained. This run does not build or exercise native inference.

Launch through the existing workflow:

```sh
gh workflow run heavy-cpu.yml --ref ocr-permissive-roadmap \
  -f script=tools/ci-heavy/qwen35_ocr_bootstrap.py \
  -f 'pip=torch==2.10.0 torchvision==0.25.0 transformers==5.5.4 gguf==0.19.0 huggingface_hub safetensors pillow numpy' \
  -f timeout_minutes=60
```

## Next bounded implementation steps

1. Complete the frontend behind `CRISPEMBED_QWEN35_OCR=1`, using `core_env::on()`:
   exact tokenizer/processor, vision, merger and multimodal positions. Connect
   native merged embeddings to the verified 24-layer decoder and its native
   recurrent/KV caches, then compare standalone image-to-text against Python.
   Decoder and recurrence evidence is recorded below; frontend is still open.
2. Full-stage comparator: absolute error/magnitudes, cosine, logits/argmax and
   final decoding against the frozen official reference. Include non-square
   images, digit/Unicode prompts, repeated calls, two independent contexts,
   empty/invalid input, long-page continuation, EOS and grounding output.
3. F32/F16 before Q8/Q4; use `crispembed-quantize` with audited tensor-name
   guards. Measure calibrated precision retention for A/B projections,
   recurrent coefficients, normalization, head/embedding and vision/projector.
   Do not infer OCR quality from hidden-state cosine.
4. Complete C ABI/orchestrator/CLI/server and Python/Rust/Dart contracts per
   [contributing](contributing.md), then HF artifact/license/provenance publication
   and generated registry hashes. No alias until native decoded parity exists.
5. OvisOCR2 and Xiaomi next; handle each model's prompt/processor/EOS/MTP
   independently. LightOnOCR-3-1B reuses the separate existing LightOn engine;
   TeleOCR uses the Qwen2.5 family; larger shared-backend models follow only
   after small checkpoints pass.

## Checkpoint — 2026-10-13

Implementation branch `ocr-permissive-roadmap`, source `aea235b7`: exporter,
official reference dumper and hosted bootstrap script are committed/pushed.
Python AST parsing and `git diff --check` pass; the formatter excludes these
Python files. [Hosted ARM run 38459422250](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38459422250)
is queued; no checkpoint conversion/reference result or native parity exists yet.
Earlier task-owned Linux/ARM queued runs were cancelled while selecting the
runner and correcting aligned multimodal types in the full-prefix reference.

Next action: collect this run's full conclusion and small `bootstrap.json`,
`export-manifest.json` and official-reference JSON reports. Fix its first failing
stage before native implementation. Keep the implementation branch/worktree
until this model-backed gate completes; the public roadmap/blueprint are on main.

## First hosted result and native recurrence checkpoint — 2026-10-13

[Initial ARM run 38459422250](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38459422250)
at `aea235b7` validates **473 tensor byte/shape roundtrips**, F16 export
**1,717,205,376 bytes**, and rejection of malformed embedding shape and unknown
adapter weights. It fails before official inference because the processor's
video component imports torchvision even for still images. Add matching
`torchvision==0.25.0` to the pinned Torch 2.10 reference environment.
[Initial report](../tests/results/qwen35-ocr/bootstrap-initial.json) is retained;
this is export/schema evidence, not output parity.

The implementation branch now contains a CPU ggml delta boundary gated by
`core_env::on("CRISPEMBED_QWEN35_OCR")`. It applies official Q/K sum-based L2
normalization, preserves ggml's internal query scale and transposes HF recurrent
state to/from ggml's layout. The comparator checks exact element counts, cosine,
absolute error and norms for output and state, then feeds **native** prefill
state into continuation and checks a fresh reset. Disabled/invalid calls and
a deliberately corrupted output are controls. No production OCR API exists.

Pending checks: [full official reference plus real-checkpoint delta boundary](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38462297091),
and [independent official single-token/multi-batch 65-token chunk cases](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38462494620).
Do not claim native recurrence parity until their assertion logs are collected.

## Verified native foundation — 2026-10-13

[ARM actual-checkpoint run 38463754247](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38463754247)
and [Mac CPU single-token/multi-batch run 38464514789](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38464514789)
pass at `63b46720`. The earlier raw cached-forward harness passed a full-length
position sequence to a one-token query and failed in partial RoPE. The corrected
reference uses the actual Transformers generation position/input preparation,
including its slicing, for prefill and continuation.

- Export: all 473 tensors byte/shape-exact through GGUF; both malformed controls
  rejected. F16 export 1,717,205,376 bytes, with small/critical vectors and convs F32.
- Official reference: 267 recorded stages; every required stage present, including
  12 vision layers and 24 decoder layers for prefill/cached/full-prefix arms.
  Actual positions are `[4,1,78]`, `[1,1,1]` and `[4,1,79]` respectively.
  Cached/full-prefix logits max absolute difference **2.3365e-5**, argmax identical.
- Actual-checkpoint native recurrence/core: reported minimum cosine **1.000000000**
  throughout; recurrence output max abs **7.45e-8**, carried state max abs
  **1.61e-6**, gated normalization max abs **7.15e-7**, convolution history
  byte-identical. Prefill, native-state continuation and fresh reset pass, as do
  explicitly disabled, invalid-input and corruption controls.
- Independent official-function cases: batch 1/token 1 and batch 2/tokens 65,
  sixteen 128-wide heads, cached continuation and fresh reset. Reported cosine
  **1.000000000**, maximum state error **6.26e-7**.
- Python emits `Hello OCR  \n12345` and stops at EOS **248046**, with nine
  generated IDs. This is a synthetic Python transcript, not native OCR output.

[Initial cached-input failure](../tests/results/qwen35-ocr/cached-input-initial-failure.json),
[passed bootstrap](../tests/results/qwen35-ocr/bootstrap-passed.json) and
[per-stage errors/magnitudes](../tests/results/qwen35-ocr/native-core-parity.json)
are retained. Printed cosine 1 does not mean bit equality; absolute errors above
state the actual limits. Full native model parity remains unimplemented.

The next feature checkpoint materializes actual GGUF projection weights as F32
for structural parity (avoiding FP16 activation-dot rounding), and compares all
18 trained linear-attention blocks through their output projections with native
convolution/recurrent state carried into the cached arm. Each block still receives
its input from the official capture: this is component coverage, not a complete
native decoder or transcript. Full attention, residual/norm/FFN depth composition,
vision/tokenizer/positions and actual C ABI decoding remain necessary.

## Verified all trained linear-attention blocks — 2026-10-13

[ARM run 38468659301](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38468659301)
at `25169f68` passes **288 comparisons**: eight stages for each of eighteen
linear-attention blocks, both prefill and native-state cached continuation.
Actual checkpoint projection weights are loaded from the exported GGUF,
materialized as F32 and computed through ggml. Each block receives its normalized
input from the official reference; native outputs are **not yet composed into
a complete decoder**, and this is not native transcript parity.

| Compared stage | Maximum absolute error | Reported minimum cosine |
|---|---:|---:|
| QKV projection | 5.3406e-5 | 1.000000000 |
| Z projection | 2.2888e-5 | 1.000000000 |
| A / B projections | 1.8120e-5 / 2.4796e-5 | 1.000000000 |
| Gated normalization | 1.5259e-5 | 1.000000000 |
| Output projection | 5.2452e-6 | 1.000000000 |
| Convolution history | 2.8610e-5 | 1.000000000 |
| Recurrent state | 5.0068e-6 | 1.000000000 |

[Full stage errors/magnitudes](../tests/results/qwen35-ocr/trained-linear-parity.json)
and [complete bootstrap result](../tests/results/qwen35-ocr/trained-linear-bootstrap.json)
are retained. Expanding the dumper initially used the moving implementation's
`block_type` attribute; pinned 5.5.4 uses `layer_type`. The corrected selection
uses `text_config.layer_types` directly, with required per-block stage coverage.

The experimental native library now has `linear_cpu` and `linear_core_cpu`,
covering projections, causal convolution, A/B controls, normalized ggml delta
recurrence, gated RMSNorm and output projection. Scope is equal K/V head counts
and widths (the first checkpoint's geometry); reject incompatible dimensions
before adapting larger variants. This CPU path materializes F16/F32 weights as
F32 for structural proof; it is not a tuned deployment or quantized path.

Next: implement full-attention query gating, zero-centered Q/K RMSNorm and partial
interleaved MRoPE; then residual/norm/FFN depth composition, vision and exact
frontend/tokenizer/prompt/positions, full native logits/IDs/transcript and public
C ABI/orchestrator/bindings. A supplementary original-module batch-2 weighted
projection/cache test is queued at [38469337004](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38469337004).
No new OCR registry alias, HF weight publication or runtime release.

## Supplemental multi-batch weighted module check — complete

[Mac run 38469337004](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38469337004)
at `db4c843d` passes another **288 checks** using original Transformers GatedDeltaNet
modules with synthetic weights, batch two, 65-token prefill and one-token
continuation. Reported cos_min **1.000000000**, max abs **2.5034e-6**.
[Complete errors/magnitudes](../tests/results/qwen35-ocr/synthetic-linear-parity.json)
are retained. Cache snapshots are copied before the original module mutates its
convolution history in place. These checks exercise the compiled projection/core
path across a chunk boundary and batches; they are not trained-model accuracy.

The converter/reference/native linear-attention checkpoint is landed on main.
Its implementation branch is closed; the remaining first-model work above is
open and must use a fresh isolated worktree. No supported OCR model/alias or HF
weight publication is claimed by this checkpoint.

## Full-attention checkpoint — 2026-10-14

[Hosted ARM run 38514712654](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38514712654),
source `771f0bfa`, verifies all six trained full-attention blocks with the actual
experimental F16/F32 GGUF and original pinned Python implementation. **150 checks**
pass across prefill, native KV-carried continuation and a fresh full-prefix arm.
Every block uses its official normalized input at this component checkpoint.
Reported minimum cosine is **1.000000000**; maximum absolute errors:

| Stage | Maximum absolute error |
| --- | ---: |
| Q/K/V projections (worst) | 1.9074e-5 |
| Zero-centered Q/K head norms (worst) | 2.2888e-5 |
| Partial/interleaved RoPE key cache | 2.2888e-5 |
| Value cache | 1.9074e-5 |
| Attention pooling after sigmoid query gate | 9.6560e-6 |
| Output projection | 3.5763e-6 |

[Complete errors and magnitudes](../tests/results/qwen35-ocr/trained-attention-parity.json)
and [bootstrap result](../tests/results/qwen35-ocr/trained-attention-bootstrap.json)
are retained. The native path rotates only 64/256 dimensions, uses chronological
causality independently of multimodal rotary coordinates, repeats KV heads by
group, and preserves the query/gate split within each head. Tests reject `=0`,
invalid rotary dimensions and malformed KV caches; a perturbed pooling control
must exceed the parity threshold. This is an unpadded CPU reference path.

The same trained-checkpoint checks pass again in the composed-decoder run below.
Supplementary batch-2 attention initially stopped before comparison because the
harness imported `Qwen3_5RotaryEmbedding`; pinned 5.5.4 names it
`Qwen3_5TextRotaryEmbedding`. [Initial failure](../tests/results/qwen35-ocr/synthetic-attention-initial-failure.json)
is retained. The corrected original-module test is at
[38515525129](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38515525129).
No registry alias, quantization calibration or weight publication yet.

## Composed native decoder — 2026-10-14

[ARM run 38514911908](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38514911908),
source `0d33acaa`, passes **183 comparisons**: 72 decoder-layer outputs (24 each
for prefill, cached continuation and a fresh full-prefix), three final norms,
three vocabulary logits, 96 convolution/recurrent/KV cache fields and nine
original generation score vectors. This arm supplies official merged embeddings
and multimodal position coordinates, then composes every native decoder layer;
it does **not** supply hidden states between layers or Python caches.

| Stage | Maximum absolute error | Reported minimum cosine |
| --- | ---: | ---: |
| All decoder-layer outputs | 2.0266e-5 | 1.000000000 |
| Final zero-centered norm | 2.0791e-4 | 1.000000000 |
| Vocabulary logits / generation scores | 4.8637e-5 | 1.000000000 |
| Native-carried cache fields | 8.5831e-5 | 0.999999940 |

The free-running native greedy decoder selects its own next token and embedding,
with native caches carried each step. **9/9 IDs match the original generation,
including EOS248046**: `9419 77796 2228 16 17 18 19 20 248046`.
Those IDs decode with the official tokenizer to `Hello OCR  \n12345` (two
Markdown spaces before the newline). This is decoder token parity on one
controlled fixture, not native image-to-text or a scored OCR benchmark.
[Complete stage errors/magnitudes](../tests/results/qwen35-ocr/composed-decoder-parity.json)
and [bootstrap result](../tests/results/qwen35-ocr/composed-decoder-bootstrap.json)
are retained. Norm absolute error is higher than layer-output error; the report
keeps magnitudes rather than treating rounded cosine1 as bit equality.

Next: native vision with all 12 layer/merger checks, then image preprocessing,
patch order, Qwen3.5 BPE/chat template, multimodal positions and standalone
image-to-text parity; wire C ABI/orchestrator/CLI/server/bindings only after those
checks. This F16-weight/F32-arithmetic CPU reference is not a tuned or quantized
runtime. The current linear core requires equal K/V head counts/widths, and
attention supports unpadded inputs. No latency/GPU/Q4 or model-support claim.

## Supplementary full-attention batching — complete

[Mac run 38515525129](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38515525129),
source `a5bac46c`, passes **150 additional comparisons** against the original
pinned attention modules with synthetic weights, two batches, 65-token prefill,
cached continuation and a fresh full-prefix replay. Each rotary axis differs,
with a separate batch offset; head norm weights are nonzero. This tests causal
masking, partial interleaved RoPE, GQA groups, per-head gates and immutable native
KV carry beyond the single OCR fixture. Reported cos_min1.000000000,
max abs2.3842e-6. [Complete stage evidence](../tests/results/qwen35-ocr/synthetic-attention-parity.json).
The earlier import failure is fixed; thresholds remain unchanged.

The full-attention/composed-decoder lane is complete and landed. Remaining
first-model work is native vision/frontend, standalone image-to-text validation,
C ABI/orchestrator/CLI/server/bindings, Q8/Q4 critical-tensor calibration and
validated HF publication/registry pins. The permissive queue remains active,
starting with this checkpoint until those completion gates hold.

## Native vision / connected decoder lane — 2026-10-14

Implemented in `src/qwen35_vision.{h,cpp}` behind the existing value-parsed
`CRISPEMBED_QWEN35_OCR` gate. The CPU reference uses actual F16/F32 GGUF weights
and F32 projection arithmetic: flattened Conv3d patch projection plus bias,
learned bilinear position interpolation in merge-group order, full 2D vision
RoPE, ordinary LayerNorm, per-frame non-causal attention, tanh-GELU blocks and
an exact-GELU patch merger. It preserves the checkpoint's `v.blocks.*` names.
No Python, ONNX Runtime or reference array is part of the native library.

Hosted trained-checkpoint run
[38526207651](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38526207651),
source `67430932`, compares all 12 layers and the patch/position/rotary/merger
boundaries. A connected `test-qwen35-decoder --native-vision` arm constructs
native text embeddings, inserts native vision output at the image-token slots,
then verifies the whole decoder, vocabulary logits, native cache states and
free-running greedy IDs. **Only official packed pixels, prompt IDs and position
coordinates are supplied**; this does not verify a raw-image frontend/tokenizer.
The run's conclusion and numerical evidence are still pending.

Original-module non-square/multi-frame controls are at
[38526798047](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38526798047),
source `5a427124`: two non-square grids, three frames, independent per-frame
Python replay, native composed layers and reset/invalid controls. The initial
harness stopped before comparison because `Qwen3_5VisionConfig` is not exported
from Transformers' top-level namespace; the corrected import is from its model
configuration module. [Initial failure](../tests/results/qwen35-ocr/synthetic-vision-initial-failure.json)
is retained. Production alias/API, quantization and publication remain open.
