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
The checkpoint's tokenizer config advertises a combining-mark-aware split, but
its [serialized tokenizer](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/tokenizer.json)
contains **NFC plus the older Qwen regex**. Pinned Transformers 5.5.4's generic
[TokenizersBackend](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/tokenization_utils_tokenizers.py)
loads that file directly; configuration hints do not replace its normalizer or
pretokenizer. Record the instantiated backend before choosing native behavior.
Its numeric regex splits digits individually. Added tokens bypass BPE and include
`<think>`/`</think>`, which a scanner for only `<|...|>` misses.

The checkpoint's top-level EOS is 248046, while nested text EOS is 248044.
Preserve effective generation EOS, tokenizer EOS and padding separately. Capture
actual templates for empty transcription and `grounding` prompts, and validate
normalized boxes independently of transcription.

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
  retained. The current script also builds and compares native recurrence,
  trained linear/full attention, vision, decoder, native-position and raw RGB/BPE
  arms. Two owned image-to-text fixtures pass with the explicit F32 vision recipe;
  production contexts, corpus validation and integration remain unfinished.

Launch through the existing workflow:

```sh
gh workflow run heavy-cpu.yml --ref main \
  -f script=tools/ci-heavy/qwen35_ocr_bootstrap.py \
  -f args=--frontend-vision-f32 \
  -f runner=ubuntu-24.04-arm \
  -f 'pip=torch==2.10.0 torchvision==0.25.0 transformers==5.5.4 tokenizers==0.22.2 gguf==0.19.0 huggingface_hub safetensors pillow numpy' \
  -f timeout_minutes=60
```

## Next bounded implementation steps

1. The gated native RGB/BPE frontend now passes its independent input checks and
   two connected trained fixtures with the explicit F16 text/F32 vision recipe.
   Next, create an owning inference context with load/free, file/raw-RGB entry
   points, bounded native generation, confidence and clear/reset semantics.
   No reference arrays belong in that context. Preserve the source pixel budget;
   reject pages beyond the current1024-patch bound instead of silently reducing them.
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
The first F16-weight run fails the original absolute gate at raw vision block11:
max abs0.167602539 despite reported cosine1.000000000 and L2 magnitudes
24610.5508/24610.543. Earlier blocks pass; the merger output is much closer,
max abs5.2571e-5. The connected arm was not reached in that initial run.
[Initial per-stage errors/magnitudes](../tests/results/qwen35-ocr/trained-vision-initial-failure.json)
are retained. No threshold was increased; F32 and explicit critical-tensor
retention are now being audited at
[38527589341](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38527589341).

Original-module non-square/multi-frame controls are at
[38526798047](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38526798047),
source `5a427124`, **passes 21 comparisons**: two non-square grids, three frames,
independent per-frame Python replay, native composed layers and reset/invalid
controls. Reported cos_min1.000000000, max abs7.1526e-7; official packed versus
separate frame max abs1.1176e-8. [Complete stage evidence](../tests/results/qwen35-ocr/synthetic-vision-parity.json). The initial
harness stopped before comparison because `Qwen3_5VisionConfig` is not exported
from Transformers' top-level namespace; the corrected import is from its model
configuration module. [Initial failure](../tests/results/qwen35-ocr/synthetic-vision-initial-failure.json)
is retained. Production alias/API, quantization and publication remain open.

The converter now accepts repeatable `--keep-f32 GGUF_NAME` for exact tensor
names. Unknown names reject before output creation; retained names and actual
per-tensor dtypes/hashes are recorded in the manifest. The default experimental
F16 recipe is preserved for comparison. No precision recipe is promoted until
its full native stage/logit/token checks pass.

Current CPU reference bounds: at most 4096 packed patches total and 1024 per
frame. Larger-page capacity and attention performance remain deployment work;
this is not a latency-tuned public engine.

### Trained source-precision audit — 2026-10-14

[ARM 38527589341](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38527589341),
source `80c12d91`, verifies all 473 tensor byte/shape roundtrips for each additional
precision recipe. Raising source precision does **not** resolve the original
strict gate. All four connected arms still select **9/9 Python IDs through EOS**;
this decoded-token result does not erase the intermediate discrepancies.

| Recipe | Bytes | Raw vision block11 maximum absolute error |
| --- | ---: | ---: |
| Original experimental F16 | 1,717,205,376 | 0.167602539 |
| F16, FFNs5/11 retained F32 | 1,736,079,744 | 0.165191650 |
| F16 text, all vision matrices F32 | 1,918,138,752 | 0.175903320 |
| All F32 | 3,421,927,808 | 0.175903320 |

The all-F32 arm first exceeds 0.002 inside block11 at its second LayerNorm
(max abs0.004591), then the FFN amplifies that into 0.176 raw output error.
The worst raw-layer element is approximately -367.63 versus -367.81. Final
vision merger output is much closer; connected vocabulary scores pass while
prefill/full-prefix final decoder norms exceed 0.002. This points to accumulated
native arithmetic, rather than a source-weight precision fix.
[All stage errors, magnitudes and selected IDs](../tests/results/qwen35-ocr/vision-precision-audit.json)
and [bootstrap](../tests/results/qwen35-ocr/vision-precision-bootstrap.json) are
retained; no source-precision recipe is promoted.

The next A/B at
[38529255416](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38529255416)
uses separate default-off `CRISPEMBED_QWEN35_VISION_PRECISE_NORM` and
`CRISPEMBED_QWEN35_VISION_PRECISE_ATTN` gates for double-precision reductions,
with F32 boundary tensors/projections. It compares each flag and both together,
plus an all-F32 weight control, against the same original absolute/cosine/IDs
gates. The original path remains available with both flags0. The pinned
[PyTorch LayerNorm primitive](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)
and [softmax primitive](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)
were inspected for reduction and affine ordering; native code does not call Torch.

### Official CPU self-parity and calibrated comparison

[ARM 38531977370](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38531977370),
source `c8e395e0`, runs the **original model** with the same F32 weights/inputs,
then disables MKLDNN or splits independent patch convolution into single-patch
calls. Disabling MKLDNN changes raw vision block11 by **0.193328857**, final
vision merger by **4.8757e-5**, final decoder norm by **0.005830765** and logits
by **7.4387e-5**. IDs remain **9/9 identical through EOS**. Patch chunking is
exactly equal on this fixture. [Independent self-parity report](../tests/results/qwen35-ocr/official-cpu-self-parity.json).
The old uniform absolute0.002 gate rejects the official model against itself at
those amplified composed states; this is not a source-weight precision defect.

The precision-reduction experiments also retain all nine IDs but do not meet
that old uniform gate. [All alternatives and magnitudes](../tests/results/qwen35-ocr/vision-reduction-audit.json)
are preserved. Both reduction flags remain default-off; no alternative is
promoted because a particular mean/maximum moved in a favorable direction.

The test-only comparison now records both the original strict-absolute result
and an independent reference-variation budget. Vision budgets use the named
stage's maximum F32 backend self-variation, final decoder norms use each
component's variation: **0.002 + 2 × self-variation**. CPU backend controls must
retain the original greedy IDs before a calibrated reference is emitted.
Tensor counts/shapes, finite values and minimum cosine0.99999 remain required;
**decoder hidden states, caches and vocabulary scores keep absolute0.002, and
native greedy IDs/EOS must match exactly**. Production inference never reads
reference controls or budgets. Synthetic tests without controls keep the old
absolute gate.

An initial per-component vision envelope was too narrow for errors distributed
across correlated feature channels; [38533775792](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38533775792)
retains that failure, while its connected decoder **passes186 checks** and
all nine IDs. A per-patch envelope at
[38534784004](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38534784004)
was also too narrow at some locations: a single CPU backend pair does not bound
a third implementation's error at every channel/patch. The final comparison uses
the independently observed **named-stage** variation for vision, together with
unchanged minimum cosine **per patch**; it retains elementwise final-norm bounds
and strict downstream gates. Deliberate corruption beyond the observed budget
must reject. Verification is at
[38535674747](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38535674747).
The source-precision/default inference path is unchanged. This calibrated gate
is not a claim of raw bit equality, standalone image-to-text or corpus OCR quality.

## Native vision + connected decoder checkpoint — landed 2026-10-14

[ARM 38535674747](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38535674747),
source `0902229d`, passes **37 vision-boundary comparisons and 186 connected
decoder comparisons**, including reset/invalid/gate0 and beyond-budget corruption
controls. It uses the original experimental F16/F32 GGUF, F32 projections and
both precise-reduction flags **off**. No extra source-precision recipe is needed
for this calibrated checkpoint; higher-precision alternatives remain available.

Native patch projection, learned positions, RoPE, all 12 vision blocks and merger
feed native text embeddings and all 24 decoder blocks. **9/9 native-selected IDs
match Python through EOS248046**; the official tokenizer decodes the IDs to
`Hello OCR  \n12345`. Vocabulary logits/scores remain under the original absolute
limit, worst **9.2268e-5**, reported cosine1.000000000.

| Scope | Minimum reported cosine | Maximum raw absolute error | Maximum calibrated-budget ratio |
| --- | ---: | ---: | ---: |
| 37 vision boundaries | 0.999999940 | 0.167755127 | 0.486603 |
| 186 connected decoder stages | 0.999999881 | 0.006835938 | 0.788510 |

The raw errors above are **not** erased or called bit equality. The report marks
which stages fail uniform absolute0.002 and which use the independent reference
budget. Vision uses named-stage CPU self-variation and retains per-patch cosine;
final decoder norms use per-element variation. Decoder hidden states, cache
fields and vocabulary scores still use absolute0.002; IDs/EOS remain exact.
[Vision evidence](../tests/results/qwen35-ocr/trained-vision-calibrated-parity.json),
[connected evidence](../tests/results/qwen35-ocr/trained-connected-calibrated-parity.json)
and [bootstrap](../tests/results/qwen35-ocr/vision-connected-bootstrap.json) are retained.
The two rejected local-envelope experiments remain linked above.

[Mac 38534221704](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38534221704)
rechecks the comparator on the original synthetic non-square/multi-image/frame
model: **21 checks** pass, cos_min1.000000000, max abs7.1526e-7, original
absolute0.002 gate because no CPU-control arrays are present. Original packed
versus separate-frame replay max abs1.1176e-8.
[Full control evidence](../tests/results/qwen35-ocr/synthetic-vision-calibrated-comparator-parity.json).

The vision/connected-decoder lane is landed and its active-work row removed.
**Next:** raw RGB resize/normalization/patch order, Qwen3.5 BPE/chat template and
multimodal positions, then standalone image-to-text/corpus checks, C ABI,
orchestrator/CLI/server/bindings, larger-page performance, Q8/Q4 calibration and
validated HF publication/hash pins. Packed pixels, prompt IDs and position
coordinates are still supplied by the official frontend. No supported alias,
standalone OCR, speed, GPU or corpus accuracy claim is made by this checkpoint.

## Native position generation — 2026-10-14

`qwen35_ocr::image_positions` derives the chronological text plane and three
multimodal axes from contiguous modality spans and image grids, using the pinned
original `get_rope_index`/`get_vision_position_ids` semantics. Image coordinates
advance by the larger merged grid dimension, rather than the number of image
tokens. `cached_positions` derives the next coordinates from the native cache's
sequence length plus the prefill RoPE delta. Negative deltas are valid.

Both helpers use the default-off `CRISPEMBED_QWEN35_OCR` gate. Scope is unpadded
batch-one still images, including multiple images separated by text/delimiters.
Videos, temporal grids other than one, mismatched image-span lengths, unused
or missing grids, invalid modalities and context overflow are rejected, with
outputs cleared. This does not add production C ABI exposure.

The existing official-coordinate decoder arms remain. The new
`test-qwen35-decoder --native-positions` arm computes its own prefill/full-prefix
positions and every continuation position from its native carried cache; saved
reference positions are comparison targets. Packed pixels, prompt IDs and
modality IDs still come from the official processor. Standalone raw-image OCR,
tokenizer/template rendering, decoded-text processing and corpus quality remain
unverified.

The next frontend comparison must record the resolved official processor class
and backend. The pinned [processor config](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/processor_config.json)
names `Qwen2VLImageProcessor`, while the
[legacy preprocessing config](https://huggingface.co/lightonai/LightOnOCR-3-0.8B/blob/36d8636925327c671bd8e479c517a004f7566bd9/preprocessor_config.json)
names `Qwen2VLImageProcessorFast`. Transformers 5.5.4's
[Qwen2VL processor implementation](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
is a `TorchvisionBackend`; its
[auto-loader](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/auto/image_processing_auto.py)
can select/fall back between backends. Verify the actual reference environment,
then compare resized non-square raw RGB fixtures. The current shared native
float bicubic helper documents residual differences from torchvision's uint8
coefficient quantization; the 256x256 no-resize fixture cannot establish resize
parity.

The pinned [torchvision backend implementation](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/image_processing_backends.py)
also fuses rescale/normalize: RGB mean/std 0.5 become 127.5 before the F32
subtraction/division. Match this order when comparing normalized patches.
Torch 2.10's [CPU resize implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp)
computes antialias coefficients in double, chooses an int16 coefficient precision
from the largest weight, and rounds/clamps each separable uint8 pass. A float
Catmull-Rom kernel alone does not reproduce those steps. Capture original RGB,
resized bytes, normalized packed patches and grids before diagnosing the vision
projection; retain independent up/downscale and aspect-ratio controls.

[Trained ARM run 38538187682](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38538187682)
at `ba44114c` passes **188 comparisons and 11 exact coordinate checks**: prefill,
cached/full-prefix and all eight generated continuation steps. The native cache
starts at 78 tokens; the derived offset is **−56**. All nine native greedy IDs,
including EOS 248046, match the official run. Vocabulary logits/scores retain
reported cosine **1.000000000**, maximum absolute error **9.22679901e-5**.
[Per-stage report](../tests/results/qwen35-ocr/native-positions-connected-parity.json)
keeps absolute errors, norms and the two final-normalization stages that exceed
the strict 0.002 floor and pass the unchanged official self-backend budget.
There is no tolerance change in this lane. The older oracle-position arm also
passes in the same run.

The independent weight-free [Linux run 38541059168](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38541059168)
at `9d7b3dee` passes **18/18 layouts** against original Transformers methods:
text-only, image-only, rectangular grids in both orientations, two/three images,
4090-token text and one-to-four-token cached blocks. All four prefill planes,
RoPE deltas and expanded cached coordinates match exactly (**max error0**).
Bounds, malformed/missing/unused grids, bad span lengths, temporal/video rejection,
negative offsets, output reset and `CRISPEMBED_QWEN35_OCR=0` controls pass.
[Independent report](../tests/results/qwen35-ocr/independent-positions-parity.json)
retains case geometry, offsets and native assertion lines. The initial
[Mac dispatch](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38538187763)
was cancelled after over 23 minutes without a runner, before any code executed.
This verifies position semantics, not multiple-image OCR quality.

A separate geometry difference exists before resampling: pinned Python
`smart_resize` uses ties-to-even `round`, while the shared native helper uses
`std::round` and clamps to at least one factor before the pixel-budget check.
With factor32 and this model's pixel limits, **80x1024 becomes64x1024 in Python
and96x1024 in the shared helper**. Python uses double for budget square roots;
the shared helper uses float. Add half-factor and extreme-aspect fixtures and
match original grids exactly in the new frontend, preserving existing engines'
preprocessing behind their current paths. This finding is a source comparison,
not a newly measured end-to-end resize result.


## Native RGB/BPE frontend lane — 2026-10-14

The default-off `CRISPEMBED_QWEN35_OCR` frontend now implements:

- CPU uint8 bicubic using double coefficients and Torch's per-axis int16 precision,
  signed coefficient rounding, horizontal/vertical uint8 rounding/clamping.
- Source `smart_resize` ties-to-even, double pixel-budget arithmetic, RGB .5
  fused normalization and still-frame P16/T2/M2 patch packing.
- GGUF BPE vocabulary/merges, typed added/special IDs and effective generation EOS.
  Added tokens split before NFC. Canonical ordering/composition includes algorithmic
  Hangul and source-pinned Unicode9 data under the existing Unicode license.
- The audited single-image user/assistant template, image-token expansion and
  modality labels, native positions/caches, and byte-level decoded text with
  replacement for incomplete/invalid UTF-8.

This frontend explicitly accepts only the immutable LightOnOCR-3-0.8B source
profile. It supports raw contiguous RGB, unpadded single-image prompts and the
current vision bound of1024 patches (262144 resized pixels). Larger default-profile
pages are rejected, without silently lowering the publisher's pixel budget.
A caller-supplied pixel budget can be compared with the same official override.
There is no file decoder/C ABI/CLI/binding/registry promotion in this lane.

The combining-mark pattern prototype remains available behind default-off
`CRISPEMBED_QWEN35_TOKENIZER_MARKS=1`; the checkpoint's serialized pattern is the
default. Preserve older shared Qwen/LFM pretokenizer behavior.

[Hosted frontend checks](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38551847357)
at `c9e08c18` pass all nine owned image layouts, eleven text cases, 5102 NFC
cases against the actual Rust normalizer, partial UTF-8 decoding and bounds/reset
controls. Resized RGB bytes, normalized patches, grids, IDs, templates and modality
labels match exactly. The existing 39 Qwen and 246 other pretokenizer checks pass.
The first [Unicode failure](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38547813683)
is retained: Unicode16 tables disagreed in112 cases with the serialized tokenizer's
Unicode9 normalizer. Correcting the source version removed those mismatches;
no comparison threshold changed. [Input evidence](../tests/results/qwen35-ocr/frontend-input-parity.json)
and [initial failure](../tests/results/qwen35-ocr/frontend-unicode-version-failure.json).

[Trained RGB-to-text checks](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38547814421)
match both owned fixtures' native IDs9/9 through EOS and decoded text exactly.
The native arm derives its own pixels, IDs, embeddings and positions; reference
fields are comparison targets. The unchanged256x256 fixture passes, but the
211x301 resized fixture fails the strict cache absolute gate: layer7 cached keys
have max error0.00201964378 against0.002, despite vocabulary-score cosine1 and
max error8.41617584e-5. This is a numeric failure, not a passing model-support gate.
[Full initial evidence](../tests/results/qwen35-ocr/frontend-initial-resized-failure.json).

[F32 key retention](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38550962287)
does not improve the failing cache (0.00202131271); isolated full-attention checks
pass. Both native/reference mean greedy confidences include EOS and are reported
as0.995937912/0.995937876 for the resized fixture. An independent
[original F32 CPU control](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38552512562)
has layer7 key variation0.00162005424, below the strict0.002 gate. Native attention
input traces show upstream error amplification; the control does not justify
calling the native result a strict pass. Retained evidence:
[key precision](../tests/results/qwen35-ocr/frontend-key-precision-audit.json),
[CPU/cache boundaries](../tests/results/qwen35-ocr/frontend-cpu-cache-audit.json).

The [F32 vision trial](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38554130487)
at `3dfe209e` retains52 vision matrices at F32 while leaving text matrices F16.
It passes both native raw-RGB fixtures and all unchanged comparison gates.
The resized layer7 keys improve from0.00201964378 to0.00183296204; vocabulary-score
cosine reports1, with worst score error8.17775726e-5. Both fixtures preserve all
9 IDs through EOS and exact decoded text. This retains the existing original
F32 self-backend calibration for vision/final norms; it does not imply bit equality
or that diagnostic attention-input traces satisfy an absolute0.002 limit.
[Complete comparisons, norms, budgets, IDs and confidence](../tests/results/qwen35-ocr/frontend-vision-f32-parity.json).

The export is1,918,139,328 bytes, versus1,717,205,952 for the current metadata-bearing
F16 baseline: an extra200,933,376 bytes (about11.7%). Keep this recipe explicit;
the converter's F16 default remains available with its measured resized failure.
API integration, larger pages, quantization, corpus quality, performance and HF
publication remain open. These two fixtures are implementation parity evidence,
not an OCR accuracy benchmark.

The earlier source `c8cdb38b` dispatches
[frontend](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38546388161) and
[trained](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38546388295) were
cancelled while queued after the serialized NFC discrepancy was found; they
produced no inference evidence.

Repository GPU-teardown guard also identified seven Qwen test entry points that
missed `core_util::clean_exit`; source `c4d8a634` adds those wrappers, with no model
math change. The static guard passes all163 one-shot binaries.


A second arithmetic control, default-off `CRISPEMBED_QWEN35_PRECISE_RMS=1`,
retains F32 squared inputs, scale, epsilon and zero-centered weights, but sums
squares in F64 for decoder and query/key RMS norms. It is an experiment, not
a source-exact reduction or a promoted recipe. Hosted
`qwen35_ocr_bootstrap.py --frontend-precise-rms` explicitly sets the gate to1;
other bootstrap modes explicitly set0. The original Python reference and all
comparison thresholds remain unchanged. The [hosted RMS control](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38556070004)
at `46368a4c` fails the resized fixture's unchanged layer7 key gate at0.00202226639.
Both texts/IDs still match exactly. This control remains default-off and unpromoted.
[Full precision-control evidence](../tests/results/qwen35-ocr/frontend-rms-precision-audit.json).

The reference dump and input checker now reject versions other than
Transformers5.5.4/Tokenizers0.22.2 before inference. The NFC source-version finding
requires pinning the tokenizer implementation as well as the checkpoint JSON.

[Latest Linux input check](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38556570758)
at `307b19e8` passes all nine images, eleven text cases,5102 NFC cases and285 legacy
checks with Tokenizers0.22.2 explicitly pinned and the latest C++ changes.
[Fresh pinned evidence](../tests/results/qwen35-ocr/frontend-pinned-input-parity.json).


To reproduce the passing experimental export (run conversion on a hosted runner):

```bash
python models/convert-qwen35-ocr-to-gguf.py \
  --revision 36d8636925327c671bd8e479c517a004f7566bd9 \
  --dtype f16 --precision-recipe vision-f32 \
  --output lightonocr-3-0.8b-vision-f32-experimental.gguf
```

The named recipe matches all52 retained tensors from the successful hosted
manifest exactly. Hosted `qwen35_ocr_bootstrap.py --frontend-vision-f32` uses the
same recipe. `--keep-f32` remains available for additional exact tensor names;
unknown names reject. This command does not register or publish the checkpoint.


The current-source [ARM recheck38597484044](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38597484044)
passes at `937ab2a0`:189 comparisons per owned fixture, all native inputs/IDs/EOS
and decoded text exact. It exercises the named shared recipe, explicit Tokenizers
pin and default-off RMS control together against the same original reference and
unchanged numerical gates. Resized keys remain0.00183296204; score max abs8.17775726e-5,
reported cosine1. [Current-source errors, norms, budgets and outputs](../tests/results/qwen35-ocr/frontend-current-source-parity.json).
The reproducible frontend lane is complete; the owning-context and production
integration lanes remain open.


## Owning context lane — 2026-10-14

The experimental `crispembed-qwen35-delta` target now includes
[`qwen35_ocr.h`](../src/qwen35_ocr.h) and its owning CPU engine. Initialization
checks the pinned source/format/tokenizer and all473 tensors' names, mapped shapes
and F32/F16 types. It owns backend buffers and the tokenizer, and frees weights
before the backend. Every recognition starts native vision, positions and fresh
recurrent/KV state; no reference arrays enter the engine.

The engine C interface provides init/free, bounded max-token configuration,
file/raw recognition, generated IDs, selected-token probabilities, mean confidence,
stop reason and an error string. Returned buffers remain valid until the next
recognition or free; failures clear prior text/IDs/confidence and zero `out_len`.
Independent contexts own independent state; one context is not reentrant.
Null prompt uses the audited empty OCR prompt. Grayscale is replicated to RGB;
RGBA drops alpha, matching the original PIL RGB conversion. The existing native
pixel/patch bounds still apply, without lowering the publisher's pixel budget.

Generation reports EOS, max-token truncation or native context exhaustion
separately. Confidence is stable whole-vocabulary softmax for each greedy selected
ID, including EOS. Returned IDs include EOS; decoded text skips source special
tokens. Invalid token settings reject without changing the previous setting.

[Hosted context check38600487132](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38600487132)
at `0a5efdc6` is queued. It uses the passing F32 vision/F16 text recipe, actual PNG
files, a separately compiled C header consumer and the pinned original Python
reference. Checks cover raw/file outputs, repeated calls, two simultaneously
loaded contexts, freeing one while using the other, grayscale/RGBA, token-limit
prefixes, IDs/EOS/text/confidence, missing weights/metadata and invalid-input,
error-reset/recovery and master=0 controls. No result is claimed before completion.
Long-context exhaustion, larger pages and arbitrary-file/corpus coverage remain
unverified. The standalone test supplies the image decoder normally provided by
`image_preprocess.cpp`; the engine is not yet wired into the main library.

Unified `crispembed_ocr_model_*`, orchestrator, CLI/server, bindings, registry,
quantization, model publication and a runtime release remain separate work.
