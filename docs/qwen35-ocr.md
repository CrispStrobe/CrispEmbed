# Qwen3.5 OCR: first checkpoint and native port blueprint

Started 2026-10-13 with **LightOnOCR-3-0.8B**, the smallest missing checkpoint
in the permissive OCR queue. The experimental gated port now has C API,
orchestrator, CLI/server and Python/Rust/Dart integration, plus bounded F16/Q8/Q4
comparisons. Four synthetic cases preserve IDs/EOS/text; Q4 logit drift is larger
than Q8. Registry aliases, broader page/corpus validation, performance and HF
publication remain pending. See the [measured precision sweep](#measured-precision-sweep)
for the current results and limits; the checkpoints below retain initial evidence.

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
   An owning engine context now passes load/free, file/raw-RGB, native generation,
   confidence, repeated/two-context and clear/reset checks. Next, wire this context
   into the unified public API and orchestrator/CLI/bindings.
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
The resized layer 7 keys improve from0.00201964378 to0.00183296204; vocabulary-score
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
checks the pinned source/format/tokenizer and all 473 tensors' names, mapped shapes
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
at `0a5efdc6` passes all eight context cases and the lifecycle/input controls. It uses
the passing F32 vision/F16 text recipe, actual PNG
files, a separately compiled C header consumer and the pinned original Python
reference. Checks cover raw/file outputs, repeated calls, two simultaneously
loaded contexts, freeing one while using the other, grayscale/RGBA, token-limit
prefixes, IDs/EOS/text/confidence, missing weights/metadata and invalid-input,
error-reset/recovery and master=0 controls. Full requests match all9 IDs through
EOS and17 decoded bytes; max per-token confidence error2.66547188e-7 under the
unchanged1e-6 gate. Token-limited prefixes preserve the original IDs/text and report
MAX_TOKENS. [Complete case/confidence/stop evidence](../tests/results/qwen35-ocr/context-lifecycle-parity.json).
Long-context exhaustion, larger pages and arbitrary-file/corpus coverage remain
unverified. The standalone test supplies the image decoder normally provided by
`image_preprocess.cpp`; the engine is not yet wired into the main library.

Unified `crispembed_ocr_model_*`, orchestrator, CLI/server, bindings, registry,
quantization, model publication and a runtime release remain separate work.


## Public integration lane — 2026-10-14

The default-off experimental context is now wired into the main library's
metadata-dispatched `crispembed_ocr_model_*` API. Raw RGB/grayscale/RGBA, float
[0,1] grayscale, prompt/reset, generation cap/default-reset and confidence use
the owning engine. Orchestrator engine20 is appended to the existing public
mapping; no shipped IDs move. Its stage forwards `vlm_prompt`, applies
`vlm_max_tokens` and restores the engine default when the stage cap is0.
The CLI/server recognize `qwen35-ocr` and `qwen35_ocr`; this engine requires an
explicit local GGUF rather than an unverified download alias.

Python `CrispOcrModel`, Rust `OcrModel` and Dart `CrispOcrModel` use the same
architecture dispatcher. Python adds `CrispOcrOrchestrator(stages=...)`, Dart adds
`CrispOcrOrchestrator.fromStages`, and Rust's existing `CrispOcrPipeline::from_stages`
selects engine20. Rust validates exact grayscale/RGB/RGBA buffer sizes before FFI;
Dart exposes raw channels, prompt/cap and copied token-confidence accessors.

Source review found incomplete legacy pipeline-parameter mirrors: Python72 bytes
and Dart's80-byte manual allocation versus the current native136-byte layout on
64-bit platforms. Both now use all20 native fields, with typed Dart `Struct`
layouts. This is a source-level initial finding, not a reproduced native crash.
[Initial audit](../tests/results/qwen35-ocr/public-initial-abi-layout-audit.json).
Hosted consumers compare cleanup/stage/params sizes and Python stage offsets
against the compiled C header before claiming ABI parity.

[Hosted public check38610331501](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38610331501)
at `815234d5` passes. It includes the existing full-stage checks, an independently
captured original custom-prompt reference and native full-stage comparison,
unified C ABI, Python/Rust/Dart consumers, CLI single/stage JSON and a loopback
server `/ocr/pipeline` request. It uses the explicit52-tensor F32 vision recipe;
no numeric gate changes. Earlier run38609791536 was cancelled while queued before
inference to add the default-cap reset control; it provides no execution evidence.

The custom-prompt native frontend/decoder passes189 stage comparisons and exact
9/9 IDs/EOS/17 decoded bytes. Generated-score reported minimum cosine1.0,
maximum absolute error8.63075256e-5; cache-key maximum0.00198709965 remains
under the unchanged0.002 limit, with little headroom on this fixture. Unified C
confidence maximum error2.64690611e-7; the custom-prompt error is6.83976417e-8,
under1e-6. Prompt/reset, cap0-to-default2048, truncation, NaN-gray rejection and
recovery, and master0 rejection pass. No tolerance was relaxed.

The actual compiled ABI gives stage168 bytes, cleanup60, params136; Python
matches all three and the cap/prompt offsets136/144, and Dart matches all three
sizes. Python/Rust/Dart each pass the model and explicit-stage consumers. Both
CLI JSON routes and HTTP `/ocr/pipeline` preserve the exact17-byte transcript;
HTTP returns one region and rounded mean confidence0.9956. Dart analysis and
execution use SDK3.13.5; only the changed binding classes take its formatter
output. [Full public evidence and prompt stage errors/norms](../tests/results/qwen35-ocr/public-integration-parity.json).

Experimental usage after a verified build:

```sh
CRISPEMBED_QWEN35_OCR=1 crispembed -m model.gguf --ocr input.png --ocr-max-tokens 12
CRISPEMBED_QWEN35_OCR=1 crispembed --ocr-pipeline input.png \
  --ocr-engine qwen35-ocr --ocr-rec model.gguf --ocr-max-tokens 12
```

```python
from crispembed import CrispOcrModel, CrispOcrOrchestrator
ocr = CrispOcrModel("model.gguf")  # set CRISPEMBED_QWEN35_OCR=1 in the environment
ocr.set_max_tokens(12)
text = ocr.recognize("input.png")
orch = CrispOcrOrchestrator(router=False, stages=[
    {"engine": 20, "model_a": "model.gguf", "vlm_max_tokens": 12},
])
text = orch.run("input.png")["text"]
```

This integration checkpoint covered F32/F16 weights and the1024-patch bound.
The subsequent quantization lane adds Q8/Q4 matrices behind a separate gate;
the source-profile and page bounds remain.
Model registry, quantization, broader page/corpus coverage, HF publication and
runtime release are not part of this integration checkpoint. CrispSorter provider
and catalog selection remain a TODO once a validated model artifact is published;
that application is outside this checkout. The pinned app release remains separate.

## Native quantization calibration lane — 2026-10-14

The owning engine and shared embedding/projection helpers now have a separate
value-parsed default-off `CRISPEMBED_QWEN35_QUANT` gate for contiguous Q8_0/Q4_K
matrices. The master OCR gate is still required. Small norms, biases, recurrent
vectors and convolution kernels remain floating point. Quantized matrices and
the tied embedding/head are materialized as F32 for the existing CPU arithmetic;
this is compressed storage, not a reduced-RAM or optimized quantized-matmul claim.

The hosted sweep runs `crispembed-quantize` on the calibrated `vision-f32` export,
with `--tensor-type 'v.*=f32'` and explicit matrix overrides. It retains Q8 and Q4
controls, then compares tied head F16, recurrent A/B projection F16, and attention
Q8/F16 variants. Every export audits all 473 logical shapes/types; vision and
non-matrix bytes must remain exact. Effective stored matrix types must match the
recipe. Converting an already rounded F16 vision tensor back to F32 does not
recover the calibrated source weights; start from the named converter recipe.

The quantization harness reports `crispembed-diff` layer/cache errors, norms and
whole-vocabulary scores without replacing the strict F16 comparator. Teacher-forced
score comparisons are explicitly diagnostic and use the original Python prefix;
a separate owning-context call generates freely from real image/prompt/GGUF
weights. IDs/EOS/text and paired CER/WER are recorded, including mismatches.
The original/resized images, custom prompt and a new owned invoice image are
included; literal0 quant-gate initialization rejection is tested per export.
The strict comparator's capture length bound is extended from12 to64 to permit
longer fixtures; all numerical thresholds and calibration logic stay unchanged.
Any strict F16 failure on a new fixture is retained and prevents a production
parity claim. [Hosted quant sweep38614951999](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38614951999)
passes at `6b970ea8`. Larger pages, a corpus accuracy claim,
registry hashes and HF publication remain open.

### Measured precision sweep

All seven quantized exports plus the F16 baseline preserve all41 generated IDs
across four owned cases, including each EOS, exact decoded text and paired CER/WER0.
All teacher-forced argmaxes also match. The new invoice has14 generated IDs;
its194 strict stage comparisons and the custom prompt's189 comparisons pass
unchanged F16 numerical gates. Overall2792 diagnostic vectors are retained.
Every export passes the473-tensor logical-shape/type and critical-byte audit,
finite-inference and literal0 quant-gate controls.

| Recipe (all keep F32 vision) | Decimal GB | Minimum generated-score cosine | Maximum score absolute error | Exact IDs/EOS/text cases |
|---|---:|---:|---:|---:|
| F16 baseline | 1.918139 | 1.000000000 | 0.00009465 | 4/4 |
| Q8 | 1.213238 | 0.999463260 | 0.379626 | 4/4 |
| Q4 + Q8 tied head | 0.964431 | 0.949601591 | 4.313274 | 4/4 |
| Q8 + F16 tied head | 1.451625 | 0.999561191 | 0.334938 | 4/4 |
| Q4 + F16 tied head | 1.202818 | 0.949943483 | 4.329383 | 4/4 |
| Q4 + F16 tied head/A/B controls | 1.203666 | 0.949801862 | 4.364334 | 4/4 |
| Q4 FFN + Q8 attention/F16 head/controls | 1.320058 | 0.977389455 | 3.220321 | 4/4 |
| Q4 FFN + F16 attention/head/controls | 1.538293 | 0.977991045 | 3.215855 | 4/4 |

The cheapest tested Q4 file is964,430,848 bytes, with186 Q4_K matrices, one Q8
embedding/tied head and286 F32 tensors. Q8 is1,213,238,272 bytes, with187 Q8 matrices
and286 F32 tensors. The head-F16 restoration costs238,387,200 additional bytes and
barely improves Q4 score cosine. Keeping A/B controls F16 adds847,872 bytes and
slightly worsens the measured worst cosine here. Higher attention precision
improves cosine, but both tested attention recipes are larger and less faithful
in logits than the all-Q8 text export. The remaining Q4 FFN error is already
visible at decoder layer0 even when attention/head/controls are F16; it compounds
through24 layers. These findings support testing FFN precision/importance rather
than spending more on the tied head or all attention.

Quantized logit errors exceed the F16 tolerance. Matching these short transcripts
is a bounded quality result. Q8 is the stronger candidate for broader validation;
Q4 is the smallest tested synthetic-output match. Gold raw CER includes the
reference's Markdown two-space line breaks (2/15 on the Hello fixtures,2/24 on
the invoice); gold WER is0. This is explicitly distinct from paired CER/WER0.
[Every tensor audit, layer/cache error/norm, teacher score and native output](../tests/results/qwen35-ocr/quant-calibration-parity.json)
is retained; no thresholds were loosened and no weight payload was downloaded
from the hosted artifact for this handoff.

Experimental quantized usage requires both gates:

```sh
CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_QWEN35_QUANT=1 \
  crispembed -m model-q4.gguf --ocr input.png --ocr-max-tokens 64
```

The following FFN sweep extends this evidence. Preserve the Q8 candidate,
expand page/corpus coverage and evaluate row-wise F32 matmul memory/performance
before publication. The current scope remains1024 patches, compressed storage
with F32 materialization, and four owned synthetic cases. No registry alias,
HF publication, runtime release, latency or corpus-accuracy claim.

## Native FFN importance/precision lane — 2026-10-14

The hosted sweep keeps the Q8 tied head and calibrated F32 vision. It adds
Q6_K matrix loading under the existing quant gate and tests FFN down-projection
Q8, all-FFN Q6/Q8, compact Q8 attention without F16-head overhead, and native
importance-weighted Q4. The preceding seven-recipe evidence remains unchanged.

A synchronous observer local to the calling thread is default-off behind
`CRISPEMBED_QWEN35_IMATRIX`. The hosted collector registers it for six disjoint
owned calibration images and records actual native F16 FFN input sums of squares
for all72 gate/up/down matrices. Image and generated-token inputs are native;
no Python reference activations or teacher-forced IDs feed calibration. Both
literal0 registration and runtime bypass are checked, and the first image's
IDs/text must match with collection off/on. Clearing the observer is always
allowed. Normal inference has no observer or statistics file writes.

The output uses the existing CrispEmbed imatrix GGUF format (F32 sum-square
vectors plus UINT64 row counts). The quantizer must report all72 FFN vectors
applied. Calibration image hashes/texts, dimensions/counts/statistics hashes and
native outputs are recorded, with no image/text overlap against the four held-out
synthetic evaluation fixtures. All use the same renderer family, so this does
not establish corpus generalization. Export file SHA-256 is now recorded too.
Existing F16 numeric gates and all prior runtime alternatives remain unchanged.
[Hosted FFN sweep38627759408](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38627759408)
passed at `af8a7967f2d7af2c4e4c47ba06ea00a53e1fe717`, with lint and secret scan
also passing. Collection covers511 native activation rows per matrix; all six
calibration examples reach EOS. The importance GGUF is549,440 bytes, SHA-256
`223fbfbe1a4ea52114cfc7ca0d387829ac7a2013ac2cb048146122b0f337d8e5`.

All ten exports pass all41 generated IDs/EOS/text on the same four evaluation
cases, paired CER/WER0 and all teacher-forced argmaxes equal. The sweep retains
3490 diagnostic vectors and473 tensor audits per export. Both new fixtures pass
the unchanged strict F16 stage gates. These are synthetic quality checks;
quantized scores still exceed F16 tolerances.

| Recipe (all keep F32 vision and Q8 tied head) | Actual bytes | Minimum generated-score cosine | Maximum score absolute error |
|---|---:|---:|---:|
| F16 text baseline (F16 tied head) | 1,918,139,328 | 1.000000000 | 0.00009465 |
| Q8 text | 1,213,238,272 | 0.999463260 | 0.379626 |
| Q4 text | 964,430,848 | 0.949601591 | 4.313274 |
| Q4 text + Q8 FFN down | 1,008,471,040 | 0.954575598 | 4.491977 |
| Q4 text + Q6 FFNs | 1,032,555,520 | 0.962701976 | 3.451684 |
| Q4 text + Q8 FFNs | 1,096,551,424 | 0.962654233 | 3.338670 |
| Q4 FFNs + Q8 attention/F16 A/B | 1,081,670,656 | 0.977639556 | 3.252047 |
| Q6 FFNs + Q8 attention/F16 A/B | 1,149,795,328 | 0.998158216 | 0.887069 |
| Q4 text + native FFN imatrix | 964,430,848 | 0.949486971 | 5.220582 |
| Q4 FFNs + Q8 attention/F16 A/B + native FFN imatrix | 1,081,670,656 | 0.980404854 | 3.001625 |

The Q8-attention/Q6-FFN candidate saves63,442,944 bytes (5.23%) against Q8,
with lower score fidelity. Its effective types are79 Q8_0 matrices,72 Q6_K FFN
matrices,36 F16 A/B controls and286 F32 tensors; no Q4 tensors remain despite
the sweep arm's historical `q4-attention-q8-ffn-q6` name. Q8 remains the stronger
score-fidelity control. Neither candidate is selected for production.

Selective FFN restoration alone leaves attention drift. Q8 down improves worst
cosine but worsens maximum score error; all-FFN Q8 adds more bytes than Q6 without
improving the worst cosine here. Native importance slightly worsens plain Q4
and its maximum error, while improving the Q8-attention arm at unchanged size.
Calibration is recipe-dependent; these six examples do not establish a general
improvement. All negative results remain in the
[full report](../tests/results/qwen35-ocr/ffn-calibration-parity.json), including
export SHA-256, effective tensor types, layer/cache errors/norms, scores, complete
native outputs and calibration provenance.

To reproduce the mixed candidate from the calibrated F16-text/F32-vision source:

```sh
crispembed-quantize model-vision-f32.gguf model-attention-q8-ffn-q6.gguf q4_k \
  --tensor-type 'v.*=f32' \
  --tensor-type 'l.layers.*.self_attn.*.weight=q8_0' \
  --tensor-type 'l.layers.*.linear_attn.in_proj_qkv.weight=q8_0' \
  --tensor-type 'l.layers.*.linear_attn.in_proj_z.weight=q8_0' \
  --tensor-type 'l.layers.*.linear_attn.out_proj.weight=q8_0' \
  --tensor-type 'l.layers.*.linear_attn.in_proj_a.weight=f16' \
  --tensor-type 'l.layers.*.linear_attn.in_proj_b.weight=f16' \
  --tensor-type 'l.layers.*.mlp.*_proj.weight=q6_k'
```

Next: broader independent page/corpus coverage for Q8 and the Q8/Q6 candidate,
then row-wise F32 matmul memory/performance A/B. The1024-patch bound and full F32
weight materialization remain. No HF publication, registry alias, runtime release,
latency or peak-memory claim follows from this sweep.

## Fixed real-document panel — 2026-10-15

The hosted `--pages` mode compares the unchanged F16-text/F32-vision recipe,
Q8 text, and Q8 attention/Q6 FFNs. Inputs come from the repository's checksummed
CC0/public-domain catalog: the Arabic printed line, complete form, complete
receipt explicitly resized to448×576, and a German page header cropped at
`[0,0,1920,650]` then explicitly resized to768×256. These are evaluation image
derivatives, not changes to production preprocessing or its1024-patch limit.
The original full-size documents remain unchanged.
This is an external-document panel including rendered receipt/form artifacts and
a historical scan, not four independent camera scans or a scored corpus.

The compact reference calls the pinned original Transformers5.5.4 F32 eager
model and processor. Every captured array must equal the full original capture
on the existing anchor before page evaluation. It stores the77 prefill diagnostic
boundaries plus official generated scores, IDs and text. The native probe uses
independent image/prompt/BPE/free generation; teacher-forced scores stay separate.
Explicit per-case caps64/192/384/128 retain EOS versus max-token stops. Matching
truncated outputs are not complete-document parity. Empty decoded output remains
distinguishable from failure. The existing strict F16 anchor checks are unchanged.

Reports retain all 473 effective tensor types/shapes, protected-byte checks,
export hashes, layer/cache magnitudes/errors, native outputs, confidence, lines,
timing scopes and paired CER/WER. Existing receipt/form annotations additionally
provide raw gold CER/WER; form ambiguity and annotation confidence remain visible.
No crop or Arabic annotation is guessed. Reference agreement is separate from
accuracy against annotations. Timings are single-run observations: Python
generation excludes processor/model load and collects scores; native recognition
includes image decoding/preprocessing, excludes context load and teacher probes.
They do not establish a latency benchmark.

[Hosted panel38674281196](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38674281196)
completed at `d6c8941662e84fcb410de7838036afc7ba971e9b`; lint and secret scan pass.
All 90 shared compact anchor arrays equal the original full capture exactly; the
compact file has91 arrays including its additional decoded-length field. The panel
retains three exports,12 native cases,2586 stage vectors and554 teacher-forced
score vectors per export. All 473 tensor audits and literal0 controls pass; export
SHA-256 matches the preceding calibration run for all three files.

| Recipe | Exact IDs/text/stop cases | Exact EOS-completed cases (of3) | Min score cosine | Max score absolute error |
|---|---:|---:|---:|---:|
| F16 text/F32 vision | 4/4 | 3/3 | 1.000000000 | 0.00144529 |
| Q8 text/F32 vision | 3/4 | 2/3 | 0.998763442 | 0.680237 |
| Q8 attention/Q6 FFNs/F32 vision | 2/4 | 1/3 | 0.994996905 | 1.935345 |

The receipt produces384 tokens and stops at the cap in Python and all native
arms, ending inside an HTML closing tag. Its prefix matches exactly, including
stop reason, but the receipt is incomplete. The other Python outputs reach EOS:
Arabic5 IDs, form148 IDs and German header17 IDs. Resolved patch counts are
272/560/1008/768 respectively; the original processor upsizes the small Arabic
line according to its minimum-pixel policy.

Q8 first diverges on the form at generated step75, with paired CER0.117431 and
WER0.133333; there are two teacher-forced argmax differences. Q8/Q6 diverges on
the form at step10 (CER0.526606/WER0.733333), and on German at step8
(CER0.068966/WER0.166667); six teacher argmax differences in total. These are
zero-based steps. Native free generation confirms the differences independently.
Both quantized arms first show storage-related drift at decoder layer0; the F32
vision merger is identical across recipes. High score cosine did not preserve
these outputs. The earlier short synthetic matches did not establish this parity.

F16 preserves the decoded outputs but exceeds the prior0.002 cache-key error
budget on form0.00295281, receipt0.00264263 and German0.00622153 (Arabic0.00041670).
Some receipt/German cached values also exceed0.002. Final-norm raw error reaches
0.0147171; the new compact captures lack official backend-off controls for
assessing the existing qualified norm budget.
These are diagnostic measurements, **not new strict numerical-parity passes**.
The original strict anchor gates still pass unchanged. Isolate F16 text rounding
and original CPU backend variation on the larger inputs before any numerical
promotion; do not enlarge the budget to accommodate these results.

The upstream form output is a figure description with a label list, not a literal
field transcription; F16 reproduces it exactly. Raw annotation CER/WER is
1.676113/2.027778 for Python/F16 and1.651822/2.000000 for Q8.
Q8/Q6 gives1.975709/2.361111. The truncated receipt's raw annotation
CER/WER is1.275184/0.842857 for every arm. Insertions, HTML/figure markup, output
intent and truncation dominate these raw comparisons with plain-text annotations;
insertion errors can make raw CER/WER exceed1.
They are retained evidence, not an OCRBench or corpus-accuracy score. Quantized
reference disagreement alone also does not prove worse annotation accuracy.

Single native recognize-file observations (context loading excluded) were:

| Input | Python generate seconds | F16 native seconds | Q8 native seconds | Q8/Q6 native seconds |
|---|---:|---:|---:|---:|
| Arabic line | 1.326 | 9.998 | 7.095 | 7.996 |
| Form | 12.201 | 225.124 | 124.466 | 172.131 |
| Receipt (384-token prefix) | 31.092 | 578.030 | 351.114 | 411.234 |
| German header | 3.816 | 37.505 | 27.872 | 29.841 |

The timing scopes differ as described above; form/German token counts also differ
between recipes. These are not controlled speed ratios. The material native
latency gap requires profiling and gated graph/F32-materialization work. Q8/Q6
is smaller but slower than Q8 in these observations and loses more output parity.

[All outputs, annotations, errors/norms, stop states, confidence and timings](../tests/results/qwen35-ocr/page-parity.json)
are retained. Neither quantized recipe is promoted. Next: isolate larger-input
cache drift with unrounded text weights/official CPU controls, complete a receipt
under an explicit adequate cap, and profile the native hot path before publication.
No corpus score, registry alias, HF publication or runtime release follows.

## Larger-input precision and original backend controls

[Hosted38698415749](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38698415749)
at `263d6abe67fd53049986dff8912437e2b904305c` separates text-weight precision,
RMS reduction and original CPU convolution variation on the same form, receipt
and German derivatives. All source/input hashes match the preceding panel;
90 shared compact/full original anchor arrays remain bitwise exact. The reference
is original Transformers5.5.4 F32 eager. HF metadata for the pinned checkpoint
reports 852,985,920 BF16 parameters: F32 here expands the original stored values.
No higher-precision upstream checkpoint is invented.

The native `--prefill-only` diagnostic takes real image/prompt/tokenizer inputs
and runs the native vision/decoder. It measures 77 stages and applies the unchanged
strict comparator at each boundary. A successful experiment means all finite,
shape, capture and report checks completed; **it does not mean numerical parity
passed**. The report retains every failed stage and budget ratio. No reference
array enters the production inference path. No full-generation, EOS, corpus
accuracy or speed claim is added by this probe.

| Arm | Storage GB (decimal) | Form max key error | Receipt max key error | German max key error | Strict failed stages (form/receipt/German) |
|---|---:|---:|---:|---:|---|
| F16 text/F32 vision | 1.918 | 0.00295281 | 0.00264263 | 0.00622153 | 5 /7 /8 |
| All-F32 | 3.422 | 0.00295234 | 0.00264335 | 0.00625157 | 5 /7 /8 |
| All-F32 + precise RMS | 3.422 | 0.00295186 | 0.00264016 | 0.00619769 | 5 /7 /8 |

The F16 baseline is byte-identical to the preceding panel. The all-F32 export is
3,421,928,384 bytes, SHA-256
`cab4631dbea7aa08512ed9473a72e695add5ea51f0a3d652939fda0212cbcc48`.
All 473 names/shapes are checked for every arm; vision, vectors and convolutions
keep identical bytes. The F16 arm has 187 F16 and 286 F32 tensors; the other two
have 473 F32 tensors. More text precision and the existing F64-sum-of-F32-squares
RMS alternative do not resolve this drift. No precision recipe is promoted.

Pixels match exactly. The first measured native difference is the vision merger:
maximum errors 0.000139356/0.000193000/0.0000952482 for form/receipt/German, with
norms 47.346/69.233/55.774. Input embeddings inherit those errors. In the F16 arm,
all decoder hidden layers stay under0.002 (maxima 0.00126052/0.00087929/0.00191617).
The first strict cache failure is layer 11 keys for form and layer 7 keys for the
other two; later keys and some receipt/German values also exceed0.002. This
locates measured boundaries, not yet the first responsible operator.
All first-token argmaxes match across nine native cases; maximum prefill-logit
error is 0.000196576 across all arms. Stage cosines remain above0.99999, while
absolute budgets still fail. All norms and errors are retained, not just cosine.

Original Python backend-off versus default variation reaches key errors
0.00212538/0.00320449/0.00398135 and final-norm errors
0.00990486/0.01163006/0.00962448. Per-patch original convolution is identical to
default at all 76 captured boundaries on each input. These controls measure
original implementation variability; they do not relax the cache budget.
The existing elementwise qualified final-norm gate passes receipt (maximum budget
ratio 0.65814) but fails form 1.01615 and German 1.17035 in the F16 arm. Using a
stage-wide norm allowance would conceal those failures and is not accepted.

The report contains 693 native metric/budget vectors, 456 original control vectors,
all 60 strict failures and the unchanged anchor/component/context controls.
Next isolate vision operators and decoder projection/normalization/RoPE with
identical inputs in test-only probes, then measure runtime costs. The0.002 cache
budget, runtime math/defaults and prior full-output evidence remain unchanged.
[Complete retained report](../tests/results/qwen35-ocr/prefill-drift.json).

## Same-input decoder isolation

[Hosted38702930399](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38702930399)
at `799bfefe26e97f46ae3a5ec859801e935a6afd23` keeps the same three inputs,
original captures, tensor audits and unchanged comparison gates. It repeats the
composed F16 baseline and adds F16/all-F32 decoder arms with the original full
input-embedding vector supplied **only inside the diagnostic test executable**.
Native vision still runs and is measured independently. The explicit
`--prefill-reference-embeddings` flag and `ISOLATION` output distinguish these
arms; production inference and public bindings do not accept reference arrays.

| Arm | Max key error | Max final-norm error | Failed stages (of 231) |
|---|---:|---:|---:|
| Composed native F16 text/F32 vision | 0.00622153 | 0.0147171 | 20 |
| F16 decoder, original input embeddings (test-only) | 0.0000406504 | 0.000167608 | 0 |
| All-F32 decoder, original input embeddings (test-only) | 0.0000371933 | 0.000216484 | 0 |

The baseline reproduces every scalar stage metric from the preceding experiment.
Native vision merger metrics are identical across all arms. The isolated decoder
inputs are exact; all isolated stage errors pass the raw 0.002 absolute gate,
including final norms without needing the qualified backend-variation allowance.
Minimum stage cosines are 0.999999821 and 0.999999881 for the two isolated arms.
All nine first-token argmaxes match. The report retains 693 native metric/budget
vectors, 456 original CPU-control vectors, all 473 tensor records per arm and every
baseline failure.

These controls localize the larger-input failures to differences entering the
decoder, with the first measured difference already present in the native vision
merger. The preceding all-F32 composed control did not remove them. Prioritize
native vision patch/projection, position, norm, attention/FFN and merger operator
isolation before changing decoder arithmetic or text precision. The exact
responsible vision operator is still unknown. The composed native baseline still
fails its strict gates: these are conditional component passes, not a replacement
frontend, full decoding, recognition benchmark, speed result or publication gate.
[Complete retained measurements and scope](../tests/results/qwen35-ocr/prefill-isolation.json).

## Vision operator audit contract

`tools/dump_qwen35_ocr_page_reference.py` can observe every original vision block
and its operator inputs without rebuilding the reference arithmetic. The hosted
`--vision-operators` lane first checks the unchanged compact anchor and all shared
expanded arrays against the full original dumper. It then retains original
backend-off/per-patch controls on the same three document derivatives.

`test-qwen35-vision-ops` uses `crispembed_diff::Ref` and the unchanged qualified
`qwen35_compare` for 77 composed native vision boundaries and 101
same-input operator calls spanning all twelve blocks. The production vision
source is byte-for-byte unchanged and SHA-checked. The existing gated projection
observer captures 51 actual projection inputs, exposing norm, attention-pooling
and activation boundaries across all twelve blocks. Existing block5/11 captures
are checked for exact equality, and an observer-off rerun checks unchanged outputs.
Replay code includes that
implementation in a separate test translation unit, renaming its public entry
points to keep the ordinary linked implementation in its original compilation
context. Only the test target defines
`CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT`. No replay code or original arrays enter
production inference. Scalar activation replay has a separate compilation context;
composed native measurements and the existing golden guards remain authoritative.

The sweep retains baseline, existing precise-norm, precise-attention and combined
modes, plus the existing 77-stage composed decoder measurements for each. Reports
retain finite/shape/control assertions, all errors/norms, raw absolute failures
and unchanged qualified verdicts. Failed comparisons make the test executable
exit nonzero; the driver retains all arms before failing the overall audit when
qualified failures remain. Experiment completion does not establish
composed numerical parity, full decoding, corpus accuracy or a speed result.


## Completed larger-input vision operator localization

[38719205543](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38719205543)
at `8418621b` completes the contract above:3060 stage vectors and1302 original
CPU-control vectors across three derivatives and four existing precision arms.
All126 shared full-anchor arrays are exact;106 additional original-hook arrays
have the expected schema. The production source is SHA-checked,51 projection
inputs are observed per call, observer-off native outputs are exact, and all231
previous baseline scalar prefill measurements reproduce exactly. All old
small-fixture component/frontend/context guards pass again. Restoring the entire
production translation unit and isolating replay recovers the baseline; the
specific compiler transformation behind the initial instrumentation regression
has not been established.

| Arm | Form failed prefill stages | Receipt | German | Worst key error |
|---|---:|---:|---:|---:|
| Baseline | 5 | 7 | 8 | 0.00622153 |
| Precise norm | 1 | 7 | 5 | 0.00534558 |
| Precise attention | 1 | 16 | 6 | 0.00410581 |
| Both | 0 | 7 | 4 | 0.00408316 |

These use the unchanged qualified comparator: cache/hidden/logit absolute gates
remain0.002; only the existing vision/final-norm backend qualification applies.
Both gates pass all77 form checks, but raw final-norm error is0.00835323 and is
qualified by the original backend control. Receipt/German still fail strict cache
gates. All12 first argmaxes match. The driver finishes every arm, retains67
failed comparisons and then exits nonzero. No precision default is promoted.

Same-input operators report cosine1.000000000. Of303 per arm,302 satisfy raw0.002;
form block11 FFN down reaches0.00219727. Precise norm reduces worst local norm
error from0.000183105 to0.0000305176. In composed baseline inference, patch
projection differs by3.58e-6–4.81e-6, and raw0.002 is first exceeded at block8
norm2 for form/receipt and block8 norm1 for German. Composed block11 FFN-down
errors reach0.136841/0.443115/0.956604, versus isolated0.002197/0.001953/0.001709.
All vision comparisons still pass existing backend-qualified budgets; the raw
violations remain recorded separately and cannot substitute for strict decoder
parity. Scalar activation replay retains its separate compilation-context limit.

This narrows follow-up to complete-block/prefix replay and pre-norm residual/FFN
amplification. Establish replay agreement on ordinary native block inputs before
interpreting original-input block controls. The data does not identify one faulty
operator or prove that higher precision alone solves the port. There is no new
full decoding, EOS, recognition-quality, latency or publication result.
[Complete retained data and original failures](../tests/results/qwen35-ocr/vision-operators.json);
[rejected initial instrumentation](../tests/results/qwen35-ocr/vision-operators-initial.json).


## Native block and suffix replay contract

The hosted `--vision-blocks` audit links the unchanged production vision engine;
it does not include a private implementation copy. A validation-only identity
patch matrix and zero patch bias/position table deliver a captured block input.
The prepared embedding must equal that input exactly. Each of the twelve blocks
must then reproduce ordinary native block output and four observed projection
inputs exactly before any original-input block measurements are interpreted.
Observer-off output and residual-sum controls also remain mandatory.

Pre-norm residuals are reconstructed from the public native attention projection
and float addition; adding the independently replayed native FFN-down projection
must recover the ordinary block output exactly. Original expected values still
come from the unchanged Python dumper/hooks and `crispembed-diff`. The identity
frontend, selected layer lists and original inputs are confined to the test binary.
Literal0/master-off and malformed-input controls reject replay before inference.

Each case compares111 composed boundaries,108 original-block-input boundaries,
and47 suffix boundaries starting at blocks0/4/8/10/11, then77 ordinary composed
decoder stages. Baseline and the existing combined precision gates are retained;
the other two precision arms remain in the preceding operator report. Native
rotary geometry remains in use and is explicitly compared with the original;
this is block-input isolation, not exact original-geometry isolation. No parity
budget, production source, runtime default or publication status changes. The
report is checkpointed per completed case; failures remain nonzero after all
valid arms have been collected.


## Completed native block and suffix replay

[38726666455](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38726666455)
at `7437aed1` completes2058 stage vectors and678 exact controls. Every identity
frontend, selected native block output, observed projection input and reconstructed
residual sum agrees with ordinary native execution. All462 prior prefill records
and402 shared vision records reproduce exactly, including the prior direct
block5/11 attention/down scalar captures. Reference anchors retain126 exact shared
arrays and106 expected additions;1302 original CPU-control vectors are retained.

| Baseline input | Ordinary layer11 max error | Original-input block11 | Original-input suffix8 layer11 | Original-input suffix10 layer11 |
|---|---:|---:|---:|---:|
| Form | 0.136475 | 0.007202 | 0.013855 | 0.006836 |
| Receipt | 0.441040 | 0.005859 | 0.065247 | 0.008545 |
| German header | 0.954773 | 0.006104 | 0.017822 | 0.007690 |

All independent block0–10 outputs meet raw0.002 in both arms; block11 outputs
still fail in all six cases. With original block11 input, baseline pre-norm2
residual error is only2.19e-5–2.86e-5, growing through norm2/activation/down to
0.00586–0.00720. This separates local last-block amplification from the much larger
upstream contribution, without identifying one structural defect. Original-input
block cosine reports1.000000000 despite those absolute failures.

Earlier resets can worsen the result. Baseline receipt suffix4 produces layer11
error1.116821 and German suffix0 gives1.288086; both fail qualified vision gates.
All outcomes remain in the report. The unchanged ordinary decoder still fails20
baseline /11 combined-precision stages. All six first argmaxes match, and all old
small-fixture frontend/context/decoded-output guards pass. The complete audit
correctly exits nonzero with33 verdict failures; no default or threshold changes.
Native rotary values remain in use, with original differences up to5.96e-8.
No full-document, EOS, speed, memory, corpus or publication claim follows.
[Every measurement, control and retained failure](../tests/results/qwen35-ocr/vision-blocks.json).

The next kernel audit should observe original mean/rstd via native_layer_norm,
first checking its output against the existing LayerNorm hook. The pinned
[PyTorch2.10 F32 LayerNorm kernel](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp)
uses the same centered affine ordering as the native path, while
[RowwiseMoments](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/moments_utils.h)
uses vector Welford/cascade reductions instead of the native sequential two-pass
statistics. Record dispatch/vector width rather than assuming one architecture.
This source difference is a testable numerical hypothesis, not an established
cause. Evaluate original Python blocks with native rotary values as an additional
conditional control before attributing replay differences to normalization or
attention arithmetic. Keep all original references and composed gates unchanged.

## Mean/rstd and original-Python/native-geometry audit contract

`--norm-statistics` extends original module hooks with the original
`aten.native_layer_norm` output/mean/rstd. Its output must equal each module hook
exactly before mean/rstd are saved. Existing shared anchor arrays stay unchanged;
statistics add50 arrays, giving267 prefill boundaries per original CPU control.
CPU capability and the Torch source revision are recorded. Python does not expose
the reference vector width here; candidate widths4/8/16 are tested explicitly.

The linked production vision path stops through the existing observer immediately
after LayerNorm, before QKV projection. A test-only identity frontend supplies
each captured norm1/norm2/merger input. Reconstructed two-pass F32/F64 statistics
must reproduce those actual normalized outputs exactly; all input and output
controls are retained, and failed reconstruction prevents interpreting a candidate
as a validated native correction. Welford/cascade candidates are validation-only
scalar-lane adaptations of the pinned PyTorch kernel, with its BSD notice retained.
Original-moment fused/split affine controls separate reduction error from final
multiply-add rounding; split-affine Welford variants are retained as well. These
original statistics are conditional test inputs only. The current eleven-mode
harness expects825 output/mean/rstd vectors and12 geometry-control vectors via
`crispembed-diff` and unchanged qualified verdicts, plus77 ordinary prefill stages.

Native rotary arrays are generated by the unchanged engine, with exact zero-input
versus alternating-input independence controls. Original Python blocks first
reproduce their original hook outputs exactly, then run with the native rotary
values as a separate conditional reference control. These controls isolate
geometry without changing the primary reference. Ordinary native prefill and
old small-fixture decoded guards remain required. All negative results stay
nonzero after collection; no production reduction, threshold, default or
publication change follows merely from completing the audit.

## Completed original statistics and geometry audit

[Hosted38739442000](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38739442000)
at `c55832bc` completed 2,517 comparisons and 1,602 original CPU-control vectors.
All 126 shared anchor arrays and 231 prior baseline prefill records remain exact;
all earlier small-fixture guards pass. Original native_layer_norm outputs equal
250 original module captures exactly. All 36 original block replays and native
rotary input-independence controls pass. The complete report retains every raw
record, including 75 failed F32 native reconstruction controls (maximum 0.0000305).
All 75 F64 reconstructions and all 150 delivered-input controls are exact.
The F32 reconstruction is therefore not validated as the native baseline.

The four-lane Welford candidate computes all 75 original mean/rstd/output triples
exactly. Eight- and sixteen-lane candidates do not. Original-moment fused affine
controls match all 75 outputs, while split affine differs by up to 0.00000763;
these original moments remain test-only and exact statistics there are by
construction. Torch reports CPU capability DEFAULT, not an exposed vector width;
these results establish the tested candidate's agreement on this runner only.

Original Python blocks with native rotary values differ by up to 0.00109863
(form), 0.000732422 (receipt/German), despite rotary differences below 5.97e-8.
All these comparisons satisfy existing budgets. Ordinary prefill retains all
20 failures and exact first argmaxes. The job correctly exits nonzero for those
20 verdicts plus 75 invalidating reconstruction controls. No runtime/default,
threshold, decoded-quality or publication improvement is claimed. Next repair
and validate F32 reconstruction, then test computed four-lane normalization in
an opt-in runtime path against actual native outputs and composed inference.
[All measurements and controls](tests/results/qwen35-ocr/norm-statistics.json).

## Gated computed cascade experiment

`CRISPEMBED_QWEN35_VISION_CASCADE_NORM=1` selects computed four-lane cascade
statistics and explicit fused affine for width768/epsilon1e-6 vision norms.
Literal0 and unset preserve the existing two-pass path; unsupported shapes
retain it as well. When selected, cascade takes precedence over PRECISE_NORM;
PRECISE_ATTN remains independent. This is a candidate, not a promoted default.
Its arithmetic is in a separate production translation unit with the pinned
PyTorch BSD notice. No original moments, arrays or test dependencies enter it.

Hosted `--cascade-norm` requires75 actual native norm-output controls per case,
including the candidate on original inputs; it tests ordinary baseline, cascade,
and cascade plus the existing attention gate through all77 prefill stages.
Every previous baseline scalar record must reproduce exactly. Three actual
small-document decode arms must match original IDs/text and stop at EOS.
F32 reconstruction now explicitly separates variance multiplication/addition,
with fused accumulation retained as a contrast. Its equality controls must pass
before that reconstruction is accepted. Both earlier sets of75 failed controls
remain retained. No budgets are changed.

## Initial native cascade measurements

[Hosted38759613908](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38759613908)
at `16e3399d` tests actual native four-lane normalization, without original arrays
in production. All75 candidate norm-output controls and all75 F64 controls pass;
all225 delivered-input controls pass. All231 prior baseline scalar records remain
exact. Baseline/cascade/cascade-plus-attention small-document runs match all nine
original IDs, UTF-8 text and EOS; no large-document decode was added.

Strict prefill failures are20 baseline,11 cascade and14 cascade-plus-attention.
Cascade gives form/receipt/German counts0/6/5, versus baseline5/7/8. Its worst key
error is0.00549555 versus baseline0.00622153; adding precise attention gives
0.00423408 but increases the failure count. The existing combined F64 precision
arm also had11 failures, with a smaller0.00408316 worst key error; cascade is not
an overall best-path promotion. All alternatives remain default-off.

Explicit fused variance accumulation fails to repair all75 F32 reconstruction
controls; their measurements reproduce the initial negatives. The complete audit
retains2,979 vectors and exits nonzero for45 prefill verdicts plus75 controls.
Next test explicitly split variance multiplication/addition, retaining the fused
variant as a measured contrast. Do not relabel the failed reconstruction as
validated. [All initial runtime results and failures](tests/results/qwen35-ocr/cascade-norm-initial.json).

The completed follow-up uses explicit split variance accumulation and retains
the fused variant separately. It compares837 statistics/geometry vectors per case,
plus baseline/cascade/cascade-attention prefill and unchanged decoded controls.
All75 native norm-output controls per case pass in the final run.

## Verified native controls and current limits

[Final38764189561](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38764189561)
at `65c859b2` passes all 225 actual native norm-output and 225 delivered-input
controls. Explicitly split variance multiplication/addition fixes the validation
reconstruction; all 75 fused contrasts reproduce the old mismatch exactly.
[Actual compiler output](../tests/results/qwen35-ocr/norm-codegen.json) confirms
separate variance vector products and ordered additions at width 768, with fused
affine. This is build-specific evidence; unsupported widths retain the old path.

All 693 earlier prefill scalar records reproduce exactly. Failure counts remain
20/11/14 for baseline/cascade/cascade-attention; all nine first argmaxes and three
small-document ID/text/EOS controls match. The final diagnostic retains all 45
prefill failures and exits nonzero, with zero failed native controls. Cascade
merger maxima worsen on all three inputs, and the earlier combined F64 arm also
had 11 failures with a smaller worst key error. Keep all alternatives default-off;
no overall quality, speed or publication promotion follows from component parity.

[All 3,204 vectors and variance contrasts](../tests/results/qwen35-ocr/cascade-norm.json),
[completed lane](../HISTORY.md#2026-10-15--qwen35-native-normalization-controls-resolved)
and [remaining tasks](../PLAN.md). Large-document decoded A/B and remaining
attention/projection/rotary isolation remain open; original arrays never enter
production inference. Package/install/vendor licenses retain the kernel's BSD notice.

## Four-arm native block replay contract

`qwen35_ocr_bootstrap.py --cascade-blocks` reuses the actual linked native block
and suffix replay with baseline, combined F64, cascade and cascade plus precise
attention. All four arms explicitly set the cascade/precision gates, including
literal0 for the baseline. The validation executable and production sources are
unchanged; original inputs enter only through the checked test-only identity
frontend. Native rotary values stay in use, so this is conditional block parity.

Each arm covers266 vision vectors and 77 ordinary composed prefill vectors per
larger document. Across the three inputs this gives 4,116 vectors and 1,356 exact
identity/block/projection/residual controls. Baseline and combined F64 must
reproduce all 1,596 prior block vectors and their controls exactly. All four arms
must reproduce924 prior prefill records, including their unchanged budgets and
verdicts; every first argmax must agree. A changed historical metric is an
invalidating control failure, separate from the retained parity failures.

Four actual small-fixture decode controls additionally compare complete IDs,
UTF-8 bytes, text and EOS with the original reference. They reuse one document
across four arithmetic settings; they do not establish larger-document quality.
All raw failures remain in `cascade-blocks.json`, and the job exits nonzero when
any qualified comparison or decoded control fails. No runtime/default, tolerance,
model publication, speed or memory promotion follows from this diagnostic alone.


## Completed cascade block and suffix replay

[Hosted38775800804](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38775800804)
at `f7143914` completes 4,116 stage vectors and all 1,356 exact native replay controls.
All 924 prior composed prefill records,1,596 prior baseline/combined-F64 block
vectors and 678 prior control records reproduce exactly. The original anchor
retains 126 exact shared arrays and 106 additions;1,302 original CPU-control vectors
remain recorded. Production kernels and the C++ replay executable are unchanged
from the preceding normalization lane.

With original block inputs, cascade and cascade-attention produce exact norm1
outputs in all 36 blocks each. All 132 independent block0–10 outputs satisfy raw
0.002 across the four arms, while all 12 block11 outputs still exceed it. Cascade
reduces isolated block11 error from 0.007202/0.005859/0.006104 to
0.005493/0.005127/0.005585 on form/receipt/German. These block vectors report cosine
1.000000000 despite the absolute failures; qualified vision gates still apply.
[Four-arm error table](../PERFORMANCE.md#qwen35-cascade-native-block-replay-2026-10-15).

In cascade block11, exact norm1 is followed by pooled errors 1.37e-5–2.43e-5,
residual errors 2.00e-5–2.67e-5, norm2 errors 0.000170–0.000191, activation errors
0.00104–0.00127 and down/output errors 0.00513–0.00558. This localizes a remaining
post-norm1 attention-path contribution and later amplification. It does not yet
separate QKV projection, native rotary geometry and attention reductions.
Original geometry remains the primary reference; native geometry is retained
in these conditional block replays.

Composed cascade layer11 improves receipt0.441040→0.279785, but worsens form
0.136475→0.163574 and German0.954773→0.997681. Earlier resets remain nonmonotonic:
cascade receipt suffix4 gives 1.0625 and German suffix0 gives 1.285950, both qualified
failures. Precise attention removes those two qualified suffix failures in both
norm variants, but no arm is uniformly best and ordinary prefill remains
20/11/11/14 failures for baseline/combined F64/cascade/cascade-attention.

All 12 first argmaxes and four actual small-fixture ID/text/EOS controls pass.
The workflow correctly fails for 56 prefill and four suffix verdicts, with no
invalidating replay/reproduction controls. Every raw measurement and failure is
retained in [the complete report](../tests/results/qwen35-ocr/cascade-blocks.json).
No tolerance/default, large-document quality, speed, memory or publication change
follows. Next capture selected original attention boundaries with exact native
controls and compare full decoded outputs on the independent documents.

## Selected attention boundary audit contract

`qwen35_ocr_bootstrap.py --attention-trace` observes original eager calls in vision
blocks 5/11, returning their original outputs unchanged. Reconstructed F32 scores
and softmax must reproduce the returned probabilities exactly; the second matrix
product must reproduce the returned pooled tensor, which is also checked against
the original projection input. These checks apply in prefill and both original
CPU-control phases. The observer is restored before generation and on exceptions.

Full rotated Q/K/V and pooled tensors are retained. Scores/probabilities cover 16
deterministic query positions spanning the token axis, with every head and key;
this preserves each sampled row's full softmax normalization. Unsampled score rows
are not captured. The compact original reference and 126 shared full-anchor arrays
must remain exact. Expanded hooks add13 arrays beyond the earlier rich anchor.

The test-only native arithmetic trace must match actual linked production QKV,
full pooled output and sampled probability-to-value replay exactly before its
intermediate values can be interpreted. Every failed control is retained and
invalidates that interpretation. No trace code or original arrays enter the
production library; its source hash and the existing cascade helper are pinned.

Four arithmetic arms compare original QKV with original/native geometry, native
projection from original norm1 with original/native geometry, original rotated
Q/K/V reductions and original probabilities/value pooling. These conditional
experiments distinguish input projection, geometry and reduction effects. Their
scores/probabilities remain compared with the primary original geometry. Each
arm/case has93 shared-diff vectors and six native equality controls. Across four
arms and three larger inputs this gives 1,116 component vectors and 72 controls,
plus 924 unchanged ordinary prefill vectors and four small-fixture full ID/text/EOS
checks. All prior prefill, 108 shared composed vision and 1,302 original CPU-control
records must reproduce exactly; new original CPU boundaries bring that total to
1,374. All raw failures and unchanged qualified budgets remain visible. Completing
this diagnostic alone does not promote runtime arithmetic, full-document quality,
performance, quantization or publication.


## Initial attention trace controls

[Hosted 38791137435](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38791137435)
at `05d4705c` passes all 60 original function/projection reconstruction controls and
reproduces all 924 prior native prefill records and 1,302 original CPU-control
records exactly. It retains 2,040 vectors, all 56 prefill verdict failures, 12 exact
first argmaxes and four matching small-fixture ID/text/EOS controls.

All 24 native QKV and 24 probability-to-value controls pass, but all 24 full pooled
trace controls fail by 6.56e-7–7.27e-6. Even though all 1,116 component qualified
verdicts pass, the copied arithmetic is not validated as faithful production
behavior. The initial composed pooled fields use the copy too, invalidating all 12
aggregate shared-vision case checks;84 individual records and all 924 actual
prefill records still reproduce. [All initial failures](../tests/results/qwen35-ocr/attention-trace-initial.json).

The test-only follow-up makes rotary FMA ordering explicit, retains the original
expression as a measured contrast and compares composed pooled output directly
from the public production result. The [compiler inspection](../tests/results/qwen35-ocr/attention-initial-codegen.json)
at `4abeee6a` confirms that the production object disassembly is byte-identical to
the prior norm audit. Native scalar rotary rounds sine products before a fused
cosine multiply-add/subtract. The initial trace's vector fast path instead
contracts the sine subtraction for the left component. Exact retained excerpts
show both instruction sequences. This is build-specific compiler evidence;
the corrected trace still needs all native equality controls to pass. Production
kernels, gates and comparison budgets are unchanged.


## Rotary-only trace correction and remaining controls

[Hosted 38795690126](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38795690126)
at `07e8ea00` passes all 36 native controls in the precise-attention arms after the
explicit rotary pin. All 24 QKV and 24 probability-to-value controls pass across
all four arms. Twelve full pooled controls still fail in the F32-attention arms,
so those traces remain invalid. Their largest mismatch is 8.43e-6.

All 924 actual native prefill and 108 shared composed vision records reproduce
exactly. Composed pooled measurements now use the public native result directly,
independently of trace fidelity. All 1,302 prior original CPU-control records and
60 original function/projection controls remain exact. The workflow retains 56
prefill failures and 12 invalidating native controls, while all four small-fixture
ID/text/EOS arms pass. [Every measurement](../tests/results/qwen35-ocr/attention-trace-fma-initial.json).

The [actual compiler output](../tests/results/qwen35-ocr/attention-rotary-codegen.json)
shows why a further correction is needed: native QK uses scalar FMA at 0x3ab0,
while the trace's F32 loop multiplies four lanes at 0x1574 then adds the rounded
products. Its scalar remainder uses FMA, but is skipped at head width 64. The
production disassembly still matches the earlier norm audit. A test-only explicit
QK FMA pin targets this mismatch; exact native controls must verify it. This does
not change production attention, tolerances or default precision gates.


## Completed faithful attention trace

[Hosted 38800963470](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38800963470)
at `f93ebcd3` passes all 72 actual native QKV/full-pooled/sampled probability-to-
value controls exactly after explicit rotary and scalar QK FMA ordering. This
repairs validation fidelity; production attention is unchanged. The initial 24
failed full-pooled controls and rotary-only 12 F32 failures remain retained.
All 24 current legacy rotary contrasts still differ (maximum 6.32e-6), while the
pinned trace has zero error in every contrast. The production object disassembly
again equals the prior norm audit. [Actual final compiler provenance/excerpts](../tests/results/qwen35-ocr/attention-final-codegen.json).

All 60 original function/projection controls, 126 shared anchor arrays, 924 prior
prefill, 108 shared composed vision and 1,302 original CPU-control records remain
exact. All 558 F64 trace vectors from the rotary-only run remain exact. All 420
conditional vectors shared across norm-gate pairs also match. The full audit
compares 2,040 stage vectors and 1,374 original CPU-control vectors; its rich anchor
adds119 arrays beyond the 126 shared full-reference arrays.

Every one of the 840 conditional comparisons satisfies raw 0.002 and reports cosine
1.000000000. With original rotated Q/K/V, F32 sampled QK scores match exactly,
while subsequent probabilities differ by up to1.88e-6 and full pooled output by
1.67e-5. Given original probabilities/V, sampled F32 pooling matches exactly.
This isolates a remaining softmax-stage rounding contribution on sampled rows.
Original norm1 with native projection and original geometry produces full pooled
error up to2.43e-5. Supplying original geometry does not eliminate all projection/
reduction differences. These maxima are conditional errors against the same
original reference, not additive error attributions or complete score-matrix
coverage. [F32/F64 measurement table](../PERFORMANCE.md#qwen35-faithful-selected-attention-trace-2026-10-15).

All 1,116 qualified attention verdicts pass;51 raw composed attention failures
remain visible. Ordinary prefill retains 20/11/11/14 failures across the four arms,
so the workflow correctly fails for 56 prefill verdicts with zero failed native or
reproduction controls. All 12 first argmaxes and four actual small-fixture complete
ID/text/EOS arms match. The trace is now suitable for further diagnosis on this
validated build; the subsequent fixed-cap full-output A/B is recorded below. F64 attention is not uniformly closer to the original F32 result.

[Complete final report](../tests/results/qwen35-ocr/attention-trace.json),
[initial failures](../tests/results/qwen35-ocr/attention-trace-initial.json) and
[rotary-only failures](../tests/results/qwen35-ocr/attention-trace-fma-initial.json)
preserve every raw metric and control. Original arrays stay test-only. Production
kernels, precision defaults, weights, thresholds and publication status are unchanged.

## Four-arm decoded panel driver

`tools/ci-heavy/qwen35_ocr_bootstrap.py --decoded-panel` runs the fixed-cap decoded
gate through hosted `heavy-cpu.yml` on the same ARM64 runner and pinned Python
packages as the attention audit. `qwen35_decoded_panel.py` freezes the original
Python references first, then compares actual owning-context free generation
under baseline, combined F64, cascade and cascade-attention settings. The four
existing documents, image derivatives, prompt and token caps stay fixed. Full
IDs, UTF-8 bytes/text, EOS versus cap, confidence and available CER/WER are
reported; teacher-forced score comparisons remain separate from free generation.
Reference confidence normalizes the actual original F32 generation scores in
F64, including EOS, matching the native API's definition; it is not calibrated
OCR accuracy. This adds metadata without changing original scores or generation.

The driver also reruns the three larger inputs' strict prefill comparisons with
the unchanged shared `crispembed-diff`/`qwen35_compare` gates. It checks all 924
prior records, 1,302 original CPU-control records, original full outputs and the
previous baseline decode measurements exactly. Exact output cannot waive a
strict failure. The receipt remains a 384-token prefix/cap control until the
separate long-document lane. No model, production arithmetic, tolerance, default
or publication changes are part of this diagnostic driver. Model-free positive
and malformed-output controls run in `qwen35-decoded-audit.yml` and before the
hosted model audit. The [initial run](../tests/results/qwen35-ocr/decoded-panel-initial.json)
reaches its 120-minute limit after 14/16 exact ID/text/stop comparisons and preserves
770 earlier prefill records with 45 failures. `--decoded-panel-tail` completes only
the missing cascade-attention receipt/German cases, after requiring fresh original
references, CPU controls and confidence to match the initial capture exactly.
The [completion run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38830295916)
at `9ba370a8` finishes both missing cases. [Combined records](../tests/results/qwen35-ocr/decoded-panel.json)
retain per-case source/workflow provenance and both raw report hashes. All 16
ID/text/stop comparisons and 2,216 teacher argmaxes match. Each arm reaches EOS
on Arabic/form/German and the 384-token cap on the receipt. Paired Python/native
CER/WER is zero; maximum mean-confidence error is 3.28822e-6. All 924 earlier
prefill records, 1,302 CPU controls and four baseline decode measurements remain
exact; fresh completion references reproduce 868 original CPU controls plus
confidence exactly. All 56 prefill failures remain (20/11/11/14); the completion
workflow correctly fails on its 11 retained stages. The initial timeout remains
visible. No default, numerical-gate, full-receipt or transcription-quality promotion.

Existing form/receipt annotations still disagree with original Python output;
reference parity does not establish OCR accuracy. Native receipt-prefix recognition
is 585.82 s versus original generate 31.67 s in single observations with different
scopes. [All timing/quality limits](../PERFORMANCE.md) are retained. Profiling
projection materialization, backend/graph setup, vision and decode is now first
in PLAN; every change must preserve full outputs and prior numerical records.
