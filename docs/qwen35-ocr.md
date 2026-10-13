# Qwen3.5 OCR: first checkpoint and native port blueprint

Started 2026-10-13 with **LightOnOCR-3-0.8B**, the smallest missing checkpoint
in the permissive OCR queue. This is an export/reference bootstrap, **not native
support**. No registry alias, output-parity score, calibrated Q4 quality result
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
  -f 'pip=torch==2.10.0 transformers==5.5.4 gguf==0.19.0 huggingface_hub safetensors pillow numpy' \
  -f timeout_minutes=60
```

## Next bounded implementation steps

1. Native graph behind `CRISPEMBED_QWEN35_OCR=1`, using `core_env::on()`:
   exact tokenizer/processor, vision and merger first, then decoder prefill and
   recurrent/KV cache continuation. Reuse vendored `ggml_gated_delta_net`; verify
   its input normalization, scaling, layout and state convention against the
   official captured recurrence before calling it equivalent.
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
