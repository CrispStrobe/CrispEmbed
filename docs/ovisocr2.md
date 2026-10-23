# OvisOCR2 source contract and bounded integration lane

The source/license/header and original metadata audits pass. Experimental F32/F16
conversion and all473-tensor roundtrips pass39992662147; PR77 is merged.
[Export producer and independent review](../tests/results/qwen35-ocr/ovisocr2-export.json)
record exact sources, payload/manifest hashes, artifact expiry and review limits.
Model payloads remain remote. Exact frontend parity passes40017014043; PR78
merged918c1c8d. Native draftPR79 prepares inference and audits; model-backed
numerical/decoded and independent quality results remain pending.
[Full configuration, hashes and473-tensor header](../tests/results/qwen35-ocr/ovisocr2-source-contract.json).

Sources inspected23 October2026:

- [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2/tree/cbdede82e048659b83310476ff807da825f8fe78),
  revision `cbdede82e048659b83310476ff807da825f8fe78`: card and LICENSE Apache-2.0.
- Declared [Qwen3.5-0.8B base](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17),
  revision `2fc06364715b967f1860aea9cf38778875588b17`: card and LICENSE Apache-2.0.
- [Original inference example](https://huggingface.co/ATH-MaaS/OvisOCR2/blob/cbdede82e048659b83310476ff807da825f8fe78/README.md)
  uses vLLM0.22.1, Triton GDN prefill, greedy temperature0 and16384 new tokens.
  Its benchmark numbers are upstream claims, not CrispEmbed measurements.

The text/vision dimensions match the small dense Qwen3.5 family already ported
for LightOn:24 text blocks,1024 width,18 linear/six full attention,12 vision
blocks at768 width, patch16/temporal2/merge2 and tied248320-token embedding/head.
Sharing dimensions does not qualify an alias.

The473-entry safetensors header describes852,985,920 stored parameters in a
1,706,030,496-byte file. Only two HTTP206 byte ranges were read:0–7 and8–58655.
The58,648-byte header hash and complete names/shapes/dtypes/offsets are retained.
That initial range-only source audit did not inspect payloads. The later
qualified export39992662147 independently accounts for and roundtrips every
original/cast tensor; its complete evidence is linked above.

## Initial blockers and original blueprint differences

1. Config declares `mtp_num_hidden_layers=1`; no header tensor name contains
   `mtp`. The unchanged converter rejects it with its explicit MTP-policy error.
   The qualified exact-source profile now accepts the reviewed non-speculative
   main-model schema only. Actual vLLM model execution remains pending; unknown
   config/tensors are still rejected.
2. `generation_config.json` and `processor_config.json` are absent; the converter
   originally required both. Nested text EOS248044 and tokenizer EOS
   `<|im_end|>` need separate effective-generation/stop tracing in vLLM. Do not
   invent LightOn EOS248046 defaults or fabricate missing source files.
3. The original example uses an explicit content-extraction prompt, thinking
   disabled, and pixel limits448²–2880². The saved preprocessor defaults differ
   (65536–16777216 pixels). Capture actual rendered prompt, processor packed RGB,
   grids and image positions with the example overrides. The native frontend
   currently pins LightOn identity and uses its fixed prompt framing. Generalize
   only with separate pinned profiles and exact token/mask/Unicode controls.
4. Native vision currently rejects more than1024 patches (roughly512² pixels),
   below the upstream maximum. Establish a bounded supported resolution first;
   report rejection explicitly. Increasing it needs separate memory/geometry and
   long-page qualification; silently shrinking the upstream input is invalid.
5. Original parsing strips whitespace, optionally removes whole bbox-image-tag
   blocks (enabled by default), then cleans repetitive truncated tails. Preserve
   raw IDs/text/EOS/completion separately from rendered Markdown. Validate HTML
   tables, LaTeX, reading order and scaled `[0,1000)` image boxes independently;
   a cleanup routine cannot make a capped raw decode count as complete.

## Next implementation and acceptance

Start with a metadata-only model profile and exporter preparation on an isolated
branch; keep the current source checks and all unknown-tensor failures. Pin and
inspect vLLM0.22.1 Qwen3.5 loading/generation code and its Transformers dependency;
record effective EOS/sampling and exactly how absent metadata is resolved.
Before weight download, add model-free wrong-MTP/schema/missing-metadata and
wrong-profile controls. No new alias or publication follows from a header audit.

Run the first weight-backed conversion and official/native comparison remotely.
Reuse `tools/dump_qwen35_ocr_reference.py`, the page-reference dumper,
`tools/ci-heavy/qwen35_ocr_bootstrap.py` and `crispembed-diff`; add original probes
where the original contract differs. Do not silently replace the vLLM example
with an unqualified Transformers generate path. Require every tensor accounted
for, roundtrip bytes, original preprocessing/IDs, all layers/caches/logits,
argmax and final raw/processed outputs, including EOS/cap, repeated/two-context
and empty/invalid cases. Separate local operator replay from composed output.

Freeze declared development and independent evaluation inputs before selecting
precision. Retain the LightOn numerical debt and every failed alternative. Start
F32/F16 text with audited vision precision; only then use `crispembed-quantize`
for Q8/Q4 and measure quality, latency/RSS and critical-tensor retention.
No component cosine or upstream benchmark claim substitutes for task accuracy.

## New upstream checkpoint observed

The OvisOCR2 card points to [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd).
Its metadata declares Apache-2.0/Qwen3.5-0.8B and the same small dimensions, with
MTP0. Only card/config were inspected. Audit its actual LICENSE, schema,
processor/prompt and original decoding before considering it as a sibling;
its recency is not a reason to skip the pinned OvisOCR2 evidence or claim support.

## Pinned execution-blueprint follow-up — 23 October2026

[vLLM0.22.1](https://github.com/vllm-project/vllm/tree/0decac0d96c42b49572498019f0a0e3600f50398)
resolves to0decac0d. [Source hashes, isolated stop probe and next gates](../tests/results/qwen35-ocr/ovisocr2-vllm-blueprint.json)
retain the inspected files. No vLLM/model installation or inference occurred.

- The [ordinary model forward/loader](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/vllm/model_executor/models/qwen3_5.py)
  uses the main language model and skips MTP-prefixed weights; speculative state
  expansion depends on an explicit speculative configuration. Inspected HF5.16.1
  main-model source only mentions MTP in unexpected-key ignore lists. This
  supports auditing a pinned non-speculative policy; it does not authorize
  silently ignoring arbitrary config or tensors. Official meta-schema comparison
  with the complete473-entry header still precedes export.
- [Missing generation metadata falls back to the model configuration](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/vllm/transformers_utils/config.py).
  Inspected Transformers5.16.1 fills unset generation values from the nested text
  config. Ovis config EOS248044 is endoftext; tokenizer EOS248046 is im_end.
  The [vLLM stop updater](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/vllm/sampling_params.py)
  combines both. Its original AST method, executed in isolation with those
  supplied fields, yields primary248046 plus stop248044; removing the derived
  generation field demonstrably removes the second stop. This is a source probe,
  not installed end-to-end resolution. Native currently accepts/stops on only
  248046; a pinned stop-set representation is needed before aliasing.
- [Processor loading](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/vllm/transformers_utils/processor.py)
  can obtain processor_class from the existing preprocessor/tokenizer metadata,
  then delegates to AutoProcessor. The saved preprocessor names Qwen3VLProcessor
  and Qwen2VLImageProcessorFast. Missing processor_config alone is not proof
  the official path cannot load; our exporter must preserve missing-source
  provenance and measured effective metadata, rather than inventing a file.

The example pins vLLM but its dependency specifies a Transformers range. Lock
and verify an actual compatible environment before claiming effective behavior.
Next run a hosted **metadata-only** original probe (no model weights): resolved
generation fields and both stops, processor/tokenizer classes, rendered prompt
and packed synthetic RGB/grid/positions under the example overrides. Then audit
meta schema and add model-free pinned-profile/missing-source/dual-stop controls.
The unchanged LightOn converter/frontend and their failed numerical/quality
records remain intact. Weight-backed parity still needs the actual vLLM path.

The next probe is executable in `tools/ci-heavy/ovisocr2_metadata.py` through
`ovisocr2-metadata.yml` (manual heavy job, push runs only three model-free controls).
It pins CPU Torch2.14.0/TorchVision0.29.0/Transformers5.16.1, downloads only
audited metadata/tokenizer files, rejects fabricated missing-source files,
checks loaded model/generation source hashes and compares a meta-only official
state dictionary to all473 header tensors. Three owned synthetic RGB fixtures
preserve original packed arrays/IDs/grid plus exact repeat checks remotely.
It executes no model forward and does not install or qualify a complete vLLM
runtime. Inspect the full report and retain any failure before a profile change.

Initial hosted metadata run39974181719 at60679016 was superseded while
queued; original metadata39983101540 subsequently passed. Push-only duplicates39974180947/39974177044 were
cancelled before execution; the manual run retains its required controls.
The source blueprint was also independently reviewed with read-only web tools;
its installed-vLLM qualification caveat remains. Prior header and source audits
are preserved, not repeated in response to generic review suggestions.

The initial39974181719 was superseded while still queued, before processor
execution or original captures. A read-only Sol review found that repeat equality
alone did not prove example pixel overrides. The corrected producer adds a
96×64 owned image and a freshly constructed default processor control, saves
both packed arrays/grids before checking the minimum-pixel branch, and rejects
ignored overrides. This demonstrates the448² minimum only; the2880² maximum
remains unexercised and is labelled accordingly. No large local image/model
work is introduced. Phase checkpoints and bounded URL/token/path-redacted error
details preserve the earliest failure. Five model-free groups pass locally.

## Experimental pinned export profile

`models/qwen35_ovis_profile.py` consumes the reviewed39983101540 proof and exact
ten-file metadata set, including README/prompt and license. It accepts MTP1
only for the pinned source/config/non-speculative main schema, requires the
complete checkpoint size/SHA before tensor reads, and rejects shard indexes or
fabricated absent metadata. The ordinary converter path still rejects MTP.

The explicit `--profile ovisocr2` branch preserves raw missing-file provenance,
derived generation/processor values, both stop IDs and the original single-image
prompt in experimental format2. LightOn output stays format1; no native alias,
reference tensors or quality claim is added. Ten pure source/proof/CLI/config/
checkpoint controls plus five metadata controls pass locally. Source review
caught an initial missing-README allowlist before any weight job; the exact
source-file allowlist now comes from the shared profile.

On hosted CI, run `ovisocr2-metadata.yml` with `phase=export` at the reviewed
feature source. `tools/ci-heavy/ovisocr2_export.py` checks original service
qualification before dependencies/weights, executes actual F32/F16 converter
CLIs and verifies all473 original casts, names/shapes/dtypes and GGUF payload
hashes independently. Models/manifests/license remain remote in an experimental
artifact; download only the reports for review. No local weight conversion is
authorized by this protocol on a constrained shared host.

Full-weight conversion/roundtrip is not yet qualified. Next native work must
compare actual prompt IDs, packed RGB/grids/positions and both stopping IDs
against the preserved original arrays before adapting the LightOn frontend.
Use original-module intermediate dumpers and crispembed-diff for the model
forward and decoded EOS/text/cleanup; metadata success cannot replace that.
Maximum resolution, long pages, existing numerical debt and publication gates
remain explicit.

The roundtrip auditor at5e197bae derives record names, flattened patch shapes
and requested F32/F16 precision directly from the qualified original schema.
It rejects duplicate/extra records or GGUF tensor names rather than trusting
exporter manifest declarations. Negative controls mutate precision, shapes,
source coverage and names. The initial full export39991277170 was cancelled
while its weight-backed job remained queued, before execution. The replacement
[39992662147](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39992662147)
uses that strengthened source. Inspect both F32/F16 reports and original-cast
payload checks before any conversion qualification; no duplicate local work.

## Separately gated native frontend audit

`feat/ovisocr2-frontend` adds a frontend-only opt-in
`CRISPEMBED_OVISOCR2_FRONTEND=1` alongside the master Qwen OCR gate. It admits
only the pinned experimental metadata contract and uses the exact625-byte
original prompt,448²–2880² image limits and both stop IDs248044/248046. Empty
custom prompt selects that original template; nonempty overrides are rejected
until independently qualified. Setting either gate to0 disables already-loaded
Ovis tokenization. The existing LightOn input surface remains separate. Native
OCR initialization still requires format1, so this does not enable an Ovis model
loader or registry alias.

`tools/ci-heavy/ovisocr2_frontend.py` verifies the original metadata run/artifact
and every frozen NPZ/RGB/field hash, then writes a test-only GGUF and builds
`test-ovisocr2-frontend` on hosted CI. Three original packed-image cases cover
784/840/864 patches; the native test compares grids, packed values with the
unchanged1.2e-7 absolute budget, exact IDs/masks/modalities/template, dual stops
and16 wrong-profile mutations. Oversized images are rejected at the unchanged
1024-patch bound; the upstream maximum is not silently resized to that bound.
Six driver service/artifact controls run without model-framework imports.
Run `ovisocr2-metadata.yml` with `phase=frontend` at the pinned feature source.
Retain the original reference, failed output and earliest absolute errors before
fixing any mismatch; no model-forward quality claim follows from frontend parity.

### Initial frontend failure and tokenizer correction

Run40004747653 at48aefc30 compiled and passed CPU-baseline checks, but exited
before numerical comparisons. The Ovis source regex contains combining marks
(`\p{M}`) in letter runs and excludes them from punctuation; the initial loader
expected LightOn's different regex. The
[initial receipt](../tests/results/qwen35-ocr/initial-ovisocr2-frontend-pattern-failure.json)
preserves the original report, log hashes and30MB reference hash/artifact.

The correction selects Ovis's exact pattern and the existing mark-aware BPE
path by source profile; LightOn retains its original pattern and gated
alternative. Twelve new original `tokenizers==0.23.3` probes capture NFC,
pre-tokenizer chunks, IDs and decoded text for combining marks, Indic/Arabic,
punctuation, whitespace, emoji and special tokens. Native comparisons exercise
both values of the older marks override without overriding Ovis's source
contract. Original packed-image captures and the1.2e-7 budget remain unchanged.
Actual corrected frontend parity awaits a hosted retry.

## Current execution sequence

Full export and exact native frontend qualification are complete. The
[controlled original GPU reference](../tests/results/qwen35-ocr/ovisocr2-vllm-reference.json)
passes at5fe3fc90: T4 auto-FP16, explicit TRITON_ATTN decoder, auto Torch-SDPA
vision and requested Triton GDN. Both original images reach EOS248046. The
unchanged automatic FlashInfer startup failure remains retained; no automatic-
backend or newer-GPU BF16 equivalence is inferred. Exact input/report/log bytes
are [preserved immutably](../tests/results/qwen35-ocr/ovisocr2-vllm-reference-preservation.json).

Native draft [PR79](https://github.com/CrispStrobe/CrispEmbed/pull/79) and model
[40039159696](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40039159696)
at d0f64542 are the next measured gate. Initial40034685690 stopped before forward
on a Python tuple/list processor comparison; all three saved processor fields
and framework source hashes had matched. The correction compares exact JSON
representations and records each field/type; hosted confirmation is pending.
It captures original HF CPU/F32 modules and genuine MKLDNN-off controls before
native outputs, then compares every layer/cache/logit/score and actual decoded
IDs/EOS/text through the existing harness. Raw native vectors are retained
remotely; scalar cosine alone does not qualify task accuracy.

The candidate's explicit pipeline selector is engine21 (`ovisocr2`) with an
exact model file and all three gates: `CRISPEMBED_QWEN35_OCR`,
`CRISPEMBED_OVISOCR2_FRONTEND`, `CRISPEMBED_OVISOCR2_INFERENCE`. Actual owning
C/Python/CLI/HTTP/Rust/Dart tests are prepared, including multiple stage model
paths, prompt/profile/gate and ABI controls. Generic APIs expose text/confidence;
actual IDs/stops are retained separately by the test-only owning runner.

Follow the pending [PLAN sequence](../PLAN.md) for original-first independent
quality, protected Q8/Q4, four-development-image/72-FFN importance calibration
and conditional publication. Keep every failed precision candidate and the
unchanged LightOn debt. Only a qualified candidate with complete license/base
provenance, pinned remote hashes and an actual immutable empty-cache download
may enter the registry/catalog. No Ovis model alias or weights are published.
