# OvisOCR2 source contract and bounded integration lane

The source/license/header audit is complete; native support is pending. No model
payloads were fetched and no conversion/inference or benchmark was run.
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
No tensor value, tied-head equality or native schema roundtrip was checked.

## Concrete blockers and original blueprint differences

1. Config declares `mtp_num_hidden_layers=1`; no header tensor name contains
   `mtp`. The unchanged converter rejects it with its explicit MTP-policy error.
   Establish non-speculative original execution and expected official meta schema
   before adding a pinned export policy. Never broadly ignore MTP or unknown keys.
2. `generation_config.json` and `processor_config.json` are absent; the converter
   currently requires both. Nested text EOS248044 and tokenizer EOS
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
