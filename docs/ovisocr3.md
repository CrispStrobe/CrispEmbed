# OvisOCR3 pinned source contract

**Status: original execution and capture reviewed; source quality unqualified.**
The pinned Apache-2.0 checkpoint has actual controlled vLLM/T4 execution,
installed-processor evidence and a 943-stage original HF CPU/F32 reference.
[The eleven-case original capture](../tests/results/qwen35-ocr/ovisocr3-original-quality.json)
is complete, but its two-column case reaches the unchanged 1024-token cap
without EOS (CER 0.52365, WER 0.83993). Publication and a model alias are blocked.
Native F32/F16 and paired-output audits can establish port fidelity separately;
prepared workflows do not establish a positive result. No runtime release or
application upgrade follows from this work.

## Pinned provenance and license

| Source | Audited revision | Observed license |
| --- | --- | --- |
| [ATH-MaaS/OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd) | `5f6ce6932efee9ec10c30746db55bd780214b7fd` | [Card](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/README.md) declares Apache-2.0; [LICENSE](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/LICENSE) contains Apache License 2.0 |
| [Declared Qwen3.5-0.8B base](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | `2fc06364715b967f1860aea9cf38778875588b17` | Card and [LICENSE](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/LICENSE) declare Apache-2.0 |
| [Base model's declared ancestor](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Card and [LICENSE](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/LICENSE) declare Apache-2.0 |
| [vLLM 0.22.1](https://github.com/vllm-project/vllm/tree/0decac0d96c42b49572498019f0a0e3600f50398), required by the model example | `0decac0d96c42b49572498019f0a0e3600f50398` | [LICENSE](https://github.com/vllm-project/vllm/blob/0decac0d96c42b49572498019f0a0e3600f50398/LICENSE) contains Apache License 2.0 |

The Qwen revisions pin the public snapshots inspected for license terms.
The OvisOCR3 card identifies the base repository, but does not identify the
exact Qwen commit used to initialize training. Do not present these audit
revisions as training provenance.

## Checkpoint and reusable architecture

The pinned [config](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/config.json)
is Qwen3.5 conditional generation: 24 text layers at width 1024, 12 vision
layers at width 768, vocabulary 248,320, tied embeddings, image token 248056,
and MTP hidden layers **0**. Compared as JSON with the pinned
[OvisOCR2 config](https://huggingface.co/ATH-MaaS/OvisOCR2/blob/cbdede82e048659b83310476ff807da825f8fe78/config.json),
its only value change is `text_config.mtp_num_hidden_layers` from 1 to 0.

The config's `transformers_version="4.57.0.dev0"` is saved metadata, not
an installed-runtime pin. Record the actual Transformers/vLLM source and
resolved processor in the first hosted metadata run.

The public single `model.safetensors` LFS declaration is 1,706,030,528 bytes,
SHA-256 `aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600`.
Two HTTP 206 ranges read only the 8-byte header length and 58,680-byte JSON
header. Its 473 BF16 tensor entries account contiguously for 852,985,920
stored parameters; no MTP-named tensor or untied `lm_head.weight` occurs.
Every name, shape, dtype **and data offset** matches the reviewed
[OvisOCR2 source header](../tests/results/qwen35-ocr/ovisocr2-source-contract.json).
The checkpoint LFS hashes differ. The source-only header audit did not download weight payloads. The later
reviewed original T4 run verifies the full OvisOCR3 payload SHA and executes it;
equal headers still do not establish native parity.

The tokenizer JSON, merges, vocabulary, tokenizer config, chat template, video
preprocessor and preprocessor config share the same Git/LFS identities as
OvisOCR2 at its reviewed revision. Their saved
[preprocessor config](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/preprocessor_config.json)
names `Qwen3VLProcessor` and `Qwen2VLImageProcessorFast`, with default
pixel area 65,536–16,777,216. OvisOCR2's installed Transformers 5.16.1
metadata run resolved a slow `Qwen2VLImageProcessor` despite this saved
Fast declaration. [Hosted metadata40172525903](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40172525903)
now resolves OvisOCR3 to `Qwen3VLProcessor` / slow `Qwen2VLImageProcessor`
with torchvision backend. Independent review verifies three authored RGB inputs,
all five packed-field hashes, IDs/masks/grids, repeat identity and the separately
rendered531-byte plain OCR template. Small-input default versus explicit448²
minimum changes packed pixels, so the official override is required.
[Producer and independent review](../tests/results/qwen35-ocr/ovisocr3-metadata.json)
retain exact versions/source hashes. This weight-free probe establishes no model
forward or installed vLLM runtime parity. Neither repository has
`processor_config.json`.

## Official inference and changed contracts

The pinned [README inference example](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/README.md)
uses vLLM 0.22.1, one or two RGB images per request, Triton GDN prefill,
`tensor_parallel_size=1`, `gpu_memory_utilization=0.8`, and an
image-per-prompt limit of 2. It renders a user message containing one image
content item per image followed by the task text, with
`add_generation_prompt=True` and `enable_thinking=False`. Each request
passes nested `mm_processor_kwargs.images_kwargs` with
`min_pixels=448²` and `max_pixels=2880²`. Greedy
`SamplingParams(max_tokens=16384, temperature=0.0)` returns raw text and a
finish reason; reaching the cap emits a warning. The example does not repair
or normalize general OCR output.

The model supplies 18 separately hashed [task prompts](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd/prompts).
The [plain OCR prompt](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/prompts/plain_ocr.txt)
asks for plain text and explicitly excludes Markdown/HTML/JSON. The
[document Markdown prompt](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/prompts/document_markdown.txt)
has a different output contract; the README applies OvisOCR2-style
postprocessing only to that mode, while retaining raw Markdown. Its
[cross-page script](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/document_crosspage.py)
repeats the same vLLM request settings for two-image work.

OvisOCR3 adds a real
[generation_config.json](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/generation_config.json):
EOS IDs `[248044, 248046]`, BOS and pad 248044, sampling disabled. The nested
text config still says EOS 248044 and the tokenizer says `<|im_end|>` ID
248046. OvisOCR2 lacked this file and its vLLM stop contract was derived from
fallback metadata. Capture OvisOCR3's **actual** loaded generation config and
effective vLLM stop set before declaring native EOS/completion parity.

## Bounded implementation gate

1. Add an **OvisOCR3-specific, exact-source profile** only after a hosted
   weight-free metadata run records the installed model/processor/tokenizer
   source versions, all 473 official state shapes, rendered task prompts,
   RGB/packed pixels, grids, image placeholders, generation settings and
   effective stops. Preserve OvisOCR2's own proof marker and MTP=1 policy;
   the source change to MTP=0 is not a general bypass. Continue rejecting
   unknown tensor names and missing required metadata.
2. Convert full checkpoint payloads on hosted resources with complete file
   SHA/size verification, independent tensor coverage, dtype/shape policy and
   roundtrip checks. Keep the source header audit separate from that result.
3. Freeze original vLLM outputs first for the intended prompt modes,
   including raw IDs, EOS/finish reason, images, processing arrays, hidden
   stages/caches and final text. Compare native one-image support under its
   current 1024-patch bound before any larger-page or two-image extension.
   Explicitly reject unsupported upstream image geometry.
4. Require gated F32/F16 numerical and decoded agreement, representative
   licensed OCR quality, public API checks and fresh alias-download proof
   before publication. Decide quantization only after a qualified F16 baseline,
   retaining protected tensors and failed candidates. Upstream benchmark
   tables are upstream claims, not CrispEmbed measurements.

The matching source schema makes OvisOCR3 a practical sibling candidate.
It does **not** qualify the existing OvisOCR2 GGUF, fixed prompt, cleanup,
vision preset, precision selection or production alias for the new weights.

## Active original-first qualification

[Original reference40177416485](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40177416485)
completed at immutable source2ce86168. Its independently reviewed
[receipt](../tests/results/qwen35-ocr/ovisocr3-original-reference.json) binds the
service/head/artifact and report hashes, three processor anchors, all 943 stages,
72 decoder-layer records, 96 cache states and nine generated score steps. The
owning output is `Hello OCR\n12345`, with nine IDs ending at EOS 248046.
The original CPU-backend-off vision diagnostic retains a 0.0816956 layer-11
absolute difference despite identical decoded output; this does not waive any
native numerical budget. The separate [eleven-case panel](../tests/results/qwen35-ocr/ovisocr3-quality-panel.json)
freezes eight authored and three previously reviewed licensed document inputs
before native results. Scoring uses plain raw-text CER/WER, exact paired IDs/text/
stops and cap records. Ovis2's HTML presentation scorer does not apply.

The new selector is append-only public engine22 / CLI `ovisocr3`; it requires an
explicit GGUF and all value-parsed `CRISPEMBED_QWEN35_OCR`,
`CRISPEMBED_OVISOCR3_FRONTEND`, `CRISPEMBED_OVISOCR3_INFERENCE` gates. Its cache
is separate from engines20/21, requires exact format3 and rejects custom task
prompts. This is implementation under audit, with no qualified alias or release.

Source-only `--profile ovisocr3` exports carry false qualification flags and are
rejected by inference. A separately proof-bound original-HF export may qualify
that explicit EOS-list policy after actual reference evidence. Such a result
must retain the distinction from installed vLLM's effective stop/runtime policy.
Original eleven-case capture is reviewed. [Final-source native14](../tests/results/qwen35-ocr/ovisocr3-native-model.json)
passes all14 F32/F16 arms with exact nine-ID smoke output, context recovery and
profile-isolated1024 bounds. Paired native F16 quality40187405735 and six public
interfaces remain acceptance gates. Fresh immutable download/publication requires a
qualified source-quality protocol and is blocked by the current result. Other17
tasks, multi-image inference and the official2880² pixel maximum remain outside
the initial single-image plain OCR claim.

## Reviewed controlled original GPU execution

The [GPU receipt](../tests/results/qwen35-ocr/ovisocr3-vllm-reference.json) preserves
actual T4 FP16 execution under vLLM0.22.1, controlled TRITON_ATTN decoder
attention and Triton GDN prefill. Three processor anchors and the plain prompt
match reviewed metadata exactly. The installed runtime stop set is
`[248044,248046]`; square/readable requests finish at EOS with11/9 IDs under the
bounded256 cap. Readable raw text is `Hello OCR\n12345`. The patterned square
has no OCR gold. Source/log/score hashes and independent checks are retained.

This establishes the controlled original GPU path and its stop policy.
Original CPU/F32 module capture and native/quality/public consumer comparisons
remain separate gates. No automatic-backend, BF16 or public benchmark score is
claimed by these two requests.

## Original plain OCR quality decision

[Original eleven-case40178328864](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40178328864)
is fully captured and independently reviewed. Eight authored cases have zero
raw CER/WER; the receipt and inventory reach EOS. The frozen two-column page
hits1024 tokens without EOS, with repeated/incorrect raw text (CER0.52365,
WER0.83993). [The complete receipt](../tests/results/qwen35-ocr/ovisocr3-original-quality.json)
preserves this source-model failure. No cap, prompt, gold or scoring change follows.

Publication remains blocked by the unchanged completion screen. Native stage
parity and a complete paired F16 capture can establish port fidelity separately.
Any diagnostic public-interface result must explicitly retain the negative
source quality and cannot admit a model alias or distribution. Other17 upstream
task modes need their own declared original-first protocols and untouched
evaluation sets.
