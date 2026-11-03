# Xiaomi-OCR-0 experimental native profile

This lane adds the exact `xiaomi-research/Xiaomi-OCR-0` checkpoint at
`bbd8cf8d21f256b3644b44cb860f3a7cecb93c30` to the existing small Qwen3.5
CPU implementation. It is opt-in and has no registry/download alias, published
GGUF, benchmark result, precision qualification, or change to existing models.
The [source contract](../tests/results/qwen35-ocr/xiaomi0-source-contract.json)
records the original metadata, tensor header, framework sources, task prompts,
and layout dependency. No original/native model inference was performed while
preparing this source-only change.

## Exact source and reuse boundary

The [official checkpoint](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0/tree/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30)
contains 488 BF16 tensors: 473 main-model tensors and 15 MTP tensors. The main
model has the existing 12-layer vision tower and 24-layer hybrid Qwen3.5 decoder:
1024 text width, 768 vision width, 248320 vocabulary entries, three linear
attention layers followed by full attention, and tied input/output embeddings.
The pinned Transformers 5.17.0
[original implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
explicitly ignores `^mtp.*` while loading ordinary generation. The converter
checks all 15 exact names and shapes, verifies the complete checkpoint hash,
records every omitted name, then checks the remaining 473 tensors against the
original model instantiated on the meta device. Unknown names remain errors.
This does not implement speculative decoding.

The source model card declares Apache-2.0 and names
[Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
as its base. That exact base revision includes an Apache-2.0 license. The Xiaomi
model repository has no separate LICENSE file; the audited
[upstream code repository](https://github.com/SeerRay-Lab/Xiaomi-OCR-0/tree/b1a9bb4c07060c453785e9151715ccda6b96f0b1)
contains an Apache-2.0 license. Source fixtures retain both licenses.

The source has neither `generation_config.json` nor
`preprocessor_config.json`. Their absence is preserved. The nested image
processor is in `processor_config.json`. Source analysis of
`GenerationConfig.from_model_config` selects the non-default outer EOS 248046,
not the nested text EOS 248044. Hosted metadata validation must reproduce that
resolution before export. It also checks the effective tokenizer: the serialized
tokenizer and Qwen2Tokenizer class use the plain-letter split regex, while
`tokenizer_config.json` declares a combining-mark variant. A mismatch rejects;
the converter does not silently rewrite either source file.

## Native contract

The export uses architecture `qwen35_ocr`, format 4, profile `xiaomi-ocr-0`.
Admission requires all three gates:

- `CRISPEMBED_QWEN35_OCR=1`
- `CRISPEMBED_XIAOMI_OCR0_FRONTEND=1`
- `CRISPEMBED_XIAOMI_OCR0_INFERENCE=1`

`0` disables each gate. F32 and F16 tensors are supported in this experimental
lane; quantized Xiaomi tensors are rejected. The four exact original task prompts
are accepted by `qwen35_ocr_recognize_file` / `qwen35_ocr_recognize_raw`:

| Task | Prompt | Raw result |
|---|---|---|
| Text | `Extract the text in the image.` | Text |
| Table | `Parse the table in the image into OTSL.` | OTSL |
| Formula | `Identify the formula in the image and represent it using LATEX format.` | LaTeX |
| Document | The exact long document prompt in the source contract; an empty caller prompt selects it | Markdown with OTSL tables and LaTeX formulas |

The native message is image first, then task text, with the original non-thinking
assistant prefix. Only EOS 248046 stops this HF profile. The context uses the
existing 4096 total-token limit and at most 1024 vision patches; the default
output cap is 4096, but total context can stop generation sooner. The original
processor permits much larger images. Native inputs beyond the implemented
patch bound reject rather than receiving an implicit smaller resolution. These
limits are material for whole documents and require separate measured expansion.

Existing LightOn and Ovis profile admission, numerical operations, stop policies,
and qualification records are preserved. The two diagnostic callbacks accept
read-only actual Xiaomi logits and prefill stages. They default to null and
cannot be attached to another profile; no reference tensor is passed into the
production inference path.

## The two-stage document dependency

The original document pipeline uses **PP-DocLayoutV3**, region crops, task-specific
recognition, and page assembly. It is not implemented by registering Xiaomi as a
whole-page OCR engine. The source audit pins the actual Transformers detector:
`PaddlePaddle/PP-DocLayoutV3_safetensors` at
`e4f489dc536556fc1be02b973ba06756e0d4a2ac`, including its config and
133270468-byte checkpoint SHA-256. No native PP-DocLayoutV3 graph is added here.
An existing generic layout detector is not accepted as this dependency.

`tools/xiaomi0_layout_contract.py` is the bounded, hosted **original layout**
producer. It verifies the actual PP checkpoint, imports the pinned upstream
layout postprocessor, executes `PPDocLayoutV3ForObjectDetection`, and preserves
region order, labels, normalized boxes, skipped visual regions, and exact region
prompts. More than the requested region budget rejects instead of truncating.
It returns normalized `[0,1]` boxes for evaluation and preserves original
`[0,1000]` boxes for cropping. It requires a local checkout of the exact upstream
source revision and the exact layout checkpoint directory.

The upstream vLLM region request places text before the image and defaults to
16384 output tokens. The HF profile here places the image first and has the
native limits above. The contracts record that difference. Neither a successful
region run nor this external layout producer reproduces the upstream end-to-end
benchmark. Native layout, polygon/PaddleX variants, page assembly, full-document
scoring, and vLLM parity remain separate work.

## Hosted verification recipe

The original README pins Transformers 5.17.0, torch 2.14.0, and torchvision
0.29.0. The hosted recipe additionally requires `huggingface_hub`, `safetensors`,
`gguf`, `numpy`, `pillow`, and the repository's build dependencies. Layout reference
production additionally needs the pinned upstream Python dependencies.

```bash
python tools/ci-heavy/xiaomi0_reference.py --output "$RUNNER_TEMP/xiaomi0-metadata"
python tools/ci-heavy/xiaomi0_reference.py --output "$RUNNER_TEMP/xiaomi0-parity" --inference
```

Both commands require hosted CI. They first verify source files, loaded framework
sources, actual generation defaults, prompts, tokenizer, meta tensor schema, and
original/native image preprocessing. The second then executes the frozen
[two-image protocol](../tests/fixtures/xiaomi0-bounded/protocol.json): exactly two
original F32 cases and the same two native cases in F32 and F16, each capped at
16 generated tokens. It uses portable CPU builds with both native compiler flags
disabled. No quantization is attempted.

The result retains every generated logit row, 40 prefill stage captures, IDs,
text, confidence, stop reason, cap, source and image hashes. It reports the
owner's F32 `>0.99` / F16 `>0.98` final-logit bars beside strict `0.99999` / `0.002`
stage diagnostics and decoded agreement. Different generation prefixes are
recorded explicitly. The small protocol remains an integration experiment even
if every measured comparison passes. It does not change old failure evidence or
establish benchmark, full-document, or publication qualification.

Local checks use no framework, model weights, native build, or inference:

```bash
python -m unittest discover -s tests -p 'test_xiaomi0_*.py' -v
python tools/qwen35_source_pins.py --check
```
