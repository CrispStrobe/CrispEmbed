---
license: other
license_name: tongyi-qianwen-research
license_link: LICENSE
base_model: stepfun-ai/GOT-OCR2_0
base_model_relation: quantized
tags:
- gguf
- ocr
- crispembed
- got-ocr2
- research-only
library_name: crispembed
---
# GOT-OCR2 GGUF (CrispEmbed)

GGUF conversions of [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0/tree/979938bf89ccdc949c0131ddd3841e24578a4742)
for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)'s native ggml runtime. The original
authors get full credit for the model. These files change only the storage format and, where
noted, the numeric precision.

> **Research and evaluation use only.** You may not use these files commercially, and you may not
> use the model or its output to improve any other large language model. See the license section.
>
> **Use `got-ocr2-f16-r2.gguf`.** It is the variant CrispEmbed recommends, and the only one whose
> output exactly matches the original model. **`got-ocr2-q4_k.gguf` is EXPERIMENTAL:** its output
> drifts from the original's on two of the three test cases.

## License and attribution

- GOT-OCR2's decoder derives from [Qwen/Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B), which is licensed under the
  **Tongyi Qianwen RESEARCH LICENSE AGREEMENT**. The upstream `apache-2.0` tag cannot relax the license of the base model.
  These derived files are redistributed under the same research license.
  Evidence: [license audit](https://github.com/CrispStrobe/CrispEmbed/blob/main/tests/results/licenses/registry-qwen-audit.json).
- `LICENSE` is a verbatim copy of the license text from `Qwen/Qwen1.5-0.5B` at revision `8f445e3628f3500ee69f24e1303c9f10f5342a39`
  (7,281 bytes, SHA256 `41c2cf8c272f6fb0080a97cd9d9bd7d4604072b80a0b10e7d65ca26ef5000c0c`). The GOT-OCR2_0 repository ships no LICENSE file.
- `NOTICE` is the "Notice" text file that section 2(c) of that license requires. Its first line is the required
  attribution, verbatim: "Tongyi Qianwen is licensed under the Tongyi Qianwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."
- `UPSTREAM_README.md` is the unmodified upstream model card at revision `979938bf89ccdc949c0131ddd3841e24578a4742`
  (3,959 bytes, SHA256 `1c8697ddeb841080c25a06b2eadd9dbdf714cdef886fb457a2568b57b2079134`).
- CrispEmbed loads these files only when you set `CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1`.
- An earlier version of this card described the license as apache-2.0, named a Qwen2-0.5B decoder and recommended Q4_K. All three
  statements were wrong, and this card replaces them.

## Statement of changes (license section 2(b))

CrispEmbed modified these files from the original:

- The checkpoint weights of `stepfun-ai/GOT-OCR2_0` revision `979938bf89ccdc949c0131ddd3841e24578a4742` were
  **format-converted** to GGUF. The conversion renamed tensors, cast them to other types, and embedded the tokenizer and
  preprocessing metadata.
- The `q8_0` and `q4_k` files were also **quantized** from the F16 GGUF with `crispembed-quantize`.

No weights were trained, fine-tuned, merged or otherwise altered.

## Files

| File | Status | Bytes | SHA256 |
|---|---|---:|---|
| `got-ocr2-f16-r2.gguf` | **recommended**: exact match with the original | 1,441,830,112 | `98b431abc3e6b5715363e7296f7fc665a2eca829127ccb8693511926c4784a7f` |
| `got-ocr2-q4_k.gguf` | **experimental**: drifts from the original | 445,492,288 | `ba276f2e246273af7d1bcd58f35ac7bae09c30340b923ed4440effe82d5e7613` |
| `got-ocr2-q8_0.gguf` | unqualified: not measured against the original | 599,632,960 | `15920499f65b71bcfbf4d44d0e92ffbaf7f7c4523ad13f7b7bef160dc5edf098` |
| `got-ocr2-f16.gguf` | **defective, do not use**: kept for provenance | 1,441,830,112 | `aeaf1b44925eeb449584d716f181fc486cdd6becf596ec896276c5d633e996bb` |
| `got-ocr2-ref-full.gguf` | diagnostic reference dump | 170,013,952 | `27de920f22b7b44c65d4fb32087e51f35b7bc38c9e45d438b2251df05849ffe5` |
| `LICENSE` | verbatim Tongyi Qianwen RESEARCH LICENSE | 7,281 | `41c2cf8c272f6fb0080a97cd9d9bd7d4604072b80a0b10e7d65ca26ef5000c0c` |
| `NOTICE` | section 2(c) attribution | 391 | `c3fe8e03056e0610a7f69c38e12066bfe7e4931223d046eaf5596604788ee892` |
| `UPSTREAM_README.md` | upstream card, unmodified | 3,959 | `1c8697ddeb841080c25a06b2eadd9dbdf714cdef886fb457a2568b57b2079134` |

The original `got-ocr2-f16.gguf` contains an 8 MiB block of zero bytes inside `l.embed_tokens.weight` (token rows 43-2091).
As a result it outputs repeated `colorcolor...` on every backend. `got-ocr2-f16-r2.gguf` is a fresh conversion. CrispEmbed
read back every one of its 471 tensors and found each byte-identical to the original checkpoint.

## Quality against the original model

The reference is the original `stepfun-ai/GOT-OCR2_0` `chat()` remote code, run on CPU in F32 with transformers 4.37.2 and
greedy decoding. The panel has three cases: `fox`, `receipt_example` and `simple_form`. The figures come from hosted run
41588457181 ([receipt](https://github.com/CrispStrobe/CrispEmbed/tree/main/tests/results/got-ocr2-f16/qualify-41588457181-orig-ref)).

| Variant | Status | Token-ID agreement with the original | First divergent token | Character ratio vs original | Stage cosine vs original (min) | Final-logit cosine |
|---|---|---|---|---|---|---|
| F16-r2 | recommended | 3/3 cases exact | none | 1.0 / 1.0 / 1.0 | 1.000000 (19 stages) | not measured |
| Q4_K | **experimental** | 1/3 cases exact (`fox`) | `receipt_example` token 83, `simple_form` token 20 | 1.0 / 0.8041 / 0.9679 | vision 0.997745, decoder 0.995157 | not measured |
| Q8_0 | unqualified | not measured | not measured | not measured | not measured | not measured |

The Q4_K drift is a real loss of quality, not merely a numerical difference. On `receipt_example` the text diverges after
token 83, and only about 80% of the characters match. Use F16-r2 when results matter. CrispEmbed has not yet run a
mixed-precision Q4 search to find a recipe that matches the original. Until one exists, Q4_K stays experimental.

Not verified: GPU backends other than the M1 F16 receipt, multi-page documents, and task-level benchmarks.

## Usage

```bash
CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1 crispembed --ocr got-ocr2-f16 image.png
```

## Provenance and EU AI Act Art. 53 note

- **Upstream model:** [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0), published by `stepfun-ai`.
  The decoder base is Qwen/Qwen1.5-0.5B.
- **What was done here:** format conversion and quantization only. There was no training, fine-tuning, merging or
  distillation.
- **Provider status:** the upstream authors remain the provider of this model. Training data is documented upstream, if
  anywhere.
