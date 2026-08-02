---
license: apache-2.0
language:
- en
- de
- fr
- es
- it
- pt
- nl
- ru
- ar
- zh
- ja
- ko
tags:
- gguf
- ocr
- tesseract
- lstm
- crispembed
library_name: crispembed
pipeline_tag: image-to-text
---

# Tesseract LSTM OCR — GGUF models for CrispEmbed

Apache-2.0 Tesseract LSTM line-recognition models converted to GGUF for
CrispEmbed. The repository contains English, German, French, Spanish,
Italian, Portuguese, Dutch, Russian, Arabic, Simplified Chinese, Japanese,
and Korean variants.

These are line recognizers: normalize and crop individual text lines before
recognition, or pair them with a detector such as DBNet for full-page OCR.

## Precision variants

Each language has an F32 reference, an F16 deployment model, and Q8_0/Q4_K
variants. The quantized models preserve `output.weight` and `output.bias` at
the source precision (F16 for the multilingual F16-derived models; F32 for
German regenerated from its original `.traineddata`). Only recurrent matrices
are quantized. This preserves the CTC decision boundary and keeps critical
character logits stable.

| Variant | Purpose |
|---|---|
| `*-f32.gguf` | Canonical reference and parity baseline |
| `*-f16.gguf` | High-fidelity deployment |
| `*-q8_0.gguf` | Recommended compact deployment |
| `*-q4_k.gguf` | Smallest deployment; validate on the target corpus |

## Usage

```bash
crispembed -m tesseract-eng-q8_0.gguf --ocr line.png
```

The native runtime performs height normalization and CTC greedy decoding. Word
spacing and page reading order must be supplied by the surrounding OCR
pipeline; the Tesseract DAWG language models are not part of this GGUF graph.
The converter also supports an opt-in `--embed-dawgs` mode that preserves the
three LSTM DAWG components as GGUF metadata for future runtime scoring; the
current runtime intentionally ignores those arrays.

## Conversion

```bash
python models/convert-tesseract-to-gguf.py \
  --model eng.traineddata --output tesseract-eng-f32.gguf

# Optional: retain the source LSTM DAWG bytes for a future scorer.
python models/convert-tesseract-to-gguf.py \
  --model eng.traineddata --output tesseract-eng-dawg-f32.gguf --embed-dawgs

crispembed-quantize tesseract-eng-f32.gguf tesseract-eng-q8_0.gguf q8_0
crispembed-quantize tesseract-eng-f32.gguf tesseract-eng-q4_k.gguf q4_k
```

The exact source URL/revision and SHA-256 are stored in each GGUF's metadata.

## License

Apache-2.0, following the upstream Tesseract language data. Preserve the
upstream attribution and source checksum when redistributing derivatives.
