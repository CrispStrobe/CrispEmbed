---
license: apache-2.0
tags:
  - gguf
  - ocr
  - scene-text
  - parseq
  - crispembed
base_model: baudm/parseq
---

# PARSeq — Scene Text Recognition (GGUF)

GGUF conversions of [PARSeq](https://github.com/baudm/parseq) (ECCV 2022) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

PARSeq is a scene text recognition model that reads text from natural images (signs, labels, documents). It recognizes 94 printable ASCII characters (digits, letters, punctuation).

## Architecture

- **Encoder**: 12-layer pre-LN ViT (patch 4×8, input 32×128 RGB, 128 tokens, GELU FFN)
- **Decoder**: 1-layer two-stream Transformer (XLNet-style position queries + context self-attention, then cross-attention to encoder memory)
- **Head**: Linear → 95 classes (94 printable ASCII chars + EOS)
- **Inference**: Autoregressive greedy decode (max 25 characters)

## Variants

| File | Variant | Params | Size | Notes |
|------|---------|--------|------|-------|
| `parseq-f32.gguf` | Base | 24M | 91 MB | Full precision |
| `parseq-q8_0.gguf` | Base | 24M | 24 MB | Best quantized |
| `parseq-q4_k.gguf` | Base | 24M | 13 MB | Smallest base |
| `parseq-tiny-f16.gguf` | Tiny | 6M | 12 MB | Half precision |
| `parseq-tiny-q8_0.gguf` | Tiny | 6M | 6 MB | Smallest overall |

All quantization levels produce identical output on test images.

## Usage

```bash
# CLI
crispembed -m parseq-q8_0.gguf --ocr image.png

# Auto-download
crispembed -m parseq --auto-download --ocr image.png
```

```python
from crispembed import CrispMathOcr
ocr = CrispMathOcr("parseq-q8_0.gguf")
text = ocr.recognize("sign.png")
```

## Benchmark (94-char, PARSeq-base)

| Dataset | Accuracy |
|---------|----------|
| IIIT5k | 99.1% |
| SVT | 97.9% |
| IC13-1015 | 98.1% |
| IC15-2077 | 89.2% |
| SVTP | 96.9% |
| CUTE80 | 98.6% |

## Source

- Paper: [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966) (ECCV 2022)
- Code: [baudm/parseq](https://github.com/baudm/parseq) (Apache-2.0)
- Converted with `models/convert-parseq-to-gguf.py` from CrispEmbed
