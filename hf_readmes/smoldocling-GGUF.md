---
license: apache-2.0
base_model: ds4sd/SmolDocling-256M-preview
tags:
  - ocr
  - document-understanding
  - doctags
  - vision-language
  - gguf
  - crispembed
  - ggml
language:
  - en
---

# SmolDocling-256M GGUF

GGUF conversions of [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) inference.

Ultra-compact document conversion model (256M params). Generates DocTags structured markup from page images — OCR, layout, tables, formulas, code, charts.

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `smoldocling-f16.gguf` | F16 | 491 MB | Full precision |
| `smoldocling-q8_0.gguf` | Q8_0 | 261 MB | Recommended |
| `smoldocling-q4_k.gguf` | Q4_K | 153 MB | Max compression |

## Architecture

- **Vision**: SigLIP ViT (12L, 768d, 12 heads, patch=16, 512px)
- **Connector**: Pixel shuffle (scale=4, 1024→64 tokens) + Linear(12288→576)
- **LLM**: SmolLM2-135M (30L, 576d, GQA 9/3, SwiGLU, RoPE)
- **Parameters**: 256M total (93M vision + 135M LLM + connector)
- **Output**: DocTags (structured XML-like document markup)

Parity vs HF reference: vision cos=0.9998, connector cos=0.9999.

## Usage

```bash
# CLI
./crispembed -m smoldocling-q8_0.gguf --ocr document.png

# Server
./crispembed-server --ocr smoldocling-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/math/ocr -F "image=@document.png"
```

```python
from crispembed import CrispMathOcr

ocr = CrispMathOcr("smoldocling-q8_0.gguf")
doctags = ocr.recognize("document.png")
print(doctags)  # <doctag><text>...</text>...</doctag>
```

## License

Apache-2.0 — same as the base model.

## Credits

Original model by [Docling Team, IBM Research](https://huggingface.co/ds4sd). GGUF conversion and inference engine by [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
