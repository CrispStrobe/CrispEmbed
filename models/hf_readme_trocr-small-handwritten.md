---
license: mit
language:
- en
tags:
- text-recognition
- ocr
- trocr
- handwriting
- gguf
- crispembed
base_model: microsoft/trocr-small-handwritten
pipeline_tag: image-to-text
---

# TrOCR-small Handwritten Text — GGUF

Handwritten text recognition model for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Recognizes handwritten text from cropped text-line images. Outputs **mixed-case** text
(unlike trocr-small-printed which uppercases).

**Architecture**: DeiT-small encoder (12L, 384d, 6 heads) + TrOCR decoder
(6L, 256d, 8 heads). XLM-R vocabulary (64,044 tokens). 61M parameters.

**Source**: [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten) (MIT).

## Model Variants

| Variant | Size | Recognition quality |
|---------|------|-------------------|
| F32 | 235 MB | exact token match vs HuggingFace (greedy) |
| F16 | 118 MB | identical to F32 |
| **Q8_0** | **63 MB** | **identical to F32** |

Q4_K not provided — 256d decoder bottleneck is too narrow for 4-bit quantization.

## Usage

```bash
# Single crop recognition
crispembed -m trocr-small-handwritten-q8_0.gguf --ocr crop.png

# Full pipeline with DBNet detection
crispembed --det dbnet-ic15-q4_k.gguf -m trocr-small-handwritten-q8_0.gguf --ocr scan.png
```

## License

MIT (same as [microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten)).
