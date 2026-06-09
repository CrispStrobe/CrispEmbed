---
license: mit
language:
- en
tags:
- text-recognition
- ocr
- trocr
- gguf
- crispembed
base_model: microsoft/trocr-base-printed
pipeline_tag: image-to-text
---

# TrOCR-base Printed Text — GGUF

Text recognition model for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Larger and more capable than trocr-small-printed. Recognizes printed text from
cropped text-line images. Uses GPT-2 BPE tokenizer (50,265 tokens).

**Architecture**: BEiT encoder (12L, 768d, 12 heads) + TrOCR decoder
(12L, 1024d, 16 heads). 333M parameters. Tied embeddings (lm_head = embed_tokens).

**Source**: [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) (MIT).

## Model Variants

| Variant | Size | Recognition quality |
|---------|------|-------------------|
| F32 | 1.3 GB | exact token match vs HuggingFace (greedy) |
| F16 | 639 MB | identical to F32 |
| **Q8_0** | **340 MB** | **identical to F32** |

Q4_K not tested — d_model=1024 should handle it better than small (256d),
but Q8_0 is recommended for this model size.

## Usage

Pair with [cstr/dbnet-ic15-GGUF](https://huggingface.co/cstr/dbnet-ic15-GGUF) for end-to-end OCR.

```bash
crispembed --det dbnet-ic15-q4_k.gguf -m trocr-base-printed-q8_0.gguf --ocr document.png
```

## License

MIT (same as [microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)).
