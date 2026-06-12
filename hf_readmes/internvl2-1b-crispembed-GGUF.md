---
license: mit
language:
  - en
  - de
  - zh
  - ja
  - ko
tags:
  - ocr
  - document-understanding
  - vision-language-model
  - gguf
  - crispembed
base_model: OpenGVLab/InternVL2-1B
library_name: gguf
pipeline_tag: image-text-to-text
---

# InternVL2-1B — CrispEmbed GGUF

GGUF conversions of [OpenGVLab/InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

Smallest competitive VLM for OCR — ideal for edge, mobile, and WASM deployment.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | InternVL2 (InternViT-300M + Qwen2-0.5B) |
| Total Parameters | ~0.9B |
| Vision Encoder | InternViT-300M-448px (24L, 1024d, identical to InternVL2.5-2B) |
| Projector | Pixel unshuffle (4:1) + LayerNorm + Linear + GELU + Linear |
| LLM Decoder | Qwen2-0.5B-Instruct (24L, 896d, GQA 14/2, SwiGLU, RMSNorm) |
| Input Resolution | 448x448 per tile, dynamic tiling (1-12 tiles) |
| License | MIT |
| OCRBench | 779 |

## Available Quantizations

| File | Size | Compression | Notes |
|------|------|-------------|-------|
| `internvl2-1b-f16.gguf` | 2.3 GB | 1x | Full precision |
| `internvl2-1b-q8_0.gguf` | 955 MB | 2.4x | Good quality |
| `internvl2-1b-q4_k.gguf` | ~600 MB | ~4x | Smallest, vision Q8_0 floor |

## Parity Verification

All components verified against Python reference (cos=1.000000):
- Vision encoder: 4/4 layers PASS
- Projector: PASS
- LLM decoder (Qwen2): 2/2 layers PASS

## Credits

- Original model: [OpenGVLab/InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B) (MIT)
- GGUF conversion: [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
