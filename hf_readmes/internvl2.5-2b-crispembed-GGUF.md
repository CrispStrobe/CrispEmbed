---
license: mit
language:
  - en
  - de
  - zh
  - ja
  - ko
  - fr
  - es
  - pt
tags:
  - ocr
  - document-understanding
  - vision-language-model
  - gguf
  - crispembed
base_model: OpenGVLab/InternVL2_5-2B
library_name: gguf
pipeline_tag: image-text-to-text
---

# InternVL2.5-2B — CrispEmbed GGUF

GGUF conversions of [OpenGVLab/InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | InternVL2.5 (InternViT-300M + InternLM2.5-1.8B) |
| Total Parameters | ~2.1B |
| Vision Encoder | InternViT-300M-448px (24L, 1024d, 16H, LayerNorm + GELU + LayerScale) |
| Projector | Pixel unshuffle (4:1) + LayerNorm + Linear + GELU + Linear |
| LLM Decoder | InternLM2.5-1.8B-chat (24L, 2048d, GQA 16/8, SwiGLU, RMSNorm) |
| Input Resolution | 448x448 per tile, dynamic tiling (1-12 tiles) |
| License | MIT |
| OCRBench | ~830 (top tier for <3B models) |

## Available Quantizations

| File | Size | Compression | Notes |
|------|------|-------------|-------|
| `internvl2.5-2b-f16.gguf` | 4.9 GB | 1x | Full precision (F16 weights, F32 norms/embeds) |
| `internvl2.5-2b-q8_0.gguf` | 2.2 GB | 2.2x | Good quality, vision weights at Q8_0 floor |
| `internvl2.5-2b-q4_k.gguf` | 880 MB | 5.6x | Smallest, vision weights kept at Q8_0 minimum |

**Note:** Vision encoder weights are kept at Q8_0 minimum even in Q4_K to preserve OCR accuracy. The Q4_K savings come primarily from the LLM decoder.

## Usage with CrispEmbed

```c
#include "crispembed.h"

// Auto-detects InternVL2 architecture from GGUF metadata
void *ctx = crispembed_math_ocr_init("internvl2.5-2b-q4_k.gguf", 4);

int len;
const char *text = crispembed_math_ocr_recognize(ctx, pixels, w, h, channels, &len);
printf("%s\n", text);

crispembed_math_ocr_free(ctx);
```

```python
from crispembed import CrispMathOcr

ocr = CrispMathOcr("internvl2.5-2b-q4_k.gguf")
text = ocr.recognize("document.png")
```

## Parity Verification

All components verified against the Python reference implementation:

| Stage | cos_sim | max_abs_diff |
|-------|---------|--------------|
| vis_patch_embed | 1.000000 | 0.000003 |
| vis_layer_0..3 | 1.000000 | <0.001 |
| vis_proj_output | 1.000000 | 0.000909 |
| llm_embed | 1.000000 | 0.000000 |
| llm_layer_0..1 | 1.000000 | <0.000005 |

## Conversion

Converted using `models/convert-internvl2-to-gguf.py` from CrispEmbed:

```bash
python models/convert-internvl2-to-gguf.py \
    --model OpenGVLab/InternVL2_5-2B \
    --output internvl2.5-2b-f16.gguf --dtype f16

# Then quantize with the C++ quantizer:
./crispembed-quantize internvl2.5-2b-f16.gguf internvl2.5-2b-q8_0.gguf q8_0
./crispembed-quantize internvl2.5-2b-f16.gguf internvl2.5-2b-q4_k.gguf q4_k
```

## Architecture

```
Image (448x448 per tile, 1-12 tiles)
  → Conv2D patch embed (14x14, stride 14) → 1024 patches
  → Prepend CLS + position embedding
  → 24x InternViT blocks (LayerNorm → MHSA → LayerScale → residual
                           LayerNorm → GELU MLP → LayerScale → residual)
  → Remove CLS → pixel unshuffle (4:1, 1024→256 tokens, dim 1024→4096)
  → LayerNorm → Linear(4096→2048) → GELU → Linear(2048→2048)
  → Splice into text token sequence
  → 24x InternLM2.5 blocks (RMSNorm → GQA(16/8) + RoPE → residual
                              RMSNorm → SwiGLU FFN → residual)
  → RMSNorm → LM head → logits → greedy decode
```

## Credits

- Original model: [OpenGVLab/InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B) (MIT License)
- GGUF conversion: [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
