---
license: apache-2.0
language:
- en
tags:
- text-detection
- ocr
- dbnet
- gguf
- crispembed
base_model: open-mmlab/mmocr
pipeline_tag: object-detection
---

# DBNet ResNet-18 ICDAR 2015 — GGUF

Text detection model for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Detects text regions (words/lines) in document images, scene photos, and screenshots.

**Architecture**: DBNet (Differentiable Binarization Network) with ResNet-18 backbone,
FPNC neck, and probability map head. All BatchNorm folded into Conv at export time.

**Source**: [MMOCR dbnet_resnet18_fpnc_1200e_icdar2015](https://github.com/open-mmlab/mmocr)
(Apache 2.0). Trained on ICDAR 2015 scene text dataset (P=0.885, R=0.758, H=0.817).

## Model Variants

| Variant | Size | Cosine vs F32 | max_abs | Detection quality |
|---------|------|--------------|---------|-------------------|
| F32 | 47 MB | baseline | — | reference |
| F16 | 24 MB | 1.000000 | 2.4e-3 | identical |
| **Q8_0** | **13 MB** | 1.000000 | 1.9e-2 | identical (21/21 regions) |
| **Q4_K** | **7 MB** | 1.000000 | 2.9e-1 | identical (21/21 regions) |

All variants detect the same text regions with scores within 0.005 of F32.
Parity validated via per-pixel diff harness against PyTorch reference.

**Recommended**: Q4_K (7 MB) for production, Q8_0 (13 MB) for maximum fidelity.

## Usage

Pair with a TrOCR recognition model ([cstr/trocr-small-printed-GGUF](https://huggingface.co/cstr/trocr-small-printed-GGUF))
for a complete OCR pipeline.

### CLI
```bash
# Full OCR pipeline (detect + recognize)
crispembed --det dbnet-ic15-q4_k.gguf \
    -m trocr-small-printed-q8_0.gguf \
    --ocr document.png

# JSON output
crispembed --det dbnet-ic15-q4_k.gguf \
    -m trocr-small-printed-q8_0.gguf \
    --ocr document.png --json
```

### C API
```c
#include "crispembed.h"

void *ctx = crispembed_ocr_init("dbnet-ic15-q4_k.gguf",
                                 "trocr-small-printed-q8_0.gguf", 4);
int n;
const crispembed_ocr_result *r = crispembed_ocr(ctx, "image.png", &n);
for (int i = 0; i < n; i++)
    printf("(%g,%g): %s\n", r[i].x, r[i].y, r[i].text);
crispembed_ocr_free(ctx);
```

## Architecture

```
Input image (resized, padded to 32x)
  |
  +-> ResNet-18 backbone (stem + 4 stages x 2 BasicBlocks)
  |     Stage 0: 64ch, stride 4   Stage 1: 128ch, stride 8
  |     Stage 2: 256ch, stride 16  Stage 3: 512ch, stride 32
  |
  +-> FPNC neck (FPN-Cat variant)
  |     4x lateral 1x1 conv -> top-down upsample+add
  |     4x smooth 3x3 conv (256->64) -> concat 4x64=256ch
  |
  +-> DBHead probability branch
        3x3 conv (256->64) + ReLU
        ConvTranspose2d (64->64, k=2, s=2) + ReLU
        ConvTranspose2d (64->1, k=2, s=2) + sigmoid
```

Post-processing: binarize at 0.3 -> connected components -> bbox extraction
with unclip expansion (ratio 1.5). Output sorted in reading order.

12.2M parameters. All BatchNorm pre-folded into Conv/ConvTranspose weights.

## Conversion

```bash
pip install gguf numpy torch mmengine

# Download MMOCR checkpoint
wget -q "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet18_fpnc_1200e_icdar2015/dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth"

# Convert
python models/convert-dbnet-to-gguf.py \
    --checkpoint dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth \
    --output dbnet-ic15-f32.gguf

# Quantize
crispembed-quantize dbnet-ic15-f32.gguf dbnet-ic15-q8_0.gguf q8_0
crispembed-quantize dbnet-ic15-f32.gguf dbnet-ic15-q4_k.gguf q4_k
```

## License

Apache 2.0 (same as the [MMOCR](https://github.com/open-mmlab/mmocr) source).
