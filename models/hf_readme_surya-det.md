---
license: other
license_name: openrail-m
license_link: https://huggingface.co/datalab-to/surya-ocr-2/blob/main/LICENSE
language:
- multilingual
tags:
- text-detection
- ocr
- gguf
- crispembed
- efficientvit
base_model: datalab-to/surya-ocr-2
---

# surya-det-GGUF — Text Line Detection

GGUF conversion of the [surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) text detector
for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

## Architecture

EfficientViT-Large segformer (38M params):
- **Encoder**: Stem + 4 stages (FusedMBConv, MBConv, EfficientVitBlock with LiteMLA linear attention)
- **Decoder**: SegFormer-style FPN decode head (multi-scale projection + fuse + classify)
- **Input**: RGB image resized to 1200×1200 (ImageNet-normalized)
- **Output**: 2-channel heatmap [300×300] (text line, separator) → bounding boxes

## Files

| File | Type | Size | Notes |
|---|---|---|---|
| `surya-det-f32.gguf` | F32 | 147 MB | Full precision |
| `surya-det-f16.gguf` | F16 | 74 MB | Recommended |
| `surya-det-q8_0.gguf` | Q8_0 | 41 MB | 3.6x compression |
| `surya-det-q4_k.gguf` | Q4_K | 23 MB | 6.5x compression |

All BatchNorm layers are folded into preceding Conv2d weights.

## Language Support

91 languages including English, German (89.7%), Chinese, Arabic, Japanese, Korean.
See [surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) for the full list.

## Usage with CrispEmbed

```cpp
#include "surya_det.h"

surya_det_context* ctx = surya_det_init("surya-det-f16.gguf", 4);
int hm_h, hm_w;
const float* heatmap = surya_det_detect(ctx, pixels, width, height, 3, &hm_h, &hm_w);

int n_boxes;
const surya_det_bbox* boxes = surya_det_get_boxes(ctx, width, height, 0.6f, 0.35f, &n_boxes);
for (int i = 0; i < n_boxes; i++) {
    printf("Text at (%.0f,%.0f)-(%.0f,%.0f) conf=%.2f\n",
           boxes[i].x0, boxes[i].y0, boxes[i].x1, boxes[i].y1, boxes[i].confidence);
}
surya_det_free(ctx);
```

## Python

```python
from crispembed import CrispTextDetect

det = CrispTextDetect("surya-det-f16.gguf")
boxes = det.detect("document.png")
for b in boxes:
    print(f"({b['x0']:.0f},{b['y0']:.0f})-({b['x1']:.0f},{b['y1']:.0f}) conf={b['confidence']:.3f}")
```

## Parity

Verified against Python reference (identical preprocessed input):
- Heatmap max: 0.9649 (exact match)
- Heatmap mean: 0.0113 (exact match)
- Per-stage activation means match to 4 decimal places

## License

OpenRail-M — free for organizations under $5M revenue/funding.
See [LICENSE](https://huggingface.co/datalab-to/surya-ocr-2/blob/main/LICENSE).

## Source

Converted from [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2)
(surya.detection.model.encoderdecoder.EfficientViTForSemanticSegmentation).
