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
base_model: microsoft/trocr-small-printed
pipeline_tag: image-to-text
---

# TrOCR-small Printed Text — GGUF

Text recognition model for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Recognizes printed text from cropped text-line images. Pair with a text detector
like [cstr/dbnet-ic15-GGUF](https://huggingface.co/cstr/dbnet-ic15-GGUF) for
end-to-end OCR.

**Architecture**: DeiT-small encoder (12L, 384d, 6 heads) + TrOCR decoder
(6L, 256d, 8 heads). XLM-R vocabulary (64,044 tokens). 61M parameters.

**Source**: [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed) (MIT).

## Model Variants

| Variant | Size | Recognition quality |
|---------|------|-------------------|
| F32 | 235 MB | exact match vs HuggingFace |
| F16 | 119 MB | exact same tokens |
| **Q8_0** | **65 MB** | **exact same tokens** |

**Recommended: Q8_0** (65 MB). Q4_K is not provided — the 256-dim decoder
bottleneck is too narrow for 4-bit quantization, causing recognition errors.

### Verification (all variants produce identical output)

| Input image | Output |
|-------------|--------|
| "Hello World" | HELLO WORLD |
| "The quick brown fox" | THE QUICK BROWN FOX |
| "42 is the answer" | 42 IS THE ANSWER |

Note: trocr-small-printed uppercases output (training data bias). For
mixed-case, use a trocr-base model.

## Usage

### Full OCR pipeline (with DBNet)
```bash
crispembed --det dbnet-ic15-q4_k.gguf \
    -m trocr-small-printed-q8_0.gguf \
    --ocr document.png
```

Output:
```
[ 0] (49,53)-(143,86)   conf=0.91  "HELLO"
[ 1] (153,52)-(270,86)  conf=0.91  "WORLD!"
[ 2] (50,122)-(124,157) conf=0.91  "THIS"
...
```

### C API
```c
#include "crispembed.h"

void *ctx = crispembed_ocr_init("dbnet-ic15-q4_k.gguf",
                                 "trocr-small-printed-q8_0.gguf", 4);
int n;
const crispembed_ocr_result *r = crispembed_ocr(ctx, "document.png", &n);
for (int i = 0; i < n; i++)
    printf("%s ", r[i].text);
crispembed_ocr_free(ctx);
```

### Pipeline size

| Detection | Recognition | Total | Throughput |
|-----------|-------------|-------|-----------|
| Q4_K (7 MB) | Q8_0 (65 MB) | **72 MB** | ~200ms/region |

## Architecture

```
Input: text crop (resized to 384x384, grayscale)
  |
  +-> DeiT-small encoder (12 layers)
  |     16x16 patch embedding -> 576+2 tokens (CLS + distillation)
  |     12x Pre-LN MHA (6 heads, 384d) + FFN (GELU, 1536d)
  |
  +-> TrOCR decoder (6 layers, autoregressive)
        Token + position embedding (64044 BPE vocab, 514 max positions)
        6x Self-attn (causal) + Cross-attn + FFN
        -> greedy argmax -> SentencePiece BPE detokenize
```

XLM-R SentencePiece tokenizer with fairseq vocab offset. Word boundaries
marked by `▁` (U+2581), converted to spaces at decode time.

## Conversion

```bash
pip install gguf numpy transformers sentencepiece safetensors

# Download model
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('microsoft/trocr-small-printed', local_dir='trocr-small-printed')"

# Convert (embeds XLM-R tokenizer via AutoTokenizer)
python models/convert-trocr-to-gguf.py \
    --model-dir trocr-small-printed/ \
    --output trocr-small-printed-f32.gguf

# Quantize (Q8_0 recommended; Q4_K degrades this model)
crispembed-quantize trocr-small-printed-f32.gguf trocr-small-printed-q8_0.gguf q8_0
```

## License

MIT (same as [microsoft/trocr-small-printed](https://huggingface.co/microsoft/trocr-small-printed)).
