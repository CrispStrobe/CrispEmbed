# PP-OCRv6

CrispEmbed supports the official PP-OCRv6 model family through one GGUF
converter and one runtime family:

| tier | detector | recognizer |
|---|---|---|
| tiny | `PP-OCRv6_tiny_det` | `PP-OCRv6_tiny_rec` |
| small | `PP-OCRv6_small_det` | `PP-OCRv6_small_rec` |
| medium | `PP-OCRv6_medium_det` | `PP-OCRv6_medium_rec` |

The source repositories are the official
`PaddlePaddle/PP-OCRv6_*_safetensors` repositories. Keep source checkpoints,
F16 GGUFs, quantized GGUFs, and parity fixtures on the external model volume:

```text
/Volumes/backups/ai/crispembed-gguf/
```

Convert one model with:

```bash
python models/convert-ppocrv6-to-gguf.py \
  --model-dir /Volumes/backups/ai/crispembed-gguf/source/PP-OCRv6_small_rec_safetensors \
  --output /Volumes/backups/ai/crispembed-gguf/PP-OCRv6_small_rec-f16.gguf
```

The converter folds inference BatchNorm into convolution weights. PP-OCRv6's
policy-q4 deployment files intentionally retain the complete detector or
recognizer graph in F16: quantizing intermediate CNN/SVTR weights compounds
error through the CTC path and fails parity. The policy therefore prioritizes
quality over file-size reduction for these compact models.

F32 small/medium conversions match the native reference through logits, while
the published F16 artifacts accumulate measurable drift through repeated
layers. A first true Q8 experiment (pointwise CNN/SVTR weights quantized,
sensitive tensors retained) degraded small-rec logit cosine to about 0.59, so
Q8 is not enabled by default until a layer-selective scheme passes real
line-crop quality tests. Q4 remains an explicit debug-only policy variant.

The official v6 preprocessing is also load-bearing: recognizers use a 48-pixel
height, aspect-ratio-preserving width with padding up to 320 pixels, RGB
conversion, rescaling by `1/255`, and the model's declared normalization. Text
detection uses the v6 736-pixel minimum-side policy and ImageNet channel
normalization. These values must remain part of the parity fixtures.

Dump the current torch reference fixture and enable native comparisons with:

```bash
python tools/dump_ppocrv6_reference.py \
  --model-dir /Volumes/backups/ai/crispembed-gguf/source/PP-OCRv6_tiny_rec_safetensors \
  --image /path/to/line.png \
  --output /Volumes/backups/ai/crispembed-gguf/PP-OCRv6_tiny_rec-ref.gguf
PPOCRV6_REF=/Volumes/backups/ai/crispembed-gguf/PP-OCRv6_tiny_rec-ref.gguf \
  ./build/crispembed -m /Volumes/backups/ai/crispembed-gguf/PP-OCRv6_tiny_rec-f16.gguf \
  --ocr /path/to/line.png
```

The same comparison is available as a standalone regression binary:

```bash
PPOCRV6_REF=/Volumes/backups/ai/crispembed-gguf/PP-OCRv6_tiny_rec-ref.gguf \
  ./build/test-ppocrv6-rec \
  /Volumes/backups/ai/crispembed-gguf/PP-OCRv6_tiny_rec-f16.gguf \
  /path/to/line.png
```
