---
license: other
license_name: lfm-1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE
base_model: VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER
tags:
  - ner
  - named-entity-recognition
  - gliner
  - zero-shot
  - gguf
  - crispembed
  - ggml
language:
  - en
  - de
  - fr
  - it
  - es
---

# GLiNER-LFM GGUF

GGUF conversions of [VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER](https://huggingface.co/VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER) for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) inference.

Zero-shot Named Entity Recognition — detect arbitrary entity types at inference time, no retraining needed.

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `gliner-lfm-f16.gguf` | F16 | 786 MB | Full precision |
| `gliner-lfm-q8_0.gguf` | Q8_0 | 419 MB | Recommended |
| `gliner-lfm-q4_k.gguf` | Q4_K | 254 MB | Max compression |

All variants produce the same entities. Score deltas vs F32: Q8_0 ≤ 0.01, Q4_K ≤ 0.03.

## Architecture

LFM2.5-350M bidirectional backbone (16 layers: 10 ShortConv + 6 GQA attention, SwiGLU FFN) + layer fusion (squeeze-and-excitation) + BiLSTM + GLiNER span-label matching head.

## Usage

```bash
# CLI
./crispembed -m gliner-lfm-q8_0.gguf \
  --ner "Tim Cook announced the new iPhone in Cupertino" \
  --ner-labels "person,organization,location,product" --json

# Server
./crispembed-server --ner gliner-lfm-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/ner/extract \
  -d '{"text": "Tim Cook at Apple", "labels": ["person", "organization"]}'
```

```python
from crispembed import CrispNER

ner = CrispNER("gliner-lfm-q8_0.gguf")
entities = ner.extract(
    "Maria Schmidt arbeitet bei Siemens in München",
    labels=["person", "organization", "location"],
)
for e in entities:
    print(f"{e['text']} => {e['label']} ({e['score']:.2f})")
```

## Parity

All 16 backbone layers cos=1.000000 vs HuggingFace Python reference. 17/17 entities match across 5 test texts.

## License

LFM Open License v1.0 — free for entities under $10M annual revenue. See [upstream license](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).

## Conversion

```bash
python models/convert-gliner-lfm-to-gguf.py \
  --model VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER \
  --output gliner-lfm-f32.gguf
./crispembed-quantize gliner-lfm-f32.gguf gliner-lfm-q8_0.gguf q8_0
```
