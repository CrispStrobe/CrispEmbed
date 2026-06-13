---
license: apache-2.0
base_model: urchade/gliner_medium-v2.1
tags:
  - ner
  - named-entity-recognition
  - gliner
  - zero-shot
  - gguf
  - crispembed
  - ggml
  - deberta
language:
  - en
---

# GLiNER DeBERTa-v3 GGUF

GGUF conversions of [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) inference.

Zero-shot Named Entity Recognition — detect arbitrary entity types at inference time, no retraining needed. Apache-2.0 licensed.

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `gliner-deberta-f32.gguf` | F32 | 747 MB | Full precision |
| `gliner-deberta-q8_0.gguf` | Q8_0 | 198 MB | Recommended |
| `gliner-deberta-q4_k.gguf` | Q4_K | 152 MB | Max compression |

Q8_0 produces identical entities to F32. Q4_K may merge adjacent spans at high compression.

## Architecture

DeBERTa-v3-base encoder (12 layers, 768 hidden, disentangled attention with log-bucketed relative positions) + 768-to-512 linear projection + BiLSTM (hidden=256) + GLiNER markerV0 span-label matching head (start+end concatenation).

209M parameters. Based on `microsoft/deberta-v3-base` with SentencePiece tokenizer (128K vocab).

## Usage

```bash
# CLI
./crispembed -m gliner-deberta-q8_0.gguf \
  --ner "Tim Cook announced the new iPhone in Cupertino" \
  --ner-labels "person,organization,location,product" --json

# Auto-download
./crispembed -m gliner-deberta \
  --ner "Barack Obama was born in Hawaii" --json

# Server
./crispembed-server --ner gliner-deberta-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/ner/extract \
  -d '{"text": "Tim Cook at Apple", "labels": ["person", "organization"]}'
```

```python
from crispembed import CrispNER

ner = CrispNER("gliner-deberta-q8_0.gguf")
entities = ner.extract(
    "Apple Inc. was founded by Steve Jobs in Cupertino, California",
    labels=["person", "organization", "location"],
)
for e in entities:
    print(f"{e['text']} => {e['label']} ({e['score']:.2f})")
# Apple Inc. => organization (1.00)
# Steve Jobs => person (1.00)
# Cupertino => location (0.99)
# California => location (0.97)
```

## Parity

C++ output matches Python GLiNER library (`gliner==0.2.26`) — same entities detected on all test inputs. Scores within 0.02-0.09 of PyTorch reference (expected for F32 precision differences in DeBERTa disentangled attention).

## License

Apache-2.0 — fully permissive, no revenue cap.

## Conversion

```bash
python models/convert-gliner-deberta-to-gguf.py \
  --model /path/to/gliner_medium-v2.1 \
  --output gliner-deberta-f32.gguf
./crispembed-quantize gliner-deberta-f32.gguf gliner-deberta-q8_0.gguf q8_0
./crispembed-quantize gliner-deberta-f32.gguf gliner-deberta-q4_k.gguf q4_k
```
