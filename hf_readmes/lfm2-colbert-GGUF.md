---
license: other
license_name: lfm-1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-ColBERT-350M
tags:
  - colbert
  - retrieval
  - multi-vector
  - late-interaction
  - gguf
  - crispembed
  - ggml
language:
  - en
  - es
  - de
  - fr
  - it
  - pt
  - ar
  - sv
  - "no"
  - ja
  - ko
---

# LFM2.5-ColBERT-350M GGUF

GGUF conversions of [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) inference.

ColBERT multi-vector late interaction retrieval — per-token 128-dimensional embeddings with MaxSim scoring for fine-grained passage matching.

## Model variants

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `lfm2-colbert-f32.gguf` | F32 | 677 MB | Full precision (dev only) |
| `lfm2-colbert-q8_0.gguf` | Q8_0 | 361 MB | Recommended |
| `lfm2-colbert-q5_k.gguf` | Q5_K | 258 MB | Good quality/size tradeoff |
| `lfm2-colbert-q4_k.gguf` | Q4_K | 224 MB | Max compression |

Parity vs F32 reference (cos similarity): F32 = 0.999995, Q8_0 = 0.998, Q5_K = 0.977, Q4_K = 0.959.

## Architecture

LFM2.5-350M bidirectional backbone (16 layers: 10 ShortConv + 6 GQA attention, 1024-dim hidden, SwiGLU FFN) + Dense 128-dim projection head with L2 normalization. Per-token output for ColBERT late interaction retrieval.

- **Backbone**: Hybrid ShortConv/GQA architecture with bidirectional processing
- **ColBERT head**: Linear(1024, 128) + L2 normalize per token
- **Scoring**: MaxSim — max over doc tokens of cosine similarity per query token, summed

## Usage

```bash
# CLI — encode ColBERT vectors
./crispembed -m lfm2-colbert-q8_0.gguf --colbert "query: what is deep learning?"

# Server
./crispembed-server --embed lfm2-colbert-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/colbert/score \
  -d '{"query": "what is deep learning?", "documents": ["Deep learning is a subset of ML", "The weather is nice"]}'
```

```python
from crispembed import CrispVit

model = CrispVit("lfm2-colbert-q8_0.gguf")
assert model.has_colbert

# Encode multi-vector representations
query_vecs = model.encode_multivec("query: what is deep learning?")   # (n_tokens, 128)
doc_vecs = model.encode_multivec("document: Deep learning uses neural networks")

# MaxSim scoring
score = model.maxsim(query_vecs, doc_vecs)
print(f"Score: {score:.4f}")
```

```rust
use crispembed::CrispEmbed;

let mut model = CrispEmbed::new("lfm2-colbert-q8_0.gguf", 4)?;
assert!(model.has_colbert());

let query = model.encode_multivec("query: what is deep learning?");
let doc = model.encode_multivec("document: Neural networks learn representations");
```

## Prefixes

Following the original model convention:
- **Queries**: `query: ` prefix
- **Documents**: `document: ` prefix

## License

LFM Open License v1.0 — see [LICENSE](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE).

## Credits

Original model by [LiquidAI](https://huggingface.co/LiquidAI). GGUF conversion and inference engine by [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
