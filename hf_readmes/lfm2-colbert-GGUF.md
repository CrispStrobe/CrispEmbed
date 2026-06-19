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

# LFM2.5-ColBERT-350M — CrispEmbed GGUF

CrispEmbed-native GGUF quantizations of [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M).

Multi-vector (ColBERT-style) retrieval: per-token embeddings projected to 128 dimensions, L2-normalized. Uses late interaction (MaxSim) scoring for fine-grained token-level matching.

**Format note:** These GGUFs use CrispEmbed's internal tensor naming (`lfm.*` prefix, arch=`lfm2`). They include the `colbert.projection.weight` tensor from the `1_Dense` module. **Not** compatible with llama.cpp.

## Model variants

| File | Quant | Size | ColBERT cos vs F32 |
|------|-------|------|--------------------|
| `lfm2-colbert-f32.gguf` | F32 | 677 MB | 0.999995 |
| `lfm2-colbert-q8_0.gguf` | Q8_0 | 361 MB | 0.998 |
| `lfm2-colbert-q5_k.gguf` | Q5_K | 258 MB | 0.977 |
| `lfm2-colbert-q4_k.gguf` | Q4_K | 224 MB | 0.959 |

## Architecture

- **Backbone**: LFM2.5-350M bidirectional hybrid (16 layers: 10 ShortConv + 6 GQA attention, 1024-dim hidden, SwiGLU FFN)
- **ColBERT head**: Linear(1024, 128) + L2 normalize per token
- **Scoring**: MaxSim — max over doc tokens of cosine similarity per query token, summed
- **Parameters**: 350M + 128K projection head
- **Languages**: EN, ES, DE, FR, IT, PT, AR, SV, NO, JA, KO (11 languages)
- **Task prefixes**: `"query: "` for queries, `"document: "` for passages

## Usage

```bash
# ColBERT multi-vector encode
./crispembed -m lfm2-colbert-q8_0.gguf --colbert "query: what is deep learning?"

# JSON output (per-token vectors)
./crispembed -m lfm2-colbert-q8_0.gguf --colbert --json "query: machine learning"

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

## Conversion

Convert from the source model yourself:

```bash
git clone https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed

# Convert (loads 1_Dense/model.safetensors for ColBERT projection)
python models/convert-lfm2-embed-to-gguf.py \
    --model LiquidAI/LFM2.5-ColBERT-350M \
    --output lfm2-colbert-f32.gguf --dtype f32

# Quantize
./build/crispembed-quantize lfm2-colbert-f32.gguf lfm2-colbert-q8_0.gguf q8_0
./build/crispembed-quantize lfm2-colbert-f32.gguf lfm2-colbert-q5_k.gguf q5_k
./build/crispembed-quantize lfm2-colbert-f32.gguf lfm2-colbert-q4_k.gguf q4_k
```

## License

[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE) — same as the base model.

## Credits

Original model by [LiquidAI](https://huggingface.co/LiquidAI). GGUF conversion and inference engine by [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
