# CrispEmbed

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.
Supports BERT encoder AND decoder (Qwen3/LLaMA) embedding models.

## Status

**8 models verified** bit-identical to HuggingFace (cos≥0.999):

| Model | Type | Dim | CosSim |
|-------|------|-----|--------|
| all-MiniLM-L6-v2 | BERT | 384 | 0.999999 |
| gte-small | BERT | 384 | 0.999999 |
| arctic-embed-xs | BERT | 384 | 1.000000 |
| Octen-Embedding-0.6B | Qwen3 | 1024 | 0.999891 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | 0.999420 |
| Jina v5 Small | Qwen3 | 1024 | 0.999941 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | 0.999959 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 |

**Server + Python wrapper** — working with OpenAI-compatible API.

## Quick start

```bash
# Build
cmake -S . -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build -j

# Convert a BERT model
python models/convert-bert-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2.gguf

# Convert a decoder model
python models/convert-decoder-embed-to-gguf.py \
    --model Octen/Octen-Embedding-0.6B \
    --output octen-0.6b.gguf

# Encode text
./build/crispembed -m model.gguf "Hello world"

# JSON output
./build/crispembed -m model.gguf --json "Hello world" "Goodbye world"

# Start server
./build/crispembed-server -m model.gguf --port 8080
curl -X POST http://localhost:8080/embed \
    -d '{"texts": ["Hello world"]}'

# OpenAI-compatible
curl -X POST http://localhost:8080/v1/embeddings \
    -d '{"input": ["Hello world"], "model": "all-MiniLM"}'
```

## Python

```python
from crispembed import CrispEmbed

model = CrispEmbed("all-MiniLM-L6-v2.gguf")
vectors = model.encode(["Hello world", "Goodbye world"])
print(vectors.shape)  # (2, 384)
```

## Architecture

**BERT encoder** (all-MiniLM, gte, arctic-embed, e5):
- Token + Position + Type embeddings → Post-LN transformer → Mean/CLS pooling

**Qwen3 decoder** (Octen, F2LLM, Jina v5, Harrier):
- Token embeddings + RoPE → RMSNorm + GQA with causal mask + SwiGLU → Last-token pooling

Both via ggml graphs. Quantisation (Q4_K/Q8_0) supported.
See [PLAN.md](PLAN.md) for the full roadmap.

## Credits

- [ggml](https://github.com/ggml-org/ggml) — inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) — shared core (gguf_loader, bpe.h, attention patterns)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — SentencePiece tokenizer reference
- [sentence-transformers](https://www.sbert.net/) — ground-truth validation
