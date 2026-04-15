# CrispEmbed

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.
Supports BERT encoder, XLM-R encoder, AND decoder (Qwen3/Gemma3) embedding models.
Includes a C++ quantizer for Q4_K/Q5_K/Q8_0.

## Status

**13 models verified** bit-identical to HuggingFace (cos>=0.999):

| Model | Type | Dim | F32 CosSim | Q8_0 | Q4_K |
|-------|------|-----|------------|------|------|
| all-MiniLM-L6-v2 | BERT | 384 | 0.999999 | 0.9995 | 0.97 |
| gte-small | BERT | 384 | 1.000000 | 0.9998 | 0.99 |
| arctic-embed-xs | BERT | 384 | 1.000000 | 0.9999 | 0.99 |
| multilingual-e5-small | XLM-R | 384 | 1.000000 | 0.9999 | 0.99 |
| PIXIE-Rune-v1.0 | XLM-R | 1024 | 0.999993 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | XLM-R | 1024 | 0.999993 | 0.9989 | 0.95 |
| Octen-Embedding-0.6B | Qwen3 | 1024 | 0.999891 | 0.9995 | 0.96 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | 0.999420 | 0.9952 | -- |
| Jina v5 Nano | Qwen3 | 768 | 0.999020 | 0.9983 | -- |
| Jina v5 Small | Qwen3 | 1024 | 0.999941 | 0.9997 | 0.97 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | 0.999959 | 0.9999 | 0.99 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 | 0.9996 | 0.97 |
| Harrier-OSS-v1-270M | Gemma3 | 640 | 0.999948 | 0.9998 | 0.99 |

Q8_0 = all PASS (cos > 0.99). Q4_K = most PASS; `--` = use Q5_K or Q8_0 for this model.

**Server + Python wrapper** -- working with OpenAI-compatible API.

## Quick start

```bash
# Build
cmake -S . -B build
cmake --build build -j

# Convert a BERT model
python models/convert-bert-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2.gguf

# Convert a decoder model (Qwen3/Gemma3)
python models/convert-decoder-embed-to-gguf.py \
    --model Octen/Octen-Embedding-0.6B \
    --output octen-0.6b.gguf

# Quantize
./build/crispembed-quantize octen-0.6b.gguf octen-0.6b-q4_k.gguf q4_k
./build/crispembed-quantize octen-0.6b.gguf octen-0.6b-q8_0.gguf q8_0

# Encode text
./build/crispembed -m model.gguf "Hello world"

# JSON output
./build/crispembed -m model.gguf --json "Hello world" "Goodbye world"

# Start server
./build/crispembed-server -m model.gguf --port 8080
curl -X POST http://localhost:8080/embed \
    -d '{"texts": ["Hello world"]}'
```

## Quantization

The C++ quantizer (`crispembed-quantize`) supports all ggml quant types:

| Type | Compression | Quality | Notes |
|------|-------------|---------|-------|
| Q8_0 | ~3.8x | Excellent (cos>0.995) | Recommended default |
| Q5_K | ~5x | Very good | Good balance |
| Q4_K | ~7x | Good (cos>0.95) | Max compression, some models degrade |
| Q6_K | ~4.5x | Near-lossless | Premium quality |

Embedding tables are preserved at F32 for all quant types (they're precision-sensitive).
K-quants require 256-aligned row widths; the tool auto-falls back to Q4_0/Q5_0 for smaller tensors.

## Python

```python
from crispembed import CrispEmbed

model = CrispEmbed("all-MiniLM-L6-v2.gguf")
vectors = model.encode(["Hello world", "Goodbye world"])
print(vectors.shape)  # (2, 384)
```

## Architecture

**BERT encoder** (all-MiniLM, gte, arctic-embed-xs):
- Token + Position + Type embeddings -> Post-LN transformer -> Mean/CLS pooling

**XLM-R encoder** (PIXIE-Rune, multilingual-e5, arctic-embed-l-v2):
- Token + Position(+offset) embeddings -> Post-LN transformer -> CLS/Mean pooling
- SentencePiece Unigram tokenizer (Viterbi DP)

**Qwen3 decoder** (Octen, F2LLM, Jina v5, Harrier-0.6B, Qwen3-Embed):
- Token embeddings + RoPE -> RMSNorm + GQA with causal mask + SwiGLU -> Last-token pooling
- GPT-2 BPE tokenizer

**Gemma3 decoder** (Harrier-270M):
- Token embeddings * sqrt(H) + RoPE -> Gemma3 RMSNorm(1+w) + GQA + GeGLU -> Last-token pooling
- SentencePiece BPE tokenizer, BOS/EOS tokens

All via ggml graphs. Quantization supported via `crispembed-quantize`.
See [PLAN.md](PLAN.md) for the full roadmap.

## Credits

- [ggml](https://github.com/ggml-org/ggml) -- inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) -- shared core (gguf_loader, bpe.h, attention patterns)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) -- SentencePiece tokenizer reference
- [sentence-transformers](https://www.sbert.net/) -- ground-truth validation
