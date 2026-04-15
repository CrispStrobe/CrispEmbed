# CrispEmbed — Plan

Lightweight, dependency-free text embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any BERT/MiniLM/E5/GTE/Arctic-embed model from a GGUF file
2. Tokenizes input text (WordPiece or SentencePiece)
3. Runs the transformer encoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_0 / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, and Python wrapper

## Architecture

```
Input text
    │
    ▼
Tokenizer (WordPiece / SentencePiece from GGUF metadata)
    │
    ▼
Token embeddings + Position embeddings [+ Type embeddings]
    │
    ▼
N × Transformer layer:
    LayerNorm → Multi-head self-attention → residual
    LayerNorm → FFN (Linear → GELU → Linear) → residual
    │
    ▼
Pooling (mean / CLS / last-token)
    │
    ▼
Optional projection head + L2 normalization
    │
    ▼
float[embed_dim] output vector
```

## Target models (Phase 1)

| Model | Params | Dim | Tokenizer | HF ID |
|-------|--------|-----|-----------|-------|
| snowflake-arctic-embed-xs | 22M | 384 | WordPiece | Snowflake/snowflake-arctic-embed-xs |
| multilingual-e5-small | 118M | 384 | SentencePiece | intfloat/multilingual-e5-small |
| all-MiniLM-L6-v2 | 22M | 384 | WordPiece | sentence-transformers/all-MiniLM-L6-v2 |
| nomic-embed-text-v1.5 | 137M | 768 | WordPiece | nomic-ai/nomic-embed-text-v1.5 |
| gte-small | 33M | 384 | WordPiece | thenlper/gte-small |

## Phases

### Phase 0 — Scaffold (this session)

- [ ] Create repo structure
- [ ] Copy ggml as submodule (same version as CrispASR)
- [ ] CMakeLists.txt with crispembed library + cli + server targets
- [ ] Stub C API header (crispembed.h)

### Phase 1 — Single-model proof of concept

- [ ] GGUF converter: `convert-bert-to-gguf.py` (HF → GGUF)
  - Token/position/type embeddings
  - N encoder layers (LN + MHA + FFN)
  - Pooler / projection head
  - Tokenizer vocab in GGUF metadata
- [ ] WordPiece tokenizer in C++ (from GGUF vocab metadata)
- [ ] ggml graph: build full encoder + pooling
- [ ] C API: `crispembed_init()`, `crispembed_encode()`, `crispembed_free()`
- [ ] CLI: `crispembed -m model.gguf "query text"` → prints vector
- [ ] Verify: compare output with HF sentence-transformers on 10 test strings
- [ ] Convert + test all-MiniLM-L6-v2

### Phase 2 — Quantisation + multi-model

- [ ] crispembed-quantize tool (reuse CrispASR pattern)
- [ ] Convert all Phase 1 models, upload to HF
- [ ] Batch encoding: multiple texts in one call
- [ ] SentencePiece tokenizer (for multilingual-e5)
- [ ] Benchmark: tokens/sec, memory, vs ONNX fastembed

### Phase 3 — Server + Python

- [ ] HTTP server: `crispembed --server --port 8080`
  - POST /embed: `{"texts": ["hello", "world"]}` → `{"embeddings": [[...], [...]]}`
  - GET /health
  - POST /load (hot-swap model)
- [ ] Python wrapper: `pip install crispembed`
  - `from crispembed import CrispEmbed`
  - `model = CrispEmbed("model.gguf")`
  - `vectors = model.encode(["hello", "world"])`
- [ ] OpenAI-compatible /v1/embeddings endpoint

### Phase 4 — Optimisation

- [ ] GPU graph dispatch (reuse CrispASR ggml_backend_sched pattern)
- [ ] KV cache for prefix-shared batches
- [ ] SIMD-optimised L2 norm + cosine similarity
- [ ] Matryoshka dimension truncation support

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy or shared lib |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crispasr_cache | examples/cli/ | adapt for crispembed |

## File layout

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md
├── ggml/                      (submodule)
├── src/
│   ├── crispembed.h           C API
│   ├── crispembed.cpp         ggml graph encoder
│   ├── tokenizer.h            WordPiece + SentencePiece
│   ├── tokenizer.cpp
│   └── core/                  shared helpers (from CrispASR)
│       ├── gguf_loader.h
│       ├── gguf_loader.cpp
│       ├── attention.h
│       └── ffn.h
├── examples/
│   ├── cli/
│   │   └── main.cpp           CLI binary
│   └── server/
│       ├── server.cpp          HTTP server
│       └── httplib.h
├── models/
│   └── convert-bert-to-gguf.py
├── python/
│   ├── crispembed/
│   │   ├── __init__.py
│   │   └── _binding.py        ctypes wrapper
│   └── setup.py
└── tests/
    └── test_encode.py          compare vs HF reference
```

---

## Status (April 2026)

### Verified working — 12 models, cos >= 0.999 vs HF

| Model | Type | Dim | Pooling | CosSim |
|-------|------|-----|---------|--------|
| all-MiniLM-L6-v2 | BERT | 384 | mean | 0.999999 |
| gte-small | BERT | 384 | mean | 1.000000 |
| arctic-embed-xs | BERT | 384 | CLS | 1.000000 |
| multilingual-e5-small | XLM-R | 384 | mean | 1.000000 |
| PIXIE-Rune-v1.0 | XLM-R | 1024 | CLS | 0.999993 |
| arctic-embed-l-v2 | XLM-R | 1024 | CLS | 0.999993 |
| Octen-Embedding-0.6B | Qwen3 | 1024 | last-token | 0.999891 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | last-token | 0.999420 |
| Jina v5 Small | Qwen3 | 1024 | last-token | 0.999941 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | last-token | 0.999959 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | last-token | 0.999895 |
| Harrier-OSS-v1-270M | Gemma3 | 640 | last-token | 0.999948 |

### Supported architectures

| Architecture | Tokenizer | Key features | Models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, GTE, arctic-xs |
| XLM-R encoder | SentencePiece Unigram (Viterbi) | Post-LN, GELU FFN, pos_offset=2 | PIXIE-Rune, e5, arctic-l-v2 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA, causal mask | Octen, F2LLM, Jina, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU, embed*sqrt(H), extra norms | Harrier-270M |

### Optimization next steps

- Add ggml_backend_sched path for GPU offload
- Quantize all models (Q4_K/Q8_0) and benchmark vs ONNX fastembed
- Batch encoding (multi-text parallelism)
- Matryoshka dimension truncation support
