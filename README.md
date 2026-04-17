# CrispEmbed

[![Build](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml/badge.svg)](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml)

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.
Supports BERT, XLM-R, Qwen3, and Gemma3 embedding models with GPU acceleration
(CUDA/Vulkan/Metal) and BLAS support (OpenBLAS/MKL).

**Multi-vector retrieval**: dense, sparse (SPLADE/BGE-M3), ColBERT multi-vector, and cross-encoder rerankers — all in one binary, all GPU-accelerated.

## Status

**13 models verified** bit-identical to HuggingFace (cos>=0.999), 30 models in registry:

| Model | Type | Dim | F32 CosSim | Q8_0 | Q4_K |
|-------|------|-----|------------|------|------|
| all-MiniLM-L6-v2 | BERT | 384 | 0.999999 | 0.9995 | 0.97 |
| gte-small | BERT | 384 | 1.000000 | 0.9998 | 0.99 |
| arctic-embed-xs | BERT | 384 | 1.000000 | 0.9999 | 0.99 |
| multilingual-e5-small | XLM-R | 384 | 1.000000 | 0.9999 | 0.99 |
| PIXIE-Rune-v1.0 | XLM-R | 1024 | 0.999993 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | XLM-R | 1024 | 0.999993 | 0.9989 | 0.95 |
| Octen-Embedding-0.6B | Qwen3 | 1024 | 0.999891 | 0.9995 | 0.97 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | 0.999420 | 0.9952 | -- |
| Jina v5 Nano | Qwen3 | 768 | 0.999020 | 0.9983 | -- |
| Jina v5 Small | Qwen3 | 1024 | 0.999941 | 0.9997 | 0.97 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | 0.999959 | 0.9999 | 0.99 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 | 0.9996 | 0.97 |
| Harrier-OSS-v1-270M | Gemma3 | 640 | 0.999948 | 0.9998 | 0.99 |

Q8_0 = all PASS (cos > 0.99). Q4_K = most PASS; `--` = use Q5_K or Q8_0 for this model.

**Performance** (Apple M1, Metal):

| Engine | Single text | Batch (10) |
|--------|------------|------------|
| **CrispEmbed Python** (ctypes) | **3.6 ms** / 280 t/s | **12.7 ms** / **787 t/s** |
| fastembed-rs (Rust ONNX) | 3.8 ms / 263 t/s | 18.9 ms / 528 t/s |
| HuggingFace (PyTorch) | 12.2 ms / 82 t/s | 29.8 ms / 335 t/s |
| CrispEmbed Server (HTTP) | 21.3 ms / 46 t/s | 32.9 ms / 303 t/s |

Model: all-MiniLM-L6-v2. See [PERFORMANCE.md](PERFORMANCE.md) for full multi-model benchmarks.

**Ollama-compatible**: All 13 models export as Ollama-compatible GGUFs. Works with our [Ollama fork](https://github.com/CrispStrobe/ollama/tree/feat/xlmr-embedding) (adds XLM-R, Viterbi SentencePiece tokenizer, GELU_ERF, multi-tokenizer BERT support).

## Quick start

```bash
# Clone with submodule
git clone --recursive https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed

# Build (CPU)
cmake -S . -B build
cmake --build build -j

# Encode text
./build/crispembed -m model.gguf "Hello world"

# Matryoshka truncation (e.g. 128 dims from a 384-dim model)
./build/crispembed -m model.gguf -d 128 "Hello world"

# Start server (model loaded once, fast repeated queries)
./build/crispembed-server -m model.gguf --port 8080
curl -X POST http://localhost:8080/embed \
    -d '{"texts": ["Hello world"]}'
```

## Building

### Linux / macOS

```bash
# CPU only (default)
cmake -S . -B build && cmake --build build -j

# With OpenBLAS acceleration
cmake -S . -B build -DGGML_BLAS=ON && cmake --build build -j

# With Intel MKL
cmake -S . -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp

# With CUDA (NVIDIA GPU)
cmake -S . -B build -DGGML_CUDA=ON && cmake --build build -j

# With Vulkan (cross-platform GPU)
cmake -S . -B build -DGGML_VULKAN=ON && cmake --build build -j

# macOS with Metal (recommended)
./build-macos.sh              # Metal + Accelerate + embedded shaders
./build-macos.sh --cpu        # CPU only, no Metal
./build-macos.sh --shared     # Also build shared lib for Python
```

### Windows

Requires Visual Studio 2022 Build Tools + Ninja.

```batch
:: CPU build
build-windows.bat

:: Vulkan GPU build (needs Vulkan SDK)
build-vulkan.bat

:: CUDA GPU build (needs CUDA Toolkit)
build-cuda.bat
```

If you get "ggml does not contain a CMakeLists.txt", run:
```
git submodule update --init --recursive
```

### Dependencies

- **Required**: C++17 compiler, CMake 3.14+
- **Optional**: OpenBLAS (`apt install libopenblas-dev`), Intel MKL, CUDA Toolkit, Vulkan SDK

## Converting models

```bash
# BERT / XLM-R encoder models
pip install torch transformers gguf
python models/convert-bert-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2.gguf

# Qwen3 / Gemma3 decoder models
python models/convert-decoder-embed-to-gguf.py \
    --model Octen/Octen-Embedding-0.6B \
    --output octen-0.6b.gguf

# Quantize (Q8_0 recommended, Q4_K for max compression)
./build/crispembed-quantize model.gguf model-q8_0.gguf q8_0
./build/crispembed-quantize model.gguf model-q4_k.gguf q4_k
```

Pre-converted models: [HuggingFace cstr/](https://huggingface.co/cstr)

## Quantization

| Type | Compression | Quality (cos vs F32) | Notes |
|------|-------------|---------------------|-------|
| Q8_0 | ~3.8x | >0.995 | Recommended default |
| Q5_K | ~5x | >0.98 | Good balance |
| Q4_K | ~5.5x | >0.95 | Max compression |
| Q6_K | ~4.5x | >0.99 | Premium quality |

Embedding tables quantized to Q8_0 even in Q4_K mode (quality-sensitive).

## BGE-M3 / Sparse / ColBERT / Reranker

CrispEmbed supports all three BGE-M3 retrieval modalities plus cross-encoder rerankers.

```bash
# Convert BGE-M3 (writes sparse_linear.weight + colbert_linear.weight into GGUF)
pip install torch transformers gguf FlagEmbedding
python models/convert-bert-to-gguf.py --model BAAI/bge-m3 --output bge-m3.gguf --crisp

# Validate all three heads against FlagEmbedding ground truth
python tests/test_bgem3.py --gguf bge-m3.gguf --lib build/libcrispembed.so
```

```python
from crispembed import CrispEmbed

model = CrispEmbed("bge-m3.gguf")

# Dense (L2-normalised)
vec = model.encode("Hello world")                   # Vec<f32> len 1024

# Sparse (SPLADE-style term weights)
if model.has_sparse():
    sparse = model.encode_sparse("Hello world")     # {token_id: weight}

# ColBERT multi-vector
if model.has_colbert():
    multi = model.encode_multivec("Hello world")    # [[f32; 128]; n_tokens]
```

Cross-encoder rerankers:

```python
reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
score = reranker.rerank("query text", "document text")   # raw logit
```

## Python

Requires the shared library (`--shared` flag or `-DCRISPEMBED_BUILD_SHARED=ON`).

```python
from crispembed import CrispEmbed

model = CrispEmbed("all-MiniLM-L6-v2.gguf")

# Single text
vec = model.encode("Hello world")      # shape (384,)

# Batch — single C call, true batched Metal/GPU inference
vectors = model.encode(["Hello world", "Goodbye world"])
print(vectors.shape)  # (2, 384)

# Matryoshka dimension truncation
model.set_dim(128)
vec128 = model.encode("Hello world")   # shape (128,)

# Prompt prefix (for models that need it)
model.set_prefix("query: ")           # auto-prepended before tokenization

# Sparse (BGE-M3)
model = CrispEmbed("bge-m3.gguf")
if model.has_sparse:
    sparse = model.encode_sparse("Hello world")   # {token_id: weight}

# ColBERT multi-vector
if model.has_colbert:
    multi = model.encode_multivec("Hello world")   # (n_tokens, 128)

# Cross-encoder reranking
reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
score = reranker.rerank("query", "document")       # raw logit

# Bi-encoder reranking (any embedding model, cosine similarity)
results = model.rerank_biencoder("query", ["doc1", "doc2", "doc3"], top_n=2)
for r in results:
    print(f"  [{r['index']}] {r['score']:.4f}: {r['document']}")
```

## Rust

```toml
[dependencies]
crispembed = { git = "https://github.com/CrispStrobe/CrispEmbed" }
```

```rust
use crispembed::CrispEmbed;

let mut model = CrispEmbed::new("model.gguf", 0)?;
let vec = model.encode("Hello world");

// Prompt prefix
model.set_prefix("query: ");

// Sparse + ColBERT (BGE-M3)
if model.has_sparse() {
    let sparse = model.encode_sparse("query");   // Vec<(i32, f32)>
}
if model.has_colbert() {
    let multi = model.encode_multivec("query");  // Vec<Vec<f32>>
}

// Bi-encoder reranking (cosine similarity)
let ranked = model.rerank_biencoder("query", &["doc1", "doc2"], Some(2));
for (idx, score) in &ranked {
    println!("  doc {} score {:.4}", idx, score);
}
```

## Benchmarking

```bash
./benchmark.sh                          # single model, all engines
./benchmark.sh --multi                  # 3 models, all engines
./benchmark.sh -n 100 --skip-fastembed  # CrispEmbed + HF only, 100 runs

# RAG retrieval quality benchmark
python tests/bench_rag.py --lib build/libcrispembed.so --gguf model.gguf

# Reranking benchmark
python tests/bench_rerank.py --lib build/libcrispembed.so \
    --embed-gguf model.gguf --reranker-gguf reranker.gguf
```

Compares CrispEmbed (CLI, Python ctypes, HTTP server) against HuggingFace
sentence-transformers, FastEmbed (ONNX), and fastembed-rs (Rust ONNX).
Auto-creates a `.bench-venv` for Python dependencies.

## Architecture

**BERT encoder** (all-MiniLM, gte, arctic-embed-xs):
- Token + Position + Type embeddings → Post-LN transformer → Mean/CLS pooling

**XLM-R encoder** (PIXIE-Rune, multilingual-e5, arctic-embed-l-v2):
- Token + Position(+offset) embeddings → Post-LN transformer → CLS/Mean pooling
- SentencePiece Unigram tokenizer (Viterbi DP)

**BGE-M3 multi-modal** (`BAAI/bge-m3`):
- Same BERT encoder trunk with three output heads:
  - **Dense**: mean-pool → L2 normalize → `float[1024]`
  - **Sparse**: `Linear(H,1)` + ReLU → scatter via input_ids → `{token_id: weight}`
  - **ColBERT**: `Linear(H,128)` → per-token L2 normalize → `float[n_tokens][128]`

**Cross-encoder reranker** (BGE-reranker-v2-m3, etc.):
- `[CLS] query [SEP] document [SEP]` pair tokenization → CLS hidden state → `Linear(H,1)` → scalar score

**Qwen3 decoder** (Octen, F2LLM, Jina v5, Harrier-0.6B, Qwen3-Embed):
- Token embeddings + RoPE → RMSNorm + GQA with causal mask + SwiGLU → Last-token pooling

**Gemma3 decoder** (Harrier-270M):
- Token embeddings * sqrt(H) + RoPE → Gemma3 RMSNorm(1+w) + GQA + GeGLU → Last-token pooling

All via ggml graphs with GPU dispatch (ggml_backend_sched).
See [PLAN.md](PLAN.md), [LEARNINGS.md](LEARNINGS.md), [PERFORMANCE.md](PERFORMANCE.md).

## Credits

- [ggml](https://github.com/ggml-org/ggml) -- inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) -- shared core (gguf_loader, bpe.h)
- [sentence-transformers](https://www.sbert.net/) -- ground-truth validation
