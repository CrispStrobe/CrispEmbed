# CrispEmbed Performance

Benchmark results on Intel Xeon Skylake (4 threads), CPU-only, no GPU.

## Server Mode Latency (model loaded once)

Single-text encoding latency via HTTP server (`/embed` endpoint).

| Model | Quant | Params | Dim | Avg (ms) | Texts/s |
|-------|-------|--------|-----|----------|---------|
| all-MiniLM-L6-v2 | F32 | 22M | 384 | 15.5 | 64 |
| arctic-embed-xs | F32 | 22M | 384 | 15.5 | 64 |
| gte-small | F32 | 33M | 384 | 30 | 33 |
| octen-0.6b | Q8_0 | 600M | 1024 | 308 | 3.2 |
| octen-0.6b | Q4_K | 600M | 1024 | 294 | 3.4 |

## macOS Metal (Apple M1)

Benchmarked with Metal backend + embedded shaders, `./benchmark.sh --multi -n 20`.

### all-MiniLM-L6-v2 (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 3.9 ms / 258 t/s | 19 ms / 533 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 4.0 ms / 248 t/s | 62 ms / 161 t/s |
| HuggingFace sentence-transformers | 11.4 ms / 88 t/s | 23 ms / 431 t/s |
| CrispEmbed Server (Metal + HTTP) | 21.9 ms / 45 t/s | 31 ms / 318 t/s |
| FastEmbed Python (ONNX) | 33.5 ms / 30 t/s | -- |

### gte-small (33M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 4.1 ms / 243 t/s | 21 ms / 479 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 6.4 ms / 155 t/s | 70 ms / 142 t/s |
| HuggingFace sentence-transformers | 22.6 ms / 44 t/s | 226 ms / 44 t/s |
| CrispEmbed Server (Metal + HTTP) | 24.9 ms / 40 t/s | 52 ms / 190 t/s |

### arctic-embed-xs (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| **CrispEmbed Python** (Metal, ctypes) | 3.7 ms / 267 t/s | 46 ms / 220 t/s |
| fastembed-rs (Rust, ONNX) | 4.0 ms / 251 t/s | 29 ms / 342 t/s |
| FastEmbed Python (ONNX) | 4.1 ms / 244 t/s | -- |
| CrispEmbed Server (Metal + HTTP) | 22.2 ms / 44 t/s | 35 ms / 287 t/s |

CrispEmbed Python wrapper (ctypes, Metal) matches or beats fastembed-rs for
single-text latency. Batch throughput gap is due to per-text Python loop --
a C-level batch API would close it.

## Ollama Integration (Q8_0, Apple M1)

All CrispEmbed models verified in Ollama fork with Ollama-compatible GGUF export.

### Encoder Models (Q8_0 and Q4_K vs HuggingFace F32)

| Model | Dim | Q8_0 cos | Q4_K cos | Q8_0 Size | Q4_K Size |
|-------|-----|----------|----------|-----------|-----------|
| all-MiniLM-L6-v2 | 384 | 0.9998 | 0.970 | 24 MB | 18 MB |
| gte-small | 384 | 0.9999 | 0.991 | 34 MB | 24 MB |
| arctic-embed-xs | 384 | 0.9999 | 0.995 | 24 MB | 18 MB |
| multilingual-e5-small | 384 | 0.9999 | 0.990 | 126 MB | 115 MB |
| pixie-rune-v1 | 1024 | cross-lingual OK | cross-lingual OK | 581 MB | 437 MB |
| arctic-embed-l-v2 | 1024 | L2-norm=1.0 | L2-norm=1.0 | 581 MB | 437 MB |

### Decoder Models (Q8_0 and Q4_K in Ollama)

| Model | Arch | Dim | Q8_0 Size | Q4_K Size | L2-Norm | Diversity |
|-------|------|-----|-----------|-----------|---------|-----------|
| qwen3-embed-0.6b | Qwen3 | 1024 | 610 MB | 300 MB | 1.000 | 0.599 |
| octen-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.649 |
| f2llm-v2-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.711 |
| harrier-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.504 |
| harrier-270m | Gemma3 | 640 | 287 MB | 239 MB | 1.000 | 0.922 |
| jina-v5-nano | Qwen3 | 768 | 222 MB | 168 MB | 1.000 | 0.237 |
| jina-v5-small | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.746 |

All 13 Q4_K models verified: L2-normalized, semantically correct embeddings.
Diversity = 1 - avg cosine similarity between 4 different test texts (higher = better discrimination).

## GPU Inference (CUDA)

Tested on NVIDIA RTX A1000 Laptop GPU (4GB VRAM), via HTTP server.

| Model | Quant | Avg (ms) | Texts/s | Batch (10) |
|-------|-------|----------|---------|------------|
| all-MiniLM-L6-v2 | F32 | 7.4 | 135 | 211/s |

GPU inference **matches HuggingFace PyTorch** (10.6ms vs 10.8ms) and
**beats fastembed ONNX** (10.6ms vs 13.4ms). Both HF and CrispEmbed use
CUDA on this hardware. The ggml_backend_sched dispatcher offloads
matmul, flash attention, and norm ops to CUDA.

True batched encoding: single graph with 4D flash attention for B texts.
Batch mode (10 texts): 190-211 texts/s on CUDA. HF gets 347/s via
PyTorch's native batch parallelism (more mature GPU batching).

## CPU Batch Mode

| Model | Batch Latency | Per-text | Texts/s |
|-------|--------------|----------|---------|
| all-MiniLM-L6-v2 | 114ms | 11.4ms | 88 |

Optimizations: graph caching, flash attention (fused QKV), buffer reuse,
sorted batch processing (group by token count for graph cache hits).

## Comparison with HuggingFace and fastembed (ONNX)

Single-text latency, same hardware (CPU, 4 threads).

| Model | CrispEmbed | HF PyTorch | fastembed ONNX | vs HF | vs ONNX |
|-------|-----------|------------|----------------|-------|---------|
| MiniLM-L6-v2 | **15.5ms** | 54ms | 29.5ms | **3.5x faster** | **1.9x faster** |
| gte-small | **30ms** | 79ms | -- | **2.6x faster** | -- |
| arctic-embed-xs | **15.5ms** | -- | 4.9ms | -- | 0.32x |

Optimizations: graph caching, flash attention, pre-merged QKV weights, buffer reuse.

CrispEmbed is **1.9-3.5x faster than HF PyTorch** and **1.9x faster than fastembed ONNX**
for MiniLM on pure CPU. Fastembed ONNX is 3x faster for arctic-embed-xs due to ORT's
Level3 graph JIT compilation (operator fusion, fused LayerNorm, layout optimization).
We apply QKV weight fusion and flash attention but cannot match ORT's runtime compilation.

Key advantages:
- No Python runtime overhead (direct C++ inference)
- No ONNX runtime dependency
- Graph + work buffer reuse across calls
- ~20MB binary vs ~500MB Python + ONNX environment

## Model Sizes

| Model | F32 | Q8_0 | Q4_K | Q8_0 ratio |
|-------|-----|------|------|------------|
| all-MiniLM-L6-v2 | 87 MB | 24 MB | 19 MB | 3.6x |
| gte-small | 128 MB | 35 MB | 25 MB | 3.7x |
| arctic-embed-xs | 87 MB | 24 MB | 19 MB | 3.6x |
| multilingual-e5-small | 453 MB | 123 MB | 113 MB | 3.7x |
| pixie-rune-v1 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| arctic-embed-l-v2 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| octen-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| f2llm-v2-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| jina-v5-nano | 585 MB | 219 MB | 164 MB | 2.7x |
| jina-v5-small | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-270m | 741 MB | 279 MB | 231 MB | 2.7x |
| qwen3-embed-0.6b | 1.6 GB | 607 MB | 291 MB | 2.7x |

## Quantization Quality

Cosine similarity between F32 and quantized models (1.0 = identical).

| Model | Q8_0 | Q4_K |
|-------|------|------|
| all-MiniLM-L6-v2 | 0.9995 | 0.97 |
| gte-small | 0.9998 | 0.99 |
| arctic-embed-xs | 0.9999 | 0.99 |
| multilingual-e5-small | 0.9999 | 0.99 |
| pixie-rune-v1 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | 0.9989 | 0.95 |
| octen-0.6b | 0.9995 | 0.97 |
| harrier-0.6b | 0.9999 | 0.99 |
| harrier-270m | 0.9998 | 0.99 |
| qwen3-embed-0.6b | 0.9996 | 0.97 |

Q8_0: all > 0.995. Q4_K: most > 0.95.

## BLAS Acceleration

OpenBLAS 0.3.26, Intel Xeon Skylake, 4 threads.

| Model | Quant | no-BLAS | BLAS | Speedup |
|-------|-------|---------|------|---------|
| gte-small | F32 | 114ms | 123ms | 0.9x |
| gte-small | Q8_0 | 116ms | 116ms | 1.0x |
| octen-0.6b | Q8_0 | 422ms | 410ms | 1.0x |

BLAS provides minimal benefit because quantized kernels use ggml's SIMD paths.
Use Q8_0 for CPU speed, GPU (CUDA/Vulkan) for maximum throughput.

## Notes

- CrispEmbed uses ggml inference with SIMD-optimized quantized matmul
- Graph and work buffers are reused across calls (3.2x throughput improvement)
- When built with CUDA/Vulkan/Metal, `ggml_backend_sched` auto-dispatches to GPU
- Decoder models (Qwen3/Gemma3) are 10-15x slower than encoders (28 layers vs 6)
- Server mode eliminates model loading overhead (~100-300ms per cold start)
