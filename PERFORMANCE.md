# CrispEmbed Performance

Benchmark results on Intel Xeon Skylake (4 threads), CPU-only, no GPU.

## Server Mode Latency (model loaded once)

Single-text encoding latency via HTTP server (`/embed` endpoint).

| Model | Quant | Params | Dim | Avg (ms) | Texts/s |
|-------|-------|--------|-----|----------|---------|
| all-MiniLM-L6-v2 | F32 | 22M | 384 | 16.8 | 59 |
| arctic-embed-xs | F32 | 22M | 384 | 18.0 | 56 |
| gte-small | F32 | 33M | 384 | 35 | 29 |
| octen-0.6b | Q8_0 | 600M | 1024 | 308 | 3.2 |
| octen-0.6b | Q4_K | 600M | 1024 | 294 | 3.4 |

**Batch mode** (10 texts per request):

| Model | Batch Latency | Per-text | Texts/s |
|-------|--------------|----------|---------|
| all-MiniLM-L6-v2 | 114ms | 11.4ms | 88 |

Optimizations: graph caching, flash attention (fused QKV), buffer reuse,
sorted batch processing (group by token count for graph cache hits).

## Comparison with HuggingFace and fastembed (ONNX)

Single-text latency, same hardware (CPU, 4 threads).

| Model | CrispEmbed | HF PyTorch | fastembed ONNX | vs HF | vs ONNX |
|-------|-----------|------------|----------------|-------|---------|
| MiniLM-L6-v2 | **16.8ms** | 54ms | 29.5ms | **3.2x faster** | **1.8x faster** |
| gte-small | **35ms** | 79ms | -- | **2.3x faster** | -- |
| arctic-embed-xs | **18.0ms** | -- | 4.9ms | -- | 0.27x |

CrispEmbed is **1.8-3.2x faster than HF PyTorch** and **1.8x faster than fastembed ONNX**
for MiniLM on pure CPU. Fastembed ONNX is faster for arctic-embed-xs due to ORT's Level3
graph optimizations (operator fusion, fused LayerNorm, batched QKV projection).

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
