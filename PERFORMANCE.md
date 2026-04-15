# CrispEmbed Performance

Benchmark results on Intel Xeon Skylake (4 threads), CPU-only, no GPU.

## Server Mode Latency (model loaded once)

Single-text encoding latency via HTTP server (`/embed` endpoint).

| Model | Quant | Params | Dim | Avg (ms) | P50 (ms) | Texts/s |
|-------|-------|--------|-----|----------|----------|---------|
| gte-small | F32 | 33M | 384 | 36 | 35 | 27.8 |
| gte-small | Q8_0 | 33M | 384 | 36 | 35 | 27.8 |
| harrier-270m | Q8_0 | 270M | 640 | 209 | 211 | 5.1 |
| octen-0.6b | Q8_0 | 600M | 1024 | 369 | 356 | 2.7 |
| octen-0.6b | Q4_K | 600M | 1024 | 346 | 346 | 2.9 |

Note: gte-small throughput improved 3.2x (8.8 -> 27.8 texts/s) after graph buffer
reuse optimization. Decoder models not yet optimized with buffer reuse.

## Comparison with HuggingFace sentence-transformers

Single-text latency, same hardware (CPU, 4 threads, PyTorch with MKL).

| Model | CrispEmbed (ms) | HF PyTorch (ms) | Ratio |
|-------|-----------------|-----------------|-------|
| gte-small F32 | 116 | 79 | 1.5x |
| gte-small Q8_0 | 116 | 79 | 1.5x |

CrispEmbed is ~1.5x slower than HF PyTorch with MKL-optimized BLAS on CPU.
This is expected: HF PyTorch uses highly optimized BLAS (Intel MKL/OpenBLAS),
while CrispEmbed uses ggml's pure C matmul (portable, no external BLAS dep).

With GPU (CUDA/Vulkan/Metal), CrispEmbed would be significantly faster due
to ggml's GPU graph dispatch via `ggml_backend_sched`.

## Model Sizes

| Model | F32 | Q8_0 | Q4_K | Compression |
|-------|-----|------|------|-------------|
| all-MiniLM-L6-v2 | 87 MB | 24 MB | 19 MB | 4.6x |
| gte-small | 128 MB | 35 MB | 25 MB | 5.1x |
| arctic-embed-xs | 87 MB | 24 MB | 19 MB | 4.6x |
| multilingual-e5-small | 453 MB | 123 MB | 113 MB | 4.0x |
| pixie-rune-v1 | 2.2 GB | 580 MB | 436 MB | 5.0x |
| arctic-embed-l-v2 | 2.2 GB | 580 MB | 436 MB | 5.0x |
| octen-0.6b | 1.6 GB | 607 MB | 397 MB | 4.1x |
| f2llm-v2-0.6b | 1.6 GB | 607 MB | 397 MB | 4.1x |
| jina-v5-nano | 585 MB | 219 MB | 164 MB | 3.6x |
| jina-v5-small | 1.6 GB | 607 MB | 397 MB | 4.1x |
| harrier-0.6b | 1.6 GB | 607 MB | 397 MB | 4.1x |
| harrier-270m | 741 MB | 279 MB | 231 MB | 3.2x |
| qwen3-embed-0.6b | 1.6 GB | 607 MB | 291 MB | 5.6x |

Q8_0 compression: ~3-4x. Q4_K compression: ~4-6x. Embedding tables use Q8_0
even in Q4_K mode for quality preservation.

## Quantization Quality

Cosine similarity between F32 and quantized models (higher = better, 1.0 = identical).

| Model | Q8_0 cos | Q4_K cos |
|-------|----------|----------|
| all-MiniLM-L6-v2 | 0.9995 | 0.97 |
| gte-small | 0.9998 | 0.99 |
| arctic-embed-xs | 0.9999 | 0.99 |
| multilingual-e5-small | 0.9999 | 0.99 |
| pixie-rune-v1 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | 0.9989 | 0.95 |
| octen-0.6b | 0.9995 | 0.97 |
| f2llm-v2-0.6b | 0.9952 | -- |
| jina-v5-nano | 0.9983 | -- |
| jina-v5-small | 0.9997 | 0.97 |
| harrier-0.6b | 0.9999 | 0.99 |
| harrier-270m | 0.9998 | 0.99 |
| qwen3-embed-0.6b | 0.9996 | 0.97 |

Q8_0: all models > 0.995. Recommended for production use.
Q4_K: most models > 0.95. Use Q5_K for f2llm and jina-v5-nano.

## Memory Usage

Approximate runtime memory (RSS) during server mode:

| Model | Quant | Model Size | RSS |
|-------|-------|-----------|-----|
| gte-small | F32 | 128 MB | ~180 MB |
| gte-small | Q8_0 | 35 MB | ~90 MB |
| octen-0.6b | Q8_0 | 607 MB | ~750 MB |
| octen-0.6b | Q4_K | 397 MB | ~550 MB |

RSS is approximately model size + ~50-150 MB for graph workspace and tokenizer.

## BLAS Acceleration

Tested with OpenBLAS 0.3.26 on Intel Xeon Skylake, 4 threads.

| Model | Quant | no-BLAS (ms) | BLAS (ms) | Speedup |
|-------|-------|-------------|-----------|---------|
| gte-small | F32 | 114 | 123 | 0.9x |
| gte-small | Q8_0 | 116 | 116 | 1.0x |
| octen-0.6b | Q8_0 | 422 | 410 | 1.0x |

BLAS provides minimal benefit for embedding models because:
- Quantized (Q8_0/Q4_K) kernels use ggml's SIMD-optimized paths, not BLAS
- Encoder models have moderate matrix sizes (not large enough for BLAS overhead to pay off)
- BLAS helps primarily with large F32 matmul (e.g. 4096x4096+)

**Recommendation**: Use Q8_0 quantization for CPU speed, not BLAS. BLAS is only
useful if you must run F32 models for exact precision.

## Notes

- CrispEmbed uses ggml with pure C matmul (no external BLAS dependency)
- When built with CUDA/Vulkan/Metal, the `ggml_backend_sched` dispatcher
  automatically offloads graph computation to GPU
- Batch encoding currently processes texts sequentially; batch optimization
  would improve throughput significantly
- Server mode eliminates the ~100-300ms model loading overhead per request
