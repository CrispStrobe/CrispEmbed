# CrispEmbed

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.
Supports BERT, XLM-R, Qwen3, and Gemma3 embedding models with GPU acceleration
(CUDA/Vulkan/Metal) and BLAS support (OpenBLAS/MKL).

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
| Octen-Embedding-0.6B | Qwen3 | 1024 | 0.999891 | 0.9995 | 0.97 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | 0.999420 | 0.9952 | -- |
| Jina v5 Nano | Qwen3 | 768 | 0.999020 | 0.9983 | -- |
| Jina v5 Small | Qwen3 | 1024 | 0.999941 | 0.9997 | 0.97 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | 0.999959 | 0.9999 | 0.99 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 | 0.9996 | 0.97 |
| Harrier-OSS-v1-270M | Gemma3 | 640 | 0.999948 | 0.9998 | 0.99 |

Q8_0 = all PASS (cos > 0.99). Q4_K = most PASS; `--` = use Q5_K or Q8_0 for this model.

**Server throughput**: ~28 texts/sec (gte-small Q8_0, 4 threads, CPU).
See [PERFORMANCE.md](PERFORMANCE.md) for full benchmarks.

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

# With Metal (macOS GPU)
cmake -S . -B build -DGGML_METAL=ON && cmake --build build -j
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

**Gemma3 decoder** (Harrier-270M):
- Token embeddings * sqrt(H) + RoPE -> Gemma3 RMSNorm(1+w) + GQA + GeGLU -> Last-token pooling

All via ggml graphs with GPU dispatch (ggml_backend_sched).
See [PLAN.md](PLAN.md), [LEARNINGS.md](LEARNINGS.md), [PERFORMANCE.md](PERFORMANCE.md).

## Credits

- [ggml](https://github.com/ggml-org/ggml) -- inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) -- shared core (gguf_loader, bpe.h)
- [sentence-transformers](https://www.sbert.net/) -- ground-truth validation
