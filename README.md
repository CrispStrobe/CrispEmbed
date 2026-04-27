# CrispEmbed

[![Build](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml/badge.svg)](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml)

Lightweight text embedding inference via ggml. No Python runtime, no ONNX.
10 architectures: BERT, XLM-R, MPNet, NomicBERT, ModernBERT, GTE v1.5, DeBERTa-v2,
Qwen3, Gemma3, SPLADE. GPU acceleration (CUDA/Vulkan/Metal), BLAS (OpenBLAS/MKL).

**Multi-vector retrieval**: dense, sparse (SPLADE/BGE-M3), ColBERT multi-vector,
cross-encoder rerankers, bi-encoder reranking — all in one binary, all GPU-accelerated.

**9.5x faster** than FastEmbed (ONNX) on MiniLM-L6. Python/Rust/Dart APIs.
iOS (Metal) + Android (Vulkan) builds. 29 HuggingFace model repos.

### Part of the Crisp ecosystem

| Project | Role |
|---|---|
| **CrispEmbed** | This repo — text embedding engine (ggml), dense + sparse + ColBERT + reranking |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | Speech recognition engine (ggml) — 11 ASR backends, same philosophy for audio |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Flutter transcription app powered by CrispASR — desktop + mobile, fully offline |
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python ASR GUI with 9 backends (faster-whisper, mlx-whisper, voxtral, ...) |

## Status

**23+ models verified** (cos>=0.96), 43+ in registry:

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
| all-mpnet-base-v2 | MPNet | 768 | 0.999997 | 0.9998 | 0.99 |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 0.999441 | 0.9994 | -- |
| multilingual-e5-base | XLM-R | 768 | 0.999995 | 0.9999 | 0.99 |
| multilingual-e5-large | XLM-R | 1024 | 0.999997 | 0.9999 | 0.99 |
| granite-embedding-278m | XLM-R | 768 | 0.999984 | 0.9999 | 0.99 |
| granite-embedding-107m | XLM-R | 384 | 0.999986 | 0.9999 | 0.99 |
| bge-small-en-v1.5 | BERT | 384 | 0.999999 | 0.9999 | 0.99 |
| bge-base-en-v1.5 | BERT | 768 | 0.999994 | 0.9999 | 0.99 |
| bge-large-en-v1.5 | BERT | 1024 | 0.999992 | 0.9999 | 0.99 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000032 | 0.9999 | 0.99 |

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

**BidirLM-Omni Q4_K verified locally**: text, audio, and raw vision-patch embedding all load and emit 2048-d vectors from `bidirlm-omni-2.5b-q4_k.gguf`. Current CPU smoke benchmark on Apple M1:
text batch of 4 = 373.2 ms, JFK audio = 618.7 ms, synthetic 2x2 vision patches = 11.0 ms. Regression gate used for graph changes: cosine >= 0.99999 and max abs diff <= 1e-3 against saved baseline vectors.

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

# Prefix + capability inspection
./build/crispembed -m model.gguf --prefix "query: " --capabilities

# BidirLM-Omni text/audio/raw-vision patch embedding
# Put local paths in a gitignored `.env.local`, then source it here.
source .env.local
./build/crispembed -m bidirlm-omni-2.5b "Hello world"
./build/crispembed -m bidirlm-omni-2.5b --audio "$CRISPEMBED_BIDIRLM_AUDIO"
./build/crispembed -m bidirlm-omni-2.5b \
    --image-raw "$CRISPEMBED_BIDIRLM_PATCHES" --grid-thw 1,14,14

# Sparse / ColBERT retrieval (BGE-M3)
./build/crispembed -m bge-m3.gguf --sparse "Hello world"
./build/crispembed -m bge-m3.gguf --colbert "Hello world"

# Cross-encoder and bi-encoder reranking
./build/crispembed -m bge-reranker-v2-m3.gguf --rerank "capital of france" \
    "Paris is the capital of France." "Bicycles have two wheels."
./build/crispembed -m model.gguf --biencoder "capital of france" --top-n 2 \
    "Paris is the capital of France." "Berlin is the capital of Germany."

# CLI parity test
python tests/test_cli_parity.py --cli ./build/crispembed \
    --dense-model "$CRISPEMBED_DENSE_MODEL" \
    --retrieval-model "$CRISPEMBED_RETRIEVAL_MODEL" \
    --reranker-model "$CRISPEMBED_RERANKER_MODEL"

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

Wrapper parity script:

```bash
python tests/feature_parity.py \
  --dense-model "$CRISPEMBED_DENSE_MODEL" \
  --retrieval-model "$CRISPEMBED_RETRIEVAL_MODEL" \
  --reranker-model "$CRISPEMBED_RERANKER_MODEL"
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

Wrapper parity script:

```bash
cargo run -p crispembed --example feature_parity -- \
  "$CRISPEMBED_DENSE_MODEL" \
  "$CRISPEMBED_RETRIEVAL_MODEL" \
  "$CRISPEMBED_RERANKER_MODEL"
```

## Dart / Flutter

```yaml
# pubspec.yaml
dependencies:
  crispembed:
    path: <local Flutter plugin path>
```

```dart
import 'package:crispembed/crispembed.dart';

final model = CrispEmbed('model.gguf');

// Dense encoding
final vec = model.encode('Hello world');           // Float32List(384)
final batch = model.encodeBatch(['Hello', 'World']); // List<Float32List>

// Matryoshka truncation + prefix
model.setDim(128);
model.setPrefix('query: ');

// Bi-encoder reranking
final ranked = model.rerankBiencoder('query', ['doc1', 'doc2']);

// Sparse / ColBERT / cross-encoder (BGE-M3, rerankers)
if (model.hasSparse) {
  final sparse = model.encodeSparse('text');  // Map<int, double>
}

model.dispose();
```

Wrapper parity script:

```bash
cd flutter/crispembed
dart run example/feature_parity.dart \
  "$CRISPEMBED_DENSE_MODEL" \
  "$CRISPEMBED_RETRIEVAL_MODEL" \
  "$CRISPEMBED_RERANKER_MODEL" \
  "$CRISPEMBED_LIB"
```

Works on iOS (Metal GPU), Android (Vulkan/NEON), macOS, Linux, Windows.

## Feature Parity

Python, Rust, Dart, and the `crispembed` CLI now cover the same core inference features from the
shared C API: dense encode, batch encode, Matryoshka truncation, prefix control, sparse retrieval,
ColBERT multi-vector retrieval, cross-encoder reranking, and bi-encoder reranking.

The CLI still keeps some convenience-only UX that the wrappers do not mirror directly:

- CLI-only conveniences: `--list-models`, model-name auto-download, `--cache-dir`, `-f FILE`, `--json`, `--dim`, `--capabilities`.
- Wrapper-only convenience helpers: prefix getters and in-process ranking helper return types.

Inference capability parity is now aligned across all four entry points.

## Mobile (iOS / Android)

```bash
# iOS — xcframework with Metal GPU acceleration
./build-ios.sh                    # arm64 device + simulator
./build-ios.sh --device           # device only

# Android — NDK cross-compilation
./build-android.sh                # arm64-v8a + armeabi-v7a + x86_64
./build-android.sh --abi arm64-v8a --vulkan  # single ABI with Vulkan GPU
```

Output:
- iOS: `build-ios/CrispEmbed.xcframework/`
- Android: `build-android/<abi>/libcrispembed.so`

## Benchmarking

```bash
./benchmark.sh                          # single model, all engines
./benchmark.sh --multi                  # 6 models, all engines
./benchmark.sh -n 100 --skip-fastembed  # CrispEmbed + HF only, 100 runs

# RAG retrieval quality benchmark
python tests/bench_rag.py --lib build/libcrispembed.so --gguf model.gguf

# Reranking benchmark
python tests/bench_rerank.py --lib build/libcrispembed.so \
    --embed-gguf model.gguf --reranker-gguf reranker.gguf

# BidirLM-Omni text/audio/raw-vision benchmark with cosine regression check
PYTHONPATH=python python tests/benchmark_bidirlm.py \
    --model "$CRISPEMBED_MODEL" \
    --lib "$CRISPEMBED_LIB" \
    --save-baseline "$CRISPEMBED_BIDIRLM_BASELINE"
PYTHONPATH=python python tests/benchmark_bidirlm.py \
    --model "$CRISPEMBED_MODEL" \
    --lib "$CRISPEMBED_LIB" \
    --compare-baseline "$CRISPEMBED_BIDIRLM_BASELINE"
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

**MPNet encoder** (all-mpnet-base-v2):
- Token + Position(+offset) embeddings → Post-LN transformer with relative position bias → Mean pooling
- T5-style logarithmic bucket relative attention bias (32 buckets × n_heads)

**NomicBERT encoder** (nomic-embed-text-v1.5):
- Token embeddings (no position) + RoPE → Post-LN transformer + SwiGLU FFN → Mean pooling
- Rotary position embeddings (same as decoder path), no absolute position embeddings

**Cross-encoder reranker** (BGE-reranker-v2-m3, ms-marco-MiniLM, mxbai-rerank, etc.):
- `[CLS] query [SEP] document [SEP]` pair tokenization → CLS hidden state → `Linear(H,1)` → scalar score

**Qwen3 decoder** (Octen, F2LLM, Jina v5, Harrier-0.6B, Qwen3-Embed):
- Token embeddings + RoPE → RMSNorm + GQA with causal mask + SwiGLU → Last-token pooling

**Gemma3 decoder** (Harrier-270M):
- Token embeddings * sqrt(H) + RoPE → Gemma3 RMSNorm(1+w) + GQA + GeGLU → Last-token pooling

**BidirLM-Omni** (BidirLM-Omni-2.5B-Embedding):
- Text: bidirectional Qwen3 body (RoPE, GQA, RMSNorm, q_norm/k_norm, SwiGLU) → Mean pooling → 2048-d
- Audio (when CrispAudio is available): Whisper-shape encoder (Conv2D stem + 24-layer pre-LN encoder + proj1/GELU/proj2) → Mean pooling → same 2048-d shared space, enabling cross-modal cosine similarity
- Audio path uses the shared `crisp_audio` library from the configured CrispASR checkout; CMake auto-discovers it and `-DCRISP_AUDIO_DIR=...` can override that lookup.
- Vision: BidirLM/Qwen2VL-style vision tower in ggml (patch embed, interpolated position embedding, rotary attention, patch merger, DeepStack). The Python binding accepts images through the HF processor. The CLI accepts already-preprocessed float32 patch rows via `--image-raw patches.f32 --grid-thw T,H,W`; direct JPG/PNG preprocessing in C++ is still future work.
- Decoder text embedding uses `ggml_backend_sched`, matching the encoder/vision execution path while preserving baseline cosine outputs.
- Standalone image embedding skips DeepStack materialization because pooled image vectors do not use it; raw vision output and text-with-image still compute DeepStack.
- Local model cache convention: point `CRISPEMBED_CACHE_DIR` at your backing store and keep large GGUF/cache files out of the repo tree.

All via ggml graphs with GPU dispatch (ggml_backend_sched).
See [PLAN.md](PLAN.md), [LEARNINGS.md](LEARNINGS.md), [PERFORMANCE.md](PERFORMANCE.md).

## Credits

- [ggml](https://github.com/ggml-org/ggml) -- inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) -- shared core (gguf_loader, bpe.h)
- [sentence-transformers](https://www.sbert.net/) -- ground-truth validation
