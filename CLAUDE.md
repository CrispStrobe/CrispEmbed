# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CrispEmbed is a C/C++ text embedding inference engine built on ggml. It loads GGUF models and produces embedding vectors with GPU acceleration (Metal/CUDA/Vulkan). No Python runtime, no ONNX dependency. Supports 13 verified embedding models across BERT, XLM-R, Qwen3, and Gemma3 architectures.

## Build commands

```bash
# macOS (Metal GPU) — the standard dev build
./build-macos.sh                    # Metal + Accelerate + embedded shaders
./build-macos.sh --shared           # Also build libcrispembed.dylib for Python
./build-macos.sh --clean            # Clean rebuild (needed after ggml changes)

# Manual cmake (Linux/cross-platform)
cmake -S . -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCRISPEMBED_BUILD_SHARED=ON
cmake --build build -j

# GPU variants
cmake -S . -B build -DGGML_CUDA=ON      # NVIDIA
cmake -S . -B build -DGGML_VULKAN=ON    # Cross-platform GPU
```

Binaries land in `build/`: `crispembed`, `crispembed-server`, `crispembed-quantize`.

## Running

```bash
# CLI (auto-downloads models on first use)
./build/crispembed -m all-MiniLM-L6-v2 "Hello world"

# Server (model loaded once, HTTP API)
./build/crispembed-server -m all-MiniLM-L6-v2 --port 8080

# Quantize
./build/crispembed-quantize model.gguf model-q8_0.gguf q8_0
```

## Testing

```bash
# Validate GGUF output matches HuggingFace (needs pip: sentence-transformers)
python tests/test_vs_hf.py --model sentence-transformers/all-MiniLM-L6-v2 \
    --gguf ~/.cache/crispembed/all-MiniLM-L6-v2.gguf --binary ./build/crispembed

# Test all cached models
python tests/test_all_models.py

# Benchmark (creates .bench-venv for Python deps automatically)
./benchmark.sh -n 20                      # single model
./benchmark.sh --multi -n 20              # 3 models, all engines
./benchmark.sh --skip-hf --skip-fastembed  # CrispEmbed only
```

## Architecture

### Inference pipeline

Text → **Tokenizer** (auto-detected from GGUF: WordPiece/SentencePiece/BPE) → Token+Position+Type embeddings → N × Transformer layers → **Pooling** (mean/CLS/last-token) → L2 normalization → `float[dim]`

### Source layout

- **`src/crispembed.h`** — Public C API: `crispembed_init`, `crispembed_encode`, `crispembed_encode_batch`, `crispembed_free`. This is the interface everything else builds on.
- **`src/crispembed.cpp`** — Core inference engine (~860 lines). Builds ggml computation graphs for encoder models (BERT/XLM-R). Handles backend init (`ggml_backend_init_best()` picks Metal/CUDA/CPU), graph caching, QKV weight fusion, batched encoding.
- **`src/decoder_embed.cpp`** — Separate forward pass for decoder-based embedders (Qwen3/Gemma3). Uses RoPE, GQA, causal attention, last-token pooling.
- **`src/tokenizer.cpp`** — WordPiece tokenizer (BERT-style).
- **`src/tokenizer_spm.cpp`** — SentencePiece Unigram tokenizer (Viterbi DP, NOT bigram merge — see LEARNINGS.md).
- **`src/tokenizer_bpe.cpp`** — GPT-2/Gemma BPE tokenizer.
- **`src/core/gguf_loader.cpp`** — GGUF file parser, tensor weight loading.

### Model architecture dispatch

The model type is auto-detected from GGUF metadata:
- **Encoder models** (BERT, XLM-R): `crispembed.cpp` → `encode_tokens()` / `encode_tokens_batch()`
- **Decoder models** (Qwen3, Gemma3): `decoder_embed.cpp` → `decoder_encode_tokens()`

Detection heuristic: presence of `blk.0.ffn_gate` tensor → decoder path.

### Tokenizer dispatch

Auto-detected from GGUF `tokenizer.ggml.type`: 0=WordPiece, 1=BPE, 2=SentencePiece. Fallback heuristic: vocab > 100K → SentencePiece.

### Server API endpoints

`examples/server/server.cpp` exposes four API dialects:
- `POST /embed` — native: `{"texts": ["..."]}`
- `POST /v1/embeddings` — OpenAI-compatible
- `POST /api/embed` — Ollama-compatible (batch)
- `POST /api/embeddings` — Ollama-compatible (legacy single)

### Python wrapper

`python/crispembed/_binding.py` — ctypes binding to `libcrispembed.dylib`. Single-text calls use `crispembed_encode()`, multi-text calls use `crispembed_encode_batch()` (single C call, true GPU-batched inference).

### Model auto-download

`examples/cli/model_mgr.cpp` contains a registry of 13 models with HuggingFace URLs. Models are cached to `~/.cache/crispembed/`. The CLI and server accept short names (e.g., `-m all-MiniLM-L6-v2`) and resolve + download automatically.

## Key technical details

- **Metal shaders must be embedded**: `build-macos.sh` passes `-DGGML_METAL_EMBED_LIBRARY=ON`. Without this, Metal init silently falls back to CPU. Verify with stderr output: "using embedded metal library" + "using MTL0 backend".
- **ggml is a git submodule**: Run `git submodule update --init --recursive` if `ggml/CMakeLists.txt` is missing.
- **Graph caching**: The ggml computation graph is built once per sequence length and reused. `encode_tokens_batch()` sorts texts by token count for cache hits.
- **QKV fusion**: For encoder models, separate Q/K/V weight matrices are fused into a single `[3H, H]` matrix at load time, reducing kernel launches.
- **Batch encoding**: `crispembed_encode_batch()` runs all texts through a single ggml graph with 4D flash attention. This is significantly faster than per-text encoding.
- **GGUF conversion scripts**: `models/convert-bert-to-gguf.py` (encoder models) and `models/convert-decoder-embed-to-gguf.py` (Qwen3/Gemma3 decoder models). Pre-converted models available at huggingface.co/cstr/.
