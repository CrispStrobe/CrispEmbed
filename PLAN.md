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

- [x] GPU graph dispatch via ggml_backend_sched (encoder + decoder)
- [x] Matryoshka dimension truncation (-d N flag)
- [x] Graph/work buffer reuse (3.2x server throughput improvement)
- [x] BLAS/MKL build support (cmake -DGGML_BLAS=ON)
- [x] Windows build scripts (build-windows.bat, build-vulkan.bat, build-cuda.bat)
- [x] C++ quantizer with Q4_K/Q5_K/Q6_K (tools/quantize.cpp)
- [ ] True batched graph (single compute for multiple texts)
- [ ] KV cache for prefix-shared batches
- [ ] Reranker model support

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

### Optimizations completed

- ggml_backend_sched GPU dispatch (encoder + decoder full-graph)
- All 21+ models quantized (Q8_0 + Q4_K) and uploaded to HuggingFace
- Graph/work buffer reuse: 27.8 texts/s server throughput (gte-small)
- Matryoshka dimension truncation via -d N flag
- BLAS/MKL/CUDA/Vulkan/Metal build support
- Windows build scripts
- C++ quantizer with K-quant fallback chain
- QKV weight fusion (1 matmul vs 3 per layer)
- Flash attention with optional position bias mask

### RAG feature parity (April 17, 2026)

- [x] Full Python/Rust/Dart wrapper: sparse, ColBERT, reranker, set_dim, set_prefix
- [x] Bi-encoder reranking API (Python + Rust + Dart): cosine similarity ranking
- [x] Prompt prefix system (C/Rust/Python/Dart): auto-prepend query/passage prefixes
- [x] 21 verified embedding models (cos >= 0.999 vs HuggingFace)
- [x] 5 reranker models (bge-reranker-base, ms-marco L6/L12, mxbai-rerank xsmall/base)
- [x] 27 HuggingFace repos with GGUF models + README cards
- [x] RAG retrieval quality benchmark (tests/bench_rag.py): MRR@10, NDCG@10, Recall@k
- [x] Reranking benchmark (tests/bench_rerank.py): cross-encoder vs bi-encoder
- [x] Head-to-head benchmark vs FastEmbed (tests/bench_head2head.py):
  - MiniLM-L6: CrispEmbed **9.5x faster** single, **10.8x faster** batch
  - BGE-small: FastEmbed 1.7x faster (ONNX graph JIT optimization)
  - Arctic-M: tied on batch (126 vs 127ms)
  - cos = 0.999999–1.000000 cross-engine on all models
- [x] Demo apps (Python + Rust) for both CrispEmbed and CrispASR

### Architecture support (11 + omnimodal)

| Architecture | Status | Key features | Example models |
|---|---|---|---|
| BERT encoder | Complete (cos≥0.999) | Post-LN, GELU, WordPiece | MiniLM, GTE-small, BGE, arctic-xs |
| XLM-R encoder | Complete (cos≥0.999) | Post-LN, GELU, SentencePiece Viterbi, pos_offset=2 | E5, PIXIE-Rune, arctic-l-v2, granite |
| MPNet encoder | Complete (cos≥0.999) | Post-LN, GELU, relative position bias (T5-style buckets) | all-mpnet-base-v2 |
| NomicBERT encoder | Complete (cos=0.999) | Post-LN, SwiGLU, RoPE, no biases | nomic-embed-text-v1.5 |
| ModernBERT encoder | Complete (cos=0.97) | Pre-LN, fused ggml_geglu, RoPE, per-layer theta, BPE | gte-modernbert-base |
| GTE v1.5 encoder | Converter done | Pre-LN, fused ggml_geglu, RoPE, QKV+bias, CLS pooling | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | Partial | Post-LN, c2c only (no c2p/p2c disentangled) | mxbai-rerank (converter works) |
| Qwen3 decoder | Complete | RMSNorm, SwiGLU, RoPE, GQA, causal mask | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | Complete | Gemma RMSNorm(1+w), GeGLU, embed*sqrt(H) | Harrier-270M |
| BidirLM-Omni text | Complete (cos≥0.999) | Bidirectional Qwen3 body, mean-pool, MRoPE-aware (text-only collapses to NEOX) | BidirLM-Omni-2.5B-Embedding |
| BidirLM-Omni audio | Complete (cos=0.995) | crisp_audio Whisper-shape encoder, mean-pool to shared 2048-d | BidirLM-Omni-2.5B-Embedding |
| BidirLM-Omni vision | Complete (cos=0.999) | Qwen2VL ViT, 4-corner pos-embed gather, 2D rotate-half RoPE, patch merger + DeepStack | BidirLM-Omni-2.5B-Embedding |
| BidirLM-Omni text+image | Phase 3 (parity test in `tests/test_bidirlm_image_text.py`) | DeepStack injection at first 3 decoder layers, 3D interleaved-MRoPE positions | BidirLM-Omni-2.5B-Embedding |

### Bindings and platforms

| Binding | CrispEmbed | CrispASR |
|---|---|---|
| C API | Complete | Complete (whisper.h) |
| Python (ctypes) | Complete + tested | Complete + tested |
| Rust (crate) | Complete + tested | Complete + compiled |
| Dart/Flutter (FFI) | Created | Created |
| iOS (Metal) | CI green | CI green |
| Android (NDK) | CI green (arm64/armv7/x86_64) | CI green |
| Windows | CI green | CI green |
| macOS (Metal) | CI green | CI green |
| Linux | CI green | CI green |

### Remaining feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| CLIP-style image embedding | Medium | Medium | Pure-vision tower exists for BidirLM-Omni; CLIP would reuse the patch-embed + ViT scaffolding with its own preprocessor and contrastive head |
| In-process image preprocessing (C++) | Medium | Medium | Currently relies on HF `Qwen2VLImageProcessorFast` in Python; need a C++ port of `smart_resize + normalize + patchify` for the CLI / mobile bindings |
| SPLADE sparse model | Medium | Medium | Different sparse architecture from BGE-M3 |
| DeBERTa full disentangled | Low | High | c2p/p2c need per-layer position projections in ggml graph |
| Nomic v2 MoE | Low | High | MoE routing layer in encoder |
| Qwen3-VL multimodal | Low | High | Reuse BidirLM-Omni vision tower + DeepStack scaffolding; main delta is vision_config keys + cross-attention layout |
| Qwen3 4B/8B | Low | Low | Existing decoder path, just needs memory for larger models |

### CrispEmbed advantages over fastembed-rs

- **ColBERT multi-vector** retrieval (fastembed-rs doesn't have it)
- **Matryoshka dimension truncation** (fastembed-rs doesn't have it)
- **GGUF quantization** (Q8_0, Q4_K — smaller than ONNX INT8/INT4)
- **9.5x faster on MiniLM-L6** (most popular embedding model)
- **GPU dispatch** via ggml_backend_sched (CUDA/Metal/Vulkan)
- **Ollama-compatible** server with 4 API dialects
- **Flutter/Dart** wrapper for mobile apps
- **iOS/Android** build scripts with full CI
- **20MB binary** vs ~500MB Python+ONNX environment

### Original remaining items

- True batched graph for decoder models
- KV cache for prefix-shared decoder batches

### Multimodal status (April 2026)

- [x] BidirLM-Omni text path through `decoder_embed.cpp` (cos ≥ 0.999 vs HF bf16)
- [x] BidirLM-Omni audio path through `bidirlm_audio.cpp` + crisp_audio (cos = 0.995 vs HF)
- [x] BidirLM-Omni vision tower in `bidirlm_vision.cpp` (cos ≥ 0.999 vs HF bf16, image_embeds + deepstack slabs)
- [x] DeepStack injection + 3D interleaved-MRoPE in `decoder_embed.cpp` (Phase 3) — **validated cosine = 0.998903 vs HF bf16** on `bidirlm-omni-2.5b-q8_0.gguf` via `tests/test_bidirlm_image_text_lite.py`. q4_k drops to ~0.94 on the same path, identical to text-only q4_k (intrinsic quant floor — see LEARNINGS.md "q4_k quantization cosine ceiling"), not a multimodal-injection bug.
- [x] `crispembed_encode_text_with_image` C ABI + Python `encode_text_with_image()` wrapper
- [x] `crispembed_encode_with_image_ids` (pre-tokenized variant for parity tests)
- [x] CLI `--image FILE` (in-process preprocessor) + `--image-raw patches.f32 --grid-thw T,H,W`
- [x] Decoder `ggml_backend_sched` initialization (was previously CPU-only fallback)
- [x] Memory-efficient lite parity test (`tests/test_bidirlm_image_text_lite.py`):
      loads HF text + vision separately, reproduces `BidirLMOmniModel.forward` manually.
      Skips audio_tower, fits in 16 GB RAM, 4–5 min wall-clock.
- [x] In-process C++ image preprocessor (`src/image_preprocess.{h,cpp}`):
      smart_resize + Catmull-Rom bicubic+antialias + CLIP normalize + Qwen2VL patchify
      via stb_image. Cosine ≈ 0.97 vs HF Python preprocessor (residual is JPEG decoder
      differences PIL/libjpeg-turbo vs stb). C ABI: `crispembed_encode_image_file`,
      `crispembed_encode_text_with_image_file`, `crispembed_preprocess_image`.
- [x] Stale-GGUF fallbacks (`load_decoder_model`): recover image_token_id /
      vision_start / vision_end from `tokenizer.ggml.tokens` string match,
      spatial_merge_size from `bidirlm.vision.*`, mrope_section default to
      [24, 20, 20] for BidirLM-Omni when decoder.* keys are missing.
- [ ] JPEG-decoder parity (libjpeg-turbo vs stb_image — close 0.97 → 0.99 cosine)
- [ ] Image batching in `encode_text_with_image` (HF's `image_grid_thw` already supports it)
