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

## Status (May 2026)

### Verified working — 40+ models, cos >= 0.965 vs HF

25 encoder models + 11 decoder models + 12 rerankers + 1 SPLADE + 2 multimodal
= 45 models in registry, 151 GGUF variants published on HF
(typical per-model: F32 + Q8_0 + Q4_K; about a dozen also have Q5_K /
Q6_K / F16). The 4B and 8B Qwen3-Embedding / Octen-Embedding variants
land at Q4_K cos ≥ 0.965 vs **bf16** HF reference (full f32 doesn't
fit in 16 GB RAM for the 8B path).

Key parity results (cos vs HuggingFace reference):

| Model | Type | Dim | CosSim |
|-------|------|-----|--------|
| all-MiniLM-L6-v2 | BERT | 384 | 1.000000 |
| bge-small/base/large-en-v1.5 | BERT | 384/768/1024 | 1.000000 |
| gte-base/large-en-v1.5 | GTE | 768/1024 | 1.000000 |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 1.000000 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000000 |
| all-mpnet-base-v2 | MPNet | 768 | 1.000000 |
| multilingual-e5-small/base/large | XLM-R | 384/768/1024 | 1.000000 |
| snowflake-arctic-embed-m/l | BERT/XLM-R | 768/1024 | 1.000000 |
| bge-m3 (dense+sparse+ColBERT) | XLM-R | 1024 | 1.000000 |
| splade-pp-en-v1 | BERT SPLADE | 768 | 1.000000 |
| granite-embedding-278m/107m | XLM-R | 768/384 | 1.000000 |
| pixie-rune-v1 | XLM-R | 1024 | 0.999993 |
| octen-0.6b | Qwen3 | 1024 | 0.999891 |
| octen-8b | Qwen3 | 4096 | 0.965 (Q4_K vs bf16 HF) |
| qwen3-embed-4b | Qwen3 | 2560 | 0.974 (Q4_K vs bf16 HF) |
| harrier-0.6b / harrier-270m | Qwen3/Gemma3 | 1024/640 | 0.999959/948 |
| jina-v5-nano/small | Qwen3 | 1024 | 0.999941 |
| bge-reranker-v2-m3 | XLM-R reranker | - | verified |
| ms-marco-MiniLM-L-6/12-v2 | BERT reranker | - | verified |

Pending HF publication (cpp registry wired, GGUFs being produced via the
Kaggle qwen3-quantize notebook): `octen-4b`, `qwen3-embed-8b`.

### Supported architectures

| Architecture | Tokenizer | Key features | Models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, MPNet, SPLADE |
| NomicBERT | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| GTE v1.5 | WordPiece | Pre-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| MPNet | WordPiece | Post-LN, relative attn bias | all-mpnet-base-v2 |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU FFN, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M |
| ViT (SigLIP) | - | Conv2D patch embed, transformer | siglip-base |
| CNN (SCRFD) | - | FPN, anchor decode, NMS | scrfd-det-10g |
| CNN (ArcFace) | - | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |

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
      smart_resize + Catmull-Rom bicubic+antialias + per-channel normalize +
      Qwen2VL patchify via stb_image. C ABI: `crispembed_encode_image_file`,
      `crispembed_encode_text_with_image_file`, `crispembed_preprocess_image`,
      `crispembed_preprocess_image_rgb` (caller-supplied RGB bytes for byte-tight
      JPEG-decoder parity when needed). Empirical cosine vs HF Python preprocessor
      on `/tmp/cat.jpg`: pixel_values 0.999989, encode_image embedding 0.999984.
      Sub-1e-5 residual is sub-pixel torchvision-uint8 bicubic kernel quantization
      (PyTorch uses int16 weights for the uint8 AA path; we use float weights).
- [x] **BPE special-token handling** (`src/tokenizer_bpe.cpp`): GPT-2 byte-level
      BPE now pre-splits on `<|...|>`-shaped vocab entries (Qwen-style added
      tokens like `<|im_start|>`, `<|image_pad|>`, `<|vision_start|>`). Unblocks
      the fully-native `encode_text_with_image_file` path (no `transformers`
      runtime dependency). End-to-end cosine vs HF-tokenized + HF-preproc
      reference on q4_k: **0.999590**.
- [x] Stale-GGUF fallbacks (`load_decoder_model`): recover image_token_id /
      vision_start / vision_end from `tokenizer.ggml.tokens` string match,
      spatial_merge_size from `bidirlm.vision.*`, mrope_section default to
      [24, 20, 20] for BidirLM-Omni when decoder.* keys are missing.
- [x] Converter writes `bidirlm.vision.image_mean / image_std / min_pixels /
      max_pixels` into the GGUF so the C++ runtime can read preprocessing
      hyperparameters from the model file (currently consumed via hardcoded
      defaults; trivial to wire when a non-BidirLM VL model lands).
- [x] Image batching in `encode_text_with_image`. The C ABI accepts `n_images > 1`
      via concatenated `pixel_patches` + multi-row `grid_thw`. Smoke-validated
      on `bidirlm-omni-2.5b-q4_k.gguf` with the cat image replicated twice:
      588 image_pad tokens, output shape (2048,) norm 1.0, deterministic on
      rerun, distinct from 1-image embedding (cos 0.984 — expected since
      prompts and DeepStack injection positions differ). Run with
      `tests/test_bidirlm_image_text_lite.py --n-images 2 --gguf …` for the
      formal HF-parity cosine when system memory permits.

---

## Phase 8: Vision — Image Embeddings, Face Detection & Recognition

CrispEmbed already has a ViT vision tower for BidirLM-Omni (text+image+audio
cross-modal embedding). This phase extends vision support to standalone image
embedding models and face analysis, using only **commercially permissive**
(Apache 2.0 / MIT) models.

### 8A. CLIP / SigLIP Image Embedding — DONE ✓

**Status:** cos=0.996 vs HF. Uploaded to [cstr/siglip-base-GGUF](https://huggingface.co/cstr/siglip-base-GGUF).

- [x] GGUF converter (`models/convert-siglip-to-gguf.py`) — handles SigLIP + CLIP
- [x] ViT forward path (`src/vit_embed.cpp`) — conv2d patch embed → transformer → mean pool
- [x] Image preprocessing (stb_image load → bilinear resize → normalize)
- [x] CLI: `crispembed -m siglip-base.gguf --image photo.jpg`
- [x] Unit test (`tests/test_siglip_converter.py`) — structure + weight parity
- [ ] SigLIP attention pooling head (mean pool works, attn pool for full parity)
- [ ] CLIP text encoder (causal mask variant)
- [ ] Quantization (conv2d needs F32 kernel — selective quant needed)
- [ ] Python wrapper `encode_image()`
- [ ] Convert + upload SigLIP-large, CLIP-base, CLIP-large

### 8B. Face Detection — SCRFD — DONE ✓

**Status:** Scores match ONNX Runtime (max 0.80 vs 0.80 on Lenna).
Uploaded to [cstr/scrfd-det-10g-GGUF](https://huggingface.co/cstr/scrfd-det-10g-GGUF).

- [x] GGUF converter (`models/convert-face-to-gguf.py`) — ONNX→GGUF with BN precompute
- [x] Generic ONNX graph replayer (Conv, ReLU, Add, Pool, Resize, Concat, Sigmoid, BNPrecomputed)
- [x] FPN: lateral convs + top-down upsample + bottom-up downsample
- [x] Detection heads: cls/reg/kps at 3 strides (8, 16, 32)
- [x] Anchor decode: grid centers + distance regression
- [x] NMS with IoU 0.4 threshold
- [x] CLI: `crispembed -m scrfd.gguf --detect photo.jpg [--json]`
- [x] 5-point landmark output (kps decoded at all strides)
- [x] Face alignment: 5-landmark similarity transform to 112×112 (face_align.cpp)
- [x] Alignment verified: MAE=0.00 vs InsightFace norm_crop
- [x] Letterbox preprocessing + coordinate scaling to original image dimensions
- [x] C API: `crispembed_detect_faces()`
- [x] Python wrapper (`CrispFace.detect()`)
- [x] Configurable confidence threshold from CLI (`--conf`)
- [x] SCRFD detection quality fixed (anchor decode indexing: channel-last vs interleaved)
- [x] Detection counts now match InsightFace exactly (1/1/4/4/8 faces on test images)
- [ ] Configurable input size (currently hardcoded 640×640)

### 8C. Face Recognition — AuraFace + SFace — DONE ✓

**Status:** cos=0.9999 vs ONNX for both models. Same-person matching verified.
Uploaded to [cstr/auraface-v1-GGUF](https://huggingface.co/cstr/auraface-v1-GGUF)
and [cstr/sface-GGUF](https://huggingface.co/cstr/sface-GGUF).

- [x] GGUF converter with BN folding (SFace) and BN precompute (AuraFace)
- [x] SFace MobileFaceNet: hardcoded sequential path (27 conv, PReLU, 128-D)
- [x] AuraFace ResNet-100: generic graph replay (255 nodes, 49 residual Add, 512-D)
- [x] CLI: `crispembed -m sface.gguf --face face.jpg`
- [x] Parity verified vs ONNX Runtime on real face photos
- [x] Face alignment: similarity transform with correct normal equations (4 sign fixes)
- [x] w600k_r50 (InsightFace buffalo_l ArcFace) also converted + verified (cos=0.995 vs IF)
- [x] Cross-image face matching verified on real photos (same person cos=0.56-0.76)
- [x] C API: `crispembed_encode_face()`, `crispembed_face_pipeline()`
- [x] Python wrapper (`CrispFace`, `CrispFacePipeline`)
- [x] Rust wrapper (`CrispFace`, `CrispFacePipeline`)
- [x] Dart/Flutter wrapper (`CrispFace`, `CrispFacePipeline`)
- [ ] Quantization (Q8_0, Q4_K)
- [ ] YuNet lightweight detection alternative

### Implementation order — REVISED

Phase 8 core is complete. Face alignment bug (4 sign errors in normal equations)
has been fixed and verified against InsightFace norm_crop (MAE=0.00).
Full pipeline (detect→align→encode) produces cos=0.994-0.999 vs InsightFace ArcFace.
Cross-image face matching works correctly on real photos.

Completed (May 12, 2026):
- [x] Letterbox preprocessing + coordinate scaling (cnn_embed::detect_file)
- [x] CLI pipeline mode: `crispembed --face-pipeline --det scrfd.gguf -m arcface.gguf img1.jpg img2.jpg`
- [x] C API: `crispembed_face_init/detect_faces/encode_face/face_pipeline/face_free`
- [x] Python wrapper: `CrispFace`, `CrispFacePipeline`
- [x] Rust wrapper: `CrispFace`, `CrispFacePipeline` in crispembed crate
- [x] Dart wrapper: `CrispFace`, `CrispFacePipeline` in flutter/crispembed
- [x] Server API: `/detect`, `/face` endpoints (crispembed-server --det --rec)

Remaining work:
1. **CrispLens integration** — update `crispembed_client.py` for face pipeline
2. **Additional models** — SigLIP-large, CLIP-large, YuNet, SFace int8

### Known issues (v0.4.0)

1. **NomicBERT** — **RESOLVED**. Root cause: gate/up weights (fc11↔fc12) were
   swapped in old GGUF; also needed Ollama tensor name fallback (`blk.N.ffn_gate.weight`).
   Reconverted to Ollama format. F32 cos=1.0, Q8_0 cos=0.998.
   SwiGLU is too sensitive for aggressive quants (Q5_K cos~0.95, Q4_K cos~0.85) —
   only F32 and Q8_0 uploaded to HF.

2. **EmbeddingGemma-300m parity** — **RESOLVED** (cos=1.0000 F32, 0.9998 Q8_0,
   0.9954 Q5_K). Root causes identified and fixed:
   - Missing `is_bidirectional=1` GGUF key (model has `use_bidirectional_attention: true`)
   - Wrong pooling: EmbeddingGemma uses SentenceTransformer mean-pool + Dense(768→3072)
     + Dense(3072→768) + L2-normalize; was doing last-token pooling with no Dense
   - BPE merges not loading (converter used `hf_hub_download` with local snapshot path)
   - Dense layers being quantized (runtime reads them as F32; `dense.*` now excluded)
   - cmake build target: `crispembed` builds only libcrispembed-static.a; executable
     needs `crispembed-cli` target
   All four v5 GGUFs (F32, Q8_0, Q5_K, Q4_K) uploaded to `cstr/embeddinggemma-300m-GGUF`.

3. **Jina v5 nano/small** — **RESOLVED**. Root cause: models use task-specific
   LoRA adapters (retrieval, text-matching, clustering, classification); old GGUFs
   had base weights only. Fixed: converter now merges `retrieval` adapter via
   `model.merge_and_unload()`. Nano F32 cos=1.0, Small F32 cos=0.9999.

4. **all-mpnet-base-v2** — **RESOLVED**. Root cause: old GGUF was missing
   `relative_attention_bias.weight` [32,12]. Reconverted with bias tensor.
   cos=0.987–0.999 (short sequences slightly sensitive to rel-bias numerics).

5. **gte-modernbert-base** — **RESOLVED**. Validation wrongly required `ln1`
   for pre-LN models; ModernBERT only has `ln2` (pre-FFN norm). Fixed validation.
   cos=0.9999.

6. **Full regression sweep (2026-05-17)**: 34 models tested, all pass.
   8 BERT, 7 XLM-R, 3 special encoders (Nomic/ModernBERT/SPLADE),
   7 decoder (Qwen3/Gemma3), 3 rerankers, 1 MPNet. 5 models fixed and
   re-uploaded to HF (EmbeddingGemma, Nomic, Jina v5 nano/small, MPNet).

7. **DeBERTa-v2 disentangled attention not implemented**: mxbai-rerank-xsmall-v1
   and mxbai-rerank-base-v1 use DeBERTa-v2's content-to-position (c2p) and
   position-to-content (p2c) relative attention bias with log-bucket encoding
   (position_buckets=256). This is a fundamentally different attention mechanism
   from standard BERT/RoPE and requires a custom implementation:
   - Relative position embedding matrix (learned, 512 entries)
   - Log-bucket position encoding for positions > bucket_size
   - Three attention score components: c2c + c2p + p2c
   All other rerankers (ms-marco-MiniLM, bge-reranker, jina-reranker) work
   correctly — use those instead.

### Commercially permissive stack (no NC restrictions)

The full pipeline uses only Apache 2.0 / MIT models:
- Text: any CrispEmbed encoder model (BERT/XLM-R/etc.)
- Image: SigLIP (Apache 2.0) or CLIP (MIT)
- Face detection: SCRFD (Apache 2.0) or YuNet (Apache 2.0)
- Face recognition: AuraFace-v1 512-D (Apache 2.0) or SFace 128-D (Apache 2.0)
- Face landmarks: MediaPipe FaceLandmarker (Apache 2.0)
- Audio: CrispASR (our own, Apache 2.0)

This replaces ALL non-commercial dependencies in CrispLens.
AuraFace-v1 is schema-compatible with InsightFace buffalo_l (same 512-D
ArcFace embedding space) — existing CrispLens databases work without
re-embedding when switching from buffalo_l to AuraFace.

### Testing methodology for Phase 8

Every new model/forward path follows this validation protocol:

1. **Python reference dump** (`tools/dump_reference.py`)
   - Load HF model, run forward pass with hooks on every layer
   - Capture: patch_embed output, each enc_layer output, pooling output, final embedding
   - Write to GGUF reference archive

2. **C++ per-layer diff** (`src/crispembed_diff.h`)
   - Load reference GGUF
   - Run CrispEmbed forward path with dump mode (tag intermediates)
   - Compare each stage: cosine_min, max_abs, rms
   - First layer where cos < 0.999 = bug location

3. **End-to-end parity** (`tests/test_all_parity.py`)
   - F32 cos ≥ 0.95 vs HF reference (PASS/FAIL gate)
   - Q8_0 and Q4_K quant degradation measurement
   - Cross-input diversity check (detect degenerate outputs)

4. **Live tests on example files**
   - Vision: test on standard images (cat, dog, cityscape, face)
   - Face detection: test on multi-face images, verify bounding boxes
   - Face recognition: test identity matching on LFW or similar dataset
   - Cross-modal: text "a photo of a cat" vs actual cat image → cos > 0.2

5. **Unit tests** (in `tests/`)
   - Converter output validation (tensor names, shapes, metadata)
   - Tokenizer / image preprocessor correctness
   - Individual op verification (conv2d, batch norm, attention pooling)

---

## Phase 9: Handwritten Math OCR — HMER (DenseNet-121 + GRU Attention)

Port the Pytorch-HMER model (MIT license, whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition)
to CrispEmbed as a GGUF model with C++ inference. This complements the existing
printed-math DeiT+TrOCR pipeline with a dedicated handwritten-math recognizer.

### Source model

- **Repo:** <https://github.com/whywhs/Pytorch-Handwritten-Mathematical-Expression-Recognition>
- **License:** MIT
- **Dataset:** CROHME 2016 (handwritten math expressions)
- **Checkpoints:** `model/encoder_*.pkl` (21.5 MB) + `model/attn_decoder_*.pkl` (10.3 MB)
- **Total params:** ~7.9M (encoder ~5.3M + decoder ~2.6M)
- **Estimated Q4_K size:** 4–5 MB

### Architecture

**Encoder: Modified DenseNet-121 (3 dense blocks, 2 input channels)**

```
Input: (B, 2, H, W)  — ch0 = grayscale [0,1], ch1 = padding mask (1=valid, 0=pad)

features.conv0_m:      Conv2d(2, 64, 7×7, stride=2, pad=3, bias=False)
features.norm0:        BatchNorm2d(64)
features.relu0:        ReLU
features.pool0:        MaxPool2d(3, stride=2, pad=1)
                       → (B, 64, H/4, W/4)

features.denseblock1:  6 DenseLayers, growth=32
  denselayer{1..6}:    BN(in) → ReLU → Conv1×1(in, 128, bias=F) → BN(128) → ReLU → Conv3×3(128, 32, pad=1, bias=F)
  output channels:     64 + 6×32 = 256
features.transition1:  BN(256) → ReLU → Conv1×1(256, 128, bias=F) → AvgPool2×2
                       → (B, 128, H/8, W/8)

features.denseblock2:  12 DenseLayers, growth=32
  output channels:     128 + 12×32 = 512
features.transition2:  BN(512) → ReLU → Conv1×1(512, 256, bias=F) → AvgPool2×2
                       → (B, 256, H/16, W/16)

features.denseblock3:  24 DenseLayers, growth=32
  output channels:     256 + 24×32 = 1024

features.norm5:        BatchNorm2d(1024)
ReLU (in forward)
                       → (B, 1024, H/16, W/16)
```

Spatial downsampling: 16× in both dimensions.

Each DenseLayer (bottleneck):
- BN(in_ch) → ReLU → Conv1×1(in_ch → 4×growth=128) → BN(128) → ReLU → Conv3×3(128 → growth=32)
- Output: concat(input, new_features) along channel dim

**Decoder: AttnDecoderRNN (GRU + Bahdanau attention + coverage)**

Instantiated as `AttnDecoderRNN(hidden_size=256, output_size=112, dropout_p=0.5)`.

Layers (with GGUF tensor names):

| PyTorch name (module.prefix stripped) | Shape | GGUF name |
|---|---|---|
| `embedding.weight` | [112, 256] | `dec.embedding.weight` |
| `gru1.weight_ih` | [768, 256] | `dec.gru1.weight_ih` |
| `gru1.weight_hh` | [768, 256] | `dec.gru1.weight_hh` |
| `gru1.bias_ih` | [768] | `dec.gru1.bias_ih` |
| `gru1.bias_hh` | [768] | `dec.gru1.bias_hh` |
| `gru.weight_ih` | [768, 1024] | `dec.gru.weight_ih` |
| `gru.weight_hh` | [768, 256] | `dec.gru.weight_hh` |
| `gru.bias_ih` | [768] | `dec.gru.bias_ih` |
| `gru.bias_hh` | [768] | `dec.gru.bias_hh` |
| `hidden.weight` | [256, 256] | `dec.hidden.weight` |
| `hidden.bias` | [256] | `dec.hidden.bias` |
| `hidden2.weight` | [128, 256] | `dec.hidden2.weight` |
| `hidden2.bias` | [128] | `dec.hidden2.bias` |
| `emb2.weight` | [128, 256] | `dec.emb2.weight` |
| `emb2.bias` | [128] | `dec.emb2.bias` |
| `ua.weight` | [256, 1024] | `dec.ua.weight` |
| `ua.bias` | [256] | `dec.ua.bias` |
| `uf.weight` | [256, 1] | `dec.uf.weight` |
| `uf.bias` | [256] | `dec.uf.bias` |
| `v.weight` | [1, 256] | `dec.v.weight` |
| `v.bias` | [1] | `dec.v.bias` |
| `wc.weight` | [128, 1024] | `dec.wc.weight` |
| `wc.bias` | [128] | `dec.wc.bias` |
| `out.weight` | [112, 128] | `dec.out.weight` |
| `out.bias` | [112] | `dec.out.bias` |
| `conv1.weight` | [1, 1, 3, 3] | `dec.conv1.weight` |
| `conv1.bias` | [1] | `dec.conv1.bias` |
| `conv_tan.weight` | [256, 256, 3, 3] | `dec.conv_tan.weight` |
| `conv_tan.bias` | [256] | `dec.conv_tan.bias` |
| `bn1.weight` | [256] | `dec.bn1.weight` |
| `bn1.bias` | [256] | `dec.bn1.bias` |
| `bn1.running_mean` | [256] | `dec.bn1.running_mean` |
| `bn1.running_var` | [256] | `dec.bn1.running_var` |

Unused in forward (skip): `emb`, `conv_et`, `bn`, `relu`.

Decoder forward pass (single step):

```
1. embedded = Embedding(prev_token)                    → (256,)
2. st = GRUCell_1(embedded, hidden)                    → (256,)  [query state]
3. hidden1 = Linear_hidden(st)                         → (256,)  reshaped to (1,1,256)
4. decoder_attention = Conv2d_1(prev_attention)         → (1, H', W')  [3×3 conv, 1→1 ch]
5. attention_sum += decoder_attention                   → (1, H', W')  [cumulative coverage]
6. encoder_out1 = Linear_ua(enc_features)              → (H', W', 256)
7. attention_sum1 = Linear_uf(attention_sum)            → (H', W', 256)
8. et = hidden1 + encoder_out1 + attention_sum1         → (H', W', 256)
9. et = Conv2d_tan(et)                                  → (256, H', W')  [3×3 conv, 256→256]
10. et = et * mask                                      → mask out padding
11. et = BatchNorm2d_1(et)
12. et = tanh(et)
13. et = Linear_v(et)                                   → (H', W', 1)  → squeeze → (H', W')
14. alpha = softmax(et) over spatial dims                → (1, H', W')  [attention weights]
15. ct = sum(alpha * encoder_outputs)                    → (1024,)  [context vector]
16. hidden_next = GRUCell_2(ct, st)                     → (256,)
17. hidden2 = Linear_hidden2(hidden_next)               → (128,)
18. embedded2 = Linear_emb2(embedded)                    → (128,)
19. ct2 = Linear_wc(ct)                                 → (128,)
20. output = log_softmax(Linear_out(hidden2 + embedded2 + ct2))  → (112,)
```

Greedy decoding: start with `<sos>` (token 111), argmax until `<eol>` (token 0) or max 48 steps.

Initial hidden: `tanh(randn(1, 256) * mean(encoder_output))` — for inference we use zeros.

### Vocabulary

112 tokens. Index 0 = `<eol>`, index 111 = `<sos>`, indices 1–110 = LaTeX tokens.
Stored in GGUF as `tokenizer.tokens` string array (same pattern as pix2tex).

### Image preprocessing

1. Load image as grayscale, values [0, 255] uint8
2. Convert to float, divide by 255 → [0, 1]
3. Create mask channel: 1.0 where image exists, 0.0 where padded
4. Stack as 2-channel input: (2, H, W)
5. No fixed resolution — variable size, constrained to H×W ≤ 100,000 pixels

### Implementation plan

#### Step 1: GGUF converter (`models/convert-hmer-to-gguf.py`)

Convert the PyTorch state_dict checkpoints to a single GGUF file.

- Load encoder `.pkl` and decoder `.pkl` via `torch.load()`
- Strip `module.` prefix (DataParallel wrapping)
- For BatchNorm layers: store weight, bias, running_mean, running_var
  (fold at inference time in C++ for conv-BN fusion, or precompute in converter)
- **BN folding strategy:** For Conv+BN pairs (all DenseNet convs are followed by BN),
  precompute fused weight/bias in the converter:
  ```
  scale = gamma / sqrt(running_var + eps)
  W_fused = W * scale.reshape(-1, 1, 1, 1)
  B_fused = beta - running_mean * scale    (convs have no bias, so just this)
  ```
  This eliminates all BN tensors from the GGUF — cleaner and faster inference.
  Exception: `dec.bn1` is applied after `conv_tan` in the attention mechanism
  and cannot be folded (conv_tan output is added to other terms first).
- Store dictionary as `tokenizer.tokens` array in GGUF metadata
- Store hyperparameters as GGUF KV:
  - `hmer.encoder.growth_rate` = 32
  - `hmer.encoder.block_config` = [6, 12, 24]
  - `hmer.encoder.num_init_features` = 64
  - `hmer.decoder.hidden_size` = 256
  - `hmer.decoder.output_size` = 112
  - `hmer.decoder.sos_token` = 111
  - `hmer.decoder.eol_token` = 0
  - `hmer.decoder.max_seq_len` = 48

Tensor naming convention:
- Encoder: `enc.features.{layer_path}.weight` / `.bias`
- Decoder: `dec.{layer_name}.weight` / `.bias`

Expected tensor count: ~280 encoder (42 DenseLayers × ~4 tensors + transitions + stem)
+ ~34 decoder = ~314 tensors before BN folding; fewer after folding.

#### Step 2: C++ inference — DenseNet encoder (`src/hmer_ocr.cpp`)

New file alongside `math_ocr.cpp`. Shares `core/gguf_loader.h`.

DenseNet forward pass in C++:
- Stem: Conv2d(2→64, 7×7, s=2, p=3) → BN → ReLU → MaxPool(3, s=2, p=1)
- DenseBlock: for each layer, compute bottleneck and concat to running feature map
- Transition: BN → ReLU → Conv1×1 → AvgPool2×2
- Final: BN → ReLU

Key GGML ops needed:
- `ggml_conv_2d` — already used in CNN embed (SCRFD, SigLIP)
- `ggml_concat` — for DenseNet feature concatenation (dim=channel)
- `ggml_pool_2d` — MaxPool and AvgPool
- `ggml_relu` — already available
- `ggml_norm` or manual BN — if BN is not folded into conv

For BN-folded convs, it's just `ggml_conv_2d` + `ggml_add` (bias) + `ggml_relu`.

Since the encoder output is spatial (1024, H/16, W/16) and variable-size,
we need dynamic graph building (not fixed-size like DeiT's 384×384).

#### Step 3: C++ inference — GRU attention decoder (`src/hmer_ocr.cpp`)

GRU cell implementation (standard formulas):
```
z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
h' = (1 - z) * n + z * h
```

Where GRU weight_ih = [W_iz; W_ir; W_in] stacked (768 = 3×256).

Attention mechanism: follows the 20-step decoder forward pass above.

State between steps:
- `hidden`: (256,) — GRU hidden state
- `attention_sum`: (1, H', W') — cumulative coverage
- `decoder_attention`: (1, H', W') — previous step's attention weights

#### Step 4: C API (`src/hmer_ocr.h`)

```c
typedef struct hmer_ocr_context hmer_ocr_context;

hmer_ocr_context * hmer_ocr_init(const char * model_path, int n_threads);
void               hmer_ocr_free(hmer_ocr_context * ctx);

// Image-based recognition (grayscale float [0,1])
const char * hmer_ocr_recognize(
    hmer_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len);

// Raw bytes (RGB/grayscale uint8) — converts internally
const char * hmer_ocr_recognize_raw(
    hmer_ocr_context * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    int * out_len);
```

Same pattern as `math_ocr_recognize` / `math_ocr_recognize_raw`.

#### Step 5: Validation

1. Python reference: run PyTorch model on CROHME test images, dump encoder output + decoder token sequence
2. C++ per-step comparison: encoder cosine sim ≥ 0.99, decoder argmax tokens identical
3. End-to-end: run on 10+ test images, verify LaTeX output matches PyTorch

#### Step 6: Quantization + integration

1. Quantize: F32 → F16 → Q8_0 → Q4_K using existing `crispembed-quantize`
2. Test each quant level for accuracy degradation on CROHME test set
3. Add to CrispCalc `ocr_model_manager.dart` as `hmer-hw-q4_k.gguf`
4. Wire into `ocr_providers_init.dart` FFI bindings
5. Upload to HuggingFace: `cstr/hmer-handwritten-math-gguf`

### File changes

New files:
- `models/convert-hmer-to-gguf.py` — PyTorch → GGUF converter
- `src/hmer_ocr.h` — C API header
- `src/hmer_ocr.cpp` — DenseNet encoder + GRU attention decoder inference
- `tests/test_hmer_parity.py` — end-to-end parity test vs PyTorch

Modified files:
- `CMakeLists.txt` — add hmer_ocr.cpp to build
- `src/crispembed.cpp` — expose hmer_ocr via unified C API (optional)

### Key differences from existing math_ocr (DeiT+TrOCR)

| | math_ocr (printed) | hmer_ocr (handwritten) |
|---|---|---|
| Encoder | DeiT ViT (12 transformer layers) | DenseNet-121 (42 conv layers, 3 dense blocks) |
| Decoder | TrOCR (6 transformer layers, cross-attn) | 2× GRU cells + Bahdanau attention + coverage |
| Input | Fixed 384×384, 1ch grayscale | Variable size, 2ch (grayscale + mask) |
| Vocab | 1200 BPE tokens | 112 LaTeX tokens |
| Params | 29.3M | 7.9M |
| Q4_K | 17 MB | ~4-5 MB |
| KV cache | Transformer KV cache (per-layer K,V) | GRU hidden state + attention/coverage maps |
| Activation | GELU (encoder), ReLU (decoder) | ReLU (encoder), tanh/sigmoid (decoder GRU) |
| Norm | LayerNorm throughout | BatchNorm (encoder), BatchNorm (decoder attention) |
