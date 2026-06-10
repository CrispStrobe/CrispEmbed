# CrispEmbed — History

Completed milestones and work log. See PLAN.md for current roadmap.

---

## June 2026 — RT-DETRv2 Layout Detection

### Document layout analysis: ResNet-50 + HybridEncoder + deformable decoder
- Architecture: ResNet-50-D backbone + HybridEncoder (AIFI self-attention +
  FPN/PAN with CSP-RepVGG blocks) + 6-layer transformer decoder with
  deformable multi-scale cross-attention (300 queries, 17 classes)
- 14 parity bugs found and fixed via systematic layer-by-layer diff:
  AIFI pos/LN/residual, PAN lateral features, cpu_linear weight convention,
  converter weight transposition (Gemm/Split/Transpose patterns),
  decoder_input_proj Conv convention, valid_mask, query_pos_head architecture,
  bilinear resize, grid_sample alignment
- All encoder stages cos=1.0 with exact input (verified via crispembed-diff)
- Detection score 0.934 on test images (HF reference: 0.955)
- Fully wired: C ABI, CLI (`--layout`), server (`POST /layout/detect`),
  Python (`CrispLayout`), Rust (`CrispLayout`), Dart/Flutter
- Source: docling-project/docling-layout-heron (Apache-2.0, 42M params)

---

## June 2026 — WASM build (math OCR in browser)

### CrispEmbed compiled to WebAssembly via Emscripten
- `build-wasm.sh`: emcmake cmake, CPU-only, SIMD128, MODULARIZE=1
- Output: `crispembed_ocr.js` (62K) + `crispembed_ocr.wasm` (999K)
- `wasm/ocr_wrapper.c`: thin C entry point exposing `wasm_ocr_init`,
  `wasm_ocr_recognize_gray`, `wasm_ocr_recognize`, `wasm_ocr_free`
- Emscripten guards: `model_mgr.cpp` (disable curl/wget),
  `gguf_loader.cpp` (skip mmap, use fread fallback)
- `cmake/FindThreads.cmake`: stub override creates no-op Threads::Threads
  target, avoiding -pthread and SharedArrayBuffer/COOP/COEP requirement
- Integrated into CrispCalc web/PWA: `dart:js_interop` bridge, IndexedDB
  model caching, conditional import selects WASM provider on web
- All existing OCR models work: pix2tex, HMER, BTTR, PosFormer, Texo,
  PP-FormulaNet-L (auto-detected from GGUF architecture tag)
- Tested end-to-end: model load (16.8 MB, 1.5s) + encoder (578 tokens)
  + decoder (201 tokens) → LaTeX output in Node.js

### HuggingFace Space
- `hf-space/`: Docker build (two-stage) + Gradio UI (3 tabs: text
  embeddings, math OCR, health)
- Pattern: C++ `crispembed-server` on :8090 + Gradio on :7860
- Default models: all-MiniLM-L6-v2 (text) + hmer-hw (OCR)
- Tested: cos=0.785 for similar texts, `x² + 1 = 0` → `x ^ { 2 } + 1 = 0`
- Live at https://huggingface.co/spaces/cstr/CrispEmbed

### CI
- `build-wasm.yml`: builds WASM on push/PR, uploads artifacts
- `deploy-hf-space.yml`: auto-deploys `hf-space/` to HuggingFace on push

---

## June 2026 — PP-FormulaNet-L OCR (181M params)

### Full in-graph ViT encoder with decomposed RPE
- **Full ggml graph encoder**: all 12 ViT layers run as single ggml graphs
  with attention + decomposed relative position bias entirely in-graph
- Window batching: all 16 windows × 12 heads processed as one batch dimension
  via reshape + permute, enabling efficient batched matmuls
- Decomposed RPE in-graph: two matmuls (rp_h@Q, rp_w@Q_permuted) with
  broadcast-add to attention scores (Granite NLE pattern)
- LN ordering fix: for windowed layers, LayerNorm1 applied on CPU before
  window partition to match HF's LN→pad→QKV ordering. Prevents LN(0)=bias
  corruption of padding tokens (cos jumped from 0.973 to 0.9999)
- Per-layer parity: cos ≥ 0.99997 on all 12 layers
- Performance: 60s encoder (was 77s hybrid, ~500s scalar)

### Printed math OCR: SAM-ViT encoder + MBart decoder
- New architecture: SAM-style ViT encoder (12 layers, 768d, 12 heads)
  with windowed attention (ws=14) + global attention (layers 2,5,8,11)
  and decomposed relative position bias
- Neck: Conv1x1 + LayerNorm2d + Conv3x3 + LayerNorm2d (768 → 256)
- Multi-modal projector: 2× Conv3x3(stride=2) + 2× Linear (256 → 512)
  Output: (144, 512) sequence for decoder
- MBart PRE-LN decoder: 8 layers, 16 heads, d_model=512, FFN=2048
- 768x768 RGB input, UniMERNet preprocessing pipeline
- Encoder parity: cos=0.999962 vs HuggingFace reference (F32)
- Quantization: F32 (692 MB), F16 (347 MB), Q8_0 (241 MB, cos=0.999940),
  Q4_K (122 MB, cos=0.997595) — all produce identical decoded LaTeX
- Smart Q8_0: critical tensors (embeddings, LN, rel_pos, lm_head) in F16
- Auto-detected from GGUF metadata (`general.architecture = ppformulanet_l`)
- Wired into unified `--ocr` CLI, C ABI, model registry, CrispCalc Dart catalog
- Source: PaddlePaddle/PP-FormulaNet-L_safetensors (Apache-2.0)
- New GGUF loader helper: `kv_i32_array()` for int32 metadata arrays

### Full-stack wiring
- HTTP server: `POST /math/ocr` endpoint (`--ocr` flag, stb_image load, JSON response)
- Python bindings: `CrispMathOcr` class with `recognize()` and `recognize_gray()`
- Updated contributing.md with server + Python binding steps
- Updated public C header comments to list all supported architectures

## June 2026 — PPFormulaNet-S / Texo-Distill OCR

### Printed math OCR: HGNetv2 + MBart decoder (20M params)
- New architecture: HGNetv2 CNN encoder (StemBlock, 4 HG_Stages, LightConvBNAct)
  + MBart Transformer decoder (2 layers, 16 heads, 384 d_model)
- Conv-BN folding in GGUF converter: all BatchNorm absorbed into preceding Conv2d
- CPU-side CNN forward pass for encoder (all standard ops: conv2d, relu, maxpool, concat)
- MBart PRE-LN decoder: LayerNorm before attention/FFN, residual skips LN
- UniMERNet preprocessing: aspect-ratio-preserving resize + black pad + grayscale
  normalize (mean=0.7931, std=0.1738)
- ODR fix: renamed internal dec_layer → ppfn_dec_layer to avoid linker collision
  with decoder_embed_internal.h
- Added `--ocr` CLI flag for unified auto-detection (pix2tex/hmer/bttr/ppformulanet)
- Quantized: F16 (39 MB), Q8_0 (22 MB, identical quality), Q4_K (13 MB, degraded)
- GGUF models published: huggingface.co/cstr/texo-distill-gguf
- Diff regime: encoder cos=1.000000, decoder verified via layer-by-layer debug traces
- Source: Texo (AGPL-3.0) distilled from PaddleOCR PP-FormulaNet-S (Apache-2.0)
  trained on UniMER-1M (CC-BY-4.0)

## June 2026 — Nomic v2 MoE Encoder

### Mixture-of-Experts encoder support
- First MoE embedding model: nomic-embed-text-v2-moe (8 experts, top-2, GELU)
- Fully in-graph MoE routing: ggml_top_k + ggml_get_rows + ggml_mul_mat_id
- Mixed architecture: odd layers = MoE FFN, even layers = dense GELU FFN
- Converter handles GPT2-style config (NomicBERT extends GPT2Config),
  per-layer MoE/dense auto-detection, expert weight stacking [n_exp, dim, dim]
- Fixed missing Wqkv + out_proj biases (present in v2-moe but not v1.5)
- Exact erf-GELU activation (NomicBERT uses nn.GELU(approximate='none'))
- Parity: cos=1.000000 vs HuggingFace on all test texts
- Quantized variants: F16 (1344 MB), Q8_0 (1122 MB, cos=0.9996), Q4_K (1095 MB, cos=0.966)
- GGUFs published to cstr/nomic-embed-text-v2-moe-GGUF on HuggingFace
- Extended parity_layers_bert.py harness with --arch nomic (QKV split, MoE expert tensor diff)
- Added CRISPEMBED_DUMP_LAYERS env var for per-layer intermediate tensor dumps

---

## June 2026 — LoRA Hot-Swap, Batched Decoder, Face Pipeline

### LoRA adapter hot-swap
- Runtime switching between Jina v5 per-task LoRA adapters (retrieval,
  classification, clustering, text-matching) without re-loading the model
- Pre-compute approach: `W' = W + (α/r)·B@A` on CPU at switch time (~10-50ms)
- Converter `--lora-mode=separate` stores base weights + per-adapter A/B
  tensors (F16) in a single GGUF with metadata
- Lazy base weight snapshot with dequant→merge→requant for quantized models
- C API: `crispembed_set_lora/get_lora/list_lora`
- CLI: `--lora NAME`, `--list-lora`
- Python: `set_lora()`, `lora` property, `list_lora()` on CrispEmbed

### Batched decoder graph
- Single ggml graph compute for N decoder texts (was: N sequential passes)
- Block-diagonal causal mask (text i cannot attend to text j), padding
  positions get self-attention to prevent softmax NaN
- Independent RoPE positions per text, pad to T_max
- Per-text last-token / mean pooling after graph compute
- **3.3x speedup** on batch of 4 (Jina v5 nano, CPU)
- Parity: cos >= 0.999 vs sequential encoding on all test texts

### Face pipeline Python completion
- `CrispFacePipeline` exported in `__init__.py`
- `from_registry()` class methods on `CrispFace` and `CrispFacePipeline`
  for auto-download by registry name
- Unit tests (`tests/test_face_python.py`): 12 tests covering detection,
  recognition, pipeline, match, edge cases
- Example script (`examples/face_search.py`): index faces from directory,
  query by image, top-K cosine matches

### BTTR beam search decoder
- Beam search with configurable width (default 5) for BTTR handwritten
  math OCR — improves exact-match accuracy over greedy decoding

### Windows CI fix
- `M_PI` undefined on MSVC: added `#ifndef M_PI` portable fallback in
  `bttr_ocr.cpp`

---

## June 2026 — CLIP/SigLIP Vision + Text, YuNet, HMER/BTTR OCR

### YuNet lightweight face detection
- 228 KB GGUF (vs SCRFD 16 MB), ShuffleNetV2 backbone, 640×640 input
- IoU 0.99 vs OpenCV FaceDetectorYN, score diff < 0.01, landmark diff < 2px
- Converter unchanged (existing `convert-face-to-gguf.py` handles YuNet's ops)
- Key gotcha: ggml Transpose op does real 2D transpose for n_dims==2 tensors,
  requiring different spatial indexing for 1-channel (cls/obj) vs multi-channel
  (bbox/kps) outputs
- Uploaded to `cstr/yunet-GGUF`, in auto-download registry

### CLIP text encoder (new module)
- `clip_text_embed.{h,cpp}`: pre-LN transformer with causal mask, EOS pooling,
  text_projection, BPE tokenizer embedded in GGUF
- `convert-clip-text-to-gguf.py`: extracts text tower + tokenizer from HF CLIP
- C API (`crispembed_clip_text_*`), Python wrapper (`CrispClipText`), server
  `/clip/text` endpoint
- Cross-modal text↔image search works end-to-end
- Uploaded: `cstr/clip-text-base-GGUF` (244 MB), `cstr/clip-text-large-GGUF` (474 MB)

### CLIP / SigLIP vision models
- Fixed `vit_embed.cpp`: CLS token prepend, CLS pooling for CLIP, quick_gelu
  via FP32 ggml primitives, attention pooling residual skip connection
- Converted and uploaded 6 vision GGUFs:
  - `cstr/clip-vit-base-patch16-GGUF` (329 MB, MIT)
  - `cstr/clip-vit-large-patch14-GGUF` (1.2 GB)
  - `cstr/clip-vit-large-patch14-336-GGUF` (1.2 GB)
  - `cstr/siglip-large-256-GGUF` (1.2 GB, Apache 2.0)
  - `cstr/siglip-so400m-patch14-384-GGUF` (1.6 GB)

### Handwritten math OCR (HMER + BTTR)
- HMER: DenseNet-121 encoder + GRU attention decoder (with coverage).
  Source: whywhs/Pytorch-HMER (MIT), trained on CROHME 2016.
  112 LaTeX tokens, ~6.8M params, ~4-5 MB Q4_K.
  `hmer_ocr.{h,cpp}`, `convert-hmer-to-gguf.py`. CLI: `--hmer FILE`.
  Auto-detect image polarity and invert if needed. Dequant support.

- BTTR: DenseNet encoder (growth=24, 3 blocks) + Transformer decoder
  (3 layers, 8 heads, d=256). Source: Green-Wood/BTTR (MIT),
  trained on CROHME 2014. 113 LaTeX tokens, 6.5M params.
  52.8% exact match on 36 CROHME test images (vs 38.9% for HMER).
  `bttr_ocr.{h,cpp}`, `convert-bttr-to-gguf.py`.
  BN folded into conv, fused QKV preserved.

### SFace quantization (conv2d quant support)
- Converter flattens 4D conv weights to 2D [OC, IC*KH*KW] for quantization
- Runtime: dequant Q8/Q4→F32, reshape back to 4D, cast to F16 for ggml_conv_2d
- SFace results: F32=37MB, Q8_0=10MB (cos=0.9999), Q4_K=6MB (cos=0.974)
- Uploaded to `cstr/sface-GGUF` (F32 + Q8_0 + Q4_K)
- Same pattern applies to AuraFace and SCRFD (reconverted with flat conv)
- AuraFace: 249 MB (Q8_0 only compresses FC → 212 MB; conv rows too small for Q8_0)
- SCRFD: 17 MB (minimal Q8_0 gain — detection heads are small)
- AuraFace F16: 249→125 MB (2.0x, lossless — conv2d casts to F16 anyway)
- SCRFD F16: 17→8 MB (2.0x, lossless)
- Added F16 support to quantizer (Q8_0/Q4_K need row÷32; F16 has no alignment limit)

### Face model quantized graph replay fixed
- YuNet F16/Q8_0 inference via graph replayer now works (was crashing)
- Three fixes: (1) parse Conv group attrs before 2D→4D reshape for
  correct depthwise IC detection, (2) handle ggml_n_dims returning 2
  for 4D weights with trailing 1s via element count validation,
  (3) dequant Q→F32 before F16 cast (ggml only supports Q→F32)
- Q8_0 detection matches F32 with minor quantization drift (conf 0.731 vs 0.749)
- Old-style 4D-weight GGUFs and new-style 2D-flattened GGUFs both work
- YuNet parity verified: sub-pixel match vs OpenCV FaceDetectorYN on both
  single-face and multi-face images (x/y/w/h diff < 0.5px, conf diff < 0.01)
- Raw tensor cos vs ONNX (0.35-0.85) is a false alarm — planar (ggml) vs
  interleaved (ONNX) layout of the same correct data; decoded coords match

### SigLIP text encoder verified
- cos=1.000000 vs HuggingFace on all test texts
- SentencePiece BPE vocab decoded correctly by Viterbi unigram algorithm
- Key finding: SP BPE training doesn't change inference algorithm — Viterbi works

### Model registry expansion
- 13 new auto-download entries: 2 face detection (yunet, scrfd-det-10g),
  2 face recognition (auraface-v1, sface), 8 vision/text (CLIP + SigLIP),
  1 SigLIP-base
- Total registry: ~58 models

### Vision parity fixed: cos 0.8 → 1.0
- Root cause: patch embedding `ggml_permute(2,1,0)` produced column-major
  spatial ordering (t = ow*OH + oh), but HuggingFace uses row-major
  (t = oh*OW + ow via flatten(2)). Every patch beyond (0,0) got the
  wrong position embedding.
- Fix: `ggml_permute(1,2,0,3)` produces [D, OW, OH] which flattens to
  row-major matching HF. Per-layer cos goes from ~0.3 to 1.000000.
- Final embedding cos = 0.9998 vs HuggingFace (SigLIP-base-384)
- CLIP ViT also verified: cos=1.000000, max_diff=0.000001 (clip-vit-base-patch16)
- Was NOT "FP32 non-associativity" as previously hypothesized — it was
  a simple permutation index bug that scrambled patch positions

---

## v0.7.0 — May 2026

### Registry status

45 models in registry, 151 GGUF variants published on HF:
25 encoder models + 11 decoder models + 12 rerankers + 1 SPLADE + 2 multimodal.
Typical per-model: F32 + Q8_0 + Q4_K; about a dozen also have Q5_K / Q6_K / F16.

Key parity results (cos vs HuggingFace reference):

| Model | Type | Dim | CosSim |
|-------|------|-----|--------|
| all-MiniLM-L6-v2 | BERT | 384 | 1.000000 |
| bge-small/base/large-en-v1.5 | BERT | 384/768/1024 | 1.000000 |
| gte-base/large-en-v1.5 | GTE | 768/1024 | 1.000000 |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 1.000000 |
| nomic-embed-text-v2-moe | NomicBERT MoE | 768 | 1.000000 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000000 |
| all-mpnet-base-v2 | MPNet | 768 | 1.000000 |
| multilingual-e5-small/base/large | XLM-R | 384/768/1024 | 1.000000 |
| snowflake-arctic-embed-m/l | BERT/XLM-R | 768/1024 | 1.000000 |
| bge-m3 (dense+sparse+ColBERT) | XLM-R | 1024 | 1.000000 |
| splade-pp-en-v1 | BERT SPLADE | 768 | 1.000000 |
| granite-embedding-278m/107m | XLM-R | 768/384 | 1.000000 |
| gte-modernbert-base | ModernBERT | 768 | 0.9999 |
| pixie-rune-v1 | XLM-R | 1024 | 0.999993 |
| octen-0.6b | Qwen3 | 1024 | 0.999891 |
| octen-8b | Qwen3 | 4096 | 0.965 (Q4_K vs bf16 HF) |
| qwen3-embed-4b | Qwen3 | 2560 | 0.974 (Q4_K vs bf16 HF) |
| harrier-0.6b / harrier-270m | Qwen3/Gemma3 | 1024/640 | 0.999959/948 |
| jina-v5-nano/small | Qwen3 | 1024 | 0.999941 |
| bge-reranker-v2-m3 | XLM-R reranker | - | verified |
| ms-marco-MiniLM-L-6/12-v2 | BERT reranker | - | verified |

### Optimizations completed

- ggml_backend_sched GPU dispatch (encoder + decoder full-graph)
- All 45 models quantized (Q8_0 + Q4_K) and uploaded to HuggingFace
- Graph/work buffer reuse: 27.8 texts/s server throughput (gte-small)
- Matryoshka dimension truncation via -d N flag
- BLAS/MKL/CUDA/Vulkan/Metal build support
- Windows build scripts
- C++ quantizer with K-quant fallback chain
- QKV weight fusion (1 matmul vs 3 per layer)
- Flash attention with optional position bias mask
- ggml graph decoder for math OCR (27x speedup over scalar)

### Bindings and platforms

| Binding | CrispEmbed | CrispASR |
|---|---|---|
| C API | Complete | Complete (whisper.h) |
| Python (ctypes) | Complete + tested | Complete + tested |
| Rust (crate) | Complete + tested | Complete + compiled |
| Dart/Flutter (FFI) | Complete | Created |
| iOS (Metal) | CI green | CI green |
| Android (NDK) | CI green (arm64/armv7/x86_64) | CI green |
| Windows | CI green | CI green |
| macOS (Metal) | CI green | CI green |
| Linux | CI green | CI green |

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

### Commercially permissive stack (no NC restrictions)

The full pipeline uses only Apache 2.0 / MIT models:
- Text: any CrispEmbed encoder model (BERT/XLM-R/etc.)
- Image: SigLIP (Apache 2.0) or CLIP (MIT)
- Face detection: SCRFD (Apache 2.0) or YuNet (Apache 2.0)
- Face recognition: AuraFace-v1 512-D (Apache 2.0) or SFace 128-D (Apache 2.0)
- Face landmarks: MediaPipe FaceLandmarker (Apache 2.0)
- Audio: CrispASR (our own, Apache 2.0)

### Resolved known issues

1. **NomicBERT** — Root cause: gate/up weights (fc11/fc12) were swapped in old GGUF;
   also needed Ollama tensor name fallback. F32 cos=1.0, Q8_0 cos=0.998.

2. **EmbeddingGemma-300m** — cos=1.0000 F32, 0.9998 Q8_0, 0.9954 Q5_K.
   Root causes: missing `is_bidirectional=1`, wrong pooling, BPE merges not loading,
   Dense layers being quantized. All fixed.

3. **Jina v5 nano/small** — Models use task-specific LoRA adapters; converter now
   merges `retrieval` adapter. Nano F32 cos=1.0, Small F32 cos=0.9999.

4. **all-mpnet-base-v2** — Old GGUF was missing `relative_attention_bias.weight`.
   Reconverted with bias tensor. cos=0.987-0.999.

5. **gte-modernbert-base** — Validation wrongly required `ln1` for pre-LN models.
   Fixed validation. cos=0.9999.

6. **DeBERTa-v2 disentangled attention** — c2p/p2c relative position bias with
   log-bucket encoding now fully implemented. mxbai-rerank-xsmall-v1 and
   mxbai-rerank-base-v1 both working.

7. **Full regression sweep (2026-05-17)**: 34 models tested, all pass. 5 models
   fixed and re-uploaded to HF.

---

## May 2026 — Multimodal & Vision

### BidirLM-Omni (text + audio + image)

- [x] Text path through `decoder_embed.cpp` (cos >= 0.999 vs HF bf16)
- [x] Audio path through `bidirlm_audio.cpp` + crisp_audio (cos = 0.995 vs HF)
- [x] Vision tower in `bidirlm_vision.cpp` (cos >= 0.999 vs HF bf16)
- [x] DeepStack injection + 3D interleaved-MRoPE (cos = 0.998903 vs HF bf16)
- [x] `crispembed_encode_text_with_image` C ABI + Python wrapper
- [x] `crispembed_encode_with_image_ids` (pre-tokenized variant for parity tests)
- [x] CLI `--image FILE` + `--image-raw patches.f32 --grid-thw T,H,W`
- [x] Decoder `ggml_backend_sched` initialization
- [x] Memory-efficient lite parity test
- [x] In-process C++ image preprocessor (smart_resize + Catmull-Rom bicubic)
- [x] BPE special-token handling for Qwen-style tokens
- [x] Stale-GGUF fallbacks for missing metadata
- [x] Image batching in `encode_text_with_image`

### Phase 8: Vision — Image Embeddings, Face Detection & Recognition

#### 8A. SigLIP Image Embedding (DONE)

cos=0.996 vs HF. Uploaded to cstr/siglip-base-GGUF.
- GGUF converter, ViT forward path, image preprocessing
- CLI: `crispembed -m siglip-base.gguf --image photo.jpg`

#### 8B. Face Detection — SCRFD (DONE)

Scores match ONNX Runtime. Uploaded to cstr/scrfd-det-10g-GGUF.
- Generic ONNX graph replayer (Conv, ReLU, Add, Pool, Resize, Concat, Sigmoid)
- FPN + multi-scale detection heads + NMS
- Letterbox preprocessing + coordinate scaling
- C API, Python, Rust, Dart wrappers

#### 8C. Face Recognition — AuraFace + SFace (DONE)

cos=0.9999 vs ONNX for both models.
- BN folding/precomputation, 512-D/128-D embeddings
- Full detect-align-encode pipeline
- C API, Python, Rust, Dart wrappers
- Server API: `/detect`, `/face` endpoints

---

## April 2026 — RAG Feature Parity

- [x] Full Python/Rust/Dart wrapper: sparse, ColBERT, reranker, set_dim, set_prefix
- [x] Bi-encoder reranking API (Python + Rust + Dart): cosine similarity ranking
- [x] Prompt prefix system (C/Rust/Python/Dart): auto-prepend query/passage prefixes
- [x] 21 verified embedding models (cos >= 0.999 vs HuggingFace)
- [x] 5 reranker models (bge-reranker-base, ms-marco L6/L12, mxbai-rerank xsmall/base)
- [x] 27 HuggingFace repos with GGUF models + README cards
- [x] RAG retrieval quality benchmark (tests/bench_rag.py): MRR@10, NDCG@10, Recall@k
- [x] Reranking benchmark (tests/bench_rerank.py): cross-encoder vs bi-encoder
- [x] Head-to-head benchmark vs FastEmbed:
  - MiniLM-L6: CrispEmbed **9.5x faster** single, **10.8x faster** batch
  - BGE-small: FastEmbed 1.7x faster (ONNX graph JIT optimization)
  - Arctic-M: tied on batch (126 vs 127ms)
  - cos = 0.999999-1.000000 cross-engine on all models
- [x] Demo apps (Python + Rust) for both CrispEmbed and CrispASR

---

## May 12, 2026 — Face Pipeline Complete

Full detect -> align -> encode pipeline for face recognition.

### RAG parity: prompt prefixes + new models

- Added auto-prefix system: BGE, E5, Nomic, Jina models get query/passage
  prefixes auto-applied.
- Converted 3 new models: SPLADE-PP-en-v1, granite-embedding-278m/107m.
- Model registry: 47 models total.

### SCRFD preprocessing + anchor decode fixes (3 bugs)

1. RGB-BGR channel swap
2. Anchor center offset (integer grid, no 0.5 offset)
3. Top-left placement (not centered letterbox)

### SCRFD anchor decode fix (data layout mismatch)

Channel-last vs interleaved indexing. After fix: detection counts match
InsightFace exactly on all test images.

### Face alignment fix (4 sign errors in normal equations)

After fix: alignment matches InsightFace `norm_crop` with MAE=0.00.
Per-face embedding cos=0.994-0.999 vs InsightFace ArcFace.

### Pipeline implementation

- `cnn_embed::detect_file()` — letterbox resize, coordinate scaling
- `cnn_embed::encode_aligned()` — 5-point landmark similarity transform + encode
- `cnn_embed::face_pipeline()` — detect -> align -> encode in one call
- CLI, C API, Server API, Python/Rust/Dart wrappers all complete

### Models converted

- SCRFD-10GF (16.1 MB)
- w600k_r50 ArcFace (166 MB)
- AuraFace-v1 (248.6 MB)
- SFace (36.8 MB)

---

## May 11-12, 2026 — Vision Models & Parity Fixes

### SigLIP image embedding
- Converter: `models/convert-siglip-to-gguf.py`
- Forward path: `src/vit_embed.cpp` — cos=0.996 vs HF mean-pool
- Native `--image` flag with stb_image preprocessing
- Uploaded: cstr/siglip-base-GGUF

### Face detection (SCRFD)
- Generic ONNX graph replayer in `src/cnn_embed.cpp`
- FPN backbone + multi-scale detection heads
- Anchor decode + NMS at strides 8/16/32
- Semicolon delimiter for ONNX tensor names with commas

### Face recognition (SFace + AuraFace)
- SFace MobileFaceNet: cos=0.9999 vs ONNX, 128-D
- AuraFace ResNet-100: cos=0.9999 vs ONNX, 512-D
- BN folding/precomputation at converter time
- PReLU: relu(x) + slope * (x - relu(x))
- Conv F32->F16 auto-cast for ggml_conv_2d

### Text model parity fixes (35 models)
- GTE v1.5: post-LN + GeGLU half swap + NTK RoPE
- Jina reranker v2: post-LN + position offset
- NomicBERT: SwiGLU fc11/fc12 swap
- Ollama format: auto-strip prefix, dual metadata keys, pooling type mapping

---

## Apr 12, 2026 — v0.1.0 Release

30-commit session: FastConformer extraction, granite 3.x support,
NeMo FC-CTC, omniASR, Silero LID, CI, Windows, Vulkan, benchmarks.
Tagged v0.1.0 release with multi-platform binaries.
