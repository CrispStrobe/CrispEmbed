# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.7.0)

```
Input text / image / audio
    │
    ├─► Text ──► Tokenizer (WordPiece / SentencePiece / BPE)
    │              │
    │              ├─► Encoder path (BERT, XLM-R, MPNet, NomicBERT,
    │              │     ModernBERT, GTE v1.5, DeBERTa-v2, SPLADE)
    │              │     Token + Pos [+ Type] embeddings
    │              │     N × Transformer layer (LN → MHA → FFN → residual)
    │              │     Pooling (mean / CLS) + optional heads
    │              │     → dense / sparse / ColBERT / reranker output
    │              │
    │              └─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │                    Token embeddings + RoPE
    │                    N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │                    Last-token / mean pooling + L2 normalize
    │
    ├─► Image ──► ViT path (SigLIP/CLIP: vit_embed.cpp)
    │               Conv2D patch embed → transformer → mean pool → L2
    │
    ├─► Image ──► BidirLM-Omni vision (bidirlm_vision.cpp)
    │               Qwen2VL ViT + patch merger + DeepStack
    │               → image_embeds spliced into decoder
    │
    ├─► Image ──► CNN path (cnn_embed.cpp)
    │               SCRFD/YuNet face detection (FPN + anchor decode + NMS)
    │               ArcFace/SFace/AuraFace face recognition
    │
    ├─► Audio ──► BidirLM-Omni audio (bidirlm_audio.cpp)
    │               crisp_audio Whisper-shape encoder → mean pool → 2048-d
    │
    ├─► Math  ──► DeiT encoder + TrOCR decoder (math_ocr.cpp)
    │               Printed math → LaTeX via ggml graph compute
    │
    ├─► Math  ──► HMER: DenseNet-121 + GRU attention (hmer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2016)
    │
    ├─► Math  ──► BTTR: DenseNet + Transformer decoder (bttr_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2014, 49% exact match)
    │
    └─► Math  ──► PosFormer: DenseNet + Transformer + ARM (posformer_ocr.cpp)
                    Handwritten math → LaTeX (CROHME 2014, 56% raw / 61.4% parsed)
                    NOTE: SJTU weights = academic-only; training NC weights on Kaggle
```

### Math OCR roadmap — next models to evaluate

| Model | Year | CROHME'14 | Params | Code | Verdict |
|-------|------|-----------|--------|------|---------|
| Uni-MuMER | 2025 | ~82% | 3B VLM | Yes | Too large for mobile |
| NAMER | 2024 | 60.5% | ~20M | No | Best arch (non-AR), no code |
| WaveMER | 2026 | 60.6% | ~20M | Yes | Weak accuracy, complex |
| SCAN | 2020 | ~55% | ~15M | No | Online-only (stroke input) |
| CAN | 2022 | ~57% | ~10M | Yes (MIT) | Compact, good license |
| CoMER | 2022 | ~59% | ~10M | Yes | Coverage-enhanced transformer |

**Status**: PosFormer (our port) outperforms all practical candidates in greedy
l2r mode: 56.0% raw / 61.4% parsed on CROHME 2014.

**Training pipeline**: `tools/kaggle/posformer-train/posformer_train.py`
- Kaggle kernel: https://www.kaggle.com/code/chr1str/posformer-train-on-mathwriting
- W&B dashboard: https://wandb.ai/cze-github/posformer-hmer
- CROHME data: https://huggingface.co/datasets/cstr/posformer-training-data
- Supports CROHME (8.8K) and MathWriting (230K) datasets
- GPU training on P100 (cu118) or T4, hourly HF checkpoints, W&B monitoring
- Uses CrispASR kaggle_harness for auth + progress (clone at runtime)

**License path**: All handwritten math datasets are NC. Current SJTU weights are
academic-only (stricter). Retraining on CROHME/MathWriting produces NC weights
(CC BY-NC-SA), which are fine for the app's "download with terms acceptance" model.
Do NOT wire into CrispCalc until NC (or better) weights are available.

**For non-NC weights** (fully commercial): synthetic handwriting generation
(render LaTeX with handwriting fonts + augmentation) or train on Im2LaTeX-100K
(CC0, printed math) + fine-tune on self-created handwritten samples.

## Supported architectures (v0.7.0)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| ModernBERT encoder | BPE | Pre-LN, GeGLU, RoPE, per-layer theta | gte-modernbert-base |
| GTE v1.5 encoder | WordPiece | Post-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | WordPiece | Post-LN, c2p/p2c disentangled attn | mxbai-rerank-xsmall/base-v1 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M, EmbeddingGemma-300m |
| BidirLM-Omni | GPT-2 BPE | Bidirectional Qwen3, MRoPE, DeepStack | BidirLM-Omni-2.5B |
| ViT (SigLIP/CLIP) | — | Conv2D patch embed, CLS/mean/attn pool | siglip-base, clip-vit-base |
| CLIP text | CLIP BPE | Pre-LN, causal mask, EOS pool | clip-text-base/large |
| CNN (SCRFD/YuNet) | — | FPN, anchor decode, NMS | scrfd-det-10g, yunet |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |

## File layout (current)

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md                     architecture + roadmap (this file)
├── HISTORY.md                  completed milestones
├── LEARNINGS.md                technical notes
├── PERFORMANCE.md              benchmarks
├── ggml/                       (submodule)
├── src/
│   ├── crispembed.h            C API
│   ├── crispembed.cpp          encoder graph + C API impl
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr.{h,cpp}        HMER handwritten math OCR
│   ├── bttr_ocr.{h,cpp}        BTTR handwritten math OCR
│   ├── tokenizer.h             WordPiece + SentencePiece + BPE
│   ├── tokenizer_bpe.cpp       GPT-2 byte-level BPE
│   ├── model_mgr.{h,cpp}       registry + auto-download
│   └── core/                   shared helpers (from CrispASR)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/
│   ├── convert-bert-to-gguf.py
│   ├── convert-decoder-embed-to-gguf.py
│   ├── convert-siglip-to-gguf.py
│   ├── convert-clip-text-to-gguf.py
│   ├── convert-face-to-gguf.py
│   ├── convert-hmer-to-gguf.py
│   ├── convert-bttr-to-gguf.py
│   └── upload_to_hf.py
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending roadmap

### Performance

- [x] True batched graph for decoder models (single compute for N texts, block-diagonal causal mask, ~3x speedup)
- [ ] KV cache for prefix-shared decoder batches
- [x] SigLIP attention pooling head (mean pool works; attn pool for full parity)

### Models

- [x] CLIP text encoder (causal mask variant)
- [x] SigLIP-large, CLIP-large conversion + upload
- [x] SigLIP / ViT quantization (conv2d needs F32 kernel — selective quant)
- [x] YuNet lightweight face detection alternative
- [x] SFace INT8 quantization (Q8_0 cos=0.9999, Q4_K cos=0.974; 37→10→6 MB)
- [x] Face model quantized inference via graph replayer (YuNet F16/Q8_0 working; fixed depthwise IC, ggml_n_dims trailing-1s, Q→F32 dequant path)
- [x] ViT parity: cos 0.8→1.0 (was patch ordering bug — permute(2,1,0) gave column-major spatial, fixed to permute(1,2,0,3) for row-major matching HF)
- [ ] Nomic v2 MoE (MoE routing layer in encoder)
- [x] LoRA adapter hot-swap (Jina v5 per-task adapters, pre-compute merge on CPU, ~10-50ms switch)

### Bindings

- [x] Python wrapper `encode_image()` for standalone SigLIP/CLIP
- [x] CrispFacePipeline export + from_registry() + Python unit tests + face_search example
- [ ] CrispLens integration — update `crispembed_client.py` for face pipeline

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| Nomic v2 MoE | Low | High | MoE routing layer in encoder |
| Qwen3-VL multimodal | Low | High | Reuse BidirLM-Omni scaffolding |

### Ideas (unscoped)

- [ ] **Streaming ColBERT late interaction scoring** — Server-side MaxSim
  scoring between a query's ColBERT multi-vector and pre-stored document
  token vectors. Stream partial scores via SSE so the client can show
  progressive ranking.  Needs: `/colbert/score` endpoint accepting query
  multi-vec + list of doc multi-vecs, chunked response with cumulative
  top-K.  Builds on the existing `/embed` endpoint and
  `crispembed_encode_multivec()` C API.

- [ ] **WASM build target** — Compile CrispEmbed to WebAssembly
  (Emscripten) for browser-based embedding inference.  Requires: ggml
  WASM backend (CPU-only, no GPU), JS wrapper exporting `encode()` /
  `encode_batch()`, a demo page.  ggml already has partial Emscripten
  support (whisper.cpp ships a WASM build).  Main challenges: SIMD
  (relaxed-simd flag), memory limits (large models need streaming GGUF
  loading or smaller quants), and thread support (SharedArrayBuffer +
  Web Workers for multi-threaded ggml).

- [ ] **INT4 GGUF for face models** — Apply Q4_K quantization to
  Conv2D weights in SCRFD / AuraFace / SFace.  Currently conv weights
  are stored F32 or F16 because `ggml_conv_2d` only supports
  F32/F16 kernels; quantized conv would require dequant→F32 at graph
  build time (same pattern as HMER/BTTR).  Expected size savings:
  AuraFace 249 MB → ~65 MB, SCRFD 17 MB → ~5 MB.  Quality gate:
  cos ≥ 0.99 vs F32 for recognition, IoU ≥ 0.95 for detection.

---

## Implementation blueprints

Detailed specs for pending roadmap items. Each blueprint is self-contained
so a fresh agent can implement it independently.

### Blueprint: LoRA adapter hot-swap

**Goal**: Load multiple LoRA adapters from a single GGUF and switch at
runtime without re-loading the model. Primary use case: Jina v5 per-task
LoRA (retrieval, classification, clustering, text-matching).

**Current state**: LoRA is baked at convert time. The converter
(`models/convert-decoder-embed-to-gguf.py` lines 142-156) calls
`model.set_adapter("retrieval")` then `model.merge_and_unload()`, producing
a single merged weight set. Switching tasks requires re-converting.

**LoRA math**: `y = Wx + (a/r) * B(Ax)` where W is the base weight
`[out, in]`, A is `[r, in]`, B is `[out, r]`, a is scaling, r is rank
(typically 8-16 for Jina v5).

**Step 1 -- Converter** (`models/convert-decoder-embed-to-gguf.py`):
- Add `--lora-mode=separate` flag. Instead of merging, store base weights
  without LoRA and separately store per-adapter tensors:
  `lora.{adapter}.{layer}.{matrix}.A` `[r, in]` and `.B` `[out, r]`.
- Write metadata: `decoder.lora_adapters` (comma-separated names),
  `decoder.lora_rank`, `decoder.lora_alpha`.

**Step 2 -- Loading** (`src/decoder_embed.cpp`):
- Detect `decoder.lora_adapters` in GGUF metadata.
- Load A/B tensors into a secondary backend buffer (reuse the QKV fusion
  allocation pattern from `vit_embed.cpp` lines 224-263).
- Store as `ctx->lora[adapter_name][layer_idx] = {q_A, q_B, k_A, k_B, ...}`.

**Step 3 -- Graph** (`src/decoder_embed.cpp` forward):
- When LoRA is active, for each augmented matmul:
  `y = mul_mat(W, x) + scale(mul_mat(B, mul_mat(A, x)), alpha/r)`
  Two extra matmuls per LoRA weight (tiny: r x D and D x r).
- Alternative: pre-compute `W' = W + (a/r)*B@A` on CPU at switch time,
  then use W' directly. Faster inference, slower switching.

**Step 4 -- API**: `crispembed_set_lora(ctx, "retrieval")`,
`crispembed_list_lora(ctx, &names, &count)`.

**Testing**: Convert Jina v5 with `--lora-mode=separate`, verify each
adapter matches the baked version (cos >= 0.9999).

**Files**: `models/convert-decoder-embed-to-gguf.py`, `src/decoder_embed.cpp`,
`src/crispembed.{h,cpp}`, `examples/cli/main.cpp`

**Effort**: Medium (4-5 days)

---

### Blueprint: Nomic v2 MoE encoder

**Goal**: Support Mixture-of-Experts FFN layers in the BERT encoder so
nomic-embed-text-v2 (and similar MoE embedding models) can run.

**Current state**: NomicBERT v1.5 (non-MoE) works: Post-LN, SwiGLU, RoPE.
Standard FFN: `y = FC2(act(FC1(x)))`. See encoder forward in
`src/crispembed.cpp` line ~700+.

**MoE architecture**: Replace dense FFN with N experts + router:
`router_logits = matmul(gate_w, x)` -> topk -> weighted expert dispatch.

**ggml support**: `ggml_mul_mat_id(as, b, ids)` (ggml.h:1423) provides
indirect matmul -- dispatches rows to different weight matrices by ID.
Supported on CPU/CUDA/Metal/Vulkan.

**Step 1 -- Converter** (`models/convert-bert-to-gguf.py`):
- Detect MoE: check for `encoder.layer.{i}.mlp.experts.{k}.fc1.weight`
  or `gate.weight` in state dict.
- Stack expert weights: `enc.{i}.ffn.expert_fc1.weight` shape
  `[N_experts, inter, hidden]` for `ggml_mul_mat_id`.
- Store router: `enc.{i}.ffn.gate.weight` shape `[N_experts, hidden]`.
- Metadata: `bert.num_experts`, `bert.num_experts_per_tok` (top-K).

**Step 2 -- Encoder forward** (`src/crispembed.cpp`):
- Per-layer, after LN, check `L.expert_fc1_w`:
  ```
  logits = mul_mat(gate_w, x)    // [N, T]
  ids, weights = topk(softmax(logits), K)
  up = mul_mat_id(expert_fc1, x, ids)
  up = act(up)
  down = mul_mat_id(expert_fc2, up, ids)
  x = weighted_combine(down, weights)
  ```
- Top-K selection: ggml may lack a topk op. Fallback: compute router
  logits on CPU (extract via `ggml_backend_tensor_get` after a partial
  graph compute), determine top-K IDs, pass back as input tensor.
  Alternatively, implement topk via `ggml_argsort` + `ggml_get_rows`.

**Step 3 -- Testing**: Convert nomic-embed-text-v2, compare per-layer
with `dump_reference.py` + `crispembed_diff.h`. Non-MoE layers should
match exactly; MoE layers match if routing is identical.

**Files**: `models/convert-bert-to-gguf.py`, `src/crispembed.cpp`,
`src/crispembed.h` (layer struct needs expert fields)

**Effort**: High (7-10 days). The topk routing is the hardest part.

---

### Blueprint: Batched decoder graph

**Goal**: Run N decoder texts in one graph compute instead of N sequential
passes. Expected 2-4x speedup for batches of 4-8.

**Current state**: Encoder has true batching (`encode_tokens_batch`,
crispembed.cpp:1226). Decoder is sequential (crispembed.cpp:1689).

**Approach -- padded batching** (recommended first):

**Step 1 -- New function** `decoder_encode_tokens_batch()` in
`decoder_embed.cpp`:
- Pad all B sequences to T_max = max(len(tokens[i])) with pad token.
- Flatten batch into sequence: `[D, T_max*B]` (same as encoder batching).
- Build block-diagonal causal attention mask: text i cannot attend to
  text j, and causal within each text, and padding positions masked.
  Pre-compute on CPU, pass as `kq_b` to `ggml_flash_attn_ext`.
- RoPE: independent positions per text (0..len[i]-1, then 0 for padding).

**Step 2 -- Pooling**:
- For last-token pooling: extract token at `len[i]-1` offset within each
  text's block. Use `ggml_get_rows` with custom index tensor.
- L2-normalize per text.

**Step 3 -- Dispatch**: In `crispembed_encode_batch()`, call the new batch
function for decoder models instead of the sequential loop.

**KV cache**: Low priority for embeddings (each text is independent). Only
useful if many texts share a prompt prefix. Defer.

**Files**: `src/decoder_embed.cpp` (new batch function),
`src/crispembed.cpp` (dispatch), `src/decoder_embed.h`

**Effort**: High (6-8 days). Block-diagonal mask construction is the
tricky part.

---

### Blueprint: CrispLens face pipeline integration

**Goal**: Python API for face detection + recognition so CrispLens can
call it for face search/verification.

**Current state**: Face C API is complete (`crispembed.h` lines 408-475):
`crispembed_detect_faces()`, `crispembed_encode_face()`,
`crispembed_face_pipeline()`. Missing: Python wrapper.

**Step 1 -- Python wrapper** (`python/crispembed/_binding.py`):
- ctypes bindings for face functions.
- `CrispFace` class: `detect(image_path)`, `encode(image_path, landmarks)`,
  `pipeline(image_path)` returning dicts with bbox/confidence/embedding.

**Step 2 -- High-level API** (`python/crispembed/__init__.py`):
- `from crispembed import CrispFace`
- `CrispFace.from_registry("yunet", "auraface-v1")` for auto-download.

**Step 3 -- Example** (`examples/face_search.py`):
- Index faces from a directory, query by image, return top-K matches.

**Files**: `python/crispembed/_binding.py`, `python/crispembed/__init__.py`,
`examples/face_search.py`

**Effort**: Low (1-2 days). C API is already complete and tested.
