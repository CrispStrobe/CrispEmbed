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
    │               Handwritten math → LaTeX (CROHME 2014, 53% exact match)
    │
    ├─► Math  ──► PosFormer: DenseNet + Transformer + ARM (posformer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2014, 56% raw / 61.4% parsed)
    │               NOTE: SJTU weights = academic-only; training NC weights on Kaggle
    │
    └─► Math  ──► PPFormulaNet-S / Texo-Distill (ppformulanet_ocr.cpp)
                    HGNetv2 CNN encoder + MBart decoder (20M params)
                    Printed math → LaTeX (BLEU 0.90 on UniMER SPE)
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

**Current status**:
- Retrained CROHME weights (epoch 93, ~57% beam=1) wired into CrispCalc
  with NC confirmation gate. On HF: cstr/posformer-crohme-GGUF.
- Training continues to epoch 300 (plain cosine decay, label smoothing).
- Cosine warm restarts caused crash (57%→38%→60%→reverted to epoch 93).

**Dataset landscape** (verified 2026-06-09):

| Dataset | License | Commercial? | Type | Size |
|---------|---------|------------|------|------|
| UniMER ArXiv+Pix2tex | **Apache 2.0** | **Yes** | Printed | 978K |
| CROHME 2014/16 | CC BY-NC-SA 3.0 | No | Handwritten | 8.8K |
| MathWriting (Google) | CC BY-NC-SA 4.0 | No | Handwritten | 230K |
| HME100K / MLHME-38K (TAL) | Proprietary | No | Handwritten | 74K/38K |
| Im2LaTeX-100K | CC0 | Yes | Printed | 100K |
| figshare CROHME+HME100K | Claims CC BY 4.0 | **No** — unauthorized re-upload | Mixed | 83K |

**Training strategy**:
1. Pretrain on UniMER Apache 2.0 (978K printed) → **commercial weights**
2. Fine-tune on CROHME (8.8K handwritten) → NC weights for handwritten
3. Ship both: printed model (no gate) + handwritten model (NC gate)

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
| PPFormulaNet (Texo) | BPE | HGNetv2 CNN enc, MBart PRE-LN dec | texo-distill (20M) |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |

## OCR Model Catalog — available models and integration status

### Implemented

| Model | Params | Encoder | Decoder | License | Status |
|---|---|---|---|---|---|
| pix2tex-mfr | ~28M | DeiT ViT | TrOCR (6L, 256d) | MIT | math_ocr.cpp, shipped |
| HMER | ~7M | DenseNet-121 | GRU + coverage attn | MIT | hmer_ocr.cpp, shipped |
| BTTR | ~6M | DenseNet | Transformer (3L, 256d) | MIT | bttr_ocr.cpp, shipped |
| PosFormer | ~6M | DenseNet | Transformer+ARM (3L) | CC BY-NC-SA 3.0 | posformer_ocr.cpp, shipped |
| Texo-distill | 20M | HGNetv2 CNN | MBart PRE-LN (2L, 384d) | **AGPL-3.0** | ppformulanet_ocr.cpp, shipped |

### Available for integration (priority order)

#### Tier 1: PP-FormulaNet-L (safetensors, Apache-2.0, 200M)
- HF: `PaddlePaddle/PP-FormulaNet-L_safetensors`
- Encoder: SAM-style ViT (768d, 12L, windowed attn + global attn at [2,5,8,11], rel pos)
  + post-conv CNN (256→512→1024)
- Decoder: MBart (8L, 16H, d_model=512, FFN=2048)
- Image: 768x768, vocab: 50000
- **New ggml work**: windowed attention with relative position bias (SAM ViT),
  post-conv CNN neck. Decoder is larger MBart (same architecture as Texo-distill).
- Native HF `pp_formulanet` model type support in transformers.
- **Recommended next** — Apache-2.0, safetensors ready, best official accuracy.

#### Tier 2: PP-FormulaNet_plus-L (safetensors, Apache-2.0, 200M)
- HF: `PaddlePaddle/PP-FormulaNet_plus-L_safetensors`
- Same encoder architecture as PP-FormulaNet-L
- Decoder: MBart (8L, 16H, d=512), max_pos=2560 (longer sequences)
- Plus variant adds parallel decoding (speculative decode)
- **Same ggml work as Tier 1** — encoder code shared.

#### Tier 3: UniMERNet family (PyTorch .pth, Apache-2.0)
- HF: `wanderkid/unimernet_tiny` (107M), `unimernet_small` (76M?), `unimernet_base` (145M?)
- Encoder: **Swin Transformer** (windowed self-attn, shifted windows, rel pos bias)
  depths=[6,6,6,6] (Swin-based, NOT SAM-style)
- Decoder: MBart (8L, d=512/768/1024)
- Image: 192x672 (rectangular!)
- Vocab: 50000 (UniMERNet tokenizer)
- **New ggml work**: Swin Transformer (different from SAM ViT — shifted windows,
  cyclic shifts, window merging). Substantial effort.
- Best published BLEU scores. Code is Apache-2.0.

#### Tier 4: PP-FormulaNet S/M (Paddle inference only, Apache-2.0)
- HF: `PaddlePaddle/PP-FormulaNet-S`, `PP-FormulaNet_plus-S`, `PP-FormulaNet_plus-M`
- Only `.pdiparams` (Paddle PIR inference format), **no safetensors**
- S: HGNetv2 encoder (same as Texo-distill), 57M total
- M: intermediate size
- Would need Paddle PIR→GGUF converter or wait for safetensors release.
- S encoder already implemented (ppformulanet_ocr.cpp) — only need trained decoder.

#### Tier 5: PP-FormulaNet-L ONNX (Apache-2.0)
- HF: `ningpp/PP-FormulaNet-L-ONNX`, `ningpp/PP-FormulaNet_plus-L-ONNX`,
  `PaddlePaddle/PP-FormulaNet_plus-L_onnx`
- Could extract weights from ONNX → GGUF (like pix2tex converter did).

#### Tier 6: TexTeller 3.0 (Apache-2.0, 300M)
- HF: `OleehyO/TexTeller`
- Encoder: ViT (Swin-variant), Decoder: RoBERTa
- 300M params, trained on 80M image-formula pairs
- Too large for mobile (300M), desktop-only.

#### Not recommended
- **MixTeX** — AGPL-3.0, unknown architecture, no safetensors
- **Texo-transfer** (687 vocab) — AGPL-3.0, smaller vocab than distill

### Architecture reuse matrix

| Encoder type | Models using it | ggml status |
|---|---|---|
| HGNetv2 CNN | Texo-distill, PPFormulaNet-S/M | **Done** (ppformulanet_ocr.cpp) |
| SAM-style ViT (windowed+global+relpos) | PPFormulaNet-L, PPFormulaNet_plus-L | **Not started** |
| Swin Transformer | UniMERNet (all sizes) | **Not started** |
| DeiT ViT | pix2tex-mfr | **Done** (math_ocr.cpp) |
| DenseNet | HMER, BTTR, PosFormer | **Done** (bttr_ocr.cpp) |
| Standard ViT | TexTeller 3.0 | Partial (vit_embed.cpp) |

| Decoder type | Models using it | ggml status |
|---|---|---|
| MBart PRE-LN (2-8 layers) | Texo-distill, PPFormulaNet-L, UniMERNet | **Done** (ppformulanet_ocr.cpp) |
| TrOCR POST-LN | pix2tex-mfr | **Done** (math_ocr.cpp) |
| Transformer POST-LN | BTTR, PosFormer | **Done** (bttr_ocr.cpp) |
| GRU + coverage | HMER | **Done** (hmer_ocr.cpp) |

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
- [x] General OCR: DBNet text detection (ResNet-18+FPNC, 7 MB Q4_K, cos=1.0 parity)
- [x] General OCR: TrOCR text recognition (reuses math_ocr engine, exact token match vs HF)
- [x] General OCR: pipeline glue (detect → crop → recognize), C API, CLI --ocr

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

- [ ] **Unified dump harness** — CrispASR already has a unified
  `tools/dump_reference.py` with a `--backend` flag and a plug-in
  registry (`tools/reference_backends/<name>.py`). CrispEmbed has 3
  one-off scripts (`dump_reference.py`, `dump_trocr_intermediates.py`,
  `dump_math_ocr_reference.py`). Adopt the CrispASR contract: single
  CLI, backend registry, consistent stage names, shared GGUF writer.
  Each model family (encoder, decoder, math_ocr, dbnet, trocr) becomes
  a ~60-line backend module. Reduces duplication, makes adding new
  models (like DBNet, general TrOCR) trivial.

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

**Testing**: DONE. Jina v5 small converted with `--lora-mode=separate`.
4 adapters (retrieval, text-matching, clustering, classification) load
and hot-swap correctly. retrieval vs baked cos=0.999984. Round-trip
switching produces bit-identical embeddings. Bugs fixed: converter PEFT
`.base_layer` key stripping, decoder `gguf_free` use-after-free.

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

---

### Blueprint: General OCR — Text Detection (DBNet) + General TrOCR

**Goal**: Add a complete open-source OCR pipeline to CrispEmbed: detect text
regions in any image, then recognize each region into Unicode text. All via
ggml/GGUF, quantized, cross-platform — a portable, open alternative to
proprietary engines like WeChat OCR.

**Motivation**: CrispEmbed already has encoder-decoder vision models
(DeiT+TrOCR for math, PosFormer for handwriting) and CNN detection models
(YuNet/SCRFD for faces). General OCR combines both patterns: a CNN detector
to find text regions + a TrOCR-style encoder-decoder to read them. The
ggml graph-building infrastructure for both patterns already exists.

**Architecture overview**:

```
Input image
    │
    ├─► Text Detection (DBNet/DBNet++)
    │     ResNet-18 backbone + FPN + differentiable binarization head
    │     → binary map → contour extraction → oriented bounding boxes
    │     Model size: ~3 MB (Q4_K), ~12 MB (F16)
    │
    └─► Text Recognition (TrOCR-base or TrOCR-small)
          ViT/DeiT encoder + GPT-2/RoBERTa decoder (autoregressive)
          Per-crop: resize to 384×384 → encoder features → decode tokens
          Vocabulary: ~50K BPE tokens (multilingual capable)
          Model size: ~60 MB (Q4_K small), ~130 MB (Q4_K base)
```

**Why DBNet**: Lightweight (ResNet-18 backbone), fast, handles arbitrary
orientations, proven in PaddleOCR/MMOCR. The differentiable binarization
head is a simple sigmoid+threshold — no anchor boxes, no NMS. Same FPN
pattern as SCRFD (already implemented in cnn_embed.cpp).

**Why TrOCR**: We already have DeiT+TrOCR for math (math_ocr.cpp). General
TrOCR uses the same architecture with different training data. Microsoft
publishes pretrained checkpoints for English, and community fine-tunes
exist for many languages. Encoder-decoder transformer is our strongest
ggml pattern.

#### Phase 1 — Text Detection: DBNet

**Step 1 — Converter** (`models/convert-dbnet-to-gguf.py`):
- Load DBNet from MMOCR or PaddleOCR checkpoint (PyTorch or Paddle format).
- MMOCR preferred: `mmocr/dbnet_resnet18_fpnc_1200e_icdar2015`.
- Extract ResNet-18 backbone (conv1 + 4 stages), FPN neck (lateral + smooth
  convs), DBNet head (probability map + threshold map convs).
- Store as GGUF with architecture metadata:
  `ocr_det.arch = "dbnet"`, `ocr_det.backbone = "resnet18"`,
  `ocr_det.fpn_channels = 256`, `ocr_det.input_size = 640`.
- Weight naming: `det.backbone.conv1.{weight,bias}`,
  `det.backbone.stage{i}.block{j}.conv{k}.{weight,bias}`,
  `det.fpn.lateral{i}.{weight,bias}`, `det.fpn.smooth{i}.{weight,bias}`,
  `det.head.prob.{weight,bias}`, `det.head.thresh.{weight,bias}`.

**Step 2 — Inference** (`src/ocr_detect.{h,cpp}`):
- ResNet-18 forward: Conv2D → BN (folded at convert time) → ReLU → MaxPool
  → 4 stages of BasicBlock (skip connections). Reuse conv2d patterns from
  cnn_embed.cpp (SCRFD).
- FPN: lateral 1×1 convs + upsample + add + smooth 3×3 convs. Concatenate
  multi-scale features. Same pattern as SCRFD FPN.
- DBNet head: 3×3 conv → BN → ReLU → 1×1 conv → sigmoid → probability map.
- Post-processing (CPU, not ggml):
  - Threshold probability map at 0.3 → binary map
  - Find contours (simple scanline or OpenCV-free implementation)
  - Fit minimum bounding rectangles (oriented or axis-aligned)
  - Filter by area (min 100px²) and score (>0.5)
  - Expand boxes by 1.5× for recognition padding
  - Return list of `ocr_box { float x[4], y[4]; float score; float angle; }`

**Step 3 — Image preprocessing**:
- Reuse `image_preprocess.{h,cpp}` for resize, normalize.
- DBNet expects: resize longest side to 640 (or 960), pad to multiple of 32,
  normalize with ImageNet mean/std.

#### Phase 2 — Text Recognition: General TrOCR

**Step 4 — Converter** (`models/convert-trocr-to-gguf.py`):
- Load Microsoft `trocr-small-printed` or `trocr-base-printed` from HF.
- Encoder: DeiT/BEiT ViT (same as math_ocr converter, lines ~50-120).
- Decoder: GPT-2 style with cross-attention (same as math_ocr converter).
- Key difference from math TrOCR: vocabulary is full BPE (~50K tokens)
  instead of math-only (~500 tokens). Store full vocab in GGUF.
- Metadata: `ocr_rec.arch = "trocr"`, `ocr_rec.vocab_size`,
  `ocr_rec.max_length = 128`, `ocr_rec.bos_token_id`, `ocr_rec.eos_token_id`.

**Step 5 — Inference** (`src/ocr_recognize.{h,cpp}`):
- Heavily based on math_ocr.cpp. Main differences:
  - Input: arbitrary text crop (not just equation) resized to 384×384.
  - Vocabulary: BPE with ~50K tokens (reuse tokenizer_bpe.cpp for decoding).
  - Output: Unicode string (not LaTeX).
  - Beam search optional (greedy is fine for most text).
- Forward: DeiT encoder → cross-attention decoder → argmax per step → BPE
  detokenize.

#### Phase 3 — Pipeline Integration

**Step 6 — Pipeline API** (`src/ocr_pipeline.{h,cpp}`):
- `crispembed_ocr_detect(ctx, image_path, &boxes, &n_boxes)` — detection only
- `crispembed_ocr_recognize(ctx, image_path, crop_box, &text)` — recognition only
- `crispembed_ocr(ctx, image_path, &results, &n_results)` — full pipeline:
  detect → sort boxes (top-to-bottom, left-to-right) → recognize each →
  return `ocr_result { ocr_box box; char *text; float confidence; }[]`

**Step 7 — C API** (`src/crispembed.h`):
```c
// OCR text detection
int crispembed_ocr_detect(crispembed_ctx *ctx,
                          const char *image_path,
                          crispembed_ocr_box **boxes,
                          int *n_boxes);

// OCR text recognition (single crop)
int crispembed_ocr_recognize(crispembed_ctx *ctx,
                             const char *image_path,
                             crispembed_ocr_box *box,   // NULL = full image
                             char **text,
                             float *confidence);

// Full OCR pipeline (detect + recognize all)
int crispembed_ocr(crispembed_ctx *ctx,
                   const char *image_path,
                   crispembed_ocr_result **results,
                   int *n_results);

void crispembed_ocr_free(crispembed_ocr_result *results, int n);
```

**Step 8 — CLI/Server**:
- CLI: `crispembed --ocr image.png` → prints detected text with bounding boxes.
- Server: `/ocr` endpoint accepting image upload, returning JSON array of
  `{ "text": "...", "box": [[x1,y1], ...], "confidence": 0.95 }`.

**Step 9 — Python/Rust/Dart bindings**: Extend wrappers with `ocr()` method.

#### Testing & Quality Gates

- Detection: IoU ≥ 0.80 vs MMOCR on ICDAR 2015 test set (500 images).
- Recognition: Character accuracy ≥ 95% on printed English (ICDAR 2015 crops).
- End-to-end: F1 ≥ 0.75 on ICDAR 2015 end-to-end (word-level).
- Quantization: Q4_K detection cos ≥ 0.99 vs F32; Q4_K recognition
  accuracy drop < 1%.
- Total pipeline size target: ~70 MB (Q4_K detection + Q4_K TrOCR-small).

#### Model Registry Entries

```
| ID             | Arch    | Size (Q4_K) | Notes              |
|----------------|---------|-------------|--------------------|
| dbnet-ic15     | DBNet   | ~3 MB       | ICDAR 2015 trained |
| trocr-small-en | TrOCR   | ~60 MB      | English printed    |
| trocr-base-en  | TrOCR   | ~130 MB     | English printed    |
```

#### What We Already Have (reuse map)

| Existing code | Reuse for |
|---|---|
| `cnn_embed.cpp` (SCRFD FPN) | DBNet backbone + FPN |
| `math_ocr.cpp` (DeiT+TrOCR) | TrOCR recognition (nearly identical) |
| `image_preprocess.cpp` | Image resize/normalize/crop |
| `tokenizer_bpe.cpp` | BPE detokenization for TrOCR output |
| `convert-hmer-to-gguf.py` | DenseNet/CNN conversion patterns |
| `models/convert-bert-to-gguf.py` | GGUF metadata writing patterns |

#### Execution Order

1. DBNet converter + inference (detection standalone)
2. General TrOCR converter (adapt from math_ocr converter)
3. General TrOCR inference (adapt from math_ocr.cpp)
4. Pipeline glue (detect → crop → recognize)
5. C API + CLI + server endpoint
6. Python/Rust/Dart bindings
7. Quality benchmarks on ICDAR 2015

**Effort**: High (10-14 days). Detection is the most new code; recognition
is largely a vocabulary swap from math_ocr.

---

## CrispASR reuse map — don't reinvent the wheel

CrispASR (sibling repo at `../CrispASR`) already has significant
infrastructure that overlaps with CrispEmbed's OCR needs. Before writing
new C++ code, check whether CrispASR already provides it.

**Already in CrispASR — DO NOT REBUILD in CrispEmbed:**

| Capability | CrispASR location | Notes |
|---|---|---|
| Translation (MADLAD-400/T5) | `src/t5_translate.{h,cpp}` | Full T5 encoder-decoder, beam search, 200+ languages |
| Translation (M2M-100) | `src/m2m100.{h,cpp}` | Encoder-decoder, 100 languages |
| Language identification | (LID model in CrispASR) | Route text to correct translation pair |
| Cross-attention (enc→dec) | `src/core/cross_attn.h` | KV projection + step helpers |
| Beam search decoding | `src/core/beam_decode.h` | Replay-from-prefix style |
| Greedy decoding | `src/core/greedy_decode.h` | Step-by-step token generation |
| Self-attention + KV cache | `src/core/attention.h` | Multi-variant, quantized KV, CPU offload |
| FFN (SwiGLU/GeGLU) | `src/core/ffn.h` | Shared with CrispEmbed (copy) |
| SentencePiece tokenizer | `src/core/sentencepiece.h` | Unigram Viterbi, header-only |
| BPE tokenizer | `src/core/bpe.h` | GPT-2 byte-level, header-only |
| WordPiece tokenizer | `src/core/wordpiece.h` | Greedy longest-prefix |
| GGUF loader | `src/core/gguf_loader.{h,cpp}` | Two-pass metadata+weights |

**Current sharing model**: CrispEmbed copies `gguf_loader`, `attention.h`,
`ffn.h`, `bpe.h` from CrispASR. BidirLM-Omni audio links against
`libcrispasr` as a shared library.

**Open question**: Should `core/` become a shared git submodule instead of
ad-hoc copies? Both repos evolve these files. Current drift is manageable
but will grow as CrispEmbed adds more encoder-decoder models (OCR, etc.).

**For the OCR blueprint specifically**: The general TrOCR decoder reuses
the existing `math_ocr.cpp` inline decoder (which was written before the
CrispASR core/ copy existed). It does NOT need a new decoder from scratch.
The main new code is DBNet detection (CNN, same pattern as cnn_embed.cpp).

**For the full detect → OCR → translate pipeline**: OCR runs in CrispEmbed,
then hand off the recognized text to CrispASR's translation API (already
exposed as a C library). No need to port translation into CrispEmbed.

## Future OCR extensions (unscoped)

- [ ] **Document layout analysis** — YOLOv8 or RT-DETR to classify page
  regions (text, table, figure, equation) before routing to OCR or math OCR.

- [ ] **PaddleOCR PP-OCRv4 alternative** — SVTR recognizer instead of TrOCR.
  Smaller (~10 MB) but less accurate. Good for mobile/edge.

- [ ] **Multilingual TrOCR** — Fine-tune or find community checkpoints for
  CJK, Arabic, Devanagari. The BPE tokenizer already supports Unicode.

- [ ] **Handwriting recognition (general)** — Extend PosFormer/TrOCR
  training to IAM Handwriting dataset for general handwritten English.
