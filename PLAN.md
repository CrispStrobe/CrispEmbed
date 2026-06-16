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
    ├─► Math  ──► PosFormer: BTTR + ARM coverage (posformer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME, improved over BTTR)
    │
    ├─► Math  ──► MixTex: Swin-Tiny + RoBERTa (mixtex_ocr.cpp)
    │               Chinese+English LaTeX OCR (25681 BPE vocab)
    │
    ├─► Math  ──► PP-FormulaNet-S: HGNetv2 + MBart (ppformulanet_ocr.cpp)
    │               57M params, 384×384 input
    │
    ├─► Math  ──► PP-FormulaNet-L: SAM-ViT + MBart (ppformulanet_l_ocr.cpp)
    │               181M params, 768×768 input
    │
    ├─► OCR   ──► DBNet + TrOCR pipeline (ocr_pipeline.cpp)
    │               Text detection → recognition → reading-order sort
    │
    ├─► OCR   ──► Surya-OCR-2 detector (surya_det.cpp)
    │               EfficientViT + SegFormer, 38M, 91 languages
    │
    ├─► OCR   ──► Qwen2.5-VL (qwen2vl_ocr.cpp)
    │               3B VLM, German business document OCR
    │
    ├─► Layout ─► RT-DETRv2 docling-heron (layout_detect.cpp)
    │               ResNet-50 + deformable xattn, 17 document classes
    │
    │   ── PLANNED ──
    │
    ├─► OCR   ──► PARSeq (24M, MIT) — fast text-line recognition EN+DE
    ├─► OCR   ──► InternVL2.5-2B (2.1B, MIT) — OCRBench ~830, EN+DE
    ├─► OCR   ──► InternVL2-1B (0.9B, MIT) — edge/WASM OCR
    ├─► OCR   ──► Granite Vision 3.3-2B (3B, Apache) — OCRBench 852
    └─► OCR   ──► H2OVL-Mississippi-2B (2.1B, Apache) — OCRBench 782
```

## Supported architectures (v0.7.0)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| NomicBERT MoE encoder | SentencePiece | Post-LN, MoE 8-expert top-2, GELU, RoPE | nomic-embed-text-v2-moe |
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
| PosFormer | — | DenseNet + Transformer + ARM | posformer (handwritten math) |
| MixTex | BPE (25681) | Swin-Tiny + RoBERTa 4L decoder | mixtex (CN+EN LaTeX) |
| PP-FormulaNet-S | BPE (50000) | HGNetv2 CNN + MBart 2L decoder | ppformulanet (57M) |
| PP-FormulaNet-L | BPE (50000) | SAM-ViT + MBart 8L decoder | ppformulanet-l (181M) |
| DBNet | — | ResNet-18 + FPN + DB head | text detection (12M) |
| Surya-Det | — | EfficientViT + SegFormer | surya-ocr-2 detector (38M, 91 langs) |
| RT-DETRv2 | — | ResNet-50 + deformable xattn | layout-heron (17 classes) |
| Qwen2.5-VL | tiktoken | ViT-32L + spatial merger + Qwen2.5 LLM | german-ocr-3 (3B) |

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

#### GPU + quantization audit (2026-06-16, updated after all gaps filled)

All 35 inference engines now GPU-enabled. Zero CPU-only gaps remaining.
Every engine uses `ggml_backend_init_best()` and has a `<ENGINE>_FORCE_CPU=1`
env var override. A/B verified on CPU — identical outputs, no regression.

**FULL GPU** — `ggml_backend_init_best()` + ggml graph compute (CUDA/Vulkan/Metal).
These run the full forward pass on GPU when available:

| Engine | Env override | Quant | GGUF sizes |
|--------|-------------|-------|------------|
| crispembed (BERT/XLM-R/etc.) | `CRISPEMBED_FORCE_CPU=1` | Q4_K/Q8_0 | varies |
| decoder_embed (Qwen3/Gemma3) | (inherits from crispembed) | Q4_K/Q8_0 | varies |
| bidirlm_vision | (auto GPU device) | Q8_0/F16 | varies |
| fireredpunc | (auto) | Q8_0 | ~400 MB |
| pcs | (auto) | Q8_0 | ~1 GB |
| gliner_ner | `CRISPEMBED_FORCE_CPU=1` | Q4_K/Q8_0 | varies |
| got_ocr | (auto) | Q4_K/Q8_0 | ~700 MB |
| surya_det | `SURYA_DET_FORCE_CPU=1` | Q4_K/Q8_0 | 30-73 MB |
| tesseract_lstm | (auto) | Q4_K/Q8_0 | ~3 MB/lang |
| vit_embed | `VIT_EMBED_FORCE_CPU=1` | Q4_K/Q8_0 | 50-350 MB |
| clip_text_embed | `CLIP_TEXT_FORCE_CPU=1` | F32 | ~250 MB |
| cnn_embed | `CNN_EMBED_FORCE_CPU=1` | Q4_K/Q8_0 | 5-250 MB |
| ocr_detect | `OCR_DETECT_FORCE_CPU=1` | Q4_K/Q8_0 | 7 MB |
| parseq_ocr | `PARSEQ_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 6-90 MB |
| layout_detect | `LAYOUT_DETECT_FORCE_CPU=1` | Q8_0 | ~42 MB |
| internvl2_ocr | `INTERNVL2_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 1-4 GB |
| qwen2vl_ocr | `QWEN2VL_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 2.6-7.6 GB |
| glm_ocr | `GLM_OCR_FORCE_CPU=1` | Q8_0 | ~950 MB |
| math_ocr | `MATH_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | varies |
| ppformulanet_l_ocr | `PPFNL_OCR_FORCE_CPU=1` | Q8_0 | ~180 MB |
| lilt_kie | `LILT_KIE_FORCE_CPU=1` | Q4_K/Q8_0 | ~350 MB |
| bert_ner | `BERT_NER_FORCE_CPU=1` | varies | varies |

**GPU-SAFE** — `ggml_backend_init_best()` + `ggml_backend_tensor_get` for
weight reads, scalar CPU forward pass. Weights on GPU, compute on CPU.
Full GPU compute needs ggml graph rewrite (depthwise conv, PixelShuffle, etc.):

| Engine | Env override | Quant | GGUF sizes |
|--------|-------------|-------|------------|
| hmer_ocr | `HMER_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 12-48 MB |
| bttr_ocr | `BTTR_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 12-48 MB |
| posformer_ocr | `POSFORMER_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 10-25 MB |
| nafnet_denoise | `NAFNET_FORCE_CPU=1` | Q4_K/Q8_0 | 15-56 MB |
| mixtex_ocr | `MIXTEX_OCR_FORCE_CPU=1` | Q4_K/Q8_0 | 85-329 MB |
| ppformulanet_ocr | `PPFN_OCR_FORCE_CPU=1` | Q8_0 | ~20 MB |
| pan_sr | `PAN_SR_FORCE_CPU=1` | Q4_K/Q8_0/F16 | 543K-577K |
| tbsrn_sr | `TBSRN_SR_FORCE_CPU=1` | Q4_K/Q8_0/F16 | 686K-2.2M |
| text_sr | `TEXT_SR_FORCE_CPU=1` | F16 | varies |
| safmn_sr | `SAFMN_SR_FORCE_CPU=1` | Q4_K/Q8_0/F32 | 947K-970K |
| esrgan_sr | `ESRGAN_SR_FORCE_CPU=1` | Q4_K/Q8_0/F32 | 358K-2.4M |
| restormer | `RESTORMER_FORCE_CPU=1` | Q4_K/Q8_0/F16 | 28-50 MB |
| tps_locnet | `TPS_LOCNET_FORCE_CPU=1` | Q4_K/Q8_0/F32 | 88K-449K |
| scunet_denoise | `SCUNET_FORCE_CPU=1` | Q4_K/Q8_0 | varies |
| swinir_sr | `SWINIR_SR_FORCE_CPU=1` | Q4_K/Q8_0 | varies |

**Not model engines** (no GPU needed):
bidirlm_audio (shared backend from bidirlm_vision), hat_sr (uses
esrgan_sr engine internally).

**Summary (2026-06-16, final):**
- 22 engines: full GPU acceleration (ggml graph compute)
- 15 engines: GPU-safe weight storage (scalar CPU compute)
- 0 engines: CPU-only
- All 37 inference engines have `<ENGINE>_FORCE_CPU=1` env var overrides
- All SR/restoration models quantized to Q8_0 + Q4_K (zero quant gaps)

### Models

- [x] CLIP text encoder (causal mask variant)
- [x] SigLIP-large, CLIP-large conversion + upload
- [x] SigLIP / ViT quantization (conv2d needs F32 kernel — selective quant)
- [x] YuNet lightweight face detection alternative
- [x] SFace INT8 quantization (Q8_0 cos=0.9999, Q4_K cos=0.974; 37→10→6 MB)
- [x] Face model quantized inference via graph replayer (YuNet F16/Q8_0 working; fixed depthwise IC, ggml_n_dims trailing-1s, Q→F32 dequant path)
- [x] ViT parity: cos 0.8→1.0 (was patch ordering bug — permute(2,1,0) gave column-major spatial, fixed to permute(1,2,0,3) for row-major matching HF)
- [x] Nomic v2 MoE (MoE routing layer in encoder) — cos=1.000000 vs HF
- [x] LoRA adapter hot-swap (Jina v5 per-task adapters, pre-compute merge on CPU, ~10-50ms switch)

### OCR — next-gen models to port

#### Specialized detection + recognition (lightweight pipeline models)

- [x] surya-ocr-2 (0.7B, OpenRail-M free <$5M) — detector ported, FULL PARITY VERIFIED (heatmap max+mean exact match). Encoder moved to ggml graph (13x speedup). GGUF on HF.
- [x] **PARSeq (24M, MIT)** — ViT encoder + Transformer decoder, scene-text recognition. DONE: engine, converter, reference dumper, test, GGUF. Wired into orchestrator.
- [x] GLM-OCR (0.9B, MIT) — CogViT + GLM-0.5B, 8 languages. DONE: engine, converter, reference dumper, test, GGUF. Wired into orchestrator.
- [x] GOT-OCR2_0 (0.7B, Apache-2.0) — SAM-ViT + Qwen-0.5B, end-to-end doc OCR. DONE: engine, converter, reference dumper, test, GGUF. Wired into orchestrator.

#### VLM-based OCR (OCRBench-ranked, document understanding)

- [x] **Keyven/german-ocr-3 — Qwen2.5-VL base engine DONE**. Full engine, converter, parity, orchestrator wiring.
- [x] **InternVL2 (MIT)** — InternViT + InternLM2.5 VLM. DONE: engine (`internvl2_ocr.{h,cpp}`), converter, reference dumper, diff test, e2e test. Wired into orchestrator. GGUFs: 1B and 2B variants.
- [x] Keyven/german-ocr-3.1 (2B, Apache-2.0) — Qwen2.5-VL fine-tune, German business docs. DONE: wrote merge-llamacpp-qwen2vl-gguf.py to convert split llama.cpp GGUFs (LLM+mmproj) to single CrispEmbed GGUF. Loads and passes unit tests on qwen2vl_ocr engine. Q4_K 1.3GB.
- [x] Nanonets-OCR2-1.5B (1.5B, Apache-2.0) — Qwen2-VL pruned fine-tune (16L vs 28L), 12+ languages incl. German. DONE: runs on existing qwen2vl_ocr engine unchanged. GGUF converted (F16 3.6GB, Q4_K 1.3GB). Model registry entry added.
- [ ] Qari-OCR (2B, Apache-2.0) — Qwen2-VL fine-tune, Arabic with diacritics (parity bug: hallucinated text)
- [ ] **Granite Vision 3.3-2B (~3B, Apache-2.0)** — OCRBench 852 (highest in class). English-only.
- [x] **H2OVL-Mississippi-2B (~2.1B, Apache-2.0)** — OCRBench 782. DONE: InternViT-300M + Danube-2-1.8B (Mistral arch). Runs on existing internvl2_ocr engine unchanged. GGUF converted (F16 1.2GB, Q4_K 457MB). Model registry entry added.

#### OCRBench leaderboard reference (small VLMs, ≤3B)

| Rank | Model | LLM | Params | OCRBench | License | llama.cpp | Priority |
|------|-------|-----|--------|----------|---------|-----------|----------|
| 1 | Granite Vision 3.3-2B | Granite-3.1-2B | 3B | 852 | Apache-2.0 | Not yet | Medium (EN-only) |
| 2 | InternVL2.5-2B* | InternLM2.5-1.8B | 2.1B | ~830 | MIT | **Yes** | **High** |
| 3 | MiniMonkey | InternLM2-1.8B | ~2B | 806 | — | No | Low |
| 4 | H2OVL-Mississippi-2B | H2O-Danube-1.8B | 2.1B | 782 | Apache-2.0 | No | Medium |
| 5 | InternVL2-1B | Qwen2-0.5B | 0.9B | 779 | MIT | **Yes** | **High (edge)** |
| 6 | InternVL2-4B | Phi-3-mini | ~4B | 776 | MIT | Yes | Low (too big) |
| 7 | H2OVL-Mississippi-0.8B | H2O-Danube3-0.5B | 0.8B | 751 | Apache-2.0 | No | Medium (tiny) |

*InternVL2.5-2B not on original leaderboard slice but scores higher than InternVL2-2B (768).

### Bindings

- [x] Python wrapper `encode_image()` for standalone SigLIP/CLIP
- [x] CrispFacePipeline export + from_registry() + Python unit tests + face_search example
- [ ] CrispLens integration — update `crispembed_client.py` for face pipeline

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| ~~Nomic v2 MoE~~ | ~~Low~~ | ~~High~~ | ~~MoE routing layer in encoder~~ DONE |
| Qwen3-VL multimodal | Low | High | Reuse BidirLM-Omni scaffolding |

### Pending improvements

- [x] **Layout decoder → ggml graph** — Self-attention + FFN now use
  ggml graph with BLAS-accelerated matmuls. Decoder: 16.4s → 8s (2x).
  Total: 36s → 19.6s. Key fix: weight data must be transposed when
  creating ggml input tensors (Gemm convention ↔ ggml mul_mat stride).
  Cross-attention stays CPU scalar (deformable grid sampling).

- [x] **PPFormulaNet-L: BLAS enabled, Q8_0 — 53s (was 60s)** — The 4 global
  layers (2304² attention matrix = 5.3M per head × 12 heads) dominate
  encoder time. Options: (a) use flash_attn_ext for memory efficiency,
  (b) tile/block the computation to improve cache behavior,
  (c) Q8_0 quantized attention for reduced bandwidth.

- [x] **Upload layout-heron GGUFs to HuggingFace** — The Q8_0 model
  (43 MB) exists at `/mnt/storage/models/layout-heron-q8_0.gguf` but
  hasn't been published to `cstr/layout-heron-gguf`. Also need F16
  variant and model card.

- [x] **CrispCalc Dart catalog** — Add `OcrModelVariant` entries for
  layout-heron (Q8_0, F16, F32) in `lib/engine/ocr_model_manager.dart`.
  Register at appropriate priority tier in `ocr_providers_init.dart`.

- [x] **Layout detection** — RT-DETRv2 docling-heron, full parity (7/7 detections exact match). Score gap (0.934 vs 0.955) is
  Root cause: bilinear resize pixel differences (PIL vs custom C++)
  cascading through 50+ backbone Conv layers. Fix: match PIL's exact
  coordinate mapping or use stb_image_resize2 with matching filter.

- [x] **Fix Q8_0 layout model** — DONE: cast quantized tensors to F32 before
  `ggml_add`/`ggml_mul` (commits `0ac0d91`, `24798e3`).

- [x] (duplicate — see line 189)
  share a prompt prefix (e.g. Jina v5 instruction prefix), compute KV
  for the shared prefix once and reuse across the batch.

- [ ] **Streaming ColBERT late interaction scoring** — Server-side MaxSim
  scoring via `/colbert/score` endpoint with SSE streaming.

- [x] **Quantized GGUF for face models** — Quantizer now flattens 4D conv
  weights to 2D before quantizing. SFace: Q8_0 cos=0.9996 (37→10 MB),
  Q6_K cos=0.9966 (37→8 MB). Q4_K cos=0.936 (too low for recognition).
  SCRFD detection: Q8_0 17→10 MB, Q4_K 17→8.7 MB.

- [ ] **Live-test LoRA with Jina v5** — LoRA hot-swap is implemented but
  not end-to-end tested with real Jina v5 adapters. Need to: convert with
  `--lora-mode=separate`, verify each adapter (retrieval, classification,
  clustering, text-matching) matches the baked version (cos >= 0.9999),
  confirm switching works correctly, test with the Python wrapper.

- [x] **3D tensor quantization for MoE experts** — DONE. Quantizer now
  handles 3D tensors by quantizing each 2D slice independently. Results:
  nomic-v2-moe Q8_0: 1122→487 MB (3.8x), Q4_K: 1095→352 MB (5.2x).
  Quality: Q8_0 cos=0.9995, Q4_K cos=0.964 vs F32.

---

## Implementation blueprints

Detailed specs for pending roadmap items. Each blueprint is self-contained
so a fresh agent can implement it independently.

### Blueprint: LoRA adapter hot-swap — DONE

Implemented June 2026. See HISTORY.md. C API: `crispembed_set_lora()`,
`crispembed_list_lora()`. Pre-compute merge `W' = W + (a/r)*B@A` on CPU.

---

### Blueprint: Nomic v2 MoE encoder — DONE

Implemented June 2026. cos=1.000000 vs HuggingFace. See HISTORY.md.
Key learning: NomicBERT v2-moe has Wqkv + out_proj biases (v1.5 does not).
Parity harness extended with `--arch nomic` for QKV split + MoE expert diff.

---

### Blueprint: Batched decoder graph — DONE

Implemented June 2026. `decoder_encode_tokens_batch()` in decoder_embed.cpp.
Block-diagonal F32 causal mask, per-text RoPE, last-token pooling. ~3x speedup.

---

### Blueprint: KV cache for prefix-shared decoder batches

**Goal**: When N texts share a prompt prefix (e.g. Jina v5 instruction
`"Retrieve semantically similar text.\nQuery: "`), compute KV for the
prefix once and reuse across the batch, saving ~40% of compute.

**Current state**: `decoder_encode_tokens_batch()` (decoder_embed.cpp:1158)
pads all sequences to T_max and builds a block-diagonal mask. Each text
recomputes the full prefix independently.

**Approach**:
1. Detect common prefix: compare token IDs across batch, find longest
   shared prefix length `P`. Typical Jina v5 prefix is ~15 tokens.
2. Compute prefix KV: Run a single forward pass for `[P]` tokens.
   Extract K/V tensors for each layer → `prefix_kv[layer] = {K, V}`.
3. Extend graph: For each text's remaining `T-P` tokens, use pre-computed
   prefix K/V. Modify `ggml_flash_attn_ext` call to pass concatenated
   K = `[prefix_K ; text_K]`, V = `[prefix_V ; text_V]`.
4. Mask: Each text attends causally to `[prefix + own_tokens]`, not to
   other texts' tokens.

**Challenge**: ggml's `flash_attn_ext` takes K/V as single tensors. Need
either: (a) pre-concatenate K/V on CPU before graph compute, or (b) use
`ggml_concat` in-graph (adds one concat op per layer, negligible cost).

**Files**: `src/decoder_embed.cpp`, `src/decoder_embed.h`

**Effort**: Medium (3-4 days). Prefix detection is trivial; KV caching
mechanics are well-understood from LLM inference.

---

### Blueprint: Batched decoder improvements (F16 mask + Gemma3 NaN fix)

**Goal**: Two targeted fixes for the batched decoder path.

**F16 attention mask**: Current mask is F32 (`ggml_new_tensor_2d(gctx,
GGML_TYPE_F32, T_total, T_total)` at decoder_embed.cpp:1297). For
T_total=512 that's 1 MB; for T_total=2048 it's 16 MB. `ggml_flash_attn_ext`
supports F16 `kq_b` natively. Change to:
```cpp
ggml_tensor * attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_total, T_total);
```
CPU-side mask fill: cast `-INFINITY` and `0.0f` to `ggml_fp16_t` via
`ggml_fp32_to_fp16()`. Reduces mask memory 2x.

**Gemma3 NaN fix**: Some Gemma3 GGUFs produce NaN embeddings on certain
inputs. Root cause: `(1+w)*rms_norm(x)` can overflow when `w` is large
and `rms_norm(x)` has extreme values. The Ollama-format GGUFs pre-bake
`1+w` into norm weights (flag `gemma_norm=false`), but CrispEmbed-native
GGUFs use raw `w` with `gemma_norm=true`. Fix: clamp RMSNorm output to
`[-65504, 65504]` (F16 max) before the `(1+w)` multiply, or detect NaN
after the first layer and fall back to F32 compute for that batch.

**Files**: `src/decoder_embed.cpp`

**Effort**: Low (1 day). Both are mechanical changes.

---

### Blueprint: Streaming ColBERT late interaction scoring

**Goal**: Add `/colbert/score` endpoint to the HTTP server for MaxSim
scoring between a query's multi-vector embeddings and pre-stored document
token vectors, with streaming partial results.

**Current state**: `crispembed_encode_multivec()` (crispembed.cpp) produces
per-token ColBERT embeddings for BGE-M3. No server endpoint for scoring.
No MaxSim implementation.

**MaxSim**: For query tokens Q `[nq, d]` and doc tokens D `[nd, d]`:
`score = sum_i(max_j(cos(Q[i], D[j])))`. O(nq * nd * d) per document.

**Step 1 -- C API**: Add `crispembed_colbert_score(ctx, query_vecs, n_query,
doc_vecs, n_doc, dim)` returning a float score. Pure CPU computation (no
ggml graph needed — it's just a batched dot product + max reduction).

**Step 2 -- Server endpoint** (`examples/server/server.cpp`):
- `POST /colbert/score`: JSON body with `query_vecs` (or `query_text` to
  encode on the fly), `documents` (list of pre-computed doc vecs or raw
  texts). Returns ranked list with scores.
- SSE streaming: After each document's score is computed, emit a partial
  result via `text/event-stream`. Client sees progressive ranking.

**Step 3 -- Batched scoring**: For K documents, parallelize MaxSim across
documents (each document is independent). Use OpenMP.

**Files**: `src/crispembed.cpp` (C API), `src/crispembed.h`,
`examples/server/server.cpp`

**Effort**: Medium (2-3 days). MaxSim is simple math; SSE streaming is
the main engineering work.

---

### Blueprint: WASM build target

**Goal**: Compile CrispEmbed to WebAssembly for browser-based embedding.

**Current state**: ggml has Emscripten support (whisper.cpp WASM demo
exists). CrispEmbed uses standard ggml APIs. CMakeLists.txt has no WASM
target.

**Step 1 -- CMake toolchain**:
- Add `cmake/wasm.cmake` toolchain file for Emscripten.
- Build static library + JS wrapper. Disable audio/face/vision paths
  (browser doesn't need them). Target: `crispembed.js` + `crispembed.wasm`.
- Flags: `-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -msimd128` (relaxed SIMD).
- Threading: `-pthread -s SHARED_MEMORY=1` requires `SharedArrayBuffer`
  (needs COOP/COEP headers). Start single-threaded, add workers later.

**Step 2 -- JS API** (`wasm/crispembed.js`):
- `async function loadModel(url)` — fetch and load GGUF into WASM heap.
- `function encode(text)` — returns Float32Array of embedding.
- `async function encodeBatch(texts)` — batch encode.

**Step 3 -- Demo page** (`wasm/demo.html`):
- Load a small model (all-MiniLM-L6-v2, 24 MB Q4_K).
- Text input → encode → show embedding dims + cosine vs reference.

**Challenges**:
- Memory: 2 GB WASM limit. MiniLM Q4_K fits (24 MB); larger models need
  streaming GGUF or won't fit.
- First load: GGUF download + WASM compilation. Use `caches` API.
- No GPU: CPU-only, but ggml SIMD kernels should be fast enough for
  small models.

**Files**: `cmake/wasm.cmake`, `wasm/crispembed.js`, `wasm/demo.html`,
`CMakeLists.txt` (add wasm target)

**Effort**: Medium (3-4 days). Main risk is ggml WASM compatibility.

---

### Blueprint: INT4 GGUF for face models

**Goal**: Quantize Conv2D weights in face models (SCRFD, AuraFace, SFace)
to Q4_K for 4x size reduction.

**Current state**: `tools/quantize.cpp` skips tensors with ndims > 2
(line 172: "skipping N-D tensor (conv2d)"). Conv2D weights are stored as
2D `[OC, IC*KH*KW]` in the GGUF (flattened from 4D for quantization
compatibility). The graph replayer (`src/cnn_embed.cpp`) reshapes them
back to 4D and dequants Q→F32 before `ggml_conv_2d`.

**The fix is simple**: The quantizer already handles 2D tensors. The issue
is that face model conv weights are ALREADY 2D in the GGUF (flattened by
the converter). The quantizer skips them because of a name-based guard
(`patch_embed` check) or because certain tensors are small (< 256 elements).

**Step 1 -- Quantizer** (`tools/quantize.cpp`):
- Remove or relax the `patch_embed` / ndims guard for face models.
- Add `--include-conv` flag to opt-in to conv weight quantization.
- Keep bias/norm tensors at F32 (1D tensors, already excluded).

**Step 2 -- Graph replayer** (`src/cnn_embed.cpp`):
- Already handles Q→F32 dequant (commit 2fcde98). Verify it works for
  Q4_K (not just Q8_0).

**Step 3 -- Testing**: Quantize YuNet, SCRFD, AuraFace, SFace to Q4_K.
Compare detection boxes (IoU ≥ 0.95) and recognition embeddings (cos ≥ 0.99).

**Expected sizes**:
- AuraFace: 249 MB → ~65 MB (Q4_K)
- SCRFD: 17 MB → ~5 MB
- SFace: 37 MB → ~6 MB (already have Q8_0 at 10 MB)

**Files**: `tools/quantize.cpp`, `src/cnn_embed.cpp` (verify only)

**Effort**: Low (1 day). The infrastructure exists; this is mostly
testing.

---

### Blueprint: Live-test LoRA with Jina v5

**Goal**: End-to-end verification that LoRA hot-swap works correctly with
real Jina v5 adapters, not just unit tests.

**Current state**: LoRA infrastructure is implemented (decoder_embed.cpp,
C API, Python wrapper). Converter supports `--lora-mode=separate`. But no
end-to-end parity test exists against HuggingFace per-adapter outputs.

**Step 1 -- Convert**: `python models/convert-decoder-embed-to-gguf.py
--model jinaai/jina-embeddings-v5-text-small --lora-mode=separate --output
jina-v5-small-lora.gguf`

**Step 2 -- Per-adapter parity** (new test `tests/test_lora_parity.py`):
- For each adapter (retrieval, classification, clustering, text-matching):
  - HF: `model.set_adapter(name)` → encode test texts
  - CrispEmbed: `crispembed_set_lora(ctx, name)` → encode same texts
  - Compare: cos >= 0.999 per text
- Also test: switching adapters mid-session, listing available adapters,
  setting invalid adapter name (should fail gracefully).

**Step 3 -- Python wrapper test**: Verify `ctx.set_lora("retrieval")`,
`ctx.list_lora()`, `ctx.lora` property work end-to-end.

**Files**: `tests/test_lora_parity.py` (new), test data

**Effort**: Low (1 day). Infrastructure exists; this is testing only.

---

### Blueprint: 3D tensor quantization for MoE experts

**Goal**: Extend `tools/quantize.cpp` to quantize 3D tensors (MoE expert
weights) instead of copying them as F32.

**Current state**: Quantizer skips ndims > 2 (line 172). For nomic-v2-moe,
expert_fc1 `[8, 3072, 768]` and expert_fc2 `[8, 768, 3072]` stay F32,
limiting Q8_0 to 1122 MB (vs potential ~600 MB if experts were quantized).

**Approach**: Iterate over the outermost dimension (n_experts=8) and
quantize each 2D slice `[inter, hidden]` independently. The ggml quantize
functions work on flat arrays with a specified number of elements per row.

**Step 1** (`tools/quantize.cpp`):
```cpp
if (ndims == 3) {
    // Quantize each 2D slice of the 3D tensor
    int64_t slice_size = ne[0] * ne[1];
    for (int64_t k = 0; k < ne[2]; k++) {
        quantize_slice(src + k * slice_size, dst + k * q_slice_size,
                       ne[0], ne[1], target_type);
    }
}
```

**Step 2** -- Verify: `ggml_mul_mat_id` must support quantized `as` tensor.
Check ggml source: `ggml_mul_mat_id` calls `ggml_compute_forward_mul_mat`
per expert, which supports quantized weights. Should work.

**Step 3** -- Test: Quantize nomic-v2-moe with 3D quant, verify cos >= 0.999
(Q8_0) and cos >= 0.96 (Q4_K).

**Expected sizes** (nomic-v2-moe):
- F32: 1818 MB → Q8_0: ~600 MB (vs current 1122 MB)
- Expert weights: 6 layers × 2 × [8, 3072, 768] = ~845 MB F32 → ~225 MB Q8_0

**Files**: `tools/quantize.cpp`

**Effort**: Low (half day). Small code change + testing.

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

### Blueprint: surya-ocr-2 (full-page OCR, 91 languages) — IN PROGRESS

**Goal**: Port surya-ocr-2 for multilingual full-page OCR with text
detection, recognition, and layout analysis.

**Source**: datalab-to/surya-ocr-2 (0.7B, OpenRail-M — free for <$5M)
GitHub: https://github.com/VikParuchuri/surya

**Architecture**: Two separate models:
1. **Detector** (38M params, 147 MB F32, 73 MB F16):
   EfficientViT-Large segformer. Input [3,1200,1200] → 2-channel
   heatmap [2,300,300] (text line, separator) → polygon bounding boxes.
   - Stem: Conv3x3(s2)+BN+Hardswish + ConvBlock residual → [32,600,600]
   - Stage 0: 2× FusedMBConv (expand 16x/4x) → [64,300,300]
   - Stage 1: 2× FusedMBConv (expand 16x/4x) → [128,150,150]
   - Stage 2: 7× MBConv (expand 16x/4x, ReLU6) → [256,75,75]
   - Stage 3: 1× MBConv (expand 24x) + 6× EfficientVitBlock → [512,38,38]
     (each VitBlock = LiteMLA linear attention + MBConv, with residuals)
   - DecodeHead: 4× Linear proj → bilinear upsample → cat(reversed) →
     Conv1x1+BN+ReLU → Conv1x1 classifier → sigmoid

2. **Recognizer** (0.7B): Qwen3.5-VL fine-tune. GGUF already exists at
   datalab-to/surya-ocr-2-gguf. Runs via llama-server with OpenAI API.
   Prompts: "OCR this block image to HTML." / "OCR this image to HTML.
   Each block is a div with data-label and data-bbox (x0 y0 x1 y1,
   normalized 0-1000)."

**Key op — LiteMLA** (O(N) linear attention):
  Q,K = ReLU(qkv_split), V_pad = pad(V,1)
  KTV = K^T @ V_pad  (head_dim × (head_dim+1), spatial aggregation)
  out = Q @ KTV / (Q @ K^T @ ones + eps)  (per-position normalization)
  Multi-scale: DW-Conv5x5 + grouped-Conv1x1 aggregation, cat with original

**Status** (2026-06-11):
- [x] Reference dumper: `tools/dump_surya_det_reference.py` (52 activations)
- [x] GGUF converter: `models/convert-surya-det-to-gguf.py` (148 tensors, BN folded)
- [x] C++ engine: `src/surya_det.{h,cpp}` (CPU-scalar, compiles+runs)
- [x] Test binaries: `test-surya-det`, `test-surya-det-diff`
- [x] Model registry entry: `surya-det`
- [x] Parity verification — heatmap max=0.9649 exact, mean=0.0113 exact
- [x] Heatmap → polygon post-processing (connected components + bbox)
- [x] Move encoder to ggml graph — 13min→1min (13x). Stages 0-2 + block0: 17s via graph. LiteMLA + decode scalar.
- [x] Upload GGUF to HuggingFace — https://huggingface.co/cstr/surya-det-GGUF (F32, F16, Q8_0, Q4_K)
- [x] GPU enablement (ggml_backend_init_best, SURYA_DET_FORCE_CPU=1)
- [x] CUDA/GPU testing — all 35 engines GPU-enabled (ggml_backend_init_best)
- [x] Image format support: test binaries use stb_image for PNG/JPG/BMP

**GGUFs**: `/mnt/storage/gguf-models/surya-det-{f32,f16}.gguf`

**Files**: `src/surya_det.{h,cpp}`, `models/convert-surya-det-to-gguf.py`,
`tools/dump_surya_det_reference.py`, `tests/test_surya_det{,_diff}.cpp`

**Effort**: Medium remaining (3-4 days for parity + ggml graph + postproc).

---

### Blueprint: MixTex ZhEn-Latex-OCR (86M, Apache-2.0) — IN PROGRESS

**Goal**: Chinese+English math LaTeX OCR. Smallest new model, introduces
Swin encoder as new building block.

**Source**: MixTex/ZhEn-Latex-OCR (85.9M, Apache-2.0)

**Architecture**: VisionEncoderDecoderModel
1. **Encoder**: Swin-Tiny (28M params)
   - Patch embed: Conv2d [96, 3, 4, 4] stride 4
   - 4 stages: depths=[2,2,6,2], heads=[3,6,12,24], dims=[96,192,384,768]
   - Window size=7, shifted window attention, relative position bias
   - Patch merging (2×2 concat + Linear) between stages
   - Output: [N, 768] where N=H/32×W/32 (180 for 400×500 input)

2. **Decoder**: RoBERTa 4-layer with cross-attention (57M params)
   - hidden=768, 12 heads, FFN=3072, GELU
   - Post-LN: self-attn → LN → cross-attn → LN → FFN → LN
   - BPE tokenizer, 25681 tokens (LaTeX + CJK)
   - max_position=300, greedy decode

**Status** (2026-06-11):
- [x] Reference dumper: `tools/dump_mixtex_reference.py`
- [x] GGUF converter: `models/convert-mixtex-to-gguf.py` (345 tensors)
- [x] GGUF files: F32=329MB, F16=165MB at `/mnt/storage/gguf-models/`
- [x] C++ engine: `src/mixtex_ocr.{h,cpp}` — runs end-to-end, Swin+RoBERTa
- [x] [x] Parity VERIFIED — encoder cos=1.000000, decoder cos=1.000000 (confirmed by running Python decoder with same encoder output). Previous cos=-1.0 was caused by reference GGUF inconsistency (dec_layer captured from different preprocessing → cross-attn → FFN pipeline.

**Key new op**: Swin shifted-window attention with relative position bias.
Window partition → local MHSA + RPB lookup → window reverse → shift.

**Files**: `tools/dump_mixtex_reference.py`, `models/convert-mixtex-to-gguf.py`

**Effort**: Medium (3-4 days). Swin encoder is new; decoder is same as TrOCR.

---

### PDF + document processing improvements

Implement from first principles using the PDF specification
(ISO 32000, public) and Apache-2.0 reference implementations.
Primary reference: **Tesseract's pdfrenderer.cpp** (Apache-2.0,
976 LOC). Also: qpdf (Apache-2.0), pdfcpu (Apache-2.0), libharu (zlib).
Listed in priority order:

- [x] **PDF text matrix positioning (Tm)** — use text matrix with affine
  coefficients instead of simple Td. Enables rotation-aware rendering.
  Reference: Tesseract pdfrenderer.cpp (Apache-2.0). DONE.

- [x] **Glyph-width-aware text positioning** — scale font size so rendered
  text width matches OCR bounding box width (Helvetica avg glyph ≈ 0.55*size).
  Makes text selection in the PDF accurately span words. DONE.

- [x] **Searchable PDF with embedded page image** — embed original page image
  as JPEG XObject (/DCTDecode), drawn as full-page background before invisible
  text layer. True searchable PDF output. DONE.

- [x] **Image downsampling calculator** — `compute_downsample_factor()`:
  DPI-based + max_pixels constraint. Pure math. DONE.

- [x] **PDF page DPI profiling** — zero-dependency PDF parser that extracts
  image metadata to compute effective page DPI. Weighted harmonic mean for
  mixed-resolution pages. 36/36 tests, exact DPI at 72-600 DPI. DONE.

- [x] **OCR quality scoring** — `ocr_quality_score()`: binary-search
  dictionary lookup for word match ratio. DONE.

- [x] **PDF/A-2b metadata** — `ocr_render_set_pdfa()` adds XMP metadata
  stream (pdfaid:part=2, conformance=B) + sRGB OutputIntent. DONE.

---

### Advanced OCR capabilities (Apache-2.0 references available)

- [x] **TPS Spatial Transformer** — learnable thin-plate-spline dewarping.
  20-point control prediction CNN (108K params, PaddleOCR RARE, Apache-2.0).
  TPS math + localization net + auto_dewarp pipeline. Parity cos=1.000000.
  Full integration: C API, CLI, server, Python, Rust, Dart. DONE.

- [x] **Text angle classification** — `detect_text_angle()`: heuristic
  0°/180° detection via ascender/descender asymmetry + page-level ink
  distribution. No model needed. Returns 0 or 180 + confidence. DONE.

- [x] **Table structure recognition** — rule-based HTML extraction via
  morphological line detection + projection splitting. 14/14 tests pass.
  Full integration (C API, CLI, server, Python, Dart, Rust). DONE.
  Future: model-based TableMaster for borderless tables.

- [x] **Key Information Extraction (KIE)** — OCR + NER pipeline for
  structured field extraction from documents. LiLT layout-aware model
  (dual-stream BiACM, 25/25 layers cos=1.000000). Plus simple
  OCR→NER chaining. Full integration. DONE.

- [x] **Text super-resolution** — NAFNet-SR (tiled full-page, Hann
  blending, 2x/4x), TBSRN (per-line, PaddleOCR Telescope, 1.1M params),
  PAN (whole-image, 272K params). Auto-trigger in orchestrator when
  DPI < threshold. Parity-tested. Full integration. DONE.

- [x] **Refined DBNet postprocessing** — DONE. Moore contour tracing,
  convex hull (Andrew's monotone chain), min-area rotated rectangle
  (rotating calipers), polygon-interior probability scoring (ray-casting),
  Unclip on rotated rect. Replaces axis-aligned bbox with rotated quad.

---

### Image restoration — next-gen models to port

Current engines (all DONE with full integration):
- NAFNet denoise (29M, SIDD), NAFNet-SR (tiled full-page, no trained model)
- TBSRN (per-line, cos=0.999985), PAN (whole-image 272K, cos=0.999654)
- SAFMN (228K, cos=1.000000), SwinIR-light (0.9M-4.2M, cos=0.986)
- Restormer (26M, denoise+SR+deblur), SCUNet (18M, Swin-Conv-UNet denoise)
- Real-ESRGAN (SRVGGNetCompact, 0.9M-17M)

#### DONE — Priority 1 (all ported)

| # | Model | Params | Task | License | Status |
|---|-------|--------|------|---------|--------|
| 1 | **SAFMN** | ~228K | SR | Apache-2.0 | DONE, parity cos=1.000000 |
| 2 | **SwinIR-light** | ~0.9M-4.2M | SR | Apache-2.0 | DONE, parity cos=0.986 (F16 through 24 attn layers) |
| 3 | **Restormer** | 26M | Denoise/SR/Deblur | Apache-2.0 | DONE, full integration |
| 4 | **SCUNet** | ~18M | Denoise | Apache-2.0 | DONE, parity cos=1.000000 |

#### Priority 2 — Evaluate later (heavier or GAN-based)

| # | Model | Params | Task | License | Status |
|---|-------|--------|------|---------|--------|
| 5 | **HAT** | 21M | SR | MIT | **DONE** (engine+converter+C API+all bindings, cos=0.9999 parity) |
| 6 | **DAT** | ~830K (light) | SR | Apache-2.0 | **DONE** (engine+converter+C API+all bindings, cos=0.9999 parity) |
| 7 | **Real-ESRGAN** | ~0.9M-17M | SR | BSD-3 | **DONE** |
| 8 | **AdaIR** | 28.8M | All-in-one | MIT | **DONE** (Restormer+AFLB, cos=0.999924, all bindings) |
| 9 | **InstructIR** | ~15M | All-in-one | Apache-2.0 | **DONE** (text-guided restoration, all bindings) |
| 10 | ~~PromptIR~~ | ~26M | All-in-one | Academic (NC) | **REJECTED** — non-commercial license, replaced by InstructIR |
| 11 | ~~AirNet~~ | ~9M | All-in-one | No license | **REJECTED** — no LICENSE file in repo |

#### Implementation notes

- **SAFMN** (Priority 1): 200K params, spatially-adaptive feature modulation.
  Architecture: shallow feature extraction → N SAFM blocks → PixelShuffle.
  Each SAFM block: channel mixing (1×1 conv) + spatial mixing (multi-scale
  DW-conv with channel split). ~200 LOC C++ engine. Converter adapts from
  existing NAFNet pattern (Conv+BN folding). Reference: `liang-j20/SAFMN`.

- **SwinIR-light** (Priority 1): Reuses Swin window attention from
  `mixtex_ocr.cpp`. Architecture: shallow embed → N RSTB (residual Swin
  transformer blocks) → upsample (PixelShuffle). ~400 LOC. Reference:
  `JingyunLiang/SwinIR`.

- **Restormer** (Priority 1): Multi-Dconv head transposed attention
  (transpose spatial to channel dimension, then attention). U-Net
  encoder-decoder with skip connections. Follows NAFNet inference pattern
  closely. ~600 LOC. Reference: `swz30/Restormer`.

- **SCUNet** (Priority 1): Swin-Conv-UNet. Alternates Swin transformer
  blocks with residual conv blocks in a U-Net. Follows `nafnet_denoise.cpp`
  pattern for the U-Net structure, adds Swin blocks from `mixtex_ocr.cpp`.
  ~500 LOC. Reference: `cszn/SCUNet`.

- **DAT** (Priority 2, DONE): Dual Aggregation Transformer. Split-channel
  windowed spatial attention + L2-normalized transposed channel attention,
  both with AIM (Adaptive Interaction Module). SGFN feed-forward with
  DW-conv gating. ~1400 LOC C++ engine. cos=0.999956 parity vs PyTorch.
  
- **HAT** (Priority 2): Hybrid Attention Transformer. Same Swin-based
- **HAT** (DONE): Hybrid Attention Transformer (21M, MIT, CVPR 2023 SOTA).
  Swin + overlapping cross-attention. cos=0.999990. All bindings wired.

- **Real-ESRGAN** (DONE): SRVGGNetCompact (~620K, BSD-3). All bindings.

- **AdaIR** (DONE): Adaptive all-in-one restoration (28.8M, MIT).
  Restormer + AFLB. cos=0.999924. All bindings wired.

- **InstructIR** (DONE): Text-guided restoration (~15M, Apache-2.0).
  All bindings wired.

- **PromptIR/AirNet** (Priority 2): All-in-one models. PromptIR uses
  learnable prompts; AirNet uses contrastive encoder.
---

### Blueprint: GLM-OCR (0.9B, MIT, GGUF exists)

**Goal**: Integrate GLM-OCR for general document OCR. GGUF already
converted by ggml-org — may only need inference integration, not
conversion.

**Source**: zai-org/GLM-OCR (0.9B, MIT)
GGUF: ggml-org/GLM-OCR-GGUF (Q8_0: 950 MB, F16: 1.79 GB)

**Architecture**: CogViT vision encoder + GLM-0.5B language decoder.
Lightweight cross-modal connector with token downsampling. 8 languages.

**Approach**: Since GGUF already exists and runs via llama-server, the
fastest path may be to study the GGUF tensor layout and write a native
CrispEmbed inference engine that loads it directly, rather than
re-converting.

**Files**: `src/glm_ocr.{h,cpp}`, possibly no converter needed

**Effort**: Medium (3-4 days). GGUF exists; main work is C++ inference.

---

### Blueprint: GOT-OCR2_0 (0.7B, Apache-2.0)

**Goal**: End-to-end document OCR that handles plain text, LaTeX math,
tables, and formatted output in a single model.

**Source**: stepfun-ai/GOT-OCR2_0 (0.7B, Apache-2.0)

**Architecture**: SAM-style ViT-B vision encoder (12 layers, 768-dim,
16x16 patches, custom "Vary" backbone) + Qwen-0.5B causal LM decoder
(24 layers, 1024-dim). tiktoken tokenizer. Requires `trust_remote_code`.

**Reuse**: Qwen decoder path already in CrispEmbed. SAM-ViT encoder is
similar to PPFormulaNet-L's encoder (already ported). Main new work is
the vision-language connector and GOT-specific prompt templates.

**Step 1 — Converter**: `models/convert-got-ocr-to-gguf.py`
- Export vision encoder + connector + LM decoder.
- Handle custom modeling code (GOTQwenForCausalLM).

**Step 2 — C inference**: `src/got_ocr.{h,cpp}`

**Files**: `src/got_ocr.{h,cpp}`, `models/convert-got-ocr-to-gguf.py`

**Effort**: Medium (4-5 days). Custom vision backbone needs careful
mapping; decoder side reuses existing Qwen3 path.

---

### Blueprint: Nanonets-OCR2-1.5B (Apache-2.0)

**Goal**: Multilingual document OCR (12+ languages incl. German).

**Source**: nanonets/Nanonets-OCR2-1.5B-exp (1.5B, Apache-2.0)
Fine-tune of Qwen2-VL-2B-Instruct.

**Architecture**: Qwen2-VL vision encoder + Qwen2-VL language decoder.
Standard Qwen2-VL architecture — well-supported in llama.cpp, GGUF
already exists.

**Approach**: Since this is a standard Qwen2-VL fine-tune with existing
GGUF, integration may follow the same pattern as GLM-OCR — load
existing GGUF and write native inference.

**Files**: `src/nanonets_ocr.{h,cpp}` or reuse VLM dispatch

**Effort**: Medium (3-4 days). Standard Qwen2-VL, well-documented.

---

### Blueprint: Qari-OCR (Arabic, 2B, Apache-2.0) — CONVERTED, PARITY BUG OPEN

**Goal**: Arabic OCR with diacritics support (tashkeel).

**Source**: NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct (Apache-2.0)
LoRA fine-tune (r=16, 324 pairs) of Qwen2-VL-2B-Instruct on 50K Arabic samples.

**Architecture**: Qwen2-VL-2B — same family as existing qwen2vl_ocr.cpp:
- Vision: 32L ViT (embed_dim=1280, hidden_size=1536), merger 5120→1536
- LLM: 28L Qwen2 (1536d, GQA 12Q/2KV, FFN=8960)

**Status**:
- [x] LoRA merge (324/324 pairs) via Kaggle kernel
- [x] GGUF conversion (F16 4.7GB, Q8_0 2.3GB, Q4_K 1.6GB)
- [x] Converter fixed for Qwen2-VL config (embed_dim/mlp_ratio/in_chans)
- [x] HuggingFace upload: `cstr/qari-ocr-crispembed-GGUF`
- [x] Registry entry: `qari-ocr`
- [x] mRoPE grid_thw fix (was using dummy [1,1,1])
- [x] Reference GGUF captured (11 tensors: vis layers, merger, LLM layers, logits)
- [ ] **PARITY BUG**: CrispEmbed generates hallucinated prompt text instead of OCR

**Diagnostic findings** (Kaggle diff harness):
- PyTorch output: `'This image contains the text "Hello World 2024".'`
- CrispEmbed output: `'Below is the plain text representation...'` (prompt echo)
- PyTorch top-1 at last position: `This` (18.13)
- CrispEmbed top-1: `Below` (14.19) — different token, lower score
- Vision activations: shapes correct (132×1280, merger 33×1536)
- Token IDs: 58 tokens, 33 image_pad, grid [1,6,22] — all match
- **Divergence is in prefill logits** — vision embeds reach LLM but
  produce different attention patterns. Likely cause: Qwen2-VL uses
  GELU vision FFN (fc1/fc2) vs Qwen2.5-VL's SwiGLU (gate/up/down),
  and the C++ forward pass may have a subtle difference in activation
  or normalization order for the Qwen2-VL variant.
- Reference GGUF at `cstr/qari-ocr-crispembed-GGUF/qari-ocr-ref.gguf`
  for offline per-layer cos comparison via test-qwen2vl-diff.

---

### Blueprint: Keyven/german-ocr (German docs, Apache-2.0) — IN PROGRESS

**Goal**: German business document OCR with structured JSON output.

**Base model**: Qwen2.5-VL-3B-Instruct (Keyven/german-ocr-3 is a
fine-tune of this). Architecture: 32-layer ViT (1280d) + spatial merger
(2×2→2048d) + 36-layer Qwen2.5 LLM (2048d, GQA 16/2, mRoPE).

**Status (2026-06-12):**

| Step | Status | Notes |
|------|--------|-------|
| C++ inference engine | DONE | `src/qwen2vl_ocr.{h,cpp}` — vision + LLM + generation |
| GGUF converter | DONE | `models/convert-qwen2vl-to-gguf.py` (lazy tensor loading) |
| Reference dumper | DONE | `tools/dump_qwen2vl_reference.py` (safetensors, ~600MB peak) |
| Parity: vision encoder | DONE | 32/32 layers cos=1.000 |
| Parity: spatial merger | DONE | cos=1.000, max_abs=6e-4 |
| Parity: LLM decoder | DONE | 2/2 layers cos=1.000 with mRoPE |
| E2E generation (Q4_K) | DONE | "Um die Rechnung im Bild als" — coherent German |
| Quantization | DONE | F16 (7.6GB), Q8_0 (3.9GB), Q4_K (2.6GB, vision Q8_0 floor) |
| HuggingFace upload | DONE | `cstr/qwen2.5-vl-3b-crispembed-GGUF` (F16 + Q8_0 + Q4_K) |
| Wire into C ABI | TODO | Add to `crispembed.cpp` dispatch |
| CLI + model registry | TODO | Add to `model_mgr.cpp`, `--ocr` dispatch |
| Python bindings | TODO | Wire via `CrispMathOcr` auto-dispatch |
| CrispCalc catalog | TODO | `OcrModelVariant` entries |
| KV cache | TODO | O(n²)→O(n) per generated token |
| C++ image preprocessor | TODO | Currently uses Python-generated patches |
| Load Keyven fine-tune | TODO | Same architecture, just different weights |

**GGUFs**: `cstr/qwen2.5-vl-3b-crispembed-GGUF` on HuggingFace:
- `qwen2.5-vl-3b-f16.gguf` — 7.57 GiB
- `qwen2.5-vl-3b-q8_0.gguf` — 3.93 GiB (2x compression)
- `qwen2.5-vl-3b-q4_k.gguf` — 2.61 GiB (3x, vision Q8_0 preserved)

**Key learnings:**
- Vision weights need Q8_0 floor in quantizer (Q4_K degrades OCR)
- `ggml_set_output()` required on intermediates to prevent memory reuse
- GQA interleave via 4D reshape+repeat (not `ggml_repeat` which tiles)
- mRoPE uses neghalf rotation with `GGML_ROPE_TYPE_MROPE`
- Token splicing: `x = embed * keep_mask + image_patches`
- `AutoConfig` varies by transformers version — read config.json directly

**Files**: `src/qwen2vl_ocr.{h,cpp}`, `models/convert-qwen2vl-to-gguf.py`,
`tools/dump_qwen2vl_reference.py`, `tools/qwen2vl_tokenize.py`,
`tests/test_qwen2vl_diff.cpp`, `tests/test_qwen2vl_e2e.cpp`,
`tools/kaggle/qwen2vl-convert/`

---

### Blueprint: PARSeq — Lightweight Text-Line Recognition (24M, MIT)

**Goal**: Fast, accurate text-line recognizer for EN+DE to replace TrOCR
in `ocr_pipeline.cpp`. 3-14x smaller than TrOCR-small (62M) with
comparable or better accuracy.

**Source**: baudm/parseq (MIT), docTR multilingual variant for German.
Paper: "Scene Text Recognition with Permuted Autoregressive Sequence Models"

**Architecture** (24M base):
1. **Encoder**: ViT-Small (12 layers, 384d, 6 heads, patch 8×4)
   - Input: 128×32 grayscale or RGB text-line crop
   - Patch embed: Conv2d [384, 3, 8, 4] → flatten → [N_patches, 384]
   - Standard ViT with class token, GELU FFN, post-LN
   - Output: [N_patches+1, 384]

2. **Decoder**: Permuted Autoregressive Transformer (2 layers, 384d, 6 heads)
   - Cross-attention to encoder output
   - Position queries (max 26 tokens)
   - Greedy or refinement decode (iterative)
   - 94-char charset (digits + upper/lower + punctuation)

**Variants**:
- PARSeq-tiny: ~12M, ~93% accuracy — ideal for WASM/mobile
- PARSeq-base: ~24M, ~96% accuracy — sweet spot
- docTR multilingual: Latin charset covering German umlauts (ä, ö, ü, ß)

**GGML ops needed**: All standard — ggml_mul_mat, ggml_flash_attn_ext,
ggml_norm, ggml_gelu, ggml_conv_2d. Identical to existing TrOCR decoder
path. No exotic ops.

**Steps**:
1. Converter: `models/convert-parseq-to-gguf.py` — export ViT encoder +
   Transformer decoder. Simple weight mapping (no custom code).
2. Reference dumper: `tools/dump_parseq_reference.py`
3. C++ engine: `src/parseq_ocr.{h,cpp}` — encoder + decoder graph
4. Integration: replace TrOCR in `ocr_pipeline.cpp` (or add as option)
5. Test: parity vs Python + DBNet+PARSeq pipeline end-to-end

**GGUFs**: `/mnt/storage/gguf-models/parseq-{base,tiny}-{f32,f16,q8_0}.gguf`

**Files**: `src/parseq_ocr.{h,cpp}`, `models/convert-parseq-to-gguf.py`,
`tools/dump_parseq_reference.py`, `tests/test_parseq.cpp`

**Effort**: Medium (3-4 days). Pure ViT — reuses patterns from TrOCR/math_ocr.

---

### Blueprint: InternVL2.5-2B — Document OCR + Understanding (2.1B, MIT)

**Goal**: Compact VLM for full document understanding: OCR + VQA + KIE +
layout comprehension. OCRBench ~830. Multilingual (EN+DE). MIT license.

**Source**: OpenGVLab/InternVL2_5-2B (MIT)
Already in llama.cpp (InternVL2.5 1B+4B confirmed in multimodal.md).

**Architecture** (2.1B total):
1. **Vision encoder**: InternViT-300M-448px-V2.5 (300M)
   - Standard ViT: 24 layers, 384d, 6 heads, patch 14
   - Input: 448×448 per tile, dynamic tiling (1-12 tiles)
   - Pixel unshuffle: 4:1 spatial token reduction
   - Output: ~256 visual tokens per tile (1024 for 448px / 14 / 2)

2. **Projector**: 2-layer MLP with GELU (randomly initialized)
   - Linear(384→1536) → GELU → Linear(1536→1536)

3. **LLM decoder**: InternLM2.5-1.8B-chat (1.8B)
   - 24 layers, 1536d, 12 heads (GQA 12/4), RoPE
   - SiLU FFN (gate/up/down), RMSNorm
   - 8k context window

**Dynamic resolution**: Images split into 448×448 tiles based on aspect
ratio. Each tile processed independently by ViT. Thumbnail tile always
included. Max 12 tiles = 12×256 = 3072 visual tokens.

**Two port paths**:

**Path A — Leverage llama.cpp (fastest, 1-2 days)**:
- llama.cpp already has InternVL2.5 support
- Use their GGUF converter: `convert_hf_to_gguf.py`
- Write thin wrapper in CrispEmbed that calls the same graph
- Pro: minimal new code. Con: depends on llama.cpp graph structure.

**Path B — Native CrispEmbed engine (3-5 days)**:
- `src/internvl2_ocr.{h,cpp}` — vision + projector + LLM from scratch
- Vision encoder: identical ViT to existing `vit_embed.cpp` (same ops)
- Pixel unshuffle: `ggml_reshape_4d` + `ggml_permute` (trivial)
- MLP projector: 2× `ggml_mul_mat` + `ggml_gelu` (trivial)
- LLM decoder: standard transformer, reuse patterns from `qwen2vl_ocr.cpp`
  (GQA, RoPE, SiLU FFN — all identical). RoPE is standard 1D (simpler
  than Qwen2.5-VL's mRoPE).
- Pro: full control, follows CrispEmbed patterns. Con: more code.

**Recommended**: Path B. The architecture is simpler than Qwen2.5-VL
(which is already done). No mRoPE, no spatial merger, no 3D RoPE.
Standard 1D RoPE + GQA + SiLU FFN.

**Steps**:
1. Converter: `models/convert-internvl2-to-gguf.py`
2. Reference dumper: `tools/dump_internvl2_reference.py`
3. C++ engine: `src/internvl2_ocr.{h,cpp}`
4. Parity: vision encoder → projector → LLM layer-by-layer
5. E2E generation test with German + English documents
6. Quantize: F16, Q8_0, Q4_K (vision at Q8_0 floor, like Qwen2.5-VL)

**GGUFs**: `/mnt/storage/gguf-models/internvl2.5-2b-{f16,q8_0,q4_k}.gguf`

**Expected sizes**:
- F16: ~4.2 GB
- Q8_0: ~2.2 GB
- Q4_K: ~1.4 GB (vision Q8_0 preserved)

**Files**: `src/internvl2_ocr.{h,cpp}`, `models/convert-internvl2-to-gguf.py`,
`tools/dump_internvl2_reference.py`, `tests/test_internvl2_ocr.cpp`

**Effort**: Medium (3-5 days). Simpler than Qwen2.5-VL (no mRoPE).

---

### Blueprint: InternVL2-1B — Edge/WASM OCR (0.9B, MIT)

**Goal**: Tiniest competitive VLM for OCR. 0.9B params, quantizes to
~500MB Q4_K. OCRBench 779. Ideal for WASM, mobile, or resource-
constrained edge deployment.

**Source**: OpenGVLab/InternVL2-1B (MIT)

**Architecture** (0.9B total):
1. **Vision encoder**: InternViT-300M-448px (same as 2.5-2B)
2. **Projector**: MLP (same pattern)
3. **LLM decoder**: Qwen2-0.5B-Instruct (0.5B)
   - 24 layers, 896d, 14 heads (GQA 14/2), RoPE
   - SiLU FFN, RMSNorm

**Reuse**: Shares vision encoder + projector with InternVL2.5-2B.
Only the LLM decoder differs (Qwen2-0.5B vs InternLM2.5-1.8B).
If InternVL2.5-2B is ported first, this is a ~1 day adaptation.

**Steps**:
1. Reuse converter with `--model OpenGVLab/InternVL2-1B`
2. Swap LLM config (Qwen2 0.5B params)
3. Test: OCR quality on EN+DE docs, compare vs 2.5-2B

**GGUFs**: `/mnt/storage/gguf-models/internvl2-1b-{f16,q8_0,q4_k}.gguf`

**Expected sizes**:
- F16: ~1.8 GB
- Q8_0: ~1.0 GB
- Q4_K: ~0.5 GB

**Effort**: Low (1-2 days) if InternVL2.5-2B is done first.

---

### Blueprint: Granite Vision 3.3-2B — Highest OCRBench (3B, Apache-2.0)

**Goal**: Highest OCRBench score (852) in the small VLM class. English-
focused document OCR. Apache-2.0 license. LLaVA-style architecture.

**Source**: ibm-granite/granite-vision-3.3-2b (Apache-2.0)

**Architecture** (3B total):
1. **Vision encoder**: SigLIP2-400M (google/siglip2-so400m-patch14-384)
   - Standard ViT: patch 14, 384×384 input, ~400M params
   - AnyRes: multi-scale grid for high-res documents (up to 768px)
   - CrispEmbed already has SigLIP v1 in `vit_embed.cpp`

2. **Projector**: 2-layer MLP with GELU
   - Standard LLaVA connector

3. **LLM decoder**: Granite-3.1-2B-Instruct (2B, 128k context)
   - Standard transformer, details TBD (likely GQA + RoPE + SiLU)

**Key consideration**: English-only. Despite highest OCRBench score, no
German support out of the box. Fine-tuning on German data would be needed.
Also largest at 3B (vs 2.1B for InternVL2.5-2B).

**Port approach**: SigLIP2 is a standard ViT — diff vs SigLIP v1 is
minor (same patch14, similar dims). Granite LLM is a standard decoder.
Main work is verifying SigLIP2 compatibility with existing `vit_embed.cpp`
and writing the LLaVA-style token splicing.

**Steps**:
1. Verify SigLIP2 vs SigLIP v1 differences (likely: attention pooling,
   minor norm changes)
2. Converter: `models/convert-granite-vision-to-gguf.py`
3. C++ engine: `src/granite_vision_ocr.{h,cpp}` (or reuse generic VLM dispatch)
4. Parity test on English documents

**GGUFs**: `/mnt/storage/gguf-models/granite-vision-3.3-2b-{f16,q8_0,q4_k}.gguf`

**Expected sizes**:
- F16: ~6 GB
- Q8_0: ~3.2 GB
- Q4_K: ~2.0 GB

**Files**: `src/granite_vision_ocr.{h,cpp}`, `models/convert-granite-vision-to-gguf.py`

**Effort**: Medium (3-4 days). SigLIP v1→v2 delta is small; Granite LLM
is standard. Lower priority than InternVL due to English-only + larger size.

---

### Blueprint: H2OVL-Mississippi-2B — InternVL + H2O-Danube (2.1B, Apache-2.0)

**Goal**: Alternative 2B VLM trained on 17M image-text pairs. OCRBench
782. Uses InternVL vision pipeline with H2O-Danube LLM.

**Source**: h2oai/h2ovl-mississippi-2b (Apache-2.0)

**Architecture** (2.1B total):
1. **Vision encoder**: InternViT-based (same InternVL pipeline)
   - Same 448px tiling, pixel unshuffle, MLP projector
2. **LLM decoder**: H2O-Danube2-1.8B
   - Standard transformer, GQA, RoPE, SiLU FFN

**Reuse**: If InternVL2.5-2B is ported, this shares the vision encoder
and projector. Only the LLM backbone differs (Danube vs InternLM).

**Port priority**: Lower than InternVL2.5-2B (which scores higher and
has llama.cpp support). Port after InternVL family as a ~1 day swap of
the LLM backbone.

**GGUFs**: `/mnt/storage/gguf-models/h2ovl-mississippi-2b-{f16,q8_0,q4_k}.gguf`

**Effort**: Low (1-2 days) if InternVL2.5-2B is done first.
