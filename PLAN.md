# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

> Completed milestones live in `HISTORY.md`; technical deep-dives in
> `LEARNINGS.md`. This file tracks the current architecture and what is
> still **pending**.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.11)

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
    │              ├─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │              │     Token embeddings + RoPE
    │              │     N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │              │     Last-token / mean pooling + L2 normalize
    │              │
    │              └─► LFM2 path (LFM2.5, lfm2_embed.cpp)
    │                    RMSNorm + GQA, 350M, BOS-only tokenization
    │                    → dense / ColBERT multi-vector output
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
    ├─► OCR   ──► Qwen2.5-VL / Qwen2-VL (qwen2vl_ocr.cpp)
    │               VLM doc OCR; german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets
    │
    ├─► Layout ─► RT-DETRv2 docling-heron (layout_detect.cpp)
    │               ResNet-50 + deformable xattn, 17 document classes
    │
    ├─► OCR   ──► PARSeq scene text recognition (parseq_ocr.cpp)
    │               ViT + Transformer, 24M, 94-char ASCII, Apache-2.0
    │
    ├─► OCR   ──► InternVL2 (internvl2_ocr.cpp)
    │               InternViT + InternLM2.5 VLM, 1B/2B, MIT (+ H2OVL)
    │
    ├─► OCR   ──► GLM-OCR (glm_ocr.cpp)
    │               CogVLM2 + GLM-4, 0.9B, 8 languages, MIT
    │
    ├─► OCR   ──► GOT-OCR2 (got_ocr.cpp)
    │               SAM ViT-B + Qwen2-0.5B, document+math+table, Apache-2.0
    │
    ├─► OCR   ──► LightOnOCR-2-1B (lightonocr.cpp)
    │               Pixtral ViT + Qwen3, 1B, OCR Arena #2, Apache-2.0
    │
    ├─► OCR   ──► DeepSeek-OCR-2 (deepseek_ocr2.cpp)
    │               SAM ViT + Qwen2 + MoE decoder, 3.4B, multilingual
    │
    ├─► OCR   ──► Granite Vision 3.3-2B (granite_vision_ocr.cpp)
    │               SigLIP2 + Granite-3.1-2B, OCRBench 852, Apache-2.0
    │
    ├─► OCR   ──► Tesseract LSTM (tesseract_lstm.cpp)
    │               DBNet detection + per-line LSTM, 126 languages
    │
    ├─► NER   ──► BERT/XLM-R token classification (bert_ner.cpp)
    │               Fixed-label NER: PER/LOC/ORG/MISC, auto-detected
    │
    ├─► NER   ──► GLiNER zero-shot (gliner_ner.cpp)
    │               LFM2.5/DeBERTa-v3 + BiLSTM + span matching
    │
    ├─► KIE   ──► OCR + NER pipeline (kie_pipeline.cpp)
    │               Phase 1: OCR→NER. Phase 2: LiLT layout-aware
    │
    ├─► KIE   ──► LiLT layout transformer (lilt_kie.cpp)
    │               Dual-stream RoBERTa + BiACM, 130M, FUNSD, MIT
    │
    ├─► LID   ──► Text language identification (crisp_lid)
    │               CLD3 / GlotLID, Tesseract auto-select
    │
    ├─► Table ──► Rule-based table structure (table_parse.cpp)
    │               Line detection + grid + cell OCR → HTML
    │
    ├─► OCR   ──► PaddleOCR-VL (qwen2vl_ocr.cpp) — DONE
    │               NaViT ViT + ERNIE-4.5-0.3B, 109 langs, Apache-2.0
    │               OmniDocBench SOTA 96.3% (1.6) / 0.9B variant
    │
    │   ── PLANNED ──
    │
    └─► OCR   ──► SmolDocling (256M, Apache-2.0) — DONE: SigLIP + SmolLM2, DocTags
                    Idefics3/SmolVLM, IBM Research, DocTags output (tiny, EN-only)
```

(Evaluated and **rejected** for licensing: dots.ocr — supplemental PRC
agreement (rednote/Xiaohongshu), not pure MIT; MinerU2.5-Pro — commercial
thresholds + gated HF; Hunyuan-OCR — custom Tencent license, excludes
EU/UK/South Korea. See the next-gen table below.)

## Supported architectures (v0.11)

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
| LFM2 (bidirectional) | GPT-2 BPE | Pre-norm RMSNorm, GQA, RoPE, BOS-only | LFM2.5-Embedding-350M, LFM2.5-ColBERT |
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
| Qwen2.5-VL / Qwen2-VL / Qwen3-VL | tiktoken | ViT-32L + spatial merger + Qwen LLM; runtime ne-fix for transposed-weight GGUFs | german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets, PaddleOCR-VL |
| InternVL2 | tiktoken | InternViT + InternLM2.5 LLM | internvl2-1b/2b, H2OVL |
| GLM-OCR | BPE | CogVLM2 + GLM-4 decoder | glm-edge-ocr (0.9B) |
| GOT-OCR2 | BPE | SAM ViT-B + Qwen2-0.5B | got-ocr2 (0.7B) |
| LightOnOCR | tiktoken | Pixtral ViT + Qwen3 decoder | lightonocr-2-1b (1B) |
| DeepSeek-OCR-2 | tiktoken | SAM ViT + Qwen2 + MoE decoder | deepseek-ocr2 (3.4B) |
| Granite Vision | tiktoken/BPE | SigLIP2 ViT + Granite-3.1 LLM | granite-vision-3.3-2b |
| PARSeq | — | ViT + AR/NAR Transformer | parseq (24M, 94-char) |
| Tesseract LSTM | — | DBNet det + LSTM line rec | 126 languages |
| LiLT | RoBERTa BPE | RoBERTa + layout transformer + BiACM | lilt-funsd (130M) |
| BERT NER | WordPiece/SP | BERT/XLM-R + Linear classifier | bert-ner, xlmr-ner-hrl |
| Table parser | — | Rule-based morphology + grid detection | table_parse (no model) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |
| crisp_punc | CrispASR/crisp_punc/ | shared library (FireRedPunc + PCS) |
| crisp_lid | CrispASR/crisp_lid/ | shared library (CLD3 + GlotLID) |
| crisp_truecase | CrispASR/crisp_truecase/ | shared library (stat + CRF + BiLSTM) |

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
│   ├── crispembed.{h,cpp}      C API + encoder graph + OCR-model dispatch
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── lfm2_embed.cpp          LFM2.5 dense + ColBERT multi-vector
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr / bttr_ocr / posformer_ocr / mixtex_ocr / ppformulanet*  math OCR
│   ├── qwen2vl_ocr / internvl2_ocr / glm_ocr / got_ocr / lightonocr      VLM OCR
│   ├── deepseek_ocr2 / granite_vision_ocr / parseq_ocr / tesseract_lstm  OCR engines
│   ├── tokenizer*.{h,cpp}      WordPiece + SentencePiece + BPE
│   └── core/                   shared helpers (gguf_loader, bpe, mel, cpu_ops)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/                     GGUF conversion scripts
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending roadmap

### GPU + quantization audit (2026-06-16)

All inference engines are GPU-enabled (zero CPU-only gaps). Every engine uses
`ggml_backend_init_best()` and has a `<ENGINE>_FORCE_CPU=1` env override.
A/B verified on CPU — identical outputs, no regression.

**FULL GPU** — `ggml_backend_init_best()` + ggml graph compute (CUDA/Vulkan/Metal):
crispembed (BERT/XLM-R/etc.), decoder_embed (Qwen3/Gemma3), bidirlm_vision,
fireredpunc, pcs, gliner_ner, got_ocr, surya_det, tesseract_lstm, vit_embed,
clip_text_embed, cnn_embed, ocr_detect, parseq_ocr, layout_detect,
internvl2_ocr, qwen2vl_ocr, glm_ocr, math_ocr, ppformulanet_l_ocr, lilt_kie,
bert_ner.

**GPU-SAFE** — weights on GPU, scalar CPU forward pass (depthwise conv /
PixelShuffle not yet in ggml graph): hmer_ocr, bttr_ocr, posformer_ocr,
nafnet_denoise, mixtex_ocr, ppformulanet_ocr, pan_sr, tbsrn_sr, text_sr,
safmn_sr, esrgan_sr, restormer, tps_locnet, scunet_denoise, swinir_sr.

**Summary**: ~22 engines full-GPU, ~15 GPU-safe, 0 CPU-only. All engines have
`<ENGINE>_FORCE_CPU=1`; all SR/restoration models quantized to Q8_0 + Q4_K.

### OCR — next-gen models to port

| # | Model | Params | OmniDocBench | License | Architecture | Status |
|---|-------|--------|-------------|---------|-------------|--------|
| ~~1~~ | ~~dots.ocr~~ | ~~3B~~ | ~~88.4%~~ | ~~NOT pure MIT~~ | — | REJECTED: supplemental PRC license (rednote/Xiaohongshu) |
| 2 | **PaddleOCR-VL-0.9B** | 0.9B | — | Apache-2.0 | NaViT + ERNIE-4.5-0.3B | **DONE**: reuses qwen2vl_ocr engine, Q8_0/Q4_K on HF |
| 3 | **PaddleOCR-VL-1.6** | 0.9B | 96.3% SOTA | Apache-2.0 | NaViT + ERNIE-4.5-0.3B (same arch, improved training) | **DONE**: Q8_0/Q4_K on HF |
| ~~4~~ | ~~MinerU2.5-Pro~~ | ~~1.2B~~ | ~~90.7%~~ | ~~NOT pure Apache~~ | — | REJECTED: commercial thresholds, mandatory attribution, gated HF |
| 5 | **SmolDocling** | 256M | — | Apache-2.0 | Idefics3/SmolVLM, IBM Research | DONE: engine + parity cos=0.9999, HF `cstr/smoldocling-GGUF` |
| ~~6~~ | ~~Hunyuan-OCR~~ | ~~1B~~ | — | ~~Custom Tencent~~ | — | REJECTED: excludes EU/UK/South Korea |
| 7 | **Qari-OCR** | 4B | Apache-2.0 | Qwen2-VL fine-tune (Arabic only) | Pending (4B = large, Arabic-only, parity bug) |

**Remaining**: Qari-OCR (Arabic-only, 4B, parity bug). FireRed-OCR (Qwen3-VL 2B) and german-ocr-3 reuse the qwen2vl_ocr engine; runtime ne-fix handles GGUF converters that store weights in PyTorch (out, in) order.

#### OCRBench leaderboard reference (small VLMs, ≤3B)

| Rank | Model | LLM | Params | OCRBench | License | Status |
|------|-------|-----|--------|----------|---------|--------|
| 1 | Granite Vision 3.3-2B | Granite-3.1-2B | 3B | 852 | Apache-2.0 | **Ported** |
| 2 | InternVL2.5-2B* | InternLM2.5-1.8B | 2.1B | ~830 | MIT | **Ported** |
| 3 | MiniMonkey | InternLM2-1.8B | ~2B | 806 | — | Low priority |
| 4 | H2OVL-Mississippi-2B | H2O-Danube-1.8B | 2.1B | 782 | Apache-2.0 | **Ported** |
| 5 | InternVL2-1B | Qwen2-0.5B | 0.9B | 779 | MIT | **Ported** (edge) |
| 6 | InternVL2-4B | Phi-3-mini | ~4B | 776 | MIT | Low (too big) |
| 7 | H2OVL-Mississippi-0.8B | H2O-Danube3-0.5B | 0.8B | 751 | Apache-2.0 | Low (tiny) |

*InternVL2.5-2B not on the original leaderboard slice but scores higher than
InternVL2-2B (768).

### Handwritten math OCR — permissive-license models to port

Current handwritten math models (PosFormer/BTTR/HMER) are CC BY-NC-SA 3.0
(non-commercial). Best accuracy: 57% on CROHME 2014. These candidates are
all Apache-2.0 and would be a major accuracy upgrade.

| # | Model | Params | CROHME 2014 | License | Architecture | Effort | Status |
|---|-------|--------|-------------|---------|-------------|--------|--------|
| 1 | **Uni-MuMER-Qwen3-VL-2B** | 2.1B | ~82% (3B variant) | Apache-2.0 | Qwen3-VL fine-tune (multi-task: recognition + symbol counting + position) | Low — reuses existing `qwen2vl_ocr.cpp` engine, same GGUF converter | [ ] Pending |
| 2 | **Uni-MuMER-Qwen2.5-VL-3B** | 3.4B | 82.25% | Apache-2.0 | Qwen2.5-VL fine-tune | Low — same engine | [ ] Pending |
| 3 | **TexTeller 3.0** | 0.3B | unknown | Apache-2.0 | ViT + Transformer decoder, 80M training pairs, supports handwritten | Medium — needs new GGUF converter (TrOCR-like arch but not identical) | [ ] Pending |
| 4 | PP-FormulaNet-L | 181M | ~57% | Apache-2.0 | SAM-ViT + MBart | — | Already integrated (mostly printed math) |

**Recommended priority:**

1. **Uni-MuMER-Qwen3-VL-2B** (best ROI) — same Qwen3-VL base we already
   support. Convert via `convert-qwen3vl-to-gguf.py` → quantize Q4_K/Q8_0
   → add to CrispCalc catalog as "Handwritten Math (82%, Apache-2.0)".
   The fine-tune adds handwriting-specific LoRA/full weights on top of the
   base Qwen3-VL architecture — the GGUF converter should work as-is since
   the architecture key is still `qwen3vl`. Verify with CROHME 2014 test.

   Source: [github.com/BFlameSwift/Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)
   Weights: HuggingFace (released 2025-06-02), also Google Drive

2. **TexTeller 3.0** (lightweight option) — at 300M params, small enough
   for mobile. Architecture is ViT encoder + Transformer decoder (similar
   to pix2tex but trained on 80M pairs including handwritten). Would need
   a new converter or adaptation of `convert-pix2tex-to-gguf.py`. Benchmark
   on CROHME before committing to full integration.

   Source: [github.com/OleehyO/TexTeller](https://github.com/OleehyO/TexTeller)
   Weights: [huggingface.co/OleehyO/TexTeller](https://huggingface.co/OleehyO/TexTeller)

3. **Uni-MuMER-Qwen2.5-VL-3B** (fallback) — if Qwen3-VL variant has
   issues, the 3B Qwen2.5-VL variant has proven 82.25% accuracy and uses
   the same engine. Slightly larger (3.4B) but still under 4B.

**Impact**: replacing NC-licensed 57% models with Apache-2.0 82% models
eliminates the license acceptance gate in the UI AND nearly doubles
handwritten accuracy. This is the single highest-value remaining OCR task.

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| Qwen3-VL multimodal | Low | High | Reuse BidirLM-Omni scaffolding |

### DeepSeek-OCR-2 performance (remaining levers)

The pipeline is now mostly on Metal (encoder, MoE decode, SAM convs + patch
embed, LM head) — full OCR ~9 min (never completed) → ~12 s warm. Profiled
warm breakdown: load ~9 s cold / 0.8 s warm · SAM ~4.7 s · decode ~3.8 s ·
enc+proj ~1.1 s. Remaining levers, ranked by leverage:

- [x] **#1 Load-path prefetch — DONE, but not the bottleneck.** Added
  `madvise(MADV_SEQUENTIAL/WILLNEED)` to `core_gguf::load_weights` (correct
  practice, helps genuinely disk-bound cold loads on other systems). On *this*
  machine it didn't move the needle, and the diagnostic explains why: the disk
  reads 2.1 GB in **1.17 s** and a warm load is **0.8 s** — so the ~9–18 s cold
  loads are **memory-pressure / swap**, not readahead. During a run the process
  holds ~5 GB (2.1 model + 1.3 stacked experts + 0.65 embed-f32 + Metal) on a
  16 GB box, so file pages and new allocations contend and swap. → the real load
  lever is **reducing the footprint** (#3, #4), not prefetch.
- [x] **#2 Decode graph reuse (~1–1.5 s) — DONE.** Persistent T=1 decode graph
  with fixed max-KV, incremental KV-cache mask; 2× faster decode stage.
  (`fcb5b11 perf(ocr2): persistent T=1 decode graph reuse`)
- [ ] **#3 Per-row embedding dequant (~0.5 s + 655 MB).** Decode dequants the
  whole 128k×1280 embed table to F32 just to look up ~32 rows; dequant per row.
- [ ] **#4 Converter-emitted stacked experts (memory, ~0.6 s).** Emit
  `ffn_{gate,up,down}_exps [in,out,n_exp]` from the converter (needs a Kaggle
  reconvert + loader tweak) so the runtime skips `stack_moe_experts` and the
  +1.3 GB duplication → footprint 3.4 → 2.1 GB → better cache retention (helps
  #1's cold/warm swing). Primarily a memory win.
- [ ] **#5 SAM flash-attention (marginal, skip unless needed).** The SAM
  attention uses a decomposed rel-pos bias (rel_h/rel_w added to scores), which
  blocks `ggml_flash_attn_ext` unless the bias is materialized as a [T,T] mask —
  fiddly, and the win is small (~3–4 s SAM is mostly the genuine 4096-token
  global attention compute).

All deepseek perf paths are env-gated with validated CPU fallbacks
(`DS_QWEN2_SCALAR`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU`, `DS_MMAP`,
`DS_REF` parity harness, `DS_DBG` timers).

### Refactoring

- [x] **Extract shared VLM building blocks to `core/` headers** (Phase 1 done) —
  - [x] `core/cpu_ops.h` — to_f32, layernorm (raw + tensor overloads), layernorm2d,
    rmsnorm, linear (raw + tensor overloads), conv2d (with groups), gelu, gelu_erf,
    silu, softmax, hardswish, relu6, relu, mha_1q_cpu. Replaced in 6 engine files
    (surya_det, got_ocr, ppformulanet_l_ocr, ppformulanet_ocr, deepseek_ocr2,
    mixtex_ocr) — 728 lines deleted. 88 unit tests in test_core_cpu_ops.cpp.
  - [x] `core/vlm_attention.h` — RoPE (neghalf + interleaved), GQA attention with
    KV cache, SwiGLU FFN. Replaced in smoldocling + granite_vision (134 lines deleted).
    97 unit tests in test_core_vlm_attention.cpp. Commit `c730539`.
  - [ ] `core/vlm_decoder.h` — unified decode loop (deferred: only 2 scalar engines,
    premature to abstract)

---

### Optimization TODOs (June 2026 audit)

Full line-by-line code review of all ~57K lines across 60+ runtimes.
Organized by priority (P0 = highest impact, P3 = nice-to-have).

#### P0 — Critical performance wins

- [x] **SIMD in `core/cpu_ops.h`** — Added `dot_product()` with AVX2+FMA (x86-64)
  and NEON (ARM) inner loops. `linear_cpu` and `mha_1q_cpu` now use it.
  737 `vfmadd231ps` instructions emitted in libcrispembed.so. `-march=native`
  enabled via `CRISPEMBED_NATIVE` cmake option (ON by default).
  `conv2d_cpu` still scalar — requires im2col restructure (separate TODO).

- [x] **Dequantized weight caching** — Added `DequantCache` struct to
  `cpu_ops.h`: `unordered_map<void*, vector<float>>` keyed on tensor data
  pointer, dequantizes on first access, returns cached F32 thereafter.
  Migrated: smoldocling_ocr (replaced wbufs), granite_vision_ocr (replaced
  wcache). Remaining runtimes still need migration.

- [ ] **Adopt F16 ggml KV cache** — Port to: deepseek_ocr2 (F32 std::vector).
  pix2struct: **DONE** (`088d359`) — F32 std::vector KV cache + cross-attn pre-compute.
  lightonocr: **DONE** (`485cb97`, branch `lighton-perf`) — 2.09x total speedup.
  granite_vision_ocr: **DONE** (`66b8de2`).
  smoldocling_ocr: **DONE** (`bc329e4`, branch `feat/smoldocling-kvcache-prefill`).
  qwen2vl_ocr: **DONE** — already had F16 kvc; fixed CPU round-trip in seeding
  (`48948a6`, branch `feat/qwen2vl-kvcache`).

- [x] **Move granite_vision_ocr vision encoder to ggml graphs** — DONE
  (feat/granite-vision-ggml-graph). SigLIP ViT (27 layers, T=729,
  D=1152, n_heads=16) as a single ggml graph with Metal backend.

- [x] **granite_vision projector + LLM decoder → ggml graphs** — DONE
  (`66b8de2`). `gv_run_projector_graph` (2-layer MLP on Metal) and
  `gv_run_llm_body` (40-layer Granite-3.1: RMSNorm + GQA with
  ggml_rope_ext NEOX + F16 KV cache + ggml_flash_attn_ext + SwiGLU FFN,
  scaled residuals). LM head stays CPU (linear_cpu, SIMD). Scalar fallback
  preserved in `gv_llm_decode_step` (used by dump_llm parity).
  - **Crash fix (`feat/granite-vision-ne-fix`)**: the projector + LLM graphs
    aborted on `GGML_ASSERT(ggml_can_mul_mat)` — the converter stores 2D
    weights in PyTorch `[out,in]` order, so non-square weights need a
    `ggml_reshape_2d(w, ne[1], ne[0])` before `ggml_mul_mat` (the vision FFN
    already did this). Applied to projector linear_1 and LLM k/v/gate/up/down.
  - **ggml LLM decode is NOT yet validated**: it runs but produces wrong
    tokens (decode runs away to max_tokens). Gated behind
    `CRISPEMBED_GRANITE_LLM_GRAPH=1`; default is the diff-validated scalar
    decode (`gv_llm_decode_step`, crispembed-diff cos=1.0). Vision + projector
    ggml graphs are correct and stay on Metal.
  - **Memory**: the scalar fallback's DequantCache materializes ~9 GB of F32
    weights (swaps on a 16 GB machine). Q4_K vec_dot would keep it bounded
    (~2 GB); see `tools/dump_granite_llm_reference.py` for the parity harness.

- [x] **Batched prefill for granite** — DONE (`66b8de2`). All prompt tokens
  (vision + text, 759 total) assembled into one buffer and passed to
  `gv_run_llm_body` as a single T=759 call. Replaces 759 serial decode
  steps with 1 batched ggml graph invocation.

- [x] **F16 KV cache + batched prefill for smoldocling** — DONE (`bc329e4`,
  branch `feat/smoldocling-kvcache-prefill`). SmolLM2-135M (30L, 576d, GQA
  9/3). Batches entire prompt in one `sd_run_llm_body` call. Scalar fallback
  via `sd_llm_decode_step` preserved. Uses CPU backend with Accelerate BLAS.

- [x] **Eliminate CPU round-trips in qwen2vl KV seeding** — DONE (`48948a6`,
  branch `feat/qwen2vl-kvcache`). Moved `alloc_kv_cache` before prefill;
  `run_llm_forward(populate_kvc=true)` writes K/V directly into kvc via
  `ggml_cpy` in the prefill graph (F32→F16 in graph, no CPU bounce).

- [x] **Move pix2struct to ggml graphs + add KV cache** — DONE (`088d359`,
  `51a3008`). Encoder as single ggml graph, decoder with incremental self-attn
  KV cache + pre-computed cross-attn K/V via ggml graph. DequantCache for all
  weight access. Per-step heap allocations hoisted to context scratch buffers.
  Parity: encoder cos=0.9999, decoder cos=1.0000.

- [x] **scunet per-pixel heap allocations** — Hoisted `std::vector<float>` pix,
  pix_out, pix_norm, h allocations outside the spatial loops. Also cached LN2
  weights outside the MLP per-pixel loop (was re-dequantizing 65536 times).
  Eliminates 100K+ heap allocs per swin block for 256×256 images.

#### P1 — High-impact targeted improvements

- [ ] **Flash attention everywhere** — use `ggml_flash_attn_ext` in:
  - `decoder_embed.cpp`: **DONE** (`29d8a08`) — both single-text and batch paths
  - `bidirlm_vision.cpp`: **DONE** (`fd8cd09`) — F16 mask, halves mask memory
  - `lilt_kie.cpp`: SKIPPED — BiACM adds text+layout scores before softmax,
    incompatible with fused flash_attn kernel
  - `qwen2vl_ocr.cpp`: already uses flash_attn
  - `deepseek_ocr2.cpp` all attention paths

- [ ] **Move remaining scalar encoders to ggml graphs**:
  - `deepseek_ocr2` Qwen2 encoder (lines 777-931): 24-layer bidirectional
    transformer, all scalar. O(T^2 * heads * head_dim) attention loops.
  - `bttr_ocr` / `posformer_ocr` / `hmer_ocr` DenseNet encoders: 7-nested-loop
    scalar convolutions dominate runtime.
  - `mixtex_ocr` Swin encoder: 12500-token window attention, scalar.
  - `ppformulanet_ocr` HGNetv2 CNN: 57M-param CNN at 384x384, scalar `conv2d_cpu`.

- [x] **Patch embedding conv → ggml matmul** — Most VLM runtimes now use ggml
  graph (internvl2, granite, smoldocling, qwen2vl) or im2col+matmul (got,
  lightonocr, pix2struct). Remaining: glm, deepseek (minor, scalar fallback).

- [x] **Pre-compute RoPE frequency tables** — Added `RoPEFreqTable` struct to
  `vlm_attention.h` with `precompute(head_dim, theta)` and `apply()` methods.
  Eliminates `powf` per-element. Migrated: smoldocling_ocr (NEGHALF),
  granite_vision_ocr (NEGHALF). Remaining `core_vlm` users still on `apply_rope()`.

- [x] **Batch linear → GEMM in SR/restoration attention** — DONE. dat_sr
  (`a71c123`), swinir_sr (`dcf6556`), hat_sr (`b199741`), scunet (`52250ef`),
  mixtex (`816a88a`): replaced per-token scalar linear with
  linear_batch_cpu (SIMD), SIMD dot_product in attention.

- [ ] **Sequential region recognition → batched** — `ocr_pipeline.cpp` (line 112)
  and `table_parse.cpp` (line 337) recognize each detected text region one at a
  time. Batch crops into a single encoder pass for PARSeq/TrOCR.

- [x] **Eliminate redundant image loading in orchestrator** — Pre-load image once
  per stage in the accept-gate loop, pass pixel buffer to all 9 VLM engines.
  Eliminates N-1 redundant JPEG/PNG decodes per multi-stage run.

- [x] **LSTM gate SIMD** — `tesseract_lstm.cpp` inner dot-product loops in both
  `lstm_forward` and `summ_lstm_forward` now use `core_cpu::dot_product()`.
  AVX2+FMA accelerated on x86-64, NEON on ARM.

- [x] **Sliding-window min/max pool** — Replaced O(K) per-pixel brute-force in
  `scan_cleanup.cpp` with monotonic deque sliding window — O(1) amortized per
  pixel. For K=51 this is ~50x fewer comparisons.

- [x] **Weight dequant caching in SR runtimes** — Migrated 7 Pattern-A runtimes
  (hat_sr, swinir_sr, pan_sr, text_sr, nafnet_denoise, restormer, tbsrn_sr)
  from `wbufs` append to `core_cpu::DequantCache`. 4 Pattern-B runtimes
  (instructir, adair, esrgan, safmn) use inline dequant — separate TODO.

- [x] **Migrate duplicated helpers to `core/cpu_ops.h`** — bttr_ocr, hmer_ocr,
  posformer_ocr: replaced duplicated conv2d/relu/layernorm/linear with
  `core_cpu` shared versions (SIMD-accelerated). Replaced per-context
  `dequant_cache` map with `core_cpu::DequantCache`. Kept unique helpers
  (maxpool, avgpool, apply_bn) as-is.

- [ ] **deepseek_ocr2: single multi-layer LLM graph** — currently builds 12
  separate ggml graphs per decode token (line 1288-1295). A single graph
  covering all attention layers would reduce graph construction cost by 12x.

- [x] **glm_ocr / got_ocr: scalar downsample/merger → ggml** — DONE. glm `host_matmul`
  (lines 493-502) and got neck (lines 699-773) use scalar CPU for Conv+matmul
  projectors. Should be ggml graph ops.

- [x] **gliner_ner BiLSTM SIMD** — Gate computation now uses
  `core_cpu::dot_product()` (AVX2+FMA/NEON). ~3M MACs per timestep accelerated.

- [ ] **LiteMLA graph implementation** — `surya_det.cpp` line 792 has TODO:
  `g_litemla` returns nullptr, the graph-accelerated LiteMLA is stubbed out.
  Currently falls back to CPU-scalar attention.

- [x] **Add tiling to SR runtimes without it** — ALL DONE. Hann-window overlap
  tiling added to esrgan_sr, safmn_sr, nafnet_denoise, scunet_denoise,
  instructir, adair. All env-configurable. Small images bypass tiling.

#### P2 — Moderate improvements

- [x] **LFM2 sched + T-bucketing** — migrated `lfm2_embed` from raw
  `ggml_gallocr` to `ggml_backend_sched` with sequence-length bucketing (same
  pattern as encoder path in `crispembed.cpp`). Eliminates per-call allocation
  overhead for same-bucket inputs (~2ms → ~0.7ms graph+alloc). Compute
  dominates at ~700ms for the 350M Q8_0 model. Architecturally aligns LFM2
  with the rest of the codebase and enables future GPU dispatch.

- [ ] **Graph caching** — most runtimes still rebuild the ggml graph on every
  call. Caching graph structure and only updating input data would eliminate
  per-call graph construction + allocation overhead.

- [x] **`ggml_gallocr` reuse** — moved gallocr from per-call to per-context
  for 7 engines: vit_embed, clip_text_embed, parseq_ocr, cnn_embed,
  ocr_detect, surya_det, layout_detect. Eliminates ~1-3ms malloc/free
  overhead per call; significant for small/fast models (DBNet 12M, PARSeq 24M).

- [x] **Native GQA in flash_attn (all VLMs)** — DONE. Removed `ggml_repeat`
  KV head expansion before `flash_attn_ext` in internvl2 (`7cffe56`),
  lightonocr, got_ocr, glm_ocr (`fbae7ba`). flash_attn handles GQA via
  broadcast factors (rk2 = neq2/nek2). -76 lines total.

- [ ] **internvl2: batch vision tiles** — `encode_vision()` (line 1200-1226)
  processes tiles one at a time with separate graph allocations per tile.
  Batch multiple tiles into one graph.

- [ ] **Eliminate redundant CHW↔HWC layout conversions** — `dat_sr.cpp`
  converts at every block boundary. Low priority: layouts are comments/
  documentation, actual transposes are minimal after SIMD linear_batch refactor.

- [x] **Pre-compute attention masks and position biases** — Already resolved:
  swinir_sr masks loaded from GGUF model file (stored as tensors, cached via
  DequantCache). hat_sr has no runtime mask computation. dat_sr position bias
  depends on spatial dims which change per tile — not precomputable.

- [x] **Fuse BatchNorm into conv weights at model load** — TBSRN: fused 11
  conv+BN pairs (2 per SRB × 5 + 1 final) at init. `dat_sr` still pending.

- [x] **qwen2vl: token embedding via direct read** — DONE. Embed is now part of
  the main LLM graph (ggml_get_rows runs on GPU). lightonocr uses direct
  tensor read (embed_tokens_cpu).

- [ ] **lightonocr / deepseek: decode graph reuse** — graph structure is
  identical across decode steps. Build once, update input data only.

- [x] **qwen2vl: F32 causal mask → F16** — already F16 (GGML_TYPE_F16)
  (half the memory).

- [ ] **gliner_ner: DeBERTa relative position expansion** — creates [H, T*T]
  F32 tensor on CPU every call. T=200 → 117MB. Cache or compute incrementally.

- [x] **Pre-compute 2D positional encoding** — TBSRN: cached at init (fixed
  16×64 dims, reused across 5 SRB blocks). BTTR/PosFormer: cached for
  last-used (h, w) — skips ~327K sinf/cosf evals on repeated calls.

- [ ] **mel.cpp: OpenMP on STFT loop** — each frame's FFT is independent
  (line 73-84). `#pragma omp parallel for` on the `t` loop.

- [x] **mel.cpp: SIMD for mel projection** — DONE. Float MelsFreqs layout uses
  `core_cpu::dot_product()` (AVX2+FMA/NEON). Double-precision path kept scalar.

- [x] **gguf_loader: `madvise(MADV_SEQUENTIAL)`** — already done (line 244).
  Also has MADV_WILLNEED (line 247).

- [x] **gguf_loader: `std::unordered_map` for tensor lookup** — DONE
  (`0777f30`). Replaced std::map with std::unordered_map in WeightLoad
  and all 14 model files. O(1) avg lookups.

- [x] **instructir: SCA weight dequant inside per-channel loop** — DONE
  (`06b3190`). Hoisted sca_w/sca_b dequant outside per-channel loop.

- [x] **Otsu threshold: extract shared utility** — Added
  `core_cpu::otsu_threshold()` to `cpu_ops.h`. Replaced duplicated
  implementations in cc_detect, table_parse, classical_preproc, dewarp.
  scan_cleanup float variant kept separate (different input type).

- [ ] **OpenMP in pixel-level ops** — `image_preprocess`, `dewarp`,
  `scan_cleanup`, `face_align` all accept `n_threads` but run single-threaded
  pixel loops.

- [x] **pcs: cache FC weights at load** — DONE. All 17 FC head tensors cached
  in `fc_cache` struct at init. No per-call `ggml_backend_tensor_get`.

- [x] **restormer: dead `rst_gdfn()` stub** — DONE (`06b3190`). Removed.
- [x] **restormer: `rst_layernorm_bf` computes variance twice** — DONE
  (`06b3190`). Removed dead first-pass sum-of-squares.

#### P3 — Nice-to-have / minor

- [x] **bpe.h: priority queue for BPE merges** — DONE (`eae73de`).
  Linked list + min-heap, O(N log N). Both bpe.h and tokenizer_bpe.cpp.
- [x] **tokenizer_bpe.cpp: same O(N^2) merge issue** — DONE (`eae73de`).

- [ ] **tokenizer.cpp: trie for WordPiece** — currently linear scan for longest
  match. Trie would be O(len).

- [x] **cpu_ops.h: `layernorm2d_cpu` cache-hostile access** — already fixed
  (gather-norm-scatter with contiguous buf). Now uses thread-local buf
  to eliminate per-call heap alloc (`113202d`).

- [x] **vlm_attention.h: pre-allocate scores vector outside head loop** —
  DONE (`9172ba1`). Thread-local buffer replaces per-head std::vector.
- [x] **vlm_attention.h: pre-allocate `swiglu_ffn` intermediates** — DONE
  (`9172ba1`). Thread-local buffers for gate/up vectors.
- [x] **cpu_ops.h mha_1q_cpu alloc elimination** — DONE (`9172ba1`).
  Write directly to output, thread-local scores buffer, optional external buf.
- [x] **parseq_ocr decoder alloc hoist** — DONE (`38177e2`). ~18 per-step
  vectors moved to pre-allocated dec_scratch struct.

- [ ] **Nearest-neighbor → bilinear resize** — 4 of 7 math OCR runtimes
  (math_ocr, mixtex, ppformulanet, ppformulanet_l) and several others
  (got_ocr, surya_det, parseq_ocr) use nearest-neighbor. Quality issue.

- [x] **bttr beam search: top-K selection** — DONE (`113202d`).
  Replaced std::sort with std::partial_sort for O(N·log(K)) top-K.

- [x] **bttr_ocr decoder alloc hoist + SIMD attention** — DONE (`4febeb6`).
  Pre-allocated scratch buffers, core_cpu::dot_product in MHA, pre-alloc KV cache.
- [x] **posformer_ocr decoder alloc hoist + SIMD attention** — DONE (`080c75e`).
  Same pattern as bttr, including ARM-specific buffers (raw_scores, cov_bias).
- [x] **hmer_ocr GRU decoder alloc hoist + SIMD** — DONE (`98e6daf`).
  Pre-allocated scratch for GRU/Bahdanau attention, SIMD v() dot product,
  SIMD enc_ua precomputation via core_cpu::linear_cpu.
- [x] **math_ocr SIMD linear + scalar decoder allocs** — DONE (`ac3a362`).
  Replaced scalar linear_cpu with core_cpu::linear_cpu (SIMD), SIMD dot_product
  in mha_1q, pre-allocated scalar decoder scratch.

- [ ] **Add beam search to math OCR runtimes** — only bttr_ocr has it.
  mixtex, math_ocr, hmer, posformer, ppformulanet, ppformulanet_l are
  greedy-only. Beam width=3 typically helps math OCR accuracy.

- [ ] **morph_fast: decomposed dilation** — horizontal dilation iterates
  over all shift values (O(hsize * wpl * h)). Leptonica-style decomposed
  2-pass (power-of-2) would be much faster for large kernels.

- [x] **pdf_info: mmap instead of full file read** — DONE (`5f027aa`).
  Memory-mapped on POSIX with MADV_SEQUENTIAL, fread fallback on Windows.

- [ ] **tps_warp: coarse grid + bilinear interpolation** — evaluates all N
  control points per output pixel (O(W*H*N) with sqrt+log). Pre-compute
  coarse displacement grid, interpolate at render time.

- [x] **Debug fprintf gating (layout_detect)** — DONE (`614132e`). ~30
  unconditional printfs converted to LDBG() macro (gated behind LAYOUT_DEBUG).
  Remaining files (surya_det, ocr_detect) still have some unconditional prints.

- [ ] **hmer coverage conv per step** — conv2d(256, 256, 3x3) per decoder
  step is the attention mechanism. Expensive but architecturally required.

- [x] **ppformulanet_l: ggml meta buffer reuse across layers** — DONE
  (`b7bc237`). Hoisted 8MB meta_buf before 12-layer loop.

- [x] **math_ocr: global dequant cache → per-context** — already done.
  Uses `core_cpu::DequantCache dcache` per-context (line 93).

- [x] **Remove dead scalar fallback encoder in ppformulanet_l** — DONE
  (`c7bd92c`). Removed 370 lines of unused scalar encoder code.

---

## Per-Backend Performance Optimization (Q4_K, A/B benchmarked)

Systematic per-backend optimization pass. Every change is A/B benchmarked
using `CRISPEMBED_<MODULE>_BENCH=1` on Q4_K models. Constraint: 8GB VPS,
single-threaded, must not OOM.

### lightonocr (Pixtral ViT 24L + Qwen3 28L, 1B) — 2.09x done

**Baseline** (400×100 image, 240 patches, q4_k, CPU 4-thread):
  vision=64.5s, projection=0.2s, prefill=36.4s, decode(6tok)=123.6s, total=245.2s

**Done:**
- [x] Flash attention default — 1.5x vision, 1.4x prefill
- [x] Direct embed lookup (no ggml graph per token) — eliminates per-step overhead
- [x] F16 ggml KV cache — persistent F16 backend tensors, ggml_view + ggml_cpy,
      zero CPU↔backend transfer per step. Halves KV memory.

**After all optimizations**: vision=20.6s, prefill=14.0s, decode=69.5s, total=117.5s (**2.09x**)

- [x] Patch embedding → ggml matmul (im2col + mul_mat, scalar fallback gated)

**Remaining:**
- [ ] **Decode graph reuse** — graph still rebuilt per step (tensor shapes change
      with Lk). Need fixed-max-KV graph with ggml_view variable-length reads.

### qwen2vl — DONE (already optimized)
  F16 KV cache, flash attn, ggml patch embed, direct embed lookup, F16 mask — all done.

### deepseek_ocr2 — PENDING (needs q4_k model, 3.4B too large for 8GB VPS f16-only)

### got_ocr (SAM ViT-B + Qwen2-0.5B, 0.7B) — DONE
- [x] Patch embedding → ggml matmul (same im2col pattern, scalar fallback gated)
- [x] Neck+downsample+projector → ggml graph (conv2d_direct + LN2d via permute+norm + mul_mat)
  Gated: CRISPEMBED_GOT_OCR_SCALAR_NECK=1 / CRISPEMBED_GOT_OCR_SCALAR_PATCH=1

### glm_ocr (CogVLM2 + GLM-4, 0.9B) — DONE
- [x] Downsample + merger → ggml graph (conv2d_direct + batched SwiGLU + LayerNorm)
  Gated: CRISPEMBED_GLM_OCR_SCALAR_MERGER=1. Merger: 383ms on q4_k.
### granite_vision — PENDING (LLM decoder ggml graph not yet validated)

### smoldocling (SigLIP + SmolLM2, 256M) — IN PROGRESS
- [x] F16 KV cache + batched prefill (done earlier, `bc329e4`)
- [x] Patch embedding → ggml matmul (im2col + mul_mat, F16 bias cast)
  Gated: CRISPEMBED_SMOLDOCLING_SCALAR_PATCH=1
- [x] LLM decoder → ggml graphs — DONE. Was already implemented but blocked by
  F16 norm weight type mismatch on Q4_K models. Fixed with ggml_cast (`91b1f89`).
  Tested: prefill=2.3s, decode=62s (128 steps).
### internvl2 — DONE (already optimized)
  F16 KV cache, flash attn, ggml patch embed, ggml vision graph — all done.
  Remaining: native GQA in flash_attn (skip ggml_repeat), batch vision tiles.
### SR/denoise — DONE (SIMD + batched linear)
- [x] dat_sr: `linear_batch_cpu` + SIMD `linear_cpu`, batch QKV/proj/FFN (`a71c123`)
- [x] swinir_sr: batch per-token linear + SIMD dot product (`dcf6556`)
- [x] hat_sr: SIMD dot product in SA/OCA + SIMD channel attention (`b199741`)
- [x] scunet_denoise: batch QKV/proj via `linear_batch_cpu` + SIMD dot (`52250ef`)
- [x] mixtex_ocr: SIMD dot product in Swin attention (`816a88a`)
- [x] instructir: SCA dequant hoist (`06b3190`)
- [x] restormer: dead code removal + variance fix (`06b3190`)

### Embedding — DONE (flash attention)
- [x] decoder_embed: `ggml_flash_attn_ext` in single-text + batch paths (`29d8a08`)

---

## Implementation blueprints

Detailed specs for pending roadmap items. Each blueprint is self-contained
so a fresh agent can implement it independently. (Blueprints for completed
work have been moved to `HISTORY.md`.)

### Blueprint: KV cache for prefix-shared decoder batches — DONE

Implemented in `decoder_encode_tokens_batch()` (decoder_embed.cpp:1188).
- `detect_common_prefix()` finds longest shared prefix across batch
- Layout: `[prefix_0..P-1 | suf0_pad | suf1_pad | ...]` — prefix appears once
- Custom attention mask: each suffix attends causally to shared prefix + own suffix
- Saves `(B-1)*P` tokens of redundant compute (~40% for Jina v5 batches)
- Minimum prefix threshold: 4 tokens (not worth mask complexity for shorter)

---

### Blueprint: Batched decoder improvements (F16 mask + Gemma3 NaN fix) — DONE

Both fixes are implemented in `decoder_embed.cpp`:
- **F16 attention mask**: `ggml_new_tensor_2d(gctx, GGML_TYPE_F16, T_total, T_total)` (line 1386). 2x memory reduction.
- **Gemma3 NaN fix**: `ggml_clamp(gctx, x, -1000.0f, 1000.0f)` before `(1+w)*x` (line 668). Prevents overflow in CrispEmbed-native GGUFs with `gemma_norm=true`.

---

### Blueprint: WASM build target — DONE

Implemented via `build-wasm.sh` (Math OCR) and `build-embed-wasm.sh`
(text embeddings). CI workflows in `.github/workflows/build-wasm.yml`
and `build-wasm-embed.yml`. HuggingFace Space demo at `hf-space/`.
README mentions: "Math OCR compiles to WebAssembly (1 MB) via build-wasm.sh.
Runs entirely client-side — no server, no API key."
