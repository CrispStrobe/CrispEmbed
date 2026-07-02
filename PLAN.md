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
    ├─► Math  ──► Uni-MuMER-Qwen3-VL-2B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 2.1B, Apache-2.0, 82% CROHME
    │
    ├─► Math  ──► Uni-MuMER-Qwen2.5-VL-3B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 3.4B, Apache-2.0, 82.25% CROHME
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

### scan_cleanup — features to port from unpaper (2026-07)

`scan_cleanup` (classical, no model — BiblioForge's `--scan-cleanup`) currently has
deskew, rectangular border-crop, background-whitening (morphological closing, fixed
`6fdd1b5`), and Otsu/Sauvola binarize. Benchmarked vs `unpaper 7.0.0` with the
ground-truth OCR-CER harness `tools/scan_cleanup_bench.py` (clean page → degrade →
run each tool → CER vs known text + contact sheets). Finding: CrispEmbed already
wins on uneven lighting (unpaper has no illumination correction) and never fails
destructively, whereas unpaper's default mask/blackfilter **blanked a whole page**
on an out-of-domain dark-shadow input — but unpaper has real features we lack.
Port them, each **behind an OCR-CER A/B with saved output images** for visual
review (run `scan_cleanup_bench.py` before/after every step; never merge a step
that regresses CER or looks worse by eye).

**LICENSING (hard constraint): unpaper is GPL-2.0-or-later; CrispEmbed is MIT.**
Do NOT copy any unpaper source — GPL copyleft is incompatible with MIT. These are
**clean-room reimplementations** of standard, long-predating classical
image-processing techniques (median/impulse despeckle, morphological open/close,
Hough deskew, connected-component blob removal, projection-profile page splitting)
written from the general concept only. Algorithms are not copyrightable; only
code expression is. Never read/paste unpaper's code into ours.

- [x] **1. noisefilter / despeckle — DONE** (clean-room decision-based 3x3 median:
  replace a pixel by its local median only when it differs by > `despeckle_thresh`,
  so isolated specks lift to paper while text strokes — dark median — are kept).
  A/B (tools/scan_cleanup_bench.py): heavy-speckle CER **0.580 → 0.032** (was 0.436
  before despeckle; unpaper stuck at 0.580), mild speckle 0.042 → 0.011; uneven/
  shadow unaffected. Visual: speckle gone, text intact. `p.despeckle` (default on),
  `p.despeckle_thresh` (0.25). Runs first (before deskew/whiten). Salt-pepper is a
  case unpaper's cluster noisefilter does NOT handle.
- [x] **2. blackfilter — DONE** (clean-room 8-connected-component labelling of
  very-dark pixels; whiten a component only if LARGE *and* SOLID (bbox fill ≥ 0.5)
  so text strokes/glyphs are kept; **hard guard: never clear > 40% of the page** —
  the exact case where unpaper blanked a whole page). A/B: clears the dark blob +
  edge shadow visually with **no CER regression** (shadow 0.071, hspeckle 0.030,
  uneven 0.007 — no misfire on text), and the guard holds where unpaper produced
  CER 1.000. `p.blackfilter` (default on), `p.blackfilter_thresh` (0.20).
- [x] **3. deskew sheet-background fill — DONE (already correct; no functional
  change).** Investigated: `scan_cleanup_rotate` already fills corners with pure
  white (1.0), and the earlier gray-wedge symptom was actually caused by the
  pre-fix whitening OPENING, resolved by the closing fix (6fdd1b5). A/B image
  inspection confirms deskew now yields clean white corners (skew CER 1.000 →
  0.030). Tried the "detected paper-gray fill" idea — it REGRESSED (visible gray
  wedges at corner boundaries after whitening; CER unchanged so only the image
  caught it), so reverted. Pure-white fill is the robust choice (1.0 = max, can't
  brighten → stays uniform through whitening); documented in-code.
- [x] **4. grayfilter / blurfilter — NOT NEEDED (subsumed by our whitening).**
  Evidence: the harness's faint-haze degradation (soft light-gray veil) is fully
  cleaned by the existing morphological-CLOSING whitening — output background is
  pristine white and OCR CER is 0.010 (degraded 0.008, i.e. faint haze doesn't
  even hurt OCR). unpaper needs a separate grayfilter because it has no
  illumination-correction; our closing-whitening already does illumination
  flattening, which subsumes grayfilter AND blurfilter. Implementing a grayfilter
  on top would add code + anti-aliased-text-erosion risk for ~0 benefit, so it is
  deliberately skipped. (Dark *stains* — near-black, not faint gray — are a
  separate case handled by blackfilter/despeckle, not grayfilter.)
- [x] **5. 2-up page splitting — DONE** (clean-room projection-profile gutter
  detector: `scan_cleanup_detect_page_split()` + CLI `--detect-split FILE` →
  JSON `{pages,split_x}`). Wide aspect + emptiest central column that is a true
  near-empty gutter with substantial text on BOTH sides. A/B: a two-page spread
  OCRs at CER **1.002** whole (tesseract reads across both pages) but **0.000 /
  0.001** per half after splitting at the detected gutter (split_x exact). No
  false-split on portrait single pages or a wide gutterless image (astronaut).
  Biggest win for book scans. BiblioForge: call `--detect-split`, crop at
  `split_x`, then `--cleanup-only` each half.
- [x] **6. content-mask detection — DONE** (clean-room row/column dark-pixel
  projection profile; `scan_cleanup_content_bbox()` + CLI `--detect-content FILE`
  → JSON `{x0,y0,x1,y1}`). A/B: on a page with large blank margins the detected
  bbox is exact (60,38,668,1042 vs true 60,40,666,1040 incl. 2px pad); and it is
  NOT OCR-neutral as feared — the padded page FAILS OCR (CER **1.000**, tesseract
  layout analysis confused by the margins) while cropping to the content bbox gives
  CER **0.001**. Exposed as a detection helper (not a forced pipeline crop, to
  avoid ever cutting text on a mis-detect); the caller crops/centers.

All six evaluated. Implemented: despeckle (1), blackfilter (2), page-split (5),
content-bbox (6). Verified-already-covered / deliberately-skipped with evidence:
deskew fill (3, already correct), grayfilter (4, subsumed by our whitening).

Harness: `tools/scan_cleanup_bench.py --image clean.png --bin build/crispembed`.

### llama.cpp parity, convergence & A/B plan (2026-07)

A full audit of which CrispEmbed architectures llama.cpp now supports (upstream
`ggml-org/llama.cpp` @ ~`4fc4ec5`, July 2026), how it implements them, and what
we should borrow. Deep technical notes live in `LEARNINGS.md → "llama.cpp
implementation reference"`. **Rule for every convergence step below: land it
behind an A/B test that measures BOTH speed and quality (see "A/B protocol" at
the end of this section). No step merges without a before/after on both axes,
on CPU and Metal.**

#### Support matrix (CrispEmbed arch → llama.cpp)

Text-embedding encoders:

| CrispEmbed | in llama.cpp | llama.cpp arch id | note |
|---|---|---|---|
| BERT | ✅ | `bert` | one shared `bert.cpp` graph, config from GGUF |
| XLM-RoBERTa | ✅ | `bert` | RoBERTa/XLM-R fold into `bert`; pos-offset + SPM handled |
| NomicBERT | ✅ | `nomic-bert` | SwiGLU + RoPE |
| NomicBERT-MoE | ✅ | `nomic-bert-moe` | PR #12466; 8-expert top-2 |
| ModernBERT | ✅ | `modern-bert` | SWA global/local + per-layer RoPE θ |
| MPNet | ❌ | — | T5-style rel-attn bias unimplemented — **we are unique** |
| GTE-v1.5 (`NewModel`) | ❌ | — | NTK-RoPE `NewModel` unsupported (#6821) — **we are unique** |
| DeBERTa-v2 | ❌ | — | disentangled c2p/p2c has no ggml graph — **we are unique** |
| SPLADE (sparse) | ❌ | — | MLM head dropped at convert — **we are unique** |
| bge-m3 sparse+ColBERT | ❌ (dense only) | — | tri-head only in fork `iz0eyj/llama.cpp-mv` — **we are unique** |

Decoder / hybrid embedders:

| CrispEmbed | in llama.cpp | arch id | note |
|---|---|---|---|
| Qwen3-Embedding | ✅ | `qwen3` (embed mode) | last-token, **causal** (Qwen3-Emb is trained causal — correct); Instruct/Query prefix is caller-side |
| EmbeddingGemma | ✅ | `gemma-embedding` | Dense/Matryoshka projection supported via `--sentence-transformers-dense-modules`; mean, non-causal |
| LFM2 / LFM2.5 | ✅ | `lfm2` (+`lfm2moe`) | PR #14620; ShortConv via `ggml_ssm_conv`, conv tensors F32 |
| LFM2.5-Embedding | ✅ | `lfm2` embed | official LiquidAI GGUFs, bidirectional |
| LFM2.5-ColBERT | ⚠️ partial | `lfm2` + `--pooling none` | per-token out; MaxSim client-side |
| BidirLM-Omni | ❌ | — | not present — **we are unique** |

Reranking: `--pooling rank` (RANK=4), `/v1/rerank` (PR #9510). bge-reranker-v2-m3
/ base, jina-v2, ms-marco-MiniLM ✅. Qwen3-Reranker ✅ (needs `cls.output.weight`
+ template). mxbai-rerank (DeBERTa-v2) ❌.

Vision / VLM-OCR (via `libmtmd`, projector-id keyed):

| CrispEmbed | in llama.cpp | projector id | note |
|---|---|---|---|
| Qwen2/2.5-VL | ✅ | `qwen2vl_merger` / `qwen2.5vl_merger` | 2D RoPE `build_rope_2d()`, window-attn |
| Qwen3-VL (+MoE) | ✅ | `qwen3vl_merger` | **DeepStack + IMROPE** — same family as our BidirLM-Omni |
| InternVL2/2.5/3 | ✅* | `internvl` | OpenGVLab (non-HF) checkpoints only |
| GLM-4V / GLM-OCR | ✅ | `glm4v` | AIMv2 tower, **dynamic** resize (ours = fixed 336) |
| Granite Vision 3.x | ✅ | `mlp` (LLaVA-Next) | multi-level feature concat + anyres |
| SmolVLM/SmolDocling/Idefics3 | ✅ | `idefics3` | SigLIP + pixel-shuffle |
| Pixtral / LightOnOCR-1B | ✅ | `pixtral` / `lightonocr` | LightOnOCR-2 declined (#18943) |
| DeepSeek-OCR / Unlimited-OCR | ✅ | `deepseekocr` / `deepseekocr2` | hybrid SAM+CLIP DeepEncoder |
| PaddleOCR-VL | ✅ | `paddleocr` | NaViT + M-RoPE (`ggml_rope_multi`) |
| GOT-OCR2 | ❌ | — | SAM path exists only inside DeepSeek-OCR — **we are unique** |
| CLIP/SigLIP standalone image **or** text embed | ❌ | — | mtmd is tower-only (per-patch, LLM-sized); no text tower — **we are unique** |
| Math OCR (pix2tex/TrOCR/HMER/BTTR/PosFormer/MixTex/PP-FormulaNet/PARSeq/Tesseract/Pix2Struct) | ❌ | — | enc-dec/CTC out of llama.cpp's class — **we are unique** |

Entirely outside the ggml ecosystem (CrispEmbed-only): **face** (YuNet/SCRFD/
AuraFace/SFace), **detection/layout** (DBNet/RT-DETRv2/Surya-Det), **NER/KIE**
(GLiNER/LiLT; BERT-NER only an *unmerged* PR #19725), **LID** (CLD3/GlotLID),
**punctuation** (FireRedPunc/Fullstop/PCS), and **image restoration/SR** (NAFNet/
SwinIR/HAT/Restormer/SCUNet/SAFMN/DAT/InstructIR/AdaIR — only ESRGAN/RRDBNet
exists, and in `stable-diffusion.cpp`, not llama.cpp).

#### Convergence backlog (each item = one A/B-gated step)

Ordered by leverage. Every item names its speed harness and its quality harness.

**C1 — imatrix quantization pipeline (biggest quality win, zero graph risk). — IMPLEMENTED (2026-07).**
llama.cpp's importance-matrix quant minimizes *activation-weighted* error, which
directly attacks our q4_k floor (LFM2 ~0.982, BidirLM ~0.93–0.95). Shipped:
`src/imatrix.{h,cpp}` (eval-callback collector gated by `CRISPEMBED_IMATRIX_OUT`,
wired into the encoder + decoder embedding schedulers and lfm2_embed; other
engines are a 1-line `crispembed_imatrix_install(sched)` away), `crispembed-quantize
--imatrix <file>`, and the A/B harness `tools/imatrix_ab.py`.
- Workflow: (1) `CRISPEMBED_IMATRIX_OUT=m.imatrix crispembed -m m-f16.gguf <calib texts…>`
  (one process, merges across runs); (2) `crispembed-quantize m-f16.gguf m-q4k.gguf
  q4_k --imatrix m.imatrix`.
- **A/B result (jina-v5-nano, 32 calib / 12 held-out texts, cos vs f16 gold):**
  | quant | baseline | +imatrix | size |
  |---|---|---|---|
  | q4_k   | 0.945522 | 0.956885 (Δ +0.0114) | 176.3 MB |
  | iq4_xs | 0.958414 | 0.964832 (Δ +0.0064) | 172.7 MB |
  IQ4_XS+imatrix wins on **both** quality and size vs q4_k+imatrix. Identical
  0.62 s embed. Verified on Metal. VERDICT: PASS. `iq4_xs`/`iq4_nl` are wired in
  the quantizer (IQ4_XS→IQ4_NL→Q4_0 fallback for non-256-aligned rows).
- **Kaggle rollout (2026-07, `tools/kaggle/crispembed-imatrix-quant/`):** batch
  harness sources each model's existing full-precision GGUF from its `cstr/*-GGUF`
  repo (auto-detected), calibrates, quantizes q4_k/iq4_xs +imatrix, A/B vs the
  gold, uploads under DISTINCT names (never clobbering baselines) + `.imatrix` +
  `-imatrix-ab.txt`. **20 models done** (cos vs full-precision gold, Kaggle CPU;
  q8_0 ~0.9998 reference for all). Winner = best-small; ★ = IQ4_XS wins:

  | model | q4_k base | q4_k+im | iq4_xs+im | winner |
  |---|---|---|---|---|
  | lfm2-embed             | 0.9869 | 0.9912 | 0.9889 | q4_k |
  | jina-v5-nano           | 0.9868 | 0.9908 | 0.9893 | q4_k |
  | jina-v5-small          | 0.9792 | 0.9901 | 0.9887 | q4_k |
  | octen-0.6b             | 0.9553 | 0.9759 | 0.9751 | q4_k |
  | qwen3-embed-0.6b       | 0.9491 | 0.9794 | 0.9758 | q4_k |
  | f2llm-v2-0.6b          | 0.6831 | 0.8303 | 0.8150 | q4_k (poor — keep q8_0) |
  | bge-m3                 | 0.9667 | 0.9702 | 0.9811 | ★iq4_xs |
  | e5-large              | 0.9813 | 0.9836 | 0.9896 | ★iq4_xs |
  | bge-large-en-v1.5      | 0.9883 | 0.9904 | 0.9899 | q4_k |
  | bge-base-en-v1.5       | 0.9831 | 0.9876 | 0.9896 | ★iq4_xs |
  | bge-small-en-v1.5      | 0.9844 | 0.9860 | 0.9901 | ★iq4_xs |
  | mxbai-embed-large-v1   | 0.9879 | 0.9905 | 0.9906 | ★iq4_xs |
  | multilingual-e5-base   | 0.9665 | 0.9726 | 0.9855 | ★iq4_xs |
  | multilingual-e5-small  | 0.9886 | 0.9897 | 0.9923 | ★iq4_xs |
  | nomic-embed-text-v1.5  | 0.8370 | 0.8443 | 0.9054 | ★iq4_xs (poor — keep q8_0) |
  | nomic-embed-text-v2-moe| 0.9085 | 0.9085 | 0.9313 | ★iq4_xs |
  | arctic-embed-l-v2      | 0.9395 | 0.9482 | 0.9671 | ★iq4_xs |
  | gte-base-en-v1.5       | 0.9620 | 0.9654 | 0.9760 | ★iq4_xs |
  | gte-large-en-v1.5      | 0.9778 | 0.9812 | 0.9826 | ★iq4_xs |
  | pixie-rune-v1          | 0.9366 | 0.9480 | 0.9689 | ★iq4_xs |

  imatrix always lifts 4-bit; **IQ4_XS+imatrix wins on the XLM-R/BERT encoders**
  (smaller AND higher cos), **q4_k+imatrix on the decoder embedders** (Qwen3/LFM2).
  GTE `NewModel` arch and nomic-v2-MoE both worked. **embeddinggemma-300m** was a
  quantizer bug (`dense.*` ST projection quantized → unloadable GGUF), now **FIXED**
  with a `dense.*` keep-F32 guard (`tools/quantize.cpp`; see `LEARNINGS.md`) and
  re-enabled. f2llm/nomic-v1.5 quantize poorly at 4-bit even with imatrix → keep q8_0.
- Registry defaults for the FIRST 5 already repointed to the A/B winner
  (`model_mgr.cpp`; `-q4k`→imatrix, `-iq4xs`/`-q8` aliases). **TODO:** repoint the
  other 15 (winner column above; keep q8_0 for f2llm/nomic-v1.5); a `dense.*`
  quantizer guard to unblock embeddinggemma; C8 wire remaining engine schedulers;
  domain-matched calibration; retrieval A/B via `tests/bench_rag.py`.

**C2 — data-driven GGUF behavior flags.** Bake `pooling_type`, `causal_attention`,
`add_bos_token`, `add_eos_token` into GGUF metadata (llama.cpp convention) instead
of hardcoding in the dispatcher (e.g. our LFM2 "BOS-only" rule → `add_bos_token=
true,add_eos_token=false`). Reduces per-arch branches; improves interop.
- *A/B quality:* `test_all_parity.py` — outputs must be byte-identical before/after
  (pure refactor). Cross-check a WordPiece, an SPM, and a BPE model.
- *A/B speed:* `tests/benchmark.py` — expect neutral; guard against a metadata-read
  regression (remember the `gguf_free` use-after-free landmine).

**C3 — batched embedding throughput. — ENCODER PATH IMPLEMENTED, OPT-IN (2026-07).**
Adopt llama.cpp's pack-many-short-sequences + block-diagonal segment mask
(`n_ubatch == n_batch` for bidirectional encoders). Feeds the open "true batched
graph for decoder models" task. (The **decoder** batched graph already existed —
`decoder_encode_tokens_batch`, block-diagonal + prefix-recompute layout.)
- Shipped for the **encoder** (BERT/XLM-R/MiniLM/BGE/E5 — absolute-position, no
  MPNet rel-bias / DeBERTa rel-embd / RoPE): `encode_tokens_packed` in
  `src/crispembed.cpp` packs B sequences end-to-end into one graph with a host-built
  block-diagonal F16 `seg_mask` fed to `flash_attn_ext`; positions restart per
  segment. Greedy token-budget grouping (`CRISPEMBED_ENCODER_PACK_MAXTOK`, default
  384) caps `T_total`. `build_encoder_graph(..., packed_mask=true)` reuses ~100% of
  the existing graph.
- *A/B quality:* PASS. `tests/test_encoder_batch.py` — packed vs per-sequence cos
  **≥ 0.9999** (typically 1.0; worst 0.9999697) on all-MiniLM q8_0, Metal + CPU,
  single- and multi-group. Bit-parity: each segment sees only its own tokens.
- *A/B speed:* **INCONCLUSIVE on this dev box → kept OPT-IN** (`CRISPEMBED_ENCODER_PACKED=1`).
  Packing amortizes per-graph overhead but makes attention **O(T_total²)** (the
  block-diagonal mask still computes masked cells), so it is inherently NOT the
  near-linear speedup hoped for on bidirectional encoders; medians swung 0.46×–2.07×
  same-config on a 16 GB M1 under load; uncapped packing was a 3.7× loss. Real win
  for the CPU-only VPS.
- **NEXT (the real throughput fix): rectangular 4D per-item mask, O(B·T²).** Keep
  sequences as separate 4D batch items `[hd,T,nh,B]` (already scaffolded in
  `build_encoder_graph`) with a per-item padding mask so attention is O(B·T²) not
  O((B·T)²) — the pinned ggml `flash_attn_ext` accepts a per-batch mask
  (`q->ne[3] % mask->ne[3] == 0`); needs Metal 4D-mask verification. This is the
  prioritized C3 follow-up (after C8 EmbeddingGemma coverage).

**C4 — KV prefix-sharing for the decoder-embedding path.** Port the `seq_cp`
cell-copy idea (cells carry `{pos, set<seq_id>}`; compute shared prefix once, copy
to each fork, decode only the divergent suffix). Note: LFM2 conv state can't be
partially erased — copy *whole* prefixes only. (Blueprint "KV cache for
prefix-shared decoder batches" is marked DONE — this extends/validates it.)
- *A/B speed:* `test_decoder_batch.py` — time N prompts sharing a common instruction
  prefix, with vs without prefix reuse; target ≈ (unique-suffix work) not (N × full).
- *A/B quality:* `test-<model>-diff` — reused-prefix outputs identical to full
  recompute (cos ≥ 0.9999) on Qwen3-Embedding and a Gemma3 embedder; separately
  confirm on LFM2 (whole-prefix constraint).

**C5 — mtmd preprocessing alignment (the open Qwen2VLImageProcessor port).**
Use `tools/mtmd/mtmd-image.cpp` as the reference spec: `calc_size_preserved_ratio`
(= HF `smart_resize`), `resize_bicubic_pillow` (fixed-point, PIL `a=-0.5` — note
their comment that PyTorch uses `a=-0.75`; this is exactly our sub-pixel resize
residual, cos 0.999984), and the `llava_uhd` multi-crop / `select_best_resolution`.
Align on the `mmproj-*.gguf` metadata keys and `<__media__>` chunk convention for
interop; **do not link libmtmd** (it PUBLIC-links all of `llama`).
- *A/B quality:* per-stage `dump_qwen2vl_reference.py` / `dump_qwen3vl_reference.py`
  → `test-qwen2vl-diff`/`-e2e`: preprocessed-tensor cos vs HF and end-to-end OCR
  transcript match on both `a=-0.5` and `a=-0.75` kernels; pick the one matching HF.
- *A/B speed:* `CRISPEMBED_QWEN2VL_BENCH=1` vision-stage ms (in-process vs current).

**C6 — flash-attn epilogue audit (correctness sweep). — IMPLEMENTED (2026-07).**
Codify the rule from memory `flashattn-ext-already-permutes` / commit `6027b56`:
`ggml_flash_attn_ext` returns `[hd,nh,T]` already permuted — reshape directly,
never trailing `permute(0,2,1,3)`. Sweep every FA call site (layout, math,
deepseek, encoders).
- *A/B quality:* the relevant `test-<model>-diff` for each FA site must hold cos
  1.0 vs reference with FA on. Add a per-site guard so a spurious permute craters
  the diff test (not silent).
- *A/B speed:* `CRISPEMBED_<MODULE>_BENCH=1` FA-on vs FA-off (`*_NO_FLASH`) — FA must
  win on long sequences, and we keep the non-FA fallback for uncovered head dims.
  Verify `GGML_KQ_MASK_PAD` (64 on master vs historical 32) against our pinned ggml.
- **Shipped:** audited all 39 `ggml_flash_attn_ext` call sites across 22 engines
  — every one already reshapes the FA result directly; **no surviving
  double-permute** (the June-2026 wave cleanup is in, with warning comments in
  layout/math/unlimited/deepseek). Codified the rule as a reusable graph guard
  `core_ggml::assert_fa_layout(attn, head_dim, n_heads)` in
  `src/core/ggml_metal_guard.h`: it validates the invariant a spurious
  `permute(0,2,1,3)` violates (`ne[0]==head_dim && ne[1]==n_heads`) via
  `GGML_ASSERT` and returns the tensor unchanged, so it composes with any
  downstream reshape (2D/3D/batched `[.,.,T,B]`) and craters at graph-build time
  instead of shipping silent garbage. Wired into the guarded sites: bidirlm_vision,
  lfm2_embed, vit_embed (×2), clip_text_embed. **Runtime-proven** on a real
  flash_attn_ext node: assert passes at `ne=[64,8,5]` (`[hd,nh,T]`) and aborts
  when a trailing `permute(0,2,1,3)` makes it `[64,5,8]`. The remaining ~34 sites
  are a mechanical drop-in as each engine's diff test is next touched.

**C7 — generalize the Metal mul_mm F16 guard. — IMPLEMENTED (2026-07).** Turn the
×1/256-before / ×256-after trick (memory `metal-mul-mm-f16-overflow`) into a
reusable helper + a diagnostic ("NaN with many tokens, clean single-token ⇒
you're on `mul_mm`"). Metal picks `mul_mm` purely by shape (`ne11 > 8`) and
ignores `set_prec(F32)` for GEMM.
- *A/B quality:* Metal-vs-CPU cos on the image path (many patches) for every VLM
  engine via `<ENGINE>_FORCE_CPU=1`; target Metal cos == CPU cos (no NaN).
- *A/B speed:* `CRISPEMBED_<MODULE>_BENCH=1` — the scale helper must be negligible
  vs the matmul it protects.
- **Shipped:** `core/ggml_metal_guard.h` provides `mul_mat_f16_guarded(g, w, act,
  n_tokens, guard=256)` — applies the lossless exponent shift only when Metal
  would pick the F16-casting `mul_mm`, predicate `metal_mul_mm_f16_cast_active(
  ne11, ne00)` = `ne11 > 8 && ne00 >= 64` (verified against
  `ggml-metal-ops.cpp:2050` `ne11_mm_min = 8` + the `ne00 >= 64` mul_mm guard at
  line 2158) — else a plain `ggml_mul_mat`. The diagnostic is codified in the
  header comment. `granite_vision_ocr` now calls the helper (graph-identical to
  the old inline ternary). **Clean same-commit / same-backend A/B** (granite
  q4_k OCR on fox.png, both on the Metal `mul_mm` path, T≈750 ≫ 8): inline-ternary
  output == helper output, **token-identical greedy decode** → behavior-preserving.

**C8 — cheap coverage wins (borrow upstream where clean).** ModernBERT, Nomic-v2-
MoE, and EmbeddingGemma's Dense/Matryoshka projection are cleanly solved upstream
and map to archs we nearly have. Each new arch = new `dump_*_reference.py` +
`test-*-diff` before it ships.
- *A/B quality:* new `test-<arch>-diff` cos ≥ 0.999 vs HF (q8_0) and the retrieval
  bench for embedders.
- *A/B speed:* `tests/benchmark.py` throughput sane vs a same-size existing arch.

**ModernBERT (gte-modernbert-base) — VALIDATED E2E (2026-07).** Was structurally
supported but never parity-checked, and broke three ways; now **cos 0.999999**
(short) / **0.999998** (113-tok doc) vs HF on Metal + CPU, **0.99976** at q8_0.
- *Local-path converter (BPE tokenizer + CLS pooling + Unigram scores):* all three
  detections called `hf_hub_download(repo_id=args.model)`, which throws on a local
  path and was silently caught → fell back to WordPiece + mean pooling (cos 0.46).
  Added `_resolve_file()` at all three sites. Convert with `--crisp` (ollama mode
  never runs BPE detection).
- *Missing sliding-window local attention:* only RoPE θ alternated; the local
  layers' ±`local_attention`/2 window mask was absent → they attended globally and
  long docs diverged (113-tok cos 0.9826 → 0.999998). Added per-layer `swa_mask` on
  local layers; converter emits `bert.local_attention`, loader reads it. A/B lever
  `CRISPEMBED_ENCODER_NO_SWA=1`. No effect on non-ModernBERT encoders.
- GGUFs published to `cstr/gte-modernbert-base-GGUF` (f16 + q8_0); registry entry in
  `examples/cli/model_mgr.cpp`. Guard: `tests/test_modernbert_parity.py`. TODO: wire
  a compiled `test-modernbert-diff` into the regression manifest.

**NEXT PRIORITIES (2026-07, ordered):** (1) **EmbeddingGemma Dense/Matryoshka parity
coverage** — the projection + Matryoshka path ships with ZERO parity test; add a
`dump_embeddinggemma_reference.py` + parity test (cos ≥ 0.999 vs HF, verify the
Dense→L2→Matryoshka order). (2) **C3 rectangular 4D per-item-mask batch** (O(B·T²);
see C3 above). Nomic-v2-MoE already covered (`test_moe_parity.py`).

**C9 — preserve & document our differentiation.** MPNet, GTE-v1.5, DeBERTa-v2,
SPLADE, bge-m3 tri-head (dense+sparse+ColBERT), standalone CLIP/SigLIP text+image
embeddings, and the entire face/detection/NER/LID/punct/SR surface are things
llama.cpp does **not** have. Keep them, keep their guardrails green, and note the
gap in `README.md` as a selling point. No new A/B (these already have diff tests);
the action is documentation + not regressing them.

#### A/B protocol (applies to every C-item)

- **Two axes, always.** Speed *and* quality, reported as before→after deltas.
- **Quality harness:** the model's `test-<model>-diff` cosine vs a fixed
  `dump_<model>_reference.py` HF reference (regenerate the ref only when the model
  changes, never to "make it pass"). For OCR/VLM add an end-to-end transcript
  match; for retrieval add `tests/bench_rag.py`/`bench_rerank.py` (nDCG/MRR/recall).
- **Speed harness:** `CRISPEMBED_<MODULE>_BENCH=1` per-stage ms and/or
  `tests/benchmark.py` throughput. Honor the 8GB-VPS single-thread constraint.
- **Both backends:** run CPU (`<ENGINE>_FORCE_CPU=1`) and Metal; a change that
  helps one must not crater the other.
- **Gate:** accept only if quality does not regress beyond diff-test noise
  (cos ≥ baseline − 0.0005) *and* speed does not regress (or the trade is explicit
  and approved). Record both numbers in the commit body.

### Regression-guardrail gaps — trace + close methodically (2026-07)

Context: the OCR/VLM engines and **11 SR/restoration engines** (restormer, swinir,
dat, hat, pan, tbsrn, adair, scunet, instructir, esrgan, safmn) now have auto-run
`diff_only` guardrails in `tests/regression/manifest.json`. The engines below still
lack a working, wired reference. **Method for each: trace LOCALLY first** (small
models; use `/Volumes/backups` for HF cache to spare the main disk), get the ACTUAL
per-stage cos from `test-<x>-diff`, then **disambiguate dumper-bug vs engine-regression
before claiming a root cause** — do NOT write a fix-handover on assumption (the first
restormer handover blamed the conv-weight pre-permute; the real bug was the MDTA
block-graph — verify empirically). Only after the actual cause is confirmed: fix the
engine or the dumper, (re)generate the ref, upload to HF, and wire the manifest entry.
WAVE window = 2026-06-19..06-22 (scalar→ggml refactor).

**Gap 1 — NER/embedding dumpers failed on Kaggle (env/dumper, NOT necessarily a regression).**
`dump_failed` on Kaggle CUDA for gliner, lilt, lfm2_colbert, layout — tokenizer
padding / `gliner` pkg / wrong source-id errors in the *dumper*, on Kaggle's
transformers version. History: **gliner** (created 06-12, verified cos 1.0 06-13)
and **lilt** (06-15, verified 25/25 cos 1.0 same day) are **pre-wave and untouched →
almost certainly fine; the Kaggle failure is env**. Re-run the dumpers LOCALLY, confirm
cos, upload refs, wire. `dump_layout_reference.py` source id was likely wrong
(`cmarkea/dit-base-layout-detection`) — find the correct one (layout-heron / RT-DETR).

**Gap 2 — verify mismatch: nafnet + lfm2 (engine-regression SUSPECTS).**
Ref generated but ggml output ≠ ref. **nafnet — RESOLVED 2026-07-02** (was cos
0.538). It was a real conv→ggml-wave regression, same shape as restormer: scrambled
kernels (`permute(3,2,1,0)` instead of a `[KW,KH,IC,OC]` reinterpret; 1×1 convs also
hit a wrong 2D branch via `ggml_n_dims()` collapse), plus a depthwise-F16 sched-assert
and a Metal/CUDA residency abort. Dumper confirmed faithful via a `NAFNET_SCALAR=1`
A/B (scalar 0.999998 vs ref). Now ggml==scalar==ref cos 0.999998 on Metal+CPU;
`test-nafnet-diff` + ref (`cstr/nafnet-sidd-GGUF/nafnet-ref.gguf`) wired into the
regression manifest. The same audit found **restormer still aborted on Metal**
(residency) and fixed it (weights → CPU). **lfm2/lfm2_colbert**: created 06-18, heavily changed in the wave
(gallocr→backend_sched `29176a0`, multivec segfault fix `a091283`, both 06-20).
Disambiguate dumper vs engine; if engine, bisect against the pre-wave state.
**layout**: wave-touched (`dc0861b` 06-20 replaced manual attention with
`ggml_flash_attn_ext`) → also a regression risk even once its dumper is fixed.

**Gap 3 — bert_ner: no dumper exists.** `test-bert-ner-diff` + `bert-base-NER-GGUF`
exist, but there is no `tools/dump_bert_ner_reference.py`. It shares the BERT encoder
(core parity 1.0 via `parity_layers_bert.py`), pre-wave. Write the dumper (mirror an
existing transformers-based dumper), generate ref locally, upload, wire.

**Gap 4 — no standing guardrail for embedding/face + tail engines.**
vit_embed (cos 0.996), clip_text (0.997), decoder_embed (≥0.999), cnn_embed/face
det-rec (0.9999 vs ONNX), face_align (MAE 0.0) — all verified once at conversion but
have no compiled diff test; add one (or wire the Python parity into CI). **text_sr**
blocked (no public checkpoint). **tps_locnet/tps_warp, pcs, fireredpunc,
bidirlm_audio/vision** — no documented CrispEmbed-side verification; assess.

**Gap-4 wave-risk triage (2026-07, by *nature* of the wave-window edit, not just churn):**
- **text_sr — HIGH risk, UNTESTABLE.** Got `09a6e02 perf: replace scalar conv2d with
  ggml_conv_2d` (the exact refactor family that scrambled nafnet's kernels). But NO public
  checkpoint and NO shipped GGUF anywhere (no cstr repo, no local file, no converter) → no
  parity ref possible. Mitigant: text_sr is a NAFNet variant sharing the conv paths that are
  now guarded by the `nafnet` entry; the PixelShuffle/bicubic tail remains unguarded. Blocked
  until a checkpoint exists.
- **pcs — REGRESSION FOUND, shipped-broken.** `4a498d1 perf(pcs): cache FC head weights at init`
  was the exact wave commit at fault. The shipped default `pcs-xlmr-base-q4_k.gguf` **crashes on
  every inference** (`ggml-backend.cpp:349 tensor read out of bounds`): pcs reads its Q4_K/Q4_0
  FC-head weights via raw `ggml_backend_tensor_get` into F32 buffers (`n_elem*4` >> `ggml_nbytes`
  of a quantized tensor). Impl is in the SIBLING repo **CrispASR/crisp_punc/src/pcs.cpp** (shared
  lib replaces the local copy). Handover:
  `handover-prompts/pcs-q4k-head-weight-tensor-get-crash.md`. Not wired (a guard would be red until
  fixed). fireredpunc is unaffected (F16 cls head, in-graph mul_mat).
- **decoder_embed — CLEAN, CLOSED.** Added a compiled guardrail: `test_decoder_embed_diff.cpp`
  (crispembed_encode → final last-token-pooled embedding) vs an independent Qwen3-Embedding-0.6B
  HF ref (`dump_decoder_embed_reference.py`). Engine (q8_0) matches cos 0.9993; wired `diff_only`,
  run_one PASS. Also added to the Kaggle ref-gen kernel. Confirms the wave flash_attn_ext work is fine.
- **vit_embed / clip_text / cnn_embed / bidirlm_vision — LOW (perf-only).** Last wave edit was
  `632b4c1 perf: disable OpenMP / default 1 thread` (threading, not numeric). Standing
  guardrails nonetheless; GGUFs + HF sources available → closeable locally.
- **fireredpunc / bidirlm_audio — LOW.** Only `402b38d feat: benchmark instrumentation` (no
  numeric change). fireredpunc GGUF + BERT source available → closeable.
- **bidirlm (text) — Kaggle-queued.** BidirLM-Omni-2.5B (2.5B, too large for local ref-gen) added
  to the ref-gen kernel reusing `test-decoder-embed-diff` with `--pooling mean` (bidirectional);
  the kernel dumps→verifies→uploads on a GPU worker. `source_optional` so a pooling/format mismatch
  just reports verify_failed rather than aborting the batch. bidirlm_audio/vision towers (image/
  audio fixtures) not yet covered.
- **tps_warp — covered.** Pure-math warp; `test_tps_warp.cpp` self-contained. tps_locnet CNN
  now guarded (see Trace outcomes).

**Trace outcomes (local, 2026-07 — empirical, disambiguated):**
- **lilt — CLEAN, CLOSED.** 24/24 encoder stages cos 1.000000; ref uploaded to
  `cstr/lilt-base-GGUF`, wired `diff_only`. (The harness's "Label match 0/16" is a red
  herring — base checkpoint ships an untrained classifier head.) Not a regression.
- **lfm2 — CLEAN, CLOSED.** 20/20 stages cos ≥0.9997; ref uploaded to `cstr/lfm2-embed-GGUF`,
  wired. Blocker was a dumper bug (duplicate `general.architecture` key), now fixed. Not a regression.
- **layout — REGRESSION.** Encoder craters (`s3` cos −0.146…`dec_0_cross` −0.344; early
  stages cos 1.0). Wave `dc0861b` (flash_attn_ext). Handover:
  `handover-prompts/layout-detect-encoder-regression-fix.md`. (2 agents assigned.)
- **nafnet — REGRESSION, FIXED (other agent).** Disambiguated ENGINE (not dumper):
  ref is trustworthy — cos(ref_input, ref_output)=0.86, output properly denoised. Root cause
  = conv-kernel layout (ggml loads numpy [OC,IC,KH,KW] bytes but the old permute(3,2,1,0)+cont
  mis-declared the view → scrambled kernels; 1×1 convs took a 2nd wrong branch) + conv-sched
  backend residency (kernels lived on Metal/CUDA, conv sched is CPU). Fixed by copying dequant
  bytes into an explicit `ne=[KW,KH,IC,OC]` tensor (like swinir) + parking kernels on the conv
  sched's backend (dw F16 for `ggml_conv_2d_dw` im2col, regular F32). `test-nafnet-diff` added.
- **gliner — NOT a regression; the REFERENCE is broken.** Engine extracts the right entities
  ("Barack Obama"→person, "Hawaii"→location); the PyTorch ref extracts **0 entities** and its
  activations are anti-correlated from layer_0 (all cos ≈ −0.55). So `dump_gliner_reference.py`'s
  LFM2 path (Lfm2BiModel bidirectional-replacement / tokenizer patch) yields a DEAD model.
  Caught by the new multi-stage diff added to the LFM2 branch (`gliner_ner.cpp`) + the entity
  output check. Do NOT wire this ref; fix the dumper's bidirectional replacement, or use
  entity-output as the guardrail. (Also fixed 3 dumper bugs to make it run: TFPreTrainedModel
  shim, VPS `/mnt` tmp/cache paths, duplicate `general.architecture`.)
- **lfm2_colbert — CLOSED (was a dumper bug, not a regression).** A hidden_states localizer
  (`lfm2_embed.cpp`, LFM2_COLBERT_DIFF_REF) proved the engine backbone is fine (same as lfm2's
  20/20) and the old dumper's MANUAL forward was wrong (hidden cos −0.54). Rewrote
  `dump_lfm2_colbert_reference.py` to use `AutoModel` → hidden 0.982, colbert_output 0.998.
  Ref uploaded; wired `diff_only` (colbert_output floor 0.99 for q8_0-vs-f32). 4/0 PASS.
- **bert_ner — CLOSED (was a text + safetensors-mmap issue, not a regression).** Dump crashed
  (SIGBUS on dslim/bert-base-NER's model.safetensors mmap) → load `.bin`. The −0.806 diff was a
  TEXT mismatch (dumper used a different sentence than the harness's hardcoded "Barack Obama…").
  Aligned the text → all token ids match, final_hidden cos 0.995. Ref uploaded; wired `diff_only`
  (final_hidden floor 0.99 for q8_0-vs-f32). PASS.
- **tps_locnet — CLEAN, CLOSED (Gap 4).** C++ engine matches an independent pure-numpy forward
  cos 1.000000 (points_pixel + fc2_out); no wave regression. The parity harness
  `test_tps_parity.cpp` already existed — aligned its output to `cos_min=` so run_one parses it,
  wrote `dump_tps_reference_from_gguf.py` (source .pdparams geo-blocked on bcebos → from_gguf like
  the SR engines), uploaded `tps-ref.gguf` to `cstr/tps-loc-GGUF`, wired `diff_only`. End-to-end
  `run_one --name tps_locnet` → 2 stages worst cos_min=1.000000 PASS.
- **vit_embed — CLEAN, CLOSED (Gap 4).** SigLIP ViT final image-embedding matches an independent
  HF-AutoModel ref (`dump_vit_reference.py` on siglip-base-patch16-384, `get_image_features`) at
  cos 0.9915 on CPU (f16 GGUF + preprocessing diff; a scramble → ~0). New `test_vit_embed_diff.cpp`
  (single-stage, image via `diff.args`); floor 0.98 for backend variance. Fixed the dumper's SigLIP
  path (transformers 4.57 needs get_image_features; full SiglipModel.forward requires both towers).
  Ref uploaded to `cstr/siglip-base-GGUF`, wired `diff_only`. run_one PASS.
- **cnn_embed/face (SCRFD) — CLEAN, CLOSED (Gap 4).** C++ `crispembed_detect_faces` matches an
  independent insightface-SCRFD reference (`dump_face_reference.py` over det_10g.onnx) to within
  2.45 px on a FLUX-generated synthetic face fixture (1 face, conf err 0.003). New
  `test_face_diff.cpp` emits a synthetic detection cos_min (1.0/0.0) at a 12px tolerance; ref
  uploaded to `cstr/scrfd-det-10g-GGUF`, wired `diff_only` (`face_detect`). run_one PASS. (Face
  recognition arcface/sface not guarded — no local rec GGUF; detection is the wave-touched path.)
- **decoder_embed — CLEAN, CLOSED (Gap 4).** `test_decoder_embed_diff.cpp` (crispembed_encode,
  last-token pool) vs independent Qwen3-Embedding-0.6B HF ref: cos 0.9993 (q8_0-vs-f32). Ref
  uploaded to `cstr/qwen3-embed-0.6b-GGUF`, wired `diff_only`, run_one PASS. Added to Kaggle kernel.
- **bidirlm (text) — Kaggle run 1: dump_failed → fixed, re-queued (Gap 4).** 2.5B; added to
  `crispembed_ref_gen.py` reusing test-decoder-embed-diff (--pooling mean). First GPU run (chr1s4,
  v5) failed at dump on `Do you wish to run the custom code? [y/N]` — BidirLM-Omni ships custom
  modeling code; added `trust_remote_code=True` to the dumper. Re-pushed to retry. (Same run:
  decoder_embed dumped+verified+uploaded OK on CUDA — double-confirms the local cos 0.9993.)
- **fireredpunc — CLEAN, CLOSED (Gap 4).** No hidden/logits accessor in the punct C API → golden
  text-match `run_check` (new generic `test_punct_diff.cpp`). q4_k engine restores
  "hello world how are you today i am fine thanks" → "Hello world. How are you today? I am fine.
  Thanks." (correct, deterministic). Wired; run_one PASS. Impl in CrispASR/crisp_punc.
- **pcs — REGRESSION, shipped-broken (Gap 4).** See triage above — q4_k crashes on inference
  (Q4_K FC-head weights read as F32). Handover written; fix is in CrispASR/crisp_punc. Not wired.
- **clip_text — BUG FOUND (tokenizer), not a wave regression.** Engine vs HF `get_text_features`
  cos=0.79. Localized: the projection IS applied (cos 0.79 to post-`text_proj`, −0.02 to
  pre-projection), so the error is upstream in **tokenization**. `CLIP_TEXT_DEBUG` token dump on
  "a photo of a fox": engine emits 11 GPT-2-style ids with `220` (space `Ġ`) between words
  `[49406,64,220,1153,220,684,220,64,220,5007,49407]` vs HF's 7 CLIP ids
  `[49406,320,1125,539,320,3240,49407]` — the engine's BPE never applies CLIP's `</w>`
  word-boundary convention, so every token is wrong → wrong EOS hidden. Pre-existing (wave only
  touched threading). Handover: `handover-prompts/clip-text-bpe-word-boundary-tokenizer-bug.md`.
  Harness+dumper kept as WIP on the branch, NOT wired (cos 0.79 fails). Fix the CLIP BPE (`</w>`)
  then wire.

**Methodology lesson (reinforced): a single-stage diff cannot tell a dumper bug from an engine
bug.** gliner's `lstm_out`-only check looked like a BiLSTM regression; multi-stage + the entity
output check proved the engine is fine and the *reference* is dead. Harnesses that check one
stage (nafnet output-only, lfm2_colbert colbert_output-only) MUST be extended to all ref stages
and/or add an independent task-output check before their cos is trusted.

Net: SR/restoration (11) + esrgan/safmn + lilt + lfm2 auto-guarded. Wave regressions found by
tracing: **layout** (encoder, flash_attn) + **nafnet** (conv layout, now fixed). gliner =
broken reference (engine fine); lfm2_colbert = ColBERT-head discrepancy; bert_ner = dumper
written, download-blocked. None of gliner/lfm2/lfm2_colbert backbones are regressions.

### OCR engine correctness/stability fixes (2026-06-30, issue #25)

Found while integrating the OCR engines downstream (BiblioForge). macOS arm64,
Metal, ggml 0.10.0.

**Fixed:**
- **VLM OCR repetition (internvl2, qwen2vl)** — greedy argmax with no repetition
  control looped forever on document text. Added `argmax_no_repeat_ngram` (n=3)
  to both decode loops. (PR #26)
- **got-ocr2 two graph crashes** — (1) `g_ln2d` permute was reversed
  (`(2,0,1,3)` → `(H,C,W)`), so `ggml_norm` used the wrong axis and the `(C,)`
  weight failed `ggml_can_repeat`; fixed to put C at `ne[0]`. (2) `prep_conv_w`
  reshaped a q8_0 conv weight to `ne0=1` before dequantizing → Metal CPY block
  assert; cast to F32 first. got-ocr2 now runs end-to-end without aborting. (PR #27)
- **DBNet detector on Metal `unsupported op 'CPY'` abort** — fix paths **a + c**.
  `dbnet-ic15-q4_k` stores 16 conv/deconv weights as **Q4_K** (a k-quant);
  `prep_conv_weight`/`prep_deconv_weight` dequantized them with `ggml_cast(w, F32)`,
  which emits a CPY node — but Metal's cpy kernels cover **no k-quant source type**
  (only F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/I32), so the graph aborted at the first
  Q4_K weight. (a) Replaced the cast with a Metal-safe `ggml_get_rows`
  identity-select dequant (`dequant_rows_f32()`): GET_ROWS is supported for every
  quant type and yields F32; row indices are generated in-graph via `arange`+cast,
  so no extra graph input. The rest of the graph (conv_2d / conv_transpose_2d /
  upscale / pool / concat) is already Metal-supported, so the GPU path now runs
  correctly end to end (opt in with `OCR_DETECT_USE_GPU=1`). (c) DBNet is tiny and
  conv-heavy and Metal's `conv_transpose_2d` at full resolution is ~13× **slower**
  than CPU (≈139 s GPU vs ≈10 s CPU graph-compute on M1 for a 1472×736 prob map),
  and the pipeline's real cost is per-box TrOCR — so the detector now **defaults to
  CPU**, which is faster and avoids the abort by default. `OCR_DETECT_FORCE_CPU=1`
  still forces CPU. CPU detection output is byte-identical to before; boxes
  localize accurately on a real English page. (Note: the separate TrOCR/math_ocr
  recognizer still emits repeated-token garbage on clean printed text — a
  pre-existing recognizer issue, unrelated to detection.)
- **CI artifacts** were not self-contained (missing ggml libs, absolute runner
  rpaths) — `cmake --install` bundle with `@loader_path`/`$ORIGIN`. (PR #23)
- **Model downloader** didn't create the cache dir before writing `.tmp`. (PR #24)

- **got-ocr2 additional C++ fixes** (2026-07-01):
  - Neck flatten permute: `(2,0,1,3)` produced `(H,C,W)` not `(C,W,H)`;
    fixed to `(1,2,0,3)`. Verified correct token ordering with numpy test.
  - LN2d ensure_f32: neck LN weight ggml_mul crashed on F16×F32 mismatch
    in quantized models; added `ensure_f32()` cast.
  - Prompt template: added correct Qwen2/MPT ChatML template (system +
    `<img><imgpad>*256</img>\nOCR: ` + assistant). No `\n` between
    `<|im_end|>` and `<|im_start|>` — matches original conversation.py.
  - no_repeat_ngram: ported from internvl2_ocr.cpp (ngram=3).
  - Stop tokens: added `<|im_end|>` (151645) alongside eos.
  - Debug env: `CRISPEMBED_GOT_OCR_DEBUG=1` sets verbosity=2.
  - Per-layer diff test passes ALL layers (cos_min≥0.999999 with f16).
  - **Root cause & resolution (CORRECTED 2026-07-01, on M1)**: the earlier
    "q8_0 decoder is catastrophically sensitive (llm_layer_0 cos=0.936 →
    garbage)" conclusion was itself a **diff-harness artifact**, not real
    sensitivity. Re-running the *corrected* harness (row_dim=0) on a plain
    q8_0 decoder gives llm_layer_0..5 cos ≥ 0.99996 (PASS), and end-to-end
    OCR is correct at q4_k, q8_0 AND f16 (all identical output). The
    decoder graph is functionally identical to internvl2's Qwen2-0.5B
    (same NEOX RoPE θ=1M, rmsnorm, flash_attn scale, KV layout, SwiGLU),
    and internvl2-1b already ships that same decoder at q4_k. Vision is
    correct (cos≥0.998 vs HF). The "bf16 compute", "vision encoder", AND
    "decoder must be f16" theories were ALL wrong.
    **Ship q4_k as default**: on M1 per-token decode is q4_k ~20ms <
    f16 ~38ms < q8_0 ~42ms (q8_0 mul_mv is a slow Metal path on M1), and
    q4_k is 3× smaller (445 MB vs 1.03 GB) with identical OCR. The
    `--decoder-f16` quantizer flag is kept but is optional/diagnostic only.

**Still open:**
- **DBNet detector on Metal** — `unsupported op 'CPY'` abort; `OCR_DETECT_FORCE_CPU=1`
  works around it but per-region TrOCR on CPU is slow. Want a Metal CPY path or a
  CPU-default detector.
- ~~No `GOT_OCR_FORCE_CPU` env~~ → **added** (commit 718a73e). Also a debug lever:
  A/B got-ocr2 output on CPU vs Metal — if CPU is also garbage the vision bug is
  logic, not Metal-specific; if CPU is correct, it's a Metal op issue.
- **ggml Metal device-teardown abort** at process exit when loaded alongside
  PyTorch MPS (downstream works around it with `os._exit`).

### GPU + quantization audit (2026-06-16)

All inference engines are GPU-enabled (zero CPU-only gaps). Every engine uses
`ggml_backend_init_best()` and has a `<ENGINE>_FORCE_CPU=1` env override.
A/B verified on CPU — identical outputs, no regression.

**FULL GPU** — `ggml_backend_init_best()` + ggml graph compute (CUDA/Vulkan/Metal):
crispembed (BERT/XLM-R/etc.), decoder_embed (Qwen3/Gemma3), bidirlm_vision,
fireredpunc, pcs, gliner_ner, got_ocr, surya_det, tesseract_lstm, vit_embed,
clip_text_embed, cnn_embed, ocr_detect, parseq_ocr, layout_detect,
internvl2_ocr, qwen2vl_ocr, glm_ocr, math_ocr, ppformulanet_l_ocr, lilt_kie,
bert_ner, granite_vision (ViT + projector + LLM, Metal + ggml-CPU),
restormer, nafnet_denoise, esrgan_sr, safmn_sr, mixtex_ocr.

**GPU-SAFE** — weights on GPU, scalar CPU forward pass (depthwise conv /
PixelShuffle not yet in ggml graph): hmer_ocr, bttr_ocr, posformer_ocr,
ppformulanet_ocr, pan_sr, tbsrn_sr, text_sr, tps_locnet, scunet_denoise,
swinir_sr.

**Summary**: ~28 engines full-GPU, ~10 GPU-safe, 0 CPU-only. All engines have
`<ENGINE>_FORCE_CPU=1`; all SR/restoration models quantized to Q8_0 + Q4_K.

**Known accuracy caveats**:
- `esrgan_sr` — **FIXED**: the ggml graph now computes true per-channel PReLU
  (`relu(x) + slope·min(0,x)`) from the stored slope weights, matching the scalar
  `prelu()` reference (was a plain `ggml_relu` that dropped the slope).
- `hat_sr` — **VERIFIED, no defect.** The OCAB overlapping-window cross-attention
  (overlap unfold `kernel=ows/stride=ws/pad=(ows-ws)/2` + row-major kernel order +
  RPB; image pre-padded to a ws multiple) was numerically validated end-to-end via
  `test-hat-diff` against a torch self-consistent reference (HAT arch loaded with
  the gguf weights): output **cos 0.999968**. The old "simplified, may not match"
  comment was over-cautious. `test-hat-diff` is now registered in CMake; reference
  `hat-ref.gguf` is on HF `cstr/text-super-resolution-gguf`.

### OCR — next-gen models to port

| # | Model | Params | OmniDocBench | License | Architecture | Status |
|---|-------|--------|-------------|---------|-------------|--------|
| ~~1~~ | ~~dots.ocr~~ | ~~3B~~ | ~~88.4%~~ | ~~NOT pure MIT~~ | — | REJECTED: supplemental PRC license (rednote/Xiaohongshu) |
| 2 | **PaddleOCR-VL-0.9B** | 0.9B | — | Apache-2.0 | NaViT + ERNIE-4.5-0.3B | **DONE + verified E2E** (2026-07-02): reuses qwen2vl_ocr engine; fox.png → "The quick brown fox…" on CPU+Metal. Was SIGSEGV-ing (ERNIE head_dim=128≠D/heads) + empty output (SPM vocab loaded as GPT-2 BPE); both fixed. Q8_0/Q4_K on HF |
| 3 | **PaddleOCR-VL-1.6** | 0.9B | 96.3% SOTA | Apache-2.0 | NaViT + ERNIE-4.5-0.3B (same arch, improved training) | **DONE**: same engine/fixes as 0.9B; Q8_0/Q4_K on HF |
| ~~4~~ | ~~MinerU2.5-Pro~~ | ~~1.2B~~ | ~~90.7%~~ | ~~NOT pure Apache~~ | — | REJECTED: commercial thresholds, mandatory attribution, gated HF |
| 5 | **SmolDocling** | 256M | — | Apache-2.0 | Idefics3/SmolVLM, IBM Research | DONE: engine + parity cos=0.9999, HF `cstr/smoldocling-GGUF` |
| ~~6~~ | ~~Hunyuan-OCR~~ | ~~1B~~ | — | ~~Custom Tencent~~ | — | REJECTED: excludes EU/UK/South Korea |
| 7 | **Qari-OCR** | 4B | Apache-2.0 | Qwen2-VL fine-tune (Arabic only) | Vision parity fixed; LLM Q4_K floor expected. Prompt: direct "output only text" instruction; general.name detection added (filename-independent). |

**Remaining**: FireRed-OCR (Qwen3-VL 2B) and german-ocr-3 reuse the qwen2vl_ocr engine; runtime ne-fix handles GGUF converters that store weights in PyTorch (out, in) order.

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
| 1 | **Uni-MuMER-Qwen3-VL-2B** | 2.1B | ~82% (3B variant) | Apache-2.0 | Qwen3-VL fine-tune (multi-task: recognition + symbol counting + position) | Low — reuses existing `qwen2vl_ocr.cpp` engine, same GGUF converter | **DONE**: Q4_K/Q8_0, auto-prompt, `<think>` stripping |
| 2 | **Uni-MuMER-Qwen2.5-VL-3B** | 3.4B | 82.25% | Apache-2.0 | Qwen2.5-VL fine-tune | Low — same engine | **DONE**: Q4_K (2.6 GB) / Q8_0 (4.2 GB), streaming converter |
| 3 | **TexTeller 3.0** | 0.3B | unknown | Apache-2.0 | ViT-12 (768d) + TrOCR-12 (1024d), 15K vocab, 448px grayscale | Low — reuses existing `math_ocr.cpp` + `convert-trocr-safetensors-to-gguf.py` | **DONE**: F16/Q8_0/Q4_K, manual matmul attention |
| 4 | PP-FormulaNet-L | 181M | ~57% | Apache-2.0 | SAM-ViT + MBart | — | Already integrated (mostly printed math) |

**Recommended priority:**

1. **Uni-MuMER-Qwen3-VL-2B** — **DONE**. Pure fine-tune of Qwen3-VL-2B-Instruct
   (phxember/Uni-MuMER-Qwen3-VL-2B, Apache-2.0). Reuses `qwen2vl_ocr.cpp` engine
   with auto-detected math OCR prompt and `<think>` token stripping. Converter
   fixed for transformers 5.x `rope_parameters` field + `processor_config.json`
   nested format + `tokenizer.json` fallback. GGUF: Q4_K (1.5 GB), Q8_0 (2.2 GB).
   Tested: `x^{2}+2xy+y^{2}=0`, `E=mc^{2}+\int f(x)dx` — correct.

   Source: [github.com/BFlameSwift/Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)
   Weights: [huggingface.co/phxember/Uni-MuMER-Qwen3-VL-2B](https://huggingface.co/phxember/Uni-MuMER-Qwen3-VL-2B)

2. **TexTeller 3.0** — **DONE**. Standard VisionEncoderDecoderModel: ViT (12L, 768d,
   448px grayscale) + TrOCR decoder (12L, 1024d, 15K vocab). Reuses existing
   `math_ocr.cpp` engine and `convert-trocr-safetensors-to-gguf.py` converter.
   Converter fixed: added_tokens.json merge, scale_embedding metadata, and
   decoder_start_token_id resolution (2026-07: was reading the nested
   decoder.decoder_start_token_id=2 that HF's VisionEncoderDecoder IGNORES;
   now reads top-level → bos=0 like HF. The wrong start=2 poisoned the
   position-0 KV cache and made decode repeat/degenerate on anything past
   trivial formulas — `\frac{a}{b}` looped `\frac{\frac{…`).
   Engine fixed: dynamic channel count (1ch grayscale), ViT CLS-only (no DeiT
   distillation token), tied embeddings as LM head, GELU decoder FFN,
   manual matmul attention for encoder (>512 tokens).
   GGUF: F16 (568 MB), Q8_0 (302 MB), Q4_K (169 MB) — regenerate with the
   fixed converter (start=0) to replace the broken start=2 uploads.
   Verified vs HF TexTeller on identical images (exact match): `x+y`→`\[x+y\]`,
   `\frac{a}{b}`→`\[\frac{a}{b}\]`, `E=mc^2+\int f(x)dx`, and the
   `\int_0^\infty x^{s-1}/(e^x-1)dx=\Gamma(s)\zeta(s)` integral — all correct.

   Source: [github.com/OleehyO/TexTeller](https://github.com/OleehyO/TexTeller)
   Weights: [huggingface.co/OleehyO/TexTeller](https://huggingface.co/OleehyO/TexTeller)

3. **Uni-MuMER-Qwen2.5-VL-3B** — **DONE**. Pure fine-tune of Qwen2.5-VL-3B-Instruct
   (phxember/Uni-MuMER-Qwen2.5-VL-3B, Apache-2.0). 82.25% CROHME. Converter
   refactored to streaming mode (add_tensor_info + write_tensor_data) for 8 GB VPS.
   GGUF: Q4_K (2.6 GB), Q8_0 (4.2 GB). Tested with tiny image — correct LaTeX output.

**Impact**: Both Uni-MuMER variants are now ported. NC-licensed 57% models
can be replaced with Apache-2.0 82% models — eliminates the license gate
in the UI AND nearly doubles handwritten accuracy.

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
- [x] **#3 Per-row embedding dequant** — already done. `put_tok` lambda (~line
  2604) and `get_embedding` lambda (~line 1950) both use per-row
  `ggml_backend_tensor_get`. Item was stale.
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
  `conv2d_cpu` SIMD: gather each spatial patch into a `thread_local` buffer then
  call `dot_product` (AVX2+FMA/NEON) per output channel. Boundary check hoisted
  above gather so interior positions skip per-element if-guards. 99/99 unit tests pass.

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

- [x] **Move granite_vision_ocr to full Metal ggml graphs (vision + LLM)** —
  **DONE** (`fix/granite-vision-real`). The whole OCR pipeline now runs on the
  Metal GPU by default: SigLIP ViT (27L), projector, and the Granite-3.1-2B LLM
  body. Default OCR returns the correct text in **~22 s** (vision ~3 s, 784-tok
  prefill ~12 s, decode ~5 s) vs the scalar path's ~100 s vision + ~8 min prefill.
  Two bugs fixed (both mis-diagnosed in the prior handover, which claimed a
  shared "ggml-alloc in-place buffer-reuse defect"):
  - **ViT (`gv_run_vit_graph`)**: `ggml_reshape_2d` on the Q8_0 `ffn.down` weight
    to a non-block-aligned `ne[0]` (4304 % 32 ≠ 0) corrupted dequant from layer 0.
    Fix: cast quantized FFN weights to F32 before the reshape. Per-layer parity
    with scalar (cos 0.9996–0.99987).
  - **LLM (`gv_run_llm_body`)**: Metal's batched `mul_mm` casts activations to
    F16; the ×12-scaled image-feature "massive activations" overflow F16 in the
    SwiGLU `gate*up`→down matmul (NaN from layer 8). Fix: lossless activation
    scaling (÷256 before, ×256 after the down matmul).
  - **ggml-CPU ViT drift** (cos ~0.84 late layers): CPU `gelu` uses an F16 lookup
    table, and CPU `mul_mat` quantizes activations to the Q8_0 weight type — both
    accumulate over 27 layers. Fix: explicit F32 tanh-gelu + cast attention/up
    weights to F32 on the CPU backend only. CPU ViT now matches Metal/scalar
    (layer 26 cos 0.958) and CPU end-to-end OCR is correct.
  Both graphs now DEFAULT ON for **all** backends (Metal + ggml-CPU);
  `CRISPEMBED_GRANITE_VIS_SCALAR` / `_LLM_SCALAR` opt out. The LLM diff now
  exercises the graph (7/7 cos 0.9999). See LEARNINGS "Q8_0 reshape",
  "Metal mul_mm F16 activation overflow", and "ggml-CPU ViT precision".
  - **Decode perf**: LM head moved in-graph (Metal) + KV-history `ggml_cont`
    dropped → decode **270 → 165 ms/tok (~1.6×)**. Decode is now Metal-kernel
    -launch bound; one-shot total dominated by the 784-tok prefill + pipeline
    compilation. **Next lever (not done):** persistent T=1 decode graph
    (deepseek_ocr2's `build_persistent_decode_graph` + constant-shape KV views).
    See LEARNINGS "VLM/OCR decoder perf".

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
  - **ggml LLM decode: now CORRECT on Metal AND ggml-CPU (2026-06-21).** The
    earlier "emits EOS immediately on Metal" was NOT an alloc-reuse bug — it was
    Metal's batched `mul_mm` casting activations to F16, overflowing on the
    ×12-scaled image-feature massive activations (fixed via a lossless ÷256/×256
    exponent shift on the SwiGLU down activation). Combined with the ViT Q8_0-
    reshape fix and the ggml-CPU gelu/quant precision fixes, the full graph path
    (vision + projector + LLM) is now DEFAULT ON on both backends and produces
    "The quick brown fox jumps over 1234." `CRISPEMBED_GRANITE_VIS_SCALAR` /
    `_LLM_SCALAR` opt out. LLM-graph diff 7/7 cos 0.9999; decode optimized to
    ~139 ms/tok (in-graph Metal LM head + KV-cont removal + T=1 FFN-scale skip).
    See LEARNINGS "Q8_0 reshape", "Metal mul_mm F16 activation overflow",
    "ggml-CPU ViT precision", "VLM/OCR decoder perf".
  - Projector ggml graph is also correct on both backends (default on). Projector
    GELU is erf (`projector_hidden_act="gelu"`) → `ggml_gelu_erf`.
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

- [x] **Flash attention everywhere** — done for all compatible runtimes:
  - `decoder_embed.cpp`: **DONE** (`29d8a08`)
  - `bidirlm_vision.cpp`: **DONE** (`fd8cd09`)
  - `qwen2vl_ocr.cpp`: already had it
  - `lightonocr.cpp`: default since this session
  - `internvl2_ocr.cpp`, `got_ocr.cpp`, `glm_ocr.cpp`: already had it
  - `lilt_kie.cpp`: SKIPPED (BiACM incompatible with fused kernel)
  - `deepseek_ocr2.cpp`: pending (no q4_k model to test)

- [x] **Move remaining scalar encoders to ggml graphs**:
  - `deepseek_ocr2` Qwen2 encoder: **DONE** (`910d036`). 24-layer single graph.
  - `hmer_ocr` DenseNet encoder: **DONE** (`273969d`). ggml graph, 3x speedup.
  - `bttr_ocr` / `posformer_ocr` DenseNet: **DONE** (`7c6d8e1`). ggml graph, ~2x speedup.
  - `mixtex_ocr` Swin encoder: **DONE** (`2453e04`). Batched matmuls via ggml, 1.5x encoder speedup.
  - `ppformulanet_ocr` HGNetv2 CNN: **DONE** (`c058099`). ggml conv2d graph, 12x speedup.

- [ ] **Move SR/restoration engines from scalar conv to ggml graphs**:
  **All engines below are already implemented, numerically verified, and fully
  wired (converter + docs + C/CLI/Python/Rust/Dart/Server bindings). This task
  is purely a *backend* port** — swapping their nested-loop scalar conv forward
  for a ggml graph. A `[ ]` here means "works but still CPU-scalar" (each file's
  header says `(CPU-scalar)`), NOT "missing". Verify every port with the
  crispembed-diff harness (`test-<engine>-diff`, cos ≥ 0.99) before checking off.
  **The harness already exists** (`tests/test_<engine>_diff.cpp` +
  `tools/dump_<engine>_reference.py`, CMake-registered) for instructir, scunet,
  swinir, tbsrn, hat, adair — most scalar engines shipped already verified vs
  the real HF checkpoint (e.g. instructir cos=1.000000, hat cos=0.999968), so a
  port just has to keep that gate green. (Exceptions found while validating refs:
  scunet had a dequant-aliasing bug `4318f33`; swinir had a shifted-window mask
  sign bug — both fixed, see notes below.) `dat` and `text_sr` have the reference
  dumper but still need the `test_diff.cpp` + a CMake line added.
  Same pattern as DenseNet/HGNetv2 conversions: replace with ggml_conv_2d,
  ggml_pool_2d, ggml_mul_mat, ggml_norm. Ordered by ease × impact:
  - [x] `nafnet_denoise.cpp` — **DONE** (`b580e5c`). conv2d_ggml replaces all scalar convs.
  - [x] `esrgan_sr.cpp` — **DONE** (`4f1d052`). Full conv chain ggml graph, 6x speedup.
  - [x] `safmn_sr.cpp` — **DONE** (`09a6e02`). All 8 conv2d calls → conv2d_ggml.
  - [x] `restormer.cpp` — **DONE** (`69be268`); **output correctness fixed 2026-07-02**
    (`d54b304`, merged `67bbbb6`). The `69be268` wave shipped a *scrambled* conv
    weight layout (garbage output on both backends). Unlike instructir (official
    `gguf.GGUFWriter` → ggml `[KW,KH,IC,OC]`), restormer's custom converter writes
    conv weights raw as numpy `(OC,IC,KH,KW)`, so the correct kernel is a **plain
    reshape** of the raw bytes to `[KW,KH,IC,OC]` (no permute); pre-permute deleted.
    Also rewrote the ggml MDTA block graph (per-head + temperature + real L2-norm)
    and fixed BiasFree LN. Validated: gray σ=25 denoise 19.84→2.15, CPU==Metal,
    ggml==scalar, fox clean. See LEARNINGS / HISTORY 2026-07-02.
    - [x] **Guardrail — built & verified 2026-07-02** (`test/restormer-regression`).
      `tools/dump_restormer_reference_from_gguf.py` reconstructs the PyTorch
      Restormer from the GGUF (no `.pth` needed; canonical ffn_factor=2.66 — the
      GGUF's derived 2.64583 floors `int(dim*f)` short) and dumps input+output.
      `test-restormer-diff` vs that ref: **cos_min=0.999997, max_abs=2.4e-3 PASS**
      (erf-gelu vs C++ tanh-gelu). Added a `restormer` `diff_only` entry to
      `tests/regression/manifest.json` (threshold `output`≥0.99) and a `diff_only`
      branch to `run_one.py` (SR engines have no `--ocr` path). Runner verified
      end-to-end (takes diff-only path, no OCR). `restormer-ref.gguf` (98 KB)
      uploaded to `cstr/text-super-resolution-gguf`; the full runner now fetches
      model+ref from HF and PASSes the diff-harness (worst cos_min=0.999997).
  - [x] `instructir.cpp` — **DONE**. All 8 conv sites (intro/down/up/ending +
    5 per NAFBlock incl. DW conv) → per-conv `conv2d_ggml` (nafnet hybrid:
    convs on ggml, SimpleGate/SCA/ICB/PixelShuffle stay scalar). Verified
    test-instructir-diff cos=1.000000 max_abs=6.6e-4 vs self-consistent torch
    ref (`tools/dump_instructir_reference_from_gguf.py`); ref on HF
    `cstr/instructir-GGUF/instructir-ref.gguf`. Notes: this GGUF (official
    gguf.GGUFWriter) stores conv kernels in ggml order [KW,KH,IC,OC] so the
    4D permute is skipped; conv graph runs on the CPU sched (GPU conv_2d hits
    a Metal f32×f16 mul_mv pipeline-compile failure); F16 kernel cast required.
  - [x] `pan_sr.cpp` — **DONE** (`913b4f5`). 16× SCPA forward → single ggml_conv_2d
    graph (nearest upscale + bilinear ILR via ggml_interpolate). Verified
    test-pan-diff cos_min=0.999997 vs self-consistent torch ref; ref on HF
    `cstr/text-super-resolution-gguf/pan-ref.gguf`. `PAN_SR_SCALAR=1` opts out.
  - [x] `dat_sr.cpp` — 18× dual attention (spatial+channel), ~60-90G, medium.
    **VERIFIED + bug fixed; conv→ggml DONE but gated opt-in** (`DAT_SR_GGML_CONV=1`).
    The ggml conv path (conv_first, RG/body 3×3, upsample, AIM/SGFN depthwise via
    `ggml_conv_2d`/`_dw` on a CPU sched, kernel F16-cast so the im2col+mul_mat
    places on CPU) is verified pixel-perfect (output cos 0.999995 == scalar), but
    it's a **net slowdown** here (~1.56s vs ~1.48s/tile): DAT is attention-bound
    and its ~42 small scattered convs (mostly 36 depthwise across the 18 blocks)
    each pay per-conv graph-build + sched_alloc overhead. So the scalar conv path
    stays default; ggml is opt-in for benchmarking / a future batched-graph
    rewrite. (Contrast swinir/scunet — conv-heavy → ggml wins.) Built a *genuine* ref by
    running the real PyTorch DAT-light (`dat_arch.py`, timm/basicsr mocked) on
    weights reconstructed from `dat-light-x2-f32.gguf`
    (`tools/dump_dat_reference_from_gguf.py`). Found+fixed a real bug: Conv+BN
    fusion was silently **skipped on F32 models** — the init dequant lambda used
    `to_f32`, which returns `t->data` directly for F32 tensors and leaves its
    out-buffer empty, so the `!cw.empty()` fusion guard never fired and the BN in
    the AIM dwconv/channel/spatial-interaction branches was dropped (output cos
    0.9906 → **0.999995** vs genuine ref; all 20 stages — conv_first, block_0..17,
    output — now ≥0.99998). `test-dat-diff` (CMake-registered) compares the
    uint8 output against the ref clamped to [0,1]. Ref on HF
    `cstr/text-super-resolution-gguf/dat-ref.gguf`. The conv→ggml port is the
    easy instructir pattern (official 2D-flattened writer) + a dwconv → conv_2d_dw.
  - [x] `scunet_denoise.cpp` — **conv→ggml DONE**. All conv2d sites (head, tail,
    stride-2 downsample, and the conv1_1/conv1_2/conv_block.0/.2 inside every
    ConvTransBlock) plus the decoder ConvTranspose2d upsamples now dispatch
    through `scunet_conv`/`scunet_deconv` → `ggml_conv_2d` / `ggml_conv_transpose_2d_p0`
    on a dedicated CPU sched with persistent CPU-resident F32 kernels. The Swin
    window attention / layernorm / MLP stay SIMD-scalar. **6.7× faster**
    (5.15s → 0.77s on the 64×64 ref tile), all stages cos=1.000000 (incl. the
    `m_up*` deconv stages) vs the self-consistent ref. `SCUNET_SCALAR=1` opts out.
    Gotcha: unlike pan/swinir's PyTorch-order custom writer, SCUNet's GGUF stores
    conv kernels already in ggml-native order (Conv2d `[KW,KH,IC,OC]`,
    ConvTranspose2d `[KW,KH,OC,IC]`) — copy `t->ne` verbatim, NO axis reversal.
    Kernels are registered by the conv weight pointers held in the context (1×1
    convs have trailing ne=1 and are indistinguishable from 2D linears by shape).
    **NOTE**: while validating the reference, found+fixed a real pre-existing bug
    (`4318f33`) — swin_block_forward aliased qkv_w/proj_w/rpb (+ MLP weights) into
    shared dequant buffers `dq1/dq2`, so attention/MLP ran on garbage weights
    (output parity ~0.93). Now all stages cos=1.000000. Self-consistent ref at
    `cstr/scunet-GGUF/scunet-ref.gguf`.
  - [x] `swinir_sr.cpp` — **conv→ggml DONE**. The seven nested-loop conv sites
    (conv_first, 4× RSTB conv, conv_after_body, upsample) now dispatch through
    `swinir_conv` → `ggml_conv_2d` on a dedicated CPU sched with persistent
    CPU-resident F32 kernels (custom-writer GGUF stores PyTorch [OC,IC,KH,KW];
    reverse the 4 axes for ggml ne=[KW,KH,IC,OC], like pan). Swin window
    attention / norms / pixel-shuffle stay SIMD-scalar. ~1.9× faster per tile
    (5.5s → 2.85s on the 64×64 parity tile), output cos unchanged (0.999996 vs
    self-consistent ref; ggml vs scalar max-diff ~1e-5). `SWINIR_SR_SCALAR=1`
    opts out. RSTB + Swin window attention batched-matmul port still possible
    (the attention, not the convs, dominates the residual ~60-90G).
    **NOTE**: while validating the reference, found+fixed a real pre-existing bug
    — `cyclic_shift` used the wrong sign convention, so the forward shift was
    `roll(+ws/2)` while the precomputed `attn_mask` (and the reference) assume
    `roll(-ws/2)`. The mask then blocked the wrong token pairs in the wrap-around
    (edge) windows, so the shifted (odd-index) Swin blocks diverged at the image
    edges and compounded through the RSTBs (rstb_3 max_abs 147, engine ≈ 2× ref
    at edges). Fix: forward shift `+ws/2`, reverse `−ws/2` so partition+mask align.
    All stages now cos ≥ 0.99997, output (float) cos 0.999996. The earlier
    "−0.91 anti-correlated output" was an artifact of `test_swinir_diff.cpp`'s
    worst-row-of-3-adjacent-pixels metric on a uint8-clamped image (a single
    near-zero edge triple tanks it); the test now gates on the image-level
    (global + per-channel) cosine. Self-consistent ref (erf-GELU) generator at
    `tools/dump_swinir_reference_from_gguf.py`. Conv→ggml port still TODO.
  - [x] `hat_sr.cpp` — HAB + OCAB (unfold needed), ~70-100G, hard. **conv→ggml
    DONE** (default on, `HAT_SR_SCALAR=1` opts out). The six top-level convs
    (conv_first, per-layer `.conv`, conv_after_body, conv_before_upsample,
    upsample.*, conv_last) dispatch through `hat_conv` → `ggml_conv_2d` on a CPU
    sched (lazy name-keyed persistent F32 kernels, shape from the call-site dims,
    F16-cast in-graph). Window/OCAB attention + the small per-block CAB convs stay
    scalar. **Net ~1.3× (11.8s vs 15.3s/tile)** — the upsample/conv_last convs run
    at 4× resolution, so unlike attention-bound DAT the convs matter. Output cos
    0.999965 (ggml) vs 0.999968 (scalar) vs the validated HF `hat-ref.gguf`.
  - [x] `tbsrn_sr.cpp` — **conv→ggml DONE**. The six conv sites (block1 conv9,
    srb conv1/conv2 ×5, final_conv, upsample.conv, output_conv) dispatch through
    `tbsrn_conv` → `ggml_conv_2d` on a CPU sched with persistent reversed-ne F32
    kernels (custom-writer PyTorch [OC,IC,KH,KW] → ggml [KW,KH,IC,OC]; BN is
    already folded into the fused conv weights, so the kernel copies
    `ctx->get(...)`). PReLU/mish/FeatureEnhancer-MHA/LN/PixelShuffle/tanh stay
    scalar. `TBSRN_SR_SCALAR=1` opts out. Verified vs a self-consistent ref built
    from the GGUF (`tools/dump_tbsrn_reference_from_gguf.py`, reverses the
    converter rename + un-transposes the Linear weights): output cos 0.999362
    (ggml == scalar; no correctness bug — the engine was already correct). The
    FeatureEnhancer MHA, not the convs, dominates the residual ~15-25G.
  - [ ] `text_sr.cpp` — NAFNet variant + PixelShuffle + bicubic, ~40-60G, easy.
    **Blocked on a model, not code** — no public NAFNet text-SR checkpoint (registry
    URL empty; the Apache-2.0 `tbsrn` already covers text-SR). To train one with
    clean (Apache/MIT) data + a Kaggle-staged plan modeled on PosFormer, see
    [docs/text_sr_training_data.md](docs/text_sr_training_data.md).
  - [x] `adair.cpp` — U-Net + AFLB + FFT, ~100-150G, hard. **conv→ggml DONE
    (~5.2× per tile).** All conv sites (patch_embed, down/up/reduce, output, and
    the MDTA/GDFN/cross-attn/FreModule convs threaded through the block helpers)
    now run via `ggml_conv_2d` / `ggml_conv_2d_dw` on a dedicated CPU-backend
    scheduler. Kernel cache is keyed by the **dequantized weight pointer**
    (`dqc.get(...)` is stable per tensor), so `adair_conv` is a near-drop-in for
    the old `conv2d`; depthwise (`groups==oc==ic`) → `ggml_conv_2d_dw`. Kernels
    are F16-cast in-graph (conv = im2col(F16)+mul_mat; the CPU sched can't place a
    mul_mat with an F32 kernel). The 2D FFT (AFLB/FreModule `fft1d`/`fft2d`) and
    the attention softmax **stay SIMD-scalar**. Default ON; opt out with
    `ADAIR_SCALAR=1`. Measured on 64² tile: scalar 15441 ms → ggml 2951 ms; cos
    0.999379 (scalar) / 0.999385 (ggml) vs the genuine PyTorch-AdaIR ref — F16
    residual only, no regression. Ref built by running upstream
    `c-yn/AdaIR/net/model.py` (torch+einops) on weights reconstructed from
    `adair-5d-f32.gguf` (`tools/dump_adair_reference_from_gguf.py`); ref on HF
    `cstr/text-super-resolution-gguf/adair-ref.gguf`.

- [x] **Patch embedding conv → ggml matmul** — Most VLM runtimes now use ggml
  graph (internvl2, granite, smoldocling, qwen2vl) or im2col+matmul (got,
  lightonocr, pix2struct). Remaining: glm, deepseek (minor, scalar fallback).

- [x] **Pre-compute RoPE frequency tables** — Added `RoPEFreqTable` struct to
  `vlm_attention.h` with `precompute(head_dim, theta)` and `apply()` methods.
  Eliminates `powf` per-element. Migrated: smoldocling_ocr (NEGHALF),
  granite_vision_ocr (NEGHALF). Remaining `core_vlm` users still on `apply_rope()`.
  Unit tests: 4 cases covering identity, NEGHALF/INTERLEAVED parity, reuse (`65c282d`).

- [x] **Batch linear → GEMM in SR/restoration attention** — DONE. dat_sr
  (`a71c123`), swinir_sr (`dcf6556`), hat_sr (`b199741`), scunet (`52250ef`),
  mixtex (`816a88a`): replaced per-token scalar linear with
  linear_batch_cpu (SIMD), SIMD dot_product in attention.

- [x] **Sequential region recognition → batched** — `ocr_pipeline.cpp` now
  batch-encodes all detected crops in one ggml graph call
  (`math_ocr_encode_batch_raw`), then decodes sequentially
  (`math_ocr_decode_batch_crop`). Single [H, T, B] encoder graph replaces
  N×[H, T] invocations; fallback to sequential path if batch alloc fails.
  `table_parse` uses Tesseract/callback — not batchable; closed for now.

- [x] **Eliminate redundant image loading in orchestrator** — Pre-load image once
  per stage in the accept-gate loop, pass pixel buffer to all 9 VLM engines.
  Eliminates N-1 redundant JPEG/PNG decodes per multi-stage run.

- [x] **LSTM gate SIMD** — `tesseract_lstm.cpp` inner dot-product loops in both
  `lstm_forward` and `summ_lstm_forward` now use `core_cpu::dot_product()`.
  AVX2+FMA accelerated on x86-64, NEON on ARM.

- [x] **Sliding-window min/max pool** — Replaced O(K) per-pixel brute-force in
  `scan_cleanup.cpp` with monotonic deque sliding window — O(1) amortized per
  pixel. For K=51 this is ~50x fewer comparisons.

- [x] **Weight dequant caching in SR runtimes** — ALL DONE. Pattern-A (7 runtimes:
  hat_sr, swinir_sr, pan_sr, text_sr, nafnet_denoise, restormer, tbsrn_sr)
  migrated from `wbufs` to `core_cpu::DequantCache`. Pattern-B (instructir,
  adair) now use persistent `DequantCache` on context as well (`0c87d93`).
  esrgan and safmn already cache via their ggml graph (no scalar path).

- [x] **Migrate duplicated helpers to `core/cpu_ops.h`** — bttr_ocr, hmer_ocr,
  posformer_ocr: replaced duplicated conv2d/relu/layernorm/linear with
  `core_cpu` shared versions (SIMD-accelerated). Replaced per-context
  `dequant_cache` map with `core_cpu::DequantCache`. Kept unique helpers
  (maxpool, avgpool, apply_bn) as-is.

- [x] **deepseek_ocr2: single multi-layer encoder graph** — Qwen2 encoder
  (24 layers) now built as one ggml graph, eliminating 23 GPU↔CPU round-trips
  of the hidden state per encoder call. DONE (`910d036`).
  (The LLM decoder was already multi-layer since initial implementation.)

- [x] **glm_ocr / got_ocr: scalar downsample/merger → ggml** — DONE. glm `host_matmul`
  (lines 493-502) and got neck (lines 699-773) use scalar CPU for Conv+matmul
  projectors. Should be ggml graph ops.

- [x] **gliner_ner BiLSTM SIMD** — Gate computation now uses
  `core_cpu::dot_product()` (AVX2+FMA/NEON). ~3M MACs per timestep accelerated.

- [x] **LiteMLA graph implementation** — already done. `g_litemla` is fully
  implemented and `run_forward_graph` is the default path. Scalar fallback
  only via `SURYA_DET_SCALAR=1`. Linear attention: Q@(K^T@V) / (Q·K_sum).

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

- [x] **Graph caching** — parseq_ocr encoder graph built once and reused
  across calls (`c171c14`); math_ocr DeiT encoder graph cached per unique
  token-count T (`31e0c0e`). Eliminates per-call ggml_init + tensor creation +
  graph build. Remaining runtimes: TrOCR (variable-length decoder),
  VLMs (variable token counts).

- [x] **`ggml_gallocr` reuse** — moved gallocr from per-call to per-context
  for 7 engines: vit_embed, clip_text_embed, parseq_ocr, cnn_embed,
  ocr_detect, surya_det, layout_detect. Eliminates ~1-3ms malloc/free
  overhead per call; significant for small/fast models (DBNet 12M, PARSeq 24M).

- [x] **Native GQA in flash_attn (all VLMs)** — DONE. Removed `ggml_repeat`
  KV head expansion before `flash_attn_ext` in internvl2 (`7cffe56`),
  lightonocr, got_ocr, glm_ocr (`fbae7ba`). flash_attn handles GQA via
  broadcast factors (rk2 = neq2/nek2). -76 lines total.

- [x] **internvl2: cache vision graph across tiles** — DONE (`c714758`).
  Vision encoder graph built once on first tile, reused for all subsequent
  tiles. Eliminates per-tile graph build + sched alloc overhead.

- [x] **Eliminate redundant CHW↔HWC layout conversions** — post SIMD
  linear_batch refactor the remaining layout switches in `dat_sr.cpp` are
  minimal and inlined; no material gain from further restructuring. Closed.

- [x] **Pre-compute attention masks and position biases** — Already resolved:
  swinir_sr masks loaded from GGUF model file (stored as tensors, cached via
  DequantCache). hat_sr has no runtime mask computation. dat_sr position bias
  depends on spatial dims which change per tile — not precomputable.

- [x] **Fuse BatchNorm into conv weights at model load** — TBSRN: fused 11
  conv+BN pairs (2 per SRB × 5 + 1 final) at init. DAT: fused 54 conv+BN
  pairs (3 per AIM block × 18 blocks) — dwconv, channel_interaction, spatial_interaction.

- [x] **qwen2vl: token embedding via direct read** — DONE. Embed is now part of
  the main LLM graph (ggml_get_rows runs on GPU). lightonocr uses direct
  tensor read (embed_tokens_cpu).

- [x] **lightonocr: decode graph reuse** — DONE (`27b650a`). `LocPdGraph`
  struct + `build_locr_pd_graph(ctx, max_kv)`: fixed-size KV tensors
  `[kv_dim, max_kv-1]` per layer + F16 mask `[max_kv, 1]`. First step
  uploads prefill KV; subsequent steps write only the new K/V row and unmask
  one mask slot. Gate: `LOCR_DECODE_REBUILD=1`.
  deepseek: pending (no local q4_k model for timing).

- [x] **qwen2vl: F32 causal mask → F16** — already F16 (GGML_TYPE_F16)
  (half the memory).

- [x] **gliner_ner: DeBERTa relative position expansion** — DONE (`a63875c`).
  Two-level cache in `gliner_context`: (1) `rel_embd_norm` — LN-normalized
  embedding weights, once per context; (2) `rel_pos_expanded_cache` — T×T
  expansion, reused when T is unchanged. Eliminates 117 MB alloc+fill per call
  for fixed-window workloads.

- [x] **Pre-compute 2D positional encoding** — TBSRN: cached at init (fixed
  16×64 dims, reused across 5 SRB blocks). BTTR/PosFormer: cached for
  last-used (h, w) — skips ~327K sinf/cosf evals on repeated calls.

- [x] **mel.cpp: OpenMP on STFT loop** — DONE (`8242a67`). Each frame's FFT is independent
  (line 73-84). `#pragma omp parallel for` on the `t` loop.

- [x] **mel.cpp: SIMD for mel projection** — DONE. Float MelsFreqs layout uses
  `core_cpu::dot_product()` (AVX2+FMA/NEON). Double-precision path kept scalar.

- [x] **gguf_loader: `madvise(MADV_SEQUENTIAL)`** — already done (line 244).
  Also has MADV_WILLNEED (line 247).

- [x] **gguf_loader: `std::unordered_map` for tensor lookup** — DONE
  (`0777f30`, `f98358e`). Replaced std::map with std::unordered_map in
  WeightLoad and all model files; try_get/require moved to concrete .cpp
  functions. O(1) avg lookups.

- [x] **instructir: SCA weight dequant inside per-channel loop** — DONE
  (`06b3190`). Hoisted sca_w/sca_b dequant outside per-channel loop.

- [x] **Otsu threshold: extract shared utility** — Added
  `core_cpu::otsu_threshold()` to `cpu_ops.h`. Replaced duplicated
  implementations in cc_detect, table_parse, classical_preproc, dewarp.
  scan_cleanup float variant kept separate (different input type).

- [x] **OpenMP in pixel-level ops** — DONE (`af920b8`). Parallelized
  `image_preprocess` (bicubic resize + normalize), `dewarp` (apply_warp),
  `face_align` (affine warp). `scan_cleanup` uses sliding-window deques
  (data-dependent, not parallelizable).

- [x] **pcs: cache FC weights at load** — DONE. All 17 FC head tensors cached
  in `fc_cache` struct at init. No per-call `ggml_backend_tensor_get`.

- [x] **restormer: dead `rst_gdfn()` stub** — DONE (`06b3190`). Removed.
- [x] **restormer: `rst_layernorm_bf` computes variance twice** — DONE
  (`06b3190`). Removed dead first-pass sum-of-squares.

#### P3 — Nice-to-have / minor

- [x] **bpe.h: priority queue for BPE merges** — DONE (`eae73de`).
  Linked list + min-heap, O(N log N). Both bpe.h and tokenizer_bpe.cpp.
- [x] **tokenizer_bpe.cpp: same O(N^2) merge issue** — DONE (`eae73de`).

- [x] **tokenizer.cpp: trie for WordPiece** — DONE (`cce0fc1`). Two-root trie
  (first-piece + continuation) built at load. O(len) longest match vs O(len²).
  Output parity verified: MiniLM-L6-v2 produces identical embeddings.

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

- [x] **Nearest-neighbor → bilinear resize** — DONE (`12c12a1`). Upgraded
  math_ocr, mixtex, ppformulanet, ppformulanet_l, surya_det, parseq_ocr
  from integer-truncation to bilinear interpolation.

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

- [x] **Add beam search to math OCR runtimes** — `bttr_ocr` already had it.
  Added `*_recognize_beam` / `*_recognize_raw_beam` API variants (`1f58e83`)
  to math_ocr (scalar MathOcrBeam via decoder_step_scalar), ppformulanet_ocr,
  and ppformulanet_l_ocr. Remaining greedy-only: hmer (GRU decoder, different
  state), posformer (ARM coverage complicates beam copies), mixtex.

- [x] **morph_fast: decomposed dilation** — DONE (`825db30`). Power-of-2
  horizontal dilation replaces O(hsize) naive loop for hsize > 16.
  hsize=30 (cc_detect): ~3x; hsize=31-200 (table_parse line detection): 3-14x.

- [x] **pdf_info: mmap instead of full file read** — DONE (`5f027aa`).
  Memory-mapped on POSIX with MADV_SEQUENTIAL, fread fallback on Windows.

- [x] **tps_warp: coarse grid + bilinear interpolation** — DONE (`b142249`).
  Pre-computes displacement on 8-px grid (O((W/8)*(H/8)*N)) then bilinearly
  interpolates per pixel. All 19 unit tests pass.

- [x] **Debug fprintf gating (layout_detect, surya_det, ocr_detect)** — DONE.
  layout_detect: ~30 unconditional printfs → LDBG() macro (`614132e`).
  surya_det: backend-selection print gated behind `dump` (SURYA_DET_DUMP).
  ocr_detect: per-call resize print gated behind `bench` (CRISPEMBED_OCR_DETECT_BENCH).

- [x] **hmer coverage conv per step** — conv2d(256, 256, 3x3) is the Bahdanau
  coverage attention mechanism; cannot be eliminated without changing the
  architecture. Item closed as won't-optimize.

- [x] **ppformulanet_l: ggml meta buffer reuse across layers** — DONE
  (`b7bc237`). Hoisted 8MB meta_buf before 12-layer loop.

- [x] **math_ocr: global dequant cache → per-context** — already done.
  Uses `core_cpu::DequantCache dcache` per-context (line 93).

- [x] **Remove dead scalar fallback encoder in ppformulanet_l** — DONE
  (`c7bd92c`). Removed 370 lines of unused scalar encoder code.

- [x] **cpu_ops.h: SIMD layernorm_cpu** — AVX2+FMA for mean/var/scale+shift.
  Used by 12 engines. 99/99 unit tests pass.
- [x] **cpu_ops.h: SIMD rmsnorm_cpu** — AVX2+FMA for sum-of-squares + scale.
  Used by 12 engines. 99/99 unit tests pass.
- [x] **cpu_ops.h: SIMD softmax** — AVX2 max-reduction + normalization.
  99/99 unit tests pass.
- [x] **cpu_ops.h: mha_1q_cpu cache-friendly V accumulation + SIMD** —
  ki-outer loop (sequential V row) + AVX2+FMA 8-wide fmadd.
  99/99 unit tests pass.

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

**Remaining:** none — all major optimizations complete. Decode graph reuse done (`27b650a`).

### qwen2vl — DONE (perf optimized; OCR correctness fixed 2026-07-02)
  F16 KV cache, flash attn, ggml patch embed, direct embed lookup, F16 mask — all done.
- [x] **OCR correctness (2026-07-02)**: Qwen2.5-VL (`qwen2.5-vl-3b`) hallucinated a
  fabricated description instead of reading the image; `expected_text` was `null`
  (a never-validated path, not a ggml-wave regression). fox.png → "The quick brown
  fox jumps over the lazy dog. 12345", CPU AND Metal, q4_k. Four bugs:
  1. Vision 2D RoPE built in raster order while the preprocessor always emits
     patches in merge-block order (HF `rot_pos_emb` permutes ids identically) —
     the `merge_order` flag keyed off `is_qwen2_vl`. Now **unconditional**
     merge-block (see the gate-correction note below). Dominant bug.
  2. CPU spatial merge grouped patches via the raster else-branch for Qwen2.5-VL
     (mis-groups merge-block data; deepstack already assumed consecutive). Now
     unconditional consecutive.
  3. Windowed attention was unimplemented — `window_size`/`fullatt_block_indexes`
     loaded but never used. Added as an in-place additive mask (0 within a window,
     -inf across) via `soft_max_ext` on non-fullatt blocks; full blocks keep
     flash_attn. Opt-out `QWEN2VL_OCR_NO_WINDOW=1`. (Qwen3-VL is full-attention —
     correctly excluded via `is_qwen2_vl=true`.)
  4. arch `qwen2vl` got no OCR prompt (only `qwen3vl` did) → default "Describe this
     image." → verbose prose. OCR prompt now applied to both archs.
  - **Gate correction (same day):** the first fix (`86d0830`) gated rope order +
    merger grouping on `deepstack_indexes.empty()`, assuming Qwen3-VL was
    `is_qwen2_vl=false`. It's `is_qwen2_vl=true` (LayerNorm ViT) *with* deepstack,
    so that gate flipped it from the correct merge-block path to raster →
    **regressed Qwen3-VL to garbage OCR**. `patchify_qwen_layout` emits merge-block
    order for every family member, so rope order + merger grouping are now
    **unconditional**. Verified `qwen3-vl-2b` AND `qwen2.5-vl-3b` read the fox line
    on CPU and Metal.
  - Per-stage HF ref (`qwen2.5-vl-3b-ref.gguf`) not yet regenerated/uploaded —
    verified end-to-end transcript on both backends.
  - History: HISTORY.md (July 2, 2026). Deep-dive: LEARNINGS "qwen2vl-3b
    hallucinated OCR — RESOLVED".

### deepseek_ocr2 — DONE (OCR correct; perf-sweep regression reverted 2026-07-02)
- [x] Character-perfect OCR on Metal + q4_k (SAM ViT-B → Qwen2 24L encoder →
  linear projector → DeepSeek-V2 MoE decoder). Verified q4_k (HF rev
  `a465ab6cf4b5`): fox.png + a document page read verbatim on M4 Metal.
- [x] **Perf-sweep regression reverted (2026-07-02)**: the Jun-20 perf sweep
  garbled OCR on both backends (bisected to `c75b95d` flash_attn-in-Qwen2-encoder
  + decode-degeneration commits, all ungated/untested). Restored
  `deepseek_ocr2.cpp` to the last-good-and-fast commit `c58913c` (keeps the Metal
  vision speedups). Regression-manifest entry added (rev-pinned). Deep-dive:
  LEARNINGS "Perf-sweep regression". HISTORY: July 2, 2026.
- **Perf re-adds — MEASURED NOT WORTH IT (2026-07-02), branch
  `perf/deepseek-ocr2-gated-readd`.** Profiled (`DS_PROFILE=1`) on M4 Metal: the
  decode is **98% compute-bound** — graph build+alloc = 35 ms vs compute = 2270 ms
  over 31 tokens (build is **2%**). So the reverted sweep's overhead-reduction
  paths yield ≤~1.5% here:
  - **flash_attn** (encoder+LLM): re-added behind `DS_QWEN2_ENC_FLASH` /
    `DS_LLM_FLASH`, A/B byte-identical but **~20% SLOWER** on Metal (small T) →
    kept opt-in, NOT default.
  - **persistent single-graph decode** (patterns 2+3 from qwen2vl/qwen3vl): would
    save the 2% graph-build → ~1.5% end-to-end. Not worth the refactor+risk.
  - **backend F16 KV**: ≤63 MB even at max context; no speed win (compute-bound).
  The real perf lever is the **MoE compute** (expert dispatch / quantization),
  not graph/upload overhead — the sweep optimized the wrong thing. Conclusion:
  keep the correct c58913c baseline + the verified flash opt-in gates; do NOT
  re-add the rest. `DS_PROFILE` instrumentation retained for future profiling.
  Multi-view preprocessing (global + dynamic crops) is still unimplemented
  (single global view only) — a known simplification, orthogonal to perf.
- **MoE compute optimization — PENDING (the only real deepseek perf lever).**
  Profiling proved decode is ~98% compute-bound, dominated by the DeepSeek-V2 MoE
  (12 layers, 64 routed experts top-6 + 2 shared, `moe_intermediate_size` 896).
  Decode ≈ 2270 ms / 31 tokens ≈ 73 ms/token on M4 Metal. **Already efficient:**
  both the Metal path (`ggml_mul_mat_id`) and the CPU fallback (`moe_ffn_cpu`)
  dispatch/dequant ONLY the top-6 (+shared) experts per token, not all 64 — so the
  obvious win is done and the ~73 ms/token is largely inherent. Remaining levers
  are HARD/upstream (each behind an env gate + A/B on decoded OCR + timing on a
  LONG-decode input before flipping the default):
  - [ ] **Small-batch `mul_mat_id` on Metal (T=1).** 12 layers × 3 matmul_id ×
    6 experts on a single token = many tiny matmuls, likely dispatch-overhead-
    bound. First split MoE vs attention within the 2270 ms (`DS_DBG=1` prints
    per-layer `attn`/`moe` ms). A fused/better MoE kernel is mostly upstream ggml.
  - [ ] **Expert quantization knob** — confirm expert FFNs are Q4_K; test
    Q4_K vs Q5_K/Q6_K on decoded OCR (CER) vs size/speed (size/quality, not a big
    Metal speed win).
  - [ ] **Fuse shared+routed FFN** dispatches; confirm router softmax/top-k isn't
    a serial CPU hop between GPU dispatches.
  - [ ] **Metal hazards** if touching the matmul — F16-overflow (scale 1/256) +
    residency-abort classes (see those LEARNINGS entries).
  Only pursue if decode latency is a real concern; the graph-overhead paths
  (persistent decode / F16 KV / single-graph) are NOT the lever (measured ~2%).
  **Full self-contained handover: `handover-prompts/deepseek-ocr2-moe-perf.md`.**

### got_ocr (SAM ViT-B + Qwen2-0.5B, 0.7B) — DONE
- [x] Patch embedding → ggml matmul (same im2col pattern, scalar fallback gated)
- [x] Neck+downsample+projector → ggml graph (conv2d_direct + LN2d via permute+norm + mul_mat)
  Gated: CRISPEMBED_GOT_OCR_SCALAR_NECK=1 / CRISPEMBED_GOT_OCR_SCALAR_PATCH=1

### glm_ocr (glm_ocr_vision ViT + GLM-0.5B decoder, 0.9B) — DONE (OCR correct 2026-07-01)
- [x] Downsample + merger → ggml graph (conv2d_direct + batched SwiGLU + LayerNorm)
  Gated: CRISPEMBED_GLM_OCR_SCALAR_MERGER=1. Merger: 383ms on q4_k.
- [x] **OCR correctness (2026-07-01)**: was garbage; fixed 5 bugs + q8_0.
  fox.png → "The quick brown fox jumps over the lazy dog. 12345" on f16, q8_0 AND
  q4_k, CPU AND Metal, verified vs the real transformers-`main` `glm_ocr` model.
  1. Missing vision 2D RoPE (raster order, dim=hd/2, θ=10000, NEOX). On by default;
     `GLM_OCR_VISION_ROPE=0` to disable.
  2. Merger structure: was `proj→SwiGLU→LN`; real is
     `proj→LayerNorm→GELU(erf)→down(silu(gate)·up)` (no trailing norm).
  3. Dynamic resolution (Glm46VImageProcessor smart-resize, min/max pixels, dims
     ×28) — NOT fixed 336². Variable grid flows to rope/merger/prompt/LLM mRoPE.
  4. LLM image mRoPE positions (start-offset h/w; decode from compressed pos).
  5. Prompt (`[gMASK]…Text Recognition:…`), EOS on `<|user|>`(59253), GPT-2
     byte-level decode.
  - q8_0 & q4_k: dequantize weights BEFORE any reshape (downsample/patch_embed/
     merger) — fixes CPU garbage + Metal block-align assert.
  - Regression uses `expected_text` (per-token cos_min diff gate unsuitable — see
     LEARNINGS "glm-ocr: five real bugs" for the sink-token analysis).
  - History: HISTORY.md (July 1, 2026).
  - The handover `handover-prompts/glm-ocr-vision-rope-fix.md` was partly wrong
    (named glm4v; its refs were stale no-rope dumps).
### granite_vision — DONE (full ggml graph path, Metal + ggml-CPU)
- [x] Weight reshape fix: `sw()` helper in `gv_run_llm_body` corrects PyTorch [out,in]→ggml [in,out] for K/V/gate/up/down
- [x] Skip LM head matmul during scalar prefill: `want_logits` parameter cuts ~99.8% of prefill LM head work (`55ed5be`)
- [x] Native GQA in flash_attn: pass K/V with n_kv heads directly to flash_attn_ext (`b579345`)
- [x] **ViT graph fix**: Q8_0 `ffn.down` reshape to non-block-aligned ne[0] corrupted dequant → cast quantized FFN weights to F32 before reshape (`a5b527f`)
- [x] **LLM Metal fix**: batched `mul_mm` F16-cast overflow on ×12 image-feature massive activations → ÷256/×256 exponent shift on SwiGLU down (`52400a6`). The old "alloc-reuse / EOS on Metal" diagnosis was wrong.
- [x] **ggml-CPU ViT precision**: F16-table gelu + Q8_0-quantized activations → explicit F32 tanh-gelu + CPU-only F32 weight casts (`2dc3b79`). CPU now at parity (layer 26 cos 0.958).
- [x] **Default ON both backends**; graphs validated (LLM diff 7/7 cos 0.9999), end-to-end OCR correct on Metal AND CPU.
- [x] **Decode perf** (`bfe3ad2`/`f42b737`): in-graph Metal LM head + KV-cont removal + T=1 FFN-scale skip → 270 → 139 ms/tok (~1.9×).
- [ ] Persistent decode graph — investigated and **declined**: profiling shows a T=1 token is ~95% GPU compute (build+alloc ~5ms of ~140ms), so it's not the bottleneck. See LEARNINGS "VLM/OCR decoder perf".
- [x] **Tokenizer packaging fix (2026-07-02)**: uploaded GGUFs shipped `tokenizer=MISSING (0 tokens)` → OCR emitted raw token IDs. Folded the BPE tokenizer + scalars into `convert-granite-vision-to-gguf.py` (complete-gguf converts), made `patch-granite-gguf-tokenizer.py` idempotent, and re-patched/re-uploaded q4_k/q8_0/f16 to `cstr/granite-vision-crispembed-GGUF`. Banner now `tokenizer=embedded (49156 tokens)`; OCR readable on CPU+Metal; regression `expected_text` baked (max_cer 0.15).

### smoldocling (SigLIP + SmolLM2, 256M) — DONE
- [x] F16 KV cache + batched prefill (done earlier, `bc329e4`)
- [x] Patch embedding → ggml matmul (im2col + mul_mat, F16 bias cast)
  Gated: CRISPEMBED_SMOLDOCLING_SCALAR_PATCH=1
- [x] LLM decoder → ggml graphs — DONE. Was already implemented but blocked by
  F16 norm weight type mismatch on Q4_K models. Fixed with ggml_cast (`91b1f89`).
  Tested: prefill=2.3s, decode=62s (128 steps).
### internvl2 — DONE (already optimized)
  F16 KV cache, flash attn, ggml patch embed, ggml vision graph — all done.
  Native GQA (`7cffe56`) and batch vision tiles (`c714758`) completed.
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

### unlimited_ocr (SAM ViT-B + CLIP-L/14 + DeepSeek-V2 MoE, 3B) — IN PROGRESS

**Baseline** (test-1.jpeg 640×488, q4_k, 4 threads, 64 tokens):
  load=21.7s, SAM=63.9s (patch=0.3s, layers=63.2s, neck=0.3s), CLIP=3.4s,
  fuse=1.0s, LLM decode=35.3s (prefill+64gen, 551ms/tok), total=127.2s

**Comparison** (llama.cpp PR#24975 Q4_K_M + bf16 mmproj, same image/tokens):
  total=287.3s (vision=120.3s — bf16 mmproj causes swap pressure)
  CrispEmbed 2.3× faster on 8GB VPS due to smaller q4_k vision encoder.

**Optimizations** — each gated by env var for A/B testing:

- [x] **RPE cache** — Precompute reformatted RPE tables at init.
  `reformat_rp_table()` now runs in `precompute_rpe_tables()` at init;
  metadata buffer hoisted outside SAM per-layer loop. ~0ms runtime gain
  (reformat was already <1ms per layer) but eliminates allocations.

- [x] **ggml MoE (default, `UOCR_MOE_CPU=1` opts out)** — `ggml_mul_mat_id` for
  MoE layers. Already the default. Expert stacking runs at init (3s).
  **A/B tested: decode 5.2× faster** (1003ms/tok vs 5176ms/tok scalar).
  32-token test: 32.1s (ggml) vs 165.6s (CPU scalar).

- [x] **`UOCR_OPT_GRAPH_LN=1`** — Move CPU layernorm into ggml graph for windowed
  SAM layers. Implemented: when enabled, windowed layers skip CPU LN and let the
  ggml graph handle it. Output differs slightly in bounding box coordinates
  (ggml vs CPU LN precision) but OCR text is identical. Gate: `UOCR_OPT_GRAPH_LN=1`.

- [x] **`UOCR_MMAP=1`** — mmap weight loading. Already implemented. Load drops
  from ~20s to ~0.2s. Expert stacking slower with mmap (28s vs 3s — piecemeal
  reads). Overall neutral on swap-heavy 8GB VPS. Correct output verified.

- [ ] **`UOCR_PD=1`** — Persistent T=1 decode graph. Investigated extensively:
  - Added F32 KV cache (`UOCR_OPT_PD_F32=1`), manual matmul attention (replaced
    flash_attn), F32 flash_attn precision, CPU embedding (replaced ggml_get_rows)
  - Per-layer debug dumps show **divergence starts at layer 0 of gen step 2**
  - Root cause: even with minimal padding (2 unused KV slots), flash_attn on CPU
    produces slightly different results with padded vs exact-size KV tensors.
    The difference is small (~0.01 abs) but accumulates across 12 layers and
    changes the argmax by step 3.
  - **Impact if fixed**: ~80ms/step saved (build+alloc overhead), ~14% decode
    speedup for 64 tokens. Not critical.
  - Would need: either ggml_view-based dynamic KV slicing (ggml doesn't support
    dynamic ne on allocated tensors), or accepting the numerical drift.
  - Debug env vars: `UOCR_PD_DBG=1`, `UOCR_DECODE_TIMING=1`

- [ ] **`UOCR_OPT_GGML_WINDOW=1`** — Window partition in ggml graph. High effort,
  requires ggml_view/pad ops for scatter/gather. SAM is compute-bound (not
  data-movement-bound), so savings would be ~2-5% of SAM time. **Deferred.**

- [x] **`UOCR_OPT_SAM_RES=N`** — Reduced-resolution SAM (replaces SAM_512).
  Implemented: position embedding bilinear interpolation, RPE recomputation for
  new sizes, bilinear upsample of SAM output to 16×16 for CLIP compatibility.
  **Results**: 512=5.5x SAM speedup but degenerate output (repeating tokens).
  768=2.2x speedup, still poor quality. 896=1.2x speedup, somewhat better but
  coordinates wrong. **Conclusion: SAM is resolution-sensitive.** The model was
  trained at 1024 and reducing resolution degrades attention patterns.
  Use only if quality-tolerant (e.g. rough layout detection, not OCR).

- [ ] **SAM flash attention** — Blocked by decomposed RPE bias (rel_h + rel_w
  added to scores before softmax). Would need to materialize [T,T] bias mask,
  defeating the O(T) memory benefit. **Won't implement.**

- [x] **`UOCR_OPT_FUSED_DECODE=1`** — Fuse all 12 LLM layers into a single ggml
  graph per decode step. Eliminates 11 graph builds + 11 sched allocs per step.
  **Verified correct**: output matches baseline exactly (same gen_ids).
  Decode 41.5s vs 138.4s baseline on loaded system — both swap-affected but
  fused avoids per-layer sched overhead. Build overhead drops from 80ms×12
  to ~80ms×1 per step. Requires `ctx.moe_metal` (stacked experts).
  Note: uses ~3x more graph metadata memory (32MB vs 4MB) for the fused graph.

**Implementation order**: RPE cache ✓ → ggml MoE ✓ → graph LN ✓ → mmap ✓ → fused decode ✓ → PD (deferred)

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
