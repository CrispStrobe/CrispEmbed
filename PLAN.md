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

- [x] **Do-no-harm fix (blackfilter sharpness gate).** A heavy but still-readable
  dark vignette/stain was being DESTROYED: blackfilter cleared the stain's solid
  dark core to white and whitening smeared the leftover gradient edge to black
  (`darkvignette` harness case, CER degraded 0.561 → cleanup **0.474 + a black
  amoeba over the text**). Fix: blackfilter now clears a dark region only if it is
  bordered by BRIGHT paper (a sharp-edged shadow/blob); a soft dark gradient is
  bordered by more gradient, so it is left for the whitening step. Result:
  darkvignette 0.561 → **0.006** (pristine), shadow still cleared (0.075 ≈ prior
  0.071), all other cases unchanged. Added the `darkvignette` degradation to the
  harness as a permanent do-no-harm guard.

- [x] **Consensus deskew + per-params deskew on all image paths (2026-07-06,
  `ce7f1c4`).** The Hough-energy angle is now cross-checked against the
  independent differential-square-sum detector (`classical_preproc.h`) before
  rotating — sign agreement + a 1.5° magnitude band (DSS is opposite-signed and
  overestimates by ~0.5–1.2° depending on resolution); conflict → no rotation.
  New `deskew_consensus` param (default on) mirrored through the C ABI, Python
  and Rust bindings. New `scan_cleanup_deskew_rgb` (consensus detect + bilinear
  rotate, channel-preserving, white fill) gives every image-*embedding* path an
  optional per-params deskew, off by default: `image_preproc::config.deskew`
  (covers `encode_image_file` / `encode_text_with_image_file` /
  `preprocess_image[_rgb]` via `crispembed_set_image_deskew`),
  `vit_embed::set_deskew` / `crispembed_vit_set_deskew`, CLI `--deskew`, and
  granular cleanup toggles (`--no-deskew`, `--no-crop-borders`, `--no-whiten`,
  `--binarize`, `--no-deskew-consensus`, `--deskew-max-angle`). Verified: 22/22
  in `test-scan-cleanup`; a 3°-skewed synthetic scan comes out at 0.00°
  residual; library preprocess grid changes only while the toggle is on. Also
  fixed: Rust `from_stages` `ScanCleanupParams` literal was missing the four
  despeckle/blackfilter fields (did not compile).

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

**Reverse interop (import a stock llama.cpp mmproj INTO CrispEmbed):** shipped +
validated for the three rows where both a working CrispEmbed loader and a
downloadable mmproj exist — `qwen2vl_merger`, `idefics3`, `internvl` — via the
auto-detecting `models/merge-llamacpp-gguf.py` (see the status block below and
README "Importing a stock llama.cpp VL model"). Qwen2-VL is bidirectional
(export too). The rest need either a non-crashing dynamic-preproc loader
(`glm4v`) or an mmproj llama.cpp doesn't ship (`GOT-OCR2`).

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
  `-imatrix-ab.txt`. **31 models done** (batch 1: 20 below; group 2: 10 more —
  all-MiniLM-L6/L12, all-mpnet, gte-small, arctic-embed-xs, snowflake-arctic-m/l,
  paraphrase-multilingual, harrier-270m/0.6b; + embeddinggemma once fixed). The 4B/8B
  decoder embedders (octen/qwen3-embed) run via the **big-base path** (calibrate/gold
  on the q8_0 that fits Kaggle RAM, quantize from the f32 base, stage in `/tmp`).
  cos vs full-precision gold, Kaggle CPU; q8_0 ~0.9998 reference; ★ = IQ4_XS wins:

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
- **All embedders DONE (2026-07-03).** The 4B/8B big-base runs are complete:
  octen-4b (q4_k+im 0.9889), qwen3-embed-4b (0.9881), octen-8b (0.9902),
  qwen3-embed-8b (0.9934) — all cos-vs-q8_0. qwen3-embed-8b was the last of the
  roster. **Caveat found + fixed:** the first pass on the big decoders produced
  *empty* imatrix files (`-q4_k-imatrix.gguf` bit-identical to baseline) because
  the collector flushed via `atexit`, which the CLI's `clean_exit()`/`_exit()`
  skips — see `LEARNINGS.md → "The collector wrote nothing"`. Re-quantized after
  the `crispembed_free()` flush fix (commit 07439db).
- **Registry: optimally wired for every model.** Each auto-download default in
  `model_mgr.cpp` resolves to its **max-cosine A/B flavor** (decoder embedders →
  q4_k+imatrix, BERT/XLM-R encoders → iq4_xs+imatrix), with
  `-q4k`(imatrix)/`-iq4xs`/`-q8` aliases; f2llm-v2-0.6b + nomic-embed-text-v1.5
  kept at q8_0 (both collapse to <0.91 at 4-bit even with imatrix). Verified by
  cross-checking all 30 defaults against the uploaded `-imatrix-ab.txt` files.
- **All 38 dense embedders DONE (2026-07-03).** Backfilled the last three —
  granite-embedding-278m (q4_k+im 0.9960), granite-embedding-107m (iq4_xs 0.9935),
  gte-modernbert-base (iq4_xs 0.9892) — defaults repointed to their max-cosine
  flavor.
- **C1b — rerankers DONE (2026-07-03).** Extended the harness with a `rerank`
  MODE: calibration runs the `--rerank` path (collector fires unchanged), A/B =
  mean **Kendall-tau** on the doc ranking vs full-precision gold (+ mean|dscore|
  tiebreaker). All 7 rerankers quantized. Defaults: jina-v2 → q4_k+im,
  ms-marco-L6/L12 → iq4_xs (τ=1.0); bge + mxbai stay q8_0 (τ<1.0 at 4-bit). The
  mxbai DeBERTa rerankers only became quantizable after the `rel_embd` dequant fix
  (commit 73a016e — a raw-F32 read of a quantized 2-D weight aborted them on ANY
  quant; also unblocks gliner-deberta NER).
- **C1b-finalize (2026-07-04) — reranker A/B on a finer corpus + a shipped-broken
  model fixed.** Expanded the paired corpus to 16 EN+DE queries × 6 docs (graded
  relevance → real score spread; `tools/kaggle/.../imatrix_quant.py` RERANK_EVAL).
  - **BUG: `bge-reranker-base` was shipped HEADLESS.** Its GGUFs were converted as a
    plain XLM-R *encoder* (197 tensors, `pooling_type`, **no classifier head**) — the
    CLI correctly refused: "model is not a cross-encoder reranker". So the old
    "τ<1.0 keep q8_0" call was measured on a non-functional model. Reconverted with
    the head-aware converter (2-layer `classifier.dense`+`out_proj` now baked in),
    re-uploaded to `cstr/bge-reranker-base-GGUF` (verified on HF: q8 header has the
    classifier). Same headless-model trap as splade-pp. **Audited all 7 rerankers:
    only bge-reranker-base was affected**; v2-m3, jina, mxbai×2, ms-marco×2 all have
    heads.
  - **Full cross-family head audit (2026-07-04, tokenless gguf-header curl).** Every
    task-head model verified to carry its required head: bert-base-NER + xlmr-ner-hrl
    (`ner.classifier`), splade-pp (`mlm_transform`), gliner-deberta + gliner-lfm
    (`prompt_rep`/`span.out_project`), lfm2-colbert (`colbert.projection`, `--colbert`
    1024→128 confirmed), and embeddinggemma's Matryoshka `dense.0/1` (768-dim output
    confirmed). **bge-reranker-base was the ONLY affected model across all families.**
    (Header read must exceed the tokenizer-vocab KV — ~9 MB for 30–250 K vocabs,
    ~20 MB for Gemma's 256 K — or the tensor-info section is missed → false HEADLESS.)
    **Automated as `tests/audit_gguf_heads.py`** (manifest-driven, tokenless HTTP
    range reads, stdlib-only, exits 1 on any missing head; negative-tested). Run it
    as a release gate after any converter change or re-upload; it currently passes
    all 14 task/projection-head models.
  - **imatrix does NOT help rerankers.** On the working bge-reranker-base (τ vs f16,
    16×6): q8_0 **1.000** (top1 16/16) → q4_k 0.967 → q4_k+imat 0.958 → iq4_xs 0.942.
    4-bit keeps the top doc but reorders the tail, and imatrix slightly *hurts* (the
    score head is argmax/threshold-sensitive, like the punct argmax case). **q8_0 is
    the correct reranker default** — now a *measured* call.
  - **Re-checked jina + ms-marco on the 16×6 corpus (2026-07-04) → ALL rerankers
    now default Q8_0.** They DID reorder the tail: ms-marco-L6 iq4_xs **τ=0.958**,
    jina-v2 q4_k+im **τ=0.925** (both top1 16/16, min-τ ~0.73). So every reranker
    4-bit default was a coarse-corpus artifact; repointed jina + ms-marco-L6/L12 to
    q8_0 (their q8_0 GGUFs already shipped — registry-only change, `-iq4xs`/`-q4k`
    aliases kept). Net rule: **rerankers ship Q8_0, always** (exact ranking; 4-bit is
    a size alias only). bge + mxbai were already q8_0.
- **C1c — fixed-label NER DONE (2026-07-03).** Harness `ner` MODE: calibrate over
  `--ner` texts (BERT-NER's encoder is a shared crispembed_context, so the
  collector fires unchanged), A/B = micro **span-F1** vs full-precision gold.
  bert-base-NER (iq4_xs span-F1 1.0) + xlmr-ner-hrl (iq4_xs 1.0) done — both
  repointed to iq4_xs. Required a `bert_ner` classifier-dequant fix (commit
  85feaeb): the Q8_0/Q4_K `ner.classifier.weight` was read as raw F32, so both
  **failed to load on any quant** (their q8_0 defaults were broken).
- **C1d — GLiNER DONE (2026-07-03).** Routed GLiNER's 4 compute sites through an
  opt-in `ggml_backend_sched` (built only when `CRISPEMBED_IMATRIX_OUT` is set) so
  the collector's eval-callback attaches — the gallocr path has no hook. Flush in
  `gliner_ner_free` (clean_exit skips atexit). gliner-deberta → iq4_xs (span-F1
  1.0); gliner-lfm → q8_0 (q4_k+im 0.971 — the F1 dip is a 0.5-threshold artifact,
  not a bug: uniform 2% score shift tips 3 borderline detections; verified by
  score-level diff).
- **C1e — ColBERT + Sparse DONE (2026-07-03).** ColBERT (`lfm2-colbert`, per-token
  cosine A/B) → q4_k+im 0.9975. Sparse (`splade-pp-en-v1`, sparse-vector cosine)
  → iq4_xs 0.996. **splade-pp was functionally broken** (GGUF shipped without the
  MLM head) — a general **converter bug** (SPLADE mis-detected as 2-label NER via
  HF random-init); fixed with checkpoint-authoritative head detection
  (`convert-bert-to-gguf.py`), reconverted, sparse restored.
- **Eval corpora — bilingual EN+DE, DONE (2026-07-03).** The harness now ships
  bilingual EN+DE calibration/eval corpora across all modes: embed/sparse/ColBERT
  text (`calib_corpus.txt`/`eval_corpus.txt`, `tools/gen_eval_corpora.py`) and
  hand-written German parallels for the structured rerank/NER/ColBERT corpora (n
  roughly doubled — e.g. RERANK_EVAL 5→9, NER_EVAL 6→10). **All bundled text is
  self-authored and released CC0** — no third-party content license (avoids CC-BY /
  Wikipedia-CC-BY-SA), so it's usable under MIT/Apache/BSD-3. Verified both
  languages A/B cleanly (gte-modernbert q4_k vs q8: EN 0.987, **DE 0.975** — a real
  gap the old English-only n=5 set couldn't show).
  - *For SOTA benchmark scoring* (not license-free): report against **MMTEB**
    (Apache-2.0 framework); real permissive-ish retrieval data = **MIRACL**
    (Apache-2.0 card, Wikipedia text underneath) / **Tatoeba** (CC-BY-2.0) /
    GermanQuAD (CC-BY-4.0). No clean MIT/Apache EN+DE *gold NER* exists.
- **Bilingual re-calibration — NOT worth it (verified 2026-07-03).** A controlled
  A/B (`tools/kaggle/crispembed-calib-ab/`) showed English-only vs bilingual imatrix
  calibration makes a *noise*-level difference on German (bge-m3 DE +0.0001, xlmr-ner
  1.0→1.0): the imatrix is language-agnostic (per-column activation stats set by the
  weights, not the calib language). So the existing English-calibrated quants stand;
  the bilingual corpora are used for A/B *reporting*, not calibration.
- **Collector wiring complete (2026-07-03).** Every text-producing engine now
  hooks the imatrix collector: encoder + decoder + lfm2 (existing), GLiNER + lilt_kie
  (sched + install), clip_text/SigLIP-text (opt-in sched, gallocr path otherwise).
  cnn_embed is a face encoder (vision) — not a text imatrix target.
- **jina-reranker-v2 crash — FIXED (commit 9c25c75).** Surfaced by the calib-ab run:
  its Q8_0 768×768 `classifier.dense.weight` was read raw-F32 in `apply_classifier`
  → abort on the first rerank of ANY input (looked German-specific only because that
  was the first call). Read via `core_cpu::to_f32` (dequant-safe); the whole
  quantized reranker was broken. 5th instance of the quantized-2D-weight-read-raw
  class this session.
- **Retrieval-quality A/B DONE (2026-07-03).** `tests/bench_rag.py` now A/Bs all
  shipped flavors on MRR@10/Recall@10 (rank quality, not just cosine) over a
  hardened 35-doc IR set. Result: 4-bit preserves ranking (MRR/Recall ≥ F32) and
  imatrix lifts cosine monotonically (q4_k < q4_k+im < iq4_xs), recovering
  all-MiniLM's plain-q4_k MRR dip 0.948→0.950. The shipped imatrix defaults are
  retrieval-safe.
- **Architecture coverage audit DONE (2026-07-03).** No clean cheap-coverage gap
  remains: the encoder family (BERT/RoBERTa/XLM-R/Nomic-MoE/ModernBERT), decoder
  embedders (Qwen3/Gemma3/LFM2), rerankers, ColBERT, sparse, and NER are all
  shipped; the reranker RANK-head activations were verified to match llama.cpp
  (tanh for BERT/XLM-R, GELU for the DeBERTa pooler). Remaining upstream borrows are
  *optimizations*, not coverage — chiefly LFM2 ShortConv → `ggml_ssm_conv` for
  better Metal kernel coverage (perf, regression risk on a working engine).
- **Imatrix-less audit + unblock (2026-07-03).** A full-roster HF sweep found 9
  text/structured models still without imatrix. Resolved:
  - **CLIP + SigLIP text (3) — BUG FIXED.** They didn't just lack imatrix, they
    *crashed on any quant*: `position_embd.weight` is Q8_0 and was added via a raw
    `ggml_view_2d` → `binary_op: unsupported types ... src1 q8_0` (cos_vs_f32=0).
    Dequant-cast the position embedding to F32 before the add in
    `clip_text_embed.cpp` (token_embd is fine via get_rows; Q4_K matmul weights via
    mul_mat). Now: q4_k 0.9916, **q4_k+imat 0.9932**, iq4_xs+imat 0.9916 vs f32.
    Same F32-src1 family as the LFM2/Metal `ggml_mul` landmine.
  - **fireredpunc + fullstop-punc (2) — WIRED.** `fireredpunc.cpp` already builds a
    CPU-last `ggml_backend_sched` for both the chinese-BERT and XLM-R (fullstop)
    paths; added `crispembed_imatrix_install` + flush-in-free (one-shot binaries
    exit past atexit). Verified 73 tensors collected, punctuation output unchanged.
  - **LiLT (2) — VERIFIED, no code needed.** Already wired (sched + install). Not
    image-dependent — input is `{input_ids, bbox}`, and every embedding (pos, x/y/w/h
    box, type) goes through `ggml_get_rows`, which dequantizes → no clip-style add
    hazard. q4_k flips 3/16 KIE token labels vs f32 (the low-confidence ones);
    **q4_k+imat recovers all 3 → 16/16 label match.** imatrix collected (142 tensors).
  - **The "deferred 3" were RE-TESTED and mostly DONE (2026-07-03).** The earlier
    deferral reasons were largely wrong:
    - **pcs-xlmr-base — SHIPPED.** Quantizes cleanly (q4_k output = f32, no hazard).
      The shared-file blocker was solved with `#if __has_include("imatrix.h")` — the
      collector hook is active in CrispEmbed and compiles out in CrispASR (both build;
      pcs.cpp stays logically identical, paired commits CrispEmbed ff36015 / CrispASR
      f35185b8). Fine metric (365 tokens vs f32): q4_k+imat cuts KL **4.2×**
      (0.00132→0.00031) and restores argmax 1.0. Default → q4_k-imatrix.
    - **bidirlm-omni-2.5b-textonly — SHIPPED.** NOT unwired — the *decoder* path
      already installs the collector (crispembed.cpp:2067), and there's a dedicated
      text-only variant with an f16 base. q4_k+imat cos 0.948 vs f16 (+0.036 over
      plain); 2.5B is quant-sensitive so imatrix quants are SIZE options, default
      stays q8_0.
    - **bidirlm-omni-2.5b (full multimodal) — FULLY multimodal imatrix SHIPPED
      (2026-07-03).** Wired the vision tower (`bidirlm_vision.cpp`, own sched) and
      the audio tower (`crisp_audio/audio_tower.cpp`, `__has_include` guard — the
      collector resolves via crispembed-core's include path in the CrispEmbed build,
      compiles out in standalone CrispASR). A combined text+image+audio calibration
      run merges (collector's `merge_existing()`) into one imatrix covering **442 of
      452** tensors (196 text + 99 vision + 147 audio). Held-out A/B vs f16:
      text +0.036, **image +0.0072 (0.988→0.9955)**, audio +0.00001 (audio tower is
      already 4-bit-lossless). Shipped `-q4_k-imatrix-multimodal.gguf`, supersedes the
      text-only quant. **Found+fixed a latent bug:** `crisp_audio_compute_mel` read
      the 2-D `mel_filters` raw → aborted on ANY quantized bidirlm (audio worked only
      at f16); now dequant-safe (CrispASR 06c5f1d4) + `mel_filters`/`mel_window` kept
      F32 in the quantizer. Registry default stays q8_0 (quant-sensitive); this is a
      size option (`bidirlm-omni-2.5b-mm`).
  - **Rollout DONE (2026-07-03), 6 of 7 shipped + defaults repointed to their
    max-quality flavor:** clip-text-base (q4_k+im 0.9932), clip-text-large (0.9918),
    siglip-text-base (0.9623 — 4-bit-sensitive, -q8 alias kept), lilt-funsd (16/16
    KIE labels), fireredpunc (q4_k+im, **KL 0.0033 vs plain 0.0093 = 2.8x closer to
    f16**). All uploaded to their `cstr/*-GGUF` repos with q8_0/q4_k/q4_k-imatrix/
    iq4_xs/.imatrix/-imatrix-ab.txt + `-iq4xs/-q8/-f32` registry aliases.
    - **fullstop-punc — kept plain q4_k, no imatrix.** No f32/f16 base exists on HF
      (only q8_0/q4_k), so imatrix could only be calibrated+quantized *from q8_0*;
      measured against q8_0 the plain q4_k is already near-lossless (prob-cos 0.9996,
      KL 0.0012) and imatrix adds nothing. A proper f16-calibrated version would need
      reconverting from `oliverguhr/...` safetensors (no punct converter in `models/`)
      — not worth it for a sub-0.0012-KL gap.
    - **lilt-base — skipped.** Headless base/pretraining checkpoint (0 labels), no
      inference target and no meaningful A/B; users fine-tune it first.
  - **Metric lesson (why fireredpunc first looked worthless).** Restored-string
    exact-match / argmax-label agreement are THRESHOLDED — the final decision
    saturates while the probability distribution still drifts, so plain-q4_k and
    q4_k+imat both score "perfect" and imatrix looks useless (n=5 → "no value").
    imatrix acts on the LOGITS. Added a `$FIREREDPUNC_DUMP_LOGITS` hook and a
    continuous A/B (`/tmp/punct_ab.py`: mean per-token softmax prob-cosine + KL vs
    gold over hundreds of tokens); that revealed the real 2.8x KL win. Also caught a
    false negative from A/B-ing a **half-written gguf** (mid-quantize iq4_xs read as
    0/5; complete file is argmax-perfect). See `LEARNINGS.md`.
- **imatrix rollout COMPLETE (2026-07-03/04).** Every quantizable text/structured/
  multimodal model now has an imatrix quant with its default repointed to the
  max-quality flavor: 38 dense embedders, 7 rerankers, 2 NER, 2 GLiNER, ColBERT,
  SPLADE, clip/siglip-text ×3, lilt-funsd, fireredpunc, pcs-xlmr-base, and
  bidirlm-omni-2.5b (textonly + fully-multimodal). Intentionally NOT done, with
  reasons recorded: **fullstop-punc** (no f16 base on HF → q8_0-requant imatrix is
  a measurable no-op; would need a safetensors reconvert for a sub-0.0012-KL gain),
  **lilt-base** (headless pretraining checkpoint, 0 labels, no inference target),
  and a *fully-clean* bidirlm multimodal re-quant with F32 mel_filters (the shipped
  file uses quantized mel + the dequant read-fix; the quantizer now keeps mel F32,
  so a future re-quant would be marginally cleaner — audio is already 0.9979).
### Pending work — consolidated backlog (2026-07-12) — HANDOVER BRIEFS

Nothing below is a regression or a blocker; the shipped surface is correct.
Each item is written so a fresh agent can execute it without re-deriving
context. **Before starting ANY item: read LEARNINGS "measure the DOMINANT cost
before fixing a flagged micro-gap" and "The build dir was silently CPU-only";
verify `GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`; check `git worktree
list` + `git log main..<branch>` for a concurrent session's finished work; all
edits in a worktree (ggml symlink dance, see CLAUDE.md).** In priority order:

> **Status after the 2026-07-12 session (all on `main`, all validated):**
> DONE — C2 GGUF flags; **C4 cross-call prefix KV cache** (bit-equal CPU /
> cos ≥ 0.9999995 Metal, ≈2.07× compute); **Tier-1 2b decode fusion** (math_ocr
> cont-removal, ~30% faster decode, byte-identical); **C5 mmproj interop BOTH
> directions** — export (`models/export-mmproj-llamacpp.py`, validated via
> `llama-mtmd-cli`) and import (fixed `merge-llamacpp-qwen2vl-gguf.py`: stock
> llama.cpp Qwen2-VL-2B now OCRs correctly in `crispembed --ocr`, no 2.5-VL
> regression).
>
> **mmproj interop hardening + generalized to 3 families (2026-07-12, follow-up):**
> `tests/test_mmproj_interop.py` — a real-scripts round-trip regression (found +
> fixed two shipped bugs: export read stale legacy names; merge hardcoded F16
> patch dtype). Then **generalized to a family-dispatch on shared
> `models/gguf_merge_core.py`** — one unified entry point
> `models/merge-llamacpp-gguf.py` auto-detects the family from
> `clip.projector_type` and routes to the per-family merge. THREE families now
> import a stock llama.cpp VL model into CrispEmbed, each **validated
> end-to-end** (correct OCR + diff-harness intermediate parity with the native
> converter, isolated against the source `llama-mtmd-cli`):
>   - **Qwen2-VL** (`qwen2vl_merger` → `qwen2vl_ocr`): identity names, Conv3d
>     temporal patch. `merge-llamacpp-qwen2vl-gguf.py` (bidirectional: also
>     `export-mmproj-llamacpp.py`).
>   - **SmolVLM/Idefics3** (`idefics3` → `smoldocling`): SigLIP, `ffn_down`=fc1,
>     arch=llama LLM → q/k **un-permute**. `merge-llamacpp-smolvlm-gguf.py`
>     (256M + 500M validated).
>   - **InternVL2.5/3** (`internvl` → `internvl2_ocr`): InternViT, QKV re-fusion,
>     layer-scale, class token, MLP connector, `ffn_up`=fc1 (inverse of SmolVLM),
>     dynamic-tiling metadata, arch=qwen2 LLM → q/k copied **VERBATIM** (NEOX
>     RoPE — un-permuting scrambles it; this was the bug).
>     `merge-llamacpp-internvl-gguf.py` (1B validated).
> Key cross-cutting rules (LEARNINGS.md): un-permute q/k is **arch-dependent**
> (llama yes, qwen2 no); map ViT FFN fc1/fc2 by **output dim**, never name; the
> diff-harness catches masked bugs the output hides (InternVL OCR'd correctly at
> `vis_patch_embed cos=-0.936`). Four Python tests + README "Importing a stock
> llama.cpp VL model" in the `regression.yml` smoke tier. These three are the
> complete tractable set (both a working CrispEmbed loader AND a downloadable
> llama.cpp mmproj at a sane size); GLM-4V would mean a 9B download vs a loader
> still needing dynamic-preprocessing work.
>
> RESOLVED in the 2026-07-12 backlog sweep: **C5(a) bicubic** (measured — HF uses
> PIL a=−0.5, already correct; a=−0.75 is cos<0.00002 worse); **`<__media__>`
> marker** (not-applicable — mtmd-internal, CrispEmbed expands tokens per-engine;
> no CLI prompt entry point); **reranker corpus** (expanded 16→30 EN+DE groups;
> τ-eval is Kaggle-only); **internvl2 diff-harness input guard** (dump stamps
> `diff.input_mode`, harness refuses image-vs-gradient mismatch).
> STILL OPEN (all P3, low-EV/blocked): CrispASR `gpu_backend_pref.h` sync (3-line
> change applied on disk, uncommitted — commit in the CrispASR session); LFM2
> ShortConv → `ggml_ssm_conv` (P3, regression risk, needs the model); bidirlm
> multimodal clean re-quant (P3).


- **C2 data-driven GGUF behavior flags — DONE (2026-07-12).** Survey found it
  was already mostly data-driven: pooling (`bert.pooling_method`/`pooling_type`
  read at load), causal-attention (`is_bidirectional` arch KV in
  `decoder_embed.cpp:96`), decoder-BPE `add_bos_token` (crispembed.cpp ~:2101),
  and the converters already EMIT `tokenizer.ggml.add_bos_token`/`add_eos_token`.
  What landed now: the remaining readers — SPM encode() wrap gated by
  `set_add_flags` (encode_pair keeps the canonical cross-encoder layout),
  encoder-BPE CLS/SEP gated via the -1-id convention (survives the merges
  reload), and LFM2's hardcoded BOS-only rule replaced by flag reads with
  defaults (true, false) = the historical rule. `kv_bool` is BOOL-typed-only →
  absent/foreign-typed KVs fall back to exact current behavior.
  *Verified:* byte-identical embeddings main-vs-change on all four tokenizer
  families (all-MiniLM WordPiece, multilingual-e5-small SPM with flags
  present, gte-modernbert BPE with flags absent, lfm2-embed with flags
  absent), 3 texts each incl. multibyte; negative test: patching
  `add_eos_token=false` into an e5 copy changes the embedding (flag is live).

- **P2 — C4 KV/prefix-sharing ACROSS decoder-embedding calls. DONE 2026-07-12
  (see the detailed C4 brief below for the landed delta + numbers).** Persists
  a per-context prefix KV cache so a second call with the same instruction
  prefix skips recomputing it. Ships Qwen3 (octen) + Gemma3 (harrier); LFM2 is
  a separate path (`lfm2_embed.cpp`, never routed through
  `decoder_encode_tokens_cached`) so its ShortConv whole-prefix constraint does
  not apply. jina-v5 is `is_bidirectional=1` → correctly ineligible (causal
  prefix independence does not hold). Default ON, opt out with
  `CRISPEMBED_DECODER_PREFIX_CACHE=0`.
  *Verified:* CPU bit-equal (cos 1.0, max_abs 0.0) cached-vs-full on octen-0.6b
  q8 + harrier-270m q8; Metal cos ≥ 0.9999995; no-prefix byte-identical to the
  pre-C4 main binary (cos 1.0, max_abs 0.0). Speed (octen q8 Metal, load 1.2,
  σ≈0.02s): 40 long-prefix prompts 2.16→1.30s end-to-end; ≈2.07× compute-only
  after subtracting ~0.50s fixed load. Test: `tests/test_prefix_cache.py`.

- **P2 — Tier-1 2b op-count reduction, one decoder end-to-end. DONE 2026-07-12
  — decode-step ~30% faster, byte-identical.** Measured first (`MATH_OCR_NODES=1`
  prints decode-step `n_nodes`; `CRISPEMBED_MATH_OCR_BENCH=1` prints encoder/
  decoder ms), then fixed the *actual* overhead — which was NOT where the brief
  pointed:
  - Baseline: decode-step = **355 nodes**; encoder 200 ms vs decoder 44 ms
    (trocr q8 Metal) → decode ~18% of compute, encoder 82%.
  - The step already uses `ggml_flash_attn_ext` (brief item 1 done); the flagged
    QKV concat is only ~1.3% of compute (below threshold, skipped).
  - **The real overhead was the redundant `ggml_cont` copies in the single-query
    decoder attention** (`g_mha_1q`): `flash_attn_ext` only needs row-contiguous
    src (`nb0==type_size`), which `permute(0,2,1,3)` preserves, so the 3 conts
    (×2 self+cross ×6 layers = 36 copy-kernels) were pure waste. Removed →
    **355→319 nodes, decode 45.5→31.5 ms (~30% faster, interleaved, no overlap)**,
    transcript byte-identical on Metal AND CPU (fox + scan_strip). Default now
    cont-off; `MATH_OCR_ATTN_CONT=1` restores them.
  - **Negative result (gate caught it):** converting the 578×578 *encoder*
    attention (`g_mha`) from manual F32 mul_mat+soft_max to `flash_attn_ext` is
    byte-identical on Metal but **DIVERGES on the CPU kernel** (GOOD→DSP.90,
    CARDS→RECEIPT). Kept manual F32 there (conts are load-bearing for mul_mat).
    Documented inline so nobody "optimizes" it again. The 200 ms encoder remains
    the real cost but is out of this decode-step task's scope (and its safe
    fusion path is blocked by the CPU-flash divergence).
  - Generalization to long-output VLM decoders (qwen2vl/got/glm/internvl2/
    lightonocr): the same cont-removal applies to any per-step attention that
    already uses flash + permute; audit each on a quiet box, per-engine.

- **P3 — LFM2 ShortConv → `ggml_ssm_conv` — WON'T-DO, resolved by analysis
  (2026-07-12).** The premise (Metal kernel coverage) is already satisfied:
  `lfm2_short_conv` uses `ggml_conv_1d_dw`, which decomposes to
  `ggml_im2col` + `ggml_mul_mat` (ggml.c ~L4583) — **both have Metal kernels**,
  so the conv already runs fully on-device with no CPU fallback. Refactoring
  gains nothing on coverage AND is a semantic mismatch: `ggml_ssm_conv` is
  **causal** (left-padded sliding window `[t, t+nc)`), while LFM2-embed's
  ShortConv is **bidirectional/symmetric** (centre-padded, `[t-1, t+1]`), so a
  drop-in swap would change the output; matching it would need fiddly custom
  symmetric padding of the ssm_conv input, adding risk to a working, correct,
  Metal-covered engine for no measured benefit. Closed.

- **P3 — C5 remnants (the preprocessor port itself is DONE). Assessed
  2026-07-12 — both correctly deferred, for concrete reasons (not just the
  brief).**
  - (a) bicubic A/B — **RESOLVED by measurement 2026-07-12 (no model download
    needed).** The A/B doesn't need a 4 GB VL model: it's about which cubic
    coefficient the HF processors match. Measured locally (PIL vs torch on a
    structured test image): HF `image_transforms.resize` uses **PIL/Pillow**
    (`a=-0.5`), which is what `image_preprocess.cpp` already uses — so `a=-0.5`
    is correct for HF parity (residual cos 0.999984). `a=-0.75` (OpenCV
    `INTER_CUBIC` / `torch.nn.functional`) differs from `a=-0.5` by only cos
    0.999983–0.999995 (max ~0.13/255) AND would move *away* from the PIL path —
    so switching is strictly worse. No change; corrected the stale code comment
    (it wrongly listed OpenCV under `a=-0.5`).
  - (b) mmproj interop — **DONE + GENERALIZED (see the status block above;
    the narrative below is the original blow-by-blow, kept for the debugging
    trail).** Both directions ship for Qwen2-VL; the import path is now a
    family-dispatch on `models/gguf_merge_core.py` with a validated 2nd family
    (SmolVLM/Idefics3 → `smoldocling`, `merge-llamacpp-smolvlm-gguf.py`).
    Regression: `tests/test_mmproj_interop.py` + `tests/test_mmproj_smolvlm.py`.
    Original notes: **EXPORT half DONE + validated 2026-07-12**
    (`models/export-mmproj-llamacpp.py`). Converts a CrispEmbed combined
    Qwen2-VL GGUF → a llama.cpp `mmproj-*.gguf` (metadata: general.architecture=
    clip / clip-vision, the full `clip.*` key set incl. projector_type=
    qwen2vl_merger, image_size/patch_size/embedding_length/block_count/
    head_count/projection_dim/feed_forward_length/image_mean/std/ln_eps; tensors
    renamed `vis.*`/`proj.*` → `v.blk.*`/`v.patch_embd`/`v.post_ln`/`mm.*`). The
    name + metadata maps are the exact INVERSE of the shipped, proven merge
    script, and the **complete clip.* schema was extracted empirically** from a
    real reference (`ggml-org/Qwen2-VL-2B-Instruct-GGUF` mmproj, 27 KV / 520
    tensors) — no guessing. `--self-test REF` round-trips a reference mmproj
    (rename → export → write GGUF → re-read) and asserts **byte-identical**
    tensors (520/520) + matching clip.* metadata — self-contained validation, no
    VL-model download or inference. So the output is byte-identical to a
    known-good llama.cpp mmproj (which llama.cpp loads by definition). Do NOT
    link libmtmd (it PUBLIC-links all of llama) — this is offline conversion.
    - EXPORT validated END-TO-END 2026-07-12: merge (shipped) → export → the
      exported mmproj + LLM run in `llama-mtmd-cli` and OCR fox.png correctly
      ("The quick brown fox jumps over the lazy dog. 12345"). So CrispEmbed→
      llama.cpp mmproj interop works.
    - **REVERSE direction (llama.cpp → CrispEmbed) — DONE.** The merge was
      BROKEN: it renamed tensors to `vis.blocks.*`/`llm.layers.*`, which the
      current `qwen2vl_ocr` loader does NOT read (it reads llama.cpp-native
      `v.blk.*`/`blk.*` + CrispEmbed `l.blk.*`), so its output SIGSEGV'd on load
      (vision misdetected as 2.5-VL). Fixed: (1) merge keeps native names +
      concatenates the split temporal patch embed (`v.patch_embd.weight` +
      `.weight.1` → `[in*T*H*W, out]`); (2) loader gains `v.post_ln` merger-norm
      + tied-`lm_head`(=`token_embd`) fallbacks. (A transient `ggml_can_mul_mat`
      abort was first mis-blamed on `qwen2vl.vision.intermediate_size` metadata —
      **red herring**; the real cause was the ViT-FFN fc1/fc2 mapping, below.)
    - The abort was the ViT-FFN **fc1/fc2 role mapping**: llama.cpp's qwen2vl
      mmproj INVERTS `ffn_up`/`ffn_down` vs the
      projection direction (`ffn_down`=fc1 hidden→intermediate, `ffn_up`=fc2 —
      proven by biases: `ffn_up.bias`=[hidden], `ffn_down.bias`=[intermediate]).
      The loader blindly aliased `ffn_up→fc1`; fixed to map fc1 = the
      larger-output projection (by bias/weight dim). The weights are already
      correct ggml `[in,out]` order (no transpose needed — Qwen2-VL's null
      `qkv_w` means the fix_ne gate correctly never fires).
    - **DONE — reverse interop WORKS end-to-end on Metal AND CPU.** A stock
      llama.cpp Qwen2-VL-2B (ggml-org GGUFs) → `merge-llamacpp-qwen2vl-gguf.py`
      → `crispembed --ocr` reads fox.png as *"The quick brown fox jumps over the
      lazy dog. 12345"* (identical to `llama-mtmd-cli`). The final bug was NOT
      the vision: a proper **HF diff-harness** (Qwen2-VL-2B vision tower fed
      CrispEmbed's exact patches) showed the vision output at cos 0.957 — and
      **injecting HF's perfect embeds still produced "text not visible"**, and
      zeros/random/HF embeds gave IDENTICAL output → the image was being
      **silently dropped**. Root cause: `qwen2vl.image_token_id` is absent from
      llama.cpp GGUFs, so the vision-text splice used its default `0` while the
      prompt builder emitted `<|image_pad|>=151655` → the splice found no image
      positions. Fixed both sides: `image_token_id` default is now 151655
      (matching the prompt's `image_pad_id`), and the merge writes
      `qwen2vl.image_token_id`/`vision_start`/`vision_end` (151655/151652/151653)
      so the GGUF is self-describing. (Method note for future VL interop: the
      inject-embeds + zeros/random discriminator instantly separates a vision
      bug from an LLM-conditioning bug — do that FIRST before diffing the ViT.)
      Do NOT link libmtmd. **`<__media__>` prompt-marker — RESOLVED as
      not-applicable (2026-07-12).** Verified by reading the code: it's an
      mtmd-INTERNAL marker that mtmd expands into engine-specific image tokens.
      CrispEmbed already does that expansion itself — each OCR/VLM engine inserts
      its own image token at its canonical position (`internvl2_ocr`
      `<IMG_CONTEXT>`, `qwen2vl_ocr` `<|image_pad|>`, `smoldocling`
      `<image>`) inside `build_prompt`, and the image is supplied as a path /
      pixel buffer, not an inline prompt marker. There is no `--prompt` CLI flag,
      so no entry point where a user would type `<__media__>`, and no consumer
      for a marker parser. Adding one would be speculative dead code that
      contradicts the deliberate image-path design. Migration mapping for a
      llama.cpp/mtmd user: a `<__media__>` prompt → CrispEmbed `--image FILE`
      (image auto-placed per engine). Closed; not a follow-up.

- **P3 — reranker corpus expansion — corpus DONE (2026-07-12); τ-eval pending
  Kaggle.** `RERANK_EVAL` extended from 16 to **30** self-authored-CC0 EN+DE
  graded 6-doc groups (7 EN + 7 DE new distinct topics: antibiotics, tides,
  four-stroke engine, balanced diet, earthquakes, dreams, rainbows). All eval
  loops iterate `len(RERANK_EVAL)` so nothing else changed. The **actual
  Kendall-τ vs f16 gold runs on Kaggle** (needs the reranker models + GPU) — not
  runnable locally. Expectation unchanged: the 16×6 corpus already showed 4-bit
  reorders tails on EVERY reranker (τ 0.925–0.967), so the larger corpus will
  almost certainly re-confirm the q8_0 default, not flip it; only repoint a
  reranker to 4-bit if it scores τ=1.0 on the expanded set.

- **P3 — CrispASR `gpu_backend_pref.h` sync.** Commit `0622c1d` added a
  metal→mtl alias (ggml registry is named "MTL"); CrispASR's copy needs the
  same 3 lines. Its tree had uncommitted work on 2026-07-12 — sync when
  clean, keep the files logically identical (see the pcs.cpp convention).

- **P3 — esrgan tile-loop parallelism.** Intra-op threading measured SLOWER
  (see negative result above); the real lever is running whole 128px tiles
  concurrently, which needs per-thread backend+sched replication (the tile
  loop shares one `ctx->enc_sched`). A real concurrency project; verify on a
  quiet box.

- **P3 — bidirlm multimodal clean re-quant** with F32 `mel_filters`
  (cosmetic; shipped file works via the dequant read-fix). OOM-prone on the
  16 GB Mac — Kaggle only.

- **P3 — non-embedding OCR/vision perf** (CUDA Class-B divergence needs
  Turing/Pascal HW; deepseek_ocr2 F16-KV port; unified `core/vlm_decoder.h`)
  — tracked in the OCR sections below.

- **Stale-claims corrections (2026-07-12):** the old backlog said the decoder
  batched graph "still needs the block-diagonal + prefix-recompute layout" —
  it EXISTS (`decoder_encode_tokens_batch`, prefix-sharing included); and C5
  was listed as an open port — the port shipped, only the remnants above are
  open.

**C2 — data-driven GGUF behavior flags. DONE 2026-07-12 (see backlog brief above for the landed delta + verification).** Original scope: bake `pooling_type`, `causal_attention`,
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
- **Rectangular 4D per-item mask, O(B·T²) — IMPLEMENTED, OPT-IN (`CRISPEMBED_ENCODER_4D=1`).**
  Keeps sequences as separate 4D batch items `[hd,T,nh,B]` with a per-item padding
  mask `pad_mask` [T,T,1,B] (−inf on padded keys per item), so attention is O(B·T²)
  not O((B·T)²) — the real throughput fix vs packing. `encode_tokens_4d` length-sorts
  + chunks into groups of `CRISPEMBED_ENCODER_4D_GROUP` (default 32) to minimize
  padding. Bit-parity: `tests/test_encoder_batch.py::TestEncoder4DBatchParity` cos
  **1.0 / 0.9999697** (single/multi group). **Consistently faster than sequential AND
  packed** (1.18×–1.48× at N=8/32/128, stable). ggml `flash_attn_ext` takes the
  per-batch mask (`q->ne[3] % mask->ne[3] == 0`) and the Metal kernel indexes it per
  `iq3 % ne33` (confirmed in ggml source). **NOTE: this env is CPU-only
  (`GGML_METAL=OFF`; sandbox has no GPU) — Metal not empirically exercised** → kept
  opt-in pending a real-Metal A/B before flipping the default over packed.
  - **Real-Metal A/B (2026-07-12, M1) — CORRECTED same day.** The first
    "Metal" verdict here was measured against a build whose `build/` was
    silently `GGML_METAL=OFF` (stale since 2026-07-07; see LEARNINGS) — those
    numbers were CPU. On an actual Metal build: **4D parity PASSES cleanly**
    (worst cos 0.9999996/0.9999997 at group=2/100 — the earlier 0.99989
    "failure" was the CPU-only build), and the interleaved 3-way bench
    (round-robin, 9 reps, all-MiniLM q8_0) gives:
    | shape | packed vs seq | 4D vs seq | 4D vs packed |
    |---|---|---|---|
    | uniform N=8/32/128 | **5.2× / 6.0× / 6.6×** | 2.1× / 2.3× / 1.5× | 0.40× / 0.38× / 0.22× |
    | mixed N=8/32/128 | **6.6× / 7.4× / 7.3×** | 2.0× / 1.1× / 2.0× | 0.30× / 0.15× / 0.27× |
    **Verdict: on Metal, PACKED is the batching mode** — 5–7× vs sequential,
    parity cos 1.0, consistent on uniform AND mixed. 4D beats sequential but
    always trails packed there; 4D remains the CPU-backend tool (its 1.18–1.48×
    CPU result stands).
  - **Backend-conditional default SHIPPED (2026-07-12).** `packed_batch_enabled`
    now defaults ON when the primary backend is a GPU and OFF on CPU;
    `CRISPEMBED_ENCODER_PACKED=1/0` overrides in either direction, 4D stays
    opt-in. Verified: Metal default output is **bitwise identical** to forced
    packed, worst cos vs sequential 0.9999996, warm B=30 mixed 3.2× (loaded
    box; 5–7× at lower load); CPU default output identical to sequential
    (cos 1.0); full test_encoder_batch suite green.
    En route: `tests/test_encoder_batch.py` had an env leak — the 4D parity
    class left `CRISPEMBED_ENCODER_4D=1` set, so the throughput test's "seq"
    and "packed" legs silently ran the 4D path (fixed: bench pops both envs).

**C4 — KV prefix-sharing for the decoder-embedding path. DONE 2026-07-12.**
The decoder-embed path is a single-shot prefill (flash-attn over the whole
sequence, no autoregressive KV cache), so with causal attention the prefix
tokens' per-layer post-rope K/V and final output-normed hidden are INDEPENDENT
of any suffix. Landed:
- `dec_prefix_cache` in `decoder_embed_internal.h` (host-side per-layer K/V +
  prefix hidden + prefix ids), a member of `crispembed_context`.
- `decoder_encode_tokens_cached` (`decoder_embed.cpp`): on a repeated leading
  prefix (LCP vs the previous call, exact-match, MIN_PREFIX=4) it BUILDS the
  cache via a prefix-only graph then runs a SUFFIX-ONLY graph whose queries
  attend to `[cached prefix K/V | fresh suffix K/V]` (rectangular flash-attn,
  n_q=S < n_kv=P+S); on a later call starting with the cached prefix it reuses
  directly. The full/cold/miss path is the UNTOUCHED `decoder_encode_tokens`
  (byte-identical). Wired into the single-text encode in `crispembed.cpp`,
  invalidated on LoRA hot-swap. Bidirectional models ineligible.
- Both the build and suffix graphs compute on a SINGLE-backend `ggml_gallocr`
  (not the sched): the sched aliases the 2·n_layer interior `set_output` K/V
  snapshots to one reused buffer (read back identical). The injected per-layer
  prefix-K/V inputs are marked `set_output` too so gallocr keeps them in
  distinct, non-reused buffers.
- **Landmine (cost most of the debug time):** in the build graph V was
  `ggml_reshape_3d(v_proj)` — a VIEW; `set_output` on a view does NOT protect
  the underlying v_proj buffer, so the post-compute V readback was stale garbage
  (K, a fresh rope output, was fine; prefix_hidden was correct because the
  forward's flash read V in time). Fix: `ggml_cont` K and V before marking them
  output. See LEARNINGS.
- *A/B quality:* PASS. `tests/test_prefix_cache.py` cached-vs-full: octen-0.6b
  q8 + harrier-270m (Gemma3) q8, CPU **bit-equal** (cos 1.0, max_abs 0.0),
  Metal cos ≥ **0.9999995**. Same-prefix-twice, prefix-change (invalidation),
  no-prefix (byte-identical, incl. vs the pre-C4 binary) all covered.
- *A/B speed:* octen q8 Metal, load 1.2, σ≈0.02s: 40 long-prefix prompts
  2.16→1.30s end-to-end, **≈2.07× compute-only** (after ~0.50s fixed load).

**C5 — mtmd preprocessing alignment. PORT DONE (`src/image_preprocess.{h,cpp}`, wired into qwen2vl/bidirlm/mixtex); only the remnants below are open.**
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
  `examples/cli/model_mgr.cpp`. Guards: `tests/test_modernbert_parity.py` **and** a
  compiled `test-modernbert-diff` wired into the regression manifest
  (`dump_modernbert_reference.py` → `modernbert-ref.gguf`; q8_0-vs-f32 final_hidden
  0.9919, floor 0.99; disabling SWA craters cos to −0.87 so the entry catches an SWA
  regression). Runs green via `run_one.py --name modernbert`.
- *Bug found + fixed en route:* `crispembed_encode_tokens_raw` (+ a sibling raw path)
  branched only SPM/WordPiece — **missing the BPE case**, so BPE encoders (ModernBERT)
  were mis-tokenized via WordPiece in the raw API (113→103 tokens). Added the `use_bpe`
  branch. Also: the CrispEmbed BPE tokenizer diverges from HF on some longer/varied
  texts (edge case, not chased; the guardrail text is verified-aligned).

**NEXT PRIORITIES (2026-07, ordered):** **C3 4D batch — DONE (opt-in, see C3).**
**EmbeddingGemma — ALL CLOSED (2026-07-12).** Dense(×2)/mean/Matryoshka pipeline
verified correct (~0.997, an accepted small Gemma3-backbone residual amplified by
the Dense bottleneck). The parity test + Python `matryoshka_dim` in `encode()`
landed earlier than this note assumed (`6659252`,
`tests/test_embeddinggemma_parity.py`); the registry pooling labels
("last-token" → mean-pool, 4 embeddinggemma entries) are fixed now that the
imatrix-repoint churn on `model_mgr.cpp` has landed. Nomic-v2-MoE already covered
(`test_moe_parity.py`).

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
- **pcs — FULL ONNX PARITY, CLOSED.** Started as a q4_k crash (`ggml-backend.cpp:349 tensor read out
  of bounds`: Q4_K/Q4_0 FC-head weights read via raw `ggml_backend_tensor_get` into F32 buffers);
  fixed by per-row dequant (`to_float` trait, sized by `ggml_nbytes`). Then a full diff-harness pass
  vs the source ONNX model (`tools/dump_pcs_reference.py` + `PCS_DEBUG`/`PCS_FORCE_CPU`/
  `PCS_DUMP_HIDDEN`/`PCS_DUMP_LAYER`) found **six** root causes of engine-vs-reference deviation and
  fixed all. After: tok+post+pre+seg predictions match the reference **11/11** on every test
  sentence, encoder hidden cos **0.999997** (was 0.996), and q8_0/f32 reproduce the ONNX output
  exactly — "Hello world, how are you today? I am fine, thanks." (q4_k: 1 truecase char off, genuine
  4-bit quant floor). The two `pcs.cpp` copies are now **unified** (CrispEmbed = CrispASR modulo the
  `pcs.h` include; the unproven `fc_cache`/`bench` scaffolding was dropped to end the divergence).
  Root causes:
  1. **Tokenizer (dominant):** XLM-R's SP model is **Unigram** → needs Viterbi max-score
     segmentation. Greedy longest-match mis-split multi-subword words ("delayed"), corrupting
     embeddings (per-token cos 0.13). Added Viterbi using new `tokenizer.ggml.scores` (converter
     emits `sp.GetScore`; new `core_gguf::kv_f32_array`); greedy kept as scores-absent fallback.
     **Requires re-converted GGUFs** (re-uploaded to `cstr/pcs-xlmr-base-GGUF` with scores; added q8_0).
  2. **Decode** re-counted subtokens greedily → dropped final punctuation on multi-subword words;
     now partitions the actual token_ids by the ▁ word-start boundary.
  3. **SBD/seg** used argmax; ONNX thresholds `softmax P(boundary) > 0.05`.
  4. **Truecase** conditioning used the current token's sbd; ONNX feeds the SHIFTED
     is-sentence-initial flag (`argmax(seg[t-1])`, token 0 = initial).
  5. **FFN GELU** was `ggml_gelu` (tanh approx); ONNX uses exact **erf** GELU (`ggml_gelu_erf`).
  6. **LayerNorm eps** was `1e-12`; XLM-R/ONNX use `1e-5`.
  Also: manual F32 attention (`PCS_FLASH_ATTN=1` restores flash). Guard `pcs` now pins the exact
  golden on **q8_0** (exact + backend-robust; q4_k has the 1-char quant flip). Registry adds `pcs-q8`.
  fireredpunc was unaffected (WordPiece + F16 cls head, in-graph mul_mat).
- **encoder-parity audit (pcs follow-up) — CLOSED.** Swept both repos for the pcs bug classes.
  Confirmed + fixed: **fullstop-punc** (XLM-R-large via the fireredpunc SP path) had the full set —
  greedy tokenizer→Unigram Viterbi (needs `tokenizer.ggml.scores`; GGUFs re-converted+re-uploaded),
  eps 1e-12→1e-5 (conditional on `is_sentencepiece`), tanh→erf GELU; verified exact vs HF
  (oliverguhr/fullstop-punctuation-multilang-large), guard added. **GELU tanh→erf** (config-confirmed
  `hidden_act="gelu"`=exact): `gliner_ner.cpp` (DeBERTa-v3), `lilt_kie.cpp` layout FFN (text was
  already erf), `fireredpunc.cpp` (both BERT + XLM-R), `bert_encoder.cpp` (CrispASR/MeloTTS).
  **Quant-read crash** (pcs class): `crispembed.cpp` MLM/SPLADE head read quantized `token_embd`
  /`mlm_transform_w` as raw F32 → now `core_cpu::to_f32` (dequant-safe). Fused-QKV was already guarded.
  The dead `src/{pcs,fireredpunc}.cpp` fallback duplicates (built only without crisp_punc) were
  unified to the fixed crisp_punc copies. Not portable (pcs-specific): SBD 0.05 threshold, truecase
  shifted conditioning, decode subtoken-count. m2m100 greedy-SP is a self-labeled placeholder (left).
- **decoder_embed — CLEAN, CLOSED.** Added a compiled guardrail: `test_decoder_embed_diff.cpp`
  (crispembed_encode → final last-token-pooled embedding) vs an independent Qwen3-Embedding-0.6B
  HF ref (`dump_decoder_embed_reference.py`). Engine (q8_0) matches cos 0.9993; wired `diff_only`,
  run_one PASS. Also added to the Kaggle ref-gen kernel. Confirms the wave flash_attn_ext work is fine.
- **vit_embed / clip_text / cnn_embed / bidirlm_vision — LOW (perf-only).** Last wave edit was
  `632b4c1 perf: disable OpenMP / default 1 thread` (threading, not numeric). Standing
  guardrails nonetheless; GGUFs + HF sources available → closeable locally.
  **bidirlm_vision CLOSED (2026-07):** `test-bidirlm-vision-diff` +
  `tools/dump_bidirlm_vision_reference.py` (HF BidirLMOmniVisionModel, visual.* only)
  + `diff_only` manifest entry; ref on `cstr/bidirlm-omni-2.5b-GGUF`. q8_0 image_embeds
  cos 0.997, deepstack 0.9998/0.9938 (per-token mean); run_one PASS. q4_k 0.97 quant floor.
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
- **layout — REGRESSION, FIXED (other agent, `6027b56`).** Encoder cratered (`s3` cos −0.146…
  `dec_0_cross` −0.344; early stages cos 1.0). Wave `dc0861b` swapped manual attention for
  `ggml_flash_attn_ext`, which ALREADY applies `permute(0,2,1,3)` internally — the leftover manual
  permute double-permuted the RT-DETR encoder output. Fix: drop the spurious post-flash_attn
  permute. Same class fixed in math_ocr + deepseek (`dd4b4fd`).
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
- **bidirlm (text) — CLOSED (was a converter bug in the SHIPPED GGUF, not pooling/mrope).** The
  shipped `bidirlm-omni-2.5b*` GGUFs (both the full-omni and textonly repos) cratered the TEXT tower
  to **cos 0.047** while vision passed 0.997 — the engine warns "stale GGUF — re-export". Proven
  2026-07-03 the mean-pool ref was correct all along: a **fresh re-export with the current
  `convert-decoder-embed-to-gguf.py`** gives text **cos 1.000000 (f16) / 0.9992 (q8_0)** AND still
  passes vision (image_embeds 0.9966, deepstack 0.9997/0.9921). NOT an mrope issue (the `mrope_section`
  warning persists on the fresh GGUF too but is a red herring — text-only shares all 3 rope channels);
  the real bug was the tensor weights/layout the old converter produced. Re-quantized f16→q8_0 with
  `crispembed-quantize`, uploaded the corrected `bidirlm-omni-2.5b-q8_0.gguf` + `bidirlm-text-ref.gguf`
  to `cstr/bidirlm-omni-2.5b-GGUF`, wired `bidirlm-text` (test-decoder-embed-diff, one GGUF serves
  both text + vision). FOLLOW-UP: the repo's f16 / imatrix q4_k/q5_k/q6_k + the whole `-textonly`
  repo are still the OLD (text-broken) conversion — regenerate them from the fresh f16 (imatrix
  variants via the imatrix pipeline). decoder_embed (Qwen3) double-confirmed on CUDA at cos 0.9993.
- **Kaggle ref-gen kernel drift (investigated 2026-07, run v8):** the batch re-runs pre-closed
  engines too; findings — **lfm2 = ok** (was a FALSE verify_failed: it prints `PASS: 20 FAIL: 0`
  which tripped the kernel's `"FAIL" in out` heuristic; fixed to accept `fail: 0`). **lfm2_colbert =
  cos 0.57 on CUDA** (0.998 on CPU/Metal) — a REAL CUDA bug. Localizer (LFM2_COLBERT_DIFF_REF on
  P100) shows the backbone `cur` itself is corrupted (hidden cos −0.70) in the ColBERT graph while
  the identical dense-graph backbone passes 20/20 on the same CUDA → a compute-time divergence.
  FIRST FIX ATTEMPT DISPROVEN: `hidden=ggml_cont(cur)` (set_output-on-live-intermediate theory) gave
  byte-identical CUDA numbers (set_output can't change computed values) — reverted. Real cause is
  graph-structural (extra projection output / `sched_reserve` / shared `ctx->sched` between the
  dense and colbert paths); needs per-layer CUDA localization. Handover updated
  `handover-prompts/lfm2-colbert-cuda-multivec-divergence.md`. **RESOLVED 2026-07 (branch
  `fix/lfm2-colbert-cuda-multivec`)**: the graph-structural cause was `encode_multivec`
  re-allocating the SAME graph it passed to `ggml_backend_sched_reserve` (stale `tensor->buffer`
  after `sched_reset`); the dense path already rebuilds a fresh graph after reserve, ours did not.
  Fix = rebuild after reserve (mirror the dense path). Verified by an on-P100 A/B (same Tesla P100,
  compute 6.0): **main colbert_output cos 0.571643 (FAIL, backbone hidden −0.702160) → fix cos
  0.995885 (PASS ≥0.99, hidden +0.922054)** — baseline reproduces the handover numbers to 6 decimals,
  the guardrail now passes on CUDA. (Also: a codebase-wide sweep found ~9 engines with the same
  set_output-on-live-intermediate pattern — got_ocr/glm_ocr/internvl2/qwen2vl/math_ocr/pcs — but
  since that pattern was REFUTED as lfm2_colbert's cause, they are NOT confirmed bugs; do not
  mass-apply a cont fix.) **lilt** = FALSE verify_failed (encoder
  6 PASS/0 FAIL; the "Label match 0/16" untrained-head red herring trips the harness exit + kernel
  heuristic). **layout** = dump_failed on `RTDetrV2 embed_dim 256 not divisible by num_heads 12`
  (model-config, not transformers-version). **gliner** = known dead reference. All of these have LIVE
  manifest refs from prior sessions, so their guardrails still work — only the kernel's regeneration
  is affected.
- **fireredpunc — CLEAN, CLOSED (Gap 4).** No hidden/logits accessor in the punct C API → golden
  text-match `run_check` (new generic `test_punct_diff.cpp`). q4_k engine restores
  "hello world how are you today i am fine thanks" → "Hello world. How are you today? I am fine.
  Thanks." (correct, deterministic). Wired; run_one PASS. Impl in CrispASR/crisp_punc.
- **pcs — REGRESSION FIXED, CLOSED (Gap 4).** See triage above — q4_k crashed on inference
  (Q4_K/Q4_0 FC-head weights read as F32). Fixed by per-row dequant of the head weights in
  CrispASR/crisp_punc/src/pcs.cpp (+ CrispEmbed/src/pcs.cpp mirror); wired crash guard `pcs`.
- **clip_text — BUG FIXED, CLOSED (Gap 4).** Was cos=0.79 vs HF `get_text_features`: localized to
  **tokenization** — the BPE emitted 11 GPT-2-style ids (`220` space between words) vs HF's 7 CLIP
  ids, i.e. it never applied CLIP's `</w>` word-boundary convention. Fixed (`fa66a02`) by passing
  `clip_style=true` to `BPETokenizer::load` in clip_text_embed.cpp (the tokenizer already
  implemented the `</w>` path). Now `final_embedding` cos **1.000000**. Wired `clip-text` in the
  manifest (ref `cstr/clip-text-base-GGUF/clip-text-ref.gguf`). Pre-existing (not a wave regression).

**Methodology lesson (reinforced): a single-stage diff cannot tell a dumper bug from an engine
bug.** gliner's `lstm_out`-only check looked like a BiLSTM regression; multi-stage + the entity
output check proved the engine is fine and the *reference* is dead. Harnesses that check one
stage (nafnet output-only, lfm2_colbert colbert_output-only) MUST be extended to all ref stages
and/or add an independent task-output check before their cos is trusted.

Net: SR/restoration (11) + esrgan/safmn + lilt + lfm2 auto-guarded. Wave regressions found by
tracing: **layout** (encoder, flash_attn) + **nafnet** (conv layout, now fixed). gliner =
broken reference (engine fine); lfm2_colbert = ColBERT-head discrepancy; bert_ner = dumper
written, download-blocked. None of gliner/lfm2/lfm2_colbert backbones are regressions.

### CUDA-backend gaps (2026-07-02, from the Kaggle GPU regression run)

Ran the full 36-model manifest on **Kaggle CUDA** (T4/P100) via
`tools/kaggle/ocr-portfolio-regression` — the GPU counterpart to the CPU
`regression.yml`. This is the FIRST time most engines were exercised on CUDA
(their `expected_text`/refs were all captured on Metal/CPU). Two distinct
CUDA-only problem classes surfaced. **Neither is a regression** — proven: the
2026-07-02 tree-wide clang-format commit is whitespace-only (its non-ws bytes are
just `SortUsingDeclarations` re-ordering + `BreakStringLiterals` splitting, both
semantics-preserving), and `got_ocr.cpp` was reformatted yet got-ocr2 still scores
`cer=0.000` on CUDA.

**How to reproduce (self-contained).** Account = **chr1s4** (NOT chr1str — see
`../kaggle_usage.md`; `export KAGGLE_API_TOKEN=<chr1s4 token from that file>`).
```
kaggle kernels push -p tools/kaggle/ocr-portfolio-regression   # clones main, CUDA build, runs all models
kaggle kernels status  chr1s4/crispembed-ocr-portfolio-regression
kaggle kernels output  chr1s4/crispembed-ocr-portfolio-regression -p ./out   # log = <slug>.log
```
The `.log` is a JSON array of `{stream_name,time,data}`; reconstruct text with
`"".join(e["data"] for e in json.load(open(log)))`. Per-model timeout is now 300s
(`ocr_portfolio_regression.py`); the datasets `chr1s4/crispasr-hf-token` +
`chr1s4/crispembed-ccache` must be attached (already in `kernel-metadata.json`).

**Gap 5 — CUDA teardown crashes (correct output, then a crash on exit).**
Affected: **swinir, dat, tbsrn** (SIGSEGV / signal 11) and **gliner, lfm2_colbert,
layout-heron, lfm2_embed** (SIGABRT / signal 6). In every case the diff prints
COMPLETE, PASSING cosines first (e.g. `swinir  output cos=0.998403 … PASS`,
`swinir_sr: done (256x256)`) and only THEN `ERROR: diff harness died from signal N`.
So this is a **teardown/atexit crash, not a correctness failure**; run_one currently
marks it FAIL because `run_diff` treats `returncode < 0` as fatal *before* parsing.
- **It is NON-DETERMINISTIC (a race).** Across two back-to-back CUDA runs (v6→v7),
  `swinir` SIGSEGV'd once and passed once, while dat/hat/pan/tbsrn failed both. A
  run-to-run teardown race points squarely at the residency-style **background
  heartbeat thread** (or a CUDA stream/context) still touching a buffer while the
  global device is being destroyed — i.e. the harness-tolerance fix below is the
  right primary mitigation (you can't rely on the crash reproducing on any single run).
- **NOT reproducible locally.** Dev box is macOS 26.2 (residency sets active) yet
  `./build/test-swinir-diff …` exits **0**. The crash is CUDA-exclusive; there is no
  local CUDA, so each fix attempt costs a ~50-min Kaggle round-trip (chr1s4 has a
  30h/week GPU quota).
- **SIGSEGV root cause (swinir/dat/tbsrn), high-confidence by inspection:** they call
  `ggml_backend_free(backend)` *immediately after* `load_weights` (swinir_sr.cpp:379,
  dat_sr.cpp:1296, tbsrn_sr.cpp:350), leaving `ctx->wl`'s weight buffers on a
  torn-down backend; `swinir_sr_free` → `core_gguf::free_weights(ctx->wl)` frees them
  later. Harmless on Metal, a segfault against a dead CUDA device at teardown. The
  engines I already moved to CPU weights (nafnet/restormer/esrgan/safmn) do NOT crash
  — corroborating this. **Candidate fix:** keep the weight backend alive in `ctx` and
  free it in the engine `*_free()` AFTER `free_weights` (mirror `gliner_ner.cpp` which
  keeps `ctx->backend`, freed at :1024). Metal-verified-safe, but only CUDA-verifiable.
- **SIGABRT cluster (gliner/lfm2_colbert/layout-heron) is a DIFFERENT bug:** gliner
  already keeps+frees `ctx->backend`, so it is not the free-after-load issue — it is a
  ggml `GGML_ASSERT` firing on CUDA teardown. Diagnose separately (get the assert
  message: it's truncated in the JSON log's stderr; re-run with fewer models or grep
  the full stderr blob). Likely the CUDA analogue of the Metal residency-set assert.
- **The Metal analogue (context, do not conflate):** ggml v0.10.0 added Metal
  *residency sets* — a keep-alive GPU-memory cache with `keep_alive_s = 3*60` and a
  background heartbeat thread (`ggml-metal-device.m:536-588`), plus a STRICT teardown
  `GGML_ASSERT([rsets->data count] == 0)` at `ggml-metal-device.m:612`. It fires if any
  Metal buffer outlives the global device (macOS ≥15). Interim Metal workaround:
  `GGML_METAL_NO_RESIDENCY=1` (`ggml-metal-device.m:775` disables `use_residency_sets`).
  **This env var is Metal-only and does NOTHING for the CUDA crashes above.**
- **Recommended path (what a fresh agent should do):**
  1. **Harness-tolerance fix FIRST (reliable, locally verifiable):** in
     `tests/regression/run_one.py::run_diff` (~line 255), parse stage lines BEFORE the
     `returncode < 0` check; if the diff produced complete, passing stages, PASS with a
     printed WARNING (`[<name>] WARN teardown signal N after valid results`) instead of
     `die()`. This makes the suite honest — a correct-then-teardown-crash is green, a
     crash-before-output stays red. It does NOT hide the crash (warning + still fixable).
  2. Then attempt the SIGSEGV root fix (backend lifetime) on swinir/dat/tbsrn and
     re-run the kernel to confirm empirically.
  3. Diagnose the SIGABRT assert separately (capture the full assert string).

**Gap 6 — CUDA-garbage VLMs (work on Metal/CPU, garbage/hang only on CUDA).**
Affected: **glm-ocr** (`cer=4.245`), **internvl2-1b** (`cer=5.837`), **qwen2vl-3b**
(TIMEOUT), **deepseek-ocr2** (FAIL; exact reason not yet captured — re-run and read its
`model.deepseek-ocr2` block). `cer > 4` means the OCR text is total garbage (the
no-garbage guard still passes because it's varied garbage, not repetition). These all
read the fox line correctly on Metal/CPU (that's what their `expected_text` was captured
on). **It is engine-specific, not blanket CUDA:** got-ocr2 (`cer 0.000`) and qwen3vl-2b
both PASS on CUDA.
- **Not the repetition bug (already fixed):** internvl2/qwen2vl got `argmax_no_repeat_ngram`
  (n=3) in PR #26 (see "OCR engine correctness/stability fixes" below). So CUDA garbage is
  either (a) the vision encoder producing garbage vision tokens on CUDA → the LLM then
  hallucinates/loops, or (b) a separate CUDA decode issue. (a) is the likely culprit — the
  vision towers use conv/attention/flash_attn paths that can diverge numerically on CUDA.
- **Blocker for per-stage diagnosis:** the diff refs `internvl2-1b-ref.gguf`,
  `qwen2.5-vl-3b-ref.gguf`, `paddleocr-vl-ref.gguf` are **NOT on HF** (only the model
  GGUFs), so `run_one` SKIPs their per-layer diff. (glm-ocr DOES have `glm-ocr-ref-full.gguf`.)
- **How a fresh agent should diagnose (on Kaggle CUDA, the only place it repro's):**
  1. Confirm it's the CUDA *backend*, not the model: run the engine with
     `CRISPEMBED_FORCE_CPU=1` (or the per-engine `<ENGINE>_FORCE_CPU`) on the SAME Kaggle
     box — if CPU output is correct there, the CUDA backend is the cause (expected).
  2. Localize with per-stage dumps: internvl2/qwen2vl/glm each have a `test-<eng>-diff`
     binary. Generate the missing refs (add to `tools/kaggle/crispembed-ref-gen/crispembed_ref_gen.py`
     and run its GPU kernel, OR dump CPU output as the "ref" and diff CUDA-vs-CPU on the
     Kaggle box). First-diff the stages: a vision-encoder stage that craters on CUDA but is
     ~1.0 on CPU pinpoints the op (likely a conv im2col, windowed attention, or
     `ggml_flash_attn_ext` that misbehaves on CUDA). Fix at that op; got-ocr2/qwen3vl-2b are
     the working-on-CUDA references to compare graph construction against.
  3. qwen2vl-3b TIMEOUT specifically: it may be hanging (not just slow) on CUDA — check
     whether it's stuck in the vision encoder or an infinite decode; the 300s timeout now
     kills it fast for iteration.

### UPDATE 2026-07-03 — local Ampere CUDA changes the picture (Gap 5 + Gap 6)

A **local NVIDIA CUDA GPU is now available** (RTX A1000 Laptop, Ampere **sm_86**,
4 GB, CUDA 13.0; `build-cuda`, `CMAKE_CUDA_ARCHITECTURES=86-real`). This obsoletes
the "there is no local CUDA, each fix costs a ~50-min Kaggle round-trip" note above
(that was the macOS box). CUDA bugs are now reproducible in minutes — **but only the
backend-agnostic ones**; sm_86 tolerates the older-arch faults. Two distinct fault
classes fell out cleanly:

**Class A — device-pointer weight reads (backend-agnostic; reproduces on any
device-local backend: CUDA/Vulkan/SYCL/HIP).** Several engines dequantized/read a
MODEL WEIGHT on the host by dereferencing `t->data` directly (`memcpy(t->data)`,
`(fp16*)t->data`, `to_float(t->data)`, `return (const float*)t->data`). On a
device-resident weight that pointer is device memory → host deref **SIGSEGV**. Safe
on CPU and **Metal** (Apple unified memory is host-visible — why it "worked on
Metal/CPU"). Fix everywhere: keep the zero-copy fast path only when
`!t->buffer || ggml_backend_buffer_is_host(t->buffer)`, else read via
`ggml_backend_tensor_get`.
- **deepseek-ocr2** — was the Gap-6 "FAIL". SIGSEGV in `precompute_rpe_tables`
  reading SAM `rel_pos` weights. **Reproduced + fixed + runtime-verified on local
  CUDA** (correct fox OCR). (commit 42ef0ea)
- **dat / tbsrn** — listed under Gap 5, but their real crash is Class A, not
  teardown: they SIGSEGV'd 3/3 on Ampere during load-time BatchNorm fusion (dat
  `to_f32` returned `t->data`; tbsrn BN lambda `memcpy`'d `t->data`; note the earlier
  F32-fusion `buf.assign(p,…)` correctness fix is what began dereferencing the device
  pointer). **Fixed + verified: dat cos 0.999995, tbsrn 0.999362, exit 0.** (28fb9b1)
- **unlimited_ocr, math_ocr, smoldocling_ocr, parseq_ocr, tesseract_lstm** — same
  antipattern on weights, fixed by inspection (compile-verified; need their models /
  a Vulkan build to exercise). (42ef0ea)
- **Codebase audit COMPLETE (2026-07-03):** full `->data` census (52 refs / 14 files)
  + ggml host-accessor check (`ggml_get_f32_1d` etc. — none used). Every remaining
  `->data` is safe: the 8 fixed helpers, `granite_vision_ocr` (has the host guard),
  `instructir`/nafnet/safmn (already route via tensor_get), `decoder_embed` (GPU path
  uses `ggml_backend_tensor_set`; direct writes only in the `// CPU fallback` branch),
  `imatrix` (host-allocated gguf ctx), a `%p` debug print, and comments. **No Class-A
  instance remains.**

**Class B — arch-specific vision garbage (Turing/Pascal only; NOT reproducible on
Ampere).** **glm-ocr, internvl2-1b, and qwen2vl-3b all produce CORRECT OCR on local
Ampere sm_86** (glm/internvl2 char-perfect; qwen2vl-3b hits EOS at gen[17]). So the
Kaggle `cer>4` / TIMEOUT are an **older-arch vision-encoder numerical divergence**
(conv/windowed-attn/`flash_attn_ext` diverging on sm_75/sm_60), NOT a graph bug. The
qwen2vl-3b "TIMEOUT" is this garbage driving runaway generation to `max_tokens=2048`
— not a hang or OOM. glm's per-stage diff "FAIL" is a **stale-reference artifact**
(identical crater on CPU; the ref was made by the stale no-rope dump script). Still
open; genuinely needs a Turing/Pascal GPU (Kaggle) to localize the diverging op —
compare against got-ocr2 / qwen3vl-2b which pass on CUDA.

**Gap 5 teardown (swinir/dat/tbsrn free-after-load):** the free-after-`load_weights`
backend-lifetime bug does NOT reproduce on Ampere (sm_86 tolerates freeing the
buffer's backend early), but is real by inspection. Hardened all three: keep the
weight-load backend in `ctx->wl_backend`, free it in `*_free()` AFTER `free_weights`
(mirrors gliner). Kaggle-verifiable. (28fb9b1)

**Still open on CUDA:** Class B (glm/internvl2/qwen2vl-3b, Turing/Pascal); and a
Kaggle T4/P100 run to confirm the Class-A + Gap-5 fixes flip FAIL→PASS on the
original arch.

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
- ~~**ggml Metal device-teardown abort** at process exit when loaded alongside
  PyTorch MPS~~ → **FIXED (2026-07, `fix/metal-v0.10-regressions`)**. Root cause was
  the ggml v0.10.0 bump (`8be60f83`): it added Metal **residency sets** (a 180 s
  GPU keep-alive cache) with a hard teardown assert `[rsets->data count]==0` in
  `ggml_metal_device_free`, so any Metal buffer still alive when the process-global
  device is freed by a C++ static dtor at exit aborts (SIGABRT / exit 134) *after*
  results print — flipping exit codes and making passing one-shot CLI / `test-*-diff`
  runs (and `run_one.py`) report false "signal 6" failures. Fix: a library
  constructor in `src/core/gguf_loader.cpp` sets ggml's own kill-switch
  `GGML_METAL_NO_RESIDENCY` by default (before any Metal init), restoring pre-bump
  behavior. **Env `CRISPEMBED_METAL_RESIDENCY=1`** opts the cache back in for a
  long-lived host (the server / a long-running binding process) — safe because
  those free their contexts via `crispembed_free` on shutdown (verified leak-clean
  with residency re-enabled). The one-shot CLI and all bindings (Python/Rust/Dart
  load `libcrispembed` as a shared lib, so the constructor runs at load) get the
  safe default automatically. **The same leak also crashes CUDA** (SIGSEGV/SIGABRT;
  no `NO_RESIDENCY` switch there), so the real backstop is `core_util::clean_exit(rc)`
  (`src/core/clean_exit.h`): flush + `std::_Exit`, skipping the static-dtor GPU-device
  teardown for the **one-shot** binaries only (CLI `main` wrapper + all 88
  `tests/*.cpp` mains). Backend-agnostic; preserves pass/fail exit codes. Long-lived
  hosts keep `crispembed_free`. Generalizes (and retires) the old `os._exit`
  PyTorch-MPS-coexistence workaround.
- **lfm2 Metal scheduler abort** — also a v0.10.0 regression: `ggml_backend_sched_new`
  now asserts the last backend is CPU. **FIXED** same branch — `lfm2_embed_load`
  appends a CPU fallback backend (fireredpunc issue-#68 pattern).

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

### TrOCR recognizer — improve accuracy + speed (2026-07-07)

The DBNet-ic15 + trocr-small-printed pipeline gives poor results on real
documents. Isolated the cause this session — it is the **models**, not the port
or the runtime:

- **WASM ≡ native, token-for-token.** The PROXY_TO_PTHREAD multithreaded WASM
  pipeline decodes bit-identical tokens to the native CPU run (same 6 boxes,
  same `best=` token every decode step; logits differ only ~3rd decimal from
  SIMD/thread reduction order). So the WASM/threading path introduces zero
  accuracy change.
- **GGUF ≈ HF.** Parity vs `microsoft/trocr-small-printed` (HF transformers) on
  CrispEmbed's own crops: 3/5 identical (`MAMMA`, `LIKE`, `WE`), and HF *itself*
  reads the crops as uppercase fragments (`Mamma`→`MAMMA`, `too`→`TOO`). So the
  conversion is faithful; the low quality is trocr-small's ceiling on
  scene-text-detector crops.
- ~~**The one real GGUF↔HF discrepancy: a trailing repeated subword**~~ — **FIXED**
  (`6791af5`): ported `argmax_no_repeat_ngram`(=3) from qwen2vl_ocr.cpp to
  `math_ocr.cpp`'s graph decoder. Bans tokens completing an already-seen trigram.
  eos/length-penalty parity still TODO but the main repeat bug is resolved.

Accuracy — the bigger levers:
- **Detection under-covers documents.** DBNet-ic15 is an ICDAR-2015 *scene-text*
  detector; on a dense book page it found 6 boxes out of ~40 words. Swap in a
  **document-text detector** (Surya / a PP-OCR DBNet trained on documents) for
  dense-page coverage.
- **Prefer the doc-VLMs for real documents.** PaddleOCR-VL / SmolDocling (below)
  read a whole page directly and beat the DBNet+trocr-small line pipeline; steer
  document OCR there and keep DBNet+TrOCR for the scene/line-crop case.

Speed — the pipeline's dominant cost is **per-region autoregressive decode**
(the token loop is inherently sequential): ~19 s/region on WASM (4 threads),
~25 s/region native (1 thread). PROXY_TO_PTHREAD parallelizes each matmul but
cannot parallelize the token loop, so CPU stays slow. Real speed path is the
**GPU (WebGPU / Metal) recognizer decode**; also consider batching regions
through the encoder and a shorter `--ocr-max-tokens` for line crops.

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

### Optical Music Recognition (OMR) — models to port (2026-07-12)

OMR is "OCR for staff notation": the winning modern approach is exactly the
TexTeller shape — vision encoder + autoregressive transformer decoder emitting
a linearized notation token sequence. This reuses the existing
VisionEncoderDecoder machinery (`math_ocr.cpp` path). Output format is
irrelevant to us (bekern / **kern / MusicXML / LilyPond are all parseable
downstream), so we optimize for arch fit + license, not output dialect.

**Two distinct problems:** printed staff notation (tractable, MIT weights
exist) and handwritten (hard; the real license risk is on the *training
data*, not the code — see landmine below).

| # | Model | Params | License (code / weights) | Architecture | Output | Handles | Effort | Status |
|---|-------|--------|--------------------------|-------------|--------|---------|--------|--------|
| 1 | **Sheet Music Transformer (SMT / SMT++)** | **21.4M** | **MIT / MIT** | ConvNext encoder + vanilla Transformer decoder | bekern | Printed polyphonic / pianoform | **Low** — TexTeller-clone; only new piece is the ConvNext backbone (conv + LN + GELU, no attention — simpler than any ViT already ported) | **TODO — recommended first** |
| 2 | **oemer** | 2× U-Net | **MIT / MIT** (GH releases) | 2 semantic-segmentation U-Nets (staff/symbol) + numpy reconstruction | MusicXML | Printed, phone photos, skewed | **High** — multi-model + heavy rule-based reconstruction; poor ggml fit | Reference/fallback only |
| 3 | Polyphonic-TrOMR (NetEase) | ~22M | **Apache-2.0 / Apache-2.0** (weights committed in-repo) | ViT + multi-head Transformer decoder (parallel rhythm/pitch/lift/note heads) | symbolic text (`clef-G2+keySignature-…`) | Printed polyphonic | Medium | Viable fallback; `homr` (AGPL) wraps it but weights are the clean Apache-2.0 ones |
| 4 | Flova/omr_transformer | ~ | Apache-2.0 / Apache-2.0 (HF) | Donut VisionEncoderDecoder | LilyPond | artificial + **handwritten** + whiteboard ("simple notes" toy) | Medium | Only permissive handwritten lead; low quality |
| ~~5~~ | ~~homr (liebharc)~~ | — | ~~**AGPL-3.0**~~ | pipeline + TrOMR | MusicXML | printed/camera | — | **REJECTED — AGPL** |

**Recommended priority:**

1. **SMT (printed)** — port target. MIT code *and* MIT weights, only 21.4M
   params (quantizes to near-nothing), ConvNext + standard transformer decoder.
   Weights: `antoniorv6/smt-grandstaff`, `-camera-grandstaff`,
   `-string-quartets` (all MIT). Trained on GrandStaff (Ideal + Camera) and
   Quartets. Plan: new `models/convert-smt-to-gguf.py` mirroring the TexTeller
   converter; reuse the `math_ocr.cpp` decoder graph; add a ConvNext encoder
   (new, but the simplest backbone in the roster); bekern tokenizer is a small
   finite lookup vocab (no BPE/Unigram needed). Validate parity vs HF weights
   the usual way.

2. **Handwritten (phase 2)** — no MIT-weights handwritten model with SMT's
   polish exists. Reach handwritten by *fine-tuning SMT on synthetic +
   license-clean handwritten-style data*, same graph. `Flova/omr_transformer`
   is the only permissive handwritten lead but is a toy.

3. **Polyphonic-TrOMR (viable fallback)** — weights confirmed available and
   clean: `tromr/workspace/checkpoints/img2score_epoch47.pth` (86.3 MB) is
   committed directly into the Apache-2.0 repo (not LFS → covered by the repo
   license), with a 4-file tokenizer set (`tokenizer_{lift,pitch,rhythm,note}.json`).
   Architectural wrinkle vs SMT: TrOMR is **not** a single autoregressive stream
   — it has *parallel classification heads* (rhythm / pitch / lift / note) per
   decoder timestep, so a port needs 4 output projections + a merge step, not
   one LM head. Prefer SMT unless we specifically need TrOMR's real-world/camera
   robustness. `homr` wraps this same model but is AGPL — take the weights from
   the NetEase repo, not homr.

**Reuse map (assessed 2026-07-12, feat/smt-omr worktree):** ~70% of the SMT
port reuses existing infra —
- **Decoder + decode loop + C ABI:** `src/math_ocr.cpp` is already SMT's exact
  shape ("Hybrid CNN + ViT encoder → cross-attention Transformer decoder → token
  sequence"): KV-cached decoder, greedy + beam decode, batched encode, per-token
  confidences. SMT's "classic Transformer decoder" == TrOCR == this; port by
  config, not new graph code.
- **Converter:** `models/convert-trocr-safetensors-to-gguf.py` already handles
  the decoder + top-level `decoder_start_token_id`. New `convert-smt-to-gguf.py`
  = that file + a ConvNext encoder tensor mapping.
- **ConvNext encoder (the one new piece):** CrispASR has ConvNeXt blocks in
  `f5_tts / vibevoice / qwen3_tts / kugelaudio / outetts_wavtok` (1-D/audio, but
  identical block: dwconv → LN → pwconv → GELU → pwconv → layer-scale → residual)
  + `core/activation.h`; CrispEmbed has mature `ggml_conv_2d` engines (`swinir`,
  `nafnet`, `cnn_embed`, `adair`, `tbsrn`) for the 2-D image side. Adapt, not
  invent.
- **Shared load/preproc/vocab:** math_ocr grayscale-resize-normalize;
  `core/{gguf_loader,cpu_ops,bpe}.h`; bekern = fixed lookup vocab (simpler than
  any in-tree BPE).
- New work = 2-D ConvNext encoder + bekern vocab + encoder-side converter.

**Confirmed SMT architecture (2026-07-12, from SMT++ source + safetensors header):**
Total **21.4M params, F32, 360 tensors, 85.5 MB** `model.safetensors`. Greedy
manual decode (no HF `.generate()`), seed `<bos>=4426`, stop `<eos>=8822`,
`pad=0`, up to `maxlen=1281` steps.
- **⚠ Convert against SMT++ tensor names, NOT SMT-main.** The shipped
  grandstaff/camera-grandstaff weights only match `SMT-plusplus/smt_model/
  modeling_smt.py` (`input_attention`/`cross_attention`/`ffNet`/`out_layer`); the
  SMT-main repo has a rewritten module whose names match no checkpoint.
  `smt-string-quartets` ships **no weights** (README only).
- **Encoder** = stock HF `ConvNextModel(num_channels=1, num_stages=3,
  hidden_sizes=[64,128,256], depths=[3,3,9])`. Plain ConvNeXt, no attention. Stem
  Conv2d(1→64,k4,s4)+LN; stage-1/2 downsample Conv2d(k2,s2); **16× H/W reduction**.
  Last stage already outputs 256 = `d_model`, so **no encoder→decoder projection**.
  `encoder.layernorm` (pooler LN) is in the ckpt but **dead** on the inference path
  (`last_hidden_state` is pre-pooler). Tensors:
  `encoder.embeddings.patch_embeddings.{weight[64,1,4,4],bias}`,
  `encoder.encoder.stages.{0,1,2}.layers.{i}.{dwconv,layernorm,pwconv1,pwconv2,layer_scale_parameter}`,
  `encoder.encoder.stages.{1,2}.downsampling_layer.{0=LN,1=Conv2d}`.
- **Decoder** = 8 layers, d_model=256, **4 heads** (hd=64), **FFN dim=256 (1×, not
  4×)**, activation **ReLU** (+ `end_relu` before the head). Post-norm:
  self-attn→norm1→cross-attn→norm2→FFN→norm3. Token emb `nn.Embedding[20578,256]`;
  **embeddings NOT tied** to head. LM head = `Conv1d(256→20578,k1)` →
  `decoder.out_layer.weight[20578,256,1]` (squeeze trailing 1 → Linear) + bias.
  Tensors: `decoder.embedding.weight`, `decoder.decoder.layers.{0..7}.
  {input_attention,cross_attention}.{lq,lk,lv,out_proj}.{weight,bias}`,
  `.ffNet.{0,3}.{weight,bias}`, `.{norm1,norm2,norm3}.{weight,bias}`.
- **Positional encodings are NOT in the checkpoint — bake as constants.** (a) 1-D
  sinusoidal added to decoder token embeddings; (b) 2-D sinusoidal
  (`dim=256`, first 128ch=row H, last 128ch=col W, `div=exp(-arange(0,dim//2,2)/dim·ln1e4)`).
- **⚠ Cross-attention key≠value:** encoder output flattened over H×W;
  the 2-D PE is added to the **KEYS only**; **VALUES are the raw** flattened
  features. Query = decoder states. Cross-attn has no mask; self-attn is causal.
- **Preprocessing:** grayscale, **always color-invert** (`RandomInvert(p=1.0)` —
  mandatory, not augmentation), `ToTensor` → **[0,1], NO mean/std normalize**.
  `cv2.resize` bilinear at `reduce_ratio=0.5`, height floored/capped ~256px
  (`maxh=256`, `maxw=3056`).
- **bekern vocab** = fixed word-level lookup (NOT BPE), `out_categories=20578`,
  identical across grandstaff/camera. `w2i`/`i2w` embedded in `config.json`
  (875 kB) and as `vocab/*.npy`. Split GT on whitespace/`·` delimiter; layout
  tokens `<b>` break / `<s>` space / `<t>` tab.
- **SMT vs SMT++:** identical neural graph; SMT++ gains are training-side
  (curriculum + synthetic full pages). Full-page = same graph, bigger images +
  longer decode + layout tokens, no extra module. **Target single-system
  grandstaff first** (the only checkpoints with published weights).

**Port progress (2026-07-12, feat/smt-omr worktree):**
- ✅ `models/convert-smt-to-gguf.py` — torch-free, verbatim SMT++ names, squeezes
  `out_layer` 1×1 conv→Linear, bakes 1-D decoder PE, records `smt.scale_attention=
  False`. Verified GGUF: arch `smt_ocr`, 361 tensors, 20578-tok vocab, 83 MB.
- ✅ `tools/dump_smt_reference.py` — loads REAL SMT++ model (hooks, not a
  re-forward), dumps 18 per-stage F32 tensors → `smt_ref.gguf`. Validated on a
  real GrandStaff test image: enc 336×128→`(256,8,21)` (16× reduction, 168 mem
  tokens), decode emits correct bekern (`**ekern_1.0 <t> … *clefG2 <b> …`).
  Test assets in scratchpad: `smt-grandstaff/`, `SMT-plusplus/` clone, `gs_test0.png`
  (+ `.gt.txt`), `smt_ref.gguf`. Note: cloned `SMTConfig` needs a
  `super().__init__(**kwargs)` patch to load under transformers 4.57.
- ✅ `src/smt_ocr.{h,cpp}` ggml engine (ConvNext encoder + cross-attn decoder +
  greedy decode) + `tests/test_smt_diff.cpp` + CMake wiring. **Full per-stage
  parity vs `smt_ref.gguf` (CPU):** enc_stage0/1/2 + enc_output + mem_key
  cos_min ≥ 0.999996; dec_tok_emb + dec_layer0–7 + logits cos_min = **1.000000**.
  Native greedy decode emits correct bekern (header/clefs/meter/barlines match
  GT exactly; `*k[]` vs GT `*k[b-]` is the model's own prediction — the Python
  ref emits `*k[]` too). Bugs found & fixed during bring-up: (a) off-trunk
  `enc_stageN` snapshots weren't in the graph (`to_tokens` forks off the trunk)
  → `ggml_build_forward_expand` each; (b) `crispembed_diff.h` GGUF reader only
  decodes F32 (its I32 branch checks a stale type id 5, but this ggml tags I32
  as 26) → dumper now stores `token_ids` as F32.
- ✅ Preprocessing parity: `recognize_raw` now does cv2-bilinear resize +
  RandomInvert + BGR-as-RGB grayscale → native decode is **token-identical to
  HF** on real GrandStaff scores (100% on 3/4; 4th matched to the ref cap), CPU
  and Metal.
- ✅ Wiring: `src/crispembed.cpp` dispatcher (`arch=="smt_ocr"` → all 4 switches),
  so `crispembed -m smt.gguf --ocr score.png` works end-to-end (verified 69/69 vs
  HF); `smt_ocr_recognize_raw` added; `examples/cli/model_mgr.cpp` registry entry
  (`smt-grandstaff`). Server/bindings inherit via the generic `crispembed_ocr_model_*`.
- ✅ Quantize: `tools/quantize.cpp` keeps SMT conv kernels (`dwconv`/`downsampling`)
  and the baked PE (`positional`) F32; engine reshapes the quantizer's flattened
  2-D conv headers back to 4-D. **q8_0 (24 MB) decodes identically to HF (100%);
  q4_k (17 MB) is too lossy for the AR decode (~32%) — ship f32 + q8_0 only.**
- ✅ KV-cache: incremental decode (cross K/V precomputed once, self K/V grown per
  step via concat). Token-identical to the full-recompute path (kept behind
  `SMT_OCR_FULL_DECODE=1` for A/B) and to HF, CPU + Metal. **5.4× faster** (0.37 s
  vs 1.98 s for ~100 tokens); the gain grows with sequence length.
- ✅ GGUF upload: `cstr/smt-grandstaff-GGUF` (f32 83 MB + q8_0 24 MB + MIT model
  card; card license verified `mit`). Registry auto-download works end-to-end.
- ✅ **Preprocessing corrected + port re-validated (this was wrong before).** The
  earlier engine used `reduce_ratio=0.5` + a cv2-BGR channel swap — WRONG. The
  authoritative pipeline (SMT-main `data.py::prepare_data`) is RGB, `reduce_ratio=
  1.0`, `width=min(w,3056)`, `height=max(h,256)`, no swap. Fixed in `recognize_raw`
  + the dumper. **Port parity re-verified on 10 fresh GrandStaff images: C++ ==
  Python blueprint = 100.00% token agreement** (8/10 byte-identical; 2 differ only
  because the Python ref was maxlen-capped — prefixes 100% identical).
- ⚠️ **Model accuracy caveat (NOT a port bug).** vs ground truth the model scores
  only ~30% on the clean `antoniorv6/grandstaff` test split — it reads clefs but
  misreads key/time signatures and degenerates (no `<eos>`) on some images. This
  is faithfully reproduced by the port (SMT-plusplus forward is the correct one;
  SMT-main's forward gives 0% garbage on this checkpoint). Ruled out: measurement,
  preprocessing, image quality, weight-load (360/360), attention-scaling. Open
  hypothesis: data-distribution mismatch (model tagged `camera_grandstaff` vs the
  clean test images). **Verify on the true Camera-GrandStaff distribution before
  claiming production OMR quality.** The C++ port itself is exact.
  Lesson recorded: [[validate-intermediates-and-outputs]] — "100% vs my own
  same-preprocessing reference" hid both a preprocessing bug and a broken
  accuracy metric; only decoded-output-vs-ground-truth surfaced them.

**Landmines:**
- **⚠ SMT attention is UNSCALED.** `MHA.forward` computes `bmm(q,k)` then softmax
  with **no** `1/sqrt(head_dim)` — `self.scale_factor` is defined but never
  applied (verified in source, not the abstract). The C++ must NOT scale QK^T
  (converter records `smt.scale_attention=False`). Also: token embeddings are
  **not** scaled by `sqrt(d_model)` (no `scale_embedding`).
- **Cross-attn key≠value:** memory_key = flattened encoder features **+ 2-D PE**;
  memory_value = **raw** flattened features. Easy to wire both to the same tensor.
- **Encoder `last_hidden_state` is pre-pooler-LN** → `encoder.layernorm` in the
  ckpt is dead weight; don't apply it. Feature map is `(256, H/16, W/16)`.
- **Handwritten training-data license trap:** the canonical handwritten OMR
  datasets — **MUSCIMA++ / CVC-MUSCIMA — are CC BY-NC-SA (non-commercial)**.
  Training weights on them contaminates the *weights* for commercial use (same
  pattern as the old PosFormer/BTTR/HMER math models). PrIMuS / Camera-PrIMuS /
  GrandStaff are printed/synthetic and license-clean. Keep handwritten training
  data NC-free from day one if shipped weights must be commercially usable.
- **VisionEncoderDecoder `decoder_start_token_id`** comes from the *top-level*
  config, not the nested decoder config (the TexTeller start-token bug that
  poisoned position-0 KV — see the TexTeller 3.0 entry above). SMT's converter
  must resolve the start token the same way.
- Watch F16 Metal matmul overflow on large activations (see
  [[metal-mul-mm-f16-overflow]]) as with all VED ports.

**Sources:** SMT [github.com/antoniorv6/SMT](https://github.com/antoniorv6/SMT) ·
[SMT++](https://github.com/antoniorv6/SMT-plusplus) ·
[HF smt-grandstaff (MIT)](https://huggingface.co/antoniorv6/smt-grandstaff) ·
[PRAIG collection](https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd).
oemer [github.com/BreezeWhite/oemer (MIT)](https://github.com/BreezeWhite/oemer).
TrOMR [github.com/NetEase/Polyphonic-TrOMR (Apache-2.0, weights `img2score_epoch47.pth` 86 MB in-repo)](https://github.com/NetEase/Polyphonic-TrOMR).
[Flova/omr_transformer (Apache-2.0)](https://huggingface.co/Flova/omr_transformer).
homr [github.com/liebharc/homr (AGPL-3.0)](https://github.com/liebharc/homr).

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

- [x] **Adopt F16 ggml KV cache** — all decoder engines now use persistent device KV.
  pix2struct: **DONE** (`088d359`) — F32 std::vector KV cache + cross-attn pre-compute.
  lightonocr: **DONE** (`485cb97`, branch `lighton-perf`) — 2.09x total speedup.
  granite_vision_ocr: **DONE** (`66b8de2`).
  smoldocling_ocr: **DONE** (`bc329e4`, branch `feat/smoldocling-kvcache-prefill`).
  qwen2vl_ocr: **DONE** — already had F16 kvc; fixed CPU round-trip in seeding.
  math_ocr (TrOCR): **DONE** (`230f190`, `perf/trocr-persistent-kv`) — persistent
  device-side F32 KV cache with ggml_view/ggml_cpy. 4x speedup (19s→4.4s/region).
  Cross-attn K/V uploaded once. Also fixed WASM full-pipeline crash (was OOM from
  per-step re-uploads).
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

### Blueprint: WASM build target — DONE (reworked July 2026, issue #31)

`build-wasm.sh` (OCR: single-model + DBNet/TrOCR pipeline + scan cleanup;
2.2 MB wasm) and `build-embed-wasm.sh` (text embeddings). Three tiers:
plain SIMD CPU, `--threads` (COOP/COEP; coi-sw.js service worker makes
GitHub Pages crossOriginIsolated), `--webgpu` (emdawnwebgpu/JSPI,
Chromium-only; six local WGSL kernels in patches/ggml-webgpu-ops.patch —
NORM/IM2COL/POOL_2D/CONV_TRANSPOSE_2D/UPSCALE/ARANGE, upstream draft at
CrispASR tools/upstream-prs/22). Demo (examples/wasm-ocr) runs inference
in a Web Worker, auto-picks the best tier, deploys to
https://crispstrobe.github.io/CrispEmbed/ via deploy-pages.yml.
Verification: node suites + headless-Chromium e2e in build-wasm.yml on
every push (byte-equality vs native CLI ground truth); ggml
test-backend-ops executes in-browser for the WebGPU kernels; release
bundles attach via release-wasm.yml on version tags. Measured (M1):
pix2tex ~2.8x vs wasm CPU on WebGPU; DBNet detection ~60x; det+rec
pipeline ~1.8x. History/learnings: HISTORY.md July 4-5 2026 entries.

Done: OPFS cache; --webgpu-compat (Asyncify) tier; per-engine sweep (6/6
correct; trocr 4x); decoder-on-CPU split (MATH_OCR_DEC_CPU=1, on for the
demo's webgpu tiers — decode at CPU speed, pipeline ~wash, parity improved).
Remaining idea (unscheduled): cross-region batched decode in the pipeline.

## Runtime speedup roadmap (2026-07-11 sweep)

Source: full runtime re-verification, 2026-07-11 — see `PERFORMANCE.md →
"Runtime Optimization Audit — Re-verification (2026-07-11)"` for the verified
state tables and the corrected June-audit claims. This is the actionable
backlog. **Every item needs a target GGUF model to verify against (q8_0
preferred, to isolate the change from q4_k quant noise) and a before/after
parity + latency measurement — do NOT land a "perf" change on a compile-only
check.** The June audit's "flip to init_best" for the SR engines was a mirage
(they are CPU-pinned deliberately); treat all "easy win" labels as unverified
until the code confirms them.

### Tier 1 — higher-ceiling (each needs its own design pass + a target model)

#### 1. Decode-step graph cache (per-backend gated) — the #1 unrealized lever

Problem: no runtime reuses the built cgraph. Every autoregressive decoder does a
full graph rebuild + `ggml_backend_sched_reset` + `ggml_backend_sched_alloc_graph`
**per generated token**. Device-resident F16 KV caches landed across the VLM
decoders, but the graph *around* the KV is still rebuilt each step.

Plan:
1. Pick a reference decoder that already has persistent KV + a single decode
   graph — `deepseek_ocr2` (`deepseek_ocr2.cpp:1558`, `lag.gf` built once) or
   `math_ocr` (dkv). Instrument per-step: graph-build vs alloc-plan vs compute.
2. Cache the decode-step graph + a persistent gallocr/sched reservation keyed by
   a **bucketed KV length** (pad T to a bucket; rebuild only on bucket cross).
   Templates in-tree: text encoder `sched_reserve`+T-bucket
   (`crispembed.cpp:1202-1217`), lfm2 (`lfm2_embed.cpp:452-457`).
3. **Cache the decode-step graph only — NOT the encoder graph.** Encoder-graph
   caching is a measured dud on compute-bound work and a known GPU
   use-after-free (CrispASR #235 disabled it in 8 backends).
4. **Per-backend gating mandatory.** Graph reuse traps `unreachable` on WebGPU
   (`09dc519`) — keep OFF there. Respect the scheduler landmine
   (`ggml_backend_sched_reset` does not null `tensor->buffer`; run side graphs
   before the main alloc — CLAUDE.md / LEARNINGS).
5. Generalize decoder-by-decoder behind an env flag, measuring each.

Verify: q8_0 model per decoder; output cosine vs pre-change baseline ≈1.0 (graph
structure identical, only reuse changes) + per-token latency drop.

#### 2. ggml-metal ICB (indirect command buffer) replay — Apple-side decode

Problem: Metal LLM/TTS decode is per-op-dispatch bound (~100 ms/step on ~280-op
graphs). The CUDA side already solves this via CUDA-graph capture (CrispASR
§210, ~9–13× on RTX). ggml-metal has no ICB replay path.

**FEASIBILITY MEASURED 2026-07-11 (for CrispEmbed decoders).** The shared ggml
submodule already carries CrispASR's §210 ICB-feasibility probe
(`ggml-metal-context.m:438`, env `CRISPASR_METAL_PROFILE=1`), which splits each
`graph_compute` into host-encode time (what an ICB replay collapses) vs
GPU-execute time (what it cannot). Ran it on a trocr decode:

| graph | nodes | encode_us (ICB removes) | gpu_us (ICB can't) |
|---|---|---|---|
| encoder (ViT) | 386 | 10188 | 58658 |
| decode step, cold (1st) | 355 | 9679 (pipeline compile) | 6733 |
| **decode step, warm** | 355 | **731 (18%)** | **3335 (82%)** |

**Verdict for CrispEmbed decoders: ICB is a ~18% win at best** — it collapses only
the host-encode portion of a warm decode step; the GPU-execute 82% (per-kernel
launch latency across ~355 sequential ops) is untouchable by ICB. The cold-step
9.7ms encode is one-time pipeline compilation (a persistent decode graph /
pipeline cache addresses that, once). So a full ggml-metal ICB port is **NOT**
justified for CrispEmbed's (light) decoders — the GPU-execute majority is the real
cost, so the lever is op-count / kernel-efficiency reduction (fewer, bigger ops
per step), not ICB.

**Caveat:** this is trocr (355-node decode). CrispASR's heavy LLM decoders
(granite/voxtral4b, the ~100 ms/step floor) may have a larger encode fraction —
their §210 probe measures it there, and CUDA-graph capture already gave them
9–13× on CUDA (where launch overhead is worse than Metal's encode). Re-measure
per-decoder with `CRISPASR_METAL_PROFILE=1` before committing to any ICB work.

Original plan (only if a decoder measures encode-bound):
1. Depends on a stable, reused per-step graph (item 1) to capture an ICB once.
2. Prototype ICB encoding in ggml-metal; replay per step, updating only buffer
   offsets / position inputs. Upstream ggml contribution; coordinate with CrispASR.
Verify: Apple-GPU target model; per-step latency before/after; output parity.

#### 2b. Op-count / kernel-efficiency reduction (the lever the ICB probe points to)

Since the ICB probe showed warm Metal decode is GPU-execute-bound (~82%, i.e.
per-kernel launch latency across ~355 sequential ops), the tractable in-tree lever
is **fewer, bigger ops per decode step** — each fused op is one fewer Metal
dispatch. Candidates: fuse the per-layer norm+scale+bias chains, fuse QKV, fuse the
GLU/SwiGLU elementwise chain, prefer `ggml_soft_max_ext` (scale+mask+softmax in
one op) over the 3-op manual form. This is **per-decoder graph surgery** in each
`build_decoder_step_graph` — risky (easy to change numerics) and must be verified
per model (output cosine ≈1.0 + node-count + per-step latency). NOT a blanket
change; scope one decoder, count nodes before/after, measure. Deferred — needs a
stable benchmark and per-model care; lower confidence than the SIMD-hot-path wins.

#### 3. Real SR-GPU fix — conv weight residency (unblocks 4 SR engines)

Problem: `esrgan_sr`, `safmn_sr`, `restormer`, `instructir` are CPU-pinned
**deliberately** — conv weights loaded via `init_best` land in a Metal/CUDA
buffer the CPU conv scheduler can't read → "pre-allocated tensor in a buffer
that cannot run" abort on Metal / segfault on CUDA (same class as the nafnet
residency bug). `instructir` also hits a Metal `mul_mv` f32×f16 pipeline-compile
bug. NOT the "flip to init_best" the June audit implied.

**CORRECTION (2026-07-11 diagnosis — the audit's premise was wrong):** there is
NO GPU sibling to match. The *entire* SR family computes convolutions on a
CPU-only `enc_sched` — `swinir_sr.cpp:447` literally prints
`conv path = ggml_conv_2d (CPU sched)`. The `init_best` in dat/hat/swinir is only
the weight-LOAD backend; they then allocate a *separate* CPU-resident F32 weight
context (`swinir_sr.cpp:439`, `dat_sr.cpp:316`, `hat_sr.cpp:453`) and copy the
dequantized weights into it, so the CPU conv can read them. esrgan/safmn/
restormer/instructir just skip that copy and load straight on CPU. So SR-on-GPU
is a genuine unsolved item (needs Metal `ggml_conv_2d` for these shapes + a
GPU-resident weight/graph path the whole family currently avoids), NOT a
residency toggle. Reprioritize DOWN — it is research, not a quick win, and none
of the "working" engines demonstrate it.

Real, tractable SR-CPU wins found in the same diagnosis (see Tier 2): safmn
ignored its `n_threads` (hardcoded 1-thread conv sched) — DONE, ~2.3× on an
8-core Mac, bit-identical output.

### Tier 2 — safe, self-contained wins (verified against code)

| Win | File | Status | Note |
|---|---|---|---|
| text_sr scalar conv → SIMD `conv2d_cpu` | `text_sr.cpp:33` | **DONE (merged)** | numerically-equivalent delegation; compiles clean; runtime parity pending a model (none provisioned, no registry URL) |
| safmn honor `n_threads` (was hardcoded 1) | `safmn_sr.cpp:181,255` | **DONE — verified** | ~2.3× (16.2s→7.1s, 8-core Mac) on safmn-x4, bit-identical output; convs run on CPU sched (not Metal) |
| tps_locnet weight dequant cache | `tps_locnet.cpp:262-314` | **DONE** | conv + FC weights now dequantized (and FC-transposed) once at load; predict() reuses them. Bit-identical by construction; helps repeated-predict callers (the bundled `tps_auto_dewarp` does one predict per load, so it's neutral there). Compile-verified; no model in registry to runtime-measure |
| scunet WMSA window-loop threading (mixtex pattern) | `scunet_denoise.cpp:218` | **DONE — byte-identical; measured ~1.15× end-to-end** | per-window attention loop was serial regardless of `-t` (nW=1024 at 256²); now threads across `scunet_init`'s n_threads (file-scope `g_wmsa_threads`, per-thread scratch), `SCUNET_WMSA_SCALAR=1` = serial baseline. Output byte-identical to pre-change main at `-t 1` AND `-t 8` (PPM cmp). Re-measured 2026-07-12 (calmer box, in-process CRISPEMBED_SCUNET_BENCH totals, same-binary env-toggled): serial 15.7 s vs threaded 13.7 s at `-t 8` on scunet-color 256² — **~1.15× end-to-end**, NOT mixtex's 1.94×, because post-MLP-GEMM the window attention is a minority cost (convs/projections dominate every stage). Honest verdict: keep (byte-identical, opt-in via `-t`), but scunet's next real lever would be its conv/projection stages, not more attention work. |
| scunet Swin MLP → SIMD GEMM | `scunet_denoise.cpp:366-387` | **DONE — verified 1.69×** | WMSA was already SIMD (dot_product QK^T + linear_batch_cpu projections); the surviving scalar hot loop was the per-pixel MLP. Now batched into two `linear_batch_cpu` GEMMs. **11.74s→6.96s on scunet-color 256², output byte-identical (0 pixel diff).** (The "uncached dequant" at `:32` is once/block, not per-pixel — confirmed marginal, skipped.) |
| gliner mlp_2layer + fuser out-proj → SIMD GEMM | `gliner_ner.cpp:978-996,1430-1438` | **DONE — output-identical, but marginal on deberta** | hand-rolled per-token scalar matmuls → `linear_batch_cpu`. Output bit-identical (all 7 entities + scores match). Measured on gliner-deberta: head-passes ~5.9→5.6ms (~0.15% of the 203ms total). gliner-deberta is **encoder-bound** (177ms) and does NOT exercise the fuser path; the fuser conversion targets the audit's flagged layer-fusion `[enc_hidden,enc_hidden]`/token hotspot for **multi-layer variants** (unverified — need such a model). **Real gliner lever = the DeBERTa encoder** — DONE below. |
| gliner DeBERTa encoder: dedup disentangled rel-pos projection | `gliner_ner.cpp` c2p/p2c blocks + `prepare_deberta_rel_pos` | **DONE — verified 1.28–1.71×, byte-identical** | Step-0 `CRISPASR_METAL_PROFILE` split showed the encoder is **~96% GPU-execute / ~4% host-encode** (942 nodes, encode ~3.3ms vs gpu ~70–90ms) — so the lever is GPU compute, not op-count/dispatch. The c2p+p2c position matmuls (`k_w/q_w @ [H, T*T]`) were **~88% of per-layer matmul FLOPs** yet projected the full `T*T` pair-grid, when only `≤2T-1` *distinct* rel-pos buckets exist. Fix: project the unique bucket embeddings once (`[H, n_used]`), then `ggml_get_rows` to expand — output-identical (same floats, same per-column reductions; also drops the p2c `[H,T*T]` transpose-cont). **Encoder short (T≈40) 73→57ms (1.28×); long (T≈90, the ~177ms regime) 279→163ms best (1.71×)** — win scales with T (saving is O(T²−T)). All entities/spans/scores byte-identical vs a same-commit baseline build. Metal-confirmed (MTL0), edge cases (single-word T small) sane. |
| gate debug `fprintf` behind verbosity | layout_detect, surya_det, ocr_detect | **DONE (all 3)** | **layout_detect** (verified on layout-heron-f32 + `scan_page_pd.png`, spew→0 lines, same 20 regions): `detect()` printed per-layer/per-call tensor-stat spew (`dec*_after_sa`, `dec0 values`, `dec0 cross_out …`, `dec%d_norm2/3`, `NOT FOUND`, `%zu detections`) with NO guard, and the backbone per-block block ran ~20 `ggml_backend_tensor_get` GPU→CPU read-backs + min/max scans **every call** with only the *print* LDBG-gated — all gated on `layout_debug()` (bonus: skips the debug read-backs/scans in production). **surya_det**: `surya_det: loaded …` init line gated on `ctx->dump`. **ocr_detect**: `loading …` / `prob_thresh …` / `loaded N tensors` init lines gated on new `OCR_DETECT_DEBUG` (the `missing stem/deconv` lines stay — real load errors). All side-effect-free guard wraps (detection output unchanged by construction); surya/ocr compile-verified (CLI text-detect path not exercised — machine saturated). |
| **layout_detect Phase-2 `cpu_linear` scalar matmul → contiguous SIMD AXPY** | `layout_detect.cpp:1018` | **DONE — byte-identical, ~1.26× Phase-2 (best-of, noisy box)** | Measured first (`CRISPEMBED_LAYOUT_DETECT_BENCH`): Phase-2 decoder dominates (Phase 1 ~11–13s, heads ~5ms), and the cost is `cpu_linear` (`:1018`) — a scalar, un-SIMD, **stride-N** triple-loop matmul (dims up to 256×256×8400, ~10×/layer×6) — NOT the deformable sampling loop (a corrected guess). Rewrote the inner compute as `y[o,:] = b[o] + Σ_i W[o,i]·x[i,:]` **contiguous AXPYs over N**: `x[i,:]` and `y[o,:]` are contiguous in the col-major `[dim,N]` layout, so it vectorizes, while the **per-output accumulation order (i ascending) is unchanged → byte-identical** (verified: `diff` of the 20-region output vs the pre-AXPY build is empty). A/B of two back-to-back binaries (pre-AXPY vs AXPY, layout-heron-f32 + `scan_page_pd.png`) under ambient loadavg ~20: **Phase-2 best-of BASE 13548 ms → AXPY 10782 ms (~1.26×, −20%)**; AXPY was also markedly more load-stable (10.8–11.7s vs BASE 13.5–36s — better cache behavior under memory contention). Best-of is the right metric here (noise only inflates times); the box was too loaded (competing python/flutter, spikes to loadavg 137) for a tighter number, but the direction is unambiguous across all reps. cpu_linear is only *part* of Phase 2 (FFN/self-attn go through the ggml BLAS graph), so ~1.26× Phase-2 implies the matmul itself sped up substantially. |
| **layout_detect backbone conv: `ggml_conv_2d_direct` → `ggml_conv_2d` (im2col+GEMM), default flipped** | `layout_detect.cpp` `conv2d_dispatch` | **DONE — verified ~9.8× Phase-1, cos≈1, default flipped** | metal-prof on the backbone graph: **505 nodes, ~99.6% GPU-execute (gpu_us 11.69s / encode 44ms)** — pure Metal conv, and `ggml_conv_2d_direct` is a poor Metal kernel for the 640² shapes while `mul_mm` (GEMM) is highly optimized. Swapping the two `conv_relu`/`conv_silu` sites to `ggml_conv_2d` (im2col+GEMM, F32 since `prep_conv` casts the kernel to F32) drops **Phase-1 backbone 11436 ms → ~1200 ms (~9.8×)**. Output is cos≈1 vs direct — **same 20 regions, same labels, same rank order**, only ≤0.001 score / ≤0.1px bbox jitter (FP reduction order; both F32). Backbone is conv-heavy (the dev-doc "flip it" regime, not attention-bound), and the `test_layout_diff` regression gate is **cos≥0.99** (backbone stages), which cos≈0.9999 clears (PyTorch ref is itself im2col-based, likely *closer*). **Default flipped to im2col**; `LAYOUT_CONV_DIRECT=1` restores the old direct path (verified byte-identical to the pre-change baseline). Biggest single layout win — Phase-1 was the dominant cost after the Phase-2 AXPY. |
| other `ggml_conv_2d_direct` users (glm_ocr, got_ocr) → im2col? | `glm_ocr.cpp:806`, `got_ocr.cpp:842-852` | **MEASURED marginal — do NOT swap** | Sweep after the layout win found only glm_ocr (1 vision patch-embed conv) + got_ocr (4 SAM-neck convs). **got_ocr measured** (`CRISPEMBED_GOT_OCR_BENCH`, got-ocr2-q4_k + `scan_page_pd.png`, quiet box loadavg 5): total 6738 ms = vision_encoder 3219 + **neck_projector 396 (~5.9%, and the conv_2d_direct convs are only *part* of it)** + prefill 271 + **decode 6214 (~92%, 499 steps × ~12 ms)**. So the conv swap is ~4% at absolute best — the flagged-micro-gap-that-isn't-the-dominant-cost case; skipped. glm_ocr (single patch-embed conv, same decode-dominated VLM shape) is marginal by the same reasoning. **The real got_ocr cost is decode**, and a decode-step metal-prof showed each 940-node step is **~89% GPU-execute / ~11% host** (2.5 ms encode / ~20 ms gpu, synced) → **compute-bound**, so the decode-step graph cache / ggml-metal ICB would only touch the ~11% host slice (~10–17% per step) — a modest project. **UPDATE: the graph cache is now DONE** (`GOT_OCR_DECODE_CACHE=1`, sched-free gallocr reserved at max KV — host build+alloc 4.66→0.61 ms/step, byte-identical; see Tier-1 item 1). surya_det/ocr_detect already use `ggml_conv_2d` — no action. |
| conv2d_cpu → true im2col+GEMM + multithread | `core/cpu_ops.h:345` | **CLOSED — measured marginal, skip** (2026-07-11) | Two corrections. (1) The row's premise was wrong: the gather is done ONCE per output position and shared across all out-channels (`patch_buf` + per-oc `dot_product`) — there is no per-oc regather. The remaining headroom is cache-blocking across positions (real but bounded). (2) Consumer audit: no default path with a verifiable model is dominated by it. hmer/bttr/posformer encode via ggml graphs (measured: hmer enc 650 ms ggml, bttr 547 ms ggml); surya_det's scalar path is a fallback (default graph path restored, see the surya grouped-conv fix); got_ocr/deepseek/unlimited use it only in fallbacks; text_sr has no model; ppformulanet-S has no GGUF on HF. The one measurable default-path consumer is **ppformulanet-L's neck+projector** — first measured at ~10% of total on what turned out to be the CPU-only build; re-measured on a real Metal build (2026-07-12): ViT graph 3.4 s, encoder 8.3 s (→ neck+projector ≈ 4.9 s, ~59% of encoder, **~18% of the 27 s total**), decoder 18.7 s (**~69% — the dominant cost**). The skip verdict stands (decode dominates; the real ppformulanet-L lever is its decoder, the op-count/dispatch class), but if this engine ever matters, the neck is now a meaningful secondary target — move it into the ggml graph rather than optimizing the CPU helper. Same class as the got_ocr neck (~4%, skipped). If ppformulanet-L ever matters, the better lever is moving its neck into the ggml graph, not optimizing the CPU helper. |
| surya_det default graph path: grouped pointwise conv | `surya_det.cpp:667` (`g_conv`) | **FIXED (fix/surya-grouped-conv)** | Found while auditing conv2d_cpu consumers: the DEFAULT graph path crashed on EVERY detection — `GGML_ASSERT(a->ne[2] == b->ne[2])` at `ggml.c:4472`, because LiteMLA's `agg_pw` conv is grouped (groups=3·heads, neither depthwise nor groups=1) and `ggml_conv_2d` has no groups support. Pre-existing since the port (verified: 06c02ee crashes identically); only `SURYA_DET_SCALAR=1` worked, which is how the port was verified. Fix: 1×1 grouped conv expressed as ONE batched `ggml_mul_mat` over the group dim. Verified: all 39 boxes byte-identical to the scalar reference on Metal AND forced-CPU; graph path ~2× the scalar path (18.5 s vs 38 s) and is the default again. Lesson (LEARNINGS): verify the DEFAULT path, not just the reference/A-B path. |
| restormer single-pass variance | `restormer.cpp:101` | **Resolved — audit was WRONG, no change** | Re-read `rst_layernorm_bf`/`_wb` (`restormer.cpp:101,120`): each computes mean once then variance once (three passes, ONE variance) — there is no double-variance dead work. (Fusing mean+sumsq into one pass would break byte-identity for a negligible gain on tiny C; restormer is conv-dominated by `rst_conv2d`'s 6-deep scalar loop at `:75` anyway.) Classic verify-handover-claims case — premise didn't survive a read. |

Already-done (audit was stale, do NOT "fix"): tbsrn PE2D is already cached
(`tbsrn_sr.cpp:425`, `pe_cache`).

Negative results (measured — do NOT re-chase as cheap wins): **esrgan intra-op
threading**. `esrgan_process_float_ggml` pins the sched to 1 thread every compute
(`esrgan_sr.cpp:266`, `fn(be, 1)`), so the init-time thread count is irrelevant.
Wiring line 266 to honor `n_threads` and measuring properly: `-t 1` = 21s,
`-t 8` = **33s** (SLOWER), output bit-identical. esrgan tiles into 128px pieces;
each tile's conv is too small to amortize thread overhead, and `-t 8` on 4 P-cores
oversubscribes. The 1-thread pin is the *safe* choice — reverted the change.
The real esrgan lever is **tile-loop parallelism** (run whole tiles concurrently,
each single-threaded), which needs per-thread backend+scheduler replication
(the tile loop shares one `ctx->enc_sched`) — a real concurrency project, not a
one-liner, and hard to verify reliably on a loaded dev machine. safmn's fix WAS a
real 2.3× because it convolves the *whole* image in one graph (no tiling), so its
convs thread-scale; esrgan's tiled convs do not. Different situations.

### Prioritization update (2026-07-11) — cheap wins are exhausted

Three flagged micro-gaps in a row (esrgan threading, decode-step graph cache,
scunet dequant) measured as marginal/inert because the real bottleneck is
elsewhere (see LEARNINGS "measure the DOMINANT cost before fixing a flagged
micro-gap"). The genuine remaining levers are projects, each needing a stable
benchmark harness:

1. **SIMD/ggml-ify the scalar-compute hot paths** — the actual dominant costs:
   ~~scunet WMSA+MLP (`scunet_denoise.cpp:310-390`)~~, ~~mixtex Swin window attention
   (`mixtex_ocr.cpp:126`)~~, ~~layout_detect deformable cross-attention~~. Highest
   ROI; verifiable per-engine with a downloadable model; do one engine end-to-end
   as a proof. ALL THREE DONE — scunet closed 2026-07-12: MLP GEMM (caaf082,
   1.69×) + the WMSA window loop threaded (mixtex pattern, byte-identical,
   below); the remaining WMSA attention math was already SIMD.
   - **mixtex DONE (2026-07-11) — but the lever was THREADING, not SIMD.** The
     Swin attention math was already SIMD (`816a88a` dot-product) + ggml-batched
     (`2453e04`); measuring showed the encoder (2395 ms, ~48% of a 5009 ms total
     on mixtex_pow) was still bottlenecked by the **serial per-window loop**
     (`mixtex_ocr.cpp:741`) — 270 independent windows in stage 0, run
     single-threaded regardless of `-t`. Each `window_mhsa` is self-contained
     (own scratch, disjoint output slice), so the loop parallelizes
     byte-identically. Now honors `ctx->n_threads` (default `n_threads=1` keeps
     the old behavior; `MIXTEX_WMSA_SCALAR=1` forces serial). **Isolated A/B at
     `-t 8`, best-of-3: encoder 1420 → 733 ms (1.94×)** (full no-`-t` baseline
     was ~2350 ms → ~733 ms once ggml's other-op threading is also on).
     Byte-identical LaTeX on mixtex_pow + formula_quadratic. The lesson matches
     safmn/esrgan: for an already-SIMD scalar kernel over independent units, the
     next lever is **loop-level parallelism**, not more SIMD.
   - **mixtex DECODER DONE (2026-07-11) — a redundant-work bug, not a kernel.**
     The decoder is the other ~52% (2607 ms). Instrumenting it: the autoregressive
     step loop is 91%, and its cost was **re-dequantizing the (constant) f16
     decoder weights on EVERY step** — ~11 `to_f32()` calls per layer × 4 layers ×
     30 steps. Hoisted all of them into a per-layer f32 cache built once before
     the loop (`decw`). Same-binary same-load A/B (`MIXTEX_DEC_DEQUANT_PER_STEP=1`
     restores per-step dequant, kept as a regression gate), best-of-3 min:
     **decoder 2923 → 1008 ms (~2.9×)**, byte-identical. First guess (threading
     the vocab projection) was measured as ~4% of the step and discarded — the
     redundant dequant was the real cost. (Cross-attn K/V precompute is a separate
     6%, already hoisted.) Lesson: on an AR decoder, check for constant work
     re-run per step before optimizing any kernel.
   - **layout_detect DONE (2026-07-11) — and the flagged target was wrong.** The
     roadmap flagged the *deformable cross-attention* loop, but instrumenting it
     showed it is only **~1.5% of Phase 2** (~30 ms of ~1920 ms) — a dud, the
     classic flagged-micro-gap-that-isn't-the-dominant-cost. The real Phase-2
     cost is the `cpu_linear` matmuls (self-attn / projections / FFN over 6
     layers, up to N=8400 memory tokens for cross_value). Already AXPY'd
     (`477a4b5`, ~1.26×); the remaining lever is **threading its independent
     output-row loop** (`layout_detect.cpp:1038`, honors `ctx->n_threads`, default
     1 = old behavior). Best-of-5 min at `-t 8`: Phase 2 **2345 → 1572 ms
     (1.49×; median 1.24×)**, byte-identical (21 regions). Sub-2× because the big
     cross_value matmuls are partly **memory-bandwidth-bound** (don't scale to 8×)
     and per-call `std::thread` spawn adds overhead — a shared thread pool would
     tighten it, but that's a cross-engine project. Same lesson again: measure to
     find the real hot loop, then parallelize the independent axis.
2. **ggml-metal ICB replay** — per-op Metal dispatch dominates decode (measured:
   trocr compute ~18ms/step is dispatch, not math). CUDA-graph capture already
   solves the CUDA side (CrispASR §210). Large upstream-ggml design + prototype.
3. Optional marginal cleanups (each <5%): scunet dequant cache, debug-`fprintf`
   gating in layout_detect/surya_det/ocr_detect, honor `--gpu-backend` in
   `crispembed.cpp:81`, LTO/IPO, broaden the Metal F16 mul_mm guard.

Session's two REAL wins (merged): safmn whole-image threading (2.3×), tps_locnet
dequant hoist (for reuse-callers). Both were where the gap WAS a meaningful
fraction.

### Tier 3 — see PERFORMANCE.md re-verification gap table

layout_detect deformable cross-attn (scalar); WebGPU embedding tier (no GPU
path in `build-embed-wasm.sh`); mixtex Swin window attn (scalar);
qwen2vl/granite/smoldocling per-step graph rebuild (folds into Tier-1 item 1);
build/infra (LTO/IPO, `GGML_BLAS=ON` Accelerate, honor `--gpu-backend` instead
of `init_best()` in `crispembed.cpp:81`, broaden the Metal F16 mul_mm guard from
5/~40 GPU files).

### Tier-1 item 1 — concrete design: decode-step graph cache

**MEASURED ROI caveat (2026-07-11).** This is model-dependent and generally
MODEST — set expectations before implementing. metal-prof of got-ocr2-q4_k decode:
each 940-node step is **~89% GPU-execute / ~11% host** (2.5 ms Metal-encode / ~20 ms
gpu, synced) → compute-bound, so caching the graph (which removes graph-build +
alloc-planning, i.e. part of the host slice) caps at **~10–17%** there. Lighter
decoders skew more host/dispatch-bound but the earlier trocr-small measurement was
only **2–6%** (build+alloc 0.47 ms/step vs 18.5 ms Metal compute). So the win is
real but small, and the value is highest for **many-token decodes on light models**
(unlimited/formula OCR) and lowest for heavy q4_k LLM decoders. Worth doing for the
correctness/tidiness and the cumulative effect across long outputs, but don't expect
a headline multiple. A handover prompt for this exists (see session notes).

**Verified current state (2026-07-11 code read).** No decoder caches the built
cgraph. Even the "best" ones rebuild every step:
- `deepseek_ocr2` decode loop (`deepseek_ocr2.cpp:1915-1955`) calls
  `build_llm_layer_attn` **per layer per step**, then `sched_reset` +
  `sched_alloc_graph` + `ggml_free(lag.gctx)` each iteration. KV is
  device-persistent (`alloc_ds_kv_cache`), the *graph* is not.
- `qwen2vl`/`granite`/`smoldocling` rebuild their decode graph each step too.

So the per-step cost is graph **build + allocation-planning**, paid every token
(deepseek: every token × every layer).

**The invariant that makes caching possible.** Across steps the graph STRUCTURE
is identical; only shapes tied to `n_past` change — the K/V history view length
`[0 .. n_past+T)` and the mask width. Fix those to a *bucket* and the graph is
reusable within the bucket.

**Design.**
1. Bucket the KV length: `B = ceil((n_past + T) / S) * S` (S = 64 or 128). Build
   K/V history views of length `B`; the attention mask zeroes the unused
   `[n_past+T, B)` tail. Rebuild + re-reserve only when `n_past` crosses into a
   new bucket (amortized: a rebuild every S tokens, not every token).
2. Persist `lag.gctx` + `lag.gf` and a `ggml_backend_sched` reservation across
   steps in a bucket. Each step, update only the input tensors (`layer_input`,
   `pos_ids`, `mask`) via `ggml_backend_tensor_set` — do NOT rebuild.
   Reservation template already in-tree: text encoder `sched_reserve`+T-bucket
   (`crispembed.cpp:1202-1217`), lfm2 (`lfm2_embed.cpp:452-457`).
3. **Landmines:** `ggml_backend_sched_reset` does not null `tensor->buffer`
   (LEARNINGS) — with a persistent reserved graph, do not `reset` between alloc
   and compute on the same graph; run side graphs (embedding lookups) before the
   main alloc. Cache the *decode-step* graph only, never the encoder graph
   (measured dud + the CrispASR #235 GPU use-after-free).
4. **Per-backend gating (mandatory):** env flag; OFF on WebGPU (graph reuse traps
   `unreachable`, `09dc519`); validate separately on Metal / CPU / CUDA.

**Reference decoder to prototype:** `deepseek_ocr2` — it already has device KV +
a clean single-layer graph builder (`build_llm_layer_attn`) to make persistent,
and `DS_PROFILE` instrumentation.

**Caveat — measure build-vs-compute first.** `DS_PROFILE` (commit `ded09a8`)
found deepseek decode already **compute-bound**, so caching the graph may buy
little THERE; the win is largest on lighter decoders (smaller matmuls → graph
build/alloc is a bigger fraction). Step 0 of implementation: per-token breakdown
(build vs alloc-plan vs compute) on each candidate decoder; only pursue the ones
where build+alloc is a meaningful fraction.

**Step 0 result — MEASURED 2026-07-11 (trocr-small q8_0, a deliberately LIGHT
decoder: D=256, V=1200, 6 layers).** Instrumented the math_ocr decode loop
(`build_decoder_step_graph` + `sched_reset` + `sched_alloc_graph` vs
`sched_graph_compute`):

| backend | build+alloc / step | compute / step | build fraction |
|---|---|---|---|
| Metal | 0.47 ms | 18.5 ms | **2%** |
| CPU   | 0.46 ms | 6.9 ms  | **6%** |

**Verdict: the decode-step graph cache is a 2–6% win here, not the "#1 lever."**
And it's an *upper bound* — build+alloc is ~constant per step (fixed node count;
only KV tensor *shapes* grow), while compute grows with `n_kv`, so `build_frac`
only shrinks for longer sequences. Even on the lightest decoder where the roadmap
predicted the biggest payoff, graph construction is a minority cost. The dominant
per-step cost is compute/dispatch (Metal ~18ms vs CPU ~7ms for the same tiny
graph → Metal per-op dispatch overhead, not math). **Reprioritize: the real
decode lever is cutting per-step dispatch — Tier-1 item 2 (ggml-metal ICB replay)
and op-count reduction — NOT the graph cache.** (Sample = 2 decode steps at small
`n_kv`; the constant-build-cost argument makes the conclusion robust to step
count. Re-measure on a heavier decoder if ever revisited, but expect an even
smaller fraction.)

**Verify (if ever pursued):** q8_0 model per decoder; output cosine vs baseline
≈1.0 (structure identical, only reuse changes); per-token latency before/after.

**IMPLEMENTED 2026-07-11 (got-ocr2-q4_k, Metal) — `GOT_OCR_DECODE_CACHE=1`,
default OFF, output byte-identical.** Rather than the PLAN's persist-`gf` +
re-point-the-KV-write-views design (fragile with the Metal sched — see the
landmine list), the shipped cache exploits a cheaper invariant: the decode graph
is **single-backend** (all got-ocr2 weights + KV live on `ctx.backend`) and its
**node count is constant per step** (only tensor *shapes* grow with `n_past`).
So `run_cached_step` keeps rebuilding `gf` each token (build ≈ 0.4 ms — cheap)
but swaps the per-step `ggml_backend_sched` reset+alloc+compute for a **dedicated
`ggml_gallocr` reserved once at `kvc.max_seq`** + a **sched-free
`ggml_backend_graph_compute(ctx.backend, gf)`**. Because the reservation covers
the longest graph, every subsequent `ggml_gallocr_alloc_graph` takes the
no-realloc fast path (`ggml_gallocr_needs_realloc` only checks node count +
per-node `size_max`), and the sched-free compute skips `split_graph` entirely.

Measured, back-to-back under identical load, median of 494 warm steps (cold-5
dropped), full 499-token `scan_page_pd.png` decode:

| path | build+alloc / step | alloc component | compute (identical ops) |
|---|---|---|---|
| OFF (sched) | 4.66 ms | 4.26 ms | 23.6 ms |
| ON (gallocr, sched-free) | **0.61 ms** | **0.22 ms** | 27.6 ms |

Host graph build+alloc drops **~87 %** (the 4 ms compute delta is pure GPU-
contention noise on a loaded box — output is byte-identical, so compute does the
same work). On a *quiet* box (Step-0 numbers: sched alloc ≈ 1.9 ms, compute
≈ 24 ms) the end-to-end per-step win is **~6 %**; it grows for long decodes and
busy machines because the sched `alloc` scales with sequence length + CPU load
(1.9 → 4.3 ms measured) while the gallocr path stays **flat at ~0.22 ms**.
Verified byte-identical on 3 images (incl. the 499-step page); CPU backend also
output-identical. This confirms the roadmap's framing — a modest, real,
worth-shipping-as-opt-in win, NOT a headline multiple; the dominant per-step cost
remains compute/dispatch (Tier-1 item 2).

**EXTENDED to 4 more decoders (2026-07-11), all output byte-identical.** Same
sched-free-gallocr mechanism, one env gate each, default OFF:
`internvl2_ocr` (`INTERNVL2_DECODE_CACHE=1`), `glm_ocr` (`GLM_OCR_DECODE_CACHE=1`),
`lightonocr` (`LOCR_DECODE_CACHE=1`, a CPU-only decoder — its inline decode graph
was hoisted into a `build_step_graph(n_past)` lambda so it can be reserved).
**Critical correctness gate learned here: cache DECODE steps only (`n_past > 0`).**
Unlike `got_ocr` (whose prefill is a separate code path), these three route
*prefill* through the same `run_cached_step`; sending the prefill graph (vision
splice / full-sequence mRoPE, a different node count) through a decode-shaped
gallocr reservation + sched-free compute corrupts output — glm degenerated into
repetition until the `n_past > 0` gate was added (internvl2 happened to survive
sched-free prefill, but is now correctly gated too). Verified byte-identical ON
vs OFF, cache engagement confirmed via a stderr marker: internvl2-1b-q4_k
(scan_page_pd 21 lines + fox), glm-ocr-q8_0 (scan_page_pd 44 lines),
lightonocr-1b-q4_k (scan_strip + fox).

**Quiet-box wall-clock A/B (2026-07-11, best-of-3 back-to-back, loadavg ~7).**
The saving reconciles across metrics and scales inversely with per-step compute:
got_ocr-q4_k STEP_PROFILE build+alloc **0.85 → 0.28 ms/step** (alloc 0.73 → 0.08,
the gallocr flat vs the sched growing), and decode_total **6108 → 5933 ms best-of-3
(~3%, ~0.35 ms/step, 499 steps @ ~12 ms/step)** — the two agree within noise. On
the heavier internvl2-1b (~36 ms/step) the same ~0.5 ms host saving is **below the
noise floor (best-of-3 15473 vs 15706 ms, ~0%)**. So the decode-cache is a **~3%
win on LIGHT decoders, ~0% on heavy ones** — and its real value is
**load-insensitivity**: the sched's `alloc` ballooned to ~4.3 ms/step under loadavg
~24 (→ a ~4 ms/step, ~15% saving there) while the gallocr path stays flat at
~0.1–0.2 ms regardless of load. glm-q8_0 wall-clock not measured (heavier than
internvl2 → predictably within noise; correctness verified).

**`qwen2vl` does NOT fit — attempted and reverted (2026-07-11).** Its decode graph
is single-graph and constant-shape (KV views span `[0..max_seq-1]` + `kv_mask` —
it was already designed so `sched_alloc_graph` skips gallocr realloc, see the
comment at `build_decode_step_graph`), BUT it is **multi-backend**:
`GGML_SCHED_DEBUG=2` shows per-layer `SPLIT: CPU` for the attention (the
`qwen_kv_k/v (view)(reshaped)(permuted)` + `(cont)` tensors run on CPU, rest on
MTL0). So computing it sched-free on `ctx.backend` (Metal) forces those CPU ops
onto Metal → empty/garbage output (verified: fox OCR went 2 lines → 0). And
because the shape is already constant, the reserve buys nothing (no realloc to
skip), while `split_graph` is *mandatory* for a multi-backend graph — so there is
no win to capture. **Lesson: "single-graph decoder" is necessary but not
sufficient; the decode graph must also be single-backend** (got_ocr/internvl2/glm
are all-Metal, lightonocr all-CPU; qwen2vl is Metal+CPU). Fixing qwen2vl would
mean getting its attention onto Metal (a separate op-coverage task), not the
graph cache.

**`math_ocr` (TrOCR enc-dec) DONE (`MATH_OCR_DECODE_CACHE=1`).** The 4th backend
in the list above is really the 5th engine — added after qwen2vl: despite being
an encoder-**decoder** with a second (cross-attn) KV set, its decode is
single-graph, single-backend (self+cross KV + weights on `ctx->backend`; the
`GGML_SCHED_DEBUG=2` decode split is all one backend), and decode-only. Reserved
at `max_steps`; the cross-attn KV read length is fixed (`n_enc`) while only the
self-attn read grows with `n_kv`, so the constant-node-count invariant holds.
Byte-identical + engaged on fox + scan_strip (trocr-small-printed-q8_0). It's the
lightest decoder ported (D=256) → best relative win of the set. Still open (each
needs the single-backend decode check first): `smoldocling` (CPU LM head outside
the graph), `granite` (shares the vision sched), and `deepseek_ocr2`
(per-layer-per-step → needs the persistent-graph variant).
