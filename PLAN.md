# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## 🚧 Active work in flight (update + push to `main` at EVERY checkpoint)

Multiple sessions/worktrees run in parallel and push to `main` concurrently.
Before starting a task, add a row; at every checkpoint update it and push this
file to `main` so others see what's claimed (avoids duplicate work + CI-cancel
races). Remove the row when the branch lands.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-07-13 | `feat/transcoda-omr` | Transcoda-59M zero-shot OMR (full-page score → Humdrum `**kern`; ConvNeXt-V2-tiny enc + 8-layer RoPE cross-attn decoder). **Clean-room** (weights CC-BY-4.0, code AGPL — engine written from paper + config + oracle only) | **IN PROGRESS** — worktree + branch up; arch facts locked from `config.json`+safetensors (d512/8L/8H/ffn1024/vocab3000/RoPE θ1e4; enc=HF ConvNextV2Model+GRN; projector 768→2048→512; untied LM head). Next: converter + oracle dumper. |
| 2026-07-13 | `feat/tromr-engine` | Polyphonic-TrOMR OMR (engine + wiring + quants + fixture) | **DONE** — engine `src/tromr_ocr.cpp` on `main` (cos 1.0 / 100% argmax / byte-exact); HF `cstr/tromr-GGUF` (f32 + q8_0 31 MB w/ F16 backbone + Apache-2.0 card); registry + regression fixture (cer 0.000). |
| 2026-07-13 | `feat/flova-omr` | Flova/omr_transformer — handwritten/whiteboard OMR (donut-swin + mBART VED → LilyPond, Apache-2.0) | **DONE (on `main`)** — engine `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact incl. native preproc), `tests/test_flova_diff.cpp`, CMake, CLI dispatcher + registry. HF `cstr/flova-omr-GGUF` (f32 573 MB + q8_0 162 MB byte-exact + Apache-2.0 card). Regression fixture landed (`feat/flova-regression-fixture`): `staff_flova.png` (model card sample1.png) + golden LilyPond `c'2 a''8 c''8 r4 c'1 e'8 c'8 c'8 a''8 f'4 a'8 c'8`, run_one cer 0.000. **Fully done.** |
| 2026-07-13 | `feat/flova-regression-fixture` | Flova OMR regression fixture (manifest entry + `staff_flova.png`) | **Landed `main` (`67ddc99`).** `run_one.py --name flova` PASS (garbage-guard + text cer 0.000 vs q8_0 from `cstr/flova-omr-GGUF`, CPU==Metal). |
| 2026-07-13 | `feat/smt-regression-fixture` | SMT OMR regression fixture (manifest entry + `staff_smt.png`) — completes the OMR guardrail trio (SMT/TrOMR/Flova) | **DONE — validated, pending push to `main`.** `run_one.py --name smt` PASS (garbage-guard + text cer 0.000 vs `smt-grandstaff-q8_0.gguf` from `cstr/smt-grandstaff-GGUF`, CPU==Metal identical, deterministic bekern decode). |
| 2026-07-13 | `feat/smt-fp-fullpage` | SMT++ **full-page** pianoform OMR (`PRAIG/smt-fp-grandstaff`) | **IN PROGRESS — engine+converter done, validating parity.** Key correction to handover: fp checkpoint is `antoniorv6/SMT` (main rewrite), NOT SMT-plusplus — so beyond config it needs: (A) **scaled** attention `d_head^-0.5` [engine read `smt.scale_attention` but hardcoded 1.0 — now applied]; (B) **no** ReLU before head; (C) decoder tensor **rename** (self_attn/cross_attn/ffn/norm_layers/vocab_projection→engine names); (D) preproc `reduce_ratio=1.0`+invert; (E) head Linear not Conv1d. Converter now auto-detects scheme + maps names + writes flags (`smt.head_pre_relu`, `smt.preproc.reduce_ratio`/`.invert`); base path untouched. Engine: `mha_core` scale arg, head-relu gate, fp preproc, exact enc-grid dims (H/16 rounding fix). `smt-fp-f32.gguf` loads clean (361 tensors), runs on Metal. Next: per-stage diff (dumper adapted to main-SMT API) + decoded roundtrip, then q8_0 + upload + registry. |
| 2026-07-13 | opus-1m (perf sweep) | DBNet detection postprocess — scanline box scoring | **Landed `main`** (`74b8ac5`, 28× faster, byte-identical) |
| 2026-07-13 | opus-1m (perf sweep) | Decoder op-fusion investigation | **Done** — measured marginal on compute-bound + Metal-auto-fused decoders (`58a3751`); QKV concat-matmul deferred |
| 2026-07-13 | opus-1m (perf sweep) | Kaggle CUDA confirmation (Class-A + Gap-5) | ✅ **DONE** — clean re-run (v9, `/tmp` ENOSPC fix `8f175cb`). Class-A/Gap-5 **confirmed PASS on CUDA**: deepseek-ocr2, dat, swinir, qwen2vl-3b, lfm2_colbert. The 14 FAILs are NOT regressions in the fixed engines: glm-ocr/internvl2 = known Class-B (Turing/Pascal); pcs/fireredpunc/fullstop = `test-punct-diff` not built in this config; layout-heron = SIGABRT teardown; granite-vision = text PASSES, only 3 diff stages cos 0.95–0.97; hat = harness no-parse. **Follow-ups landed:** `be6ec54` (teardown-tolerance + run_check-skip) took v10 **14→9**; then `2af57b1` fixed the diff-output parser (ANSI codes, colon-less `cos_min=`, table formats) — `lfm2`/`lilt`/`layout`/`hat`/`pan`/`tbsrn` were **false "no-parse" FAILs** (verified locally: lfm2's 20 stages all pass). **v11 final: 46 models, 4 FAIL** (was 14) — the parser fix cleared hat/pan/tbsrn/lilt/lfm2. All harness follow-ups DONE; the 4 remaining all need a CUDA box to diagnose: `glm-ocr`+`internvl2` (Class-B vision garbage, Turing/Pascal), `granite-vision` (projector cos drift, text passes), `layout-heron` (`test-layout-diff` SIGABRT *before* output — a real CUDA abort in the deformable-attn diff, needs the assert message from a CUDA run). |
| 2026-07-13 | opus-1m (interop/SR) | Kaggle reranker τ-eval — full 7-reranker roster on the n=30 corpus (`crispembed-imatrix-quant`) | **DONE** (both batches, all imatrix quants re-uploaded to `cstr/*-GGUF`). **Key finding:** imatrix ALWAYS cuts q4_k score-drift (dscore, 7/7) but its effect on ranking **τ is model-dependent** — big win on ms-marco-L-12 (0.853→0.929) + jina (0.929→0.942), neutral on bge, but **degrades** both mxbai rerankers −0.076 (iq4_xs beats q4_k+imatrix there). So `q4_k+imatrix` is **not** a universal reranker recommendation; validate per-model. The old n=5 corpus missed both the mxbai regression and the ms-marco-L-12 win. jina q4_k-imatrix also validated locally on Metal (EN+DE rerank correct). |

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

## Pending work

Only genuinely-open, in-progress, or reference material lives below. **Completed
milestones — the imatrix quant rollout (C1), batched-encoder throughput (C3),
prefix KV cache (C4), mtmd-preprocessing port (C5), flash-attn epilogue audit
(C6), mmproj interop, the June-2026 optimization-TODO sweep, per-backend perf
passes, the SR conv→ggml sweep, the regression-guardrail closure, the CUDA
device-pointer fixes, and the scan_cleanup / unpaper feature ports — have moved
to `HISTORY.md`** (deep technical notes in `LEARNINGS.md`). Before starting any
item: read LEARNINGS "measure the DOMINANT cost before fixing a flagged
micro-gap" and "the build dir was silently CPU-only"; verify
`GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`; check `git worktree list` +
`git log main..<branch>` for a concurrent session's finished work; all edits in
a worktree (ggml symlink dance, see CLAUDE.md).

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

**Licensing methodology — AGPL *code* is NOT a blocker (verified 2026-07-13).**
The gate is the **weights** license (we redistribute GGUFs) and the **engine
authorship**, which are independent of an upstream repo's *code* license:
- If the **weights** are permissive (MIT / Apache / **CC BY**), the GGUF is
  redistributable regardless of the training-code license. AGPL/GPL on the
  *code* does not attach to CC-BY *weights*. (Training-data license only matters
  if we redistribute the data or retrain — not for shipping pretrained weights.)
- The **engine** is written **clean-room**: run the upstream Python as an
  *oracle* (reference-activation dumps — no derivative) and implement from a
  **facts-only spec** (architecture, tensor shapes, op order, hparams, eps/scale
  — all uncopyrightable) + the paper + configs. Never transcribe AGPL source
  line-by-line. Two-team wall: the brief-writer may read the AGPL `.py`; the
  implementer sees only the facts brief. (Permissive blueprints don't need this.)
- Hard rejects shrink to: **gated / unlicensed / non-permissive weights**, or an
  **11B+ base under a restrictive model license**, or a **non-single-model
  pipeline** (poor ggml fit).

| # | Model | Params | License (code / weights) | Architecture | Output | Handles | Effort | Status |
|---|-------|--------|--------------------------|-------------|--------|---------|--------|--------|
| 1 | **Sheet Music Transformer (SMT)** | 21.4M | **MIT / MIT** | ConvNext encoder + Transformer decoder | bekern | Printed polyphonic | Low | **DONE** — `src/smt_ocr.cpp` shipped (per-stage cos 1.0, 96.3% GrandStaff) |
| 1b | **SMT++ full-page** | ~10.9M | **MIT / MIT** — `PRAIG/smt-fp-grandstaff` (public, **not gated**, verified HF card) | full-page extension of SMT (curriculum-trained) | bekern | **Full-page pianoform** (no separate layout stage) | **Low–Med** | **DOABLE — top permissive target.** Verify arch delta vs base SMT first: deep-research *refuted* (2-1) the "same-arch, curriculum-only" claim, so confirm the graph before assuming free reuse. If same graph → near-free extension of shipped SMT |
| 2 | **Transcoda-59M-zeroshot** | 58.8M | **AGPL code / CC BY 4.0 weights** (`btrkeks/transcoda-59M-zeroshot-v1`, verified HF card) | ConvNeXt-V2-Tiny enc + 8L Transformer dec (d512/8h, **RoPE**) | **kern | **Full-page + historical scans** (zero-shot); **current OMR-NED SOTA** (Polish 63.97%, Verovio 18.46% — beats SMT++ & Legato) | **Med** | **DOABLE — accuracy leader.** Weights CC BY 4.0 → GGUF redistribution clean (attribute). Engine **clean-room** (code is AGPL). Arch fully in-tree: ConvNeXt-V2 ≈ SMT's ConvNext, RoPE decoder ≈ Qwen3; add 3000-token kern BPE tok; optional GBNF grammar-constrained decode. Training data `polish-scores` = `license: other` (irrelevant to CC-BY weight redistribution) |
| 3 | Polyphonic-TrOMR (NetEase) | ~22M | **Apache-2.0 / Apache-2.0** | ViT + multi-head Transformer decoder (rhythm/pitch/lift/note) | symbolic text | Printed polyphonic photos | Medium | **DONE** — `src/tromr_ocr.cpp` (cos 1.0 / 100% argmax / byte-exact); `cstr/tromr-GGUF` |
| 4 | **Flova/omr_transformer** | 143M | Apache-2.0 / Apache-2.0 | Donut VED (DonutSwin + 4L mBART) | LilyPond | artificial + **handwritten** + whiteboard (monophonic) | Medium | **DONE** — `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact); `cstr/flova-omr-GGUF` (f32 + q8_0); CLI + registry wired |
| 5 | oemer | 2× U-Net | MIT / MIT | 2 segmentation U-Nets + numpy reconstruction | MusicXML | Printed, photos, skewed | High | Reference-only — multi-model + rule-based reconstruction, poor ggml fit |
| ~~6~~ | ~~Legato~~ | ~11B | MIT (trained delta) / **Llama-3.2 license + GATED** | frozen Llama-3.2-11B-Vision + trained decoder | ABC | full-page | — | **REJECTED** — 11B base under Meta's Llama license + contact-gated weights; MIT covers only the delta. Too big + non-permissive base |
| ~~7~~ | ~~starry / FindLab~~ | — | **no code license / gated, unlicensed weights** | 7-microservice pipeline (PyTorch+TF+ONNX) | LilyPond/kern | complex polyphonic | — | **REJECTED** — not a single model (poor ggml fit) *and* weights token-gated with no stated license |
| ~~8~~ | ~~Clarity-OMR~~ | — | (unverified) | PDF→MusicXML **pipeline** | MusicXML | printed | High | Reference-only — multi-stage pipeline, not a single VED model |
| ~~9~~ | ~~homr (liebharc)~~ | — | **AGPL-3.0** (code) | pipeline + TrOMR | MusicXML | printed/camera | — | **REJECTED** — pipeline (poor ggml fit); the underlying TrOMR is already shipped separately (Apache-2.0) |

**Recommended priority (updated 2026-07-13 — SMT/TrOMR/Flova all shipped):**

1. **SMT++ full-page** — best permissive next step. MIT + public weights
   (`PRAIG/smt-fp-grandstaff`), and it extends the *already-shipped* `smt_ocr.cpp`.
   First task is cheap and decisive: **verify the arch delta vs base SMT** (the
   deep-research verifier *refuted* the "curriculum-only, identical arch" claim
   2-1, so don't assume free reuse). If the graph matches → near-free full-page
   pianoform (no separate layout stage). If it differs → scope the delta.

2. **Transcoda-59M** — accuracy leader + only permissive route to historical
   scans. Weights are **CC BY 4.0** (redistribute the GGUF freely, attribute);
   the code is AGPL so the engine is written **clean-room** (Python-as-oracle +
   facts-spec — see "Licensing methodology" above). Arch is fully in-tree:
   ConvNeXt-V2-Tiny ≈ SMT's ConvNext backbone, 8L RoPE decoder ≈ Qwen3 decoder;
   the only new pieces are a 3000-token **kern BPE tokenizer and (optional) GBNF
   grammar-constrained decode. Highest accuracy on the OMR-NED benchmark.

3. **Handwritten *polyphonic* — the real remaining gap.** No permissive model
   fills it: Flova (shipped) is monophonic-toy; the strong performers are all
   rejected (Legato = Llama-11B/gated, starry = gated/unlicensed pipeline, homr =
   AGPL pipeline). Reach it by *fine-tuning* a shipped graph (SMT or Transcoda)
   on synthetic + license-clean handwritten-style data — same engine, new weights.

3. **Polyphonic-TrOMR — DONE (2026-07-13).** Genuinely accurate model (reads
   clefs/keys/rhythms/pitches correctly on real photos). The ggml engine
   `src/tromr_ocr.cpp` (ResNetV2 SAME-pad backbone + hybrid ViT encoder →
   x-transformers 12-sublayer decoder with SIGLU attn-on-attn + GEGLU FF → 4
   parallel heads, autoregressive over rhythm/pitch/lift streams) is written,
   wired (dispatcher + CMake + `test-tromr-diff` + CLI `--ocr` auto-detect), and
   **validated CPU-only vs the reference model**: every diff-harness stage cos
   **1.0** (backbone, ViT context, all 12 decoder blocks, all 4 logit heads),
   **100% per-position argmax agreement** teacher-forced (66/66, 85/85), greedy
   decode **byte-exact** vs the authors' `examples/{1,2,3}.txt`, Metal == CPU.
   q8_0 also decodes byte-exact. **Remaining:** HF upload `cstr/tromr-GGUF`
   (f32 + q8_0) + `model_mgr.cpp` registry entry.
   Corrections vs the (now-removed) handover brief found in validation: ViT scale
   is **32^-0.5** not 64^-0.5; the converter emitted tensor names >64 chars that
   the ggml loader rejects (`GGML_MAX_NAME`) → shortened the backbone prefix to
   `enc.bb`; the quantizer must keep `enc.bb`/`enc.proj` convs unquantized
   (flatten+quantize → reshape-to-4D abort). See LEARNINGS.md.
   Weights: `tromr/workspace/checkpoints/img2score_epoch47.pth` (86.3 MB)
   committed directly into the Apache-2.0 repo (not LFS → covered by the repo
   license), with a 4-file tokenizer set (`tokenizer_{lift,pitch,rhythm,note}.json`).
   Architectural wrinkle vs SMT: TrOMR is **not** a single autoregressive stream
   — it has *parallel classification heads* (rhythm / pitch / lift / note) per
   decoder timestep, so the port needs 4 output projections + a merge step, not
   one LM head. `homr` wraps this same model but is AGPL — weights taken from
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
- ✅ **Preprocessing fixed → SMT WORKS at 96.3%.** The engine had been *inverting*
  the image (SMT-plusplus's `convert_img_to_tensor` has `RandomInvert(p=1.0)`), but
  `smt-grandstaff` is an **SMT-main** model whose preprocessing is `Grayscale→
  ToTensor` with **NO invert**. Inverting → ~30%; correct (non-inverted) → **96.3%**
  on the clean `antoniorv6/grandstaff` test split (per-image 91.8/96.2/96.7/99.6%).
  Full pipeline: RGB (no cv2-BGR swap), `reduce_ratio=1.0`, `width=min(w,3056)`,
  `height=max(h,256)`, grayscale, no invert. Fixed in `recognize_raw` + the dumper.
- ✅ **Fully validated:** per-stage diff cos=1.0; C++ decode == Python blueprint
  (100% token agreement, 10 fresh images); **C++ engine vs ground truth = 96.3%.**
  SMT-plusplus's unscaled forward confirmed correct (SMT-main's forward → 0% garbage
  on this checkpoint). The port was exact all along — the invert was the only bug.
  Lesson: [[validate-intermediates-and-outputs]] — a "reads-structure-not-detail"
  pattern across models was a preprocessing/input bug, not model quality; derive
  preprocessing from the model's OWN repo (SMT-main, not the SMT-plusplus fork).

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

### llama.cpp parity — support matrix (reference)

A living audit of which CrispEmbed architectures llama.cpp supports (upstream
`ggml-org/llama.cpp` @ ~`4fc4ec5`, July 2026), how it implements them, and where
we remain unique. Deep technical notes live in `LEARNINGS.md → "llama.cpp
implementation reference"`. **The convergence backlog derived from this audit
(C1 imatrix quant, C3 batched throughput, C4 prefix KV, C5 mtmd preprocessing,
C6 flash-attn epilogue, mmproj interop) all shipped — see HISTORY.md.** This
section is kept only as the capability reference. Any future borrow must still
land behind an A/B on BOTH speed and quality, on CPU and Metal.

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
| GLM-4V / GLM-OCR | ✅ | `glm4v` | AIMv2 tower, **dynamic** resize — ours matches now (Glm46VImageProcessor Qwen2VL smart-resize, shipped `dfd5653`; verified OCR 2026-07-13) |
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

### Open performance levers

Each needs a target GGUF (q8_0 preferred, to isolate from q4_k noise) and a
before/after parity + latency measurement — never land a "perf" change on a
compile-only check. A/B every change against ground truth and gate behind an env
var (see `../crispasr-crispembed-dev.md` "A/B every perf optimization").

- **SR/restoration — fused ggml graphs: COMPLETE (2026-07-13).** Every engine
  now runs a fused ggml graph, not per-conv mini-graphs. Ported this session:
  - **SAFMN** (`8594cee`): whole forward = ONE fused graph (erf-GELU) — **2.2×
    faster AND more accurate (cos 1.000000 vs 0.994)**. Tiny/overhead-bound, so
    fusion is a big win; Metal is a net loss here (default CPU, `SAFMN_SR_METAL`).
  - **NAFNet** (`14a8393`) + **InstructIR** (`e1eb1dc`): fused per-block graph,
    cos ≥ 0.999998, output identical to legacy. NAFNet-family = **compute-bound**,
    so fusion is perf-NEUTRAL (cleaner, not faster). NAFNet defaults to Metal
    (modest ~15%; `NAFNET_CPU`); InstructIR is CPU-only (GPU conv_2d hits a Metal
    f32×f16 mul_mv pipeline issue). Gates: `NAFNET_LEGACY` / `INSTRUCTIR_LEGACY`.
  - **Restormer**: was ALREADY fused — `rst_transformer_block_ggml` (MDTA + GDFN
    in one graph) is the default; `RESTORMER_SCALAR` is the fallback (cos 0.999997
    both). Only the stale "CPU-scalar" header was corrected.
  - **scunet, swinir, tbsrn, hat, adair, dat**: already build a single graph
    (`forward_expand=1`, no per-conv helpers) — verified sensible (swinir 0.9984,
    dat 0.99999, hat 0.89 q8_0). No work needed; the "CPU-scalar" labels were loose.
  **Key finding:** the fusion win depends on overhead-bound (tiny SAFMN → 2.2×)
  vs compute-bound (NAFNet/InstructIR → perf-neutral). Metal helps only where
  per-dispatch overhead is small relative to compute. Env gates per engine.
- **SR-on-GPU — conv weight residency (research, deferred).** The entire SR
  family computes convs on a CPU-only `enc_sched` with CPU-resident F32 kernels;
  there is no GPU sibling to match. Real SR-on-GPU needs Metal `ggml_conv_2d` for
  these shapes + a GPU-resident weight/graph path the family currently avoids —
  research, not a residency toggle. Reprioritized down.
- **Decode-step graph cache — remaining decoders.** Shipped (sched-free gallocr,
  reserved once at max KV, byte-identical, per-engine env gate) for got_ocr,
  internvl2, glm_ocr, lightonocr, math_ocr. **Still open, each needs the
  single-backend decode check first:** `smoldocling` (CPU LM head outside the
  graph), `granite` (shares the vision sched), `deepseek_ocr2` (per-layer-per-step
  → needs the persistent-graph variant). Modest win (~3% light decoders, ~0% heavy;
  real value is load-insensitivity). `qwen2vl` does NOT fit (multi-backend decode).
- **ggml-metal ICB replay / op-count reduction (the real Metal decode lever).**
  Warm Metal decode is ~82% GPU-execute (per-kernel launch across ~355 sequential
  ops), so ICB (which only collapses the ~18% host-encode) caps at ~18% and is
  NOT justified for CrispEmbed's light decoders. The tractable in-tree lever is
  **fewer, bigger ops per step** — fuse per-layer norm/scale/bias chains, QKV,
  the GLU elementwise chain, prefer `ggml_soft_max_ext`. Per-decoder graph surgery
  in each `build_decoder_step_graph`; verify output cos ≈ 1.0 + node-count +
  latency per model. Re-measure heavy decoders with `CRISPASR_METAL_PROFILE=1`
  before any ICB work. **Caveat (measured 2026-07-13):** the math_ocr ~30%
  cont-removal does NOT generalize to decoder-only VLM engines — got_ocr's cached
  decode already feeds K/V as cache views, so only Q's cont was removable
  (byte-identical, but latency within noise; `5011848`, `GOT_OCR_ATTN_CONT=1`).
  **Op-fusion measured marginal too (2026-07-13):** (a) Metal already auto-fuses
  (`use fusion=true`; `kernel_norm_mul_add`, `kernel_bin_fuse` kernels handle the
  norm/scale/bias + GLU elementwise chains at dispatch), so graph-level elementwise
  fusion is redundant there; (b) attention is already flash-fused; (c) these
  decode steps are compute-bound (got_ocr ~89% GPU-execute), capping any dispatch
  reduction at the ~11% host slice; (d) the trocr decoder is already lean (319
  nodes, 55 ms/16 tok — the ViT *encoder* at 212 ms is trocr's real cost, not the
  decoder). The only non-auto-fusable win is **QKV concat-matmul** (3→1), but a
  probe (`GOT_OCR_QKV_FUSE`, 2026-07-13) confirmed it's not worth it: `ggml_concat`
  **mishandles q4_k** (garbage output) and re-concatenating per step is 3× slower,
  so a correct fusion needs manual load-time q4_k row-block byte-stacking — and on
  a memory-bound T=1 decode that only saves ~2 matmul launches/layer (~4%).
  Deferred; see HISTORY.
  (DeepSeek-OCR-2's MoE-compute lever is detailed in its own subsection above.)
- **unlimited_ocr — remaining deferred items.** `UOCR_PD=1` persistent T=1 decode
  graph (blocked on a small flash-attn padded-vs-exact-KV numerical drift that
  changes argmax by ~step 3; ~14% decode win if solved); `UOCR_OPT_GGML_WINDOW=1`
  (SAM window partition in-graph, ~2–5%, deferred); SAM flash-attn (won't — the
  decomposed RPE bias defeats the O(T) benefit).
- **text_sr — blocked on a public checkpoint** (NAFNet text-SR; registry URL
  empty, no shipped GGUF). Conv paths are guarded transitively by the `nafnet`
  entry; PixelShuffle/bicubic tail unguarded. To train one on clean (Apache/MIT)
  data see `docs/text_sr_training_data.md`.
- **esrgan tile-loop parallelism (concurrency project, deferred).** Intra-op
  threading measured SLOWER (tiled convs don't thread-scale). The real lever is
  running whole 128px tiles concurrently → needs per-thread backend+sched
  replication (the tile loop shares one `ctx->enc_sched`). Verify on a quiet box.
- **TrOCR recognizer accuracy/speed.** eos/length-penalty parity is still TODO
  (the trigram-repeat bug is fixed). The bigger levers: swap DBNet-ic15
  (scene-text) for a document-text detector on dense pages; steer document OCR to
  the doc-VLMs (PaddleOCR-VL / SmolDocling); GPU (WebGPU/Metal) recognizer decode
  is the real speed path (the per-region AR token loop is CPU-bound).

### Open correctness / infrastructure

- **CUDA regression — the 4 remaining FAILs: diagnosis & fix plan.** After the
  2026-07-13 harness fixes, the Kaggle CUDA portfolio run is **46 models, 4 FAIL**
  (`glm-ocr`, `internvl2-1b`, `granite-vision`, `layout-heron`). All four PASS on
  local Ampere (sm_86) + Metal + CPU and fail only on Kaggle **T4 (Turing sm_75) /
  P100 (Pascal sm_60)** — so every one is an **older-arch ggml-CUDA** issue that
  reproduces ONLY on that HW. The full self-contained plan is below; see
  `HISTORY.md → "Kaggle CUDA regression"` for the run history.

  **Shared reproduction + tooling (read once).**
  - Kernel: `tools/kaggle/ocr-portfolio-regression` (clones `main`, CUDA build,
    runs `tests/regression/run_one.py` per model). Account = **chr1s4** (auth in
    local `kaggle_usage.md`; the token is NOT in `../.env`, which is chr1str).
    Push: `kaggle kernels push -p tools/kaggle/ocr-portfolio-regression`; poll
    `kaggle kernels status chr1s4/crispembed-ocr-portfolio-regression`; fetch
    `kaggle kernels output … -p OUT` (the JSON `.log` reconstructs via
    `"".join(e["data"] for e in json.load(...))`). Model + ref GGUFs stage under
    `/tmp` (the `/kaggle/working` ENOSPC fix, `8f175cb`) — keep it.
  - Iterate a SINGLE model to save the 30 h/week GPU quota: set
    `CRISPEMBED_BRANCH` in `ocr_portfolio_regression.py` to a debug branch and
    trim the manifest loop, or run `run_one.py --name <model>` from a minimal
    kernel. Each full run is ~30–45 min (cold build ~21 min unless the
    `chr1s4/crispembed-ccache` dataset warms it to ~3 min — refresh that dataset
    after a good build, see `kaggle_usage.md`).
  - **Confirm it's the CUDA backend, not the model:** re-run the failing engine
    with `<ENGINE>_FORCE_CPU=1` (or `CRISPEMBED_FORCE_CPU=1`) on the SAME Kaggle
    box — if CPU is correct there, the CUDA path is the cause (expected for all 4).
  - **The two references that PASS on CUDA are the control group:** `got-ocr2`
    (SAM ViT-B + Qwen2, `flash_attn_ext`) and `qwen3vl-2b` — diff the graph
    construction of a failing engine against these to spot the divergent op.

  **(1) `glm-ocr` + `internvl2-1b` — Class-B vision garbage (cer > 4).** The
  vision encoder emits garbage tokens on old-arch CUDA → the LLM hallucinates.
  - *Discriminator FIRST (cheap, decisive — from [[verify-handover-claims-independently]]):*
    inject known-good vision embeds (dump them on CPU) into the CUDA decode; if
    the OCR text is then correct, the bug is the vision tower; if still garbage,
    it's LLM conditioning. Also try zeros/random embeds — if output is identical
    regardless, the image is being silently dropped (a splice/token-id bug, not
    numerics — the same class as the Qwen2-VL `image_token_id` default bug).
  - *Localize the vision op:* `glm-ocr`'s ref + `test-glm-ocr-diff` BOTH exist
    (`cstr/glm-ocr-crispembed-GGUF/glm-ocr-ref-full.gguf`) → add a manifest `diff`
    block for glm-ocr and run on Kaggle to first-diff the vision stages (verify the
    ref is fresh first — an earlier note flagged a stale no-rope glm ref; if stale,
    re-dump with the current rope-correct engine). `internvl2` needs a ref dumped
    from **InternVL2-1B** (InternViT-300M + Qwen2-0.5B — NOT the local
    InternVL2.5-1B, a different arch), `tools/dump_internvl2_reference.py` (lazy
    safetensors, ~fits 16 GB), upload to `cstr/internvl2-1b-crispembed-GGUF`.
  - *Likely cause + candidate fixes (verify each on Kaggle):* the first vision
    stage whose cos craters names the op — prime suspects on sm_75/sm_60 are (a)
    an F16-accumulating `mul_mm`/`flash_attn_ext` (fix: F32 activation scaling like
    the Metal `metal-mul-mm-f16-overflow` ÷256/×256, or `ggml_mul_mat` set to F32
    precision on that op), (b) a conv/im2col kernel, or (c) a `get_rows` whose
    index is a non-contiguous view (CUDA asserts `nb[0]==type_size` — see the
    `flashattn`/`get_rows` contiguity note in `../crispasr-crispembed-dev.md`;
    fix: `ggml_cont` the index). Gate any fix behind an env var and A/B on Kaggle
    (cos vs the ref + decoded OCR).

  **(2) `granite-vision` — projector cos drift (text still PASSES).** 3 diff
  stages read cos **0.95–0.97** on CUDA (the **projector** MLP, 4608→2048→2048),
  but the LLM is robust so the OCR text passes (cer 0.163 < 0.180).
  - *Strong lead already in the code:* `gv_run_projector_graph`
    (`granite_vision_ocr.cpp:648`) notes that using the **tanh** GELU instead of
    the exact **erf** GELU drops projector cos to **0.954** — exactly the observed
    CUDA range. So the likely cause is the CUDA backend using a lower-precision
    GELU (or an F16 `mul_mm` cast) on the projector. The Metal ÷256/×256 fix
    (`:847`) guards only the **LLM** SwiGLU, NOT the projector.
  - *Fixes to try (verify on Kaggle, don't touch the passing CPU/Metal/Ampere
    paths blind):* force **F32 `ggml_gelu_erf`** on the projector regardless of
    backend; and/or force F32 precision on the two projector `mul_mat`s; and/or
    extend the ÷256/×256 activation scaling to the projector. Run
    `test-granite-vision-diff` on Kaggle before/after (target the projector stage
    back to cos ≥ 0.99). Isolate with `CRISPEMBED_GRANITE_VIS_SCALAR` /
    `_LLM_SCALAR` — if the scalar (CPU-math) projector passes on the Kaggle box,
    it confirms the divergence is in the ggml-CUDA projector graph.

  **(3) `layout-heron` — `test-layout-diff` SIGABRT (signal 6) BEFORE any stage
  output.** A genuine ggml **abort during** the diff on CUDA (not a teardown — the
  harness tolerance correctly does not mask it), so the layout graph itself hits a
  `GGML_ASSERT` on old-arch CUDA.
  - *Get the assert message (step 0):* the log truncates it. Run `test-layout-diff`
    standalone on the Kaggle box (or `GGML_ABORT`-verbose) and capture full stderr
    — the assert prints `op` + `file:line`, which names the failing kernel
    immediately. Backtrace is inside `test-layout-diff` (RT-DETRv2: ResNet-50 +
    HybridEncoder + 6-layer deformable-cross-attn decoder).
  - *Prime suspect:* the CUDA `get_rows`/view contiguity assert
    (`GGML_ASSERT(src1->nb[0] == ggml_type_size(...))`) — CUDA requires a
    CONTIGUOUS index/operand where CPU/Metal tolerate a strided view. The
    deformable cross-attention builds strided sampling indices; `flash_attn_ext`
    is used at `layout_detect.cpp:601/691/1468`. Audit any `get_rows`/view feeding
    a kernel in the decoder path and `ggml_cont` it (output-neutral). This is the
    exact class that bit dia-TTS on P100 (worked on M1) — see the CUDA-contiguity
    note in `../crispasr-crispembed-dev.md`.

  **Priority + notes.** layout-heron (3) is likely the cheapest — one assert
  message + a `ggml_cont` — do it first. granite (2) has the clearest lead
  (GELU/precision) and its text already passes, so it's low-urgency but tractable.
  Class-B (1) is the hardest (garbage output, needs ref-gen + op localization +
  a numerical fix) — use the inject-embeds discriminator to avoid chasing the
  wrong half. None of these is verifiable on this Mac (Ampere passes); every fix
  must be A/B'd on the Kaggle T4/P100 kernel behind an env gate before flipping.
- **DBNet detector — mostly resolved (2026-07-13).** The CPY abort was already
  fixed (`dequant_rows_f32` via get_rows); the real cost was the CPU postprocess
  (43 s → 1.5 s, scanline box scoring `74b8ac5`, see HISTORY). Detection graph
  compute is only ~3 s on CPU and Metal `conv_transpose_2d` is still ~13× slower,
  so **CPU stays the correct default** — a faster Metal `conv_transpose_2d` (or a
  1/4-res prob-map + cheap upscale) is the only remaining, low-value, upstream
  lever for GPU-default detection.
- **bidirlm-omni GGUF re-quant follow-up.** The text-tower converter bug is fixed
  and `bidirlm-omni-2.5b-q8_0.gguf` re-uploaded (text cos 1.0 f16 / 0.9992 q8_0),
  but the repo's f16 + imatrix q4_k/q5_k/q6_k and the whole `-textonly` repo are
  still the OLD (text-broken) conversion — regenerate them from the fresh f16
  (imatrix variants via the imatrix pipeline). Kaggle-only (16 GB Mac OOMs).
- **Regression-guardrail residuals.** `bert_ner` dumper written but its ref is
  download-blocked; face *recognition* (arcface/sface) unguarded (no local rec
  GGUF; detection is guarded). All SR/restoration (11) + esrgan/safmn + lilt +
  lfm2 + the closed engines are auto-guarded in `tests/regression/manifest.json`.
- **`core/vlm_decoder.h` — deferred.** A unified scalar decode loop; only 2 scalar
  engines remain, so abstracting is premature. Revisit if a 3rd appears.
