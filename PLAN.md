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
| 2026-07-17 | `feat/embeddinggemma-community-gguf` | **Community/official `gemma-embedding` EmbeddingGemma GGUFs crash on load** (encoder QKV reshape overrun — it's GQA not MHA). Routing to `decoder_embed.cpp` + **real bug = TOKENIZER**. | **IN PROGRESS — routing + tokenizer FIXED locally, NOT shipped.** (1) Arch-gated routing (3 edits): loads 24L/768d, no crash. (2) **DOMINANT bug was the TOKENIZER** (like sibling modern-bert): decoder loader hardcoded BPE, but this GGUF is llama.cpp SPM (`model=llama`, scores, NO merges) → loaded as char-level (BPE w/ 0 merges). Fixed: decoder loader now detects merge-less+scored vocab → SentencePiece; added SPM-BPE bigram-merge mode + `add_space_prefix` to `SentencePieceTokenizer` (Viterbi kept for XLM-R). **Tokens now match HF token-for-token; garbage-guard margin 0.038 → 0.393.** Next: backbone parity vs HF (confirm no norm bug) + quantify residual Dense gap; decide whether to bake Dense. Recipe: `handover-prompts/embeddinggemma-community-gguf.md`. |
| 2026-07-16 | (landed `main`) | **JSON I/O hardening + `core_json` + community-GGUF compat** — #34/#33, A1–A4, B1/B2, scalar migration, ground-truth HF parity + precision-control automation, CI drift guards. | **DONE — all on `main`** (see HISTORY.md 2026-07-16). |
| 2026-07-16 | (landed `main` `77b829b`+`d3f447b`) | **Community `modern-bert` GGUFs load to garbage** — true blocker was BPE-tokenizer support for community GGUFs (not just loader aliases). | **DONE — landed + pushed to `main`; docs in LEARNINGS/HISTORY.** Fixed in order: model-string-authoritative tokenizer dispatch (gpt2→BPE over the vocab-size heuristic), BPE merges from the `tokenizer.ggml.merges` KV array, GPT-2 ByteLevel regex pre-tokenizer, loader aliases (attn_norm/ffn_norm/output_norm, GeGLU-by-shape reroute of fused `ffn_up` [H,2*inter]) + metadata (pre_ln, inverted dual RoPE theta, sliding-window→local/global) + exact-erf GeGLU. **Per-stage q8_0 vs HF: emb_ln_out cos=0.999928 (structural gate PASS) + all 22 layers 0.9999+; final CLS-pool cos=0.999602. f16 control = cos=1.000000 at EVERY stage + 0.999999 final (graph exact, gap is quant).** Tokens now match HF `[50281,25521,1533,50282]`. Matrix entry `gte-modernbert-base` + garbage-guard margin 0.51 (was −0.089). Full 5-model matrix still PASS (no regression). |
| 2026-07-15 | `chore/pub-crispembed-dart` | pub.dev quality: crispembed **0.15.1** to 160/160 pana points (add example/README, enable `lints/core` + brace/dangling-doc fixes) | **DONE — publishing 0.15.1.** Docs/lint only, no behaviour change. |
| 2026-07-13 | `feat/handwritten-fixtures` | Close the bttr/hmer/posformer `expected_text: null` gap with in-domain CROHME fixtures | **DONE — validated, pending push.** Confirmed **no bug** — the 3 CROHME models were guarded on a *printed* image (out-of-domain); on correctly-rendered CROHME 2014 all three read simple formulas correctly + deterministically (CPU==Metal). Added a `sample_hf` harness mechanism: fetches one CROHME image from `Kitajiang/test2_CROHME2014` (pinned rev, row 23 `C_t=C+C=2C`) at test time so CC-BY-NC-SA data stays OUT of the MIT repo. Pinned `expected_text` for all 3 (`run_one` cer 0.000). |
| 2026-07-13 | `feat/transcoda-omr` | Transcoda-59M zero-shot OMR (full-page score → Humdrum `**kern`; ConvNeXt-V2-tiny enc + 8-layer RoPE cross-attn decoder). **Clean-room** (weights CC-BY-4.0, code AGPL — engine written from paper + config + oracle only) | **DONE (on `main`).** Engine `src/transcoda_ocr.{h,cpp}` + converter + oracle + wiring (CMake/dispatcher/CLI/registry) + quantizer conv keep-guard. f32 `test-transcoda-diff` **all stages cos=1.000000** (CPU & Metal) + **argmax 191/191**, native preproc bit-exact. **Both f32 and q8_0 greedy decodes byte-identical to the HF reference** (460 chars / 203 tokens). HF `cstr/transcoda-omr-GGUF` (f32 224 MB + q8_0 65 MB + **CC-BY-4.0 card w/ attribution**, license verified landed). Registry entry + regression fixture (`page_transcoda.png` from CC-BY-4.0 verovio-synth-omr, **cer 0.000**, garbage-guard PASS). Fixed: KV view-stale, rep-penalty per-unique-token, `/`-separator. Full wiring (README/omr.dart; bindings auto-dispatch by arch). **Perf: persistent device-KV decode 2.4–4× faster, byte-identical** (host path behind `TRANSCODA_OCR_HOST_KV=1`). Deferred to backlog: beam-3 + `**kern` grammar-constrained decode. **Fully done** (milestone in HISTORY.md; deep-dive in LEARNINGS.md). |
| 2026-07-13 | `feat/tromr-engine` | Polyphonic-TrOMR OMR (engine + wiring + quants + fixture) | **DONE** — engine `src/tromr_ocr.cpp` on `main` (cos 1.0 / 100% argmax / byte-exact); HF `cstr/tromr-GGUF` (f32 + q8_0 31 MB w/ F16 backbone + Apache-2.0 card); registry + regression fixture (cer 0.000). |
| 2026-07-13 | `feat/flova-omr` | Flova/omr_transformer — handwritten/whiteboard OMR (donut-swin + mBART VED → LilyPond, Apache-2.0) | **DONE (on `main`)** — engine `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact incl. native preproc), `tests/test_flova_diff.cpp`, CMake, CLI dispatcher + registry. HF `cstr/flova-omr-GGUF` (f32 573 MB + q8_0 162 MB byte-exact + Apache-2.0 card). Regression fixture landed (`feat/flova-regression-fixture`): `staff_flova.png` (model card sample1.png) + golden LilyPond `c'2 a''8 c''8 r4 c'1 e'8 c'8 c'8 a''8 f'4 a'8 c'8`, run_one cer 0.000. **Fully done.** |
| 2026-07-13 | `feat/flova-regression-fixture` | Flova OMR regression fixture (manifest entry + `staff_flova.png`) | **Landed `main` (`67ddc99`).** `run_one.py --name flova` PASS (garbage-guard + text cer 0.000 vs q8_0 from `cstr/flova-omr-GGUF`, CPU==Metal). |
| 2026-07-13 | `feat/smt-regression-fixture` | SMT OMR regression fixture (manifest entry + `staff_smt.png`) — completes the OMR guardrail trio (SMT/TrOMR/Flova) | **DONE — validated, pending push to `main`.** `run_one.py --name smt` PASS (garbage-guard + text cer 0.000 vs `smt-grandstaff-q8_0.gguf` from `cstr/smt-grandstaff-GGUF`, CPU==Metal identical, deterministic bekern decode). |
| 2026-07-13 | `feat/smt-fp-fullpage` | SMT++ **full-page** pianoform OMR (`PRAIG/smt-fp-grandstaff`) | **DONE — merged `main`, corrected + optimized.** fp checkpoint = `antoniorv6/SMT` (main rewrite): needs scaled attn `d_head^-0.5`, no pre-head ReLU, decoder tensor rename, head Linear, `reduce_ratio=1.0`. **Correctness fix: NO invert** — I'd copied SMT-plusplus `RandomInvert`, but the checkpoint's repo (SMT-main `convert_img_to_tensor`) is plain Grayscale+ToTensor; WITH invert the real HF model itself degenerates into `8 . r` repetition, WITHOUT it reads the page correctly (key/time/notes match golden, terminates). Per-stage cos was 1.0 either way — only the decoded roundtrip vs a no-invert reference caught it. **Perf: 485→~26 ms/step (~18×)** via persistent device KV (Pattern A) + reserved gallocr sched-free (Pattern B) + cross-K/V stored once in mul_mat-ready layout (skips per-step O(n_enc) cont); whole page ~2 min, was not-finishing. Output byte-identical CPU==Metal, f32==q8_0 (2312 tok, coherent). q5_k shipped (13MB, 0.04% token-CER); q4_k NOT shipped (plain degenerates; even encoder+head-protected q4_k is dominated by q5_k). Quantizer guards added (smt LM head `decoder.out_layer` + ConvNext encoder → Q8_0). HF `cstr/smt-fp-grandstaff-GGUF` re-uploaded (invert=false). Registry `smt-fp` + README on `main`. Fixture skipped (full-page decode too slow for CI). |
| 2026-07-13 | opus-1m (perf sweep) | DBNet detection postprocess — scanline box scoring | **Landed `main`** (`74b8ac5`, 28× faster, byte-identical) |
| 2026-07-13 | opus-1m (perf sweep) | Decoder op-fusion investigation | **Done** — measured marginal on compute-bound + Metal-auto-fused decoders (`58a3751`); QKV concat-matmul deferred |
| 2026-07-13 | opus-1m (perf sweep) | Kaggle CUDA confirmation (Class-A + Gap-5) | ✅ **DONE** — clean re-run (v9, `/tmp` ENOSPC fix `8f175cb`). Class-A/Gap-5 **confirmed PASS on CUDA**: deepseek-ocr2, dat, swinir, qwen2vl-3b, lfm2_colbert. The 14 FAILs are NOT regressions in the fixed engines: glm-ocr/internvl2 = known Class-B (Turing/Pascal); pcs/fireredpunc/fullstop = `test-punct-diff` not built in this config; layout-heron = SIGABRT teardown; granite-vision = text PASSES, only 3 diff stages cos 0.95–0.97; hat = harness no-parse. **Follow-ups landed:** `be6ec54` (teardown-tolerance + run_check-skip) took v10 **14→9**; then `2af57b1` fixed the diff-output parser (ANSI codes, colon-less `cos_min=`, table formats) — `lfm2`/`lilt`/`layout`/`hat`/`pan`/`tbsrn` were **false "no-parse" FAILs** (verified locally: lfm2's 20 stages all pass). v11 running, expected **9→3**. Only genuinely-open remainder needs Turing/Pascal HW: Class-B vision (glm-ocr/internvl2) + granite projector cos drift. |
| 2026-07-14 | `feat/uocr-stacked-experts` | Unlimited-OCR: stacked MoE experts (same win as ds-ocr2 #4) | **DONE — validated, pending merge.** Verbatim port of ds-ocr2 #4 (same DeepSeek-V2 MoE). Kaggle-reconverted `baidu/Unlimited-OCR` (byte-validated vs source; the v1 run hung 3h → fixed by the dev-guide single-thread OMP/BLAS converter prefix), uploaded f16+q4_k `-stacked` to `cstr/unlimited-ocr-crispembed-GGUF` (rev `b11fef884fee`, non-clobber). **M1 Metal q4_k A/B: output BYTE-IDENTICAL on all 3 loader paths; peak footprint 4.32→3.11 GB (−1.21 GB, −28%).** Registry promoted to stacked-default; first regression entry `unlimited-ocr-stacked` added. |
| 2026-07-13 | `feat/ds-ocr2-stacked-experts` | DeepSeek-OCR-2 #4: converter-emitted stacked MoE experts (drop ~1.3 GB dup) | **DONE — validated end-to-end, pending merge.** Converter emits `l.blk.{i}.ffn_{gate,up,down}_exps.weight` `[in,out,n_exp]` (byte-identical to runtime `stack_moe_experts`). Loader loads them directly (skips the copy) + per-expert views for the `DS_MOE_CPU` fallback + backward-compat legacy path. Kaggle kernel reconverted `deepseek-ai/DeepSeek-OCR-2`, **byte-validated stacked slices vs source** (all 5 checks), uploaded f16+q4_k to `cstr/deepseek-ocr2-crispembed-GGUF` as `-stacked` files (non-clobbering, rev `ec0fda03901c`). **Local M1 Metal A/B (q4_k, back-to-back): decoded output IDENTICAL ("The quick brown fox…" cer 0.0) on all 3 loader paths (prestacked / DS_MOE_CPU views / legacy); peak footprint 5.27→3.97 GB (−1.30 GB, −25%).** Regression entry `deepseek-ocr2-stacked` added (guards prestacked path); legacy entry kept (guards backward-compat). **Registry promoted: auto-download default → `deepseek-ocr2-q4_k-stacked.gguf` (distinct cache name → clean re-download; loader stays backward-compatible).** Fully landed. |
| 2026-07-13 | `debug/layout-cross` | Last portfolio FAIL: layout-heron `dec_0_cross_out` | **LANDED `main`** (`d7f0480` fix + `e9bba14` docs). Root cause: NOT an inference bug — the 300 decoder queries are picked by `partial_sort` over ~8400 near-tie encoder proposals (`layout_detect.cpp:1318`), so a tiny backend FP delta in enc_output (cos 0.99999) reorders near-tie ranks and index-aligned cross_out cos craters (mean 0.79/min −0.08 Metal) even though the VALUES are correct (final boxes unaffected — score-sort+NMS). Fixed by comparing `dec_0_cross_out` **permutation-tolerantly** (best-cosine match; PASS Metal 0.947/0.999, CPU 0.967/0.999; scrambles still collapse to ≤0.08 vs the 0.85 gate). Portfolio **14→0 FAIL**. LEARNINGS + HISTORY updated. |
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

### Ecosystem-compat + input-parsing hardening (A1–A4, opened 2026-07-16)

Four items surfaced while fixing issues #33/#34 (`fa5bd9e`). #34 fixed the JSON
input parser for the 3 endpoints the issue named; #33 taught the loader the
`nomic-bert-moe.*` GGUF names. Both fixes were *instance-level* — these items
close the underlying **bug classes**. Ordered by value; A1/A2 are independent.

**A1 — finish the JSON parser migration (live bug).** `examples/server/json_input.h`
(`json_extract_strings`, escaping-aware) now backs `/v1/embeddings`, `/api/embed`
and `/api/embeddings`. Four endpoints still use the old delimiter-scan and carry
the *identical* `]`-in-string / `\"` / `\\` cardinality bug:

| server.cpp | endpoint | field |
|---|---|---|
| ~292 | **`/embed`** | `"texts"` array — an embedding endpoint, missed by the #34 scope |
| ~1113 | `/rerank` | `"documents"` array |
| ~1635 | `/ner/extract` | `"labels"` array |
| ~1770 | `/kie/extract` | `"labels"` array |

A `/rerank` document containing `]` silently mis-splits today. Steps: route all
four through `json_extract_strings`; gate the OLD scan behind
`CRISPEMBED_SERVER_LEGACY_JSON=1` (regression-bisection per the env-gate rule —
never remove the gate); A/B old-vs-new over a payload corpus (must be *identical*
on unescaped payloads, differ only where the old one was wrong); extend
`test-server-json-input`. Gate: unit test + live `tests/test_server_live.py`.

**A2 — arch-driven hparams + strict mode (systemic version of #33).** `load_model`
hardcodes `bert.*`/`xlmr.*` key chains; #33 appended `nomic-bert-moe.*`, which is
an alias list that grows one model at a time (the fork's PR did the same and was
still incomplete). llama.cpp/Ollama *always* write `<general.architecture>.*`
(`embedding_length`, `block_count`, `attention.head_count`, …), so read
`general.architecture` and derive the prefix generically — every future community
GGUF then works with no new code, and the `nomic-bert-moe.*` lines collapse into it.
**The sharper half:** missing keys currently fall back to *silent defaults*
(384-dim / 6-layer / 1e-12 eps). #33 got lucky and failed loudly on a missing
tensor; for an arch whose tensor names *do* resolve, this silently emits a garbage
embedding with exit code 0. Add an opt-in hard-fail. Gates:
`CRISPEMBED_ARCH_HPARAMS=0` disables arch-derived lookup (default ON — purely
additive, only fires when the bert.*/xlmr.* keys are absent);
`CRISPEMBED_STRICT_HPARAMS=1` hard-fails on a missing required hparam (default OFF
— hard-fail could break models legitimately relying on a default). A/B: existing
models must be **byte-identical** with the gate on vs off; nomic-moe loads only
with it on. Unit test the key-resolution logic host-side.

**A3 — community-GGUF import matrix.** The registry ships
`nomic-embed-text-v2-moe` (model_mgr.cpp:435) pointing at **our own `cstr/`**
conversions, which load fine — while the *community* GGUF (`nomic-ai/…-GGUF`,
what users reach for first) did not, which is exactly what #33 was. Nothing tests
the ecosystem's output of models we claim to support. Add a harness that loads
community (llama.cpp/Ollama) GGUFs and asserts load + cosine-vs-reference; env-gated
so it skips without models. This is what catches the next #33 before a user does.

**A4 — red-main + toolchain-drift signal.** `main` was red 2026-07-14 → 07-16 and
nobody noticed until a push landed on it: `setup-emsdk` was unpinned, `latest`
drifted 6.0.2 → 6.0.3, and 6.0.3's clang segfaults compiling `layout_detect.cpp`
(fixed by pinning 6.0.2 in `0e1a1b9`). Audit the workflows for other
"latest"-alias toolchain installs with the same drift exposure, and add a signal
so a red `main` self-reports.

### FOUND (2026-07-16): community `modern-bert` GGUFs — DEEPER than aliases (validated diagnosis)

**RESOLVED + SHIPPED 2026-07-16 (`feat/modernbert-community-gguf`).** The fix
followed the recipe below exactly, each step gated per-stage. Landed:
`src/crispembed.cpp` (model-string-authoritative tokenizer dispatch; BPE merges
from the `tokenizer.ggml.merges` KV array; modern-bert metadata block with the
inverted dual RoPE theta + SWA + pre_ln + `geglu_erf`; loader aliases
attn_norm→ln1/ffn_norm→ln2/output_norm→final_norm + GeGLU-by-shape reroute),
`src/tokenizer.h` + `src/tokenizer_bpe.cpp` (`set_gpt2_regex_pretok` + the GPT-2
ByteLevel regex pre-tokenizer), `tests/community_gguf_matrix.json`
(`gte-modernbert-base` entry) + `tests/prove_quant_control.py` (`control_repo`
override). **Validation: per-stage q8_0 vs HF fp32 emb_ln_out cos=0.999928 (gate
PASS) + 22 layers 0.9999+; f16 control cos=1.000000 at EVERY stage (graph exact);
final CLS-pool cos q8_0=0.999602 / f16=0.999999. Tokens match HF
[50281,25521,1533,50282]. Full 5-model matrix PASS (no regression).** The recipe
below is retained for reference.

**Executable fresh-agent recipe: `handover-prompts/modernbert-community-gguf.md`.**

**UPDATE after attempting the fix + HF per-stage validation (fix/modernbert-load,
NOT shipped):** the loader-alias theory below is necessary but NOT sufficient, and
the real first divergence is the TOKENIZER, not the graph. Sequence of findings:

1. Added the aliases + metadata (attn_norm/ffn_norm/output_norm; pre_ln=true;
   rope_theta=freq_base_swa/local + rope_theta_global=freq_base/global;
   sliding_window→local_attention_window; sliding_window_pattern→global_attn_every_n;
   GeGLU-by-shape reroute of `ffn_up` [H,2*inter]→ffn_up_gate_w). Result: it LOADS
   (22L/768d, RoPE θ=10000 local, pre-LN) and EMITS a 768-dim vector.
2. **But the embedding is garbage** — garbage-guard cos(related)=0.068 <
   cos(unrelated)=0.157 (NEGATIVE margin). "Loads + emits" proved nothing (HARD
   RULE #3), so I ran the per-stage HF diff vs `Alibaba-NLP/gte-modernbert-base`.
3. **The STRUCTURAL GATE fails**: `emb_ln_out cos=0.583` (|ours|=40.8 |ref|=30.8),
   i.e. divergence is BEFORE block 0 — tokenization/embeddings, not GeGLU/attn.
4. **Root cause — tokenizer dispatch ignores the standard key.** The GGUF declares
   `tokenizer.ggml.model = "gpt2"` (BPE) + `tokenizer.ggml.pre = "modern-bert"`, and
   has NO `tokenizer.ggml.type`. crispembed's dispatch reads ONLY its own numeric
   `tokenizer.ggml.type` (default 0) and the `n>100000→SPM` heuristic; 50368 vocab
   + absent type → it picks **WordPiece**. HF tokenized to BPE ids
   [50281,25521,1533,50282]; crispembed's WordPiece produces different ids →
   different embeddings → garbage from token 0. (e5/granite only worked by luck:
   their 250K vocab tripped the `n>100000→SPM` heuristic, which happened to be right.)

**Why nothing was shipped:** the loader-alias change on its own turns a LOUD
failure ("missing required tensor") into SILENT garbage (loads, exit 0, wrong
embedding). That is strictly worse and is exactly the silent-default failure mode
A2/STRICT_HPARAMS exists to prevent. So `src/crispembed.cpp` was reverted; only
this diagnosis is recorded. Do NOT ship the aliases without the tokenizer fix.

**Real fix (bigger than #33 — it's "support community BPE-tokenizer encoder
GGUFs"), in order, each validated by the per-stage harness:**
  a. Tokenizer dispatch: when `tokenizer.ggml.type` is absent, map
     `tokenizer.ggml.model` string → type (`gpt2`→BPE, `bert`→WordPiece,
     `t5`/`unigram`→SPM). Authoritative over the vocab-size heuristic.
  b. BPE merges: community GGUFs store them in the `tokenizer.ggml.merges` KV
     STRING ARRAY (confirmed present), not the `tokenizer.merges` TENSOR crispembed
     reads. Load from the KV array for these.
  c. gpt2 byte-level BPE + the `modern-bert` pre-tokenizer regex (verify
     crispembed's BPE covers gpt2 byte-to-unicode + the pre-tokenizer). GATE:
     emb_ln_out cos must reach ~0.99999 before trusting any layer.
  d. ONLY THEN validate the graph: the 4 GeGLU variants
     (`ggml_geglu`/`_swapped`/`_erf`/`_erf_swapped` — ModernBERT uses EXACT gelu, so
     `_erf`; split order per llama.cpp's `ffn_up` layout), plus rope local/global
     assignment and the SWA mask, via `tests/test_encoder_diff.py` +
     `tools/dump_encoder_reference.py` (both dumps already reproduce for this model).

Loader-alias code (correct, reverted — re-apply as step (d)'s prerequisite):
tensors attn_norm→ln1, ffn_norm→ln2, output_norm→final_norm, GeGLU-by-shape when
`fc1_w->ne[1]==2*n_intermediate`; metadata as in step 1 above.

--- original (shallower) diagnosis below, superseded by the above ---

Wider matrix-coverage survey load-tested 3 new community GGUFs of shipped models:
  - `intfloat/multilingual-e5-small` (arch **bert**, 12L/384d): LOADS ✓
  - `bartowski/granite-embedding-107m-multilingual` (arch **bert**, 6L/384d): LOADS ✓
  - `eranmazur/gte-modernbert-base-Q8_0` (arch **modern-bert**, 22L/768d): **FAILS** —
    `missing required tensor layer=0 name=ln1.weight` (×22). A genuine new
    #33-class bug: users who grab a community modern-bert GGUF can't load it.

**crispembed ALREADY has the ModernBERT graph** (fused GeGLU `ggml_geglu`,
sliding-window attention `modernbert_swa_enabled`, global-attn-every-N, dual RoPE
theta) — so this is an alias + shape-detect gap, NOT a port. Fully diagnosed from
the GGUF's tensors + metadata:

Tensors (community modern-bert names → what the loader looks for):
  - `blk.N.attn_norm.weight`  → add alias to `ln1_w` (currently ln1/attn_output_norm)
  - `blk.N.ffn_norm.weight`   → add alias to `ln2_w` (currently ln2/layer_output_norm)
  - `output_norm.weight`      → add alias to `final_norm_w`
  - `blk.N.attn_qkv` / `attn_output` / `token_embd_norm`: already aliased ✓
  - `blk.N.ffn_up.weight` is **[768, 2304] = [H, 2*inter]** — the fused GeGLU
    weight, but named `ffn_up` (same name as a PLAIN ffn up). The loader must
    DETECT GeGLU BY SHAPE (ne[1] == 2*n_intermediate) and route it to
    `ffn_up_gate_w`, not `fc1_w`. feed_forward_length=1152, so 2304=2*1152.
  - layer 0's attn_norm is Identity in ModernBERT (may be absent) — `ln1_w` must
    be optional per-layer (the pre-LN path already guards `if (pre_ln && L.ln1_w)`).

Metadata (modern-bert.* — A2 arch-derived already reads dims/pooling; these need
mapping, and the RoPE theta is INVERTED vs crispembed's naming):
  - `pre_ln = true` (architectural; ModernBERT is pre-LN — no metadata key, force
    it for arch==modern-bert)
  - `rope.freq_base = 160000` is the GLOBAL theta; `rope.freq_base_swa = 10000` is
    the LOCAL theta. crispembed uses `rope_theta`=local base, `rope_theta_global`
    for global layers → set rope_theta=10000, rope_theta_global=160000 (do NOT
    just read freq_base into rope_theta — that's the bug crispembed currently has,
    it reported theta=160000).
  - `attention.sliding_window = 128`         → `local_attention_window`
  - `attention.sliding_window_pattern = 3`   → `global_attn_every_n` (global at
    il%3==0, matching ModernBERT's 0,3,6,…)
  - `pooling_type = 2` (CLS) — A2 maps this ✓

**Validation is MANDATORY before claiming (HARD RULE):** this is graph-routing
(GeGLU-by-shape, dual-theta assignment, SWA pattern). Build, then per-stage vs the
HF `Alibaba-NLP/gte-modernbert-base` (needs `trust_remote_code`) via
`tools/dump_encoder_reference.py` + `tests/test_encoder_diff.py`, plus the q8_0
final-embedding cosine, plus an f16 control (`prove_quant_control.py`). Add a
matrix entry once it passes. **NOT done here:** diagnosed only — deferred because
the fix touches the compute graph and the box was at load ~143 from other sessions
(a build + HF-modernbert validation shouldn't be rushed/contended). Pick up on a
quiet box with the recipe above.

e5-small + granite-107m matrix coverage — **RESOLVED 2026-07-16
(`feat/modernbert-community-gguf`), one added, one is a genuine community-export
bug:**
  - **`granite-embedding-107m-multilingual` — ADDED + validated.** Community
    `lmstudio-community` q4_k GGUF: arch `bert`, `tokenizer.ggml.model='t5'`
    (250002-token unigram + scores) → SPM via the new model-string dispatch;
    CLS pooling. **Per-stage q4_k vs HF fp32: emb_ln_out cos=0.999951 (gate PASS)
    + 6 layers 0.9928–0.9969; final CLS-pool cos=0.996145; f16 control
    cos=1.000000 at every stage + 1.000000 final.** Garbage-guard margin 0.31.
    This is the matrix's FIRST SentencePiece entry.
  - **`multilingual-e5-small` — NOT added; it does NOT parity-match, and it is an
    under-specified community export, not our bug.** The `rodion-m` fp32 GGUF
    (arch `bert`, `tokenizer.ggml.model='bert'` → 250037-vocab → SPM by the
    `n>100000` heuristic) has a `position_embd.weight` tensor but NO
    `bert.position_offset` key → crispembed uses offset 0, while
    `intfloat/multilingual-e5-small` is XLM-RoBERTa (padding_idx=1 → position
    offset **2**). Result: structural gate `emb_ln_out` cos=**0.467** (norms match
    16.4/16.5, direction wrong = pure position-embedding shift). NOT generically
    auto-detectable: granite has the SAME RoBERTa bos=0/eos=2 SPM tokenizer yet
    needs offset 0, so there is no tokenizer-side signal to key offset-2 off. The
    fix belongs upstream (the community GGUF must emit `position_offset`), or
    users should use a GGUF that declares it (our `cstr/*` e5 sets it). Left as a
    documented finding; no speculative heuristic shipped. **A position_embd
    row-count heuristic was CONSIDERED and RULED OUT (verified 2026-07-16):**
    RoBERTa allocates 2 extra rows (max_pos = ctx+2), so `position_embd` rows
    beyond `context_length` might have signalled offset 2 — but the community e5,
    granite, and bge GGUFs ALL have `position_embd`=[384,512] with ctx=512 (the
    e5 converter dropped the 2 padding rows), so there is NO row-count signal
    either. Conclusion: with an identical 512-row table AND an identical RoBERTa
    SPM tokenizer, e5 (needs offset 2) and granite (offset 0) are
    indistinguishable from GGUF metadata alone — crispembed cannot auto-detect it;
    the offset must be carried in the GGUF. Thread CLOSED (not fixable our side).
  - Regression note: both still load as SentencePiece under the new
    model-string-authoritative dispatch (t5→SPM directly; bert+250K→SPM via the
    retained heuristic), so the dispatch change did not regress the bert/SPM path.

### Encoder ground-truth parity harness (A3 follow-on, 2026-07-16)

A3 shipped with ONE model and no Python ground truth: its checks were rc/shape,
a semantic garbage guard, and a cross-conversion A/B — none of which is ground
truth (two of our own conversions can agree and both be wrong). Extended to 4
community GGUFs, each now gated against the ORIGINAL HF/PyTorch model, per-stage
as well as final.

**Tools.** `tests/hf_parity_community.py` (final-embedding cosine vs
sentence-transformers) · `tools/dump_encoder_reference.py` (HF per-stage
intermediates -> GGUF) · `tests/test_encoder_diff.py` (per-stage compare) ·
`CRISPEMBED_DUMP_LAYERS_GGUF=<path>` (our side dumps full tensors; the pre-existing
`CRISPEMBED_DUMP_LAYERS=1` only printed a 6-float peek, which cannot be compared).

**Measured q4_k vs HF fp32:** bge-small 0.9962 · MiniLM 0.9919 · nomic-v2-moe
0.9797 · nomic-v1.5 **0.9515**.

**The precision control is the whole method.** A low cosine alone never
distinguishes "quant floor" from "our bug" — re-run the SAME code path at f16/f32:

| model | q4_k vs HF | f16/f32 vs HF (per-stage) |
|---|---|---|
| bge-small-en-v1.5 (bert: split QKV, abs pos) | 0.9962 | **f32: cos=1.000000 at all 12 layers** |
| nomic-embed-text-v1.5 (nomic-bert: fused QKV, RoPE) | 0.9515 | **f16: cos=1.000000 at all 12 layers** |

So BOTH encoder paths are exactly correct and every q4_k delta is quantization.
nomic-v1.5's 0.9515 is a real quality fact, not a bug: it is concentrated in the
LAST block (layer_10 0.9977 -> layer_11 0.9499 — a step, not smooth drift), i.e.
that model's final layer is unusually quant-sensitive. Prefer f16/q8 for it.

**Three harness bugs this found — all invisible by reading the code:**
1. `layer_{n-1}` never existed in our dump: the graph renames the last block's
   output to `encoder_out`, so the block that FEEDS POOLING was silently absent —
   and the comparer `continue`d past missing stages, so absence looked like a
   pass. Missing stages now FAIL; `encoder_out` is dumped and aliased.
2. `NomicBertModel.forward()` rejects `output_hidden_states`, so the dumper only
   worked for stock HF models. Added a forward-hook fallback (finds the block
   ModuleList by probing known paths).
3. The structural gate read cos=0.69 on nomic while every layer read 1.000000 —
   impossible for a real input mismatch, therefore the HARNESS was wrong: BERT's
   `embeddings` module includes the LayerNorm, nomic's does not, so it compared
   pre-LN against our post-LN. Fixed by capturing block 0's INPUT via a
   forward_pre_hook (pre-block-0 by definition, architecture-agnostic). The gate
   now prints |ours| and |ref| so this class of artifact is visible at a glance.

**Precision control — now automated (2026-07-16).** The manual f16/f32 control
is wired into the matrix: entries carry `control_file` + `control_min_cos`, and
`tests/prove_quant_control.py --all` runs the control GGUF per-stage vs the HF
reference and asserts every stage clears the floor. "Prove it's quant, not a bug"
is now one command. Covers bge-small (f32), nomic-v1.5 (f16), nomic-v2-moe (f16),
gte-modernbert (f16), granite-107m (f16).

#### Community-matrix coverage roadmap — candidate archs to add (2026-07-16)

Current entries (6): bge-small + all-MiniLM (`bert`, split-QKV/abs-pos/WordPiece),
nomic-v1.5 (`nomic-bert`), nomic-v2-moe (`nomic-bert-moe`), gte-modernbert
(`modern-bert`, gpt2-BPE), granite-107m (`bert` + `t5`/SPM, CLS). Each remaining
family below exercises a DISTINCT loader/graph path not yet guarded against a
third-party GGUF. Ordered by coverage value; every one is a load + shape +
garbage-guard + HF per-stage entry (the granite recipe), each MUST be gated on the
per-stage structural cosine (a garbage-guard-only pass hides an e5-style shift).
Availability probed 2026-07-16 (repos listed are candidates, not yet validated):

| Candidate | arch / path it covers | Fits dense driver? | Candidate community GGUF | Watch-out |
|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | `qwen3` DECODER embed — last-token pool, **causal**, gpt2-BPE decoder path (distinct from modern-bert's ENCODER BPE) | ✅ (last-token) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (official) + many | **ADDED + validated (2026-07-16), CLEAN — no loader change.** decoder_embed.cpp already takes blk.N.* + the gpt2-BPE KV-merges path is handled. Final cosine vs HF: q8 mean 0.999727, **f16 mean 1.000000** (graph exact); garbage margin 0.58 |
| **EmbeddingGemma-300m** | `gemma-embedding` — mean pool, **Dense bottleneck + Matryoshka** projection (see LEARNINGS "non-orthogonal Dense bottleneck") | ✅ (mean) | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`, `unsloth/…`, `lmstudio-community/…` | **FOUND TWO GAPS (2026-07-16), NOT a clean add — deferred to a dedicated session.** (1) The community GGUF CRASHES on load — but the CORRECTED diagnosis (2026-07-16,
VERIFIED) is a **routing bug, NOT a shared-encoder rewrite**: arch `gemma-embedding`
is absent from the `is_dec` allow-list (`src/crispembed.cpp:2253`, which has `gemma3`
but not the hyphenated `gemma-embedding`), so it falls to the generic MHA encoder and
aborts at `src/crispembed.cpp:1036-1038` (Q/K/V all reshaped to `n_heads`; GQA
`head_count_kv=1` overruns). **`decoder_embed.cpp` already implements the full Gemma3
block** (GQA, q/k-norm, sandwich `post_attention_norm`/`post_ffw_norm`, embed-scale,
gelu-tanh, mean pool, `dense.N.weight` Dense stack, sliding window, bidirectional). A
**3-line arch-gated routing change** (add `gemma-embedding` to `is_dec` + to
`decoder_embed.cpp:63` arch_pfx + extend the gemma3 branch `~:115` with
`is_bidirectional=true`) makes it LOAD correctly — VERIFIED: `24L/768d, 3 heads (1 kv),
pool=mean, bidirectional`, no crash. **But it must NOT ship:** garbage-guard margin is
only **0.039** (all cos ≈0.80), because (2)↓. Full executable recipe:
`handover-prompts/embeddinggemma-community-gguf.md`. (2) The ggml-org GGUF has NO Dense/Matryoshka tensors (only `token_embd`+`output_norm`), so even once it loads it yields the RAW mean-pool, not EmbeddingGemma's Dense-projected embedding — full parity needs a GGUF that carries the Dense modules (check `unsloth/…F32`) or applying Dense out-of-band. crispembed's own gemma-embedding path also still has the open ~0.002 backbone discrepancy. GGUF downloaded (`embeddinggemma-300m-qat-Q8_0.gguf`). |
| **LFM2.5-Embedding-350M** | `lfm2` bidirectional hybrid — ShortConv + attention, **BOS-only wrap** | ✅ (CLS, pooling_type=2) | `LiquidAI/LFM2.5-Embedding-350M-GGUF` (official) | **FIXED + SHIPPED (2026-07-16), added to matrix.** Was a loader gap — `lfm2_embed` requires our `lfm.*` tensor names + `lfm2.<our>` hparam keys + a `lfm2.layer_types` c/a string; the official llama.cpp export uses `blk.N.*` + canonical `lfm2.*` keys + no layer-types string. Same class as modern-bert (alias gap), bigger. **Complete fix recipe (exact tensor + hparam maps, layer-type-from-tensor-presence, per-stage gate) in the "FOUND (2026-07-16): official `lfm2`…" subsection just below.** GGUFs already downloaded. Needs a quiet box for the build + `test-lfm2-diff` per-stage validation |
| **GTE-v1.5 (gte-base-en-v1.5)** | `NewModel` NTK-RoPE + GeGLU **tanh** (the path the modern-bert `geglu_erf` gate was explicitly kept OFF for) | ✅ | `cstr/gte-base-en-v1.5-GGUF` (our own; llama.cpp ❌ so third-party rare) | **ADDED + validated per-stage (2026-07-16).** q8 vs HF fp32: emb_ln_out gate 0.999927, all layers PASS (encoder_out 0.9926). Guards the tanh-GeGLU branch stays correct next to modern-bert's erf branch. Arch coverage (own GGUF), not ecosystem-compat |
| **MPNet (all-mpnet-base-v2)** | MPNet two-stream / T5-style rel-attn bias — **we are unique** | ✅ | `cstr/all-mpnet-base-v2-GGUF` (our own; no third-party — llama.cpp ❌) | Not a true ecosystem gap (no community export exists); lower priority |
| **XLM-R-large / multilingual-e5-large** | `bert`+SPM XLM-R at 1024-dim | ✅ | `soichisumi/…-Q8_0-GGUF`, `phate334/…`, `walsons/…` | **EXPECT the e5-small position-offset FAILURE** (XLM-R needs offset 2; community `bert`-arch GGUFs omit `position_offset`). Add ONLY if a community GGUF declares the offset — else it documents the same known gap |
| **SPLADE-v3 (sparse)** | MLM/sparse head — `has_sparse` path, NOT dense | ❌ (needs a sparse-specific check, not the garbage guard) | `mradermacher/Splade-V3-GGUF` | Driver has no sparse mode; would need a top-term overlap gate. Separate work |
| **DeBERTa-v2** | disentangled c2p/p2c rel-attn (`rel_embd`, `position_buckets`) — **we are unique**, highest-complexity encoder path | ✅ | **none found** (llama.cpp ❌, no community GGUF exists) | Blocked on the absence of any third-party GGUF; only our own conversion exists |

Status (2026-07-16 autodownload sweep): **Qwen3-Embedding ADDED (clean)**,
**LFM2.5-Embedding FIXED+ADDED** (loader gap), **granite-107m ADDED**, **e5-small
CLOSED** (under-specified export), **EmbeddingGemma DEFERRED** (crashes on load +
GGUF lacks Dense modules — see its row). Remaining candidates: **GTE-v1.5** (own
GGUF, guards the tanh-GeGLU branch) is the cleanest next; **MPNet** (own GGUF only,
no ecosystem export); **XLM-R-large** is expected to reproduce the e5 offset gap
(add only as a documented negative or if a GGUF declares the offset); **SPLADE**
needs a sparse-mode driver first; **DeBERTa-v2** is blocked on GGUF availability.
Do each on a quiet box (250K-vocab SPM reads + HF forwards are slow under
contention) and gate on the per-stage structural cosine.

#### FOUND + FIXED (2026-07-16): official `lfm2` LFM2.5-Embedding GGUF now loads

**RESOLVED + SHIPPED (`feat/modernbert-community-gguf`).** Fixed in `src/lfm2_embed.cpp`
exactly per the recipe below: tensor aliases (`lfm.*`↔`blk.N.*`/`token_embd*`),
hparam-key fallbacks (`lfm2.<our>`↔canonical `lfm2.*`; `head_count_kv` read as a
per-layer array → max), conv/attn layer-types derived from tensor presence, and a
memory-preserving reshape of the depthwise-conv weight to `[K,1,C]` (the export
ships it 2D `[K,C]`, which crashed `ggml_conv_1d_dw`). **Validated per-stage via
`build/test-lfm2-diff` vs the raw HF `Lfm2BidirectionalModel`: q8_0 = 0.9999 at
EVERY stage (post_embed structural gate + 16 layers + `cls_norm` final pooled) on
a short AND a long text; f16 control = cos=1.000000 at every stage (graph exact,
q8 gap is quant); garbage-guard margin 0.76.** Matrix entry `LFM2.5-Embedding-350M`
added (+ an LFM2-banner regex in the driver). The ST final-cosine (0.99 mean) is
looser only because sentence-transformers applies its own pooling, not the GGUF's
CLS — `test-lfm2-diff` (same-pooling, vs the raw model) is the authoritative gate.

--- original diagnosis (retained) ---

Surfaced by the autodownload matrix work. `LiquidAI/LFM2.5-Embedding-350M-GGUF`
(llama.cpp `lfm2` arch) aborts: `[lfm2_embed] required tensor
'lfm.embed_tokens.weight' not found` (×many). `src/lfm2_embed.cpp` was written for
OUR converter's `lfm.*` tensor names + `lfm2.<our>` hparam keys + a
`lfm2.layer_types` `c`/`a` string; the llama.cpp export uses `blk.N.*` tensors,
canonical `lfm2.*` keys, and no layer-types string. Same class as the shipped
modern-bert fix, bigger. crispembed already has a validated LFM2 graph and the
loader already reads the `tokenizer.ggml.merges` KV array (`:174`) + BOS-only wrap
— so this is an alias/hparam/layer-type job, NOT a port. GGUFs (q8_0 + f16 control)
already downloaded to `~/crispembed-live-cache/`. NOT shipped — loader aliases must
not land without the per-stage `test-lfm2-diff` gate (modern-bert lesson), which
needs a quiet box (this box was at load 78–130). Executable recipe:

- **Tensor aliases** (crispembed `lfm.*` → llama.cpp): `embed_tokens`→`token_embd`;
  `embedding_norm`→`token_embd_norm` (⚠ verify it's a FINAL vs INPUT norm — see
  landmine); per-block `operator_norm`→`attn_norm`, `ffn_norm`→`ffn_norm`,
  `ff.w1/w2/w3`→`ffn_gate`/`ffn_down`/`ffn_up` (⚠ verify the gate/up split per-stage),
  `conv.{conv,in_proj,out_proj}`→`shortconv.{conv,in_proj,out_proj}`,
  `attn.{q,k,v,out}_proj`→`attn_{q,k,v,output}`,
  `attn.{q,k}_layernorm`→`attn_{q,k}_norm`. Implement as two-name `get_any` lookups.
- **Hparam-key aliases** (`:144–154`): `hidden_size`→`embedding_length`,
  `n_layers`→`block_count`, `n_heads`→`attention.head_count`,
  `n_kv_heads`→`attention.head_count_kv`, `head_dim`→`attention.key_length`,
  `ff_dim`→`feed_forward_length`, `rope_theta`→`rope.freq_base`,
  `norm_eps`→`attention.layer_norm_rms_epsilon` (confirm exact names vs the GGUF KV).
- **Layer types (conv vs attn): DERIVE from tensor presence** — `blk.N.attn_q.weight`
  present → `'a'`, else `'c'` (llama.cpp doesn't write the string; the 350M pattern
  is real: blk.0=shortconv, blk.2=attention).
- **Validate (mandatory):** loads (16L/1024d) → `tools/dump_lfm2_reference.py` +
  `build/test-lfm2-diff` vs HF `LiquidAI/LFM2.5-Embedding-350M` (trust_remote_code,
  the BIDIRECTIONAL `Lfm2BidirectionalModel` — causal `AutoModel` gives ~0 cosine) →
  f16 control ~1.0 per-stage → final CLS-pool cosine → matrix entry.
- **Landmines:** never ship aliases without the per-stage gate; `token_embd_norm`
  placement (input vs final norm — the diff catches a misplacement); a swapped
  ff gate/up gives fluent garbage; Metal `ggml_mul` src[1] F32 (already handled in
  `lfm2_rms_norm`). Local convenience copy: `handover-prompts/lfm2-community-gguf.md`
  (gitignored — this section is the durable record).

### Transcoda OMR decode enhancements (deferred, 2026-07-13)

The shipped `transcoda_ocr` engine uses greedy decode (byte-identical to the HF
reference; persistent device-KV, 2.4–4×). The paper's two higher-accuracy decode
modes are **deferred** — both are large, and neither is byte-exactly validatable,
so they were intentionally NOT shipped (byte-exact-or-bust discipline). Concrete
plans for a follow-up session:

- **Beam search (width 3)** — the paper's headline (OMR-NED 18.46% vs greedy
  ~higher on Verovio-synth). HF config: `num_beams=3, length_penalty=1.0,
  repetition_penalty=1.1, early_stopping=True`.
  - *Where*: a `decode_beam(ctx, n_beams)` in `src/transcoda_ocr.cpp`, gated
    `TRANSCODA_OCR_NUM_BEAMS=N` (opt-in; greedy stays the default). Per-beam
    next-token logits via either B independent persistent KV caches (extend
    `pk_*` to a `[..., B]` beam dim) or the full-recompute `run_decoder` per beam
    (simplest, O(B·L²) — fine for opt-in).
  - *Algorithm* (mirror HF `BeamSearchScorer`): keep B live beams (init scores
    `[0,-inf,-inf]`), each step apply per-unique-token rep-penalty + `log_softmax`,
    add to beam score, take top-`2B` over the flattened `B×vocab`, route eos
    candidates to a finished pool with score `/(len**length_penalty)`, keep the
    top-B non-eos as the next beams; early-stop when B finished hypotheses exist;
    return the best finished (or best live) hypothesis.
  - *Validation*: (1) on the confident synth page `sample_page.png`, HF beam-3 ==
    greedy, so mine must be **byte-exact == greedy** there (a real regression
    gate); (2) on a real Polish scan (`btrkeks/polish-scores`, license "other" —
    LOCAL validation only, do NOT commit the image), HF beam-3 diverges from
    greedy at accent/ornament tokens (`16b#JJ`→`16bJJ`) and spine markers
    (`*^`/`*v`) — target **CER-close** to the HF beam-3 dump (byte-exact over a
    512-token uncapped scan is not realistically achievable; cascading). HF
    references already captured: `scratch-transcoda/oracle_beam3.kern.txt`,
    `polish_beam3.kern.txt`.

- **Grammar-constrained decode** — guarantees structurally-valid `**kern`
  (paper's `grammars/kern.gbnf` via xgrammar logits processors). Large: needs a
  GBNF parser + a per-step token-mask constraint engine (llama.cpp's
  `llama-grammar` is the reference, ~1k LOC). *Where*: a `kern_grammar.{h,cpp}`
  constraint module + a mask hook in the decode loop, gated
  `TRANSCODA_OCR_GRAMMAR=1`. *Validation*: structural only (every output parses as
  valid kern); no byte-exact HF target (xgrammar's tie-breaking differs). Lowest
  priority — greedy already emits valid kern on clean inputs.

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
- [x] **#4 Converter-emitted stacked experts (memory) — DONE
  (`feat/ds-ocr2-stacked-experts`).** Converter emits `ffn_{gate,up,down}_exps
  [in,out,n_exp]` (byte-identical to `stack_moe_experts`); loader loads them
  directly + per-expert views for the `DS_MOE_CPU` fallback + backward-compat.
  Kaggle-reconverted + byte-validated vs source; f16/q4_k on HF as `-stacked`
  (non-clobber). **M1 Metal q4_k A/B: peak footprint 5.27→3.97 GB (−1.30 GB),
  decoded output identical on all 3 loader paths.** Registry auto-download default
  **promoted to `deepseek-ocr2-q4_k-stacked.gguf`** (loader backward-compatible).
  Deep-dive in LEARNINGS.
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

- **ENCODER (embedding) path — the domain the 2026-07-16 community-GGUF work
  landed in, and NOT otherwise in this backlog (encoders are fast: 6–22 layers,
  batched).** One concrete micro-lever spotted:
  - **MoE FFN redundant `ggml_repeat` (nomic-bert-moe / nomic-embed-text-v2-moe).**
    The MoE FFN in `src/crispembed.cpp` explicitly expands the input
    `cur [H,TB] → [H,K,TB]` with `ggml_repeat` before `ggml_mul_mat_id`. llama.cpp's
    canonical MoE reshapes to `[H,1,TB]` and lets `mul_mat_id` BROADCAST the
    singleton expert-slot dim, so the repeat materializes K copies of the
    activations per MoE layer for nothing (6 MoE layers × K=2 on nomic-v2-moe).
    Gate landed on `main` (`5abc4de`), broadcast path behind
    `CRISPEMBED_MOE_NO_REPEAT=1` (default keeps the repeat).
    **Correctness VALIDATED (2026-07-16):** default vs `CRISPEMBED_MOE_NO_REPEAT=1`
    on `nomic-embed-text-v2-moe` is **BYTE-IDENTICAL (max_abs_diff=0.0, cos=1.0)**
    at BOTH f16 and q4_k (50-token input) — the broadcast is exactly the repeat, so
    HF cosine is unchanged by construction. **Latency INCONCLUSIVE / neutral:** a
    7-run bench A/B (graph-compute, T=50, Metal) gave repeat median ~188 ms vs
    norepeat ~195 ms but with ±100 ms run-to-run swings at load ~9 — the
    distributions fully overlap, so no reliable delta (matches the "may be
    perf-neutral" expectation; the repeat materializes only ~1.8 MB total). **Flip
    decision deferred:** per the dev-guide rule (flip only when it wins on speed AND
    quality), a clean flip needs a genuine quiet box (load <3) for a back-to-back
    median; until then keep opt-in — correctness is no longer the blocker, only a
    trustworthy latency number is.
- **HEADLINE remaining lever — GPU (Metal/WebGPU) recognizer AR decode.**
  PERFORMANCE.md calls the per-region CPU-bound token loop "the real speed path".
  Substantial project: a persistent single-step decode graph on the GPU (the
  moonshine/OMR persistent-graph pattern in the dev guide — build once, gallocr
  once at max KV, dispatch sched-free per step, re-set all inputs each compute).
  Needs a quiet box + a real CUDA box (Kaggle) for the decoded-roundtrip gate
  before flipping a GPU default (CUDA has stricter per-op contiguity asserts than
  CPU/Metal — LEARNING 35). High value for document-OCR-at-volume.

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

- **CUDA regression — the 4 FAILs are RESOLVED / explained (P100-verified 2026-07-13).**
  A diagnostic kernel (`tools/kaggle/crispembed-cuda-diag`, Tesla P100 / Pascal
  sm_60) diagnosed each under its env gates, then a 2nd run verified the fix:
  - **`layout-heron` — FIXED (`49cb38a`).** The flash→manual attention fallback
    removed the `fattn.cu:602` abort; P100 CUDA now runs `test-layout-diff` to
    **8/8 stages PASS, DIFF PASSED** (dec_0_cross_out 0.977). ✅
  - **`glm-ocr` + `internvl2` — FIXED (`7998f3c`): it was a stdout banner, NOT
    vision garbage.** Both engines printed their load banner (`glm_ocr: loading…
    Vision:… LLM:… KV cache… Ready`) via `printf` → **stdout**, and `run_one`'s
    `--ocr` text-match captures stdout — so `actual` = the banner (cer 4.3/5.4,
    mis-read as "Class-B CUDA vision garbage"). The P100 diagnostic proved both
    OCR the fox **correctly** on CUDA *and* CPU; only the harness saw the banner.
    Routed all banners to stderr to match the passing engines (qwen2vl_ocr, …). ✅
  - **`granite-vision` — text OCR PASSES**; the projector diff drift is
    cross-toolchain FP strictness (identical CUDA=CPU=scalar on P100), threshold
    already 0.95. ✅
  - **Bottom line: NONE of the 4 were real CUDA vision divergences.** It was one
    genuine CUDA bug (layout flash-abort on Pascal) + a stdout-banner harness bug
    (glm/internvl2) + cross-toolchain FP threshold strictness (granite). The
    diagnostic-first approach (test on the box via env gates) was essential — a
    blind "fix the Class-B vision divergence" would have chased a non-existent bug.
  - **RESULT: portfolio 14 → 0 FAIL** across the fix waves (harness `be6ec54`;
    parser `2af57b1`; layout flash→manual `49cb38a`; banner→stderr `7998f3c`;
    parser value-dump/nameless `c26abc4`; layout perm-tolerant `debug/layout-cross`).
    glm-ocr, internvl2, granite all PASS on P100 now. **All original FAILs fixed** —
    every "Class-B" one was a harness/output bug, not CUDA vision divergence.
  - **The last FAIL (`layout-heron` `dec_0_cross_out`) — ROOT-CAUSED + FIXED
    (`debug/layout-cross`).** NOT flaky and NOT an inference bug. The apparent
    "non-determinism" (0.977 v2 vs −0.034 v14 on P100; −0.08/−0.19 on Metal
    manual/flash) is a **query-permutation comparison artifact**. The 300 decoder
    queries are chosen by `partial_sort` over ~8400 near-tie encoder proposals
    (`layout_detect.cpp:1318`); a tiny backend FP delta in enc_output (Metal/CUDA
    vs the CPU/Python reference — max_abs 0.02, cos 0.99999) reshuffles near-tie
    ranks, so "query i" in our output is a *different physical proposal* than the
    reference's "query i". Instrumented proof: the initial queries themselves show
    per-query cos mean 0.78 / 111 below 0.9 (matching cross_out's mean 0.79), the
    top-5 ranks agree, and the cross_out **values are correct** (best-cosine
    matching each ref query → cos_mean 0.999, 299/300 unique = clean bijection).
    Final boxes are unaffected (score-sort + NMS). **Fix:** `test_layout_diff.cpp`
    compares this stage permutation-tolerantly (`perm_tolerant_cos`); now PASS on
    Metal (0.947/0.999), Metal+flash (0.947), CPU (0.967/0.999). Guardrail keeps
    full power — simulated scrambles (feature-shuffle/sign-flip/roll) collapse to
    ≤0.08 vs the 0.85 gate, and s3..enc_output still guard the encoder-scramble
    class strictly at 0.99. Manifest threshold 0.97→0.85 + comment corrected (the
    old "backend-independent" note was wrong).

  Original diagnostic detail (the run that overturned 3 of the 4 assumptions):
  - **`layout-heron` — REAL CUDA bug (fixable).** `test-layout-diff` aborts:
    `ggml/src/ggml-cuda/fattn.cu:602 fatal error` in `ggml_cuda_flash_attn_ext`
    → `GGML_ABORT` because Pascal (sm_60) has **no flash-attention kernel**
    (`get_best_fattn_kernel == BEST_FATTN_KERNEL_NONE`). With
    `LAYOUT_DETECT_FORCE_CPU=1` **all 8 stages PASS (cos 1.0)** — so the graph is
    correct; the engine just runs `flash_attn_ext` on a single CUDA backend that
    bypasses the scheduler's `supports_op` CPU-fallback. **Fix:** don't use the
    CUDA flash kernel where it's unsupported — either (a) route layout attention
    through a scheduler that honours `ggml_cuda_flash_attn_ext_supported` (returns
    false on sm_60 → runs on CPU), or (b) give `layout_detect` a manual masked
    attention fallback (`mul_mat`+`soft_max_ext`+`mul_mat`, mask=nullptr = full
    attn) selected when flash is unsupported. Verify: `test-layout-diff` PASSES on
    P100. NOTE T4 (Turing sm_75) HAS flash — this only bites Pascal.
  - **`granite-vision` — NOT a CUDA bug.** The projector stages fail **identically
    on CUDA, `GRANITE_VIS_SCALAR`, AND full-CPU (`GRANITE_CPU`)** on the P100 box
    (cos 0.952 / 0.958 / 0.955 — same to 2 dp across all three), while they PASS on
    the Mac. So it is a **cross-toolchain FP-strictness gap** (Kaggle gcc vs Mac
    clang on high-magnitude projector activations, max_abs ~2.7–4.3), NOT a CUDA
    divergence, and the **OCR text passes** (cer 0.163). **Fix:** relax the
    projector-stage diff thresholds (≈0.95, they gate a real crater by going
    negative) — a parity-harness strictness fix, not a model change.
  - **`glm-ocr` — NOT a CUDA bug.** `test-glm-ocr-diff` vis_layers 14–23 fail at
    cos 0.96–0.98 **identically on CUDA and CPU** on P100 (vis_layer_23: CUDA
    0.9630 vs CPU 0.9632; max_abs up to 217) — same cross-toolchain strictness as
    granite. And on a clean generated fox image glm reads it **correctly on CUDA**
    (`"The quick brown fox jumps over the lazy dog 12345"`). So glm's vision is not
    CUDA-garbage. Its portfolio FAIL is the **text-match on the repo `fox.png`
    (800×200)** specifically — untested CPU-vs-CUDA yet (see below).
  - **`internvl2` — reads a generated fox CORRECTLY on P100 CUDA** (identical to
    CPU). No ref uploaded, so no per-stage diff. Its portfolio FAIL is likewise the
    text-match on the repo `fox.png` (800×200), not universal vision garbage.
  - **Open sub-question (glm + internvl2 portfolio garbage):** the repo `fox.png`
    is 800×200 (the diagnostic used a 640×96 render). Next diagnostic run must OCR
    the **repo** `tests/regression/images/fox.png` under default vs `*_FORCE_CPU`
    for both engines — if CPU is also garbage there, it's a Kaggle-BUILD issue
    (like granite/glm diff), not CUDA; if only CUDA is garbage, it's a genuine
    larger-image CUDA vision divergence to localize. The vis-diff being CPU=CUDA
    identical strongly suggests the former.
  - **Full data:** the diagnostic log is on Kaggle
    (`chr1s4/crispembed-cuda-diagnostic-4-remaining-fails`, transcript in
    `/kaggle/working/diag.log`); see HISTORY.md.
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
