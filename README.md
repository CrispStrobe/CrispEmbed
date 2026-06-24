# CrispEmbed

[![Build](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml/badge.svg)](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml)

Lightweight embedding inference via ggml. No Python runtime, no ONNX.
Text, image, and face embeddings in one binary.

**Text**: 10 architectures (BERT, XLM-R, MPNet, NomicBERT, ModernBERT, GTE v1.5,
Qwen3, Gemma3, SPLADE, DeBERTa-v2). Dense, sparse (SPLADE/BGE-M3), ColBERT
multi-vector, cross-encoder rerankers, bi-encoder reranking.

**NER**: Zero-shot (GLiNER, DeBERTa-v3/LFM2.5) and fixed-label (BERT/XLM-R) Named
Entity Recognition. Auto-detected from GGUF — same `--ner` API for both.
Fixed-label: `bert-base-ner` (EN), `xlmr-ner-hrl` (10 languages). CLI, server, Python, Dart.

**KIE**: Key Information Extraction — chains OCR + NER to extract structured
key-value fields from document images (receipts, invoices, forms). No new model
needed. CLI (`--kie`), server (`POST /kie/extract`), Python (`CrispKIE`), Dart.

**LID**: Text language identification (CLD3/GlotLID) — auto-selects Tesseract model
by detected language. Server (`POST /lid/detect`), Python (`CrispTextLID`).

**LiLT**: Layout-aware document understanding via dual-stream encoder (RoBERTa +
layout transformer with BiACM). Token classification for form understanding
(FUNSD: question/answer/header). 130M params, MIT. Python (`CrispLiLT`), Dart.

**Vision**: CLIP and SigLIP text-image cross-modal search (encode text and images
into the same vector space). YuNet/SCRFD face detection, ArcFace/SFace/AuraFace
face recognition. Full detect-align-encode pipeline.

**General OCR**: Full text detection + recognition pipeline.
DBNet (ResNet-18 + FPNC, 7 MB Q4_K) detects text regions, TrOCR-small
(DeiT + Transformer decoder, 63 MB Q8_0) recognizes each crop.
~200ms per region. CLI (`--ocr-det`/`--ocr-rec`), server (`POST /ocr`),
Python (`CrispOcrPipeline`), Rust (`OcrPipeline`), Flutter (`CrispOcrPipeline`).

**Scene Text OCR**: PARSeq (ViT encoder + Transformer decoder, 24M params,
94-char ASCII, Apache-2.0). Base 91 MB F32 / 24 MB Q8_0, Tiny 12 MB F16 / 6 MB Q8_0.
State-of-the-art accuracy for scene text recognition (ECCV 2022).
CLI (`--ocr`), server (`POST /ocr/model`), Python (`CrispOcrModel`), auto-detected.

**Math OCR**: Seven engines for math-image → LaTeX:
PP-FormulaNet-L (printed, SAM-ViT+MBart 181M, Apache-2.0, 122 MB Q4_K),
MixTex (CN+EN, Swin-Tiny+RoBERTa 86M, Apache-2.0),
Texo-Distill (printed, HGNetv2+MBart 20M, BLEU 0.90),
DeiT+TrOCR (printed, 17 MB Q4_K),
PosFormer (handwritten, DenseNet+Transformer+ARM, 57% CROHME),
BTTR (handwritten, DenseNet+Transformer, 53% CROHME),
HMER (handwritten, DenseNet+GRU attention).
All auto-detected from GGUF metadata, ~3-5s decoder time.

**Tesseract LSTM OCR**: 12 languages (eng, deu, fra, spa, ita, por, nld, rus, ara,
chi_sim, jpn, kor) in tiny GGUF models (435 KB–1.7 MB Q8_0). Converted from
tessdata_best via custom binary parser. VGSL forward pass (Conv+LSTM stack+CTC).
Emit spaces and punctuation natively. Auto-detected via `--ocr`.

**Text Detection**: Surya EfficientViT segformer (38M, 91 languages, GPU-accelerated).
Heatmap → polygon bounding boxes. Pairs with any OCR recognizer for full-page OCR.
CC-based model-free fallback (zero downloads, CPU-only, 4ms/page).

**Document Preprocessing**: Two tiers — classical (CPU, model-free, instant) and
learned (GPU, model-based, higher quality). Classical: adaptive Otsu binarization,
differential-square-sum deskew, CC despeckle, background normalization, page
dewarping (cubic baseline fitting + disparity warp), 1-bit DWA morphology (21x
faster than float). Learned: NAFNet denoise, TPS spatial transformer (learned
dewarping with 20-point control prediction, 108K params, Apache-2.0), text
super-resolution (NAFNet-SR + TBSRN). All cherry-picked from Leptonica (BSD-2),
reimplemented as self-contained C++ with no dependencies.

**PDF DPI Profiling**: Zero-dependency PDF parser that extracts image metadata
to compute effective page DPI. Auto-selects OCR resolution: downsample high-DPI
scans or trigger super-resolution on low-DPI images. Parses xref tables, page
geometry, image XObjects, and content stream CTMs. CLI (`--pdf-dpi`), server
(`POST /pdf/dpi`), Python, Rust, Dart.

**OCR Output Formats**: Plain text, hOCR (XHTML), ALTO 3.1 (XML), searchable PDF.
Multi-page accumulation, XML escaping, configurable page separators.

**OCR Pipeline Orchestrator**: Source-type routing (screenshot/scan/photo),
per-stage cleanup + engine selection, accept-gate cascading with VLM fallback.
Configurable via C API, CLI, Python, Rust, Dart, HTTP server.

**9.5x faster** than FastEmbed (ONNX) on MiniLM-L6. Python/Rust/Dart APIs.
GPU acceleration (CUDA/Vulkan/Metal). iOS + Android + **WASM** builds.
102 models in registry (text, vision, face, OCR, NER, scan cleanup), 200+ GGUF variants on HF.

**Browser**: Math OCR compiles to WebAssembly (1 MB) via `build-wasm.sh`.
Runs entirely client-side — no server, no API key. GGUF models fetched on
demand and cached in IndexedDB.

**Demo**: [HuggingFace Space](https://huggingface.co/spaces/cstr/CrispEmbed) —
text embeddings (cosine similarity) + math OCR (image → LaTeX), auto-deployed
from `hf-space/`.

### Part of the Crisp ecosystem

| Project | Role |
|---|---|
| **CrispEmbed** | This repo — text embedding engine (ggml), dense + sparse + ColBERT + reranking, plus the per-token `encode_tokens` API used by SimAlign-style word aligners |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | Speech recognition engine (ggml) — 11 ASR backends, same philosophy for audio. Also ships text-to-text NMT (m2m100 / wmt21 / madlad / gemma4-e2b) |
| **[crisp-docx](https://github.com/CrispStrobe/crisp-docx)** | OOXML (`.docx`) surgery + LLM/NMT document translation pipeline. Consumes CrispEmbed for transformer-grade word alignment under the `align` feature, and CrispASR as an offline NMT backend under the `nmt` feature |
| **[CrispSorter](https://github.com/CrispStrobe/CrispSorter)** | Tauri 2 desktop document organiser. Translate tab wires the crisp-docx pipeline; LanceDB indexer uses CrispEmbed for embeddings |
| **[CrisperWeaver](https://github.com/CrispStrobe/CrisperWeaver)** | Flutter transcription app powered by CrispASR — desktop + mobile, fully offline |
| **[Susurrus](https://github.com/CrispStrobe/Susurrus)** | Python ASR GUI with 9 backends (faster-whisper, mlx-whisper, voxtral, ...) |

## Status

**28 embedding models** shown below (all pass, cos>=0.965 vs HF), **78 models** total in registry (including rerankers, SPLADE, CLIP/SigLIP vision, CLIP text, YuNet/SCRFD face detection, AuraFace/SFace recognition):

| Model | Type | Dim | F32 CosSim | Q8_0 | Q4_K |
|-------|------|-----|------------|------|------|
| all-MiniLM-L6-v2 | BERT | 384 | 0.999999 | 0.9995 | 0.97 |
| gte-small | BERT | 384 | 1.000000 | 0.9998 | 0.99 |
| arctic-embed-xs | BERT | 384 | 1.000000 | 0.9999 | 0.99 |
| multilingual-e5-small | XLM-R | 384 | 1.000000 | 0.9999 | 0.99 |
| paraphrase-multilingual-MiniLM-L12-v2 | BERT+SP | 384 | 1.000000 | 0.9999 | 0.99 |
| PIXIE-Rune-v1.0 | XLM-R | 1024 | 0.999993 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | XLM-R | 1024 | 0.999993 | 0.9989 | 0.95 |
| Octen-Embedding-0.6B | Qwen3 | 1024 | 0.999891 | 0.9995 | 0.97 |
| F2LLM-v2-0.6B | Qwen3 | 1024 | 0.999420 | 0.9952 | -- |
| Jina v5 Nano | Qwen3 | 768 | 1.000000 | 0.9987 | 0.96 |
| Jina v5 Small | Qwen3 | 1024 | 0.999899 | 0.9995 | 0.97 |
| EmbeddingGemma-300m | Gemma3 | 768 | 1.000000 | 0.9998 | 0.98 |
| Harrier-OSS-v1-0.6B | Qwen3 | 1024 | 0.999959 | 0.9999 | 0.99 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 | 0.9996 | 0.97 |
| Harrier-OSS-v1-270M | Gemma3 | 640 | 0.999948 | 0.9998 | 0.99 |
| all-mpnet-base-v2 | MPNet | 768 | 0.9874 | 0.9998 | 0.99 |
| gte-modernbert-base | ModernBERT | 768 | 0.999991 | 0.9999 | -- |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 1.000000 | 0.9980 | -- |
| nomic-embed-text-v2-moe | NomicBERT MoE | 768 | 1.000000 | 0.9996 | 0.966 |
| multilingual-e5-base | XLM-R | 768 | 0.999995 | 0.9999 | 0.99 |
| multilingual-e5-large | XLM-R | 1024 | 0.999997 | 0.9999 | 0.99 |
| granite-embedding-278m | XLM-R | 768 | 0.999984 | 0.9999 | 0.99 |
| granite-embedding-107m | XLM-R | 384 | 0.999986 | 0.9999 | 0.99 |
| bge-small-en-v1.5 | BERT | 384 | 0.999999 | 0.9999 | 0.99 |
| bge-base-en-v1.5 | BERT | 768 | 0.999994 | 0.9999 | 0.99 |
| bge-large-en-v1.5 | BERT | 1024 | 0.999992 | 0.9999 | 0.99 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000032 | 0.9999 | 0.99 |
| Qwen3-Embedding-4B | Qwen3 | 2560 | — | — | 0.974 |
| Octen-Embedding-8B | Qwen3 | 4096 | — | — | 0.965 |

Q8_0 = all PASS (cos >= 0.995). Q4_K = most PASS; `--` = SwiGLU/GeGLU too sensitive for aggressive quants.
Q4_K cosines for the 4B/8B decoder rows are min-cos vs **bf16** HF (the native training precision; full f32 would not fit in 16 GB RAM).

### Known issues

- **NomicBERT quantization**: SwiGLU gate/value projections are sensitive to
  aggressive quantization. Q5_K (cos~0.95) and Q4_K (cos~0.85) degrade
  significantly. **Use F32 or Q8_0 only** for this model.
- **Jina v5 LoRA adapters**: Jina v5 models use task-specific LoRA adapters.
  GGUFs have the `retrieval` adapter merged. Other tasks (text-matching,
  clustering, classification) require separate GGUFs.
**Performance** (Apple M1, Metal):

| Engine | Single text | Batch (10) |
|--------|------------|------------|
| **CrispEmbed Python** (ctypes) | **3.6 ms** / 280 t/s | **12.7 ms** / **787 t/s** |
| fastembed-rs (Rust ONNX) | 3.8 ms / 263 t/s | 18.9 ms / 528 t/s |
| HuggingFace (PyTorch) | 12.2 ms / 82 t/s | 29.8 ms / 335 t/s |
| CrispEmbed Server (HTTP) | 21.3 ms / 46 t/s | 32.9 ms / 303 t/s |

Model: all-MiniLM-L6-v2. See [PERFORMANCE.md](PERFORMANCE.md) for full multi-model benchmarks.

**Ollama-compatible**: All 45 registry models export as Ollama-compatible GGUFs by default (`--ollama` converter flag). 13 models verified end-to-end in our [Ollama fork](https://github.com/CrispStrobe/ollama/tree/feat/xlmr-embedding) (adds XLM-R, Viterbi SentencePiece tokenizer, GELU_ERF, multi-tokenizer BERT support). See [PERFORMANCE.md](PERFORMANCE.md) for Ollama Q8_0/Q4_K cosine results.

**BidirLM-Omni Q4_K verified locally**: text, audio, and raw vision-patch embedding all load and emit 2048-d vectors from `bidirlm-omni-2.5b-q4_k.gguf`. Current CPU smoke benchmark on Apple M1:
text batch of 4 = 373.2 ms, JFK audio = 618.7 ms, synthetic 2x2 vision patches = 11.0 ms. Regression gate used for graph changes: cosine >= 0.99999 and max abs diff <= 1e-3 against saved baseline vectors.

## Quick start

```bash
# Clone with submodule
git clone --recursive https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed

# Build (CPU)
cmake -S . -B build
cmake --build build -j

# Encode text
./build/crispembed -m model.gguf "Hello world"

# Matryoshka truncation (e.g. 128 dims from a 384-dim model)
./build/crispembed -m model.gguf -d 128 "Hello world"

# Prefix + capability inspection
./build/crispembed -m model.gguf --prefix "query: " --capabilities

# BidirLM-Omni text/audio/raw-vision patch embedding
# Put local paths in a gitignored `.env.local`, then source it here.
source .env.local
./build/crispembed -m bidirlm-omni-2.5b "Hello world"
./build/crispembed -m bidirlm-omni-2.5b --audio "$CRISPEMBED_BIDIRLM_AUDIO"
./build/crispembed -m bidirlm-omni-2.5b \
    --image-raw "$CRISPEMBED_BIDIRLM_PATCHES" --grid-thw 1,14,14

# Sparse / ColBERT retrieval (BGE-M3)
./build/crispembed -m bge-m3.gguf --sparse "Hello world"
./build/crispembed -m bge-m3.gguf --colbert "Hello world"

# Cross-encoder and bi-encoder reranking
./build/crispembed -m bge-reranker-v2-m3.gguf --rerank "capital of france" \
    "Paris is the capital of France." "Bicycles have two wheels."
./build/crispembed -m model.gguf --biencoder "capital of france" --top-n 2 \
    "Paris is the capital of France." "Berlin is the capital of Germany."

# CLIP text-image search (cross-modal)
./build/crispembed -m clip-text-base "a photo of a cat"
./build/crispembed -m clip-vit-base-patch16 --image photo.jpg

# Face detection (YuNet: 0.2 MB, or SCRFD: 16 MB)
./build/crispembed -m yunet --detect photo.jpg --json

# CLI parity test
python tests/test_cli_parity.py --cli ./build/crispembed \
    --dense-model "$CRISPEMBED_DENSE_MODEL" \
    --retrieval-model "$CRISPEMBED_RETRIEVAL_MODEL" \
    --reranker-model "$CRISPEMBED_RERANKER_MODEL"

# Start server (text + vision + face + CLIP + OCR + NER)
./build/crispembed-server -m model.gguf \
    --vit clip-vit-base-patch16.gguf \
    --clip-text clip-text-base.gguf \
    --det yunet.gguf \
    --ocr ppformulanet-l-q8_0.gguf \
    --ner gliner-lfm-f32.gguf --port 8080
curl -X POST http://localhost:8080/embed -d '{"texts": ["Hello world"]}'
curl -X POST http://localhost:8080/clip/text -d '{"text": "a photo of a cat"}'
curl -X POST http://localhost:8080/vit/encode -d '{"image": "photo.jpg"}'
curl -X POST http://localhost:8080/ocr/model -d '{"image": "formula.png"}'
curl -X POST http://localhost:8080/ner/extract \
  -d '{"text": "Tim Cook at Apple", "labels": ["person", "organization"]}'
```

## OCR

More than a dozen engines for image → text, most auto-detected from GGUF metadata via
the unified `crispembed_ocr_model_*` C API (formerly `crispembed_math_ocr_*`,
kept as deprecated aliases). Available through CLI (`--ocr`), HTTP server
(`POST /ocr/model`, alias `POST /math/ocr`), Python (`CrispOcrModel`), Rust
(`OcrModel`), and Dart/Flutter (`CrispOcrModel`).

| Model | Architecture | Params | Q4_K Size | Use case | License |
|-------|-------------|--------|-----------|----------|---------|
| **BTTR** | DenseNet + Transformer | 6.5M | — | Handwritten math | MIT |
| **DeepSeek-OCR-2** | SAM ViT-B + Qwen2 enc + DeepSeek-V2 MoE | 3B | 2.2 GB | Document OCR (runs on Metal + q4_k) | Apache-2.0 |
| **DeiT+TrOCR** | DeiT-S + TrOCR | 65M | — | Printed math | Apache-2.0 |
| **GLM-OCR** | CogViT + GLM-0.5B | 0.9B | 849 MB | Document OCR (OmniDocBench #1, 8 langs) | MIT |
| **GOT-OCR2** | SAM ViT-B + Qwen2-0.5B | 0.7B | 422 MB | Document OCR (text+LaTeX+tables) | Apache-2.0 |
| **HMER** | DenseNet + GRU attention | 6M | — | Handwritten math | MIT |
| **InternVL2-1B** | InternViT-300M + Qwen2-0.5B | 0.9B | 724 MB | Edge/WASM VLM OCR (OCRBench 779) | MIT |
| **InternVL2.5-2B** | InternViT-300M + InternLM2.5-1.8B | 2.1B | 1.4 GB | EN+DE VLM OCR (OCRBench ~830) | MIT |
| **MixTeX** | Swin-Tiny + RoBERTa | 86M | — | Chinese+English LaTeX | Apache-2.0 |
| **PosFormer** | DenseNet + Transformer+ARM | 6.5M | 10 MB | Handwritten math (60.5%) | [ours](https://huggingface.co/cstr/posformer-crohme-GGUF): CC-BY-NC, orig: Academic |
| **Qari-OCR** | Qwen2-VL-2B + LoRA | 2B | ~1.3 GB | Arabic OCR with diacritics | Apache-2.0 |
| **PP-FormulaNet-L** | SAM-ViT + MBart | 181M | 100 MB | Printed math (best) | Apache-2.0 |
| **Qwen2.5-VL-3B** | 32L ViT + 36L Qwen2.5 LLM | 3.6B | 2.6 GB | German/multilingual VLM OCR | Apache-2.0 |
| **Qwen3-VL-2B** | 24L ViT + DeepStack + 28L Qwen3 (IMROPE) | 2.4B | 1.5 GB | General VLM OCR | Apache-2.0 |
| **Texo-Distill** | HGNetv2 + MBart | 20M | 14 MB | Printed math (small) | AGPL-3.0 |
| **Unlimited-OCR** | SAM ViT-B + CLIP-L/14 + DeepSeek-V2 MoE | 3.3B | 2.0 GB | Full-page document OCR with layout grounding | MIT |

**PP-FormulaNet-L** (recommended for printed math): SAM-ViT encoder with
windowed + global attention and decomposed relative position bias, full ggml
graph compute. Encoder parity cos=0.999962 vs HuggingFace reference.

```bash
# CLI
./build/crispembed -m ppformulanet-l --ocr formula.png

# Server
./build/crispembed-server --ocr ppformulanet-l-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/ocr/model -d '{"image": "formula.png"}'

# Python
from crispembed import CrispOcrModel
ocr = CrispOcrModel("ppformulanet-l-q8_0.gguf")
latex = ocr.recognize("formula.png")

# C API
void *ctx = crispembed_ocr_model_init("ppformulanet-l-q8_0.gguf", 4);
const char *latex = crispembed_ocr_model_recognize(ctx, pixels, w, h, ch, &len);
```

**Qwen3-VL-2B** (general VLM OCR): Smallest Qwen3-VL model with DeepStack
multi-scale vision features and interleaved mRoPE. Full parity cos≥0.999.

```bash
# Auto-downloads q4_k from HuggingFace
./build/crispembed -m qwen3vl-2b --ocr document.png

# Faster on CPU: reduce resolution (1.6x speedup, minor quality loss)
CRISPEMBED_MAX_PIXELS=65536 ./build/crispembed -m qwen3vl-2b --ocr document.png
```

`CRISPEMBED_MAX_PIXELS` works with all variable-resolution VLM engines
(Qwen2.5-VL, Qwen3-VL, InternVL2, LightOnOCR).

**Unlimited-OCR** (full-page document OCR with layout grounding): a dual vision
encoder (SAM ViT-B → CLIP-L/14, "DeepLIP") feeding a DeepSeek-V2 MoE decoder
(64 routed experts top-6 + 2 shared). It emits grounded regions —
`<|det|>x1 y1 x2 y2<|/det|>text` per line. The model's own prompt is
`<image>document parsing.` and it requires a sliding-window `no_repeat_ngram`
(size 35, window 128); both are applied automatically. Run it through the OCR
orchestrator:

```bash
# Auto-downloads q4_k from HuggingFace (cstr/unlimited-ocr-crispembed-GGUF)
./build/crispembed --ocr-pipeline document.png \
    --ocr-engine unlimited_ocr --ocr-rec unlimited-ocr

# Tunables (defaults match the model card): UOCR_NO_REPEAT_NGRAM, UOCR_NGRAM_WINDOW
```

VLM engines ingest the original image and letterbox internally, so the pipeline
skips scan-cleanup (deskew/crop/whiten) for them — cleaning the input distorts
the page and shifts content out of the vision grid.

**Flutter integration**: The `flutter/crispembed/` plugin provides `CrispEmbedOcr`
for Dart FFI access. Used by [CrispCalc](https://github.com/CrispStrobe/CrispCalc)
for camera-based math input. Platform dirs (Linux/Windows/macOS/iOS/Android) with
CI-built native libraries.

## Layout Detection

Document layout analysis via RT-DETRv2 (ResNet-50 + HybridEncoder + deformable
cross-attention decoder). Detects 17 region types: text, title, table, figure,
formula, caption, section_header, list_item, footnote, page_header, page_footer,
code, document_index, checkbox_selected, checkbox_unselected, form, key_value_region.

Auto-detected from GGUF metadata (`general.architecture = layout`). Available
through CLI (`--layout`), HTTP server (`POST /layout/detect`), Python (`CrispLayout`),
Rust (`CrispLayout`), and Dart/Flutter.

```bash
# CLI
./build/crispembed -m layout-heron --layout document.png --json

# Server
./build/crispembed-server --layout layout-heron-f32.gguf --port 8080
curl -X POST http://localhost:8080/layout/detect -d '{"image": "page.png"}'

# Python
from crispembed import CrispLayout
layout = CrispLayout("layout-heron-f32.gguf")
regions = layout.detect("page.png")
for r in regions:
    print(f"{r['label']} ({r['score']:.2f})")
```

Encoder parity: all stages cos=1.0 vs HF reference. Detection score 0.93 (HF: 0.95).
Performance: 21s with BLAS (F32), Q8_0 model 43 MB.
Source: [docling-project/docling-layout-heron](https://huggingface.co/docling-project/docling-layout-heron) (Apache-2.0).
Models: [`cstr/layout-heron-gguf`](https://huggingface.co/cstr/layout-heron-gguf)

## Text Detection

Text line detection via surya-ocr-2's EfficientViT-Large segformer (38M params,
91 languages). Segmentation-based: input 1200×1200 → 2-channel heatmap → polygon
bounding boxes. GPU-accelerated via `ggml_backend_init_best()` (Metal/CUDA).

```bash
# CLI
./build/crispembed -m surya-det --text-detect document.png --json

# Server
./build/crispembed-server --text-det surya-det-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/text/detect -d '{"image": "page.png"}'

# Python
from crispembed import CrispTextDetect
det = CrispTextDetect("surya-det-q8_0.gguf")
regions = det.detect("page.png")
```

Heatmap parity: exact match vs Python reference. ggml graph acceleration (35x).
Source: [datalab-to/surya-ocr-2](https://github.com/VikParuchuri/surya) (OpenRail-M, free <$5M).
Models: [`cstr/surya-det-GGUF`](https://huggingface.co/cstr/surya-det-GGUF) — F32 (147 MB), F16 (73 MB), Q8_0 (41 MB), Q4_K (30 MB).

## Scan Cleanup

Document scan preprocessing — pure C++, no external tool dependencies.
Two tiers: classical image processing (no model) and learned denoising
via NAFNet CNN (GGUF model, MIT license).

**Tier 1 (classical, no model):** deskew (Hough transform), Otsu/Sauvola
binarization, border crop, background whitening (morphological open).

**Tier 2 (learned, NAFNet):** NAFNet-SIDD-width32 U-Net denoising CNN
(29M params, 30 MB Q8_0). Pre-trained on SIDD smartphone image denoising.

```bash
# CLI — classical only
./build/crispembed --cleanup-only scan.png

# CLI — with NAFNet denoising before OCR
./build/crispembed --cleanup -m ocr_model.gguf --ocr scan.png

# Server (always available, no model needed for tier 1)
curl -X POST http://localhost:8080/scan/cleanup -d '{"image": "scan.png"}'

# Python
from crispembed import CrispScanCleanup
cleanup = CrispScanCleanup()                          # tier 1 only
cleanup = CrispScanCleanup("nafnet-sidd-w32-q8_0.gguf")  # tier 1 + 2
cleaned = cleanup.process("scan.png")                 # numpy RGB array
```

## Text Super-Resolution

Upscale low-resolution text images before OCR — two production engines plus an
NAFNet-SR scaffolding for custom trained models. Both models are Apache-2.0 and
auto-downloaded via the model registry.

| Model | Architecture | Params | Size | Use case | License |
|-------|-------------|--------|------|----------|---------|
| **PAN** (`pan-sr-x4`) | SC-PA blocks + PixelShuffle(4) | 272K | 0.5 MB | Full-page 4× upscale | Apache-2.0 |
| **TBSRN** (`tbsrn-sr`) | TSA residual groups + PixelShuffle(2) | 1.1M | 2 MB | Per-line 2× upscale | Apache-2.0 |
| **NAFNet-SR** (`text-sr`) | NAFNet U-Net + configurable upsample tail | custom | — | Custom trained model | Apache-2.0 |

**PAN** (Pixel Attention Network): whole-page 4× super-resolution via
depthwise-separable convolutions + pixel attention gates. 0.5 MB GGUF.
Parity cos=0.999654 vs PyTorch reference.

**TBSRN** (Text Before Super-Resolution Network): per-line 2× super-resolution
with transformer-style spatial token attention (telescope training scheme).
2 MB GGUF. Parity cos=0.999985 vs PyTorch reference.

```bash
# CLI — PAN 4× whole-page super-resolution
./build/crispembed --pan-sr document.png --output upscaled.png

# CLI — TBSRN 2× per-line super-resolution (text-line crop)
./build/crispembed --tbsrn-sr line_crop.png --output upscaled_line.png

# CLI — NAFNet-SR with custom trained model
./build/crispembed --sr-model my-nafnet-sr.gguf input.png --output upscaled.png

# Server
./build/crispembed-server --pan-sr pan-x4-f16.gguf \
    --tbsrn-sr tbsrn-telescope-f16.gguf --port 8080
curl -X POST http://localhost:8080/pan/sr -d '{"image": "scan.png"}'
curl -X POST http://localhost:8080/tbsrn/sr -d '{"image": "line_crop.png"}'

# Python — PAN
from crispembed import CrispPanSr
sr = CrispPanSr("pan-x4-f16.gguf")
upscaled = sr.upscale("scan.png")          # PIL Image or numpy array in → numpy out
upscaled = sr.upscale(pil_image)
upscaled = sr.upscale(numpy_uint8_array)   # (H, W, 3) uint8 → (4H, 4W, 3) uint8

# Python — TBSRN
from crispembed import CrispTbsrnSr
sr = CrispTbsrnSr("tbsrn-telescope-f16.gguf")
upscaled = sr.upscale("line_crop.png")     # (H, W, 3) uint8 → (2H, 2W, 3) uint8
```

Models: [`cstr/crispembed-gguf`](https://huggingface.co/cstr/crispembed-gguf) —
`pan-x4-f16.gguf` (0.5 MB), `tbsrn-telescope-f16.gguf` (2 MB).

## Named Entity Recognition

Zero-shot NER via GLiNER with an LFM2.5-350M bidirectional backbone (16 layers:
10 ShortConv + 6 GQA attention, SwiGLU FFN). Detects arbitrary entity types
specified at inference time — no retraining needed. Ported from CrispASR's LFM2
backbone with bidirectional attention and symmetric convolutions.

Architecture: BPE tokenize → LFM2.5-bi backbone → layer fusion (squeeze-and-excitation)
→ BiLSTM → GLiNER span-label matching head (SpanMarkerV1 + dot-product scorer).

```bash
# CLI
./build/crispembed -m gliner-lfm-f32.gguf --ner "Maria Schmidt arbeitet bei Siemens in München"

# Server
./build/crispembed-server --ner gliner-lfm-f32.gguf --port 8080
curl -X POST http://localhost:8080/ner/extract \
  -d '{"text": "Maria Schmidt arbeitet bei Siemens", "labels": ["person", "organization"], "threshold": 0.5}'

# Python
from crispembed import CrispNER
ner = CrispNER("gliner-lfm-f32.gguf")
entities = ner.extract("Maria Schmidt arbeitet bei Siemens in München",
                       labels=["person", "organization", "location"])
for e in entities:
    print(f"{e['text']} => {e['label']} ({e['score']:.2f})")
```

Parity: all 16 backbone layers cos=1.000000 vs HF reference. Layer fusion and
BiLSTM cos=1.000000. 17/17 entities match across 5 test texts.
Source: [VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER](https://huggingface.co/VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER) (LFM Open License v1.0).

### BERT / XLM-R Fixed-Label NER

Fixed-label token classification NER using existing BERT/XLM-R encoders with a
Linear classifier head. Auto-detected from GGUF metadata (`ner.classifier.weight`).

| Model | Languages | Labels | Params | License |
|-------|-----------|--------|--------|---------|
| `bert-base-ner` | English | 9 (PER/LOC/ORG/MISC) | 110M | MIT |
| `xlmr-ner-hrl` | 10 (en/de/fr/es/pt/it/nl/ar/zh/hi) | 9 (PER/LOC/ORG/MISC/DATE) | 278M | MIT |

```bash
./build/crispembed -m bert-base-ner --ner "Apple CEO Tim Cook visited Cupertino"
# Apple (ORG), Tim Cook (PER), Cupertino (LOC)
```

Same `--ner` flag and `crispembed_ner_*` API as GLiNER — backend auto-detected from GGUF.

## Language Identification (LID)

Text-based language identification via shared `crisp_lid` library from CrispASR.
Two backends auto-detected from GGUF: CLD3 (109 languages, Apache-2.0) and
GlotLID (2102 ISO 639-3, Apache-2.0).

Integrated into the OCR orchestrator for automatic Tesseract model selection:
set `--lid-model cld3` + `--tess-model-dir /path/to/models/` and the pipeline
detects the document language then picks `tesseract-{lang}-q8_0.gguf`.

```bash
# Server
curl -X POST http://localhost:8080/lid/detect \
  -d '{"text": "Hallo Welt, wie geht es Ihnen?"}'
# → {"lang": "de", "confidence": 0.99}

# Python
from crispembed import CrispTextLID
lid = CrispTextLID("cld3-f16.gguf")
lang, conf = lid.predict("Bonjour le monde")  # ("fr", 0.98)

# OCR pipeline with LID
./build/crispembed --ocr-pipeline doc.png --lid-model cld3-f16.gguf --json
# → {"n_regions": 5, ..., "detected_lang": "de", "lang_confidence": 0.97, ...}
```

## Key Information Extraction (KIE)

Chains the OCR pipeline (text detection + recognition) with GLiNER zero-shot NER
to extract structured key-value fields from document images — receipts, invoices,
forms, business cards. No new model needed: uses existing OCR + NER models.

```bash
# CLI — extract fields from a receipt image
./build/crispembed -m gliner-lfm-f32.gguf \
    --ocr-det dbnet-det-f16.gguf --ocr-rec trocr-printed-q8_0.gguf \
    --kie receipt.png --kie-labels "total,date,vendor" --json

# Server — auto-enabled when NER + OCR det/rec are loaded
./build/crispembed-server --ner gliner-lfm-f32.gguf \
    --ocr-det dbnet-det-f16.gguf --ocr-rec trocr-printed-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/kie/extract \
  -d '{"image": "receipt.png", "labels": ["total", "date", "vendor"], "threshold": 0.5}'

# Python
from crispembed import CrispKIE
kie = CrispKIE("dbnet-det-f16.gguf", "trocr-printed-q8_0.gguf", "gliner-lfm-f32.gguf")
result = kie.extract("receipt.png", labels=["total", "date", "vendor"])
for f in result["fields"]:
    print(f"{f['label']} = {f['value']} (score={f['score']:.2f}, bbox={f['bbox']})")
```

Output (JSON mode):
```json
{"n_ocr_regions": 12, "ocr_confidence": 0.85, "fields": [
  {"label": "total", "value": "$42.50", "score": 0.92, "bbox": [120.0, 340.0, 200.0, 30.0]},
  {"label": "date", "value": "2026-06-15", "score": 0.88, "bbox": [50.0, 20.0, 150.0, 25.0]}
]}
```

### LiLT — Layout-Aware Document Understanding (KIE Phase 2)

LiLT (Language-independent Layout Transformer) is a dual-stream encoder
that combines RoBERTa (768d text) with a parallel layout transformer (192d)
via BiACM (bidirectional attention complementation). It takes OCR text +
bounding boxes as input and performs token classification — identifying
questions, answers, and headers in forms and documents.

Architecture: 130.7M params, 12 layers, 12 heads. MIT license
([SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base)).

```bash
# Python — direct LiLT token classification
from crispembed import CrispLiLT
lilt = CrispLiLT("lilt-funsd-q8_0.gguf")
tokens = lilt.classify(
    input_ids=[0, 10566, 35, 291, 5480, 35, 68, 3818, 4, 2466, 2],
    bbox=[[0,0,0,0], [10,50,90,80], [90,50,110,80], [120,50,200,80],
          [250,50,330,80], [330,50,350,80], [360,50,390,80],
          [390,50,430,80], [430,50,440,80], [440,50,470,80], [0,0,0,0]],
)
for t in tokens:
    print(f"{t['label']:15s} score={t['score']:.2f}")
# B-QUESTION      score=1.00   (Date)
# I-QUESTION      score=0.95   (:)
# B-ANSWER        score=0.72   (2026...)
# B-QUESTION      score=1.00   (Total)
# I-QUESTION      score=1.00   (:)
# B-ANSWER        score=0.56   ($)
# I-ANSWER        score=0.95   (48.60)
```

Parity: 25/25 layers cos=1.000000 vs HF reference. 16/16 token labels match.

| Variant | Size | Format |
|---------|------|--------|
| `lilt-funsd` | 498 MB / 134 MB / 90 MB | F32 / Q8_0 / Q4_K |
| `lilt-base` | 498 MB / 134 MB / 90 MB | F32 / Q8_0 / Q4_K |

FUNSD labels: O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER.
See [docs/kie.md](docs/kie.md) for architecture details and the BiACM mechanism.

## Model licenses

The auto-download registry (`-m <name>`) covers models under multiple
licenses. CrispEmbed itself is permissive, but the model you download is
governed by its **upstream** license — converting a checkpoint to GGUF
does not relicense it. Always check `--list-models` (the **License**
column) or the upstream model card before using a model commercially.

| License class | Models in registry | What you can do |
|---|---|---|
| **Permissive** (Apache-2.0 / MIT) | most BERT/XLM-R/MPNet, BGE, E5, Granite, Snowflake, MXBai, Nomic, MS-Marco, Qwen3, Harrier, BidirLM-Omni, GTE-v1.5, `gliner-deberta` (NER), `lilt-funsd`, `lilt-base` (KIE) | commercial use OK with normal attribution |
| **CC BY-NC 4.0** (non-commercial) | `jina-v5-nano`, `jina-v5-small`, `jina-reranker-v2-base-multilingual` | research/evaluation only; commercial use requires a paid license from Jina (sales@jina.ai) |
| **LFM Open License v1.0** | `lfm2-embed`, `lfm2-embed-q4k`, `lfm2-colbert`, `gliner-lfm` (NER) | free under $10M annual revenue; above that requires commercial license from Liquid AI |
| **Gemma Terms of Use** | `embeddinggemma-300m` | commercial use permitted **subject to** Google's [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) |

Restricted-license entries (NC + Gemma/vendor terms such as LFM) are marked with `*` in
`--list-models`. Auto-download for them requires explicit consent:

```bash
# Interactive (TTY): you'll be shown the license + model card URL and
# prompted to accept.
./build/crispembed -m jina-v5-nano "hello"

# Non-interactive (CI, scripts): pass the SPDX tag.
./build/crispembed -m jina-v5-nano --accept-license cc-by-nc-4.0 "hello"

# Or via env var (useful for shared scripts).
export CRISPEMBED_ACCEPT_LICENSE=cc-by-nc-4.0
./build/crispembed -m jina-v5-nano "hello"

# Special value "all" accepts every license tag — intended for trusted
# environments where you've already vetted the registry.
export CRISPEMBED_ACCEPT_LICENSE=all
```

`--accept-license` is an affirmative acknowledgement that the **caller**
accepts the upstream terms; it does not grant rights you don't otherwise
have. For commercial use of a CC BY-NC model you still need a separate
license from the model author.

To audit what the registry actually carries (cross-checks the upstream
license, the `cstr/*-GGUF` re-host's declared license, and the claim in
`models/upload_to_hf.py`):

```bash
python tests/check_registry_licenses.py            # human-readable table
python tests/check_registry_licenses.py --json     # for CI
```

## Building

### Linux / macOS

```bash
# CPU only (default)
cmake -S . -B build && cmake --build build -j

# With OpenBLAS acceleration
cmake -S . -B build -DGGML_BLAS=ON && cmake --build build -j

# With Intel MKL
cmake -S . -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp

# With CUDA (NVIDIA GPU)
cmake -S . -B build -DGGML_CUDA=ON && cmake --build build -j

# With Vulkan (cross-platform GPU)
cmake -S . -B build -DGGML_VULKAN=ON && cmake --build build -j

# macOS with Metal (recommended)
./build-macos.sh              # Metal + Accelerate + embedded shaders
./build-macos.sh --cpu        # CPU only, no Metal
./build-macos.sh --shared     # Also build shared lib for Python
```

### Windows

Requires Visual Studio 2022 Build Tools + Ninja.

```batch
:: CPU build
build-windows.bat

:: Vulkan GPU build (needs Vulkan SDK)
build-vulkan.bat

:: CUDA GPU build (needs CUDA Toolkit)
build-cuda.bat
```

If you get "ggml does not contain a CMakeLists.txt", run:
```
git submodule update --init --recursive
```

### Dependencies

- **Required**: C++17 compiler, CMake 3.14+
- **Optional**: OpenBLAS (`apt install libopenblas-dev`), Intel MKL, CUDA Toolkit, Vulkan SDK

### Installing as a system library

`cmake --install build --prefix /usr/local` (or any other prefix) lays out
a standard distro tree:

```
<prefix>/
  bin/{crispembed, crispembed-server, crispembed-quantize}
  lib/
    libcrispembed.so.0.3.0        (real file)
    libcrispembed.so.0            (SONAME symlink — SOVERSION 0)
    libcrispembed.so              (linker symlink)
    libggml*.so*                  (ggml backend siblings)
    cmake/crispembed/             (find_package(crispembed) plumbing)
    pkgconfig/crispembed.pc       (pkg-config --cflags --libs crispembed)
  include/
    crispembed.h
    ggml*.h
```

The installed `.so`/`.dylib` carries `RPATH=$ORIGIN` (Linux) / `@loader_path`
(macOS) so it finds its `libggml*` siblings without `LD_LIBRARY_PATH`. The
installed binaries carry `RPATH=$ORIGIN/../lib` / `@loader_path/../lib`.

Downstream CMake consumers:

```cmake
find_package(crispembed REQUIRED)
target_link_libraries(my_app PRIVATE crispembed::crispembed)
```

Downstream pkg-config consumers:

```sh
$ pkg-config --cflags --libs crispembed
-I/usr/local/include -L/usr/local/lib -lcrispembed
```

The pkg-config file is **relocatable** (`prefix=${pcfiledir}/../..`), so
extracting a release tarball into `/opt/foo` and pointing
`PKG_CONFIG_PATH=/opt/foo/lib/pkgconfig` Just Works without editing the
`.pc` file.

## Converting models

```bash
# BERT / XLM-R encoder models
pip install torch transformers gguf
python models/convert-bert-to-gguf.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2.gguf

# Qwen3 / Gemma3 decoder models
python models/convert-decoder-embed-to-gguf.py \
    --model Octen/Octen-Embedding-0.6B \
    --output octen-0.6b.gguf

# Quantize (Q8_0 recommended, Q4_K for max compression)
./build/crispembed-quantize model.gguf model-q8_0.gguf q8_0
./build/crispembed-quantize model.gguf model-q4_k.gguf q4_k
```

Pre-converted models: [HuggingFace cstr/](https://huggingface.co/cstr)

## Quantization

| Type | Compression | Quality (cos vs F32) | Notes |
|------|-------------|---------------------|-------|
| Q8_0 | ~3.8x | >0.995 | Recommended default |
| Q5_K | ~5x | >0.98 | Good balance |
| Q4_K | ~5.5x | >0.95 | Max compression |
| Q6_K | ~4.5x | >0.99 | Premium quality |

Embedding tables quantized to Q8_0 even in Q4_K mode (quality-sensitive).

## BGE-M3 / Sparse / ColBERT / Reranker

CrispEmbed supports all three BGE-M3 retrieval modalities, LFM2.5-ColBERT multi-vector retrieval, plus cross-encoder rerankers.

```bash
# Convert BGE-M3 (writes sparse_linear.weight + colbert_linear.weight into GGUF)
pip install torch transformers gguf FlagEmbedding
python models/convert-bert-to-gguf.py --model BAAI/bge-m3 --output bge-m3.gguf --crisp

# Validate all three heads against FlagEmbedding ground truth
python tests/test_bgem3.py --gguf bge-m3.gguf --lib build/libcrispembed.so
```

```python
from crispembed import CrispEmbed

model = CrispEmbed("bge-m3.gguf")

# Dense (L2-normalised)
vec = model.encode("Hello world")                   # Vec<f32> len 1024

# Sparse (SPLADE-style term weights)
if model.has_sparse():
    sparse = model.encode_sparse("Hello world")     # {token_id: weight}

# ColBERT multi-vector
if model.has_colbert():
    multi = model.encode_multivec("Hello world")    # [[f32; 128]; n_tokens]
```

**LFM2.5-ColBERT** (128d per-token, LFM Open License v1.0):

```python
model = CrispEmbed("lfm2-colbert-q8_0.gguf")       # 361 MB, cos=0.998
q = model.encode_multivec("query: machine learning")
d = model.encode_multivec("document: deep learning is a subset of ML")
import numpy as np
score = (q @ d.T).max(axis=1).sum()                 # MaxSim late interaction
```

| Quant | Size | ColBERT cos vs fp16 |
|-------|------|---------------------|
| F32 | 677 MB | 0.999995 |
| Q8_0 | 361 MB | 0.998 |
| Q5_K | 258 MB | 0.977 |
| Q4_K | 224 MB | 0.959 |

Cross-encoder rerankers:

```python
reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
score = reranker.rerank("query text", "document text")   # raw logit
```

## Python

Requires the shared library (`--shared` flag or `-DCRISPEMBED_BUILD_SHARED=ON`).

```python
from crispembed import CrispEmbed

model = CrispEmbed("all-MiniLM-L6-v2.gguf")

# Single text
vec = model.encode("Hello world")      # shape (384,)

# Batch — single C call, true batched Metal/GPU inference
vectors = model.encode(["Hello world", "Goodbye world"])
print(vectors.shape)  # (2, 384)

# Matryoshka dimension truncation
model.set_dim(128)
vec128 = model.encode("Hello world")   # shape (128,)

# Prompt prefix (for models that need it)
model.set_prefix("query: ")           # auto-prepended before tokenization

# Sparse (BGE-M3)
model = CrispEmbed("bge-m3.gguf")
if model.has_sparse:
    sparse = model.encode_sparse("Hello world")   # {token_id: weight}

# ColBERT multi-vector
if model.has_colbert:
    multi = model.encode_multivec("Hello world")   # (n_tokens, 128)

# Cross-encoder reranking
reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
score = reranker.rerank("query", "document")       # raw logit

# Bi-encoder reranking (any embedding model, cosine similarity)
results = model.rerank_biencoder("query", ["doc1", "doc2", "doc3"], top_n=2)
for r in results:
    print(f"  [{r['index']}] {r['score']:.4f}: {r['document']}")

# BidirLM-Omni: text, audio, image, and image-conditioned text in one shared 2048-d space
omni = CrispEmbed("bidirlm-omni-2.5b")
text_vec  = omni.encode("a small cat on a chair")
if omni.has_audio:
    audio_vec = omni.encode_audio(pcm_f32, sr=16000)             # 1-D float32 PCM
if omni.has_vision:
    # Two preprocessing paths:
    #   - encode_image(...)      uses HF Qwen2VL processor (tight parity with HF, requires `transformers`)
    #   - encode_image_file(...) uses the in-process C++ preprocessor (no transformers dep, ~0.97 cos vs HF)
    img_vec   = omni.encode_image("cat.jpg")
    img_vec_native = omni.encode_image_file("cat.jpg")
    img_raw, deepstack = omni.encode_image_raw("cat.jpg")        # un-pooled (n_merged, 2048)
    # Image-conditioned text — text must contain image_token_id placeholders
    text_with_img        = omni.encode_text_with_image(prompt, "cat.jpg")
    text_with_img_native = omni.encode_text_with_image_file(prompt, "cat.jpg")
```

Math OCR:

```python
from crispembed import CrispOcrModel

ocr = CrispOcrModel("ppformulanet-l-q8_0.gguf")
latex = ocr.recognize("formula.png")           # file path
latex = ocr.recognize(pil_image)               # PIL Image
latex = ocr.recognize(numpy_uint8_array)       # (H, W, C) uint8
latex = ocr.recognize_gray(float32_array)      # (H, W) float32 [0..1]
```

Wrapper parity script:

```bash
python tests/feature_parity.py \
  --dense-model "$CRISPEMBED_DENSE_MODEL" \
  --retrieval-model "$CRISPEMBED_RETRIEVAL_MODEL" \
  --reranker-model "$CRISPEMBED_RERANKER_MODEL"
```

## Rust

```toml
[dependencies]
crispembed = { git = "https://github.com/CrispStrobe/CrispEmbed" }
```

```rust
use crispembed::CrispEmbed;

let mut model = CrispEmbed::new("model.gguf", 0)?;
let vec = model.encode("Hello world");

// Prompt prefix
model.set_prefix("query: ");

// Sparse + ColBERT (BGE-M3)
if model.has_sparse() {
    let sparse = model.encode_sparse("query");   // Vec<(i32, f32)>
}
if model.has_colbert() {
    let multi = model.encode_multivec("query");  // Vec<Vec<f32>>
}

// Bi-encoder reranking (cosine similarity)
let ranked = model.rerank_biencoder("query", &["doc1", "doc2"], Some(2));
for (idx, score) in &ranked {
    println!("  doc {} score {:.4}", idx, score);
}
```

Wrapper parity script:

```bash
cargo run -p crispembed --example feature_parity -- \
  "$CRISPEMBED_DENSE_MODEL" \
  "$CRISPEMBED_RETRIEVAL_MODEL" \
  "$CRISPEMBED_RERANKER_MODEL"
```

## Dart / Flutter

```yaml
# pubspec.yaml
dependencies:
  crispembed:
    path: <local Flutter plugin path>
```

```dart
import 'package:crispembed/crispembed.dart';

final model = CrispEmbed('model.gguf');

// Dense encoding
final vec = model.encode('Hello world');           // Float32List(384)
final batch = model.encodeBatch(['Hello', 'World']); // List<Float32List>

// Matryoshka truncation + prefix
model.setDim(128);
model.setPrefix('query: ');

// Bi-encoder reranking
final ranked = model.rerankBiencoder('query', ['doc1', 'doc2']);

// Sparse / ColBERT / cross-encoder (BGE-M3, rerankers)
if (model.hasSparse) {
  final sparse = model.encodeSparse('text');  // Map<int, double>
}

model.dispose();
```

Wrapper parity script:

```bash
cd flutter/crispembed
dart run example/feature_parity.dart \
  "$CRISPEMBED_DENSE_MODEL" \
  "$CRISPEMBED_RETRIEVAL_MODEL" \
  "$CRISPEMBED_RERANKER_MODEL" \
  "$CRISPEMBED_LIB"
```

Works on iOS (Metal GPU), Android (Vulkan/NEON), macOS, Linux, Windows.

## Feature Parity

Python, Rust, Dart, and the `crispembed` CLI now cover the same core inference features from the
shared C API: dense encode, batch encode, Matryoshka truncation, prefix control, sparse retrieval,
ColBERT multi-vector retrieval, cross-encoder reranking, and bi-encoder reranking.

The CLI still keeps some convenience-only UX that the wrappers do not mirror directly:

- CLI-only conveniences: `--list-models`, model-name auto-download, `--cache-dir`, `-f FILE`, `--json`, `--dim`, `--capabilities`.
- Wrapper-only convenience helpers: prefix getters and in-process ranking helper return types.

Inference capability parity is now aligned across all four entry points.

## Mobile (iOS / Android)

```bash
# iOS — xcframework with Metal GPU acceleration
./build-ios.sh                    # arm64 device + simulator
./build-ios.sh --device           # device only

# Android — NDK cross-compilation
./build-android.sh                # arm64-v8a + armeabi-v7a + x86_64
./build-android.sh --abi arm64-v8a --vulkan  # single ABI with Vulkan GPU
```

Output:
- iOS: `build-ios/CrispEmbed.xcframework/`
- Android: `build-android/<abi>/libcrispembed.so`

## Benchmarking

```bash
./benchmark.sh                          # single model, all engines
./benchmark.sh --multi                  # 6 models, all engines
./benchmark.sh -n 100 --skip-fastembed  # CrispEmbed + HF only, 100 runs

# RAG retrieval quality benchmark
python tests/bench_rag.py --lib build/libcrispembed.so --gguf model.gguf

# Reranking benchmark
python tests/bench_rerank.py --lib build/libcrispembed.so \
    --embed-gguf model.gguf --reranker-gguf reranker.gguf

# BidirLM-Omni text/audio/raw-vision benchmark with cosine regression check
PYTHONPATH=python python tests/benchmark_bidirlm.py \
    --model "$CRISPEMBED_MODEL" \
    --lib "$CRISPEMBED_LIB" \
    --save-baseline "$CRISPEMBED_BIDIRLM_BASELINE"
PYTHONPATH=python python tests/benchmark_bidirlm.py \
    --model "$CRISPEMBED_MODEL" \
    --lib "$CRISPEMBED_LIB" \
    --compare-baseline "$CRISPEMBED_BIDIRLM_BASELINE"
```

Compares CrispEmbed (CLI, Python ctypes, HTTP server) against HuggingFace
sentence-transformers, FastEmbed (ONNX), and fastembed-rs (Rust ONNX).
Auto-creates a `.bench-venv` for Python dependencies.

## Architecture

The model type is auto-detected from GGUF metadata at load time:
- **Encoder models** (BERT/XLM-R/MPNet/NomicBERT/ModernBERT/GTE-v1.5/DeBERTa-v2/SPLADE) → `src/crispembed.cpp` → `encode_tokens()` / `encode_tokens_batch()`. Encoder variants auto-detect from tensor names: no `pos_embd` ⇒ RoPE (NomicBERT/ModernBERT/GTE-v1.5), `rel_attn_bias` ⇒ relative position bias (MPNet), `pre_ln` ⇒ pre-LayerNorm (ModernBERT/GTE-v1.5), `ffn_up_gate` ⇒ fused `ggml_geglu`.
- **Decoder models** (Qwen3/Gemma3/BidirLM-Omni text and image-conditioned text) → `src/decoder_embed.cpp` → `decoder_encode_tokens()`. Detection heuristic: presence of `blk.0.ffn_gate` ⇒ decoder path.
- **Vision** (BidirLM-Omni) → `src/bidirlm_vision.cpp`, opens lazily on the first `crispembed_encode_image*` call when `visual.*` tensors are present.
- **Audio** (BidirLM-Omni) → `src/bidirlm_audio.cpp` wrapping the shared `crisp_audio` library, opens lazily on the first `crispembed_encode_audio` call.

Tokenizer dispatch reads `tokenizer.ggml.type`: `0=WordPiece`, `1=BPE`, `2=SentencePiece`. Heuristic fallback: vocab > 100K ⇒ SentencePiece.

Server (`examples/server/server.cpp`) exposes text embedding (4 API dialects),
face detection/recognition, ViT/CLIP vision, math OCR, and NER:
- `POST /embed` — native `{"texts": [...]}`
- `POST /v1/embeddings` — OpenAI-compatible
- `POST /api/embed` — Ollama batch
- `POST /api/embeddings` — Ollama legacy single
- `POST /detect`, `POST /face` — face detection/recognition
- `POST /vit/encode`, `POST /clip/text` — image/text encoding
- `POST /ocr/model` — OCR recognition `{"image": "path"}` → `{"latex": "..."}` (alias: `/math/ocr`)
- `POST /ner/extract` — NER `{"text": "...", "labels": [...]}` → `{"entities": [...]}`
- `POST /lid/detect` — LID `{"text": "..."}` → `{"lang": "de", "confidence": 0.99}`
- `POST /kie/extract` — KIE `{"image": "doc.png", "labels": ["total", ...]}` → `{"fields": [...]}`
- `POST /ocr/document` — multi-page OCR: upload images → searchable PDF / hOCR / ALTO / text
- `POST /preprocess/skew` — find skew angle `{"image": "..."}` → `{"angle": F}`
- `POST /preprocess/dewarp` — straighten curved text `{"image": "..."}` → PGM or JSON
- `POST /preprocess/cc-detect` — model-free line detection `{"image": "..."}` → `{"regions": [...]}`
- `POST /render/ocr` — render OCR results `{"results": [...], "format": "hocr"}` → document

**BERT encoder** (all-MiniLM, gte, arctic-embed-xs, paraphrase-multilingual-MiniLM-L12-v2):
- Token + Position + Type embeddings → Post-LN transformer → Mean/CLS pooling
- Tokenizer is WordPiece by default; `model_type=bert` with `vocab > 100K` (paraphrase-multilingual, multilingual-e5-small) loads the XLM-R SentencePiece-Unigram vocab via Viterbi DP, still with `pos_offset=0`.

**XLM-R encoder** (PIXIE-Rune, multilingual-e5-base/large, arctic-embed-l-v2):
- Token + Position(+offset) embeddings → Post-LN transformer → CLS/Mean pooling
- SentencePiece Unigram tokenizer (Viterbi DP), `pos_offset=2`, `model_type=xlm-roberta`

**BGE-M3 multi-modal** (`BAAI/bge-m3`):
- Same BERT encoder trunk with three output heads:
  - **Dense**: mean-pool → L2 normalize → `float[1024]`
  - **Sparse**: `Linear(H,1)` + ReLU → scatter via input_ids → `{token_id: weight}`
  - **ColBERT**: `Linear(H,128)` → per-token L2 normalize → `float[n_tokens][128]`

**MPNet encoder** (all-mpnet-base-v2):
- Token + Position(+offset) embeddings → Post-LN transformer with relative position bias → Mean pooling
- T5-style logarithmic bucket relative attention bias (32 buckets × n_heads)

**NomicBERT encoder** (nomic-embed-text-v1.5):
- Token embeddings (no position) + RoPE → Post-LN transformer + SwiGLU FFN → Mean pooling
- Rotary position embeddings (same as decoder path), no absolute position embeddings

**NomicBERT MoE encoder** (nomic-embed-text-v2-moe):
- Token + Type embeddings + emb_ln + RoPE → Post-LN transformer with mixed MoE/dense FFN → Mean pooling
- 8 experts, top-2 routing, GELU activation; MoE on odd layers, dense GELU FFN on even layers
- Fully in-graph routing: ggml_top_k + ggml_get_rows + ggml_mul_mat_id (no CPU-side partial compute)

**Cross-encoder reranker** (BGE-reranker-v2-m3, ms-marco-MiniLM, mxbai-rerank, etc.):
- `[CLS] query [SEP] document [SEP]` pair tokenization → CLS hidden state → `Linear(H,1)` → scalar score

**Qwen3 decoder** (Octen, F2LLM, Jina v5, Harrier-0.6B, Qwen3-Embed):
- Token embeddings + RoPE → RMSNorm + GQA with causal mask + SwiGLU → Last-token pooling

**Gemma3 decoder** (Harrier-270M):
- Token embeddings * sqrt(H) + RoPE → Gemma3 RMSNorm(1+w) + GQA + GeGLU → Last-token pooling

**BidirLM-Omni** (BidirLM-Omni-2.5B-Embedding) — text + audio + image, shared 2048-d space:
- **Text**: bidirectional Qwen3 body (RoPE, GQA, RMSNorm, q_norm/k_norm, SwiGLU) → Mean pooling → 2048-d.
- **Audio** (when CrispAudio is available): Whisper-shape encoder (Conv2D stem + 24-layer pre-LN encoder + proj1/GELU/proj2) → Mean pooling → same 2048-d shared space. Built on the shared `crisp_audio` library from the configured CrispASR checkout (CMake auto-discovers it; override with `-DCRISP_AUDIO_DIR=…`).
- **Vision**: BidirLM/Qwen2VL-style vision tower in ggml (patch embed, interpolated position embedding, rotary attention, patch merger, DeepStack hooks at layers 8/16/24). Two preprocessing paths: the Python binding's `encode_image(image)` uses HF `Qwen2VLImageProcessorFast` (byte-tight HF parity, requires `transformers`); `crispembed -m … --image FILE` and `model.encode_image_file(path)` use CrispEmbed's in-process C++ preprocessor (smart_resize + Catmull-Rom bicubic with antialias + OpenAI CLIP normalize + Qwen2VL patchify, via `stb_image`) — no `transformers` runtime dep, cosine ≈ 0.97 vs HF on photographs (the gap is JPEG decoder differences PIL/libjpeg-turbo vs stb).
- **Image-conditioned text**: `crispembed_encode_text_with_image()` runs the vision tower, splices `image_embeds` into `inputs_embeds` at every `image_token_id` placeholder, adds `deepstack[k]` features at the first 3 decoder layers, and uses 3D interleaved-MRoPE position ids derived from `grid_thw`. A lower-level `crispembed_encode_with_image_ids()` accepts pre-tokenized ids for clean parity with external tokenizers. See `tests/test_bidirlm_image_text.py` for the parity test against HF `BidirLMOmniModel.forward(input_ids, pixel_values, image_grid_thw)` (cosine ≥ 0.99).
- **Pooled-only image path**: `crispembed_encode_image()` skips DeepStack materialization since the mean-pooled image vector doesn't use it; `encode_image_raw` and `encode_text_with_image` keep DeepStack on.
- Decoder text/text+image embedding both run through `ggml_backend_sched`, matching the encoder and vision execution paths.

Cache convention: point `CRISPEMBED_CACHE_DIR` at your backing store to keep large GGUF/cache files out of the repo tree (default: `~/.cache/crispembed/`).

All via ggml graphs with GPU dispatch (ggml_backend_sched).
See [PLAN.md](PLAN.md) for architecture and roadmap, [HISTORY.md](HISTORY.md) for completed milestones, [LEARNINGS.md](LEARNINGS.md) for technical detail (3D MRoPE workaround, DeepStack splice via mask+add, decoder scheduler init), and [PERFORMANCE.md](PERFORMANCE.md) for benchmarks.

## Credits

- [ggml](https://github.com/ggml-org/ggml) -- inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) -- shared core (gguf_loader, bpe.h)
- [sentence-transformers](https://www.sbert.net/) -- ground-truth validation
