# CrispEmbed

[![Build](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml/badge.svg)](https://github.com/CrispStrobe/CrispEmbed/actions/workflows/build.yml)

**A single C++/ggml binary for retrieval and document understanding — no Python
runtime, no ONNX.** Text/image/face embeddings, sparse & multi-vector retrieval,
rerankers, a full OCR stack (general, scene-text, math, music), layout analysis,
NER/KIE, and document preprocessing — all auto-detected from GGUF metadata and
GPU-accelerated (CUDA / Vulkan / Metal), with Python, Rust, Dart, HTTP, and
**WebAssembly** front-ends.

Where llama.cpp focuses on text *generation*, CrispEmbed covers the *retrieval,
understanding, and document-processing* half of the ggml world. **9.5× faster
than FastEmbed (ONNX)** on MiniLM-L6; runs on Linux, macOS, Windows, iOS,
Android, and in the browser.

> **Live demos:** [WASM OCR (client-side)](https://crispstrobe.github.io/CrispEmbed/)
> · [HuggingFace Space](https://huggingface.co/spaces/cstr/CrispEmbed) (embeddings + math OCR)

---

## Capabilities at a glance

| Domain | What it does | Highlights |
|---|---|---|
| **Text embeddings** | Dense vectors from 10 encoder/decoder architectures | BERT, XLM-R, MPNet, NomicBERT (+MoE), ModernBERT, GTE-v1.5, DeBERTa-v2, Qwen3, Gemma3. Matryoshka truncation, prompt prefixes. cos ≥ 0.965 vs HF |
| **Retrieval** | Sparse + multi-vector + reranking | SPLADE / BGE-M3 sparse term weights, ColBERT per-token + MaxSim, cross-encoder & bi-encoder rerankers |
| **OCR** | 15+ engines, image → text/LaTeX/notation | General (DBNet+TrOCR), scene-text (PARSeq), 7 math engines, 4 music (OMR) engines, 6 document VLMs, 12-language Tesseract-LSTM |
| **Document AI** | Understand page structure | RT-DETRv2 layout (17 classes), Surya text detection, LiLT layout-aware KIE, hOCR/ALTO/searchable-PDF output |
| **NER / KIE / LID** | Extract structured info | Zero-shot (GLiNER) + fixed-label (BERT/XLM-R) NER, receipt/form KIE, CLD3/GlotLID language ID |
| **Vision & face** | Cross-modal + biometrics | CLIP/SigLIP text-image search, YuNet/SCRFD detect, ArcFace/SFace/AuraFace recognize |
| **Preprocessing** | Clean & upscale before OCR | Classical deskew/binarize/dewarp, NAFNet denoise, TPS dewarp, 8 super-resolution engines, PDF-DPI auto-tuning |

Everything ships in one library with a unified C ABI. Over **100 models** (200+
GGUF variants) are in the auto-download registry — run `crispembed --list-models`
for the authoritative, always-current list with per-model license tags.

---

## Quick start

```bash
# Clone (with the ggml submodule) and build
git clone --recursive https://github.com/CrispStrobe/CrispEmbed
cd CrispEmbed
cmake -S . -B build && cmake --build build -j        # macOS: ./build-macos.sh (Metal)

# Text embedding (auto-downloads the model by name, or pass a local .gguf)
./build/crispembed -m all-MiniLM-L6-v2 "Hello world"
./build/crispembed -m model.gguf -d 128 "Hello world"          # Matryoshka: 128 dims
./build/crispembed -m model.gguf --prefix "query: " "Hello"    # prompt prefix

# Retrieval modalities (BGE-M3)
./build/crispembed -m bge-m3 --sparse  "Hello world"
./build/crispembed -m bge-m3 --colbert "Hello world"
./build/crispembed -m bge-reranker-v2-m3 --rerank "capital of france" \
    "Paris is the capital of France." "Bicycles have two wheels."

# OCR / document AI (engine auto-detected from GGUF metadata)
./build/crispembed -m ppformulanet-l  --ocr formula.png       # math → LaTeX
./build/crispembed -m flova           --ocr score.png         # music → LilyPond
./build/crispembed -m transcoda       --ocr page.png          # full-page score → **kern
./build/crispembed -m qwen3vl-2b      --ocr document.png      # VLM document OCR

# Cross-modal & face
./build/crispembed -m clip-vit-base-patch16 --image photo.jpg
./build/crispembed -m yunet --detect photo.jpg --json

# HTTP server (text + vision + face + CLIP + OCR + NER in one process)
./build/crispembed-server -m all-MiniLM-L6-v2 --ocr ppformulanet-l-q8_0.gguf --port 8080
curl -X POST http://localhost:8080/embed -d '{"texts": ["Hello world"]}'
```

For modular document parsing, start the orchestrator with independent layout,
table, and formula modules:

```bash
./build/crispembed-server --ocr-pipeline \
  --ocr-det dbnet-ic15-q4_k.gguf --ocr-rec trocr-small-printed-q8_0.gguf \
  --layout layout-heron-f32.gguf \
  --table tesseract-eng-f16.gguf --tables \
  --formula ppformulanet-l-q4_k.gguf --formulas
curl http://localhost:8080/capabilities
```

The server also provides `/health/live` and `/health/ready`. `/health` doubles as
a capability probe (`reranker`, `sparse`, `colbert`, `ocr_pipeline`, `layout`, …),
but those keys are **present-when-active**: an inactive capability is omitted
rather than reported as `false`, so test for the key's presence, not its value.
Text recognition can be swapped independently for TrOCR, Tesseract-LSTM,
PP-OCRv6, EasyOCR, PARSeq, or a VLM.

---

## Install & build

### From source

```bash
# Linux / macOS — CPU
cmake -S . -B build && cmake --build build -j

# GPU backends
cmake -S . -B build -DGGML_CUDA=ON   && cmake --build build -j   # NVIDIA
cmake -S . -B build -DGGML_VULKAN=ON && cmake --build build -j   # cross-platform
cmake -S . -B build -DGGML_BLAS=ON   && cmake --build build -j   # OpenBLAS / MKL

# macOS (recommended: Metal + Accelerate + embedded shaders)
./build-macos.sh            # add --cpu for CPU-only, --shared for the Python lib

# Windows (VS 2022 Build Tools + Ninja)
build-windows.bat           # or build-vulkan.bat / build-cuda.bat
```

**Requirements:** C++17 compiler, CMake ≥ 3.14. Optional: OpenBLAS, Intel MKL,
CUDA Toolkit, or Vulkan SDK. If you see *"ggml does not contain a CMakeLists.txt"*,
run `git submodule update --init --recursive`.

### CPU requirements & redistributable builds

A source build defaults to `-march=native`: it targets **the machine you build
on**, which is the fastest option and the right one for local use.

Binaries you intend to copy to another machine must instead be pinned to a
fixed baseline, or they die with `Illegal instruction` on any CPU that lacks an
extension the build machine had:

```bash
cmake -S . -B build -DGGML_NATIVE=OFF     # portable; CRISPEMBED_NATIVE follows it
python scripts/check-cpu-baseline.py build   # fails the build if it isn't portable
```

The prebuilt release archives and wheels are all built this way. Their floor is:

| Platform | Baseline | Runs on |
|----------|----------|---------|
| x86_64 (Linux / Windows) | SSE4.2 + AVX + AVX2 + FMA + F16C + BMI2 | Intel Haswell (2013) / AMD Excavator (2015) and newer |
| aarch64 (Linux) | `armv8.2-a+fp16+dotprod` | Neoverse N1/V1, Cortex-A76 and newer |
| arm64 (macOS) | Apple clang default for arm64 | Apple Silicon (M1 and newer) |

No release artifact uses AVX-512 or AMX: no current Intel consumer CPU has
them. On a pre-AVX2 CPU, build from source with `-DGGML_NATIVE=OFF
-DGGML_AVX2=OFF -DGGML_AVX=OFF` (much slower, but it runs).

### Linux runtime requirements

The Linux tarballs are self-contained apart from the base C/C++ runtime. They
need only `libc`, `libm`, `libstdc++`, `libgcc_s` and the loader — no BLAS, no
OpenMP, nothing to install. `scripts/check-bundled-deps.py` enforces this at
packaging time:

```bash
python scripts/check-bundled-deps.py pkg    # every DT_NEEDED bundled or base-system
```

That guard exists because it was not always true: up to and including v0.17.0,
`libggml-blas.so.0` carried a hard dependency on `libopenblas.so.0` that the
archive never shipped, so `crispembed-server` died in the dynamic loader with
**exit code 127 and no output** on any machine without OpenBLAS installed
(SubtitleEdit#13205). On an affected release the workaround is
`sudo apt install libopenblas0` / `sudo pacman -S openblas`.

**CUDA on Linux — two archives, pick by what the host already has:**

| archive | size | host must provide |
|---------|------|-------------------|
| `crispembed-linux-x86_64-cuda.tar.gz` | small | NVIDIA driver **and** the CUDA 12.x toolkit runtime (`libcudart`, `libcublas`) |
| `crispembed-linux-x86_64-cuda-bundled.tar.gz` | large | NVIDIA driver only (`libcuda.so.1`) |

The slim archive is the original and keeps its name so existing pins stay
valid — but note its requirement is the **toolkit**, not merely a driver. A
driver alone provides `libcuda.so.1`; `libcudart`/`libcublas` come from the
toolkit, and because `libggml.so` hard-links `libggml-cuda.so`, a driver-only
machine fails at load with exit 127 rather than falling back to CPU (#42). If
you are not sure the toolkit is installed, take the bundled one. Both archives
are checked against their own contract at packaging time.

**glibc floor:** the Linux tarballs are built inside
`quay.io/pypa/manylinux_2_28` and require **glibc ≥ 2.28** — measured at
2.27 / GLIBCXX 3.4.22 — so they run on Ubuntu 18.04+, Debian 10+, RHEL/EL 8+
and any current rolling distro. The packaging gate enforces this
(`check-bundled-deps.py --max-glibc 2.28`), so a future base-image change
cannot silently raise it.

Up to and including v0.17.2 they were built on the runner's Ubuntu 24.04 and
needed **glibc 2.38 / GLIBCXX 3.4.32**, which meant they would not start on
Ubuntu 22.04 or Debian 12 — a second, independent startup failure on top of
the OpenBLAS one (#42). The CUDA archives are still built outside the
container and keep the higher floor.

### Mobile & browser

```bash
./build-ios.sh              # CrispEmbed.xcframework (Metal GPU)
./build-android.sh          # arm64-v8a + armeabi-v7a + x86_64 (Vulkan/NEON)
./build-wasm.sh             # client-side OCR (SIMD / multithreaded / WebGPU tiers)
```

The WASM build runs the full DBNet+TrOCR pipeline, scan-cleanup, and every
auto-detected single-model OCR engine — math → LaTeX, scene text, and **music
(OMR: SMT / TrOMR / Flova / Transcoda)** — entirely client-side (no server, no API key). Three
tiers: SIMD CPU, multithreaded (COOP/COEP service worker, works on GitHub Pages),
and experimental **WebGPU** (~2.8× on ViT recognition, ~60× on DBNet detection vs
WASM CPU). The whole engine set is one 2.3 MB `.wasm`. See `examples/wasm-ocr/README.md`.

### As a system library

`cmake --install build --prefix /usr/local` lays out a standard tree with a
versioned `.so`/`.dylib` (SONAME, `RPATH=$ORIGIN`), CMake package config, and a
relocatable pkg-config file:

```cmake
find_package(crispembed REQUIRED)
target_link_libraries(my_app PRIVATE crispembed::crispembed)
```

---

## Text embeddings & retrieval

Ten architectures, auto-detected from GGUF tensor names. Dense, sparse, and
multi-vector heads all run through ggml graphs with GPU dispatch.

### Verified parity (cos vs HuggingFace)

30 embedding models validated at cos ≥ 0.965; a representative slice:

| Model | Type | Dim | F32 | Q8_0 | Q4_K |
|-------|------|-----|-----|------|------|
| all-MiniLM-L6-v2 | BERT | 384 | 0.999999 | 0.9995 | 0.97 |
| multilingual-e5-large | XLM-R | 1024 | 0.999997 | 0.9999 | 0.99 |
| gte-modernbert-base | ModernBERT | 768 | 0.999991 | 0.9999 | — |
| granite-embedding-311m-r2 | ModernBERT | 768 | 1.000000 | 0.9998 | — |
| granite-embedding-97m-r2 | ModernBERT | 384 | 1.000000 | 0.9996 | — |
| nomic-embed-text-v2-moe | NomicBERT MoE | 768 | 1.000000 | 0.9996 | 0.966 |
| EmbeddingGemma-300m | Gemma3 | 768 | 1.000000 | 0.9998 | 0.98 |
| Qwen3-Embedding-0.6B | Qwen3 | 1024 | 0.999895 | 0.9996 | 0.97 |
| Octen-Embedding-8B | Qwen3 | 4096 | — | — | 0.965 |

Q8_0 = all PASS (cos ≥ 0.995). `—` in Q4_K = SwiGLU/GeGLU too sensitive for
aggressive quants (defaults to Q8_0). The full table lives in
[PERFORMANCE.md](PERFORMANCE.md).

CrispEmbed also loads the **official/community `gemma-embedding` GGUFs** directly
(llama.cpp SPM exports, e.g. `ggml-org/embeddinggemma-300m-*-GGUF`). These ship
without the SentenceTransformers Dense head — llama.cpp applies it from an
external file — so their raw output is the backbone mean-pool. Bake the Dense
head in with `models/add-st-dense-to-gguf.py` for HF-compatible embeddings
(cos 0.984 vs HF), or just pull the ready-made `embeddinggemma-300m-qat`.

### Sparse, ColBERT & reranking (BGE-M3)

```python
from crispembed import CrispEmbed
model = CrispEmbed("bge-m3.gguf")

vec    = model.encode("Hello world")                 # dense, L2-normalized (1024,)
sparse = model.encode_sparse("Hello world")          # {token_id: weight}   (SPLADE-style)
multi  = model.encode_multivec("Hello world")        # (n_tokens, 128)      (ColBERT)

reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
score = reranker.rerank("query", "document")         # cross-encoder logit
ranked = model.rerank_biencoder("query", ["d1","d2"], top_n=2)   # cosine
```

LFM2.5-ColBERT (128-d per token) and all eight cross-encoder rerankers
(including ModernBERT-based Ettin) are supported. Sparse/ColBERT heads are
written into the GGUF by the converter and detected via `has_sparse` /
`has_colbert`.

**Byte-level BPE tokenizers transcribe the pre-tokenizer regex the checkpoint
declares** (`src/core/bpe.h`), one per family — Qwen2/Qwen3, LFM2.5, and the
DeepSeek-OCR-2 / Unlimited-OCR Split sequence — validated against HuggingFace's
own `pre_tokenize_str()` in model-free CI. Older builds used a whitespace-split
approximation that collapsed whitespace runs and dropped newlines, so multi-line
documents and non-ASCII punctuation (German typographic quotes, dashes,
currency) tokenized differently from the reference. Set
`CRISPEMBED_BPE_LEGACY_WHITESPACE=1` to restore the old ids for bisection.

---

## OCR & document AI

15+ engines for image → text, most auto-detected from GGUF metadata via the
unified `crispembed_ocr_model_*` C API. Available through CLI (`--ocr`), server
(`POST /ocr/model`), Python (`CrispOcrModel`), Rust, and Dart/Flutter.

### OCR engine matrix

| Model | Architecture | Params | Use case | License |
|-------|-------------|--------|----------|---------|
| **PP-OCRv6** | PPLCNetV4 + light-SVTR + CTC | 5–63 MB | General doc pipeline, best measured CER (see below) | Apache-2.0 |
| **EasyOCR** | VGG/ResNet + BiLSTM + CTC | 16 MB | General doc pipeline, EasyOCR-compatible output | Apache-2.0 |
| **PARSeq** | ViT + Transformer | 24M | Scene text (SOTA, ECCV'22) | Apache-2.0 |
| **DBNet + TrOCR** | ResNet-18+FPNC → DeiT+Transformer | 7+63M | General doc pipeline (~200ms/region) | MIT / Apache-2.0 |
| **Tesseract-LSTM** | VGSL Conv+LSTM+CTC | <2 MB | 12 languages, tiny GGUFs | Apache-2.0 |
| **PP-FormulaNet-L** | SAM-ViT + MBart | 181M | Printed math (best) | Apache-2.0 |
| **MixTeX** | Swin-Tiny + RoBERTa | 86M | CN+EN LaTeX | Apache-2.0 |
| **Texo-Distill** | HGNetv2 + MBart | 20M | Printed math (small) | AGPL-3.0 |
| **PosFormer / BTTR / HMER** | DenseNet + Transformer/GRU | 6–7M | Handwritten math (CROHME) | MIT / CC-BY-NC |
| **SMT** | ConvNext + Transformer | 21M | Printed music (systems) → bekern (96.3% GrandStaff) | MIT |
| **SMT++ full-page** | ConvNext + Transformer | 11M | Whole pianoform *page* → bekern (no segmentation) | MIT |
| **Polyphonic-TrOMR** | ResNetV2+ViT + 4-head decoder | ~22M | Printed music photos → symbolic | Apache-2.0 |
| **Flova/omr_transformer** | DonutSwin + mBART-4L | 143M | Handwritten/whiteboard music → LilyPond | Apache-2.0 |
| **Transcoda-59M** | ConvNeXt-V2 + 8L RoPE cross-attn | 59M | Zero-shot full-page score → Humdrum `**kern` (real-scan SOTA) | CC-BY-4.0 |
| **GOT-OCR2** | SAM ViT-B + Qwen2-0.5B | 0.7B | Doc OCR (text+LaTeX+tables) | Apache-2.0 |
| **GLM-OCR** | CogViT + GLM-0.5B | 0.9B | Doc OCR (OmniDocBench #1, 8 langs) | MIT |
| **InternVL2 / 2.5** | InternViT + Qwen2/InternLM2.5 | 0.9–2.1B | Edge/WASM & EN+DE VLM OCR | MIT |
| **Qwen2.5-VL / Qwen3-VL** | ViT (+DeepStack) + Qwen LLM | 2.4–3.6B | General/multilingual VLM OCR | Apache-2.0 |
| **DeepSeek-OCR-2 / Unlimited-OCR** | dual ViT + DeepSeek-V2 MoE | 3–3.3B | Full-page doc OCR + layout grounding | Apache-2.0 / MIT |
| **Qari-OCR** | Qwen2-VL-2B + LoRA | 2B | Arabic OCR with diacritics | Apache-2.0 |

#### Measured against the engines we port

`tests/ocr_synth_corpus.py` renders 20 fixtures that carry their own exact
ground truth, and `tests/ocr_external_parity.py` runs the native lanes beside
system Tesseract, Python EasyOCR and Python PaddleOCR on the same images.
Character/word error over that corpus:

| engine | kind | CER | WER |
|---|---|--:|--:|
| `crispembed --ocr-engine ppocrv6` | native | **0.0031** | **0.0178** |
| PaddleOCR 2.10 (Python) | external | 0.0185 | 0.1153 |
| Tesseract 5.5.2 (`--psm 6`) | external | 0.0256 | 0.0890 |
| `crispembed --ocr-engine tesseract` | native | 0.0290 | 0.1623 |
| EasyOCR 1.7.2 (Python) | external | 0.0769 | 0.2363 |
| `crispembed --ocr-engine easyocr` | native | 0.0808 | 0.3190 |

Latency for one CLI invocation of the same page, on an idle machine with
`tesseract` measured before and after as a control (0.12/0.15 s):
`tesseract-cli` 0.135 s, `--ocr-engine tesseract` 0.48 s, `--ocr-engine ppocrv6`
1.39 s, `--ocr-engine easyocr` 1.47 s. `tesseract-cli` is the like-for-like
comparison — one subprocess per image, model load included on both sides.

Reproduce with `python tests/ocr_synth_corpus.py --output <dir>` then
`python tests/ocr_external_parity.py --images <dir> --model-dir <gguf dir>`.
The harness also reports latency, in two columns that are deliberately not
comparable: `proc_ms` covers a whole invocation including model load, while
`engine_ms` excludes it. Run it on an **idle** machine — it prints the
`tesseract-cli` arm as a load control, and if that is far above ~150 ms the
timings are measuring contention rather than the engines.

Formula/music engines validated per-stage against their HF references (typically
cos ≥ 0.999, byte-exact greedy decode). VLM engines ingest the full page and
letterbox internally — the pipeline skips scan-cleanup for them.
`CRISPEMBED_MAX_PIXELS` trades resolution for CPU speed on all variable-resolution
VLMs.

### Optical Music Recognition (OMR)

Three permissively-licensed engines, all auto-detected via `--ocr`:

- **SMT** (MIT, `smt-grandstaff`) — printed polyphonic staff systems → bekern.
  Reproduces the reference exactly (per-stage cos = 1.0, 96.3% vs GrandStaff).
- **SMT++ full-page** (MIT, `smt-fp`) — a whole pianoform *page* → bekern in one
  pass, no staff/system segmentation (per-stage cos ≥ 0.9998, byte-exact greedy
  decode vs the HF reference).
- **Polyphonic-TrOMR** (Apache-2.0) — staff photos → rhythm/pitch/lift streams.
  Robust on real photos; byte-exact decode on the reference examples.
- **Flova/omr_transformer** (Apache-2.0) — the only permissive *handwritten*
  music model; whiteboard "simple notes" → LilyPond, byte-exact incl. the native
  no-`transformers` preprocessing path.
- **Transcoda-59M** (CC-BY-4.0, `transcoda`) — zero-shot *full-page* score →
  Humdrum `**kern` in one pass. ConvNeXt-V2-Tiny encoder + 8-layer RoPE
  cross-attention decoder; OMR-NED SOTA on real historical scans. Clean-room
  engine (per-stage cos = 1.0, byte-exact greedy decode vs the HF reference).

### Layout, detection & preprocessing

- **Layout detection** — RT-DETRv2 (ResNet-50 + deformable decoder), 17 region
  types. `--layout`, `POST /layout/detect`. Encoder cos = 1.0 vs HF; Q8_0 43 MB.
- **Text detection** — Surya EfficientViT segformer (38M, 91 languages,
  GPU-accelerated), plus a model-free connected-component fallback (0 downloads,
  4 ms/page). `--text-detect`.
- **Scan cleanup** — Tier 1 classical (deskew with dual-detector consensus,
  Otsu/Sauvola binarize, border crop, background whitening, cubic-baseline
  dewarp, 1-bit DWA morphology — 21× faster than float, all reimplemented from
  Leptonica). Tier 2 learned NAFNet denoise. `--cleanup-only` / `--cleanup`.
- **Text super-resolution** — PAN (4× whole-page, 0.5 MB), TBSRN (2× per-line,
  2 MB), NAFNet-SR scaffold; parity cos ≥ 0.9996. Eight SR backbones total (HAT,
  DAT, ESRGAN, SwinIR, TBSRN, SAFMN, Restormer, SCUNet).
- **PDF DPI profiling** — zero-dependency PDF parser computes effective page DPI
  to auto-select OCR resolution (downsample high-DPI, super-resolve low-DPI).
- **Output formats** — plain text, hOCR, ALTO 3.1 XML, searchable PDF, with
  multi-page accumulation. An **orchestrator** routes by source type
  (screenshot/scan/photo) with accept-gate cascading and VLM fallback.

---

## NER, KIE & language ID

- **NER** — zero-shot **GLiNER** (LFM2.5-350M bidirectional backbone; arbitrary
  entity types at inference, all 16 layers cos = 1.0 vs HF) and fixed-label
  **BERT/XLM-R** (`bert-base-ner` EN, `xlmr-ner-hrl` 10 languages). One `--ner`
  API, backend auto-detected.
- **KIE** — chains OCR + GLiNER to pull key-value fields from receipts/invoices/
  forms, no new model. `--kie`, `POST /kie/extract`, `CrispKIE`.
- **LiLT** — layout-aware document understanding (RoBERTa + layout transformer
  via BiACM), 130M, MIT, FUNSD token classification. 25/25 layers cos = 1.0.
- **LID** — CLD3 (109 langs) / GlotLID (2102 ISO 639-3) text language ID, used to
  auto-select the Tesseract model in the OCR pipeline.

```bash
./build/crispembed -m gliner-lfm --ner "Maria Schmidt arbeitet bei Siemens in München"
# Maria Schmidt → person, Siemens → organization, München → location
```

---

## Vision & face

CLIP and SigLIP text-image cross-modal search (shared vector space), plus a full
face pipeline: YuNet (0.2 MB) / SCRFD (16 MB) detection → ArcFace / SFace /
AuraFace recognition.

```bash
./build/crispembed -m clip-text-base "a photo of a cat"
./build/crispembed -m clip-vit-base-patch16 --image photo.jpg
./build/crispembed -m yunet --detect photo.jpg --json
```

> **Face recognition is biometric processing.** A face template is
> special-category personal data (GDPR Art. 9), and searching a gallery of them
> (1:N identification) is a high-risk AI system under the EU AI Act. Loading a
> recognition model therefore requires a one-time acknowledgement —
> `--accept-biometric` on the CLI and server, `CRISPEMBED_ACCEPT_BIOMETRIC=1` in
> any process, or `crispembed_accept_biometric_use()` and its Python / Rust /
> Dart equivalents. The check sits in `crispembed_face_init()`, so it holds for
> the bindings too, not just the CLI; detection alone is not gated.
> CrispEmbed provides **no gallery, enrolment, index or 1:N search**
> primitive, and prints cosine similarity without a match/no-match verdict —
> thresholds must be calibrated and documented per deployment.
>
> On the server the acknowledgement is made **once per process**, and there is
> no authentication: every client that can reach the port inherits it, and
> `/face` reads its input by server-side path. It binds `127.0.0.1` by default —
> if you move it off loopback with `--host`, put an authenticating proxy in
> front. Read **[POLICY.md](POLICY.md)** before building on this.
> `examples/face_verify.py` shows 1:1 verification.

**BidirLM-Omni** unifies text, audio, and image into one shared 2048-d space
(bidirectional Qwen3 body + Whisper-shape audio encoder + Qwen2VL vision tower
with DeepStack). Q4_K verified locally across all three modalities.

---

## Language bindings

Python, Rust, Dart, and the CLI expose the same core inference features from the
shared C ABI (dense/batch encode, Matryoshka, prefix, sparse, ColBERT, rerank).

```python
# Python  (needs the shared lib: --shared or -DCRISPEMBED_BUILD_SHARED=ON)
from crispembed import CrispEmbed
model = CrispEmbed("all-MiniLM-L6-v2.gguf")
vecs = model.encode(["Hello world", "Goodbye world"])   # (2, 384), one batched GPU call
model.set_dim(128); model.set_prefix("query: ")
```

```rust
// Rust  —  crispembed = "0.16"          (crates.io; see "Rust crate" below)
let mut model = crispembed::CrispEmbed::new("model.gguf", 0)?;
let vec = model.encode("Hello world");
```

```dart
// Dart / Flutter  (iOS Metal, Android Vulkan/NEON, desktop)
final model = CrispEmbed('model.gguf');
final vec = model.encode('Hello world');   // Float32List(384)
```

```c
/* C ABI */
void *ctx = crispembed_ocr_model_init("ppformulanet-l-q8_0.gguf", 4);
const char *latex = crispembed_ocr_model_recognize(ctx, pixels, w, h, ch, &len);
```

Per-language parity scripts (`tests/feature_parity.py`, the Rust/Dart
`feature_parity` examples) verify the wrappers against the CLI. All 45+ registry
models also export as **Ollama-compatible** GGUFs (`--ollama` converter flag).

### Rust crate

```toml
[dependencies]
crispembed = "0.16"          # safe wrapper; crispembed-sys is the raw FFI layer
```

`crispembed-sys` vendors the C/C++ sources and builds them with cmake, so the
crate needs **cmake and a C++17 compiler** but nothing preinstalled — and it
works offline, on docs.rs, and on any target. ggml is linked statically into a
single `libcrispembed`, so there is one library to ship. A cold build takes a
few minutes.

To skip that compile, point the crate at a prebuilt library from the
[releases](https://github.com/CrispStrobe/CrispEmbed/releases) — matching the
crate version — and build.rs links it instead of invoking cmake:

```bash
gh release download v0.16.1 -R CrispStrobe/CrispEmbed -p 'crispembed-macos-arm64.tar.gz'
mkdir -p lib && tar xzf crispembed-macos-arm64.tar.gz -C lib
export CRISPEMBED_SYS_LIB_DIR=$PWD/lib
cargo build       # links the prebuilt; no cmake, no source build
```

No C/C++ sources are needed on this path — not the repository, not the ggml
submodule, not the vendored copy. `build.rs` looks for a prebuilt library
first and only resolves sources if it actually has to compile them. (Between
`a3156a2a` and v0.17.0 it resolved sources unconditionally, so this documented
workflow failed with *"crispembed sources not found"* even with a valid
`CRISPEMBED_SYS_LIB_DIR`; `rust-crates.yml` now regression-tests it.) If the
variable points somewhere without a library, build.rs emits a `cargo:warning`
and falls back rather than failing silently.

The release tarballs ship `libcrispembed` alongside separate `libggml*`
libraries, so with this path all of them have to travel with your binary — the
from-source default produces just the one. Release assets exist for linux
x86_64/arm64, macOS arm64, Windows x86_64 (plus CUDA/Vulkan variants), Android
and iOS; other targets (including x86_64 macOS) use the source build.

GPU backends are cargo features: `metal`, `cuda`, `vulkan`.

---

## Converting & quantizing models

```bash
# Encoders (BERT / XLM-R) and decoders (Qwen3 / Gemma3)
pip install torch transformers gguf
python models/convert-bert-to-gguf.py --model sentence-transformers/all-MiniLM-L6-v2 --output out.gguf --crisp
python models/convert-decoder-embed-to-gguf.py --model Octen/Octen-Embedding-0.6B --output octen.gguf

# Quantize (Q8_0 recommended; Q4_K for max compression)
./build/crispembed-quantize model.gguf model-q8_0.gguf q8_0

# Import a stock llama.cpp VL model (LLM GGUF + mmproj) byte-for-byte, no re-quant
python models/merge-llamacpp-gguf.py --llm InternVL2_5-1B-Q8_0.gguf \
    --mmproj mmproj-InternVL2_5-1B-f16.gguf --output internvl2_5-1b-crispembed.gguf
```

GGUFs are quantized with an **importance matrix** (imatrix, activation-weighted),
A/B-validated per model class with a task-appropriate metric (mean cosine for
embedders, Kendall-τ for rerankers, span-F1 for NER, etc.). `-m <model>`
auto-downloads each model's best-tested small flavor; `-q8` / `-q4k` / `-iq4xs`
suffixes pick a specific variant.

| Type | Compression | Quality (cos vs F32) |
|------|-------------|----------------------|
| Q8_0 | ~3.8× | > 0.995 (recommended) |
| Q6_K | ~4.5× | > 0.99 |
| Q5_K | ~5× | > 0.98 |
| Q4_K | ~5.5× | > 0.95 (max compression) |

Pre-converted models: **[huggingface.co/cstr](https://huggingface.co/cstr)**.

---

## Performance

Apple M1, Metal, all-MiniLM-L6-v2:

| Engine | Single text | Batch (10) |
|--------|-------------|------------|
| **CrispEmbed** (Python ctypes) | **3.6 ms** / 280 t/s | **12.7 ms** / **787 t/s** |
| fastembed-rs (Rust ONNX) | 3.8 ms / 263 t/s | 18.9 ms / 528 t/s |
| HuggingFace (PyTorch) | 12.2 ms / 82 t/s | 29.8 ms / 335 t/s |

Full multi-model and Ollama Q8_0/Q4_K numbers in [PERFORMANCE.md](PERFORMANCE.md).
Benchmark with `./benchmark.sh [--multi]`.

### One-shot CLI startup

Those are *warm* numbers. A one-shot `crispembed -m model.gguf --json "text"`
also pays a fixed init, which used to be ~0.9 s on an M1 and dominated
short-text CLI/scripting use. Measured 2026-08-05, multilingual-e5-small q8_0,
same binary, output byte-identical:

| | one-shot total | Metal device init |
|---|--:|--:|
| before | 895 ms | 683 ms |
| after | **186 ms** (4.8x) | 29 ms |

Almost all of it was ggml-metal's persistent `MTLBinaryArchive` pipeline cache
(`~/Library/Caches/ggml-metal/*.archive`): it is append-only across every
engine and binary on the machine, had grown to 683 MB, and cost ~1 ms/MB to
open — while a one-shot CLI can never write an entry back to it (the archive is
serialised from a static destructor, which `_exit()` skips). CrispEmbed now
skips an archive larger than 64 MB. Env gates:

| Variable | Effect |
|---|---|
| `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB=N` | archive size cap in MB (default 64). `0` = uncapped = pre-2026-08-05 behaviour |
| `CRISPEMBED_INIT_BENCH=1` | print the per-component init profile (gguf parse / vocab / tokenizer / backend / weights) to stderr |
| `CRISPEMBED_GGUF_REPARSE=1` | restore the second GGUF metadata parse the loader used to do (~29 ms on a 250k-token vocab) |
| `CRISPEMBED_GPU_PREF_CPU_LEGACY=1` | restore the old behaviour where `--gpu-backend cpu` fell through to the GPU |
| `CRISPEMBED_ONESHOT_CPU=1` | opt in to the CPU backend for a CLI run that did not pass `--gpu-backend` |
| `CRISPEMBED_FORCE_CPU=1` | force the CPU backend for the embedding engine (pre-existing) |

`--gpu-backend cpu` now genuinely skips GPU device init (0.14 s one-shot); it
previously matched no GPU device, warned, and fell back to Metal anyway.
Delete the archive to reclaim the disk — nothing depends on it.

---

## Where CrispEmbed fits

Part of the Crisp ecosystem, and complementary to llama.cpp (shared ggml backend,
different problem space — retrieval/understanding vs generation):

| Project | Role |
|---|---|
| **CrispEmbed** | This repo — embedding + retrieval + document-AI engine (ggml) |
| **[CrispASR](https://github.com/CrispStrobe/CrispASR)** | Speech recognition (11 ASR backends) + text NMT; shares the ggml core |
| **[crisp-docx](https://github.com/CrispStrobe/crisp-docx)** | `.docx` surgery + document translation; uses CrispEmbed word alignment |
| **[CrispSorter](https://github.com/CrispStrobe/CrispSorter)** | Tauri desktop organiser; LanceDB indexer on CrispEmbed embeddings |

Capabilities llama.cpp does not cover: sparse/ColBERT retrieval, cross-encoder
reranking, the OCR/layout/detection/cleanup/super-resolution document stack, face
detect+recognize, NER/KIE, and a client-side WASM OCR build.

---

## Architecture

Model type is auto-detected from GGUF metadata at load:

- **Encoders** (BERT/XLM-R/MPNet/NomicBERT/ModernBERT/GTE-v1.5/DeBERTa-v2/SPLADE)
  → `src/crispembed.cpp`. Variants detected from tensor names (RoPE vs learned
  positions, rel-attn-bias, pre-LN, fused GeGLU).
- **Decoders** (Qwen3/Gemma3/BidirLM-Omni) → `src/decoder_embed.cpp`.
- **Vision / audio** (BidirLM-Omni) → `src/bidirlm_vision.cpp` /
  `src/bidirlm_audio.cpp`, opened lazily.

The HTTP server exposes four embedding dialects (native, OpenAI, Ollama batch &
legacy) plus face, ViT/CLIP, OCR, NER, LID, KIE, document-OCR, and preprocessing
endpoints. See [PLAN.md](PLAN.md) (roadmap), [HISTORY.md](HISTORY.md) (milestones),
and [LEARNINGS.md](LEARNINGS.md) (deep dives) for detail.

---

## Model licenses

Converting a checkpoint to GGUF **does not relicense it** — each downloaded model
is governed by its **upstream** license. Check the **License** column in
`--list-models` (or the upstream model card) before commercial use.

| License class | Examples | What you can do |
|---|---|---|
| **Permissive** (Apache-2.0 / MIT / CC-BY-4.0) | most BERT/XLM-R/MPNet, BGE, E5, Granite, MXBai, Nomic, Qwen3, Harrier, GTE-v1.5, SMT, TrOMR, Flova, Transcoda (CC-BY-4.0), LiLT | commercial use OK with normal attribution |
| **CC BY-NC 4.0** (non-commercial) | `jina-v5-*`, `jina-reranker-v2`, PosFormer (ours) | research/eval only; commercial needs a vendor license |
| **LFM Open License v1.0** | `lfm2-embed*`, `lfm2-colbert`, `gliner-lfm` | free under $10M annual revenue |
| **Gemma Terms** | `embeddinggemma-300m` | commercial OK, subject to Google's Prohibited Use Policy |
| **Other restricted** | Surya weights (OpenRAIL-M, free < $5M), Texo (AGPL) | see the model card |

Restricted entries are flagged with `*` in `--list-models` and require explicit
consent to auto-download (interactive prompt, `--accept-license <spdx>`, or
`CRISPEMBED_ACCEPT_LICENSE`). `--accept-license` acknowledges the caller accepts
upstream terms — it does not grant rights you don't otherwise have. Audit the
whole registry with `python tests/check_registry_licenses.py`.

**Check the base model, not just the fine-tune's tag.** A fine-tune cannot
relicense what it was built on, and the declared tag is sometimes the author's
optimism. The live trap in this family: **Qwen2.5-VL-3B-Instruct is
`qwen-research`** (research-only), while Qwen2.5-VL-7B and Qwen2-VL-2B are
Apache-2.0. So a "Qwen2.5-VL fine-tune" tagged Apache-2.0 is worth checking
before commercial use — we verified `german-ocr-3.1` from its tensor shapes
(hidden 1536, 28 layers ⇒ Qwen2-VL-2B, Apache-2.0) rather than trusting its
card, which credits a nonexistent "Qwen3.5". Note also that HF stores that
licence in the `license_name` field, not `license`, so a checker that reads only
`license` sees nothing at all.

The VLM OCR engines' upstream licences, all verified against the base chain:

| Registry model | Upstream | Base | Licence |
|---|---|---|---|
| `german-ocr-3.1` | keyvan-ai/german-ocr-3.1 | Qwen2-VL-2B-Instruct | Apache-2.0 |
| `nanonets-ocr2-1.5b` | nanonets/Nanonets-OCR2-1.5B-exp | Qwen2-VL-2B-Instruct | Apache-2.0 |
| `nanonets-ocr-s` | nanonets/Nanonets-OCR-s | Qwen2.5-VL-**7B** | Apache-2.0 |
| `h2ovl-mississippi-800m` | h2oai/h2ovl-mississippi-800m | Danube3-500m + InternViT-300M | Apache-2.0 (+ MIT) |
| `h2ovl-mississippi-2b` | h2oai/h2ovl-mississippi-2b | Danube2-1.8b + InternViT-300M | Apache-2.0 (+ MIT) |

---

## Intended purpose & acceptable use

CrispEmbed is a **component**, not a finished AI system: it returns vectors and
text, and every decision made from them happens in your code. Its intended
purpose is retrieval, document understanding, and document preprocessing.

**[POLICY.md](POLICY.md)** states the intended purpose, the prohibited uses
(no scraping facial-image databases, no emotion inference, no biometric
categorisation, no social scoring), and which obligations transfer to you when
you deploy — biometric processing under GDPR Art. 9, EU AI Act Annex III for 1:N
face identification, Art. 50 transparency for generated or manipulated imagery,
and the limits of OCR accuracy. Read it before building on the face pipeline or
the super-resolution engines.

CrispEmbed ships no emotion-recognition, biometric-categorisation, age, gender,
or ethnicity model, and no scraping tooling — absent by design. That constrains
what ships, not what the code can be aimed at: CLIP and SigLIP score an image
against whatever labels you pass, so the caller supplies the classifier. See
[POLICY.md §3](POLICY.md).

**If you operate it, Art. 4 (AI literacy) is yours.** In force since 2 February
2025, it binds deployers as well as providers, and the open-source exemption
does not reach it. In practice it means whoever runs the thing understands that
OCR output is a reconstruction and not a copy, that a cosine similarity is not
a match decision, that face-recognition error rates vary by demographic group,
and that a restored image is a plausible completion rather than recovered
evidence. [POLICY.md §8](POLICY.md).

### Serving it safely

`crispembed-server` has **no authentication**, and its endpoints read images by
*server-side path* (`{"image": "/path/on/the/server"}`). Bound to loopback
(the default) that is a local tool; bound to a routable address it lets any
client read any file the process can — worst on `/face`, which turns one into a
biometric template.

```bash
# Confine every {"image": ...} read to one subtree (resolves .. and symlinks).
crispembed-server --det yunet.gguf --rec arcface.gguf \
    --image-root /srv/scans --host 0.0.0.0 --accept-biometric
```

Set `--image-root` whenever the port is not loopback-only, and put an
authenticating proxy in front of it. Starting `--rec` on a non-loopback bind
warns about exactly this.

### H2OVL and MSAC

`h2ovl-mississippi-2b` sets `use_msac` (Multi-Scale Adaptive Cropping): it was
trained on a **two-scale** tile stack — a coarse grid plus a finer grid that
divides neither of its axes, concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`.
CrispEmbed implements this, so the model reads pages correctly. It matters
because the failure mode is silent: given ordinary single-scale tiles the model
loads, prefills, and returns confident nonsense rather than erroring. The 800m
sibling sets `use_msac: false` and needs none of it. Geometry is pinned by
`tests/test_msac_tiling.cpp` against the upstream algorithm.

### Provenance on image outputs (EU AI Act Art. 50(2))

Image outputs are **PNG by default** and always carry a machine-readable `iTXt`
provenance chunk naming the engine that touched the pixels — Netpbm has no
metadata container, which is why the format changed:

```bash
crispembed --esrgan-model m.gguf --esrgan in.png > out.png
python -c "from PIL import Image; print(Image.open('out.png').info['CrispEmbed'])"
# generated=true / software=CrispEmbed / engine=esrgan-sr
# digitalSourceType=.../algorithmicallyEnhanced
```

The IPTC term is `algorithmicallyEnhanced`, not `trainedAlgorithmicMedia`: the
input is a real capture we enhanced, and claiming wholly-synthetic media would
be false. `CRISPEMBED_IMAGE_FORMAT=ppm` restores the old raw Netpbm output.

**Content Credentials (C2PA)** are added when a signing identity is configured:

```bash
cmake -S . -B build -DCRISPEMBED_C2PA_FETCH=ON     # pull the c2pa-rs native lib
./scripts/make-c2pa-cert.sh                        # per-installation chain
export CRISPEMBED_C2PA_CERT=~/.config/crispembed/c2pa/cert.pem
export CRISPEMBED_C2PA_KEY=~/.config/crispembed/c2pa/key.pem
```

Full reference: [docs/provenance.md](docs/provenance.md).

**No signing key ships with CrispEmbed, on purpose.** A private key in a public
repo would let anyone mint manifests naming CrispEmbed for images it never
touched. A locally generated chain shows as *unverified signer* — it attests
what was done, not who did it. See [POLICY.md §5](POLICY.md).

### Model integrity

Auto-downloaded GGUFs are **SHA-256 pinned** against
[`examples/cli/model_hashes.h`](examples/cli/model_hashes.h), generated from
HuggingFace's LFS object IDs by `tools/fetch_model_hashes.py`. A payload whose
digest does not match is deleted rather than installed, non-HTTPS URLs are
refused, and an unpinned URL is refused unless you set
`CRISPEMBED_ALLOW_UNPINNED_MODEL=1`. A GGUF is a graph this process executes,
so "the download succeeded" is not an integrity statement. After adding a model
to the registry, re-run:

```bash
python tools/fetch_model_hashes.py          # refresh pins
python tools/fetch_model_hashes.py --check  # CI: fail if stale
```

---

## License

CrispEmbed's own code is **MIT** (see [`LICENSE`](LICENSE)), consistent with its
ggml/llama.cpp foundation.

Per-model **weights** are covered by their respective upstream/HuggingFace model
licenses (see [Model licenses](#model-licenses) and `--list-models`) — converting
a checkpoint to GGUF does not change its license. The `crispembed` binary itself
links model runtimes that are mostly permissively licensed (MIT / Apache-2.0 /
CC-BY-4.0 for weights); a few registry models carry non-commercial or
vendor-specific terms and are flagged accordingly.

---

## Credits

- [ggml](https://github.com/ggml-org/ggml) — inference engine
- [CrispASR](https://github.com/CrispStrobe/CrispASR) — shared core (gguf_loader, bpe, crisp_audio)
- [sentence-transformers](https://www.sbert.net/) — ground-truth validation
- The upstream model authors — see each model's card for architecture credit
