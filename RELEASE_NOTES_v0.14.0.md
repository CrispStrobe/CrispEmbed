# CrispEmbed v0.14.0

The browser release. Since v0.13.0 (two days, unusually dense): the WASM OCR
stack went from "never actually ran in a browser" to a fully verified,
GPU-accelerated client-side demo with 12 validated engines — plus
importance-matrix quantization rolled out across the embedding roster, a
sweep of GPU-teardown/residency crash classes, and reranker packaging fixes.

Live demo: https://crispstrobe.github.io/CrispEmbed/ — fully client-side,
models stream from Hugging Face and cache locally.

## Browser / WebAssembly (issue #31)

The `wasm_ocr` integration reported broken in #31 was broken in five
independent ways; all fixed, and every step is now verified end-to-end by a
headless-Chromium test suite that runs in CI on every push (output checked
byte-for-byte against the native CLI).

- **Root cause, engine**: a use-after-free — the cached encoder graph's
  metadata pool was a stack-local buffer, and Emscripten's allocator reused
  the freed block as the compute work buffer, corrupting tensor structs.
  This was the long-standing "ViT models exceed WASM limits" crash
  (pix2tex, TrOCR) and an occasional native segfault. The fragile graph
  cache is gone entirely (rebuild is microseconds).
- **Integration fixes**: heap views (`HEAPU8` …) are now exported (modern
  Emscripten no longer attaches them — the JS wrapper's `recognize()` threw
  in every browser); the demo's default model URLs pointed at nonexistent
  HF repos; `serve.py` forced COEP unconditionally.
- **Web Worker architecture**: all inference runs off the main thread with
  live engine progress ("recognizing region 3/8", elapsed ticker). The page
  never freezes; an explicit **Process** button decouples image and model
  selection (either order).
- **Four runtime tiers, auto-selected**: SIMD CPU → multithreaded
  (COOP/COEP via a bundled service worker — works on GitHub Pages) →
  **WebGPU (JSPI)** → **WebGPU (Asyncify)** for JSPI-less browsers.
- **WASM SIMD was silently off**: under `emcmake`,
  `CMAKE_SYSTEM_PROCESSOR` is `x86`, so ggml compiled generic scalar quant
  kernels. `-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` enables the real SIMD128
  kernels (~2× on quantized matmuls).
- **WebGPU backend in the browser** (experimental, Chromium + Safari 26
  class): six WGSL kernels added to our ggml port
  (`patches/ggml-webgpu-ops.patch`; upstream draft in CrispASR
  `tools/upstream-prs/22`) — LayerNorm, IM2COL, POOL_2D,
  CONV_TRANSPOSE_2D, UPSCALE (nearest+bilinear), ARANGE — putting the full
  OCR graph on GPU. The backend's silent skip of unhandled ops (garbage
  with no diagnostic on sched-less paths) now warns.
- **Verified engine matrix**: 12 OCR engines produce correct text in the
  browser on both CPU and WebGPU (each A/B'd, ggml's `test-backend-ops`
  executed in headless Chromium for the kernels — 241 cases):

  | Engine | wasm CPU | WebGPU | |
  |---|---|---|---|
  | TexTeller-3 (177 MB) | 29.2 s | 5.5 s | 5.4× |
  | TrOCR small printed | 6.9 s | 1.7 s | 4.0× |
  | PP-FormulaNet-L | 113 s | 43 s | 2.6× |
  | pix2tex | 6.3 s | 2.4 s | 2.6× |
  | DBNet detection (pipeline) | ~90 s | ~1.5 s | ~60× |
  | HMER / BTTR / PosFormer / Texo / MixTeX / PARSeq / Tesseract-LSTM / TrOCR-handwritten | — | — | correct, parity |

  Autoregressive decode is steered to CPU on the WebGPU tiers
  (`MATH_OCR_DEC_CPU=1`) — per-token GPU submit overhead made GPU decode
  ~5× slower than CPU decode.
- **Model picker + OPFS cache**: the demo has a grouped preset picker (13
  verified presets) with a free-text URL override; downloaded GGUFs cache
  in origin-private storage — revisits load models with zero network
  (~150 ms).
- **Release/CI plumbing**: `release-wasm.yml` could never trigger
  (`on: release` doesn't fire for token-created releases) — it now runs on
  version tags and attaches `crispembed-ocr-wasm.tar.gz` +
  `crispembed-embed-wasm.tar.gz`; `deploy-pages.yml` publishes the demo on
  every push to `main`; `build-wasm.yml` runs the node suites and the
  browser e2e.

## Embeddings: imatrix quantization rollout

- **31 embedders re-quantized with importance matrices** (calibrated on a
  CC0 Common Voice EN+DE corpus after A/B showed English-only calibration
  regresses); registry defaults switched per model to the best measured
  flavor. Collector wired through sched-based engines (GLiNER needed an
  opt-in sched + flush fix — `clean_exit` bypassed `atexit`).
- **imatrix for rerankers, ColBERT, sparse**: lfm2-colbert q4_k+im 0.9975;
  splade-pp iq4_xs 0.996 (converter bug fixed); gliner-deberta iq4_xs
  span-F1 1.0.
- **Reranker packaging**: bge-reranker-base was shipped **headless**
  (missing classifier) — reconverted and re-uploaded; DeBERTa rel-embd
  read crashed q8_0/q4_k rerankers (now dequantized); all reranker
  registry defaults moved to Q8_0 after finer-grained A/B (4-bit reorders
  the ranking tail).
- **Batched encoder throughput (C3)**: packed block-diagonal and padded-4D
  batch paths (opt-in) for absolute-position encoders; ModernBERT
  validated end-to-end; EmbeddingGemma verified.
- **bidirlm-omni 2.5B**: fully-multimodal imatrix quants (q4_k/q5_k, with
  and without audio tower); vision-tower imatrix wired; mel filterbank
  kept F32.

## GPU crash classes fixed

- **ggml v0.10.0 teardown regressions**: the submodule bump changed two
  runtime contracts (backend-buffer lifetime, sched cross-backend
  resolution) that crashed GPU runs across engines; all call sites fixed.
- **Metal residency aborts** swept across 9 conv-front-end engines
  (weights on `init_best` while the conv sched is CPU — referencing the
  GPU leaf aborted graph alloc on Metal / segfaulted on CUDA).
- **Device-pointer weight reads** fixed across 8 engines (host `memcpy`
  from GPU-resident tensors — crashed on discrete-memory CUDA).
- **lfm2-colbert CUDA multivec**: graph-reuse corruption fixed,
  P100-verified.
- **InternVL2 multi-tile**: cached-graph/shared-sched crash fixed with a
  fresh graph per tile.

## Regression infrastructure

- **Gap-4 guardrails**: five previously untested engines now have standing
  diff tests against independent references (surfaced three real shipped
  bugs, all fixed).
- **Manifest-driven GGUF head audit**: a repo-wide gate that catches
  headless/incomplete shipped GGUFs (the class behind the bge-reranker
  incident) — only that one repo was broken.
- **Mandatory A/B rule** (after the DeepSeek-OCR-2 perf-sweep regression):
  perf changes keep both paths env-gated until decoded output is proven
  equal; regression manifest entries required.
- **Browser e2e in CI**: every push builds the wasm tiers and drives the
  real demo page in headless Chromium against native ground truth — beyond
  what upstream ggml projects run.

## Punctuation / misc

- FireRedPunc + PCS punctuation models: imatrix quants shipped; fine-grained
  A/B hook (`$FIREREDPUNC_DUMP_LOGITS`).
- `tools/upstream-prs/22`: WGSL kernel PR draft for ggml-org (NORM, IM2COL,
  POOL_2D, CONV_TRANSPOSE_2D, UPSCALE, ARANGE + skip-warning).
- Ecosystem survey of browser inference stacks (wllama, whisper.cpp,
  transformers.js, web-llm …) recorded in `LEARNINGS.md`; OPFS caching and
  the compat-build pattern adopted from it (MIT sources).

## Compatibility notes

- WASM: the WebGPU tiers require `navigator.gpu` (Chromium; Safari 26
  class devices). JSPI build for `WebAssembly.Suspending` browsers,
  Asyncify build otherwise. The threaded tier needs cross-origin
  isolation (provided by the bundled `coi-sw.js` on static hosts).
- Native: no breaking API changes. New env knobs: `MATH_OCR_DEC_CPU`,
  `OCR_DETECT_USE_GPU` (existing), `CRISPEMBED_ENCODER_PACKED`,
  `CRISPEMBED_MATH_OCR_BENCH`.
