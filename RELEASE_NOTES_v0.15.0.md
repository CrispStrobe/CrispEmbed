# CrispEmbed v0.15.0

190+ commits since v0.14.0. The headline is a **new modality — Optical Music Recognition (OMR)** — landing as four engines (printed, polyphonic, handwritten, and zero-shot full-page sheet music → symbolic notation). Alongside it: a math-OCR engine that had been **shipping garbage was root-caused and fixed**, the llama.cpp interop became **bidirectional**, a broad **performance sweep** touched detection, encoders, decoders, super-resolution, and layout, and the **OCR regression suite** was hardened with real end-to-end guards (including a mechanism for license-restricted fixtures).

## Optical Music Recognition (OMR) — new capability

"OCR for staff notation." Three permissively-licensed engines, each a vision encoder + autoregressive decoder emitting a linearized notation sequence, wired end-to-end (converter → GGUF → CLI `--ocr` auto-detect → registry auto-download → WASM):

- **Sheet Music Transformer (SMT)** — ConvNeXt encoder + cross-attention Transformer decoder → bekern. Printed pianoform/grand-staff. **96.3% on the GrandStaff test split** once the preprocessing was corrected (a mistaken color-invert had capped it at ~30%). KV-cached decode is 5.4× faster and token-identical. `cstr/smt-grandstaff-GGUF` (MIT).
- **Polyphonic-TrOMR** — ResNetV2 + hybrid ViT encoder → x-transformers decoder with four parallel rhythm/pitch/lift/note heads → symbolic text. Validated byte-exact vs the reference (cos 1.0 / 100% argmax, CPU==Metal). `cstr/tromr-GGUF` (Apache-2.0, q8_0 with an F16 backbone).
- **Flova/omr_transformer** — Donut (DonutSwin + 4-layer mBART) VED → LilyPond. Handwritten/whiteboard "simple notes." Byte-exact incl. native preprocessing. `cstr/flova-omr-GGUF` (Apache-2.0, f32 + q8_0 + a browser-sized q4_k).
- **Transcoda-59M** — a **clean-room** port (weights CC-BY-4.0; engine written from paper + config + oracle only) of a ConvNeXt-V2-tiny encoder + 8-layer RoPE cross-attention decoder → full-page Humdrum `**kern`. Per-stage cos 1.0; a repetition-penalty subtlety (HF applies it once per *unique* token, not per occurrence) was the last mile to matching greedy decode.

The printed SMT engine also gained **SMT++ full-page** support. The OMR engines are exported to the WASM build and carry byte-exact regression fixtures.

## Math OCR — TexTeller was broken, now fixed

TexTeller (ViT + TrOCR) had shipped producing **garbled LaTeX** — validated only per-stage (encoder cos ~0.999), never on the decoded output. Two pix2tex-specific constants were hardcoded in the shared `math_ocr` engine:

- **Decoder FFN activation was hardcoded ReLU**; TexTeller's TrOCR decoder uses **GELU**. Wrong activation on every FFN layer drifted the decode into garbage.
- **Input preprocessing** hardcoded pix2tex's mean/std 0.5 + squash-resize; TexTeller needs its own grayscale normalization + trim/aspect/white-pad transform.

Both are now **data-driven from GGUF metadata** (`decoder.activation_function`, `encoder.image_mean/std/preprocess_pad`), with defaults that reproduce the pix2tex path byte-for-byte. Native decode is now byte-exact vs the HF reference on both formula fixtures (f16 == q8_0, CPU == Metal); `cstr/texteller-3-GGUF` was re-uploaded with the metadata. A converter-level `decoder_start_token_id` + empty-vocab fix landed for the same VED family.

## llama.cpp interop — now bidirectional

- **Export**: emit a CrispEmbed Qwen2-VL vision tower as a stock llama.cpp `mmproj` (bidirectional round-trip verified).
- **Import**: load stock llama.cpp `mmproj` VL models into CrispEmbed — SmolVLM, InternVL2.5/3, and Qwen2-VL — via a unified auto-detecting merge dispatcher (`models/merge-llamacpp-gguf.py`) with a shared merge core and projector-type routing. Two latent interop bugs (a silently-dropped image; fc1/fc2 role mapping) were caught by a new round-trip regression.

## Performance

A broad sweep, each change A/B'd against ground truth and env-gated:

- **Detection**: DBNet postprocess rewritten as scanline box scoring — **28× faster**, byte-identical.
- **Layout (RT-DETRv2)**: im2col+GEMM backbone conv (**~9.8× Phase-1**, default flipped) + threaded decoder matmuls.
- **Encoders**: default **packed batching on GPU backends** (5–7× on Metal, cos 1.0).
- **Decoders**: persistent device-side KV caches (math_ocr, deepseek_ocr2) and sched-free decode-step graph caches (got-ocr, internvl2, glm-ocr, lightonocr) that skip per-step host build+alloc; mixtex decoder weight-dequant hoisted out of the step loop (~2.9×) + threaded Swin window-attention (1.94×).
- **Super-resolution / restoration**: fused single-graph forwards — **SAFMN 2.2× (and more accurate, cos 1.0)**; NAFNet / InstructIR / Restormer fused (compute-bound → perf-neutral but cleaner); n_threads honored across the SR family.
- **SIMD GEMM**: GLiNER DeBERTa rel-pos dedup + batched matmuls (1.28–1.71×), SCUNet Swin-MLP (1.69×), tps_locnet weight precompute.

## OCR regression suite — real end-to-end guards

The suite that exists because a vision-neck permute regression once shipped garbage OCR undetected got materially stronger:

- **New fixtures with pinned golden text**: the OMR trio (SMT/TrOMR/Flova), pix2tex-mfr, TexTeller, and the three handwritten-math CROHME models (bttr/hmer/posformer) — closing every math/OCR `expected_text: null` gap.
- **`sample_hf`**: a new manifest mechanism that fetches a fixture image from an HF **dataset** parquet at a pinned revision at test time, so license-restricted images (CROHME = CC-BY-NC-SA) stay out of this MIT/Apache repo while still giving an in-domain text-match guard.
- **Harness robustness**: the diff-parser now accepts every diff-binary output format (ANSI / colon-less / table); teardown-crash-after-valid-stages is tolerated; an absent optional test binary SKIPs instead of failing.
- **CUDA confirmation + older-arch hardening**: the full portfolio was re-run on Kaggle T4/P100 — the Class-A device-pointer and Gap-5 teardown fixes confirmed FAIL→PASS. Follow-up P100 diagnostics overturned several assumptions and drove targeted fixes (e.g. a manual-attention fallback where CUDA flash-attn aborts on Pascal / sm_60); the remaining FAILs are known older-arch vision divergences, not regressions.

## WASM

- The OMR engines (SMT/TrOMR/Flova) are exported to the browser build.
- A multithreaded, deadlock-free in-browser OCR pipeline via `PROXY_TO_PTHREAD`, with a no-deadlock regression test.

## Other fixes & improvements

- **Deskew**: consensus skew detection (Hough + DSS sign/band gate) applied on all image paths.
- **EmbeddingGemma**: Matryoshka `matryoshka_dim` exposed in the Python `encode()`; registry pooling label corrected (last-token → mean) on four entries.
- **Tokenizer**: `add_bos`/`add_eos` GGUF flags honored across the remaining paths; `no_repeat_ngram` blocking in the math decoder (kills `TOOO→TOO` loops).
- **Backend**: `--gpu-backend` preference threaded through every engine.
- **Fixes**: surya grouped-pointwise-conv graph crash; gated debug spew behind flags (surya/layout/detect); Kaggle download/repo-name/ENOSPC fixes.
- **Build**: both repos now consume the shared `CrispStrobe/ggml` fork (v0.10.2 + our CPU/CUDA/Metal/Vulkan/WebGPU ops) as a submodule.

## Reranker imatrix — a caution documented

A full 7-reranker τ-evaluation (n=30 EN+DE graded corpus) found that imatrix **always** reduces q4_k score-drift but its effect on ranking **τ is model-dependent** — a large win on ms-marco-L-12 and jina, neutral on bge, but a **regression on both mxbai rerankers** (where iq4_xs beats q4_k+imatrix). `q4_k+imatrix` is therefore not a universal reranker recommendation; validate per model.

---

**Full log**: `git log v0.14.0..v0.15.0`. Deep technical notes live in `LEARNINGS.md`; completed milestones in `HISTORY.md`.
