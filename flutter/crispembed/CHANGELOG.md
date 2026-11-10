## 0.17.18

- Native library update: experimental LightOn4B OCR code, default-off and
  env-gated, not qualified, no new model alias. Unqualified LightOn4B GGUF
  profiles are rejected at OCR model load.
- Android: no x86 CPU-feature probe in mobile builds.

## 0.17.17

- Add phone-photo preparation APIs (projective crops, measured rotation,
  validated corners) and exact registry model metadata.
- Add bounded reviewed-PDF writers and PDF/A-2b archival output. Right-to-left
  text in PDF text layers is still refused safely (capabilities omit BIDI).
- Add per-call licence acceptance: `resolveModelAccepting` and
  `modelLicenseInfo`.
- Add resumable, source-linked document extraction with checkpoints.
- iOS pod: keep the `-force_load` path quoted so plugin paths with spaces link.

## 0.17.16

- Add versioned OCR review snapshots (geometry and confidence provenance) and
  checked OCR document rendering (searchable PDF with lossless raster, Unicode
  text and per-page physical size; hOCR/ALTO checks; safe failure on
  unsupported bidi). Existing entry points are unchanged.
- Add default-off PP-OCRv6 detector options to accept valid empty graph results
  and rebuild the persistent graph when video frames change size. Preserve the
  scalar reference fallback and comparison mode.
- MinerU2.5-Pro's full-document route is on by default for F16/F32 checkpoints
  and stays opt-in for quantized GGUFs.
- Register experimental, SHA-pinned GOT-OCR2 Q8_0 / Q8+Q4_K and
  Unlimited-OCR-2 Q8_0 aliases; defaults are unchanged.

## 0.17.15

- Fix film grain blocking real-background propagation in the opt-in CLI temporal
  inpainting stream; retain motion and brightness-change guards.
- Add default-off LightOnOCR-3-1B corrections for the original Pixtral vision
  norm epsilon and an F32 CPU KV cache. Together they match all twelve retained
  F32/F16 free outputs; strict intermediate failures and capped outputs remain.
  Legacy profiles and runtime defaults remain unchanged.
- Correct the LightOn3 loader/converter checkpoint namespace and OCR audit setup;
  retain original failures and bounded precision/decoded results separately.
- No new model aliases, application pin changes or pub.dev publication.

## 0.17.14

- Native libraries track the v0.17.14 release. Downstream apps can drop
  `CRISPEMBED_PREBUILT_VERSION`; Android builds from `main` fetch matching
  libraries again.
- macOS artifacts install `CrispEmbed`/`ggml` dylibs under unversioned install
  names, so bundled binaries resolve without `DYLD_*` overrides.
- Measured default flips since 0.17.12 (EasyOCR state/line recovery, GOT-OCR2
  pipeline parity, SPM space collapse, XLM-R pair layout, position-table
  truncation, CLIP EOS, qualified CUDA/Metal engine defaults, MiniCPM-OCR and
  Unlimited-OCR-2 speed paths) each keep a `=0` opt-out documented in
  `docs/environment-variables.md`.
- New MinerU2.5-Pro-2605, LightOnOCR-3-1B and Xiaomi-OCR-0 profiles are
  default-off and unqualified; the app-facing engine set is unchanged.
- Server: usage/metrics accounting, request limits (429) and an opt-in
  OpenAI-shaped `POST /v1/chat/completions` + `GET /v1/models` surface.

## 0.17.9

- Add native MOST Embed DE support with exact Tekken tokenization, bidirectional
  Ministral3/YaRN execution, retrieval prompts, and gated CC-BY-NC-4.0 downloads.
- Fix mmap-backed GGUF weight ownership and batch tokenizer override precedence.
- Expand measured embedding/reranker language coverage to Arabic and Korean.

## 0.16.1

- Native libraries are now fetched from the CI release on install (no manual
  placement); iOS/Android build only the `crispembed` library target.
- Server gains `POST /rerank` (cross-encoder) and `POST /sparse` (SPLADE/BGE-M3)
  routes; `/health` reports `reranker`/`sparse`/`colbert` capabilities.
- First pub.dev release carrying the v0.16.x native library (0.16.0 shipped as
  GitHub release binaries only).

## 0.16.0

- Community & official GGUF interoperability: load third-party llama.cpp / Ollama
  GGUFs across BPE/WordPiece/SentencePiece tokenizers (ModernBERT, EmbeddingGemma,
  LFM2.5-Embedding, Qwen3-Embedding, granite-embedding, GTE-v1.5, nomic-v2-moe).
- SPLADE-v3 sparse retrieval model; escaping-aware JSON server input; community-GGUF
  regression matrix with HF ground-truth parity.

## 0.15.1

- Docs/quality: add an example, and enable the flow-control brace lint with
  the corresponding fixes (pub.dev static-analysis + example points).

## 0.15.0

- Initial pub.dev release of the CrispEmbed Flutter/Dart FFI package.
- Exposes bindings for embeddings, reranking, OCR/OMR, layout detection, scan cleanup and document-processing helpers.
- Supports desktop, mobile and FFI-based native library loading.
