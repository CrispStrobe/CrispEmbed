## 0.17.15 (unreleased)

- Fix film grain blocking real-background propagation in the opt-in CLI temporal
  inpainting stream; retain motion and brightness-change guards.
- Add a default-off LightOnOCR-3-1B experiment using the original Pixtral vision
  norm epsilon. Existing profiles, text/projector norms and default behavior
  remain unchanged. Hosted retained-reference A/B qualification is pending.
- Correct the LightOn3 loader/converter checkpoint namespace and OCR audit setup;
  retain original failures and bounded precision/decoded results separately.
- No new runtime assets, model aliases, application pin or pub.dev publication.

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
