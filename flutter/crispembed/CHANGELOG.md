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
