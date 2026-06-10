---
title: CrispEmbed
sdk: docker
app_port: 7860
pinned: false
---

# CrispEmbed — Text, Image, Face & Math OCR Embedding Demo

Lightweight embedding inference via ggml. No Python runtime, no ONNX.

**Text Embeddings**: Dense, sparse (SPLADE/BGE-M3), ColBERT multi-vector,
cross-encoder reranking. 10 encoder/decoder architectures, 58 models.

**Math OCR**: Six engines for math-image → LaTeX (printed + handwritten).

**Image Embeddings**: CLIP/SigLIP cross-modal text-image search.

**Face**: Detection (SCRFD/YuNet) + Recognition (ArcFace/AuraFace/SFace).

Powered by the [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
C++ engine. Models auto-download on first use.
