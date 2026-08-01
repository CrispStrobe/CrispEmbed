---
title: CrispEmbed
sdk: docker
app_port: 7860
pinned: false
---

# CrispEmbed — Text Embedding & Math OCR Demo

Lightweight embedding inference via ggml. No Python runtime, no ONNX.

This Space demonstrates four things:

**Similarity**: cosine similarity between two texts.

**Semantic Search**: rank a small corpus against a query.

**Math OCR**: math image → LaTeX.

**Batch Embed**: OpenAI-compatible `/v1/embeddings` batch endpoint.

This Space processes **text and math images only**. It does no face detection,
no face recognition, and no biometric processing of any kind. An image uploaded
to the Math OCR tab is written to a temporary file, transcribed, and deleted by
the application immediately afterwards; it is not stored or logged by this app.
(The Gradio/Hugging Face platform layer may cache uploads independently of the
application.)

The upstream engine additionally supports image/face embeddings, full-page OCR,
layout analysis and NER; those are not exposed here. See
[POLICY.md](https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md) for
intended purpose and acceptable use.

Powered by the [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
C++ engine. Models auto-download on first use.
