---
library_name: crispembed
license: apache-2.0
tags: [ocr, pp-ocrv6, gguf, crispembed]
---

# PP-OCRv6 medium recognizer — CrispEmbed GGUF

Files: `PP-OCRv6_medium_rec-f16.gguf` and `PP-OCRv6_medium_rec-crispasr-q4_k-policy.gguf`.

The policy-q4 container intentionally keeps the complete PP-OCRv6 recognizer graph in F16: quantizing intermediate CNN/SVTR weights caused compounding CTC drift. Source: PaddlePaddle PP-OCRv6, Apache-2.0.

The native F32 conversion matches the reference through logits (cosine 1.0). The published F16 artifact is an inference-speed artifact, not a parity claim: repeated half-precision layers accumulate measurable drift, so production quality should use the Q8/F32-quality policy artifact once published.
