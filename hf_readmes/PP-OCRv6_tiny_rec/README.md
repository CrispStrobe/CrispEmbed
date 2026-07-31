---
library_name: crispembed
license: apache-2.0
tags: [ocr, pp-ocrv6, gguf, crispembed]
---

# PP-OCRv6 tiny recognizer — CrispEmbed GGUF

Files: `PP-OCRv6_tiny_rec-f16.gguf`, `PP-OCRv6_tiny_rec-f32.gguf`,
`PP-OCRv6_tiny_rec-q8-head.gguf`, and
`PP-OCRv6_tiny_rec-crispasr-q4_k-policy.gguf`.

The policy-q4 container intentionally keeps the complete PP-OCRv6 detector/recognizer graph in F16: quantizing intermediate CNN/SVTR weights caused compounding CTC drift. Source: PaddlePaddle PP-OCRv6, Apache-2.0.

Parity on `tests/regression/images/fox.png` using the CrispEmbed diff harness: input 0.999999, stage4 0.999982, head_input 0.999988, logits 0.999992 (F16 and policy-q4; all reported stages pass the 0.999 threshold).

The tiny head-only-Q8 artifact is generated from F32 and retains all
quality-sensitive tensors at F32; its CTC head is not Q8-eligible because its
input width is 80 rather than a 32-multiple.
