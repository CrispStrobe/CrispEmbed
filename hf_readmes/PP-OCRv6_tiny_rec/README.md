---
library_name: crispembed
license: apache-2.0
tags: [ocr, pp-ocrv6, gguf, crispembed]
---

# PP-OCRv6 tiny recognizer — CrispEmbed GGUF

Files: `PP-OCRv6_tiny_rec-f16.gguf` and `PP-OCRv6_tiny_rec-crispasr-q4_k-policy.gguf`.

The policy-q4 container intentionally keeps the complete PP-OCRv6 detector/recognizer graph in F16: quantizing intermediate CNN/SVTR weights caused compounding CTC drift. Source: PaddlePaddle PP-OCRv6, Apache-2.0.

Parity on `tests/regression/images/fox.png` using the CrispEmbed diff harness: input 0.999999, stage4 0.999982, head_input 0.999988, logits 0.999992 (F16 and policy-q4; all reported stages pass the 0.999 threshold).
