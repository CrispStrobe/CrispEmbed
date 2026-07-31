---
library_name: crispembed
license: apache-2.0
tags: [ocr, pp-ocrv6, gguf, crispembed]
---

# PP-OCRv6 medium recognizer — CrispEmbed GGUF

Files: `PP-OCRv6_medium_rec-f16.gguf` and `PP-OCRv6_medium_rec-crispasr-q4_k-policy.gguf`.

The policy-q4 container intentionally keeps the complete PP-OCRv6 recognizer graph in F16: quantizing intermediate CNN/SVTR weights caused compounding CTC drift. Source: PaddlePaddle PP-OCRv6, Apache-2.0.
