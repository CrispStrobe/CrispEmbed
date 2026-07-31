---
library_name: crispembed
license: apache-2.0
tags: [ocr, pp-ocrv6, gguf, crispembed]
---

# PP-OCRv6 tiny detector — CrispEmbed GGUF

Files: `PP-OCRv6_tiny_det-f16.gguf` and `PP-OCRv6_tiny_det-crispasr-q4_k-policy.gguf`.

The detector path is kept at F16 in the q4-policy artifact for output parity. `crispembed-diff` final probability-map cosine: **0.999997616**.

Source: PaddlePaddle PP-OCRv6, Apache-2.0. Large files are generated and tested with CrispEmbed’s PP-OCRv6 runtime.
