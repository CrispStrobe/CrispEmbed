### SPM collapse A/B (7b5b5cc9, 77 texts)

| model | file | texts | ids = HF fast on / off | ids = SentencePiece on / off | cos min on / off (embed) or max abs score diff on / off (rerank) |
|---|---|---:|---|---|---|
| pixie-rune-v1 | pixie-rune-v1-q8_0.gguf | 77 | 72 / 67 | None / None | 0.99654 / 0.96830 |
| arctic-embed-l-v2 | arctic-embed-l-v2-q8_0.gguf | 77 | 72 / 67 | 77 / 67 | 0.99405 / 0.96164 |
| arctic-embed-m-v2 | arctic-embed-m-v2-q8_0.gguf | 77 | 77 / 67 | None / None | nan / nan |
| nomic-embed-text-v2-moe | nomic-v2-moe-q8_0.gguf | 76 | 76 / 66 | 76 / 66 | 0.99820 / 0.91759 |
| granite-embedding-278m | granite-embedding-278m-multilingual-q4_k-imatrix.gguf | 76 | 76 / 66 | 76 / 66 | 0.99267 / 0.98401 |
