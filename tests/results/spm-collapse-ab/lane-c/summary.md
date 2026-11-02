### SPM collapse A/B (7b5b5cc9, 77 texts)

| model | file | texts | ids = HF fast on / off | ids = SentencePiece on / off | cos min on / off (embed) or max abs score diff on / off (rerank) |
|---|---|---:|---|---|---|
| bge-m3 | bge-m3-q8_0.gguf | 77 | 72 / 67 | 77 / 67 | 0.99843 / 0.96845 |
| bge-reranker-v2-m3 | bge-reranker-v2-m3-q8_0.gguf | 77 | 77 / 67 | 77 / 67 | 1.85951 / 1.85951 |
| bge-reranker-base | bge-reranker-base-q8_0.gguf | 76 | 71 / 66 | 76 / 66 | 1.60198 / 1.60198 |
| jina-reranker-v2-base-multilingual | jina-reranker-v2-base-multilingual-q8_0.gguf | 77 | 77 / 67 | None / None | 0.13726 / 0.16352 |
| mxbai-rerank-xsmall-v1 | mxbai-rerank-xsmall-v1-q8_0-g7c.gguf | 77 | 77 / 67 | 77 / 67 | 0.06501 / 0.33449 |
| mxbai-rerank-base-v1 | mxbai-rerank-base-v1-q8_0-g7c.gguf | 77 | 77 / 67 | 77 / 67 | 0.07865 / 0.20354 |
