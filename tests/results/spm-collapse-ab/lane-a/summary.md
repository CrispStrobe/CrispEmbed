### SPM collapse A/B (7b5b5cc9, 77 texts)

| model | file | texts | ids = HF fast on / off | ids = SentencePiece on / off | cos min on / off (embed) or max abs score diff on / off (rerank) |
|---|---|---:|---|---|---|
| multilingual-e5-small | multilingual-e5-small-q8_0.gguf | 76 | 71 / 66 | 76 / 66 | 0.98524 / 0.97478 |
| multilingual-e5-base | multilingual-e5-base-q8_0.gguf | 76 | 76 / 66 | 76 / 66 | 0.99873 / 0.97826 |
| multilingual-e5-large | multilingual-e5-large.gguf | 76 | 76 / 66 | 76 / 66 | 1.00000 / 0.98211 |
| paraphrase-multilingual-MiniLM-L12-v2 | paraphrase-multilingual-MiniLM-L12-v2-q8_0.gguf | 76 | 76 / 66 | 76 / 66 | 0.65650 / 0.65650 |
| xlmr-ner-hrl | xlmr-ner-hrl-iq4_xs.gguf | 76 | 12 / 12 | 12 / 12 | - |
| granite-embedding-107m | granite-embedding-107m-multilingual-iq4_xs.gguf | 76 | 71 / 66 | 76 / 66 | 0.98741 / 0.97334 |
