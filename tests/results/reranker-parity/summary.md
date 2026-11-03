### Reranker parity (0fb9e738, portable, 77 texts)

| model | file | HF pair layout | max abs Δ default / sep=0 | mean abs Δ default / sep=0 | order equal default / sep=0 | worst stage (default) |
|---|---|---|---|---|---|---|
| bge-reranker-v2-m3 | bge-reranker-v2-m3-q8_0.gguf | `<s> ▁a ▁b </s> </s> ▁c ▁d </s>` | 0.2290 / 1.8595 | 0.0084 / 0.0560 | False / False | None nan (len None/None) |
| bge-reranker-base | bge-reranker-base-q8_0.gguf | `<s> ▁a ▁b </s> </s> ▁c ▁d </s>` | 0.1762 / 1.6020 | 0.0075 / 0.1037 | False / False | None nan (len None/None) |
| jina-reranker-v2-base-multilingual | jina-reranker-v2-base-multilingual-q8_0.gguf | `<s> ▁a ▁b </s> </s> ▁c ▁d </s>` | 0.0261 / 0.1373 | 0.0059 / 0.0207 | False / False | None nan (len None/None) |
| mxbai-rerank-xsmall-v1 | mxbai-rerank-xsmall-v1-q8_0-g7c.gguf | `[CLS] ▁a ▁b [SEP] ▁c ▁d [SEP]` | 0.0650 / 0.0650 | 0.0228 / 0.0228 | False / False | None nan (len None/None) |
| mxbai-rerank-base-v1 | mxbai-rerank-base-v1-q8_0-g7c.gguf | `[CLS] ▁a ▁b [SEP] ▁c ▁d [SEP]` | 0.0787 / 0.0787 | 0.0197 / 0.0197 | False / False | None nan (len None/None) |
| ms-marco-MiniLM-L-6-v2 | ms-marco-MiniLM-L-6-v2-q8_0-g7c.gguf | `[CLS] a b [SEP] c d [SEP]` | 0.0484 / 0.0484 | 0.0049 / 0.0049 | False / False | None nan (len None/None) |
| ettin-reranker-17m | error | ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported. | | | | |

| encoder | HF truncated len | default rc / len / ids = HF | POS_EMBD_TRUNCATE=0 rc | cos vs ST (default) |
|---|---:|---|---|---|
| multilingual-e5-small | 512 | 0 / 512 / True | 0 | 0.9999189041932519 |
| multilingual-e5-base | 512 | 0 / 512 / True | -6 | 0.9989851932775721 |
| multilingual-e5-large | 512 | 0 / 512 / True | -6 | 0.9999999999572801 |
| granite-embedding-107m | 512 | 0 / 512 / True | -6 | 0.9906887322662787 |
| granite-embedding-278m | 512 | 0 / 512 / True | -6 | 0.994570923171504 |
| paraphrase-multilingual-MiniLM-L12-v2 | 512 | 0 / 512 / True | 0 | 0.6117490992396875 |
| bge-reranker-base | 512 | 0 / 512 / True | -6 | None |
| xlmr-ner-hrl | 512 | 0 / 512 / False | -6 | None |
| all-MiniLM-L6-v2 | 512 | 0 / 512 / True | 0 | 0.9711914432481936 |
