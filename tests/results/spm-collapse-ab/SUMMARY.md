# CRISPEMBED_SPM_COLLAPSE_SPACES A/B against the original HF models (3 November 2026)

Harness: `tools/ci-heavy/spm_collapse_ab.py` (heavy-cpu.yml, ubuntu-latest). The panel is the below-bar panel
(13 CUDA-audit texts plus the 65-text eval corpus, 77 non-empty texts, no prefix). Inputs that the original
truncates are left out (column "texts"), because the native path aborts on them in both arms (see Open).

The models are every registry model whose GGUF resolves to the SentencePiece family on the encoder path. They
were found by reading `tokenizer.ggml.type`, `tokenizer.ggml.model` and `general.architecture` from the header of
one registry file per Hub repo (161 repos). Columns:

- **HF fast**: the HF tokenizer.json via `tokenizers` 0.22 and transformers 4.x.
- **SentencePiece**: the model's own `sentencepiece` model (slow tokenizer).
- **cos**: final-embedding minimum cosine against sentence-transformers fp32, batch 1.
- **Δscore**: maximum absolute difference against the HF sequence-classification logit, query
  "What did Einstein publish in 1905?".

| model | texts | IDs = HF fast, on / =0 | IDs = SentencePiece, on / =0 | cos min on / =0 (rerank: max abs Δscore) |
|---|---:|---|---|---|
| multilingual-e5-small | 76 | 71 / 66 | **76 / 66** | 0.98524 / 0.97478 |
| multilingual-e5-base | 76 | **76 / 66** | **76 / 66** | 0.99873 / 0.97826 |
| multilingual-e5-large | 76 | **76 / 66** | **76 / 66** | 1.00000 / 0.98211 |
| paraphrase-multilingual-MiniLM-L12-v2 | 76 | **76 / 66** | **76 / 66** | 0.65650 / 0.65650 ¹ |
| granite-embedding-107m-multilingual | 76 | 71 / 66 | **76 / 66** | 0.98741 / 0.97334 |
| granite-embedding-278m-multilingual | 76 | **76 / 66** | **76 / 66** | 0.99267 / 0.98401 |
| pixie-rune-v1 | 77 | 72 / 67 | n/a ² | 0.99654 / 0.96830 |
| snowflake-arctic-embed-l-v2 | 77 | 72 / 67 | **77 / 67** | 0.99405 / 0.96164 |
| snowflake-arctic-embed-m-v2 | 77 | **77 / 67** | n/a ² | n/a (reference needs xformers) |
| nomic-embed-text-v2-moe | 76 | **76 / 66** | **76 / 66** | 0.99820 / 0.91759 |
| bge-m3 (dense GGUF) | 77 | 72 / 67 | **77 / 67** | 0.99843 / 0.96845 |
| bge-reranker-v2-m3 | 77 | **77 / 67** | **77 / 67** | 1.860 / 1.860 ³ |
| bge-reranker-base | 76 | 71 / 66 | **76 / 66** | 1.602 / 1.602 ³ |
| jina-reranker-v2-base-multilingual | 77 | **77 / 67** | n/a ² | 0.137 / 0.164 |
| mxbai-rerank-xsmall-v1 | 77 | **77 / 67** | **77 / 67** | 0.065 / 0.334 |
| mxbai-rerank-base-v1 | 77 | **77 / 67** | **77 / 67** | 0.079 / 0.204 |
| xlmr-ner-hrl (IDs only) | 76 | 12 / 12 ⁴ | 12 / 12 ⁴ | - |

**Verdict: no model regresses, so the default stays global.**
- With the switch on, every model matches its SentencePiece model on every text, on all 13 models that have a
  slow tokenizer, except xlmr-ner-hrl (⁴).
- With `=0`, the 10 whitespace-heavy texts (code, tables, invoices) mismatch on every model.
- Every cosine and reranker score is equal or better with the switch on.

The only remaining disagreement is with the transformers-4 fast tokenizer. It is a lone trailing `▁` on the 5
texts that end in whitespace, on 7 of the 16 models. SentencePiece itself and transformers 5 (checked locally:
5.15, `AutoTokenizer` for e5-small and bge-m3) both drop that piece, as the native path does. The models'
tokenizer.json normalizers differ: e5/bge-m3/arctic/granite declare `Replace(" {2,}")`, while
paraphrase-multilingual uses a `WhitespaceSplit` pre-tokenizer and no Replace. Both end up at 76/76, so no spec
switch is needed.

Notes. Each of these is identical in both arms, so it predates this switch:
1. paraphrase-multilingual: sentence-transformers truncates at `max_seq_length` 128, the native path at 512.
   The worst text is long.
2. No slow tokenizer could be loaded (`'bool' object is not callable` in the custom tokenizer config).
3. bge-reranker-v2-m3 and bge-reranker-base: native scores differ from the HF logit by up to 1.6-1.9 and the
   ranking order differs.
4. xlmr-ner-hrl: the GGUF carries no Unigram scores ("250002 tokens, 0 scores"), so its segmentation is wrong
   for most texts.

Open, pre-existing: inputs longer than 512 tokens abort with `GGML_ASSERT(i01 >= 0 && i01 < ne01)` instead of
truncating. This affects e5-base/large, granite-107m/278m, bge-reranker-base and xlmr-ner-hrl (`long_input_probe`
rc -6). e5-small, paraphrase-multilingual and nomic truncate correctly.

The initial lane A (`lane-a-initial/`, run 41492919452) failed on those aborts and on a harness error
(xlmr-ner `UnboundLocalError`). It is kept on record.
