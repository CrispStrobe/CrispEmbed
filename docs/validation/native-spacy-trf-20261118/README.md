# Native spaCy transformer pipelines: fr_dep_news_trf — 18 November 2026

`fr_dep_news_trf` 3.8.0 (CamemBERT) through the v2 profile, built with
`-DCRISPEMBED_SPACY_TRF=ON` (reference SentencePiece C++ v0.2.1). Reference:
original spaCy on the same runner; inputs are UD French GSD r2.14 test texts as
416 sentences and as 7 long documents (9,786 tokens; multi-window spans).

## Exactness and speed (hosted runners; native times include loading the 441 MB model)

Every configuration below is identical to original spaCy on 416/416 sentences
and 7/7 documents (48/48 configurations): reference vs fused graph
(`CRISPEMBED_SPACY_TRF_FUSED`), BLAS off/on (`-DCRISPEMBED_SPACY_BLAS=ON` +
`CRISPEMBED_SPACY_TRF_BLAS=1`), 1 and 4 threads (`CRISPEMBED_SPACY_THREADS`).

| Runner | Corpus | PyTorch 1T / 4T | Native reference 1T / 4T | Native fused 1T / 4T | Fused + BLAS 1T |
| --- | --- | --- | --- | --- | --- |
| Linux x86-64, 4 vCPU ([44382473604](bench-44382473604.json)) | sentences | 132 / 235 | 240 / 419 | 244 / **430** | 280 |
| | documents | 253 / 476 | 226 / 397 | 244 / 420 | 380 |
| Linux arm64, 4 vCPU ([44382479345](bench-44382479345.json)) | sentences | 240 / 267 | 106 / 312 | 106 / **314** | 177 |
| | documents | 501 / 568 | 87 / 278 | 87 / 284 | 259 |
| macOS arm64, 3 vCPU ([44382485291](bench-44382485291.json)) | sentences | 206 / 193 | 131 / 21 | 176 / 23 | 163 |
| | documents | 311 / 376 | 148 / 67 | 156 / 193 | **257** |

Tokens per second. The fused graph (batched spans, Q/K/V weight views,
`ggml_flash_attn_ext`) is never slower than the reference graph and is now the
default for transformer pipelines (`CRISPEMBED_SPACY_TRF_FUSED=0` restores the
reference graph). BLAS combined with several ggml threads oversubscribes small
runners (e.g. 47 tok/s); use BLAS with `CRISPEMBED_SPACY_THREADS=1`, or no BLAS
with threads up to the physical core count. 4 threads on the 3-vCPU macOS
runner oversubscribe as well.

Remaining gap to PyTorch on long documents comes from exact erf GELU (scalar
`erff` in ggml), per-op thread synchronisation and model loading included in
the native times. F16/quantized weights for the encoder need their own decoded
A/B before use.
