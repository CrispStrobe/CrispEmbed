# Native spaCy v2 profile: English and French — 18 November 2026

The frozen German `de_core_news_sm` v1 path (`spacy_decode.h`, `model.gguf`
`fae2e6ab…`) is unchanged. Version-2 pipelines add the official spaCy 3.8
`*_core_*_sm` CNN family through a separate converter, GGUF architecture
(`spacy_pipeline_v2`), resource format (`format_version: 2`) and runtime path.
Reference: original spaCy 3.8.18 with the pinned wheel, run on the same host.
Inputs are raw UD r2.14 test-set texts (CC BY-SA 4.0), used as input only.

## Results (macOS arm64; hosted Linux receipts below)

| Pipeline | Test text | Texts | Tokens | Annotations exact | Entities |
| --- | --- | ---: | ---: | --- | --- |
| en_core_web_sm 3.8.0 | UD English-EWT test | 2,077/2,077 | 25,531 | tag, pos, morph, lemma, dep, head, sentence starts | 1,876/1,876 |
| fr_core_news_sm 3.8.0 | UD French-GSD test | 416/416 | 9,786 | same | 469/469 |

Comparison runs through the C ABI only (`tools/spacy/spacy_smoke.cpp` vs
`tools/spacy/pipeline_oracle.py`, compared by `tests/spacy_smoke_compare.py`):
[English](mac-en-ewt-compare.json), [French](mac-fr-gsd-compare.json).
English stage drift against the original thinc layers (300 texts): tok2vec max
abs 2.6e-6, tagger scores 1.3e-5. `tests/test_spacy_v2_controls.py` (gate,
model/resource hash, foreign resources, v1/v2 mixing, text-only capabilities,
numeric service and `attach_ner` rejection) passes for both languages. German
is unchanged: 1,776/1,776 GSD sentences, NER 1,758/1,758, CTest 4/4, text API.

| Bundle | model.gguf SHA-256 | Size | Tensors |
| --- | --- | ---: | ---: |
| English | `17b18871a2c1e766be4572c1dd9a9902ba01c4a46679797ccb3c4d0048d24e7d` | 12.8 MB | 66 |
| French | `8b8f9ab4aa322191faa1735d1b7d53e0e744e18f4025290fe37242faa9a8e53e` | 12.8 MB | 66 |

French `resources.json` is 35 MB (217k-entry lemma lookup).

Hosted Linux x86-64 (converted on the runner from the pinned wheels, built from
source, compared through the C ABI) reproduces both results exactly and
produces byte-identical `model.gguf` files:
[English run 44363961502](linux-en-run-44363961502.json),
[French run 44363955799](linux-fr-run-44363955799.json).

## What v2 adds

- **Converter** `models/convert-spacy-pipeline-to-gguf.py`: layers found by
  name, pinned wheels, canonical MORPH values, POS-named lemma tables, French
  `token_match` exported as 266 top-level alternatives (rejoined text equals the
  original; 0 disagreements with the original regex on 3,649 GSD words).
- **Runtime** `src/spacy_v2.{h,cpp}` in source component order: tagger,
  morphologizer (overwrite, no extend), parser (shared state machine and
  deprojectivization), attribute ruler (spaCy Matcher subset used by these
  pipelines: 1–8 token patterns over TAG/POS/LOWER/DEP/IS_SPACE/ORTH with
  equality, IN, NOT_IN and REGEX; all rules matched first, applied in
  (rule, start, end) order), English and French rule lemmatizers, NER with any
  move inventory (`NerModel` now takes class names; German order pinned).
- **Lexical** format 2: generic bounds, ordered NORM tables, optional anchored
  `token_match`; Python `lower`/`isalpha`/regex exposed for the ruler and
  lemmatizers. Format 1 keeps every exact German inventory check.
- **C ABI**: `crispembed_spacy_create_with_resources` dispatches on the GGUF
  architecture. v2 contexts offer tokenize/analyze/entities; the prepared
  numeric services and `attach_ner` reject them.

## Model-size and language survey

[Hosted run 44355661567](survey-run-44355661567.json) (`tools/ci-heavy/spacy_model_survey.py`,
gold UD tokens): md agrees with sm on ~99% of tags/lemmas (+0.1 point); trf
gains are largest for French (UAS 0.777→0.867, LAS 0.719→0.830, UPOS
0.922→0.968) and modest for English; trf runs ~30–40× slower than sm on CPU in
Python. LatinCy needs its own Python components. Not yet ported: transformer
pipelines, Latin, Hebrew (DictaBERT-joint, CC-BY-4.0) and Arabic (current
trained options use non-commercial data).
