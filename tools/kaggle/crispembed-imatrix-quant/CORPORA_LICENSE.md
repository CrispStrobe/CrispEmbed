# Evaluation corpora — source & license

`calib_corpus.txt` and `eval_corpus.txt` are the imatrix **calibration** and **A/B
evaluation** text used for the embed / sparse (SPLADE) / ColBERT modes — a bilingual
**English + German** set of parallel sentence pairs across diverse domains.

- **Source:** self-authored for this project (see `tools/gen_eval_corpora.py`).
- **License:** released into the **public domain (CC0)** — no attribution required,
  usable under MIT / Apache-2.0 / BSD-3 / any downstream terms.

No third-party content is included, so there is no CC-BY / CC-BY-SA obligation. The
A/B compares a quant vs the full-precision model on identical text, so only realistic,
diverse, multilingual text is needed — not labels. Regenerate with
`python tools/gen_eval_corpora.py`.
