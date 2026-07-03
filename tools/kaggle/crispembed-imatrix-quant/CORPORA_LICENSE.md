# Evaluation corpora — sources & licenses

`calib_corpus.txt` and `eval_corpus.txt` are the imatrix **calibration** and **A/B
evaluation** text used for the embed / sparse (SPLADE) / ColBERT modes. They are a
bilingual **English + German** sample drawn from the **Tatoeba** collection via the
`mteb/tatoeba-bitext-mining` (`deu-eng`) dataset.

- Source: Tatoeba — https://tatoeba.org  (via https://huggingface.co/datasets/mteb/tatoeba-bitext-mining)
- License: **CC-BY 2.0 FR** (attribution). © Tatoeba contributors.
- Regenerate: `python tools/gen_eval_corpora.py` (see that script).

Only permissive corpora are used (Apache-2.0 / MIT / BSD-3 / CC-BY). See
`PLAN.md` and `LEARNINGS.md` for the SOTA EN+DE benchmark notes (MMTEB / MIRACL /
Tatoeba / GermanQuAD).
