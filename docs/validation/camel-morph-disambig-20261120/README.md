# Camel Morph context disambiguator — validation (20 November 2026)

The `noprob` Camel Morph database has no Penn Arabic Treebank probabilities,
so its analyses come unranked (camel_tools then orders by diacritized form).
This receipt covers a licence-clean replacement: a bigram hidden Markov model
over the analyzer's own analyses, trained without annotations on Arabic
Wikipedia, and its native forward-backward
(`crispembed_ar_morph_disambiguate`, ABI 3, default-off behind
`CRISPEMBED_AR_MORPH_DISAMBIG`). Format:
[arabic-morph-disambig-format.md](../../arabic-morph-disambig-format.md).

## Data and licences

| Use | Resource | Licence |
| --- | --- | --- |
| Training text | Arabic Wikipedia `20231101.ar` (HF `wikimedia/wikipedia`, shards 2–4), 12,000,927 tokens / 885,657 sentences; the 1,600 articles drawn as the Wikipedia evaluation sample are excluded | CC BY-SA 4.0 |
| Analyses, lexemes | Camel Morph MSA v1.0 `proofing-bin` (noprob) | CC BY 4.0 |
| Evaluation only | UD Arabic-PUD test (GitHub master, 2026-11-20), 1,000 sentences | CC BY-SA 3.0 |
| Reference only | Camel Morph `full` database (PATB-derived probabilities) for the comparison column | not shipped |

The model is published under CC BY-SA 4.0 with both attributions. No LDC,
PADT (CC BY-NC-SA) or other non-commercial resource was used.

## Approach and selection

EM over candidates `(tag, lex, pos)`: transitions between tags, emissions
`P(lex, pos | pos)`. Tag-set, initialization and corpus-size variants were
compared on PUD sentences 0–499 only (log in [accuracy.json](accuracy.json)):

- tags `pos` alone let EM relabel frequent function words (في as an
  abbreviation, من as a verb); adding the proclitics, `enc0`, `asp`, `per` and
  `num` fixed most of it (UPOS 67.8 → 76.8 on all of PUD);
- a dense add-α transition table over ~4,900 tags was 97 MB; backing off to
  the tag unigram stores only seen transitions (4.5 MB) at equal accuracy;
- doubling the corpus to 25M tokens did not help (dev 76.4 vs 76.6 UPOS);
- scaling down the initial weight of `noun_prop`, `abbrev` and `interj`
  readings (EM otherwise keeps proper-noun readings of من, في, قبل) gave the
  best dev result and was chosen.

## Result (top-1 in context, gold UPOS + lemma)

| | noprob (no ranking) | model | full (PATB) |
| --- | ---: | ---: | ---: |
| All PUD, 15,482 tokens: UPOS | 47.0% | **77.9%** | 77.9% |
| lemma | 72.3% | **90.4%** | 88.2% |
| both | 40.9% | **73.2%** | 74.1% |
| Held-out sentences 500–999, 7,869 tokens: UPOS / lemma / both | 47.6 / 72.9 / 41.0 | **78.1 / 90.8 / 73.4** | 77.5 / 87.8 / 73.5 |

Tokens are host tokens that are exactly one UD word (UD splits clitics; those
tokens have no single gold). Gold UPOS+lemma is among the analyses for 84.9%.
The model closes the whole UPOS gap and 97% of the UPOS+lemma gap between
noprob and the treebank-trained `full` ranking. Many remaining "errors" are
the coarse Camel→UD mapping (مع, خلال as nouns; هذا as DET), shared by `full`.
For comparison with the earlier word-type receipt
([camel-morph-msa-20261119](../camel-morph-msa-20261119/README.md), 79.6% vs
57.1% UPOS out of context), the same type-level computation here gives 79.4%
vs 57.0%.

Calibration (all PUD): the best candidate has probability ≥ 0.9 for 61.8% of
tokens, with UPOS+lemma 81.5% correct there (gap to 100% mostly the mapping).

## Native parity and speed

`crispembed-ar-morph --model … --sentences pud-tokens.txt` (C ABI) matches
the Python reference on all 17,907 analyzed tokens: same best candidate,
largest score difference 5.3e-7. 1,000 sentences / 18,222 tokens analyze and
disambiguate in 3.8 s (4,769 tokens/s, Apple M1 shared with other jobs, load
6–15). The contract test checks the native scores against the Python
reference on a hand-made model, the default-off gate, model and manifest
pinning, corrupted bytes, and that the buffer starts with the compact one.

## Publication

`cstr/camel-morph-msa-disambig@07fd9e65200a3ea92a081efa9b0714817178df5f`,
`bigram-v1/` (manifest SHA-256
`02b8fc1cdfd8c925f6737029eafd79aea22148f2ce0af31b8b0e07a95053cbaa`, model
4,487,296 bytes, SHA-256
`987250d64ec36e631b125f9f7ba52f4e5fe48da694ea3a6a21786b4ac4aabf61`).
Readback into an empty cache matched every file and the remote tree holds no
other file ([hf-readback.json](hf-readback.json)); bundle manifest in
[manifest-bigram-v1.json](manifest-bigram-v1.json).
