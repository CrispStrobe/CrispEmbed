# Native Camel Morph MSA analyzer — 19 November 2026

`src/arabic_morph.{h,cpp}` is a C++ port of the camel_tools 1.6.0 (MIT,
tag `v1.6.0`, commit `be79ca9f`) `MorphologyDB` loader (analysis flag) and
`Analyzer`, including all five backoff modes, `merge_features`, the
diacritic/CAPHI/tokenization rewrite rules, `dediac_ar`, the default
normalization map and the Unicode punctuation/symbol classes. It reads the
Camel Morph MSA v1.0 ALMOR database (CC BY 4.0), which is published separately
as data. The reference is camel_tools itself, run on the same host with the
same database files.

## Exactness

Input: every unique word of the 1,000 `# text` lines of UD Arabic-PUD r2.14
test (camel_tools `simple_word_tokenize` tokens plus raw whitespace tokens,
which keep attached punctuation; 9,447 words; evaluation only) and 124 stress
inputs: diacritized words, tatweel, hamza/alef/ya/ta-marbuta variants,
clitic-heavy forms, Latin/Greek/Cyrillic/CJK, Western/Arabic-Indic/Persian
digits, punctuation and symbols, NBSP/ideographic space, ZWJ/ZWNJ, embedded
NUL and newline, backslashes, empty and whitespace-only input.

| Database / backoff | Words exact | Analyses equal | Feature values | Words where camel_tools raises (native also errors) |
| --- | ---: | ---: | ---: | ---: |
| full / NONE (camel_tools default) | 9,571/9,571 | 217,053/217,053 | 9,740,479 | 0 |
| full / NOAN_PROP | 9,571/9,571 | 273,703/273,703 | 12,289,729 | 1 |
| full / ADD_ALL | 9,571/9,571 | 1,704,695/1,704,695 | 76,491,407 | 2 |
| noprob / NONE | 9,571/9,571 | 217,053/217,053 | 9,740,479 | 0 |

"Exact" means the same analyses in the same order, the same keys, identical
values and value types (string, `null`, float compared as doubles), and the
same key order, with one documented exception: `merge_features` appends keys
the stem lacks (`d1tok`…`bwtok`, `catib6`, `ud`, `pos_logprob`, …) by
iterating Python frozensets, whose order depends on `PYTHONHASHSEED` and so
differs between camel_tools processes. The port appends them in camel_tools'
source order; the comparison sorts each such run. Receipt:
[exactness.json](exactness.json). The camel_tools exceptions are `re.error`
raised when a backoff stem containing `\` is used as an `re.sub` template; the
port reproduces CPython 3.12's template parser and returns
`CRISPEMBED_AR_MORPH_INPUT_ERROR` for the same words.

Tools: `tools/arabic_morph/camel_reference.py` (`words`, `analyze`,
`compare`, `rank`) and the C-ABI-only CLI `crispembed-ar-morph`
(`tools/arabic_morph/ar_morph_cli.cpp`). `tests/test_arabic_morph.cpp`
(CTest `arabic-morph-contract`, CI workflow `arabic-morph.yml`) checks the ABI
against camel_tools output on a synthetic database, plus hash, manifest,
UTF-8, size and error contracts.

## Performance

Apple M1, one thread, shared machine (load average 4–8), 9,571 words, backoff
NONE, full bundle ([performance.json](performance.json)):

| | Load | Words/s | Memory |
| --- | ---: | ---: | --- |
| native (3 runs) | 1.30–1.48 s incl. SHA-256 of all bundle files | 2,309–3,185 | 238 MB resident after load, 310 MB peak |
| camel_tools 1.6.0 | 3.0 s (no hashing) | 333 | 790 MB peak |

ADD_ALL backoff produces 7.9× more analyses and runs at 287 words/s natively.
The peak during load comes from holding the 85 MB text while parsing; a
compact binary database would cut load time and peak memory.

## Bundles and publication

`models/convert-camel-morph-db.py` pins the source file
(`CAMeL-Lab/camel_morph` commit `aaa39653c2a53ebacf087f0b98e7f7cfb55cca81`,
last change `e818de08`, `official_releases/lrec-coling2024_release/databases/camel-morph-msa/camel_morph_msa_v1.0.db`,
84,959,404 bytes, SHA-256 `110e03baa282874062d076e23d9671e985b6f76520b891d58cba4e55de0b8f60`,
git blob `19f52866`) and writes two self-contained bundles, each with
`manifest.json` (size and SHA-256 of every file plus the source pin), the
database, `NOTICE` (required attribution, citation, provenance, changes) and
`LICENSE` (CC BY 4.0), under a Hugging Face card with `license: cc-by-4.0`.

| Variant | Database SHA-256 | manifest.json SHA-256 |
| --- | --- | --- |
| `full/` (unmodified) | `110e03baa282874062d076e23d9671e985b6f76520b891d58cba4e55de0b8f60` | `6fc4a6fe6eab991d5ae62219116bbdf9e7354d835df1a2d9ac71402e57716e35` |
| `noprob/` (78,083,392 bytes) | `2dfbec413288e7da96723d7926c0e74f2e90ef6994c6f7e995a27a8df079ecff` | `96b949981e76c0d2b4892ec7aac1f1e519813ac6d55d4e39303d94822df68050` |

Published to [cstr/camel-morph-msa-v1.0](https://huggingface.co/cstr/camel-morph-msa-v1.0)
at commit `b31cf4a83f8ad63bb1989cf2879e72f23fadbfb0`. An anonymous download of
all 11 files at that commit into an empty cache matched every size and SHA-256,
and the remote tree holds no other files
([hf-readback.json](hf-readback.json), `tools/arabic_morph/hf_publish.py verify`).
Hosts open a bundle with `crispembed_ar_morph_open(manifest_path, manifest_sha256, backoff, …)`;
the manifest hash is the trust anchor, and the loader checks every listed file
before parsing.

## Probability fields (provenance)

camel_morph's `db_maker.py` writes `lex_logprob` and `pos_lex_logprob` for each
stem from a pickle produced by `sandbox/get_logprob_atb.py` over Penn Arabic
Treebank (ATB) MAGOLD data, which the release README says is not public. Of
175,963 stem entries, 38,159 have a `pos_lex_logprob` other than the `-99`
placeholder.
`pos_logprob` is declared but carries no values. Prefixes and suffixes carry
no probabilities (`correctness` on prefixes is a hand label). The paper also
says the lexicon was seeded from CALIMA-Star/SAMA and reworked by hand. That
provenance applies to both variants; the noprob variant removes only the
treebank-derived numbers (351,926 values) and keeps every other byte.

camel_tools analysis sets are unchanged without them: the same analyses
for 7,567/7,567 analyzable words, and the three fields become the constant
-99.0. Ranking changes. With camel_tools' MLE out-of-context ordering
(probability, then `pos_lex_logprob`, `lex_logprob`, `diac`) on the 7,127
ambiguous words, the top-1 analysis stays the same for 3,275 (46.0%) and the
whole order for 2,513 (35.3%). Against UD Arabic-PUD gold (5,352 unsplit
tokens with one annotation; the gold pair is among the analyses for 4,816):

| Top-1 matches gold | full | noprob |
| --- | ---: | ---: |
| UPOS (coarse Camel→UD map) | 4,261 (79.6%) | 3,055 (57.1%) |
| lemma (dediacritized, normalized) | 4,444 (83.0%) | 3,512 (65.6%) |
| both | 3,907 (73.0%) | 2,514 (47.0%) |

Without the probabilities, out-of-context ranking falls back to diacritized
form order and loses about 26 points of top-1 UPOS+lemma accuracy. A host that
ships noprob needs its own ranking, for example a context model trained on
licensable data. Receipt: [ranking-full-vs-noprob.json](ranking-full-vs-noprob.json).

## Licensing

- Code: `src/arabic_morph.cpp` is MIT; it ports MIT camel_tools code and keeps
  the camel_tools copyright and permission notice in its header.
  `tools/arabic_morph/THIRD_PARTY_NOTICES` lists camel_tools, Ryu (shortest
  float formatting, Apache-2.0 option of Apache-2.0/BSL-1.0), Unicode and
  CPython character-property data. The standalone target links no other
  library.
- Data: Camel Morph MSA v1.0 © 2024 by New York University Abu Dhabi, CC BY
  4.0; downloaded by the user inside the app as data and never compiled into a
  binary. Apps must show the attribution in `NOTICE`.
- Not used: camel_tools' default `calima-msa-r13` database (GPL v2) or any
  LDC resource. The treebank-derived probabilities exist only in the `full`
  variant; the CALIMA-Star/SAMA seeding caveat applies to both variants.

## Open

Quds Office integrated the noprob bundle (Quds PR #189). The compact
binary database is done; see
[camel-morph-binary-20261119](../camel-morph-binary-20261119/README.md).
iOS and Android packages of the standalone target remain open, and
camel_tools' disambiguators (MLE, BERT) are not ported.
