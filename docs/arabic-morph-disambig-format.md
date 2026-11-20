# Camel Morph disambiguation model (`crispembed-camel-morph-disambig-v1`)

The native Camel Morph analyzer (`src/arabic_morph.cpp`) returns every
analysis of a word, unordered when the database is the `noprob` variant
(see [the binary database format](arabic-morph-binary-format.md)). The
context disambiguator ranks those analyses in their sentence. Its model is
data, downloaded separately like the database, and read by
`crispembed_ar_morph_disambig_open` (bundle) or
`crispembed_ar_morph_disambig_open_file` (bare file).

The disambiguator is **default-off**: every disambiguation function fails
with `CRISPEMBED_AR_MORPH_DISABLED` unless `CRISPEMBED_AR_MORPH_DISAMBIG` is
set to an on value (`core_env::on`: anything but empty, `0`, `false`, `off`,
`no`). `CRISPEMBED_AR_MORPH=0` disables it as well. The analyzer itself is
unchanged.

## Model

A bigram hidden Markov model whose states are **candidates**. A candidate
groups the analyses of one token that share `(tag, lex, pos)`:

- the **tag** joins the values of the model's tag features with `|`, in
  order. A `value` feature contributes its value, or `-` when the analysis
  lacks it or holds null. A `flag` feature contributes `0` when the value is
  absent, null, empty, `0` or `na`, and `1` otherwise. The first tag feature
  is always `pos` by value;
- a token without analyses (unknown, or rejected by the analyzer) is the
  single candidate with tag, lex and pos `<oov>`.

A sentence of candidates `c_1 … c_n` scores

```
T(tag_1 | <s>) · Π T(tag_i | tag_i-1) · T(<s> | tag_n) · Π L(lex_i, pos_i | pos_i)
```

`T` is the tag transition table, with `<s>` (tag 0) as the boundary. It
lists the transitions seen in training; any other one backs off to the tag
unigram, scaled per previous tag. `L`
holds the log-probability of a lexeme given its pos, and the pos table gives
the value of a lexeme the model does not list. A lexeme of an unknown pos
gets `-30`. A transition from or to a tag the model does not know gets the
probability `1e-12`.

The **score** of a candidate is its forward-backward posterior probability
in the sentence. The candidates of a token sum to 1, and every analysis
carries its candidate's score and rank. The forward pass rescales at every
token; a token whose forward mass underflows restarts from uniform weights.
`tools/arabic_morph/disambig_train.py` (`posteriors`) is the Python
reference, and the contract test compares the native scores with it.

## Layout

All integers are little-endian. Sections start at 8-byte-aligned offsets,
and the padding after each section is zero. A big-endian host refuses the
file. The container is the one of the binary database:

### Header (64 bytes)

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `char[8]` | magic `CEMDISAM` |
| 8 | u32 | format version, `1` |
| 12 | u32 | header size, `64` |
| 16 | u32 | section count (at most 64) |
| 20 | u32 | flags, `0` |
| 24 | u64 | file size |
| 32 | u32 | CRC-32 (IEEE) of the section directory |
| 36 | u32 | CRC-32 of the 64 header bytes with this field zeroed |
| 40 | `u8[24]` | zero |

The directory follows the header: one 24-byte entry per section,
`u32 tag, u32 CRC-32 of the section, u64 offset, u64 size`. Every section
below occurs exactly once.

| Tag | Content |
| --- | --- |
| `STRS` | `u32 n`, `u32 offsets[n+1]`, UTF-8 bytes. Unique and sorted by bytes; every name below is a string id. |
| `SPEC` | `u32 n` (1–32), then `n × {u32 feature name id, u32 mode}`; mode 0 = value, 1 = flag. The first entry is `pos`, value. |
| `TAGS` | `u32 n` (at most 8192), `u32 tag string id[n]`. Unique; tag 0 is `<s>`. |
| `TRAN` | `u32 n` (the tag count), `f32 log U[n]` (tag unigram), `f32 log B[n]` (backoff weight per previous tag), `u32 m`, then `m × {u32 prev, u32 cur, f32 log T(cur \| prev)}`, strictly ascending by (prev, cur). An unlisted transition is `B[prev] · U[cur]`. |
| `UNSN` | `u32 n`, then `n × {u32 pos id, f32 log-probability}`: the value of a lexeme of that pos that `LEXP` does not list. |
| `LEXP` | `u32 n`, then `n × {u32 pos id, u32 lex id, f32 log L(lex, pos \| pos)}`, strictly ascending by (pos id, lex id). |
| `META` | A JSON object: training corpus, sources and licences, hyperparameters. `crispembed_ar_morph_disambig_info` embeds it. |

Every log-probability is finite and at most 0. The loader checks the magic,
version, sizes, all CRCs and paddings, string ids, UTF-8, order and
uniqueness before use; a corrupt file is rejected, never read out of bounds.
Lexeme strings are the `lex` values of the database the model was trained
with (`pos` and `lex` must be in its feature profile, as are the tag
features).

## Bundle manifest

```json
{
  "format": "crispembed-camel-morph-disambig-bundle",
  "format_version": 1,
  "model": "camel_morph_msa_v1.0.disambig.cmdis",
  "model_format": "crispembed-camel-morph-disambig-v1",
  "files": [{"path": "…", "size": 0, "sha256": "…"}]
}
```

As for the database bundle, the host supplies the manifest's SHA-256, and
every listed file (model, `NOTICE`, `LICENSE`) must sit next to it and match
its size and SHA-256 before the model is parsed. Further keys (licence,
attribution, the database the model was trained for, the training
description) are informational.

## Output

`crispembed_ar_morph_disambiguate` takes the words of one or more sentences
(as the host tokenizes them; punctuation, numbers and Latin words are tokens
too) and the number of words of each sentence. It returns the
`crispembed_ar_morph_analyze_compact` buffer for those words, followed by a
score section (layout in `src/arabic_morph.h`): per word, the number of
analyses and candidates and, per analysis, its candidate's rank and score.
Hosts that already decode the compact buffer read the score section after
it.

## Training

`tools/arabic_morph/disambig_train.py` trains the model without annotated
data:

1. `sentences` and `eval-sample` cut raw text (Arabic Wikipedia) into
   sentences and tokens with the host tokenizer. Articles drawn for
   evaluation are excluded from training.
2. `analyze` runs every distinct token through the native analyzer
   (`crispembed-ar-morph --compact --dedup`).
3. `train` builds the candidates of every token and runs expectation
   maximization (`tools/arabic_morph/ar_morph_disambig_em.cpp`). The first E
   step weights each candidate by the frequency of its tag among tokens
   whose candidates all share one tag (`--init unambiguous`). Transitions
   are smoothed towards the tag unigram,
   `T(cur | prev) = (C(prev, cur) + κ·U(cur)) / (C(prev) + κ)` with an
   add-α unigram `U`, and lexemes add-β per pos. The counts of the last E
   step are smoothed the same way and written in this format; transitions
   with an expected count below `--min-trans` are left to the backoff, and
   lexemes below `--min-count` to the `UNSN` value.
4. `evaluate` measures top-1 UPOS and lemma accuracy in context on a CoNLL-U
   file (UD Arabic-PUD, evaluation only), next to the out-of-context
   rankings of the `noprob` and `full` databases.

`tools/arabic_morph/gen_disambig_fixture.py` writes the contract test's
hand-made model (`tests/arabic_morph_disambig_fixture.h`).
