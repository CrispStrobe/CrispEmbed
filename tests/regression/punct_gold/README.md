# Punctuation-restoration gold set

120 sentences — 2349 words, 350 marks — for scoring punctuation restoration
against text whose punctuation is **known**, which the blueprint harnesses
structurally cannot do (there, the reference is one of the candidates).

40 sentences each from three public-domain works, so no single language
dominates:

| source | language | work |
|---|---|---|
| `en_mobydick.txt` | English | Melville, *Moby-Dick* |
| `de_34811.txt` | German | Mann, *Buddenbrooks* |
| `fr_bovary.txt` | French | Hugo, *Les Misérables* |

Extracts from public-domain works (Project Gutenberg plain text). The gold is a
real editor's punctuation rather than sentences written by whoever is running
the evaluation — which matters, because self-written gold encodes the writer's
own habits and would quietly score the models against those.

```bash
python tools/eval_punct_gold.py --gold tests/regression/punct_gold/gold120.json \
    build /path/to/model.gguf [more.gguf ...]
```

Rebuild with `tools/build_punct_gold.py <out.json> <text.txt> ...`.

## What the selection rejects, and why

Every filter removes a way the SCORE could be wrong, not merely a sentence that
looked untidy:

- **Quotes, em-dashes, semicolons, parentheses, ellipses → rejected, not
  stripped.** Those are exactly where period-vs-colon is a stylistic coin flip.
  An earlier 102-mark attempt on literary prose spent most of its signal there:
  gold read `…nothing to do: once or twice…` and *every* model produced
  `…nothing to do. Once or twice…`, which is a style difference, not an error.
- **Tables of contents and headings → rejected** by a Title-Case ratio test.
  Moby-Dick's TOC is long paragraphs of chapter titles that pass every other
  test and are not sentences; scoring a punctuation model on
  `The Deck Towards the End of the First Night Watch.` measures nothing.
- **Gutenberg boilerplate → rejected.** Prose about the edition, not from it.
- **Verse and indented blocks → rejected**, and 8–40 words, so no single
  sentence dominates or contributes a single mark.

## Two metrics

`markF1` asks whether the model emitted the mark the editor did. It is harsh and
partly a style contest — a period where the author wrote a colon counts as both
a miss and a false positive.

`bndF1` asks only whether a sentence *ended* there, treating `.` and `?` as the
same event. Much less style-dependent: whether a sentence stops is close to
determined, which mark stops it is not. **When the two disagree, believe the
boundary number.**

A **paired bootstrap** (2000 resamples) reports the 95% interval on the
difference between models. Paired because every model sees identical sentences,
and resampling *sentences* rather than *marks* because marks inside a sentence
are not independent draws.

## Why the size matters

An earlier version of this evaluation used 102 marks and reported the shipped
`punctuate-all` artifact beating the hybrid by 0.087 F1. At 350 marks they are
**tied**, and the bootstrap interval on that difference is
`[-0.027, +0.024]` — it straddles zero. The first result was noise, and the only
reason it was not published as a regression is that 102 marks was flagged as too
few at the time. Treat any punctuation comparison under a few hundred marks as
undecided.
