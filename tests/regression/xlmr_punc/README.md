# XLM-R punctuation blueprint reference

`blueprint_ref.txt` pins token ids, per-token 6-class logits, argmax predictions
and the blueprint's own punctuated text for the ten lines in `corpus.txt`,
dumped from **`kredor/punctuate-all`** in f32 torch via `transformers`.

Checked in at 8 KB rather than hosted, so the comparison runs with no torch and
no 945 MB checkpoint:

```bash
python tests/xlmr_punc_parity.py build /path/to/punctuate-all.gguf \
    tests/regression/xlmr_punc/blueprint_ref.txt
```

Regenerate with:

```bash
python tools/dump_xlmr_punc_reference.py --model kredor/punctuate-all \
    --corpus tests/regression/xlmr_punc/corpus.txt \
    --output tests/regression/xlmr_punc/blueprint_ref.txt
```

## What it found on its first run

The engine handles two model families. FireRedPunc (BERT/WordPiece) got ground
truth first and it immediately turned up a wrong forward pass. This is the
SentencePiece/XLM-R half, which had none — its empty-output bug had been
verified only by "the two arms now agree", which shows consistency, not
correctness.

| artifact | token ids | cos_min | max_abs | preds | decoded |
|---|--:|--:|--:|--:|--:|
| local `punctuate-all-f16.gguf` | 5/6 | **−0.284548** | 9.6347 | 65/67 | 5/6 |
| re-converted from `kredor/punctuate-all` | **6/6** | **0.999999** | 0.0062 | **67/67** | **6/6** |

**Scope, checked rather than assumed: the bad artifact is LOCAL-ONLY.**
`punctuate-all` has no model-registry entry, so it was never distributed — it is
a bench file on this box. The registry's `fullstop-punc-*` entries were
downloaded and inspected separately: they DO carry `tokenizer.ggml.scores` and a
correct `general.name`, so they do not have this defect. An earlier draft of
this file said "shipped", which overstated the blast radius.

### What `punctuate-all-f16.gguf` actually is

Determined by comparing it tensor-by-tensor against a fresh conversion, after an
earlier version of this file attributed the whole divergence to the missing
tokenizer scores. That was wrong: the scores explain one line, and the dominant
effect is something else.

**198 of its 199 tensors are byte-identical to `kredor/punctuate-all`.** The one
that differs is `emb.tok_emb.weight`, on 9539 rows in four contiguous runs. And
on those rows it is not corrupt — it holds **xlm-roberta-base's original
embeddings**, while the source model holds **zeros**:

| row | this GGUF | kredor/punctuate-all | xlm-roberta-base |
|--:|--:|--:|--:|
| 6816 | 5.6786 | **0.0000** | 5.6786 |
| 10912 | 5.7197 | **0.0000** | 5.7197 |
| 51895 | 5.9867 | **0.0000** | 5.9867 |
| 33600 | 6.3655 | 6.3655 | 6.3592 |

(row L2 norms; the last row is one kredor did *not* zero, and there the GGUF
follows kredor, not base.)

`kredor/punctuate-all` zeroes 9531 embedding rows — four contiguous ranges
(4086–5449, 6816–9545, 10912–12276, 51895–55966), which reads as deliberate
pruning of token ranges for languages it does not serve. This artifact was built
from a path that kept base XLM-R's values there instead.

So there are **two independent differences from the reference**, and they land on
different lines:

| line | token in a zeroed range | max_abs vs reference |
|--:|---|--:|
| 0 | `▁world` (8999) | 9.6347 |
| 1 | — | 0.0055 |
| 2 | `▁geht` (8644), `▁dir` (5402) | 5.7803 |
| 3 | — | 0.0035 |
| 4 | — (this is the tokenizer one) | 1.0763 |
| 5 | — | 0.0046 |

Perfect correlation: every line touching a zeroed row diverges by ~6–10, every
line that does not matches to f16 rounding. Line 4 is the separate
missing-scores defect (`fox` → `▁fo`+`x` instead of `▁`+`fox`).

**Which behaviour is "right" is genuinely open.** A zero embedding is unlikely to
be intended semantics, and feeding base XLM-R's vectors may well produce better
text for those 9531 tokens. But `transformers` loads the zeros, so that is what
the blueprint does, and parity is measured against the blueprint. Do not treat
the divergence as proof this file is worse — only that it is not
blueprint-faithful, and that it additionally lacks the tokenizer scores, which
IS an unambiguous defect.

⚠ **`fullstop-punc` is a different model** (XLM-R large, 24L/1024) and needs
its own reference — `--model oliverguhr/fullstop-punctuation-multilang-large`.
Having scores is necessary, not sufficient.

Decoded-text case differences are expected: the blueprint works from lowercased
token surface forms, the runtime re-emits the user's original words on purpose.
