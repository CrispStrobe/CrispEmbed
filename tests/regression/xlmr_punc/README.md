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

**Scope — corrected twice, so here is the verified position.**

This artifact **IS distributed**. `cstr/punctuate-all-GGUF` (87 downloads at time
of writing) and CrispASR's `--punc-model punctuate-all` shortcut, which
auto-downloads `punctuate-all-q4_k.gguf` from it
(`src/crispasr_punc_model.h:48`). It is also in CrispASR's README model table
and has a model card in `hf_readmes/punctuate-all-GGUF.md`.

An earlier draft of this file said "local-only, never distributed". That was
wrong and the mistake is worth naming: CrispEmbed's `model_mgr.cpp` has no
punctuate-all entry, I checked only there, and concluded it was never shipped.
The shortcut lives in the *other* repo. Checking one registry does not establish
that something is undistributed.

Everything measured here was **SHA256-verified against the published LFS
hashes** — `punctuate-all-f16.gguf` is `be54280d…`, matching byte-for-byte, so
none of this is a bad download. The same check passed for the fullstop-punc and
pcs artifacts.

**What users actually get** (`punctuate-all-q4_k.gguf`, the shortcut's default)
against the kredor blueprint: **preds 64/67, decoded text 4/6**. It carries both
differences below — no `tokenizer.ggml.scores`, and the base-XLM-R embedding
rows inherited from the f16 it was quantised from.

The `fullstop-punc-*` entries were downloaded and checked separately: they DO
carry scores and a correct `general.name`, and measure clean (q8_0 cos_min
0.998680). They do not have this defect.

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

## Can we have both? (fox AND "Hello world") — YES, measured

The two differences are independent, so an artifact can have the correct Unigram
scores *and* the non-zero embeddings. CrispASR's
`convert-fullstop-punc-to-gguf.py --restore-zeroed-embeddings-from xlm-roberta-base`
builds it, and it delivers both behaviours:

| artifact | `fox` | lines matching HF | `hello world` output |
|---|---|--:|---|
| shipped | `▁fo`+`x` ✗ | 0/7 | `Hello world.` |
| kredor-faithful | `▁`+`fox` ✓ | 6/7 | `Hello. World.` |
| **hybrid** | `▁`+`fox` ✓ | **6/7** | `Hello world.` |

Scored against `tests/regression/punct_gold/` — 120 sentences, 350 marks, real
editors' punctuation in EN/DE/FR:

| artifact | markF1 | bndF1 | per-word exact |
|---|--:|--:|--:|
| shipped | 0.767 | 0.926 | 0.941 |
| kredor-faithful | 0.713 | 0.884 | 0.931 |
| **hybrid** | **0.767** | 0.919 | **0.943** |

Paired bootstrap, 2000 resamples, 95% interval on the markF1 difference:

```
shipped  - hybrid           -0.0006  [-0.0269, +0.0237]  not distinguishable
shipped  - kredor-faithful  +0.0536  [+0.0192, +0.0886]  SIGNIFICANT
kredor   - hybrid           -0.0542  [-0.0838, -0.0245]  SIGNIFICANT
```

So: **restoring the zeroed embeddings is a real, significant gain** (+0.054
markF1, +0.035 boundary F1 — those zero rows genuinely cost quality), and
**fixing the tokenization costs nothing measurable**. The hybrid matches the
shipped artifact's quality while being tokenization-correct. That is best of
both worlds, and it is now an interval rather than an impression.

⚠ **An earlier 102-mark version of this evaluation said the opposite** — shipped
0.516 vs hybrid 0.429 — and was flagged at the time as ~1–2 standard errors and
unable to settle anything. At 350 marks they are tied and the interval straddles
zero. That gap was noise. It is recorded here because the lesson generalises:
under a few hundred marks, a punctuation comparison is undecided, whatever the
two decimals say.

