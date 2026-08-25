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
| shipped `punctuate-all-f16.gguf` | 5/6 | **−0.284548** | 9.6347 | 65/67 | 5/6 |
| re-converted from `kredor/punctuate-all` | **6/6** | **0.999999** | 0.0062 | **67/67** | **6/6** |

**The shipped artifact is stale and should be replaced.** It carries only
`tokenizer.ggml.tokens` and no `tokenizer.ggml.scores`, so the runtime falls
back to greedy longest-match — and XLM-R's SP model is *Unigram*, where greedy
is not an approximation but the wrong algorithm. `fox` has no `▁fox` piece:
Viterbi gives `▁`+`fox` (6, 147797), greedy takes the longest prefix `▁fo` and
is left with `x` (5775, 425). Different ids, different embeddings.
`src/fireredpunc.cpp` predicts this in a comment; nothing had ever checked it.

A negative cosine at f16 is not precision, and the magnitudes say so more
plainly than the cosine does: 9.63 versus the 0.006 the same graph achieves on
a correct artifact.

⚠ **Unresolved, and deliberately not guessed at.** Three of six lines diverged
on the stale artifact, but only ONE had mismatched token ids — lines 0 and 2 fed
byte-identical ids and still produced different logits, while lines 1/3/5 matched
to `max_abs` 0.004 (proving the weights are the same model). Something else in
that file differs too. It is archaeology on an artifact being replaced, so it was
left uninvestigated rather than explained away.

⚠ **Scope of the claim.** What is measured here is the local
`punctuate-all-f16.gguf`. Whether the registry's `fullstop-punc-*.gguf` entries
have the same defect is UNVERIFIED — they are a different model (XLM-R large,
24L/1024) and were not downloaded. Check them the same way before trusting them.

Decoded-text case differences are expected: the blueprint works from lowercased
token surface forms, the runtime re-emits the user's original words on purpose.
