# FireRedPunc blueprint reference

`blueprint_ref.txt` is a per-token reference dumped from the **Python
blueprint** — `FireRedTeam/FireRedASR2S`'s `fireredpunc/punc.py` +
`models/fireredpunc_bert.py` — run against the official `FireRedTeam/FireRedPunc`
checkpoint (Apache-2.0) in f32 torch. It pins token ids, per-token 5-class
logits, argmax predictions and the blueprint's own punctuated text for the ten
lines in `corpus.txt`.

It is checked in (9 KB) rather than hosted, unlike the audio `-ref.gguf`
archives, because it is text and tiny — which means the parity check runs with
no torch, no `transformers` and no 407 MB checkpoint download:

```bash
python tests/firered_punc_parity.py build /path/to/fireredpunc.gguf \
    tests/regression/fireredpunc/blueprint_ref.txt
```

Regenerate (needs the checkpoint and torch) with:

```bash
python tools/dump_fireredpunc_reference.py \
    --model-dir /path/to/FireRedPunc \
    --corpus tests/regression/fireredpunc/corpus.txt \
    --output tests/regression/fireredpunc/blueprint_ref.txt
```

## Measured

Against this reference, after the `[SEP]` fix (the blueprint prepends `[CLS]`
and appends nothing; this port used to append `[SEP]`, and BERT being
bidirectional that shifted every token's distribution):

| GGUF | cos_min | max_abs | preds | decoded (ignoring case) |
|------|---------|---------|-------|-------------------------|
| f16  | 1.000000 | 0.0021 | 119/119 | 10/10 |
| q8_0 | 0.999234 | 0.2147 | 119/119 | 10/10 |
| q4_k | 0.935078 | 1.3229 | 118/119 | 9/10 |
| f16, `CRISPEMBED_FIREREDPUNC_SEP=1` (old shape) | 0.931090 | 1.8431 | 118/119 | 9/10 |

Read the last two rows together: q4_k's dip is the quantiser, and the proof is
that the f16 arm of the same graph is exact. The old shape produced a q4_k-sized
error *at f16*, which is what identified it as structural.

**Decoded-text case differences are expected and are not a port bug.** Upstream's
`RuleBaedTxtFix.fix` lowercases the whole string and then re-capitalises, so it
emits `google` for `Google`. CrispEmbed deliberately emits the user's original
words: `--punct-model` is a post-processor over OCR text, where silently
rewriting characters would be a regression. Punctuation differences ARE a bug.
