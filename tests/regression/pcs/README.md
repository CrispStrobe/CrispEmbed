# PCS blueprint reference

Ground truth for the **PCS** engine (punctuation + capitalisation + sentence
boundary, `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`), the third
and last of the three punctuation engines to get one. Four registry entries
(`pcs`, `pcs-iq4xs`, `pcs-q4k`, `pcs-q8`), four classification heads, and until
now nothing checking any of them.

```bash
python tests/pcs_parity.py build /path/to/pcs-xlmr-base.gguf \
    tests/regression/pcs/blueprint_ref.txt
```

Regenerate (needs `pip install punctuators` plus the ONNX model):

```bash
python tools/dump_pcs_reference.py --model-dir /path/to/pcs-onnx \
    --corpus tests/regression/pcs/corpus.txt \
    --output tests/regression/pcs/blueprint_ref.txt
```

The blueprint is not reimplemented: the model ships `pipeline.py`, which
delegates to `punctuators.models.PunctCapSegModelONNX`, so the dumper runs that
package. Its ONNX graph exports ARGMAXED predictions, not logits, so there is no
cosine here — the comparison is on the discrete decisions, which is the stronger
gate since they are what produce the text.

## Measured — the port is correct, and the imatrix default is earned

| artifact | post-punc preds | decoded text |
|---|--:|--:|
| local `pcs-xlmr-base.gguf` | 67/67 | **6/6 exact** |
| local `pcs-xlmr-base-q4_k.gguf` (no imatrix) | 67/67 | 5/6 |
| registry `pcs-xlmr-base-q8_0.gguf` | 67/67 | **6/6 exact** |
| registry `pcs-xlmr-base-q4_k-imatrix.gguf` (**the default**) | 67/67 | **6/6 exact** |

Decoded text is gated on EXACT equality here, unlike the other two punctuation
harnesses. Those excuse case differences because their runtimes deliberately
re-emit the user's original words; PCS truecases itself, so there is no
deliberate deviation to excuse and the strings must match character for
character. They do.

**The single failure localises the quantisation cost precisely.** Plain q4_k
gets every one of the 67 punctuation decisions right and still loses one
truecasing — `I'm OK` becomes `I'm ok`. So the punctuation head is robust to
q4_k and the cost falls entirely on the TRUECASE head. The imatrix build
recovers it and matches the f32 reference exactly, which is the first
decoded-output evidence for the registry's `q4_k-imatrix` default; its
description ("4.2x lower KL vs f32") had only ever been a divergence number.

## All four heads are compared

`PCS_DUMP_LOGITS` only ever covered post-punc, so the other three heads could be
checked only through their effect on the restored string — which is why the
`I'm OK` regression above first showed up as a text diff with nothing to say
which head caused it. `PCS_DUMP_PRE`, `PCS_DUMP_SEG` and `PCS_DUMP_CAP` close
that, and the localisation is immediate:

```
plain q4_k, against this reference
  post preds  : 67/67
  pre         : 67/67
  seg         : 67/67
  cap         : 66/67   token 37 (`▁ok`): ref=111 ours=000
  decoded text: 5/6      I'm ok  vs  I'm OK
```

One token, one head. The punctuation path is untouched by q4_k; the whole cost
is the truecase head dropping a single token's capitalisation.

All three hooks are written after every head has finished, in one block, so the
four dumps are aligned by construction rather than by separate call sites
agreeing about ordering. They **append**, like `PCS_DUMP_LOGITS` — a consumer
must delete the file first or a stale run silently shifts every comparison.

Two details the format does not make obvious:

- **`PCS_DUMP_SEG` has two columns.** Column 0 is
  `softmax(logits)[boundary] > 0.05`, the ONNX `seg_preds` output — a low tuned
  threshold, *not* argmax. Column 1 is the hard argmax, which conditions the
  truecase head's "is-sentence-initial" input. The blueprint exports only the
  former, so only that is gated; the latter is dumped so a cap mismatch can be
  traced to its conditioning rather than blamed on the cap head.
- **`PCS_DUMP_CAP` is per CHARACTER of the token, `▁` included.** `▁hello`
  reads `1100…`: bit 0 covers the `▁` (ignored downstream) and bit 1
  capitalises the `h`. It is not one flag per token.
- **Bit 0 of a `▁`-initial piece is DON'T-CARE too.** `▁` is the word-boundary
  marker, not a character that gets cased, and the reconstruction never reads
  its bit. CI's `q4_k-imatrix` emits `011` for `▁ok` where this box emits `111`,
  with byte-identical decoded text — a hardware-dependent quant difference on a
  bit that cannot matter. The harness skips it.
- **Bits past the piece's character count are DON'T-CARE, and must not be
  compared.** The head always emits 16; the reconstruction reads bit `c` only
  for character `c`. Comparing all 16 made a *correct* artifact fail — the
  registry's `q4_k-imatrix` differs from the f32 reference on bits 4..12 of
  `▁ok` (3 characters) while producing byte-identical text. The harness
  truncates to the piece length, after which imatrix matches and plain q4_k
  still mismatches on bit 0, which is the real defect. A test that flags
  don't-care values is not stricter, it is broken — and this one was, three
  separate times: comparing all 16 bits, then comparing the ▁ bit, each caught
  only by a correct artifact failing (the second by CI, on hardware that
  quantises differently from this box).

The hooks live in all three copies of pcs.cpp (CrispEmbed's, and both of
CrispASR's), verified by CrispASR's `test-copies-in-sync` and by running this
harness through the sibling build — 67/67 on every head either way.
