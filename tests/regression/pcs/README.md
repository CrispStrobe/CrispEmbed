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

## Not covered

`pre_preds`, `cap_preds` and `seg_preds` have no runtime dump hook, so they are
checked only through their effect on the decoded text — which is why the `I'm
OK` regression above shows up as a text diff rather than as a head-level one. A
change that altered truecasing while leaving punctuation alone would be caught
but not localised. `PCS_DUMP_CAP` / `PCS_DUMP_SEG` alongside the existing
`PCS_DUMP_LOGITS` would close that.
