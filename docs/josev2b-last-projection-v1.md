# Josev final-projection precision isolation

This development experiment follows [protected recipe run40838785499](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40838785499).
All four candidates completed with strict F32 controls and unchanged original
references; none qualifies for publication. Both final-pair Q4 candidates pass
the two ordinal gates but miss two probability gates: approximately0.000145 and
0.000179 against0.0001. Keeping all24 down projections in F32 instead of F16
changes those errors only around1e-7 on these four cases. This bounded observation
does not establish general tensor insensitivity.

The new [closed protocol](josev2b-last-projection-v1.json) predeclares three
recipes, each retaining all292 critical tensors byte-exact F32:

| Recipe | Final layer23 gate | Final layer23 up | Other projection matrices |
| --- | --- | --- | --- |
| q4-last-gate | Q4_K | F16 | F16 |
| q4-last-up | F16 | Q4_K | F16 |
| q8-last-pair | Q8_0 | Q8_0 | F16 |

Use [the hosted workflow](../.github/workflows/josev-last-projection-v1.yml)
on its exact reviewed source branch. It authenticates retained originals without
original forwards, executes the unchanged actual C quantizer, validates all442
storage roles/metadata/finite values, independently checks F16 rounding and
protected F32 bytes, and runs strict F32 before candidates. The native ordinary,
observed and repeated calls must agree before numerical interpretation. Every
candidate retains all227 original stage comparisons, actual crispembed-diff
results, logits, typed decisions, probability and ordinal errors. Strict failures
remain negative even when the diagnostic collection completes successfully.

The same old four development cases are the only model inputs. NEW20 remains
closed. No threshold, original, old recipe, runtime arithmetic or default changes.
Passing discrete decisions or an intermediate cosine alone cannot qualify a
candidate. Selection requires the fixed development screen followed by the
separately declared held-out/public-consumer protocol; this experiment does not
publish weights or select a recipe. Activation/imatrix calibration is a separate
future experiment, not a property of these storage-only trials.
