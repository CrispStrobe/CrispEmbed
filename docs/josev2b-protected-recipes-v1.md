# Josev protected Q8 and final-layer Q4 development protocol

The [actual A/B/C experiment40787077302](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40787077302)
at06a96cbc completed all three candidates, with a positive strict F32 control,
227 intermediate comparisons per candidate, actual `crispembed-diff`, ordinary /
observed / repeat identity and protected292 F32 tensors. The
[independent review](../tests/results/josev2b/mixed-recipes-v1-review.json)
preserves all numerical and final-output failures. None qualifies.

| Recipe | F32 / F16 / Q4 tensors | First failing layer | Failed stages /227 | Two ordinal absolute errors |
|---|---|---|---|---|
| A |292 /78 /72 |0 |203 |0.002969876;0.027582288 |
| B |316 /78 /48 |0 |201 |0.019949734;0.004645333 |
| C |316 /118 /8 |20 |118 |0.001585498;0.000765428 |

C restores all24 FFN down-projections to F32 and stores every projection before
layer20 in F16 or F32. In the first development case, its layer20 cosine is
0.99826210, maximum error0.110197 and RMS0.010144, versus original magnitude
17.50167. Earlier stages pass the unchanged0.002/0.99999 screen. The first
ordinal still exceeds its separate0.001 budget. All five discrete decisions
matching does not qualify probabilities or ordinal expectations. These role
ablations motivate finer storage experiments; they do not isolate an individual
operator's causal error or prove that down-projections alone explain drift.

The [hosted Q8 controls40829558531](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40829558531)
at6781befb independently pass actual C quantization/header readback for13
methods, Q8/Q4 profile and critical-tensor rejection controls, and variable-length
STRING metadata stamping with unchanged payloads. The
[receipt](../tests/results/quantization/portfolio-q8-profile-controls-review.json)
is model-free. It is permission to test new profiles, not model qualification.

The new closed caller uses that exact runtime/backend, the actual C
`crispembed-quantize` implementation and `crispembed-josev-packed-mixed-v2`.
It predeclares these four recipes before any new candidate outputs:

| Recipe | F32 | F16 | Q8_0 | Q4_K | Storage rule |
|---|---:|---:|---:|---:|---|
| `q8-all` |292 |0 |150 |0 |Quantize only the150 reviewed projection matrices. |
| `q8-f32-down` |316 |0 |126 |0 |Additionally restore all24 FFN down-projections to F32. |
| `q4-last-f32-down` |316 |124 |0 |2 |Only layer23 gate/up are Q4; all down-projections F32. |
| `q4-last-f16-down` |292 |148 |0 |2 |Same two Q4 matrices, but down-projections F16. |

Every head, embedding, norm, bias and other nonselected tensor remains byte-exact
F32. Source identity,442 semantic shapes, all finite source values, independently
rounded F16, finite packed scales and per-tensor hashes are mandatory. The actual
C writer stamps the existing profile STRING before laying out the header; there
is no in-place length-changing edit or Q8-as-Q4 label. Q8 uses standard header7,
Q4_K header14 and genuine tensor types8/12 respectively. No production arithmetic,
default, budget, original dump or old recipe changes.

Run `josev-protected-recipes-v1.yml` on its exact reviewed feature source after
source-closure verification. The workflow authenticates retained original arrays
without original inference, then requires strict F32 parity before any candidate.
All four candidates collect227 mapped stages, actual Ref/independent metrics,
absolute/RMS/magnitude/cosine errors, logits/probabilities, discrete decisions and
ordinal expectations. Unset/empty/zero quant gates must reject; recovery and
ordinary/observed/repeat controls remain mandatory. Numerical negatives are
retained while later predeclared recipes continue. Reports and full models / raw
arrays are retained separately as90-day remote artifacts.

This is the original four-case development panel only. NEW20 native outputs
remain unopened and selection/publication flags stay false. Review actual
candidate results before a separately versioned held-out owner admits any model.
If none passes the declared development gates, preserve every rejection and
scope source-fed replay at the earliest failing boundary; do not widen budgets
or describe component cosine as task qualification.
