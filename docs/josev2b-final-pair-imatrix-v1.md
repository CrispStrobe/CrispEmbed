# Josev final-pair native-F32 activation calibration

This separately versioned development experiment follows the retained
[protected recipe rejection](josev2b-protected-recipes-v1-result.md) and
[final projection ablations](josev2b-last-projection-v1-result.md).
The Q8 final pair preserves the limited old decision screen but still fails75
strict stage checks and full JSON equality. Individual Q4 gate/up arms also
remain rejected. No previous producer, threshold, alias or model is replaced.

The [protocol](josev2b-final-pair-imatrix-v1.json) freezes a new caller and exact
unchanged runtime678/a06. The actual route is
`josev2b_backbone::run_impl → decoder_hidden_cpu → qwen35_decoder.cpp`.
Immediately after final-layer post-attention RMS normalization, the same F32
`normalized` buffer feeds gate and up projections, each2048 input columns. This
is a CPU vector route; the generic scheduler imatrix collector is not installed
here. The original251 arrays contain layer23 output, not this pre-FFN operand.
New activation data is therefore labeled **native F32 calibration**, never
Python/reference activation gold.

A disposable observer-only overlay captures that const buffer without changing
normalization/projection math or adding graph nodes. The ordinary archive must
exclude the capture functions and strings; the observed archive must contain
them. The copied test probe arms capture only around the successful primary
request, after the existing malformed-request check. Exactly one layer23
capture is required per case. Model-free compiled controls reject wrong stored
types/names/geometry, nonfinite values, duplicate/stale callbacks and incomplete
captures, then prove recovery.

The four old inputs remain `mixed-de`, `null-and-options`, `json-unicode` and
`state-only-clip`:319/197/165/512 tokens. Ordinary/observed/repeat original227
Ref/independent comparisons and complete typed responses must pass at F32
before capture data is admitted or any candidate quantization runs. Observed
and repeat normalized operands must be bit exact. Capture occurs within those
existing F32 control calls, so it adds no extra model forward. No NEW20 data or
new original forward is accessed.

Only the first observed buffer from each case contributes statistics. All
1,193 actual prefill rows are included once; repeat captures are verification,
not duplicated calibration samples. Each input column accumulates the ordered
case/row/column F64 sum of squared native F32 values, with separate multiply/add
and one F32 cast at storage. This declares the statistic arithmetic; it does
not assert bit identity to a collector that never ran on this route. The GGUF
uses the existing `crispembed-imatrix`, version1 format: two F32 `[2048]` raw-sum
vectors, named exactly `l.layers.23.mlp.gate_proj.weight` and
`l.layers.23.mlp.up_proj.weight`, plus matching U64 `count.<name>=1193` metadata.
The vectors are identical because the input is shared. Readback reproduces the
existing loader's one F64 reciprocal of the count, multiplication by each
stored sum, and final F32 mean-squared values.
Finite, nonnegative, nonzero coverage and exact hashes/axes/counts are required.

The two predeclared storage arms are:

| Arm | Stored types | Calibration |
| --- | --- | --- |
| `uncalibrated-pair` |292 critical F32 +148 F16 +2 final gate/up Q4_K |None; exact old model SHA`289798eb4066585e59dec70c1c5447b02eece199699fce1b6a6e38933d61d31d` must reproduce |
| `calibrated-pair` |Same roles, names, shapes and4,879,929,824-byte extent |Actual C quantizer `--imatrix`; exactly two vectors must load and both named Q4 weights must report actual use |

Both use the actual unchanged C quantizer, standard14/QNTv2/mixed-v2 metadata,
independently checked F16 rounding, finite packed scales and all292 critical
F32 payload hashes. There is no Q8 imatrix, fallback encoding or header relabel.
The generic loader's nonfatal empty/invalid-imatrix behavior is not acceptance:
this caller rejects missing, malformed, skipped, unused or unexpected vectors.
The calibrated model may change only the two Q4 payloads; all other stored
payloads and model metadata remain determined by the identical recipe.

Each model has three existing four-case primary passes: ordinary, observed and
repeat. F32 uses12 full prefills; each candidate also retains the old two-forward
loaded-model recovery test, giving14 per candidate and40 total maximum. The36
primary and4 recovery forwards are reported separately. Malformed-schema and
disabled-gate controls reject before a model forward. No original or holdout
forward is included.

All227 stage flags, raw logits/probabilities, ordinal expectations and full typed
JSON are retained for both arms under the original0.002/.99999, logits0.002,
probability0.0001 and ordinal0.001 gates. A calibration change is not a new
quantization tolerance or task benchmark. Numerical failures remain negative;
no selected recipe, NEW20 activation, publication, runtime default, release or
app qualification follows. The
[standalone hosted workflow](../.github/workflows/josev-final-pair-imatrix-v1.yml)
retains canonical models/captures/adapters remotely for90 days and small source,
statistics, quantizer-use, compiler, Ref and typed reports for review. Local work
is source/stdlib only; execution remains pending peer review.
