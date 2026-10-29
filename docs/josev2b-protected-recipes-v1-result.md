# Josev protected Q8 and final-pair Q4 result

[Actual run40838785499](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40838785499)
at`bf00e4524e5d2d1f261ec20d666c3045138f65d4` completes all four
predeclared recipes. The [independent review](../tests/results/josev2b/protected-recipes-v1-review.json)
authenticates the complete small ZIP and its133 members, independently checks
reported threshold decisions, all442 tensor role counts and protected292 F32
payload hashes. Large models and arrays remain remote; this review does not
independently reread their payloads. Hosted actual crispembed-diff agrees with
independent metrics. The227-stage/four-response F32 control passes, and ordinary,
observed and repeated native outputs agree. No original forward or NEW20 native
output is produced.

| Recipe | Stored bytes | Strict stage failures | First failure | Ordinal errors |
| --- | ---: | ---: | --- | --- |
| Q8 all150 projections | 3,630,289,376 | 200 | layer0 | 0.00933951 /0.00442733 |
| Q8 with24 down projections F32 | 4,517,384,672 | 200 | layer0 | 0.00622739 /0.000343859 |
| Last gate/up Q4; down F32, other projections F16 | 5,483,909,600 | 106 | layer23 | 0.000156209 /0.000317603 |
| Last gate/up Q4; other148 projections F16 | 4,879,929,824 | 106 | layer23 | 0.000156745 /0.000317678 |

Each recipe retains all292 critical tensors F32. Source values and packed scales
are finite; F16 rounding has an independent integer oracle. The actual C quantizer
writes typed standard file types7/14 and profile metadata; source F32 correctly
omits the two quantization metadata fields. The initial header-assumption failure
[40834737477](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834737477)
and its separate receipt remain preserved.

For the first case, Q8-all layer0 has maximum error0.00653386, RMS0.000472059,
cosine0.99989551 and original absolute magnitude1.92782. Final-pair Q4 with F16
down first fails at layer23: maximum0.820132, RMS0.0195813, cosine0.99865779,
original magnitude16.7336. The minimum cosine across that recipe's first failing
stages is0.99848469. These metrics and every case's raw logits/probabilities and
ordinal errors are retained separately from typed decisions.

Both final-pair Q4 recipes pass the fixed0.001 ordinal gates and retain all five
discrete decisions. Neither passes the probability screen: F16-down errors are
0.000145703554 and0.000179201365, against0.0001. F32-down errors are
0.000145375729 and0.000179156661. Complete serialized typed responses also differ.
The tiny down-precision effect is bounded to these four development cases.

No recipe is selected or qualified. The next storage-only experiment isolates
only final gate Q4, only final up Q4 and final pair Q8, with all other projections
F16 and the same292 critical F32 tensors. Preserve original budgets and all
negative flags; do not open NEW20 until its predeclared development screen passes.
Calibration and source-fed operator attribution remain distinct follow-ups.
