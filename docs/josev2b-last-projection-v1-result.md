# Josev final gate/up precision result

[Actual40853720400](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40853720400)
at`3cc96dbabcc75b7c7665164e70a26db93da653e7` completes all three predeclared
recipes after strict227-stage/four-typed F32 control. The
[independent review](../tests/results/josev2b/last-projection-v1-review.json)
authenticates110 small ZIP members, checks all442 storage roles/292 protected F32
hash declarations, actual Ref agreement and unchanged stage/final gate decisions.
Large models/arrays remain remote; no original forward or NEW20 native access.

| Recipe | Actual stored tensors | Bytes | Strict stage failures | Worst probability error |
| --- | --- | ---: | ---: | ---: |
| Final gate Q4_K |1 Q4 +149 F16 +292 F32|4,898,017,760|106|0.000347852707|
| Final up Q4_K |1 Q4 +149 F16 +292 F32|4,898,017,760|105|0.000383540988|
| Final pair Q8_0 |2 Q8 +148 F16 +292 F32|4,892,512,736|75|0.0000433921814|

All five discrete decisions and both ordinal gates pass in each recipe. Both
one-Q4 recipes fail probability0.0001, despite some smaller layer23 hidden
errors than the older two-Q4 recipe. More protected precision therefore does
not produce monotonic final-task improvement in these four development cases.
The two-Q8 recipe passes the limited discrete/probability/ordinal screen:
ordinal errors0.0000458658/0.0000219792 and worst probability0.0000433922.
This limited screen is not the full strict comparator's result.

All three first fail strict hidden gates at layer23. First-case Q8-pair error is
max0.05596924/RMS0.00155646/cos0.99999103278, original maximum magnitude16.73357.
The worst case maximum is0.06473923. Its75 strict stage failures, raw logit
failures and nonexact serialized typed responses remain negative. Complete JSON
changes can occur even within the probability budget because displayed values
are rounded. No threshold is changed and no recipe is selected or published.
These conservative mixed files are predominantly floating-point storage, not
broadly Q8/Q4-sized artifacts.

Next independently review a selector/holdout protocol before any NEW20 access.
For Q4 repair, collect source-bound native FP32 activation importance on the old
four cases only, with original227-stage controls and ordinary/observer neutrality;
then use actual C imatrix quantization under a separately frozen calibrated
recipe. Do not imatrix-calibrate Q8 (its encoding ignores importance), tune on
NEW20, replace old negatives or infer final quality from a hidden cosine.
