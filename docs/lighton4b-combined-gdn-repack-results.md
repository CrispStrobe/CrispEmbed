# Combined GDN/repack: local table match, form failure

**The combined policy remains unqualified. Keep both experimental gates default-off; do not ship or promote.** Matching one observed table does not override the form character regression or the unresolved held-out receipt failure. Neither individually negative alternative is promoted.

The bounded Kaggle run (kernel `lighton4b-combined-gdn-repack-ab-v1`), version1 at source `374d0438765c9e111b11499785eac86a46773452`, completes in879.92 seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-combined-gdn-repack-ab/v1/combined-gdn-repack-ab.json) SHA256 is `4428c4a2d8a877f64488847dfd7145fb0ccd1ba26810e6c0edf2229e6e40eac9`. Exact tested runtime is `cd50e9e691530525fb998edda8a449cc0725be91`; physical model SHA remains `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.

| Observed case | OFF → combined ON tokens, all EOS | Character edits against original BF16 | Word edits | Actual upstream edits against same reference |
| --- | --- | --- | --- | --- |
| Table | 282 → 282 | 349 →348 /607 | 97 →97 /106 | 348/607,97/106 |
| Form | 167 →167 | 175 →**194** /656 | 45 →40 /90 | **189**/656,46/90 |

Table ON exactly matches all282 actual upstream IDs and raw UTF-8 text. Form ON improves word edits but exceeds the upstream character count, so it fails the predeclared joint floor. No aggregate rescue or normalization change applies. Native versus upstream form raw-text distance is196 characters/45words; matching-prefix length31 is not a quality gate.

OFF reproduces all282/167 frozen rows, all38 text anchors, vision/processor arrays, IDs and EOS. Exactly two generic flags change: `CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK` and `CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS`. Actual pixels, merged vision, prompt, embeddings and positions remain identical. Every emitted row is finite with actual greedy argmax; ON retains216 quantized buffers charged1,662,861,312bytes, with zero converted F32 weight entries/bytes. Both original component proofs and current supported cache/attention/GDN controls pass before acquisition.

First-score RMS against actual upstream is table0.188629→0.188863 and form0.105889→0.123101; maximum errors are1.061775→1.004997 and0.558670→0.670184. Table source/native ON score RMS magnitudes are1.785518/1.817411, form2.334931/2.308214. Numeric error and decoded equality can move independently.

Native paired stages first differ at layer0: table RMS3.42160e-5/max0.000956059 and form4.79429e-5/0.00196552. At layer31 differences are0.0821637/3.39539 and0.0769153/2.04667. These are OFF/ON native arrays, not actual upstream form hidden states. Actual same-weight upstream form text/GDN/FFN internals are not captured here.

Child wall times are table131.86→143.52 seconds and form95.97→113.15; sampled RSS4.28→5.95GB and4.33→6.01GB. Order is fixed and capture/load costs are included; this is not a balanced throughput benchmark or a speed claim. One worker runs four independent greedy cap512 arms with one teacher row for first-score identity, not a177-position teacher panel. Neither original nor upstream model is rerun; no held-out case is executed or selected.

The independent audit rehashes14 selected closed outputs/profile/pack files and authenticates the four reference files against receipts `712d4c86…`/`36a9b14b…`. It recounts raw character/word edits against both references, checks actual IDs/EOS/summaries, the exact two-gate difference, cache counts and immutable candidate SHA. Large vocabulary/stage arrays and the exact binary/backend bundle remain remote; their full finite/hash/argmax closures are producer checks, not an independent second full-array replay. Prior individual negatives and all held-out evidence remain preserved.
