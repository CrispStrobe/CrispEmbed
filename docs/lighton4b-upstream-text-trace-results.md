# Live upstream text trace and exact operator attribution

The observed table now has a **raw-neutral upstream capture and an authenticated, aligned stage comparison**. The original source arms reproduce all282 frozen full-vocabulary score hashes, IDs, text, probability records, EOS, vision features and actual buffer-dispatch records exactly. The artifact-only reader rehashed all915 retained files (633,351,140 bytes); its [raw result](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/reader-final/text-trace-analysis.json) is SHA256 `5a8a1aaedb41aa4a485d1f9927c9e0ae5c19e02c69f9ba0d2c34fc319a4dc9b3`. This is diagnostic evidence, not decoded qualification or permission to ship Q4.

The capture executes upstream `6c01da6278ab6c69ed67d9124806b50d2b299ba7` with source `195b181432251beb5a42b31689aed2b68648b2cc`; the completed artifact reader uses `6e5c98977`. Neither artifact reader builds or runs a model. The pinned publisher Q4_K_M and native physical import have the previously verified same packed weights and scalar semantics. The native denominator remains the frozen actual source-feature table execution, independently reproduced by the corrected composed vision run. Original BF16 remains the decoded-quality denominator.

All220,160 actual prompt embedding values (86×2560) and all three packed position axes are bit-exact. Upstream actual prompt batches are `[4,72,6,4]`; native uses16-token chunks. The live capture copies tensors during synchronized evaluation, before buffer reuse. The two-token startup compatibility probe is retained separately, with exact IDs `[0,0]` and packed positions `[0,1,0,1,0,1,0,0]`; pinned upstream clears its state before the request. Missing/cropped source outputs remain explicit.

The first16 layer-zero normalization inputs match exactly. On the first four text tokens, QKV also matches exactly; Z, alpha and beta are the earliest differing measured stages:

| Live stage | RMS error | Maximum error | Source RMS magnitude |
|---|---:|---:|---:|
| Input norm, first16 | 0 | 0 | 1.15668 text /1.32048 image |
| QKV, first4 text | 0 | 0 | 2.01861 |
| Z, first4 text | 3.60387e-7 | 3.81470e-6 | 1.42230 |
| Alpha, first4 text | 8.15839e-7 | 2.86102e-6 | 5.72008 |
| Beta, first4 text | 3.89155e-7 | 1.19209e-6 | 1.86581 |
| QKV, next12 image | 2.08227e-7 | 4.76837e-6 | 1.26759 |

The [independent audit](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/audit.json) closes the **entire** live text Z/alpha/beta outputs against the earlier authenticated operator receipt `4dc28c17717082a36e9f2cad3729a31a83a97c3234b07d3c00002232744a66f5`. Live source outputs equal its upstream preferred-buffer N4 hashes. After the verified V-head row map, they equal native preferred-buffer N4 hashes. Actual native N16 outputs equal the retained borrowed-buffer N16 hashes; the operator's N4-to-N16 prefix comparison is bit-exact. All three complete live residuals exactly equal the retained preferred-versus-borrowed residuals.

Actual dispatch confirms the distinction: QKV is Q6_K/CPU_Mapped with no extra buffer; Z/alpha/beta are Q4_K/CPU_REPACK with `extra=1`. Their input normalization is identical. Thus the earliest text residual is explained by the selected quantized CPU buffer/kernel route, rather than a weight-map or BF16-rounding mismatch. The image Q6 N72-versus-native16 residual is separate and is not attributed by these N4 Q4 closures.

Tiny differences grow through the decoder. First-four text attention output RMS error is6.85e-8, postnorm1.26e-6, FFN gate4.95e-5, and layer-zero output7.90e-5. The aggregate hidden progression below weights each retained token row equally; layer31 is cropped to the last prompt token and must not be mistaken for86-row coverage.

| Layer output | Token rows | RMS error | Maximum error | Source RMS magnitude |
|---|---:|---:|---:|---:|
| 0 | 86 | 0.0000403245 | 0.000956059 | 0.201899 |
| 1 | 86 | 0.000444859 | 0.0129466 | 0.206178 |
| 3 | 86 | 0.00157140 | 0.0356417 | 0.207637 |
| 7 | 86 | 0.00470810 | 0.103399 | 0.250738 |
| 15 | 86 | 0.00735692 | 0.221208 | 0.276182 |
| 23 | 86 | 0.0197919 | 0.605328 | 0.503685 |
| 30 | 86 | 0.0528045 | 0.851820 | 0.800086 |
| 31 | 1 | 0.129631 | 0.689209 | 1.37735 |
| Final norm | 1 | 0.299452 | 1.70888 | 2.89563 |
| First logits | 248,320 values | 0.188629 | 1.06178 | 1.78552 |

At the first independent decoding disagreement, position126, all preceding126 IDs agree. Source chooses token18 (`3`) with top-two gap0.0419292; native chooses20 (`5`) with gap0.00976944. The actual aligned score error is RMS0.0603224/max0.321229; source score RMS magnitude1.74290. Source position126 hidden arrays are retained, but native hidden arrays at that position do not exist and are not inferred.

Both runs emit282 IDs and EOS. Against the same original BF16 text, native has349/607 character edits and97/106 word edits; upstream has348/607 and97/106. Native versus upstream has9/822 character edits and7/162 word edits. The one-character **floor miss** is not a claim that the transcriptions differ in only one character. Exact texts and probability records are retained in the source responses and prior composed-calibration evidence.

**No selective repack policy is promoted.** The prior full-repack decoded negative remains binding. The exact component attribution establishes how to reproduce these three source operators, but does not establish which combination improves decoded quality: changing one path can remove error cancellation elsewhere. A subsequent bounded same-input investigation can distinguish selective Z/alpha/beta correction, FFN sensitivity and actual image N72 dispatch before any runtime proposal. Calibration and held-out evidence remain separate.

The [initial PNG-container admission negative](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-text-trace-v1-setup/audit.json), [complete neutral capture with original reader failure](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/capture/text-trace.json), [omitted failed-input mount negative](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/reader-mount-negative/upstream-text-trace-bootstrap.json) and [reserved-axis reader negative](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/reader-axis-negative/text-trace-analysis.json) are preserved. The private relay copied915 files exactly without credentials in Git/logs/outputs; completed outputs retain the large capture remotely. The [manifest](../tests/results/qwen35-ocr/lightonocr3-4b-neutral-text-trace/manifest.json), raw graph metadata and small source fixtures permit independent review. No model, inference or reference was repeated to repair artifact readers.
