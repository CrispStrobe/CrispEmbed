# Actual FFN input-code threshold amplification

A small upstream residual/postnorm difference crosses one activation-quantization boundary and explains the observed early FFN projection amplification. This is not evidence of a quantizer rule defect, changed weights or a safe release. Production remains unchanged; the [single-gate decoded negatives](lighton4b-gdn-full-ab-results.md) remain binding.

Source `2d78bc6a84ac6af811293f3eeb6d4f83253b32ba` completed the bounded CPU operator worker in93.173 seconds. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-ffn-postnorm-codes/v1/ffn-postnorm-probe.json) SHA256 is `c93689677044fa43a1fe98335abd6ad63db1127dfa0c2acf464772ef2bceb6db`. Both independent reviews and real synthetic RAW/preferred/prequantized/packed controls pass. All6 actual source4/nativeOFF16/ON16 projections, all32 forced-Q8 versus RAW-F32 output controls and every packed-code endpoint close exactly. Native4 RAW replay also exactly closes the actual native16 prefix.

Inputs inherit the authenticated [SwiGLU operator](lighton4b-swiglu-source-probe-results.md) source4/nativeOFF16/ON16 arrays and their original neutral-source/full-native receipts. Only the two unchanged published Q4_K gate/up tensors,26,542,080 bytes, were range/hash verified. No full-model inference, original-model reference run, runtime edit, tensor recipe, held-out case or global-repack change ran in this worker.

| Source versus actual native prefix4 | Postnorm input RMS / max error | Reconstructed Q8 input RMS / max error | Changed integer qs / sums | Changed scales / blocks |
| --- | ---: | ---: | ---: | ---: |
| GDN OFF | 1.26113e-6 / 8.79820e-6 | 1.08077e-4 / .0109365 | 1 /1 | 36 /40 |
| GDN ON | 1.27199e-6 / 7.74860e-6 | 1.08077e-4 / .0109365 | 1 /1 | 38 /40 |

Postnorm reference RMS magnitude is .446533. There are no scale-sign changes. Public converter vectors are diagnostic interventions, not directly observed CPU_REPACK buffers; forced-Q8 controls qualify their RAW output interpretation.

The differing code is token index2, feature1292, block25/lane12, located automatically from actual code differences. The source scaled input is −7.500008583 and rounds to−8; native OFF is −7.499681950 and ON is −7.499789238, both rounding to−7. This coordinate is an explanation, never a runtime conditional. Native/source Q8_K reference quantizer bodies are text-identical at the pinned commits. Changing the quantizer or nudging this boundary is not a proposed fix.

| OFF-prefix gate/up effect | Gate RMS / max error | Up RMS / max error |
| --- | ---: | ---: |
| Total native RAW versus actual source | 4.94877e-5 / .000365913 | 4.69959e-5 / .000345781 |
| Identical input, backend revision plus buffer route | 1.00390e-7 / 1.19209e-6 | 9.02799e-8 / 5.36442e-7 |
| Fixed source-preferred route, actual input change | 4.94869e-5 / .000365794 | 4.69944e-5 / .000345692 |
| Integer-code-only RAW intervention, source scales | 4.94858e-5 / .000365853 | 4.69922e-5 / .000346005 |
| Scale-only RAW intervention, source integers | 2.73410e-7 / 4.76837e-6 | 2.46601e-7 / 2.50340e-6 |

Gate/up reference RMS magnitudes are .227767/.199472. Code-space corners retain each integer vector's matching sums; they are synthetic counterfactuals, not model states. This separates the measured integer-boundary effect from floating-scale drift. RMS ratios are not unique causal variance shares. The GDN-ON case retains the same one-code difference and comparable amplification.

## Upstream cause before the FFN norm

A separate [authenticated artifact audit](../tests/results/qwen35-ocr/lightonocr3-4b-ffn-postnorm-codes/v1/upstream-cause-audit.json) links complete native OFF/ON16 attention hashes to the prior actual-input GDN replay. Embedding and initial input norm are bit-exact to source. Source/native residual additions reproduce every prefix4 F32 bit. The drift already exists in attention output: OFF6.85157e-8 and ON6.88288e-8 RMS against source magnitude .0595816. Residual errors remain6.85169e-8/6.88301e-8 against magnitude .0592997, before the postnorm errors above.

For ON, GDN postnorm and attention N4 versus N16 prefixes are exact at this cut. On identical projected inputs, changing only the Q4 output buffer route changes attention6.55971e-8 RMS/max5.96046e-7. With preferred output held fixed, replacing projected Z/alpha/beta by actual source projections changes4.20031e-8 RMS/max3.57628e-7. These are full-vector interventions, not additive RMS shares or selective quality recipes.

At the identified coordinate, actual ON RAW attention is−.0200053453; preferred output is−.0200054646, exactly the actual source coordinate. The resulting actual ON/source residuals are−.0039317608/−.0039318800. Thus upstream projection arithmetic changes the input to normalization before the code boundary. No residual-add defect appears.

The pinned native/upstream RMS kernel bodies are text-identical (SHA256 `0240273e8a8b271ecea17cfeac7d45ffbac91f92f3a3f1584a77fa1c2bb44a6a`); the Q8_K reference bodies are identical (SHA256 `c83270e32c1003a2585b77b390c8a7516558ac64cc2fb2c5727968b62e198bf1`). Text identity is source evidence, not an additional same-input normalization-kernel execution on these residuals.

The independent threshold audit authenticated2,812,128 bytes of selected arrays and recomputed the boundary, code counts and gate/up corners. The upstream audit authenticated3,485,696 bytes of existing source/native/GDN arrays. [Both audits and hashes](../tests/results/qwen35-ocr/lightonocr3-4b-ffn-postnorm-codes/v1/manifest.json) are retained; full arrays remain remotely in completed Kaggle `lighton4b-ffn-postnorm-codes-v1`, version1, and its immutable source producers. A second independent reviewer separately checked8 arrays and the one-code/corner result.

The next predeclared observed table/form diagnostic combines the existing generic Q4-repack and source-GDN gates, with OFF anchored to all retained rows/stages. It makes no token-specific adjustment and adds no SwiGLU runtime path. Prior single-gate decoded regressions remain binding; no promotion follows without complete decoded/panel evidence.
