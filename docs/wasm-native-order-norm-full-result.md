# WASM live-input norm full A/B result

[Actual40852835039](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40852835039)
at`64eef68493deb240e81287e249f55def08d5753c` completes and is rejected by
unchanged0.002/0.99999 gates. The [independent review](../tests/results/packaging/wasm-native-order-norm-full-review.json)
authenticates all119 small ZIP members, checks36 actual Ref/independent numerical
decisions and204 returned-byte records. Raw traces/modules remain remote; no
model or raw artifact was downloaded for this review.

All12 component native norm byte recoveries pass. Native20 final/16 stage
anchors, old OFF byte controls, observers/repeats, IDs and context isolation pass.
The full ON run passes4/20 final comparisons and10/16 stage comparisons;
16 finals and6 stages fail. Both tested native thread targets show the same
numerical outcome. Frontend helpers are scalar while the historical ggml backend
uses SIMD; preserve that actual per-translation-unit compiler profile.

| Captured boundary | Maximum error | RMS | Minimum row cosine | Original maximum magnitude |
| --- | ---: | ---: | ---: | ---: |
| First attention0 difference | 1.19209e-7 | 2.15272e-8 | approximately1 | retained in receipt |
| First failing layer3 | 0.00291892886 | 0.00023248944 | 0.999999184331 | 10.50798 |
| Encoder output | 0.02850568295 | 0.00623421436 | 0.999848220828 | 5.83756 |

For the first sentence, normalized final error is0.00156465905 but cosine
0.999954849513 fails. The worst final-case cosine is0.999723622438; small absolute
error alone does not qualify it. Preserve all failures and the old580 baseline.

The candidate is active: ON/OFF output bytes differ in9/10 cases, while explicit
context0/1/0 returns OFF/ON/OFF bytes exactly. These same-module observations and
the sole environment consumer support route activation. They do not establish
that every live norm input/output matches its native counterpart. The retained
full-model trace contains only eight boundary tensors, with no live norm pairs.

Next capture only the frozen first sentence: prove model geometry/backend and13
connected norms, retain every live norm input/output plus layer3's operator
operands, and require ordinary/observer/repeat neutrality. Replay those exact
inputs through actual native a06 norm and the candidate helper with actual
crispembed-diff before changing arithmetic or precision again. If local norms
agree, inspect actual quantized FC1 inputs/scales and upstream propagation.
No full ten-case rerun, default flip or release qualification follows yet.
