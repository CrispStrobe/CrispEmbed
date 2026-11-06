# Mixed decoder precision: decoded correction and remaining numerical debt

Rounding the native final head output to BF16 before greedy selection reproduces
all 178 genuine mixed-BF16 IDs, full text and EOS on the observed form. The same
binary/model/input with the gate off generates 156 tokens and first differs at
zero-based decision33. Native OFF agreement against genuine mixed BF16 is CER
`19.85185%` / WER `31.57895%`; ON is exact, CER/WER0. This is a concrete output
dtype correction on this example, not full numerical or model qualification.
The official unquantized BF16 original still produces a different177-token text.

At shared-prefix decision33, the genuine source has an exact BF16 tie:

| Actual head output | ID401 | ID1778 | Greedy winner |
| --- | ---: | ---: | ---: |
| Genuine mixed BF16 | 18.5 | 18.5 | 401 |
| Native OFF | 18.5511207581 | 18.5583839417 | 1778 |
| Native ON | 18.5 | 18.5 | 401 |

An independent scalar IEEE-F32 nearest-even calculation on **every248,320 actual
OFF value** exactly equals the actual ON vector, SHA
`f8adc42a9a53d1f6798e4c56457ea9c55446c4895e7b934776b449e37fe60b58`.
All47 same-prefix nonhead arrays are byte-identical OFF/ON; no hidden change or
source value insertion explains the correction. Gate-off also preserves all229
shared prior full-native artifact hashes and156 IDs. The runtime agent owns the
default-off implementation/A-B integration; this evidence changes no defaults.

The source-only replay at `8eeb67bdeb1bf0e55b0d9ffd833a600967615765`
captures50 actual arrays at decision33. All34 score hashes and IDs are byte-exact
against the retained178-token original mixed run, with35 audited calls including
the one-token official anchor. Actual input ID is68100, rotary positions60/60/60.
The original position tensor has four planes; its separate text-position plane
is preserved, and only the final three rotary axes are compared with native.
Finalnorm/head-input and actual head-output/generated-score boundaries are exact.
All249 effective mixed parameter hashes and723 original hashes match the prior
oracle, with474 protected originals and tied head intact. CPU is AVX2, Torch git
`cf30153c4c131c8164ee7798e5022d810682e2cb`, two threads; oneDNN enablement alone
does not identify the internal GEMM route. All20 hosted source controls pass.

The single read-only comparison
[42174566100](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42174566100),
source `5222b2619709bcee8204762952d458651c58a4ed`, authenticates that source and
native A/B [42168142185](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42168142185)
at `b98883abeccd0cae39e0026571c150e444b71786`. Both native prefixes match source;
96 raw actual-stage comparisons complete, with separate nearest-even controls.
Actual Ref equal/rejection, source admission and rounding controls all pass.
The Ref binary remains
`d6898491984110b0536dcd43ddb1091f6559495a79dfb23389baedb8dc4ae246`, with unchanged
maximum0.002 / cosine0.99999 budgets. The red job is complete numerical rejection,
status `compared`, no setup/retrieval error or new model inference.

| Decision33 actual stage, OFF | Maximum error | RMS | Absolute source/native magnitude |
| --- | ---: | ---: | --- |
| Input embedding | 0.000105858 | 0.0000190419 | 0.0529785 / 0.0529885 |
| Layer0 input norm | 0.0165167 | 0.00256818 | 6.375 / 6.36532 |
| Layer0 QKV | 0.0316496 | 0.00270113 | 10.0625 / 10.06502 |
| Layer0 hidden | 0.00110197 | 0.000163324 | 2.09375 / 2.09485 |
| Layer31 hidden | 0.0837212 | 0.0123159 | 11.75 / 11.74075 |
| Final norm/head input | 0.133360 | 0.0322251 | 32.5 / 32.55272 |
| Head output | 0.136086 | 0.0299801 | 18.5 / 18.55838 |

Earliest raw difference is the embedding. Rounding a diagnostic copy of native
embedding to BF16 makes it bit-exact with source, demonstrating the effective
weight/activation precision difference at that boundary. Earliest budget failure
is layer0 input norm; rounding its output alone still has maximum0.015625. These
actual inputs already differ, so this is not proof of a standalone RMS bug.
Likewise QKV output rounding alone reaches0.0625 and cannot substitute for
same-input/effective-weight operator attribution. Raw and rounded metrics for
every stage remain separately preserved; neither is silently promoted.
The first hidden-layer output budget failure is layer1; layer0 hidden passes.

Head ON still differs numerically from source: maximum0.125, RMS0.0302963,
cosine0.999949694, and all178 whole-vocabulary score hashes differ despite exact
decoded IDs/text/EOS. Native retains F32 dequantized weights and earlier hidden
states, while genuine original parameters/operators use BF16 semantics. The
head correction therefore closes this decoded residual, not all internal
precision or numerical debt.

The same consumer reuses26 score rows where official BF16, genuine mixed BF16
and native have identical causal token prefixes (25 previous IDs; include
decision25). All five prepared inputs, original framework versions and actual
model/config source hashes agree. No later different contexts are compared.

| Same-context row | Official BF16 → mixed BF16 max / RMS | Mixed BF16 → native OFF max / RMS | Official BF16 → native OFF max / RMS |
| --- | --- | --- | --- |
| 0 | 1.578125 / 0.285799921 | 0.163356781 / 0.027429562 | 1.48261547 / 0.284899443 |
| 5 | 10.578125 / 1.54315937 | 0.656324387 / 0.078595199 | 10.1740723 / 1.51052773 |
| 25 | 1.1953125 / 0.234242961 | 0.119170189 / 0.021261929 | 1.19766212 / 0.231959641 |

Thus the previously large row5 failure already exists inside the original BF16
model with quantized weights. This includes dequantization-to-BF16 parameter
rounding. The native-versus-mixed residual includes runtime and effective-weight
precision. Scalar maxima/RMS cannot be subtracted to assign causal contributions.
The corrected native178 output remains different from official177, with official
agreement CER27.43902% / WER44.44444%, not human OCR accuracy. The observed form
remains excluded from held-out calibration and architecture-quality selection.

Complete evidence:

- [All96 raw stages and separate rounding controls; all26 causal triples](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/step33-comparison.json)
- [Structured metrics, decoded comparison, exact sources and artifact identities](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/review.json)
- [Original50-stage manifest, parameter hashes and observer controls](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/source-trace.json)
- [Original final audit](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/source-audit.json) and [source worker receipt](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/source-bootstrap.json)
- [Complete native head A/B receipt](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/native-report.json)
- [ZIP member preservation and independent actual head-transform proof](../tests/results/qwen35-ocr/lightonocr3-4b-mixed-step33/archival-and-head-transform.json)

Artifact `13483410105`,115,613,694 bytes, digest
`f1ac52d246452ea592db7b32ae4a289ae3db01e39d79097804e5b56b18b463ed`, expires
`2027-02-04T06:27:43Z`. It preserves all178 original mixed score rows
(176,803,840 bytes), all50 source stages,48 native stages per arm,26 official and26
native same-context rows, and all Ref/control/build logs. Counts/sizes are
independently checked through its ZIP directory; only small reports and two
needed head rows were range-read locally. The native A/B artifact remains
`13482930295`, digest
`fdfd24ebcfa7b0167fb79deb747c423c904d205f30a7d968f47e77f3fe28f1dc`, expiry
`2027-02-04T05:23:28Z`. No model payload is uploaded. All earlier official,
quantized, capped, calibration and control negatives remain intact; no model
qualification, release/app pin, default or new PR follows.
