# Retained LightOn4B upstream drift

Draft PR151 stacks PR150 `9ce83605e`. It analyzes the immutable synthetic
32-layer panel [41979168300](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41979168300)
and executes only a bounded retained-input layer1 diagnostic. Neither original
nor native32-layer forward is repeated. Production runtime, tested block headers,
frozen arrays and existing thresholds are unchanged.

## Earliest measured amplification

The retained prefill layer outputs show smooth cumulative drift, rather than an
isolated jump at the later layer21 cache rejection. Relative-L2 error means the
L2 norm of the original/native difference divided by original output L2 norm.
The earliest and largest adjacent multiplicative increase is layer0→1:29.84%.
This selects a diagnostic target; it introduces no new quality threshold.

| Prefill layer | Type | Max absolute error | Original max magnitude | Relative-L2 error |
|---|---|---:|---:|---:|
| 0 | Linear | 3.21865082e-6 | 1.78922069 | 1.37052144e-6 |
| 1 | Linear | 5.96046448e-6 | 2.71021986 | 1.77949970e-6 |
| 3 | Full | 1.12652779e-5 | 4.56952763 | 2.21826008e-6 |
| 7 | Full | 2.54511833e-5 | 8.80249786 | 2.92306122e-6 |
| 15 | Full | 6.72340393e-5 | 17.8235741 | 3.90458039e-6 |
| 20 | Linear | 9.60826874e-5 | 24.0699482 | 4.43497423e-6 |
| 21 | Linear | 1.06334686e-4 | 25.5360756 | 4.54611888e-6 |
| 31 | Full | 1.80721283e-4 | 34.5850983 | 5.42456156e-6 |

The exact32-row progression is preserved in the review receipt. The original
fixture explicitly shares one linear weight set across24 distinct modules and
one full-attention set across8 modules. This curve characterizes that synthetic
fixture, not independently trained checkpoint layers. The retained panel observes
layer outputs, so it cannot locate the first operator difference inside layer0.

## Exact layer1 and operator replay

Hosted [41981848201](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41981848201)
at `132c800b63a888f9639dcdc3094b6b38ae0c538b` rehydrates the exact original/native
layer0 outputs remotely. It regenerates all26 synthetic weight hashes exactly
and verifies all six original/native layer1 hidden/conv/recurrent frozen anchors
byte-exact. Only the genuine original decoder layer1 and the native block run
on each retained input. Source-fed normalization/FFN probes exist only in the
validation executable; no source cache is fed into native execution.

All22 same-input block comparisons,4 source-normalized GDN boundaries and8
source-fed FFN comparisons pass the unchanged budgets. On the retained original
input, the earliest locally observed difference is input RMS output:
max2.3841857910e-6 against magnitude3.452250242. This locates an observed operator
boundary; it does not identify a particular reduction implementation cause.

| Compared boundary | Source-input case max error | Native-input case max error |
|---|---:|---:|
| Ordinary FFN gate | 2.26497650e-6 | 2.86102295e-6 |
| Original post-norm fed to gate projection | 2.14576721e-6 | 2.26497650e-6 |
| Original post-norm fed to up projection | 2.23517418e-6 | 2.50339508e-6 |
| Ordinary FFN activation | 4.76837158e-6 | 6.43730164e-6 |
| Original gate/up fed to activation | 2.38418579e-7 | 2.38418579e-7 |
| Original activation fed to down projection | 2.05636024e-6 | 2.53319740e-6 |
| Ordinary final hidden | 2.62260437e-6 | 3.87430191e-6 |

Dense-projection differences persist with exact original inputs and verified
weights; the activation operator itself contributes a much smaller discrepancy.
Thus amplification cannot be assigned only to activation or only to upstream
normalization. Changing only retained input in the original layer1 produces
max hidden difference4.36604023e-6 against2.71021986. Together with local
same-input discrepancies, this bounds the first observed layer-to-layer growth.
No specific CPU instruction, summation order or corrective algorithm is proven.

The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-upstream-drift.json)
verifies all seven direct source records and independently recomputes the32-row
progression from frozen report metrics. It retains every stage magnitude/error,
weight count and anchor control. Large raw arrays remain in hosted artifacts;
this review does not claim local reanalysis of every remote vector.

The earlier layer21 replay remains distinct: changing only retained input there
reproduces the tiny-row cache rejection. The complete decoder remains negative
at290/291 strict comparisons. No threshold, precision, runtime default, full-model
qualification, vocabulary projection, tokenizer/text/EOS evidence or release is
changed by these diagnostics.
