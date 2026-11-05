# LightOn4B same-input checkpoint layer0 attribution

The complete Q8/Q4 decoded rejection remains unchanged. A bounded checkpoint
layer0 probe separates shared upstream vision differences from local decoder
operators: every arm receives the same authenticated original154-token input,
not native image features. Diagnostic source is
[audit/lighton4b-layer0-source-fed-20261105](https://github.com/CrispStrobe/CrispEmbed/tree/7d8efee5f).
No new model reference, full-model forward or generation was executed.

Original BF16 layer0 replay in its source environment at4498559a22 reproduces all
394,240 retained hidden values bit-exactly, with exact hook neutrality. The
[producer report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-bf16-source-environment.json)
retains module/package/input/range/tensor hashes. Only the immutable official
226MB layer0 interval is fetched; the whole9GB checkpoint is not rehashed.
Interval bytes, recovered tensors and input match both hosted CPU attempts.

Initial hosted CPU [42022722133](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42022722133)
at7b7cb1a84 correctly stops before native/candidate execution: BF16 hidden differs
from frozen by max0.0078125/RMS0.00009840313 against magnitude8.5625, despite exact
hook neutrality. Its [negative report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-source-fed-initial.json)
is preserved. Source-environment and hosted input-norm hashes match; first
QKV/z/a/b projections differ. The first hosted attempt enabled deterministic
algorithms, while the frozen BF16 source used defaultOFF.

Explicit unanchored diagnostic [42023499031](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42023499031)
at567e59fa0 matches BF16 deterministicOFF but retains a red frozen replay:
max0.015625/RMS0.0001076533/minimum row cosine0.9999971916. Host and deterministic
changes are not a controlled isolated pair; no causal backend attribution follows.
Its status is `diagnostic-compared-unanchored`, with parity/quality false. The
[complete producer report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-source-fed-v2.json)
retains every original/native intermediate metric and all source/weight identities.

All80 same-input/same-weight native checks pass unchanged0.002/0.99999 budgets:
16 stages across official same-weight F32 and both Q8/Q4 native-quant/native-F32
arms. These F32 operations are single-layer operator diagnostics, not a new F32
model reference or prerequisite. Actual quantized storage is preserved; native
projections already dequantize matrices to F32 before ggml multiplication.
Independent `gguf` dequantization feeds original Python and the separate native
F32 arm. Native-quant and native-F32 metrics agree at every observed stage;
output byte equality was not independently reviewed from the small report.

| Same-input operator comparison | Hidden maximum absolute error | Largest stage error | Minimum stage row cosine |
| --- | ---: | ---: | ---: |
| Native official weights vs original F32 |0.00000858307 |0.0001449585 input norm |0.9999993977 |
| Native Q8 vs original independently dequantized F32 |0.0000114441 |0.0001449585 input norm |0.9999998231 |
| Native Q4 vs original independently dequantized F32 |0.00000762939 |0.0001449585 input norm |0.9999999886 |

The largest absolute error already occurs at input norm against much larger
activations (BF16 norm magnitude63.75). No localized layer0 decoder defect is
identified at these unchanged budgets. This does not prove remaining31 layers,
full quantization, or decoded quality.

Original Python with candidate weights itself reproduces larger hidden
changes against frozen BF16 on the identical original input:

| Original execution | Hidden maximum error vs frozen BF16 | RMS | Minimum row cosine |
| --- | ---: | ---: | ---: |
| Official same-weight F32 diagnostic |0.0358066559 |0.0003975073 |0.9999829255 |
| Independently dequantized Q8 F32 |0.0397329330 |0.0005752510 |0.9999495341 |
| Independently dequantized Q4 F32 |0.185863494 |0.0056318391 |0.9932813875 |

These rows include the F32-versus-BF16 precision difference; they are not pure
quantization errors against a common F32 baseline. The existing source stages
show original BF16 casts after RMSNorm/projections, convolution/SiLU, gated norm,
activation products and residual sums, while native retains F32 activations.
GDN Q/K normalization also occurs before conversion to F32 recurrence math.
Any BF16-boundary candidate must preserve quantized storage, stay default-off,
and compare the earliest changed operators and actual decoded output separately.

Large arrays/models remain remote. The v2 raw artifact13413366893 is306,071,097
bytes; only the86KB report was range-fetched for this review. No numerical gate,
default, alias, release or application pin changes.

## First default-off activation-boundary candidate: rejected

[42026091939](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42026091939)
atf201bc8c9 implements default-off BF16 activation boundaries for the linear block
only, preserving quantized storage/dequantized weight values. Its original Python
oracle retains original block/GDN/cache code while F32-weight linear/conv operators
return BF16 activations. GDN Q/K products/sum/rsqrt/multiply precede F32 recurrence;
state stays F32. The oracle is a distinct diagnostic, never replacement gold.
Full attention and final norm remain outside this bounded candidate.

The [complete report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-bf16-boundary-v1.json)
is **numerically negative**. All three boundary arms first fail input norm:
max0.015625, source BF16 norm magnitude63.75. QKV then reaches0.0625; final hidden
is official/Q80.03125 and Q40.015625 against the corresponding activation oracle.
The existing small F32 reduction differences can cross BF16 rounding boundaries;
first follow the identical-input RMS operator before another arithmetic change.
All legacy same-weight stages still pass and gate-on→legacy recovery output
hashes are exact. No decoded quality or default promotion follows.

The producer's then-existing wrapper exits0 even when native stage flags fail.
Thus its green workflow conclusion is a reporting defect; the actual comparison
is rejected. The successor aggregates local native failures into a nonzero exit
while preserving every arm. Original BF16 cross-host failures also remain.
Large raw artifact13414661349 is486,492,268 bytes; only its203KB report is locally
range-fetched. Source heads and initial reports remain immutable.

This same run independently quantifies **pure quantization** on identical original
input in the original F32 layer, separating native logic and BF16 precision:

| Candidate vs official F32 | First strict stage | QKV maximum / RMS | QKV original magnitude | Hidden maximum / RMS |
| --- | --- | ---: | ---: | ---: |
| Q8 |QKV |0.0834102631 /0.0056331582 |24.3049431 |0.0174388885 /0.0004315853 |
| Q4 |QKV |0.906708241 /0.0727442667 |24.3049431 |0.159342766 /0.0056135881 |

Input-normalization outputs are bit-exact across these original F32 arms. Both
candidate changes are reproduced by original code without any native operator;
this supports quantization loss at this boundary, not a local native decoder bug.
The GDN normalized minimum row cosines include nearly zero/sign-changing rows;
retain their absolute errors and tensor magnitudes rather than treating those
cosines alone as evidence. Full decoded Q8/Q4 rejections stay unchanged.
