# Exact retained dense-projection arithmetic

Draft PR152 stacks PR151 `69e131159`. This trace executes only the retained
layer1 FFN gate projection: original post-norm input `[1,2,2560]`, verified
synthetic F32 weight `[9216,2560]`, complete18432-element output. No decoder or
block forward is repeated; no production source, budget, precision or default
changes. All26 regenerated synthetic parameter hashes must match the frozen
panel. The original and native complete output anchors must each be byte-exact.

Initial [41983685028](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41983685028)
at `4139df53f` reproduces the anchors and eight-lane arithmetic, but static link
wrappers observe zero calls because ggml is shared. Preserve that observation
gap; the successful job alone did not establish its actual route.

Corrected [41984176263](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41984176263)
at `10489019e84bd1d6919c412daeb3d56d42d052b2` uses validation-only shared-library
interposition and explicitly fails unless actual native routing is observed.
It records two llamafile calls, two accepted calls, two exact full-matrix/input
operand matches and zero fallback vec-dot calls. All operands/types remain F32.
Both output anchors stay byte-exact, proving observation did not change results.
Native arithmetic is from the actual ggml shared object, SHA-256
`e615a1116f83a70b323c3ede52cd0b254b3f071792b27d3fe8dfab6647bead7b`;
285 relevant arithmetic instructions are located and a bounded excerpt retained.
The ggml revision is `3a27b6c665b191cb230be367db5c52d81dd3e2ac`.

## Observed arithmetic and hypotheses

All18432 native outputs match the fixed eight-lane FMA recurrence and its
llamafile horizontal-reduction tree exactly. None of the tested alternative
models reproduces all original Torch GEMM outputs:

| Arithmetic model | Exact native outputs | Exact original outputs |
|---|---:|---:|
| Eight-lane FMA | 18432 | 1209 |
| Eight-lane separate multiply/add | 3457 | 1206 |
|32-lane FMA | 1542 | 1227 |
| Actual fallback vec-dot | 1542 | 1227 |
| Sequential FMA | 563 | 522 |
| Sequential separate multiply/add | 570 | 578 |
| F64 products/sum, rounded to F32 | 1563 | 1116 |

The first row-major original/native output difference is token0,row0:
`-2.4665756225585938` versus `-2.466576099395752`, error4.7683715820e-7.
The worst is token0,row234: `1.5401862859725952` versus `1.5401884317398071`,
error2.1457672119e-6. Original full-output magnitude is3.798306942; unchanged
operator budgets pass. F64 is a diagnostic reference, never replacement gold.

For both selected rows, all2560 standalone rounded products agree bit-exactly
between Torch and native controls. In the validated native eight-lane recurrence,
the first fused versus separate multiply/add difference from the same accumulator
occurs at zero-based product10,lane2. It is one F32 ULP at that intermediate
magnitude:5.8207660913e-11 for row0 and1.1641532183e-10 for row234. This is an
actual arithmetic distinction in the reconstructed native recurrence against its
unfused control, not an observed original Torch GEMM internal partial sum.

Independent small-artifact review verifies six source digests/four ggml source
hashes, both retained input/output hashes, all18432 hypothesis match counts and
maximum errors. It checks5120 partial updates using exact rational products,
nearest-even F32 rounding and exact accumulator-chain/reduction reconstruction.
The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-dense-trace.json)
retains both initial/corrected reports, exact source pins, route observations,
selected values and verified first differing updates. Large synthetic weights
and decoder arrays remain remote; only the bounded operator artifact is reviewed.

## Limits and preserved guards

The actual original Torch GEMM reduction order/kernel partials remain unobserved.
Different Torch dot/sum and native reduction controls produce different answers
on the same values. The evidence therefore does not establish that disabling FMA
or adopting any tested alternate tree would reproduce original GEMM, improve
real OCR, or fix the complete decoder rejection. Checkpoint weights, logits and
actual decoded text/EOS remain absent; the earlier290/291 strict decoder panel
stays negative.

The runtime loader's inherited GDN function-hash field is source verification,
not executed-function coverage for this projection-only case; the review states
this and a reporting-only successor labels it explicitly. Historical stdlib
extractor assertions retain their complete old job span. A separately closed
read-only/no-secret selector control now covers the additive native job; no old
security assertion is removed. No release, alias, source closure or threshold
promotion follows from this operator trace.
