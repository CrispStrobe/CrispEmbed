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
## Earliest input-normalization reduction: precise alone remains negative

[42027758298](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42027758298)
at28e4180b8 isolates original154-token input norm, using only the226MB official
layer interval and retained original input. No quantized candidate is transferred.
The [complete norm report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-norm-attribution-v1.json)
retains scales, magnitudes, maximum-error locations and all original/legacy
controls. Existing sequential and precise RMS are compared before BF16 rounding:

| Existing RMS reduction | F32 maximum / RMS | BF16 maximum / RMS | BF16 strict result |
| --- | ---: | ---: | --- |
| Sequential/default |0.0001449585 /0.00000202514 |0.015625 /0.0000894558 |Fail |
| Precise opt-in |0.00000762939 /0.0000000834215 |0.00390625 /0.00000695560 |Fail |

At flat330868, native precise pre-round value0.587890625 is exactly a BF16
midpoint and rounds to0.5859375; original0.5878906846046448 is one F32 ULP above
and rounds to0.58984375. The resulting0.00390625 exceeds unchanged0.002. Precise
RMS improves the local error but is not a valid exact correction or reason to
promote a BF16 decoder. Native same-weight F32 layer and incompatible-cache-mode
rejection controls pass.

The producer's completion wrapper omits norm-row failure aggregation; its green
workflow does not turn either failed BF16 comparison into parity. The successor
returns nonzero on failed norm comparisons while retaining all arms. Raw
artifact13415470128 is93,989,965 bytes; only its31KB report was range-read locally.

Bounded source mean/scale isolation42028839817 atbdefd15c4 now requires exact
original norm reconstruction, then compares source-mean native reciprocal-square-
root alternatives and source-scale native multiply/affine output. Reference
inputs remain test-only. No broader decoder or decoded claim is made while that
first-operator evidence is pending.

## Source mean/scale isolation closes rsqrt and affine

[42028839817](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42028839817)
atbdefd15c4 reconstructs the original F32 input-normalization output bit-exactly
from original means/scales/weights before any native comparison. The
[complete producer](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-norm-source-isolation-v1.json)
records actual Torch commitcf30153c4c131c8164ee7798e5022d810682e2cb and CPU capability
AVX2. The job ends red for unchanged rejected sequential/precise BF16 alternatives;
all source-fed controls complete before that expected numerical rejection.

| Native operator supplied original inputs | Scale exactness | F32 norm exactness | BF16 norm exactness |
| --- | --- | --- | --- |
| Original mean → F32 sqrt/reciprocal |154/154 bit-exact |394,240/394,240 |394,240/394,240 |
| Original mean → F64-rsqrt alternative |Not exact; max7.62939e-6 |110,275 differing values |3 differences; max0.00390625 |
| Original scale → multiply/affine |154/154 bit-exact |394,240/394,240 |394,240/394,240 |

Thus source-fed reciprocal square root, affine multiplications and RNE conversion
are exact at this boundary. The earliest remaining local discrepancy lies in
mean reduction before rsqrt, not those downstream operators. This is bounded to
the exact original154×2560 input and does not establish full decoder/quality parity.
Original vector inputs stay in the test-only probe.

The [pinned CPU mean source](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/ReduceOps.cpp)
uses sum followed by division; the
[inner-sum source](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/SumKernel.cpp)
groups stream/lane partial sums, while
[AVX2 rsqrt](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cpu/vec/vec256/vec256_float.h)
uses sqrt then reciprocal division. Source-derived test-only order alternatives
are executing in42029945232 at215f463bc, with exact source hashes and mean-vector
comparison. This is a measured operator continuation, not an observed assembly
route or a new production reduction/default. Raw100,408,910-byte artifact
13414434567 remains remote; only the37KB producer report was range-read.

## Exact source-derived mean order at the retained boundary

[42029945232](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42029945232)
at215f463bc authenticates the exact Torch sum/mean/AVX2 source hashes and compares
source-derived reduction orders before any production change. The
[full grouped-order report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-norm-source-tree-v1.json)
retains exact original reconstruction, identical checkpoint interval/input and
passing legacy layer/cache controls. Small-report SHA is
f2da425824223efff8c5aa0175ac4ea145075fbdb6d93c77f5da089235933d71.

| Native test-only mean order | Means/scales | F32 norm | BF16 norm |
| --- | --- | --- | --- |
| Eight-lane, four-stream, sixteen-row grouping |All154 bit-exact |All394,240 bit-exact |All394,240 bit-exact |
| Sixteen-lane grouping |Not exact |131,753 differences |3 differences/max0.00390625 |
| F64 sum rounded to F32 before F32 mean |Not exact |90,777 differences |2 differences/max0.00390625 |

The eight-lane reconstruction embeds no original tensor values; its grouping
comes from the pinned real CPU reduction blueprint. This is actual identical-input
output evidence, not proof of a recorded assembly dispatch. Both wider-vector
and precision-only alternatives remain negative under unchanged budgets. The
job is red for preserved old rejected alternatives, not a missing/failed capture.
Raw106,800,638-byte artifact13415577818 remains remote; only its46KB report was
range-read locally. No model/decoded qualification follows.

Source5ba80426d applies the measured order only inside the default-off linear
BF16 candidate, retaining shared legacy RMS. Actual-helper42031297677 separately
checks2560-wide input norm and128-wide original GDN pre-norm data before another
composed layer0 replay. No expanded decoder/default promotion follows while that
operator and the remaining projection/decoded gates are unresolved.

## Actual gated RMS helper: both widths exact

[42031297677](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42031297677)
at5ba80426d executes the actual default-off candidate RMS helper, separately from
the earlier test-only reconstruction. The
[complete actual-helper report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-gated-rms-helper-v1.json)
retains exact input norm reconstruction and original model pre-hook input for the
128-wide GDN normalization. No original values enter the production helper.

| Actual helper boundary | Values checked | Maximum/RMS error | Exactness |
| --- | ---: | ---: | --- |
|2560-wide input norm scales |154 |0/0 |Bit-exact |
|2560-wide input norm F32 output |394,240 |0/0 |Bit-exact |
|2560-wide input norm BF16 output |394,240 |0/0 |Bit-exact |
|128-wide original GDN pre-weight normalization, BF16 |630,784 |0/0 |Bit-exact |

The GDN result covers normalization before norm-weight/gate multiplications,
not the complete GDN operator. Original input norm magnitude is63.75; GDN
normalized magnitude11.3125. All row/global cosines are1. Legacy full-layer and
incompatible-cache/direct-entry gate controls pass. The overall job correctly
remains red for old rejected RMS alternatives. This validates the bounded actual
helper; full decoder, quantized output and decoded qualification remain open.
Raw113,846,599-byte artifact13416283193 remains remote; only49KB was range-read.

Composed layer0 replay42032626361 at5b47ed44e now uses the exact helper with the
original quant-preserving activation oracle. Earlier recovery arms used fresh
processes. This successor additionally interleaves both modes in one native
process/weight owner, rejects mixed-mode cache continuation, and requires every
legacy intermediate/hidden/cache to reproduce exactly after the alternate call.
No expanded decoder or generation is run, and no corrected decoded result is
claimed while the next differing projection is under measurement.

## Composed layer0 after exact RMS: next QKV failure retained

[42032626361](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42032626361)
at5b47ed44e completes all original/native arms and correctly exits red with
`comparison_completed=true`, `numerical_rejection=true`. The
[complete composed report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-bf16-boundary-exact-rms-v2.json)
retains the exact-RMS candidate and all previous alternatives. Input norm is now
exact in every BF16 activation arm; the next first failure is QKV:

| Candidate vs corresponding quant-preserving BF16 activation oracle | Input norm max | QKV max | Hidden max before→after exact RMS |
| --- | ---: | ---: | ---: |
| Official weights |0 |0.015625 |0.03125→0.00390625 |
| Q8 weights |0 |0.03125 |0.03125→0.0078125 |
| Q4 weights |0 |0.03125 |0.015625→0.00390625 |

All hidden errors still exceed unchanged0.002. Source QKV magnitudes are about
24.25; numerical improvement does not establish full-block or decoded parity.
Recurrent-state and later FFN flags remain recorded. Existing source quantization
loss and complete legacy Q8/Q4 decoded rejections remain unchanged.

All112 stage checks across seven legacy same-weight/recovery arms pass. Every
native arm also passes actual **same-process** interleaving: the alternate mode
uses the same native weight owner, incompatible cache continuation rejects, and
all original intermediates/hidden/cache values reproduce exactly after restoring
the mode. Earlier fresh-process recovery remains separately limited. No reference
arrays are consumed by production code and no full decoder or generation runs.
Raw498,725,772-byte artifact13417242200 remains remote; only its209KB report was
range-read locally.

Focused Q8 QKV dot42036721968 atf62b28879 now preserves the original normalized
input and rounded-projection hashes, observes actual native F32 route/complete
operands and compares pre-round F32 with BF16-rounded output. Selected traces and
complete accumulation hypotheses precede any dot fix. The earlier native producer
omitted native payload SHA fields, so reproduction of its old native QKV payload
bytes is not claimed. No expanded decoder or default promotion precedes actual
operator evidence and decoded acceptance.

## Real Q8 QKV dot: tiny pre-round error crosses BF16 boundaries

[42036721968](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42036721968)
atf62b28879 executes one source-fed Q8 projection, with no decoder/GDN/full-model
forward. Frozen source BF16-normalized inputs and the original rounded Q8
projection hashes reproduce exactly. The
[complete dot report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-qkv-dot-v1.json)
records all1,261,568 outputs, actual native source/library hashes and selected
operand/product traces. Its red exit is a completed numerical rejection.

Validation-only shared-library interposition observes20 actual native llamafile
calls, all accepted. Every complete F32 matrix/input byte sequence matches
independently dequantized Q8 weights and source-normalized inputs; no fallback
vec-dot call occurs. Plain versus observed outputs and the selected-trace repeat
are bit-exact, so observation changes no output. Native composed payload SHA
was absent from the older producer; reproduction of that old native payload is
not claimed by this fresh controlled projection.

| Actual native vs original F32 linear operator | Maximum absolute error | RMS | Original magnitude |
| --- | ---: | ---: | ---: |
| Pre-round F32 QKV |0.0000267029 |0.000000738513 |24.2978497 |
| BF16-rounded QKV |0.03125 |0.0000753585 |24.25 |

The pre-round comparison passes unchanged0.002/0.99999; the rounded output has
841 differences and fails0.002 despite minimum row cosine0.999999975. The first
rounded difference is token0/row437. At the maximum rounded difference,
token2/row2129 (flat18513), original7.890624523162842 and native7.890625476837158
straddle the BF16 midpoint7.890625. They round to7.875 and7.90625, a0.03125
change from9.53674e-7 pre-round error. F64 products/fsum is7.8906255900801625,
on the native side of that midpoint; greater precision is not replacement gold.

| Complete projection arithmetic hypothesis | Exact native F32 outputs | Exact original F32 outputs | Original BF16 output differences |
| --- | ---: | ---: | ---: |
| Eight-lane FMA |1,261,568 |73,425 |841 |
| Sixty-four-lane FMA |100,621 |111,643 |564 |
| Sixty-four-lane separate multiply/add |100,253 |111,530 |568 |
| F64 products/sum rounded to F32 |87,460 |123,511 |502 |

Every alternative still reaches BF16 maximum0.03125. Standalone native/original
rounded products agree for all2560 entries of both selected rows. The eight-lane
FMA model reproduces the complete actual native output. Its first fused versus
unfused update traces are diagnostic native recurrences, not observed Torch
GEMM partial sums; the original GEMM internal reduction remains unobserved.
No tested alternate dot is a proven correction, and none is integrated.

This separates identical-input/dequantization correctness from accumulation and
BF16 rounding amplification. It does not remove intrinsic quantization loss or
qualify recognition. Legacy Q8's160-ID EOS output still differs from the177-ID
original, Q4 still caps256/noEOS with incomplete HTML, and the separate mixed
calibration remains unqualified. No default/alias/release/app-pin changes follow.
Raw46,684,813-byte artifact13418767187 stays remote; only its16KB report was
range-read locally.

Verified default-off boundary/RMS code alone is integrated into existing PR155
at7d1e73464, preserving the old path and this known next-QKV failure. Reference
probes and all dot hypotheses stay outside production. Model-free assembled
controls and vision integration remain separate; no full decoder/decoded or
platform qualification is inferred from this scoped code integration.
