# Retained-input LightOn4B layer0 attribution

The retained stage comparison first differs in shared vision arithmetic. Both
native recipes use identical decoder input but different quantized layer0
matrices. Their first recipe-specific layer0 difference alone cannot distinguish
quantization from a decoder implementation error.

The test-only `tools/ci-heavy/lighton4b_layer0_source_fed.py` uses the exact
original154-token decoder input. It requires original BF16 layer0 hidden output
to reproduce its retained anchor exactly, then executes diagnostic arms:

- Official BF16-derived weights in original/native F32 operators, with identical
  input. This is an operator diagnostic, not replacement reference gold.
- Each frozen Q8/Q4 candidate's independently dequantized layer0 weights in the
  original F32 layer, compared against actual native quantized execution.
- The same dequantized weights in native F32 execution, separating quantized
  matmul activation/kernel effects from the remaining decoder operators.

The original executes one full154-token layer. Native uses the existing
sixteen-token chunk/cache path, matching production's prefill schedule. Every
arm retains input norm, attention output/residual, post norm, gate/up/activation,
down/hidden and final convolution/recurrent cache arrays. Reports include
original/native absolute magnitudes, maximum absolute error, row/global cosine,
unchanged strict thresholds and earliest failing stage. Ordinary original output
must match hook-observed output. Protected original BF16 backend settings are
separate from the audited F32 component setup.

The official checkpoint is fetched only as its contiguous226MB layer0 interval
from an immutable public revision. The frozen header, interval size and HTTP
Content-Range are verified; interval and per-tensor hashes are retained. This
does not rehash the entire9GB checkpoint. Exact frozen BF16 hidden reproduction
is a required anchor, and failures stop all later attribution. Each quantized
candidate is fetched and fully SHA-256 authenticated in turn; large models and
raw arrays remain hosted artifacts. No generation/image forward/full decoder or
new gold is produced; no decoded-quality, publication or release claim follows.

Hosted execution is pending. Production `src/`, ggml and VERSION are unchanged.

## Initial exact-anchor replay and separate diagnostic continuation

[Initial42022722133](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42022722133)
at7b7cb1a84 compiles the actual native runner and authenticates input, original
checkpoint interval and all source modules/packages, then stops as required at
original BF16 layer0 hidden replay. Hooks preserve ordinary hidden output exactly,
but the retained cross-host anchor differs: max0.0078125, RMS0.00009840313,
minimum row cosine0.9999984685, original/replayed absolute magnitude8.5625. No
Q8/Q4 arm executed. [Initial producer report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-source-fed-initial.json)
preserves the failure; it is not candidate rejection or native implementation proof.
The source environment also used default deterministic-algorithms OFF, while the
component loader enables it. The successor explicitly matches original BF16 OFF
and retains audited F32 diagnostic ON; exact effects require actual measurement.

Protocol `v1` still stops on any original BF16 anchor difference. The separately
selected `--protocol v2-unanchored-diagnostic` may execute local operator arms
while retaining the frozen BF16 failure prominently and leaving parity false.
Its status is `diagnostic-compared-unanchored` when BF16 differs. This does not
waive the original numerical gate, create replacement gold or qualify a model.

`--bf16-only --retained-root <mounted-stage-directory>` supports a CPU-only
source-environment replay using the original retained kernel output mount. It
executes no native build, F32 operator or quantized candidate; exact source/input
hashes and the same v1 BF16 anchor remain mandatory. The host provider is recorded
explicitly. This bounded replay does not repeat original full-model inference.

## Completed original-environment anchor

The bounded BF16-only source-environment replay at4498559a22 completes without
full-model or native execution. All394,240 original layer0 hidden values are
bit-exact against the retained original: max/RMS0 and row/global cosine1.
Original hooks remain exactly neutral. [Producer report](../tests/results/qwen35-ocr/lightonocr3-4b-layer0-bf16-source-environment.json)
SHA6519caa2e53e6fd33eeb9dc540a26cab2ef4cec3fcbfb97a4f75835d73bd7b4f
retains exact official range/input/module/package hashes and all intermediate
stage declarations; large arrays remain remote. This validates the recovered
checkpoint layer/input/blueprint, not complete decoder or decoded parity.

Compared with initial hosted CPU replay, input-layernorm output hashes are exact
but the first QKV/z/a/b projection hashes already differ. Later GDN/FFN output
hashes also differ. The initial replay additionally enabled deterministic
algorithms; the v2 hosted replay matches original OFF to separate that setup
factor from cross-host BF16 GEMM arithmetic. No rounding/default/budget change
is justified by the anchor alone.

## Default-off linear-block BF16 activation candidate

`CRISPEMBED_LIGHTON4B_BF16_LINEAR=1` separately rounds LightOn4B linear-block
activations at the source's BF16 boundaries while retaining all quantized matrix
bytes and F32 dequantization. It covers input/post RMSNorm, QKV/z/a/b projections,
causal convolution then SiLU, beta, Q/K BF16 L2 normalization before F32 delta
recurrence, gated normalization, output/FFN projections, activation products and
residual sums. Recurrent state stays F32. The explicit pre-normalized delta entry
preserves the old delta entry's F32 normalization unchanged. Unset/empty/0 keep
the existing path; full-attention blocks and final decoder norm remain separate.
This is a bounded linear-block candidate, not full BF16 decoder admission.

`--bf16-boundaries` adds a same-input original Python activation oracle: original
block/GDN/cache code and unmodified quant-dequant weight values are retained;
linear/conv operators compute with F32 weights and return BF16 activations,
and gated norm follows the source's two BF16 rounding boundaries. The oracle is
explicitly a new diagnostic arm, never replacement BF16 gold. Native comparison
retains all16 stages and gate-on→legacy recovery checks, plus rounding ties.
Same-precision candidate-versus-official F32 stage metrics separately quantify
quantization. Hosted compilation/operator checks are pending. Real decoded A/B
and remaining attention/final-norm semantics remain required before promotion.

Actual first boundary candidate42026091939 atf201bc8c9 is numerically negative:
first input norm max0.015625, then QKV max0.0625 and final hidden official/Q8
max0.03125, Q4 max0.015625 against the activation oracle. Legacy recovery hashes
remain exact. The producer records all strict failures, but its then-existing
wrapper incorrectly exits0 despite local numerical failures; successor aggregation
returns nonzero on those failures. The result is rejected regardless of green
workflow conclusion. No full decoded run is justified by this component result.

The next `--norm-only` probe authenticates the same official layer/input and
compares existing sequential/precise RMS reduction alternatives, F32 scales and
BF16 outputs against original Torch on the identical input. It also runs the
legacy same-weight layer0 control with newly tagged cache precision modes;
incompatible legacy/BF16 caches must reject before execution. No quantized model
or full-model reference is downloaded by this norm-only scope. Production default
and all strict budgets stay unchanged.

Norm-only42027758298 at28e4180b8 completes actual legacy/cache controls and
unchanged sequential/precise RMS comparisons. Precise F32 maximum improves
0.0001449585→0.00000762939; BF16 maximum improves0.015625→0.00390625 but still
fails0.002. At flat330868, native pre-round0.587890625 is exactly a BF16 midpoint
and rounds to0.5859375; original0.5878906846046448 lies one F32 ULP above and
rounds to0.58984375. No precise-RMS default or corrected BF16 claim follows.
The diagnostic completion wrapper also omitted norm-row failure aggregation;
its small report remains negative and the successor returns nonzero while
retaining every comparison.

`--norm-source-isolation` reconstructs original input norm from original means,
scales and weights and requires exact output identity. It then feeds original
means to native F32-sqrt/reciprocal and F64-rsqrt diagnostic alternatives, and
feeds exact original scales to the ordinary multiplication/affine path. This
separates reduction, reciprocal-square-root and multiply/rounding effects without
another quantized candidate transfer or any expanded decoder execution. Original
mean/scale inputs are test-only; no production source consumes reference vectors.

Source-fed42028839817 atbdefd15c4 completes exact original norm reconstruction.
Feeding original means to native F32 sqrt/reciprocal reproduces all154 scales and
394,240 F32/BF16 outputs bit-exactly; exact source scales also reproduce all
outputs. F64-rsqrt instead leaves three BF16 differences/max0.00390625. This
isolates the remaining local discrepancy to mean reduction before rsqrt. Actual
Torch source commit iscf30153c4c131c8164ee7798e5022d810682e2cb, CPU capabilityAVX2.
The job exits red for preserved rejected sequential/precise alternatives while
retaining completed source-fed controls.

A test-only successor reconstructs the pinned CPU `SumKernel.cpp` inner reduction
with eight- and sixteen-lane/four-stream/sixteen-row grouping, separately from
F64-sum→F32-sum→F32-mean rounding. It authenticates the exact Torch sum/mean/vec
source hashes and compares actual mean vectors before full norm outputs. No
original values are embedded in the reconstruction and no production RMS path
changes until this bounded source-derived alternative is measured.
