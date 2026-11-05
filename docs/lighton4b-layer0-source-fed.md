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
