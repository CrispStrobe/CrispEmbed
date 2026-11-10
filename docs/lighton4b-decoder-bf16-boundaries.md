# Default-off LightOn4B decoder linear-block BF16 boundaries

`CRISPEMBED_LIGHTON4B_BF16_LINEAR=1` preserves immutable quantized weight storage
and F32 dequantized matrix values while rounding linear-block activations at the
original BF16 boundaries. Unset/empty/0 retain the existing path. This is a
bounded CPU linear-block experiment; full-attention blocks/final norm are unchanged.
No decoded-quality, default, model-alias or release promotion follows.

The path covers input/post RMSNorm, QKV/z/a/b projections, convolution then SiLU,
beta, BF16 Q/K L2 normalization before F32 delta recurrence, gated norm cast order,
attention/FFN projections, activation products and residual sums. State stays F32.
Linear caches carry a precision-mode tag and reject mismatched continuation;
previous cache/weight data remain immutable. The explicit pre-normalized delta
entry requires the new gate; existing `delta_cpu` retains old F32 normalization.

RMS reduction follows the exact pinned Torch AVX2 grouping verified in
[the retained attribution](https://github.com/CrispStrobe/CrispEmbed/blob/5b47ed44e/docs/lighton4b-layer0-source-fed.md):
actual-helper42031297677 matches154 scales,394,240 input-norm values and630,784
real GDN pre-weight normalization values bit-exactly. Composed42032626361 reaches
exact input norm but remains numerically rejected at QKV (official0.015625,
Q8/Q40.03125); hidden errors improve but still exceed0.002. All112 legacy stages
and same-process mode/cache controls pass. The dedicated QKV attribution remains
separate from runtime integration. All strict flags/old alternatives remain.

Model-free `test-lighton4b-bf16-linear-controls` covers gate values, RNE ties,
normalizer shape/zero/one controls, explicit delta-entry gating and legacy delta
isolation. Actual source-fed checkpoint probes remain test-only on their frozen
source branches and large arrays remotely; no reference vectors enter production.
The assembled source still requires hosted build/control checks and later actual
source/decoded/platform qualification. VERSION, ggml and the app pin are unchanged.
