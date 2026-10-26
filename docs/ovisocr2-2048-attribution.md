# Retained-vector 2048 attribution

This analysis consumes the exact rejected boundary case from
[40177774574](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40177774574)
at`ac1bb9cb33f58fb0e7082cc4ba2ddb38793c48cb`. The public producer is
`tests/results/qwen35-ocr/ovisocr2-larger-input-producers/large-boundary.json`;
its ZIP/reference/native member hashes remain immutable. The old result is8/14,
with six composed norm failures and unchanged IDs/text/EOS. No gate or input is
changed to turn that result positive.

The first bounded stage uses the existing943-stage dump and all37 retained
vision comparisons, plus prefill embeddings/all24 hidden layers/final norm for
prepared and composed vision in both precisions. Native-positions/frontend
manifest hashes are checked separately against native-vision; differences are
reported rather than assumed absent. Earliest non-bit-exact stages, strict
failures, absolute/RMS errors, magnitudes, cosine and original backend-calibrated
ratios remain distinct. Check patch and interpolated-position branches before
their embedding sum; insertion order of the old capture dictionary is not the
computation order. Large ZIP/reference/native arrays stay on hosted CI.

For prefill, cached and full-prefix, supply each retained layer23 output to the
actual pinned Transformers5.16.1 `Qwen3_5RMSNorm` with the original checkpoint
weight. The full checkpoint hash and norm-vector hash are verified. Before
attribution, that same module must reproduce the original frozen layer23→norm
anchor bit-for-bit; any drift is retained as failure, without replacing gold.
Then compare native norm with original-module norm on those identical native
input bits using unchanged0.002 maximum absolute/0.99999 minimum row cosine.
This separates local operator error from propagated input error. Report both
against immutable original full-model norm, including its unchanged per-element
backend calibration. Preserve the error decomposition and small local output
arrays separately from old references. No full model forward, reconversion,
requantization or native inference is repeated in this first stage.

A successful analysis workflow only establishes the declared local isolation.
It does not qualify the2048 bound, prove every vision operator or waive the
six composed failures. If RMSNorm matches but input error propagates, continue
at the earliest recorded vision boundary with original-input operator probes
before proposing a gated correction. Any correction must pass the old frozen14,
all owning decoded checks at every declared grid and eleven-case first-F16
controls. Keep default1024 and the existing experimental alternative.

Dispatch metadata workflow `phase=attribution-2048`. Review only the small
`ovisocr2-2048-attribution-reports-RUN`; preserve the local operator output
artifact and old source lineage. The driver rejects insufficient scratch and
checks ZIP/member bytes before computation. Initial failures remain separate;
never widen historical source allowlists to run a different native version.

## Original-input position contract

The completed norm isolation is recorded separately in
`tests/results/qwen35-ocr/ovisocr2-2048-norm-attribution.json`. The next operator
probe pins `transformers.vision_utils` to SHA
`bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356`.
Its align-corners path executes F32 `index * (side - 1) / max(size - 1, 1)`
and linear hat weights. Native's preserved path instead uses linspace step and
opposite-end subtraction. Mathematical equivalence does not imply identical
coordinates/weights under these different operation orders.

`CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION=1` is a default-off alternative
matching the inspected source expression. The legacy path remains available;
unset and literal0 must be bit-identical, and its position/patch output must
reproduce the retained ordinary native boundary arrays exactly. Private position
helpers are used by the production encoder and included in a test-only translation
unit; no reference observer or gold data enters a production object.

Before candidate comparisons, observe the actual original
`Qwen3_5VisionModel.fast_pos_embed_interpolate` method and interpolation utility
on the saved grid/table. Hooks/profile observation must leave output bits
unchanged, and boundary position must reproduce the immutable frozen anchor
bit-for-bit. Capture its actual indices, weights and per-axis `src`, without
reimplementing expected math. Five declared grids cover the boundary, transposed
boundary, square, small and repeated-frame geometry. These are operator cases,
not a public multi-image/video support claim. Every source-gated native index,
weight and coordinate must match exactly. Position output retains0.002/0.99999;
remaining reduction error is reported independently. The patch control uses the
actual original patch module and original input, retaining frozen and any distinct
local target separately under unchanged gates. All initial failures are saved.

Dispatch metadata `phase=position-probe`. A positive local result authorizes
only the next gated composed A/B, not a default flip or2048 qualification. The
existing first-FFN manual audit keeps its historical source-tree pin; generic
source-contract unit tests use an explicit frozen fixture plus drift rejection,
so unrelated future source additions cannot silently rewrite that old contract.
