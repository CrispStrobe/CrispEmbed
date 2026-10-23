# PosFormer trained forward and released default

PosFormer applies the trained embedding LayerNorm, positional addition, then
input LayerNorm. Fixed2×2/stride2 encoder pools duplicate missing edge samples
for PyTorch ceil-mode valid-sample max/mean behavior. The source backport is
selective; the historical release branch was not merged wholesale.

The trained forward is the default, matching published0.17.12.
`CRISPEMBED_POSFORMER_TRAINED_FORWARD=0` restores the legacy vendor-main path;
explicit1 and unset select the trained forward. Mode is captured at model init.
Required norm/position tensors are validated only for the trained path.

Actual published Linux0.17.12 and the prior opt-in candidate match all50 frozen
case token IDs/text/EOS stops. Confidence differences reach2.0265579223632812e-6;
these are retained without claiming bitwise parity or fitting a new tolerance.
[Released-binary evidence](../tests/results/posformer/released-compatibility.json)
records actual libraries, asset/model/input hashes and raw measurements. This
is the ground truth for restoring the default, not a new recognition recipe.

Hosted Linux validation checks actual unset output against the frozen released
Linux binary records. Both platforms check explicit1/unset agreement, zero/legacy
agreement, full report provenance,
50-case normal API, meaningful primitive negative controls and both bounded
original-code/exported-F32 reference paths. Pure controls reject changed tokens,
text, stops, images/models, case coverage, confidence shape and nonfinite or
impossible zero top-1 probabilities. Confidence drift remains measured separately.
Test probes belong only to the separate hosted test library.

Recognition remains7/50 on this panel. Original training-checkpoint/BN statistics
and full-sequence parity are unproved; the exported-F32 reference covers five
cases/five decoder steps. The initial trace-anchor failure remains archived.
Development manifests0.17.13 create no tag, publication or app upgrade. Keep
immutable app0.17.12 until actual successor artifact/hash/consumer qualification.
