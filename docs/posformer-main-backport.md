# PosFormer selective main backport

The source is selectively ported from PR60 at78493a31 onto current main; the old
release branch, release workflow changes and tag are not merged. The trained
decoder applies embedding LayerNorm, positional encoding, then input LayerNorm.
The fixed2×2/stride2 encoder pools duplicate missing edge samples for PyTorch
ceil-mode valid-sample max/mean behavior.

The trained forward is the default, matching published0.17.12.
`CRISPEMBED_POSFORMER_TRAINED_FORWARD=0` restores the legacy vendor-main path;
explicit1 and unset use the trained forward. The value is captured at model initialization.
The prior path remains available. Required trained norm/position tensors are
validated only for the trained path. Test probes are added only to a separate
hosted test library and must be absent from the production library.

Hosted `posformer-runtime-validation.yml` checks original exported weights,
normal public API transcription on all50 frozen cases (prior result7/50),
Linux/macOS, scalar/default reference encoders, meaningful wrong normalization/
pooling controls, unset/explicit1 equality and zero/legacy output equality and printed OCR.
A passing component test is not recognition improvement. All source/model/corpus
identities are pinned; any missing artifact or changed anchor is a failure.

The development manifests are synchronized to0.17.13 for this actual runtime
addition. No release tag, package publication, default promotion or app pin change
is implied. PR72 merged at7f1ecd33 after all sourcef7c53803 checks and report review;
historical PR60 is closed. Release compatibility with actual published0.17.12
remains separate: that release already uses trained-forward math by default,
while development main retains the legacy default. Validate this before publishing
or changing defaults; the opt-in source merge is not a successor release.

Actual released Linux0.17.12 and the prior opt-in candidate matched all50 frozen
case token IDs/text/EOS stops before this default restoration was proposed.
Confidence drift is retained in the released-compatibility report, without a
new fitted tolerance. The new-default workflow must reproduce those actual
released outputs; pure controls reject changed IDs, text, stop, image, model,
case coverage and nonfinite confidence. This is not a general-quality claim.
