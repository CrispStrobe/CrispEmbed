# Closed physical upstream Q4_K_M experiment

The default-off `CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL` path admits a separate
format-6 experimental container tagged `lighton4b-upstream-q4km-physical-v1`.
It retains the publisher's actual 216 Q4_K and 33 Q6_K text matrices, including
the tied Q6 embedding/head, and transports authoritative small F32 parameters.
Whole-row inverse permutations retain all quantization blocks; 24 GDN output
matrices retain physical columns and receive a corresponding activation
permutation. Norm gamma and pretransformed negative A values are consumed
directly. Format-5 behavior, other precision recipes, public model admission,
aliases and releases remain unchanged; the public shipping fence rejects this
experimental format even when prototype gates are enabled.

The Kaggle-only packer authenticates both entire immutable source files and all
723 persisted tensor spans. The generated candidate has SHA-256
`843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`;
297 protected vision tensors remain from the original native container.
This is exact physical **text** weight transport, not a claim of identical
multimodal weights: upstream uses its F16 projector. The
[transport and setup evidence](https://github.com/CrispStrobe/CrispEmbed/blob/47005f2a61c55f3883a01db91fe5af245ab95f27/docs/lighton4b-upstream-physical-import.md)
records all sources, raw tensor readbacks and initial dependency/setup failure.

A second, independently default-off gate,
`CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS`, requires the physical owner and its
prototype admission gates. It uses retained quantized text/head matmul and ggml
RMS normalization. The older dequantized F32 projection and scalar RMS paths
remain available with this gate off. Decoder, attention and linear-cache tags
reject toggling arithmetic mid-request. WASM cannot enable the physical path;
conflicting BF16 modes are rejected rather than combined silently.

The focused controls at `dffc7780c8baf791c32a26259efdbd65953d4988` use matched
LLAMAFILE-on builds. Runtime RMS and QKV/Z/a/b projections match pinned upstream
byte for byte on reconstructed four-token-prefix inputs, with both normalized
input variants separated. Gate/cache/nonphysical controls (68), older streamed
Q6 controls (84), and the actual native document driver build pass. An initial
strict mismatch caused by the reference LLAMAFILE profile is retained; no
tolerance was relaxed. See the
[component evidence](https://github.com/CrispStrobe/CrispEmbed/blob/ef9bd2248031854887fce3bc21a4550f683b8304/docs/lighton4b-physical-prefix-primitives.md)
for scope and raw reports. Actual-main intermediate closure and full decoded
quality remain separate acceptance gates. Reconstructed component parity is
not a model or decoded-quality result.

Reference helpers and model-free targets are not installed into product assets.
The external primitive probe and dequant audit are excluded from default builds.
The source-feature override belongs only to the document test executable, which
can preserve partial IDs/text and stage arrays for the bounded same-feature
comparison. No original BF16 forward, full F32 model, image-specific fitted
constants or release change is introduced.

## Actual composed negative: keep native-ops disabled

The matched-profile source-fed projection/RMS controls pass, but the actual
same-feature form replay is a numerical regression with
`CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS=1`: firstscore RMS0.1023485→0.1296631,
max0.576516→0.752944 versus pinned upstream. New OFF closes all38 stage hashes
to fresh ac34 exactly. Every arm emits only `<figure` at cap2/noEOS. Faster
ordered prefix timing is not decoded-quality proof; keep this gate default-off,
retain all shipping/alias fences, and do not ship the current Q4 candidate.
The [immutable negative and actual-main closure](https://github.com/CrispStrobe/CrispEmbed/blob/678d3ca790fbdbf8eec69dda9cbcbc2996170186/docs/lighton4b-physical-native-ops-negative.md)
preserve raw receipts and limitations. Controlled ablations and source-fed
GDN/attention/cache/softmax attribution are separate pending work.
