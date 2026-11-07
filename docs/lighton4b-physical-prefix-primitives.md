# Reconstructed same-input physical prefix operators

Exact source `f155c591e3e8fa7e3cfb59f719a477a886976fc7` ran on one Kaggle CPU
worker, version2. The [raw report](../tests/results/qwen35-ocr/lightonocr3-4b-physical-prefix-primitives/prefix-primitives.json)
and [artifact hashes](../tests/results/qwen35-ocr/lightonocr3-4b-physical-prefix-primitives/artifact-index.json)
retain both pinned ggml builds and all small output identities. Initial version1
failed in bootstrap before tensor transfer/operator execution: the bootstrap
retained an old worker and nonexistent protocol path. Its failure remains saved.
Version2 fixed those bootstrap paths; all repository paths, exact source and
receipt admission were statically checked. No model inference was executed.

The full154 prompt-ID file and original NPY hashes authenticate prefix IDs
248045,846,198,248053. Complete physical tensor range hashes authenticate the
Q6 embedding and five layer0 weights (544675840 bytes transferred). Four rows
are reconstructed from the authenticated embedding. This is **operator-only
reconstruction**, not an actual upstream graph trace or native-main capture.
The primitive source, build options and procedure are at
[the pinned workflow](https://github.com/CrispStrobe/CrispEmbed/tree/f155c591e3e8fa7e3cfb59f719a477a886976fc7/tools/physical-primitive-probe).

Under version2 standalone **LLAMAFILE OFF**, native ggml `3a27b6c6` and upstream
ggml under llama.cpp `6c01da62` produce bit-identical RMS, quantized-direct and dequantized-F32 projections on identical
inputs. Reconstructed native scalar RMS differs from upstream ggml RMS by
RMS5.2285653e-7/max5.7220459e-6 (reference magnitude8.9882).

| Same-input projection | Direct quantized vs dequantized F32 RMS | Maximum absolute error |
|---|---:|---:|
| QKV, Q6_K | 0.006449859 | 0.05174017 |
| Z, Q4_K | 0.007109885 | 0.04078341 |
| a, Q4_K | 0.01521108 | 0.04484177 |
| b, Q4_K | 0.006502788 | 0.02707866 |

Thus direct-quantized activation arithmetic is a concrete larger component
residual on this prefix; backend version alone did not explain it. These numbers
do not establish the full physical firstscore error's cause. Native/source
batching, later operators and cache behavior remain separate. Actual native
first4 trace hashes and row-permuted comparison are pending; no decoded-quality,
shipping, heldout or speed claim follows. A proposed physical-only opt-in
retained-quantized/RMS alternative must pass focused controls and actual replay.


## Matched actual build profile and scoped fix

Focused version3 source `a3f4b87aac82662393548b46ec19215618cfc038` compiled
the actual native driver and passed68 new controls (gate-off byte identity,
nonphysical owners, streamed head and cache toggle rejection), old stream controls
and84 Q6 controls/65792 independent scalar values. Runtime RMS and retained-Q6
QKV matched the reference exactly. The strict legacy F32-dequant QKV comparison
failed RMS3.953195e-7/max5.722046e-6: standalone ggml defaulted LLAMAFILE OFF,
while both actual parent runtimes use ON. The [negative](../tests/results/qwen35-ocr/lightonocr3-4b-physical-prefix-primitives/v3/prefix-primitives.json)
is preserved; it is a build-profile mismatch, with no tolerance waiver.
The version2 cross-version conclusion above is limited to its disabled profile.

Corrected version4 source `dffc7780c8baf791c32a26259efdbd65953d4988` explicitly
sets and admits LLAMAFILE ON in both reference builds and the native runtime,
with AVX2 ON/NATIVE OFF/AVX512 OFF/BLAS OFF/four threads. Its [raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-physical-prefix-primitives/v4/prefix-primitives.json)
SHA256 is `79dff60f82e76d689811d2151d3c76469e0b80c3a6e5d64eabe6eba0c085f398`.
All three implementations are byte-identical for RMS and direct/dequantized
QKV/Z/a/b on both identical reconstructed scalar-norm and upstream-norm inputs.
Independent raw review closes17 three-way output groups and recomputes all8
reported arithmetic differences; all3 actual build profiles admit ON. The68 new,
old stream and84 Q6 controls pass again. No learned model forward was executed.

The opt-in `CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS` alternative requires an
explicit physical owner and existing prototype/native/quant/Q6 gates, excludes
BF16 modes and returns false on WASM. It retains host-backed Q4/Q6 matrices for
matmul (including the tied head), with no full-model F32 cache; text/final and
full-attention Q/K RMS use ggml. Decoder, linear and attention caches bind the
arithmetic mode and reject toggles. Gate-off and nonphysical paths remain legacy.
GDN post-recurrence norm, convolution, SiLU, attention and later layers remain
separate attribution stages. Main-path trace closure, same-feature firstscore
and full decoded output remain pending; component exactness is not qualification.
