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

Native ggml `3a27b6c6` and upstream ggml under llama.cpp `6c01da62` produce
bit-identical RMS, quantized-direct and dequantized-F32 projections on identical
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
