# Retained-input WASM F32 operator result v1

[Run40771950208](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40771950208) at diagnostic source`2f864d4beb2f5213a323a21557415dc5e6fab9ad` completed successfully. The [independent small-report receipt](../tests/results/packaging/wasm-f32-operators-v1.json) records the complete review ZIP/member identities, frozen source580/a06, actual compiler profiles and all48 comparison metrics. Only the61,468-byte review ZIP was downloaded for this review; raw arrays/modules remain in artifact12856035451. The original model, sentence, weights, inputs and gates were unchanged; no model forward was performed.

All64 own-input output anchors pass before cross-backend comparison. Native1/2 outputs and repeated executions are byte identical; the cross-origin-isolated SIMD128/pthread browser has no page errors, with repeated outputs byte identical. Native and browser positive, wrong-shape, nonfinite and recovery controls pass. Actual `crispembed-diff::Ref` controls return0/1/2 for positive/numeric/structural cases, respectively. The48 Ref comparisons and independently computed raw metrics agree and pass maximum absolute0.002/minimum row cosine0.99999.

| Retained input axis | Passing boundaries | Byte identical | Largest absolute error | Smallest row cosine |
|---|---:|---:|---:|---:|
| Native |16/16|13|9.5367431640625e-7|0.9999999999999738|
| Legacy browser |16/16|13|4.76837158203125e-7|0.9999999999999847|
| Rounding-candidate browser |16/16|12|1.9073486328125e-6|0.9999999999999871|

This rules out a local discrepancy above the declared gates for these16 F32 operators on these exact identical inputs. Ten comparisons remain nonidentical. Their small rounding differences can still contribute to natural input drift and later activation-quantization changes; the result does not exclude normalization, attention or other operations as participants in the composed discrepancy.

The earlier natural full-graph comparison remains negative. Its first above-gate stage is `layer3_fc2_raw`: maximum absolute0.0033110380172729492, RMS0.0002794367648346981 and global cosine0.9999999411795382. The actual down-projection uses IQ4_XS weights/type23 and Q8_K activations/type15. Across the distinct natural native/browser GELU inputs,38/72 activation scales differ and two signed-quant block hashes change at token5, blocks3 and5. All12 earlier identical-input projection replays have exact activation bytes/scales across backends. These observations motivate a separately frozen natural-input/workspace cross-admission test; they do not yet establish a causal correction.

The original link failure40749050064 and compiler-profile rejection40756747682 remain preserved. This successful diagnostic does not qualify the earlier full text panel, change a production default, promote rounding, or establish retrieval quality. The next causal protocol must preserve all historical evidence and unchanged gates; no F32-weight runtime alternative is implemented by this result.
