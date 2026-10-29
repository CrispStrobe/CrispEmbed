# Layer3 F32 weightcast: retained initial negative

[Run40791643276](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40791643276)
failed at frozen source`e3df7cfd2e38f6f1bd20d68f1eb05dce11f7cd9d`.
The [initial independent review](../tests/results/packaging/wasm-layer3-f32-risk-initial.json)
preserves all thirty small-artifact member hashes, six exact quantized anchors,
coefficient identity, compiler/neutrality controls and all25 actual Ref rows.
No production weightcast promotion or reference replacement follows.

The operator arms completed. Native1/2 and browser reproduce all six old
quantized outputs. The2359296-byte F32 coefficient arrays agree with the literal
IQ4_XS decoder and across backends, SHA256
`f85e1f1a4a06267c41760e982ad4981c7adf282acb0a048aaed62e69a4a44ef3`.
Ordinary/coefficient-observed/repeat/read-only-input and rejection/recovery
controls pass. Native one/two-thread outputs are identical.

The saved actual Ref report rejects18 of25 comparisons at unchanged0.002
absolute/0.99999 minimum-row-cosine gates. Seven pass: native/browser F32
agreement, native thread agreement and natural-input sensitivity. Against the
unchanged old quantized native target the worst error is0.07845640182495117,
RMS0.008432731963694096 and minimum row cosine0.9997050166130066. The original
maximum magnitude is47.73543930053711 and the cast magnitude47.74108123779297.
Native/browser agreement alone cannot qualify that changed arithmetic.

A later optional F64 dot diagnostic called the existing F32-only metric helper
with a float64 reference. That reporting error prevented final report assembly;
it did not cause or invalidate the already saved18 Ref numerical rejections.
The separately declared [compare-only replay](wasm-layer3-f32-reanalysis-v1.md)
will recollect independent F32-array metrics from immutable retained buffers.
It omits that optional dot rather than rounding/relabeling an F64 reference.
No model, original, operator or build repeats are needed.

Only the28168-byte small ZIP was reviewed locally, SHA256
`aac1e16891a6fc39d74e1d6501db996c106877a62ced2f03a22cd45124c6cf31`.
Raw artifact12865511872 remains remote:8648225 bytes, SHA256
`58f82e19b7dac340c33d1961f8547d11cc2efe848b48412e77cf4a67e18ff392`,
expiring27 January2027. Saved source/compiler/hash/Ref records were checked;
full raw-array metric recollection is pending hosted reanalysis. The original
whole-model negative remains open.
