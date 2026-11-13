# Second image chunk: first spike reaches quantized FFN down

CPU execution144617617/version1 passes the single cache-closed second-chunk
capture. The [safe receipt](../tests/results/qwen35-ocr/lightonocr3-4b-book-second-image-layer0-review.json)
records controller `3b4b5d9cb`, exact test-only overlay `b6c86e63f` and its four
file hashes, unchanged compiled runtime `a61642056`, ggml `3a27b6c6`, and the
33,872-byte raw report SHA256. Worker/bootstrap times are279.55/303.06 seconds,
including source/build/model transport; these are not production latency claims.
Five preparation controls pass. No push, source forward or decoded generation.

All82 historical processor/vision/prefill/firstscore hashes close. Full layer0
tile outputs and final cache match ordinary main, previous caches remain unchanged,
and projected-core replay matches actual block attention/convolution/recurrent
bytes. The selected image rows16–31 consume native cache20 and produce cache36.
The spike at image23 is therefore inside the observed actual second image chunk.

Source's actual252-row image graph and native agree bit-for-bit on embeddings,
input norm, QKV, Z, alpha/beta, gated GDN values, attention projection/residual,
postnorm and FFN gate/up. SwiGLU first differs by RMS2.67391e-9/max1.19209e-7.
The same chunk's quantized down projection differs RMS4.217399e-6/max8.72016e-5;
resulting layer0 hidden differs4.217493e-6 with the same maximum, reproducing the
independently localized window spike. Source down RMS magnitude is0.0526088.

This localizes the first large residual to the activation→quantized-down boundary,
with earlier captured GDN/projection/normalization inputs exact. The subsequent [same-input operator replay](lighton4b-book-second-swiglu-quant.md)
closes source-fed fused down output exactly and identifies 98 scale changes plus
one integer-code change at image23; it requires no further full-model forward. All prior numerical/decoded negatives and default-off gates remain.
