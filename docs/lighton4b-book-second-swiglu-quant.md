# Second image spike: closed SwiGLU to Q8_K attribution

The [reviewed operator receipt](../tests/results/qwen35-ocr/lightonocr3-4b-book-second-swiglu-quant.json)
records a successful hosted CPU replay (157.733 seconds), following the
[cache-closed second-image observation](lighton4b-book-second-image-layer0.md).
Controller `3b4b5d9cb` plus the four exact test-only overlay hashes identify the
executed source. Native GGML `3a27b6c6` and upstream `6c01da627` use the same
explicit AVX2/FMA/F16C profile. The historical native tiled Q6 helper comes
unchanged from `a61642056`. Only the original, whole 19,353,600-byte Q6 down
matrix is fetched; there is no model forward, decoded generation or runtime edit.

All 147,456 actual gate/up elements match before this replay. Native scalar
SwiGLU reproduces native activation exactly; both GGML fused implementations
reproduce actual source activation exactly. Scalar activation fed to the historical
native tiled helper reproduces actual native down exactly. Fused activation fed to
either that helper or the upstream raw graph reproduces actual source down
exactly. All twelve public-Q8 forced-output controls are bit-exact. The generic
native raw reduction remains separately measured: it differs from actual native
down by RMS1.64427e-8/max3.57628e-7, so it cannot replace the tiled closure.

| Actual boundary | RMS error | Maximum absolute error | Source RMS magnitude |
|---|---:|---:|---:|
| Scalar versus fused activation |2.67390736e-9|1.19209290e-7|0.05243855|
| Scalar versus fused tiled down |4.21739905e-6|8.72015953e-5|0.05260878|
| Down, image row23 only |1.68695903e-5|8.72015953e-5|0.05555850|

Across 576 Q8_K blocks, scalar→fused changes 98 scale words, one integer code
and one block sum. The sole integer change is local row7/image23, block10,
lane143 (feature2703): code10→11. Its scalar/fused F32 activation values are
0.02059226669371128/0.02059226855635643; converter scales are
0.0019611683674156666/0.001961168134585023. Reconstructed Q8 values therefore
change 0.0196116836742→0.0215728494804. These are operator quantization codes,
not vocabulary IDs. The forced-output closures tie this public converter
intervention to the reproduced outputs; internal CPU buffers were not captured.
An independent stdlib audit rehashes 57 selected small artifacts and reproduces
the 98/1/1 counts directly from the selected operator arrays. It does not claim
a complete remote raw-artifact audit.

The initial hosted attempt stopped at controller-source admission because a
read-only git query inherited the worker directory. Its report hash is preserved;
no build, parameter transfer or operator computation occurred. Explicit repository
paths repair that orchestration error without changing the numerical contract.

This establishes the local cause of this observed down-projection spike: scalar
SwiGLU changes Q8_K rounding before unchanged Q6 weights. It does not establish
whole-model or decoded quality. A separate default-off source-operator candidate
would still require disabled-path identity, actual stage/cache closure and the
unchanged decoded panel. Preserve the prior timeout, cap, EOS and content
negatives; no default, recipe, alias, application pin or release changes follow.
