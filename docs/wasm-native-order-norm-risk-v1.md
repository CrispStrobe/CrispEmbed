# Retained native-order norm risk v1

This is source preparation for a test-only operator study. Production source,
weights, defaults, original captures and whole-model gates are unchanged. No
numerical result or model qualification is claimed by this declaration.

The retained [same-input F32 operator run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40771950208)
passes its64 byte anchors and48 comparisons, with small nonzero norm differences.
The [F32 weightcast risk run](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40791643276)
rejects18 of25 unchanged-target comparisons. Neither result resolves the earlier
whole-model negative. This study retains the actual stored IQ4_NL weights and
Q8_0 activation path, instead of casting coefficients to F32.

The blueprint is pinned GGML [a06fc0b](https://github.com/CrispStrobe/ggml/tree/a06fc0bb46b73cc10011fc90ae4054aee8438aeb):
`src/ggml-cpu/vec.h` sequential F64 sum then F32 mean;
`src/ggml-cpu/vec.cpp` AVX2/FMA `ggml_vec_cvar_f32` subtracts and squares in
F32, combines each group of eight with high/low128 and pairwise F32 additions,
then accumulates those chunk sums in F64 and divides by384 before F32 conversion;
`src/ggml-cpu/ops.cpp` applies F32 epsilon1e-12, square root, reciprocal and scale.
The portable test emulator preserves those rounding points. Actual compiler
preprocessor output must prove AVX2/FMA without AVX512/fast-math for the native
baseline, and SIMD128 for the browser. Complete original/current CPU commands
and selected arithmetic flags must agree. A failure to recover native bytes
stops the study; this description is not evidence that recovery already works.

The zero-input [workflow](../.github/workflows/wasm-native-order-norm-risk-v1.yml)
uses two build workers and a60-minute hosted budget. Its ordered barriers are:

1. Authenticate service, ZIP, complete member inventories and immutable reports
   for the original layer3 trace and F32 operator proof. Authenticate the model
   and exact stored `blk.3.ffn_up.weight` type20 IQ4_NL, shape384×1536,
   331776 bytes. Retain its original SHA and all input/parameter/output hashes.
2. Rebuild the frozen580/a06 backend with unchanged production arithmetic flags
   and a test-only selected-converter capture hook. Native1/2 and the isolated
   SIMD128/pthread browser must reproduce30 old norm/affine/projection/workspace
   byte anchors across the saved native and wasm1 residual inputs. Actual Ref
   positive/numeric/shape controls and local invalid-input/recovery controls run.
3. Recompute attention and FFN norm from each of those two residual inputs with
   the portable native-order emulator. All12 native1/2/browser outputs must be
   byte-exact to the retained native norm on the corresponding input **before
   any candidate projection**. The browser and native phases enforce the same
   barrier independently.
4. Recompute attention norm in the candidate function, apply the original F32
   gamma/beta, and execute the real IQ4_NL matrix multiplication. Its selected
   Q8_0 conversion is captured after the actual backend conversion barrier:
   native policy0 and context-local browser rounding policy1. The function takes
   no saved norm, original output or workspace argument. Ordinary/observed/repeat
   results and read-only inputs must match; the actual kernel input must equal
   the freshly computed affine output. Report every Q8_0 block scale and signed
   activation-byte hash, with changes against the retained native workspace.
5. Collect all56 actual `crispembed_diff::Ref` comparisons plus independent F32
   max/mean/RMS errors, magnitudes, norms and minimum row cosine. Coverage is12
   norm recovery pairs,24 candidate chain stages against unchanged old native
   targets,12 natural-input sensitivity pairs and8 cross-backend chain pairs.
   Fixed gates remain max absolute0.002 and minimum row cosine0.99999. Numeric
   rejection is retained before a nonzero job exit; structural failure records
   its phase separately. Small reports and raw diagnostics are retained90 days.

Only two saved12-token residual inputs and layer3 attention/FFN norms plus
attention-norm/affine/FC1 are covered. FFN norm is a control; no FFN-down or later
layer candidate executes. A local positive cannot qualify a runtime change.
All ten old final targets and the first sentence's eight named stages still
need unchanged0.002/0.99999 gates in a separately reviewed full-model A/B before
any production proposal. No decoded/task-quality, release or default promotion
is part of this workflow. Large arrays, modules and model bytes remain remote.

The initial generic formatting run [40831805927](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40831805927)
failed before this operator study. Its capped log includes inherited formatting
violations, preserved in the [initial receipt](../tests/results/packaging/wasm-norm-risk-initial-lint.json).
The successor formats only the new norm operator, requiring an identical C++
token stream to frozenf781. Historical protected files remain byte-exact; fixes
to unrelated inherited formatting belong to their respective integration owners.
