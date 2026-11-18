# PCS matched-operator investigation

This continues the retained PCS casing failure. It does not repair a model,
change production arithmetic, regenerate a distributed artifact, or replace the
unchanged PCS/fullstop acceptance gates.

## Original code and matched quantities

`tools/dump_pcs_reference.py --steps` extends the existing PCS dumper. It runs
the pinned original ONNX operators, first with original weights, then with the
same 214 stored tensors used by native PCS. Original and stored-weight observer
predictions must each match their unobserved graph byte for byte. The original
predictions must also match the frozen reference for the selected request.

The model's Python pipeline delegates to `PunctCapSegModelONNX`; its collector
strips BOS/EOS. Encoder observations retain BOS/EOS, while head observations
explicitly remove them. Native full IDs, original graph position IDs and original
graph type IDs are compared exactly. The existing original collector/control
and all six frozen references remain unchanged.

| Boundary | Original operator/code | Native computation and comparison |
|---|---|---|
| Token/position/type lookups | Three Gather nodes | Actual get-rows outputs; selected original/stored weight rows compared separately |
| Embedding sum | token + type, then position | token + position, then type; partial sums are not falsely matched |
| Embedding normalization | LayerNormalization and recorded attributes | norm with epsilon 1e-5, scale/bias; original epsilon and weights recorded |
| Q/K/V | MatMul and bias Add | Each multiplication and bias output captured separately |
| Attention | QK, scale/mask, Softmax, value aggregation | Raw QK, probabilities and flattened context; selected no-padding request retains aligned inputs |
| Attention output | Dense, residual, LayerNormalization | Separate dense/bias/residual and completed normalization outputs |
| FFN | Dense, original Erf GELU chain, dense, residual, LayerNormalization | Separate multiplication/bias/GELU/residual and completed normalization; no tanh GELU substitution |
| Post/pre heads | Both linear layers, biases and ReLU | Matching graph outputs after explicit BOS/EOS alignment |
| SBD/casing heads | Concatenation, first affine, ReLU, output affine | Actual scalar-loop inputs, bias-initialized accumulation, ReLU and logits; fused accumulation is not labelled a bias-free MatMul |
| Conditioning/decoding | Original predictions/collector and frozen decoded reference | Actual SBD hard/threshold decisions, sentence-start bit, all casing logits and decoded text retained |

Named captures map to real graph nodes in `tools/pcs_steps.py`. The observer is
included only in a generated copy behind the explicit diagnostic build option.
Production `src/pcs.cpp` stays unchanged. Original driver versus observer OFF,
and OFF versus ON, must retain four head dump bytes and decoded output exactly
for each matrix path.

## Composed execution and local operators

`crispembed-diff-pcs` uses the existing `crispembed_diff::Ref` implementation.
Each stage must exist with its exact shape; nonfinite values fail. Reports
include per-row cosine, maximum/mean absolute error, RMS, tensor norms/peaks and
the worst element. Input identities require zero error. The new diagnostic
absolute budget is explicitly 0.0001; shared cosine defaults and existing PCS
quality budgets remain unchanged. Negative comparisons return nonzero after
the other arms finish collecting.

The legacy matrix path and the previously rejected F32-materialized control
remain separate. Neither is changed to obtain a passing comparison.

Reference-input replay records each real native operator result, then feeds
the independent same-weight reference output to its consumers. Q/K receive the
independent normalized embedding; bias Add receives the independent MatMul;
GELU receives the independent affine output. Fused normalization and
bias-initialized scalar accumulation remain explicit composite boundaries.
Replay is local-error attribution, not composed inference or quality evidence.

The dispatch-only workflow retains small intermediates and original failures.
It does not upload original/expanded model graphs, weights, libraries or
production artifacts. Prepared source is not an executed model result.
