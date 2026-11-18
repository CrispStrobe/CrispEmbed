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

## First measured panel and observer storage limit

Run [44334585175](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44334585175)
at `21dd802a7b57249a960fab32075bacc10001d789` builds both variants and passes
all 13 source controls. Both original graph arms and all six native arms
execute. All four head dumps and decoded bytes are neutral with observation
enabled. The 214 mapped tensor identities match the retained replay exactly;
expanded original/stored captures remain within 0.0001 of the retained anchors.

Independent raw-array review verifies all five 266-stage reports. Original versus
stored weights first differs at token lookup, maximum 0.04326669872; position
lookup differs by 0.04800090566. Type embeddings and embedding normalization
scale/bias are byte-identical. Legacy versus the same stored weights first
exceeds the diagnostic budget at layer-0 query multiplication, maximum
0.07043683529. Reference-input
replay retains that local difference (0.07043671608), separating it from
upstream amplification. F32-materialized reference-input replay passes all
266 boundaries; this does not repair the stored-weight decoded failure.

The composed attention-context snapshots have a separate observer defect:
contiguous reshape tensors are views, not owned snapshots. The pinned allocator
can release their unmarked backing allocation after the last inference consumer
even when the view itself is marked as output. These context captures cannot
be interpreted as native arithmetic errors. Independent raw inspection finds
the entire layer-0 context capture identical to the later attention-bias capture;
none of the six token/head/channel permutations explains it. At full-sequence
token 0/channel 720 it reads 8.556599617 instead of −0.09441738576. Other reported
negatives, including small QK absolute-budget exceedances, remain preserved.

The observer now prepares independent DUP output tensors. Their future graph
consumers retain the source allocation until copying; the copies then own
retained storage. This changes diagnostic observation only. The fresh
qualification below applies to the correction; the first panel is never
relabelled as having used the corrected observer.

The first panel independently confirms four-head/decoded OFF/ON equality for
both native paths. Original-driver versus legacy OFF captured decoded equality
only, not original four-head bytes. The corrected controller includes that
missing four-head control, measured in the next panel.

## Corrected observer: actual frozen-request qualification

Run [44347337837](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44347337837)
uses exact corrected source `9860370c46b93fd311259d37f307c5903e6a1d24`.
Build and all 13 controls pass; both original graph arms, all six native arms
and all five 266-stage comparisons complete. Independent raw-array review
verifies source/input identities, the same 214 stored tensors, shapes, numerical
metrics and unchanged reference anchors.

Original-driver versus legacy observer OFF now matches all four head dump bytes
and decoded bytes exactly. Both legacy and F32-materialized OFF/ON pairs match
those same bytes exactly. All 12 composed F32 attention-context snapshots pass;
their largest absolute error is 5.185604095e-6, with no later-bias aliases. This
qualifies the corrected observer for this one frozen request and CPU profile.

The numerical and quality failures remain. Composed F32 has 256/266 passing
boundaries and ten QK absolute-budget failures: first layer1 QK, maximum
0.0002994537354; largest at layer4 QK, 0.0003986358643. Independent-input F32
replay passes 266/266 with maximum 9.155273438e-5, below the unchanged 0.0001
budget. Legacy same-weight computation still first exceeds the budget at layer0 query
multiplication; original versus stored weights still first differs at token
lookup. No threshold, original scorer or reference is changed.

The first nonzero native/same-weight reference difference precedes those budget
failures: embedding summation differs by one ULP (1.192092896e-7) at full-token
1/channel588, native bits `bfc5f763` versus reference `bfc5f762`. A bounded offline
check of the captured operands reproduces all 9,984 output values bit-for-bit
with the two actual source orders: native `(token+position)+type`, original
`(token+type)+position`. This is within budget; it is not evidence that the
summation difference alone causes the decoded failure. First nonzero difference
and first budget failure are reported separately.

Row37's original `o` logit remains +1.083245516, stored-graph −0.238312081,
legacy native −0.098158233 and materialized native −0.238311455. Native composed
output remains `I'm oK, how are you? I'm fine.` Numeric operator passes and
observer qualification do not fix PCS or replace the original full PCS/fullstop
gates. Those full gates were not rerun by this single-request diagnostic.
