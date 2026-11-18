# PCS causal diagnostic checkpoint — 18 November 2026

Status: diagnostic work stopped at the user's final checkpoint. PCS remains
quality-negative. No production weights, production PCS source, dependency
pins, defaults, frozen references or acceptance thresholds were changed.
See the [operator comparison contract and measured results](pcs-step-diff.md).

## Completed evidence

Corrected observer source `9860370c46b93fd311259d37f307c5903e6a1d24` was
qualified for the one frozen request in
[run 44347337837](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44347337837),
artifact `14446463047`. Original-driver/OFF and OFF/ON comparisons match all
four head dumps and decoded output exactly. All 12 corrected context snapshots
pass. The earlier context captures aliasing later bias values remain preserved
as observer failures, not native arithmetic evidence.

Commit `0b54ef8ba69dbcf886ca5fb177f887b91b712419` adds the offline
`tools/pcs_operand_cause.py` analysis and six passing controls. It uses existing
qualified captured operands and the existing reference-input native arm;
it performs no model or capture run. At the first composed QK budget failure,
layer1/head8/query11/key9, operand propagation contributes +0.0002894699423
to the observed +0.0002994537354 discrepancy. Reference-input local error is
+0.00001907348633, with -0.000009089693243 input-dependent remainder.
All four retained original head biases match stored dequantized bytes.
Token and position table contributions are separately measured before
embedding normalization; this does not identify final decoded causality.

The first nonzero same-weight native difference remains the one-ULP embedding
sum difference, explained by actual source addition orders. It is separate
from the first budget failures. All ten composed F32 QK budget failures remain.
Independent-input replay's 266/266 operator passes do not repair decoded PCS.
Row37's original `o` logit is +1.083245516 versus stored-graph -0.238312081;
composed output still contains `oK`. Original full PCS/fullstop gates were not
rerun by these offline analyses. No model-fixed or full-gate closure claim.

## Last metadata inspection: selective payload availability

The established `dump_pcs_reference.py --steps` path hashes and loads the full
models, then runs inference. It cannot be invoked unchanged for the final
request, which authorizes selective tensor access but no new model run.

The local provider cache contains an original ONNX candidate under revision
`d1769a597ce8dfaa070d436bc67d4ee761f58884`, size 1,112,481,438 bytes. Its
snapshot path and size match retained origin metadata, whose expected SHA-256
is `c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89`.
Content identity was not newly verified; path and size alone are insufficient.
No initializer payload was read or extracted.

The exact stored revision `27f17edbd7e33949f42a9a4fe4322b5af8172e6b`
was absent from the inspected provider snapshot and known model locations in
the main and two diagnostic checkouts. This is a scoped absence finding,
not a claim that no other retained copy exists. The existing uploaded operator
artifacts retain activations, parameter hashes and selected embedding rows,
not the full model files or these projection payloads.

The admitted mapping identifies the required tensors:

| Native name | Original initializer | Mapped F32 shape | F32 bytes | Stored type |
| --- | --- | --- | ---: | --- |
| `head.tc.fc1.weight` | `onnx::MatMul_1843` | 769 × 128 | 393,728 | F32 |
| `head.tc.fc2.weight` | `onnx::MatMul_1844` | 128 × 16 | 8,192 | Q4_0 |

The pair totals 401,920 F32 bytes per model. These are planned output bounds,
not extracted or verified payloads. Stored dequantized tensor hashes are:

- FC1: `14e7dc7e9cebb203d8856a54c5d5369058f640c34fb05cc6392f12cfcb18dcd3`
- FC2: `2a2476b19c5173fbea512d8dd3cf54aeae44ab33b9a9f95634f33386da699aa1`

No selective reader, payload admission or causal intervention was completed
before the stop instruction. This inspection started zero model/capture runs,
downloads, large model hashes or regeneration operations. The 5 GiB user-slice
limits remain unchanged; the last checkpoint had about 1.92 GiB current usage
and more than 1 GiB headroom. No heavy operation was admitted or lock acquired
for this metadata inspection.

## Remaining blocker and bounded handoff

The product blocker is the unresolved tensor/conversion cause of the
stored-weight decoded error. The immediate evidence dependency is the exact
stored casing projection payloads, plus integrity admission of the cached
original before extracting its corresponding tensors.

1. Locate an existing exact stored artifact through known provenance, without
   broad worktree recrawls, full downloads or repeated large hashes.
2. Bind the original candidate to retained immutable integrity evidence.
   Do not infer identity from the pinned snapshot pathname or file size.
3. Add a bounded selective mode to the established dumper. Before payload
   access, record exact source, file identity, initializer names, offsets,
   storage types, shapes, byte/output caps, memory and time admission. Skip
   unrelated payloads without loading the whole ONNX/GGUF. Retain the original
   shared-lock, CPU, host memory, disk and user-slice headroom guards.
4. Only if both exact tensor pairs are available and admitted, use the
   qualified captured inputs for a parameter-held-fixed head intervention.
   Validate the unchanged head against its captured outputs before attributing
   an intervention. No full model/capture replay, regeneration, production
   repair or numerical waiver is authorized by this handoff.

No unique tensor cause is established by the current correlations. If the
exact payloads or integrity evidence cannot be obtained within this scope,
record that dependency and yield rather than rerunning unchanged audits.
