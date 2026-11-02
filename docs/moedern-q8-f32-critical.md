# MoEdern genuine Q8_0/F32-critical packed profile v1

This separately named experiment adds packed Q8 admission for1.5M,27M and115M.
It is source preparation, with no compiled or model result. The current full
SciFact F32 failures still block Q8 and Q4 qualification:1.5M has three failing
documents/six matrices;27M and115M reject one/two documents at the exact router
top-k cutoff. Preserve those negatives and all frozen originals. A weight error
or authored graph pass is not vector/retrieval or publication proof.

[The source contract](moedern-q8-f32-critical-v1.json) starts at reviewed shared
metadata source `7d886594d2f487cdaa9420712a8df39eda89e744`. It protects root build,
actual bindings, ggml and all unrelated source objects; only five declared MoE
admission/routing files change. The scalar1.5M engine, family fast-graph body,
old F32 packed profile, padded Q4 profile and historical converter/guards remain
unchanged. The existing family backend already supports Q8_0; no kernel, router,
activation, normalization, attention, expert weighting or output math is added.

The new default-off `CRISPEMBED_MOEDERN_Q8_0` uses `core_env::on`: unset, empty
and literal0 are OFF. It requires the existing COLBERT/FAMILY/QUANT/FAST gates,
and packed text additionally requires the existing TEXT_FRONTEND/TEXT_COMPOSED/
PACKED gates. Concurrent Q4_K_PADDED and Q8_0 admission is rejected. The frontend
sets an explicit family revision for all named Q8 profiles, including1.5M; only
that explicit1.5M profile routes into the existing family fast graph. Ordinary
F32 one/two-thread scalar behavior and old27M/115M diagnostic Q8/Q4_0 routes stay
available. A named Q8 file cannot enter the generic family initializer instead.

The [new profile](../src/moedern_q8_profile.h) requires:

- `general.file_type` UINT32 standard7, quantization-version2 and genuine actual
  Q8_0 tensor type8. No Q4_K/Q4_0/F16 storage or header relabel is accepted.
- `moedern.packed.format=crispembed-moedern-q8_0-f32-critical-v1` and
  `modernbert_hash_moe_v10.quant_recipe=q8_0_f32_critical_v1`, with original model,
  revision, tokenizer and original-report metadata pins unchanged.
- Exact32/32/94 tensors and1,484,448/27,124,000/114,696,576 semantic parameters.
  Exactly20/20/52 attention/expert/conv-in/out tensors use Q8_0; the remaining
  12/12/42 hash-table/hash-projection/router/dense-head/norm/bias/depthwise tensors
  retain F32. Every semantic row is already32-aligned; there is no column padding.
- Bounded4MiB header/512MiB file, unique typed names, contiguous exact extents,
  zero padding, no overlaps/trailing data, exact semantic shapes and finite
  dequantized rows before a graph. The family loader enforces each role's type.

[The new converter](../tools/moedern-colbert/quantize_q8_critical.py) authenticates
the exact published F32 whole-file SHA before optional array imports, calls the
actual `crispembed-quantize q8_0`, forces every critical tensor F32 and overrides
only the two existing STRING profile keys. It verifies every actual type/shape/
extent, F32 critical payload SHA and unchanged raw metadata outside those keys
and file-type/quantization-version fields. Q8 row dequantization errors are
retained separately from output-quality gates. Source files are never mutated;
old Python Q8/Q4_0 diagnostic conversions remain distinct historical evidence.

Hosted-only controls are prepared under [tests/moedern-q8](../tests/moedern-q8):
actual standard-header/profile negatives for all three variants, pre-allocation
storage mutations, unset/empty/0/concurrent gate rejection and six actual ggml
Q8 direct/expert graphs. Authored values make both weight and activation Q8
encodings exact, so the0.002 component comparison does not hide activation
quantization drift. The separate tiny C-quantizer fixture verifies real standard7,
2D/3D Q8 blocks, independent half-scale/integer decode and all seven critical
role payloads. These fixtures are not actual model qualification. Build and run
only on hosted CPU resources; no root CMake or public ABI change is needed.

Before a model can qualify, resolve the original full-F32 blockers, then run
unchanged inputs/token IDs/three masks, intermediate actual Ref/independent
comparisons, all5,483 vectors, all300 rankings and actual public consumers for
each size. Retain0.002/0.99999 F32 vector gates,0.002 MaxSim, maximum0.001
NDCG/MRR/Recall degradation and minimum0.99 top1 agreement, plus all strict
quantized flags. Evidence-based critical-precision successors must be declared
before measuring their outputs. No alias, publication, runtime default, release
or application pin follows from this preparation.
