# MoEdern padded Q4_K experiment

This separately versioned, default-off source candidate supports all three
pinned MoEdern profiles. It has no native or quality result yet. The existing
F32 loaders, original captures, historical guards and published models remain
strict and unchanged. No publication or default promotion follows from this
source preparation.

The recipe keeps hash embeddings/projection, routing, dense projection, norms,
biases and depthwise convolutions F32. Eligible attention, expert and pointwise
convolution weights use actual GGML Q4_K blocks (256 elements / 144 bytes).
Semantic columns 128 and 640 have storage widths 256 and 768; width 1024 needs
no padding. Expert/output axes and every metadata architecture dimension remain
original. Narrow weights without a byte saving remain F32. Expected actual
Q4_K/F32 tensor counts are 20/12, 20/12 and 52/42 for 1.5M, 27M and 115M.

The new `crispembed-moedern-q4k-padded-v1` format records each selected name and
its logical/storage columns. Before allocating GGUF metadata the new bounded
header reader permits only F32/Q4_K, checks row alignment, contiguous extents,
padding, duplicates and file/header bounds. The subsequent loader requires the
exact original semantic schema and complete declarations. Only each selected
Q4_K matrix receives zero columns at its graph input; attention products and
F32 matrices keep their original shape. Both ordinary matmul and expert
`mul_mat_id` use the same checked padding helper. This is storage padding, with
no fused rows or implicit reinterpretation of semantic dimensions.

Every path requires value-parsed `CRISPEMBED_MOEDERN_Q4_K_PADDED=1`,
`CRISPEMBED_MOEDERN_QUANT=1`, `CRISPEMBED_MOEDERN_FAST=1` and the existing
family/text/public gates. Unset, empty and zero gates reject. The new 1.5M
quantized route explicitly uses the family FAST graph; ordinary 1.5M F32 keeps
the earlier engine. Test code and callbacks are excluded from the ordinary
library. Authored hosted controls exercise actual Q4_K encoding, matmul and
expert matmul at all three widths, exact zero columns and excess-padding
rejection. Model-backed controls toggle gates within a loaded context and
require recovery to preserve final vector bits.

The [converter](../tools/moedern-colbert/quantize_q4k_padded.py) accepts only
the three exact published self-contained F32 files. It preserves all tokenizer
metadata, writes padded F32 scratch rows, then delegates block encoding to the
unchanged C++ quantizer. Readback rejects a fallback type, missing declaration,
unexpected shape, nonfinite decoded weight or any change to an excluded F32
payload. The report includes actual types, bytes and per-tensor hashes. The
result must be smaller than its F32 input.

The [standalone hosted workflow](../.github/workflows/moedern-q4k-padded.yml)
starts with an exact source/ggml closure. `build-only` uses no model payloads;
`panel` authenticates existing original reports and all twenty selected frozen
arrays before fetching a pinned public F32 file. It binds the 1.5M GitHub
producer and the 27M/115M Kaggle original version labels `v3`/`v1`; no latest
fallback or original rerun is accepted.

`run_panel(original_dir, f32_model, variant, build, out)` can also consume
authenticated mounted originals on hosted infrastructure. It first compares
the actual raw-text F32 consumer against all ten frozen cases/twenty vectors
and three MaxSim scores. A separate family FAST F32 control uses IDs emitted
by that real native frontend. Only then may conversion and quantized inference
begin. All IDs, supplied/effective/scoring masks and query expansion positions
must remain exact. Actual `crispembed-diff::Ref` and independent NumPy report
the unchanged 0.002 absolute / 0.99999 row-cosine vector gates, error/magnitude
metrics and 0.002 score gates. Quantized numerical rejections are retained
across all cases, followed by repeat and recovery controls; they cannot be
renamed successful F32 parity. Structural failures reject separately.

No frozen retrieval-quality acceptance contract for all three quantized models
has been reviewed. The source declaration records this dependency as pending;
the diagnostic cannot set publication qualification. Before any publication,
freeze original/F32 retrieval outputs and acceptance gates, evaluate each
candidate on those same inputs and preserve failures. Component cosine and
the ten-case owned panel alone do not qualify retrieval accuracy, a release,
other platforms or the app's immutable runtime.

### Integrated source closure, 28 October

The initial prepared source at `645f51e8` remains independently recoverable.
The coordinated portfolio closure is now based on `3586a8915f58ac6eca5cd28204abd00a2518f679`,
including the exact proven UltraBERT Unicode tokenizer, Josev mixed loader,
MiniCPM graph path and explicit four-dimensional F16 patch-cast repair. This
new declaration pins every production source and the actual common quantizer;
it does not substitute the old standalone branch's build for an integrated run.
The MoEdern metadata boundary additionally requires actual Q4_K file type and
quantization version, with authored missing/wrong-type/version controls.
Historical source closures and numeric gates remain unchanged. Run the hosted
build-only controls on this exact new closure before its full model diagnostics.
