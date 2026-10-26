# Frozen-input UltraBERT RoPE attribution

This operator-only audit tests a distinct hypothesis after the completed local
linear and four-arm Python/M1 controls. The pinned original source computes
independent F32 inverse-frequency powers; the actual ggml CPU cache advances
phase by recurrent F32 multiplication. Their mathematical equivalence does not
establish matching coefficients or explain the accumulated hidden-state errors.

Use original run38997562520, source`0573f1847db68914052a98341a22fceeb1358518`,
through its verified preservation39913800444. The workflow binds both service
identities, the preserved ZIP and every member byte/hash before using tensors.
For each of the seven frozen cases, all24 layers retain their actual original
Q/K inputs, positions and48 cosine/sine table fields. No checkpoint is downloaded,
no full model is instantiated and no full forward or conversion is recaptured.

Actual hash-checked `RotaryEmbedding` and `apply_rotary_pos_emb` calls produce
additional local targets. Regenerated original tables and their deviations are
reported separately from frozen tables; neither can replace historical gold.
Repeated function outputs must match bit-for-bit, and arguments must remain
unchanged. The preserved source/config, CPU framework and both local/global
theta schedules remain explicit.

The test-only C++ target runs actual `ggml_rope_ext` on a split-half basis to
observe its real coefficients, then on frozen Q/K. A separate graph supplies
the literal frozen tables to ordinary ggml multiply/add operations and the
source split-half rotation. Ordinary and observed native RoPE outputs must
match exactly; tensor transfers must preserve input/table bits. Per case,192
comparisons and48 bit controls cover both frozen/current targets and the
source-table control. Hosted Python independently recomputes every dumped
metric and records shape, bytes and SHA256.

All numerical comparisons keep maximum absolute0.002/minimum row cosine0.99999.
Any numerical failures remain failures and produce a nonzero workflow exit;
all independent cases are collected first. Coefficient/table bit differences
are visible even when these numerical gates pass. Historical183 F32/1294 F16
full-stage failures and both strict replay alternatives remain unchanged.
There is no production arithmetic/gate change, model publication, release or
decoded-quality qualification in this experiment.

Run `.github/workflows/ultrabert-rope.yml` on the immutable candidate source,
or invoke it as a reusable workflow. Only the small reports should be downloaded
for review; additional operator targets/raw arrays remain in the separately
retained hosted artifact. Any subsequent correction needs its own default-off
path, unchanged complete replay and representative task-quality evidence.

## Completed bounded result

[40336758177](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40336758177)
atb689f871 passes every declared comparison/control. Frozen/current tables and
source-fed native Q/K are bit-exact; ordinary worst coefficient/Q/K errors are
3.05157e-5/1.37925e-4/1.61648e-4. Empty rotation is exact and cannot explain its
earliest pre-RoPE drift. No new production gate or composed causal claim is
justified by these within-budget local differences. Historical full-stage and
strict replay failures remain. The independently reviewed
[receipt](../tests/results/ultrabert/rope-attribution.json) retains exact provenance,
magnitudes and limits; representative quality continues in the SciFact lane.
