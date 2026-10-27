# Prepared first-attention reproducibility diagnosis

The original [40412164041](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40412164041)
at `790f7b1d0372c12d1807dfb2fc7fc3af20a3c445` remains frozen. Its observer
omitted Dense/Normalize pipeline arrays. The corrected observer
[40423259251](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40423259251)
at `8b153739503d6d817c7271a5ce319358eadaaab3` captured 271 arrays, including
the missing pipeline leaves, but correctly rejected old-output continuity.
156 of the old 250 descriptors changed, including all ten final vector outputs.
Three MaxSim changes were about `-9.54e-7`, `-1.91e-6` and `+7.63e-6`.
These are genuine recorded differences, not a JSON score-formatting issue.
Neither the original strict equality guard nor its rejected receipt is changed.

For both fully captured cases (`q-de`, `d-de`), the first differing recorded
stage is `actual.sdpa.0.output`. First-attention input, normalization, Q/K/V,
rotated/repeated tensors and mask values/shapes are exact across those runs.
Software inventories, Python 3.12.15, Torch 2.8.0+cpu, two intra-op threads,
one inter-op thread and the recorded Ubuntu image agree. Both programs used
deterministic algorithms and highest F32 matmul precision. Neither historical
receipt recorded CPU model/ISA, Torch build configuration, actual tensor strides
or the selected SDPA kernel. The saved arrays are contiguous snapshots, so
original stride/alignment metadata cannot be reconstructed from them alone.
Matching output bytes later cannot establish historical hardware or kernel identity.

The separately frozen [protocol](moedern-sdpa-reproducibility-v1.json) prepares one
hosted CPU [workflow](../.github/workflows/moedern-sdpa-reproducibility.yml), registered
as dispatcher phase `moedern-sdpa-reproducibility`. Its preparation authenticates
both actual service outcomes, all four raw/review archives, exact receipts,
checkpoint/source/loaded-weight identities and all retained array hashes before
installing or importing original frameworks. No new checkpoint, tokenizer or
reference corpus is downloaded. Original arrays and checkpoint data stay on the
hosted runner; local preparation runs only tiny stdlib/source controls.

The [operator driver](../tools/ci-heavy/moedern_sdpa_reproducibility.py) loads only
the publisher's original `HybridAttention` module and its five first-layer F32
parameters from the authenticated checkpoint. It feeds each captured first-layer
input and effective attention mask through that original source method. It
requires the observed normalization, projections, rotated/repeated Q/K/V and
boolean mask to reproduce both historical inputs exactly. It does not execute
the embedding, second block, routed experts, Dense/Normalize pipeline, tokenizer,
complete backbone, full model or any native model candidate.

Ordinary, observed and repeated module outputs must be bit-exact. The direct
operator then uses the same actual Q/K/V/mask tensor objects for default, explicitly
forced math and explicitly forced CPU-flash controls, with dropout zero and
causal false. Each arm requires repeat/observer/profiled output neutrality;
input object identity, values, layout and backend settings must remain unchanged.
Forced backend profiler evidence must identify that backend; unsupported kernels
reject without a fallback. No backend selection is adopted by the historical
producer or runtime.

The report preserves current CPU model/flags, Torch build/capability, MKLDNN
state, thread settings, actual reconstructed strides/alignment and aten profiler
kernel names. It compares each supported operator output and default module
output against both retained outputs, and compares the two historical SDPA
outputs directly. These comparisons use actual
[`crispembed_diff::Ref`](../tests/moedern-sdpa/diff.cpp) plus independent standard
library metrics: maximum/mean/signed mean error, RMS, row cosine, L2 and value
magnitudes. Lossless F32 GGUF adapters retain exact shape, row axis and payload
hashes; no reshape or precision conversion is permitted. The comparator's fixed
`0.002` absolute and `0.99999` row-cosine gates remain unchanged. Numerical
rejections remain recorded and do not prevent later diagnostic arms. Equal and
changed-row actual-Ref controls must pass first. Metric reporting consistency
uses the existing `2e-7` relative comparison and cannot waive either numerical gate.

The comparator is a small standalone C++ target using the existing Ref header;
it links no model runtime or GGML backend and changes no root CMake/API. All
compilation, original CPU packages and retained/operator arrays belong on hosted
CI, with two build workers and a 20-minute job bound. Only small reports are
downloaded for review; original/operator vectors, adapters and the exact comparator
binary are retained remotely for 90 days. Original source artifacts currently
expire January 24, 2027 and must be preserved before then.

A complete operator diagnosis can include numerical or unsupported-backend
rejections. It always records `qualification:false`, `new_gold:false`,
`native_outputs:false` and `old_continuity_accepted:false`. Its purpose is to
identify the next reproducibility boundary before a separately reviewed source
correction. It cannot supply new gold, relax exact original continuity, authorize
native inference, or qualify retrieval quality, public APIs, model publication,
defaults, releases or application upgrades. This preparation does not dispatch CI.
