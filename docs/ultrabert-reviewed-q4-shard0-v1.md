# UltraBERT reviewed conservative Q4 shard 0

[Conversion 40843704011](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40843704011)
at `c3d2fc051f6632e06d5e7ba69ac700a03a703e22` completed without original,
F32 or quantized model forwards. The unchanged
[producer](../tests/results/ultrabert/first-q4-model-40843704011.json) and
[independent review](../tests/results/ultrabert/first-q4-model-40843704011-review.json)
authenticate all 13 small archive members, all 218 declared names/shapes/types,
12 actual Q4_K matrices and 206 F32 payload-preservation assertions. This is a
source/report audit; the model and raw payloads were not read locally.

The immutable model is 576,997,440 bytes with SHA256
`087c9be0fa50ba57db74e650889521c985d306863c94aabf731764538eb4533c`.
Only gate/up/down matrices in layers 20–23 use Q4_K. All other tensors stay F32,
including vocabulary, attention projections and norms. The frozen018 C quantizer
wrote the legacy internal file-type header12. Tensor Q4_K storage is genuine;
this artifact is not qualified for standard-header publication. No metadata is
relabelled and Q8 remains closed.

The [new protocol](ultrabert-reviewed-q4-shard0-v1.json) activates only the first
original strided shard: indices `0,8,...,5480`, exactly 686 of 5,483 frozen rows.
It binds the reviewed compressed model artifact, its unchanged receipt and
whole-file model SHA, and the previously reviewed zero-forward F32/original
freeze. All old018/37/08/c3d controllers, protocols, references and workflows
remain byte-exact. The new source owns its separate admission and exposes no
model-conversion, other-shard, aggregate, resume or Q8 phase.

The [standalone workflow](../.github/workflows/ultrabert-reviewed-q4-shard0-v1.yml)
requires one exact reviewed source matching the actual GHA checkout, preserves
the frozen018 runtime/a06 Git objects and build flags, and authenticates both
retained archives before model access. The actual Ultra production object and
whole-library symbol inventory are checked with the same scoped audit. A fresh
Python interpreter must load the exact built C ABI and dependency paths before
the model phase. Native public raw-CLS and normalized calls retain every token
ID against the frozen tokenizer input, with Unicode and truncation gates held
at one. No original/F32 forward repeats.

The forward/journal/comparator bodies are copied unchanged from c3d. Each row is
checkpointed durably to output with input/vector SHA records. Timeout or failure
retains committed prefixes; resume stays closed until separately reviewed.
Actual `crispembed_diff::Ref` and independent metrics compare raw/normalized
vectors to the old original targets, preserving 0.002/0.99999 strict F32 flags.
Quantized numerical failure remains visible even when diagnostic collection
completes. No threshold is changed after output.

This first shard cannot qualify full retrieval, rank all 300 queries, select a
recipe, publish a model, flip a default or qualify release/app consumers. All
eight shards, 5,483 rows and unchanged full ranking/vector gates remain required.
Earlier F32/M1/strict-stage debts and the initial nine-hour Q4 interruption are
separate retained evidence. Large archives and arrays stay remote for 90 days;
only small review evidence should be collected locally. Workflow registration,
source review and actual dispatch belong to the coordinator.
