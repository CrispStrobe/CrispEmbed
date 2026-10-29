# UltraBERT first conservative Q4 model

[Actual conversion40843704011](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40843704011)
at`c3d2fc051f6632e06d5e7ba69ac700a03a703e22` completes without any native or
original model forward. The [unchanged producer](../tests/results/ultrabert/first-q4-model-40843704011.json)
and [independent small-report review](../tests/results/ultrabert/first-q4-model-40843704011-review.json)
bind the complete small artifact,218 semantic tensor roles, source/runtime/build
and actual quantizer records. Large model payloads remain remote; the reviewer
did not independently download or rehash them locally.

The actual candidate is576,997,440 bytes, SHA
`087c9be0fa50ba57db74e650889521c985d306863c94aabf731764538eb4533c`.
Only12 FFN gate/up/down matrices in layers20–23 use Q4_K; the other206 tensors
remain byte-exact F32. This is a conservative mixed-precision experiment with
limited compression, not a generally Q4-sized model or a qualified embedding.
The old frozen quantizer writes legacy header12; standard-header publication
qualification remains separate.

The [new reviewed shard0 source](https://github.com/CrispStrobe/CrispEmbed/tree/a5feac83e177d7e08cfa23384089f7584f0736a9)
binds only this exact model and the already reviewed original/F32 freeze.
Workflow`ultrabert-reviewed-q4-shard0-v1.yml` must dispatch with its matching
exact branch and`source_ref`. It permits only686 strided rows, with raw and
normalized public calls, exact IDs, durable per-row journals and actual
crispembed-diff/independent comparisons. It cannot convert a model, run original
or F32 inference, activate another shard, run Q8, aggregate or qualify retrieval.
Nine bounded controls pass; actual shard execution remains pending.

All5483 rows/eight shards and300-query retrieval metrics are required for full
qualification. Preserve strict historical errors and the remaining Q8 task,
public consumers and publication gates; one shard's cosine cannot close them.
