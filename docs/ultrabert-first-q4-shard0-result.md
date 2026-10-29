# UltraBERT first conservative Q4 shard

[Actual40855927708](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40855927708)
at`a5feac83e177d7e08cfa23384089f7584f0736a9` completes all686 shard0 rows
with exact input IDs, durable raw/normalized output streams and actual Ref.
The [independent small review](../tests/results/ultrabert/first-q4-shard0-review.json)
authenticates13 small members, report/manifest identities and two Ref-versus-
independent metric records. Large vectors/journal/model remain remote; they were
not independently downloaded or rehashed locally.

| Output versus original F32 | Maximum error | RMS | Minimum row cosine | Original maximum magnitude |
| --- | ---: | ---: | ---: | ---: |
| Raw CLS |2.76204991|0.62445053|0.9807522714|22.565872|
| Normalized |0.0258125439|0.00612672935|0.9807522717|0.228689|

Both strict0.002/0.99999 vector flags fail. These failures remain negative;
service-level completion does not qualify the model. The candidate still uses
only12 late FFN Q4_K matrices and206 F32 tensors. Its legacy header and pending
standard-header/publication qualification remain explicit.

The686 paired public raw/normalized calls (1,372 calls) take905.04 seconds inside
this process; this excludes compilation/downloads and is not a paired speed or
RSS benchmark. No original or F32 forward ran. No other shard or Q8 was activated.

Next independently declare the remaining seven shards using this same immutable
candidate and reviewed original/F32 freeze, without conversion or original/F32
replay. Preserve shard0 streams and every strict flag; compare all5,483 vectors
and the fixed300-query NDCG/MRR/Recall degradation/top1 gates before any task
conclusion. A single shard's cosine does not establish retrieval quality or
justify changing precision. Further calibrated/critical-tensor alternatives
need their own development protocol rather than silently replacing this recipe.
