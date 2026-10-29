# UltraBERT same-candidate full retrieval continuation

This separately closed [protocol](ultrabert-remaining-q4-v1.json) and
[standalone six-hour workflow](../.github/workflows/ultrabert-remaining-q4-v1.yml)
complete the existing last-four-FFN candidate, without new conversion, original
inference, F32 inference, precision tuning or a new reference. Runtime018 and
a06 are built once with the same CPU/root profile as the
[actual first shard40855927708](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40855927708).
The published [first-shard review](../tests/results/ultrabert/first-q4-shard0-review.json)
preserves both strict vector failures: raw maximum2.76205 and normalized0.0258125,
minimum row cosine0.980752. Its original/F32 freeze is independently positive;
quantized vector drift never becomes a F32-parity claim.

The only admitted model remains576,997,440 bytes, SHA256
`087c9be0fa50ba57db74e650889521c985d306863c94aabf731764538eb4533c`.
It contains12 Q4_K late FFN matrices and206 unchanged F32 tensors. The legacy
header remains explicit; this source does not canonicalize metadata or qualify
publication, aliases, defaults, Q8 or a release.

Before any new model call, authenticate the exact completed services, attempts,
source revisions, whole artifact ZIPs and full member manifests for the existing
original/F32 freeze, model preparation and shard0. Import shard0's exact report,
controller/protocol binding, library/comparator identities, input order, journal
and both output streams. Rehash every committed row, including signed zero;
reject torn journals, extra tails, missing rows, changed IDs/text or mixed models.
The imported thirteen-member manifest and report remain byte identical. Every
new shard also binds its current controller, runtime objects, model, original
receipt, actual library and comparator. Historical guards are neither weakened
nor retargeted.

Only shards1–7 execute, sequentially in one hosted job. Each row retains the
frozen raw-CLS then normalized public call order and exact token-ID checks.
Attempt counters are written before each call; both F32 rows are synced before
the durable journal commit. No resume is enabled in this first continuation.
Failures retain current counters, committed rows and a structural failure receipt.
The exact successful totals are4,797 new rows /9,594 new public calls, plus686
imported rows /1,372 imported calls:5,483 rows /10,966 calls across the two
producer attempts. Original, F32 and quantizer invocation counts remain zero.

All eight shards retain the two actual `crispembed_diff::Ref` comparisons and
independent F64 metrics at unchanged absolute0.002/minimum cosine0.99999. The
seven new shards continue collecting despite strict quantized-versus-F32 vector
failures; their negative flags remain explicit. Aggregation reauthenticates all
journals, checks every input exactly once, recomputes the independent metrics and
retains worst row/component errors and magnitudes. Full original/candidate
rankings and a small300-query summary remain artifacts.

Task evaluation uses the frozen018 ranking code, all300 queries and all5,183
documents. NDCG@10, MRR@10 and Recall@100 may each degrade at most0.001; top-one
agreement must be at least0.99. The workflow's final numerical exit follows these
predeclared quantized task gates, while every strict F32 vector flag is separately
reported. Complete task evaluation can reject; a numerical rejection is saved
before nonzero exit. Publication remains false even if retrieval gates pass.

All builds, package installation, model/vector handling and inference run only on
the admitted hosted CPU runner (two workers,14GiB RAM/24GiB free disk and the
existing five-hour per-shard guard inside a six-hour workflow). Large artifacts
stay remote for90 days; small receipts identify the immutable source, compiler
profile, binaries, services, all strict outcomes, rankings and call counts. This
source checkpoint has not executed the remaining shards.
