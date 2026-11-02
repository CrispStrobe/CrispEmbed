# Retained MoEdern three-document comparison: twelve negatives preserved

[Run40834748271](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40834748271)
finished its comparison at source92f75550a0ce0f1486f6d7603164a6628aeb24a0. The
job concludes **failure**, with a structurally complete report and twelve
numerical rejections. The [independent review receipt](../tests/results/moedern-colbert/three-doc-compare-v2-review.json)
binds the26,680-byte small ZIP and all nine member identities. Actual Ref and
independent metrics agree for all18 matrices. No original/native/model forward,
model or tokenizer access occurred in this compare-only job; raw arrays remain
in the immutable remote10,733,423-byte artifact.

| Comparison | Pass | Reject |
| --- | ---: | ---: |
| Frozen original vs fresh original | 2 | 4 |
| Frozen original vs retained native93 | 0 | 6 |
| Fresh original vs retained native93 | 4 | 2 |

All three documents retain both all-token and scoring-selected matrices.
Documents24928817 and5849439 show fresh-original drift beyond the frozen budget;
the fresh outputs match retained native93 within7.16e-7. This does not replace
frozen gold, clear any corpus failure or establish a backend cause.

Document24726600 remains a reproducible fresh-original/native difference while
its fresh-original/frozen comparison passes at maximum4.246830940246582e-7.
There is exactly one failing input position58/token29084: row58 of the all-token
matrix and row57 of the scoring-selected matrix. At component19, fresh original
is0.023389169946312904 and retained native93 is0.000559773703571409, absolute
0.022829396242741495 with minimum row cosine0.9957477159184743. Those exceed the
unchanged0.002/0.99999 gates. Other input positions remain reported rather than
being hidden by an aggregate metric.

A separately claimed one-document intermediate capture is justified, with two
mandatory ordinary barriers before attributing stages: the new ordinary original
must pass the unchanged frozen first-document gates; ordinary native all/selected
outputs must match the exact retained93 hashes. Observed and repeat outputs must
then be byte-neutral. The old blocked three-case guard remains unchanged.

Actual router softmax/topk/division should be captured at the executed ATen
operations under the appropriate FFN/module/occurrence context, with integer
indices kept integer. The prior after-hook recomputation is a derived diagnostic,
not an executed publisher-operation reference. SDPA capture must preserve all
arguments and the actual return/kernel path. Capture bookkeeping must avoid
recursive dispatch and later in-place overwrite. Every supplied/effective/scoring
mask and occurrence remains explicit; derived cutoff margins are labeled as
such. New source/protocol must freeze before outputs and reject if either
ordinary anchor or observer neutrality fails. No runtime/tolerance/gold change
or causal conclusion follows from this proposal.

Review authenticated the small artifact and checked all18 reported scalar gates,
Ref metrics, row diagnostics and packet identities. It did not download arrays,
weights, binaries or tokenizers, rerun models or recompute raw metrics locally.
All prior packets and18 outcomes remain immutable; publication and numerical
qualification remain false.
