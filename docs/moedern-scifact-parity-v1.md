# MoEdern full SciFact original-first parity v1

This adds missing retrieval coverage for the 1.5M, 27M and 115M models. It does
not replace their frozen ten-case original stage captures or turn the existing
1.5M quality-only measurement into an original comparison.

The [protocol](moedern-scifact-parity-v1.json) is declared before candidate
retrieval outputs. Use all 300 canonical test queries and 5,183 documents from
the pinned BEIR SciFact inputs and notices in [the existing protocol](moedern-scifact.md).
Documents remain title + one ASCII space + text, with ascending string IDs.
Queries use the original `encode_query`; documents use `encode_document`. The
actual original Transformer/Dense/MultiVectorMask/Normalize pipeline supplies
512-token truncation, 32-token query expansion and scoring masks.

Each separately rendered [hosted CPU worker](../tools/kaggle/moedern-scifact-original/kernel.py)
pins one exact Git commit and variant. Before original inference it verifies the
checkpoint files, exact runtime package/source identities, loaded weight bytes
and original module pipeline. Three first queries and the ten longest documents
(input lengths only) check ordinary/observer identity and the public selected
slice, and provide a conservative encoding estimate. An estimate above eight
hours rejects before the full panel; retained evidence must then inform a new
input-only sharding declaration. No shared-host framework or model execution is
needed. Each variant has its own session and a nine-hour command limit.

Preserve the full original token IDs, supplied/effective/scoring masks, all and
selected token vectors, byte offsets and SHA256 values remotely. MaxSim uses
float32 dot products and a float64 query-token sum, with document-ID tie breaks.
Six predeclared pairs also check the actual author's scorer against its explicit
float32 reduction and this ranking protocol. Preserve notices with the output;
these data keep their upstream terms rather than the model's Apache license.

The first phase executes **only original Python**. Review its small report,
complete 5,483-row inventory, source and raw-artifact identities before freezing
inputs for native retrieval. A subsequent native phase must authenticate that
freeze before model access, compare every token/mask and vector, and retain
actual crispembed-diff and independent error/magnitude results. Historical F32
budgets stay maximum absolute 0.002 and minimum row cosine 0.99999.

For F32 and separately labeled quantized candidates, full retrieval requires
NDCG@10, MRR@10 and Recall@100 drops no greater than 0.001 each, plus at least
0.99 top-one agreement, relative to the corresponding frozen original variant.
Quantized numerical drift remains reported under the unchanged F32 budgets;
a ranking pass does not relabel numerical failures. No authored cosine result,
partial corpus, quality-only score or completed original capture authorizes a
quantized publication. All three originals and native/quantized comparisons
remain pending until their actual reports are reviewed.
