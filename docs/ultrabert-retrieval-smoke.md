# UltraBERT owned retrieval smoke v1 — frozen before task outputs

[Fixture](../tests/regression/ultrabert/retrieval-smoke-v1.json), SHA256
`80143c78618b835d388d3133dbf508b8ff129c12659a42b94b2823983b1b6dc4`.

Eight independently authored English questions and24 owned explanatory documents
exercise semantic retrieval, including closely related distractors. Inputs and
relevance judgements are frozen before any original/native retrieval result.
No external dataset text is redistributed. This small authored smoke is neither
an upstream MTEB/BEIR reproduction nor a representative quality benchmark.

After the bounded public-API job passes, run the pinned original checkpoint
through **SentenceTransformer6.0.1**, Torch2.14.0 CPU and Transformers5.16.1, with
hash-checked custom source at367d5343. Use the exact named query prompt from the
fixture, unprefixed documents, raw CLS and explicit L2 normalization. Preserve
actual tokenizer IDs/masks, module graph, prompt/default settings, ordering and
untruncated lengths. Every input must fit512 including BOS; reject instead of
silently truncating. Preserve original outputs before running candidates.

Run the gated native F32 public text API on identical inputs, using explicit
query prefixes and unprefixed documents. Compare every raw/normalized vector
with the existing0.99999/0.002 budgets, finite values and exact IDs. Compute all
8×24 cosine scores in float64; rank by descending score then ascending document
ID. Report full rankings/scores, top margins and all relevance outcomes for both
implementations, including every miss. Compute Recall@1, MRR@10 and NDCG@10.

Prospective smoke acceptance is fixed in the fixture: all32 inputs present,
pooled vector gates pass, native/original top1 agrees for every query, and each
implementation's Recall@1 is at least7/8. Keep numerical agreement and gold
accuracy as separate results. Preserve first failures, even if the original
wrapper misses the quality gate. Never change documents/gold/prompts/precision
or tolerances after observing outputs to obtain a pass. Wrong-query-prefix,
mean-pooling and shuffled/missing-ID controls must reject before trusting results.

This smoke does not settle public pair APIs, F16/quantized/GPU/long context,
full intermediate parity or independent broad retrieval quality. Plan a separately
licensed representative benchmark after this bounded check. No model publication,
registry alias, release or app upgrade follows from smoke success alone.
All model-backed work runs on appropriate hosted resources or the authorized M1
with its resource checks; retain large payloads outside Git.
