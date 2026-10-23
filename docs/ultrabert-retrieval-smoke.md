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

## Executable scorer and controls

`tools/ultrabert_retrieval.py FIXTURE ORIGINAL.json NATIVE.json REPORT.json`
checks the exact fixture hash,32 ordered vectors, CLS/query/document metadata,
finite768-wide raw/normalized features and normalization consistency (max1e-6,
declared before model-backed task outputs). Each producer record has `pooling`,
`query_prompt`, `document_prompt`, `fixture_sha256`, and an ordered `vectors` list
of `{id, raw_cls, normalized_cls}`. This format validates records; source/model/
module/API provenance still needs the separate original/native producer audit.
The scorer exits nonzero on failed vector, top1 or gold-quality gates and retains
complete rankings/errors in the output JSON.

`python tests/test_ultrabert_retrieval.py -v` passes four model-free control
groups: valid synthetic vectors, a real vector/ranking error, identical but
poor-quality retrieval, wrong pooling/prefix/order/coverage and invalid values.
Synthetic control vectors are never model results. The lightweight
`ultrabert-retrieval.yml` workflow registers these controls; actual original
SentenceTransformer/native task execution remains pending.

## Original-first model producer

`tools/ci-heavy/ultrabert_retrieval.py` and `ultrabert-retrieval-model.yml` are a
separate lane from PR74. The workflow requires the successful API run and its
full native source. Preflight validates the service run/path/source, artifact
SHA256 and complete six-case/seven-row pooled/API receipt before heavy package
installation, model download or builds. Failed/pending receipts are rejected;
39939481805 was checked as a real negative control without any model download.

The inspected SentenceTransformer6.0.1 source is commit
[b0acd1a4d22ba2bc10072868ddea7612561c9424](https://github.com/huggingface/sentence-transformers/tree/b0acd1a4d22ba2bc10072868ddea7612561c9424).
The producer verifies installed wrapper/Pooling file hashes and the actual
`pooling_mode="cls"`/include-prompt configuration. Hooks observe attended token
IDs and the first raw hidden vector entering the real Pooling module; unhooked
wrapper results must remain bit-exact. Named query and unprefixed document calls
run through the actual wrapper, with raw and explicit normalized outputs.
Original vectors/IDs/source provenance and rankings are saved before native work.

The native phase checks out the API-qualified immutable source, validates all
four source fingerprints and builds a fresh CPU/F32 library with portable flags
and the repository baseline checker. It regenerates the exact frozen F32 GGUF
and measures actual public raw-token/normalized embedding APIs on all32 inputs.
The fresh binary is reported with its own hash; source qualification is not a
claim of byte-identical API artifacts. Earliest ID mismatches, vector max/mean/
RMS/norms, all scores/rankings and gold failures are preserved. Model payloads and
builds remain outside the report artifact. Full-stage debt remains separate.

After reviewing API39943940607 at4be9f864 as successful, dispatch exactly once:

```sh
gh workflow run ultrabert-retrieval-model.yml   --ref audit/ultrabert-retrieval-producer   -f api_run=39943940607   -f native_source=4be9f864e0e7bc154325e1edc1bdf29a12210a69
```

The API job is still pending at preparation time; this command is conditional,
not authorization to treat a pending/failed run as qualified. If a later source
is corrected, use its completed reviewed API receipt and full source instead.
Check the dispatched producer head against the reviewed branch before execution.
Seven model-free guard groups cover source/status/receipt errors, missing or
forged numeric data, safe receipt extraction, no-model preflight, immutable original input/module reuse, and actual Python binding/library identity. No model-backed task result is claimed yet.

On a native/quality failure after original capture, preserve the first original
bytes. The workflow's optional `original_run` input fetches the prior small
artifact and checks producer head, recorded original SHA256, all32 input hashes/
IDs/masks and wrapper/model source provenance before package installation. It
reuses `original.json` byte-for-byte; it does not rerun original inference.
Manual authorized runs can pass `--reuse-original FILE --original-sha256 SHA`
after independently verifying the prior service artifact. Always use fresh
scratch/output directories and preserve the initial failed scores and logs.

The producer also checks that the loaded native library and Python binding resolve
to the fresh build/qualified checkout, and records their hashes. A default-library
or installed-package fallback is rejected before candidate inference.
