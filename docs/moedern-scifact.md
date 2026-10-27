# MoEdern1.5M SciFact retrieval quality (evidence only)

An independent retrieval-quality measurement of the published
`moedern-1.5m` alias
(`cstr/SauerkrautLM-MoEdernColBERT-1.5M-CrispEmbed-GGUF` at `6f3b3514`,
SHA256 `0e2b8773abd8c78a5825280f9a73cb808b88d56f0675a8e4bb7f0509f5b6cc51`)
through the public raw-text C API. **This is quality evidence only.** There is
no pass/fail gate, it changes no default, and it claims **no comparison with the
original author model**: no frozen original-model SciFact ranking exists for
this model, so the numbers stand alone.

## Protocol

- Data: the same pinned BEIR SciFact identities as the frozen UltraBERT v2
  protocol (`tests/regression/ultrabert/scifact-protocol-v2.json`): corpus and
  queries Parquet at BEIR revision `b3b53356`, test qrels at `2938d17d`
  (Git blob `c62f58c7`), all byte sizes and SHA256 values checked before use.
  The legacy/Parquet equality and the author-column journal were established by
  that lane and are not repeated here.
- Rows: all 300 canonical test queries and all 5,183 documents, ascending string
  ID order; nothing dropped or selected from scores.
- Text: query = BEIR query text; document = BEIR title + one ASCII space + BEIR
  text, no strip or normalization. Role `query` / `document` is passed to the
  public API, whose frontend adds any role handling and truncates at 512 tokens.
- Model: alias resolved into an empty fresh cache, size and SHA256 verified;
  gates `CRISPEMBED_MOEDERN_PUBLIC_TEXT`, `_PACKED`, `_TEXT_COMPOSED`,
  `_TEXT_FRONTEND`, `_COLBERT` = 1; root library built with LLAMAFILE ON; CPU,
  2 threads, Python `CrispMoEdern`.
- Scoring: exact brute-force MaxSim over each row's selected vectors (float32
  dot products, float64 sum); ties broken by ascending document ID. A dozen pairs
  are also scored by the native pair scorer and the difference is recorded.
- Metrics: NDCG@10, MRR@10 and Recall@100 with binary canonical qrels, the same
  definitions as the UltraBERT SciFact helper (`tests/test_moedern_scifact.py`
  covers them on a tiny authored fixture).

Run: `gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/moedern_scifact.py`.
Outputs: `moedern-scifact.json` (metrics, timing, hashes), `rankings.json`
(top-100 per query), `preflight.json`, and these notices. The run refuses to
continue if a timed preflight of the 40 longest documents projects encoding
beyond 270 minutes; `--shard I --shards N` then splits the documents.

## Data attribution and terms

Dataset terms are separate from the model's Apache-2.0 declaration and this
repository's code license.

The [SciFact author LICENSE.md at 68b98a56](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md)
assigns the claims to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
and the abstracts, derived from Semantic Scholar's
[S2ORC](https://github.com/allenai/s2orc), to
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). Preserve the
author's [README and citation](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/README.md).

The pinned [BEIR SciFact card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md)
and [qrels card](https://huggingface.co/datasets/BeIR/scifact-qrels/blob/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/README.md)
declare [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Credit BEIR's authors with the citation in those cards. Joined/ordered copies
keep the share-alike declaration and are never relabeled Apache-2.0.

Raw data stays in runner scratch and is not retained; the four upstream
license/README files are downloaded with pinned SHA256 and kept in the run artifact. Git holds
only the tools and this description.
