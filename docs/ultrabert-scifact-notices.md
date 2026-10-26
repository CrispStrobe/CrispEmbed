# SciFact audit data attribution and terms

This audit uses author-issued SciFact data and the explicitly pinned BEIR
SciFact corpus/query/qrels snapshot. Dataset terms are separate from the
UltraBERT model's Apache-2.0 declaration and this repository's code license.

The [author's LICENSE.md at68b98a56](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md)
assigns claims and evidence annotations in `claims_*.jsonl` to
[CC BY4.0](https://creativecommons.org/licenses/by/4.0/), and assigns the
abstracts in `corpus.jsonl`, derived from Semantic Scholar's
[S2ORC](https://github.com/allenai/s2orc), to
[ODC-By1.0](https://opendatacommons.org/licenses/by/1-0/).
Its Apache-2.0 section applies to the author's code. Preserve the author's
[README and citation](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/README.md)
and the actual542-byte license alongside each retained dataset copy.

The pinned [BEIR SciFact card](https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md)
and [qrels card](https://huggingface.co/datasets/BeIR/scifact-qrels/blob/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/README.md)
declare [CC BY-SA4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Credit BEIR's authors and retain its full citation as supplied in those cards.
Keep those declarations and the upstream terms together. Dataset copies and
this audit's joined/ordered dataset and gold transformations retain the
BEIR share-alike declaration; they are not relabeled Apache-2.0. Honor upstream
attribution and database notices when copying or sharing them. No permission
from an incompatible third-party snapshot is assumed.

Declared transformations: join author abstract sentences with one ASCII space;
join title and abstract with one ASCII space; preserve exact source text/IDs;
select all300 canonical BEIR test query IDs and all5,183 documents; sort IDs
lexicographically; retain canonical binary BEIR qrels and separately journal all
differences from author `evidence` and `cited_doc_ids`. For model input only,
apply the declared query prefix and right truncation to512 original token IDs,
including BOS. No query, long document or failed row is dropped or tuned.

Actual raw copies, notices and prepared data remain in remote workflow artifacts,
with exact member hashes and service retention dates. Git contains the protocol
and source tools, not copied corpus/claims/qrels payloads. The preparation
receipt records the immutable author S3 object version, actual archive SHA256,
pinned BEIR payload identities, all input/gold hashes and applicable notices
before any model output. Retain these notices with future preserved copies.
